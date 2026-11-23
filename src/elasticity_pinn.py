# ============================================================
# 3D ELASTICITY PINN (Same Navier Physics + Batched BCs + Adam & L-BFGS)
# ============================================================

import os
import json
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from .config import CONFIG, device, safe_torch_load
from .physics import lhs_sample


class ElasticityPINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.depth      = len(layers) - 1
        self.activation = nn.Tanh()
        self.linears    = nn.ModuleList([
            nn.Linear(layers[i], layers[i + 1]) for i in range(self.depth)
        ])
        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, y, z, t):
        out = torch.cat([x, y, z, t], dim=1)
        for i in range(self.depth - 1):
            out = self.activation(self.linears[i](out))
        out = self.linears[-1](out)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]


def make_grad_tensor(data, dev=device):
    return torch.tensor(data, dtype=torch.float32, device=dev).requires_grad_(True)


def rand_grad(shape, dev=device, val=None):
    if val is not None:
        t = torch.full(shape, val, dtype=torch.float32, device=dev)
    else:
        t = torch.rand(shape, dtype=torch.float32, device=dev)
    return t.requires_grad_(True)


def impact_force(x, y, z, t, physics):
    x0, y0  = physics['x0'], physics['y0']
    r       = physics['r_avg']
    t_peak  = physics['t_peak_n']
    t_sigma = physics['t_sigma']
    F       = physics['pressure_norm']

    spatial  = torch.exp(-((x - x0)**2 + (y - y0)**2) / (2.0 * r**2))
    temporal = torch.exp(-((t - t_peak)**2) / (t_sigma**2))
    z_mask   = torch.exp(-((z - 1.0)**2) / (0.05**2))

    fx = torch.zeros_like(x)
    fy = torch.zeros_like(y)
    fz = -F * spatial * temporal * z_mask
    return fx, fy, fz


def compute_stress(model, x, y, z, t):
    u, v, w = model(x, y, z, t)

    def grad(f, var):
        return torch.autograd.grad(
            f, var, grad_outputs=torch.ones_like(f),
            create_graph=True, allow_unused=False
        )[0]

    u_x, u_y, u_z = grad(u, x), grad(u, y), grad(u, z)
    v_x, v_y, v_z = grad(v, x), grad(v, y), grad(v, z)
    w_x, w_y, w_z = grad(w, x), grad(w, y), grad(w, z)

    eps_xx, eps_yy, eps_zz = u_x, v_y, w_z
    eps_xy = 0.5 * (u_y + v_x)
    eps_xz = 0.5 * (u_z + w_x)
    eps_yz = 0.5 * (v_z + w_y)

    nu    = CONFIG['nu']
    lam_n = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu_n  = 1.0 / (2.0 * (1.0 + nu))
    trace = eps_xx + eps_yy + eps_zz

    sig_xx = lam_n * trace + 2.0 * mu_n * eps_xx
    sig_yy = lam_n * trace + 2.0 * mu_n * eps_yy
    sig_zz = lam_n * trace + 2.0 * mu_n * eps_zz
    sig_xy = 2.0 * mu_n * eps_xy
    sig_xz = 2.0 * mu_n * eps_xz
    sig_yz = 2.0 * mu_n * eps_yz

    return (sig_xx, sig_yy, sig_zz, sig_xy, sig_xz, sig_yz,
            u, v, w, u_x, v_y, w_z, u_y, u_z, v_x, v_z, w_x, w_y)


def loss_pde(model, x, y, z, t, physics):
    (sig_xx, sig_yy, sig_zz,
     sig_xy, sig_xz, sig_yz,
     u, v, w, *_) = compute_stress(model, x, y, z, t)

    def grad(f, var):
        return torch.autograd.grad(
            f, var, grad_outputs=torch.ones_like(f),
            create_graph=True, allow_unused=False
        )[0]

    u_tt = grad(grad(u, t), t)
    v_tt = grad(grad(v, t), t)
    w_tt = grad(grad(w, t), t)

    fx, fy, fz = impact_force(x, y, z, t, physics)

    res_x = grad(sig_xx, x) + grad(sig_xy, y) + grad(sig_xz, z) + fx - u_tt
    res_y = grad(sig_xy, x) + grad(sig_yy, y) + grad(sig_yz, z) + fy - v_tt
    res_z = grad(sig_xz, x) + grad(sig_yz, y) + grad(sig_zz, z) + fz - w_tt

    return torch.mean(res_x**2) + torch.mean(res_y**2) + torch.mean(res_z**2)


def sample_bc_points(N=300):
    zeros = torch.zeros((N, 1), device=device)
    ones  = torch.ones((N, 1), device=device)
    r = lambda: torch.rand((N, 1), device=device)

    xe = torch.cat([zeros, ones, r(),   r()],  dim=0)
    ye = torch.cat([r(),   r(),  zeros, ones], dim=0)
    ze = torch.rand((4 * N, 1), device=device)
    te = torch.rand((4 * N, 1), device=device)

    xf = torch.rand((2 * N, 1), device=device).requires_grad_(True)
    yf = torch.rand((2 * N, 1), device=device).requires_grad_(True)
    zf = torch.cat([zeros, ones], dim=0).requires_grad_(True)
    tf = torch.rand((2 * N, 1), device=device).requires_grad_(True)

    return (xe, ye, ze, te), (xf, yf, zf, tf)


def loss_bc(model, physics, N=300, bc_pts=None):
    if bc_pts is None:
        bc_pts = sample_bc_points(N)
    (xe, ye, ze, te), (xf, yf, zf, tf) = bc_pts

    ub, vb, wb = model(xe, ye, ze, te)
    loss_edges = 4.0 * (torch.mean(ub**2) + torch.mean(vb**2) + torch.mean(wb**2))

    (_, _, sig_zz, _, sig_xz, sig_yz, *_) = compute_stress(model, xf, yf, zf, tf)
    loss_faces = 2.0 * (torch.mean(sig_zz**2) + torch.mean(sig_xz**2) + torch.mean(sig_yz**2))

    return loss_edges + loss_faces


def sample_ic_points(N=300):
    xi = rand_grad((N, 1), device)
    yi = rand_grad((N, 1), device)
    zi = rand_grad((N, 1), device)
    ti = rand_grad((N, 1), device, val=0.0)
    return xi, yi, zi, ti


def loss_ic(model, N=300, ic_pts=None):
    xi, yi, zi, ti = sample_ic_points(N) if ic_pts is None else ic_pts
    u, v, w = model(xi, yi, zi, ti)
    loss_disp = torch.mean(u**2) + torch.mean(v**2) + torch.mean(w**2)

    def grad(f, var):
        return torch.autograd.grad(
            f, var, grad_outputs=torch.ones_like(f), create_graph=True
        )[0]

    loss_vel = (torch.mean(grad(u, ti)**2) +
                torch.mean(grad(v, ti)**2) +
                torch.mean(grad(w, ti)**2))
    return loss_disp + loss_vel


def sample_collocation(n, physics=None, seed=None):
    samples = lhs_sample(n, 4, seed=seed)
    x = make_grad_tensor(samples[:, 0:1], device)
    y = make_grad_tensor(samples[:, 1:2], device)
    z = make_grad_tensor(samples[:, 2:3], device)
    t = make_grad_tensor(samples[:, 3:4], device)
    return x, y, z, t


def train_elasticity_pinn(physics, run_tag='default'):
    print("\n" + "=" * 55)
    print("  ELASTICITY PINN — TRAINING (Adam + L-BFGS)")
    print("=" * 55)

    epochs         = physics['epochs']
    n_col          = physics['n_col']
    resume         = physics['resume']
    lbfgs_steps    = physics.get('lbfgs_steps', CONFIG['lbfgs_steps'])
    resample_every = physics.get('resample_every', CONFIG['resample_every'])

    ckpt_path = f"{CONFIG['ckpt_elastic']}/elasticity_{run_tag}.pth"
    log_path  = f"{CONFIG['logs']}/elasticity/loss_{run_tag}.json"

    model = ElasticityPINN(CONFIG['layers_elast']).to(device)
    start_epoch = 0

    if resume and os.path.exists(ckpt_path):
        ckpt = safe_torch_load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt['epoch']
        print(f"✅ Resumed from epoch {start_epoch}")

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=CONFIG['lr_decay_step'], gamma=CONFIG['lr_decay_rate']
    )

    x_col, y_col, z_col, t_col = sample_collocation(n_col, physics)
    history   = {'total': [], 'pde': [], 'bc': [], 'ic': [], 'epochs': []}
    best_loss = float('inf')
    save_every = max(epochs // 10, 500)

    # ── Stage 1: Adam with periodic LHS resampling ────────
    for epoch in tqdm(range(start_epoch, start_epoch + epochs), desc="Elasticity (Adam)"):
        if epoch > start_epoch and (epoch - start_epoch) % resample_every == 0:
            x_col, y_col, z_col, t_col = sample_collocation(n_col, physics)

        optimizer.zero_grad()
        l_pde = loss_pde(model, x_col, y_col, z_col, t_col, physics)
        l_bc  = loss_bc(model, physics, N=CONFIG['N_bc'])
        l_ic  = loss_ic(model, N=CONFIG['N_ic'])
        loss  = CONFIG['w_pde'] * l_pde + CONFIG['w_bc'] * l_bc + CONFIG['w_ic'] * l_ic

        loss.backward()
        optimizer.step()
        scheduler.step()

        history['total'].append(loss.item())
        history['pde'].append(l_pde.item())
        history['bc'].append(l_bc.item())
        history['ic'].append(l_ic.item())
        history['epochs'].append(epoch)

        if epoch % 500 == 0:
            tqdm.write(f"Epoch {epoch:6d} | Total: {loss.item():.5f} | "
                       f"PDE: {l_pde.item():.5f} | BC: {l_bc.item():.5f} | IC: {l_ic.item():.5f}")

        if loss.item() < best_loss or (epoch % save_every == 0 and epoch > start_epoch):
            best_loss = min(loss.item(), best_loss)
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'loss': loss.item(),
                'physics': {k: v for k, v in physics.items() if k != 'BC'},
            }, ckpt_path)

    # ── Stage 2: L-BFGS Polishing ─────────────────────────
    if lbfgs_steps > 0:
        print(f"🔧 Polishing Elasticity PINN with L-BFGS (max {lbfgs_steps} iters)...")
        x_col, y_col, z_col, t_col = sample_collocation(n_col, physics)
        bc_fixed = sample_bc_points(N=400)
        ic_fixed = sample_ic_points(N=400)

        lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=lbfgs_steps,
            history_size=50, line_search_fn='strong_wolfe'
        )
        step_counter = [start_epoch + epochs]

        def closure():
            lbfgs.zero_grad()
            l_pde = loss_pde(model, x_col, y_col, z_col, t_col, physics)
            l_bc  = loss_bc(model, physics, bc_pts=bc_fixed)
            l_ic  = loss_ic(model, ic_pts=ic_fixed)
            loss  = CONFIG['w_pde'] * l_pde + CONFIG['w_bc'] * l_bc + CONFIG['w_ic'] * l_ic
            loss.backward()

            history['total'].append(loss.item())
            history['pde'].append(l_pde.item())
            history['bc'].append(l_bc.item())
            history['ic'].append(l_ic.item())
            history['epochs'].append(step_counter[0])
            step_counter[0] += 1
            return loss

        lbfgs.step(closure)

    torch.save({
        'epoch': history['epochs'][-1], 'model_state': model.state_dict(),
        'loss': history['total'][-1],
        'physics': {k: v for k, v in physics.items() if k != 'BC'},
    }, ckpt_path)

    with open(log_path, 'w') as f:
        json.dump(history, f)

    print(f"✅ Elasticity PINN complete | Final loss: {history['total'][-1]:.6f}")
    return model, history