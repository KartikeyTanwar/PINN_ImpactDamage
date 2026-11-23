# ============================================================
# DAMAGE PINN (Cached Von Mises Stress + O(1) Scaling + Adam & L-BFGS)
# ============================================================

import os
import json
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from .config import CONFIG, device, safe_torch_load
from .elasticity_pinn import compute_stress, sample_collocation


class DamagePINN(nn.Module):
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
        return torch.sigmoid(self.linears[-1](out))


def von_mises_stress(model_e, x, y, z, t):
    (sig_xx, sig_yy, sig_zz, sig_xy, sig_xz, sig_yz, *_) = compute_stress(model_e, x, y, z, t)
    s11, s22, s33 = sig_xx.detach(), sig_yy.detach(), sig_zz.detach()
    s12, s13, s23 = sig_xy.detach(), sig_xz.detach(), sig_yz.detach()
    return torch.sqrt(0.5 * (
        (s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2 +
        6.0 * (s12**2 + s13**2 + s23**2)
    ) + 1e-8)


def loss_damage_pde(model_d, x, y, z, t, vm_cached, physics):
    """
    Dimensionless CDM evolution PDE:
      dD/dt = H(vm - sigma_y) * ((vm - sigma_y) / sigma_y)^n
    Uses cached vm_cached so model_e is not re-differentiated every epoch.
    """
    sigma_y = physics['sigma_y_norm']
    n_exp   = 2.0
    k       = 1000.0

    D   = model_d(x, y, z, t)
    D_t = torch.autograd.grad(
        D, t, grad_outputs=torch.ones_like(D), create_graph=True
    )[0]

    heaviside     = torch.sigmoid(k * (vm_cached - sigma_y))
    excess_stress = torch.clamp((vm_cached - sigma_y) / sigma_y, min=0.0)
    driving       = heaviside * (excess_stress ** n_exp)

    irrev    = torch.clamp(-D_t, min=0.0)**2
    residual = D_t - driving

    return torch.mean(residual**2) + 10.0 * torch.mean(irrev)


def sample_damage_bc(N=300):
    zeros = torch.zeros((N, 1), device=device)
    ones  = torch.ones((N, 1), device=device)
    r = lambda: torch.rand((N, 1), device=device)
    xb = torch.cat([zeros, ones, r(),   r()],  dim=0)
    yb = torch.cat([r(),   r(),  zeros, ones], dim=0)
    zb = torch.rand((4 * N, 1), device=device)
    tb = torch.rand((4 * N, 1), device=device)
    return xb, yb, zb, tb


def loss_damage_bc(model_d, N=300, bc_pts=None):
    xb, yb, zb, tb = sample_damage_bc(N) if bc_pts is None else bc_pts
    return 4.0 * torch.mean(model_d(xb, yb, zb, tb)**2)


def sample_damage_ic(N=300):
    return (torch.rand((N, 1), device=device),
            torch.rand((N, 1), device=device),
            torch.rand((N, 1), device=device),
            torch.zeros((N, 1), device=device))


def loss_damage_ic(model_d, N=300, ic_pts=None):
    xi, yi, zi, ti = sample_damage_ic(N) if ic_pts is None else ic_pts
    return torch.mean(model_d(xi, yi, zi, ti)**2)


def train_damage_pinn(model_e, physics, run_tag='default'):
    print("\n" + "=" * 55)
    print("  DAMAGE PINN — TRAINING (Cached Stress + Adam + L-BFGS)")
    print("=" * 55)

    epochs         = physics['epochs']
    n_col          = physics['n_col']
    resume         = physics['resume']
    lbfgs_steps    = physics.get('lbfgs_steps', CONFIG['lbfgs_steps'])
    resample_every = physics.get('resample_every', CONFIG['resample_every'])

    ckpt_path = f"{CONFIG['ckpt_damage']}/damage_{run_tag}.pth"
    log_path  = f"{CONFIG['logs']}/damage/loss_{run_tag}.json"

    model_e.eval()
    for p in model_e.parameters():
        p.requires_grad_(False)

    model_d = DamagePINN(CONFIG['layers_damage']).to(device)
    start_epoch = 0

    if resume and os.path.exists(ckpt_path):
        ckpt = safe_torch_load(ckpt_path, map_location=device)
        model_d.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt['epoch']
        print(f"✅ Resumed from epoch {start_epoch}")

    optimizer = torch.optim.Adam(model_d.parameters(), lr=CONFIG['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=CONFIG['lr_decay_step'], gamma=CONFIG['lr_decay_rate']
    )

    def resample_with_stress():
        xc, yc, zc, tc = sample_collocation(n_col, physics)
        vm = von_mises_stress(model_e, xc, yc, zc, tc)
        return xc.detach(), yc.detach(), zc.detach(), tc.detach().requires_grad_(True), vm

    x_col, y_col, z_col, t_col, vm_col = resample_with_stress()

    history   = {'total': [], 'pde': [], 'bc': [], 'ic': [], 'bounds': [], 'epochs': []}
    best_loss = float('inf')
    save_every = max(epochs // 10, 500)

    # ── Stage 1: Adam ─────────────────────────────────────
    for epoch in tqdm(range(start_epoch, start_epoch + epochs), desc="Damage (Adam)"):
        if epoch > start_epoch and (epoch - start_epoch) % resample_every == 0:
            x_col, y_col, z_col, t_col, vm_col = resample_with_stress()

        optimizer.zero_grad()
        l_pde = loss_damage_pde(model_d, x_col, y_col, z_col, t_col, vm_col, physics)
        l_bc  = loss_damage_bc(model_d, N=CONFIG['N_bc'])
        l_ic  = loss_damage_ic(model_d, N=CONFIG['N_ic'])
        loss  = CONFIG['w_pde'] * l_pde + CONFIG['w_bc'] * l_bc + CONFIG['w_ic'] * l_ic

        loss.backward()
        optimizer.step()
        scheduler.step()

        history['total'].append(loss.item())
        history['pde'].append(l_pde.item())
        history['bc'].append(l_bc.item())
        history['ic'].append(l_ic.item())
        history['bounds'].append(0.0)
        history['epochs'].append(epoch)

        if epoch % 500 == 0:
            tqdm.write(f"Epoch {epoch:6d} | Total: {loss.item():.5f} | "
                       f"PDE: {l_pde.item():.5f} | BC: {l_bc.item():.5f} | IC: {l_ic.item():.5f}")

        if loss.item() < best_loss or (epoch % save_every == 0 and epoch > start_epoch):
            best_loss = min(loss.item(), best_loss)
            torch.save({'epoch': epoch, 'model_state': model_d.state_dict(),
                        'loss': loss.item()}, ckpt_path)

    # ── Stage 2: L-BFGS Polishing ─────────────────────────
    if lbfgs_steps > 0:
        print(f"🔧 Polishing Damage PINN with L-BFGS (max {lbfgs_steps} iters)...")
        x_col, y_col, z_col, t_col, vm_col = resample_with_stress()
        bc_fixed = sample_damage_bc(N=400)
        ic_fixed = sample_damage_ic(N=400)

        lbfgs = torch.optim.LBFGS(
            model_d.parameters(), lr=1.0, max_iter=lbfgs_steps,
            history_size=50, line_search_fn='strong_wolfe'
        )
        step_counter = [start_epoch + epochs]

        def closure():
            lbfgs.zero_grad()
            l_pde = loss_damage_pde(model_d, x_col, y_col, z_col, t_col, vm_col, physics)
            l_bc  = loss_damage_bc(model_d, bc_pts=bc_fixed)
            l_ic  = loss_damage_ic(model_d, ic_pts=ic_fixed)
            loss  = CONFIG['w_pde'] * l_pde + CONFIG['w_bc'] * l_bc + CONFIG['w_ic'] * l_ic
            loss.backward()

            history['total'].append(loss.item())
            history['pde'].append(l_pde.item())
            history['bc'].append(l_bc.item())
            history['ic'].append(l_ic.item())
            history['bounds'].append(0.0)
            history['epochs'].append(step_counter[0])
            step_counter[0] += 1
            return loss

        lbfgs.step(closure)

    torch.save({'epoch': history['epochs'][-1], 'model_state': model_d.state_dict(),
                'loss': history['total'][-1]}, ckpt_path)
    with open(log_path, 'w') as f:
        json.dump(history, f)

    print(f"✅ Damage PINN complete | Final loss: {history['total'][-1]:.6f}")
    return model_d, history