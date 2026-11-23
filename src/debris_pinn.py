# ============================================================
# DEBRIS PINN (Adam + L-BFGS Polishing + scipy.solve_ivp Validation)
# ============================================================

import os
import json
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm
from .config import CONFIG, device, safe_torch_load


def get_debris_constants(physics, m_frag=0.01):
    """Shared normalized gravity and aerodynamic drag constants for PINN & ODE."""
    L       = physics['L']
    t_cross = physics['t_cross']
    g_norm  = 9.81 * (t_cross**2) / L
    Cd      = 0.47
    rho_air = 1.225
    r_frag  = 0.005
    A_frag  = np.pi * (r_frag**2)
    k_drag  = (0.5 * Cd * rho_air * A_frag * (t_cross**2)) / (m_frag * L)
    return g_norm, k_drag


def find_fracture_zones(model_d, physics, n_fragments=5):
    print("🔍 Checking fracture condition...\n")
    sigma_y  = CONFIG['sigma_y']
    sigma_u  = CONFIG['sigma_u']
    impact_p = physics['impact_pressure']

    print(f"   Impact pressure : {impact_p/1e6:.2f} MPa")
    print(f"   Yield stress    : {sigma_y/1e6:.2f} MPa")
    print(f"   Ultimate stress : {sigma_u/1e6:.2f} MPa")

    if impact_p < sigma_y:
        print("\n✅ ELASTIC REGIME — No fracture, no fragments formed")
        return []
    elif impact_p < sigma_u:
        damage_ratio = min((impact_p - sigma_y) / (sigma_u - sigma_y), 1.0)
        print(f"\n⚠️  PLASTIC REGIME | Damage ratio: {damage_ratio:.3f}")
        if damage_ratio < 0.5:
            print("   Damage ratio < 0.5 — severe deformation, no fragmentation")
            return []
        n_actual = max(1, int(n_fragments * damage_ratio))
    else:
        print(f"\n❌ FRACTURE REGIME — Full fracture ({n_fragments} fragments)")
        damage_ratio = 1.0
        n_actual     = n_fragments

    threshold = CONFIG['damage_thresh']
    t_peak    = physics['t_peak_n']

    N_scan = 50
    x_1d   = torch.linspace(0, 1, N_scan)
    y_1d   = torch.linspace(0, 1, N_scan)
    Xg, Yg = torch.meshgrid(x_1d, y_1d, indexing='ij')

    x_flat = Xg.reshape(-1, 1).to(device)
    y_flat = Yg.reshape(-1, 1).to(device)
    z_flat = torch.ones_like(x_flat)
    t_flat = torch.full_like(x_flat, t_peak)

    model_d.eval()
    with torch.no_grad():
        D_grid = model_d(x_flat, y_flat, z_flat, t_flat).reshape(N_scan, N_scan).cpu().numpy()

    D_physical   = D_grid * damage_ratio
    damaged_idx  = np.argwhere(D_physical > threshold)

    print(f"   Raw D range     : [{D_grid.min():.3f}, {D_grid.max():.3f}]")
    print(f"   Physical D range: [{D_physical.min():.3f}, {D_physical.max():.3f}]")
    print(f"   Points > {threshold}   : {len(damaged_idx)}")

    if len(damaged_idx) == 0:
        print("\n⚠️  No points exceed threshold after physical scaling")
        return []

    if len(damaged_idx) >= n_actual:
        step     = max(1, len(damaged_idx) // n_actual)
        selected = damaged_idx[::step][:n_actual]
    else:
        selected = damaged_idx[:n_actual]

    fragment_data = []
    v_debris = physics['debris_v_norm']
    x0, y0   = physics['x0'], physics['y0']

    for i, (xi, yi) in enumerate(selected):
        xpos = xi / (N_scan - 1)
        ypos = yi / (N_scan - 1)
        dx, dy = xpos - x0, ypos - y0
        dist   = max(np.sqrt(dx**2 + dy**2), 1e-6)

        vx0 = v_debris * (dx / dist) * 0.3 * damage_ratio
        vy0 = v_debris * (dy / dist) * 0.3 * damage_ratio
        vz0 = v_debris * 0.8 * damage_ratio

        fragment_data.append({
            'id'  : i,
            'x0'  : float(xpos),
            'y0'  : float(ypos),
            'z0'  : float(physics['z_launch_norm']),
            'vx0' : float(vx0),
            'vy0' : float(vy0),
            'vz0' : float(vz0),
            'D0'  : float(D_physical[xi, yi]),
            'mass': 0.01,
        })
        print(f"   Fragment {i+1}: pos=({xpos:.2f},{ypos:.2f}) "
              f"D_phys={D_physical[xi,yi]:.3f} v=({vx0:.3f},{vy0:.3f},{vz0:.3f})")

    print(f"\n✅ {len(fragment_data)} fragments identified")
    return fragment_data


class DebrisPINN(nn.Module):
    def __init__(self, layers, fragment_info):
        super().__init__()
        self.depth      = len(layers) - 1
        self.activation = nn.Tanh()
        self.layers     = nn.ModuleList([
            nn.Linear(layers[i], layers[i + 1]) for i in range(self.depth)
        ])
        self.frag = fragment_info
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, t):
        out = t
        for i in range(self.depth - 1):
            out = self.activation(self.layers[i](out))
        out = self.layers[-1](out)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]


def loss_debris_pde(model_frag, t, physics, m_frag=0.01):
    g_norm, k_drag = get_debris_constants(physics, m_frag)
    x, y, z = model_frag(t)

    def grad(f, var):
        return torch.autograd.grad(
            f, var, grad_outputs=torch.ones_like(f), create_graph=True
        )[0]

    x_t, y_t, z_t    = grad(x, t), grad(y, t), grad(z, t)
    x_tt, y_tt, z_tt = grad(x_t, t), grad(y_t, t), grad(z_t, t)

    speed = torch.sqrt(x_t**2 + y_t**2 + z_t**2 + 1e-8)
    res_x = x_tt + k_drag * speed * x_t
    res_y = y_tt + k_drag * speed * y_t
    res_z = z_tt + k_drag * speed * z_t + g_norm

    ground_penalty = torch.mean(torch.clamp(-z, min=0.0)**2) * 100.0
    return (torch.mean(res_x**2) + torch.mean(res_y**2) +
            torch.mean(res_z**2) + ground_penalty)


def loss_debris_ic(model_frag, frag_info):
    t0 = torch.zeros(1, 1, dtype=torch.float32, device=device, requires_grad=True)
    mt = lambda val: torch.tensor([[val]], dtype=torch.float32, device=device)

    x, y, z = model_frag(t0)
    loss_pos = (x - mt(frag_info['x0']))**2 + (y - mt(frag_info['y0']))**2 + (z - mt(frag_info['z0']))**2

    def grad(f):
        return torch.autograd.grad(f, t0, grad_outputs=torch.ones_like(f), create_graph=True)[0]

    loss_vel = ((grad(x) - mt(frag_info['vx0']))**2 +
                (grad(y) - mt(frag_info['vy0']))**2 +
                (grad(z) - mt(frag_info['vz0']))**2)
    return torch.mean(loss_pos + loss_vel)


def validate_debris_with_ode(model_f, frag, physics, n_eval=200):
    """
    Solves the exact debris equations of motion using scipy.integrate.solve_ivp
    and computes relative L2 and max absolute trajectory error vs DebrisPINN.
    """
    g_norm, k_drag = get_debris_constants(physics, frag.get('mass', 0.01))
    t_eval = np.linspace(0.0, 1.0, n_eval)

    def ode_rhs(t, s):
        _, _, _, vx, vy, vz = s
        spd = np.sqrt(vx**2 + vy**2 + vz**2)
        return [vx, vy, vz,
                -k_drag * spd * vx,
                -k_drag * spd * vy,
                -k_drag * spd * vz - g_norm]

    y0 = [frag['x0'], frag['y0'], frag['z0'],
          frag['vx0'], frag['vy0'], frag['vz0']]
    sol = solve_ivp(ode_rhs, [0.0, 1.0], y0, t_eval=t_eval, rtol=1e-9, atol=1e-11)
    ode_pos = sol.y[:3]  # shape (3, n_eval)

    model_f.eval()
    t_tensor = torch.tensor(t_eval, dtype=torch.float32, device=device).reshape(-1, 1)
    with torch.no_grad():
        xp, yp, zp = model_f(t_tensor)
    pinn_pos = np.vstack([
        xp.cpu().numpy().ravel(),
        yp.cpu().numpy().ravel(),
        zp.cpu().numpy().ravel(),
    ])

    # Evaluate over airborne segment (z_ode >= 0) if ground is struck before t=1
    airborne = ode_pos[2] >= 0.0
    if np.any(airborne):
        diff = pinn_pos[:, airborne] - ode_pos[:, airborne]
        ref  = ode_pos[:, airborne]
    else:
        diff = pinn_pos - ode_pos
        ref  = ode_pos

    rel_l2_pct  = 100.0 * float(np.linalg.norm(diff) / (np.linalg.norm(ref) + 1e-12))
    max_abs_err = float(np.max(np.linalg.norm(diff, axis=0)))

    return {
        'rel_l2_err_%': rel_l2_pct,
        'max_abs_err' : max_abs_err,
        'ode_x'       : ode_pos[0].tolist(),
        'ode_y'       : ode_pos[1].tolist(),
        'ode_z'       : ode_pos[2].tolist(),
    }


def train_debris_pinn(model_d, physics, run_tag='default'):
    print("\n" + "=" * 55)
    print("  DEBRIS PINN — TRAINING (Adam + L-BFGS + ODE Check)")
    print("=" * 55)

    epochs      = physics['epochs']
    n_fragments = physics['n_fragments']
    resume      = physics['resume']
    lbfgs_steps = physics.get('lbfgs_steps', CONFIG['lbfgs_steps'])
    ckpt_dir    = CONFIG['ckpt_debris']
    log_dir     = f"{CONFIG['logs']}/debris"

    fragment_list = find_fracture_zones(model_d, physics, n_fragments)
    if len(fragment_list) == 0:
        print("\n✅ No fragmentation — skipping debris trajectory training")
        return [], [], []

    N_t   = 500
    t_arr = torch.linspace(0, 1, N_t, device=device).reshape(-1, 1).requires_grad_(True)

    all_models, all_histories = [], []

    for frag in fragment_list:
        fid = frag['id']
        print(f"\n── Fragment {fid + 1}/{len(fragment_list)} ──")

        ckpt_path = f"{ckpt_dir}/debris_frag{fid}_{run_tag}.pth"
        log_path  = f"{log_dir}/loss_frag{fid}_{run_tag}.json"

        model_f = DebrisPINN(CONFIG['layers_debris'], frag).to(device)
        start_epoch = 0

        if resume and os.path.exists(ckpt_path):
            ckpt = safe_torch_load(ckpt_path, map_location=device)
            model_f.load_state_dict(ckpt['model_state'])
            start_epoch = ckpt['epoch']
            print(f"  ✅ Resumed from epoch {start_epoch}")

        optimizer = torch.optim.Adam(model_f.parameters(), lr=CONFIG['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=CONFIG['lr_decay_step'], gamma=CONFIG['lr_decay_rate']
        )

        history   = {'total': [], 'pde': [], 'ic': [], 'epochs': []}
        best_loss = float('inf')
        save_every = max(epochs // 5, 200)

        # ── Stage 1: Adam ─────────────────────────────────
        for epoch in tqdm(range(start_epoch, start_epoch + epochs),
                          desc=f"Fragment {fid + 1} (Adam)", leave=False):
            optimizer.zero_grad()
            l_pde = loss_debris_pde(model_f, t_arr, physics, frag['mass'])
            l_ic  = loss_debris_ic(model_f, frag)
            loss  = CONFIG['w_pde'] * l_pde + CONFIG['w_ic'] * l_ic
            loss.backward()
            optimizer.step()
            scheduler.step()

            history['total'].append(loss.item())
            history['pde'].append(l_pde.item())
            history['ic'].append(l_ic.item())
            history['epochs'].append(epoch)

            if loss.item() < best_loss or (epoch % save_every == 0 and epoch > start_epoch):
                best_loss = min(loss.item(), best_loss)
                torch.save({'epoch': epoch, 'model_state': model_f.state_dict(),
                            'frag_info': frag}, ckpt_path)

        # ── Stage 2: L-BFGS Polishing ─────────────────────
        if lbfgs_steps > 0:
            lbfgs = torch.optim.LBFGS(
                model_f.parameters(), lr=1.0, max_iter=lbfgs_steps,
                history_size=50, line_search_fn='strong_wolfe'
            )
            step_counter = [start_epoch + epochs]

            def closure():
                lbfgs.zero_grad()
                l_pde = loss_debris_pde(model_f, t_arr, physics, frag['mass'])
                l_ic  = loss_debris_ic(model_f, frag)
                loss  = CONFIG['w_pde'] * l_pde + CONFIG['w_ic'] * l_ic
                loss.backward()

                history['total'].append(loss.item())
                history['pde'].append(l_pde.item())
                history['ic'].append(l_ic.item())
                history['epochs'].append(step_counter[0])
                step_counter[0] += 1
                return loss

            lbfgs.step(closure)

        # ── Stage 3: Exact ODE Validation (solve_ivp) ─────
        val = validate_debris_with_ode(model_f, frag, physics)
        frag['ode_rel_l2_%'] = val['rel_l2_err_%']
        frag['ode_max_err']  = val['max_abs_err']
        frag['ode_traj']     = (val['ode_x'], val['ode_y'], val['ode_z'])

        torch.save({
            'epoch': history['epochs'][-1],
            'model_state': model_f.state_dict(),
            'frag_info': {k: v for k, v in frag.items() if k != 'ode_traj'},
        }, ckpt_path)

        with open(log_path, 'w') as f:
            json.dump(history, f)

        print(f"  ✅ Frag {fid + 1} | Final Loss: {history['total'][-1]:.2e} | "
              f"vs solve_ivp Rel L2 Error: {val['rel_l2_err_%']:.4f}% "
              f"(Max |dr|: {val['max_abs_err']:.2e})")

        all_models.append(model_f)
        all_histories.append(history)

    print(f"\n✅ {len(fragment_list)} fragment trajectories trained & ODE-validated")
    return all_models, all_histories, fragment_list