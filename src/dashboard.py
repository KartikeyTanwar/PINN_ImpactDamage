# ============================================================
# RESULTS DASHBOARD (7 Plots + Symmetry Metrics + ODE Comparison)
# ============================================================

from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from .config import CONFIG, device, IN_NOTEBOOK
from .damage_pinn import von_mises_stress


def _finish_plot(fig, path):
    plt.savefig(path, dpi=150, bbox_inches='tight')
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close(fig)
    print(f"✅ Saved: {path}")


def eval_on_grid(model, t_val, N=40, mode='elastic'):
    x_1d = torch.linspace(0, 1, N)
    y_1d = torch.linspace(0, 1, N)
    Xg, Yg = torch.meshgrid(x_1d, y_1d, indexing='ij')

    x_f = Xg.reshape(-1, 1).to(device)
    y_f = Yg.reshape(-1, 1).to(device)
    z_f = torch.ones_like(x_f)
    t_f = torch.full_like(x_f, t_val)

    with torch.no_grad():
        if mode == 'elastic':
            u, v, w = model(x_f, y_f, z_f, t_f)
            disp_mag = torch.sqrt(u**2 + v**2 + w**2)
            return (Xg.numpy(), Yg.numpy(),
                    u.reshape(N, N).cpu().numpy(),
                    v.reshape(N, N).cpu().numpy(),
                    w.reshape(N, N).cpu().numpy(),
                    disp_mag.reshape(N, N).cpu().numpy())
        elif mode == 'damage':
            D = model(x_f, y_f, z_f, t_f)
            return Xg.numpy(), Yg.numpy(), D.reshape(N, N).cpu().numpy()


def eval_vm_stress(model_e, t_val, physics, N=30):
    x_1d = torch.linspace(0.05, 0.95, N)
    y_1d = torch.linspace(0.05, 0.95, N)
    Xg, Yg = torch.meshgrid(x_1d, y_1d, indexing='ij')

    x_f = Xg.reshape(-1, 1).to(device).requires_grad_(True)
    y_f = Yg.reshape(-1, 1).to(device).requires_grad_(True)
    z_f = torch.ones_like(x_f).requires_grad_(True)
    t_f = torch.full_like(x_f, t_val).requires_grad_(True)

    vm = von_mises_stress(model_e, x_f, y_f, z_f, t_f)
    return Xg.numpy(), Yg.numpy(), vm.reshape(N, N).detach().cpu().numpy()


def compute_asymmetry_error(model_e, model_d, t_val=0.3, N=40):
    """
    Quantifies emergent physical symmetry on the full domain (center impact):
      w, D even in x and y | u odd in x, even in y | v even in x, odd in y.
    """
    model_e.eval()
    model_d.eval()
    _, _, u, v, w, _ = eval_on_grid(model_e, t_val, N=N, mode='elastic')
    _, _, D          = eval_on_grid(model_d, t_val, N=N, mode='damage')

    def rel_err(a, b):
        return 100.0 * float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))

    metrics = {
        'u_x_antisym_%': rel_err(u, -u[::-1, :]),
        'u_y_sym_%'    : rel_err(u,  u[:, ::-1]),
        'v_x_sym_%'    : rel_err(v,  v[::-1, :]),
        'v_y_antisym_%': rel_err(v, -v[:, ::-1]),
        'w_x_sym_%'    : rel_err(w,  w[::-1, :]),
        'w_y_sym_%'    : rel_err(w,  w[:, ::-1]),
        'D_x_sym_%'    : rel_err(D,  D[::-1, :]),
        'D_y_sym_%'    : rel_err(D,  D[:, ::-1]),
    }
    print(f"\n📐 Emergent Symmetry Errors at t={t_val} (0% = perfectly symmetric):")
    for k, val in metrics.items():
        print(f"   {k:<16}: {val:.2f}%")
    return metrics


def plot_loss_curves(hist_e, hist_d, hist_r, run_tag, physics):
    num_plots = 2 + (1 if hist_r else 0)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    fig.suptitle('Training Loss History — All PINNs (Adam + L-BFGS)',
                 fontsize=14, fontweight='bold')
    if num_plots == 1:
        axes = [axes]

    configs = [(hist_e, 'Elasticity PINN'), (hist_d, 'Damage PINN')]
    if hist_r:
        configs.append((hist_r[0], 'Debris PINN (Fragment 1)'))

    for i, (hist, title) in enumerate(configs):
        ax = axes[i]
        ax.semilogy(hist['total'], label='Total', linewidth=2, color='black')
        ax.semilogy(hist['pde'],   label='PDE',   linewidth=1.5, color='royalblue')
        if 'bc' in hist:
            ax.semilogy(hist['bc'], label='BC', linewidth=1.5, color='tomato')
        if 'ic' in hist:
            ax.semilogy(hist['ic'], label='IC', linewidth=1.5, color='seagreen')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Iteration (Adam + L-BFGS)')
        ax.set_ylabel('Loss (log scale)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _finish_plot(fig, f"{CONFIG['res_dashboard']}/loss_curves_{run_tag}.png")


def plot_displacement(model_e, run_tag, physics):
    t_vals = [0.1, 0.3, 0.7]
    labels = ['t = 0.1 (early)', 't = 0.3 (peak)', 't = 0.7 (late)']
    comp_labels = ['u (X-displacement)', 'v (Y-displacement)', 'w (Z-displacement)']
    cmaps = ['RdBu_r', 'RdBu_r', 'plasma']

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle('Displacement Field — Top Face (z=1)', fontsize=14, fontweight='bold')
    model_e.eval()

    for col, (tv, tl) in enumerate(zip(t_vals, labels)):
        Xg, Yg, u, v, w, _ = eval_on_grid(model_e, tv, N=40, mode='elastic')
        for row, fld in enumerate([u, v, w]):
            ax = axes[row, col]
            im = ax.contourf(Xg, Yg, fld, levels=40, cmap=cmaps[row])
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.plot(physics['x0'], physics['y0'], 'w*', markersize=12)
            ax.set_xlabel('x')
            ax.set_ylabel(comp_labels[row] + '\ny' if col == 0 else 'y', fontsize=9)
            if row == 0:
                ax.set_title(tl, fontsize=10)

    plt.tight_layout()
    _finish_plot(fig, f"{CONFIG['res_stress']}/displacement_{run_tag}.png")


def plot_von_mises(model_e, run_tag, physics):
    t_vals = [0.1, 0.3, 0.7]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Von Mises Stress — Top Face', fontsize=14, fontweight='bold')
    sigma_y = physics['sigma_y_norm']
    model_e.eval()

    for ax, tv in zip(axes, t_vals):
        Xg, Yg, vm = eval_vm_stress(model_e, tv, physics, N=30)
        im = ax.contourf(Xg, Yg, vm, levels=40, cmap='hot_r')
        plt.colorbar(im, ax=ax, fraction=0.046, label='sigma_vm (norm)')
        if vm.min() < sigma_y < vm.max():
            ax.contour(Xg, Yg, vm, levels=[sigma_y], colors='cyan', linewidths=2, linestyles='--')
        ax.plot(physics['x0'], physics['y0'], 'b*', markersize=14)
        ax.set_title(f't = {tv}', fontsize=11)
        ax.set_xlabel('x'); ax.set_ylabel('y')

    cyan_line = mpatches.Patch(color='cyan', label=f'Yield stress sigma_y={sigma_y:.4f}')
    axes[0].legend(handles=[cyan_line], fontsize=9, loc='upper left')
    plt.tight_layout()
    _finish_plot(fig, f"{CONFIG['res_stress']}/von_mises_{run_tag}.png")


def plot_damage(model_d, run_tag, physics):
    t_vals = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Damage Variable D(x,y,t) — Top Face\nD=0 (intact) -> D=1 (fractured)',
                 fontsize=13, fontweight='bold')
    axes = axes.ravel()
    threshold = CONFIG['damage_thresh']
    model_d.eval()

    for ax, tv in zip(axes, t_vals):
        Xg, Yg, D = eval_on_grid(model_d, tv, N=40, mode='damage')
        im = ax.contourf(Xg, Yg, D, levels=np.linspace(0, 1, 41),
                         cmap='RdYlBu_r', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046)
        if D.min() < threshold < D.max():
            ax.contour(Xg, Yg, D, levels=[threshold], colors='black', linewidths=2)
        ax.plot(physics['x0'], physics['y0'], 'k*', markersize=12)
        dmg_pct = 100.0 * np.mean(D > threshold)
        ax.set_title(f"t = {tv}\nDamaged area: {dmg_pct:.1f}%", fontsize=10)
        ax.set_xlabel('x'); ax.set_ylabel('y')

    plt.tight_layout()
    _finish_plot(fig, f"{CONFIG['res_damage']}/damage_evolution_{run_tag}.png")


def plot_debris_3d(debris_models, fragment_list, run_tag, physics):
    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_subplot(111, projection='3d')
    colors = ['red', 'blue', 'green', 'orange', 'purple',
              'brown', 'pink', 'gray', 'cyan', 'magenta']
    t_range = torch.linspace(0, 1, 200, device=device).reshape(-1, 1)

    Xg_p, Yg_p = np.meshgrid(np.linspace(-0.5, 1.5, 10), np.linspace(-0.5, 1.5, 10))
    ax.plot_surface(Xg_p, Yg_p, np.zeros_like(Xg_p), color='lightgray', alpha=0.4)

    landing_positions = []
    for i, (model_f, frag) in enumerate(zip(debris_models, fragment_list)):
        model_f.eval()
        with torch.no_grad():
            x_t, y_t, z_t = model_f(t_range)
        x_np, y_np, z_np = x_t.cpu().numpy().ravel(), y_t.cpu().numpy().ravel(), z_t.cpu().numpy().ravel()
        c = colors[i % len(colors)]

        err_str = f" | ODE err={frag['ode_rel_l2_%']:.3f}%" if 'ode_rel_l2_%' in frag else ""
        ax.plot(x_np, y_np, z_np, color=c, linewidth=2,
                label=f'Frag {i+1} (D={frag["D0"]:.2f}{err_str})')

        # Overlay exact solve_ivp trajectory as dashed black reference
        if 'ode_traj' in frag:
            ox, oy, oz = frag['ode_traj']
            ax.plot(ox, oy, oz, color='black', linestyle='--', linewidth=1.0, alpha=0.6,
                    label='Exact ODE (solve_ivp)' if i == 0 else None)

        ax.scatter(frag['x0'], frag['y0'], frag['z0'], color=c, s=100, marker='^', zorder=5)
        land_idx = np.where(z_np <= 0.0)[0][0] if np.any(z_np <= 0.0) else -1
        ax.scatter(x_np[land_idx], y_np[land_idx], 0, color=c, s=150, marker='x', linewidths=3)

        landing_positions.append({
            'fragment'    : i + 1,
            'land_x'      : float(x_np[land_idx]),
            'land_y'      : float(y_np[land_idx]),
            'max_z'       : float(z_np.max()),
            'D0'          : frag['D0'],
            'ode_rel_l2_%': frag.get('ode_rel_l2_%', float('nan')),
        })

    plate_x, plate_y = [0, 1, 1, 0, 0], [0, 0, 1, 1, 0]
    pb = physics['plate_height_m'] / physics['L']
    ax.plot(plate_x, plate_y, [pb] * 5, 'k-', linewidth=2, label='Plate bottom')
    ax.plot(plate_x, plate_y, [physics['z_launch_norm']] * 5, 'k--', linewidth=1, alpha=0.4)
    ax.scatter(physics['x0'], physics['y0'], physics['z_launch_norm'],
               color='black', s=300, marker='*', label='Impact point', zorder=10)

    ax.set_xlabel('X (norm)'); ax.set_ylabel('Y (norm)'); ax.set_zlabel('Z (norm)')
    ax.set_title('3D Debris Trajectories (Solid = PINN, Dashed = Exact ODE)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_zlim(bottom=0.0)

    plt.tight_layout()
    _finish_plot(fig, f"{CONFIG['res_debris']}/trajectories_3d_{run_tag}.png")
    return landing_positions


def plot_landing_map(debris_models, fragment_list, run_tag, physics):
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    colors = ['red', 'blue', 'green', 'orange', 'purple',
              'brown', 'pink', 'gray', 'cyan', 'magenta']
    t_range = torch.linspace(0, 1, 200, device=device).reshape(-1, 1)

    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black',
                               linewidth=3, label='Plate boundary'))

    for i, (model_f, frag) in enumerate(zip(debris_models, fragment_list)):
        model_f.eval()
        with torch.no_grad():
            x_t, y_t, z_t = model_f(t_range)
        x_np, y_np, z_np = x_t.cpu().numpy().ravel(), y_t.cpu().numpy().ravel(), z_t.cpu().numpy().ravel()
        c = colors[i % len(colors)]
        land_idx = np.where(z_np <= 0.0)[0][0] if np.any(z_np <= 0.0) else -1

        ax.plot(x_np, y_np, color=c, linewidth=1.5, alpha=0.6, linestyle='--')
        ax.scatter(frag['x0'], frag['y0'], color=c, s=120, marker='^', zorder=5)
        delta = np.sqrt((x_np[land_idx] - frag['x0'])**2 + (y_np[land_idx] - frag['y0'])**2)
        ax.scatter(x_np[land_idx], y_np[land_idx], color=c, s=200, marker='x',
                   linewidths=3, zorder=5, label=f'Frag {i+1} (delta={delta:.2f})')

    ax.scatter(physics['x0'], physics['y0'], color='black', s=400, marker='*',
               zorder=10, label='Impact point')
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('X (normalized)'); ax.set_ylabel('Y (normalized)')
    ax.set_title('Fragment Landing Map — Top View', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    _finish_plot(fig, f"{CONFIG['res_debris']}/landing_map_{run_tag}.png")


def plot_summary_table(physics, landing_positions, hist_e, hist_d, asym_metrics, run_tag):
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.axis('off')
    fig.suptitle('Analysis Summary Report', fontsize=16, fontweight='bold', y=0.98)

    input_data = [
        ['Parameter', 'Value', 'Unit'],
        ['Projectile mass',       f"{physics.get('mass', 'N/A')}",              'kg'],
        ['Projectile velocity',   f"{physics.get('velocity', 'N/A')}",          'm/s'],
        ['Kinetic energy',        f"{physics['KE']/1e3:.2f}",                   'kJ'],
        ['Impact force',          f"{physics['F_impact']/1e6:.4f}",             'MN'],
        ['Impact pressure',       f"{physics['impact_pressure']/1e6:.2f}",      'MPa'],
        ['Plate dimensions',      f"{physics['L']}x{physics['W']}x{physics['H']}", 'm'],
        ['Plate height (bottom)', f"{physics['plate_height_m']:.3f}",           'm'],
        ['Impact location',       f"({physics['x0']:.2f}, {physics['y0']:.2f})", 'norm'],
        ['P-wave speed',          f"{physics['c_p']:.1f}",                      'm/s'],
        ['Wave cross time',       f"{physics['t_cross']*1e6:.2f}",              'us'],
    ]

    results_data = [
        ['Metric', 'Value'],
        ['Elasticity PINN final loss', f"{hist_e['total'][-1]:.6f}"],
        ['Damage PINN final loss',     f"{hist_d['total'][-1]:.6f}"],
        ['w X/Y Symmetry Error',       f"{asym_metrics['w_x_sym_%']:.2f}% / {asym_metrics['w_y_sym_%']:.2f}%"],
        ['D X/Y Symmetry Error',       f"{asym_metrics['D_x_sym_%']:.2f}% / {asym_metrics['D_y_sym_%']:.2f}%"],
        ['Fragments tracked',          str(len(landing_positions))],
    ]
    for lp in landing_positions:
        results_data.append([
            f"Frag {lp['fragment']} Landing / ODE Err",
            f"({lp['land_x']:.2f},{lp['land_y']:.2f}) | {lp['ode_rel_l2_%']:.3f}%"
        ])

    for data, bbox in [(input_data, [0.0, 0.30, 0.47, 0.62]),
                       (results_data, [0.51, 0.30, 0.49, 0.62])]:
        tbl = ax.table(cellText=data[1:], colLabels=data[0],
                       cellLoc='center', loc='upper left', bbox=bbox)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        for (r, _), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor('#2c3e50')
                cell.set_text_props(color='white', fontweight='bold')
            elif r % 2 == 0:
                cell.set_facecolor('#ecf0f1')

    ax.text(0.5, 0.22,
            f"Run tag: {run_tag}  |  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            transform=ax.transAxes, ha='center', fontsize=10, color='gray')

    plt.tight_layout()
    _finish_plot(fig, f"{CONFIG['res_dashboard']}/summary_{run_tag}.png")


def generate_dashboard(model_e, model_d, debris_models, fragment_list,
                       hist_e, hist_d, hist_r, physics, run_tag):
    print("\n" + "=" * 55)
    print("  RESULTS DASHBOARD — GENERATING")
    print("=" * 55)

    asym_metrics = compute_asymmetry_error(model_e, model_d, t_val=0.3)

    plot_loss_curves(hist_e, hist_d, hist_r, run_tag, physics)
    plot_displacement(model_e, run_tag, physics)
    plot_von_mises(model_e, run_tag, physics)
    plot_damage(model_d, run_tag, physics)

    if debris_models and fragment_list:
        landing = plot_debris_3d(debris_models, fragment_list, run_tag, physics)
        plot_landing_map(debris_models, fragment_list, run_tag, physics)
    else:
        print("⚠️  No fragments formed — skipping debris plots.")
        landing = []

    plot_summary_table(physics, landing, hist_e, hist_d, asym_metrics, run_tag)
    print(f"\n✅ Dashboard complete -> {CONFIG['res_dashboard']}")
    return asym_metrics, landing