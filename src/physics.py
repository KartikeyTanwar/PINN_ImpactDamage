# ============================================================
# PHYSICS PARAMETER GENERATOR & LATIN HYPERCUBE SAMPLER
# ============================================================

import json
import numpy as np
from scipy.stats import qmc
from .config import CONFIG


def lhs_sample(n_samples, n_dims, seed=None):
    """Generate Latin Hypercube samples in [0, 1]^n_dims."""
    sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
    return sampler.random(n=n_samples)


def compute_physics_params(params):
    """Convert raw user inputs into normalized physics quantities."""
    m              = params['mass']
    v              = params['velocity']
    d              = params['diameter']
    ang            = params['angle_deg']
    L              = params['length']
    W              = params['width']
    H              = params['thickness']
    x0             = params['impact_x']
    y0             = params['impact_y']
    plate_height_m = params['plate_height_m']

    print("=" * 55)
    print("  PHYSICS PARAMETER GENERATOR")
    print("=" * 55)
    print(f"\n📥 Raw Inputs:")
    print(f"   Projectile  : mass={m} kg | velocity={v} m/s | d={d} m | angle={ang}°")
    print(f"   Plate       : {L}m x {W}m x {H}m | height={plate_height_m} m")
    print(f"   Impact point: ({x0}, {y0}) normalized")

    angle_rad  = np.radians(ang)
    t_contact  = d / v
    F_impact   = (m * v * np.sin(angle_rad)) / t_contact
    F_impact_n = F_impact / (CONFIG['E'] * L * W)

    KE              = 0.5 * m * v**2
    contact_area    = np.pi * (d / 2.0)**2
    impact_pressure = F_impact / contact_area
    pressure_norm   = impact_pressure / CONFIG['E']

    r_x   = (d / 2.0) / L
    r_y   = (d / 2.0) / W
    r_avg = max((r_x + r_y) / 2.0, 0.02)

    E, nu, rho = CONFIG['E'], CONFIG['nu'], CONFIG['rho']
    c_p      = np.sqrt(E * (1.0 - nu) / (rho * (1.0 + nu) * (1.0 - 2.0 * nu)))
    c_s      = np.sqrt(E / (2.0 * rho * (1.0 + nu)))
    t_cross  = L / c_p
    t_peak_n = min(t_contact / t_cross, 0.15)

    z_launch_abs  = plate_height_m + H
    z_launch_norm = z_launch_abs / L

    boundary_conditions = {
        'x_min': 'clamped', 'x_max': 'clamped',
        'y_min': 'clamped', 'y_max': 'clamped',
        'z_min': 'free',    'z_top': 'free',
    }

    sigma_y_norm  = CONFIG['sigma_y'] / E
    debris_v_init = v * 0.15
    debris_v_norm = debris_v_init / (L / t_cross)

    print(f"\n⚡ Impact & Wave Scaling:")
    print(f"   Contact time    : {t_contact*1e6:.3f} us | F_impact: {F_impact/1e6:.4f} MN")
    print(f"   Kinetic energy  : {KE/1e3:.2f} kJ | Pressure: {impact_pressure/1e6:.2f} MPa")
    print(f"   c_p={c_p:.1f} m/s | c_s={c_s:.1f} m/s | t_cross={t_cross*1e6:.2f} us")
    print(f"   Debris launch Z : {z_launch_abs:.3f} m ({z_launch_norm:.3f} norm)")

    physics = {
        'mass'            : float(m),
        'velocity'        : float(v),
        'diameter'        : float(d),
        'angle_deg'       : float(ang),
        'L'               : float(L),
        'W'               : float(W),
        'H'               : float(H),
        'x0'              : float(x0),
        'y0'              : float(y0),
        'z0'              : 1.0,
        'z_launch_abs'    : float(z_launch_abs),
        'z_launch_norm'   : float(z_launch_norm),
        'plate_height_m'  : float(plate_height_m),
        'F_impact'        : float(F_impact),
        'F_norm'          : float(F_impact_n),
        'impact_pressure' : float(impact_pressure),
        'pressure_norm'   : float(pressure_norm),
        'KE'              : float(KE),
        'r_x'             : float(r_x),
        'r_y'             : float(r_y),
        'r_avg'           : float(r_avg),
        'c_p'             : float(c_p),
        'c_s'             : float(c_s),
        't_cross'         : float(t_cross),
        't_contact'       : float(t_contact),
        't_peak_n'        : float(t_peak_n),
        't_sigma'         : 0.02,
        'lam_n'           : float((E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu)) / E),
        'mu_n'            : float(E / (2.0 * (1.0 + nu)) / E),
        'rho_n'           : 1.0,
        'sigma_y_norm'    : float(sigma_y_norm),
        'BC'              : boundary_conditions,
        'debris_v_norm'   : float(debris_v_norm),
        'n_fragments'     : CONFIG['n_fragments'],
        'epochs'          : int(params['epochs']),
        'n_col'           : int(params['n_col']),
        'lbfgs_steps'     : int(params.get('lbfgs_steps', CONFIG['lbfgs_steps'])),
        'resample_every'  : int(params.get('resample_every', CONFIG['resample_every'])),
        'resume'          : bool(params.get('resume', False)),
    }

    ts = params['timestamp']
    fname = f"{CONFIG['configs']}/physics_{ts}.json"
    with open(fname, 'w') as f:
        json.dump(physics, f, indent=2)

    print(f"✅ Physics params saved -> {fname}\n")
    return physics