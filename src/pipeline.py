# ============================================================
# MASTER PIPELINE (Run, Load, and List Saved Runs)
# ============================================================

import os
import json
import time
from .config import CONFIG, device, safe_torch_load
from .physics import compute_physics_params
from .elasticity_pinn import ElasticityPINN, train_elasticity_pinn
from .damage_pinn import DamagePINN, train_damage_pinn
from .debris_pinn import DebrisPINN, train_debris_pinn, validate_debris_with_ode
from .dashboard import generate_dashboard


def run_full_pipeline(params, status_bar=None):
    run_tag = params['timestamp']
    start   = time.time()

    print("\n" + "=" * 55)
    print("  PINN IMPACT DAMAGE ANALYZER — STARTING")
    print(f"  v={params['velocity']} m/s | m={params['mass']} kg | "
          f"epochs={params['epochs']} | N_col={params['n_col']}")
    print("=" * 55)

    physics = compute_physics_params(params)

    model_e, hist_e = train_elasticity_pinn(physics, run_tag=run_tag)
    model_d, hist_d = train_damage_pinn(model_e, physics, run_tag=run_tag)
    debris_models, hist_r, fragment_list = train_debris_pinn(model_d, physics, run_tag=run_tag)

    asym_metrics, landing = generate_dashboard(
        model_e, model_d, debris_models, fragment_list,
        hist_e, hist_d, hist_r, physics, run_tag
    )

    total = time.time() - start
    h, m, s = int(total // 3600), int((total % 3600) // 60), int(total % 60)

    print("\n" + "=" * 55)
    print(f"  ✅ PIPELINE COMPLETE in {h:02d}h {m:02d}m {s:02d}s | Run: {run_tag}")
    print("=" * 55)

    if status_bar is not None:
        status_bar.value = (
            f"<p style='color:green;font-size:13px'>"
            f"✅ Pipeline complete — {h:02d}h {m:02d}m {s:02d}s | Run: {run_tag}</p>"
        )

    return {
        'model_e'       : model_e,
        'model_d'       : model_d,
        'debris_models' : debris_models,
        'fragment_list' : fragment_list,
        'hist_e'        : hist_e,
        'hist_d'        : hist_d,
        'hist_r'        : hist_r,
        'physics'       : physics,
        'asym_metrics'  : asym_metrics,
        'landing'       : landing,
        'run_tag'       : run_tag,
    }


def load_and_display(run_tag):
    print(f"\n📂 Loading saved run: {run_tag}")
    ckpt_e = f"{CONFIG['ckpt_elastic']}/elasticity_{run_tag}.pth"
    ckpt_d = f"{CONFIG['ckpt_damage']}/damage_{run_tag}.pth"
    log_e  = f"{CONFIG['logs']}/elasticity/loss_{run_tag}.json"
    log_d  = f"{CONFIG['logs']}/damage/loss_{run_tag}.json"

    for path in [ckpt_e, ckpt_d, log_e, log_d]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")

    ce = safe_torch_load(ckpt_e, map_location=device)
    model_e = ElasticityPINN(CONFIG['layers_elast']).to(device)
    model_e.load_state_dict(ce['model_state'])
    model_e.eval()
    physics = ce['physics']

    cd = safe_torch_load(ckpt_d, map_location=device)
    model_d = DamagePINN(CONFIG['layers_damage']).to(device)
    model_d.load_state_dict(cd['model_state'])
    model_d.eval()

    debris_models, fragment_list, hist_r = [], [], []
    for i in range(CONFIG['n_fragments']):
        ckpt_f = f"{CONFIG['ckpt_debris']}/debris_frag{i}_{run_tag}.pth"
        log_f  = f"{CONFIG['logs']}/debris/loss_frag{i}_{run_tag}.json"
        if os.path.exists(ckpt_f):
            cf = safe_torch_load(ckpt_f, map_location=device)
            frag = cf['frag_info']
            mf = DebrisPINN(CONFIG['layers_debris'], frag).to(device)
            mf.load_state_dict(cf['model_state'])
            mf.eval()
            val = validate_debris_with_ode(mf, frag, physics)
            frag['ode_rel_l2_%'] = val['rel_l2_err_%']
            frag['ode_max_err']  = val['max_abs_err']
            frag['ode_traj']     = (val['ode_x'], val['ode_y'], val['ode_z'])
            debris_models.append(mf)
            fragment_list.append(frag)
            if os.path.exists(log_f):
                with open(log_f) as f:
                    hist_r.append(json.load(f))

    with open(log_e) as f:
        hist_e = json.load(f)
    with open(log_d) as f:
        hist_d = json.load(f)

    generate_dashboard(
        model_e, model_d, debris_models, fragment_list,
        hist_e, hist_d, hist_r, physics, run_tag + '_loaded'
    )


def list_saved_runs():
    ckpt_dir = CONFIG['ckpt_elastic']
    files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith('.pth'))
    if not files:
        print("No saved runs found yet.")
        return []
    tags = [f.replace('elasticity_', '').replace('.pth', '') for f in files]
    for idx, tag in enumerate(tags, 1):
        print(f"  {idx}. {tag}")
    return tags