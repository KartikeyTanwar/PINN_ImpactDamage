# ============================================================
# BUNDLES src/*.py -> notebooks/kaggle_run/kaggle_train.ipynb
# Includes an 18-case One-Factor-At-a-Time (OFAT) Parametric Study,
# automatic CSV export, and 6-panel comparative thesis plots.
# ============================================================

import json
from pathlib import Path

SRC_ORDER = [
    'src/config.py',
    'src/physics.py',
    'src/elasticity_pinn.py',
    'src/damage_pinn.py',
    'src/debris_pinn.py',
    'src/dashboard.py',
    'src/pipeline.py',
]

RUN_CELL = r"""
# ============================================================
# KAGGLE PARAMETRIC STUDY RUNNER (18 OFAT Cases)
# ============================================================
import csv
import gc
import time
import traceback
import numpy as np
import torch
import matplotlib.pyplot as plt

# Set QUICK_TEST = True for a fast 2-minute pipeline verification
QUICK_TEST = False

BASE_PARAMS = {
    'mass'           : 1000.0,
    'velocity'       : 200.0,
    'diameter'       : 0.3,
    'angle_deg'      : 90.0,
    'length'         : 2.0,
    'width'          : 2.0,
    'thickness'      : 0.1,
    'plate_height_m' : 3.0,
    'impact_x'       : 0.5,
    'impact_y'       : 0.5,
    'epochs'         : 30   if QUICK_TEST else 2000,
    'n_col'          : 500  if QUICK_TEST else 10000,
    'lbfgs_steps'    : 10   if QUICK_TEST else 300,
    'resample_every' : 50,
    'resume'         : False,
}

# One-Factor-At-a-Time (OFAT) Parametric Matrix
EXPERIMENTS = [
    # 0. Shared Baseline (v=200, m=1000, d=0.3, angle=90, L=W=2.0, center)
    {'sweep': 'baseline',  'label': '200 m/s (Base)', 'val': 200.0, 'tag': 'S0_baseline',       'overrides': {}},

    # 1. Velocity Sweep (all other parameters fixed at baseline)
    {'sweep': 'velocity',  'label': '10',             'val': 10.0,  'tag': 'S1_vel_010',        'overrides': {'velocity': 10.0}},
    {'sweep': 'velocity',  'label': '50',             'val': 50.0,  'tag': 'S1_vel_050',        'overrides': {'velocity': 50.0}},
    {'sweep': 'velocity',  'label': '85',             'val': 85.0,  'tag': 'S1_vel_085',        'overrides': {'velocity': 85.0}},
    {'sweep': 'velocity',  'label': '100',            'val': 100.0, 'tag': 'S1_vel_100',        'overrides': {'velocity': 100.0}},
    {'sweep': 'velocity',  'label': '150',            'val': 150.0, 'tag': 'S1_vel_150',        'overrides': {'velocity': 150.0}},

    # 2. Mass Sweep (velocity FIXED at 200 m/s)
    {'sweep': 'mass',      'label': '100',            'val': 100.0,  'tag': 'S2_mass_0100',     'overrides': {'mass': 100.0}},
    {'sweep': 'mass',      'label': '180',            'val': 180.0,  'tag': 'S2_mass_0180',     'overrides': {'mass': 180.0}},
    {'sweep': 'mass',      'label': '500',            'val': 500.0,  'tag': 'S2_mass_0500',     'overrides': {'mass': 500.0}},
    {'sweep': 'mass',      'label': '2000',           'val': 2000.0, 'tag': 'S2_mass_2000',     'overrides': {'mass': 2000.0}},

    # 3. Diameter Sweep (velocity FIXED at 200 m/s, mass FIXED at 1000 kg -> constant KE)
    {'sweep': 'diameter',  'label': '0.20',           'val': 0.20,  'tag': 'S3_diam_020',       'overrides': {'diameter': 0.20}},
    {'sweep': 'diameter',  'label': '0.50',           'val': 0.50,  'tag': 'S3_diam_050',       'overrides': {'diameter': 0.50}},
    {'sweep': 'diameter',  'label': '0.60',           'val': 0.60,  'tag': 'S3_diam_060',       'overrides': {'diameter': 0.60}},

    # 4. Impact Angle Sweep (velocity FIXED at 200 m/s)
    {'sweep': 'angle',     'label': '30',             'val': 30.0,  'tag': 'S4_ang_030',        'overrides': {'angle_deg': 30.0}},
    {'sweep': 'angle',     'label': '60',             'val': 60.0,  'tag': 'S4_ang_060',        'overrides': {'angle_deg': 60.0}},

    # 5. Impact Location Sweep (velocity FIXED at 200 m/s)
    {'sweep': 'location',  'label': '(0.25,0.50)',    'val': 1.0,   'tag': 'S5_loc_edge',       'overrides': {'impact_x': 0.25, 'impact_y': 0.50}},
    {'sweep': 'location',  'label': '(0.25,0.25)',    'val': 2.0,   'tag': 'S5_loc_corner',     'overrides': {'impact_x': 0.25, 'impact_y': 0.25}},

    # 6. Plate Span Sweep (velocity FIXED at 200 m/s)
    {'sweep': 'plate_span','label': '5x5 m',          'val': 5.0,   'tag': 'S6_span_5x5',       'overrides': {'length': 5.0, 'width': 5.0}},
]

csv_path = f"{CONFIG['exports']}/parametric_summary.csv"
CSV_FIELDS = [
    'tag', 'sweep', 'label', 'velocity', 'mass', 'diameter', 'angle_deg',
    'length', 'impact_x', 'impact_y', 'impact_pressure_MPa', 'KE_kJ',
    'regime', 'elast_loss', 'damage_loss', 'max_abs_w_t03', 'peak_vm_t03',
    'damaged_area_pct_t10', 'w_sym_err_pct', 'D_sym_err_pct',
    'n_fragments', 'mean_ode_err_pct', 'runtime_sec'
]

# Suppress inline rendering of ~126 individual plots during the loop (still saved as high-res PNGs)
_orig_in_notebook = IN_NOTEBOOK
IN_NOTEBOOK = False

records = []
with open(csv_path, 'w', newline='') as f:
    csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

batch_start = time.time()
print(f"🚀 Starting Parametric Study: {len(EXPERIMENTS)} cases")

for idx, exp in enumerate(EXPERIMENTS, 1):
    print("\n" + "#" * 65)
    print(f"  CASE {idx}/{len(EXPERIMENTS)} : {exp['tag']}  (sweep={exp['sweep']})")
    print("#" * 65)

    p = dict(BASE_PARAMS)
    p.update(exp['overrides'])
    p['timestamp'] = exp['tag']

    t0 = time.time()
    try:
        res = run_full_pipeline(p)
        phys = res['physics']

        # Extract quantitative physical response metrics on top face (z=1)
        res['model_e'].eval()
        res['model_d'].eval()
        _, _, _, _, w_t03, _ = eval_on_grid(res['model_e'], 0.3, N=40, mode='elastic')
        _, _, vm_t03         = eval_vm_stress(res['model_e'], 0.3, phys, N=30)
        _, _, D_t10          = eval_on_grid(res['model_d'], 1.0, N=40, mode='damage')

        p_mpa = phys['impact_pressure'] / 1e6
        if p_mpa < CONFIG['sigma_y'] / 1e6:
            regime = 'Elastic'
        elif p_mpa < CONFIG['sigma_u'] / 1e6:
            regime = 'Plastic'
        else:
            regime = 'Fracture'

        asym = res['asym_metrics']
        ode_errs = [lp['ode_rel_l2_%'] for lp in res['landing']
                    if not np.isnan(lp.get('ode_rel_l2_%', np.nan))]
        mean_ode = float(np.mean(ode_errs)) if ode_errs else 0.0

        row = {
            'tag'                 : exp['tag'],
            'sweep'               : exp['sweep'],
            'label'               : exp['label'],
            'velocity'            : p['velocity'],
            'mass'                : p['mass'],
            'diameter'            : p['diameter'],
            'angle_deg'           : p['angle_deg'],
            'length'              : p['length'],
            'impact_x'            : p['impact_x'],
            'impact_y'            : p['impact_y'],
            'impact_pressure_MPa' : round(p_mpa, 2),
            'KE_kJ'               : round(phys['KE'] / 1e3, 2),
            'regime'              : regime,
            'elast_loss'          : float(res['hist_e']['total'][-1]),
            'damage_loss'         : float(res['hist_d']['total'][-1]),
            'max_abs_w_t03'       : float(np.max(np.abs(w_t03))),
            'peak_vm_t03'         : float(np.max(vm_t03)),
            'damaged_area_pct_t10': float(100.0 * np.mean(D_t10 > CONFIG['damage_thresh'])),
            'w_sym_err_pct'       : float(0.5 * (asym['w_x_sym_%'] + asym['w_y_sym_%'])),
            'D_sym_err_pct'       : float(0.5 * (asym['D_x_sym_%'] + asym['D_y_sym_%'])),
            'n_fragments'         : len(res['fragment_list']),
            'mean_ode_err_pct'    : round(mean_ode, 4),
            'runtime_sec'         : round(time.time() - t0, 1),
        }
        records.append(row)

        with open(csv_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)

        print(f"📊 Recorded {exp['tag']}: P={row['impact_pressure_MPa']} MPa ({regime}) | "
              f"max|w|={row['max_abs_w_t03']:.5f} | D_area={row['damaged_area_pct_t10']:.1f}% | "
              f"w_sym={row['w_sym_err_pct']:.2f}% | Frags={row['n_fragments']}")

    except Exception as e:
        print(f"❌ Case {exp['tag']} failed: {e}")
        traceback.print_exc()

    # Free figures and GPU memory between runs
    plt.close('all')
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

IN_NOTEBOOK = _orig_in_notebook

total_min = (time.time() - batch_start) / 60.0
print("\n" + "=" * 65)
print(f"✅ PARAMETRIC STUDY COMPLETE: {len(records)}/{len(EXPERIMENTS)} runs in {total_min:.1f} min")
print(f"📄 Summary CSV saved -> {csv_path}")
print("=" * 65)

# ============================================================
# GENERATE 6-PANEL COMPARATIVE PARAMETRIC PLOT
# ============================================================
base_row = next((r for r in records if r['tag'] == 'S0_baseline'), None)

def get_sweep_rows(sweep_name, x_key):
    rows = [r for r in records if r['sweep'] == sweep_name]
    if base_row is not None:
        rows.append(base_row)
    return sorted(rows, key=lambda r: r[x_key])

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Parametric Sensitivity Analysis (OFAT around Baseline: v=200 m/s, m=1000 kg, d=0.3 m, 90 deg)',
             fontsize=13, fontweight='bold')

numeric_sweeps = [
    (axes[0, 0], 'velocity', 'velocity',  'Velocity (m/s)',       '1. Effect of Velocity'),
    (axes[0, 1], 'mass',     'mass',      'Projectile Mass (kg)', '2. Effect of Mass (v=200 m/s fixed)'),
    (axes[0, 2], 'diameter', 'diameter',  'Diameter (m)',         '3. Effect of Diameter (v=200, m=1000 fixed)'),
    (axes[1, 0], 'angle',    'angle_deg', 'Impact Angle (deg)',   '4. Effect of Angle (v=200 m/s fixed)'),
]

for ax, sname, xkey, xlabel, title in numeric_sweeps:
    rows = get_sweep_rows(sname, xkey)
    if not rows:
        continue
    xs   = [r[xkey] for r in rows]
    ws   = [r['max_abs_w_t03'] for r in rows]
    dmgs = [r['damaged_area_pct_t10'] for r in rows]

    l1 = ax.plot(xs, ws, 'o-', color='royalblue', linewidth=2, label='Max |w| (t=0.3)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Max |w| (norm)', color='royalblue')
    ax.tick_params(axis='y', labelcolor='royalblue')
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=11)

    ax2 = ax.twinx()
    l2 = ax2.plot(xs, dmgs, 's--', color='crimson', linewidth=2, label='Damaged Area % (t=1.0)')
    ax2.set_ylabel('Damaged Area (%)', color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')
    ax2.set_ylim(-5, 105)

    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines], loc='upper left', fontsize=8)

# Panel 5: Impact Location (categorical bar chart: symmetry error & peak stress)
ax_loc = axes[1, 1]
loc_rows = ([base_row] if base_row else []) + [r for r in records if r['sweep'] == 'location']
if loc_rows:
    loc_labels = ['(0.50,0.50)\nCenter' if r['tag'] == 'S0_baseline' else r['label'] for r in loc_rows]
    x_pos = np.arange(len(loc_rows))
    ax_loc.bar(x_pos - 0.18, [r['w_sym_err_pct'] for r in loc_rows], 0.35, label='w Symmetry Error (%)', color='teal')
    ax_loc.bar(x_pos + 0.18, [r['damaged_area_pct_t10'] for r in loc_rows], 0.35, label='Damaged Area (%)', color='coral')
    ax_loc.set_xticks(x_pos)
    ax_loc.set_xticklabels(loc_labels)
    ax_loc.set_ylabel('Percentage (%)')
    ax_loc.set_title('5. Effect of Impact Location (v=200 fixed)', fontsize=11)
    ax_loc.legend(fontsize=8)
    ax_loc.grid(True, axis='y', alpha=0.3)

# Panel 6: Plate Span (2x2 m vs 5x5 m)
ax_span = axes[1, 2]
span_rows = ([base_row] if base_row else []) + [r for r in records if r['sweep'] == 'plate_span']
if span_rows:
    span_labels = ['2x2 m' if r['tag'] == 'S0_baseline' else r['label'] for r in span_rows]
    x_pos = np.arange(len(span_rows))
    ax_span.bar(x_pos - 0.18, [r['peak_vm_t03'] * 1e3 for r in span_rows], 0.35, label='Peak sigma_vm (x1e-3)', color='darkorange')
    ax_span.bar(x_pos + 0.18, [r['w_sym_err_pct'] for r in span_rows], 0.35, label='w Symmetry Error (%)', color='slateblue')
    ax_span.set_xticks(x_pos)
    ax_span.set_xticklabels(span_labels)
    ax_span.set_ylabel('Value')
    ax_span.set_title('6. Effect of Plate Span (v=200 fixed)', fontsize=11)
    ax_span.legend(fontsize=8)
    ax_span.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
comp_path = f"{CONFIG['exports']}/parametric_comparison.png"
plt.savefig(comp_path, dpi=180, bbox_inches='tight')
plt.show()
print(f"✅ Comparative plot saved -> {comp_path}")
"""


def strip_relative_imports(code: str) -> str:
    clean_lines = []
    for line in code.splitlines():
        if line.lstrip().startswith('from .'):
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines) + "\n"


def make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.splitlines()],
    }


def main():
    out_dir = Path("notebooks/kaggle_run")
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = []
    for path in SRC_ORDER:
        raw = Path(path).read_text(encoding='utf-8')
        cells.append(make_code_cell(strip_relative_imports(raw)))

    cells.append(make_code_cell(RUN_CELL))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    nb_path = out_dir / "kaggle_train.ipynb"
    nb_path.write_text(json.dumps(nb, indent=2), encoding='utf-8')
    print(f"✅ Built 18-case parametric Kaggle notebook -> {nb_path}")


if __name__ == "__main__":
    main()