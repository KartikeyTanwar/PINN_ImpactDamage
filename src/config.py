# ============================================================
# GLOBAL CONFIG, ENVIRONMENT AUTO-DETECT & PATHS
# Works seamlessly on Local VS Code, Kaggle GPU, and Colab
# ============================================================

import os
import sys
import torch
import matplotlib

IN_COLAB    = 'google.colab' in sys.modules
IN_KAGGLE   = os.path.exists('/kaggle/working')
IN_NOTEBOOK = 'ipykernel' in sys.modules

# Use non-interactive backend when running headless scripts (prevents blocking)
if not IN_NOTEBOOK:
    matplotlib.use('Agg')

if IN_COLAB:
    from google.colab import drive, output
    drive.mount('/content/drive')
    output.enable_custom_widget_manager()
    PROJECT_ROOT = '/content/drive/MyDrive/PINN_ImpactDamage'
    ENV_NAME = 'Google Colab'
elif IN_KAGGLE:
    PROJECT_ROOT = '/kaggle/working/PINN_ImpactDamage'
    ENV_NAME = 'Kaggle'
else:
    PROJECT_ROOT = os.path.abspath('./outputs')
    ENV_NAME = 'Local Machine'

FOLDERS = [
    PROJECT_ROOT,
    f'{PROJECT_ROOT}/01_checkpoints/elasticity',
    f'{PROJECT_ROOT}/01_checkpoints/damage',
    f'{PROJECT_ROOT}/01_checkpoints/debris',
    f'{PROJECT_ROOT}/02_results/stress_maps',
    f'{PROJECT_ROOT}/02_results/damage_maps',
    f'{PROJECT_ROOT}/02_results/debris_plots',
    f'{PROJECT_ROOT}/02_results/dashboards',
    f'{PROJECT_ROOT}/03_logs/elasticity',
    f'{PROJECT_ROOT}/03_logs/damage',
    f'{PROJECT_ROOT}/03_logs/debris',
    f'{PROJECT_ROOT}/04_exports',
    f'{PROJECT_ROOT}/05_configs',
    f'{PROJECT_ROOT}/06_data',
]
for _folder in FOLDERS:
    os.makedirs(_folder, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
    GPU_NAME = torch.cuda.get_device_name(0)
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
    GPU_NAME = 'Apple Silicon MPS'
else:
    device = torch.device('cpu')
    GPU_NAME = 'CPU only'


def safe_torch_load(path, map_location=device):
    """Compatible with PyTorch <2.6 and >=2.6 (weights_only default change)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


CONFIG = {
    # ── Paths ──────────────────────────────────────────────
    'root'          : PROJECT_ROOT,
    'ckpt_elastic'  : f'{PROJECT_ROOT}/01_checkpoints/elasticity',
    'ckpt_damage'   : f'{PROJECT_ROOT}/01_checkpoints/damage',
    'ckpt_debris'   : f'{PROJECT_ROOT}/01_checkpoints/debris',
    'res_stress'    : f'{PROJECT_ROOT}/02_results/stress_maps',
    'res_damage'    : f'{PROJECT_ROOT}/02_results/damage_maps',
    'res_debris'    : f'{PROJECT_ROOT}/02_results/debris_plots',
    'res_dashboard' : f'{PROJECT_ROOT}/02_results/dashboards',
    'logs'          : f'{PROJECT_ROOT}/03_logs',
    'exports'       : f'{PROJECT_ROOT}/04_exports',
    'configs'       : f'{PROJECT_ROOT}/05_configs',

    # ── Material: Structural Steel ──────────────────────────
    'E'             : 200e9,    # Young's modulus (Pa)
    'nu'            : 0.30,     # Poisson's ratio
    'rho'           : 7850.0,   # Density (kg/m^3)
    'sigma_y'       : 250e6,    # Yield stress (Pa)
    'sigma_u'       : 400e6,    # Ultimate tensile strength (Pa)
    'G_c'           : 50e3,     # Fracture energy (J/m^2)
    'damage_thresh' : 0.85,     # D > threshold => fragment breaks off

    # ── PINN Training Defaults ─────────────────────────────
    'N_col'         : 5000,
    'N_bc'          : 300,
    'N_ic'          : 300,
    'epochs_elast'  : 2000,
    'epochs_damage' : 2000,
    'epochs_debris' : 1500,
    'lbfgs_steps'   : 300,      # L-BFGS polishing iterations after Adam
    'resample_every': 50,       # Resample interior LHS points every N Adam epochs
    'lr'            : 1e-3,
    'lr_decay_step' : 1000,
    'lr_decay_rate' : 0.5,

    # ── Network Architecture ────────────────────────────────
    'layers_elast'  : [4, 128, 128, 128, 128, 128, 3],
    'layers_damage' : [4, 128, 128, 128, 128, 1],
    'layers_debris' : [1, 64, 64, 64, 3],

    # ── Loss Weights ────────────────────────────────────────
    'w_pde'         : 1.0,
    'w_bc'          : 10.0,
    'w_ic'          : 10.0,

    # ── Simulation ──────────────────────────────────────────
    't_end'         : 1.0,
    'n_fragments'   : 5,
}

print(f"✅ Environment  : {ENV_NAME}")
print(f"✅ Device       : {device} ({GPU_NAME})")
print(f"✅ PyTorch      : {torch.__version__}")
print(f"✅ Output root  : {PROJECT_ROOT}")