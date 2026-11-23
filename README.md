# PINN Impact Damage & Debris Tracker

A Physics-Informed Neural Network (PINN) framework for predicting high-velocity projectile impact damage on structural plates, elastoplastic stress distributions, and trajectory tracking of ejected debris fragments.

---

## 📌 Overview

This framework implements a 3-stage physics-informed deep learning pipeline:

1. **Stage 1 — Elasticity PINN (`ElasticityPINN`)**:
   - Solves 2D elastostatic Navier-Cauchy equilibrium equations with dynamic impact pressure distribution:
     $$\nabla \cdot \boldsymbol{\sigma} + \mathbf{f} = \mathbf{0}$$
   - Infers displacement fields $(u, v)$ and stress tensor components $(\sigma_{xx}, \sigma_{yy}, \tau_{xy})$.
   - Computes Von Mises yield criteria and hydrostatic stress fields.

2. **Stage 2 — Damage PINN (`DamagePINN`)**:
   - Predicts anisotropic damage field $D(x, y) \in [0, 1]$ based on Continuum Damage Mechanics (CDM) and Johnson-Cook failure criterion.
   - Computes crater dimensions, crack path propagation, and localized damage accumulation.

3. **Stage 3 — Debris Tracking PINN (`DebrisPINN`)**:
   - Simulates post-impact fragment ejection dynamics:
     $$m \frac{d\mathbf{v}}{dt} = m\mathbf{g} - \frac{1}{2}\rho C_d A \|\mathbf{v}\|\mathbf{v}$$
   - Validated against classical Runge-Kutta numerical ODE integrators.
   - Computes fragment dispersion footprint, kinetic energy distribution, and hazard exclusion zones.

---

## 🚀 Key Features

- **Multi-Environment Support**: Seamless execution on Local Workstations, Kaggle (GPU), and Google Colab.
- **Physics-Informed Loss**: Automatic differentiation via PyTorch for residual stress equilibrium, boundary conditions, and Newtonian kinematics.
- **Automated Kaggle Bundler**: Single-click script `build_kaggle_notebook.py` to bundle all modules into a standalone Kaggle kernel.
- **Comprehensive Visualization**: Generates stress heatmaps, damage contour maps, debris ballistic trajectories, and executive summary dashboards.

---

## 🛠️ Project Structure

```text
PINN_ImpactDamage/
├── .gitignore
├── requirements.txt
├── run.py                       # CLI execution entry point
├── build_kaggle_notebook.py     # Bundler for Kaggle/Colab notebooks
├── notebooks/
│   └── kaggle_run/
│       ├── kaggle_train.ipynb   # Standalone notebook for cloud training
│       └── kernel-metadata.json
└── src/
    ├── __init__.py
    ├── config.py                # Environment auto-detection & hyperparameters
    ├── physics.py               # Analytical impact mechanics & material properties
    ├── elasticity_pinn.py       # Stage 1: Navier-Cauchy elastostatic solver
    ├── damage_pinn.py           # Stage 2: Continuum damage mechanics solver
    ├── debris_pinn.py           # Stage 3: Ballistic debris trajectory tracker
    ├── dashboard.py             # Plotting & visualization engine
    └── pipeline.py              # End-to-end execution & checkpoint loader
```

---

## ⚡ Getting Started

### 1. Prerequisites & Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/KartikeyTanwar/PINN_ImpactDamage.git
cd PINN_ImpactDamage
pip install -r requirements.txt
```

### 2. Local Training

Run quick verification with few epochs:
```bash
python run.py --velocity 200 --epochs 50 --n_col 1000 --lbfgs 20
```

Run full high-fidelity simulation:
```bash
python run.py --velocity 200 --epochs 2000 --n_col 10000 --lbfgs 300
```

### 3. CLI Parameters

| Flag | Default | Description |
|---|---|---|
| `--mass` | `1000.0` | Projectile mass (kg) |
| `--velocity` | `200.0` | Impact velocity (m/s) |
| `--diameter` | `0.3` | Projectile diameter (m) |
| `--angle` | `90.0` | Impact angle (degrees) |
| `--length` | `2.0` | Target plate length (m) |
| `--width` | `2.0` | Target plate width (m) |
| `--thickness` | `0.1` | Target plate thickness (m) |
| `--epochs` | `2000` | Adam optimization epochs |
| `--lbfgs` | `300` | L-BFGS second-order refinement steps |

---

## 📊 Outputs & Artifacts

All training runs generate structured outputs saved under `./outputs/`:
- `01_checkpoints/`: PyTorch model weights for elasticity, damage, and debris models.
- `02_results/`: High-resolution figures (stress fields, damage contours, trajectory dispersion).
- `03_logs/`: Convergence loss logs in JSON format.

---

## 👤 Author

- **Kartikey Tanwar** ([@KartikeyTanwar](https://github.com/KartikeyTanwar))
