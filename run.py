# ============================================================
# LOCAL CLI RUNNER
# Example (quick local test):
#   python run.py --velocity 200 --epochs 50 --n_col 1000 --lbfgs 20
# Example (full training):
#   python run.py --velocity 200 --epochs 2000 --n_col 10000 --lbfgs 300
# ============================================================

import argparse
from datetime import datetime
from src.pipeline import run_full_pipeline


def main():
    parser = argparse.ArgumentParser(description="PINN Impact Damage & Debris Tracker")
    parser.add_argument('--mass',           type=float, default=1000.0)
    parser.add_argument('--velocity',       type=float, default=200.0)
    parser.add_argument('--diameter',       type=float, default=0.3)
    parser.add_argument('--angle',          type=float, default=90.0)
    parser.add_argument('--length',         type=float, default=2.0)
    parser.add_argument('--width',          type=float, default=2.0)
    parser.add_argument('--thickness',      type=float, default=0.1)
    parser.add_argument('--plate_height',   type=float, default=3.0)
    parser.add_argument('--impact_x',       type=float, default=0.5)
    parser.add_argument('--impact_y',       type=float, default=0.5)
    parser.add_argument('--epochs',         type=int,   default=2000)
    parser.add_argument('--n_col',          type=int,   default=10000)
    parser.add_argument('--lbfgs',          type=int,   default=300)
    parser.add_argument('--resample_every', type=int,   default=50)
    parser.add_argument('--resume',         action='store_true')
    args = parser.parse_args()

    params = {
        'mass'           : args.mass,
        'velocity'       : args.velocity,
        'diameter'       : args.diameter,
        'angle_deg'      : args.angle,
        'length'         : args.length,
        'width'          : args.width,
        'thickness'      : args.thickness,
        'plate_height_m' : args.plate_height,
        'impact_x'       : args.impact_x,
        'impact_y'       : args.impact_y,
        'epochs'         : args.epochs,
        'n_col'          : args.n_col,
        'lbfgs_steps'    : args.lbfgs,
        'resample_every' : args.resample_every,
        'resume'         : args.resume,
        'timestamp'      : datetime.now().strftime('%Y%m%d_%H%M%S'),
    }
    run_full_pipeline(params)


if __name__ == '__main__':
    main()