"""Debug helper: run one sample row from a sensitivity design and print exceptions.

Usage:
    python debug_row.py --method sobol --row 600 --reps 5 --base-seed 1000

This imports the `run_simulation` helper from `sensitivity.py` so it uses the
same model wiring and defaults.
"""
import argparse
import traceback
from pathlib import Path

import numpy as np

from sensitivity import run_simulation, _row_to_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="sobol")
    p.add_argument("--row", type=int, required=True)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--width", type=int, default=50)
    p.add_argument("--height", type=int, default=50)
    p.add_argument("--n-agents", type=int, default=40)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--base-seed", type=int, default=1000)
    args = p.parse_args()

    samples_path = Path("results") / args.method / "samples.npy"
    if not samples_path.exists():
        raise SystemExit(f"Missing samples file: {samples_path}. Run sensitivity.py sample first.")

    samples = np.load(samples_path)
    if args.row < 0 or args.row >= len(samples):
        raise SystemExit(f"Row {args.row} out of range (0..{len(samples)-1})")

    row = samples[args.row]
    params = _row_to_params(row)
    print(f"Running row {args.row} params={params}")

    for rep in range(args.reps):
        seed = args.base_seed + args.row * args.reps + rep
        try:
            print(f"Replicate {rep} seed={seed}: starting")
            metrics = run_simulation(
                params,
                width=args.width,
                height=args.height,
                n_agents=args.n_agents,
                steps=args.steps,
                seed=seed,
            )
            print(f"Replicate {rep} succeeded: {metrics}")
        except Exception as exc:
            print(f"Replicate {rep} seed={seed} FAILED: {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
