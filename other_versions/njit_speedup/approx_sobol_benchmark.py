"""Approximate Sobol-run benchmark.

Runs many `run_simulation` evaluations in parallel using ProcessPoolExecutor
until the target duration is reached (default ~150 seconds). Uses the same
model wiring as `sensitivity.py` so runtime is representative of a real run.

Usage:
    python approx_sobol_benchmark.py --seconds 150 --workers 8 --steps 100

Notes:
- Adjust `--workers` to match the number of CPU cores you want to stress.
- The script samples random parameter combinations uniformly from the
  `sensitivity.PROBLEM` bounds.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict

import numpy as np
from SALib.sample import sobol as sobol_sample

from sensitivity import run_simulation, PROBLEM


def sample_params() -> Dict[str, float]:
    params = {}
    for i, name in enumerate(PROBLEM["names"]):
        lo, hi = PROBLEM["bounds"][i]
        params[name] = float(np.random.uniform(lo, hi))
    return params


def worker_task(args):
    params, width, height, n_agents, steps, seed = args
    return run_simulation(params, width=width, height=height, n_agents=n_agents, steps=steps, seed=seed)


def main():
    p = argparse.ArgumentParser(description="Approximate Sobol benchmark runner")
    p.add_argument("--seconds", type=int, default=150, help="Target run duration in seconds (default 150)")
    p.add_argument("--workers", type=int, default=0, help="Number of worker processes (0 => os.cpu_count())")
    p.add_argument("--width", type=int, default=50)
    p.add_argument("--height", type=int, default=50)
    p.add_argument("--n-agents", type=int, default=40)
    p.add_argument("--steps", type=int, default=225)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    try:
        import os
        cpu_count = os.cpu_count() or 1
    except Exception:
        cpu_count = 1

    n_workers = args.workers if args.workers > 0 else cpu_count

    print(f"Starting approximate sobol benchmark for ~{args.seconds}s with {n_workers} worker(s)")

    start = time.time()
    total_runs = 0
    seeds_base = args.seed

    def print_progress(elapsed_sec: float) -> None:
        frac = min(elapsed_sec / args.seconds, 1.0)
        width = 40
        bar = "#" * int(frac * width) + "-" * (width - int(frac * width))
        percent = int(frac * 100)
        sys.stdout.write(
            f"\rBenchmark progress: [{bar}] {percent}% "
            f"({elapsed_sec:.1f}/{args.seconds:.1f}s)"
        )
        sys.stdout.flush()

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = set()
        next_seed = 0
        # Keep submitting batches of tasks while under time budget
        while True:
            # Submit up to n_workers jobs
            for _ in range(n_workers - len(futures)):
                params = sample_params()
                seed = seeds_base + next_seed
                next_seed += 1
                fut = pool.submit(worker_task, (params, args.width, args.height, args.n_agents, args.steps, seed))
                futures.add(fut)

            # Collect finished futures and update counters
            done_any = False
            for fut in list(futures):
                if fut.done():
                    futures.remove(fut)
                    try:
                        _ = fut.result()
                        total_runs += 1
                    except Exception as exc:
                        print("\nWorker task failed:", exc)
                        total_runs += 0
                    done_any = True

            elapsed = time.time() - start
            print_progress(elapsed)
            if elapsed >= args.seconds:
                break
            # If nothing finished yet, sleep a little to avoid busy loop
            if not done_any:
                time.sleep(0.5)

        # wait for currently running futures to finish (but don't extend runtime target)
        for fut in as_completed(futures, timeout=30):
            try:
                _ = fut.result()
                total_runs += 1
            except Exception:
                pass

    elapsed = time.time() - start
    print_progress(elapsed)
    sys.stdout.write("\n")
    avg = elapsed / max(1, total_runs)
    print(f"Benchmark completed: {total_runs} runs in {elapsed:.1f}s (avg {avg:.2f}s per run)")

    # Estimate full Sobol completion time using the same N and chunking assumptions.
    # Use the standard design size from sensitivity.py for a representative example.
    try:
        N = 2048
        samples = sobol_sample.sample(PROBLEM, N, calc_second_order=True)
        n_rows = len(samples)
        estimated_total = n_rows * avg
        hours = estimated_total / 3600.0
        print(f"Estimated full Sobol run: {n_rows} model rows -> {estimated_total:.1f}s ({hours:.2f}h)")
    except Exception as exc:
        print(f"Could not estimate full Sobol run automatically: {exc}")


if __name__ == "__main__":
    main()
