"""Simple Morris sensitivity analysis for the fishing ABM.

Run this directly from the project root:

    uv run python morris_sensitivity.py

"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt  # Added for plotting
import numpy as np
import pandas as pd
from SALib.analyze import morris as morris_analyze
from SALib.sample import morris as morris_sample

from model import FishingModel, gini_coefficient


PROBLEM = {
    "num_vars": 8,
    "names": [
        "r",
        "q",
        "c",
        "beta",
        "C_birth",
        "sigma",
        "finders_share",
        "patch_scale",
    ],
    "bounds": [
        [0.05, 0.3],
        [0.1, 0.5],
        [0.01, 0.15],
        [0.5, 5.0],
        [1.0, 4.0],
        [0.1, 1.0],
        [0.2, 0.8],
        [0.05, 0.3],
    ],
}

OUTPUTS = [
    "mean_lambda",
    "producer_rate",
    "scrounge_rate",
    "role_switch_rate",
    "network_mean_degree",
    "network_lcc_fraction",
    "mean_fish_density",
    "fleet_size",
    "wealth_gini",
]


def row_to_params(row: np.ndarray) -> dict[str, float]:
    return {name: float(row[i]) for i, name in enumerate(PROBLEM["names"])}


def run_simulation(
    params: dict[str, float],
    *,
    width: int,
    height: int,
    n_agents: int,
    steps: int,
    seed: int,
) -> dict[str, float]:
    """Run one replicate and return terminal summary metrics."""
    model = FishingModel(
        width=width,
        height=height,
        n_agents=n_agents,
        v=3,
        r=params["r"],
        q=params["q"],
        c=params["c"],
        beta=params["beta"],
        C_birth=params["C_birth"],
        sigma=params["sigma"],
        finders_share=params["finders_share"],
        patch_scale=params["patch_scale"],
        rng=seed,
    )
    for _ in range(steps):
        model.step()

    agents = list(model.agents)
    df = model.datacollector.get_model_vars_dataframe()
    tail = df.iloc[-max(10, len(df) // 10) :]

    return {
        "mean_lambda": float(tail["Mean lambda"].mean()),
        "producer_rate": float(tail["Producer rate"].mean()),
        "scrounge_rate": float(tail["Scrounge rate"].mean()),
        "role_switch_rate": float(tail["Role switch rate"].mean()),
        "network_mean_degree": float(tail["Network mean degree"].mean()),
        "network_lcc_fraction": float(tail["Network LCC fraction"].mean()),
        "mean_fish_density": float(tail["Mean fish density"].mean()),
        "fleet_size": float(tail["Boats"].mean()),
        "wealth_gini": gini_coefficient([a.capital for a in agents]) if agents else 0.0,
    }


def evaluate_row(args: tuple) -> dict[str, float]:
    row_index, row, width, height, n_agents, steps, replicates, base_seed = args
    params = row_to_params(row)
    totals = {key: 0.0 for key in OUTPUTS}

    for rep in range(replicates):
        seed = base_seed + row_index * replicates + rep
        metrics = run_simulation(
            params,
            width=width,
            height=height,
            n_agents=n_agents,
            steps=steps,
            seed=seed,
        )
        for key in OUTPUTS:
            totals[key] += metrics[key]

    result = {"row_index": row_index}
    result.update(params)
    for key in OUTPUTS:
        result[key] = totals[key] / replicates
    return result


def analyze_results(
    samples: np.ndarray, raw_df: pd.DataFrame, out_dir: Path, num_levels: int
) -> pd.DataFrame:
    rows = []
    for out in OUTPUTS:
        indices = morris_analyze.analyze(
            PROBLEM,
            samples,
            raw_df[out].values,
            num_levels=num_levels,
            print_to_console=False,
        )
        for j, name in enumerate(PROBLEM["names"]):
            rows.append(
                {
                    "output": out,
                    "parameter": name,
                    "mu": indices["mu"][j],
                    "mu_star": indices["mu_star"][j],
                    "sigma": indices["sigma"][j],
                }
            )

    morris_df = pd.DataFrame(rows)
    morris_df.to_csv(out_dir / "morris_indices.csv", index=False)
    return morris_df


def plot_morris_results(morris_df: pd.DataFrame, out_dir: Path) -> None:
    """Generates a multi-panel subplot of mu_star vs sigma for each model output."""
    n_outputs = len(OUTPUTS)
    fig, axes = plt.subplots(n_outputs, 1, figsize=(8, 3 * n_outputs), sharex=False)
    
    if n_outputs == 1:
        axes = [axes]
        
    for ax, output_name in zip(axes, OUTPUTS):
        sub_df = morris_df[morris_df["output"] == output_name]
        
        ax.scatter(sub_df["mu_star"], sub_df["sigma"], s=50, alpha=0.7, color="steelblue")
        
        # Annotate points with parameter names
        for _, row in sub_df.iterrows():
            ax.annotate(
                row["parameter"],
                (row["mu_star"], row["sigma"]),
                textcoords="offset points",
                xytext=(5, 2),
                ha="left",
                fontsize=9,
            )
            
        ax.set_title(output_name.replace("_", " "), fontsize=12, fontweight="bold")
        ax.set_xlabel(r"$\mu^*$", fontsize=10)
        ax.set_ylabel(r"$\sigma$", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = out_dir / "morris_scatter.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Morris scatter plot layout to: {plot_path}")


def _print_progress(done: int, total: int) -> None:
    percent = int((done / total) * 100) if total else 100
    sys.stderr.write(f"\rProgress: {done}/{total} ({percent}%)")
    sys.stderr.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simple Morris sensitivity analysis for the fishing ABM"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "morris_simple")
    parser.add_argument("--N", type=int, default=20, help="Morris trajectory count")
    parser.add_argument("--num-levels", type=int, default=4, help="Morris grid levels")
    parser.add_argument("--reps", type=int, default=1, help="Replicates per Morris row")
    parser.add_argument("--steps", type=int, default=100, help="Model steps per replicate")
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--height", type=int, default=50)
    parser.add_argument("--n-agents", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=0, help="0 uses all available CPUs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = morris_sample.sample(PROBLEM, N=args.N, num_levels=args.num_levels)
    np.save(out_dir / "samples.npy", samples)
    pd.DataFrame(samples, columns=PROBLEM["names"]).to_csv(out_dir / "samples.csv", index=False)

    tasks = [
        (
            i,
            samples[i],
            args.width,
            args.height,
            args.n_agents,
            args.steps,
            args.reps,
            args.seed,
        )
        for i in range(len(samples))
    ]

    try:
        available_workers = len(os.sched_getaffinity(0))
    except AttributeError:
        available_workers = os.cpu_count() or 1
    n_workers = args.workers if args.workers > 0 else available_workers
    n_workers = max(1, min(n_workers, len(tasks)))

    print(f"Running {len(samples)} Morris rows with {n_workers} worker(s)...", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(evaluate_row, task) for task in tasks]
        rows = []
        total = len(futures)
        for done, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            _print_progress(done, total)

    sys.stderr.write("\n")
    sys.stderr.flush()

    raw_df = pd.DataFrame(rows).sort_values("row_index")
    raw_df.to_csv(out_dir / "morris_raw.csv", index=False)

    morris_df = analyze_results(samples, raw_df, out_dir, args.num_levels)
    
    # Generate scatter plots layout
    plot_morris_results(morris_df, out_dir)

    importance_df = morris_df.sort_values(["output", "mu_star"], ascending=[True, False]).copy()
    importance_df["rank"] = importance_df.groupby("output").cumcount() + 1
    importance_df = importance_df[
        ["output", "rank", "parameter", "mu_star", "sigma", "mu"]
    ]
    importance_df.to_csv(out_dir / "morris_importance_by_output.csv", index=False)

    top = (
        morris_df.sort_values("mu_star", ascending=False)
        .groupby("output")
        .head(1)[["output", "parameter", "mu_star"]]
    )
    top.to_csv(out_dir / "morris_dominant_parameters.csv", index=False)

    print("\nDominant parameter by output:")
    print(top.to_string(index=False))
    print("\nFull parameter importance table by output:")
    print(importance_df.to_string(index=False))
    print(f"\nWrote results to {out_dir}/")


if __name__ == "__main__":
    main()