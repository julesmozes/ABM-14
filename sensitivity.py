"""
Three methods (OFAT, Morris, Sobol) share a sample → run → analyze pipeline
designed for embarrassingly parallel execution on Snellius.

Usage:
    # Generate design matrices (login node)
    python sensitivity.py sample --method sobol --N 2048
    python sensitivity.py sample --method morris --r 500
    python sensitivity.py sample --method ofat --levels 21

    # Run evaluations (SLURM job array or local)
    python sensitivity.py run --method sobol --chunk 0 --n-chunks 32

    # Analyze and plot (login node)
    python sensitivity.py analyze --method sobol

    # Local smoke test
    python sensitivity.py sample --method sobol --N 16
    python sensitivity.py run --method sobol --chunk 0 --n-chunks 1 --reps 2 --steps 50
    python sensitivity.py analyze --method sobol
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.analyze import morris as morris_analyze
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import morris as morris_sample
from SALib.sample import sobol as sobol_sample

from model import FishingModel, gini_coefficient

RESULTS_ROOT = Path("results")
METHODS = ("ofat", "morris", "sobol")

OUTPUTS = [
    "mean_lambda",
    "scrounge_rate",
    "network_mean_degree",
    "network_lcc_fraction",
    "mean_fish_density",
    "fleet_size",
    "wealth_gini",
]

PROBLEM = {
    "num_vars": 4,
    "names": [
        "beta",
        "finders_share",
        "r",
        "patch_scale",
    ],
    "bounds": [
        [0.5, 4.0],
        [0.2, 0.8],
        [0.05, 0.25],
        [0.05, 0.25],
    ],
}

DEFAULTS = {
    "beta": 2.0,
    "finders_share": 0.5,
    "r": 0.1,
    "patch_scale": 0.1,
    "sigma": 0.5,
}


def method_dir(method: str, root: Path = RESULTS_ROOT) -> Path:
    return root / method


def run_simulation(
    params: dict,
    *,
    width: int,
    height: int,
    n_agents: int,
    steps: int,
    seed: int,
) -> dict[str, float]:
    """Run one ABM replicate and return terminal summary metrics."""
    model = FishingModel(
        width=width,
        height=height,
        n_agents=n_agents,
        beta=params["beta"],
        finders_share=params["finders_share"],
        r=params["r"],
        patch_scale=params["patch_scale"],
        sigma=DEFAULTS["sigma"],
        rng=seed,
    )
    for _ in range(steps):
        model.step()

    df = model.datacollector.get_model_vars_dataframe()
    tail = df.iloc[-max(10, len(df) // 10) :]

    return {
        "mean_lambda": float(tail["Mean lambda"].mean()),
        "scrounge_rate": float(tail["Scrounge rate"].mean()),
        "network_mean_degree": float(tail["Network mean degree"].mean()),
        "network_lcc_fraction": float(tail["Network LCC fraction"].mean()),
        "mean_fish_density": float(tail["Mean fish density"].mean()),
        "fleet_size": float(tail["Boats"].mean()),
        "wealth_gini": float(tail["Wealth Gini"].mean()),
    }


def _row_to_params(row: np.ndarray) -> dict[str, float]:
    return {name: float(row[i]) for i, name in enumerate(PROBLEM["names"])}


def evaluate_row(
    row_index: int,
    row: np.ndarray,
    *,
    width: int,
    height: int,
    n_agents: int,
    steps: int,
    replicates: int,
    base_seed: int,
) -> dict:
    """Average metrics over replicate seeds for one design row."""
    params = _row_to_params(row)
    accum = {key: 0.0 for key in OUTPUTS}
    successful = 0
    for rep in range(replicates):
        seed = base_seed + row_index * replicates + rep
        try:
            metrics = run_simulation(
                params, width=width, height=height,
                n_agents=n_agents, steps=steps, seed=seed,
            )
            for key in OUTPUTS:
                accum[key] += metrics[key]
            successful += 1
        except Exception as exc:
            print(f"[WARN] row {row_index} rep {rep} seed {seed}: {exc}", flush=True)

    out = {"row_index": row_index, "failed_reps": replicates - successful}
    out.update(params)
    if successful == 0:
        # All replicates failed — flag for re-queuing, don't write NaN
        raise RuntimeError(f"All {replicates} replicates failed for row {row_index}: params={params}")
    for key in OUTPUTS:
        out[key] = accum[key] / successful
    return out


def _evaluate_row_task(args: tuple) -> dict | None:
    row_index, row, width, height, n_agents, steps, replicates, base_seed = args
    try:
        return evaluate_row(
            row_index, row, width=width, height=height,
            n_agents=n_agents, steps=steps,
            replicates=replicates, base_seed=base_seed,
        )
    except RuntimeError as exc:
        print(f"[ERROR] row {row_index} skipped: {exc}", flush=True)
        return None


def generate_ofat_samples(levels: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Baseline at defaults plus `levels` evenly spaced values per parameter."""
    rows = []
    meta = []
    baseline = [DEFAULTS[name] for name in PROBLEM["names"]]
    rows.append(baseline)
    meta.append({"row_index": 0, "param": "baseline", "level": 0, "value": np.nan})

    idx = 1
    for j, name in enumerate(PROBLEM["names"]):
        lo, hi = PROBLEM["bounds"][j]
        for k, val in enumerate(np.linspace(lo, hi, levels)):
            row = baseline.copy()
            row[j] = float(val)
            rows.append(row)
            meta.append(
                {"row_index": idx, "param": name, "level": k, "value": float(val)}
            )
            idx += 1

    return np.asarray(rows, dtype=float), pd.DataFrame(meta)


def generate_samples(
    method: str,
    *,
    N: int = 2048,
    r: int = 500,
    levels: int = 21,
) -> tuple[np.ndarray, pd.DataFrame | None]:
    if method == "sobol":
        samples = sobol_sample.sample(PROBLEM, N, calc_second_order=True)
        return samples, None
    if method == "morris":
        samples = morris_sample.sample(PROBLEM, N=r, num_levels=4)
        return samples, None
    if method == "ofat":
        return generate_ofat_samples(levels)
    raise ValueError(f"Unknown method: {method}")


def cmd_sample(args: argparse.Namespace) -> None:
    out_dir = method_dir(args.method, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, meta = generate_samples(
        args.method,
        N=args.N,
        r=args.r,
        levels=args.levels,
    )

    np.save(out_dir / "samples.npy", samples)
    with open(out_dir / "problem.json", "w") as f:
        json.dump(PROBLEM, f, indent=2)

    sample_df = pd.DataFrame(samples, columns=PROBLEM["names"])
    sample_df.insert(0, "row_index", range(len(samples)))
    sample_df.to_csv(out_dir / "samples.csv", index=False)

    if meta is not None:
        meta.to_csv(out_dir / "ofat_meta.csv", index=False)

    print(f"Wrote {len(samples)} samples to {out_dir}/")


def _chunk_bounds(n_rows: int, chunk: int, n_chunks: int) -> tuple[int, int]:
    size = (n_rows + n_chunks - 1) // n_chunks
    start = chunk * size
    end = min(start + size, n_rows)
    return start, end


def cmd_run(args: argparse.Namespace) -> None:
    out_dir = method_dir(args.method, args.output_dir)
    samples_path = out_dir / "samples.npy"
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing {samples_path}; run sample first.")

    samples = np.load(samples_path)
    n_rows = len(samples)
    start, end = _chunk_bounds(n_rows, args.chunk, args.n_chunks)
    if start >= n_rows:
        print(f"Chunk {args.chunk}: empty slice (start={start}, n_rows={n_rows})")
        return

    partials_dir = out_dir / "partials"
    partials_dir.mkdir(parents=True, exist_ok=True)
    out_path = partials_dir / f"{args.method}_{args.chunk}.csv"

    try:
        n_workers = len(os.sched_getaffinity(0))
    except AttributeError:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, min(n_workers, end - start))

    tasks = [
        (
            i,
            samples[i],
            args.width,
            args.height,
            args.n_agents,
            args.steps,
            args.reps,
            args.base_seed,
        )
        for i in range(start, end)
    ]

    print(
        f"Chunk {args.chunk}/{args.n_chunks}: rows {start}-{end - 1} "
        f"({end - start} rows, {n_workers} workers)",
        flush=True,
    )

    rows = []
    out_path_tmp = out_path.with_suffix(".tmp.csv")
    wrote_header = False
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for result in pool.map(_evaluate_row_task, tasks, chunksize=1):
            if result is None:
                continue
            rows.append(result)
            # flush every N results so progress survives a crash
            if len(rows) % 50 == 0:
                partial = pd.DataFrame(rows)
                partial.to_csv(
                    out_path_tmp,
                    mode="a",
                    header=not wrote_header,
                    index=False,
                )
                rows.clear()
                wrote_header = True

    # flush remainder
    if rows:
        pd.DataFrame(rows).to_csv(
            out_path_tmp, mode="a", header=not wrote_header, index=False
        )

    out_path_tmp.rename(out_path)
    print(f"Wrote {out_path}")


def _gather_partials(method: str, root: Path) -> pd.DataFrame:
    partials_dir = method_dir(method, root) / "partials"
    files = sorted(partials_dir.glob(f"{method}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No partials in {partials_dir}")
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df = df.sort_values("row_index").drop_duplicates(subset=["row_index"], keep="last")
    return df


def _load_samples(method: str, root: Path) -> np.ndarray:
    return np.load(method_dir(method, root) / "samples.npy")


def plot_sobol_bars(sobol_df: pd.DataFrame, path: Path, index: str = "S1") -> None:
    outputs = OUTPUTS
    params = PROBLEM["names"]
    fig, axes = plt.subplots(len(outputs), 1, figsize=(8, 2.2 * len(outputs)), sharex=True)
    if len(outputs) == 1:
        axes = [axes]
    for ax, out in zip(axes, outputs):
        sub = sobol_df[sobol_df["output"] == out].set_index("parameter").loc[params]
        vals = sub[index].values
        conf = sub[f"{index}_conf"].values
        ax.bar(params, vals, yerr=conf, capsize=3, color="steelblue", alpha=0.85)
        ax.set_ylabel(index)
        ax.set_title(out.replace("_", " "))
        ymax = max(0.3, float((vals + conf).max() * 1.2))
        ax.set_ylim(0, min(1.05, ymax))
    axes[-1].tick_params(axis="x", rotation=30)
    fig.suptitle(f"Sobol {index} indices", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sobol_heatmap(sobol_df: pd.DataFrame, path: Path, index: str = "ST") -> None:
    pivot = sobol_df.pivot(index="output", columns="parameter", values=index)
    pivot = pivot.reindex(index=OUTPUTS, columns=PROBLEM["names"])
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(PROBLEM["names"])))
    ax.set_xticklabels(PROBLEM["names"], rotation=30, ha="right")
    ax.set_yticks(range(len(OUTPUTS)))
    ax.set_yticklabels([o.replace("_", " ") for o in OUTPUTS])
    ax.set_title(f"Sobol {index} heatmap")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ofat_curves(raw_df: pd.DataFrame, meta_df: pd.DataFrame, path: Path) -> None:
    params = PROBLEM["names"]
    n = len(params)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    for ax, param in zip(axes, params):
        sub_meta = meta_df[meta_df["param"] == param]
        sub_raw = raw_df.set_index("row_index").loc[sub_meta["row_index"]]
        for out in ("mean_lambda", "scrounge_rate", "network_lcc_fraction"):
            ax.plot(
                sub_meta["value"].values,
                sub_raw[out].values,
                marker="o",
                ms=3,
                label=out.replace("_", " "),
            )
        ax.set_xlabel(param)
        ax.set_title(param)
        ax.legend(fontsize=6)
    fig.suptitle("OFAT response curves", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_morris_scatter(morris_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(len(OUTPUTS), 1, figsize=(6, 2.2 * len(OUTPUTS)))
    if len(OUTPUTS) == 1:
        axes = [axes]
    for ax, out in zip(axes, OUTPUTS):
        sub = morris_df[morris_df["output"] == out]
        ax.scatter(sub["mu_star"], sub["sigma"], c="steelblue", alpha=0.85)
        for _, row in sub.iterrows():
            ax.annotate(row["parameter"], (row["mu_star"], row["sigma"]), fontsize=7)
        ax.set_xlabel("mu*")
        ax.set_ylabel("sigma")
        ax.set_title(out.replace("_", " "))
    fig.suptitle("Morris mu* vs sigma", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze_sobol(raw_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    Y = {out: raw_df[out].values for out in OUTPUTS}
    rows = []
    for out in OUTPUTS:
        Si = sobol_analyze.analyze(
            PROBLEM, Y[out], calc_second_order=True, print_to_console=False
        )
        for j, name in enumerate(PROBLEM["names"]):
            rows.append(
                {
                    "output": out,
                    "parameter": name,
                    "S1": Si["S1"][j],
                    "S1_conf": Si["S1_conf"][j],
                    "ST": Si["ST"][j],
                    "ST_conf": Si["ST_conf"][j],
                }
            )
    sobol_df = pd.DataFrame(rows)
    sobol_df.to_csv(out_dir / "sobol_indices.csv", index=False)
    plot_sobol_bars(sobol_df, out_dir / "sobol_first_order.png", index="S1")
    plot_sobol_bars(sobol_df, out_dir / "sobol_total_order.png", index="ST")
    plot_sobol_heatmap(sobol_df, out_dir / "sobol_heatmap.png", index="ST")
    return sobol_df


def analyze_morris(raw_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    Y = {out: raw_df[out].values for out in OUTPUTS}
    rows = []
    for out in OUTPUTS:
        Si = morris_analyze.analyze(
            PROBLEM, Y[out], print_to_console=False, num_levels=4
        )
        for j, name in enumerate(PROBLEM["names"]):
            rows.append(
                {
                    "output": out,
                    "parameter": name,
                    "mu": Si["mu"][j],
                    "mu_star": Si["mu_star"][j],
                    "sigma": Si["sigma"][j],
                }
            )
    morris_df = pd.DataFrame(rows)
    morris_df.to_csv(out_dir / "morris_indices.csv", index=False)
    plot_morris_scatter(morris_df, out_dir / "morris_scatter.png")
    return morris_df


def analyze_ofat(raw_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    meta_path = out_dir / "ofat_meta.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")
    meta_df = pd.read_csv(meta_path)
    raw_df.to_csv(out_dir / "ofat_results.csv", index=False)
    plot_ofat_curves(raw_df, meta_df, out_dir / "ofat_curves.png")
    return raw_df


def cmd_analyze(args: argparse.Namespace) -> None:
    out_dir = method_dir(args.method, args.output_dir)
    raw_df = _gather_partials(args.method, args.output_dir)
    n_samples = len(_load_samples(args.method, args.output_dir))
    if len(raw_df) != n_samples:
        raise RuntimeError(
            f"Incomplete results: {len(raw_df)} rows gathered, expected {n_samples}"
        )

    raw_df.to_csv(out_dir / "sensitivity_raw.csv", index=False)

    if args.method == "sobol":
        sobol_df = analyze_sobol(raw_df, out_dir)
        top = (
            sobol_df.sort_values("ST", ascending=False)
            .groupby("output")
            .head(1)[["output", "parameter", "ST"]]
        )
        print("\nDominant parameter (total-order) per output:")
        print(top.to_string(index=False))
    elif args.method == "morris":
        morris_df = analyze_morris(raw_df, out_dir)
        top = (
            morris_df.sort_values("mu_star", ascending=False)
            .groupby("output")
            .head(1)[["output", "parameter", "mu_star"]]
        )
        print("\nDominant parameter (mu*) per output:")
        print(top.to_string(index=False))
    elif args.method == "ofat":
        analyze_ofat(raw_df, out_dir)
        print(f"OFAT curves written to {out_dir}/ofat_curves.png")

    print(f"Analysis complete: {out_dir}/")


def build_parser() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--output-dir", type=Path, default=RESULTS_ROOT)

    parser = argparse.ArgumentParser(
        description="Sensitivity analysis for the fishing ABM (OFAT, Morris, Sobol)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sample = sub.add_parser("sample", parents=[parent], help="Generate parameter design matrix")
    p_sample.add_argument("--method", choices=METHODS, required=True)
    p_sample.add_argument("--N", type=int, default=2048, help="Sobol base sample size")
    p_sample.add_argument("--r", type=int, default=500, help="Morris trajectories")
    p_sample.add_argument("--levels", type=int, default=21, help="OFAT levels per param")
    p_sample.set_defaults(func=cmd_sample)

    p_run = sub.add_parser("run", parents=[parent], help="Evaluate a chunk of design rows")
    p_run.add_argument("--method", choices=METHODS, required=True)
    p_run.add_argument("--chunk", type=int, required=True)
    p_run.add_argument("--n-chunks", type=int, required=True)
    p_run.add_argument("--reps", type=int, default=10)
    p_run.add_argument("--steps", type=int, default=250)
    p_run.add_argument("--width", type=int, default=50)
    p_run.add_argument("--height", type=int, default=50)
    p_run.add_argument("--n-agents", type=int, default=40)
    p_run.add_argument("--base-seed", type=int, default=1000)
    p_run.set_defaults(func=cmd_run)

    p_analyze = sub.add_parser("analyze", parents=[parent], help="Gather partials and compute indices")
    p_analyze.add_argument("--method", choices=METHODS, required=True)
    p_analyze.set_defaults(func=cmd_analyze)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
