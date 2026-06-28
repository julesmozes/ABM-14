"""

Staged workflow (Snellius):
    1. Morris broad screen  -> identify influential parameters
    2. OFAT response curves -> interpret nonlinear effects
    3. Sobol on user-selected subset -> variance decomposition

Usage:
    python sensitivity.py sample --method morris --problem-file snellius/problems/screen.json --r 96
    python sensitivity.py run --method morris --chunk 0 --n-chunks 8
    python sensitivity.py status --method all
    python sensitivity.py analyze --method morris

    # After choosing Sobol parameters, edit snellius/problems/sobol.json then:
    python sensitivity.py sample --method sobol --problem-file snellius/problems/sobol.json --N 512
    python sensitivity.py run --method sobol --chunk 0 --n-chunks 16

    # Rerun one design row for full time-series plotting:
    python sensitivity.py rerun --method morris --row-index 42 --rep 0 --output traj.csv
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.analyze import morris as morris_analyze
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import morris as morris_sample
from SALib.sample import sobol as sobol_sample
from tqdm import tqdm

from model import FishingModel, gini_coefficient

RESULTS_ROOT = Path("results")
METHODS = ("ofat", "morris", "sobol")
SCREEN_PROBLEM_FILE = Path("snellius/problems/screen.json")

OUTPUTS = [
    "mean_lambda",
    "std_lambda",
    "scrounge_rate",
    "role_switch_rate",
    "bankruptcies",
    "network_mean_degree",
    "network_lcc_fraction",
    "mean_fish_density",
    "fleet_size",
    "wealth_gini",
]

MODEL_FIXED_DEFAULTS = {
    "beta": 2.0,
    "finders_share": 0.5,
    "r": 0.1,
    "patch_scale": 0.1,
    "sigma": 0.5,
    "q": 0.3,
    "c": 0.05,
    "v": 3,
    "C_birth": 2.0,
}


def method_dir(method: str, root: Path = RESULTS_ROOT) -> Path:
    return root / method


def _salib_problem(problem: dict) -> dict:
    """Return SALib-compatible problem dict (names + bounds only)."""
    return {
        "num_vars": len(problem["names"]),
        "names": list(problem["names"]),
        "bounds": [list(b) for b in problem["bounds"]],
    }


def load_problem_file(path: Path) -> dict:
    with open(path) as f:
        raw = json.load(f)
    names = raw["names"]
    bounds = raw["bounds"]
    if len(names) != len(bounds):
        raise ValueError(f"{path}: names and bounds length mismatch")
    defaults = dict(MODEL_FIXED_DEFAULTS)
    defaults.update(raw.get("defaults", {}))
    return {
        "names": names,
        "bounds": bounds,
        "defaults": defaults,
        "integer_params": list(raw.get("integer_params", [])),
        "morris_num_levels": int(raw.get("morris_num_levels", 4)),
    }


def save_problem(problem: dict, path: Path) -> None:
    payload = {
        "names": problem["names"],
        "bounds": problem["bounds"],
        "defaults": problem["defaults"],
        "integer_params": problem["integer_params"],
        "morris_num_levels": problem["morris_num_levels"],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_problem(method: str, root: Path = RESULTS_ROOT) -> dict:
    path = method_dir(method, root) / "problem.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run sample first.")
    return load_problem_file(path)


def row_to_params(row: np.ndarray, problem: dict) -> dict[str, float | int]:
    params = dict(problem["defaults"])
    integer = set(problem["integer_params"])
    for i, name in enumerate(problem["names"]):
        val = float(row[i])
        if name in integer:
            lo, hi = problem["bounds"][i]
            val = int(round(np.clip(val, lo, hi)))
        params[name] = val
    return params


def run_simulation(
    params: dict,
    *,
    width: int,
    height: int,
    n_agents: int,
    steps: int,
    seed: int,
    return_trajectory: bool = False,
) -> dict | tuple[dict[str, float], pd.DataFrame]:
    """Run one ABM replicate; return summary metrics (and optional full trajectory)."""
    model = FishingModel(
        width=width,
        height=height,
        n_agents=n_agents,
        v=int(params["v"]),
        r=float(params["r"]),
        q=float(params["q"]),
        c=float(params["c"]),
        beta=float(params["beta"]),
        C_birth=float(params["C_birth"]),
        sigma=float(params["sigma"]),
        finders_share=float(params["finders_share"]),
        patch_scale=float(params["patch_scale"]),
        rng=seed,
    )
    for _ in range(steps):
        model.step()

    agents = list(model.agents)
    df = model.datacollector.get_model_vars_dataframe()
    tail = df.iloc[-max(10, len(df) // 10) :]

    summary = {
        "mean_lambda": float(tail["Mean lambda"].mean()),
        "std_lambda": float(tail["Std lambda"].mean()),
        "scrounge_rate": float(tail["Scrounge rate"].mean()),
        "role_switch_rate": float(tail["Role switch rate"].mean()),
        "bankruptcies": float(tail["Bankruptcies"].mean()),
        "network_mean_degree": float(tail["Network mean degree"].mean()),
        "network_lcc_fraction": float(tail["Network LCC fraction"].mean()),
        "mean_fish_density": float(tail["Mean fish density"].mean()),
        "fleet_size": float(tail["Boats"].mean()),
        "wealth_gini": gini_coefficient([a.capital for a in agents]) if agents else 0.0,
    }
    if return_trajectory:
        return summary, df
    return summary


def evaluate_row(
    row_index: int,
    row: np.ndarray,
    problem: dict,
    *,
    width: int,
    height: int,
    n_agents: int,
    steps: int,
    replicates: int,
    base_seed: int,
) -> dict:
    """Average metrics over replicate seeds for one design row."""
    params = row_to_params(row, problem)
    accum = {key: 0.0 for key in OUTPUTS}
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
            accum[key] += metrics[key]
    out: dict = {"row_index": row_index}
    for name in problem["names"]:
        out[name] = params[name]
    for key in OUTPUTS:
        out[key] = accum[key] / replicates
    return out


def _evaluate_row_task(args: tuple) -> dict:
    (
        row_index,
        row,
        problem,
        width,
        height,
        n_agents,
        steps,
        replicates,
        base_seed,
    ) = args
    return evaluate_row(
        row_index,
        row,
        problem,
        width=width,
        height=height,
        n_agents=n_agents,
        steps=steps,
        replicates=replicates,
        base_seed=base_seed,
    )


def checkpoint_dir(method: str, chunk: int, root: Path) -> Path:
    return method_dir(method, root) / "partials" / "checkpoints" / f"{method}_{chunk}"


def checkpoint_path(method: str, chunk: int, row_index: int, root: Path) -> Path:
    return checkpoint_dir(method, chunk, root) / f"row_{row_index:06d}.csv"


def _completed_row_indices(method: str, root: Path) -> set[int]:
    partials = method_dir(method, root) / "partials"
    if not partials.exists():
        return set()
    indices: set[int] = set()
    for path in partials.rglob("row_*.csv"):
        df = pd.read_csv(path, usecols=["row_index"])
        indices.update(df["row_index"].astype(int))
    for path in partials.glob(f"{method}_*.csv"):
        if path.parent.name == "checkpoints":
            continue
        df = pd.read_csv(path, usecols=["row_index"])
        indices.update(df["row_index"].astype(int))
    return indices


def _write_row_checkpoint(row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)


def generate_ofat_samples(problem: dict, levels: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Baseline at defaults plus `levels` evenly spaced values per parameter."""
    names = problem["names"]
    defaults = problem["defaults"]
    rows = []
    meta = []
    baseline = [defaults[name] for name in names]
    rows.append(baseline)
    meta.append({"row_index": 0, "param": "baseline", "level": 0, "value": np.nan})

    idx = 1
    for j, name in enumerate(names):
        lo, hi = problem["bounds"][j]
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
    problem: dict,
    *,
    N: int = 512,
    r: int = 96,
    levels: int = 13,
) -> tuple[np.ndarray, pd.DataFrame | None]:
    salib = _salib_problem(problem)
    if method == "sobol":
        samples = sobol_sample.sample(salib, N, calc_second_order=False)
        return samples, None
    if method == "morris":
        samples = morris_sample.sample(
            salib, N=r, num_levels=problem["morris_num_levels"]
        )
        return samples, None
    if method == "ofat":
        return generate_ofat_samples(problem, levels)
    raise ValueError(f"Unknown method: {method}")


def cmd_sample(args: argparse.Namespace) -> None:
    problem = load_problem_file(args.problem_file)
    out_dir = method_dir(args.method, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, meta = generate_samples(
        args.method,
        problem,
        N=args.N,
        r=args.r,
        levels=args.levels,
    )

    np.save(out_dir / "samples.npy", samples)
    save_problem(problem, out_dir / "problem.json")

    sample_df = pd.DataFrame(samples, columns=problem["names"])
    sample_df.insert(0, "row_index", range(len(samples)))
    sample_df.to_csv(out_dir / "samples.csv", index=False)

    if meta is not None:
        meta.to_csv(out_dir / "ofat_meta.csv", index=False)

    print(f"Wrote {len(samples)} samples ({len(problem['names'])} params) to {out_dir}/")


def _chunk_bounds(n_rows: int, chunk: int, n_chunks: int) -> tuple[int, int]:
    size = (n_rows + n_chunks - 1) // n_chunks
    start = chunk * size
    end = min(start + size, n_rows)
    return start, end


def _merge_chunk_checkpoints(
    method: str, chunk: int, row_indices: list[int], root: Path
) -> Path:
    ckpt = checkpoint_dir(method, chunk, root)
    rows = []
    for i in sorted(row_indices):
        path = ckpt / f"row_{i:06d}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint {path}")
        rows.append(pd.read_csv(path).iloc[0])
    out_path = method_dir(method, root) / "partials" / f"{method}_{chunk}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("row_index").to_csv(out_path, index=False)
    return out_path


def cmd_run(args: argparse.Namespace) -> None:
    out_dir = method_dir(args.method, args.output_dir)
    samples_path = out_dir / "samples.npy"
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing {samples_path}; run sample first.")

    problem = load_problem(args.method, args.output_dir)
    samples = np.load(samples_path)
    n_rows = len(samples)
    start, end = _chunk_bounds(n_rows, args.chunk, args.n_chunks)
    if start >= n_rows:
        print(f"Chunk {args.chunk}: empty slice (start={start}, n_rows={n_rows})")
        return

    ckpt = checkpoint_dir(args.method, args.chunk, args.output_dir)
    ckpt.mkdir(parents=True, exist_ok=True)

    pending = []
    for i in range(start, end):
        if checkpoint_path(args.method, args.chunk, i, args.output_dir).exists():
            continue
        pending.append(i)

    try:
        n_workers = len(os.sched_getaffinity(0))
    except AttributeError:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, min(n_workers, max(1, len(pending))))

    skipped = (end - start) - len(pending)
    print(
        f"Chunk {args.chunk}/{args.n_chunks}: rows {start}-{end - 1} "
        f"({end - start} total, {skipped} resumed, {len(pending)} pending, "
        f"{n_workers} workers)",
        flush=True,
    )

    if pending:
        tasks = [
            (
                i,
                samples[i],
                problem,
                args.width,
                args.height,
                args.n_agents,
                args.steps,
                args.reps,
                args.base_seed,
            )
            for i in pending
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_evaluate_row_task, t): t[0] for t in tasks}
            with tqdm(total=len(pending), desc=f"{args.method} chunk {args.chunk}") as pbar:
                for fut in as_completed(futures):
                    row_index = futures[fut]
                    result = fut.result()
                    _write_row_checkpoint(
                        result,
                        checkpoint_path(args.method, args.chunk, row_index, args.output_dir),
                    )
                    pbar.update(1)

    chunk_rows = list(range(start, end))
    out_path = _merge_chunk_checkpoints(
        args.method, args.chunk, chunk_rows, args.output_dir
    )
    print(f"Wrote {out_path} ({len(chunk_rows)} rows)")


def _gather_partials(method: str, root: Path) -> pd.DataFrame:
    partials_dir = method_dir(method, root) / "partials"
    files = sorted(
        p for p in partials_dir.glob(f"{method}_*.csv") if p.parent.name != "checkpoints"
    )
    if not files:
        raise FileNotFoundError(f"No partials in {partials_dir}")
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df = df.sort_values("row_index").drop_duplicates(subset=["row_index"], keep="last")
    return df


def _load_samples(method: str, root: Path) -> np.ndarray:
    return np.load(method_dir(method, root) / "samples.npy")


def cmd_status(args: argparse.Namespace) -> None:
    methods = list(METHODS) if args.method == "all" else [args.method]
    for method in methods:
        mdir = method_dir(method, args.output_dir)
        samples_path = mdir / "samples.npy"
        if not samples_path.exists():
            print(f"{method}: no samples (run sample first)")
            continue
        n_samples = len(np.load(samples_path))
        done = len(_completed_row_indices(method, args.output_dir))
        pct = 100.0 * done / n_samples if n_samples else 0.0
        problem = load_problem(method, args.output_dir)
        print(
            f"{method}: {done}/{n_samples} rows ({pct:.1f}%) "
            f"[{len(problem['names'])} params]"
        )


def plot_sobol_bars(
    sobol_df: pd.DataFrame, params: list[str], path: Path, index: str = "S1"
) -> None:
    fig, axes = plt.subplots(len(OUTPUTS), 1, figsize=(8, 2.2 * len(OUTPUTS)), sharex=True)
    if len(OUTPUTS) == 1:
        axes = [axes]
    for ax, out in zip(axes, OUTPUTS):
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


def plot_sobol_heatmap(
    sobol_df: pd.DataFrame, params: list[str], path: Path, index: str = "ST"
) -> None:
    pivot = sobol_df.pivot(index="output", columns="parameter", values=index)
    pivot = pivot.reindex(index=OUTPUTS, columns=params)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels(params, rotation=30, ha="right")
    ax.set_yticks(range(len(OUTPUTS)))
    ax.set_yticklabels([o.replace("_", " ") for o in OUTPUTS])
    ax.set_title(f"Sobol {index} heatmap")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ofat_curves(
    raw_df: pd.DataFrame, meta_df: pd.DataFrame, params: list[str], path: Path
) -> None:
    n_params = len(params)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).flatten()
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
    for ax in axes[n_params:]:
        ax.set_visible(False)
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


def analyze_sobol(raw_df: pd.DataFrame, problem: dict, out_dir: Path) -> pd.DataFrame:
    salib = _salib_problem(problem)
    params = problem["names"]
    Y = {out: raw_df[out].values for out in OUTPUTS}
    rows = []
    for out in OUTPUTS:
        Si = sobol_analyze.analyze(
            salib, Y[out], calc_second_order=False, print_to_console=False
        )
        for j, name in enumerate(params):
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
    plot_sobol_bars(sobol_df, params, out_dir / "sobol_first_order.png", index="S1")
    plot_sobol_bars(sobol_df, params, out_dir / "sobol_total_order.png", index="ST")
    plot_sobol_heatmap(sobol_df, params, out_dir / "sobol_heatmap.png", index="ST")
    return sobol_df


def analyze_morris(
    raw_df: pd.DataFrame, problem: dict, out_dir: Path, root: Path
) -> pd.DataFrame:
    salib = _salib_problem(problem)
    params = problem["names"]
    samples = _load_samples("morris", root)
    indices = raw_df["row_index"].astype(int).to_numpy()
    X = samples[indices]
    Y = {out: raw_df[out].values for out in OUTPUTS}
    rows = []
    for out in OUTPUTS:
        Si = morris_analyze.analyze(
            salib,
            X,
            Y[out],
            print_to_console=False,
            num_levels=problem["morris_num_levels"],
        )
        for j, name in enumerate(params):
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


def analyze_ofat(raw_df: pd.DataFrame, problem: dict, out_dir: Path) -> pd.DataFrame:
    meta_path = out_dir / "ofat_meta.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")
    meta_df = pd.read_csv(meta_path)
    raw_df.to_csv(out_dir / "ofat_results.csv", index=False)
    plot_ofat_curves(raw_df, meta_df, problem["names"], out_dir / "ofat_curves.png")
    return raw_df


def cmd_analyze(args: argparse.Namespace) -> None:
    out_dir = method_dir(args.method, args.output_dir)
    problem = load_problem(args.method, args.output_dir)
    raw_df = _gather_partials(args.method, args.output_dir)
    n_samples = len(_load_samples(args.method, args.output_dir))
    if len(raw_df) != n_samples:
        if not args.allow_partial:
            raise RuntimeError(
                f"Incomplete results: {len(raw_df)} rows gathered, expected {n_samples}. "
                "Re-run missing rows or pass --allow-partial (Morris screening only)."
            )
        missing = n_samples - len(raw_df)
        print(
            f"WARNING: analyzing {len(raw_df)}/{n_samples} rows "
            f"({missing} missing; Morris indices are approximate)."
        )

    raw_df.to_csv(out_dir / "sensitivity_raw.csv", index=False)

    if args.method == "sobol":
        if args.allow_partial:
            raise RuntimeError("--allow-partial is not valid for Sobol (design must be complete).")
        sobol_df = analyze_sobol(raw_df, problem, out_dir)
        top = (
            sobol_df.sort_values("ST", ascending=False)
            .groupby("output")
            .head(1)[["output", "parameter", "ST"]]
        )
        print("\nDominant parameter (total-order) per output:")
        print(top.to_string(index=False))
    elif args.method == "morris":
        morris_df = analyze_morris(raw_df, problem, out_dir, args.output_dir)
        top = (
            morris_df.sort_values("mu_star", ascending=False)
            .groupby("output")
            .head(1)[["output", "parameter", "mu_star"]]
        )
        print("\nDominant parameter (mu*) per output:")
        print(top.to_string(index=False))
    elif args.method == "ofat":
        analyze_ofat(raw_df, problem, out_dir)
        print(f"OFAT curves written to {out_dir}/ofat_curves.png")

    print(f"Analysis complete: {out_dir}/")


def cmd_rerun(args: argparse.Namespace) -> None:
    """Run one design row + replicate and save the full model time series."""
    problem = load_problem(args.method, args.output_dir)
    samples = np.load(method_dir(args.method, args.output_dir) / "samples.npy")
    if args.row_index < 0 or args.row_index >= len(samples):
        raise ValueError(f"row_index must be in [0, {len(samples) - 1}]")

    params = row_to_params(samples[args.row_index], problem)
    seed = args.base_seed + args.row_index * args.reps + args.rep
    _, traj = run_simulation(
        params,
        width=args.width,
        height=args.height,
        n_agents=args.n_agents,
        steps=args.steps,
        seed=seed,
        return_trajectory=True,
    )
    traj.insert(0, "step", range(len(traj)))
    for name, val in params.items():
        traj.insert(0, name, val)
    traj.insert(0, "seed", seed)
    traj.insert(0, "rep", args.rep)
    traj.insert(0, "row_index", args.row_index)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    traj.to_csv(args.output, index=False)
    print(f"Wrote trajectory ({len(traj)} steps) to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--output-dir", type=Path, default=RESULTS_ROOT)

    parser = argparse.ArgumentParser(
        description="Sensitivity analysis for the fishing ABM (OFAT, Morris, Sobol)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sample = sub.add_parser("sample", parents=[parent], help="Generate parameter design matrix")
    p_sample.add_argument("--method", choices=METHODS, required=True)
    p_sample.add_argument(
        "--problem-file",
        type=Path,
        default=SCREEN_PROBLEM_FILE,
        help="JSON parameter definition (names, bounds, defaults)",
    )
    p_sample.add_argument("--N", type=int, default=512, help="Sobol base sample size")
    p_sample.add_argument("--r", type=int, default=96, help="Morris trajectories")
    p_sample.add_argument("--levels", type=int, default=13, help="OFAT levels per param")
    p_sample.set_defaults(func=cmd_sample)

    p_run = sub.add_parser("run", parents=[parent], help="Evaluate a chunk of design rows")
    p_run.add_argument("--method", choices=METHODS, required=True)
    p_run.add_argument("--chunk", type=int, required=True)
    p_run.add_argument("--n-chunks", type=int, required=True)
    p_run.add_argument("--reps", type=int, default=6)
    p_run.add_argument("--steps", type=int, default=250)
    p_run.add_argument("--width", type=int, default=50)
    p_run.add_argument("--height", type=int, default=50)
    p_run.add_argument("--n-agents", type=int, default=40)
    p_run.add_argument("--base-seed", type=int, default=1000)
    p_run.set_defaults(func=cmd_run)

    p_analyze = sub.add_parser("analyze", parents=[parent], help="Gather partials and compute indices")
    p_analyze.add_argument("--method", choices=METHODS, required=True)
    p_analyze.add_argument(
        "--allow-partial",
        action="store_true",
        help="Analyze completed rows only (Morris screening; not valid for Sobol)",
    )
    p_analyze.set_defaults(func=cmd_analyze)

    p_status = sub.add_parser("status", parents=[parent], help="Report completed design rows")
    p_status.add_argument(
        "--method",
        choices=[*METHODS, "all"],
        default="all",
        help="Method to report (default: all)",
    )
    p_status.set_defaults(func=cmd_status)

    p_rerun = sub.add_parser(
        "rerun", parents=[parent], help="Rerun one row for full time-series output"
    )
    p_rerun.add_argument("--method", choices=METHODS, required=True)
    p_rerun.add_argument("--row-index", type=int, required=True)
    p_rerun.add_argument("--rep", type=int, default=0)
    p_rerun.add_argument("--reps", type=int, default=6, help="Must match the run campaign reps")
    p_rerun.add_argument("--steps", type=int, default=250)
    p_rerun.add_argument("--width", type=int, default=50)
    p_rerun.add_argument("--height", type=int, default=50)
    p_rerun.add_argument("--n-agents", type=int, default=40)
    p_rerun.add_argument("--base-seed", type=int, default=1000)
    p_rerun.add_argument("--output", type=Path, required=True)
    p_rerun.set_defaults(func=cmd_rerun)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
