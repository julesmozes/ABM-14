# Snellius sensitivity analysis workflow

`ssh snellius` lands in your project working directory. All commands below assume you are at the repo root.

## Student campaign (current)

1. **Morris** — `benchmark.sh` with `r=48` (480 rows, 6 reps); analyze with `sensitivity.py analyze --method morris --allow-partial` (47/48 complete trajectories OK)
2. **OFAT** — 9 params, **7 levels**, tightened bounds → **64 rows** × 8 reps
3. **Sobol** — top-5 Morris params (`C_birth`, `c`, `v`, `r`, `beta`), **N=256** → **1792 rows** × 6 reps

```bash
source venv/bin/activate
bash snellius/prepare_student.sh
sbatch snellius/run_ofat.sh
sbatch snellius/run_sobol.sh
```

Monitor:

```bash
watch -n 30 '
squeue -u $USER
echo; ls sensitivity/ofat/partials/checkpoints/ofat_0 2>/dev/null | wc -l; echo / 64 OFAT
find sensitivity/sobol/partials/checkpoints -name "row_*.csv" 2>/dev/null | wc -l; echo / 1792 Sobol
python sensitivity.py status --method ofat
python sensitivity.py status --method sobol
'
```

Analyze:

```bash
python sensitivity.py analyze --method morris --allow-partial   # if Morris incomplete
python sensitivity.py analyze --method ofat
python sensitivity.py analyze --method sobol
```

## 1. One-time environment setup

```bash
bash snellius/setup_env.sh
source venv/bin/activate
```

## 2. Validate setup

```bash
bash snellius/check.sh
```

## 3. Benchmark (Morris timing + optional r=48 screen)

```bash
mkdir -p logs
sbatch snellius/benchmark.sh
```

## Parameter sets

| File | Use | Notes |
|------|-----|-------|
| `snellius/problems/screen.json` | Morris + OFAT | 9 params; tightened bounds (`q≤0.8`, `v≤5`, `C_birth≥1`) |
| `snellius/problems/sobol.json` | Sobol | 5 params from Morris ranking |

Fixed during sensitivity: ρ, grid size, fleet size, steps, K, metabolism, v_see.

## SLURM resources

| Script | Partition | Cores | Walltime | Array |
|--------|-----------|-------|----------|-------|
| `run_morris.sh` | genoa | 48 | 2:00:00 | 0-7%8 |
| `run_ofat.sh` | genoa | 48 | 2:00:00 | 0 |
| `run_sobol.sh` | genoa | 48 | 8:00:00 | 0-15%8 |
| `benchmark.sh` | genoa | 48 | 2:00:00 | — |

Re-submit failed chunks only:

```bash
sbatch --array=3 snellius/run_sobol.sh
```

Row-level checkpoints in `sensitivity/<method>/partials/checkpoints/` allow safe resume within a chunk.

## Production campaign (optional, larger)

For full `r=96` Morris + 13-level OFAT + `N=512` Sobol, use `prepare_morris_ofat.sh` and edit `sobol.json` before `prepare_sobol.sh`.

## Notes

- Use `genoa` compute nodes, not `staging`.
- BLAS threads pinned to 1 (`OMP_NUM_THREADS=1`).
- Archive `sensitivity/` before scratch purge if running from `/scratch-shared/`.
