# Snellius sensitivity analysis workflow

`ssh snellius` lands in your project working directory. All commands below assume you are at the repo root.

## 1. One-time environment setup

```bash
bash snellius/setup_env.sh
source venv/bin/activate
```

## 2. Benchmark (recommended first)

Measure seconds per evaluation and estimate walltime before the full campaign:

```bash
mkdir -p logs
sbatch snellius/benchmark.sh
# tail -f logs/benchmark_*.out
```

Adjust `--array`, `--n-chunks`, and `-t` in the run scripts if benchmark times differ from estimates.

## 3. Generate design matrices (login node)

```bash
source venv/bin/activate
python sensitivity.py sample --method sobol --N 2048
python sensitivity.py sample --method morris --r 500
python sensitivity.py sample --method ofat --levels 21
```

## 4. Submit evaluation job arrays

```bash
sbatch snellius/run_sobol.sh    # 32 chunks, 512 rows each
sbatch snellius/run_morris.sh   # 8 chunks
sbatch snellius/run_ofat.sh     # 1 chunk
```

Monitor: `squeue -u $USER`

## 5. Analyze (login node, after all partials complete)

```bash
python sensitivity.py analyze --method sobol
python sensitivity.py analyze --method morris
python sensitivity.py analyze --method ofat
```

Outputs land in `results/sobol/`, `results/morris/`, `results/ofat/`.

## Local smoke test

```bash
python sensitivity.py sample --method sobol --N 16
python sensitivity.py run --method sobol --chunk 0 --n-chunks 1 --reps 2 --steps 50
python sensitivity.py analyze --method sobol
```

## Notes

- Each array task uses the `staging` partition (32 cores) with multiprocessing internally. CPU thin nodes (`rome`/`genoa`) require a separate budget product.
- BLAS threads are pinned to 1 to avoid oversubscription.
- Partial CSVs are written to `results/<method>/partials/`; safe to re-submit individual chunks.
- Archive `results/` before the 14-day scratch purge if running from `/scratch-shared/`.
