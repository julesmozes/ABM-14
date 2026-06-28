#!/bin/bash
#SBATCH -p genoa
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/benchmark_%j.out
#SBATCH -e logs/benchmark_%j.err

# Measure wall time per model evaluation to size job arrays and walltime.
# Run once before the campaign: sbatch snellius/benchmark.sh

set -euo pipefail
mkdir -p logs

source venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

N_ROWS=48
REPS=6
STEPS=250
PROBLEM=snellius/problems/screen.json

python sensitivity.py sample --method morris --problem-file "$PROBLEM" --r "$N_ROWS"

START=$(date +%s)
python sensitivity.py run --method morris --chunk 0 --n-chunks 1 \
    --reps "$REPS" --steps "$STEPS" --width 50 --height 50 --n-agents 40
END=$(date +%s)

ELAPSED=$((END - START))
DESIGN_ROWS=$((N_ROWS * 10))  # r=48, 9 params -> 480 Morris rows
EVAL_COUNT=$((DESIGN_ROWS * REPS))
SEC_PER_EVAL=$(python -c "print(f'{$ELAPSED / $EVAL_COUNT:.3f}')")

echo "Benchmark: ${DESIGN_ROWS} rows x ${REPS} reps = ${EVAL_COUNT} evals in ${ELAPSED}s"
echo "Seconds per eval: ${SEC_PER_EVAL}"
echo ""
echo "Student campaign sizing (Morris r=48, OFAT 7 levels, Sobol N=256 5p):"
python -c "
elapsed = $ELAPSED
evals = $EVAL_COUNT
workers = 48
campaigns = [
    ('ofat', 64 * 8),
    ('sobol_5p', 1792 * 6),
]
for name, total_evals in campaigns:
    core_hours = total_evals * (elapsed / evals) / 3600
    wall_hours = core_hours / workers
    print(f'  {name}: {total_evals} evals -> ~{core_hours:.0f} core-h (~{wall_hours:.1f}h wall at {workers} workers)')
"
