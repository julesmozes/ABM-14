#!/bin/bash
#SBATCH -p staging
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -t 01:00:00
#SBATCH -o logs/benchmark_%j.out
#SBATCH -e logs/benchmark_%j.err

# Measure wall time per model evaluation to size job arrays and walltime.
# Run once before the full campaign:
#   sbatch snellius/benchmark.sh

set -euo pipefail
mkdir -p logs

source venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

N_ROWS=64
REPS=10
STEPS=250

python sensitivity.py sample --method sobol --N "$N_ROWS"

START=$(date +%s)
python sensitivity.py run --method sobol --chunk 0 --n-chunks 1 \
    --reps "$REPS" --steps "$STEPS" --width 50 --height 50 --n-agents 40
END=$(date +%s)

ELAPSED=$((END - START))
EVAL_COUNT=$((N_ROWS * REPS))
SEC_PER_EVAL=$(python -c "print(f'{$ELAPSED / $EVAL_COUNT:.3f}')")

echo "Benchmark: ${N_ROWS} rows x ${REPS} reps = ${EVAL_COUNT} evals in ${ELAPSED}s"
echo "Seconds per eval: ${SEC_PER_EVAL}"
echo ""
echo "Sizing guide (High tier, 32 workers/node on staging):"
echo "  Sobol: 16384 rows x 10 reps = 163840 evals"
echo "  Morris: 3500 rows x 10 reps = 35000 evals"
echo "  OFAT: 127 rows x 10 reps = 1270 evals"
python -c "
elapsed = $ELAPSED
evals = $EVAL_COUNT
workers = 32
for name, total_evals in [('sobol', 163840), ('morris', 35000), ('ofat', 1270)]:
    core_hours = total_evals * (elapsed / evals) / 3600
    node_hours = core_hours / workers
    print(f'  {name}: ~{core_hours:.0f} core-hours (~{node_hours:.1f} node-hours at full parallelism)')
"
