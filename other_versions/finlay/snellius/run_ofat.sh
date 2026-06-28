#!/bin/bash
#SBATCH -p genoa
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --array=0
#SBATCH -o logs/ofat_%A_%a.out
#SBATCH -e logs/ofat_%A_%a.err

# Stage 2: OFAT response curves (9 params, 7 levels -> 64 rows).
# Submit after Morris is underway or complete: sbatch snellius/run_ofat.sh

set -euo pipefail
mkdir -p logs

source venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python sensitivity.py run --method ofat \
    --chunk "$SLURM_ARRAY_TASK_ID" --n-chunks 1 \
    --reps 8 --steps 250 --width 50 --height 50 --n-agents 40
