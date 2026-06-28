#!/bin/bash
#SBATCH -p genoa
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH --array=0-15%8
#SBATCH -o logs/sobol_%A_%a.out
#SBATCH -e logs/sobol_%A_%a.err

# Stage 3: Sobol on 5-param subset (N=256 -> 1792 rows, 16 chunks).
# Run bash snellius/prepare_sobol.sh, then: sbatch snellius/run_sobol.sh

set -euo pipefail
mkdir -p logs

source venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python sensitivity.py run --method sobol \
    --chunk "$SLURM_ARRAY_TASK_ID" --n-chunks 16 \
    --reps 6 --steps 250 --width 50 --height 50 --n-agents 40
