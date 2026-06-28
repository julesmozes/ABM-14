#!/bin/bash
#SBATCH -p rome
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -t 04:30:00
#SBATCH --array=0-31%10
#SBATCH -o logs/sobol_%A_%a.out
#SBATCH -e logs/sobol_%A_%a.err

set -euo pipefail
mkdir -p logs

source venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python sensitivity.py run --method sobol \
    --chunk "$SLURM_ARRAY_TASK_ID" --n-chunks 32 \
    --reps 5 --steps 225 --width 50 --height 50 --n-agents 40
