#!/bin/bash
#SBATCH -p staging
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -t 02:00:00
#SBATCH --array=0-7%8
#SBATCH -o logs/morris_%A_%a.out
#SBATCH -e logs/morris_%A_%a.err

set -euo pipefail
mkdir -p logs

source venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python sensitivity.py run --method morris \
    --chunk "$SLURM_ARRAY_TASK_ID" --n-chunks 8 \
    --reps 10 --steps 250 --width 50 --height 50 --n-agents 40
