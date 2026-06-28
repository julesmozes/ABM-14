#!/bin/bash
#SBATCH -p genoa
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --array=0-7%8
#SBATCH -o logs/morris_%A_%a.out
#SBATCH -e logs/morris_%A_%a.err

# Stage 1: Morris broad screen (9 params, r=96 -> 960 rows, 8 parallel chunks).
# Submit first: sbatch snellius/run_morris.sh

set -euo pipefail
mkdir -p logs

source venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python sensitivity.py run --method morris \
    --chunk "$SLURM_ARRAY_TASK_ID" --n-chunks 8 \
    --reps 6 --steps 250 --width 50 --height 50 --n-agents 40
