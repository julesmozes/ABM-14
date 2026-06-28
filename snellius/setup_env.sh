#!/bin/bash
# One-time setup on Snellius login node (run from project root).
set -euo pipefail

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

python -m venv venv

module purge
source venv/bin/activate
pip install -U pip
pip install -r snellius/requirements-hpc.txt

echo "Environment ready. Activate with: source venv/bin/activate"
