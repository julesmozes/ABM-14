#!/bin/bash
# End-to-end smoke test on Snellius login or compute node.
# Usage (from repo root): bash snellius/check.sh
set -euo pipefail

mkdir -p logs

if [[ ! -d venv ]]; then
    echo "ERROR: venv not found. Run: bash snellius/setup_env.sh"
    exit 1
fi

source venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PROBLEM=snellius/problems/screen.json
SMOKE_DIR=results/_smoke

echo "=== imports ==="
python -c "import mesa, numpy, pandas, SALib, tqdm; print('OK')"

echo "=== sample (tiny Morris + OFAT) ==="
python sensitivity.py sample --method morris --problem-file "$PROBLEM" --r 4 \
    --output-dir "$SMOKE_DIR"
python sensitivity.py sample --method ofat --problem-file "$PROBLEM" --levels 3 \
    --output-dir "$SMOKE_DIR"

echo "=== run (1 chunk each) ==="
python sensitivity.py run --method morris --chunk 0 --n-chunks 1 \
    --reps 2 --steps 30 --output-dir "$SMOKE_DIR"
python sensitivity.py run --method ofat --chunk 0 --n-chunks 1 \
    --reps 2 --steps 30 --output-dir "$SMOKE_DIR"

echo "=== resume check (should skip completed rows) ==="
python sensitivity.py run --method morris --chunk 0 --n-chunks 1 \
    --reps 2 --steps 30 --output-dir "$SMOKE_DIR"

echo "=== status ==="
python sensitivity.py status --method morris --output-dir "$SMOKE_DIR"
python sensitivity.py status --method ofat --output-dir "$SMOKE_DIR"

echo "=== analyze ==="
python sensitivity.py analyze --method morris --output-dir "$SMOKE_DIR"
python sensitivity.py analyze --method ofat --output-dir "$SMOKE_DIR"

echo "=== rerun trajectory ==="
python sensitivity.py rerun --method morris --row-index 1 --rep 0 --reps 2 --steps 30 \
    --output-dir "$SMOKE_DIR" --output "$SMOKE_DIR/morris/traj_test.csv"

echo ""
echo "All checks passed. Submit production jobs with:"
echo "  bash snellius/prepare_morris_ofat.sh"
echo "  sbatch snellius/run_morris.sh"
echo "  sbatch snellius/run_ofat.sh"
