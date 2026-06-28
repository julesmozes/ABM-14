#!/bin/bash
# Production-scale Morris + OFAT (r=96, 13 levels).
# Student campaign: use benchmark.sh (r=48) + prepare_student.sh instead.
set -euo pipefail

source venv/bin/activate

PROBLEM=snellius/problems/screen.json

python sensitivity.py sample --method morris --problem-file "$PROBLEM" --r 96
python sensitivity.py sample --method ofat --problem-file "$PROBLEM" --levels 13

echo "Morris: $(python -c "import numpy as np; print(len(np.load('sensitivity/morris/samples.npy')))") rows"
echo "OFAT:   $(python -c "import numpy as np; print(len(np.load('sensitivity/ofat/samples.npy')))") rows"
