#!/bin/bash
# Generate Sobol design matrix from snellius/problems/sobol.json.
# Edit that file first to choose the parameter subset from Morris/OFAT screening.
set -euo pipefail

source venv/bin/activate

PROBLEM=snellius/problems/sobol.json
N=256

python sensitivity.py sample --method sobol --problem-file "$PROBLEM" --N "$N"

echo "Sobol: $(python -c "import numpy as np; print(len(np.load('sensitivity/sobol/samples.npy')))") rows"
python -c "import json; p=json.load(open('sensitivity/sobol/problem.json')); print('Parameters:', ', '.join(p['names']))"
