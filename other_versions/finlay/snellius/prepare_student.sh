#!/bin/bash
# Student-scale OFAT + Sobol (after Morris screening).
# Morris: benchmark r=48 (480 rows) + analyze --allow-partial on Snellius.
# Do NOT resample morris here — it would overwrite existing checkpoints.
set -euo pipefail

source venv/bin/activate

SCREEN=snellius/problems/screen.json
SOBOL=snellius/problems/sobol.json
OFAT_LEVELS=7
SOBOL_N=256

python sensitivity.py sample --method ofat --problem-file "$SCREEN" --levels "$OFAT_LEVELS"
python sensitivity.py sample --method sobol --problem-file "$SOBOL" --N "$SOBOL_N"

echo "OFAT:  $(python -c "import numpy as np; print(len(np.load('sensitivity/ofat/samples.npy')))") rows (expect 64 = 1 + 9×7)"
echo "Sobol: $(python -c "import numpy as np; print(len(np.load('sensitivity/sobol/samples.npy')))") rows"
python -c "import json; p=json.load(open('sensitivity/sobol/problem.json')); print('Sobol params:', ', '.join(p['names']))"
