#!/bin/bash
#SBATCH --job-name=hoi-mix-labels
#SBATCH --partition=batch
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
set -euo pipefail
ROOT=${1:?Specify the campaign directory}
SOURCE=${2:?Specify the frozen source directory}
RUNTIME=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128
export PATH="$RUNTIME/bin:$PATH" PYTHONPATH="$SOURCE" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$SOURCE"
mapfile -t INPUTS < <(python - "$ROOT/experiment.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); cfg=json.loads(p.read_text())
print(p.parent / cfg['table_selection'])
print(cfg['table_assets'])
print(json.load(open(cfg['ultra_experiment']))['paths']['native_hands'])
PY
)
python -m hoi_tables.prepare --selection "${INPUTS[0]}" --references-only --output "$ROOT/tables"
python -m hoi_tables.prepare_contacts --manifest "$ROOT/tables/manifest.json" --asset-bundle "${INPUTS[1]}" \
  --native-hands "${INPUTS[2]}" --workers "$SLURM_CPUS_PER_TASK" --output "$ROOT/tables/reference_contacts.pt"
touch "$ROOT/tables/prepared"
