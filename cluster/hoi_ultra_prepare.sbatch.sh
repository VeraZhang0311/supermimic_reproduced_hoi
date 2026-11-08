#!/bin/bash
#SBATCH --job-name=hoi-ultra-v11-cache
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -euo pipefail
SOURCE_ROOT=${1:?Specify the committed source checkout}
CONFIG=${2:?Specify the Ultra experiment JSON}
RUNTIME=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128
export VIRTUAL_ENV="$RUNTIME" PATH="$RUNTIME/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$SOURCE_ROOT" SUPERMIMIC_ROOT="$SOURCE_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mapfile -t INPUTS < <(python - "$CONFIG" <<'PY'
import json,sys
v=json.load(open(sys.argv[1]));assert v['profile']=='ultra'
for k in ('manifest','geometry_bundle','reference_contacts'):print(v['paths'][k])
print(v['evaluation']['canonical']);print(v['evaluation']['helper_root'])
from pathlib import Path
print(Path(v['paths']['geometry_bundle']).resolve().parent.parent/'bundle_eval80/bundle.json')
PY
)
test ! -e "${INPUTS[2]}"
export PYTHONPATH="$SOURCE_ROOT:${INPUTS[4]}"
cd "$SOURCE_ROOT"
# Repackage the already-computed v1.1 task intent, with complete SHA/T+1 checks.
# This path never recomputes hand/object reference geometry or runs FK.
python -m hoi_tables.v11_reference_contacts --manifest "${INPUTS[0]}" \
 --train-bundle "${INPUTS[1]}" --eval-bundle "${INPUTS[5]}" \
 --canonical "${INPUTS[3]}" --helper-root "${INPUTS[4]}" --output "${INPUTS[2]}"
