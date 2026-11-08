#!/bin/bash
#SBATCH --job-name=hoi-ultra-direct-inputs
#SBATCH --partition=batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
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
for k in ('manifest','asset_bundle','geometry_bundle','shape_bank','reference_contacts'):print(v['paths'][k])
print(v['evaluation']['canonical']);print(v['evaluation']['helper_root'])
PY
)
test ! -e "${INPUTS[4]}"
export PYTHONPATH="$SOURCE_ROOT:${INPUTS[6]}"
cd "$SOURCE_ROOT"
python -m hoi_tables.ultra preflight --manifest "${INPUTS[0]}" --asset-bundle "${INPUTS[1]}" \
 --geometry-bundle "${INPUTS[2]}" --shape-bank "${INPUTS[3]}" --canonical "${INPUTS[5]}"
python -m hoi_tables.ultra prepare-contacts --manifest "${INPUTS[0]}" --asset-bundle "${INPUTS[1]}" \
 --geometry-bundle "${INPUTS[2]}" --shape-bank "${INPUTS[3]}" --canonical "${INPUTS[5]}" \
 --output "${INPUTS[4]}" --workers "$SLURM_CPUS_PER_TASK" --threads 1
