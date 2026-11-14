#!/bin/bash
#SBATCH --job-name=hoi-mixed-eval
#SBATCH --partition=batch
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
REQUEST=${1:?Specify the checkpoint-bound mixed evaluation request}
SOURCE=${2:?Specify the frozen mixed source}
RUNTIME=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128
export VIRTUAL_ENV="$RUNTIME" PATH="$RUNTIME/bin:$PATH"
export ISAACLAB_PATH=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi/IsaacLab
export PYTHONPATH="$SOURCE" SUPERMIMIC_ROOT="$SOURCE" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$SOURCE"
mapfile -t INPUTS < <(python - "$REQUEST" <<'PY'
import json,sys
from hoi_tables.mixed_monitor import validate_request
r=json.load(open(sys.argv[1])); validate_request(r)
for key in ('output','training_root','training_source','iteration','parent_run_id','helper_root'):
    print(r[key])
for key in ('manifest','asset_bundle','geometry_bundle','shape_bank','reference_contacts','canonical'):
    print(r['paths'][key])
PY
)
OUTPUT=${INPUTS[0]}
mkdir "$OUTPUT" "$OUTPUT/hoi80" "$OUTPUT/tables20"
export PYTHONPATH="$SOURCE:${INPUTS[5]}"
export TMPDIR=$(mktemp -d /tmp/hoi-mixeval.XXXXXXXX)
trap 'rmdir "$TMPDIR" 2>/dev/null || true' EXIT
export WARP_CACHE_PATH="$OUTPUT/cache/warp" TORCHINDUCTOR_CACHE_DIR="$OUTPUT/cache/inductor"
mkdir -p "$WARP_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR"
for asset in {0..7}; do
    # HOI80 uses the exact frozen training code. The tables recorder is
    # released separately so existing training and checkpoint hashes hold.
    (cd "${INPUTS[2]}" && PYTHONPATH="${INPUTS[2]}:${INPUTS[5]}" SUPERMIMIC_ROOT="${INPUTS[2]}" \
    python -m hoi_tables.evaluate_hoi80 --mixed-request "$REQUEST" --batch-clips --checkpoint-root "${INPUTS[1]}" \
        --training-source "${INPUTS[2]}" --iteration "${INPUTS[3]}" --parent-run-id "${INPUTS[4]}" \
        --helper-root "${INPUTS[5]}" --manifest "${INPUTS[6]}" --asset-bundle "${INPUTS[7]}" \
        --geometry-bundle "${INPUTS[8]}" --shape-bank "${INPUTS[9]}" --reference-contacts "${INPUTS[10]}" \
        --canonical "${INPUTS[11]}" --asset "$asset" --output "$OUTPUT/hoi80/asset_0$asset") \
        > "$OUTPUT/hoi80/asset_0$asset.log" 2>&1
done
for asset in 0 1; do
    python -m hoi_tables.mixed_eval "$REQUEST" --table-asset "$asset" > "$OUTPUT/tables20/asset_0$asset.log" 2>&1
done
python -m hoi_tables.mixed_eval "$REQUEST" --aggregate
