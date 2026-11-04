#!/usr/bin/env bash
#SBATCH --job-name=superteacher-hoi80
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
# Independent requested small evaluation; periodic complete protected_eval is unchanged.
set -euo pipefail
: "${SUPERMIMIC_ROOT:?frozen evaluation source required}"
: "${SUPERMIMIC_RUNTIME:?runtime required}"
: "${ISAACLAB_PATH:?IsaacLab source required}"
: "${TRAINING_BUNDLE:?checkpoint training bundle required}"
: "${EVAL_BUNDLE:?intact protected evaluation bundle required}"
: "${CANONICAL_HOI80_BUNDLE:?historical fixed registry required}"
: "${CHECKPOINT:?frozen checkpoint required}"
: "${CHECKPOINT_SHA256:?pinned checkpoint hash required}"
: "${CHECKPOINT_ITERATION:?pinned iteration required}"
: "${SOURCE_CONTENT_SHA256:?pinned evaluation source required}"
: "${OUTPUT:?new output required}"
[[ $# == 0 && ! -e "$OUTPUT" ]]
export VIRTUAL_ENV="$SUPERMIMIC_RUNTIME" PATH="$SUPERMIMIC_RUNTIME/bin:$PATH"
export PYTHONPATH="$SUPERMIMIC_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_NO_SYNC=1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WARP_CACHE_PATH="$SUPERMIMIC_ROOT/.cache/hoi80/${SLURM_JOB_ID:-local}/warp"
export MPLCONFIGDIR="$SUPERMIMIC_ROOT/.cache/hoi80/${SLURM_JOB_ID:-local}/mpl"
cd "$SUPERMIMIC_ROOT"
source cluster/hoi_v1_runtime.sh
hoi_v1_short_tmpdir
mkdir -p "$MPLCONFIGDIR"
python - <<'PY'
import os, json, torch
from hoi.data.manifest import sha256_file
from hoi.v1.config import source_identity
assert sha256_file(os.environ['CHECKPOINT']) == os.environ['CHECKPOINT_SHA256']
assert source_identity()['content_sha256'] == os.environ['SOURCE_CONTENT_SHA256']
c = torch.load(os.environ['CHECKPOINT'], map_location='cpu', weights_only=False)
assert c['iteration'] == int(os.environ['CHECKPOINT_ITERATION'])
print(json.dumps({'frozen_checkpoint_verified': True, 'iteration': c['iteration'],
                  'job_id': os.environ.get('SLURM_JOB_ID'), 'optimizer_updates': 0}), flush=True)
PY
exec python -m hoi.v1.evaluate --bundle "$EVAL_BUNDLE" --training-bundle "$TRAINING_BUNDLE" \
  --checkpoint "$CHECKPOINT" --output "$OUTPUT" --canonical-hoi80-bundle "$CANONICAL_HOI80_BUNDLE" \
  --device cuda:0 --hand-collision --self-collision
