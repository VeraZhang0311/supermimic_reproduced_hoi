#!/usr/bin/env bash
#SBATCH --job-name=superteacher-sm-eval
#SBATCH --partition=batch
#SBATCH --account=liu-account
#SBATCH --qos=normal
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${SUPERMIMIC_ROOT:?frozen evaluation source required}"
: "${SUPERMIMIC_RUNTIME:?runtime required}"
: "${ISAACLAB_PATH:?IsaacLab required}"
: "${TRAINING_BUNDLE:?original training bundle required}"
: "${EVAL_BUNDLE:?complete protected bundle required}"
: "${SELECTION_MANIFEST:?frozen development selection required}"
: "${SELECTION_SHA256:?selection file hash required}"
: "${CHECKPOINT:?immutable checkpoint required}"
: "${CHECKPOINT_SHA256:?checkpoint hash required}"
: "${CHECKPOINT_ITERATION:?iteration required}"
: "${SOURCE_CONTENT_SHA256:?source hash required}"
: "${OUTPUT:?new output required}"
[[ $# == 0 && ! -e "$OUTPUT" ]]
export VIRTUAL_ENV="$SUPERMIMIC_RUNTIME" PATH="$SUPERMIMIC_RUNTIME/bin:$PATH"
export PYTHONPATH="$SUPERMIMIC_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_NO_SYNC=1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WARP_CACHE_PATH="$SUPERMIMIC_ROOT/.cache/sm-eval/${SLURM_JOB_ID:-local}/warp"
export MPLCONFIGDIR="$SUPERMIMIC_ROOT/.cache/sm-eval/${SLURM_JOB_ID:-local}/mpl"
cd "$SUPERMIMIC_ROOT"
source cluster/hoi_v1_runtime.sh
hoi_v1_short_tmpdir
mkdir -p "$MPLCONFIGDIR"
python - <<'PY'
import os, json, torch
from hoi.data.manifest import sha256_file
from hoi.v1.config import source_identity
assert sha256_file(os.environ['CHECKPOINT']) == os.environ['CHECKPOINT_SHA256']
assert sha256_file(os.environ['SELECTION_MANIFEST']) == os.environ['SELECTION_SHA256']
assert source_identity()['content_sha256'] == os.environ['SOURCE_CONTENT_SHA256']
c = torch.load(os.environ['CHECKPOINT'], map_location='cpu', weights_only=False)
assert c['iteration'] == int(os.environ['CHECKPOINT_ITERATION'])
print(json.dumps({'frozen_checkpoint_verified': True, 'iteration': c['iteration'],
                  'asset_batch_size': 4, 'optimizer_updates': 0}), flush=True)
PY
exec python -m hoi.v1.evaluate --bundle "$EVAL_BUNDLE" --training-bundle "$TRAINING_BUNDLE" \
  --checkpoint "$CHECKPOINT" --output "$OUTPUT" --selection-manifest "$SELECTION_MANIFEST" \
  --asset-batch-size 4 --device cuda:0 --hand-collision --self-collision
