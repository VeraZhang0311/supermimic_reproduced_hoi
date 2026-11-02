#!/usr/bin/env bash
#SBATCH --job-name=hoi-superteacher-eval
#SBATCH --gres=gpu:l40:1
#SBATCH --partition=batch
#SBATCH --account=liu-account
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
# Set the actual partition/account/qos in the independent experiment launcher or sbatch.
# One GPU evaluates all protected clips sequentially by asset; no 80-case truncation.
set -euo pipefail
: "${SUPERMIMIC_ROOT:?Set the frozen teacher source}"
: "${SUPERMIMIC_RUNTIME:?Set the read-only runtime}"
: "${ISAACLAB_PATH:?Set the read-only IsaacLab source}"
: "${TRAINING_BUNDLE:?Set the exact checkpoint training bundle}"
: "${EVAL_BUNDLE:?Set the complete protected_eval bundle}"
: "${CHECKPOINT:?Set the immutable checkpoint}"
: "${OUTPUT:?Set a new evaluation output directory}"
: "${EVALUATION_DISPATCH:?Set the source-bound periodic evaluation request}"
[[ $# == 0 ]] || { echo 'Extra evaluation CLI overrides are forbidden' >&2; exit 2; }
r=$(realpath "$SUPERMIMIC_ROOT")
cd "$r"
[[ ! -e "$OUTPUT" ]] || { echo 'Existing evaluation results must be preserved' >&2; exit 2; }
export VIRTUAL_ENV="$SUPERMIMIC_RUNTIME" PATH="$SUPERMIMIC_RUNTIME/bin:$PATH"
export UV_NO_SYNC=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Destination and optional HOI_WANDB_SDK_PATH are explicitly inherited from dispatch.
export WARP_CACHE_PATH="$r/.cache/teacher-eval/${SLURM_JOB_ID:-local}/warp"
export MPLCONFIGDIR="$r/.cache/teacher-eval/${SLURM_JOB_ID:-local}/mpl"
source "$r/cluster/hoi_v1_runtime.sh"
hoi_v1_short_tmpdir
mkdir -p "$MPLCONFIGDIR"
python -m hoi.v1.periodic_eval --verify "$EVALUATION_DISPATCH"
python -m hoi.v1.evaluate --bundle "$EVAL_BUNDLE" --training-bundle "$TRAINING_BUNDLE"   --checkpoint "$CHECKPOINT" --output "$OUTPUT" --device cuda:0 --hand-collision --self-collision
if [[ "${WANDB_SYNC:-0}" == 1 ]]; then
  python -m hoi.v1.track --evaluation "$OUTPUT/evaluation.json" --output "$OUTPUT/tracking" --online
fi
