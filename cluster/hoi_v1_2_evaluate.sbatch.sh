#!/usr/bin/env bash
#SBATCH --job-name=sm-hoi-v12-eval80
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${SUPERMIMIC_ROOT:?Set the frozen independent v1.2 source checkout}"
: "${SUPERMIMIC_RUNTIME:?Set the prepared read-only runtime}"
: "${ISAACLAB_PATH:?Set the read-only IsaacLab source}"
: "${TRAINING_BUNDLE:?Set the exact checkpoint training bundle}"
: "${EVAL_BUNDLE:?Set the full canonical eval80 bundle}"
: "${CHECKPOINT:?Set the immutable complete checkpoint snapshot}"
: "${OUTPUT:?Set a fresh evaluation output directory}"
: "${EVALUATION_DISPATCH:?Set the frozen periodic-evaluation dispatch request}"
root=$(realpath "$SUPERMIMIC_ROOT")
for old in supermimic_reproduced_hoi supermimic_reproduced_hoi_selfcollision supermimic_reproduced_hoi_handcollision supermimic_hoi_zeroshot supermimic_hoi_zeroshot_handcollision SuperMimic_HOI; do
  test "$root" != "/mnt/projects/ll/tianzizh/$old" || { echo 'Protected historical checkout' >&2; exit 2; }
done
cd "$root"
test ! -e "$OUTPUT" || { echo 'Existing evaluation results must be preserved' >&2; exit 2; }
export VIRTUAL_ENV="$SUPERMIMIC_RUNTIME" PATH="$SUPERMIMIC_RUNTIME/bin:$PATH"
export UV_NO_SYNC=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WARP_CACHE_PATH="$root/.cache/hoi-v12-eval/${SLURM_JOB_ID}/warp"
export TORCHINDUCTOR_CACHE_DIR="$root/.cache/hoi-v12-eval/${SLURM_JOB_ID}/inductor"
export MPLCONFIGDIR="$root/.cache/hoi-v12-eval/${SLURM_JOB_ID}/mpl"
source "$root/cluster/hoi_v1_runtime.sh"
hoi_v1_short_tmpdir
mkdir -p "$MPLCONFIGDIR"
python -m hoi.v1.periodic_eval --verify "$EVALUATION_DISPATCH"
python -m hoi.v1.evaluate --bundle "$EVAL_BUNDLE" --training-bundle "$TRAINING_BUNDLE" \
  --checkpoint "$CHECKPOINT" --output "$OUTPUT" --hand-collision --self-collision "$@"
if [[ "${WANDB_SYNC:-0}" == 1 ]]; then
  python -m hoi.v1.track --evaluation "$OUTPUT/evaluation.json" --output "$OUTPUT/tracking" --online
fi
