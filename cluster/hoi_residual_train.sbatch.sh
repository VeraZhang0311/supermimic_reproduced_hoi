#!/bin/bash
#SBATCH --job-name=hoi-residual-teacher
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --no-requeue
#SBATCH --output=hoi-residual-teacher-%j.log
set -euo pipefail
# Submit only after native smoke and full-data capacity validation pass.
ROOT=${1:?Specify a prepared experiment root}
SOURCE_ROOT=${2:?Specify an immutable source snapshot}
SHAPE_BANK=${HOI_SHAPE_BANK:-$ROOT/input/shape_bank_v2.json}
GMT_CHECKPOINT=${HOI_GMT_CHECKPOINT:-$ROOT/input/ckpt_004000.pt}
GMT_ITERATION=${HOI_GMT_ITERATION:-4000}
SHARED_GPU=${HOI_SHARED_GPU:-1}
case "$SHARED_GPU" in
  1) EXECUTION_ARGS=(--shared-gpu);;
  0) EXECUTION_ARGS=();;
  *) echo 'HOI_SHARED_GPU must be 0 or 1' >&2; exit 2;;
esac
# For eight dedicated GPUs, set HOI_SHARED_GPU=0 and submit with
# --gres=gpu:l40:8. The default keeps eight asset ranks on one L40.
ITERATIONS=${3:-100000}
REWARD_ARGS=()
if [[ $# -gt 3 ]]; then
  REWARD_ARGS=(--experiment-config "$4" --v11-train-bundle "${5:?Specify the v1.1 training bundle}")
fi
RUN="$ROOT/train_${SLURM_JOB_ID:?Use the scheduler}"
test ! -e "$RUN"
test -f "$GMT_CHECKPOINT"
test -f "$ROOT/input/ultra_manifest.json"
test -f "$ROOT/input/assets/assets.json"
test -f "$SHAPE_BANK"
export VIRTUAL_ENV=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128
export ISAACLAB_PATH=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi/IsaacLab
export PATH="$VIRTUAL_ENV/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$SOURCE_ROOT" SUPERMIMIC_ROOT="$SOURCE_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_NO_SYNC=1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WARP_CACHE_PATH="$ROOT/train_cache_${SLURM_JOB_ID}/warp"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/train_cache_${SLURM_JOB_ID}/inductor"
export MPLCONFIGDIR="$ROOT/train_cache_${SLURM_JOB_ID}/mpl"
export CUDA_CACHE_PATH="$ROOT/train_cache_${SLURM_JOB_ID}/cuda"
export TMPDIR="$ROOT/train_cache_${SLURM_JOB_ID}/tmp"
mkdir -p "$WARP_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH" "$TMPDIR"
cd "$SOURCE_ROOT"
sha256sum supermimic/*.py hoi_residual/*.py > "$ROOT/train_cache_${SLURM_JOB_ID}/source.sha256"
sha256sum "$GMT_CHECKPOINT" "$ROOT/input/ultra_manifest.json" "$ROOT/input/assets/assets.json" "$SHAPE_BANK" > "$ROOT/train_cache_${SLURM_JOB_ID}/inputs.sha256"
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  -m hoi_residual.train \
  --manifest "$ROOT/input/ultra_manifest.json" \
  --asset-bundle "$ROOT/input/assets/assets.json" \
  --shape-bank "$SHAPE_BANK" \
  --gmt-checkpoint "$GMT_CHECKPOINT" --gmt-iteration "$GMT_ITERATION" \
  --output "$RUN" --iterations "$ITERATIONS" --num-envs 512 "${EXECUTION_ARGS[@]}" "${REWARD_ARGS[@]}"
sha256sum --check "$ROOT/train_cache_${SLURM_JOB_ID}/source.sha256" > "$ROOT/train_cache_${SLURM_JOB_ID}/source_unchanged.txt"
sha256sum --check "$ROOT/train_cache_${SLURM_JOB_ID}/inputs.sha256" > "$ROOT/train_cache_${SLURM_JOB_ID}/inputs_unchanged.txt"
