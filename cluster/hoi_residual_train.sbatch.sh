#!/bin/bash
#SBATCH --job-name=hoi-residual-teacher
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40:8
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --no-requeue
#SBATCH --output=hoi-residual-teacher-%j.log
set -euo pipefail
# Explicit submission is required; the development task submits only smoke.
ROOT=${1:?Specify a prepared experiment root}
SOURCE_ROOT=${2:?Specify an immutable source snapshot}
ITERATIONS=${3:-100000}
RUN="$ROOT/train_${SLURM_JOB_ID:?Use the scheduler}"
test ! -e "$RUN"
test -f "$ROOT/input/ckpt_026000.pt"
test -f "$ROOT/input/ultra_manifest.json"
test -f "$ROOT/input/assets/assets.json"
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
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  -m hoi_residual.train \
  --manifest "$ROOT/input/ultra_manifest.json" \
  --asset-bundle "$ROOT/input/assets/assets.json" \
  --gmt-checkpoint "$ROOT/input/ckpt_026000.pt" \
  --output "$RUN" --iterations "$ITERATIONS" --num-envs 512
