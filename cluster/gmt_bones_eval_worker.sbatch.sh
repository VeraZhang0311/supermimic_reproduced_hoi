#!/bin/bash
#SBATCH --job-name=gmt-bones-eval512
#SBATCH --partition=batch
#SBATCH --account=liu-account
#SBATCH --qos=normal
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/projects/ll/tianzizh/gmt_bones26k_20261106_02/eval-%j.log
set -euo pipefail
ROOT=/mnt/projects/ll/tianzizh/gmt_bones26k_20261106_02
SOURCE="$ROOT/source"
RUN=/mnt/projects/ll/tianzizh/runs/gmt_bones_filtered_from26k_20k_20261106_02
CHECKPOINT= OUTPUT=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --source-root) test "$2" = "$SOURCE"; shift 2;;
    --run-dir) test "$2" = "$RUN"; shift 2;;
    --checkpoint) CHECKPOINT=$2; shift 2;;
    --output) OUTPUT=$2; shift 2;;
    *) echo 'Unexpected evaluation argument' >&2; exit 2;;
  esac
done
test -n "$CHECKPOINT" && test -n "$OUTPUT"
export VIRTUAL_ENV=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128
export ISAACLAB_PATH=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi/IsaacLab
export PATH="$VIRTUAL_ENV/bin:/usr/local/bin:/usr/bin:/bin" PYTHONPATH="$SOURCE"
export SUPERMIMIC_ROOT="$SOURCE" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_NO_SYNC=1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WARP_CACHE_PATH="$ROOT/eval_cache/warp" TORCHINDUCTOR_CACHE_DIR="$ROOT/eval_cache/inductor"
export MPLCONFIGDIR="$ROOT/eval_cache/mpl" CUDA_CACHE_PATH="$ROOT/eval_cache/cuda"
export TMPDIR
TMPDIR=$(mktemp -d /tmp/gmt-bones-eval-"$SLURM_JOB_ID".XXXXXX)
mkdir -p "$WARP_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
cd "$SOURCE"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python tools/gmt_eval_worker.py --source-root "$SOURCE" --run-dir "$RUN" \
  --checkpoint "$CHECKPOINT" --output "$OUTPUT" --eval-envs 256 --device cuda:0 \
  --expected-clips 512 --eval-motion-dir "$ROOT/reference/eval512"
