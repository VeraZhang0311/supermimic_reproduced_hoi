#!/usr/bin/env bash
#SBATCH --job-name=hoi-object-cook
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
# Resource partition/QOS and verified CUDA-compatible nodes are selected by the
# caller. This script performs geometry cooking only, never training/acceptance.
set -euo pipefail
: "${SUPERMIMIC_ROOT:?Set the frozen independent source checkout}"
: "${SUPERMIMIC_RUNTIME:?Set the existing read-only native runtime}"
: "${ISAACLAB_PATH:?Set the existing read-only IsaacLab source}"
: "${COOKING_PLAN:?Set the hash-bound native cooking plan.json}"
index="${TASK_INDEX:-${SLURM_ARRAY_TASK_ID:?Set TASK_INDEX or SLURM_ARRAY_TASK_ID}}"
r=$(realpath "$SUPERMIMIC_ROOT")
cd "$r"
export VIRTUAL_ENV="$SUPERMIMIC_RUNTIME" PATH="$SUPERMIMIC_RUNTIME/bin:$PATH"
export UV_NO_SYNC=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$r${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${NATIVE_COOK_THREADS:-3}" MKL_NUM_THREADS="${NATIVE_COOK_THREADS:-3}"
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
printf 'cooking_threads=%s allocator=%s\n' "$OMP_NUM_THREADS" "$PYTORCH_ALLOC_CONF"
cache="${NATIVE_CACHE_ROOT:-$r/.cache/hoi-object-cook/${SLURM_JOB_ID:-manual}/$index}"
export WARP_CACHE_PATH="$cache/warp" TORCHINDUCTOR_CACHE_DIR="$cache/inductor" MPLCONFIGDIR="$cache/mpl"
source "$r/cluster/hoi_v1_runtime.sh"
hoi_v1_short_tmpdir
mkdir -p "$MPLCONFIGDIR"
python "${COOKING_DRIVER:-$r/hoi/data/native_cooking.py}" run --plan "$COOKING_PLAN" --task-index "$index"
