#!/bin/bash
#SBATCH --job-name=ultra-auto-continue
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=96:00:00
#SBATCH --no-requeue
set -euo pipefail
ROOT=${1:?Ultra recovery controller}
ATTEMPT=${2:?fresh continuation attempt}
RUNTIME=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128
mapfile -t P < <("$RUNTIME/bin/python" -c 'import json,sys;v=json.load(open(sys.argv[1]));print(v["helpers"]);print(v["source"])' "$ROOT/policy.json")
export VIRTUAL_ENV="$RUNTIME" PATH="$RUNTIME/bin:/usr/local/bin:/usr/bin:/bin"
export ISAACLAB_PATH=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi/IsaacLab
export PYTHONPATH="${P[1]}" SUPERMIMIC_ROOT="${P[1]}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_NO_SYNC=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WARP_CACHE_PATH="$ATTEMPT/cache/warp" TORCHINDUCTOR_CACHE_DIR="$ATTEMPT/cache/inductor"
export CUDA_CACHE_PATH="$ATTEMPT/cache/cuda" MPLCONFIGDIR="$ATTEMPT/cache/mpl" TMPDIR="$ATTEMPT/tmp"
mkdir -p "$WARP_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$TMPDIR"
export WANDB_MODE=disabled
exec python "${P[0]}/preemption.py" run --root "$ROOT" --attempt "$ATTEMPT"
