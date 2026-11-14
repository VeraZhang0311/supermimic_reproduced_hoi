#!/bin/bash
#SBATCH --job-name=hoi-tables20-zero8000
#SBATCH --partition=batch
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
set -euo pipefail
REQUEST=${1:?Specify the protected zero-shot request}
SOURCE=${2:?Specify the frozen evaluator source}
RUNTIME=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128
export VIRTUAL_ENV="$RUNTIME" PATH="$RUNTIME/bin:$PATH"
export ISAACLAB_PATH=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi/IsaacLab
export PYTHONPATH="$SOURCE" SUPERMIMIC_ROOT="$SOURCE" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TMPDIR=$(mktemp -d /tmp/hoi-zeroeval.XXXXXXXX)
trap 'rmdir "$TMPDIR" 2>/dev/null || true' EXIT
export WARP_CACHE_PATH="$(dirname "$REQUEST")/warp_cache"
cd "$SOURCE"
for asset in 0 1; do
    python -m hoi_tables.mixed_zeroshot "$REQUEST" --table-asset "$asset" > "$(dirname "$REQUEST")/asset_${asset}.log" 2>&1
done
python -m hoi_tables.mixed_zeroshot "$REQUEST" --aggregate
