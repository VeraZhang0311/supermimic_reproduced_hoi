#!/usr/bin/env bash
#SBATCH --job-name=sm-hoi80-rubberhands
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
r=/mnt/projects/ll/tianzizh/supermimic_hoi_zeroshot_handcollision
b=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi
export VIRTUAL_ENV="$r/envs/kitless-cu128" PATH="$r/envs/kitless-cu128/bin:$PATH"
export ISAACLAB_PATH="$r/IsaacLab" UV_NO_SYNC=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export WARP_CACHE_PATH="$r/.cache/warp" TORCHINDUCTOR_CACHE_DIR="$r/.cache/inductor"
export MPLCONFIGDIR="$r/.cache/mpl" TMPDIR="$r/tmp"
cd "$r"
git diff --exit-code HEAD -- hoi supermimic
test "$(git branch --show-current)" = codex/hoi80-rubber-hand-collision
test -s results/hoi80_inputs/manifest.json
out="$r/results/hoi80_stock14000_handcollision-$SLURM_JOB_ID"
python -m hoi.run --prepared "$r/results/hoi80_inputs" \
 --checkpoint "$b/logs/lafan1_stock_seed0/ckpt_014000.pt" --output "$out"
PYTHONPATH=/mnt/projects/ll/tianzizh/SuperMimic_HOI/envs/tracking-py312/lib/python3.12/site-packages \
 "$r/tools/python-portable/cpython-3.12.4-linux-x86_64-gnu/bin/python3.12" "$r/hoi/track.py" "$out/evaluation.json" \
 --run-id zeroshot-hoi80-stock14000-rubberhands-20261022
