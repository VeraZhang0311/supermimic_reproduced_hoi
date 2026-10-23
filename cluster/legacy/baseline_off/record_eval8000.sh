#!/usr/bin/env bash
set -euo pipefail
r=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi
export VIRTUAL_ENV="$r/envs/kitless-cu128" ISAACLAB_PATH="$r/IsaacLab" UV_NO_SYNC=1
export PATH="$VIRTUAL_ENV/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export WARP_CACHE_PATH="$r/.cache/record-eval8000/warp" TORCHINDUCTOR_CACHE_DIR="$r/.cache/record-eval8000/inductor"
export MPLCONFIGDIR="$r/.cache/record-eval8000/mpl" TMPDIR="$r/tmp"
cd "$r"
python cluster/record_eval8000.py --checkpoint "$r/logs/lafan1_stock_seed0/ckpt_008000.pt" --output "$1"
