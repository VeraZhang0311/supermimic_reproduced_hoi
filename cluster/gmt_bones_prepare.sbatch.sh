#!/bin/bash
#SBATCH --job-name=gmt-bones-prepare
#SBATCH --partition=batch
#SBATCH --account=liu-account
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/ll/tianzizh/gmt_bones26k_20261106_01/prepare-%j.log
set -euo pipefail
ROOT=/mnt/projects/ll/tianzizh/gmt_bones26k_20261106_01
RUNTIME=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128
export PYTHONPATH="$ROOT/source" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
cd "$ROOT/source"
"$RUNTIME/bin/python" "$ROOT/parent_verify.py"
"$RUNTIME/bin/python" tools/prepare_bones_filtered.py \
  --source /mnt/projects/ll/tianzizh/SuperMimic_HOI/data/bones_filtered_generic_2ca8074_packed \
  --output "$ROOT/reference/full" --chunk-frames 65536 \
  --eval-output "$ROOT/reference/eval512" --eval-clips 512 --eval-seed 0
