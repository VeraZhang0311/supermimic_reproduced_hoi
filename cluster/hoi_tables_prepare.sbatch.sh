#!/bin/bash
#SBATCH --job-name=hoi-tables-prepare
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
set -euo pipefail
ROOT=${1:?Specify a new tables campaign root}
SOURCE_ROOT=${2:?Specify its frozen preparation source}
test -d "$ROOT" && test -d "$SOURCE_ROOT"
test ! -e "$ROOT/input"
export VIRTUAL_ENV=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128
export PATH="$VIRTUAL_ENV/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$SOURCE_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SOURCE_ROOT"
python -m unittest discover -s tests -p test_hoi_tables_data.py -v
python -m hoi_tables.prepare --selection "$SOURCE_ROOT/configs/hoi_tables_selection.json" --output "$ROOT/input"
