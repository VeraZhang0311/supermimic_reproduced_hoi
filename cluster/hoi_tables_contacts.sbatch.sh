#!/bin/bash
#SBATCH --job-name=hoi-tables-contacts
#SBATCH --partition=batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
ROOT=${1:?Specify the tables campaign root}
SOURCE_ROOT=${2:?Specify the frozen contact-preparation source}
OUTPUT="$ROOT/formal_inputs01"
mkdir "$OUTPUT"
export VIRTUAL_ENV=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128
export PATH="$VIRTUAL_ENV/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$SOURCE_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SOURCE_ROOT"
python -m unittest tests.test_hoi_tables_contact_cache -v
python -m hoi_tables.prepare_contacts --manifest "$ROOT/input/manifest.json" \
  --asset-bundle "$ROOT/input/assets.json" --output "$OUTPUT/reference_contacts.pt" \
  --workers "$SLURM_CPUS_PER_TASK" --threads 1 --chunk-size 32 | tee "$OUTPUT/preparation.log"
