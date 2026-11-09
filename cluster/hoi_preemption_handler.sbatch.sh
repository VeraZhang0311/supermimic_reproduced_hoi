#!/bin/bash
#SBATCH --job-name=ultra-preempt-handler
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --requeue
set -euo pipefail
ROOT=${1:?Ultra recovery controller}
PARENT=${2:?ended Ultra training job}
# Slurm stores the script outside its submitted directory; use policy identity.
PY=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128/bin/python
mapfile -t P < <("$PY" -c 'import json,sys;v=json.load(open(sys.argv[1]));print(v["helpers"]);print(v["source"])' "$ROOT/policy.json")
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="${P[1]}" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$PY" "${P[0]}/preemption.py" handle --root "$ROOT" --parent "$PARENT"
