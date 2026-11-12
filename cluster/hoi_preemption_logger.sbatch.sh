#!/bin/bash
#SBATCH --job-name=ultra-recovery-logger
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
#SBATCH --no-requeue
set -euo pipefail
ROOT=${1:?Ultra recovery controller}
ATTEMPT=${2:?continuation attempt}
RUNTIME=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128
TELEMETRY=/mnt/projects/ll/tianzizh/gmt_l40_20261105_01/telemetry_venv
COMPAT=/mnt/projects/ll/tianzizh/hoi_residual_ppo_20261106_01/source_smoke08/tools
mapfile -t P < <("$RUNTIME/bin/python" -c 'import json,sys;v=json.load(open(sys.argv[1]));print(v["helpers"]);print(v.get("eval_source",v["source"]))' "$ROOT/policy.json")
export PYTHONPATH="$COMPAT:$TELEMETRY/lib/python3.12/site-packages:$RUNTIME/lib/python3.12/site-packages:${P[1]}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export WANDB_MODE=online WANDB_DISABLE_CODE=true WANDB_CONSOLE=off WANDB_SILENT=true
exec "$TELEMETRY/bin/python" "${P[0]}/preemption.py" logger --root "$ROOT" --attempt "$ATTEMPT"
