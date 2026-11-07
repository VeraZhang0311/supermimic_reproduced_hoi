#!/bin/bash
#SBATCH --job-name=hoi80-watch
#SBATCH --partition=batch
#SBATCH --account=liu-account
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=3-02:00:00
#SBATCH --requeue
set -euo pipefail
ROOT=${1:?experiment root}
JOB=${2:?training job ID}
CODE=${3:?evaluation code directory}
export HOI_TRAINING_SOURCE=${4:-$ROOT/source_smoke08}
test -f "$HOI_TRAINING_SOURCE/tools/gmt_auto_eval.py"
export PYTHONPATH="$HOI_TRAINING_SOURCE/tools" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec python3 "$CODE/tools/hoi80_watch.py" "$ROOT" "$JOB" "$CODE"
