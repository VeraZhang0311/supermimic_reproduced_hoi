#!/bin/bash
#SBATCH --job-name=hoi-ultra-tables
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --gres=gpu:l40:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --no-requeue
set -euo pipefail
ROOT=${1:?Specify the prepared mixed campaign}
SOURCE=${2:?Specify the frozen mixed source}
RUN=${3:?Specify a fresh training output}
RUNTIME=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128
TELEMETRY=/mnt/projects/ll/tianzizh/gmt_l40_20261105_01/telemetry_venv
COMPAT=/mnt/projects/ll/tianzizh/hoi_residual_ppo_20261106_01/source_smoke08/tools
export VIRTUAL_ENV="$RUNTIME" PATH="$RUNTIME/bin:$PATH"
export ISAACLAB_PATH=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi/IsaacLab
export PYTHONPATH="$SOURCE" SUPERMIMIC_ROOT="$SOURCE" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export WANDB_DISABLE_CODE=true WANDB_CONSOLE=off WANDB_SILENT=true OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WARP_CACHE_PATH="$ROOT/cache/warp" TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/inductor"
export TMPDIR=$(mktemp -d /tmp/hoi-mixtrain.XXXXXXXX)
test -f "$ROOT/tables/prepared"
mkdir "$RUN"
finish() { code=$?; printf '%s\n' "$code" > "$RUN/training_exitcode"; rmdir "$TMPDIR" 2>/dev/null || true; exit "$code"; }
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
mkdir -p "$WARP_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR"
cd "$SOURCE"
python - "$ROOT" "$RUN" <<'PY'
import secrets,sys
from pathlib import Path
from hoi_tables.evaluate_hoi80 import validate_source
from hoi_tables.mixed import ASSETS, read_experiment, input_hashes
from hoi_tables.networks import feature_schema
from hoi_tables.telemetry import atomic_json
e=read_experiment(sys.argv[1]); p=Path(sys.argv[2])
validate_source(Path.cwd())
atomic_json(p/'run_metadata.json', {'global_config': e.config.to_dict(), 'schema': feature_schema(),
    'branch': 'hoi-retarget/tables-boxes-v1.0', 'asset_count': 10, 'gpus': 2, 'gpu_type': 'L40',
    'total_env': e.config.num_envs, 'train_clips': 2854, 'protected_eval': 20, 'evaluation_clips': 100,
    'from_scratch': False, 'manifest_sha256': input_hashes(e)['table_manifest'],
    'experiment': {'profile': 'ultra_tables', 'assets': list(ASSETS), 'lane_counts': list(e.lanes),
        'sampling': e.raw['sampling'], 'evaluation': e.raw['evaluation'], 'initialized_iteration': 8000,
        'overlap_collection': e.execution['overlap_collection']}})
(p/'run_id').write_text(secrets.token_hex(4)+'\n')
PY
RUN_ID=$(cat "$RUN/run_id")
PYTHONPATH="$COMPAT:$TELEMETRY/lib/python3.12/site-packages:$RUNTIME/lib/python3.12/site-packages:$SOURCE" \
    "$TELEMETRY/bin/python" -m hoi_tables.telemetry --root "$RUN" --id "$RUN_ID" --register-only
CONTROL="$RUN/logging"
EVAL_CONTROL="${RUN%/}.evaluation"
mkdir "$CONTROL"
monitor=$(sbatch --parsable --output="$CONTROL/monitor-%j.log" "$SOURCE/cluster/hoi_unified_monitor.sbatch.sh" \
    "$SOURCE" "$CONTROL" "$RUN" "$ROOT" "$SOURCE" "$SOURCE" "$EVAL_CONTROL" "$SLURM_JOB_ID" mixed hoi_tables.mixed_monitor)
printf '%s\n' "$monitor" > "$CONTROL/slurm_job_id"
for ((i=0;i<300;i++)); do
    if [[ -f "$CONTROL/unified_error.json" || -f "$CONTROL/eval_error.json" ]]; then
        exit 1
    fi
    if [[ -f "$CONTROL/unified_ready.json" ]]; then break; fi
    sleep 1
done
test -f "$CONTROL/unified_ready.json"
python -m hoi_tables.mixed_train --campaign "$ROOT" --output "$RUN"
