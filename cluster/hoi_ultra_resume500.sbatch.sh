#!/bin/bash
#SBATCH --job-name=ultra-resume500
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --exclude=ll-l40-1.grasp.maas
#SBATCH --gres=gpu:l40:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=96:00:00
#SBATCH --no-requeue
set -euo pipefail
CAMPAIGN=${1:?Specify a fresh campaign directory}
REPOSITORY=${2:?Specify the committed source checkout or frozen snapshot}
CONFIG=${3:?Specify the tables or Ultra experiment JSON}
RUNTIME=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128
TELEMETRY=/mnt/projects/ll/tianzizh/gmt_l40_20261105_01/telemetry_venv
COMPAT=/mnt/projects/ll/tianzizh/hoi_residual_ppo_20261106_01/source_smoke08/tools
export VIRTUAL_ENV="$RUNTIME" PATH="$RUNTIME/bin:/usr/local/bin:/usr/bin:/bin"
export ISAACLAB_PATH=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi/IsaacLab
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_NO_SYNC=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
export PYTHONPATH="$REPOSITORY" SUPERMIMIC_ROOT="$REPOSITORY"
PREP=(--config "$CONFIG" --campaign "$CAMPAIGN" --source "$REPOSITORY")
if [[ -n "${HOI_WORKERS_PER_GPU:-}" ]]; then PREP+=(--workers-per-gpu "$HOI_WORKERS_PER_GPU"); fi
# No GPU optimizer is constructed during immutable input/source preparation.
python -m hoi_tables.launch "${PREP[@]}"
SOURCE_ROOT="$CAMPAIGN/source"
RUN="$CAMPAIGN/training"
export PYTHONPATH="$SOURCE_ROOT" SUPERMIMIC_ROOT="$SOURCE_ROOT"
mapfile -t SETTINGS < <(python - "$CAMPAIGN/preparation.json" <<'PY'
import json,sys
v=json.load(open(sys.argv[1]))
for x in (v['profile'],v['workers'],v['workers_per_gpu'],','.join(v['assets'])):print(x)
PY
)
PROFILE=${SETTINGS[0]}
python - "$CAMPAIGN/preparation.json" "$CAMPAIGN/experiment.json" <<'PY'
import json,sys
v=json.load(open(sys.argv[1]));c=json.load(open(sys.argv[2]))
assert v['profile']=='ultra' and v['workers']==8 and v['workers_per_gpu']==4 and v['gpus']==2
assert v['start_iteration']==501 and v['resume']['iteration']==500
assert c['solver']['ls_iterations']==200 and c['paths']['resume_bundle']
PY
WORKERS=${SETTINGS[1]}
ASSETS=${SETTINGS[3]}
python - "$CAMPAIGN/preparation.json" <<'PY'
import json,sys,torch
v=json.load(open(sys.argv[1]))
if torch.cuda.device_count()!=v['gpus']:
    raise ValueError(f"This profile requires {v['gpus']} L40 GPUs; set sbatch --gres and workers_per_gpu consistently")
PY
EVAL_MODULE=hoi_tables.eval_monitor
if [[ "$PROFILE" == ultra ]]; then EVAL_MODULE=hoi_tables.hoi80_monitor; fi
finish() { code=$?; printf '%s\n' "$code" > "$RUN/training_exitcode"; exit "$code"; }
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WARP_CACHE_PATH="$CAMPAIGN/cache/warp" TORCHINDUCTOR_CACHE_DIR="$CAMPAIGN/cache/inductor"
export MPLCONFIGDIR="$CAMPAIGN/cache/mpl" CUDA_CACHE_PATH="$CAMPAIGN/cache/cuda" TMPDIR="$CAMPAIGN/tmp"
export WANDB_DISABLE_CODE=true WANDB_CONSOLE=off WANDB_SILENT=true WANDB_MODE=online
mkdir -p "$WARP_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH" "$TMPDIR"
COMMON=(--experiment-config "$CAMPAIGN/experiment.json")
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$WORKERS" \
 -m hoi_tables.train "${COMMON[@]}" --capacity-only --recovery-validation-steps 4096 --output "$CAMPAIGN/capacity" 2>&1 | tee "$CAMPAIGN/capacity.log"
python - "$CAMPAIGN" "$WORKERS" <<'PY'
from pathlib import Path
import json,sys
r=Path(sys.argv[1]);reports=[json.loads((r/f'capacity/rank_{i:02d}/capacity_report.json').read_text()) for i in range(int(sys.argv[2]))]
assert len(reports)==8
assert all(x['passed'] and x['production_configuration'] and x['optimizer_updates']==0 for x in reports)
for i in range(8):
    folder=r/f'capacity/rank_{i:02d}'
    guard=json.loads((folder/'guard_validation.json').read_text())
    stress=json.loads((folder/'recovery_stress_report.json').read_text())
    receipt=json.loads((folder/'resume_receipt.json').read_text())
    prefill=json.loads((folder/'resume_prefill.json').read_text())
    assert guard['passed'] and guard['native_cuda_array'] and guard['all_native_flags_remain_fatal'] and guard['nonfinite_remains_fatal']
    assert stress['passed'] and stress['real_control_steps']>=4096 and stress['optimizer_updates']==0 and stress['models_and_RMS_unchanged']
    assert stress['native']['overflow_world_count']==0 and stress['solver_health']['ls_iterations']==200
    assert receipt['iteration']==500 and receipt['next_iteration']==501 and receipt['model_optimizer_RSI_contact_counts_restored']
    assert prefill['passed'] and prefill['models_and_RMS_unchanged'] and not prefill['repeat_wm_warmup']
(r/'capacity_passed.json').write_text(json.dumps({'passed':True,'workers':len(reports)},indent=2))
PY

# W&B is created only after all8 GPU guards, stress rollouts and backwards pass.
mapfile -t REGISTRATION < <(python - "$PROFILE" <<'PY'
from datetime import datetime
from zoneinfo import ZoneInfo
import secrets,sys
print(secrets.token_hex(4))
prefix='ultra_boxes_v1-resume500-ls200-'
print(prefix+datetime.now(ZoneInfo('America/New_York')).strftime('%Y%m%d-%H%M%S'))
PY
)
PYTHONPATH="$COMPAT:$TELEMETRY/lib/python3.12/site-packages:$RUNTIME/lib/python3.12/site-packages:$SOURCE_ROOT" \
 "$TELEMETRY/bin/python" -m hoi_tables.telemetry --root "$RUN" --id "${REGISTRATION[0]}" \
 --name "${REGISTRATION[1]}" --register-only > "$CAMPAIGN/wandb_registration.log" 2>&1
CONTROL="$CAMPAIGN/unified_logging"
EVAL_CONTROL="$CAMPAIGN/evaluation"
mkdir "$CONTROL"
MONITOR=$(sbatch --parsable --job-name="hoi-unified-${REGISTRATION[0]}" --output="$CONTROL/monitor-%j.log" \
 "$SOURCE_ROOT/cluster/hoi_unified_monitor.sbatch.sh" "$SOURCE_ROOT" "$CONTROL" "$RUN" "$CAMPAIGN" \
 "$SOURCE_ROOT" "$SOURCE_ROOT" "$EVAL_CONTROL" "$SLURM_JOB_ID" "$ASSETS" "$EVAL_MODULE")
printf '%s\n' "$MONITOR" > "$CONTROL/slurm_job_id"
for ((i=0;i<180;i++)); do
 if [[ -f "$CONTROL/unified_error.json" ]]; then echo 'Unified owning-run logger failed' >&2; exit 1; fi
 if [[ -f "$CONTROL/eval_error.json" ]]; then echo 'Evaluation scheduler failed before training' >&2; exit 1; fi
 if [[ -f "$CONTROL/unified_ready.json" && -f "$CONTROL/unified_state.json" ]]; then break; fi
 sleep 1
done
python - "$CONTROL" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
ready=json.loads((root/'unified_ready.json').read_text())
state=json.loads((root/'unified_state.json').read_text())
assert state['id']==ready['id'] and state['eval_failed'] is False
assert not (root/'unified_error.json').exists() and not (root/'eval_error.json').exists()
PY
cd "$SOURCE_ROOT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$CAMPAIGN/gpu.txt"
scontrol show job "$SLURM_JOB_ID" > "$CAMPAIGN/slurm.txt"
sha256sum supermimic/*.py hoi_residual/*.py hoi_tables/*.py hoi_tables/v11/*.py > "$CAMPAIGN/runtime.sha256"
# Fresh processes restore ORIGINAL checkpoint500 again; validation state is discarded.
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$WORKERS" \
 -m hoi_tables.train "${COMMON[@]}" --output "$RUN" 2>&1 | tee "$CAMPAIGN/training.log"
sha256sum --check "$CAMPAIGN/runtime.sha256" > "$CAMPAIGN/runtime_unchanged.txt"
