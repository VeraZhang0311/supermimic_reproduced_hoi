#!/bin/bash
#SBATCH --job-name=hoi-tables-v1
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --gres=gpu:l40:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=96:00:00
#SBATCH --no-requeue
set -euo pipefail
ROOT=${1:?Specify the tables campaign root}
SOURCE_ROOT=${2:?Specify the frozen formal source}
RUN_NAME=${3:-formal_01}
[[ "$RUN_NAME" =~ ^formal_[0-9]+$ ]] || { echo 'Invalid fresh formal run name' >&2; exit 2; }
RUN="$ROOT/$RUN_NAME"
CONTACTS="$ROOT/formal_inputs01/reference_contacts.pt"
test -f "$RUN/wandb_registration.json"
test -f "$CONTACTS"
test ! -f "$RUN/training_exitcode"
mkdir "$RUN/trainer_claim"
export VIRTUAL_ENV=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128
export ISAACLAB_PATH=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi/IsaacLab
export PATH="$VIRTUAL_ENV/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$SOURCE_ROOT" SUPERMIMIC_ROOT="$SOURCE_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_NO_SYNC=1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WARP_CACHE_PATH="$RUN/cache/warp" TORCHINDUCTOR_CACHE_DIR="$RUN/cache/inductor"
export MPLCONFIGDIR="$RUN/cache/mpl" CUDA_CACHE_PATH="$RUN/cache/cuda" TMPDIR="$RUN/tmp"
export WANDB_DISABLE_CODE=true WANDB_CONSOLE=off WANDB_MODE=online
mkdir -p "$WARP_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH" "$TMPDIR"
MONITOR_PID=""
finish() {
  result=$?
  trap - EXIT TERM INT
  printf '%s\n' "$result" > "$RUN/training_exitcode"
  if [[ -n "$MONITOR_PID" ]]; then
    for ((attempt=0; attempt<60; attempt++)); do
      if ! kill -0 "$MONITOR_PID" 2>/dev/null; then break; fi
      sleep 1
    done
    if kill -0 "$MONITOR_PID" 2>/dev/null; then kill "$MONITOR_PID"; fi
    wait "$MONITOR_PID" || true
  fi
  exit "$result"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
cd "$SOURCE_ROOT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$RUN/gpu.txt"
scontrol show job "$SLURM_JOB_ID" > "$RUN/slurm.txt"
sha256sum supermimic/*.py hoi_residual/*.py hoi_tables/*.py hoi_tables/v11/*.py > "$RUN/source.sha256"
sha256sum "$ROOT/input/manifest.json" "$ROOT/input/assets.json" "$ROOT/input/shape_bank.json" "$CONTACTS" > "$RUN/inputs.sha256"
# Copy only this campaign's completed smoke compilation cache into private paths.
python - "$ROOT" "${HOI_TABLES_WARP_CACHE_SEED:-$ROOT/smoke_1512175/cache/warp}" "$WARP_CACHE_PATH" <<'PY'
from pathlib import Path
import shutil,sys
root,source,target=map(lambda p:Path(p).resolve(),sys.argv[1:])
if not source.is_relative_to(root) or not target.is_relative_to(root) or source==target:
    raise ValueError('Compilation cache copy must stay in this campaign')
if source.is_dir(): shutil.copytree(source,target,dirs_exist_ok=True)
PY
TELEMETRY=/mnt/projects/ll/tianzizh/gmt_l40_20261105_01/telemetry_venv
COMPAT=/mnt/projects/ll/tianzizh/hoi_residual_ppo_20261106_01/source_smoke08/tools
RUN_ID=$(python - "$RUN/wandb_registration.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['id'])
PY
)
PYTHONPATH="$COMPAT:$TELEMETRY/lib/python3.12/site-packages:$VIRTUAL_ENV/lib/python3.12/site-packages:$SOURCE_ROOT" \
  "$TELEMETRY/bin/python" -m hoi_tables.telemetry --root "$RUN" --id "$RUN_ID" --idle-timeout 43200 \
  > "$RUN/wandb_monitor.log" 2>&1 &
MONITOR_PID=$!
for ((attempt=0; attempt<120; attempt++)); do
  if [[ -f "$RUN/wandb_error.json" ]]; then echo 'W&B monitor failed' >&2; exit 1; fi
  if [[ -f "$RUN/wandb_ready.json" ]]; then break; fi
  if ! kill -0 "$MONITOR_PID" 2>/dev/null; then echo 'W&B monitor exited before readiness' >&2; exit 1; fi
  sleep 1
done
test -f "$RUN/wandb_ready.json"
COMMON=(--manifest "$ROOT/input/manifest.json" --asset-bundle "$ROOT/input/assets.json"
        --shape-bank "$ROOT/input/shape_bank.json" --reference-contacts "$CONTACTS"
        --num-envs 4096 --iterations 20000)
# Both stages use the same two L40 GPUs. Validation has no optimizer updates;
# its process exits completely before new from-scratch models/replay are built.
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  -m hoi_tables.train "${COMMON[@]}" --capacity-only --output "$RUN/capacity_01" \
  2>&1 | tee "$RUN/capacity.log"
python - "$RUN" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
reports=[json.loads((root/f'capacity_01/rank_{i:02d}/capacity_report.json').read_text()) for i in range(2)]
assert all(r['passed'] and r['production_configuration'] and r['optimizer_updates']==0 for r in reports)
(root/'capacity_passed.json').write_text(json.dumps({'passed':True,'reports':reports},indent=2))
PY
sha256sum --check "$RUN/source.sha256" > "$RUN/source_validated.txt"
sha256sum --check "$RUN/inputs.sha256" > "$RUN/inputs_validated.txt"
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  -m hoi_tables.train "${COMMON[@]}" --output "$RUN" 2>&1 | tee "$RUN/training.log"
sha256sum --check "$RUN/source.sha256" > "$RUN/source_unchanged.txt"
