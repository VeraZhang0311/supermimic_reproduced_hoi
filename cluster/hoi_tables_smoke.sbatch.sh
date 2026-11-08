#!/bin/bash
#SBATCH --job-name=hoi-tables-smoke
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=hoi-tables-smoke-%j.log
set -euo pipefail
ROOT=${1:?Specify the new tables experiment root}
SOURCE_ROOT=${2:?Specify a frozen source snapshot}
test -f "$ROOT/input/preparation_report.json"
test -f "$SOURCE_ROOT/hoi_tables/train.py"
RUN="$ROOT/smoke_${SLURM_JOB_ID:?GPU smoke requires Slurm}"
mkdir "$RUN"
export VIRTUAL_ENV=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128
export ISAACLAB_PATH=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi/IsaacLab
export PATH="$VIRTUAL_ENV/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$SOURCE_ROOT" SUPERMIMIC_ROOT="$SOURCE_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_NO_SYNC=1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WARP_CACHE_PATH="$RUN/cache/warp" TORCHINDUCTOR_CACHE_DIR="$RUN/cache/inductor"
export MPLCONFIGDIR="$RUN/cache/mpl" CUDA_CACHE_PATH="$RUN/cache/cuda" TMPDIR="$RUN/tmp"
export WANDB_MODE=disabled
mkdir -p "$WARP_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH" "$TMPDIR"
if [[ -n "${HOI_TABLES_WARP_CACHE_SEED:-}" ]]; then
  python - "$ROOT" "$HOI_TABLES_WARP_CACHE_SEED" "$WARP_CACHE_PATH" <<'PY'
from pathlib import Path
import shutil,sys
root,source,target=map(lambda p:Path(p).resolve(),sys.argv[1:])
if not source.is_relative_to(root) or not target.is_relative_to(root) or source==target:
    raise ValueError('Private cache copy must stay inside this tables campaign')
shutil.copytree(source,target,dirs_exist_ok=True)
PY
fi
cd "$SOURCE_ROOT"
finish() { code=$?; printf '%s\n' "$code" > "$RUN/exitcode"; exit "$code"; }
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$RUN/gpu.txt"
scontrol show job "$SLURM_JOB_ID" > "$RUN/slurm.txt"
sha256sum supermimic/*.py hoi_residual/*.py hoi_tables/*.py hoi_tables/v11/*.py > "$RUN/source.sha256"
python -m unittest discover -s tests -p 'test_hoi_tables*.py' -v > "$RUN/cpu_tests.log" 2>&1
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  -m hoi_tables.train --shared-gpu --smoke --num-envs 16 \
  --manifest "$ROOT/input/manifest.json" --asset-bundle "$ROOT/input/assets.json" \
  --shape-bank "$ROOT/input/shape_bank.json" --output "$RUN/training" 2>&1 | tee "$RUN/native.log"
sha256sum --check "$RUN/source.sha256" > "$RUN/source_unchanged.txt"
python - "$RUN" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]);reports=[json.loads((p/f'training/rank_{i:02d}/report.json').read_text()) for i in range(2)]
assert all(r['passed'] and r['smoke'] and r['world_size']==2 for r in reports)
assert {r['asset'] for r in reports}=={'largetable','smalltable'}
assert all(r['feature_schema']['wm_input_dim']==313 and r['feature_schema']['wm_output_dim']==82 for r in reports)
assert all(len(r['native_contacts']['channels'])==5 and r['horizon']==32 for r in reports)
assert all(r['native_object']['passed'] and r['model_parameters_and_normalizers_synchronized'] for r in reports)
assert all(all(r['models_changed'].values()) and r['resets']>0 for r in reports)
(p/'smoke_report.json').write_text(json.dumps({'passed':True,'scope':'two-asset shared privileged teacher, native physics and three optimizer smoke; no convergence claim','reports':reports},indent=2))
print('PASS: both tables, shared model/RMS, native physics, 313/82/contact5/H32 and WM-only isolation')
PY
