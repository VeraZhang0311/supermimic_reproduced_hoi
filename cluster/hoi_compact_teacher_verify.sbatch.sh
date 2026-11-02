#!/usr/bin/env bash
# Zero optimizer updates: verify the frozen compact data and real task sampling.
# Submit with an explicit GPU/CPU/memory allocation; this script never trains.
set -euo pipefail
: "${SUPERMIMIC_ROOT:?frozen source checkout required}"
: "${SUPERMIMIC_RUNTIME:?prepared runtime required}"
: "${ISAACLAB_PATH:?prepared IsaacLab required}"
: "${COMPACT_DATASET:?materialized compact dataset directory required}"
: "${PARENT_NATIVE_ROOT:?original full_native_acceptance_01 directory required}"
: "${VERIFY_ROOT:?fresh verification output required}"
: "${SOURCE_SNAPSHOT_SHA256:?frozen source manifest SHA required}"
export VIRTUAL_ENV="$SUPERMIMIC_RUNTIME"
export PATH="$SUPERMIMIC_RUNTIME/bin:/opt/slurm/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$SUPERMIMIC_ROOT" UV_NO_SYNC=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export WANDB_SYNC=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$SUPERMIMIC_ROOT"
source cluster/hoi_v1_runtime.sh
hoi_v1_short_tmpdir
python - <<'PY'
import hashlib, json, os
from pathlib import Path
root=Path(os.environ['SUPERMIMIC_ROOT'])
raw=(root/'source_snapshot.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()==os.environ['SOURCE_SNAPSHOT_SHA256']
for name,sha in json.loads(raw)['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
Path(os.environ['VERIFY_ROOT']).mkdir(parents=True,exist_ok=False)
PY
python -m hoi.v1.packed_probe --mode capacity --sampling-audit \
  --bundle "$COMPACT_DATASET/bundle_train/bundle.json" \
  --output "$VERIFY_ROOT/collection_probe" --num-envs "${PROBE_NUM_ENVS:-4096}" \
  --assets-per-process 28 --collector-devices cuda:0 cuda:1 \
  --worker-startup-concurrency 2 --steps "${PROBE_STEPS:-128}"
python -m hoi.data.compact_native_evidence \
  --bundle "$COMPACT_DATASET/bundle_train/bundle.json" \
  --parent-train-evidence "$PARENT_NATIVE_ROOT/train/native_evidence.json" \
  --parent-eval-evidence "$PARENT_NATIVE_ROOT/protected_eval/native_evidence.json" \
  --collection "$VERIFY_ROOT/collection_probe/collection/report.json" \
  --output "$VERIFY_ROOT/native_gate"
python -m hoi.v1.train --preflight --from-scratch \
  --config configs/hoi_v1_privileged_teacher.json \
  --bundle "$COMPACT_DATASET/bundle_train/bundle.json" \
  --eval-bundle "$COMPACT_DATASET/bundle_protected_eval/bundle.json" \
  --native-evidence "$VERIFY_ROOT/native_gate/native_evidence.json" \
  --hand-collision --self-collision --device cuda:0 \
  --collector-devices cuda:0 cuda:1 --assets-per-process 28 \
  --worker-startup-concurrency 2 --output "$VERIFY_ROOT/final_preflight" \
  > "$VERIFY_ROOT/final_preflight.stdout.json"
