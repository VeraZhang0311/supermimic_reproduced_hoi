#!/usr/bin/env bash
# Submit with 2 L40, 16 CPUs, 64 GiB RAM and an explicit time limit.
set -euo pipefail
: "${SUPERMIMIC_ROOT:?frozen training source required}"
: "${SUPERMIMIC_RUNTIME:?prepared runtime required}"
: "${ISAACLAB_PATH:?prepared IsaacLab required}"
: "${COMPACT_DATASET:?frozen dataset required}"
: "${NATIVE_EVIDENCE:?accepted native evidence required}"
: "${CAPACITY_REPORT:?accepted L40 report required}"
: "${SOURCE_CONTENT_SHA256:?training source identity required}"
: "${RUN_DIR:?fresh training directory required}"
: "${HOI_WANDB_RUN_NAME:?explicit teacher_v1 timestamp required}"
: "${HOI_WANDB_SDK_PATH:?verified tracking SDK required}"
: "${WANDB_PROJECT_VERIFIED_PRIVATE:?verify project before submission}"
[[ "$WANDB_PROJECT_VERIFIED_PRIVATE" == superteacher ]]
[[ ! -e "$RUN_DIR" ]]
export VIRTUAL_ENV="$SUPERMIMIC_RUNTIME" PATH="$SUPERMIMIC_RUNTIME/bin:$PATH"
export PYTHONPATH="$SUPERMIMIC_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_NO_SYNC=1
export WANDB_ENTITY=tianzizh-university-of-pennsylvania WANDB_PROJECT=superteacher WANDB_SYNC=1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$SUPERMIMIC_ROOT"
source cluster/hoi_v1_runtime.sh
hoi_v1_short_tmpdir
python - <<'PY'
import hashlib, json, os
from pathlib import Path
from hoi.v1.config import source_identity, bundle_identity
from hoi.v1.train import read_config
from hoi.v1.evaluate import _read_bundle_header
r = json.loads(Path(os.environ['CAPACITY_REPORT']).read_text())
assert r['passed'] and r['resource_measurement_complete'] and not r['finalization_errors']
assert r['optimizer_steps'] == 0 and r['formal_training_started'] is False
assert r['scope'] == 'full_training_bundle_resource_validation'
assert r['probe']['num_envs'] == 4096 and r['probe']['expected_groups'] == 56
assert r['assets_per_process'] == 28 and r['collector_devices'] == ['cuda:1']
assert r['worker_startup_concurrency'] == 2 and r['worker_cpu_threads'] == 3
assert r['real_start_mix']['all_static_groups_present']
assert {'production_wm_backward','full_imagination_backward','production_full_critic_target_values',
        'production_critic_backward'} <= {p['name'] for p in r['phases']}
source = source_identity()
assert source['content_sha256'] == os.environ['SOURCE_CONTENT_SHA256']
baseline = r['source_identity']
assert baseline['content_sha256'] == 'fab38afc03836ff450c272255ce592f26bb248084da58d1835ad32150693ce77'
assert set(source['files']) == set(baseline['files'])
# The only code delta from the capacity-tested source is the explicit W&B label.
assert {p for p in source['files'] if source['files'][p] != baseline['files'][p]} == {'hoi/v1/track.py'}
config, _ = read_config(Path('configs/hoi_v1_privileged_teacher.json'))
assert config.to_dict() == r['production_config']
bundle = _read_bundle_header(Path(os.environ['COMPACT_DATASET'])/'bundle_train/bundle.json')
assert bundle_identity(bundle) == r['data_identity']
assert len(bundle['clips']) == 6537
mjcf = Path(bundle['robot_fk']['mjcf_path'])
assert hashlib.sha256(mjcf.read_bytes()).hexdigest() == bundle['robot_fk']['mjcf_sha256']
robot = Path('data/g1_description')
if not robot.exists():
    robot.parent.mkdir(parents=True, exist_ok=True)
    robot.symlink_to(mjcf.parent, target_is_directory=True)
from hoi.hand_collision import HAND_SHA
for side, sha in HAND_SHA.items():
    assert hashlib.sha256((robot/'meshes'/(side+'_rubber_hand.STL')).read_bytes()).hexdigest() == sha
print(json.dumps({'capacity_verified':True,'source_sha':source['content_sha256'],
                  'run_name':os.environ['HOI_WANDB_RUN_NAME']}),flush=True)
PY
exec python -m hoi.v1.train --from-scratch --config configs/hoi_v1_privileged_teacher.json \
  --bundle "$COMPACT_DATASET/bundle_train/bundle.json" \
  --eval-bundle "$COMPACT_DATASET/bundle_protected_eval/bundle.json" \
  --native-evidence "$NATIVE_EVIDENCE" --device cuda:0 --collector-devices cuda:1 \
  --assets-per-process 28 --worker-startup-concurrency 2 --hand-collision --self-collision \
  --robot-mjcf "$SUPERMIMIC_ROOT/data/g1_description/g1_29dof.xml" \
  --evaluation-launcher "$SUPERMIMIC_ROOT/cluster/hoi_compact_teacher_evaluate.sbatch.sh" \
  --evaluation-runtime "$SUPERMIMIC_RUNTIME" --evaluation-isaaclab "$ISAACLAB_PATH" --output "$RUN_DIR"
