#!/usr/bin/env bash
# Full learner graph + one collector GPU. No optimizer step or training run.
set -euo pipefail
: "${SUPERMIMIC_ROOT:?frozen source checkout required}"
: "${SUPERMIMIC_RUNTIME:?prepared runtime required}"
: "${ISAACLAB_PATH:?prepared IsaacLab required}"
: "${COMPACT_DATASET:?frozen compact dataset required}"
: "${NATIVE_EVIDENCE:?accepted compact native evidence required}"
: "${CAPACITY_ROOT:?fresh capacity output required}"
: "${SOURCE_SNAPSHOT_SHA256:?frozen source manifest SHA required}"
export VIRTUAL_ENV="$SUPERMIMIC_RUNTIME"
export PATH="$SUPERMIMIC_RUNTIME/bin:/opt/slurm/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$SUPERMIMIC_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_NO_SYNC=1
export WANDB_SYNC=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$SUPERMIMIC_ROOT"
source cluster/hoi_v1_runtime.sh
hoi_v1_short_tmpdir
python - <<'PY'
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

source = Path(os.environ['SUPERMIMIC_ROOT'])
snapshot = (source/'source_snapshot.json').read_bytes()
assert hashlib.sha256(snapshot).hexdigest() == os.environ['SOURCE_SNAPSHOT_SHA256']
for name, digest in json.loads(snapshot)['files'].items():
    assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest, name
output = Path(os.environ['CAPACITY_ROOT'])
output.mkdir(parents=True, exist_ok=False)
bundle_path = Path(os.environ['COMPACT_DATASET'])/'bundle_train/bundle.json'
from hoi.v1.evaluate import _read_bundle_header
from hoi.v1.gates import validate_native_evidence
from hoi.v1.config import source_identity
bundle = _read_bundle_header(bundle_path)
required = sorted({c['asset_id'] for c in bundle['clips']})
assert len(required) == 56 and len(bundle['clips']) == 6537 and bundle['partition'] == 'train'
gate = validate_native_evidence(os.environ['NATIVE_EVIDENCE'],
    expected_physics={'hand_collision': True, 'self_collision': True},
    required_asset_ids=required, expected_bundle_sha256=bundle['bundle_content_sha256'],
    expected_asset_profiles={a['asset_id']: a['physical_profile'] for a in bundle['assets']})
(output/'native_admission.json').write_text(json.dumps(gate, indent=2)+'\n')
assert gate['passed'], gate['blockers']
identity = source_identity()
assert gate['actual_collection']['source_content_sha256'] == identity['content_sha256']
mjcf = Path(bundle['robot_fk']['mjcf_path'])
assert hashlib.sha256(mjcf.read_bytes()).hexdigest() == bundle['robot_fk']['mjcf_sha256']
robot = source/'data/g1_description'
if not robot.exists() and not robot.is_symlink():
    robot.parent.mkdir(parents=True, exist_ok=True)
    robot.symlink_to(mjcf.parent, target_is_directory=True)
from hoi.hand_collision import HAND_SHA
for side, digest in HAND_SHA.items():
    assert hashlib.sha256((robot/'meshes'/(side+'_rubber_hand.STL')).read_bytes()).hexdigest() == digest
del bundle, gate, snapshot

required_phases = {'production_wm_backward', 'full_imagination_backward',
                   'production_full_critic_target_values', 'production_critic_backward'}
summary = {'schema': 'hoi_compact_capacity_comparison_v1', 'optimizer_steps': 0,
           'formal_training_started': False, 'learner_device': 'cuda:0',
           'collector_devices': ['cuda:1'], 'source_identity': identity,
           'default_num_envs': 4096, 'results': []}
for number in (4096, 2048):
    destination = output/('envs_'+str(number))
    command = [sys.executable, '-m', 'hoi.v1.resource_probe', '--from-scratch',
        '--config', 'configs/hoi_v1_privileged_teacher.json', '--bundle', str(bundle_path),
        '--output', str(destination), '--robot-mjcf', str(mjcf),
        '--hand-collision', '--self-collision', '--expected-groups', '56',
        '--collect-steps', '128', '--probe-num-envs', str(number),
        '--device', 'cuda:0', '--collector-devices', 'cuda:1',
        '--assets-per-process', '28', '--worker-startup-concurrency', '2',
        '--worker-cpu-threads', '3']
    print(json.dumps({'phase': 'capacity_start', 'num_envs': number, 'command': command}), flush=True)
    started = time.monotonic()
    with (output/('envs_'+str(number)+'.log')).open('x') as log:
        code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
    path = destination/'report.json'
    report = json.loads(path.read_text()) if path.is_file() else {}
    phases = {p['name'] for p in report.get('phases', [])}
    unchanged = source_identity() == identity
    passed = (code == 0 and report.get('passed') is True and unchanged
              and report.get('optimizer_steps') == 0 and report.get('formal_training_started') is False
              and required_phases <= phases and report.get('source_identity') == identity)
    row = {'num_envs': number, 'return_code': code, 'passed': passed,
           'source_unchanged': unchanged, 'report': str(path),
           'elapsed_seconds': time.monotonic()-started,
           'completed_phases': sorted(phases), 'error': report.get('error'),
           'finalization_errors': report.get('finalization_errors', [])}
    summary['results'].append(row)
    summary['passed'] = passed
    summary['passing_num_envs'] = number if passed else None
    summary['default_4096_passed'] = passed and number == 4096
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({'phase': 'capacity_end', **row}), flush=True)
    if passed:
        break
    # Only a cleanly finalized learner CUDA OOM triggers the authorized width
    # comparison. Coverage/physics/worker/measurement failures require diagnosis.
    error = report.get('error', {})
    retry_2048 = (number == 4096 and unchanged and not row['finalization_errors']
                  and 'models_and_references' in phases
                  and error.get('type') == 'OutOfMemoryError'
                  and 'CUDA out of memory' in error.get('message', ''))
    if not retry_2048:
        break
raise SystemExit(0 if summary['passed'] else 1)
PY
