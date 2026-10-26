from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(os.environ.get("SUPERMIMIC_ROOT", Path(__file__).resolve().parents[2])).resolve()
from .stage import verify


def main():
    root, stage = verify()
    operations = Path(os.environ['OPERATIONS_ROOT']).resolve()
    manifest_path = operations/'operations.json'
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == os.environ['EXPECTED_OPERATIONS_SHA256']
    manifest = json.loads(manifest_path.read_text())
    assert str(operations) == manifest['root']
    assert manifest['stage_sha256'] == os.environ['EXPECTED_STAGE_SHA256']
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((operations/name).read_bytes()).hexdigest() == digest

    complete = json.loads((root/'results/v11_validation_complete.json').read_text())
    assert complete['passed'] and complete['source_identity'] == stage['source_identity']
    for prefix in ['native_evidence', 'resource_report']:
        assert hashlib.sha256(Path(complete[prefix]).read_bytes()).hexdigest() == complete[prefix+'_sha256']
    resource = json.loads(Path(complete['resource_report']).read_text())
    assert resource['passed'] and resource['actual_optimizer_steps'] == 0 and resource['all_model_parameters_unchanged']
    assert resource['config'] == stage['production_config'] and resource['source_identity'] == stage['source_identity']
    assert resource['noop_optimizer_step_calls'] == {'pi': 1, 'wm': 1, 'critic': 20}
    output = root/'results/hoi_v1_1_object_rms_scratch_20k'
    assert not output.exists()
    args = [sys.executable, '-m', 'hoi.v1.train', '--config', str(root/'configs/hoi_v1_1_observation_matching.json'),
            '--bundle', stage['train_bundle'], '--native-evidence', complete['native_evidence'],
            '--eval-bundle', stage['eval_bundle'], '--from-scratch', '--hand-collision', '--self-collision',
            '--robot-mjcf', stage['robot_mjcf'], '--evaluation-runtime', os.environ['SUPERMIMIC_RUNTIME'],
            '--evaluation-isaaclab', os.environ['ISAACLAB_PATH'], '--evaluation-launcher', str(root/'cluster/hoi_v1_1_evaluate.sbatch.sh')]
    preflight = root/'results/v11_formal_preflight'
    if not preflight.exists():
        subprocess.run(args + ['--output', str(preflight), '--preflight'], cwd=root, check=True)
    plan = json.loads((preflight/'plan.json').read_text())
    assert plan['source_identity'] == stage['source_identity'] and plan['config'] == stage['production_config']
    assert plan['actor_observation_contract'] == stage['actor_observation_contract']
    assert plan['native_gate']['passed'] and plan['native_dr_gate']['passed'] and not plan['training_started']
    assert not plan['smoke_deviations'] and plan['initialization_request']['from_scratch']
    assert not plan['policy_update_schedule']['wm_quality_gate_enabled']
    assert plan['periodic_evaluation']['every_iterations'] == 5000
    subprocess.run(args + ['--output', str(output)], cwd=root, check=True)


if __name__ == '__main__':
    main()
