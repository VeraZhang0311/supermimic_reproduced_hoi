"""Frozen-policy HOI80 evaluation with explicit physics and full recorded rollouts."""
from pathlib import Path
from dataclasses import asdict
import argparse
import datetime
import json
import os
import subprocess
import sys
import traceback
import uuid
from hoi.hand_collision import prepare as prepare_hands, assert_same_physics, LIMITATION
from supermimic.physics import add_physics_arguments, resolve_arguments, sha

ROOT = Path(__file__).resolve().parents[1]


def aggregate(items):
    return {'count': len(items), 'phc_pass': sum(c['phc_pass'] for c in items),
            'object_pose_pass': sum(c['object_pose_pass'] for c in items),
            'body_and_object_pose_pass': sum(c['body_and_object_pose_pass'] for c in items),
            'supermimic_n_failed': sum(c['first_supermimic_failure_step'] >= 0 for c in items),
            'supermimic_norm_return': sum(c['supermimic_norm_return'] for c in items) / len(items)}


def paired_comparison(report, baseline):
    old = {c['id']: c for c in baseline['cases']}
    assert set(old) == {c['id'] for c in report['cases']}
    return {'baseline_aggregate': baseline['aggregate'],
            'baseline_physics': baseline.get('physics', 'Legacy report: inspect its frozen source identity'),
            'aggregate_deltas': {k: report['aggregate'][k] - baseline['aggregate'][k]
                                 for k in report['aggregate'] if k != 'count'},
            'per_case': [{'id': c['id'], 'return_delta': c['supermimic_norm_return'] - old[c['id']]['supermimic_norm_return'],
                          'baseline_phc_pass': old[c['id']]['phc_pass'], 'current_phc_pass': c['phc_pass'],
                          'baseline_object_pose_pass': old[c['id']]['object_pose_pass'],
                          'current_object_pose_pass': c['object_pose_pass']} for c in report['cases']]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-report', type=Path, help='Optional paired comparison; original report is read-only.')
    parser.add_argument('--run-id', help='A new W&B run identity, never an existing experiment run.')
    add_physics_arguments(parser)
    a = parser.parse_args()
    settings, _ = resolve_arguments(a)
    a.prepared, a.checkpoint, a.output = (p.resolve() for p in (a.prepared, a.checkpoint, a.output))
    import torch
    checkpoint_hash = sha(a.checkpoint)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    iteration, checkpoint_args = int(checkpoint['iteration']), checkpoint['args']
    del checkpoint
    manifest = json.loads((a.prepared / 'manifest.json').read_text())
    assert len(manifest['cases']) == 80
    baseline = a.baseline_report.resolve() if a.baseline_report else None
    baseline_report = json.loads(baseline.read_text()) if baseline else None
    if baseline_report:
        assert baseline_report['checkpoint_sha256'] == checkpoint_hash, 'Paired comparison requires the same checkpoint'
        assert baseline_report['input_manifest_sha256'] == sha(a.prepared / 'manifest.json')
    a.output.mkdir(parents=True, exist_ok=False)
    source = {str(p.relative_to(ROOT)): sha(p) for folder in ('hoi', 'supermimic') for p in sorted((ROOT / folder).glob('*.py'))}
    identity = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'source_branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'checkpoint': str(a.checkpoint), 'checkpoint_iteration': iteration, 'checkpoint_sha256': checkpoint_hash,
        'checkpoint_training_physics': {k: checkpoint_args.get(k, 'unknown_legacy') for k in asdict(settings)},
        'input_manifest_sha256': sha(a.prepared / 'manifest.json'), 'source_sha256': source,
        'zero_shot': True, 'optimizer_updates': 0, 'normalization_updates': 0,
        'physical_objects': True, 'object_observation_channels': 0, 'no_failure_reset': True,
        'reference_count': 80, 'checkpoint_uploads': False, 'rollout_uploads': False, 'source_uploads': False,
        'physics': asdict(settings), 'hand_collision_comparison': baseline is not None,
        'baseline_report': str(baseline) if baseline else None,
        'baseline_report_sha256': sha(baseline) if baseline else None,
        'wandb_run_id': a.run_id or 'hoi80-' + uuid.uuid4().hex[:16],
        'hand_cooking_limitation': LIMITATION if settings.hand_collision else None}

    def write(name, value):
        path = a.output / name
        temporary = path.with_suffix(path.suffix + '.partial')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)

    write('manifest.json', identity)

    def evaluate(asset, directory, gate=0, robot_usd=None, physics=settings):
        command = [sys.executable, str(ROOT / 'hoi/launch.py'), str(ROOT / 'hoi/evaluate.py'),
                   '--prepared', str(a.prepared), '--checkpoint', str(a.checkpoint),
                   '--asset', asset, '--output', str(directory), *physics.flags()]
        if gate:
            command += ['--gate-steps', str(gate)]
        if robot_usd:
            command += ['--robot-usd', str(robot_usd)]
        log = a.output / (directory.name + '.stdout.log')
        print(json.dumps({'command': command, 'log': str(log)}), flush=True)
        with log.open('x') as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=2700)
        result = json.loads((directory / 'evaluation.json').read_text())
        assert result['physics'] == asdict(physics)
        assert result['checkpoint_iteration'] == iteration
        return result

    try:
        assets = sorted(manifest['assets'])
        hand_asset = prepare_hands(ROOT, a.output / 'hand_asset', a.prepared / 'metadata/ultra_g1.urdf') if settings.hand_collision else None
        identity['hand_asset'] = hand_asset
        write('manifest.json', identity)
        if hand_asset:
            off_settings = type(settings)(hand_collision=False, self_collision=settings.self_collision)
            gate_off = evaluate(assets[0], a.output / 'native_gate_off', 16, physics=off_settings)
            gate = evaluate(assets[0], a.output / 'native_gate_on', 16, hand_asset['usd'])
            assert_same_physics(gate_off['native']['hand_collision_verification'], gate['native']['hand_collision_verification'])
            write('hand-gate-passed.json', {'off': gate_off['native']['hand_collision_verification'],
                'on': gate['native']['hand_collision_verification'], 'same_robot_mass_inertia_joints': True,
                'physics': asdict(settings), 'original_authored_properties_unchanged': hand_asset['original_authored_values_unchanged']})
        else:
            gate = evaluate(assets[0], a.output / 'native_gate', 16)
        assert gate['gate_only'] and gate['native']['compiled_mujoco_allowed_object_robot_geom_pairs'] > 0
        cases = []
        for asset in assets:
            result = evaluate(asset, a.output / asset, robot_usd=hand_asset['usd'] if hand_asset else None)
            assert not result['gate_only'] and result['checkpoint_sha256'] == checkpoint_hash
            assert all(c['full_reference_completed'] for c in result['cases'])
            for case in result['cases']:
                case['rollout'] = asset + '/' + case['rollout']
                cases.append(case)
            assert sha(a.checkpoint) == checkpoint_hash
        assert len(cases) == 80 and {c['id'] for c in cases} == {c['id'] for c in manifest['cases']}
        assert all(sha(ROOT / name) == digest for name, digest in source.items())
        if baseline:
            assert sha(baseline) == identity['baseline_report_sha256']
        report = {**identity, 'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'status': 'completed', 'aggregate': aggregate(cases),
            'by_object': {obj: aggregate([c for c in cases if c['id'].startswith(obj + '_')])
                          for obj in ('smallbox', 'largebox', 'plasticbox', 'suitcase')},
            'limitations': ['Object-pose pass is an engineering pose diagnostic, not a contact/grasp-success judge.',
                'Collision settings are explicit in physics; defaults differ from historical OFF/ON baselines.',
                LIMITATION if settings.hand_collision else 'No rubber-hand colliders requested.',
                'Newton CoACD and MuJoCo-Warp differ from the original ULTRA PhysX benchmark.',
                'Canonical validation80, not a newly held-out test split.'], 'cases': cases}
        if baseline_report:
            report['paired_comparison'] = paired_comparison(report, baseline_report)
        write('evaluation.json', report)
        print(json.dumps({'output': str(a.output), 'aggregate': report['aggregate']}), flush=True)
    except BaseException as error:
        write('failure.json', {'error': repr(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
