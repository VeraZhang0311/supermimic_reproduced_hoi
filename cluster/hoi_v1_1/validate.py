from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(os.environ.get("SUPERMIMIC_ROOT", Path(__file__).resolve().parents[2])).resolve()
from .stage import verify


def main():
    root, stage = verify()
    import torch
    torch.set_num_threads(3)
    from supermimic.physics import PhysicsSettings
    from hoi.v1.config import fresh_directory, write_json
    from hoi.v1.evaluate import _read_bundle_header, validate_evaluation_identity
    from hoi.v1.workers import WorkerSpec, WorkerGroup, geometry_group
    from hoi.v1.gates import require_native_evidence
    from hoi.v1.randomization import MATCHING_DR, randomization_contract, validate_dr_evidence
    train = _read_bundle_header(stage['train_bundle'])
    validate_evaluation_identity(train, _read_bundle_header(stage['eval_bundle']))
    physics = asdict(PhysicsSettings().export())
    old = require_native_evidence(stage['base_native_evidence'], expected_physics=physics,
           required_asset_ids=range(8), expected_bundle_sha256=train['bundle_content_sha256'])
    assert old['passed']
    output = fresh_directory(root / 'results/v11_native_dr')
    specs = [WorkerSpec(int(a['asset_id']), 4, 4, geometry_group(a)) for a in train['assets']]
    reports = []
    for nominal in (False, True):
        print('NATIVE_DR_MODE', 'nominal_eval' if nominal else 'training_DR', flush=True)
        config = {'bundle': stage['train_bundle'], 'output': str(output / ('nominal' if nominal else 'randomized')),
                  'physics': physics, 'domain_randomization_profile': MATCHING_DR, 'device': 'cuda:0',
                  'robot_mjcf': train['robot_fk']['mjcf_path'], 'seed': 0, 'eval_mode': nominal,
                  'reset_noise': False, 'variant': 'hoi_v1', 'max_processes': 8,
                  'validate_domain_randomization': True, 'launcher_args': []}
        workers = WorkerGroup(specs, config, device='cuda:0')
        try:
            for report in workers.native_reports:
                check = report['domain_randomization_validation']
                assert check['passed'] and check['effective_MuJoCo_Warp_properties_verified'] and check['subset_reset_passed']
            reports.extend(workers.native_reports)
            write_json(output / ('nominal_reports.json' if nominal else 'randomized_reports.json'), workers.native_reports)
        finally:
            workers.close()
    assert len(reports) == 16
    evidence = json.loads(Path(stage['base_native_evidence']).read_text())
    evidence['domain_randomization'] = {'contract': randomization_contract(MATCHING_DR),
        'native_property_readback_passed': True, 'subset_reset_passed': True, 'nominal_eval_passed': True,
        'source_identity': stage['source_identity'], 'asset_ids': list(range(8)),
        'native_job_id': os.environ['SLURM_JOB_ID'],
        'reports': {name: {'path': str(output/name), 'sha256': hashlib.sha256((output/name).read_bytes()).hexdigest()}
                    for name in ['randomized_reports.json', 'nominal_reports.json']}}
    evidence_path = output / 'native_evidence.json'
    write_json(evidence_path, evidence)
    assert validate_dr_evidence(evidence_path, MATCHING_DR)['passed']
    print('NATIVE_DR_ALL_8_ASSETS_BOTH_MODES_PASSED', flush=True)
    env = dict(os.environ, BUNDLE=stage['train_bundle'], NATIVE_EVIDENCE=str(evidence_path),
               CONFIG=str(root/'configs/hoi_v1_1_observation_matching.json'), RUN_DIR=str(root/'results/v11_resource'))
    subprocess.run([sys.executable, '-m', 'cluster.hoi_v1_1.capacity_probe'], cwd=root, env=env, check=True)
    resource = json.loads((root/'results/v11_resource/resource_validation.json').read_text())
    assert resource['passed'] and resource['actual_optimizer_steps'] == 0
    assert resource['all_model_parameters_unchanged'] and resource['config'] == stage['production_config']
    write_json(root/'results/v11_validation_complete.json', {'passed': True, 'source_identity': stage['source_identity'],
        'native_evidence': str(evidence_path), 'native_evidence_sha256': hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        'resource_report': str(root/'results/v11_resource/resource_validation.json'),
        'resource_report_sha256': hashlib.sha256((root/'results/v11_resource/resource_validation.json').read_bytes()).hexdigest(),
        'actual_optimizer_steps': 0, 'job_id': os.environ['SLURM_JOB_ID']})
    print('V11_NATIVE_AND_PRODUCTION_CAPACITY_PASSED', flush=True)


if __name__ == '__main__':
    main()
