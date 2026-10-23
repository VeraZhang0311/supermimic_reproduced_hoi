"""Continue the completed ON experiment 14000 -> 24000, full40 evaluation every1000."""
from pathlib import Path
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time

from run_pipeline import ROOT, PORTABLE, TRACKER_SITE, write_json, stamp, sha, terminate

START, END, EVERY = 14000, 24000, 1000
ORIGINAL = ROOT / 'logs/lafan1_selfcollision_seed0'
RUN = ROOT / 'logs/lafan1_selfcollision_seed0_continue14000to24000'
SNAPSHOT = RUN / 'resume_014000_full.pt'
EXPECTED_SHA = 'b933f0949de00484328423659a903437641be629845908eb85c0a7cd891fba15'
RUN_ID = 'selfcoll-lafan1-20261022-s0'
CLOUD_PATH = 'tianzizh-university-of-pennsylvania/SuperMimic_Upstream_Reproduction/' + RUN_ID
PROTECTED = ('last.pt', 'ckpt_012000.pt', 'ckpt_014000.pt',
             'ckpt_012000_eval.json', 'ckpt_014000_eval.json', 'train_log.csv', 'eval.csv',
             'args.json', 'manifest.json', 'acceptance.json', 'pipeline.json', 'wandb-cursor.json')
LIMITATIONS = ('The original checkpoint saves models, normalizers, all three Adam states, '
    'iteration, cumulative simulator steps and training wall time. It does not save replay, '
    'simulator/contact states, observation history, motion sampler statistics or RNG state. '
    'A new seed0 simulator and replay are created; 64 original collection steps refill replay '
    'without optimizer updates, with the original online-normalizer updates. '
    'This is checkpoint continuation, not a bitwise uninterrupted trajectory. '
    'The 500 world-model-only iterations are not repeated. Global iteration-based LR schedule is unchanged.')


def effective_args(original):
    args = dict(original)
    args.update(iterations=END, ckpt_every=EVERY, log_dir=str(RUN), resume=str(SNAPSHOT))
    assert {k for k in args if args[k] != original[k]} == {'iterations', 'ckpt_every', 'log_dir', 'resume'}
    return args


def training_arguments(manifest):
    params = manifest['effective_parameters']
    argv = []
    for key in (*manifest['parameters'].keys(), 'motion_dir', 'resume'):
        argv += ['--' + key, str(params[key])]
    return argv + ['--log_dir', str(RUN)]


def validate_saved_arguments(initial, checkpoint):
    # IsaacLab's launcher mutates only these display flags after args.json was
    # written. They are not training parameters. Fail on any other difference.
    allowed = {'livestream': [-1, 0], 'kit_visualizer': [None, False],
               'enable_cameras': [None, False]}
    differences = {key: [initial.get(key), checkpoint.get(key)]
                   for key in set(initial) | set(checkpoint)
                   if initial.get(key) != checkpoint.get(key)}
    for key, values in differences.items():
        assert key in allowed and values == allowed[key], (key, values)
    return differences


def prepare():
    import torch
    assert not RUN.exists(), 'Continuation directory exists; inspect before retrying'
    assert json.loads((ORIGINAL / 'pipeline.json').read_text())['status'] == 'completed'
    assert sha(ORIGINAL / 'last.pt') == EXPECTED_SHA, '14k full checkpoint changed'
    checkpoint = torch.load(ORIGINAL / 'last.pt', map_location='cpu', weights_only=False)
    assert checkpoint['iteration'] == START
    assert set(checkpoint) >= {'wm', 'pi', 'critic', 'opt_wm', 'opt_pi', 'opt_v'}
    finite = 0
    def scan(value):
        nonlocal finite
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            assert torch.isfinite(value).all().item()
            finite += value.numel()
        elif isinstance(value, dict):
            for v in value.values(): scan(v)
        elif isinstance(value, (list, tuple)):
            for v in value: scan(v)
    scan(checkpoint)
    args = json.loads((ORIGINAL / 'args.json').read_text())
    launcher_changes = validate_saved_arguments(args, checkpoint['args'])
    manifest = json.loads((ORIGINAL / 'manifest.json').read_text())
    for relative, digest in manifest['source_sha256'].items():
        assert sha(ROOT / relative) == digest, relative
    assert sha(ROOT / 'provenance/download-manifest.json') == manifest['data_manifest_sha256']
    assert subprocess.check_output(['git', '-C', str(ROOT / 'IsaacLab'), 'rev-parse', 'HEAD'], text=True).strip() == manifest['isaaclab_commit']
    assert subprocess.check_output(['git', '-C', str(ROOT), 'diff', '--', 'supermimic', 'tools'], text=True) == ''
    protected = {p: sha(ORIGINAL / p) for p in PROTECTED}
    RUN.mkdir()
    shutil.copy2(ORIGINAL / 'last.pt', SNAPSHOT)
    assert sha(SNAPSHOT) == EXPECTED_SHA
    SNAPSHOT.chmod(0o444)
    manifest.update(stage='continuation', created_utc=stamp(), original_manifest_sha256=sha(ORIGINAL / 'manifest.json'),
        effective_parameters=effective_args(args), resume_limitations=LIMITATIONS,
        original_runtime_launcher_arg_changes=launcher_changes,
        continuation={'start_iteration': START, 'stop_iteration': END, 'extra_iterations': END - START,
            'evaluation_interval': EVERY, 'evaluation_clips': 40, 'wandb_run_id': RUN_ID,
            'checkpoint_sha256': EXPECTED_SHA, 'original_run_directory': str(ORIGINAL),
            'same_learning_rate_schedule': True, 'prefill_vector_steps': 64},
        protected_original_files=protected,
        continuation_adapter_sha256={name: sha(ROOT / 'cluster' / name) for name in
            ('resume_launch.py', 'continue_pipeline.py', 'continue_track.py', 'continue.sbatch.sh')})
    write_json(RUN / 'manifest.json', manifest)
    write_json(RUN / 'checkpoint-audit.json', {'utc': stamp(), 'sha256': EXPECTED_SHA,
        'iteration': checkpoint['iteration'], 'sim_steps': checkpoint['sim_steps'],
        'wall_time': checkpoint['wall_time'], 'finite_floating_values': finite,
        'optimizer_steps': {key: sorted({float(s['step']) for s in checkpoint[key]['state'].values()})
                            for key in ('opt_wm', 'opt_pi', 'opt_v')}})
    return manifest


def evaluated():
    path = RUN / 'eval.csv'
    if not path.exists(): return set()
    done = set()
    for row in csv.DictReader(path.open()):
        iteration = int(row['iteration'])
        result_path = RUN / f'ckpt_{iteration:06d}_eval.json'
        if not result_path.exists(): continue
        try:
            result = json.loads(result_path.read_text())
        except json.JSONDecodeError:
            continue  # Author evaluator appends CSV just before writing this JSON.
        assert len(result['per_clip']) == 40
        assert all(math.isfinite(v) for v in result['per_clip'].values())
        if math.isfinite(float(row['norm_return'])):
            done.add(iteration)
    return done


def command(script, *args):
    return [sys.executable, str(ROOT / 'cluster/compat_launch.py'), script, *args]


def main():
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.cuda.get_device_properties(0).total_memory >= 44 * 1024 ** 3
    torch.set_num_threads(min(24, int(os.environ.get('SLURM_CPUS_PER_TASK', '24'))))
    manifest = prepare()
    manifest['hardware'] = {'name': torch.cuda.get_device_name(0),
        'memory_bytes': torch.cuda.get_device_properties(0).total_memory,
        'torch': torch.__version__, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'nvidia_smi': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
            '--format=csv,noheader'], text=True).strip()}
    write_json(RUN / 'manifest.json', manifest)
    # Verify the cloud run can really be resumed before starting a training producer.
    tracker_env = dict(os.environ, PYTHONPATH=TRACKER_SITE,
        WANDB_BASE_URL='https://api.forge.coreweave.com', WANDB_CACHE_DIR=str(ROOT / '.cache/wandb'),
        WANDB_DATA_DIR=str(ROOT / '.cache/wandb-data'), WANDB_DIR=str(RUN))
    streams = [open(RUN / name, 'x') for name in ('stdout.log', 'eval.stdout.log', 'tracker.stdout.log')]
    train = evaluator = tracker = None
    status, error = 'failed', None
    try:
        tracker = subprocess.Popen([str(PORTABLE), str(ROOT / 'cluster/continue_track.py')],
            cwd=ROOT, env=tracker_env, stdout=streams[2], stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 180
        while not (RUN / 'wandb-resumed.json').exists():
            assert tracker.poll() is None, f'Tracker exited {tracker.returncode} before resume'
            assert time.monotonic() < deadline, 'Cloud resume handshake timed out'
            time.sleep(2)
        params = manifest['effective_parameters']
        # Specify every original training parameter explicitly; only the four declared
        # operational fields differ. Launcher defaults remain exactly the saved ones.
        argv = training_arguments(manifest)
        assert params['no_critic'] is False
        train = subprocess.Popen(command('cluster/resume_launch.py', *argv), cwd=ROOT,
            stdout=streams[0], stderr=subprocess.STDOUT)
        evaluator = subprocess.Popen(command('supermimic/evaluate.py', '--run_dir', str(RUN), '--watch', '--every', '1'),
            cwd=ROOT, stdout=streams[1], stderr=subprocess.STDOUT)
        write_json(RUN / 'processes.json', {'utc': stamp(), 'training_pid': train.pid,
            'evaluation_pid': evaluator.pid, 'tracker_pid': tracker.pid,
            'slurm_job_id': os.environ['SLURM_JOB_ID'], 'training_command': command('cluster/resume_launch.py', *argv)})
        while train.poll() is None:
            assert tracker.poll() is None, 'Metrics tracker stopped during training'
            assert evaluator.poll() is None, 'Full40 evaluator stopped during training'
            time.sleep(10)
        assert train.returncode == 0, f'Training exited {train.returncode}'
        expected = set(range(START + EVERY, END + 1, EVERY))
        deadline = time.monotonic() + 3600
        while evaluated() != expected:
            assert evaluator.poll() is None, 'Evaluator stopped before all ten full40 evaluations'
            assert tracker.poll() is None
            assert time.monotonic() < deadline, 'Incomplete full40 evaluation after one hour'
            time.sleep(10)
        actual_args = json.loads((RUN / 'args.json').read_text())
        assert actual_args == params, 'Effective launch parameters differ'
        for relative, digest in manifest['source_sha256'].items():
            assert sha(ROOT / relative) == digest, relative
        for name, digest in manifest['protected_original_files'].items():
            assert sha(ORIGINAL / name) == digest, name
        result = json.loads((RUN / 'ckpt_024000_eval.json').read_text())
        assert len(result['per_clip']) == 40
        write_json(RUN / 'acceptance.json', {'utc': stamp(), 'iteration': END,
            'norm_return': result['norm_return'], 'n_failed': result['n_failed'],
            'completion_fraction': 1 - result['n_failed'] / 40,
            'target_reproduced': result['norm_return'] >= .82 and result['n_failed'] == 0,
            'full40_evaluations_completed': 10, 'original_results_unchanged': True})
        status = 'completed'
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        terminate(train)
        terminate(evaluator)
        write_json(RUN / 'pipeline.json', {'utc': stamp(), 'status': status,
            'error': error, 'training_exit_code': None if train is None else train.returncode})
        if tracker is not None:
            try: tracker.wait(timeout=150)
            except subprocess.TimeoutExpired: terminate(tracker)
        for stream in streams: stream.close()
    assert tracker.returncode == 0, 'Cloud synchronization failed; local results preserved'


if __name__ == '__main__':
    main()
