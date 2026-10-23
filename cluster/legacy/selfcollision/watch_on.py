"""Read actual GRASP self-collision ON state and persist a small audit receipt; no train mutation."""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import math
import re
import subprocess

ROOT = Path('/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision')
FORMAL = ROOT / 'logs/lafan1_selfcollision_seed0'
UTC = datetime.datetime.now(datetime.timezone.utc)
JOB = (ROOT / 'provenance/gpu-job.txt').read_text().strip()


def cmd(arguments):
    return subprocess.check_output(arguments, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


def read(path):
    return path.read_text(errors='replace') if path.exists() else ''


def rows(path):
    text = read(path)
    if not text:
        return []
    if not text.endswith('\n'):
        text = text.rsplit('\n', 1)[0] + '\n'
    return list(csv.DictReader(text.splitlines()))


def numeric(row):
    return {key: float(value) for key, value in row.items()}


def finite(records):
    return all(math.isfinite(float(value)) for row in records for value in row.values())


def atomic(path, value):
    def serializable(item):
        if isinstance(item, float) and not math.isfinite(item):
            return str(item)
        if isinstance(item, dict):
            return {key: serializable(part) for key, part in item.items()}
        if isinstance(item, list):
            return [serializable(part) for part in item]
        return item
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(serializable(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


native = read(ROOT / f'logs/native-fk-gate-{JOB}.log')
smoke = rows(ROOT / f'logs/smoke_full501-{JOB}/train_log.csv')
formal = rows(FORMAL / 'train_log.csv')
roundtrip = dict(re.findall(r'roundtrip (\w+): max err ([0-9.eE+-]+)', native))
fk_match = re.search(r'FK pos err ([0-9.eE+-]+)', native)
native_valid = bool(smoke) and set(roundtrip) == {'p', 'R', 'v', 'w', 'q', 'qd'} and all(float(x) < 2e-4 for x in roundtrip.values()) and bool(fk_match) and float(fk_match[1]) < 1e-4
queue = cmd(['/opt/slurm/bin/squeue', '-u', 'tianzizh', '--noheader', '--format=%i|%j|%T|%M|%R'])
accounting = cmd(['/opt/slurm/bin/sacct', '-j', JOB, '--format=JobID,State,Elapsed,ExitCode,NodeList', '-P'])
# Derive current state from this observation, rather than retaining submission status.
queue_line = next((line for line in queue.splitlines() if line.split('|')[0] == JOB), '')
accounting_line = next((line for line in accounting.splitlines() if line.split('|')[0] == JOB), '')
slurm_state = queue_line.split('|')[2] if queue_line else accounting_line.split('|')[1] if accounting_line else 'UNKNOWN'
slurm_state = slurm_state.split()[0].rstrip('+')
if formal:
    stage = 'formal'
elif smoke:
    stage = 'smoke'
elif native or (ROOT / f'provenance/collision-on-{JOB}.json').exists():
    stage = 'native'
else:
    stage = 'pending' if slurm_state == 'PENDING' else 'setup'
receipt = {
    'utc': UTC.isoformat(), 'variant': 'selfcollision_on', 'job_id': int(JOB),
    'queue': queue, 'accounting': accounting, 'slurm_state': slurm_state, 'stage': stage,
    'native_gpu_validated': native_valid,
    'native_fk_max_m': float(fk_match[1]) if fk_match else None,
    'smoke_last_iteration': int(smoke[-1]['iteration']) if smoke else None,
    'smoke_rows': len(smoke), 'smoke_all_logged_values_finite': finite(smoke),
    'formal_directory_exists': FORMAL.exists(),
    'formal_last_iteration': int(formal[-1]['iteration']) if formal else None,
    'formal_all_logged_values_finite': finite(formal),
    'formal_last_two_rows': [numeric(row) for row in formal[-2:]],
    'formal_post_warmup_rows': sum(int(row['iteration']) > 500 for row in formal),
    'pipeline_tail': read(ROOT / f'logs/gpu-{JOB}.out').splitlines()[-15:],
    'smoke_tail': read(ROOT / f'logs/smoke501-{JOB}.stdout.log').splitlines()[-5:],
    'formal_stdout_tail': read(FORMAL / 'stdout.log').splitlines()[-8:],
    'tracker_stdout_tail': read(ROOT / 'logs/tracker.stdout.log').splitlines()[-12:],
    'source_head': cmd(['git', 'rev-parse', 'HEAD']),
    'core_source_diff': cmd(['git', 'diff', '--name-only', 'HEAD', '--', 'supermimic', 'tools']),
    'cluster_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / 'cluster').glob('*.py'))},
    'evidence': [f'logs/native-fk-gate-{JOB}.log', f'logs/smoke_full501-{JOB}/train_log.csv', 'logs/lafan1_selfcollision_seed0/train_log.csv', 'logs/tracker.stdout.log'],
    'automation_id': json.loads(read(ROOT/'provenance/automation.json') or '{}').get('id'), 'automation_interval_minutes': 15,
    'training_or_configuration_mutated': False,
}
try:
    import wandb
    api = wandb.Api(timeout=25)
    cloud = api.run('tianzizh-university-of-pennsylvania/SuperMimic_Upstream_Reproduction/selfcoll-lafan1-20261022-s0')
    summary = dict(cloud.summary)
    receipt['wandb'] = {
        'cloud_readback': True, 'url': cloud.url, 'state': cloud.state,
        'summary': {key: value for key, value in summary.items() if key.startswith(('train/', 'eval/'))},
    }
except Exception as exc:
    receipt['wandb'] = {'cloud_readback': False, 'exception_type': type(exc).__name__, 'reason': str(exc)[:500]}
collision={}
for setting in ('off','on'):
    path=ROOT/f'provenance/collision-{setting}-{JOB}.json'
    collision[setting]=json.loads(path.read_text()) if path.exists() else None
receipt['collision_backend_verification']=collision
receipt['physical_self_collision_verified']=bool(collision['off'] and collision['on'] and collision['off']['newton_allowed_self_shape_pairs']==0 and collision['off']['mujoco_allowed_nonadjacent_geom_pairs']==0 and collision['on']['newton_allowed_self_shape_pairs']>0 and collision['on']['mujoco_allowed_nonadjacent_geom_pairs']>0)
receipt['native_gpu_validated']=native_valid and receipt['physical_self_collision_verified']
previous = sorted((ROOT / 'provenance').glob('selfcollision-watch-*.json'))
if previous:
    old = json.loads(previous[-1].read_text())
    before = old.get('formal_last_iteration')
    after = receipt['formal_last_iteration']
    receipt['formal_growth_since_previous_receipt'] = before is not None and after is not None and after > before
    receipt['previous_receipt'] = str(previous[-1].relative_to(ROOT))
else:
    receipt['formal_growth_since_previous_receipt'] = False
post = [row for row in formal if int(row['iteration']) > 500]
receipt['policy_and_critic_updates_observed'] = bool(post) and any(float(row['pi_gradnorm']) > 0 and float(row['v_loss']) != 0 for row in post)
cloud_iteration = int(receipt['wandb'].get('summary', {}).get('train/iteration', 0))
cloud_row = next((row for row in formal if int(row['iteration']) == cloud_iteration), None)
receipt['wandb_metrics_match_logged_iteration'] = bool(cloud_row) and all(math.isclose(float(receipt['wandb']['summary'].get('train/' + key, 'nan')), float(cloud_row[key]), rel_tol=1e-6, abs_tol=1e-8) for key in ('wm_loss', 'pi_loss', 'v_loss', 'pi_gradnorm'))
receipt['stable_start_conditions_currently_satisfied'] = receipt['physical_self_collision_verified'] and bool(formal) and int(formal[-1]['iteration']) >= 550 and finite(formal) and receipt['policy_and_critic_updates_observed'] and receipt['formal_growth_since_previous_receipt'] and receipt['wandb']['cloud_readback'] and cloud_iteration >= 550 and receipt['wandb_metrics_match_logged_iteration']
receipt['next_action'] = 'Pause ON startup monitor, retain 14000-iteration producer and evaluation.' if receipt['stable_start_conditions_currently_satisfied'] else 'Inspect actual next native/smoke/formal state and W&B after 15 minutes; diagnose any actual failure before retry.'
destination = ROOT / ('provenance/selfcollision-watch-' + UTC.strftime('%Y%m%dT%H%M%SZ') + '.json')
atomic(destination, receipt)
status_path = ROOT / 'provenance/startup-status.json'
status = json.loads(status_path.read_text())
status.update(last_checked_utc=UTC.isoformat(), last_watch_receipt=str(destination.relative_to(ROOT)), last_queue_snapshot=queue, repair_native_gpu_validated=receipt['native_gpu_validated'], physical_self_collision_verified=receipt['physical_self_collision_verified'], formal_training_started=bool(formal), smoke_last_iteration=receipt['smoke_last_iteration'], formal_last_iteration=receipt['formal_last_iteration'], formal_stable_start_validated=receipt['stable_start_conditions_currently_satisfied'], automation_interval_minutes=15)
status.update(status=slurm_state.lower(), slurm_state=slurm_state, stage=stage,
              queue_snapshot=queue_line or accounting_line,
              last_observation_attempt_utc=UTC.isoformat(),
              latest_observation_status='actual_cluster_state_read',
              fresh_cluster_state_available=True, last_observation_error=None,
              native_self_collision_verified=receipt['physical_self_collision_verified'],
              formal_training_stable=receipt['stable_start_conditions_currently_satisfied'])
atomic(status_path, status)
print(json.dumps({'receipt_path': str(destination), 'receipt': json.loads(destination.read_text())}, indent=2))
