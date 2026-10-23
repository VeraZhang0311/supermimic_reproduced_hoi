"""Read a chosen evaluation's actual outputs; no fixed experiment/job identity."""
from pathlib import Path
import argparse
import json
import subprocess
from supermimic.physics import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    report = json.loads((root / 'evaluation.json').read_text()) if (root / 'evaluation.json').exists() else None
    job = manifest.get('slurm_job_id')
    receipt = {'output': str(root), 'job_id': job, 'physics': manifest['physics'],
               'status': report['status'] if report else 'incomplete',
               'aggregate': report['aggregate'] if report else None,
               'failure': json.loads((root / 'failure.json').read_text()) if (root / 'failure.json').exists() else None,
               'rollouts_sha256_verified': bool(report) and all(sha(root / c['rollout']) == c['rollout_sha256'] for c in report['cases'])}
    if job:
        try:
            receipt['slurm'] = subprocess.check_output(['/opt/slurm/bin/sacct', '-j', str(job), '--format=JobID,State,Elapsed,ExitCode,NodeList', '-P'], text=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            receipt['slurm_observation_error'] = str(exc)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
