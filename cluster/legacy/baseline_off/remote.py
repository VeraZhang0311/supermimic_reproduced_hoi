"""Run reviewed, UTF-8 scripts over the existing GRASP SSH connection."""
from pathlib import Path
import subprocess
import sys

HOST = 'grasp-login1.seas.upenn.edu'
if sys.argv[1] == 'upload':
    source = Path(__file__).parent
    subprocess.run(['ssh', '-o', 'BatchMode=yes', HOST,
                    'mkdir -p /mnt/projects/ll/tianzizh/supermimic_reproduced_hoi/cluster'], check=True)
    for path in source.iterdir():
        if path.is_file() and path.name != 'remote.py':
            subprocess.run(['scp', str(path), HOST + ':/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi/cluster/' + path.name], check=True)
elif sys.argv[1] == 'run':
    path = Path(sys.argv[2])
    subprocess.run(['ssh', '-o', 'BatchMode=yes', HOST, 'bash -s'],
                   input=path.read_text(encoding='utf-8').encode('utf-8'), check=True)
else:
    raise SystemExit('Expected upload or run SCRIPT')
