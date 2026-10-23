"""Run reviewed UTF-8 scripts over the authorized GRASP SSH connection."""
from pathlib import Path
import subprocess
import sys
subprocess.run(['ssh','-o','BatchMode=yes','-o','UpdateHostKeys=no',
    'grasp-login1.seas.upenn.edu','bash -s'],
    input=Path(sys.argv[1]).read_text(encoding='utf-8').encode('utf-8'),check=True)
