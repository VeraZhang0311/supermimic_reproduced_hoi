"""Bind the unchanged official graph compatibility adapter to this isolated checkout."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cluster import compat_launch
compat_launch.ROOT=ROOT
if __name__=='__main__':compat_launch.main()
