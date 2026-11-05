"""Read literal robot metadata without importing the native simulator."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = str(ROOT / "data/g1_description/g1_29dof_rev_1_0.urdf")
MJCF_PATH = str(ROOT / "data/g1_description/g1_29dof.xml")
_tree = ast.parse((ROOT / "supermimic/robot.py").read_text(encoding="utf-8"))
for _node in _tree.body:
    if isinstance(_node, ast.Assign):
        for _target in _node.targets:
            if isinstance(_target, ast.Name) and _target.id in ("G1_JOINT_NAMES", "TRACKED_BODIES"):
                globals()[_target.id] = ast.literal_eval(_node.value)
