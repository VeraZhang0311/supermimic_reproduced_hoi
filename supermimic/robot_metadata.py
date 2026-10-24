"""CPU-safe names and MJCF path, read from the unchanged robot definition.

Only literal metadata is parsed. Importing the native articulation/actuator
configuration is unnecessary for differentiable FK and offline preparation.
"""
from __future__ import annotations

import ast
from pathlib import Path
import re

_ROOT = Path(__file__).resolve().parent.parent
MJCF_PATH = str(_ROOT / "data" / "g1_description" / "g1_29dof.xml")
_TREE = ast.parse((_ROOT / "supermimic" / "robot.py").read_text(encoding="utf-8"))
_NAMES = {"G1_JOINT_NAMES", "TRACKED_BODIES", "DEFAULT_JOINT_POS"}
_VALUES = {}
for _node in _TREE.body:
    if isinstance(_node, ast.Assign):
        for _target in _node.targets:
            if isinstance(_target, ast.Name) and _target.id in _NAMES:
                _VALUES[_target.id] = ast.literal_eval(_node.value)
if set(_VALUES) != _NAMES:
    raise RuntimeError("robot.py must declare literal G1 joint and tracked-body names")
G1_JOINT_NAMES = _VALUES["G1_JOINT_NAMES"]
TRACKED_BODIES = _VALUES["TRACKED_BODIES"]
DEFAULT_JOINT_POS = _VALUES["DEFAULT_JOINT_POS"]


def default_joint_pos_list() -> list[float]:
    """Original ordered regex rule; read literals without importing IsaacLab."""
    result = []
    for name in G1_JOINT_NAMES:
        value = 0.0
        for pattern, candidate in DEFAULT_JOINT_POS.items():
            if re.fullmatch(pattern, name):
                value = candidate
        result.append(value)
    return result
