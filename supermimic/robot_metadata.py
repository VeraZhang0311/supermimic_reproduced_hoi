"""Read literal robot metadata without importing the native simulator."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = str(ROOT / "data/g1_hw3/source/urdf/g1/main_rubber_hand_meshcol_handcol_softfinger.urdf")
MJCF_PATH = str(ROOT / "data/g1_hw3/g1_hw3.xml")
# Exact source-URDF order, including sensor and fixed foot frames. The rubber
# hand pieces are collision shapes on wrist_yaw_link, not separate bodies.
G1_BODY_NAMES = [
    "pelvis", "pelvis_contour_link", "left_hip_pitch_link", "left_hip_roll_link",
    "left_hip_yaw_link", "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link", "right_knee_link",
    "right_ankle_pitch_link", "right_ankle_roll_link", "waist_yaw_link", "waist_roll_link",
    "torso_link", "logo_link", "head_link", "imu_in_torso", "imu_in_pelvis", "mid360_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link",
    "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link",
    "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link", "LL_FOOT", "LR_FOOT",
]
_tree = ast.parse((ROOT / "supermimic/robot.py").read_text(encoding="utf-8"))
for _node in _tree.body:
    if isinstance(_node, ast.Assign):
        for _target in _node.targets:
            if isinstance(_target, ast.Name) and _target.id in ("G1_JOINT_NAMES", "TRACKED_BODIES"):
                globals()[_target.id] = ast.literal_eval(_node.value)
