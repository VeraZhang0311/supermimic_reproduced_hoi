"""Shared G1 hw3 contract for GMT and future residual HOI; no simulator imports.

Source: Feng Qiao's gear_sonic@4aed2bb (2026-11-05 handoff).
Motor caps/gains: robots/g1.py; envelope: mdp/motor_model.py;
friction: mdp/friction_hw3.py; timing/materials: robots/g1_hw3.py.
MuJoCo adaptation choices are explicitly versioned separately from measured data.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from .robot_metadata import G1_JOINT_NAMES

PROFILE = "fengqiao_hw3_mujoco_v1"
PHYSICS_DT = 0.005
CONTROL_DECIMATION = 4
CONTROL_DT = PHYSICS_DT * CONTROL_DECIMATION
CONTACT_CONE = "elliptic"
CONTACT_IMPRATIO = 10.
ARCHIVE_MD5 = "4389c0463f2037ef8f04520eda1f949b"
SOURCE_COMMIT = "4aed2bb"
ARMATURE = {"7520-22": .025101925, "7520-14": .010177520,
            "5020": .003609725, "2x5020": 2 * .003609725, "4010": .00425}
# drive, brake, knee speed, no-load speed (Nm, Nm, rad/s, rad/s).
ENVELOPES = {"7520-22": (111., 131., 14.5, 22.7), "7520-14": (71., 83.3, 22.63, 35.52),
             "5020": (24.8, 31.9, 30.86, 40.13), "2x5020": (49.6, 63.8, 30.86, 40.13),
             "4010": (4.8, 8.6, 15.3, 24.76)}
CAPS = {"7520-22": (139., 20.), "7520-14": (88., 32.), "5020": (25., 37.),
        "2x5020": (50., 37.), "4010": (5., 22.)}
MOTOR_TYPES = {"hip_pitch": "7520-22", "hip_roll": "7520-22", "knee": "7520-22",
               "hip_yaw": "7520-14", "waist_yaw": "7520-14",
               "ankle_pitch": "2x5020", "ankle_roll": "2x5020", "waist_pitch": "2x5020", "waist_roll": "2x5020",
               "shoulder_pitch": "5020", "shoulder_roll": "5020", "shoulder_yaw": "5020", "elbow": "5020",
               "wrist_roll": "5020", "wrist_pitch": "4010", "wrist_yaw": "4010"}
# slow and fast no-load friction levels, in SDK joint order (Nm).
FRICTION_LEVELS = [(.893, .747), (.686, .630), (1.079, .879), (.978, .713), (.729, .612), (.468, .385),
                   (.809, .769), (.568, .577), (.963, .911), (.848, .700), (.759, .623), (.529, .438),
                   (.462, .443), (.869, .905), (.869, .905),
                   (.324, .277), (.453, .395), (.296, .271), (.412, .348), (.395, .297), (.177, .170), (.163, .162),
                   (.368, .304), (.461, .379), (.258, .212), (.444, .367), (.322, .247), (.204, .199), (.134, .137)]
DEFAULT_JOINT_POS = {".*_hip_pitch_joint": -.312, ".*_knee_joint": .669, ".*_ankle_pitch_joint": -.363,
                     ".*_elbow_joint": .6, "left_shoulder_roll_joint": .2, "left_shoulder_pitch_joint": .2,
                     "right_shoulder_roll_joint": -.2, "right_shoulder_pitch_joint": .2}
# Adaptation: native dry-friction constraint with a smooth static-to-dynamic
# threshold, not a second explicitly subtracted tanh torque. Transition speed
# is an engineering choice, NOT a measured Stribeck fit from the handoff.
FRICTION_TRANSITION_SPEED = .01
# Constant-impedance MuJoCo contact. The three-point hull/plane probe gives
# 0.54 N/mm incremental pad stiffness for 180 N/m per contact. Damping 80
# settles the 0.25..4 kg press fixtures at 5 ms; this is a backend adaptation,
# not a copy of PhysX's 130 N/m, 10 Ns/m contact material.
CONTACT_STIFFNESS = 180.
CONTACT_DAMPING = 80.
TARGET_PAD_SLOPE_N_MM = .55
SOFT_IMPEDANCE = .9
HAND_FRICTION = 1.
ASSET_DIR = Path(__file__).resolve().parents[1] / "data/g1_hw3"


def motor_type(name):
    key = name.removeprefix("left_").removeprefix("right_").removesuffix("_joint")
    return MOTOR_TYPES[key]


def motor_of(name):
    t = motor_type(name)
    return (*CAPS[t], ARMATURE[t])


def gains(name):
    arm = ARMATURE[motor_type(name)]
    omega = 20 * math.pi
    return arm * omega**2, 4 * arm * omega


def action_scale(name):
    return .25 * motor_of(name)[0] / gains(name)[0]


def profile_contract():
    return {"profile": PROFILE, "source_commit": SOURCE_COMMIT, "archive_md5": ARCHIVE_MD5,
            "physics_dt": PHYSICS_DT, "control_decimation": CONTROL_DECIMATION,
            "contact_solver": {"cone": CONTACT_CONE, "impratio": CONTACT_IMPRATIO, "integrator": "implicitfast"},
            "joints": [{"name": n, "motor": motor_type(n), "caps_armature": motor_of(n),
                        "kp_kd": gains(n), "envelope": ENVELOPES[motor_type(n)],
                        "friction_static_dynamic": [s, min(s, d)], "action_scale": action_scale(n)}
                       for n, (s, d) in zip(G1_JOINT_NAMES, FRICTION_LEVELS, strict=True)],
            "delay_steps": [2, 4], "torque_scale_reset": [.9, 1.1],
            "joint_friction_scale_startup": [.8, 2.], "leg_friction_scale_startup": [.8, 2.5],
            "friction_adaptation": {"method": "native frictionloss with speed-dependent threshold",
                                    "transition_rad_s": FRICTION_TRANSITION_SPEED, "viscous": 0.},
            "soft_hand_adaptation": {"contact_k_N_m": CONTACT_STIFFNESS, "contact_c_Ns_m": CONTACT_DAMPING,
                                     "target_pad_slope_N_mm": TARGET_PAD_SLOPE_N_MM,
                                     "impedance": SOFT_IMPEDANCE, "mu": HAND_FRICTION,
                                     "normalization": "nominal body inverse weight; validate contact response",
                                     "physx_equivalence": False},
            "surface_adaptation": "MuJoCo sliding friction; hand priority=1, mu=1; other pairs use native maximum; rolling/torsion=0",
            "surface_dr": {"dynamic": [.3, 1.2]},
            "ground_sliding_mu": {"training": .3, "nominal_evaluation": 1.},
            "surface_dr_adaptation": "source PhysX separate static surface friction/restitution DR not reproduced; one native sliding coefficient",
            "evaluation": "nominal torque/friction scales=1, shared delay=3 ticks; no parameter DR",
            "scope": "G1 motors/collisions/friction/soft hand; task mass/COM/push recipes remain separate"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physics_contract(asset_dir=ASSET_DIR):
    from .g1_assets import verify_assets
    receipt = verify_assets(asset_dir)
    # JSON round-trip makes tuples lists before checkpoint serialization.
    return json.loads(json.dumps({**profile_contract(), "asset_identity": receipt["identity"],
                                 "runtime_contact_mix": {
                                     "hand_body_object": "multiply native geom sliding friction via post-narrowphase callback",
                                     "gmt_world_planes": "preserve hand-ground mu=1; native floor mu=.3 is the body max-mixing convention",
                                     "normal_material": "hand priority preserves source-adapted soft/rigid solref and solimp"},
                                 "implementation_sha256": {p: sha(Path(__file__).with_name(p)) for p in
                                     ("g1_physics.py", "g1_assets.py", "g1_runtime.py", "g1_usd.py", "g1_contactmix.py", "actuator_state.py", "robot.py", "env.py")}}))


def friction_threshold(velocity, scale):
    levels = velocity.new_tensor(FRICTION_LEVELS)
    fs, fd = levels[:, 0], levels.min(-1).values
    return (fd + (fs - fd) * torch.exp(-(velocity / FRICTION_TRANSITION_SPEED).square())) * scale


class G1Motor:
    """Stateful 200 Hz controller. Feed the combined GMT+residual target here.

    A single 2..4 tick delay per environment applies to all joints. The FIFO
    is filled with the reset pose, preventing a previous episode's commands
    from leaking. Friction is applied by the solver, NOT subtracted here.
    """
    def __init__(self, num_envs, device="cpu", randomize=True):
        self.N, self.device, self.randomize = num_envs, device, randomize
        self.kp = torch.tensor([gains(n)[0] for n in G1_JOINT_NAMES], device=device)
        self.kd = torch.tensor([gains(n)[1] for n in G1_JOINT_NAMES], device=device)
        self.cap = torch.tensor([motor_of(n)[0] for n in G1_JOINT_NAMES], device=device)
        self.envelope = torch.tensor([ENVELOPES[motor_type(n)] for n in G1_JOINT_NAMES], device=device).T
        self.delay = torch.full((num_envs,), 3, device=device, dtype=torch.long)
        self.scale = torch.ones(num_envs, 29, device=device)
        self.friction_scale = torch.ones_like(self.scale)
        if randomize:
            high = self.kp.new_tensor([2.5 if i < 12 else 2. for i in range(29)])
            self.friction_scale = .8 + (high - .8) * torch.rand_like(self.scale)
        self.queue = torch.zeros(5, num_envs, 29, device=device)
        self.tick = 0
        self.ids = torch.arange(num_envs, device=device)

    def reset(self, ids, pose):
        self.queue[:, ids] = pose.unsqueeze(0)
        if self.randomize:
            self.delay[ids] = torch.randint(2, 5, (len(ids),), device=self.device)
            self.scale[ids] = .9 + .2 * torch.rand(len(ids), 29, device=self.device)
        else:
            self.delay[ids] = 3
            self.scale[ids] = 1.

    def compute(self, target, q, qd):
        if any(x.shape != (self.N, 29) for x in (target, q, qd)) or not all(bool(torch.isfinite(x).all()) for x in (target, q, qd)):
            raise ValueError("G1 motor requires finite (num_envs, 29) target, q and qd")
        self.queue[self.tick] = target
        delayed = self.queue[(self.tick - self.delay) % 5, self.ids]
        self.tick = (self.tick + 1) % 5
        effort = (self.kp * (delayed - q) - self.kd * qd).clamp(-self.cap, self.cap)
        drive, brake, knee, maximum = self.envelope
        peak = torch.where((effort * qd > 0) & (qd.abs() > .01), drive, brake)
        limit = peak * ((maximum - qd.abs()) / (maximum - knee)).clamp(0, 1) * self.scale
        return torch.maximum(torch.minimum(effort, limit), -limit)
