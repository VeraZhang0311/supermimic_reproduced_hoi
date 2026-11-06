"""ResMimic tracking objectives adapted to GMT's named joints and 14 tracked links.

Method source: amazon-far/ResMimic@fa20d98, G1HOICfg and G1Mimic rewards.
Independent implementation; no Isaac Gym collision/PD/termination code is copied.
All positive terms and regularizers are multiplied by the existing 0.02 s dt.
Foot-contact heuristics are deliberately not inferred from ambiguous net forces.
"""
import torch
from supermimic.motion import CONTROL_DT
from supermimic.rotations import rot_angle_sq, yaw_mat

MAX_POSITIVE_REWARD = 19.2 * CONTROL_DT


def object_distance(O, reference, points):
    """Mean paired surface-point displacement, in meters (not Chamfer/squared)."""
    p, R = O[:2]
    pr, Rr = reference[:2]
    actual = p[:, None] + points @ R.transpose(-1, -2)
    target = pr[:, None] + points @ Rr.transpose(-1, -2)
    return (actual - target).norm(dim=-1).mean(-1)


def reward(mdp, S, O, frame, action, previous_action, points, previous_qd):
    p, R, v, w, q, qd = S
    pr, Rr, vr, wr, qr, qdr = mdp.motion.ref_state(frame)
    bp = mdp.kin.forward(*S)[0]
    bpr = mdp.motion.reference_bodies(frame)[0]
    H, Hr = yaw_mat(R).transpose(-1, -2), yaw_mat(Rr).transpose(-1, -2)
    local = (H[:, None] @ (bp-p[:, None]).unsqueeze(-1)).squeeze(-1)
    local_ref = (Hr[:, None] @ (bpr-pr[:, None]).unsqueeze(-1)).squeeze(-1)
    distance = object_distance(O, mdp.motion.object_ref(frame), points)
    terms = {
        "joint_pos": 2 * torch.exp(-.15 * (q-qr).square().sum(-1)),
        "joint_vel": .2 * torch.exp(-.01 * (qd-qdr).square().sum(-1)),
        "root_height": torch.exp(-5 * (p[:, 2]-pr[:, 2]).square()),
        "root_rotation": torch.exp(-5 * rot_angle_sq(R, Rr)),
        "root_linear_vel": torch.exp(-(v-vr).square().sum(-1)),
        "root_angular_vel": torch.exp(-(w-wr).square().sum(-1)),
        "body_local": torch.exp(-10 * (local-local_ref).square().sum((-1, -2))),
        "body_global": 10 * torch.exp(-10 * (bp-bpr).square().sum((-1, -2))),
        "object_surface": 2 * torch.exp(-10 * distance),
        "action_rate": -.01 * (action-previous_action).norm(dim=-1),
        "joint_limits": -5 * (torch.relu(q-mdp.soft_hi)+torch.relu(mdp.soft_lo-q)).sum(-1),
        "joint_speed": -1e-4 * qd.square().sum(-1),
        "joint_accel": -5e-8 * ((qd-previous_qd)/CONTROL_DT).square().sum(-1),
    }
    total = sum(terms.values()) * CONTROL_DT
    metrics = {key: value * CONTROL_DT for key, value in terms.items()}
    metrics["object_surface_error_m"] = distance
    metrics["object_position_error_m"] = (O[0]-mdp.motion.object_ref(frame)[0]).norm(dim=-1)
    return total, metrics
