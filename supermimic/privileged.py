"""Robot-only privileged GMT features on Feng Qiao's hw3 38-link skeleton.

Same heading frames and x/z-column rotation6D as cc99253:hoi/v1/privileged.py;
hw3 URDF link frames, nominal COM
velocities, two lookaheads and raw robot context. No object/contact slots.
"""
import xml.etree.ElementTree as ET

import torch

from .robot_metadata import G1_BODY_NAMES, G1_JOINT_NAMES, URDF_PATH
from .rotations import axis_angle_to_mat, quat_to_mat, yaw_mat
from .actuator_state import ACTUATOR_DIM, ACTUATOR_SCHEMA

NUM_BODIES = len(G1_BODY_NAMES)
BODY_CURRENT_DIM = 15 * NUM_BODIES - 2
BODY_TARGET_ERROR_DIM = 24 * NUM_BODIES
PRIVILEGED_DIM = 2 * (BODY_CURRENT_DIM + BODY_TARGET_ERROR_DIM + 145) + 33 + ACTUATOR_DIM  # 3371
SCHEMA = {
    "version": "robot_privileged_gmt_hw3_actuator_v3", "dim": PRIVILEGED_DIM,
    "source": "codex/hoi-three-gpu-collector@cc99253:hoi/v1/privileged.py",
    "lookahead_frames": [1, 13], "rotation6d": "x/z columns",
    "bodies": NUM_BODIES, "velocity": "nominal hw3 URDF COM; root state uses link-origin velocity",
    "noise": False, "self_collision": True,
    "blocks": ["2*(body_current568+body_target_errors912+action_joint_history145)", "raw_robot_context33", "actuator_state88"],
    "actuator_state": ACTUATOR_SCHEMA,
}


def tan_norm(R):
    return torch.cat((R[..., :, 0], R[..., :, 2]), -1)


def euler_xyz_quat(rpy):
    r, p, y = (rpy / 2).unbind(-1)
    cr, sr, cp, sp, cy, sy = r.cos(), r.sin(), p.cos(), p.sin(), y.cos(), y.sin()
    return torch.stack((cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                        cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy), -1)


class RobotKinematics38:
    """URDF link frames with nominal COM-marker velocities, including fixed links."""
    def __init__(self, urdf=URDF_PATH, device="cpu"):
        tree = ET.parse(urdf).getroot()
        self.body_names = [link.get("name") for link in tree.findall("link")]
        if self.body_names != G1_BODY_NAMES:
            raise ValueError("privileged GMT requires the exact hw3 38-link G1 URDF body order")
        self.urdf_path = str(urdf)
        def vector(element, key, default):
            return torch.tensor([float(x) for x in (element.get(key, default) if element is not None else default).split()], device=device)
        self.com = torch.stack([vector(link.find("inertial/origin"), "xyz", "0 0 0") for link in tree.findall("link")])
        self.joints, reached = [], {"pelvis"}
        pending = list(tree.findall("joint"))
        dofs = []
        while pending:
            ready = [j for j in pending if j.find("parent").get("link") in reached]
            if not ready:
                raise ValueError("disconnected URDF")
            for j in ready:
                parent, child = j.find("parent").get("link"), j.find("child").get("link")
                if child in reached or j.get("type") not in ("fixed", "revolute", "continuous"):
                    raise ValueError("invalid G1 joint tree")
                dof = None if j.get("type") == "fixed" else G1_JOINT_NAMES.index(j.get("name"))
                if dof is not None:
                    dofs.append(dof)
                origin = j.find("origin")
                xyz = vector(origin, "xyz", "0 0 0")
                rotation = quat_to_mat(euler_xyz_quat(vector(origin, "rpy", "0 0 0")))
                axis = vector(j.find("axis"), "xyz", "1 0 0")
                if dof is not None and not torch.isclose(axis.norm(), torch.tensor(1., device=device)):
                    raise ValueError("nonunit URDF axis")
                self.joints.append((self.body_names.index(parent), self.body_names.index(child), dof, xyz, rotation, axis))
                reached.add(child)
                pending.remove(j)
        if len(reached) != NUM_BODIES or sorted(dofs) != list(range(29)):
            raise ValueError("URDF body/joint identity differs from G1")

    def forward(self, S):
        p, R, v, w, q, qd = S
        P, Rs, V, W = [[None] * NUM_BODIES for _ in range(4)]
        P[0], Rs[0], V[0], W[0] = p, R, v, w
        for parent, child, dof, xyz, origin_R, axis in self.joints:
            offset = (Rs[parent] @ xyz.unsqueeze(-1)).squeeze(-1)
            P[child] = P[parent] + offset
            pre_R = Rs[parent] @ origin_R
            V[child] = V[parent] + torch.cross(W[parent], offset, dim=-1)
            if dof is None:
                Rs[child], W[child] = pre_R, W[parent]
            else:
                Rs[child] = pre_R @ axis_angle_to_mat(axis, q[..., dof])
                W[child] = W[parent] + (pre_R @ axis.unsqueeze(-1)).squeeze(-1) * qd[..., dof:dof+1]
        P, Rs, V, W = torch.stack(P, -2), torch.stack(Rs, -3), torch.stack(V, -2), torch.stack(W, -2)
        V = V + torch.cross(W, (Rs @ self.com.unsqueeze(-1)).squeeze(-1), dim=-1)
        return P, Rs, V, W


class PrivilegedFeatures:
    def __init__(self, kin, motion):
        self.kin, self.motion = kin, motion

    def __call__(self, S, f, a_prev, xi, q_prev, qd_prev, actuator):
        P, R, V, W = self.kin.forward(S)
        Ht = yaw_mat(S[1]).transpose(-1, -2)
        Hb = Ht.unsqueeze(-3)
        rotate = lambda x: (Hb @ x.unsqueeze(-1)).squeeze(-1).flatten(-2)
        current = (S[0][..., 2:3], rotate(P - S[0].unsqueeze(-2))[..., 3:],
                   tan_norm(Hb @ R).flatten(-2), rotate(V), rotate(W))
        history = (a_prev, S[4], S[5], q_prev, qd_prev)
        blocks = []
        end = self.motion.clip_end[self.motion.clip_id[f]]
        for offset in (1, 13):
            target_f = torch.minimum(f + offset, end)
            with torch.no_grad():
                Pr, Rr, Vr, Wr = self.kin.forward(self.motion.ref_state(target_f))
            errors = (rotate(Pr - P), tan_norm(Hb @ (R @ Rr.transpose(-1, -2)) @ Hb.transpose(-1, -2)).flatten(-2),
                      rotate(Pr - S[0].unsqueeze(-2)), tan_norm(Hb @ Rr).flatten(-2), rotate(Vr - V), rotate(Wr - W))
            blocks.append(torch.cat((*current, *errors, *history), -1))
        result = torch.cat((*blocks, xi, actuator), -1)
        if result.shape[-1] != PRIVILEGED_DIM:
            raise ValueError("privileged feature layout drift")
        return result


def checkpoint_contract(motion, kin):
    import hashlib
    from dataclasses import asdict
    from pathlib import Path
    from .g1_physics import physics_contract
    from .mdp import RewardCfg, OBS_DIM, CONTROL_DT
    from .robot_metadata import TRACKED_BODIES
    heads = {"observation_dim": OBS_DIM, "control_dt": CONTROL_DT, "reward_cfg": asdict(RewardCfg()),
             "tracked_bodies": TRACKED_BODIES,
             "implementation_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                       for name in ("mdp.py", "kinematics.py", "rotations.py", "privileged.py", "networks.py", "actuator_state.py")}}
    return {**SCHEMA, "physics": physics_contract(), "body_names": kin.body_names, "joint_names": G1_JOINT_NAMES,
            "analytic_heads": heads,
            "urdf_sha256": hashlib.sha256(Path(kin.urdf_path).read_bytes()).hexdigest(),
            "dataset": motion.dataset, "motion_manifest_sha256": motion.manifest_sha256}


def validate_checkpoint(ck, contract):
    if ck.get("gmt_contract") != contract:
        raise ValueError("checkpoint GMT observation/physics/robot/dataset contract mismatch; start a new run")
