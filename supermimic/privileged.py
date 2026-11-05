"""Robot-only subset of cc99253:hoi/v1/privileged.py.

Same heading frames, x/z-column rotation6D, 39 URDF links, nominal COM
velocities, two lookaheads and raw robot context. No object/contact slots.
"""
import xml.etree.ElementTree as ET

import torch

from .robot_metadata import G1_JOINT_NAMES, URDF_PATH
from .rotations import axis_angle_to_mat, quat_to_mat, yaw_mat

PRIVILEGED_DIM = 3361  # 2 * (583 current + 936 target/error + 145 history) + 33
SCHEMA = {
    "version": "robot_privileged_gmt_v1", "dim": PRIVILEGED_DIM,
    "source": "codex/hoi-three-gpu-collector@cc99253:hoi/v1/privileged.py",
    "lookahead_frames": [1, 13], "rotation6d": "x/z columns",
    "bodies": 39, "velocity": "nominal URDF COM; root state uses link-origin velocity",
    "noise": False, "self_collision": True,
    "blocks": ["2*(body_current583+body_target_errors936+action_joint_history145)", "raw_robot_context33"],
}


def tan_norm(R):
    return torch.cat((R[..., :, 0], R[..., :, 2]), -1)


def euler_xyz_quat(rpy):
    r, p, y = (rpy / 2).unbind(-1)
    cr, sr, cp, sp, cy, sy = r.cos(), r.sin(), p.cos(), p.sin(), y.cos(), y.sin()
    return torch.stack((cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                        cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy), -1)


class RobotKinematics39:
    """URDF link frames with nominal COM-marker velocities, including fixed links."""
    def __init__(self, urdf=URDF_PATH, device="cpu"):
        tree = ET.parse(urdf).getroot()
        self.body_names = [link.get("name") for link in tree.findall("link")]
        if len(self.body_names) != 39 or len(set(self.body_names)) != 39 or self.body_names[0] != "pelvis":
            raise ValueError("privileged GMT requires the original 39-link G1 URDF, pelvis first")
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
        if len(reached) != 39 or sorted(dofs) != list(range(29)):
            raise ValueError("URDF body/joint identity differs from G1")

    def forward(self, S):
        p, R, v, w, q, qd = S
        P, Rs, V, W = [[None] * 39 for _ in range(4)]
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

    def __call__(self, S, f, a_prev, xi, q_prev, qd_prev):
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
            errors = (rotate(Pr - P), tan_norm(Hb @ (Rr.transpose(-1, -2) @ R) @ Hb.transpose(-1, -2)).flatten(-2),
                      rotate(Pr - S[0].unsqueeze(-2)), tan_norm(Hb @ Rr).flatten(-2), rotate(Vr - V), rotate(Wr - W))
            blocks.append(torch.cat((*current, *errors, *history), -1))
        result = torch.cat((*blocks, xi), -1)
        if result.shape[-1] != PRIVILEGED_DIM:
            raise ValueError("privileged feature layout drift")
        return result


def checkpoint_contract(motion, kin):
    import hashlib
    from pathlib import Path
    return {**SCHEMA, "body_names": kin.body_names, "joint_names": G1_JOINT_NAMES,
            "urdf_sha256": hashlib.sha256(Path(URDF_PATH).read_bytes()).hexdigest(),
            "dataset": motion.dataset, "motion_manifest_sha256": motion.manifest_sha256}


def validate_checkpoint(ck, contract):
    if ck.get("gmt_contract") != contract:
        raise ValueError("checkpoint GMT observation/physics/robot/dataset contract mismatch; start a new run")
