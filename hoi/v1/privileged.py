"""Noise-free teacher features, shared by collection, imagination and evaluation.

The kinematic/interaction blocks follow ULTRA's UltraG1Retarget teacher, not
its student or sim2sim observation builder. This is an explicit new schema,
not a 4052D ULTRA checkpoint adapter: unavailable measured body contacts are
not replaced with geometric proximity. Existing reference hand intent retains
its validity mask. The original actor inputs remain as an additive prefix.
"""
from __future__ import annotations

from copy import deepcopy
import torch

from supermimic.rotations import axis_angle_to_mat, quat_to_mat, yaw_mat

BASE_PROFILE = "v1"
TEACHER_PROFILE = "privileged_teacher_v1"
PRIVILEGED_DIM = 3906
TEACHER_SCHEMA = {
    "version": TEACHER_PROFILE,
    "dim": PRIVILEGED_DIM,
    "ultra_source_commit": "ebe4aa276dd3208c33a8604084c0a3aeb37d9061",
    "ultra_source": "ultra/env/tasks/ultra_g1_retarget.py",
    "lookahead_control_frames": [1, 13],
    "lookahead_seconds": [.02, .26],
    "lookahead_mapping": "nearest 50Hz grid to ULTRA canonical 1/60 and 16/60 seconds",
    "rotation6d": "ULTRA concatenated x and z columns, not legacy row-interleaved x/y",
    "body_state": "39 bound-URDF link poses and nominal URDF-COM marker velocities; root input velocity is link-origin",
    "per_lookahead": ["body_current583", "body_target_and_errors936", "action_q_qd_prevq_prevqd145",
                      "object_velocity_and_errors21", "interaction_graph_and_error234"],
    "shared": ["raw_robot_context33", "object_context12", "current_object_pose_and_velocity15",
               "two_lookahead_expected_hand_contacts_and_validity8"],
    "actor_noise": False,
    "context_path": "normalized direct decoder input; original robot/object goal encoders retained",
    "interaction_graph": "nearest of existing 128 object surface points; body-minus-surface; d/(norm+1e-6)*exp(-5*norm)",
    "reference_ig_frame": "each future reference robot heading, matching ULTRA reference generation",
    "contact_inputs": "reference hand-object intent+validity only; actual body contacts deferred until native+WM schema extension",
    "differences_from_ultra": ["50Hz timebase", "existing 128-point geometry", "no stochastic heading perturbation",
                               "no measured-contact slots or zero torque placeholder", "explicit physical contexts",
                               "bound-URDF FK for actual, reference and imagined states", "original dense 10-frame goals retained"],
}


def validate_profile(profile):
    if profile not in (BASE_PROFILE, TEACHER_PROFILE):
        raise ValueError(f"unknown actor observation profile: {profile}")
    return profile


def feature_schema(profile, baseline):
    validate_profile(profile)
    result = deepcopy(baseline)
    if profile == TEACHER_PROFILE:
        result["version"] = "supermimic_hoi_privileged_teacher_1"
        result["privileged_actor"] = deepcopy(TEACHER_SCHEMA)
        result["actor"] += [f"privileged_direct{PRIVILEGED_DIM}"]
    return result


def tan_norm(R):
    return torch.cat((R[..., :, 0], R[..., :, 2]), -1)


class TeacherKinematics:
    """Matrix FK of the existing, hash-bound 39-link virtual URDF.

    Avoid a matrix->quaternion round trip inside differentiable imagination.
    The URDF parser and original joint/body identity are reused unchanged.
    """
    def __init__(self, virtual, joint_names):
        self.body_names = list(virtual.body_names)
        if len(self.body_names) != 39 or len(set(self.body_names)) != 39:
            raise ValueError("teacher requires 39 distinct bound URDF bodies")
        if set(virtual.dof_names) != set(joint_names) or len(joint_names) != 29:
            raise ValueError("teacher joint identity differs from the bound URDF")
        self.order = [joint_names.index(name) for name in virtual.dof_names]
        self.joints = []
        for parent, child, dof, xyz, quat_xyzw, axis in virtual.joints:
            R = quat_to_mat(torch.cat((quat_xyzw[3:], quat_xyzw[:3])))
            self.joints.append((parent, child, dof, xyz, R, axis))
        self.com = virtual.com_offsets

    def forward(self, S):
        p, R, v, w, q, qd = S
        q, qd = q[..., self.order], qd[..., self.order]
        P, Rs, V, W = [[None] * 39 for _ in range(4)]
        P[0], Rs[0], V[0], W[0] = p, R, v, w
        for parent, child, dof, xyz, origin_R, axis in self.joints:
            offset = (Rs[parent] @ xyz.to(p).unsqueeze(-1)).squeeze(-1)
            P[child] = P[parent] + offset
            pre_R = Rs[parent] @ origin_R.to(p)
            V[child] = V[parent] + torch.cross(W[parent], offset, dim=-1)
            if dof is None:
                Rs[child], W[child] = pre_R, W[parent]
            else:
                Rs[child] = pre_R @ axis_angle_to_mat(axis.to(p), q[..., dof])
                W[child] = W[parent] + (pre_R @ axis.to(p).unsqueeze(-1)).squeeze(-1) * qd[..., dof: dof + 1]
        P, Rs, V, W = torch.stack(P, -2), torch.stack(Rs, -3), torch.stack(V, -2), torch.stack(W, -2)
        V = V + torch.cross(W, (Rs @ self.com.to(p).unsqueeze(-1)).squeeze(-1), dim=-1)
        return P, Rs, V, W


class PrivilegedFeatures:
    def __init__(self, kin, motion, shapes):
        self.kin, self.motion, self.shapes = kin, motion, shapes

    @staticmethod
    def _rotate(Ht, vectors):
        return (Ht @ vectors.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def _ig(P, O, points, Ht):
        world = (O[1].unsqueeze(-3) @ points.unsqueeze(-1)).squeeze(-1) + O[0].unsqueeze(-2)
        # The nearest-point index is discrete in ULTRA too. Keep the large
        # 39x128 search out of the H-step graph, but differentiate the selected
        # point through both the robot and object pose.
        with torch.no_grad():
            delta = P.unsqueeze(-2) - world.unsqueeze(-3)
            index = delta.square().sum(-1).argmin(-1)
        nearest = world.gather(-2, index[..., None].expand(index.shape + (3,)))
        vector = P - nearest
        vector = PrivilegedFeatures._rotate(Ht.unsqueeze(-3), vector)
        length = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        return vector / (length + 1e-6) * torch.exp(-5 * length)

    def __call__(self, S, O, f, a_prev, xi, xi_o, previous_S):
        if xi.shape[-1] != 33 or xi_o.shape[-1] != 12:
            raise ValueError("teacher requires real robot33/object12 physical context")
        P, R, V, W = self.kin.forward(S)
        Ht = yaw_mat(S[1]).transpose(-1, -2)
        Hb = Ht.unsqueeze(-3)
        rotate = lambda x: self._rotate(Hb, x).flatten(-2)
        points = self.shapes.points(self.motion.asset_ids[f])
        ig = self._ig(P, O, points, Ht)
        current = (S[0][..., 2:3], rotate(P - S[0].unsqueeze(-2))[..., 3:],
                   tan_norm(Hb @ R).flatten(-2), rotate(V), rotate(W))
        history = (a_prev, S[4], S[5], previous_S[4], previous_S[5])
        blocks, intents = [], []
        end = self.motion.clip_end[self.motion.clip_id[f]]
        for offset in (1, 13):
            target_f = torch.minimum(f + offset, end)
            ref_S = tuple(x[target_f] for x in self.motion.ref_arrays())
            ref_O = self.motion.ref_object(target_f)
            # References are constants, never a second trainable dynamics path.
            with torch.no_grad():
                Pr, Rr, Vr, Wr = self.kin.forward(ref_S)
                ref_ig = self._ig(Pr, ref_O, points, yaw_mat(ref_S[1]).transpose(-1, -2))
            errors = (rotate(Pr - P), tan_norm(Hb @ (Rr.transpose(-1, -2) @ R) @ Hb.transpose(-1, -2)).flatten(-2),
                      rotate(Pr - S[0].unsqueeze(-2)), tan_norm(Hb @ Rr).flatten(-2), rotate(Vr - V), rotate(Wr - W))
            obj = (self._rotate(Ht, O[2]), self._rotate(Ht, O[3]), self._rotate(Ht, ref_O[0] - O[0]),
                   tan_norm(Ht @ (ref_O[1].transpose(-1, -2) @ O[1]) @ Ht.transpose(-1, -2)),
                   self._rotate(Ht, ref_O[2] - O[2]), self._rotate(Ht, ref_O[3] - O[3]))
            blocks.append(torch.cat((*current, *errors, *history, *obj, ig.flatten(-2), (ref_ig - ig).flatten(-2)), -1))
            expected, valid = self.motion.expected(target_f)
            valid = valid * ((f + offset) <= end).unsqueeze(-1)
            intents.extend((expected * valid, valid))
        # Also retain explicit current object pose, absent in ULTRA's task block.
        object_state = (self._rotate(Ht, O[0] - S[0]), tan_norm(Ht @ O[1]),
                        self._rotate(Ht, O[2]), self._rotate(Ht, O[3]))
        result = torch.cat((*blocks, xi, xi_o, *object_state, *intents), -1)
        if result.shape[-1] != PRIVILEGED_DIM:
            raise ValueError(f"privileged feature layout drift: {result.shape[-1]}")
        return result


def configure_teacher(mdp, bundle, device):
    from .evaluate import _load_fk39
    from supermimic.robot_metadata import G1_JOINT_NAMES
    virtual = _load_fk39(bundle, list(G1_JOINT_NAMES), device)
    mdp.privileged_features = PrivilegedFeatures(TeacherKinematics(virtual, list(G1_JOINT_NAMES)), mdp.motion, mdp.shapes)
