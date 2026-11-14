"""Level-batched FK using the same GMT transforms, axes, and velocity conventions."""
from collections import defaultdict

import torch

from supermimic.rotations import axis_angle_to_mat


def privileged_call(features, state, frames, action, context, q_prev, qd_prev, actuator, *, bodies):
    from supermimic.privileged import PRIVILEGED_DIM, tan_norm
    from supermimic.rotations import yaw_mat

    P, R, V, W = bodies
    Hb = yaw_mat(state[1]).transpose(-1, -2).unsqueeze(-3)

    def rotate(value):
        return (Hb @ value.unsqueeze(-1)).squeeze(-1).flatten(-2)

    root = state[0].unsqueeze(-2)
    current = (state[0][..., 2:3], rotate(P - root)[..., 3:], tan_norm(Hb @ R).flatten(-2), rotate(V), rotate(W))
    history = (action, state[4], state[5], q_prev, qd_prev)
    motion = features.motion
    end = motion.clip_end[motion.clip_id[frames]]
    blocks = []
    for offset in (1, 13):
        Pr, Rr, Vr, Wr = motion.privileged_reference(torch.minimum(frames + offset, end))
        errors = (rotate(Pr - P), tan_norm(Hb @ (R @ Rr.transpose(-1, -2)) @ Hb.transpose(-1, -2)).flatten(-2),
                  rotate(Pr - root), tan_norm(Hb @ Rr).flatten(-2), rotate(Vr - V), rotate(Wr - W))
        blocks.append(torch.cat((*current, *errors, *history), -1))
    result = torch.cat((*blocks, context, actuator), -1)
    if result.shape[-1] != PRIVILEGED_DIM:
        raise ValueError("privileged feature layout drift")
    return result


class BatchedFK:
    def __init__(self, kin, *, privileged):
        self.kin = kin
        self.privileged = privileged
        groups = defaultdict(list)
        level = {0: 0}
        if privileged:
            joints = kin.joints
        else:
            joints = [(parent, child, kin.joint_idx[child], kin.off_p[child], kin.off_R[child], kin.axes[child])
                      for child, parent in enumerate(kin.parent) if parent >= 0]
        for joint in joints:
            parent, child = joint[:2]
            level[child] = level[parent] + 1
            groups[level[child]].append(joint)
        order = [0]
        self.groups = []
        for depth in sorted(groups):
            parent, child, dof, xyz, origin, axis = zip(*groups[depth], strict=True)
            device = xyz[0].device
            self.groups.append((torch.tensor([order.index(i) for i in parent], device=device),
                                torch.tensor([29 if i is None else i for i in dof], device=device),
                                torch.stack(xyz), torch.stack(origin), torch.stack(axis)))
            order.extend(child)
        wanted = range(len(order)) if privileged else kin.tracked_idx
        self.indices = torch.tensor([order.index(i) for i in wanted], device=device)
        self.all_indices = torch.tensor([order.index(i) for i in range(len(order))], device=device)

    def __getattr__(self, name):
        return getattr(self.kin, name)

    def forward(self, *state, all_bodies=False):
        if self.privileged:
            state = state[0]
        p, R, v, w, q, qd = state
        P, Rs, V, W = p.unsqueeze(-2), R.unsqueeze(-3), v.unsqueeze(-2), w.unsqueeze(-2)
        q = torch.cat((q, torch.zeros_like(q[..., :1])), dim=-1)
        qd = torch.cat((qd, torch.zeros_like(qd[..., :1])), dim=-1)
        for parent, dof, xyz, origin, axis in self.groups:
            parent_p, parent_r = P[..., parent, :], Rs[..., parent, :, :]
            parent_v, parent_w = V[..., parent, :], W[..., parent, :]
            offset = (parent_r @ xyz.unsqueeze(-1)).squeeze(-1)
            pos = parent_p + offset
            pre = parent_r @ origin
            rot = pre @ axis_angle_to_mat(axis, q[..., dof])
            ang = parent_w + (pre @ axis.unsqueeze(-1)).squeeze(-1) * qd[..., dof, None]
            if not self.privileged:
                offset = pos - parent_p
            vel = parent_v + torch.cross(parent_w, offset, dim=-1)
            P, Rs = torch.cat((P, pos), dim=-2), torch.cat((Rs, rot), dim=-3)
            V, W = torch.cat((V, vel), dim=-2), torch.cat((W, ang), dim=-2)
        indices = self.all_indices if all_bodies else self.indices
        P, Rs, V, W = P[..., indices, :], Rs[..., indices, :, :], V[..., indices, :], W[..., indices, :]
        if self.privileged:
            V = V + torch.cross(W, (Rs @ self.kin.com.unsqueeze(-1)).squeeze(-1), dim=-1)
        return P, Rs, V, W

    def bind_reference(self, motion, *, batch=8192):
        """Frozen references use the same FK every time an RSI frame recurs."""
        from hoi_residual.data import Indexed

        with torch.no_grad():
            pieces = []
            for first in range(0, motion.num_frames, batch):
                state = motion.ref_state(slice(first, first + batch))
                pieces.append(self.forward(state) if self.privileged else self.forward(*state))
            packed = tuple(torch.cat(values) for values in zip(*pieces, strict=True))
        def lookup(frame):
            return tuple(value[frame] for value in packed)
        if self.privileged:
            motion.privileged_reference = lookup
        else:
            motion.reference_bodies = lookup
            for name, value in zip(("body_p", "body_R", "body_v", "body_w"), packed, strict=True):
                setattr(motion, name, Indexed(lambda f, value=value: value[f], motion.num_frames))
