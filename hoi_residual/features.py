"""Explicit, heading-invariant oracle schema; the GMT prefix is unmodified."""
import torch
from supermimic.privileged import PRIVILEGED_DIM, tan_norm
from supermimic.rotations import yaw_mat

ORACLE_DIM = PRIVILEGED_DIM + 15 + 2 * 15 + 12 + 8 + 6
SCHEMA = "hoi_oracle_hw3_gmt3371_object71_v1"
BLOCKS = {"robot_privileged": 3371, "object_current": 15,
          "object_errors_lookahead_1_13": 30, "object_physics": 12,
          "asset_onehot": 8, "local_shape_bounds": 6}


def oracle_features(robot_privileged, S, O, motion, frames, object_context, asset_id, bounds):
    H = yaw_mat(S[1]).transpose(-1, -2)
    rot = lambda x: (H @ x.unsqueeze(-1)).squeeze(-1)
    p, R, v, w = O
    current = (rot(p - S[0]), tan_norm(H @ R), rot(v), rot(w))
    ends = motion.clip_end[motion.clip_id[frames]]
    goals = []
    for lookahead in (1, 13):
        pr, Rr, vr, wr = motion.object_ref(torch.minimum(frames + lookahead, ends))
        goals.extend((rot(pr-p), tan_norm(H @ Rr @ R.transpose(-1, -2) @ H.transpose(-1, -2)),
                      rot(vr-v), rot(wr-w)))
    identity = torch.nn.functional.one_hot(asset_id, 8).to(p)
    result = torch.cat((robot_privileged, *current, *goals, object_context, identity,
                        bounds.expand(len(p), -1)), -1)
    if result.shape[-1] != ORACLE_DIM or not torch.isfinite(result).all():
        raise ValueError("HOI oracle shape or finite-value contract violated")
    return result
