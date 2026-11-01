"""Reference-relative HOI termination; shared motion-only MDP remains unchanged."""
from __future__ import annotations

import math
import torch
from supermimic.mdp import MDP


def gravity_angle_error(actual_rotation, reference_rotation):
    """Angle between world gravity expressed in each pelvis frame, in radians.

    R maps pelvis to world, so R[2, :] is body-frame world-up (both signs may
    be negated to obtain gravity). World yaw leaves it unchanged. atan2 of
    cross-norm and dot avoids acos's singular gradient at aligned rotations.
    """
    actual = actual_rotation[..., 2, :]
    reference = reference_rotation[..., 2, :]
    if actual.dtype in (torch.float16, torch.bfloat16):
        actual = actual.float()
    reference = reference.to(actual)
    sine = torch.linalg.vector_norm(torch.cross(actual, reference, dim=-1), dim=-1)
    cosine = (actual * reference).sum(-1)
    return torch.atan2(sine, cosine)


class ReferenceGravityMDP(MDP):
    """Keep human rewards/observations, replace only the HOI tilt predicate."""
    def __init__(self, *args, tilt_limit_rad, **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(tilt_limit_rad) or not 0 < tilt_limit_rad < math.pi:
            raise ValueError("HOI tilt angle limit must be finite and in (0, pi)")
        self.tilt_limit_rad = tilt_limit_rad

    def termination_errors(self, S, f):
        height = (self.motion.root_p[f, 2] - S[0][..., 2]).abs()
        tilt = gravity_angle_error(S[1], self.motion.root_R[f])
        return height, tilt

    def termination_components(self, S, f):
        height, tilt = self.termination_errors(S, f)
        return height > self.cfg.term_height, tilt > self.tilt_limit_rad

    def termination(self, S, f):
        height, tilt = self.termination_components(S, f)
        return height | tilt

    def soft_termination(self, S, f, tz=.02, tt=.04):
        """tz is metres; tt is radians. Same errors/thresholds as hard reset."""
        height, tilt = self.termination_errors(S, f)
        alive = (1 - torch.sigmoid((height - self.cfg.term_height) / tz)) * (
            1 - torch.sigmoid((tilt - self.tilt_limit_rad) / tt))
        return 1 - alive
