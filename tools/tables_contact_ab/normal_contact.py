"""Model-local foot/object solref probe, composed inside the existing hand hook.

No model geom/material/controller field is edited. Only the first solref
component of actual foot/object contacts is optionally replaced, after
narrowphase and before constraint construction. All hand contacts remain owned
by the frozen HandContactMix callback.
"""
from __future__ import annotations


def _kernel(wp):
    @wp.kernel
    def apply(count: wp.array(dtype=wp.int32),
              pair: wp.array(dtype=wp.vec2i), roles: wp.array(dtype=wp.int32),
              solref: wp.array(dtype=wp.vec2), original: wp.array(dtype=wp.vec2),
              time_constant: float):
        i = wp.tid()
        if i < count[0]:
            value = solref[i]
            original[i] = value
            a, b = pair[i][0], pair[i][1]
            if a >= 0 and b >= 0 and time_constant > 0.0:
                ra, rb = roles[a], roles[b]
                if ((ra == 2 and (rb == 3 or rb == 4)) or
                    (rb == 2 and (ra == 3 or ra == 4))):
                    value[0] = time_constant
                    solref[i] = value
    return apply


class FootTableNormalProbe:
    def __init__(self, hand_owner, roles, capacity, time_constant=None):
        import numpy as np
        import warp as wp
        from supermimic.g1_contactmix import HandContactMix, _backend_contract
        if not isinstance(hand_owner, HandContactMix):
            raise TypeError('Probe requires the unchanged hand-contact owner')
        hand_owner.ensure_installed()
        _backend_contract()
        self.hand = hand_owner
        self.model = hand_owner.model
        self.previous = hand_owner.previous
        self.wp = wp
        self.time_constant = 0. if time_constant is None else float(time_constant)
        if self.time_constant not in (0., .01):
            raise ValueError('This fixed A/B only permits baseline or 10ms')
        roles = np.asarray(roles, dtype=np.int32)
        if roles.shape != (self.model.ngeom,) or not all((roles == r).any() for r in (2, 3, 4)):
            raise ValueError('Require explicit native table and both-foot geom roles')
        self.roles = wp.array(roles, dtype=wp.int32, device=self.model.geom_friction.device)
        self.original = wp.empty(int(capacity), dtype=wp.vec2, device=self.roles.device)
        self.kernel = _kernel(wp)
        hand_owner.previous = self

    def __call__(self, model, data):
        if model is not self.model:
            raise RuntimeError('Probe attached to another model')
        if self.previous is not None:
            self.previous(model, data)
        self.wp.launch(self.kernel, dim=data.naconmax,
            inputs=[data.nacon, data.contact.geom, self.roles, data.contact.solref,
                    self.original, self.time_constant], device=self.roles.device)

    def ensure_installed(self):
        self.hand.ensure_installed()
        if self.hand.previous is not self:
            raise RuntimeError('Foot probe callback chain changed')

    def close(self):
        self.ensure_installed()
        self.hand.previous = self.previous
