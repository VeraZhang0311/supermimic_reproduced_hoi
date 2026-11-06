"""FQ hand-material friction multiplication on the pinned MuJoCo Warp path.

MuJoCo's high hand priority preserves the soft normal material, but also
selects hand mu=1 over an object's material. PhysX's hand material instead
uses multiply. The official model-level ``callback.contactfilter`` runs
after narrowphase and before contact constraints are built; this adapter
changes only the friction vectors of actual hand contacts there.

The declared GMT world planes are excluded: their native mu=.3 is a max-
mixing convention for randomized body contacts, while the source terrain
material is nominally mu=1. Hand/ground therefore retains effective mu=1.
All other hand/body/object contacts use the product of native geom values.
No global function, source package or non-hand contact is modified.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
from typing import Any

import numpy as np


VERSION = "fengqiao_hw3_mjwarp_contact_multiply_v1"
SUPPORTED_MJWARP = "3.15.0"
# Exact callback source inspected on the matching Windows validation wheel
# (Python 3.11) and the authorized Linux runtime (Python 3.12). ast.dump
# differs across these Python versions because FunctionDef adds type_params.
SUPPORTED_COLLISION_AST = {
    "cae7ead5a311201f3833a6a7c75a0234b612acf512482b623486d840ec6f2870",
    "832158ed3957df093e76f71a2b4d5c7d28fa6bc081dba4d1b1adc678af5201d1",
}


def _backend_contract():
    import mujoco_warp
    from mujoco_warp._src import collision_driver
    if mujoco_warp.__version__ != SUPPORTED_MJWARP:
        raise RuntimeError(f"Hand contact multiplication requires verified MuJoCo Warp {SUPPORTED_MJWARP}; got {mujoco_warp.__version__}")
    try:
        node = ast.parse(inspect.getsource(inspect.unwrap(collision_driver.collision))).body[0]
        digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    except (TypeError, OSError, SyntaxError) as error:
        raise RuntimeError("Cannot verify MuJoCo Warp's post-narrowphase contact hook") from error
    if digest not in SUPPORTED_COLLISION_AST:
        raise RuntimeError(f"MuJoCo Warp collision callback order changed: AST SHA {digest}; revalidate before use")
    return {"version": VERSION, "mujoco_warp": mujoco_warp.__version__,
            "collision_ast_sha256": digest, "hook": "model.callback.contactfilter after narrowphase",
            "mixing": "hand geom friction * other geom friction; declared GMT ground contacts unchanged"}


def _mix_kernel(wp):
    @wp.kernel
    def multiply(
        geom_friction: wp.array(dtype=wp.vec3, ndim=2),
        hand: wp.array(dtype=wp.int32),
        ground: wp.array(dtype=wp.int32),
        active_count: wp.array(dtype=wp.int32),
        contact_geoms: wp.array(dtype=wp.vec2i),
        contact_world: wp.array(dtype=wp.int32),
        contact_dim: wp.array(dtype=wp.int32),
        contact_friction: wp.array(dtype=Any),
    ):
        index = wp.tid()
        if index >= active_count[0]:
            return
        pair = contact_geoms[index]
        a, b = pair[0], pair[1]
        # Flex contacts and deliberately filtered/frictionless contacts stay
        # owned by the original callback/engine.
        if a < 0 or b < 0 or contact_dim[index] < 3:
            return
        if hand[a] == 0 and hand[b] == 0:
            return
        if ground[a] != 0 or ground[b] != 0:
            return
        world = contact_world[index] % geom_friction.shape[0]
        left, right = geom_friction[world, a], geom_friction[world, b]
        friction = contact_friction[index]
        friction[0] = left[0] * right[0]
        friction[1] = friction[0]
        friction[2] = left[1] * right[1]
        friction[3] = left[2] * right[2]
        friction[4] = friction[3]
        contact_friction[index] = friction
        # Zero sliding friction on this condim=3 hand profile must become
        # frictionless, rather than feed zero coefficients to a friction cone.
        if friction[0] == 0.0 and friction[2] == 0.0 and friction[3] == 0.0:
            contact_dim[index] = 1
    return multiply


class HandContactMix:
    """Owned model callback; retain this object for the simulation lifetime."""
    def __init__(self, mjw_model, hand_geom_ids, *, ground_geom_ids=()):
        import warp as wp
        self.wp, self.model = wp, mjw_model
        self.contract = _backend_contract()
        count = int(mjw_model.ngeom)
        hand = sorted(set(map(int, hand_geom_ids)))
        ground = sorted(set(map(int, ground_geom_ids)))
        if not hand or any(i < 0 or i >= count for i in hand + ground) or set(hand) & set(ground):
            raise ValueError("Hand/ground contact masks must be nonempty, disjoint and in model geom range")
        if not hasattr(mjw_model, "callback") or not hasattr(mjw_model.callback, "contactfilter"):
            raise RuntimeError("Pinned MuJoCo Warp model lacks its post-narrowphase contact callback")
        self.previous = mjw_model.callback.contactfilter
        if isinstance(self.previous, HandContactMix):
            raise RuntimeError("Hand contact multiplication is already installed on this model")
        if self.previous is not None and not callable(self.previous):
            raise TypeError("Existing contactfilter must be callable")
        mu = mjw_model.geom_friction
        if mu.ndim != 2 or mu.shape[1] != count or mu.dtype != wp.vec3:
            raise RuntimeError("Unsupported native geom_friction layout for hand multiplication")
        masks = np.zeros((2, count), dtype=np.int32)
        masks[0, hand] = 1; masks[1, ground] = 1
        self.hand = wp.array(masks[0], dtype=wp.int32, device=mu.device)
        self.ground = wp.array(masks[1], dtype=wp.int32, device=mu.device)
        self.kernel = _mix_kernel(wp)
        self.contract.update(hand_geom_ids=hand, preserved_ground_geom_ids=ground,
                             chains_existing_callback=self.previous is not None)
        mjw_model.callback.contactfilter = self

    def __call__(self, model, data):
        if model is not self.model:
            raise RuntimeError("Hand contact callback was attached to a different solver model")
        if self.previous is not None:
            self.previous(model, data)
        self.wp.launch(self.kernel, dim=data.naconmax,
                       inputs=[model.geom_friction, self.hand, self.ground, data.nacon,
                               data.contact.geom, data.contact.worldid, data.contact.dim, data.contact.friction],
                       device=model.geom_friction.device)

    def ensure_installed(self):
        if self.model.callback.contactfilter is not self:
            raise RuntimeError("The shared hand multiply callback was replaced; revalidate callback composition")

    def close(self):
        self.ensure_installed()
        self.model.callback.contactfilter = self.previous


def install_hand_contact_multiply(mjw_model, hand_geom_ids, *, ground_geom_ids=()):
    """Install before first simulation/CUDA-graph capture; return owner/receipt."""
    return HandContactMix(mjw_model, hand_geom_ids, ground_geom_ids=ground_geom_ids)
