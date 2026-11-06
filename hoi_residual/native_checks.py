"""Read the real MuJoCo-Warp object parameters after a physics boundary.

Newton's object views are not sufficient evidence: property notifications must
have reached the solver. Call after stepping, including the first step after a
reset/DR update. This module never mutates model state or advances simulation.
"""
from __future__ import annotations

import numpy as np
import torch

from supermimic.g1_runtime import _entity_mapping
from supermimic.rotations import quat_to_mat


def _worlds(value, template_shape, nworld, name):
    """Require the pinned solver layout; expose shared rows without copying."""
    if tuple(value.shape) == tuple(template_shape):
        value = value.unsqueeze(0)
    if tuple(value.shape[1:]) != tuple(template_shape) or value.shape[0] not in (1, nworld):
        raise RuntimeError(f"unexpected MJWarp {name} layout: {tuple(value.shape)}")
    return value.expand(nworld, *value.shape[1:])


def _check(name, actual, expected, *, atol=1e-6, rtol=2e-5):
    expected = expected.to(device=actual.device, dtype=actual.dtype)
    if actual.shape != expected.shape or not bool(torch.isfinite(actual).all()):
        raise RuntimeError(f"invalid actual MJWarp object {name}")
    error = float((actual - expected).abs().max())
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise RuntimeError(f"object DR did not reach MJWarp {name}: maximum error={error}")
    return {"shape": list(actual.shape), "max_abs_error": error, "atol": atol, "rtol": rtol}


def audit_object_solver(env):
    """Verify every world's object inertia, COM, mass, friction and gravity."""
    import mujoco
    import warp as wp

    manager = env.sim.backend_manager
    solver, newton_model = manager._solver, manager.get_model()
    template, model = solver.mj_model, solver.mjw_model
    bodies, identity = _entity_mapping(template, {"Object"}, "body", newton_model)
    body = bodies["Object"]
    if (int(template.body_jntnum[body]) != 1 or
            int(template.jnt_type[int(template.body_jntadr[body])]) != int(mujoco.mjtJoint.mjJNT_FREE)):
        raise RuntimeError("native object must have exactly one free joint")
    geoms = np.flatnonzero(template.geom_bodyid == body).tolist()
    if len(geoms) != len(env.asset["collision_meshes"]):
        raise RuntimeError("native object collider count differs from its exact geometry binding")
    if any(int(template.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH) for g in geoms):
        raise RuntimeError("native object collision geometry unexpectedly changed type")
    fields = {}
    for name in ("body_mass", "body_ipos", "body_inertia", "body_iquat", "body_gravcomp", "geom_friction"):
        fields[name] = _worlds(wp.to_torch(getattr(model, name)),
                               np.asarray(getattr(template, name)).shape, env.N, name)
    randomizer = env.randomizer
    mass = fields["body_mass"][:, body]
    com = fields["body_ipos"][:, body]
    principal = fields["body_inertia"][:, body]
    quaternion = fields["body_iquat"][:, body]
    if not torch.allclose(quaternion.norm(dim=-1), torch.ones_like(mass), atol=1e-5, rtol=0):
        raise RuntimeError("native object principal-inertia rotation is invalid")
    rotation = quat_to_mat(quaternion)  # MuJoCo stores wxyz; tensor in body-link axes.
    inertia = rotation @ torch.diag_embed(principal) @ rotation.transpose(-1, -2)
    friction = fields["geom_friction"][:, geoms, 0]
    report = {
        "mass": _check("mass", mass, randomizer.mass),
        "com": _check("COM", com, randomizer.com, atol=2e-7),
        "inertia": _check("body-frame inertia", inertia, randomizer.inertia, atol=1e-8),
        "friction": _check("sliding friction", friction, randomizer.friction[:, None].expand_as(friction)),
        "gravcomp": _check("gravity compensation", fields["body_gravcomp"][:, body], torch.zeros_like(mass), atol=1e-7, rtol=0),
    }
    gravity = wp.to_torch(model.opt.gravity)
    if not bool(torch.isfinite(gravity).all()) or not bool((gravity[..., 2] < 0).all()):
        raise RuntimeError("actual MJWarp gravity must act downward")
    overflow = wp.to_torch(solver.mjw_data.overflow)
    if bool((overflow != 0).any()):
        raise RuntimeError("native HOI contact/constraint allocation overflow")
    report.update(passed=True, scope="actual MuJoCo-Warp parameters after physics step",
                  object_identity=identity["Object"], object_body_id=int(body),
                  object_geom_ids=geoms, worlds=int(env.N),
                  actual_mass=mass.detach().cpu().tolist(),
                  actual_friction=friction[:, 0].detach().cpu().tolist(),
                  gravity=gravity.detach().cpu().tolist(), overflow=False)
    return report
