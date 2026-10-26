"""Independent native DR checks, run only by explicitly requested validation workers."""
from __future__ import annotations

import math
import re

import torch

from supermimic.rotations import quat_to_mat
from .robot_context import mass_context_from


def _close(actual, expected, name):
    if not bool(torch.isfinite(actual).all()) or not torch.allclose(actual, expected, atol=2e-6, rtol=2e-5):
        error = float((actual - expected).abs().max())
        raise RuntimeError(f"native DR {name} differs; max absolute error={error}")


def _properties(host):
    import warp as wp
    e, obj = host.env, host.object
    model = e.sim.backend_manager.get_model()
    material = wp.to_torch(obj._root_view.get_attribute("shape_material_mu", model)).reshape(host.N, -1)
    return {"robot_mass": e.robot.data.body_mass.torch.clone(),
            "robot_inertia": e.robot.data.body_inertia.torch.reshape(host.N, -1, 3, 3).clone(),
            "robot_com": e.robot.data.body_com_pose_b.torch[..., :3].clone(),
            "object_mass": obj.data.body_mass.torch[:, 0].clone(),
            "object_com": obj.data.body_com_pose_b.torch[:, 0, :3].clone(),
            "object_inertia": obj.data.body_inertia.torch.reshape(host.N, 3, 3).clone(),
            "object_friction": material.clone(), "xi_o": host.xi_o.clone(),
            **host.robot_mass_fields()}


def _solver_readback(host, props):
    """Check the effective MuJoCo-Warp model, after a real integration interval."""
    import mujoco
    import warp as wp
    solver = host.env.sim.backend_manager._solver
    mj, model, n = solver.mj_model, solver.mjw_model, host.N
    def ids_for(kind):
        ids = [i for i in range(mj.nbody) if (mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, i) or "").endswith(kind)]
        if len(ids) not in (1, n):
            raise RuntimeError(f"unexpected compiled {kind} body count {len(ids)} for {n} environments")
        if len(ids) == n:
            def env_id(i):
                name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, i)
                match = re.search(r"env[_/](\d+)", name)
                if match is None:
                    raise RuntimeError(f"compiled body has no environment identity: {name}")
                return int(match[1])
            ids.sort(key=env_id)
            if [env_id(i) for i in ids] != list(range(n)):
                raise RuntimeError("compiled DR body environment IDs differ")
        return ids * n if len(ids) == 1 else ids
    def read(name, count, shape, ids):
        raw = wp.to_torch(getattr(model, name))
        width = count * math.prod(shape)
        if raw.numel() % width:
            raise RuntimeError(f"unexpected effective {name} shape: {tuple(raw.shape)}")
        value = raw.reshape(-1, count, *shape)
        if len(value) not in (1, n):
            raise RuntimeError(f"unexpected effective {name} world count: {len(value)}")
        rows = torch.arange(n, device=raw.device) if len(value) == n else torch.zeros(n, device=raw.device, dtype=torch.long)
        return value[rows, torch.tensor(ids, device=raw.device)].clone()
    errors = {}
    for kind, prefix, index in (("torso_link", "robot", host.env.torso_id), ("Object", "object", None)):
        ids = ids_for(kind)
        mass = read("body_mass", mj.nbody, (), ids)
        com = read("body_ipos", mj.nbody, (3,), ids)
        inertia = read("body_inertia", mj.nbody, (3,), ids)
        rotation = quat_to_mat(read("body_iquat", mj.nbody, (4,), ids))
        tensor = rotation @ torch.diag_embed(inertia) @ rotation.transpose(-1, -2)
        for name, actual in (("mass", mass), ("com", com), ("inertia", tensor)):
            expected = props[prefix + "_" + name]
            if index is not None:
                expected = expected[:, index]
            _close(actual, expected, "effective_solver_" + prefix + "_" + name)
            errors[prefix + "_" + name] = float((actual - expected).abs().max())
        if index is None:
            for env_id, body in enumerate(ids):
                geoms = [i for i in range(mj.ngeom) if int(mj.geom_bodyid[i]) == body]
                if not geoms:
                    raise RuntimeError("compiled object has no collision geoms")
                for geom in geoms:
                    actual = read("geom_friction", mj.ngeom, (3,), [geom] * n)[env_id, 0]
                    _close(actual, props["object_friction"][env_id, 0], "effective_solver_object_friction")
    return errors


def validate_native_randomization(host):
    if host.N < 2 or not host.torso_mass_context:
        raise ValueError("native DR validation needs at least two environments and torso mass context")
    e, dr = host.env, host.randomizer
    nominal = bool(e.eval_mode)
    if nominal != (dr is None):
        raise RuntimeError("nominal evaluation must disable DR")
    torso = e.torso_id
    props = _properties(host)
    nominal_object = {"mass": torch.tensor(host.physics_report["mass"], device=host.device),
                      "com": torch.tensor(host.physics_report["com"], device=host.device),
                      "inertia": torch.tensor(host.physics_report["inertia"], device=host.device),
                      "friction": torch.tensor(host.physics_report["object_friction"], device=host.device)}
    def check(values):
        context = mass_context_from(values)
        if nominal:
            _close(context, torch.zeros_like(context), "nominal_torso_mass_context")
            _close(e.xi[:, 1:], torch.zeros_like(e.xi[:, 1:]), "nominal_robot_DR_context")
            for name, expected in nominal_object.items():
                actual = values["object_" + name]
                if name == "friction": expected = expected[:, None].expand_as(actual)
                _close(actual, expected, "nominal_object_" + name)
        else:
            delta = values["robot_mass"][:, torso] - dr.nominal_robot_mass[:, torso]
            if bool((delta.abs() > 3.00001).any()): raise RuntimeError("torso mass outside +/-3kg")
            ratio = values["robot_mass"][:, torso] / dr.nominal_robot_mass[:, torso]
            _close(values["robot_inertia"][:, torso], dr.nominal_robot_inertia.reshape(host.N, -1, 3, 3)[:, torso] * ratio[:, None, None], "torso_mass_inertia_scaling")
            other = torch.arange(values["robot_mass"].shape[1], device=host.device) != torso
            _close(values["robot_mass"][:, other], dr.nominal_robot_mass[:, other], "other_robot_link_masses")
            lower = torch.maximum(dr.nominal_object_mass - 1.5, .1 * dr.nominal_object_mass)
            if bool(((values["object_mass"] < lower) | (values["object_mass"] > dr.nominal_object_mass + 1.50001)).any()):
                raise RuntimeError("object mass outside sampling support")
            if bool(((values["object_com"] - dr.nominal_object_com).abs() > .010001).any()):
                raise RuntimeError("object COM outside +/-1cm")
            if bool(((values["object_friction"] < .4) | (values["object_friction"] > .800001)).any()):
                raise RuntimeError("object friction outside [0.4,0.8]")
            _close(values["object_inertia"], dr.nominal_object_inertia * (values["object_mass"] / dr.nominal_object_mass)[:, None, None], "object_mass_inertia_scaling")
        _close(values["torso_mass"].squeeze(-1), values["robot_mass"][:, torso], "recorded_mass_readback")
        _close(values["xi_o"][:, 0], values["object_mass"], "recorded_object_mass")
        _close(values["xi_o"][:, 1:4], values["object_com"], "recorded_object_COM")
    check(props)
    errors, resets = [], []
    for ids in (torch.arange(0, host.N, 2, device=host.device), torch.arange(1, host.N, 2, device=host.device)):
        before = _properties(host)
        frames = e.f[ids].clone()
        host.reset(ids, frames)
        after = _properties(host)
        keep = torch.ones(host.N, dtype=torch.bool, device=host.device)
        keep[ids] = False
        for name in before:
            _close(after[name][keep], before[name][keep], "subset_untouched_" + name)
        _close(after["robot_mass"], before["robot_mass"], "static_robot_mass_across_reset")
        if not nominal and torch.equal(after["object_mass"][ids], before["object_mass"][ids]):
            raise RuntimeError("selected object masses were not resampled")
        check(after)
        pair = host._advance(torch.zeros(host.N, 29, device=host.device))
        if any(not bool(torch.isfinite(value).all()) for value in pair.values() if isinstance(value, torch.Tensor)):
            raise RuntimeError("nonfinite real native DR integration")
        actual = _properties(host)
        check(actual)
        errors.append(_solver_readback(host, actual))
        resets.append(ids.cpu().tolist())
    host.reset(torch.arange(host.N, device=host.device), e.f.clone())
    return {"passed": True, "mode": "nominal_eval" if nominal else "training_DR",
            "asset_id": int(host.asset_id[0]), "num_envs": host.N,
            "native_property_readback_passed": True, "effective_MuJoCo_Warp_properties_verified": True,
            "subset_reset_passed": True, "subset_reset_ids": resets,
            "real_integrated_intervals": 2, "solver_readback_max_absolute_errors": errors,
            "torso_mass_kg": actual["torso_mass"].squeeze(-1).cpu().tolist(),
            "torso_mass_nominal_kg": actual["torso_mass_nominal"].squeeze(-1).cpu().tolist(),
            "object_mass_kg": actual["object_mass"].cpu().tolist()}
