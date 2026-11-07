"""MuJoCo-Warp allocation settings for residual HOI and native failure readback."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import sys


DEFAULT_HORIZON = 24
_requested_horizon = DEFAULT_HORIZON
_observed_horizon = None
_native_sources = {
    "collision_convex.py": "3fd288f0cab3f5f444924d724362fd2cd2eab1281ab2dcb6bc3b48b9b3f30b92",
    "collision_gjk.py": "c288f8f37e4b7f8f9e0e48a92a46c3eee0d0efc6dd7fac65bba2522e11f39e48",
}


def capacity_profile(nconmax=512, njmax=2048, epa_horizon=DEFAULT_HORIZON):
    if any(type(value) is not int for value in (nconmax, njmax, epa_horizon)):
        raise ValueError("Native solver capacities must be integers")
    if nconmax < 512 or njmax < 2048 or epa_horizon < DEFAULT_HORIZON:
        raise ValueError("Residual HOI capacities must cover the original 512/2048/24 allocation")
    result = {"nconmax": nconmax, "njmax": njmax}
    if epa_horizon != DEFAULT_HORIZON:
        result["epa_horizon"] = epa_horizon
    return result


def checkpoint_capacity(value):
    if not isinstance(value, dict) or set(value) not in ({"nconmax", "njmax"}, {"nconmax", "njmax", "epa_horizon"}):
        raise ValueError("Invalid residual checkpoint solver capacity")
    return capacity_profile(**value)


def configure_capacity(sim_cfg, profile):
    """Apply before constructing the simulator, including its CUDA graph capture.

    MJWarp 3.15 rigid convex collision allocates its horizon on the host from
    this module constant. Each Slurm learner/evaluator is a separate process.
    Larger capacity does not change any contact equations or iteration limits.
    """
    global _requested_horizon, _observed_horizon
    profile = checkpoint_capacity(profile)
    sim_cfg.physics.solver_cfg.nconmax = profile["nconmax"]
    sim_cfg.physics.solver_cfg.njmax = profile["njmax"]
    horizon = profile.get("epa_horizon", DEFAULT_HORIZON)
    if horizon == DEFAULT_HORIZON and _requested_horizon == DEFAULT_HORIZON:
        return
    import mujoco_warp
    from mujoco_warp._src import collision_convex, collision_driver, types

    if mujoco_warp.__version__ != "3.15.0" or types.MJ_MAX_EPAHORIZON != DEFAULT_HORIZON:
        raise RuntimeError("EPA horizon expansion requires the verified 3.15.0 solver")
    modules = Path(collision_convex.__file__).resolve().parent
    for name, expected in _native_sources.items():
        source = (modules / name).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(source).hexdigest() != expected:
            raise RuntimeError(f"Unverified MJWarp EPA implementation: {name}")
    original = collision_convex.convex_narrowphase
    if (collision_convex.MJ_MAX_EPAHORIZON != _requested_horizon or
            collision_driver.convex_narrowphase not in (original, getattr(configure_capacity, "_probe", None))):
        raise RuntimeError("Another hook changed the native EPA capacity/dispatcher")
    collision_convex.MJ_MAX_EPAHORIZON = horizon
    _requested_horizon = horizon
    _observed_horizon = None
    collision_driver.convex_narrowphase = original
    if horizon == DEFAULT_HORIZON:
        return

    # Observe the actual native array once, even when Newton captures collision
    # into a CUDA graph. Retain its dimensions only, never the scratch buffer.
    native_code = inspect.unwrap(original).__code__

    def probe(*args, **kwargs):
        global _observed_horizon
        previous = sys.getprofile()

        def observe(frame, event, arg):
            global _observed_horizon
            if event == "return" and frame.f_code is native_code:
                local = frame.f_locals
                scratch = local.get("epa_horizon")
                if scratch is not None:
                    data = local["d"]
                    shape = tuple(int(n) for n in scratch.shape)
                    if shape != (int(data.naccdmax), horizon):
                        raise RuntimeError(f"Native EPA workspace shape differs from requested allocation: {shape}")
                    _observed_horizon = (id(data), shape)
            if callable(previous):
                previous(frame, event, arg)

        sys.setprofile(observe)
        try:
            return original(*args, **kwargs)
        finally:
            sys.setprofile(previous)
            if _observed_horizon is not None:
                collision_driver.convex_narrowphase = original

    configure_capacity._probe = probe
    collision_driver.convex_narrowphase = probe


def capacity_readback(physics):
    data = physics.solver.mjw_data
    result = {"contacts_capacity": int(data.naconmax), "ccd_candidate_capacity": int(data.naccdmax),
              "world_constraint_capacity": int(data.njmax), "epa_horizon": _requested_horizon}
    if _requested_horizon != DEFAULT_HORIZON:
        if _observed_horizon is None or _observed_horizon[0] != id(data):
            raise RuntimeError("Expanded EPA workspace was not observed on this native solver")
        shape = _observed_horizon[1]
        result.update(epa_horizon_shape=list(shape), epa_horizon_bytes=4 * shape[0] * shape[1],
                      additional_epa_bytes=4 * shape[0] * (shape[1] - DEFAULT_HORIZON))
    return result


def overflow_detail(physics):
    """Decode the native sticky bits and occupancy while the failed substep is still resident."""
    from mujoco_warp._src.types import OverflowType
    import numpy as np
    data = physics.solver.mjw_data
    mask = data.overflow.numpy().astype(np.int64)
    nefc = data.nefc.numpy()
    worlds = np.flatnonzero(mask)
    values = [entry for entry in OverflowType if entry.value and entry.value & (entry.value - 1) == 0]
    counts = {entry.name: int(np.count_nonzero(mask & entry.value)) for entry in values if (mask & entry.value).any()}
    result = {"overflow_world_count": int(len(worlds)), "overflow_bit_counts": counts,
              "overflow_worlds": [{"world": int(i), "mask": int(mask[i]), "nefc": int(nefc[i]),
                                   "reasons": [entry.name for entry in values if mask[i] & entry.value]}
                                  for i in worlds],
              "contacts": int(data.nacon.numpy()[0]), "contacts_capacity": int(data.naconmax),
              "world_constraints_peak": int(nefc.max()), "world_constraints_peak_id": int(nefc.argmax()),
              "world_constraint_capacity": int(data.njmax), "ccd_candidate_capacity": int(data.naccdmax),
              "broadphase_candidates": int(data.ncollision.numpy()[0]), "epa_horizon": _requested_horizon}
    if "EPA_HORIZON" in counts:
        # _add_edge reports a failed append at capacity; it never records the ultimate required count.
        result["epa_required_edges_at_least"] = _requested_horizon + 1
    return result
