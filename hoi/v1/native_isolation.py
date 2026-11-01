"""Read-only byte fingerprints for small native cross-asset isolation audits.

This is an opt-in diagnostic, not a training/resource-probe default. It neither
steps physics nor changes solver options, RNG, tolerances, or determinism mode.
Object identities are not evidence here: actual buffer contents are downloaded
and hashed, including unused contact/constraint storage and its NaN bit patterns.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import json
import struct
from types import SimpleNamespace

import numpy as np
import torch


SCHEMA = "hoi_native_inactive_asset_byte_fingerprint_v1"
BOOKKEEPING_FIELDS = ("f", "ep_step", "a_prev", "xi", "start_bin", "ep_disc_ret", "ep_ret",
                      "bin_G", "bin_w_avg", "done_eval", "failed_eval", "G_max")
HOI_FIELDS = ("asset_id", "xi_o", "contact_missing_counter", "support_reset_poses",
              "support_reset_present", "support_alignment")
NATIVE_ROOTS = ("model", "state_0", "state_1", "control", "mjw_model", "mjw_data")


def _digest(payload, *, kind, shape, dtype, **metadata):
    return {"kind": kind, "shape": list(shape), "dtype": dtype, "nbytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(), **metadata}


def _numpy_bytes(array):
    if array.dtype.hasobject:
        raise TypeError("object arrays cannot supply stable physical buffer bytes")
    return array.tobytes(order="C")


def fingerprint_tree(roots, *, is_native_array=lambda value: False):
    """Fingerprint concrete data fields without evaluating runtime properties.

    Native roots may be ordinary instances; nested records must be dataclasses,
    SimpleNamespace, or containers. Device/module/callable runtime objects are
    skipped, so constructing another scene's JIT cache cannot masquerade as a
    mutation of the inactive asset. Aliased arrays retain every field path.
    """
    buffers, scalars, skipped = {}, {}, []
    array_cache = {}

    def walk(value, path, ancestors, *, root=False):
        if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray) or is_native_array(value):
            identity = id(value)
            if identity not in array_cache:
                if isinstance(value, torch.Tensor):
                    # numpy() does not support bfloat16; view bytes first. No
                    # floating-point comparison/canonicalization is performed.
                    host = value.detach().cpu().contiguous()
                    payload = host.reshape(-1).view(torch.uint8).numpy().tobytes()
                    result = _digest(payload, kind="torch", shape=value.shape, dtype=str(value.dtype))
                else:
                    host = value if isinstance(value, np.ndarray) else value.numpy()
                    result = _digest(_numpy_bytes(host), kind="numpy" if isinstance(value, np.ndarray) else "warp",
                                     shape=value.shape, dtype=str(value.dtype),
                                     host_shape=list(host.shape), host_dtype=str(host.dtype))
                array_cache[identity] = result
            buffers[path] = dict(array_cache[identity])
            return
        if value is None:
            scalars[path] = _digest(b"", kind="scalar", shape=(), dtype="None")
            return
        if isinstance(value, (bool, int, float, str, bytes, np.generic)):
            if isinstance(value, np.generic):
                payload, dtype = value.tobytes(), "numpy:" + str(value.dtype)
            elif isinstance(value, float):
                payload, dtype = struct.pack("<d", value), "python:float64"
            elif isinstance(value, bytes):
                payload, dtype = value, "python:bytes"
            else:
                payload, dtype = str(value).encode("utf-8"), "python:" + type(value).__name__
            scalars[path] = _digest(payload, kind="scalar", shape=(), dtype=dtype)
            return
        if id(value) in ancestors:
            skipped.append({"path": path, "reason": "container_cycle"})
            return
        nested = ancestors | {id(value)}
        if isinstance(value, dict):
            for key in sorted(value, key=lambda key: (type(key).__name__, str(key))):
                if not isinstance(key, (str, int)):
                    raise TypeError("fingerprint dictionaries require string/integer field names")
                walk(value[key], path + "." + str(key), nested)
        elif isinstance(value, (list, tuple)):
            scalars[path + ".__length__"] = _digest(str(len(value)).encode(), kind="scalar", shape=(), dtype="length")
            for index, item in enumerate(value):
                walk(item, path + f"[{index}]", nested)
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                walk(getattr(value, field.name), path + "." + field.name, nested)
        elif isinstance(value, SimpleNamespace) or (root and hasattr(value, "__dict__")):
            for key, item in sorted(vars(value).items()):
                walk(item, path + "." + key, nested)
        else:
            skipped.append({"path": path, "reason": "non_data_runtime_object",
                            "type": type(value).__module__ + "." + type(value).__qualname__})

    for name, value in sorted(roots.items()):
        walk(value, name, set(), root=True)
    content = {"buffers": buffers, "scalars": scalars, "skipped": skipped}
    return {**content, "aggregate_sha256": hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


class _NativeRuntime:
    def __init__(self):
        import warp as wp
        from isaaclab.sim import SimulationContext
        from .workers import _snapshot
        self.wp, self.context_type, self.snapshot = wp, SimulationContext, _snapshot

    def require_active(self, env):
        if self.context_type.instance() is not env.env.sim:
            raise RuntimeError("native fingerprint requires the asset's active simulation slot")

    def synchronize(self, device):
        self.wp.synchronize_device(str(device))

    def is_native_array(self, value):
        return isinstance(value, self.wp.array)


def capture_native_fingerprint(env, rng, *, _runtime=None):
    """Read an inactive asset while its simulation slot (but not RNG) is leased.

    Call outside ``rng.activate()`` so saved per-asset streams are authoritative.
    Every native root must contain at least one actual buffer; API/schema drift
    fails instead of silently reporting a fingerprint with missing physics.
    """
    runtime = _NativeRuntime() if _runtime is None else _runtime
    runtime.require_active(env)
    if rng._active:
        raise RuntimeError("capture inactive asset RNG outside rng.activate()")
    runtime.synchronize(env.device)
    base, sim = env.env, env.env.sim
    manager, solver = sim.backend_manager, env.contacts.solver
    native = {"model": manager.get_model(), "state_0": manager.get_state_0(),
              "state_1": manager.get_state_1(), "control": manager.get_control(),
              "mjw_model": solver.mjw_model, "mjw_data": solver.mjw_data}
    if any(value is None for value in native.values()):
        raise RuntimeError("native fingerprint requires every initialized physical model/state/control root")
    # Snapshot reads are the same published collector observations; they never
    # advance the solver. Materialize them before synchronizing and hashing.
    snapshot = runtime.snapshot(env)
    roots = {"native." + name: value for name, value in native.items()}
    roots.update(snapshot=snapshot,
                 bookkeeping={name: getattr(base, name) for name in BOOKKEEPING_FIELDS},
                 hoi={name: getattr(env, name) for name in HOI_FIELDS},
                 recorder={name: value for name, value in vars(env.contacts).items()
                           if isinstance(value, (torch.Tensor, np.ndarray, int, float, bool)) or runtime.is_native_array(value)},
                 sim={"physics_step_count": sim._physics_step_count, "render_generation": sim._render_generation,
                      "is_playing": sim._is_playing, "is_stopped": sim._is_stopped},
                 rng={"seed": rng.seed, "device": str(rng.device),
                      "cpu_state": rng._cpu_state, "cuda_state": rng._cuda_state})
    runtime.synchronize(env.device)
    result = fingerprint_tree(roots, is_native_array=runtime.is_native_array)
    runtime.synchronize(env.device)
    coverage = {name: sum(path.startswith("native." + name + ".") for path in result["buffers"])
                for name in NATIVE_ROOTS}
    if any(count == 0 for count in coverage.values()):
        raise RuntimeError("native fingerprint omitted actual buffers: " + str(coverage))
    if not any(path.startswith("native.mjw_data.contact.") for path in result["buffers"]):
        raise RuntimeError("native fingerprint omitted nested MJW contact buffers")
    if not any(path.startswith("native.mjw_data.efc.") for path in result["buffers"]):
        raise RuntimeError("native fingerprint omitted nested MJW constraint buffers")
    return {"schema": SCHEMA, "scope": "read-only inactive asset native buffer contents; no physics steps or RNG draws",
            "native_buffer_counts": coverage, **result}


def assert_native_unchanged(before, after, *, label="inactive asset"):
    """Require identical schemas and physical bytes; no numerical tolerance."""
    if before.get("schema") != SCHEMA or after.get("schema") != SCHEMA:
        raise ValueError("native isolation fingerprint schema mismatch")
    changed = []
    for section in ("buffers", "scalars"):
        left, right = before.get(section, {}), after.get(section, {})
        changed += [section + ":" + path for path in sorted(set(left) | set(right))
                    if left.get(path) != right.get(path)]
    if before.get("skipped") != after.get("skipped"):
        changed.append("skipped_field_inventory")
    if before.get("native_buffer_counts") != after.get("native_buffer_counts"):
        changed.append("native_buffer_coverage")
    if changed or before.get("aggregate_sha256") != after.get("aggregate_sha256"):
        raise AssertionError(label + " changed during another asset operation: " + ", ".join(changed[:16]) +
                             (f" (and {len(changed) - 16} more)" if len(changed) > 16 else ""))
    return {"unchanged": True, "buffers_checked": len(before["buffers"]),
            "scalar_fields_checked": len(before["scalars"]), "fingerprint_sha256": before["aggregate_sha256"]}
