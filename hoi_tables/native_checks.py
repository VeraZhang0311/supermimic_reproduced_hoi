"""Native audit with high-precision reconstruction of float32 inertia storage.

MuJoCo stores principal inertias and a principal-frame quaternion separately.
The diagnostic normalizes the stored quaternion and reconstructs R diag(I) R.T
in float64, avoiding a second float32 rounding pass on top of native storage.
The native quaternion unit-norm hard check still executes before this readback.

Newton converts the tensor through Warp's fixed four-sweep float32 eig3,
descending eigenpair sort, proper-rotation correction and quaternion encoding.
Its approximation residual is not bounded by eight storage epsilons. The audit
therefore independently encodes the requested tensor with that exact pinned
official kernel into private temporary arrays, and requires the native result
to match that reference within eight float32 epsilons in matrix norm.

A separate physical guard requires both reconstructed tensors to be within
the inherited 2e-5 relative tolerance of the requested tensor in Frobenius norm.
This prevents the reference encoding from concealing an inaccurate conversion;
the strict reference check independently rejects stale or incorrect native data.
Mass, COM, friction, gravity, identity and overflow checks retain the exact
original logic and checker. No model property or physics is changed.
"""
from contextlib import nullcontext
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import torch

from hoi_residual import native_checks as _original

np = _original.np
_worlds = _original._worlds
_entity_mapping = _original._entity_mapping
quat_to_mat = _original.quat_to_mat

FLOAT32_EPS = torch.finfo(torch.float32).eps
ROUNDTRIP_ULPS = 8
RELATIVE_FROBENIUS_BOUND = ROUNDTRIP_ULPS * FLOAT32_EPS
PHYSICAL_RELATIVE_FROBENIUS_BOUND = 2e-5
PINNED_KERNEL_SHA256 = "9443b0e25753623c8433500eb595814fc47a5a9735270aa1b914f59b1d0d10de"
PINNED_WARP_SVD_SHA256 = "9f709c5e706bb1461e1b1f3e0b059bcb2588fcce6a8cd736fba2d63616b4da68"


@lru_cache(maxsize=1)
def _installed_conversion_kernel():
    import warp as wp
    from newton._src.solvers.mujoco import kernels
    kernel_path = Path(kernels.__file__).resolve()
    svd_path = Path(wp.__file__).resolve().parent / "native/svd.h"
    digest = lambda path: hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    kernel_sha, svd_sha = digest(kernel_path), digest(svd_path)
    if kernel_sha != PINNED_KERNEL_SHA256 or svd_sha != PINNED_WARP_SVD_SHA256:
        raise RuntimeError("Unverified Newton/Warp inertia conversion implementation")
    return kernels.update_body_inertia_kernel, {
        "kernel": "newton._src.solvers.mujoco.kernels.update_body_inertia_kernel",
        "kernel_path": str(kernel_path), "kernel_sha256": kernel_sha,
        "warp_svd_path": str(svd_path), "warp_svd_sha256": svd_sha,
        "warp_version": wp.__version__, "storage_dtype": "float32",
        "jacobi_sweeps": 4, "rotations_per_sweep": 3,
        "eigenpair_convention": "descending; determinant-positive; normalized quaternion wxyz",
        "private_temporary_outputs": True, "native_model_writes": False,
    }


def _encode_reference_inertia(expected):
    """Run the pinned official conversion on private arrays, never model state."""
    import warp as wp
    kernel, identity = _installed_conversion_kernel()
    requested = expected.detach().to(dtype=torch.float32).contiguous()
    count = requested.reshape(-1, 3, 3).shape[0]
    requested = requested.reshape(count, 3, 3)
    mapping = torch.arange(count, dtype=torch.int32, device=requested.device).reshape(count, 1)
    device = wp.get_device(str(requested.device))
    stream = wp.stream_from_torch(torch.cuda.current_stream(requested.device)) if requested.is_cuda else None
    scope = wp.ScopedStream(stream, sync_enter=True, sync_exit=True) if stream is not None else nullcontext()
    with scope:
        private_principal = wp.empty((count, 1), dtype=wp.vec3f, device=device)
        private_quaternion = wp.empty((count, 1), dtype=wp.quatf, device=device)
        wp.launch(kernel, dim=(count, 1),
                  inputs=[wp.from_torch(mapping), wp.from_torch(requested, dtype=wp.mat33f)],
                  outputs=[private_principal, private_quaternion], device=device, stream=stream)
        reference = reconstruct_body_frame_inertia(wp.to_torch(private_principal)[:, 0],
                                                    wp.to_torch(private_quaternion)[:, 0])
        # The Warp owners remain alive until all Torch reads on the shared
        # stream complete; no private allocation can be recycled too early.
        if stream is not None:
            wp.synchronize_stream(stream)
    return reference.reshape(expected.shape), dict(identity)


def check_body_frame_inertia(actual, expected, *, reference=None, kernel_identity=None):
    """Strict native/reference agreement plus independent physical accuracy.

    The default reference equals the requested tensor for unit-level direct
    checks. Native audit calls always provide the official independent encoding.
    """
    if (not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor)
            or not actual.is_floating_point() or not expected.is_floating_point()
            or actual.shape != expected.shape or actual.ndim < 2 or actual.shape[-2:] != (3, 3)
            or not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all())):
        raise RuntimeError("invalid actual/expected MJWarp object body-frame inertia")
    # Promote the native expected tensor to the readback dtype, then compute
    # norms in float64. Expected float32 values remain exactly representable.
    target = expected.to(device=actual.device, dtype=actual.dtype).double().reshape(-1, 3, 3)
    measured = actual.double().reshape(-1, 3, 3)
    if not len(measured):
        raise RuntimeError("empty MJWarp object body-frame inertia")
    norm = torch.linalg.matrix_norm(target, ord="fro", dim=(-2, -1))
    bound = norm * PHYSICAL_RELATIVE_FROBENIUS_BOUND
    native_bound = norm * RELATIVE_FROBENIUS_BOUND
    symmetry = torch.linalg.matrix_norm(target - target.transpose(-1, -2), ord="fro", dim=(-2, -1))
    if bool((norm <= 0).any()) or bool((symmetry > native_bound).any()) or bool((torch.linalg.eigvalsh(target) <= 0).any()):
        raise RuntimeError("invalid expected positive-definite MJWarp object body-frame inertia")
    delta = measured - target
    error = torch.linalg.matrix_norm(delta, ord="fro", dim=(-2, -1))
    relative = error / norm
    if reference is None:
        reference = target.reshape(expected.shape)
    if (not isinstance(reference, torch.Tensor) or reference.shape != expected.shape
            or not reference.is_floating_point() or not bool(torch.isfinite(reference).all())):
        raise RuntimeError("invalid independently encoded MJWarp inertia reference")
    encoded = reference.to(device=actual.device, dtype=torch.float64).reshape(-1, 3, 3)
    reference_error = torch.linalg.matrix_norm(encoded - target, ord="fro", dim=(-2, -1))
    native_error = torch.linalg.matrix_norm(measured - encoded, ord="fro", dim=(-2, -1))
    worst = int(torch.maximum(native_error / native_bound, torch.maximum(error, reference_error) / bound).argmax())
    maximum = float(delta.abs().max())
    passed = not bool(((error > bound) | (reference_error > bound) | (native_error > native_bound)).any())
    evidence = {
        "shape": list(actual.shape), "max_abs_error": maximum,
        "comparison": "strict_pinned_native_reference_and_independent_physical_Frobenius_guards",
        "readback_dtype": str(actual.dtype), "expected_storage_dtype": str(expected.dtype),
        "storage_epsilon": FLOAT32_EPS, "roundtrip_ulps": ROUNDTRIP_ULPS,
        "relative_frobenius_bound": PHYSICAL_RELATIVE_FROBENIUS_BOUND,
        "native_reference_relative_frobenius_bound": RELATIVE_FROBENIUS_BOUND,
        "physical_bound_source": "inherited hoi_residual.native_checks._check rtol=2e-5",
        "reference_kernel": kernel_identity,
        "max_frobenius_error": float(error.max()),
        "max_relative_frobenius_error": float(relative.max()),
        "max_native_reference_relative_frobenius_error": float((native_error / norm).max()),
        "max_reference_physical_relative_frobenius_error": float((reference_error / norm).max()),
        "expected_frobenius_norm_range": [float(norm.min()), float(norm.max())],
        "allowed_frobenius_error_range": [float(bound.min()), float(bound.max())],
        "worst_world": {
            "index": worst, "actual": measured[worst].detach().cpu().tolist(),
            "expected": target[worst].detach().cpu().tolist(),
            "independent_native_reference": encoded[worst].detach().cpu().tolist(),
            "actual_frobenius_norm": float(torch.linalg.matrix_norm(measured[worst], ord="fro")),
            "expected_frobenius_norm": float(norm[worst]),
            "error_frobenius_norm": float(error[worst]), "allowed_error_frobenius_norm": float(bound[worst]),
            "reference_physical_error_frobenius_norm": float(reference_error[worst]),
            "native_reference_error_frobenius_norm": float(native_error[worst]),
            "allowed_native_reference_error_frobenius_norm": float(native_bound[worst]),
        },
        "native_reference_match": not bool((native_error > native_bound).any()),
        "physical_guard_passed": not bool(((error > bound) | (reference_error > bound)).any()),
        "all_worlds_within_bound": passed,
    }
    if not passed:
        failure = RuntimeError("object DR did not reach MJWarp body-frame inertia: " +
                               json.dumps(evidence, separators=(",", ":"), allow_nan=False))
        failure.evidence = evidence
        raise failure
    return evidence


def _check(name, actual, expected, *, atol=1e-6, rtol=2e-5):
    if name == "body-frame inertia":
        reference, identity = _encode_reference_inertia(expected)
        return check_body_frame_inertia(actual, expected, reference=reference, kernel_identity=identity)
    return _original._check(name, actual, expected, atol=atol, rtol=rtol)


def reconstruct_body_frame_inertia(principal, quaternion):
    """Read-only float64 rotation/tensor reconstruction from native storage.

    Normalizing a float64 copy removes the radial rounding residue of a valid
    float32 unit quaternion. It does not edit the native quaternion or inertia.
    The audit separately retains its original raw-quaternion norm check.
    """
    quaternion64 = torch.nn.functional.normalize(quaternion.double(), dim=-1)
    rotation = quat_to_mat(quaternion64)
    return rotation @ torch.diag_embed(principal.double()) @ rotation.transpose(-1, -2)


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
    inertia = reconstruct_body_frame_inertia(principal, quaternion)
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
