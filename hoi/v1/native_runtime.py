"""Narrow Warp loader compatibility for forward-only native data collection.

CUDA 12.0 drivers cannot load the unused 6008-byte backward entry in the
Newton MuJoCo module. Only this module's backward code generation is disabled
on those drivers. Forward kernels, compiler math options and physics remain
unchanged. This is not a differentiable-simulator entry point.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re

SCHEMA = "hoi_native_warp_forward_compat_v1"
MODULE = "newton._src.solvers.mujoco.kernels"
TARGET_KERNEL = "convert_newton_contacts_to_mjwarp_kernel"
OPTION_REFERENCE = "https://nvidia.github.io/warp/v1.17/user_guide/configuration.html#module-settings"
DRIVER_REFERENCE = "https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters/"


def require_native_physics_driver(wp, *, cooking_only=False):
    """Reject an unsupported full-step ABI before constructing a scene.

    In this installed stack, the actual CCD forward kernel has 4296 bytes of
    parameters even with backward compilation disabled. CUDA 12.0 drivers
    cannot load it. Geometry-only cooking never invokes that kernel and keeps
    the separately verified Newton-module backward compatibility path.
    """
    if cooking_only:
        return
    version = wp.get_cuda_driver_version()
    if version is None or tuple(version) < (12, 1):
        current = "unavailable" if version is None else ".".join(map(str, version))
        raise RuntimeError(
            f"Full native physics requires CUDA driver >=12.1 (found {current}): "
            "the MuJoCo-Warp CCD forward kernel has 4296 parameter bytes, above "
            "the old driver's 4096-byte limit. Disabling backward generation "
            "cannot fix this forward ABI. Use a compatible node; only explicit "
            "cooking_only geometry export is allowed on the old driver.")


def configure_native_forward_compat(wp=None, *, physics_autograd=False):
    """Use an official per-module option; never mutate a runtime installation.

    Call before constructing the first native scene. The learned WM uses
    PyTorch autograd independently; no Warp Tape is used by this collector.
    CUDA >=12.1 remains an unmodified control. A caller needing native physics
    gradients must not enter this forward-only compatibility path.
    """
    if physics_autograd:
        raise ValueError("native forward compatibility does not support physics autograd")
    if wp is None:
        import warp as wp
    version = wp.get_cuda_driver_version()
    report = {"schema": SCHEMA, "cuda_driver_version": list(version) if version else None,
              "warp_version": str(wp.__version__), "module": MODULE,
              "physics_autograd": False, "applied": False, "changed_options": {},
              "option_reference": OPTION_REFERENCE, "driver_reference": DRIVER_REFERENCE}
    if version is None or tuple(version) >= (12, 1):
        report["reason"] = "driver does not need the large-parameter backward-kernel workaround"
        return report
    module = importlib.import_module(MODULE)
    before = dict(wp.get_module_options(module))
    wp.set_module_options({"enable_backward": False}, module=module)
    after = dict(wp.get_module_options(module))
    changed = {key: {"before": before.get(key), "after": after.get(key)}
               for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
    if after.get("enable_backward") is not False or set(changed) - {"enable_backward"}:
        raise RuntimeError("native compatibility unexpectedly changed forward compilation options")
    report.update(applied=True, reason="omit unused backward kernel requiring CUDA >=12.1",
                  changed_options=changed, enable_backward_before=before.get("enable_backward"),
                  enable_backward_after=after["enable_backward"],
                  module_source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
    return report


def forward_entry_hashes(source):
    """Hash generated forward CUDA bodies for a zero-physics A/B load probe."""
    entries = {}
    pattern = r'extern "C" __global__ void (\w+_cuda_kernel_forward)\s*\('
    for match in re.finditer(pattern, source):
        start = source.find("{", match.end())
        if start < 0:
            raise ValueError("generated forward entry has no body")
        end, depth = start, 1
        while depth and end + 1 < len(source):
            end += 1
            depth += (source[end] == "{") - (source[end] == "}")
        if depth:
            raise ValueError("generated forward entry has an incomplete body")
        body = source[match.start():end + 1]
        # Module-option changes can alter generated symbol hashes, not math.
        normalize = lambda text: re.sub(r"_[0-9a-f]{8}(?=_|\b)", "_HASH", text)
        entries[normalize(match.group(1))] = hashlib.sha256(normalize(body).encode()).hexdigest()
    if not entries:
        raise ValueError("no generated forward CUDA entries found")
    return entries


def main(argv=None):
    """Load exactly the failing module and run its zero-contact forward branch.

    This command creates no IsaacLab scene and executes zero physics steps.
    Run each mode in a fresh process/cache; the caller schedules any GPU job.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("default", "native_forward"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--baseline-cuda-source", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.cache.exists():
        raise FileExistsError("probe output and cache must both be new")
    args.cache.mkdir(parents=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import warp as wp
    wp.config.kernel_cache_dir = str(args.cache)
    wp.init()
    report = {"schema": "hoi_native_warp_load_probe_v1", "mode": args.mode,
              "physics_steps": 0, "formal_training_started": False,
              "cuda_driver_version": wp.get_cuda_driver_version(), "warp_version": wp.__version__,
              "module": MODULE, "target_kernel": TARGET_KERNEL, "passed": False,
              "baseline_cuda_source_sha256": hashlib.sha256(args.baseline_cuda_source.read_bytes()).hexdigest()}
    try:
        if args.mode == "native_forward":
            report["compilation"] = configure_native_forward_compat(wp)
        kernels = importlib.import_module(MODULE)
        # SolverMuJoCo registers this deferred kernel before its first module
        # load. Register the same entry without constructing a physics scene.
        kernels.create_convert_mjw_contacts_to_newton_kernel()
        report["deferred_contact_conversion_registered"] = True
        wp.load_module(module=kernels, device="cuda:0")
        report["module_loaded"] = True
        # Resolve the actual installed signature, without duplicating 57 input
        # descriptors. All buffers have one element; count=0 returns before
        # accessing body/shape geometry. Two written sentinels prove execution.
        from typing import get_type_hints
        kernel = getattr(kernels, TARGET_KERNEL)
        types = get_type_hints(kernel.func)
        values = {}
        for name, dtype in types.items():
            if name == "return":
                continue
            # Warp 1.17 subscript annotations are lightweight descriptors,
            # not array instances; both expose dtype and ndim metadata.
            if hasattr(dtype, "dtype") and isinstance(getattr(dtype, "ndim", None), int):
                values[name] = wp.zeros((1,) * dtype.ndim, dtype=dtype.dtype, device="cuda:0")
            elif dtype in (int, bool):
                values[name] = 1 if dtype is int else False
            else:
                raise TypeError(f"unsupported installed probe input {name}: {dtype}")
        values["contact_generation"].fill_(1)
        values["ncollision_out"].fill_(7)
        values["tid_to_cid"].fill_(7)
        wp.launch(kernel, dim=1, inputs=list(values.values()), device="cuda:0")
        wp.synchronize_device("cuda:0")
        report["empty_contact_forward"] = {name: value.numpy().tolist() for name, value in values.items()
                                           if name in ("nacon_out", "ncollision_out", "tid_to_cid")}
        if report["empty_contact_forward"] != {"nacon_out": [0], "ncollision_out": [0], "tid_to_cid": [-1]}:
            raise RuntimeError("actual contact conversion forward kernel returned unexpected sentinels")
        generated = list(args.cache.rglob("*.cu"))
        generated = [path for path in generated if TARGET_KERNEL in path.read_text()]
        if len(generated) != 1:
            raise RuntimeError("cannot identify the actual compiled target CUDA source")
        before = forward_entry_hashes(args.baseline_cuda_source.read_text())
        after = forward_entry_hashes(generated[0].read_text())
        report["forward_entries"] = len(after)
        report["baseline_forward_entries"] = len(before)
        report["missing_forward_entries"] = sorted(before.keys() - after.keys())
        report["extra_forward_entries"] = sorted(after.keys() - before.keys())
        report["changed_forward_entries"] = sorted(key for key in before.keys() & after.keys() if before[key] != after[key])
        report["forward_sources_identical"] = before == after
        report["generated_cuda_source_sha256"] = hashlib.sha256(generated[0].read_bytes()).hexdigest()
        report["generated_backward_entry_count"] = len(re.findall(r'extern "C" __global__ void \w+_cuda_kernel_backward', generated[0].read_text()))
        if before != after:
            raise RuntimeError("generated forward CUDA code differs from the unmodified baseline")
        report["passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
