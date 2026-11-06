"""Official Warp compilation settings for the simulator-only PPO rollout.

Call before importing Isaac Lab, Newton, or MuJoCo Warp. PPO differentiates its
PyTorch policy, never the simulator: no Warp backward kernels are required.
This module does not change forward kernels, physics settings, Torch grad mode,
the installed runtime, or the frozen GMT source.
"""
from __future__ import annotations


SCHEMA = "hoi_residual.warp_forward_only.v1"
NEWTON_MUJOCO_MODULE = "newton._src.solvers.mujoco.kernels"


def configure_warp_forward_only() -> dict:
    """Disable unused Warp reverse code generation through supported APIs.

    Warp copies its global ``enable_backward`` into each module at registration.
    Set the global default before simulator imports and explicitly configure the
    Newton contact-conversion module as well, including if already registered.
    Warp's normal cache key includes module options; no cache files are deleted.

    On the pinned runtime, the contact conversion's forward entry accepts 3016
    bytes of parameters and its unused backward entry 6008 bytes. R525 drivers
    cannot load the latter (>4096 bytes requires R530); omitting it avoids that
    specific compatibility requirement. Other driver/runtime requirements still
    apply and native GPU smoke remains the acceptance test.
    """
    import warp as wp

    wp.config.enable_backward = False
    wp.set_module_options({"enable_backward": False}, module=NEWTON_MUJOCO_MODULE)
    effective = wp.get_module_options(module=NEWTON_MUJOCO_MODULE)
    if wp.config.enable_backward is not False or effective.get("enable_backward") is not False:
        raise RuntimeError("Warp did not apply forward-only simulator compilation")
    return {
        "schema": SCHEMA,
        "warp_version": wp.__version__,
        "global_enable_backward": False,
        "module_overrides": {NEWTON_MUJOCO_MODULE: {"enable_backward": False}},
        "reason": "PPO uses Torch autograd; simulator backward kernels are unused",
    }
