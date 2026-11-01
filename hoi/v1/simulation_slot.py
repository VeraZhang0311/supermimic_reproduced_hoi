"""Serial ownership adapter for independent kitless Newton scenes in one process.

IsaacLab currently exposes an active SimulationContext singleton although each
Newton manager owns its own solver/model/state. This module is the only place
that leases that private singleton. Every scene operation must run inside its
slot, with the matching official ``use_stage`` scope. No runtime is patched.

The adapter is intentionally limited to the audited runtime, one host thread,
one CUDA device, identical physics settings, and no rendering or builder hooks.
Slots are never individually destroyed: clear_instance closes the global USD
stage cache, so teardown is permitted only after the whole pool stops running.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import threading


AUDITED_RUNTIME_SHA256 = {
    "isaaclab.sim.simulation_context": "df848016f045de9c9381dccb2ab1c78a43056266936847d3abd0c9484897c7f0",
    "isaaclab.sim.simulation_cfg": "5f969534a76865287f5f8cdcd9c67ee161c65f6bbc587e58f39ca809d486e149",
    "isaaclab.sim.utils.stage": "06b91a265a096e60ddd1d822d3da587b1805084acc4e41aa7bf501f4046ed62f",
    "isaaclab.app.settings_manager": "298d2fe011ad33d33f2464f973d0821c0f9b8bf7aaee0fd964024df29db05573",
    "isaaclab.app.sim_launcher": "575c2ba721621e704e5cf773f14d032738ef7692426e242b42be8de4a8734779",
    "isaaclab_newton.physics.newton_manager": "6d670e1e4af1693bc708ed858396597126a51d22708e9d74d02f3cd990e71f1e",
    "isaaclab_newton.physics.mjwarp_manager": "b77f7755e7a0d976e3f325237432e04d98cd5b3fdef19fc6fe02bee04272af2f",
    "isaaclab_newton.cloner.replicate": "d2b0ed4d39b97805a9bdca3a0e8b3434994b8ded753107d55eea7507a4681e28",
    "isaaclab.assets.asset_base": "1a8c9232eea2dab1c5d9fb88b8ccd9879f80054e151aebff9339d9597099aeed",
    "isaaclab.scene.interactive_scene": "faa870290f94328a1fafecbbdd65b15b8a7cbaa186e3e46a1e6b5c14cc51a35a",
    "isaaclab.physics.physics_manager": "3a26c1a439e459d5218043d4ff817fc9d090de13b006d575438c3122e0630eca",
}


def audit_runtime_sources(paths, expected=AUDITED_RUNTIME_SHA256):
    """Fail closed on any runtime drift instead of guessing private semantics."""
    if set(paths) != set(expected):
        raise RuntimeError("simulation slot runtime source inventory differs from the audited contract")
    actual = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in paths.items()}
    changed = [name for name in expected if actual[name] != expected[name]]
    if changed:
        raise RuntimeError("simulation slot runtime source changed; re-audit required: " + ", ".join(changed))
    return {"verified": True, "source_sha256": actual}


def validate_empty_reset_callbacks(context_type):
    callbacks = getattr(context_type, "_reset_callbacks", None)
    if not isinstance(callbacks, dict) or callbacks:
        raise RuntimeError("simulation slots require an empty global SimulationContext reset callback registry")


class _NativeRuntime:
    def __init__(self):
        modules = {name: importlib.import_module(name) for name in AUDITED_RUNTIME_SHA256}
        self.identity = audit_runtime_sources({name: module.__file__ for name, module in modules.items()})
        self.context_type = modules["isaaclab.sim.simulation_context"].SimulationContext
        self.use_stage = modules["isaaclab.sim.utils.stage"].use_stage
        self.settings = modules["isaaclab.app.settings_manager"].get_settings_manager()
        self.hook_module = modules["isaaclab_newton.cloner.replicate"]
        self.hooks = self.hook_module._PER_WORLD_BUILDER_HOOKS
        from isaaclab.utils.version import has_kit
        from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg
        from pxr import Usd
        self.has_kit, self.physics_type, self.solver_type = has_kit, NewtonCfg, MJWarpSolverCfg
        # This temporary scope stage is not inserted in the global StageCache.
        # SimulationContext creates its own fresh stage from slot.cfg.
        self.new_stage = Usd.Stage.CreateInMemory
        if "_instance" not in vars(self.context_type) or not callable(self.context_type.clear_instance):
            raise RuntimeError("audited SimulationContext ownership API is unavailable")

    def validate_globals(self):
        if self.has_kit():
            raise RuntimeError("simulation slots require kitless Newton, including when Kit is headless")
        if self.hook_module._PER_WORLD_BUILDER_HOOKS is not self.hooks or self.hooks:
            raise RuntimeError("simulation slots do not support global per-world builder hooks")
        validate_empty_reset_callbacks(self.context_type)
        for key in ("/isaaclab/has_gui", "/isaaclab/render/offscreen", "/isaaclab/render/rtx_sensors",
                    "/isaaclab/xr/enabled", "/physics/fabricUpdateTransformations"):
            if self.settings.get(key):
                raise RuntimeError("simulation slots require no rendering: " + key)

    def validate_cfg(self, cfg):
        if not isinstance(cfg.physics, self.physics_type) or not isinstance(cfg.physics.solver_cfg, self.solver_type):
            raise ValueError("simulation slots support only the audited Newton MuJoCo-Warp backend")
        if not str(cfg.device).startswith("cuda:"):
            raise ValueError("simulation slots require one explicit CUDA device")
        if cfg.physics.use_cuda_graph:
            raise ValueError("simulation slots require the unchanged HOI outer CUDA graph setting (False)")
        visualizers = cfg.visualizer_cfgs
        if not isinstance(visualizers, (list, tuple)):
            visualizers = [visualizers]
        visualizers = list(visualizers) + [getattr(cfg, "default_visualizer_cfg", None)]
        if any(getattr(viz, "backend", None) for viz in visualizers if viz is not None):
            raise ValueError("simulation slots do not support visualizers or renderers")

    def validate_instance(self, sim):
        self.validate_cfg(sim.cfg)
        if sim.has_gui or sim.has_offscreen_render or sim.is_rendering or sim.visual_shapes_required:
            raise RuntimeError("simulation slots cannot activate a rendered or camera-bound scene")
        if type(sim.backend_manager).__module__ != "isaaclab_newton.physics.mjwarp_manager":
            raise RuntimeError("simulation slot resolved to an unaudited physics manager")

    @staticmethod
    def physics_identity(cfg):
        return json.dumps({"device": str(cfg.device), "dt": cfg.dt,
                           "physics": cfg.physics.to_dict()}, sort_keys=True, default=str)

    @staticmethod
    def owned_objects(sim):
        manager = sim.backend_manager
        solver = manager._solver
        return {"simulation_context": sim, "physics_manager": manager, "backend": manager.backend,
                "newton_model": manager.get_model(), "newton_state_0": manager.get_state_0(),
                "newton_state_1": manager.get_state_1(), "newton_control": manager.get_control(),
                "solver": solver, "native_solver": getattr(solver, "_wrapped", solver), "mj_model": solver.mj_model,
                "mjw_model": solver.mjw_model, "mjw_data": solver.mjw_data}


class SimulationSlot:
    """One independent native scene; use ``activate`` for every scene access."""
    def __init__(self, owner, cfg):
        self.owner, self.cfg = owner, cfg
        self.sim = self.stage = None
        self.closed = False
        self._owned_objects = None
        self.identity_report = None

    def activate(self):
        return self.owner._activate(self)


class SimulationSlots:
    """Own all scene lifetimes and prohibit concurrent or nested activation.

    ``with slots.create() as slot: env = HOITrackingSim(slot.cfg, ...)`` builds
    one scene. ``with slot.activate(): env.step(action)`` leases it afterwards.
    The injected ``_runtime`` exists only for CPU lifecycle tests.
    """
    _live_owner = None
    _active_owner = None

    def __init__(self, sim_cfg, *, _runtime=None):
        if SimulationSlots._live_owner is not None:
            raise RuntimeError("only one simulation slot pool may own a process")
        self._runtime = _NativeRuntime() if _runtime is None else _runtime
        self._thread = threading.get_ident()
        self._cfg = deepcopy(sim_cfg)
        self._slots = []
        self._ownership_index = {}
        self._closed = self._shutdown = self._poisoned = False
        self._runtime.validate_globals()
        self._runtime.validate_cfg(self._cfg)
        if self._runtime.context_type._instance is not None:
            raise RuntimeError("cannot take ownership of an external SimulationContext")
        self._physics_identity = self._runtime.physics_identity(self._cfg)
        self.runtime_report = {**deepcopy(self._runtime.identity),
                               "adapter": "hoi_serial_simulation_slots_v1",
                               "ownership": "one active context and matching stage per host thread",
                               "stage_creation": "fresh in-memory stage per slot",
                               "teardown": "whole-pool shutdown only; never clear during slot switches",
                               "rendering": False, "global_reset_callbacks": "must remain empty",
                               "disjoint_slots_verified": 0}
        SimulationSlots._live_owner = self

    def _check_idle(self, *, shutdown=False):
        if threading.get_ident() != self._thread:
            raise RuntimeError("simulation slots are confined to their owning thread")
        if SimulationSlots._active_owner is not None:
            raise RuntimeError("simulation slot activation is not reentrant")
        if self._closed or (self._shutdown and not shutdown) or (self._poisoned and not shutdown):
            raise RuntimeError("simulation slot pool is closed, shutting down, or has failed construction")
        if self._runtime.context_type._instance is not None:
            raise RuntimeError("external SimulationContext would be overwritten")
        if not shutdown:
            self._runtime.validate_globals()

    def _validate_slot(self, slot):
        if self._runtime.physics_identity(slot.cfg) != self._physics_identity:
            raise RuntimeError("all simulation slots must preserve the same device and physics configuration")
        self._runtime.validate_instance(slot.sim)
        self._runtime.validate_globals()
        objects = self._runtime.owned_objects(slot.sim)
        if not objects or any(value is None for value in objects.values()):
            raise RuntimeError("simulation slot has incomplete mutable native ownership")
        identities = {id(value) for value in objects.values()}
        if slot._owned_objects is not None:
            # State buffer roles may exchange after a step; their owning set
            # must stay identical. A rebuilt or foreign model is not accepted.
            if identities != {id(value) for value in slot._owned_objects.values()}:
                raise RuntimeError("simulation slot mutable model/state/solver ownership changed")
        elif any(identity in self._ownership_index for identity in identities):
            raise RuntimeError("simulation slots share mutable model/state/solver ownership")
        else:
            slot._owned_objects = objects
            self._ownership_index.update({identity: slot for identity in identities})
            slot.identity_report = {"process_local_object_ids": {key: id(value) for key, value in objects.items()},
                                    "cross_slot_disjoint": True}
            self.runtime_report["disjoint_slots_verified"] += 1

    @contextmanager
    def create(self):
        self._check_idle()
        cfg = deepcopy(self._cfg)
        cfg.create_stage_in_memory = True
        slot = SimulationSlot(self, cfg)
        self._slots.append(slot)
        context_type = self._runtime.context_type
        SimulationSlots._active_owner = self
        try:
            # A public stage scope restores the previous thread-local context
            # even though the constructor installs its own newly created stage.
            with self._runtime.use_stage(self._runtime.new_stage()):
                try:
                    yield slot
                finally:
                    sim = context_type._instance
                    if sim is not None:
                        if sim.cfg is not cfg:
                            raise RuntimeError("construction published a foreign SimulationContext")
                        slot.sim, slot.stage = sim, sim.stage
                if slot.sim is None:
                    raise RuntimeError("slot construction did not publish a SimulationContext")
                if any(other is not slot and other.stage is slot.stage for other in self._slots):
                    raise RuntimeError("simulation slots must have independent USD stages")
                self._validate_slot(slot)
        except BaseException:
            self._poisoned = True
            raise
        finally:
            if context_type._instance is slot.sim:
                context_type._instance = None
            SimulationSlots._active_owner = None

    @contextmanager
    def _activate(self, slot, *, shutdown=False):
        self._check_idle(shutdown=shutdown)
        if slot.owner is not self or slot not in self._slots or slot.closed or slot.sim is None:
            raise RuntimeError("simulation slot is not a live initialized member of this pool")
        if not shutdown:
            self._validate_slot(slot)
        context_type = self._runtime.context_type
        SimulationSlots._active_owner = self
        context_type._instance = slot.sim
        try:
            with self._runtime.use_stage(slot.stage):
                yield slot
        finally:
            current = context_type._instance
            if current is slot.sim:
                context_type._instance = None
            SimulationSlots._active_owner = None
            if current is not slot.sim and not (shutdown and current is None):
                raise RuntimeError("active SimulationContext was unexpectedly replaced; foreign context preserved")

    def close(self):
        """Stop the entire pool, then clear each instance; never resume a slot.

        IsaacLab clear_instance closes the global USD StageCache. Strong stage
        references keep later slots available for teardown only. No close or
        cache clear is performed during ordinary activation switches.
        """
        if self._closed:
            return
        self._check_idle(shutdown=True)
        self._shutdown = True
        errors = []
        try:
            for slot in reversed(self._slots):
                if slot.sim is not None:
                    try:
                        with self._activate(slot, shutdown=True):
                            self._runtime.context_type.clear_instance()
                    except Exception as error:
                        errors.append(error)
                slot.closed = True
                slot._owned_objects = None
        finally:
            self._ownership_index.clear()
            self._closed = True
            if SimulationSlots._live_owner is self:
                SimulationSlots._live_owner = None
        if errors:
            raise RuntimeError("simulation slot shutdown errors: " + "; ".join(str(error) for error in errors)) from errors[0]
