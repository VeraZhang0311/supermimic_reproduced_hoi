"""CPU lifecycle checks for the narrow private-singleton adapter."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest

from hoi.v1.simulation_slot import SimulationSlots, audit_runtime_sources, validate_empty_reset_callbacks


class FakeRuntime:
    def __init__(self):
        self.stage = None
        self.cache = []
        self.cleared = []
        self.bad_global = False
        self.identity = {"verified": False, "injected_cpu_fixture": True}
        runtime = self
        class Context:
            _instance = None
            _reset_callbacks = {}
            def __init__(self, cfg):
                if Context._instance is not None:
                    raise RuntimeError("occupied")
                self.cfg = cfg
                assert cfg.create_stage_in_memory
                self.stage = object()
                runtime.stage = self.stage
                runtime.cache.append(self.stage)
                self.backend_manager = SimpleNamespace(state=0, closed=False, model=object(),
                    state_0=object(), state_1=object(), solver=object(), control=object())
                self.fail_close = False
                Context._instance = self
            def step(self, amount):
                assert Context._instance is self and runtime.stage is self.stage
                assert not self.backend_manager.closed
                self.backend_manager.state += amount
            @classmethod
            def clear_instance(cls):
                current = cls._instance
                assert current is not None and runtime.stage is current.stage
                current.backend_manager.closed = True
                runtime.cleared.append(current)
                runtime.cache.clear()  # Mirrors the audited global StageCache clear.
                runtime.stage = None
                cls._instance = None
                if current.fail_close:
                    raise RuntimeError("fixture teardown failure")
        self.context_type = Context

    def new_stage(self):
        return object()

    @contextmanager
    def use_stage(self, stage):
        previous = self.stage
        self.stage = stage
        try:
            yield
        finally:
            self.stage = previous

    def validate_globals(self):
        validate_empty_reset_callbacks(self.context_type)
        if self.bad_global:
            raise RuntimeError("global rendering changed")

    def validate_cfg(self, cfg):
        if cfg.device != "cuda:0" or cfg.physics != {"solver": "Newton", "dt": .02}:
            raise ValueError("unsupported config")

    def validate_instance(self, sim):
        self.validate_cfg(sim.cfg)

    def physics_identity(self, cfg):
        return (cfg.device, deepcopy(cfg.physics))

    def owned_objects(self, sim):
        manager = sim.backend_manager
        return {"sim": sim, "manager": manager, "model": manager.model,
                "state_0": manager.state_0, "state_1": manager.state_1,
                "solver": manager.solver, "control": manager.control}


class SimulationSlotTest(unittest.TestCase):
    def setUp(self):
        self.runtime = FakeRuntime()
        self.cfg = SimpleNamespace(device="cuda:0", physics={"solver": "Newton", "dt": .02},
                                   create_stage_in_memory=False)
        self.pool = SimulationSlots(self.cfg, _runtime=self.runtime)

    def tearDown(self):
        # Preserve intentionally injected foreign objects during assertions,
        # then explicitly relinquish them so each test can clean its own pool.
        self.runtime.context_type._instance = None
        self.pool.close()

    def build(self):
        with self.pool.create() as slot:
            self.assertTrue(slot.cfg.create_stage_in_memory)
            self.runtime.context_type(slot.cfg)
        return slot

    def test_two_slots_restore_global_stage_and_preserve_independent_state(self):
        outside = object()
        self.runtime.stage = outside
        first, second = self.build(), self.build()
        self.assertIsNot(first.sim, second.sim)
        self.assertIsNot(first.stage, second.stage)
        self.assertIsNot(first.sim.backend_manager, second.sim.backend_manager)
        self.assertEqual(len(self.runtime.cache), 2)
        self.assertIs(self.runtime.stage, outside)
        for slot, increment in ((first, 2), (second, 7), (first, 3)):
            with slot.activate():
                slot.sim.step(increment)
            self.assertIsNone(self.runtime.context_type._instance)
            self.assertIs(self.runtime.stage, outside)
        self.assertEqual(first.sim.backend_manager.state, 5)
        self.assertEqual(second.sim.backend_manager.state, 7)
        self.assertEqual(self.runtime.cleared, [])
        self.assertFalse(self.cfg.create_stage_in_memory)
        self.assertIsNot(first.cfg, second.cfg)
        self.assertIsNot(first.cfg.physics, second.cfg.physics)

    def test_operation_exception_restores_owner_and_other_slot_can_run(self):
        first, second = self.build(), self.build()
        with self.assertRaisesRegex(ValueError, "step failed"):
            with first.activate():
                first.sim.step(1)
                raise ValueError("step failed")
        self.assertIsNone(self.runtime.context_type._instance)
        with second.activate():
            second.sim.step(4)
        self.assertEqual(second.sim.backend_manager.state, 4)
        self.assertEqual(first.sim.backend_manager.state, 1)

    def test_reentrant_activation_create_and_close_are_rejected_without_destroying(self):
        first, second = self.build(), self.build()
        with first.activate():
            for enter in (first.activate, second.activate, self.pool.create):
                with self.assertRaisesRegex(RuntimeError, "not reentrant"):
                    with enter():
                        pass
            with self.assertRaisesRegex(RuntimeError, "not reentrant"):
                self.pool.close()
            self.assertIs(self.runtime.context_type._instance, first.sim)
        self.assertEqual(self.runtime.cleared, [])

    def test_other_thread_and_second_pool_cannot_lease_context(self):
        slot = self.build()
        errors = []
        def wrong_thread():
            try:
                with slot.activate():
                    pass
            except RuntimeError as error:
                errors.append(str(error))
        thread = threading.Thread(target=wrong_thread)
        thread.start(); thread.join()
        self.assertEqual(len(errors), 1)
        self.assertIn("owning thread", errors[0])
        with self.assertRaisesRegex(RuntimeError, "one simulation slot pool"):
            SimulationSlots(self.cfg, _runtime=FakeRuntime())
        self.assertIsNone(self.runtime.context_type._instance)

    def test_external_context_is_never_overwritten_or_destroyed(self):
        slot = self.build()
        foreign = object()
        self.runtime.context_type._instance = foreign
        with self.assertRaisesRegex(RuntimeError, "external SimulationContext"):
            with slot.activate():
                pass
        with self.assertRaisesRegex(RuntimeError, "external SimulationContext"):
            self.pool.close()
        self.assertIs(self.runtime.context_type._instance, foreign)
        self.assertEqual(self.runtime.cleared, [])

    def test_mid_operation_foreign_replacement_is_preserved_and_reported(self):
        slot = self.build()
        foreign = object()
        with self.assertRaisesRegex(RuntimeError, "unexpectedly replaced"):
            with slot.activate():
                self.runtime.context_type._instance = foreign
        self.assertIs(self.runtime.context_type._instance, foreign)
        self.assertEqual(self.runtime.cleared, [])

    def test_empty_construction_fails_closed_and_leaves_no_active_context(self):
        with self.assertRaisesRegex(RuntimeError, "did not publish"):
            with self.pool.create():
                pass
        self.assertIsNone(self.runtime.context_type._instance)
        self.assertEqual(self.runtime.cleared, [])
        self.pool.close()

    def test_shutdown_only_clears_every_owned_slot_once_and_disables_all_slots(self):
        first, second = self.build(), self.build()
        self.pool.close()
        self.assertEqual(self.runtime.cleared, [second.sim, first.sim])
        self.assertEqual(self.runtime.cache, [])
        self.assertTrue(first.sim.backend_manager.closed and second.sim.backend_manager.closed)
        self.assertIsNone(self.runtime.context_type._instance)
        for slot in (first, second):
            with self.assertRaisesRegex(RuntimeError, "closed"):
                with slot.activate():
                    pass
        self.pool.close()
        self.assertEqual(len(self.runtime.cleared), 2)

    def test_failed_construction_is_retained_for_whole_pool_shutdown(self):
        first = self.build()
        with self.assertRaisesRegex(ValueError, "construction failed"):
            with self.pool.create() as failed:
                self.runtime.context_type(failed.cfg)
                raise ValueError("construction failed")
        self.assertIsNone(self.runtime.context_type._instance)
        self.assertEqual(self.runtime.cleared, [])
        with self.assertRaisesRegex(RuntimeError, "failed construction"):
            with first.activate():
                pass
        self.pool.close()
        self.assertEqual(self.runtime.cleared, [failed.sim, first.sim])

    def test_shutdown_attempts_other_slots_after_one_cleanup_error(self):
        first, second = self.build(), self.build()
        second.sim.fail_close = True
        with self.assertRaisesRegex(RuntimeError, "shutdown errors"):
            self.pool.close()
        self.assertEqual(self.runtime.cleared, [second.sim, first.sim])
        self.assertIsNone(self.runtime.context_type._instance)

    def test_mutated_device_physics_or_global_rendering_is_rejected(self):
        slot = self.build()
        slot.cfg.physics["dt"] = .04
        with self.assertRaisesRegex(RuntimeError, "same device and physics"):
            with slot.activate():
                pass
        slot.cfg.physics["dt"] = .02
        self.runtime.bad_global = True
        with self.assertRaisesRegex(RuntimeError, "rendering"):
            with slot.activate():
                pass
        self.assertIsNone(self.runtime.context_type._instance)

    def test_global_reset_callbacks_are_rejected_before_activation_or_build(self):
        slot = self.build()
        self.runtime.context_type._reset_callbacks["foreign"] = lambda sim: None
        with self.assertRaisesRegex(RuntimeError, "reset callback registry"):
            with slot.activate():
                pass
        with self.assertRaisesRegex(RuntimeError, "reset callback registry"):
            with self.pool.create():
                pass
        self.assertIsNone(self.runtime.context_type._instance)
        self.assertEqual(self.runtime.cleared, [])

    def test_shared_model_or_state_is_rejected_during_second_slot_build(self):
        first = self.build()
        with self.assertRaisesRegex(RuntimeError, "share mutable"):
            with self.pool.create() as second:
                sim = self.runtime.context_type(second.cfg)
                sim.backend_manager.model = first.sim.backend_manager.model
        self.assertIsNone(self.runtime.context_type._instance)
        self.assertEqual(self.pool.runtime_report["disjoint_slots_verified"], 1)

    def test_state_buffers_can_swap_but_cannot_be_replaced_or_imported(self):
        first, second = self.build(), self.build()
        manager = first.sim.backend_manager
        manager.state_0, manager.state_1 = manager.state_1, manager.state_0
        with first.activate():
            first.sim.step(1)
        manager.state_0 = second.sim.backend_manager.state_0
        with self.assertRaisesRegex(RuntimeError, "ownership changed"):
            with first.activate():
                pass
        self.assertEqual(self.pool.runtime_report["disjoint_slots_verified"], 2)
        first_ids = set(first.identity_report["process_local_object_ids"].values())
        second_ids = set(second.identity_report["process_local_object_ids"].values())
        self.assertFalse(first_ids & second_ids)
        self.assertTrue(first.identity_report["cross_slot_disjoint"])

    def test_runtime_hash_drift_and_missing_inventory_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.py"
            path.write_bytes(b"audited runtime")
            expected = {"runtime": hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertTrue(audit_runtime_sources({"runtime": path}, expected)["verified"])
            path.write_bytes(b"changed runtime")
            with self.assertRaisesRegex(RuntimeError, "re-audit"):
                audit_runtime_sources({"runtime": path}, expected)
            with self.assertRaisesRegex(RuntimeError, "inventory"):
                audit_runtime_sources({}, expected)


if __name__ == "__main__":
    unittest.main()
