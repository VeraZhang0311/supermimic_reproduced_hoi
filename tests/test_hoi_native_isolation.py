"""Read-only byte proof, including nested physical workspaces and RNG state."""
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from hoi.v1.native_isolation import (BOOKKEEPING_FIELDS, HOI_FIELDS, SCHEMA,
                                     capture_native_fingerprint, fingerprint_tree, assert_native_unchanged)
from hoi.v1.worker_rng import AssetRNG


class NativeArray:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
        self.shape, self.dtype = self.values.shape, self.values.dtype
    def numpy(self):
        return self.values.copy()


@dataclass
class Contact:
    geom: NativeArray
    force: NativeArray


@dataclass
class Constraint:
    force: NativeArray


@dataclass
class NativeData:
    qpos: NativeArray
    contact: Contact
    efc: Constraint


class FakeRuntime:
    def __init__(self):
        self.active = None
        self.syncs = []
    def require_active(self, env):
        if self.active is not env:
            raise RuntimeError("wrong active slot")
    def synchronize(self, device):
        self.syncs.append(str(device))
    def is_native_array(self, value):
        return isinstance(value, NativeArray)
    def snapshot(self, env):
        return {"s": torch.tensor(env.env.sim.backend_manager.get_state_0().qpos.numpy()),
                "f": env.env.f.clone(), "asset_id": env.asset_id.clone()}


def fixture():
    make = lambda: SimpleNamespace(qpos=NativeArray([1., 2.]))
    model, state0, state1, control, mjw_model = [make() for _ in range(5)]
    mjw_data = NativeData(NativeArray([2.]), Contact(NativeArray([0., 1.]), NativeArray([3.])),
                          Constraint(NativeArray([4.])))
    manager = SimpleNamespace(get_model=lambda: model, get_state_0=lambda: state0,
                              get_state_1=lambda: state1, get_control=lambda: control)
    sim = SimpleNamespace(backend_manager=manager, _physics_step_count=7, _render_generation=0,
                          _is_playing=True, _is_stopped=False)
    base = SimpleNamespace(sim=sim, **{key: torch.zeros(2) for key in BOOKKEEPING_FIELDS})
    contacts = SimpleNamespace(solver=SimpleNamespace(mjw_model=mjw_model, mjw_data=mjw_data), calls=4,
                               impulse=torch.zeros(2, 16), forces=NativeArray([0., 1.]))
    env = SimpleNamespace(env=base, contacts=contacts, device="cpu",
                          **{key: torch.zeros(2) for key in HOI_FIELDS})
    env.support_alignment = {"reset_count": 1, "step_count": 4}
    runtime = FakeRuntime()
    runtime.active = env
    return env, AssetRNG(19), runtime


class NativeIsolationTests(unittest.TestCase):
    def test_capture_covers_nested_contact_constraint_bookkeeping_and_rng_without_mutating(self):
        env, rng, runtime = fixture()
        caller = torch.get_rng_state().clone()
        saved = rng._cpu_state.clone()
        first = capture_native_fingerprint(env, rng, _runtime=runtime)
        second = capture_native_fingerprint(env, rng, _runtime=runtime)
        result = assert_native_unchanged(first, second)
        self.assertTrue(result["unchanged"])
        self.assertTrue(all(first["native_buffer_counts"].values()))
        self.assertIn("native.mjw_data.contact.geom", first["buffers"])
        self.assertIn("native.mjw_data.efc.force", first["buffers"])
        self.assertIn("bookkeeping.bin_G", first["buffers"])
        self.assertIn("bookkeeping.bin_w_avg", first["buffers"])
        self.assertIn("rng.cpu_state", first["buffers"])
        self.assertIn("sim.physics_step_count", first["scalars"])
        self.assertIn("recorder.calls", first["scalars"])
        self.assertEqual(runtime.syncs, ["cpu"] * 6)
        self.assertTrue(torch.equal(caller, torch.get_rng_state()))
        self.assertTrue(torch.equal(saved, rng._cpu_state))
        self.assertEqual(env.env.sim._physics_step_count, 7)
        self.assertEqual(env.contacts.calls, 4)

    def test_real_buffer_change_is_detected_even_when_object_identity_is_unchanged(self):
        env, rng, runtime = fixture()
        before = capture_native_fingerprint(env, rng, _runtime=runtime)
        same_object = env.contacts.solver.mjw_data.efc.force
        same_object.values[0] += 1
        self.assertIs(same_object, env.contacts.solver.mjw_data.efc.force)
        after = capture_native_fingerprint(env, rng, _runtime=runtime)
        with self.assertRaisesRegex(AssertionError, "native.mjw_data.efc.force"):
            assert_native_unchanged(before, after, label="asset 0 after asset 1 step")

    def test_nan_payload_and_signed_zero_bytes_are_preserved(self):
        bits = np.array([0x7FC00001, 0x7FC00002, 0x80000000], dtype=np.uint32)
        floats = bits.view(np.float32)
        tensor = torch.from_numpy(floats)
        first = fingerprint_tree({"numpy": floats, "torch": tensor})
        second = fingerprint_tree({"numpy": floats, "torch": tensor})
        self.assertEqual(first, second)
        self.assertEqual(first["buffers"]["numpy"]["sha256"], first["buffers"]["torch"]["sha256"])
        bits[0] = 0x7FC00003  # Still NaN; a different payload must not compare equal.
        third = fingerprint_tree({"numpy": floats, "torch": tensor})
        self.assertNotEqual(first["aggregate_sha256"], third["aggregate_sha256"])
        bits[2] = 0  # Signed zero is also a byte change.
        self.assertNotEqual(third, fingerprint_tree({"numpy": floats, "torch": tensor}))

    def test_bfloat16_noncontiguous_and_scalar_tensors_hash_without_conversion(self):
        tensors = {"bf16": torch.arange(6).to(torch.bfloat16).reshape(2, 3).T,
                   "scalar": torch.tensor(7.), "empty": torch.empty((0, 3))}
        before = {key: value.clone() for key, value in tensors.items()}
        fingerprint = fingerprint_tree(tensors)
        self.assertEqual(fingerprint["buffers"]["bf16"]["nbytes"], 12)
        self.assertEqual(fingerprint["buffers"]["bf16"]["shape"], [3, 2])
        self.assertEqual(fingerprint["buffers"]["scalar"]["shape"], [])
        for key in tensors:
            torch.testing.assert_close(tensors[key], before[key])

    def test_each_native_root_and_nested_workspaces_are_required(self):
        env, rng, runtime = fixture()
        env.env.sim.backend_manager.get_control = lambda: SimpleNamespace(no_buffers=1)
        with self.assertRaisesRegex(RuntimeError, "omitted actual buffers"):
            capture_native_fingerprint(env, rng, _runtime=runtime)
        env, rng, runtime = fixture()
        env.contacts.solver.mjw_data.contact = None
        with self.assertRaisesRegex(RuntimeError, "nested MJW contact"):
            capture_native_fingerprint(env, rng, _runtime=runtime)

    def test_wrong_slot_or_active_rng_fails_before_any_snapshot_or_sync(self):
        env, rng, runtime = fixture()
        runtime.active = object()
        with self.assertRaisesRegex(RuntimeError, "wrong active slot"):
            capture_native_fingerprint(env, rng, _runtime=runtime)
        runtime.active = env
        with rng.activate(), self.assertRaisesRegex(RuntimeError, "outside rng.activate"):
            capture_native_fingerprint(env, rng, _runtime=runtime)
        self.assertEqual(runtime.syncs, [])

    def test_bookkeeping_scalar_and_saved_rng_changes_are_detected(self):
        for field in ("rng", "step_count", "calls", "bin_G"):
            env, rng, runtime = fixture()
            before = capture_native_fingerprint(env, rng, _runtime=runtime)
            if field == "rng":
                rng._cpu_state[0] ^= 1
            elif field == "step_count":
                env.env.sim._physics_step_count += 1
            elif field == "calls":
                env.contacts.calls += 1
            else:
                env.env.bin_G[0] = 1.
            after = capture_native_fingerprint(env, rng, _runtime=runtime)
            with self.subTest(field=field), self.assertRaises(AssertionError):
                assert_native_unchanged(before, after)

    def test_aliases_keep_paths_but_properties_and_runtime_caches_are_not_traversed(self):
        shared = NativeArray([1., 2.])
        class RuntimeDevice:
            def __init__(self): self.cache = {"foreign_kernel": torch.ones(3)}
            @property
            def forbidden(self): raise AssertionError("properties must not be read")
        root = SimpleNamespace(first=shared, second=shared, device=RuntimeDevice())
        first = fingerprint_tree({"model": root}, is_native_array=lambda value: isinstance(value, NativeArray))
        root.device.cache["other_scene"] = torch.ones(5)
        second = fingerprint_tree({"model": root}, is_native_array=lambda value: isinstance(value, NativeArray))
        self.assertEqual(first, second)
        self.assertEqual(set(first["buffers"]), {"model.first", "model.second"})
        self.assertEqual(first["buffers"]["model.first"], first["buffers"]["model.second"])

    def test_schema_and_removed_buffer_paths_cannot_pass(self):
        env, rng, runtime = fixture()
        before = capture_native_fingerprint(env, rng, _runtime=runtime)
        after = deepcopy(before)
        del after["buffers"]["native.mjw_data.contact.geom"]
        with self.assertRaisesRegex(AssertionError, "contact.geom"):
            assert_native_unchanged(before, after)
        with self.assertRaisesRegex(ValueError, "schema"):
            assert_native_unchanged(before, {**before, "schema": SCHEMA + "wrong"})


if __name__ == "__main__":
    unittest.main()
