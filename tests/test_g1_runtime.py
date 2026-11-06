"""Real MuJoCo-Warp CPU array tests; no Isaac Lab/GPU or training.

Warp tests skip when its optional wheels are absent. The no-runtime CLI
test still checks that dependency-only validation never reports success.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from supermimic.g1_assets import asset_cwd, load_model, verify_assets
from supermimic.g1_physics import ASSET_DIR, G1Motor
from supermimic.g1_runtime import G1PhysicsRuntime, _copy_kernels, _entity_mapping, _audit_inertials, audit_model
from supermimic.robot_metadata import G1_JOINT_NAMES

HAS_WARP = importlib.util.find_spec("warp") is not None and importlib.util.find_spec("mujoco_warp") is not None


class NativeDependencyTests(unittest.TestCase):
    def test_graph_fallback_is_capability_based_and_changes_only_scheduler(self):
        for is_cuda, supported, original in ((True, False, True), (True, True, True),
                                              (True, True, False), (False, False, True)):
            with self.subTest(is_cuda=is_cuda, supported=supported, original=original):
                calls = []
                def capability():
                    calls.append("queried")
                    return supported
                runtime = G1PhysicsRuntime.__new__(G1PhysicsRuntime)
                runtime.device = "cuda:0" if is_cuda else "cpu"
                runtime.wp = SimpleNamespace(get_device=lambda device: SimpleNamespace(is_cuda=is_cuda),
                                             is_conditional_graph_supported=capability)
                opt = SimpleNamespace(graph_conditional=original, iterations=100, ls_iterations=50,
                                      timestep=.005, cone=1, impratio=10.)
                runtime.solver = SimpleNamespace(mjw_model=SimpleNamespace(opt=opt, nworld=4096))
                before = vars(opt).copy()
                report = runtime._initialize_graph_scheduling()
                expected = False if is_cuda and not supported else original
                self.assertEqual(opt.graph_conditional, expected)
                self.assertEqual(report["selected_graph_conditional"], expected)
                self.assertEqual(report["conditional_graph_supported"], supported if is_cuda else None)
                self.assertEqual(len(calls), int(is_cuda))
                self.assertEqual(runtime.solver.mjw_model.nworld, 4096)
                self.assertEqual({key: value for key, value in vars(opt).items() if key != "graph_conditional"},
                                 {key: value for key, value in before.items() if key != "graph_conditional"})

    def test_mesh_recentering_keeps_installed_hull_but_shape_changes_fail(self):
        if not (ASSET_DIR / "manifest.json").is_file():
            self.skipTest("Prepare the verified handoff asset")
        model = load_model(ASSET_DIR / "g1_hw3.xml")
        geom = model.geom("hw3_left_wrist_yaw_link_1_left_rubber_hand_palm_hull").id
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        shift = np.array([.003, -.002, .001])
        model.mesh_vert[start:start+count] -= shift
        for i in np.flatnonzero((model.geom_type == int(mujoco.mjtGeom.mjGEOM_MESH)) & (model.geom_dataid == mesh)):
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[i])
            model.geom_pos[i] += rotation.reshape(3, 3) @ shift
        audit_model(model)
        # Moving the actual hull while keeping its frame fixed must fail the
        # unchanged 30 micrometre support-function check.
        model.mesh_vert[start:start+count, 0] += .001
        with self.assertRaisesRegex(AssertionError, "installed mesh hull support"):
            audit_model(model)

    def test_named_marker_regularization_is_bounded_without_wrist_exception(self):
        if not (ASSET_DIR / "manifest.json").is_file():
            self.skipTest("Prepare the verified handoff asset")
        expected = load_model(ASSET_DIR / "g1_hw3.xml")
        model = load_model(ASSET_DIR / "g1_hw3.xml")
        bodies = {model.body(i).name: i for i in range(1, model.nbody) if model.body_mass[i] > 0}
        for name in ("LL_FOOT", "LR_FOOT"):
            model.body_inertia[bodies[name]] = 1.0001e-6
        report = _audit_inertials(model, expected, bodies)
        self.assertIsNotNone(report["bodies"]["LL_FOOT"]["regularization"])
        self.assertGreater(report["bodies"]["LL_FOOT"]["tensor_max_abs_error_kg_m2"], 1e-6)
        model.body_inertia[bodies["LL_FOOT"], 0] += 1e-7
        with self.assertRaisesRegex(AssertionError, "named fixed-marker regularization"):
            _audit_inertials(model, expected, bodies)
        model.body_inertia[bodies["LL_FOOT"]] = 1.0001e-6
        model.body_inertia[bodies["left_wrist_roll_link"], 0] += 5e-8
        with self.assertRaisesRegex(AssertionError, "full body inertia tensor: left_wrist_roll_link"):
            _audit_inertials(model, expected, bodies)

    def test_capsule_unused_slot_does_not_hide_real_radius_or_length_changes(self):
        if not (ASSET_DIR / "manifest.json").is_file():
            self.skipTest("Prepare the verified handoff asset")
        model = load_model(ASSET_DIR / "g1_hw3.xml")
        indices = np.flatnonzero(model.geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE))
        self.assertGreater(len(indices), 0)
        # This is Newton's native representation: third slot repeats radius,
        # whereas the direct MuJoCo compiler leaves that unused slot zero.
        model.geom_size[indices, 2] = model.geom_size[indices, 0]
        audit_model(model)
        i = indices[0]
        original = model.geom_size[i].copy()
        for component in (0, 1):
            model.geom_size[i] = original
            model.geom_size[i, component] += .001
            with self.assertRaises(AssertionError):
                audit_model(model)
        model.geom_size[i] = original
        model.geom_quat[i] = [.7071067811865476, .7071067811865476, 0., 0.]
        with self.assertRaises(AssertionError):
            audit_model(model)

    def test_export_mapping_rejects_ambiguous_native_identity(self):
        names = ["hw3_pad", "/World/Robot/hw3_pad_7"]
        model = SimpleNamespace(ngeom=2, nbody=0, njnt=0,
                                geom=lambda i: SimpleNamespace(name=names[i]))
        labels = ["unused"] * 7 + ["/World/Robot/hw3_pad"]
        with self.assertRaisesRegex(RuntimeError, "Ambiguous hw3 geom identity"):
            _entity_mapping(model, ["hw3_pad"], "geom", SimpleNamespace(shape_label=labels))

    def test_exact_newton_export_labels_preserve_full_geometry_audit(self):
        if not (ASSET_DIR / "manifest.json").is_file():
            self.skipTest("Prepare the verified handoff asset")
        root = ET.parse(ASSET_DIR / "g1_hw3.xml").getroot()
        labels = SimpleNamespace(shape_label=["unused"] * 100, body_label=[], joint_label=[])
        for i, geom in enumerate(root.findall(".//geom")):
            name = geom.get("name")
            label = f"/World/envs/env_0/Robot/collisions/{name}/{name}"
            labels.shape_label.append(label)
            geom.set("name", f"{label}_{100+i}")
        for body in root.findall(".//body"):
            label = "/World/envs/env_0/Robot/" + body.get("name")
            labels.body_label.append(label)
            body.set("name", label.replace("/", "_"))
        joint_names = {}
        for joint in [*root.findall(".//joint"), *root.findall(".//freejoint")]:
            name = joint.get("name")
            label = "/World/envs/env_0/Robot/joints/" + name
            labels.joint_label.append(label)
            joint_names[name] = label.replace("/", "_")
            joint.set("name", joint_names[name])
        for actuator in root.findall("actuator/*"):
            actuator.set("joint", joint_names[actuator.get("joint")])
        with asset_cwd(ASSET_DIR):
            model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        report, geoms, dofs, _ = audit_model(model, newton_model=labels)
        self.assertEqual(len(geoms), 47)
        self.assertEqual(len(dofs), 29)
        for proof in report["identity_mapping"]["geoms"].values():
            self.assertEqual(proof["rule"], "Newton SolverMuJoCo exact label export")
            self.assertGreaterEqual(proof["newton_index"], 100)
            self.assertEqual(proof["mujoco_name"], f"{proof['newton_label']}_{proof['newton_index']}")

    @unittest.skipIf(importlib.util.find_spec("isaaclab") is not None, "This assertion is for the dependency-missing CPU host")
    def test_native_cli_reports_unavailable_without_claiming_simulation(self):
        root = Path(__file__).resolve().parents[1]
        cache = root / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            path = Path(tmp) / "native.json"
            result = subprocess.run([sys.executable, "-m", "supermimic.g1_native_smoke", "--output", str(path)],
                                    cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertFalse(report["training_started"])
            self.assertEqual(report["native_status"], "unavailable")
            self.assertIn("isaaclab", report["missing_modules"])


@unittest.skipUnless(HAS_WARP, "Optional Warp/MuJoCo-Warp CPU wheels are not installed")
class NativeFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import warp as wp
        import mujoco_warp as mjw
        cls.wp, cls.mjw = wp, mjw
        wp.config.kernel_cache_dir = str(Path(__file__).resolve().parents[1] / ".cache/warp_test_kernels")
        wp.init()
        cls.kernels = _copy_kernels(wp)

    def make_runtime(self):
        if not (ASSET_DIR / "manifest.json").is_file():
            self.skipTest("Prepare the verified handoff asset for real-array integration tests")
        from supermimic.g1_runtime import audit_model
        root = ET.parse(ASSET_DIR / "g1_hw3.xml").getroot()
        ET.SubElement(root.find("worldbody"), "geom", name="external_floor", type="plane", size="1 1 .1")
        with asset_cwd(ASSET_DIR):
            model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        with self.wp.ScopedDevice("cpu"):
            native = self.mjw.put_model(model)
        runtime = G1PhysicsRuntime.__new__(G1PhysicsRuntime)
        runtime.wp = self.wp
        runtime.solver = SimpleNamespace(mj_model=model, mjw_model=native)
        runtime.N, runtime.device, runtime.randomize = 2, "cpu", True
        runtime.report, runtime.geom_ids, runtime.dofs, runtime.template = audit_model(model)
        runtime.motor = G1Motor(2)
        runtime._buffers, runtime._arrays, runtime._destination_owners = {}, {}, {}
        runtime._copy_batched, runtime._copy_shared = self.kernels
        runtime._bind_native_fields()
        return runtime

    def test_typed_indexed_copy_preserves_unowned_rows(self):
        wp = self.wp
        batched, shared = self.kernels
        with wp.ScopedDevice("cpu"):
            indices = wp.array([1], dtype=wp.int32)
            source = wp.array([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]], dtype=wp.vec3)
            destination = wp.array(np.full((2, 2, 3), .17, dtype=np.float32), dtype=wp.vec3)
            wp.launch(batched, dim=(2, 1), inputs=[source, destination, indices])
            np.testing.assert_array_equal(destination.numpy()[:, 1], source.numpy()[:, 1])
            np.testing.assert_allclose(destination.numpy()[:, 0], .17)
            source_int = wp.array([2, 3], dtype=wp.int32)
            destination_int = wp.array([9, 9], dtype=wp.int32)
            wp.launch(shared, dim=1, inputs=[source_int, destination_int, indices])
            np.testing.assert_array_equal(destination_int.numpy(), [9, 3])

    def test_live_solver_timestep_initialization_uses_configured_interval(self):
        runtime = self.make_runtime()
        runtime.solver.mj_model.opt.timestep = .002
        runtime.solver.mjw_model.opt.timestep.fill_(.002)
        runtime.manager = SimpleNamespace(get_solver_dt=lambda: .005)
        runtime._initialize_solver_timestep()
        runtime.audit_solver_options()
        runtime.solver.mjw_model.opt.timestep.fill_(.002)
        with self.assertRaisesRegex(RuntimeError, "solver timestep differs in mujoco_warp"):
            runtime.audit_solver_options()
        runtime.manager = SimpleNamespace(get_solver_dt=lambda: .02)
        with self.assertRaisesRegex(RuntimeError, "actual solver interval must be 5 ms"):
            runtime._initialize_solver_timestep()

    def test_external_active_priority_cannot_override_soft_hand(self):
        runtime = self.make_runtime()
        report = runtime.audit_external_contact_priority()
        self.assertTrue(report["checked"])
        floor = runtime.solver.mj_model.geom("external_floor").id
        self.wp.to_torch(runtime.solver.mjw_model.geom_priority)[floor] = 1
        with self.assertRaisesRegex(RuntimeError, "object/ground priority=0"):
            runtime.audit_external_contact_priority()

    def test_floor_initialization_preserves_static_objects_and_attached_planes(self):
        wp = self.wp
        # Newton exports GeoType.PLANE=1, BOX=7, MESH=8. Only the enum import
        # is supplied here; the model fields are real Warp arrays consumed by
        # the production initialization method.
        newton_types = SimpleNamespace(GeoType=SimpleNamespace(PLANE=1),
                                       ModelFlags=SimpleNamespace(SHAPE_PROPERTIES=32))
        model = SimpleNamespace(
            shape_body=wp.array([-1, -1, -1, 0], dtype=wp.int32, device="cpu"),
            shape_type=wp.array([1, 7, 8, 1], dtype=wp.int32, device="cpu"),
            shape_material_mu=wp.array([.2, .7, .8, .9], dtype=wp.float32, device="cpu"))
        changes = []
        runtime = G1PhysicsRuntime.__new__(G1PhysicsRuntime)
        runtime.wp, runtime.randomize = wp, True
        runtime.manager = SimpleNamespace(get_model=lambda: model, add_model_change=changes.append)
        with patch.dict(sys.modules, {"newton": newton_types}):
            runtime._initialize_floor_materials()
            np.testing.assert_allclose(model.shape_material_mu.numpy(), [.3, .7, .8, .9])
            runtime.randomize = False
            runtime._initialize_floor_materials()
        np.testing.assert_allclose(model.shape_material_mu.numpy(), [1., .7, .8, .9])
        self.assertEqual(changes, [32, 32])

    def test_real_mjwarp_layout_per_world_friction_and_material_restore(self):
        runtime = self.make_runtime()
        model, native = runtime.solver.mj_model, runtime.solver.mjw_model
        wp = self.wp
        self.assertEqual(tuple(wp.to_torch(native.geom_priority).shape), (model.ngeom,))
        self.assertEqual(tuple(wp.to_torch(native.geom_condim).shape), (model.ngeom,))
        self.assertEqual(tuple(wp.to_torch(native.geom_solref).shape), (2, model.ngeom, 2))
        options = runtime.audit_solver_options()
        self.assertEqual(options["mujoco_template"]["cone"], "elliptic")
        cone = native.opt.cone
        native.opt.cone = int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
        with self.assertRaisesRegex(RuntimeError, "solver options differ"):
            runtime.audit_solver_options()
        native.opt.cone = cone
        runtime.set_surface_friction(torch.tensor([.35, .95]))
        runtime.update_friction(torch.zeros(2, 29))
        floor = model.geom("external_floor").id
        actual_mu = wp.to_torch(native.geom_friction)
        actual_mu[:, floor, 0] = .17
        # Emulate a Newton contact-property reconstruction on owned rows.
        wp.to_torch(native.geom_solref)[:, list(runtime.geom_ids.values())] = .123
        runtime._restore_native_fields()
        report = runtime.readback()
        self.assertTrue(all(v["max_abs_error"] == 0 for v in report.values()))
        torch.testing.assert_close(actual_mu[:, floor, 0], torch.full((2,), .17))
        for name, i in runtime.geom_ids.items():
            if "rubber_hand" in name or "wrist_yaw_link" in name:
                torch.testing.assert_close(actual_mu[:, i, 0], torch.ones(2))
            else:
                torch.testing.assert_close(actual_mu[:, i, 0], torch.tensor([.35, .95]))


if __name__ == "__main__":
    unittest.main()
