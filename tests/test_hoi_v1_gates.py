"""Native evidence must be real/complete; WM proposals never imply calibration."""
import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from hoi.v1.gates import (GateError, WMThresholdProposal, require_native_evidence,
                         require_wm_evidence, validate_native_evidence, validate_wm_evidence, main)


def native_report(asset_id=0):
    probes = {name: {"maximum_interval_normal_force_N": force} for name, force in (
        ("left_hand_object", [2.,0,0]), ("right_hand_object", [0,2.,0]), ("object_ground", [0,0,2.]))}
    probes["hand_ground_negative"] = {"maximum_interval_normal_force_N": [0,0,0], "maximum_hand_ground_interval_force_N": [2.,0]}
    probes.update(passed=True, checks={key: True for key in ("left_hand_object", "right_hand_object", "object_ground", "hand_ground_not_mislabelled")})
    hand = {side: {"collision_geom_count": 5, "allowed_object_pairs": 5, "cpu_probe_contacts": [{"distance": -.001}]} for side in ("left", "right")}
    return {"asset_id": asset_id, "bundle_content_sha256": "a" * 64,
        "physics": {"physics": {"hand_collision": True, "self_collision": True},
            "control_dt": .02, "substeps": 4, "integration_dt": .005, "mass": [1.], "com": [[0,0,0]],
            "inertia": [[[1.,0,0],[0,1.,0],[0,0,1.]]], "object_friction": [.5], "ground_friction": .3,
            "gravity": [[0,0,-9.81]], "velocity_reference_point": "link_origin", "object_pose_origin": "link",
            "collision_hash": "b" * 64, "flattened_object_usd_sha256": "c" * 64,
            "native_convex_hull_export": {"source": "mjModel.mesh_graph", "all_geometries_graph_backed": True,
                "object_hulls": 2, "hand_hulls": {"left": 5, "right": 5}, "object_evidence_sha256": "d"*64,
                "hand_evidence_sha256": {"left": "e"*64, "right": "f"*64}},
            "independent_native_geometry_queries": {"passed": True,
                "api": "mujoco.mj_geomDistance convex collision distance", "tolerance_m": 5e-5,
                "samples": [{"hand": side, "pose": pose, "native_distance_m": .1, "cached_surface_distance_m": .10001,
                    "abs_error_m": .00001, "cache_pairs_queried": 3, "cache_pairs_total": 10}
                    for side in ("left", "right") for pose in (0,1)]},
            "robot_fk_comparison": {"passed": True, "max_link_position_error_m": 1e-6,
                "max_rotation_matrix_error": 1e-6, "position_tolerance_m": 5e-5, "rotation_tolerance": 5e-5},
            "compiled_collision_geometry_verified_against_bundle": True, "use_cuda_graph": False},
        "contact": {"schema": "hoi_v1_pair_normal_impulse_v1", "channels": ["left_hand_object", "right_hand_object", "object_ground"],
            "interval": "(t,t+1]", "num_substeps": 4, "substep_dt": .005, "threshold_N": 1., "use_cuda_graph": False,
            "normal_force_api": "mujoco_warp.contact_force(to_world_frame=False)[0]"},
        "palm_selection_verified": True, "compiled_hand_audit": {"enabled": True, "hands": hand},
        "compiled_self_collision_audit": {"enabled": True, "allowed_nonadjacent_robot_geom_pairs": 8},
        "pair_probes": probes, "native_pair_calibration_complete": True, "zero_action_smoke_completed": True}


def wm_report():
    metric = {"valid_samples": 20, "object_normalized_error": .1, "zero_increment_normalized_error": 1., "constant_velocity_normalized_error": .5}
    horizons = {str(n): {**metric, "phases": {p: dict(metric) for p in ("no_contact", "contact", "onset", "release")}} for n in (1,8,32)}
    return {"finite": True, "scope": "training_reference_bins_reserved_from_wm_fit_not_eval80_or_test40",
        "real_executed_actions": True, "horizon_reports": horizons,
        "contact_diagnostics": {"positive_counts": [20,20,20], "negative_counts": [20,20,20],
            "precision": [.8,.8,.8], "recall": [.8,.8,.8], "brier_per_channel": [.1,.1,.1], "prevalence_baseline_brier": [.25,.25,.25]}}


class EvidenceGateTests(unittest.TestCase):
    def test_missing_native_evidence_and_random_hand_probes_close_gate(self):
        result = validate_native_evidence({})
        self.assertFalse(result["passed"])
        report = native_report()
        report["palm_selection_verified"] = False
        result = validate_native_evidence(report)
        self.assertFalse(result["passed"])
        self.assertTrue(any("palm" in x for x in result["blockers"]))
        with self.assertRaises(GateError):
            require_native_evidence(report)

    def test_native_pair_identity_timebase_bundle_and_full_group_coverage(self):
        report = native_report()
        self.assertTrue(validate_native_evidence(report, required_asset_ids=[0], expected_bundle_sha256="a"*64)["passed"])
        for field, value in (("num_substeps", 1), ("interval", "last_substep_only"), ("normal_force_api", "wrist_net_force")):
            altered = copy.deepcopy(report)
            altered["contact"][field] = value
            self.assertFalse(validate_native_evidence(altered)["passed"])
        altered = copy.deepcopy(report)
        altered["pair_probes"]["hand_ground_negative"]["maximum_interval_normal_force_N"][0] = 3
        self.assertFalse(validate_native_evidence(altered)["passed"])
        self.assertFalse(validate_native_evidence(report, expected_bundle_sha256="d"*64)["passed"])
        self.assertFalse(validate_native_evidence(report, required_asset_ids=[0,1])["passed"])
        self.assertTrue(validate_native_evidence({"reports": [report, native_report(1)]}, required_asset_ids=[0,1])["passed"])

    def test_missing_geometry_or_mass_readback_cannot_pass_cpu_mock_as_native(self):
        report = native_report()
        report["physics"]["compiled_collision_geometry_verified_against_bundle"] = False
        self.assertFalse(validate_native_evidence(report)["passed"])
        report = native_report()
        report["physics"]["mass"] = []
        self.assertFalse(validate_native_evidence(report)["passed"])
        report = native_report()
        report["physics"]["inertia"][0][0][0] = -1.
        self.assertFalse(validate_native_evidence(report)["passed"])

    def test_native_geometry_requires_engine_hulls_independent_queries_and_fk(self):
        for key in ("native_convex_hull_export", "independent_native_geometry_queries", "robot_fk_comparison"):
            report = native_report()
            del report["physics"][key]
            self.assertFalse(validate_native_evidence(report)["passed"], key)
        report = native_report()
        report["physics"]["native_convex_hull_export"]["source"] = "mjModel.mesh_face_raw_triangles"
        self.assertFalse(validate_native_evidence(report)["passed"])
        for mutation in ("tolerance", "error", "coverage", "pair_count"):
            report = native_report()
            audit = report["physics"]["independent_native_geometry_queries"]
            if mutation == "tolerance":
                audit["tolerance_m"] = .01
            elif mutation == "error":
                audit["samples"][0]["abs_error_m"] = .001
            elif mutation == "coverage":
                audit["samples"] = audit["samples"][:2]
            else:
                audit["samples"][0]["cache_pairs_total"] = 999
            self.assertFalse(validate_native_evidence(report)["passed"], mutation)
        report = native_report()
        report["physics"]["robot_fk_comparison"]["max_link_position_error_m"] = .001
        self.assertFalse(validate_native_evidence(report)["passed"])

    def test_wm_success_is_explicitly_only_an_uncalibrated_startup_proposal(self):
        result = require_wm_evidence(wm_report())
        self.assertTrue(result["passed"])
        self.assertEqual(result["threshold_proposal"]["calibration_status"], "proposal_not_empirically_calibrated")
        json.dumps(result, allow_nan=False)

    def test_wm_gate_requires_1_8_32_baselines_phases_and_pair_coverage(self):
        self.assertFalse(validate_wm_evidence({})["passed"])
        report = wm_report()
        report["horizon_reports"]["32"]["object_normalized_error"] = 1.
        self.assertFalse(validate_wm_evidence(report)["passed"])
        report = wm_report()
        report["horizon_reports"]["8"]["phases"]["onset"]["valid_samples"] = 0
        self.assertFalse(validate_wm_evidence(report)["passed"])
        report = wm_report()
        report["contact_diagnostics"]["recall"][0] = 0
        self.assertFalse(validate_wm_evidence(report)["passed"])
        report = wm_report()
        report["contact_diagnostics"]["brier_per_channel"][1] = .26
        self.assertFalse(validate_wm_evidence(report)["passed"])
        report = wm_report()
        report["real_executed_actions"] = False
        with self.assertRaises(GateError):
            require_wm_evidence(report)

    def test_cli_binds_report_bytes_and_preserves_blocked_evidence(self):
        from hoi.v1.prepare import BUNDLE_SCHEMA, _digest_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = {"schema_version": BUNDLE_SCHEMA, "clips": [{"asset_id": 0}, {"asset_id": 1}], "partition": "train"}
            bundle["bundle_content_sha256"] = _digest_json(bundle)
            bundle_path = root / "bundle.json"
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
            reports = []
            for asset in range(2):
                report = native_report(asset)
                report["bundle_content_sha256"] = bundle["bundle_content_sha256"]
                path = root / f"native_{asset}.json"
                path.write_text(json.dumps(report), encoding="utf-8")
                reports.append(path)
            output = root / "all_assets"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--bundle", str(bundle_path), "--native-report", str(reports[0]),
                                       "--native-report", str(reports[1]), "--output", str(output)]), 0)
            evidence_path = output / "native_evidence.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["inputs"][0]["sha256"], hashlib.sha256(reports[0].read_bytes()).hexdigest())
            gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
            self.assertEqual(gate["evidence_sha256"], hashlib.sha256(evidence_path.read_bytes()).hexdigest())
            blocked = root / "missing_asset"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--bundle", str(bundle_path), "--native-report", str(reports[0]), "--output", str(blocked)]), 1)
            self.assertTrue((blocked / "native_evidence.json").exists())
            self.assertFalse(json.loads((blocked / "gate.json").read_text(encoding="utf-8"))["passed"])


class RealWindowDiagnosticsTests(unittest.TestCase):
    def test_real_action_window_masks_reset_and_reports_all_horizons(self):
        from hoi.v1.learner import Learner
        from hoi.v1.state import flatten_object, flatten_state, unflatten_object, unflatten_state
        B, K = 2, 32
        ticks = torch.arange(K).float()[None].expand(B, K) * .02
        p = torch.stack((ticks, torch.zeros_like(ticks), torch.ones_like(ticks)), -1)
        eye = torch.eye(3).expand(B,K,3,3)
        velocity = torch.zeros(B,K,3)
        velocity[...,0] = 1.
        z3, z29 = torch.zeros(B,K,3), torch.zeros(B,K,29)
        S = p, eye, velocity, z3, z29, z29
        O = p + .2, eye, velocity, z3
        S1, O1 = (p + velocity * .02, *S[1:]), (O[0] + velocity * .02, *O[1:])
        contact = ((torch.arange(K) // 4) % 2).float()[None,:,None].expand(B,K,3).clone()
        mask = torch.ones(B,K)
        mask[1,11:] = 0
        window = {"s": flatten_state(S), "s1": flatten_state(S1), "o": flatten_object(O), "o1": flatten_object(O1),
            "a": torch.zeros(B,K,29), "xi": torch.zeros(B,K,33), "xi_o": torch.zeros(B,K,12),
            "asset_id": torch.zeros(B,K,dtype=torch.long), "mask": mask, "contact": contact,
            "contact_valid": torch.ones(B,K,3,dtype=torch.bool)}
        class WM:
            object_out_norm = SimpleNamespace(std=lambda: torch.ones(12))
            object_error_floor = torch.full((12,), .001)
            def step(self, S, O, a, xi, xi_o, points):
                return (S[0] + S[2] * .02, *S[1:]), (O[0] + O[2] * .02, *O[1:]), a.new_zeros((len(a),3))
            def loss(self, *args):
                return {name: torch.tensor(0.) for name in ("total", "robot", "object", "contact")}
        learner = Learner.__new__(Learner)
        learner.device, learner.wm = torch.device("cpu"), WM()
        learner.config = SimpleNamespace(wm_window=8, wm_batch=B)
        learner.buffer = SimpleNamespace(sample_windows=lambda b,k,split: window)
        learner.shapes = SimpleNamespace(points=lambda ids: torch.zeros((*ids.shape,128,3)))
        report = learner.wm_validation()
        self.assertEqual(set(report["horizon_reports"]), {"1","8","32"})
        self.assertEqual(report["horizon_reports"]["1"]["valid_samples"], 43)
        self.assertEqual(report["horizon_reports"]["8"]["valid_samples"], 2)
        self.assertEqual(report["horizon_reports"]["32"]["valid_samples"], 1)
        self.assertLess(report["horizon_reports"]["32"]["object_normalized_error"], 1e-10)
        self.assertTrue(report["finite"] and report["real_executed_actions"])
        self.assertGreater(report["horizon_reports"]["1"]["phases"]["onset"]["valid_samples"], 0)
        self.assertEqual(report["contact_diagnostics"]["positive_counts"], [20,20,20])
        self.assertFalse(validate_wm_evidence(report)["passed"])
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
