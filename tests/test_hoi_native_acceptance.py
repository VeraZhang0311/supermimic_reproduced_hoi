"""Evidence validation tests; synthetic records are not native GPU results."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch

from hoi.v1.contacts import CHANNELS
from hoi.v1.gates import validate_native_evidence
from hoi.v1.native_acceptance import capture_native_runtime, native_acceptance_blockers, probe_trace_blockers
from tests.test_hoi_v1_gates import native_report


def support_report():
    """Synthetic fields only; never an actual simulator result."""
    report = native_report()
    data = report["native_scene_supports"]
    data["declared_support_ids"] = ["table1"]
    data["clips"][0]["bindings"] = [{"support_id": "table1", "binding_sha256": "1"*64,
        "source_support_sha256": "2"*64, "target_timebase_sha256": "3"*64}]
    geom_id = len(report["contact"]["geom_roles"])
    report["contact"]["geom_roles"].append(20)
    report["contact"]["geom_names"].append("Support_table1")
    report["contact"]["geom_canonical_links"].append(None)
    data["supports"] = [{"support_id": "table1", "geometry_recipe_sha256": data["geometry_recipe_sha256"],
        "compiled_geom_ids": [geom_id], "native_body_flags": [2,2], "kinematic_verified": True,
        "instances": [{"geom_id": geom_id, "geom_name": "Support_table1", "body_id": 101,
            "role": 20, "size_m": [.3,.25,.01], "geom_friction": [1.,.005,.0001]}]}]
    data["isolation_probes"] = {"table1": {case: {
        "ticks": 2, "control_dt_s": .02, "substeps_per_interval": 4,
        "interval_normal_impulse_Ns": torch.zeros(2,2,16).tolist(),
        "interval_support_impulse_Ns": [[.04,.04],[.04,.04]], "passed": True}
        for case in ("body_support","object_support")}}
    data["alignment"] = {"reset_count": 3, "step_count": 16, "max_position_error_m": 0.,
        "max_quaternion_error": 0.}
    pose = [[0.,0.,1.,0.,0.,0.,1.], [2.,0.,1.,0.,0.,0.,1.]]
    data["reference_reset_audit"] = {"environment_ids": [0,1], "frames": [0,1],
        "clip_indices": [0,0], "clip_local_frames": [0,1], "supports": {"table1": {
            "expected_pose_xyzw": deepcopy(pose), "actual_pose_xyzw": deepcopy(pose),
            "actual_velocity": torch.zeros(2,6).tolist()}}}
    return report


class NativeAcceptanceTests(unittest.TestCase):
    def test_complete_synthetic_contract_is_accepted(self):
        self.assertEqual(native_acceptance_blockers(native_report()), [])

    def test_cpu_live_buffers_cannot_produce_cuda_runtime_evidence(self):
        recorder = SimpleNamespace(wp=SimpleNamespace(to_torch=lambda x: x), forces=torch.zeros(2, 6),
            impulse=torch.zeros(2, 16), solver=SimpleNamespace(mjw_data=SimpleNamespace(
                contact=SimpleNamespace(geom=torch.zeros(2, 2, dtype=torch.int32)), qpos=torch.zeros(2, 7))))
        with self.assertRaisesRegex(RuntimeError, "CUDA device"):
            capture_native_runtime(recorder)

    def test_runtime_missing_cpu_or_mixed_gpu_cannot_pass(self):
        for mutation in ("missing", "cpu", "mixed", "backend", "unverified"):
            report = native_report()
            if mutation == "missing":
                report.pop("native_runtime")
            elif mutation == "cpu":
                report["native_runtime"]["buffer_devices"]["qpos"] = "cpu"
            elif mutation == "mixed":
                report["native_runtime"]["buffer_devices"]["qpos"] = "cuda:1"
            elif mutation == "backend":
                report["native_runtime"]["backend"] = "CPU MuJoCo"
            else:
                report["native_runtime"]["verified_from_live_buffers"] = False
            self.assertFalse(validate_native_evidence(report)["passed"], mutation)

    def test_impulse_force_maximum_and_shape_consistency_are_recomputed(self):
        for mutation in ("trace_missing", "force", "maximum", "negative_impulse", "timing", "environment_missing"):
            report = native_report()
            probe = report["pair_probes"]["pelvis_object"]
            if mutation == "trace_missing":
                probe.pop("interval_normal_impulse_Ns")
            elif mutation == "force":
                probe["interval_mean_normal_force_N"][0][0][7] = 4.
            elif mutation == "maximum":
                probe["maximum_interval_normal_force_N"][7] = 4.
            elif mutation == "negative_impulse":
                probe["interval_normal_impulse_Ns"][0][0][7] = -.04
            elif mutation == "timing":
                probe["substeps_per_interval"] = 1
            else:
                probe["interval_normal_impulse_Ns"][0].pop()
            self.assertFalse(validate_native_evidence(report)["passed"], mutation)

    def test_maximum_in_one_world_does_not_hide_uncovered_worlds(self):
        for name in ("head_object", "head_ground_negative"):
            report = native_report()
            probe = report["pair_probes"][name]
            if name == "head_object":
                for key in ("interval_normal_impulse_Ns", "interval_mean_normal_force_N"):
                    for row in probe[key]:
                        row[1][14] = 0.
            else:
                for row in probe["interval_body_ground_impulse_Ns"]:
                    row[1][14] = 0.
            self.assertTrue(any("every environment" in x for x in native_acceptance_blockers(report)), name)
            self.assertFalse(validate_native_evidence(report)["passed"])

    def test_bound_profile_changes_or_readback_mismatch_are_rejected(self):
        for key in ("mass", "com", "inertia", "object_friction", "ground_friction", "bound_physical_profile_sha256"):
            report = native_report()
            physics = report["physics"]
            if key == "ground_friction":
                physics[key] = .4
            elif key == "bound_physical_profile_sha256":
                physics[key] = "0" * 64
            elif key in ("mass", "object_friction"):
                physics[key][1] += .1
            elif key == "com":
                physics[key][1][0] += .1
            else:
                physics[key][1][0][0] += .1
            self.assertFalse(validate_native_evidence(report)["passed"], key)

    def test_raw_compiled_mapping_cannot_hide_or_relabel_active_colliders(self):
        for mutation in ("missing", "role", "count", "audit_missing", "audit_role"):
            report=native_report()
            contact=report["contact"]
            if mutation == "missing":
                contact.pop("geom_canonical_links")
            elif mutation == "role":
                contact["geom_canonical_links"][7] = "head_link"
            elif mutation == "count":
                contact["region_collision_geom_counts"]["pelvis_object"] += 1
            else:
                audit=report["pair_probes"]["head_object"]["definition"]["region_pose_audit"]
                if mutation == "audit_missing":
                    for key in ("robot_geom_ids","robot_geom_roles","per_geom_minimum_world_z_m"):
                        audit[key].pop()
                else:
                    audit["robot_geom_roles"] = list(audit["robot_geom_roles"])
                    audit["robot_geom_roles"][0] = 17
            self.assertFalse(validate_native_evidence(report)["passed"],mutation)

    def test_non_target_penetration_is_recomputed_instead_of_trusting_passed(self):
        for name in ("pelvis_object", "pelvis_ground_negative"):
            report = native_report()
            audit = report["pair_probes"][name]["definition"]["region_pose_audit"]
            audit["per_geom_minimum_world_z_m"][0] = -.4
            self.assertTrue(any("deeply penetrates" in x for x in native_acceptance_blockers(report)))
            self.assertFalse(validate_native_evidence(report)["passed"])

    def test_source_geometry_floor_or_self_collision_checks_cannot_be_disabled(self):
        for mutation in ("missing", "visual", "self", "collision", "limit", "target"):
            report = native_report()
            audit = report["pair_probes"]["head_ground_negative"]["definition"]["region_pose_audit"]
            if mutation == "missing":
                audit.pop("per_geom_minimum_world_z_m")
            elif mutation == "visual":
                audit["compiled_surface"]["source"] = "visual_mesh"
            elif mutation == "self":
                audit["maximum_self_penetration_m"] = .08
            elif mutation == "collision":
                audit["search"]["collisions_disabled"] = True
            elif mutation == "limit":
                audit["search"]["joint_limits_preserved"] = False
            else:
                audit["compiled_surface"]["geom_id"] = 1
            self.assertFalse(validate_native_evidence(report)["passed"],mutation)

    def test_support_complete_synthetic_contract_is_accepted(self):
        result = validate_native_evidence(support_report())
        self.assertTrue(result["passed"],result["blockers"])

    def test_support_events_are_required_in_every_world_without_learned_leakage(self):
        for mutation in ("missing", "leak", "one_world", "negative", "timing"):
            report = support_report()
            probes = report["native_scene_supports"]["isolation_probes"]["table1"]
            if mutation == "missing":
                probes.pop("object_support")
            elif mutation == "leak":
                probes["body_support"]["interval_normal_impulse_Ns"][0][0][15] = .01
            elif mutation == "one_world":
                for row in probes["object_support"]["interval_support_impulse_Ns"]:
                    row[1] = 0.
            elif mutation == "negative":
                probes["body_support"]["interval_support_impulse_Ns"][0][0] = -.04
            else:
                probes["body_support"]["substeps_per_interval"] = 1
            self.assertFalse(validate_native_evidence(report)["passed"],mutation)

    def test_support_identity_physics_and_actual_reference_pose_are_bound(self):
        for mutation in ("omitted", "geometry", "friction", "role", "binding", "pose", "velocity", "world"):
            report = support_report()
            data = report["native_scene_supports"]
            if mutation == "omitted":
                report.pop("native_scene_supports")
            elif mutation == "geometry":
                data["supports"][0]["instances"][0]["size_m"][0] = .6
            elif mutation == "friction":
                data["supports"][0]["instances"][0]["geom_friction"][0] = .3
            elif mutation == "role":
                data["supports"][0]["instances"][0]["role"] = 0
            elif mutation == "binding":
                data["clips"][0]["bindings"][0]["source_support_sha256"] = None
            elif mutation == "pose":
                data["reference_reset_audit"]["supports"]["table1"]["actual_pose_xyzw"][1][0] += .1
            elif mutation == "velocity":
                data["reference_reset_audit"]["supports"]["table1"]["actual_velocity"][1][0] = .1
            else:
                data["reference_reset_audit"]["environment_ids"] = [0]
            self.assertFalse(validate_native_evidence(report)["passed"],mutation)

    def test_training_claim_cannot_pass_acceptance_only_contract(self):
        report = native_report()
        report["formal_training_started"] = True
        self.assertFalse(validate_native_evidence(report)["passed"])


if __name__ == "__main__":
    unittest.main()
