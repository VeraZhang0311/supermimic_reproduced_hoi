"""Bounded float32 reconstruction evidence for the exact two table tensors."""
import ast
import inspect
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from supermimic.rotations import quat_to_mat
from hoi_residual import native_checks as original
from hoi_tables.native_checks import (audit_object_solver, _check, check_body_frame_inertia,
                                     reconstruct_body_frame_inertia,
                                     RELATIVE_FROBENIUS_BOUND, PHYSICAL_RELATIVE_FROBENIUS_BOUND, FLOAT32_EPS)
from hoi_tables import native_checks as checks


# Exact nominal profiles read from the frozen smoke input assets.json, 2026-11-07.
# These are inherited engineering physical profiles (6kg/4kg), not measurements.
TABLE_INERTIAS = {
    "largetable": [[.19557019294412023, .0002691695387581407, -.00042472844945237357],
                   [.0002691695387581407, .29077930327253465, -.00010768657271409911],
                   [-.00042472844945237357, -.00010768657271409911, .19435333214204994]],
    "smalltable": [[.05987904430320412, -.0001425174953031608, .000027243433228650965],
                   [-.0001425174953031608, .07705228078550068, -.00014001347354658387],
                   [.000027243433228650965, -.00014001347354658387, .05985409225913318]],
}


def principal_frame_roundtrip(inertia):
    values, vectors = torch.linalg.eigh(inertia.float())
    if torch.linalg.det(vectors) < 0:
        vectors[:, 0] *= -1
    q = Rotation.from_matrix(vectors.double().numpy()).as_quat()
    quaternion = torch.tensor(np.r_[q[3], q[:3]], dtype=torch.float32)
    rotation = quat_to_mat(quaternion)
    return rotation @ torch.diag(values) @ rotation.T


class NativeAuditTests(unittest.TestCase):
    def test_actual_profiles_float32_roundtrip_exposes_old_entrywise_rejection(self):
        old_rejections, maxima = [], []
        for asset, values in TABLE_INERTIAS.items():
            nominal = torch.tensor(values, dtype=torch.float64)
            for scale in torch.linspace(.7, 1.3, 61, dtype=torch.float64):
                expected = (nominal * scale).float()
                actual = principal_frame_roundtrip(expected)
                report = check_body_frame_inertia(actual[None], expected[None])
                self.assertTrue(report["all_worlds_within_bound"])
                self.assertLessEqual(report["max_relative_frobenius_error"], RELATIVE_FROBENIUS_BOUND)
                maxima.append(report["max_relative_frobenius_error"])
                try:
                    original._check("body-frame inertia", actual[None], expected[None], atol=1e-8)
                except RuntimeError:
                    old_rejections.append((asset, float(scale), report["max_abs_error"]))
        self.assertGreater(len(old_rejections), 0, "Fixture must reproduce the old near-zero entrywise failure")
        self.assertLess(max(maxima), 8 * FLOAT32_EPS)

    def test_wrong_stale_scale_and_offdiagonal_are_rejected(self):
        for values in TABLE_INERTIAS.values():
            expected = torch.tensor(values, dtype=torch.float32)[None]
            wrong = expected.clone()
            wrong[:, 0, 1] += 1e-4
            wrong[:, 1, 0] += 1e-4
            for bad in (wrong, expected * 1.001, expected * (.7 / 1.3), torch.zeros_like(expected)):
                with self.assertRaises(RuntimeError): check_body_frame_inertia(bad, expected)

    def test_bound_is_per_world_and_evidence_records_measured_matrices(self):
        expected = torch.stack([torch.tensor(v) for v in TABLE_INERTIAS.values()])
        actual = torch.stack([principal_frame_roundtrip(v) for v in expected])
        report = check_body_frame_inertia(actual, expected)
        self.assertEqual(report["shape"], [2, 3, 3])
        self.assertEqual(report["roundtrip_ulps"], 8)
        worst = report["worst_world"]
        self.assertEqual(worst["actual"], actual[worst["index"]].double().tolist())
        self.assertEqual(worst["expected"], expected[worst["index"]].double().tolist())
        self.assertAlmostEqual(worst["allowed_error_frobenius_norm"],
                               PHYSICAL_RELATIVE_FROBENIUS_BOUND * worst["expected_frobenius_norm"])
        self.assertAlmostEqual(worst["allowed_native_reference_error_frobenius_norm"],
                               RELATIVE_FROBENIUS_BOUND * worst["expected_frobenius_norm"])
        bad = actual.clone()
        bad[1] += torch.eye(3) * 1e-4
        with self.assertRaises(RuntimeError): check_body_frame_inertia(bad, expected)

    def test_malformed_tensors_are_rejected(self):
        expected = torch.eye(3)[None]
        for bad in (torch.zeros(1, 3), torch.full((1, 3, 3), float("nan")), torch.full((1, 3, 3), float("inf"))):
            with self.assertRaises(RuntimeError): check_body_frame_inertia(bad, expected)
        with self.assertRaises(RuntimeError): check_body_frame_inertia(expected, -expected)
        with self.assertRaises(RuntimeError): check_body_frame_inertia(expected[:0], expected[:0])

    def test_other_native_checks_and_audit_logic_remain_exact(self):
        # The only changed audit statements reconstruct the inertia tensor.
        # All identity, norm, mass, COM, friction, gravity, overflow and report
        # logic must be AST-identical to the unchanged inherited implementation.
        def remaining_body(function):
            node = ast.parse(inspect.getsource(function)).body[0]
            node.body = [statement for statement in node.body if not (
                isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("rotation", "inertia")
                                                         for t in statement.targets))]
            return ast.dump(node, include_attributes=False)
        self.assertEqual(remaining_body(audit_object_solver), remaining_body(original.audit_object_solver))
        for name in ("_worlds", "_entity_mapping"):
            self.assertIs(audit_object_solver.__globals__[name], original.audit_object_solver.__globals__[name])
        self.assertIs(audit_object_solver.__globals__["_check"], _check)
        for name, atol, rtol in (("mass", 1e-6, 2e-5), ("COM", 2e-7, 2e-5),
                                  ("sliding friction", 1e-6, 2e-5), ("gravity compensation", 1e-7, 0)):
            actual, expected = torch.tensor([.3, .4]), torch.tensor([.3, .4])
            self.assertEqual(_check(name, actual, expected, atol=atol, rtol=rtol),
                             original._check(name, actual, expected, atol=atol, rtol=rtol))
            with self.assertRaises(RuntimeError): _check(name, actual + 1e-3, expected, atol=atol, rtol=rtol)

    def test_float64_normalized_readback_handles_valid_quaternion_radial_roundoff(self):
        expected = torch.tensor(TABLE_INERTIAS["smalltable"], dtype=torch.float32)
        principal, basis = torch.linalg.eigh(expected)
        if torch.linalg.det(basis) < 0: basis[:, 0] *= -1
        # A cyclic principal-axis permutation represents the identical tensor
        # and exercises a nontrivial quaternion instead of a near-identity one.
        principal, basis = principal[[1, 2, 0]], basis[:, [1, 2, 0]]
        q = Rotation.from_matrix(basis.double().numpy()).as_quat()
        quaternion = torch.tensor(np.r_[q[3], q[:3]], dtype=torch.float32)
        # The native hard norm guard admits this small storage residual. A
        # scaled quaternion denotes the same orientation after normalization.
        quaternion *= 1 + 4 * FLOAT32_EPS
        self.assertTrue(torch.allclose(quaternion.norm(), torch.tensor(1.), atol=1e-5, rtol=0))
        rotation32 = quat_to_mat(quaternion)
        readback32 = rotation32 @ torch.diag(principal) @ rotation32.T
        self.assertEqual(readback32.dtype, torch.float32)
        q_before, d_before = quaternion.clone(), principal.clone()
        readback64 = reconstruct_body_frame_inertia(principal, quaternion)
        report = check_body_frame_inertia(readback64[None], expected[None])
        self.assertEqual(readback64.dtype, torch.float64)
        self.assertEqual(report["readback_dtype"], "torch.float64")
        self.assertEqual(report["expected_storage_dtype"], "torch.float32")
        self.assertEqual(report["roundtrip_ulps"], 8)
        self.assertTrue(torch.equal(q_before, quaternion))
        self.assertTrue(torch.equal(d_before, principal))

    def test_twenty_thousand_random_scales_per_profile_with_float32_storage(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            generator = torch.Generator().manual_seed(848)
            scales = .7 + .6 * torch.rand(20000, generator=generator)
            for values in TABLE_INERTIAS.values():
                expected = torch.tensor(values, dtype=torch.float32)[None] * scales[:, None, None]
                principal, basis = torch.linalg.eigh(expected)
                basis[:, :, 0] *= torch.where(torch.linalg.det(basis) < 0, -1., 1.)[:, None]
                xyzw = Rotation.from_matrix(basis.double().numpy()).as_quat()
                quaternion = torch.tensor(np.concatenate((xyzw[:, 3:], xyzw[:, :3]), axis=1), dtype=torch.float32)
                quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
                # One representable step after float32 quaternion normalization
                # models storage residue without perturbing physical inputs.
                away = torch.where(quaternion >= 0, torch.full_like(quaternion, torch.inf),
                                   torch.full_like(quaternion, -torch.inf))
                quaternion = torch.nextafter(quaternion, away)
                self.assertTrue(torch.allclose(quaternion.norm(dim=-1), torch.ones_like(scales), atol=1e-5, rtol=0))
                actual = reconstruct_body_frame_inertia(principal, quaternion)
                report = check_body_frame_inertia(actual, expected)
                self.assertTrue(report["all_worlds_within_bound"])
                self.assertLess(report["max_relative_frobenius_error"], RELATIVE_FROBENIUS_BOUND)
        finally:
            torch.set_num_threads(previous_threads)

    def test_failure_carries_actual_expected_matrices_and_bound_evidence(self):
        expected = torch.tensor(TABLE_INERTIAS["smalltable"], dtype=torch.float32)[None]
        actual = expected.double().clone()
        actual[:, 0, 1] += 1e-4
        actual[:, 1, 0] += 1e-4
        with self.assertRaises(RuntimeError) as caught:
            check_body_frame_inertia(actual, expected)
        error = caught.exception
        evidence = error.evidence
        self.assertFalse(evidence["all_worlds_within_bound"])
        self.assertEqual(evidence["worst_world"]["actual"], actual[0].tolist())
        self.assertEqual(evidence["worst_world"]["expected"], expected[0].double().tolist())
        self.assertGreater(evidence["worst_world"]["error_frobenius_norm"],
                           evidence["worst_world"]["allowed_error_frobenius_norm"])
        serialized = str(error).split(": ", 1)[1]
        self.assertEqual(json.loads(serialized), evidence)

    def test_captured_native_conversion_residual_and_independent_guards(self):
        # Actual formal_02 native float64 reconstruction, smalltable world1407.
        # This legitimate official fixed-sweep encoding exceeded the previous
        # unsupported8eps physical bound (relative error1.170463e-6).
        expected = torch.tensor([[[.045760560780763626, -.00010891423880821094, .00002081988350255415],
                                  [-.00010891423880821094, .0588846318423748, -.00010700062557589263],
                                  [.00002081988350255415, -.00010700062557589263, .0457414910197258]]], dtype=torch.float32)
        actual = torch.tensor([[[.04576048866561587, -.00010891444539548844, .000020821565147845916],
                                [-.00010891444539548839, .05888463498443908, -.00010700101920629517],
                                [.000020821565147845916, -.00010700101920629515, .04574141843177797]]], dtype=torch.float64)
        identity = {"kernel_sha256": checks.PINNED_KERNEL_SHA256, "private_temporary_outputs": True}
        with self.assertRaises(RuntimeError): check_body_frame_inertia(actual, expected)
        report = check_body_frame_inertia(actual, expected, reference=actual.clone(), kernel_identity=identity)
        self.assertGreater(report["max_relative_frobenius_error"], RELATIVE_FROBENIUS_BOUND)
        self.assertTrue(report["native_reference_match"])
        self.assertTrue(report["physical_guard_passed"])
        self.assertEqual(report["reference_kernel"], identity)
        # A deviation small enough to pass the2e-5 physical guard must still
        # fail the independent8eps native/reference check.
        stale = actual + torch.eye(3)[None] * 5e-7
        with self.assertRaises(RuntimeError) as caught:
            check_body_frame_inertia(stale, expected, reference=actual)
        self.assertTrue(caught.exception.evidence["physical_guard_passed"])
        self.assertFalse(caught.exception.evidence["native_reference_match"])
        # Matching native/reference encodings do not excuse inaccurate physics.
        inaccurate = actual * 1.001
        with self.assertRaises(RuntimeError) as caught:
            check_body_frame_inertia(inaccurate, expected, reference=inaccurate)
        self.assertTrue(caught.exception.evidence["native_reference_match"])
        self.assertFalse(caught.exception.evidence["physical_guard_passed"])
        with patch.object(checks, "_encode_reference_inertia", return_value=(actual, identity)) as encode:
            self.assertTrue(_check("body-frame inertia", actual, expected)["all_worlds_within_bound"])
        self.assertIs(encode.call_args.args[0], expected)

    def test_reference_kernel_uses_live_inputs_and_private_outputs(self):
        # Exercise the complete buffer/mapping/launch plumbing without requiring
        # Newton or a GPU in the CPU unit-test environment.
        outputs_seen = []
        kernel = object()
        class Array:
            def __init__(self, tensor): self.tensor = tensor
        def launch(function, *, dim, inputs, outputs, device, stream):
            self.assertIs(function, kernel)
            mapping, tensor = inputs
            self.assertTrue(torch.equal(mapping.tensor.flatten(), torch.arange(dim[0], dtype=torch.int32)))
            self.assertEqual(tensor.tensor.dtype, torch.float32)
            for value in outputs:
                self.assertNotEqual(value.tensor.data_ptr(), tensor.tensor.data_ptr())
                outputs_seen.append(value.tensor.data_ptr())
            outputs[0].tensor[:, 0] = tensor.tensor.diagonal(dim1=-2, dim2=-1)
            outputs[1].tensor.zero_(); outputs[1].tensor[:, 0, 0] = 1.
        warp = SimpleNamespace(get_device=lambda name: name,
                               from_torch=lambda value, **kwargs: Array(value),
                               empty=lambda shape, dtype, device: Array(torch.empty((*shape, dtype))),
                               launch=launch, to_torch=lambda value: value.tensor,
                               vec3f=3, quatf=4, mat33f="mat33f")
        expected = torch.diag(torch.tensor([.05, .06, .08]))[None].repeat(2, 1, 1)
        before = expected.clone()
        with patch.dict("sys.modules", {"warp": warp}), \
                patch.object(checks, "_installed_conversion_kernel", return_value=(kernel, {"pinned": True})):
            one, identity = checks._encode_reference_inertia(expected)
            two, _ = checks._encode_reference_inertia(expected * .9)
        self.assertTrue(torch.equal(expected, before))
        self.assertEqual(one.dtype, torch.float64)
        torch.testing.assert_close(one, expected.double(), rtol=0, atol=0)
        torch.testing.assert_close(two, (expected*.9).double(), rtol=0, atol=0)
        self.assertEqual(identity, {"pinned": True})
        self.assertEqual(len(outputs_seen), 4)


if __name__ == "__main__":
    unittest.main()
