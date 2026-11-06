"""Independent source-table and control timing checks for the hw3 port."""
import json
from pathlib import Path
import re
import unittest

import numpy as np
import torch

from supermimic.g1_physics import (ARMATURE, CAPS, ENVELOPES, FRICTION_LEVELS, G1Motor,
                                    action_scale, friction_threshold, gains, motor_of, motor_type)
from supermimic.robot_metadata import G1_JOINT_NAMES


class MotorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(3)
        cls.source = json.loads((Path(__file__).parent / "fixtures/g1_hw3_source.json").read_text())

    def test_every_joint_matches_frozen_handoff_tables(self):
        source = self.source
        self.assertEqual(source["source_commit"], "4aed2bb")
        self.assertEqual(len(G1_JOINT_NAMES), 29)
        for i, name in enumerate(G1_JOINT_NAMES):
            matches = [t for pattern, t in source["G1_JOINT_MOTORS"].items() if re.fullmatch(pattern, name)]
            self.assertEqual(matches, [motor_type(name)])
            self.assertEqual(list(ENVELOPES[matches[0]]), source["MOTORS"][matches[0]][:4])
            self.assertEqual(list(FRICTION_LEVELS[i]), source["G1_FRICTION_LEVELS_HW3"][name])
        self.assertEqual(motor_of("left_hip_pitch_joint"), (139., 20., .025101925))
        self.assertAlmostEqual(gains("left_hip_pitch_joint")[0], 99.098427782326, places=8)
        self.assertAlmostEqual(action_scale("left_hip_pitch_joint"), .25*139/99.098427782326, places=8)

    def test_envelope_against_independent_numpy_reference(self):
        controller = G1Motor(96, randomize=False)
        controller.reset(torch.arange(96), torch.zeros(96, 29))
        controller.delay[:] = 0
        rng = np.random.default_rng(381)
        q, qd, target = [rng.normal(size=(96, 29)).astype(np.float32)*s for s in (1., 25., 3.)]
        # Include near-stall, driving and braking at knee/no-load speeds.
        qd[:5, :] = np.array([0., .009, -.009, 0.02, -.02])[:, None]
        scales = rng.uniform(.9, 1.1, size=(96, 29)).astype(np.float32)
        controller.scale[:] = torch.tensor(scales)
        expected = np.empty_like(q)
        for j, name in enumerate(G1_JOINT_NAMES):
            typ = motor_type(name)
            drive, brake, knee, maximum = self.source["MOTORS"][typ][:4]
            kp, kd = gains(name)
            cap = CAPS[typ][0]
            tau = np.clip(kp*(target[:, j]-q[:, j])-kd*qd[:, j], -cap, cap)
            peak = np.where((tau*qd[:, j] > 0) & (np.abs(qd[:, j]) > .01), drive, brake)
            limit = peak*np.clip((maximum-np.abs(qd[:, j]))/(maximum-knee), 0, 1)*scales[:, j]
            expected[:, j] = np.clip(tau, -limit, limit)
        actual = controller.compute(torch.tensor(target), torch.tensor(q), torch.tensor(qd)).numpy()
        np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=2e-5)
        self.assertTrue((actual[np.abs(qd) > 41] == 0).all())
        # Effort clip MUST happen before torque scale/envelope. At zero speed
        # the 5020 cap is 25 even though 1.1 * braking envelope exceeds it.
        self.assertLessEqual(float(np.abs(actual[:, 15:22]).max()), 25.00001)

    def test_delay_is_shared_by_joints_and_partial_reset_isolated(self):
        c = G1Motor(3, randomize=False)
        zero = torch.zeros(3, 29)
        c.reset(torch.arange(3), zero)
        c.delay[:] = torch.tensor([2, 3, 4])
        target = torch.full_like(zero, .001)
        for tick in range(5):
            effort = c.compute(target, zero, zero)
            active = effort.abs().sum(-1) > 0
            self.assertEqual(active.tolist(), [tick >= 2, tick >= 3, tick >= 4])
        c.reset(torch.tensor([1]), torch.full((1, 29), -.001))
        effort = c.compute(target, zero, zero)
        self.assertTrue(bool((effort[1] < 0).all()))
        self.assertTrue(bool((effort[[0, 2]] > 0).all()))

    def test_reset_randomization_and_startup_friction_lifetime(self):
        torch.manual_seed(23)
        c = G1Motor(128, randomize=True)
        original = c.friction_scale.clone()
        self.assertTrue(bool(((original[:, :12] >= .8) & (original[:, :12] <= 2.5)).all()))
        self.assertTrue(bool(((original[:, 12:] >= .8) & (original[:, 12:] <= 2.)).all()))
        c.reset(torch.arange(128), torch.zeros(128, 29))
        self.assertEqual(set(c.delay.tolist()), {2, 3, 4})
        self.assertTrue(bool(((c.scale >= .9) & (c.scale <= 1.1)).all()))
        scale = c.scale.clone()
        c.reset(torch.tensor([1]), torch.zeros(1, 29))
        torch.testing.assert_close(c.scale[0], scale[0])
        self.assertFalse(torch.equal(c.scale[1], scale[1]))
        torch.testing.assert_close(c.friction_scale, original)

    def test_friction_static_dynamic_and_no_viscous_growth(self):
        scale = torch.ones(1, 29)
        static = friction_threshold(torch.zeros(1, 29), scale)[0]
        positive = friction_threshold(torch.full((1, 29), 1.), scale)[0]
        negative = friction_threshold(torch.full((1, 29), -1.), scale)[0]
        source = self.source["G1_FRICTION_LEVELS_HW3"]
        torch.testing.assert_close(static, torch.tensor([source[n][0] for n in G1_JOINT_NAMES]))
        torch.testing.assert_close(positive, torch.tensor([min(source[n]) for n in G1_JOINT_NAMES]))
        torch.testing.assert_close(positive, negative)
        torch.testing.assert_close(positive*2, friction_threshold(torch.full((1, 29), 50.), scale*2)[0])

    def test_invalid_commands_fail_before_motor_history_changes(self):
        c = G1Motor(1, randomize=False)
        q = torch.zeros(1, 29)
        for bad in [torch.zeros(29), torch.full((1, 29), float("nan"))]:
            with self.assertRaises(ValueError):
                c.compute(bad, q, q)
        self.assertEqual(c.tick, 0)


if __name__ == "__main__":
    unittest.main()
