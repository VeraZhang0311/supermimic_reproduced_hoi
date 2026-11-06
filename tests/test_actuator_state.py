"""Control-boundary actuator sufficiency and differentiable advance tests."""
import unittest

import torch

from supermimic.actuator_state import (ACTUATOR_SCHEMA, ACTUATOR_STATE_DIM,
                                      advance_actuator_state, export_motor_state,
                                      normalize_actuator_state, validate_actuator_state)
from supermimic.g1_physics import (G1Motor, DEFAULT_JOINT_POS, action_scale, friction_threshold)
from supermimic.robot_metadata import G1_JOINT_NAMES


class ActuatorStateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(743)

    def motor(self, n=12):
        motor = G1Motor(n, randomize=True)
        motor.reset(torch.arange(n), torch.randn(n, 29)*.1)
        # Guarantee all permitted delays occur, alongside randomized scales.
        motor.delay[:] = 2 + torch.arange(n) % 3
        return motor

    @staticmethod
    def canonical_motor(context, phase):
        n = len(context)
        motor = G1Motor(n, randomize=False)
        motor.queue[:] = context[:, :29].unsqueeze(0)
        motor.tick = phase
        motor.delay[:] = context[:, 29].long()
        motor.scale[:] = context[:, 30:59]
        motor.friction_scale[:] = context[:, 59:88]
        return motor

    def test_raw_layout_and_export_owns_snapshot(self):
        motor = self.motor()
        context = export_motor_state(motor)
        self.assertEqual(context.shape, (12, ACTUATOR_STATE_DIM))
        self.assertEqual(ACTUATOR_SCHEMA["dim"], 88)
        torch.testing.assert_close(context[:, :29], motor.queue[0])
        torch.testing.assert_close(context[:, 29], motor.delay.float())
        torch.testing.assert_close(context[:, 30:59], motor.scale)
        torch.testing.assert_close(context[:, 59:88], motor.friction_scale)
        saved = context.clone()
        motor.reset(torch.arange(12), torch.full((12, 29), 2.))
        torch.testing.assert_close(context, saved, atol=0, rtol=0)

    def test_all_global_phases_and_random_delays_have_same_next_four_torques(self):
        n = 12
        action = torch.randn(n, 29)*.2
        xi = torch.randn(n, 33)*.01
        for phase in range(5):
            actual = self.motor(n)
            # Reach this phase through real motor ticks and uniform old target.
            old_target = torch.randn(n, 29)*.1
            for _ in range(5 + phase):
                actual.compute(old_target, torch.randn(n, 29)*.1, torch.randn(n, 29))
            self.assertEqual(actual.tick, phase)
            # Even a poisoned unused next-write slot cannot affect the next
            # interval: compute overwrites it before any allowed delayed read.
            actual.queue[actual.tick] = float("nan")
            context = export_motor_state(actual)
            compressed = self.canonical_motor(context, (phase+2) % 5)
            expected_next = advance_actuator_state(context, action, xi)
            target = expected_next[:, :29]
            for tick in range(4):
                q, qd = torch.randn(n, 29)*.15, torch.randn(n, 29)*5
                torch.testing.assert_close(actual.compute(target, q, qd), compressed.compute(target, q, qd), atol=0, rtol=0)
                torch.testing.assert_close(friction_threshold(qd, actual.friction_scale),
                                           friction_threshold(qd, compressed.friction_scale), atol=0, rtol=0)
            torch.testing.assert_close(export_motor_state(actual), expected_next, atol=0, rtol=0)
            torch.testing.assert_close(export_motor_state(compressed), expected_next, atol=0, rtol=0)

    def test_partial_reset_preserves_other_worlds_and_compact_advance_matches(self):
        actual = self.motor()
        q, qd = torch.randn(12, 29)*.1, torch.randn(12, 29)
        previous = torch.randn(12, 29)*.1
        for _ in range(4):
            actual.compute(previous, q, qd)
        before = export_motor_state(actual)
        reset_ids = torch.tensor([1, 5])
        pose = torch.randn(2, 29)*.1
        actual.reset(reset_ids, pose)
        context = export_motor_state(actual)
        others = torch.tensor([i for i in range(12) if i not in reset_ids.tolist()])
        torch.testing.assert_close(context[others], before[others], atol=0, rtol=0)
        torch.testing.assert_close(context[reset_ids, :29], pose, atol=0, rtol=0)
        compressed = self.canonical_motor(context, 0)
        action, xi = torch.randn(12, 29)*.1, torch.randn(12, 33)*.01
        advanced = advance_actuator_state(context, action, xi)
        for _ in range(4):
            torch.testing.assert_close(actual.compute(advanced[:, :29], q, qd),
                                       compressed.compute(advanced[:, :29], q, qd), atol=0, rtol=0)
        torch.testing.assert_close(export_motor_state(actual), advanced, atol=0, rtol=0)

    def test_changed_command_mid_interval_is_rejected(self):
        motor = self.motor()
        context = export_motor_state(motor)
        q, qd = torch.zeros(12, 29), torch.zeros(12, 29)
        command = context[:, :29] + .1
        for _ in range(3):
            motor.compute(command, q, qd)
            with self.assertRaisesRegex(ValueError, "control boundary"):
                export_motor_state(motor)
        motor.compute(command, q, qd)
        torch.testing.assert_close(export_motor_state(motor)[:, :29], command, atol=0, rtol=0)

    def test_different_history_with_identical_body_action_xi_is_observable(self):
        first, second = self.motor(3), self.motor(3)
        second.delay[:] = first.delay
        second.scale[:] = first.scale
        second.friction_scale[:] = first.friction_scale
        first.queue[:] = .01
        second.queue[:] = -.01
        c1, c2 = export_motor_state(first), export_motor_state(second)
        q, qd, new_command = torch.zeros(3, 29), torch.zeros(3, 29), torch.zeros(3, 29)
        tau1, tau2 = first.compute(new_command, q, qd), second.compute(new_command, q, qd)
        self.assertFalse(torch.equal(tau1, tau2))
        self.assertFalse(torch.equal(normalize_actuator_state(c1), normalize_actuator_state(c2)))

    def test_advance_uses_raw_offsets_and_retains_parameters_with_gradients(self):
        context = export_motor_state(self.motor()).requires_grad_()
        action = torch.randn(12, 29, requires_grad=True)
        xi = torch.randn(12, 33, requires_grad=True)
        next_context = advance_actuator_state(context, action, xi)
        # Compare the target with the production motor's public contract.
        import re
        default = torch.tensor([next((v for pat, v in DEFAULT_JOINT_POS.items() if re.fullmatch(pat, n)), 0.) for n in G1_JOINT_NAMES])
        alpha = torch.tensor([action_scale(n) for n in G1_JOINT_NAMES])
        torch.testing.assert_close(next_context[:, :29], default + xi[:, 1:30] + alpha*action)
        torch.testing.assert_close(next_context[:, 29:], context[:, 29:], atol=0, rtol=0)
        next_context.sum().backward()
        torch.testing.assert_close(action.grad, alpha.expand_as(action))
        torch.testing.assert_close(xi.grad[:, 1:30], torch.ones(12, 29))
        torch.testing.assert_close(xi.grad[:, [0, 30, 31, 32]], torch.zeros(12, 4))
        torch.testing.assert_close(context.grad[:, :29], torch.zeros(12, 29))
        torch.testing.assert_close(context.grad[:, 29:], torch.ones(12, 59))

    def test_fixed_normalization_supports_sequences_without_clipping_or_detach(self):
        context = export_motor_state(self.motor(6)).reshape(2, 3, 88).requires_grad_()
        normalized = normalize_actuator_state(context)
        self.assertEqual(normalized.shape, (2, 3, 88))
        self.assertTrue(torch.equal(normalized[..., 29], context[..., 29]-3))
        self.assertTrue(bool((normalized[..., 30:] >= -1.00001).all()))
        self.assertTrue(bool((normalized[..., 30:] <= 1.00001).all()))
        normalized.sum().backward()
        self.assertTrue(bool(torch.isfinite(context.grad).all()))
        self.assertTrue(bool((context.grad[..., :29] != 0).all()))
        large = context.detach().clone()
        large[..., :29] += 100
        self.assertGreater(float(normalize_actuator_state(large)[..., :29].abs().max()), 100)
        action, xi = torch.zeros(2, 3, 29), torch.zeros(2, 3, 33)
        self.assertEqual(advance_actuator_state(context, action, xi).shape, context.shape)

    def test_invalid_raw_state_and_shapes_fail_explicitly(self):
        context = export_motor_state(self.motor())
        for index, value in [(0, float("nan")), (29, 2.5), (29, 5), (30, 1.2), (59, .7), (87, 2.2)]:
            bad = context.clone(); bad[0, index] = value
            with self.assertRaises(ValueError):
                validate_actuator_state(bad)
        with self.assertRaises(ValueError):
            normalize_actuator_state(torch.zeros(12, 87))
        with self.assertRaises(TypeError):
            normalize_actuator_state(torch.zeros(12, 88, dtype=torch.long))
        with self.assertRaises(ValueError):
            advance_actuator_state(context, torch.zeros(12, 29), torch.zeros(12, 32))
        invalid_delay = self.motor(); invalid_delay.delay[:] = 0
        with self.assertRaisesRegex(ValueError, "delay"):
            export_motor_state(invalid_delay)


if __name__ == "__main__":
    unittest.main()
