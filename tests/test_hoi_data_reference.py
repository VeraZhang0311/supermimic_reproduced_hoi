"""CPU gates for HOI v1 schema, shared clock and source/derived velocities."""

import math
import unittest

import torch

from hoi.data.reference import adapt_motion, validate_motion
from hoi.data.timebase import CanonicalTimebase


JOINTS = [f"joint_{index}" for index in range(29)]


def motion(states=13, dtype=torch.float64):
    value = torch.zeros(states, 630, dtype=dtype)
    value[:, 6] = 1
    value[:, 77] = 1
    value[:, 201:357].reshape(states, 39, 4)[:, :, 3] = 1
    return value


class TimebaseTest(unittest.TestCase):
    def test_601_states_are_exactly_500_actions_and_501_states(self):
        clock = CanonicalTimebase()
        self.assertEqual(clock.control_steps(601), 500)
        self.assertEqual(clock.state_frames(601), 501)
        grid = clock.sample_grid(601)
        self.assertEqual(grid["time"].dtype, torch.float64)
        self.assertEqual(float(grid["time"][-1]), 10.0)
        self.assertEqual(int(grid["i0"][-1]), 600)
        self.assertEqual(int(grid["i1"][-1]), 600)
        self.assertEqual(float(grid["alpha"][-1]), 0.0)
        torch.testing.assert_close(grid["time"], clock.time_grid(601), rtol=0, atol=0)
        torch.testing.assert_close(grid["source_index"][:6], torch.tensor([0, 1.2, 2.4, 3.6, 4.8, 6], dtype=torch.float64))

    def test_integer_counts_and_tail_for_every_small_remainder(self):
        clock = CanonicalTimebase()
        for states in range(1, 611):
            with self.subTest(states=states):
                steps = 5 * (states - 1) // 6
                self.assertEqual(clock.control_steps(states), steps)
                grid = clock.sample_grid(states)
                self.assertEqual(len(grid["time"]), steps + 1)
                self.assertLessEqual(float(grid["source_index"][-1]), states - 1)
                self.assertGreaterEqual((states - 1) / 60 - float(grid["time"][-1]), -1e-14)
                self.assertLess((states - 1) / 60 - float(grid["time"][-1]), .02 + 1e-14)
                torch.testing.assert_close(grid["source_index"], grid["time"] * 60, rtol=0, atol=1e-12)

    def test_invalid_frame_counts(self):
        for states in (True, 2.0, "2", None):
            with self.subTest(states=states), self.assertRaises(TypeError):
                CanonicalTimebase().control_steps(states)
        for states in (0, -1):
            with self.subTest(states=states), self.assertRaises(ValueError):
                CanonicalTimebase().time_grid(states)

    def test_metadata_describes_protocol_assumption(self):
        metadata = CanonicalTimebase().metadata()
        self.assertEqual(metadata["source_timebase_status"], "protocol_assumed_not_generation_verified")
        self.assertEqual(metadata["source_fps_assumed"], 60.0)
        self.assertEqual(metadata["control_dt_s"], .02)
        self.assertNotIn("physics_dt_s", metadata)


class ReferenceAdapterTest(unittest.TestCase):
    def test_shared_clock_and_all_continuous_fields(self):
        source = motion(13)
        times = torch.arange(13, dtype=torch.float64) / 60
        source[:, 0] = 2 * times
        source[:, 71] = 3 * times
        source[:, 7:13] = torch.arange(6, dtype=torch.float64) + times[:, None] * 7
        source[:, 78:84] = torch.arange(6, dtype=torch.float64) + times[:, None] * 11
        source[:, 84:201] = torch.arange(117, dtype=torch.float64) + times[:, None] * 13
        source[:, 357:591] = torch.arange(234, dtype=torch.float64) + times[:, None] * 17
        original = source.clone()
        result = adapt_motion(source, JOINTS, JOINTS)
        time = result["time"]
        torch.testing.assert_close(result["robot"]["root_pos"][:, 0], 2 * time)
        torch.testing.assert_close(result["object"]["pos"][:, 0], 3 * time)
        torch.testing.assert_close(result["robot"]["source_v"], torch.arange(3, dtype=torch.float64) + time[:, None] * 7)
        torch.testing.assert_close(result["object"]["source_omega"], torch.arange(3, 6, dtype=torch.float64) + time[:, None] * 11)
        torch.testing.assert_close(result["body"]["pos"].flatten(1), torch.arange(117, dtype=torch.float64) + time[:, None] * 13)
        torch.testing.assert_close(result["body"]["source_v"].flatten(1), torch.arange(117, dtype=torch.float64) + time[:, None] * 17)
        torch.testing.assert_close(result["body"]["source_omega"].flatten(1), torch.arange(117, 234, dtype=torch.float64) + time[:, None] * 17)
        torch.testing.assert_close(source, original, rtol=0, atol=0)
        self.assertEqual(result["metadata"]["control_steps"], 10)
        self.assertEqual(result["metadata"]["state_frames"], 11)

    def test_source_velocities_preserved_and_derived_velocities_separate(self):
        source = motion(13)
        time = torch.arange(13, dtype=torch.float64) / 60
        source[:, 0] = 2 * time
        source[:, 71] = -3 * time
        source[:, 13:42] = time[:, None] * torch.arange(1, 30, dtype=torch.float64)
        source[:, 7:13] = 99
        source[:, 42:71] = 88
        source[:, 78:84] = 77
        result = adapt_motion(source, JOINTS, JOINTS)
        robot, obj = result["robot"], result["object"]
        torch.testing.assert_close(robot["derived_ref_v"][:, 0], torch.full((11,), 2., dtype=torch.float64))
        torch.testing.assert_close(obj["derived_ref_v"][:, 0], torch.full((11,), -3., dtype=torch.float64))
        torch.testing.assert_close(robot["derived_ref_joint_qd"], torch.arange(1, 30, dtype=torch.float64).expand(11, 29))
        torch.testing.assert_close(robot["source_v"], torch.full((11, 3), 99., dtype=torch.float64))
        torch.testing.assert_close(robot["source_joint_qd"], torch.full((11, 29), 88., dtype=torch.float64))
        torch.testing.assert_close(obj["source_v"], torch.full((11, 3), 77., dtype=torch.float64))
        torch.testing.assert_close(robot["derived_ref_omega"], torch.zeros(11, 3, dtype=torch.float64))

    def test_derived_velocity_uses_central_and_one_sided_boundary_rules(self):
        # Source pose is sampled piecewise linearly; differentiate the sampled
        # trajectory rather than the analytic parabola or source velocities.
        source = motion(11)
        source[:, 0] = (torch.arange(11, dtype=torch.float64) / 60) ** 2
        result = adapt_motion(source, JOINTS, JOINTS)
        position = result["robot"]["root_pos"]
        velocity = result["robot"]["derived_ref_v"]
        torch.testing.assert_close(velocity[1:-1], (position[2:] - position[:-2]) / .04)
        torch.testing.assert_close(velocity[0], (position[1] - position[0]) / .02)
        torch.testing.assert_close(velocity[-1], (position[-1] - position[-2]) / .02)

    def test_world_angular_velocity_with_noncommuting_initial_rotation(self):
        source = motion(61)
        time = torch.arange(61, dtype=torch.float64) / 60
        speed, tilt = 1.3, .8
        # xyzw of Rz(speed * t) @ Rx(tilt): angular velocity is world z.
        z = speed * time / 2
        quaternion = torch.stack((math.sin(tilt / 2) * z.cos(), math.sin(tilt / 2) * z.sin(), math.cos(tilt / 2) * z.sin(), math.cos(tilt / 2) * z.cos()), -1)
        source[:, 3:7] = quaternion
        source[:, 74:78] = quaternion
        result = adapt_motion(source, JOINTS, JOINTS)
        expected = torch.tensor([0., 0., speed], dtype=torch.float64).expand(51, 3)
        torch.testing.assert_close(result["robot"]["derived_ref_omega"], expected, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(result["object"]["derived_ref_omega"], expected, atol=1e-8, rtol=1e-8)

    def test_slerp_and_antipodal_quaternions_for_root_object_and_bodies(self):
        source = motion(7)
        theta = 2 * math.pi / 3
        rotated = torch.tensor([0., 0., math.sin(theta / 2), math.cos(theta / 2)], dtype=torch.float64)
        for start, stop in ((3, 7), (74, 78), (201, 357)):
            source[2:, start:stop].reshape(5, -1, 4)[:] = rotated
            source[3, start:stop] *= -1
        result = adapt_motion(source, JOINTS, JOINTS)
        expected = torch.tensor([0., 0., math.sin(theta * .2 / 2), math.cos(theta * .2 / 2)], dtype=torch.float64)
        for quaternion in (result["robot"]["root_quat_xyzw"], result["object"]["quat_xyzw"], result["body"]["quat_xyzw"][:, 0]):
            torch.testing.assert_close(quaternion[1], expected, atol=1e-10, rtol=1e-10)
            torch.testing.assert_close(quaternion.norm(dim=-1), torch.ones(6, dtype=torch.float64))
            torch.testing.assert_close(quaternion[3].abs(), rotated.abs())

    def test_contacts_are_nearest_source_flags_not_linear_or_pair_truth(self):
        source = motion(7)
        source[:, 591:630] = (torch.arange(7) % 2)[:, None]
        result = adapt_motion(source, JOINTS, JOINTS)
        self.assertEqual(result["nearest"].tolist(), [0, 1, 2, 4, 5, 6])
        torch.testing.assert_close(result["body"]["contact_flags"], source[result["nearest"], 591:630])
        self.assertEqual(result["metadata"]["source_contact_semantics"], "source_body_flags_not_hand_object_pair_truth")

    def test_joint_names_reorder_positions_and_source_qd_only_in_robot_block(self):
        source = motion(3)
        source[:, 13:42] = torch.arange(29, dtype=torch.float64)
        source[:, 42:71] = torch.arange(29, dtype=torch.float64) + 100
        result = adapt_motion(source, JOINTS, JOINTS[::-1])
        torch.testing.assert_close(result["robot"]["joint_pos"], torch.arange(28, -1, -1, dtype=torch.float64).expand(2, 29))
        torch.testing.assert_close(result["robot"]["source_joint_qd"], (torch.arange(28, -1, -1, dtype=torch.float64) + 100).expand(2, 29))
        torch.testing.assert_close(result["source_sampled"][:, 13:42], source[:2, 13:42])
        self.assertEqual(result["metadata"]["joint_order"], list(range(28, -1, -1)))

    def test_short_clips_rejected_instead_of_fabricating_velocity(self):
        for count in (1, 2):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "fewer than two control-grid states"):
                adapt_motion(motion(count), JOINTS, JOINTS)
        self.assertEqual(adapt_motion(motion(3), JOINTS, JOINTS)["metadata"]["state_frames"], 2)

    def test_metadata_includes_exact_tail_and_provenance(self):
        result = adapt_motion(motion(8), JOINTS, JOINTS)
        metadata = result["metadata"]
        self.assertEqual(metadata["source_frames"], 8)
        self.assertEqual(metadata["control_steps"], 5)
        self.assertEqual(metadata["state_frames"], 6)
        self.assertEqual(metadata["canonical_duration_s"], 7 / 60)
        self.assertEqual(metadata["last_control_time_s"], .1)
        self.assertEqual(metadata["tail_remainder_s"], 1 / 60)
        self.assertEqual(metadata["source_velocity_reference_point_status"], "unverified")
        self.assertEqual(metadata["robot_fk_status"], "not_computed_by_data_adapter")
        self.assertEqual(metadata["object_pose_origin_native_alignment_status"], "unverified")

    def test_bad_schema_finite_quaternions_and_contacts_rejected(self):
        for value in ({"frames": motion()}, [motion()], torch.zeros(3, 630, dtype=torch.int64), torch.zeros(3, 629), torch.zeros(0, 630)):
            with self.subTest(value_type=type(value).__name__), self.assertRaises((TypeError, ValueError)):
                validate_motion(value)
        for column, value in ((0, float("nan")), (13, float("inf")), (6, 0.), (77, 2.), (204, 0.), (591, .5), (592, -1.)):
            source = motion()
            source[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                validate_motion(source)

    def test_invalid_joint_names_rejected(self):
        bad = (JOINTS[:-1], JOINTS[:-1] + [JOINTS[0]], JOINTS[:-1] + ["unknown"], JOINTS[:-1] + [""], "joints")
        for names in bad:
            with self.subTest(names=names), self.assertRaises((TypeError, ValueError)):
                adapt_motion(motion(), names, JOINTS)
            with self.subTest(target=names), self.assertRaises((TypeError, ValueError)):
                adapt_motion(motion(), JOINTS, names)

    def test_float32_and_float16_inputs_keep_clock_float64(self):
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                result = adapt_motion(motion(dtype=dtype), JOINTS, JOINTS)
                self.assertEqual(result["time"].dtype, torch.float64)
                self.assertEqual(result["alpha"].dtype, torch.float64)
                self.assertEqual(result["source_sampled"].dtype, torch.float32)
                self.assertTrue(bool(torch.isfinite(result["robot"]["derived_ref_omega"]).all()))


if __name__ == "__main__":
    unittest.main()
