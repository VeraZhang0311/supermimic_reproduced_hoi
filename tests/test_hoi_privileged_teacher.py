"""CPU software/gradient contracts; these are not native physics acceptance."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from hoi.ultra_kinematics import UltraKinematics
from hoi.v1.privileged import (TeacherKinematics, PrivilegedFeatures, TEACHER_PROFILE,
                               PRIVILEGED_DIM, tan_norm)
from hoi.v1.networks import Policy, WorldModel, Critic, frozen_parameters
from hoi.v1.learner import Learner
from hoi.v1.checkpoint import make_checkpoint, load_checkpoint, validate_checkpoint_features
from hoi.v1.state import unflatten_state, unflatten_object
from supermimic.rotations import mat_to_quat, quat_to_mat, axis_angle_to_mat
from tests.test_hoi_v1_learner import make_learner, transition, fill


def fixture_virtual():
    """39 named links with 29 revolute joints and nine fixed links."""
    names = [f"body_{i}" for i in range(39)]
    joints = [f"joint_{i}" for i in range(29)]
    xml = ['<robot name="fixture">']
    for name in names:
        xml.append(f'<link name="{name}"><inertial><origin xyz="0.01 0.02 0.03"/></inertial></link>')
    for i in range(1, 39):
        typ, name = ("revolute", joints[i-1]) if i <= 29 else ("fixed", f"fixed_{i}")
        parent = max(0, i - 3)
        xml.append(f'<joint name="{name}" type="{typ}"><parent link="{names[parent]}"/>'
                   f'<child link="{names[i]}"/><origin xyz="0.02 0.01 0.03" rpy="0.1 0 0"/>'
                   '<axis xyz="0 1 0"/></joint>')
    xml.append('</robot>')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fixture.urdf"
        path.write_text("".join(xml), encoding="utf-8")
        return UltraKinematics(path, names, joints), joints


def teacher_learner(horizon=3):
    base = make_learner(horizon=horizon)
    virtual, joints = fixture_virtual()
    base.mdp.privileged_features = PrivilegedFeatures(TeacherKinematics(virtual, joints), base.motion, base.shapes)
    pi = Policy(hidden=(32, 16), object_hidden=(16,), actor_observation_profile=TEACHER_PROFILE)
    return Learner(pi, base.wm, base.critic, base.motion, base.mdp, base.shapes, base.buffer,
                   replace(base.config, actor_observation_profile=TEACHER_PROFILE))


class PrivilegedTeacherTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(271)

    def test_matrix_fk_matches_independent_quaternion_fk_and_com_velocities(self):
        virtual, joints = fixture_virtual()
        kin = TeacherKinematics(virtual, list(reversed(joints)))
        p, v, w = [torch.randn(3, 3) for _ in range(3)]
        q, qd = torch.randn(3, 29) * .3, torch.randn(3, 29)
        R = axis_angle_to_mat(torch.tensor([0., 0., 1.]), torch.tensor([.2, -.9, 1.8]))
        actual = kin.forward((p, R, v, w, q, qd))
        quat = mat_to_quat(R)
        root = torch.cat((p, quat[:, [1, 2, 3, 0]], v, w), -1)
        expected = virtual.forward(root, q[:, kin.order], qd[:, kin.order], root_velocity_is_com=False)
        expected_R = quat_to_mat(expected["body_rot"][..., [3, 0, 1, 2]])
        for got, want in zip(actual, (expected["body_pos"], expected_R, expected["body_vel"], expected["body_ang_vel"])):
            torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-5)

    def test_ultra_rotation_uses_x_z_columns(self):
        R = axis_angle_to_mat(torch.tensor([0., 0., 1.]), torch.tensor(.7))
        torch.testing.assert_close(tan_norm(R), torch.cat((R @ torch.tensor([1., 0., 0.]),
                                                          R @ torch.tensor([0., 0., 1.]))))

    def test_features_cover_object_velocity_context_and_have_finite_state_gradients(self):
        learner = teacher_learner()
        sample = transition(learner, [10, 70], 0)
        S = tuple(x.detach().requires_grad_() for x in unflatten_state(sample["s"]))
        O = tuple(x.detach().requires_grad_() for x in unflatten_object(sample["o"]))
        build = learner.mdp.privileged_features
        feature = build(S, O, sample["f"], sample["a_prev"], sample["xi"], sample["xi_o"], S)
        self.assertEqual(feature.shape, (2, PRIVILEGED_DIM))
        changed_context = sample["xi_o"].clone()
        changed_context[:, 0] += 2.
        changed = build(S, O, sample["f"], sample["a_prev"], sample["xi"], changed_context, S)
        self.assertGreater(float((changed - feature).detach().abs().sum()), 0.)
        for grad in torch.autograd.grad((feature * torch.randn_like(feature)).sum(), (*S, *O)):
            self.assertTrue(bool(torch.isfinite(grad).all()))
            self.assertGreater(float(grad.abs().sum()), 0.)

    def test_end_of_reference_masks_contact_intent_and_never_reads_next_clip(self):
        learner = teacher_learner()
        sample = transition(learner, [179, 359], 0)
        S, O = unflatten_state(sample["s"]), unflatten_object(sample["o"])
        build = lambda: learner.mdp.privileged_features(S, O, sample["f"], sample["a_prev"], sample["xi"], sample["xi_o"], S)
        features = build()
        self.assertTrue(bool((features[:, -8:] == 0).all()))
        learner.motion.q[180:195] += 10
        learner.motion.object_p[180:195] += 10
        torch.testing.assert_close(build(), features)

    def test_teacher_bypasses_sensor_noise_and_only_real_collection_updates_stats(self):
        learner = teacher_learner()
        sample = transition(learner, [60, 80], 0)
        with patch.object(learner.mdp, "add_noise", side_effect=AssertionError("teacher must be noise-free")):
            action = learner.action(sample, exploration=False)
            self.assertTrue(bool(torch.isfinite(action).all()))
            before = deepcopy(learner.pi.privileged_norm.state_dict())
            learner.action(sample, exploration=False, update_statistics=False)
            fill(learner)
            start, history = learner.buffer.sample_starts(2, 10, horizon=3, motion=learner.motion)
            learner.imagine(start, history)
        for name, value in before.items():
            torch.testing.assert_close(value, learner.pi.privileged_norm.state_dict()[name])

    def test_h32_actor_gradient_through_full_body_features_without_wm_parameter_gradients(self):
        learner = teacher_learner(horizon=32)
        fill(learner)
        start, history = learner.buffer.sample_starts(2, 10, horizon=32, motion=learner.motion)
        with frozen_parameters(learner.wm, learner.critic):
            result = learner.imagine(start, history)
            (-result["objective"]).backward()
        self.assertEqual(len(result["actions"]), 32)
        grads = [p.grad for p in learner.pi.parameters() if p.grad is not None]
        self.assertTrue(grads and all(bool(torch.isfinite(g).all()) for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)
        self.assertTrue(all(p.grad is None for p in learner.wm.parameters()))
        self.assertTrue(all(p.grad is None for p in learner.critic.parameters()))

    def test_profile_checkpoint_roundtrip_and_cross_profile_rejection(self):
        learner = teacher_learner()
        settings = {"hand_collision": True, "self_collision": True}
        identity = {"fixture": "synthetic_only"}
        checkpoint = make_checkpoint(learner.pi, learner.wm, learner.critic, learner.optimizers,
                                     iteration=0, physics=settings, data_identity=identity,
                                     initialization={"mode": "test"}, extra={"config": learner.config.to_dict()})
        validate_checkpoint_features(checkpoint)
        load_checkpoint(checkpoint, learner.pi, learner.wm, learner.critic,
                        physics=settings, data_identity=identity)
        baseline = make_learner()
        with self.assertRaisesRegex(ValueError, "actor observation profile"):
            load_checkpoint(checkpoint, baseline.pi, baseline.wm, baseline.critic,
                            physics=settings, data_identity=identity)
        bad = deepcopy(checkpoint)
        bad["extra"]["config"]["actor_observation_profile"] = "v1"
        with self.assertRaisesRegex(ValueError, "profile.*disagree"):
            validate_checkpoint_features(bad)

    def test_teacher_config_is_explicit_formal_observation_profile(self):
        from hoi.v1.train import read_config
        path = Path(__file__).resolve().parents[1] / "configs/hoi_v1_privileged_teacher.json"
        config, deviations = read_config(path)
        self.assertEqual(config.actor_observation_profile, TEACHER_PROFILE)
        self.assertEqual(config.experiment_kind, "formal")
        self.assertEqual(deviations, {})

    def test_frozen_evaluation_uses_teacher_inputs_and_preserves_all_statistics(self):
        from tests.test_hoi_v1_evaluate import fixture, SyntheticWorkers
        from hoi.v1.evaluate import evaluate_group
        _, motion, mdp, shapes, fk, metadata = fixture()
        virtual, joints = fixture_virtual()
        mdp.privileged_features = PrivilegedFeatures(TeacherKinematics(virtual, joints), motion, shapes)
        pi = Policy(hidden=(32, 16), object_hidden=(16,), actor_observation_profile=TEACHER_PROFILE)
        before = deepcopy(pi.state_dict())
        workers = SyntheticWorkers(motion, mdp)
        with tempfile.TemporaryDirectory() as directory:
            cases = evaluate_group(pi, motion, mdp, shapes, workers, [0, 1], fk39=fk,
                                   joint_names=fk.dof_names, output=Path(directory) / "teacher",
                                   metadata=metadata, verify_tracked_fk=False)
        self.assertEqual(len(cases), 2)
        self.assertEqual(workers.steps, 6)
        self.assertEqual(workers.reset_count, 1)
        for key, value in before.items():
            torch.testing.assert_close(pi.state_dict()[key], value)


if __name__ == "__main__":
    unittest.main()
