"""CPU observation, BPTT and DR contracts; no native-physics evidence."""
from copy import deepcopy
from contextlib import redirect_stdout
from dataclasses import replace
from enum import IntFlag
import json
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from supermimic.kinematics import G1Kinematics
from supermimic.rotations import axis_angle_to_mat, so3_exp, yaw_mat
from hoi.v1.buffer import ReplayBuffer
from hoi.v1.checkpoint import make_checkpoint, load_checkpoint, validate_checkpoint_features
from hoi.v1.config import TrainConfig
from hoi.v1.learner import Learner
from hoi.v1.mdp import HOIMDP
from hoi.v1.native import physical_context
from hoi.v1.networks import Policy, WorldModel, Critic, frozen_parameters
from hoi.v1.observations import ActorObservations, MATCHING, FUTURE_STEPS, rotation_xz
from hoi.v1.randomization import (MATCHING_DR, NativeRandomization, randomization_contract,
                                 randomization_enabled, sample_mass, sample_object_properties, validate_dr_evidence)
from hoi.v1.train import read_config
from test_hoi_v1_learner import reference_fixture, fill, transition


class ChainKinematics(G1Kinematics):
    """Explicit synthetic waist chain, using production FK methods, not G1 assets."""
    def __init__(self):
        self.body_names = ["pelvis", "waist_yaw_link", "waist_roll_link", "torso_link"] + [f"test_body_{i}" for i in range(10)]
        self.parent = [-1, 0, 1, 2] + [3] * 10
        self.joint_idx = [-1, 12, 13, 14] + list(range(10))
        self.axes = torch.tensor([[0., 0., 1.], [0., 0., 1.], [1., 0., 0.], [0., 1., 0.]] + [[0., 1., 0.]] * 10)
        self.off_p = torch.zeros(14, 3)
        self.off_p[1:4, 2] = torch.tensor([.08, .03, .07])
        self.off_p[4:, 0] = torch.linspace(-.2, .2, 10)
        self.off_R = torch.eye(3).expand(14, 3, 3).clone()
        self.tracked_idx = list(range(14))
        self.num_bodies = 14
        self.q_lo, self.q_hi = torch.full((29,), -2.), torch.full((29,), 2.)


def matching_learner():
    motion, _, shapes = reference_fixture()
    kin = ChainKinematics()
    motion.q[:, 12:15] = torch.tensor([.3, .2, -.1])
    motion.qd[:, 12:15] = torch.tensor([.4, -.5, .6])
    motion.body_p, motion.body_R, motion.body_v, motion.body_w = kin.forward(*motion.ref_arrays())
    mdp = HOIMDP(kin, motion, torch.zeros(29), shapes)
    config = TrainConfig(num_envs=2, buffer_steps=64, horizon=3, wm_window=8, wm_batches=1, wm_batch=4,
                         pi_batch=2, critic_epochs=1, critic_minibatches=2, experiment_kind="smoke", amp="none",
                         observation_profile=MATCHING, domain_randomization_profile=MATCHING_DR)
    pi = Policy(hidden=(32, 16), object_hidden=(16,), observation_profile=MATCHING)
    wm = WorldModel(hidden=(32, 32), contact_hidden=12, torso_mass_context=True)
    with torch.no_grad():
        wm.net[-1].weight.normal_(std=.005)
    return Learner(pi, wm, Critic(hidden=(32, 16), torso_mass_context=True), motion, mdp, shapes,
                   ReplayBuffer(64, 2, torso_mass_context=True), config)


class ObservationMatchingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(192)
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        self.learner = matching_learner()
        self.adapter = self.learner.actor_obs
        self.f = torch.tensor([8, 80])
        self.S = tuple(x.clone() for x in self.learner.motion.ref_state(self.f))
        self.O = tuple(x.clone() for x in self.learner.motion.ref_object(self.f))

    def test_named_torso_pose_matches_full_fk_and_analytic_waist_rotation(self):
        kin = self.learner.mdp.kin
        S = list(self.S)
        S[1] = so3_exp(torch.tensor([[.2, -.1, .4], [-.3, .2, -.1]]))
        S[4] = S[4].detach().requires_grad_()
        p, R = kin.body_pose("torso_link", S[0], S[1], S[4])
        full = kin.forward(*S)
        torch.testing.assert_close(p, full[0][:, 3])
        torch.testing.assert_close(R, full[1][:, 3])
        expected = S[1]
        for i in (1, 2, 3):
            expected = expected @ axis_angle_to_mat(kin.axes[i], S[4][:, kin.joint_idx[i]])
        torch.testing.assert_close(R, expected)
        grad = torch.autograd.grad(R[..., 0, 2].sum() + p.sum(), S[4])[0]
        self.assertTrue(bool(torch.isfinite(grad).all()))
        self.assertGreater(float(grad[:, 12:15].abs().sum()), 0.)

    def test_fields_dimensions_full_torso_rotation_and_existing_velocity_scale(self):
        previous, xi = torch.randn(2, 29), torch.zeros(2, 33)
        robot = self.adapter.robot(self.S, previous, xi, self.f)
        self.assertEqual(robot.shape, (2, 93))
        torch.testing.assert_close(robot[:, 3:32], self.S[4])
        torch.testing.assert_close(robot[:, 32:61], self.S[5])
        torch.testing.assert_close(robot[:, 61:90], previous)
        hg = self.adapter.human_goal(self.S, self.f).reshape(2, 7, 64)
        future = self.adapter.future_frames(self.f)
        torso = self.adapter.torso_pose(self.S)[1]
        reference = self.learner.motion.body_R[future, 3]
        torch.testing.assert_close(hg[..., :6], rotation_xz(torso.transpose(-1, -2)[:, None] @ reference))
        torch.testing.assert_close(hg[..., 6:35], self.learner.motion.qd[future] * .1)
        torch.testing.assert_close(hg[..., 35:], self.learner.motion.q[future])
        self.assertFalse(torch.allclose(hg[..., :6], rotation_xz(yaw_mat(torso).transpose(-1, -2)[:, None] @ reference)))
        self.assertEqual(self.adapter.object(self.S, self.O, self.f).shape, (2, 9))
        self.assertEqual(self.adapter.object_goal(self.S, self.O, self.f).shape, (2, 63))
        self.assertEqual(self.learner.pi.dec[0].in_features, 1148)
        torch.testing.assert_close(rotation_xz(torch.eye(3)), torch.tensor([1., 0., 0., 0., 0., 1.]))

    def test_object_uses_torso_origin_heading_and_no_velocity_or_target_error(self):
        p, R = self.adapter.torso_pose(self.S)
        Ht = yaw_mat(R).transpose(-1, -2)
        value = self.adapter.object(self.S, self.O, self.f)
        torch.testing.assert_close(value[:, :3], (Ht @ (self.O[0] - p).unsqueeze(-1)).squeeze(-1))
        changed_object = (self.O[0], self.O[1], self.O[2] + 100, self.O[3] - 100)
        torch.testing.assert_close(value, self.adapter.object(self.S, changed_object, self.f))
        motion = self.learner.motion
        saved = motion.object_p.clone()
        before_goal = self.adapter.object_goal(self.S, self.O, self.f)
        motion.object_p += .3
        torch.testing.assert_close(value, self.adapter.object(self.S, self.O, self.f))
        self.assertFalse(torch.equal(before_goal, self.adapter.object_goal(self.S, self.O, self.f)))
        motion.object_p.copy_(saved)

    def test_state_conditioned_goal_jacobians_match_finite_differences(self):
        q = self.S[4].detach().clone().requires_grad_()
        state = (*self.S[:4], q, self.S[5])
        weights = torch.linspace(-.7, .9, 63).expand(2, -1)
        score = (self.adapter.object_goal(state, self.O, self.f) * weights).sum()
        gradient = torch.autograd.grad(score, q)[0]
        self.assertGreater(float(gradient[:, 12:15].abs().sum()), 0.)
        direction = gradient / gradient.norm()
        eps = .001
        values = []
        for sign in (1, -1):
            shifted = (*self.S[:4], q.detach() + sign * eps * direction, self.S[5])
            values.append((self.adapter.object_goal(shifted, self.O, self.f) * weights).sum())
        torch.testing.assert_close((gradient * direction).sum(), (values[0] - values[1]) / (2 * eps), atol=.002, rtol=.003)

    def test_shared_seven_times_and_endpoint_repeat_without_masks_or_clip_crossing(self):
        motion = self.learner.motion
        f = motion.clip_end - 1
        S, O = motion.ref_state(f), motion.ref_object(f)
        frames = self.adapter.future_frames(f)
        torch.testing.assert_close(frames, motion.clip_end[:, None].expand(-1, 7))
        self.assertEqual(FUTURE_STEPS, (1, 2, 5, 7, 12, 18, 25))
        for value, dim in ((self.adapter.human_goal(S, f), 64), (self.adapter.object_goal(S, O, f), 9)):
            value = value.reshape(2, 7, dim)
            torch.testing.assert_close(value, value[:, :1].expand_as(value))
        self.assertEqual(self.adapter.object_goal(S, O, f).numel(), 2 * 7 * 9)

    def test_normalization_and_original_reward_critic_inputs_are_separate(self):
        pi, mdp, motion = self.learner.pi, self.learner.mdp, self.learner.motion
        raw = torch.randn(2, 10, 9) * 100
        torch.testing.assert_close(pi.object_obs_norm(raw), raw / (1 + .01 ** 2) ** .5)
        self.assertGreater(float(pi.object_obs_norm(raw).abs().max()), 5.)
        torch.testing.assert_close(pi.obs_norm(torch.full((2, 93), 100.)), torch.full((2, 93), 5.))
        legacy_goal = motion.goal(self.f, 10).clone()
        critic = self.learner._critic_args(self.S, self.O, self.f, torch.zeros(2, 33), torch.ones(2, 12), motion.asset_ids[self.f],
                                          mass_context=torch.zeros(2, 1))
        self.assertEqual((critic[0].shape[-1], critic[1].shape[-1]), (914, 190))
        self.adapter.human_goal(self.S, self.f)
        torch.testing.assert_close(motion.goal(self.f, 10), legacy_goal, atol=0, rtol=0)
        baseline = ActorObservations(mdp)
        torch.testing.assert_close(baseline.human_goal(self.S, self.f), legacy_goal, atol=0, rtol=0)

    def test_collection_and_replay_repeat_first_frames(self):
        learner = self.learner
        snapshot = transition(learner, [8, 80], 0)
        learner.action(snapshot, exploration=False)
        torch.testing.assert_close(learner.robot_history, learner.robot_history[:, -1:].expand_as(learner.robot_history))
        torch.testing.assert_close(learner.object_history, learner.object_history[:, -1:].expand_as(learner.object_history))
        self.assertEqual(learner.object_history.shape, (2, 10, 9))
        fill(learner, steps=8)
        _, history = learner.buffer.sample_starts(2, 10, horizon=3, motion=learner.motion)
        torch.testing.assert_close(history["s"][:, 0], history["s"][:, 1])

    def test_bptt_and_finite_difference_preserve_frozen_wm_critic_and_recompute_goals(self):
        learner = self.learner
        fill(learner)
        start, history = learner.buffer.sample_starts(2, 10, horizon=3, motion=learner.motion)
        noise = [torch.zeros(2, 10, 93)] + [torch.zeros(2, 93) for _ in range(3)]
        snapshots = [deepcopy(m.state_dict()) for m in (learner.wm, learner.critic)]
        with frozen_parameters(learner.wm, learner.critic):
            with patch.object(learner.actor_obs, "human_goal", wraps=learner.actor_obs.human_goal) as human, patch.object(
                    learner.actor_obs, "object_goal", wraps=learner.actor_obs.object_goal) as obj:
                loss = -learner.imagine(start, history, fixed_robot_noise=noise)["objective"]
                loss.backward()
                self.assertEqual((human.call_count, obj.call_count), (3, 3))
            for module in (learner.pi.enc, learner.pi.object_enc, learner.pi.dec, learner.pi.shape_encoder):
                grads = [p.grad for p in module.parameters() if p.grad is not None]
                self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))
                self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)
            parameter = learner.pi.dec[-1].bias
            direction = parameter.grad / parameter.grad.norm()
            analytic = (parameter.grad * direction).sum()
            original, eps = parameter.detach().clone(), .01
            with torch.no_grad(): parameter.copy_(original + eps * direction)
            plus = -learner.imagine(start, history, fixed_robot_noise=noise)["objective"].detach()
            with torch.no_grad(): parameter.copy_(original - eps * direction)
            minus = -learner.imagine(start, history, fixed_robot_noise=noise)["objective"].detach()
            with torch.no_grad(): parameter.copy_(original)
            torch.testing.assert_close(analytic, (plus - minus) / (2 * eps), atol=2e-5, rtol=.03)
        for module, before in zip((learner.wm, learner.critic), snapshots):
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, before[key], atol=0, rtol=0)
            self.assertTrue(all(p.grad is None for p in module.parameters()))

    def test_checkpoint_new_profile_roundtrip_and_old_profile_rejected_before_mutation(self):
        learner = self.learner
        kwargs = dict(iteration=3, physics={"hand_collision": True, "self_collision": True},
                      data_identity={"synthetic": True}, initialization={"mode": "CPU_fixture"},
                      extra={"config": learner.config.to_dict()})
        saved = make_checkpoint(learner.pi, learner.wm, learner.critic, learner.optimizers, **kwargs)
        load_checkpoint(saved, learner.pi, learner.wm, learner.critic, learner.optimizers,
                        physics=kwargs["physics"], data_identity=kwargs["data_identity"], domain_randomization_profile=MATCHING_DR)
        before = deepcopy(learner.pi.state_dict())
        old_pi = Policy(hidden=(32, 16), object_hidden=(16,))
        old = make_checkpoint(old_pi, WorldModel(hidden=(32, 32), contact_hidden=12), Critic(hidden=(32, 16)), learner.optimizers,
                              **{**kwargs, "extra": {"config": replace(learner.config, observation_profile="v1", domain_randomization_profile="v1").to_dict()}})
        with self.assertRaisesRegex(ValueError, "observation profile"):
            load_checkpoint(old, learner.pi, learner.wm, learner.critic,
                            physics=kwargs["physics"], data_identity=kwargs["data_identity"])
        for key, value in learner.pi.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        saved["experiment_contract"]["domain_randomization"]["object_mass_delta_kg"] = [-10., 10.]
        with self.assertRaisesRegex(ValueError, "DR contract"):
            validate_checkpoint_features(saved)

    def test_new_formal_config_preserves_optimization_profile(self):
        root = Path(__file__).resolve().parents[1]
        config, changes = read_config(root / "configs/hoi_v1_1_observation_matching.json")
        self.assertEqual(changes, {})
        self.assertEqual((config.observation_profile, config.domain_randomization_profile), (MATCHING, MATCHING_DR))
        self.assertEqual((config.num_envs, config.pi_batch, config.horizon, config.lr), (4096, 4096, 32, .0002))

    def test_new_profile_preflight_records_pending_native_dr_without_starting_workers(self):
        from hoi.v1.train import main
        bundle = {"partition": "train", "scope": {"is_subset": False}, "assets": [], "clips": [],
                  "bundle_content_sha256": "a" * 64, "manifest": {"sha256": "b" * 64},
                  "recipes": {}, "robot_fk": {}, "reference_kinematics": {}, "native_robot_asset": {},
                  "hands": {"geometry_cache": {"sha256": "c" * 64}}, "timebase": {}}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            config = folder / "config.json"
            config.write_text(json.dumps({"observation_profile": MATCHING, "domain_randomization_profile": MATCHING_DR,
                                           "iterations": 2}))
            with patch("hoi.v1.evaluate._read_bundle_header", return_value=bundle), patch(
                    "hoi.v1.workers.WorkerGroup") as workers, redirect_stdout(io.StringIO()):
                plan = main(["--bundle", str(folder / "unused.json"), "--config", str(config),
                             "--output", str(folder / "preflight"), "--preflight"])
            workers.assert_not_called()
            self.assertFalse(plan["training_started"])
            self.assertFalse(plan["native_dr_gate"]["passed"])
            self.assertEqual(plan["actor_observation_contract"]["human_goal_dim"], 448)

    def test_frozen_evaluation_records_new_shapes_full_duration_and_no_updates(self):
        from test_hoi_v1_evaluate import fixture, SyntheticWorkers
        from hoi.v1.evaluate import evaluate_group
        _, motion, _, shapes, fk, metadata = fixture()
        kin = ChainKinematics()
        motion.body_p, motion.body_R, motion.body_v, motion.body_w = kin.forward(*motion.ref_arrays())
        mdp = HOIMDP(kin, motion, torch.zeros(29), shapes)
        policy = Policy(hidden=(32, 16), object_hidden=(16,), observation_profile=MATCHING)
        before = deepcopy(policy.state_dict())
        workers = SyntheticWorkers(motion, mdp)
        with tempfile.TemporaryDirectory() as temp:
            cases = evaluate_group(policy, motion, mdp, shapes, workers, [0, 1], fk39=fk,
                                   joint_names=fk.dof_names, output=Path(temp) / "eval", metadata=metadata,
                                   verify_tracked_fk=False)
            for index, length in enumerate((3, 6)):
                raw = torch.load(cases[index]["rollout"], weights_only=True)
                self.assertEqual(raw["control"]["robot_observation"].shape, (length, 93))
                self.assertEqual(raw["control"]["object_observation"].shape, (length, 9))
                self.assertEqual(raw["control"]["human_goal"].shape, (length, 448))
                self.assertEqual(raw["control"]["object_goal"].shape, (length, 63))
                self.assertEqual(raw["states"]["s"].shape[0], length + 1)
                self.assertEqual(raw["metadata"]["actor_observation_contract"]["profile"], MATCHING)
        self.assertEqual(workers.reset_count, 1)
        for key, value in policy.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)


class RandomizationTests(unittest.TestCase):
    def test_mass_support_positive_and_inertia_scaled_from_nominal(self):
        generator = torch.Generator().manual_seed(11)
        mass = torch.tensor([.2, 1., 3., 10.]).repeat(300)
        sampled, com, inertia, friction = sample_object_properties(mass, torch.zeros(len(mass), 3),
                            torch.eye(3).expand(len(mass), 3, 3).clone(), generator=generator)
        self.assertTrue(bool((sampled >= torch.maximum(mass - 1.5, mass * .1)).all()))
        self.assertTrue(bool((sampled <= mass + 1.5).all()))
        self.assertTrue(bool((com.abs() <= .01).all()))
        self.assertTrue(bool(((friction >= .4) & (friction <= .8)).all()))
        torch.testing.assert_close(inertia[:, 0, 0], sampled / mass)
        with self.assertRaisesRegex(ValueError, "positive"):
            sample_mass(torch.tensor([0.]), 1.5)

    def test_native_gate_cannot_be_satisfied_by_old_or_cpu_only_evidence(self):
        self.assertTrue(validate_dr_evidence(None, "v1")["passed"])
        self.assertFalse(validate_dr_evidence(None, MATCHING_DR)["passed"])
        self.assertFalse(randomization_enabled(MATCHING_DR, True))
        self.assertFalse(randomization_enabled("v1", False))
        self.assertTrue(randomization_enabled(MATCHING_DR, False))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "evidence.json"
            path.write_text(json.dumps({"domain_randomization": {"contract": randomization_contract(MATCHING_DR),
                            "native_property_readback_passed": True, "subset_reset_passed": True, "nominal_eval_passed": True}}))
            self.assertTrue(validate_dr_evidence(path, MATCHING_DR)["passed"])

    def test_native_setter_mock_subset_isolation_context_and_readback(self):
        class Flags(IntFlag):
            SHAPE_PROPERTIES = 1
            BODY_PROPERTIES = 2
            BODY_INERTIAL_PROPERTIES = 4
        class Body:
            def __init__(self, mass):
                n, b = mass.shape
                self.data = SimpleNamespace(body_mass=SimpleNamespace(torch=mass.clone()),
                    body_inertia=SimpleNamespace(torch=torch.eye(3).reshape(1, 1, 9).expand(n, b, 9).clone()),
                    body_com_pose_b=SimpleNamespace(torch=torch.zeros(n, b, 7)))
                self.material = torch.full((n, 2), .6)
                self._root_view = SimpleNamespace(get_attribute=lambda name, model: self.material)
            def set_masses_index(self, *, masses): self.data.body_mass.torch.copy_(masses)
            def set_inertias_index(self, *, inertias): self.data.body_inertia.torch.copy_(inertias)
            def set_coms_index(self, *, coms): self.data.body_com_pose_b.torch[..., :3].copy_(coms)
        robot = Body(torch.tensor([[2., 8., 1.]]).expand(3, -1))
        obj = Body(torch.full((3, 1), .5))
        manager = SimpleNamespace(get_model=lambda: None, add_model_change=lambda flags: None)
        host = SimpleNamespace(N=3, env=SimpleNamespace(robot=robot, torso_id=1,
                               sim=SimpleNamespace(backend_manager=manager)), object=obj)
        host.xi_o = physical_context(torch.full((3,), .5), torch.zeros(3, 3), torch.eye(3).expand(3, 3, 3),
                                     torch.full((3,), .6), torch.full((3,), .3))
        with patch.dict(sys.modules, {"newton": SimpleNamespace(ModelFlags=Flags), "warp": SimpleNamespace(to_torch=lambda x: x)}):
            dr = NativeRandomization(host)
            torch.testing.assert_close(robot.data.body_mass.torch[:, [0, 2]], torch.tensor([[2., 1.]]).expand(3, -1))
            self.assertTrue(bool(((robot.data.body_mass.torch[:, 1] >= 5.) & (robot.data.body_mass.torch[:, 1] <= 11.)).all()))
            before = host.xi_o.clone()
            dr.reset(torch.tensor([0, 2]))
            torch.testing.assert_close(host.xi_o[1], before[1], atol=0, rtol=0)
            torch.testing.assert_close(host.xi_o[:, 0], obj.data.body_mass.torch[:, 0])
            torch.testing.assert_close(host.xi_o[:, 1:4], obj.data.body_com_pose_b.torch[:, 0, :3])
            nominal = dr.nominal_object_mass.clone()
            dr.reset(torch.tensor([0]))
            torch.testing.assert_close(dr.nominal_object_mass, nominal, atol=0, rtol=0)
            with patch.object(obj, "set_masses_index", side_effect=lambda **kw: None):
                with self.assertRaisesRegex(RuntimeError, "readback"):
                    dr.reset(torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
