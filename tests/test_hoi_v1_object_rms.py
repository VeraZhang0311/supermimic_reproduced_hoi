"""Actor object RMS lifecycle, frozen statistics and differentiable inputs."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from hoi.v1.checkpoint import make_checkpoint, load_checkpoint
from hoi.v1.evaluate import evaluate_group
from hoi.v1.mdp import HOIMDP
from hoi.v1.networks import Policy
from hoi.v1.observations import MATCHING, observation_contract
from hoi.v1.state import unflatten_state, unflatten_object
from test_hoi_v1_1_observations import matching_learner, ChainKinematics
from test_hoi_v1_learner import transition, fill
from test_hoi_v1_evaluate import fixture
from test_hoi_v1_mass_context import NominalMassWorkers


class ObjectRMSTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(352)
        old = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old)
        self.learner = matching_learner()

    def test_two_independent_rms_have_expected_dimensions_and_no_clipping(self):
        pi = self.learner.pi
        obs, goal = pi.object_obs_norm, pi.object_goal_norm
        self.assertEqual((obs.mean.numel(), goal.mean.numel()), (9, 63))
        obs.update(torch.randn(13, 9) + 2.)
        self.assertEqual((float(obs.count), float(goal.count)), (13., 0.))
        before = deepcopy(obs.state_dict())
        goal.update(torch.randn(7, 63) - 3.)
        for name, value in obs.state_dict().items(): torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        for norm, dim in ((obs, 9), (goal, 63)):
            norm.mean.zero_()
            norm.var.zero_()
            value = torch.full((2, dim), 2., requires_grad=True)
            result = norm(value)
            torch.testing.assert_close(result, torch.full_like(value, 200.))
            gradient = torch.autograd.grad(result.sum(), value)[0]
            torch.testing.assert_close(gradient, torch.full_like(value, 100.))
        contract = observation_contract(MATCHING)["object_normalization"]
        self.assertIsNone(contract["clip"])
        self.assertEqual(contract["epsilon"], .01)

    def test_real_collection_counts_current_tick_once_and_history_shares_statistics(self):
        learner = self.learner
        tr = transition(learner, [8, 80], 0)
        S, O = unflatten_state(tr["s"]), unflatten_object(tr["o"])
        observation = learner.actor_obs.object(S, O, tr["f"])
        future = learner.actor_obs.object_goal(S, O, tr["f"])
        with patch.object(learner.actor_obs, "add_noise", side_effect=lambda value: value):
            learner.action(tr, exploration=False)
        self.assertEqual(float(learner.pi.object_obs_norm.count), 2.)
        self.assertEqual(float(learner.pi.object_goal_norm.count), 2.)
        torch.testing.assert_close(learner.pi.object_obs_norm.mean, observation.mean(0))
        torch.testing.assert_close(learner.pi.object_goal_norm.mean, future.mean(0))
        hist = learner.object_history
        normalized = learner.pi.object_obs_norm(hist)
        torch.testing.assert_close(normalized[:, 0], normalized[:, -1])
        before = deepcopy(learner.pi.state_dict())
        learner.action(tr, exploration=False, update_statistics=False)
        for name, value in learner.pi.state_dict().items(): torch.testing.assert_close(value, before[name], atol=0, rtol=0)

    def test_imagination_freezes_nonempty_rms_and_backpropagates_through_them(self):
        learner = self.learner
        for step in range(12):
            tr = transition(learner, [8 + step, 80 + step], step)
            learner.action(tr, exploration=False)
            learner.observe(tr)
        start, history = learner.buffer.sample_starts(2, 10, horizon=3, motion=learner.motion)
        norms = [learner.pi.object_obs_norm, learner.pi.object_goal_norm]
        before = [deepcopy(norm.state_dict()) for norm in norms]
        self.assertTrue(all(float(norm.count) == 24 for norm in norms))
        output = learner.imagine(start, history)
        (-output["objective"]).backward()
        for module in [learner.pi.object_enc, learner.pi.dec]:
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)
        for norm, saved in zip(norms, before):
            for name, value in norm.state_dict().items(): torch.testing.assert_close(value, saved[name], atol=0, rtol=0)

    def test_normalized_future_goal_root_jacobian_matches_finite_difference(self):
        learner = self.learner
        tr = transition(learner, [8, 80], 0)
        S, O = unflatten_state(tr["s"]), unflatten_object(tr["o"])
        norm = learner.pi.object_goal_norm
        raw = learner.actor_obs.object_goal(S, O, tr["f"])
        norm.update(torch.cat((raw - .02, raw + .02)))
        p = S[0].clone().requires_grad_()
        weights = torch.linspace(-.6, .8, 63)
        def score(root):
            value = learner.actor_obs.object_goal((root, *S[1:]), O, tr["f"])
            return (norm(value) * weights).sum()
        gradient = torch.autograd.grad(score(p), p)[0]
        direction = gradient / gradient.norm()
        eps = .001
        fd = (score(p.detach() + eps * direction) - score(p.detach() - eps * direction)) / (2 * eps)
        torch.testing.assert_close((gradient * direction).sum(), fd, atol=.01, rtol=.001)

    def test_checkpoint_roundtrips_rms_and_rejects_raw_schema_before_mutation(self):
        learner = self.learner
        learner.pi.object_obs_norm.update(torch.randn(15, 9) + 3.)
        learner.pi.object_goal_norm.update(torch.randn(11, 63) - 2.)
        kwargs = dict(iteration=0, physics={"hand_collision": True, "self_collision": True},
                      data_identity={"CPU_fixture": True}, initialization={"mode": "from_scratch"},
                      extra={"config": learner.config.to_dict()})
        saved = make_checkpoint(learner.pi, learner.wm, learner.critic, learner.optimizers, **kwargs)
        self.assertEqual(saved["feature_schema"]["version"], "supermimic_hoi_v1_1_observation_matching_3")
        before = deepcopy(learner.pi.state_dict())
        learner.pi.object_obs_norm.mean.zero_()
        learner.pi.object_goal_norm.count.zero_()
        load_checkpoint(saved, learner.pi, learner.wm, learner.critic,
                        physics=kwargs["physics"], data_identity=kwargs["data_identity"])
        for name, value in learner.pi.state_dict().items(): torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        raw = deepcopy(saved)
        raw["feature_schema"]["version"] = "supermimic_hoi_v1_1_observation_matching_2"
        contract = raw["feature_schema"]["actor_observation_contract"]
        contract["normalization"] = "clipped_RMS_robot_human;raw_object_pose_goals"
        del contract["object_normalization"]
        with self.assertRaisesRegex(ValueError, "feature schema"):
            load_checkpoint(raw, learner.pi, learner.wm, learner.critic,
                            physics=kwargs["physics"], data_identity=kwargs["data_identity"])
        for name, value in learner.pi.state_dict().items(): torch.testing.assert_close(value, before[name], atol=0, rtol=0)

    def test_frozen_evaluation_preserves_populated_object_rms(self):
        _, motion, _, shapes, fk, metadata = fixture()
        kin = ChainKinematics()
        motion.body_p, motion.body_R, motion.body_v, motion.body_w = kin.forward(*motion.ref_arrays())
        mdp = HOIMDP(kin, motion, torch.zeros(29), shapes)
        pi = self.learner.pi
        pi.object_obs_norm.update(torch.randn(15, 9))
        pi.object_goal_norm.update(torch.randn(11, 63))
        before = deepcopy(pi.state_dict())
        with tempfile.TemporaryDirectory() as temp:
            cases = evaluate_group(pi, motion, mdp, shapes, NominalMassWorkers(motion, mdp), [0, 1], fk39=fk,
                joint_names=fk.dof_names, output=Path(temp)/"eval", metadata=metadata, verify_tracked_fk=False,
                torso_mass_context=True)
            raw = torch.load(cases[0]["rollout"], weights_only=True)
            self.assertIsNone(raw["metadata"]["actor_observation_contract"]["object_normalization"]["clip"])
        for name, value in pi.state_dict().items(): torch.testing.assert_close(value, before[name], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
