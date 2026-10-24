"""CPU integration tests use real HOI heads with explicitly synthetic dynamics.

These establish software/gradient contracts. They provide no native-physics or
HOI-success evidence, and the reduced networks are test-only explicit choices.
"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hoi.v1.buffer import FIELDS, ReplayBuffer
from hoi.v1.checkpoint import make_checkpoint, load_checkpoint
from hoi.v1.config import TrainConfig
from hoi.v1.learner import Learner
from hoi.v1.mdp import HOIMDP
from hoi.v1.motion import TrainingMotionBank
from hoi.v1.networks import Policy, WorldModel, Critic, frozen_parameters
from hoi.v1.state import flatten_state, flatten_object, unflatten_state, unflatten_object


class SyntheticKinematics:
    """Differentiable 14-body FK fixture; deliberately not G1 physical evidence."""
    def __init__(self):
        self.q_lo, self.q_hi = torch.full((29,), -2.), torch.full((29,), 2.)

    def forward(self, p, R, v, w, q, qd):
        offsets = torch.zeros(q.shape[:-1] + (14, 3), dtype=q.dtype, device=q.device)
        offsets[..., :, 0] = q[..., :14] * .1
        offsets[..., :, 1] = torch.linspace(-.2, .2, 14, dtype=q.dtype, device=q.device)
        offsets[..., :, 2] = torch.linspace(-.3, .3, 14, dtype=q.dtype, device=q.device)
        position = p.unsqueeze(-2) + (R.unsqueeze(-3) @ offsets.unsqueeze(-1)).squeeze(-1)
        rotation = R.unsqueeze(-3).expand(R.shape[:-2] + (14, 3, 3))
        velocity = v.unsqueeze(-2).expand(v.shape[:-1] + (14, 3))
        omega = w.unsqueeze(-2).expand(w.shape[:-1] + (14, 3))
        return position, rotation, velocity, omega


class SyntheticSphereShapes:
    """Known analytic sphere distance and deterministic 128-point surfaces."""
    def __init__(self):
        generator = torch.Generator().manual_seed(5)
        cloud = torch.randn(2, 128, 3, generator=generator)
        self.cloud = cloud / torch.linalg.vector_norm(cloud, dim=-1, keepdim=True) * .15

    def points(self, ids):
        return self.cloud[ids]

    def bbox_diagonal(self, ids):
        return torch.full(ids.shape, .52, device=ids.device)

    def hand_distances(self, kin, S, O, ids):
        palm = S[0].unsqueeze(-2).unsqueeze(-2) + S[4][..., :6].reshape(S[4].shape[:-1] + (2, 1, 3)) * .05
        palm = palm + palm.new_tensor([.3, 0., .1])
        return torch.linalg.vector_norm(palm - O[0].unsqueeze(-2).unsqueeze(-2), dim=-1) - .15


def reference_fixture():
    kin, shapes = SyntheticKinematics(), SyntheticSphereShapes()
    clips = []
    n = 180
    for asset in range(2):
        time = torch.arange(n).float() * .02
        p = torch.stack((time * .04, torch.full_like(time, asset * .2), torch.full_like(time, .85)), -1)
        R = torch.eye(3).expand(n, 3, 3).clone()
        velocity = torch.zeros(n, 3)
        velocity[:, 0] = .04
        zero = torch.zeros(n, 3)
        robot = {"root_pos": p, "root_rot": R, "derived_ref_v": velocity,
                 "derived_ref_omega": zero, "joint_pos": torch.zeros(n, 29), "derived_ref_joint_qd": torch.zeros(n, 29)}
        obj = {"pos": p + torch.tensor([.3, 0., .1]), "rot": R.clone(), "derived_ref_v": velocity.clone(),
               "derived_ref_omega": zero.clone()}
        clips.append({"asset_id": asset, "name": f"synthetic_{asset}", "reference": {"robot": robot, "object": obj},
                      "expected_contact": torch.ones(n, 2), "expected_valid": torch.ones(n, 2)})
    motion = TrainingMotionBank(clips, kin)
    mdp = HOIMDP(kin, motion, torch.zeros(29), shapes)
    return motion, mdp, shapes


def make_learner(horizon=3, no_critic=False):
    motion, mdp, shapes = reference_fixture()
    config = TrainConfig(num_envs=2, buffer_steps=64, horizon=horizon, wm_window=8, wm_batches=1, wm_batch=4,
                         pi_batch=2, critic_epochs=1, critic_minibatches=2, experiment_kind="smoke", amp="none", no_critic=no_critic)
    pi = Policy(hidden=(32, 16), object_hidden=(16,))
    wm = WorldModel(hidden=(32, 32), contact_hidden=12)
    critic = Critic(hidden=(32, 16))
    # Test a trained nonzero input Jacobian rather than zero-initialized robot output.
    with torch.no_grad():
        wm.net[-1].weight.normal_(std=.005)
        wm.net[-1].bias.zero_()
    replay = ReplayBuffer(config.buffer_steps, config.num_envs)
    return Learner(pi, wm, critic, motion, mdp, shapes, replay, config)


def transition(learner, f, ep_step, a=None, a_prev=None):
    f = torch.as_tensor(f, dtype=torch.long)
    n = len(f)
    value = {key: torch.zeros((n,) + ((shape,) if isinstance(shape, int) else shape), dtype=dtype)
             for key, (shape, dtype) in FIELDS.items()}
    S = tuple(x.clone() for x in learner.motion.ref_state(f))
    O = tuple(x.clone() for x in learner.motion.ref_object(f))
    O[0][:, 0] += .04
    a = torch.full((n, 29), .1) if a is None else a
    a_prev = torch.zeros_like(a) if a_prev is None else a_prev
    # Explicit synthetic executed dynamics. s1/o1 are neither goal nor reference.
    S1 = (S[0] + .02 * S[2], S[1], S[2] + .001 * a[:, :3], S[3], S[4] + .01 * a, S[5] + .01 * a)
    O1 = (O[0] + .02 * O[2] + .002 * a[:, :3], O[1], O[2] + .01 * a[:, :3], O[3])
    value.update(s=flatten_state(S), s1=flatten_state(S1), o=flatten_object(O), o1=flatten_object(O1),
                 a=a.clone(), a_prev=a_prev.clone(), f=f, ep_step=torch.full((n,), ep_step, dtype=torch.int32),
                 asset_id=learner.motion.asset_ids[f], contact=torch.ones(n, 3), contact_valid=torch.ones(n, 3, dtype=torch.bool))
    value["xi"][:, 0] = .75
    value["xi_o"][:, 0] = 1.
    return value


def fill(learner, steps=24):
    for step in range(steps):
        learner.observe(transition(learner, [step, 70 + step], step))


class ReplayTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def test_terminal_transition_remains_and_reset_is_masked(self):
        learner = make_learner()
        for step in range(8):
            item = transition(learner, [60 + step, 80 + step], step if step < 3 else step - 3)
            item["done"].fill_(step == 2)
            learner.buffer.add(item)
        window = learner.buffer._windows_at(torch.tensor([0]), torch.tensor([0]), 8)
        torch.testing.assert_close(window["mask"], torch.tensor([[1., 1., 1., 0., 0., 0., 0., 0.]]))
        torch.testing.assert_close(window["s1"][0, 2], learner.buffer.data["s1"][2, 0])

    def test_histories_respect_reset_asset_and_ring_boundary(self):
        learner = make_learner()
        learner.buffer = ReplayBuffer(8, 2)
        for absolute in range(12):
            episode_step = absolute if absolute < 6 else absolute - 6
            frame = absolute if absolute < 6 else 180 + episode_step
            item = transition(learner, [frame, frame], episode_step)
            item["s"][:, 0] = absolute
            item["done"].fill_(absolute == 5)
            learner.buffer.add(item)
        starts, history = learner.buffer.sample_starts(200, 10, horizon=3, motion=learner.motion)
        self.assertTrue(torch.equal(history["asset_id"], starts["asset_id"][:, None].expand_as(history["asset_id"])))
        marker = history["s"][..., 0]
        self.assertTrue(bool((marker[:, 1:] >= marker[:, :-1]).all()))
        self.assertGreaterEqual(float(marker.min()), 4.)
        for index, current in enumerate(starts["s"][:, 0]):
            first = max(4, 6 if current >= 6 else 0)
            self.assertEqual(float(marker[index, 0]), first)
            self.assertEqual(float(marker[index, -1]), float(current))

    def test_fit_and_holdout_never_share_unmasked_transition(self):
        learner = make_learner()
        fill(learner)
        fit = learner.buffer.sample_windows(32, 8, split="fit")
        held = learner.buffer.sample_windows(32, 8, split="holdout")
        fit_frames = set(fit["f"][fit["mask"].bool()].tolist())
        held_frames = set(held["f"][held["mask"].bool()].tolist())
        self.assertFalse(fit_frames & held_frames)
        self.assertTrue(all((f // 50) % 20 for f in fit_frames))
        self.assertTrue(all((f // 50) % 20 == 0 for f in held_frames))

    def test_insufficient_windows_and_horizon_fail_bounded(self):
        learner = make_learner()
        with self.assertRaises(ValueError):
            learner.buffer.sample_windows(4, 8)
        learner.observe(transition(learner, [178, 179], 0))
        with patch("hoi.v1.buffer.torch.randint", wraps=torch.randint) as randint:
            with self.assertRaisesRegex(ValueError, "horizon"):
                learner.buffer.sample_starts(2, 10, horizon=3, motion=learner.motion)
            self.assertLessEqual(randint.call_count, 64)
        with self.assertRaises(ValueError):
            learner.buffer.sample_windows(2, 0)


class LearnerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def test_collector_uses_current_history_and_counts_current_samples_once(self):
        learner = make_learner()
        snapshot = transition(learner, [60, 185], 0)
        with patch.object(learner.pi, "forward", wraps=learner.pi.forward) as forward:
            action = learner.action(snapshot, exploration=False)
            torch.testing.assert_close(learner.robot_history[:, 0], learner.robot_history[:, -1])
            self.assertEqual(forward.call_count, 1)
        for norm in (learner.pi.obs_norm, learner.pi.object_obs_norm, learner.pi.goal_norm, learner.pi.object_goal_norm):
            self.assertEqual(float(norm.count), 2)
        a = torch.full((2, 29), .237)
        executed = transition(learner, [60, 185], 0, a=a)
        learner.observe(executed)
        torch.testing.assert_close(learner.buffer.data["a"][0], a)
        torch.testing.assert_close(learner.buffer.data["o1"][0], executed["o1"])
        self.assertFalse(torch.equal(executed["o1"][:, :3], learner.motion.object_p[executed["f"] + 1]))
        self.assertEqual(learner.sim_steps, 2)
        self.assertEqual(action.shape, (2, 29))

    def test_eight_step_wm_fit_uses_executed_actions_and_changes_parameters(self):
        learner = make_learner()
        fill(learner)
        window = learner.buffer.sample_windows(3, 8, split="fit")
        original = deepcopy(learner.wm.state_dict())
        with patch.object(learner.wm, "step", wraps=learner.wm.step) as step:
            predicted_s, predicted_o, logits = learner.wm_rollout(window)
            self.assertEqual(step.call_count, 8)
            for t, call in enumerate(step.call_args_list):
                torch.testing.assert_close(call.args[2], window["a"][:, t])
        self.assertEqual(predicted_s[0].shape, (3, 8, 3))
        self.assertEqual(predicted_o[0].shape, (3, 8, 3))
        self.assertEqual(logits.shape, (3, 8, 3))
        loss = learner.wm_update()
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in loss.values()))
        self.assertTrue(any(not torch.equal(v, original[k]) for k, v in learner.wm.state_dict().items()))

    def test_actor_bptt_freezes_entire_wm_and_critic_and_recomputes_goals(self):
        learner = make_learner()
        fill(learner)
        start, history = learner.buffer.sample_starts(2, 10, horizon=3, motion=learner.motion)
        frozen = {name: deepcopy(model.state_dict()) for name, model in (("wm", learner.wm), ("critic", learner.critic))}
        noise = [torch.zeros(2, 10, 99)] + [torch.zeros(2, 99) for _ in range(3)]
        with frozen_parameters(learner.wm, learner.critic):
            with patch.object(learner.motion, "object_goal", wraps=learner.motion.object_goal) as goal:
                trajectory = learner.imagine(start, history, fixed_robot_noise=noise)
                object_gradient = torch.autograd.grad(trajectory["objects"][-1][0].square().sum(),
                                                      learner.pi.dec[-1].bias, retain_graph=True)[0]
                self.assertTrue(bool(torch.isfinite(object_gradient).all()))
                self.assertGreater(float(object_gradient.abs().sum()), 0)
                trajectory["objective"].backward()
                self.assertGreaterEqual(goal.call_count, 4)  # every actor tick plus terminal critic
                for tick in range(3):
                    torch.testing.assert_close(goal.call_args_list[tick].args[0][0], trajectory["states"][tick][0])
                    torch.testing.assert_close(goal.call_args_list[tick].args[1][0], trajectory["objects"][tick][0])
        pi_grads = [p.grad for p in learner.pi.parameters() if p.grad is not None]
        self.assertTrue(pi_grads)
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in pi_grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in pi_grads), 0)
        for name, model in (("wm", learner.wm), ("critic", learner.critic)):
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, frozen[name][key])

    def test_fixed_noise_actor_directional_derivative(self):
        learner = make_learner(horizon=2, no_critic=True)
        fill(learner)
        start, history = learner.buffer.sample_starts(2, 10, horizon=2, motion=learner.motion)
        noise = [torch.zeros(2, 10, 99), torch.zeros(2, 99), torch.zeros(2, 99)]
        parameter = learner.pi.dec[-1].bias
        initial = parameter.detach().clone()
        direction = torch.randn_like(parameter)
        with frozen_parameters(learner.wm, learner.critic):
            objective = learner.imagine(start, history, fixed_robot_noise=noise)["objective"]
            grad = torch.autograd.grad(objective, parameter)[0]
            epsilon = .01
            with torch.no_grad():
                parameter.copy_(initial + epsilon * direction)
                plus = learner.imagine(start, history, fixed_robot_noise=noise)["objective"]
                parameter.copy_(initial - epsilon * direction)
                minus = learner.imagine(start, history, fixed_robot_noise=noise)["objective"]
                parameter.copy_(initial)
            finite_difference = (plus - minus) / (2 * epsilon)
        self.assertTrue(bool(torch.isfinite(grad).all()))
        self.assertGreater(float(grad.abs().sum()), 0)
        torch.testing.assert_close((grad * direction).sum(), finite_difference, atol=2e-5, rtol=.03)

    def test_value_fitting_updates_critic_pointnet_without_wm_actor_changes(self):
        learner = make_learner()
        fill(learner)
        start, history = learner.buffer.sample_starts(2, 10, horizon=3, motion=learner.motion)
        with frozen_parameters(learner.wm, learner.critic):
            trajectory = learner.imagine(start, history)
        before = deepcopy(learner.critic.shape_encoder.state_dict())
        other = {name: deepcopy(model.state_dict()) for name, model in (("wm", learner.wm), ("pi", learner.pi))}
        loss = learner.critic_update(trajectory, start)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertTrue(any(not torch.equal(value, before[key]) for key, value in learner.critic.shape_encoder.state_dict().items()))
        for name, model in (("wm", learner.wm), ("pi", learner.pi)):
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, other[name][key])

    def test_policy_update_wrapper_changes_actor_only_before_value_fit(self):
        learner = make_learner()
        fill(learner)
        actor = deepcopy(learner.pi.state_dict())
        frozen = {name: deepcopy(model.state_dict()) for name, model in (("wm", learner.wm), ("critic", learner.critic))}
        with patch.object(learner, "critic_update", return_value=0.) as value_update:
            metrics = learner.policy_update()
        self.assertEqual(value_update.call_count, 1)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["pi_gradnorm"])))
        self.assertGreater(metrics["pi_gradnorm"], 0)
        self.assertTrue(any(not torch.equal(v, actor[k]) for k, v in learner.pi.state_dict().items()))
        for name, model in (("wm", learner.wm), ("critic", learner.critic)):
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, frozen[name][key])

    def test_checkpoint_resume_still_requires_fresh_real_replay_prefill(self):
        old = make_learner()
        fill(old)
        physics = {"hand_collision": True, "self_collision": True}
        identity = {"dataset": "synthetic_CPU_contract_fixture", "manifest_sha256": "a" * 64}
        checkpoint = make_checkpoint(old.pi, old.wm, old.critic, old.optimizers, iteration=7, physics=physics,
                                     data_identity=identity, initialization={"kind": "synthetic_CPU_test"})
        learner = make_learner()
        restored = load_checkpoint(checkpoint, learner.pi, learner.wm, learner.critic, learner.optimizers,
                                   physics=physics, data_identity=identity)
        self.assertEqual(restored["iteration"], 7)
        self.assertTrue(restored["continuation"]["fresh_replay_prefill_required"])
        self.assertFalse(restored["continuation"]["bitwise_resume"])
        self.assertEqual(learner.buffer.size, 0)
        with self.assertRaises(ValueError):
            learner.wm_update()
        fill(learner, steps=8)
        self.assertEqual(learner.buffer.size, 8)


class ConfigTest(unittest.TestCase):
    def test_production_defaults_preserved_and_invalid_shape_rejected(self):
        c = TrainConfig().validate()
        self.assertEqual((c.num_envs, c.wm_window, c.wm_batch, c.pi_batch, c.horizon, c.wm_batches), (4096, 8, 2048, 4096, 32, 10))
        self.assertEqual((c.utd, c.gamma, c.lam, c.lr, c.warmup_steps, c.wm_only_iters), (.4, .99, .95, .0002, 64, 500))
        self.assertAlmostEqual(c.lr_at(0), c.lr)
        self.assertAlmostEqual(c.lr_at(c.lr_decay_iters), .03 * c.lr)
        for change in ({"hist": 9}, {"goal_frames": 9}, {"buffer_steps": 4}, {"wm_batch": True}, {"num_envs": 0},
                       {"action_noise": float("nan")}, {"lr": float("inf")}, {"utd": float("inf")}, {"wm_only_iters": 1.5}):
            with self.assertRaises(ValueError):
                replace(c, **change).validate()


if __name__ == "__main__":
    unittest.main()
