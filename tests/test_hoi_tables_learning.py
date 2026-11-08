"""Contract, BPTT and update tests for the from-scratch tables learner."""
from dataclasses import replace
import unittest

import torch

from hoi_residual.features import ORACLE_DIM, object_observation, torso_pose
from supermimic.actuator_state import advance_actuator_state
from supermimic.mdp import flatten_state, normalize_xi
from supermimic.robot_metadata import G1_BODY_NAMES
from supermimic.rotations import so3_exp
from hoi_tables.config import TrainConfig
from hoi_tables.learner import Learner
from hoi_tables.networks import Policy, Critic, WorldModel, OracleNormalizer, frozen_parameters, feature_schema
from hoi_tables.v11.state import flatten_object, unflatten_state, unflatten_object


class FK38:
    body_names = G1_BODY_NAMES

    def forward(self, S):
        p, R, v, w, q, qd = S
        n = len(self.body_names)
        # A differentiable articulated chain fixture, with reference and actual
        # sharing the same named frames. Native asset verification is separate.
        index = torch.arange(n, device=p.device) % 29
        offsets = torch.stack((q[..., index] * .03,
                               q[..., index] * .02,
                               q[..., index] * .01), -1)
        P = p[..., None, :] + offsets
        rotations = R[..., None, :, :].expand(*R.shape[:-2], n, 3, 3)
        V = v[..., None, :].expand_as(P) + qd[..., index, None] * .01
        W = w[..., None, :].expand_as(P)
        return P, rotations, V, W


class Motion:
    partition = "train"
    for_training = True

    def __init__(self, n=100):
        self.clip_end = torch.tensor([n - 1])
        self.clip_id = torch.zeros(n, dtype=torch.long)
        self.dtype = torch.float32

    def ref_state(self, f):
        shape = f.shape
        tick = f.to(dtype=self.dtype)
        p = torch.stack((tick * .003, tick * 0, tick * 0 + .8), -1)
        R = torch.eye(3, dtype=self.dtype).expand(*shape, 3, 3)
        v = p * 0
        q = torch.zeros(*shape, 29, dtype=self.dtype)
        return p, R, v, v, q, q

    def object_ref(self, f):
        p, R, v, w, *_ = self.ref_state(f)
        return p + torch.tensor([.3, .1, -.2]), R, v, w


class Reward:
    def __call__(self, S, O, f, a, previous, contact, contact_valid=None):
        return .1 * torch.exp(-(O[0] - torch.tensor([.3, 0., .5])).square().sum(-1)) - .002 * (a - previous).square().sum(-1) + .01 * contact[..., :2].mean(-1)

    def soft_termination(self, S, O, f):
        return torch.sigmoid((O[0].square().sum(-1) - 3.) * 2)


def fixture(B=2, H=32):
    torch.manual_seed(7)
    motion, kin = Motion(H + 50), FK38()
    f = torch.arange(B) + 2
    S = motion.ref_state(f)
    O = motion.object_ref(f)
    xi = torch.zeros(B, 33)
    xi[:, 0] = .6
    actuator = torch.cat((torch.zeros(B, 29), torch.full((B, 1), 3.),
                          torch.ones(B, 29), torch.ones(B, 29)), -1)
    state = {"s": flatten_state(S), "o": flatten_object(O), "xi": xi,
             "xi_o": torch.tensor([1., 0., 0., 0., .1, .1, .1, 0., 0., 0., .6, .3]).expand(B, -1).clone(),
             "mass_context": torch.zeros(B, 1), "actuator": actuator, "f": f,
             "a_prev": torch.zeros(B, 29), "q_prev": torch.zeros(B, 29), "qd_prev": torch.zeros(B, 29),
             "points": torch.randn(B, 128, 3) * .1, "partition": "train"}
    state["object_history"] = object_observation(torso_pose(kin, S), O)[:, None].expand(-1, 10, -1).clone()
    cfg = TrainConfig(experiment_kind="smoke", wm_hidden=(16, 16), actor_hidden=(16, 8),
                      critic_hidden=(12, 8), contact_hidden=8, horizon=H, wm_batch=B, pi_batch=B,
                      wm_batches=1, wm_only_iters=1, critic_epochs=1, critic_minibatches=2, amp="none")
    learner = Learner(Policy(cfg.actor_hidden), WorldModel(cfg.wm_hidden, cfg.contact_hidden),
                      Critic(cfg.critic_hidden), motion, kin, Reward(), cfg)
    return learner, state


def window(learner, start, K=3):
    data = {}
    B = len(start["f"])
    for key, value in start.items():
        if isinstance(value, torch.Tensor):
            data[key] = value[:, None].expand((B, K) + value.shape[1:]).clone()
    data["partition"] = "train"
    data["a"] = torch.randn(B, K, 29) * .01
    data["s_next"] = data["s"].clone()
    data["s_next"][..., 0] += .001
    data["o_next"] = data["o"].clone()
    data["o_next"][..., 0] += .001
    data["contact"] = torch.zeros(B, K, 5)
    data["contact_valid"] = torch.ones(B, K, 5, dtype=torch.bool)
    data["mask"] = torch.ones(B, K)
    data["mask"][0, -1] = 0
    return data


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_313_prefix_and_actuator_suffix(self):
        learner, start = fixture(H=3)
        S, O = unflatten_state(start["s"]), unflatten_object(start["o"])
        a = torch.zeros(2, 29)
        args = (S, O, a, normalize_xi(start["xi"]), start["xi_o"], start["points"])
        x = learner.wm.inputs(*args, start["actuator"], start["mass_context"])
        altered = start["actuator"].clone()
        altered[:, 29] = 4
        x2 = learner.wm.inputs(*args, altered, start["mass_context"])
        self.assertEqual(x.shape, (2, 313))
        self.assertTrue(torch.equal(x[:, :225], x2[:, :225]))
        self.assertFalse(torch.equal(x[:, 225:], x2[:, 225:]))
        S1, O1, contact = learner.wm.step(*args, altered, start["mass_context"])
        self.assertEqual(flatten_state(S1).shape, (2, 71))
        self.assertEqual(flatten_object(O1).shape, (2, 13))
        self.assertEqual(contact.shape, (2, 5))
        self.assertEqual(learner.wm.net[-1].out_features, 82)
        self.assertEqual(feature_schema()["wm_input_dim"], 313)

    def test_oracle_dim_no_base_and_shared_history_count(self):
        learner, start = fixture(H=3)
        oracle = learner.features(start)
        self.assertEqual(oracle.shape, (2, ORACLE_DIM))
        self.assertEqual(learner.pi.net[0].in_features, 3647 + 64)
        self.assertEqual(learner.pi(oracle, start["points"]).shape, (2, 29))
        learner.pi.update_normalizers(oracle)
        self.assertEqual(int(learner.pi.norm.object_history.count), 2)
        self.assertEqual(int(learner.pi.norm.robot.count), 2)
        before = {k: v.clone() for k, v in learner.pi.named_buffers()}
        learner.action(start, exploration=False, update_statistics=False)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in learner.pi.named_buffers()))

    def test_h32_bptt_and_analytic_actuator(self):
        learner, start = fixture(H=32)
        with frozen_parameters(learner.wm, learner.critic):
            trajectory = learner.imagine(start)
            (-trajectory["objective"]).backward()
        self.assertEqual(len(trajectory["actions"]), 32)
        self.assertTrue(torch.isfinite(trajectory["objective"]))
        for step, action in enumerate(trajectory["actions"]):
            expected = advance_actuator_state(trajectory["actuators"][step], action, start["xi"])
            self.assertTrue(torch.equal(expected, trajectory["actuators"][step + 1]))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in learner.pi.parameters()))
        self.assertTrue(all(p.grad is None for m in (learner.wm, learner.critic) for p in m.parameters()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in learner.pi.parameters() if p.grad is not None))

    def test_fixed_model_actor_direction_finite_difference(self):
        learner, start = fixture(H=3)
        # Double precision eliminates noisy float32 finite differences through FK.
        learner.pi.double(); learner.wm.double(); learner.critic.double()
        learner.motion.dtype = torch.float64
        start = {k: v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in start.items()}
        parameter = learner.pi.net[-1].bias
        direction = torch.linspace(-1, 1, 29, dtype=torch.float64)
        with frozen_parameters(learner.wm, learner.critic):
            objective = learner.imagine(start)["objective"]
            gradient, = torch.autograd.grad(objective, parameter)
            analytic = (gradient * direction).sum().item()
            epsilon = 1e-4
            original = parameter.detach().clone()
            with torch.no_grad(): parameter.copy_(original + epsilon * direction)
            positive = learner.imagine(start)["objective"].item()
            with torch.no_grad(): parameter.copy_(original - epsilon * direction)
            negative = learner.imagine(start)["objective"].item()
            with torch.no_grad(): parameter.copy_(original)
        numerical = (positive - negative) / (2 * epsilon)
        self.assertAlmostEqual(analytic, numerical, delta=max(1e-7, abs(analytic) * .01))

    def test_actor_gradient_requires_world_model_state_jacobian(self):
        learner, start = fixture(H=3)
        class DynamicsOnlyReward:
            def __call__(self, S, O, f, a, previous, contact, contact_valid=None):
                return O[0][..., 0]
            def soft_termination(self, S, O, f):
                return O[0][..., 0] * 0
        learner.reward = DynamicsOnlyReward()
        with torch.no_grad():
            for p in learner.critic.parameters(): p.zero_()
        with frozen_parameters(learner.wm, learner.critic):
            trajectory = learner.imagine(start)
            gradient, = torch.autograd.grad(trajectory["objective"], learner.pi.net[-1].bias)
        # Reward has no action regularizer or contact term and value is zero.
        # A nonzero derivative must pass through WM state integration.
        self.assertGreater(float(gradient.norm()), 1e-8)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_real_wm_update_and_direct_actor_critic_update(self):
        learner, start = fixture(H=4)
        data = window(learner, start)
        before_wm = learner.wm.net[-1].weight.detach().clone()
        report = learner.wm_update(data)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in report.values()))
        self.assertFalse(torch.equal(before_wm, learner.wm.net[-1].weight))
        before_pi = learner.pi.net[-1].weight.detach().clone()
        before_critic = learner.critic.shape_encoder.out[-1].weight.detach().clone()
        report = learner.policy_update(start)
        self.assertFalse(torch.equal(before_pi, learner.pi.net[-1].weight))
        self.assertFalse(torch.equal(before_critic, learner.critic.shape_encoder.out[-1].weight))
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in report.values()))

    def test_warmup_and_evaluation_protection(self):
        cfg = TrainConfig().validate()
        self.assertEqual(cfg.stage(500), "wm_only")
        self.assertEqual(cfg.stage(501), "actor_critic")
        with self.assertRaises(ValueError): replace(cfg, wm_only_iters=0).validate()
        learner, start = fixture(H=3)
        protected = {**start, "partition": "protected_eval"}
        with self.assertRaises(ValueError): learner.action(protected, update_statistics=True)
        with self.assertRaises(ValueError): learner.policy_update(protected)
        with self.assertRaises(ValueError): learner.wm_update({**window(learner, start), "partition": "protected_eval"})

    def test_warmup_does_not_update_actor_or_critic(self):
        learner, start = fixture(H=3)
        before_pi = {k: v.clone() for k, v in learner.pi.state_dict().items()}
        before_v = {k: v.clone() for k, v in learner.critic.state_dict().items()}
        report = learner.update(1, window=window(learner, start), start=start)
        self.assertEqual(report["stage"], "wm_only")
        self.assertTrue(all(torch.equal(before_pi[k], v) for k, v in learner.pi.state_dict().items()))
        self.assertTrue(all(torch.equal(before_v[k], v) for k, v in learner.critic.state_dict().items()))
        self.assertEqual(len(learner.optimizers["pi"].state), 0)
        self.assertEqual(len(learner.optimizers["critic"].state), 0)
        report = learner.update(2, window=window(learner, start), start=start)
        self.assertEqual(report["stage"], "actor_critic")
        self.assertGreater(len(learner.optimizers["pi"].state), 0)
        self.assertGreater(len(learner.optimizers["critic"].state), 0)

    def test_gradient_and_contact_sync_callbacks(self):
        learner, start = fixture(H=3)
        calls = []
        def gradients(module):
            self.assertTrue(any(p.grad is not None for p in module.parameters()))
            calls.append(module)
        stats = []
        def statistics(positive, negative):
            stats.append((positive.clone(), negative.clone()))
            return positive * 2, negative * 2
        learner.gradient_sync, learner.stats_sync = gradients, statistics
        data = window(learner, start)
        transition = {k: v[:, 0] if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        learner.observe(transition, append=False)
        self.assertEqual(len(stats), 1)
        self.assertTrue(torch.equal(learner.contact_negative, torch.full((5,), 4., dtype=torch.float64)))
        learner.wm_update(data)
        learner.policy_update(start)
        self.assertEqual(calls.count(learner.wm), 1)
        self.assertEqual(calls.count(learner.pi), 1)
        self.assertEqual(calls.count(learner.critic), 2)
        learner.action({**start, "partition": "protected_eval"}, exploration=False, update_statistics=False)
        self.assertEqual(len(stats), 1)
        self.assertEqual(len(calls), 4)

    def test_wm_diagnostics_are_read_only_and_reserved_windows_rejected(self):
        learner, start = fixture(H=3)
        data = {**window(learner, start, K=8), "wm_split": "holdout"}
        before = {k: v.clone() for k, v in learner.wm.state_dict().items()}
        report = learner.wm_validation(data)
        self.assertTrue(report["finite"])
        self.assertFalse(report["quality_gate"])
        self.assertEqual(set(report["horizons"]), {"1", "8"})
        self.assertEqual(len(report["contact_channels"]), 5)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in learner.wm.state_dict().items()))
        with self.assertRaises(ValueError): learner.wm_update(data)

    def test_wm_fit_statistics_exclude_reserved_rows(self):
        learner, start = fixture(H=3)
        from hoi_tables.replay import ReplayBuffer
        learner.buffer = ReplayBuffer(10, 2, holdout_fraction=.1)
        data = window(learner, start)
        transition = {k: v[:, 0] if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        transition["episode_id"] = torch.zeros(2, dtype=torch.long)
        learner.observe(transition, append=False)
        expected = int(learner.buffer.fit_mask(transition).sum())
        self.assertEqual(int(learner.wm.in_norm.count), expected)
        self.assertTrue(torch.equal(learner.contact_negative, torch.full((5,), float(expected), dtype=torch.float64)))


if __name__ == "__main__":
    unittest.main()
