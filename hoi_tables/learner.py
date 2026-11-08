"""Real transitions -> v1.1 WM -> differentiable direct teacher and value.

No native simulator, checkpoint base policy, PPO or residual action is used.
All oracle features are rebuilt from each imagined state. Only reference
targets and episode-constant physical context may come from the real start.
"""
import math
from contextlib import nullcontext

import torch
from torch import nn

from supermimic.actuator_state import advance_actuator_state
from supermimic.mdp import clamp_action, normalize_xi
from supermimic.privileged import PrivilegedFeatures
from hoi_residual.features import oracle_features, torso_pose, object_observation
from .config import TrainConfig
from .networks import frozen_parameters
from .v11.contacts import CHANNELS
from .v11.state import unflatten_state, unflatten_object


def _stack(states):
    return tuple(torch.stack([s[i] for s in states]) for i in range(len(states[0])))


def _flat(states):
    return tuple(s.flatten(0, 1) for s in states)


def _finite(module, label):
    if not all(bool(torch.isfinite(p.grad).all()) for p in module.parameters() if p.grad is not None):
        raise FloatingPointError(f"Nonfinite {label} gradient")


class Learner:
    def __init__(self, policy, wm, critic, motion, kin, reward, config=None, *, shapes=None, replay=None,
                 gradient_sync=None, stats_sync=None, distributed_normalizers=False,
                 actor_backward_context_factory=None):
        self.config = (config or TrainConfig()).validate()
        self.pi, self.wm, self.critic = policy, wm, critic
        self.motion, self.kin, self.reward = motion, kin, reward
        self.shapes, self.buffer = shapes, replay
        self.gradient_sync, self.stats_sync = gradient_sync, stats_sync
        if actor_backward_context_factory is not None and not callable(actor_backward_context_factory):
            raise TypeError("actor_backward_context_factory must create a fresh context manager")
        self.actor_backward_context_factory = nullcontext if actor_backward_context_factory is None else actor_backward_context_factory
        # Distributed RunningNorm wrappers aggregate counts/moments and must
        # also participate for an empty local fit subset.
        self.distributed_normalizers = bool(distributed_normalizers)
        self.privileged = PrivilegedFeatures(kin, motion)
        self.device = next(policy.parameters()).device
        self.optimizers = {"pi": torch.optim.Adam(policy.parameters(), lr=self.config.lr, betas=(.9, .95)),
                           "wm": torch.optim.Adam(wm.parameters(), lr=self.config.lr),
                           "critic": torch.optim.Adam(critic.parameters(), lr=self.config.lr)}
        self.contact_positive = torch.zeros(len(CHANNELS), device=self.device, dtype=torch.float64)
        self.contact_negative = torch.zeros_like(self.contact_positive)
        self.iteration = 0

    def _train_only(self, values):
        partition = values.get("partition", getattr(self.motion, "partition", "train"))
        if partition != "train" or not getattr(self.motion, "for_training", True):
            raise ValueError("Protected evaluation data cannot update models, statistics or training weights")

    def _points(self, values):
        if "points" in values:
            return values["points"]
        if self.shapes is None:
            raise ValueError("Batch must contain metric FPS128 points or learner needs a shape bank")
        return self.shapes.points(values["asset_id"])

    def _oracle(self, S, O, f, a_prev, xi, q_prev, qd_prev, actuator, xi_o, history):
        robot = self.privileged(S, f, a_prev, xi, q_prev, qd_prev, actuator)
        return oracle_features(robot, torso_pose(self.kin, S), O, self.motion, f, xi_o, history)

    def features(self, values):
        return self._oracle(unflatten_state(values["s"]), unflatten_object(values["o"]),
                            values["f"], values["a_prev"], values["xi"], values["q_prev"],
                            values["qd_prev"], values["actuator"], values["xi_o"], values["object_history"])

    @torch.no_grad()
    def action(self, snapshot, *, exploration=True, update_statistics=True):
        oracle = self.features(snapshot)
        if update_statistics:
            self._train_only(snapshot)
            self.pi.update_normalizers(oracle)
            self.critic.update_normalizers(oracle)
        action = self.pi(oracle, self._points(snapshot))
        if exploration:
            action = action + self.config.action_noise * torch.randn_like(action)
        return clamp_action(action)

    @torch.no_grad()
    def observe(self, transition, *, update_statistics=True, append=True):
        self._train_only(transition)
        if update_statistics:
            fit = self.buffer.fit_mask(transition) if self.buffer is not None else torch.ones(
                len(transition["s"]), device=transition["s"].device, dtype=torch.bool)
            S, O = unflatten_state(transition["s"][fit]), unflatten_object(transition["o"][fit])
            S1, O1 = unflatten_state(transition["s_next"][fit]), unflatten_object(transition["o_next"][fit])
            self.wm.update_normalizers(S, O, transition["a"][fit], normalize_xi(transition["xi"][fit]),
                                       transition["xi_o"][fit], self._points(transition)[fit], S1, O1,
                                       allow_empty=self.distributed_normalizers)
            valid = transition["contact_valid"].bool() & fit[:, None]
            positive = ((transition["contact"] > .5) & valid).sum(0).to(self.contact_positive)
            negative = ((transition["contact"] <= .5) & valid).sum(0).to(self.contact_negative)
            if self.stats_sync is not None:
                positive, negative = self.stats_sync(positive, negative)
            self.contact_positive += positive
            self.contact_negative += negative
        if append and self.buffer is not None:
            self.buffer.append(transition)

    def wm_rollout(self, window):
        S, O = unflatten_state(window["s"][:, 0]), unflatten_object(window["o"][:, 0])
        actuator = window["actuator"][:, 0]
        states, objects, logits = [], [], []
        points = self._points(window)
        for tick in range(window["a"].shape[1]):
            active = window["mask"][:, tick].bool().nonzero().flatten()
            contact = window["a"].new_zeros((len(window["a"]), len(CHANNELS)))
            if len(active):
                point = points[active, tick] if points.ndim == 4 else points[active]
                new_s, new_o, new_c = self.wm.step(
                    tuple(v[active] for v in S), tuple(v[active] for v in O), window["a"][active, tick],
                    normalize_xi(window["xi"][active, tick]), window["xi_o"][active, tick], point,
                    actuator[active], window["mass_context"][active, tick])
                S = tuple(previous.index_copy(0, active, v) for previous, v in zip(S, new_s))
                O = tuple(previous.index_copy(0, active, v) for previous, v in zip(O, new_o))
                new_actuator = advance_actuator_state(actuator[active], window["a"][active, tick], window["xi"][active, tick])
                actuator = actuator.index_copy(0, active, new_actuator)
                contact = contact.index_copy(0, active, new_c)
            states.append(S)
            objects.append(O)
            logits.append(contact)
        transpose = lambda states: tuple(v.transpose(0, 1) for v in _stack(states))
        return transpose(states), transpose(objects), torch.stack(logits, 1)

    def wm_update(self, window=None):
        if window is None:
            if self.buffer is None:
                raise ValueError("WM update requires a window or replay buffer")
            window = self.buffer.sample_windows(self.config.wm_batch, self.config.wm_window, split="fit")
        self._train_only(window)
        if window.get("wm_split", "fit") != "fit":
            raise ValueError("Reserved WM diagnostic windows cannot update the WM")
        predicted_s, predicted_o, logits = self.wm_rollout(window)
        ratio = self.contact_negative / self.contact_positive.clamp_min(1)
        weights = torch.where(self.contact_positive > 0, ratio, torch.ones_like(ratio)).clamp(1e-3, 100).float()
        losses = self.wm.loss(predicted_s, predicted_o, unflatten_state(window["s_next"]),
                              unflatten_object(window["o_next"]), logits, window["contact"],
                              window["contact_valid"], window["mask"], pos_weight=weights)
        optimizer = self.optimizers["wm"]
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        if self.gradient_sync is not None:
            self.gradient_sync(self.wm)
        grad = nn.utils.clip_grad_norm_(self.wm.parameters(), 1.)
        if not bool(torch.isfinite(grad)):
            raise FloatingPointError("Nonfinite joint WM gradient norm")
        _finite(self.wm, "WM")
        optimizer.step()
        length = window["a"].shape[1]
        return {**{key: float(value.detach()) / length for key, value in losses.items()}, "wm_gradnorm": float(grad)}

    @torch.no_grad()
    def wm_validation(self, window):
        """Read-only autoregressive diagnostics, without a policy quality gate."""
        S, O, logits = self.wm_rollout(window)
        actual_s, actual_o = unflatten_state(window["s_next"]), unflatten_object(window["o_next"])
        mask = window["mask"]
        error = self.wm.state_error(S, O, actual_s, actual_o)
        valid = window["contact_valid"].bool() & mask[..., None].bool()
        probability = logits.sigmoid()
        positive = (window["contact"] > .5) & valid
        predicted = probability >= .5
        counts = valid.sum((0, 1)).clamp_min(1)
        horizons = {}
        for horizon in (1, 8, 32):
            if horizon > mask.shape[1]:
                continue
            active = mask[:, horizon - 1]
            squared = error[:, horizon - 1].square()
            average = lambda value: float((value * active).sum() / active.sum().clamp_min(1))
            horizons[str(horizon)] = {
                "valid_samples": int(active.sum()),
                "robot_mse": average(squared[:, :70].mean(-1)),
                "object_mse": average(squared[:, 70:].mean(-1)),
                "object_position_rmse_m": math.sqrt(average((O[0][:, horizon - 1] - actual_o[0][:, horizon - 1]).square().sum(-1))),
            }
        return {"horizons": horizons, "contact_channels": list(CHANNELS),
                "contact_valid_counts": valid.sum((0, 1)).tolist(),
                "contact_positive_counts": positive.sum((0, 1)).tolist(),
                "contact_recall": ((predicted & positive).sum((0, 1)) / positive.sum((0, 1)).clamp_min(1)).tolist(),
                "contact_precision": ((predicted & positive).sum((0, 1)) / (predicted & valid).sum((0, 1)).clamp_min(1)).tolist(),
                "contact_brier": (((probability - window["contact"]).square() * valid).sum((0, 1)) / counts).tolist(),
                "finite": bool(torch.isfinite(error).all() & torch.isfinite(logits).all()),
                "quality_gate": False}

    def imagine(self, start):
        self._train_only(start)
        c = self.config
        S, O = unflatten_state(start["s"]), unflatten_object(start["o"])
        xi, xi_o, f, previous = start["xi"], start["xi_o"], start["f"], start["a_prev"]
        q_prev, qd_prev, actuator = start["q_prev"], start["qd_prev"], start["actuator"]
        history, points = start["object_history"], self._points(start)
        mass_context, xi_n = start["mass_context"], normalize_xi(xi)
        states, objects, frames, actuators = [S], [O], [f], [actuator]
        actions, rewards, failures, contacts, oracles = [], [], [], [], []
        policy_shape = self.pi.shape_encoder(points)
        for _ in range(c.horizon):
            oracle = self._oracle(S, O, f, previous, xi, q_prev, qd_prev, actuator, xi_o, history)
            oracles.append(oracle)
            action = clamp_action(self.pi(oracle, points, shape_embedding=policy_shape))
            old_s = S
            S, O, logits = self.wm.step(S, O, action, xi_n, xi_o, points, actuator, mass_context)
            actuator = advance_actuator_state(actuator, action, xi)
            end = self.motion.clip_end[self.motion.clip_id[f]]
            has_interval = f < end
            f = torch.minimum(f + 1, end)
            probabilities = logits.sigmoid()
            rewards.append(self.reward(S, O, f, action, previous, probabilities, torch.ones_like(probabilities)))
            failure = self.reward.soft_termination(S, O, f)
            failures.append(1 - (1 - failure) * has_interval.to(failure))
            observation = object_observation(torso_pose(self.kin, S), O)
            history = torch.cat((history[:, 1:], observation[:, None]), 1)
            q_prev, qd_prev, previous = old_s[4], old_s[5], action
            states.append(S)
            objects.append(O)
            frames.append(f)
            actuators.append(actuator)
            actions.append(action)
            contacts.append(probabilities)
        oracles.append(self._oracle(S, O, f, previous, xi, q_prev, qd_prev, actuator, xi_o, history))
        reward, failure = torch.stack(rewards), torch.stack(failures)
        alive = torch.cumprod(1 - failure, 0)
        survival = torch.cat((torch.ones_like(alive[:1]), alive[:-1]), 0)
        discounts = c.gamma ** torch.arange(c.horizon, device=self.device, dtype=reward.dtype)
        returns = (discounts[:, None] * survival * reward).sum(0)
        returns = returns + c.gamma ** c.horizon * alive[-1] * self.critic(oracles[-1], points)
        return {"objective": returns.mean(), "reward": reward, "failure": failure, "survival": survival,
                "terminal_survival": alive[-1], "states": states, "objects": objects,
                "frames": frames, "actions": actions, "contact": contacts, "actuators": actuators,
                "oracles": oracles, "points": points}

    def critic_update(self, trajectory, start):
        self._train_only(start)
        c = self.config
        H, B = len(trajectory["actions"]), len(start["f"])
        # Raw features only are cached. Value fitting recomputes its PointNet
        # in every minibatch, so its own encoder receives gradients.
        raw = torch.stack([value.detach() for value in trajectory["oracles"]]).flatten(0, 1)
        points = trajectory["points"].detach()[None].expand(H + 1, -1, -1, -1).flatten(0, 1)
        with torch.no_grad():
            chunk = max(1, math.ceil(H * B / c.critic_minibatches))
            values = torch.cat([self.critic(raw[i:i + chunk], points[i:i + chunk])
                                for i in range(0, len(raw), chunk)]).reshape(H + 1, B)
            target, targets = values[-1], []
            for tick in range(H - 1, -1, -1):
                target = trajectory["reward"][tick].detach() + c.gamma * (1 - trajectory["failure"][tick].detach()) * (
                    (1 - c.lam) * values[tick + 1] + c.lam * target)
                targets.append(target)
            targets = torch.stack(targets[::-1]).flatten()
        size, total, count = H * B, 0., 0
        for _ in range(c.critic_epochs):
            for ids in torch.randperm(size, device=self.device).tensor_split(c.critic_minibatches):
                if not len(ids):
                    continue
                loss = (self.critic(raw[ids], points[ids]) - targets[ids]).square().mean()
                optimizer = self.optimizers["critic"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.gradient_sync is not None:
                    self.gradient_sync(self.critic)
                grad = nn.utils.clip_grad_norm_(self.critic.parameters(), 1.)
                if not bool(torch.isfinite(grad)):
                    raise FloatingPointError("Nonfinite critic gradient norm")
                _finite(self.critic, "critic")
                optimizer.step()
                total += float(loss.detach())
                count += 1
        return total / max(1, count)

    def policy_update(self, start=None):
        if start is None:
            if self.buffer is None:
                raise ValueError("Policy update requires starts or replay")
            start = self.buffer.sample_starts(self.config.pi_batch, horizon=self.config.horizon, motion=self.motion)
        self._train_only(start)
        optimizer = self.optimizers["pi"]
        self.optimizers["wm"].zero_grad(set_to_none=True)
        self.optimizers["critic"].zero_grad(set_to_none=True)
        # Only local forward/BPTT may be serialized among ranks sharing a GPU.
        # Forward reads RMS; no normalizer/statistics or gradient collective is
        # allowed here. Each rank retains its complete local batch and H32.
        with self.actor_backward_context_factory():
            with frozen_parameters(self.wm, self.critic):
                trajectory = self.imagine(start)
                loss = -trajectory["objective"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(p.grad is not None for module in (self.wm, self.critic) for p in module.parameters()):
                    raise RuntimeError("Actor BPTT must not populate WM or critic parameter gradients")
        # All ranks finish local BPTT before the same global gradient mean,
        # clipping and one Adam step. Serialization never changes the update.
        if self.gradient_sync is not None:
            self.gradient_sync(self.pi)
        grad = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.)
        nn.utils.clip_grad_value_(self.pi.parameters(), .1)
        if not bool(torch.isfinite(grad)):
            raise FloatingPointError("Nonfinite actor BPTT gradient norm")
        _finite(self.pi, "actor")
        optimizer.step()
        critic_loss = self.critic_update(trajectory, start)
        return {"pi_loss": float(loss.detach()), "v_loss": critic_loss, "pi_gradnorm": float(grad),
                "imag_return_per_step": float(trajectory["reward"].detach().mean()),
                "imag_survival": float(trajectory["terminal_survival"].detach().mean())}

    def update(self, iteration, *, window=None, start=None):
        stage = self.config.stage(iteration)
        rate = self.config.lr_at(iteration)
        for optimizer in self.optimizers.values():
            for group in optimizer.param_groups:
                group["lr"] = rate
        result = {"iteration": iteration, "stage": stage, "lr": rate}
        for _ in range(self.config.wm_batches):
            result.update({"wm_" + key: value for key, value in self.wm_update(window).items()})
        if stage == "actor_critic":
            result.update(self.policy_update(start))
        self.iteration = iteration
        return result
