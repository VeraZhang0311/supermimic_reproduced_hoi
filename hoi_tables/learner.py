"""Real transitions -> v1.1 WM -> differentiable direct teacher and value.

No native simulator, checkpoint base policy, PPO or residual action is used.
All oracle features are rebuilt from each imagined state. Only reference
targets and episode-constant physical context may come from the real start.
"""
import math
from contextlib import nullcontext
from time import perf_counter

import torch
from torch import nn

from supermimic.actuator_state import advance_actuator_state
from supermimic.mdp import clamp_action, normalize_xi
from supermimic.privileged import PrivilegedFeatures
from hoi_residual.features import oracle_features, torso_pose, object_observation
from .batched_fk import privileged_call
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


def _batches(values, size):
    length = len(values["f"])
    for first in range(0, length, size):
        last = min(first + size, length)
        yield ({key: value[first:last] if torch.is_tensor(value) else value
                for key, value in values.items()}, (last - first) / length)


class Learner:
    def __init__(self, policy, wm, critic, motion, kin, reward, config=None, *, shapes=None, replay=None,
                 gradient_sync=None, stats_sync=None, distributed_normalizers=False,
                 actor_backward_context_factory=None, asset_points=None, actor_microbatch=None,
                 wm_microbatch=None, profile_updates=False, compiled_actor=False):
        self.config = (config or TrainConfig()).validate()
        self.pi, self.wm, self.critic = policy, wm, critic
        self.motion, self.kin, self.reward = motion, kin, reward
        self.shapes, self.buffer = shapes, replay
        self.asset_points = asset_points
        self.actor_microbatch = actor_microbatch
        self.wm_microbatch = wm_microbatch
        self.profile_updates = profile_updates
        self._actor_step = torch.compile(self._imagine_step, dynamic=False) if compiled_actor else self._imagine_step
        self.gradient_sync, self.stats_sync = gradient_sync, stats_sync
        if actor_backward_context_factory is not None and not callable(actor_backward_context_factory):
            raise TypeError("actor_backward_context_factory must create a fresh context manager")
        self.actor_backward_context_factory = nullcontext if actor_backward_context_factory is None else actor_backward_context_factory
        # Distributed RunningNorm wrappers aggregate counts/moments and must
        # also participate for an empty local fit subset.
        self.distributed_normalizers = bool(distributed_normalizers)
        self.privileged = PrivilegedFeatures(kin, motion)
        self.torso_id = kin.body_names.index("torso_link") if "torso_link" in getattr(kin, "body_names", ()) else None
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

    def _shape_table(self, model):
        return None if self.asset_points is None else model.shape_encoder(self.asset_points)

    def _time(self):
        if self.profile_updates and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return perf_counter()

    def _oracle(self, S, O, f, a_prev, xi, q_prev, qd_prev, actuator, xi_o, history, *, bodies=None):
        if bodies is None:
            torso = torso_pose(self.kin, S)
        else:
            torso = bodies[0][..., self.torso_id, :], bodies[1][..., self.torso_id, :, :]
        if bodies is not None and hasattr(self.motion, "privileged_reference"):
            robot = privileged_call(self.privileged, S, f, a_prev, xi, q_prev, qd_prev, actuator, bodies=bodies)
        else:
            robot = self.privileged(S, f, a_prev, xi, q_prev, qd_prev, actuator)
        return oracle_features(robot, torso, O, self.motion, f, xi_o, history, check_finite=bodies is None)

    def _imagine_step(self, S, O, f, previous, xi, xi_n, q_prev, qd_prev, actuator, xi_o, history,
                      policy_shape, wm_shape, mass_context, bodies):
        oracle = self._oracle(S, O, f, previous, xi, q_prev, qd_prev, actuator, xi_o, history, bodies=bodies)
        action = clamp_action(self.pi(oracle, None, shape_embedding=policy_shape))
        next_state, next_object, logits = self.wm.step(S, O, action, xi_n, xi_o, None, actuator, mass_context,
                                                      shape_embedding=wm_shape)
        next_actuator = advance_actuator_state(actuator, action, xi)
        next_bodies = self.kin.forward(next_state)
        torso = next_bodies[0][..., self.torso_id, :], next_bodies[1][..., self.torso_id, :, :]
        observation = object_observation(torso, next_object)
        return next_state, next_object, logits, next_actuator, next_bodies, oracle, action, observation

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
        shape = self._shape_table(self.wm)
        points = self._points(window) if shape is None else None
        for tick in range(window["a"].shape[1]):
            active = window["mask"][:, tick].bool().nonzero().flatten()
            contact = window["a"].new_zeros((len(window["a"]), len(CHANNELS)))
            if len(active):
                point = None if points is None else (points[active, tick] if points.ndim == 4 else points[active])
                kwargs = {} if shape is None else {"shape_embedding": shape[window["asset_id"][active, tick]]}
                new_s, new_o, new_c = self.wm.step(
                    tuple(v[active] for v in S), tuple(v[active] for v in O), window["a"][active, tick],
                    normalize_xi(window["xi"][active, tick]), window["xi_o"][active, tick], point,
                    actuator[active], window["mass_context"][active, tick], **kwargs)
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

    def wm_update(self, window=None, *, report_assets=False):
        if window is None:
            if self.buffer is None:
                raise ValueError("WM update requires a window or replay buffer")
            window = self.buffer.sample_windows(self.config.wm_batch, self.config.wm_window, split="fit")
        self._train_only(window)
        if window.get("wm_split", "fit") != "fit":
            raise ValueError("Reserved WM diagnostic windows cannot update the WM")
        ratio = self.contact_negative / self.contact_positive.clamp_min(1)
        weights = torch.where(self.contact_positive > 0, ratio, torch.ones_like(ratio)).clamp(1e-3, 100).float()
        optimizer = self.optimizers["wm"]
        optimizer.zero_grad(set_to_none=True)
        metrics = {}
        length = window["a"].shape[1]
        asset_totals = torch.zeros(len(self.asset_points), device=self.device) if report_assets else None
        for part, fraction in _batches(window, self.wm_microbatch or len(window["f"])):
            predicted_s, predicted_o, logits = self.wm_rollout(part)
            diagnostics = {"return_window_totals": True} if report_assets else {}
            losses = self.wm.loss(predicted_s, predicted_o, unflatten_state(part["s_next"]),
                                  unflatten_object(part["o_next"]), logits, part["contact"],
                                  part["contact_valid"], part["mask"], pos_weight=weights, **diagnostics)
            if asset_totals is not None:
                asset_totals.index_add_(0, part["asset_id"][:, 0], losses.pop("window_total"))
            (losses["total"] * fraction).backward()
            for key, value in losses.items():
                metrics[key] = metrics.get(key, 0.) + float(value.detach()) * fraction / length
        if self.gradient_sync is not None:
            self.gradient_sync(self.wm)
        grad = nn.utils.clip_grad_norm_(self.wm.parameters(), 1.)
        if not bool(torch.isfinite(grad)):
            raise FloatingPointError("Nonfinite joint WM gradient norm")
        _finite(self.wm, "WM")
        optimizer.step()
        if asset_totals is not None:
            counts = window["asset_id"][:, 0].bincount(minlength=len(asset_totals))
            values = (asset_totals / (counts.clamp_min(1) * length)).tolist()
            metrics["asset_wm_total"] = {i: value for i, (value, count) in enumerate(zip(values, counts.tolist(), strict=True)) if count}
        return {**metrics, "wm_gradnorm": float(grad)}

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
        per_asset = {}
        if self.asset_points is not None and "32" in horizons:
            ids = window["asset_id"][:, 31]
            active = mask[:, 31].bool()
            square = (O[0][:, 31] - actual_o[0][:, 31]).square().sum(-1)
            sums = square.new_zeros(len(self.asset_points)).index_add_(0, ids[active], square[active])
            asset_counts = ids[active].bincount(minlength=len(sums))
            values = (sums / asset_counts.clamp_min(1)).sqrt().tolist()
            per_asset = {i: value for i, (value, count) in enumerate(zip(values, asset_counts.tolist(), strict=True)) if count}
        return {"horizons": horizons, "asset_h32_object_rmse_m": per_asset, "contact_channels": list(CHANNELS),
                "contact_valid_counts": valid.sum((0, 1)).tolist(),
                "contact_positive_counts": positive.sum((0, 1)).tolist(),
                "contact_recall": ((predicted & positive).sum((0, 1)) / positive.sum((0, 1)).clamp_min(1)).tolist(),
                "contact_precision": ((predicted & positive).sum((0, 1)) / (predicted & valid).sum((0, 1)).clamp_min(1)).tolist(),
                "contact_brier": (((probability - window["contact"]).square() * valid).sum((0, 1)) / counts).tolist(),
                "finite": bool(torch.isfinite(error).all() & torch.isfinite(logits).all()),
                "quality_gate": False}

    def imagine(self, start, *, fixed_asset=None):
        self._train_only(start)
        c = self.config
        S, O = unflatten_state(start["s"]), unflatten_object(start["o"])
        xi, xi_o, f, previous = start["xi"], start["xi_o"], start["f"], start["a_prev"]
        q_prev, qd_prev, actuator = start["q_prev"], start["qd_prev"], start["actuator"]
        history, points = start["object_history"], self._points(start)
        mass_context, xi_n = start["mass_context"], normalize_xi(xi)
        states, objects, frames, actuators = [S], [O], [f], [actuator]
        actions, rewards, failures, contacts, oracles, intervals = [], [], [], [], [], []
        policy_shape = self._shape_table(self.pi)
        if policy_shape is None:
            policy_shape = self.pi.shape_encoder(points)
        else:
            policy_shape = policy_shape[start["asset_id"]]
        wm_shape = self._shape_table(self.wm)
        wm_kwargs = {} if wm_shape is None else {"shape_embedding": wm_shape[start["asset_id"]]}
        reward = self.reward if fixed_asset is None else self.reward.by_asset[fixed_asset]
        bodies = self.kin.forward(S) if self.asset_points is not None else None
        for _ in range(c.horizon):
            old_s = S
            if bodies is None:
                oracle = self._oracle(S, O, f, previous, xi, q_prev, qd_prev, actuator, xi_o, history)
                action = clamp_action(self.pi(oracle, points, shape_embedding=policy_shape))
                S, O, logits = self.wm.step(S, O, action, xi_n, xi_o, points, actuator, mass_context)
                actuator = advance_actuator_state(actuator, action, xi)
                observation = object_observation(torso_pose(self.kin, S), O)
            else:
                S, O, logits, actuator, bodies, oracle, action, observation = self._actor_step(
                    S, O, f, previous, xi, xi_n, q_prev, qd_prev, actuator, xi_o, history,
                    policy_shape, wm_kwargs["shape_embedding"], mass_context, bodies)
            oracles.append(oracle)
            end = self.motion.clip_end[self.motion.clip_id[f]]
            has_interval = f < end
            f = torch.minimum(f + 1, end)
            probabilities = logits.sigmoid()
            intervals.append(has_interval)
            if fixed_asset is None:
                rewards.append(reward(S, O, f, action, previous, probabilities, torch.ones_like(probabilities)))
                failure = reward.soft_termination(S, O, f)
                failures.append(1 - (1 - failure) * has_interval.to(failure))
            history = torch.cat((history[:, 1:], observation[:, None]), 1)
            q_prev, qd_prev, previous = old_s[4], old_s[5], action
            states.append(S)
            objects.append(O)
            frames.append(f)
            actuators.append(actuator)
            actions.append(action)
            contacts.append(probabilities)
        oracles.append(self._oracle(S, O, f, previous, xi, q_prev, qd_prev, actuator, xi_o, history, bodies=bodies))
        if fixed_asset is None:
            reward, failure = torch.stack(rewards), torch.stack(failures)
        else:
            # Reward and survival are outputs, so evaluate all H32 intervals in a few
            # chunks after imagination without changing the rollout recurrence.
            previous_actions = [start["a_prev"], *actions[:-1]]
            for first in range(0, c.horizon, 8):
                last = min(first + 8, c.horizon)
                state, obj = _flat(_stack(states[first + 1:last + 1])), _flat(_stack(objects[first + 1:last + 1]))
                step = torch.cat(actions[first:last])
                prev = torch.cat(previous_actions[first:last])
                frame, contact = torch.cat(frames[first + 1:last + 1]), torch.cat(contacts[first:last])
                values = reward(state, obj, frame, step, prev, contact, torch.ones_like(contact))
                fail = reward.soft_termination(state, obj, frame)
                rewards.append(values.reshape(last - first, -1))
                failures.append((1 - (1 - fail) * torch.cat(intervals[first:last]).to(fail)).reshape(last - first, -1))
            reward, failure = torch.cat(rewards), torch.cat(failures)
        if bodies is not None and not all(bool(torch.isfinite(value).all()) for value in (torch.stack(oracles), reward, failure)):
            raise FloatingPointError("Nonfinite imagined HOI oracle")
        alive = torch.cumprod(1 - failure, 0)
        survival = torch.cat((torch.ones_like(alive[:1]), alive[:-1]), 0)
        discounts = c.gamma ** torch.arange(c.horizon, device=self.device, dtype=reward.dtype)
        returns = (discounts[:, None] * survival * reward).sum(0)
        critic_shape = self._shape_table(self.critic)
        critic_kwargs = {} if critic_shape is None else {"shape_embedding": critic_shape[start["asset_id"]]}
        returns = returns + c.gamma ** c.horizon * alive[-1] * self.critic(oracles[-1], points, **critic_kwargs)
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
        assets = None if self.asset_points is None else start["asset_id"].repeat(H + 1)
        points = (trajectory["points"].detach()[None].expand(H + 1, -1, -1, -1).flatten(0, 1)
                  if assets is None else None)

        def evaluate(ids, embeddings):
            kwargs = {} if embeddings is None else {"shape_embedding": embeddings[assets[ids]]}
            return self.critic(raw[ids], None if points is None else points[ids], **kwargs)

        with torch.no_grad():
            chunk = max(1, math.ceil(H * B / c.critic_minibatches))
            embeddings = self._shape_table(self.critic)
            values = torch.cat([evaluate(slice(i, i + chunk), embeddings)
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
                loss = (evaluate(ids, self._shape_table(self.critic)) - targets[ids]).square().mean()
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
        actor_started = self._time() if self.profile_updates else None
        # Only local forward/BPTT may be serialized among ranks sharing a GPU.
        # Forward reads RMS; no normalizer/statistics or gradient collective is
        # allowed here. Each rank retains its complete local batch and H32.
        with self.actor_backward_context_factory():
            with frozen_parameters(self.wm, self.critic):
                optimizer.zero_grad(set_to_none=True)
                parts, loss = [], 0.
                if self.asset_points is None or not hasattr(self.reward, "by_asset"):
                    batches = ((batch, fraction, None) for batch, fraction in _batches(start, self.actor_microbatch or len(start["f"])))
                else:
                    order = start["asset_id"].argsort(stable=True)
                    original_order = order.argsort()
                    counts = start["asset_id"].bincount(minlength=len(self.asset_points)).tolist()
                    groups = ({key: value[ids] if torch.is_tensor(value) else value for key, value in start.items()}
                              for ids in order.split(counts))
                    batches = ((batch, fraction * count / len(order), asset) for asset, (group, count) in enumerate(zip(groups, counts))
                               if count for batch, fraction in _batches(group, self.actor_microbatch or count))
                for batch, fraction, asset in batches:
                    trajectory = self.imagine(batch) if asset is None else self.imagine(batch, fixed_asset=asset)
                    local_loss = -trajectory["objective"] * fraction
                    local_loss.backward()
                    loss += float(local_loss.detach())
                    parts.append({"actions": [None] * len(trajectory["actions"]),
                                  "oracles": [value.detach() for value in trajectory["oracles"]],
                                  **{key: trajectory[key].detach() for key in ("reward", "failure", "points", "terminal_survival")}})
                    del trajectory, local_loss
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
        trajectory = parts[0]
        if len(parts) > 1:
            trajectory = {"actions": trajectory["actions"],
                          "oracles": [torch.cat(values) for values in zip(*(part["oracles"] for part in parts))],
                          **{key: torch.cat([part[key] for part in parts], dim=1 if key in ("reward", "failure") else 0)
                             for key in ("reward", "failure", "points", "terminal_survival")}}
        if self.asset_points is not None and hasattr(self.reward, "by_asset"):
            trajectory["oracles"] = [value[original_order] for value in trajectory["oracles"]]
            for key in ("reward", "failure"):
                trajectory[key] = trajectory[key][:, original_order]
            for key in ("points", "terminal_survival"):
                trajectory[key] = trajectory[key][original_order]
        actor_seconds = self._time() - actor_started if actor_started is not None else None
        critic_started = self._time() if self.profile_updates else None
        critic_loss = self.critic_update(trajectory, start)
        timing = {} if not self.profile_updates else {"actor_seconds": actor_seconds, "critic_seconds": self._time() - critic_started}
        return {"pi_loss": loss, "v_loss": critic_loss, "pi_gradnorm": float(grad), **timing,
                "imag_return_per_step": float(trajectory["reward"].detach().mean()),
                "imag_survival": float(trajectory["terminal_survival"].detach().mean())}

    def update(self, iteration, *, window=None, start=None):
        stage = self.config.stage(iteration)
        rate = self.config.lr_at(iteration)
        for optimizer in self.optimizers.values():
            for group in optimizer.param_groups:
                group["lr"] = rate
        result = {"iteration": iteration, "stage": stage, "lr": rate}
        started = self._time() if self.profile_updates else None
        for batch in range(self.config.wm_batches):
            report_assets = self.asset_points is not None and batch + 1 == self.config.wm_batches
            wm_metrics = self.wm_update(window, report_assets=True) if report_assets else self.wm_update(window)
            if report_assets:
                result["asset_wm_total"] = wm_metrics.pop("asset_wm_total")
            result.update({"wm_" + key: value for key, value in wm_metrics.items()})
        if started is not None:
            result["wm_seconds"] = self._time() - started
        if stage == "actor_critic":
            result.update(self.policy_update(start))
        self.iteration = iteration
        return result
