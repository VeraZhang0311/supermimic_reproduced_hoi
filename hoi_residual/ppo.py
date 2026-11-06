"""Direct on-policy residual PPO; no replay buffer, WM or distillation loss.

PPO/GAE follows the clipped surrogate design used in ResMimic. Learning rate is
fixed here (unlike upstream adaptive KL); the exact choice is saved in config.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int = 24
    epochs: int = 5
    minibatches: int = 4
    learning_rate: float = 2e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    value_coef: float = 1.0
    entropy_coef: float = 0.005
    max_grad_norm: float = 1.0
    learning_rate_schedule: str = "fixed"

    def __post_init__(self):
        if min(self.rollout_steps, self.epochs, self.minibatches) <= 0:
            raise ValueError("PPO rollout/epoch/minibatch counts must be positive")
        if self.learning_rate_schedule != "fixed":
            raise ValueError("This PPO implementation explicitly uses fixed learning rate")
        if not 0 <= self.gamma <= 1 or not 0 <= self.lam <= 1 or not 0 < self.clip < 1:
            raise ValueError("Invalid PPO discount/lambda/clip")
        if any(not math.isfinite(v) or v < 0 for v in
               (self.learning_rate, self.value_coef, self.entropy_coef, self.max_grad_norm)):
            raise ValueError("PPO coefficients must be finite and nonnegative")
        if self.learning_rate == 0 or self.max_grad_norm == 0:
            raise ValueError("PPO learning rate and gradient limit must be positive")


@dataclass
class RolloutBatch:
    oracle: torch.Tensor
    base_actions: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    next_values: torch.Tensor
    normalizer_revision: int

    def validate(self):
        if self.rewards.ndim != 2 or min(self.rewards.shape) == 0:
            raise ValueError("PPO rollout must have nonempty [time, environment] dimensions")
        shape = self.rewards.shape
        for name in ("log_probs", "values", "terminated", "truncated", "next_values"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"PPO rollout shape mismatch: {name}")
        for name in ("oracle", "base_actions", "actions"):
            value = getattr(self, name)
            if value.ndim != 3 or value.shape[:2] != shape:
                raise ValueError(f"PPO rollout shape mismatch: {name}")
        if self.base_actions.shape != self.actions.shape:
            raise ValueError("Base and sampled residual action dimensions differ")
        for name in ("terminated", "truncated"):
            if getattr(self, name).dtype != torch.bool:
                raise ValueError(f"PPO {name} flags must be boolean")
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                if value.device != self.rewards.device:
                    raise ValueError("PPO rollout tensors must share a device")
                if value.requires_grad:
                    raise ValueError("PPO rollout tensors must be detached")
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(f"Nonfinite PPO rollout: {name}")


@torch.no_grad()
def compute_gae(rewards, values, next_values, terminated, truncated, gamma=0.99, lam=0.95):
    """Bootstrap time limits from FINAL observations and stop cross-reset traces.

next_values[t] is V(s[t+1]) before any automatic reset. A true termination
disables its bootstrap even if a time limit coincides. Both kinds of episode
boundary stop GAE propagation into the following episode.
    """
    if rewards.ndim != 2 or any(x.shape != rewards.shape for x in (values, next_values, terminated, truncated)):
        raise ValueError("GAE requires matching [time, environment] tensors")
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise ValueError("GAE episode boundary flags must be boolean")
    advantages = torch.empty_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        bootstrap = torch.where(terminated[t], 0.0, next_values[t])
        delta = rewards[t] + gamma * bootstrap - values[t]
        continuation = ~(terminated[t] | truncated[t])
        carry = delta + gamma * lam * torch.where(continuation, carry, 0.0)
        advantages[t] = carry
    return advantages, advantages + values


class PPO:
    def __init__(self, actor_critic, config=None):
        self.actor_critic = actor_critic
        self.config = config or PPOConfig()
        parameters = [p for p in actor_critic.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("PPO requires trainable residual actor/critic parameters")
        self.optimizer = torch.optim.Adam(parameters, lr=self.config.learning_rate)
        self.updates = 0

    def update(self, rollout: RolloutBatch):
        rollout.validate()
        if rollout.normalizer_revision != self.actor_critic.normalizer_revision:
            raise RuntimeError("Residual normalizer changed after rollout collection")
        norm_before = {key: value.clone() for key, value in self.actor_critic.norm.state_dict().items()}
        cfg = self.config
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        if distributed:
            signature = torch.tensor([*rollout.rewards.shape, rollout.oracle.shape[-1],
                                      rollout.actions.shape[-1], cfg.epochs, cfg.minibatches,
                                      rollout.normalizer_revision], device=rollout.rewards.device)
            low, high = signature.clone(), signature.clone()
            dist.all_reduce(low, op=dist.ReduceOp.MIN)
            dist.all_reduce(high, op=dist.ReduceOp.MAX)
            if not torch.equal(low, high):
                raise ValueError("Distributed PPO requires equal rollout and optimizer step shapes on all ranks")
        advantages, returns = compute_gae(rollout.rewards, rollout.values, rollout.next_values,
                                          rollout.terminated, rollout.truncated, cfg.gamma, cfg.lam)
        advantages = advantages.flatten()
        if distributed:
            moments = torch.stack((advantages.double().sum(), advantages.double().square().sum(),
                                   advantages.new_tensor(advantages.numel(), dtype=torch.float64)))
            dist.all_reduce(moments)
            mean = moments[0] / moments[2]
            std = (moments[1] / moments[2] - mean.square()).clamp_min(0).sqrt().clamp_min(1e-8)
            advantages = ((advantages.double() - mean) / std).to(advantages.dtype)
        else:
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        fields = {name: getattr(rollout, name).flatten(0, 1) for name in
                  ("oracle", "base_actions", "actions", "log_probs", "values")}
        returns = returns.flatten()
        count = advantages.numel()
        if cfg.minibatches > count:
            raise ValueError("PPO minibatches exceed the number of rollout samples")
        totals = {key: 0.0 for key in ("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm")}
        sample_count = 0
        for _ in range(cfg.epochs):
            permutation = torch.randperm(count, device=advantages.device)
            for indices in torch.tensor_split(permutation, cfg.minibatches):
                result = self.actor_critic.evaluate(fields["oracle"][indices], fields["base_actions"][indices], fields["actions"][indices])
                log_ratio = result["log_prob"] - fields["log_probs"][indices]
                ratio = log_ratio.exp()
                surrogate = torch.minimum(ratio * advantages[indices],
                                          ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * advantages[indices])
                policy_loss = -surrogate.mean()
                value = result["value"]
                value_clipped = fields["values"][indices] + (value - fields["values"][indices]).clamp(-cfg.clip, cfg.clip)
                value_loss = torch.maximum((value - returns[indices]).square(), (value_clipped - returns[indices]).square()).mean()
                entropy = result["entropy"].mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if distributed:
                    # Same number of samples/minibatches on every rank, so
                    # averaging gradients exactly matches equal-asset PPO.
                    for parameter in self.actor_critic.parameters():
                        if parameter.requires_grad:
                            if parameter.grad is None:
                                raise RuntimeError("Distributed PPO encountered an unused trainable parameter")
                            dist.all_reduce(parameter.grad)
                            parameter.grad.div_(world_size)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
                self.optimizer.step()
                with torch.no_grad():
                    metrics = {"loss": loss, "policy_loss": policy_loss, "value_loss": value_loss,
                               "entropy": entropy, "approx_kl": (ratio - 1 - log_ratio).mean(),
                               "clip_fraction": ((ratio - 1).abs() > cfg.clip).float().mean(), "grad_norm": grad_norm}
                    for key, value in metrics.items():
                        totals[key] += float(value) * len(indices)
                    sample_count += len(indices)
        if any(not torch.equal(value, norm_before[key]) for key, value in self.actor_critic.norm.state_dict().items()):
            raise RuntimeError("PPO modified residual normalizer during optimizer epochs")
        if any(not bool(torch.isfinite(value).all()) for value in self.actor_critic.state_dict().values()):
            raise FloatingPointError("Nonfinite residual actor/critic after PPO update")
        return_mean, reward_mean = float(returns.mean()), float(rollout.rewards.mean())
        if distributed:
            packed = torch.tensor([*totals.values(), sample_count, return_mean, reward_mean],
                                  device=rollout.rewards.device, dtype=torch.float64)
            dist.all_reduce(packed)
            totals = dict(zip(totals, packed[:len(totals)].tolist()))
            sample_count = float(packed[-3])
            return_mean, reward_mean = (packed[-2:] / world_size).tolist()
        self.updates += 1
        return {**{key: value / sample_count for key, value in totals.items()},
                "learning_rate": cfg.learning_rate, "ppo_update": self.updates,
                "optimizer_steps": cfg.epochs * cfg.minibatches,
                "normalizer_revision": rollout.normalizer_revision,
                "return_mean": return_mean, "reward_mean": reward_mean, "world_size": world_size}

    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict(), "updates": self.updates, "config": asdict(self.config)}

    def load_state_dict(self, state):
        if state["config"] != asdict(self.config):
            raise ValueError("PPO resume configuration differs from saved checkpoint")
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state["updates"])
