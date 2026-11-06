"""Frozen privileged GMT and an oracle residual actor/critic.

ResMimic design reference: amazon-far/ResMimic@fa20d98, frozen base action
plus a Gaussian residual with PPO log probabilities on the residual alone.
This implementation uses the existing GMT robot/observation contract; it does
not import the ResMimic simulator, student observation slicing or world model.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import json
import math
from pathlib import Path
import re

import torch
import torch.distributed as dist
from torch import nn
from torch.distributions import Normal

from supermimic.networks import Policy, RunningNorm
from supermimic.privileged import PRIVILEGED_DIM, SCHEMA
from supermimic.robot_metadata import G1_BODY_NAMES, G1_JOINT_NAMES


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_frozen_gmt_contract(source, target):
    """Only the reference dataset and its manifest may differ from GMT.

The caller builds target with supermimic.privileged.checkpoint_contract using
the running code and robot assets. ULTRA is allowed as a target dataset here;
the existing motion-only fine-tuning validator intentionally stays untouched.
"""
    if not isinstance(source, dict) or not isinstance(target, dict):
        raise ValueError("Frozen GMT requires complete source and target contracts")
    for label, contract in (("source", source), ("target", target)):
        for key, expected in SCHEMA.items():
            if _canonical(contract.get(key)) != _canonical(expected):
                raise ValueError(f"Frozen GMT {label} schema mismatch: {key}")
        if contract.get("body_names") != G1_BODY_NAMES or contract.get("joint_names") != G1_JOINT_NAMES:
            raise ValueError(f"Frozen GMT {label} robot identity mismatch")
        for section in ("physics", "analytic_heads"):
            identity = contract.get(section)
            if not isinstance(identity, dict) or not identity.get("implementation_sha256"):
                raise ValueError(f"Frozen GMT {label} missing {section} implementation identity")
            for name, digest in identity["implementation_sha256"].items():
                if not isinstance(digest, str) or re.fullmatch("[a-f0-9]{64}", digest) is None:
                    raise ValueError(f"Frozen GMT {label} invalid implementation SHA: {name}")
        for key in ("urdf_sha256", "motion_manifest_sha256"):
            if not isinstance(contract.get(key), str) or re.fullmatch("[a-f0-9]{64}", contract[key]) is None:
                raise ValueError(f"Frozen GMT {label} invalid {key}")
        if not isinstance(contract.get("dataset"), str) or not contract["dataset"]:
            raise ValueError(f"Frozen GMT {label} missing dataset identity")
    for key in source.keys() | target.keys():
        if key not in {"dataset", "motion_manifest_sha256"}:
            if key not in source or key not in target or _canonical(source[key]) != _canonical(target[key]):
                raise ValueError(f"Frozen GMT forbids contract change: {key}")


class FrozenGMT(nn.Module):
    """Load pi and its RMS only; never construct a WM, critic or optimizer."""

    def __init__(self, checkpoint_path, target_contract, device="cpu", expected_iteration=26000):
        super().__init__()
        path = Path(checkpoint_path).resolve()
        digest = file_sha256(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if file_sha256(path) != digest:
            raise RuntimeError("GMT checkpoint changed while it was being read")
        if not isinstance(checkpoint, dict) or not {"pi", "args", "gmt_contract", "iteration"}.issubset(checkpoint):
            raise ValueError("GMT checkpoint is missing policy/provenance fields")
        validate_frozen_gmt_contract(checkpoint["gmt_contract"], target_contract)
        if expected_iteration is not None and checkpoint["iteration"] != expected_iteration:
            raise ValueError("GMT checkpoint iteration differs from explicitly selected parent")
        args = checkpoint["args"]
        args = dict(args) if isinstance(args, Mapping) else vars(args)
        if args.get("hist") != 10 or args.get("goal_frames") != 10:
            raise ValueError("Frozen GMT requires hist=10 and goal_frames=10")
        self.policy = Policy(99, 730, hist=10, privileged_dim=PRIVILEGED_DIM)
        expected = self.policy.state_dict()
        saved = checkpoint["pi"]
        if not isinstance(saved, Mapping) or set(saved) != set(expected):
            raise ValueError("Frozen GMT policy state keys mismatch")
        for name, value in saved.items():
            if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape or value.dtype != expected[name].dtype:
                raise ValueError(f"Frozen GMT tensor shape/dtype mismatch: {name}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Frozen GMT nonfinite policy state: {name}")
        self.policy.load_state_dict(saved, strict=True)
        self.requires_grad_(False)
        self.to(device)
        self.train(False)
        self._snapshot = {name: value.detach().cpu().clone() for name, value in self.policy.state_dict().items()}
        self.metadata = {"checkpoint": str(path), "checkpoint_sha256": digest,
                         "iteration": checkpoint["iteration"], "loaded": ["pi", "pi_normalizers"],
                         "world_model_constructed": False,
                         "source_contract": copy.deepcopy(checkpoint["gmt_contract"]),
                         "target_contract": copy.deepcopy(target_contract),
                         "source_amp": args.get("amp", "none")}

    def train(self, mode=True):
        # Calling train() on a containing teacher must never unfreeze the base.
        return super().train(False)

    @torch.no_grad()
    def forward(self, obs_hist, goal, privileged):
        result = self.policy(obs_hist, goal, privileged)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("Frozen GMT produced a nonfinite action")
        return result.detach()

    def assert_unchanged(self):
        state = self.policy.state_dict()
        if set(state) != set(self._snapshot) or any(p.requires_grad for p in self.parameters()):
            raise RuntimeError("Frozen GMT parameter identity/trainability changed")
        for name, saved in self._snapshot.items():
            if not torch.equal(state[name].detach().cpu(), saved):
                raise RuntimeError(f"Frozen GMT weight/RMS changed: {name}")
        if self.training or self.policy.training:
            raise RuntimeError("Frozen GMT entered training mode")
        return True


def _mlp(input_dim, hidden, output_dim):
    layers = []
    for index, width in enumerate(hidden):
        layer = nn.Linear(input_dim, width)
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
        layers.append(layer)
        # ResMimic uses a LayerNorm at the last hidden representation.
        if index == len(hidden) - 1:
            layers.append(nn.LayerNorm(width))
        layers.append(nn.SiLU())
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class ResidualActorCritic(nn.Module):
    """Oracle MLP conditioned on the actual frozen GMT action.

Actor and critic both observe oracle+base_action. The residual mean is exactly
zero at initialization. The residual Gaussian is sampled without action
clipping: combine with GMT, then use the existing environment action bound.
Normalizer statistics must stay fixed throughout collection AND PPO epochs.
"""

    def __init__(self, oracle_dim, action_dim=29, schema="hoi_oracle_v1",
                 hidden=(512, 512, 256, 128), init_noise_std=0.1):
        super().__init__()
        if oracle_dim <= 0 or action_dim <= 0 or not hidden or init_noise_std <= 0 or not schema:
            raise ValueError("Invalid oracle residual network configuration")
        self.oracle_dim, self.action_dim, self.schema = oracle_dim, action_dim, schema
        self.hidden = tuple(hidden)
        self.norm = RunningNorm(oracle_dim + action_dim)
        self.register_buffer("_normalizer_revision", torch.zeros((), dtype=torch.long))
        self.actor = _mlp(oracle_dim + action_dim, hidden, action_dim)
        self.critic = _mlp(oracle_dim + action_dim, hidden, 1)
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(init_noise_std)))

    @property
    def normalizer_revision(self):
        return int(self._normalizer_revision.item())

    def _inputs(self, oracle, base_actions):
        if oracle.shape[:-1] != base_actions.shape[:-1] or oracle.shape[-1] != self.oracle_dim or base_actions.shape[-1] != self.action_dim:
            raise ValueError("Residual oracle/base action shape does not match declared schema")
        inputs = torch.cat((oracle, base_actions), -1)
        if not bool(torch.isfinite(inputs).all()):
            raise FloatingPointError("Nonfinite residual oracle/base action")
        return inputs

    @torch.no_grad()
    def update_normalizer(self, oracle, base_actions):
        """Call once before collecting the next on-policy rollout, never mid-rollout."""
        inputs = self._inputs(oracle, base_actions)
        if dist.is_available() and dist.is_initialized():
            # Each rank owns a different object asset. Pool all training ranks
            # before changing the shared policy normalization, in float64 to
            # avoid cancellation in sum(x*x) - mean(x)**2.
            flat = inputs.reshape(-1, inputs.shape[-1]).double()
            moments = torch.cat((flat.sum(0), flat.square().sum(0),
                                 torch.tensor([len(flat)], device=flat.device, dtype=flat.dtype)))
            dist.all_reduce(moments)
            if self.norm.count < self.norm.max_count:
                width = inputs.shape[-1]
                count = moments[-1]
                mean = moments[:width] / count
                variance = (moments[width:2 * width] / count - mean.square()).clamp_min(0)
                total = self.norm.count.double() + count
                delta = mean - self.norm.mean.double()
                old_count = self.norm.count.double()
                new_mean = self.norm.mean.double() + delta * count / total
                new_var = (self.norm.var.double() * old_count + variance * count +
                           delta.square() * old_count * count / total) / total
                self.norm.mean.copy_(new_mean)
                self.norm.var.copy_(new_var)
                self.norm.count.copy_(total)
        else:
            self.norm.update(inputs)
        self._normalizer_revision.add_(1)

    def distribution(self, oracle, base_actions):
        inputs = self.norm(self._inputs(oracle, base_actions))
        mean = self.actor(inputs)
        std = self.log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
        return Normal(mean, std), self.critic(inputs).squeeze(-1)

    @torch.no_grad()
    def sample(self, oracle, base_actions):
        distribution, value = self.distribution(oracle, base_actions)
        actions = distribution.sample()
        return {"actions": actions, "log_prob": distribution.log_prob(actions).sum(-1),
                "value": value, "mean": distribution.mean, "std": distribution.stddev}

    def evaluate(self, oracle, base_actions, actions):
        distribution, value = self.distribution(oracle, base_actions)
        if actions.shape != distribution.mean.shape:
            raise ValueError("PPO must evaluate the stored sampled residual action")
        return {"log_prob": distribution.log_prob(actions).sum(-1),
                "entropy": distribution.entropy().sum(-1), "value": value,
                "mean": distribution.mean, "std": distribution.stddev}

    @torch.no_grad()
    def value(self, oracle, base_actions):
        return self.distribution(oracle, base_actions)[1]

    @torch.no_grad()
    def act(self, oracle, base_actions, deterministic=True):
        distribution, _ = self.distribution(oracle, base_actions)
        return distribution.mean if deterministic else distribution.sample()

    def architecture(self):
        return {"schema": self.schema, "oracle_dim": self.oracle_dim,
                "action_dim": self.action_dim, "hidden": list(self.hidden),
                "base_action_input": True, "activation": "silu", "initial_mean": "zero",
                "normalizer": "separate_oracle_plus_base_frozen_within_rollout",
                "action_distribution": "unclipped_residual_gaussian"}
