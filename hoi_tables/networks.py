"""v1.1 joint dynamics and direct privileged tables policy/value networks.

The dynamics equations, losses, 82 outputs and five-channel head are ported
from hoi/v1.1-observation-matching@66d9184:hoi/v1/networks.py. The sole WM
feature extension is the approved 88D hw3 actuator context after the original
225D inputs. Actuator memory itself is advanced analytically, never predicted.
"""
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from supermimic.actuator_state import ACTUATOR_DIM, normalize_actuator_state
from supermimic.mdp import NUM_J, invariant_features
from supermimic.networks import RunningNorm, WorldModel as RobotWorldModel, mlp, run_mlp
from supermimic.privileged import PRIVILEGED_DIM
from supermimic.rotations import so3_exp, yaw_mat
from hoi_residual.features import ORACLE_DIM, BLOCKS
from .v11.contacts import CHANNELS, LABEL_SCHEMA
from .v11.state import integrate_object, object_features, object_state_error, object_targets

SCHEMA = "hoi_tables_hw3_privileged_teacher_v1"


def feature_schema():
    return {"version": SCHEMA, "wm_input_dim": 313, "wm_output_dim": 82,
            "wm": ["legacy_robot133", "object_invariant15", "shape64", "object_context12",
                   "torso_mass_delta_over_3kg1", "normalized_hw3_actuator88"],
            "wm_output": ["robot_increment70", "object_increment12"],
            "actor_critic_oracle_dim": ORACLE_DIM, "shape_dim": 64,
            "actor_critic_blocks": dict(BLOCKS), "base_action_input": False,
            "residual_action": False, "contact": list(CHANNELS), "contact_schema": LABEL_SCHEMA,
            "actuator_prediction": "analytic_control_boundary_advance",
            "v11_source": "hoi/v1.1-observation-matching@66d9184"}


def _dim(value, size, name):
    if value.shape[-1] != size:
        raise ValueError(f"{name} must have trailing dimension {size}")


class ShapeEncoder(nn.Module):
    """The unchanged v1.1 metric FPS128 PointNet, with fresh trainable weights."""
    def __init__(self):
        super().__init__()
        self.point = mlp(3, [64, 128], 128)
        self.out = nn.Sequential(nn.ELU(), nn.Linear(128, 64))

    def forward(self, points):
        if points.shape[-2:] != (128, 3):
            raise ValueError("Shape points must have batch+(128,3) dimensions")
        return self.out(run_mlp(self.point, points).amax(-2))


class OracleNormalizer(nn.Module):
    """Independent raw blocks; object history shares 9D statistics across ticks.

    Only the newest real history tick updates its RMS. Imagined/evaluation
    calls only read buffers, preserving all feature Jacobians.
    """
    def __init__(self):
        super().__init__()
        self.robot = RunningNorm(PRIVILEGED_DIM)
        self.object_history = RunningNorm(9)
        self.object_reference = RunningNorm(63)
        self.object_errors = RunningNorm(105)
        self.object_velocity = RunningNorm(6)
        self.object_physics = RunningNorm(12)

    def _blocks(self, raw):
        _dim(raw, ORACLE_DIM, "Privileged oracle")
        return raw.split((PRIVILEGED_DIM, 90, 63, 105, 6, 12), -1)

    def forward(self, raw):
        robot, history, reference, errors, velocity, physics = self._blocks(raw)
        history = self.object_history(history.reshape(*history.shape[:-1], 10, 9)).flatten(-2)
        return torch.cat((self.robot(robot), history, self.object_reference(reference),
                          self.object_errors(errors), self.object_velocity(velocity),
                          self.object_physics(physics)), -1)

    @torch.no_grad()
    def update(self, raw):
        robot, history, reference, errors, velocity, physics = self._blocks(raw)
        self.robot.update(robot)
        self.object_history.update(history[..., -9:])
        for norm, values in ((self.object_reference, reference), (self.object_errors, errors),
                             (self.object_velocity, velocity), (self.object_physics, physics)):
            norm.update(values)


class Policy(nn.Module):
    def __init__(self, hidden=(2048, 1024, 512, 256)):
        super().__init__()
        self.norm = OracleNormalizer()
        self.shape_encoder = ShapeEncoder()
        self.net = mlp(ORACLE_DIM + 64, list(hidden), NUM_J)
        with torch.no_grad():
            self.net[-1].weight.mul_(.01)
            self.net[-1].bias.zero_()

    def forward(self, oracle, points, *, shape_embedding=None):
        shape = self.shape_encoder(points) if shape_embedding is None else shape_embedding
        if shape.shape != oracle.shape[:-1] + (64,):
            raise ValueError("Policy shape encoding does not match oracle batch")
        return run_mlp(self.net, torch.cat((self.norm(oracle), shape), -1))

    @torch.no_grad()
    def update_normalizers(self, oracle):
        self.norm.update(oracle)


class Critic(nn.Module):
    def __init__(self, hidden=(256, 128)):
        super().__init__()
        self.norm = OracleNormalizer()
        self.shape_encoder = ShapeEncoder()
        self.net = mlp(ORACLE_DIM + 64, list(hidden), 1)

    def forward(self, oracle, points, *, shape_embedding=None):
        shape = self.shape_encoder(points) if shape_embedding is None else shape_embedding
        return run_mlp(self.net, torch.cat((self.norm(oracle), shape), -1)).squeeze(-1)

    @torch.no_grad()
    def update_normalizers(self, oracle):
        self.norm.update(oracle)


class WorldModel(nn.Module):
    def __init__(self, hidden=(2048, 2048, 2048), contact_hidden=128):
        super().__init__()
        self.in_dim, self.out_dim = 313, 82
        self.net = mlp(self.in_dim, list(hidden), 82, act=nn.SiLU, layer_norm=True)
        with torch.no_grad():
            self.net[-1].weight[:70].zero_()
            self.net[-1].bias[:70].zero_()
        self.contact_head = mlp(hidden[-1], [contact_hidden], len(CHANNELS), act=nn.SiLU)
        self.shape_encoder = ShapeEncoder()
        self.in_norm, self.out_norm = RunningNorm(133), RunningNorm(70, eps=1e-4)
        self.object_in_norm, self.object_out_norm = RunningNorm(27), RunningNorm(12, eps=1e-4)
        self.register_buffer("robot_error_floor", torch.cat((torch.full((3,), .003), torch.full((3,), .005),
                             torch.full((3,), .02), torch.full((3,), .05), torch.full((29,), .003), torch.full((29,), .05))))
        self.register_buffer("object_error_floor", torch.tensor([.003] * 3 + [.005] * 3 + [.02] * 3 + [.05] * 3))

    def raw_inputs(self, S, O, a, xi_n, xi_o):
        _dim(a, 29, "Action")
        _dim(xi_n, 33, "Robot context")
        _dim(xi_o, 12, "Object context")
        return torch.cat((invariant_features(S), a, xi_n), -1), torch.cat((object_features(S, O), xi_o), -1)

    def inputs(self, S, O, a, xi_n, xi_o, points, actuator, mass_context, *, shape_embedding=None):
        old, new = self.raw_inputs(S, O, a, xi_n, xi_o)
        new = self.object_in_norm(new)
        shape = self.shape_encoder(points) if shape_embedding is None else shape_embedding
        if mass_context.shape != old.shape[:-1] + (1,):
            raise ValueError("Torso mass context must be batch+(1,)")
        return torch.cat((self.in_norm(old), new[..., :15], shape, new[..., 15:], mass_context,
                          normalize_actuator_state(actuator)), -1)

    @staticmethod
    def targets(S, O, S1, O1):
        return torch.cat((RobotWorldModel.targets(S, S1), object_targets(S, O, O1)), -1)

    @staticmethod
    def state_error(S_hat, O_hat, S, O):
        return torch.cat((RobotWorldModel.state_error(S_hat, S), object_state_error(S, O_hat, O)), -1)

    def step(self, S, O, a, xi_n, xi_o, points, actuator, mass_context, *, shape_embedding=None):
        hidden = run_mlp(self.net[:-1], self.inputs(S, O, a, xi_n, xi_o, points, actuator, mass_context,
                                                   shape_embedding=shape_embedding))
        y = run_mlp(self.net[-1], hidden)
        dr = y[..., :70] * self.out_norm.std() + self.out_norm.mean
        do = y[..., 70:] * self.object_out_norm.std() + self.object_out_norm.mean
        p, R, v, w, q, qd = S
        dp, dth, dv, dw, dq, dqd = dr.split((3, 3, 3, 3, 29, 29), -1)
        rot = lambda m, x: (m @ x.unsqueeze(-1)).squeeze(-1)
        S1 = (p + rot(yaw_mat(R), dp), R @ so3_exp(dth), v + rot(R, dv),
              w + rot(R, dw), q + dq, qd + dqd)
        return S1, integrate_object(S, O, do), run_mlp(self.contact_head, hidden)

    forward = step

    @torch.no_grad()
    def update_normalizers(self, S, O, a, xi_n, xi_o, points, S1, O1, *, allow_empty=False):
        if not len(a) and not allow_empty:
            return
        old, new = self.raw_inputs(S, O, a, xi_n, xi_o)
        target = self.targets(S, O, S1, O1)
        self.in_norm.update(old)
        self.object_in_norm.update(new)
        self.out_norm.update(target[..., :70])
        self.object_out_norm.update(target[..., 70:])

    def loss(self, S_hat, O_hat, S, O, contact_logits, contact_actual, contact_valid, mask,
             pos_weight=None, lambda_object=1., lambda_contact=.1, *, return_window_totals=False):
        error = self.state_error(S_hat, O_hat, S, O)
        robot = ((error[..., :70] / torch.maximum(self.out_norm.std(), self.robot_error_floor)) ** 2).mean(-1)
        scaled = (error[..., 70:] / torch.maximum(self.object_out_norm.std(), self.object_error_floor)) ** 2
        obj = scaled.reshape(scaled.shape[:-1] + (4, 3)).mean(-1).mean(-1)
        for value in (contact_logits, contact_actual, contact_valid):
            _dim(value, len(CHANNELS), "Contact head/label/mask")
        bce = F.binary_cross_entropy_with_logits(contact_logits, contact_actual.float(), reduction="none", pos_weight=pos_weight)
        valid = contact_valid.to(bce.dtype)
        contact = (bce * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        reduce = lambda value: (value * mask).sum() / max(1, mask.shape[0])
        r, o, c = reduce(robot), reduce(obj), reduce(contact)
        losses = {"total": r + lambda_object * o + lambda_contact * c, "robot": r, "object": o, "contact": c}
        if return_window_totals:
            losses["window_total"] = ((robot + lambda_object * obj + lambda_contact * contact) * mask).sum(-1).detach()
        return losses


@contextmanager
def frozen_parameters(*modules):
    """Keep action/state Jacobians while excluding these model parameters."""
    original = [(p, p.requires_grad) for module in modules for p in module.parameters()]
    try:
        for p, _ in original:
            p.requires_grad_(False)
        yield
    finally:
        for p, enabled in original:
            p.requires_grad_(enabled)
