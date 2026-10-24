"""Additive full-reference HOI networks; legacy feature prefixes are preserved."""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from supermimic.mdp import NUM_J, invariant_features
from supermimic.networks import RunningNorm, WorldModel as RobotWorldModel, mlp, run_mlp
from supermimic.rotations import so3_exp, yaw_mat
from .contacts import CHANNELS, LABEL_SCHEMA
from .state import integrate_object, object_features, object_state_error, object_targets

FEATURE_VERSION = "supermimic_hoi_v1_full_reference_2"
FEATURE_SCHEMA = {
    "version": FEATURE_VERSION,
    "robot_state": "p3,quat_wxyz4,v_world3,omega_world3,q29,qd29",
    "object_state": "p_origin3,quat_wxyz4,v_origin_world3,omega_world3",
    "actor": ["human_latent32", "robot_history10x99", "object_history10x18", "object_latent32", "shape64"],
    "wm": ["legacy_robot133", "object_invariant15", "shape64", "object_context12"],
    "wm_output": ["robot_increment70", "object_increment12"],
    "critic": ["legacy914", "object_goal190", "object_invariant15", "object_error15", "object_context12", "expected_and_valid4", "shape64"],
    "object_context": "mass1,COM_local3,inertia_COM_local_xx_yy_zz_xy_xz_yz6,object_friction1,ground_friction1",
    "object_invariant": "HrobotT*(pO-pR)3,rot6d(HrobotT*RO)6,HrobotT*(vO-vR)3,HrobotT*omegaO3",
    "object_observation": "HrobotT*(pO-pR)3,rot6d(HrobotT*RO)6,HrobotT*(pOref-pO)3,rot6d(ROT*ROref)6",
    "object_goal_tick": "HrobotT*(pOref-pR)3,rot6d(HrobotT*ROref)6,HrobotT*vOref3,HrobotT*omegaOref3,expectedLR2,validLR2",
    "object_increment": "HrobotT*dp3,log(ROT*ROnext)3,HrobotT*dv3,HrobotT*domega3",
    "contact": list(CHANNELS),
    "contact_time": "actual_interval_t_tplus1",
    "actual_contact_label": {"schema": LABEL_SCHEMA, "threshold_N": 1., "comparison": "strict_greater",
                             "control_dt_s": .02, "num_substeps": 4,
                             "aggregation": "sum_normal_impulses_all_contact_points_all_substeps_divided_by_control_dt"},
    "shape": "pointnet_3_64_128_128_max_128_64_ELU_FPS128_object_local_meters",
    "normalizers": "independent_legacy_new_raw_blocks_shape_unnormalized",
}


def _dim(x, dim, name):
    if x.shape[-1] != dim:
        raise ValueError(f"{name} must have dimension {dim}; got {x.shape[-1]}")


class ShapeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.point = mlp(3, [64, 128], 128)
        self.out = nn.Sequential(nn.ELU(), nn.Linear(128, 64))

    def forward(self, points):
        if points.shape[-2:] != (128, 3):
            raise ValueError("shape pointcloud must have (...,128,3) shape")
        return self.out(run_mlp(self.point, points).amax(-2))


class Policy(nn.Module):
    def __init__(self, obs_dim=99, goal_dim=730, hist=10, z_dim=32,
                 hidden=(2048, 1024, 512, 256), object_hidden=(256, 128), *, geometry_input_enabled=True):
        super().__init__()
        if type(geometry_input_enabled) is not bool:
            raise ValueError("geometry_input_enabled must be boolean")
        self.geometry_input_enabled = geometry_input_enabled
        if (obs_dim, goal_dim, hist, z_dim) != (99, 730, 10, 32):
            raise ValueError("v1 schema requires obs99, human_goal730, history10, latent32")
        self.hist = hist
        self.enc = mlp(goal_dim, list(hidden), z_dim)
        self.object_enc = mlp(190, list(object_hidden), 32)
        self.shape_encoder = ShapeEncoder()
        self.dec = mlp(1298, list(hidden), NUM_J)
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.zero_()
        self.obs_norm, self.goal_norm = RunningNorm(99), RunningNorm(730)
        self.object_obs_norm, self.object_goal_norm = RunningNorm(18), RunningNorm(190)

    def human_latent(self, human_goal):
        _dim(human_goal, 730, "human goal")
        return torch.tanh(run_mlp(self.enc, self.goal_norm(human_goal)))

    def forward(self, robot_hist, human_goal, object_hist, object_goal, points, *, shape_embedding=None):
        if robot_hist.shape[-2:] != (10, 99) or object_hist.shape[-2:] != (10, 18):
            raise ValueError("actor requires robot (...,10,99) and object (...,10,18) histories")
        _dim(object_goal, 190, "object goal")
        if shape_embedding is None:
            shape_embedding = self.shape_encoder(points)
        elif not isinstance(shape_embedding, torch.Tensor) or shape_embedding.shape != robot_hist.shape[:-2] + (64,):
            raise ValueError("actor shape embedding must match the observation batch and have 64 features")
        if not self.geometry_input_enabled:
            shape_embedding = shape_embedding * 0
        zH = self.human_latent(human_goal)
        zO = torch.tanh(run_mlp(self.object_enc, self.object_goal_norm(object_goal)))
        x = torch.cat((zH, self.obs_norm(robot_hist).flatten(-2),
                       self.object_obs_norm(object_hist).flatten(-2), zO, shape_embedding), -1)
        return run_mlp(self.dec, x)

    @torch.no_grad()
    def update_normalizers(self, robot_hist, human_goal, object_hist, object_goal):
        self.obs_norm.update(robot_hist)
        self.goal_norm.update(human_goal)
        self.object_obs_norm.update(object_hist)
        self.object_goal_norm.update(object_goal)


class WorldModel(nn.Module):
    def __init__(self, hidden=(2048, 2048, 2048), contact_hidden=128, *, geometry_input_enabled=True):
        super().__init__()
        if type(geometry_input_enabled) is not bool:
            raise ValueError("geometry_input_enabled must be boolean")
        self.geometry_input_enabled = geometry_input_enabled
        if not hidden:
            raise ValueError("joint WM requires hidden layers")
        self.in_dim, self.out_dim = 224, 82
        self.net = mlp(224, list(hidden), 82, act=nn.SiLU, layer_norm=True)
        # Preserve the legacy zero-increment start for the robot block. New
        # object rows are independently initialized and immediately trainable.
        with torch.no_grad():
            self.net[-1].weight[:70].zero_()
            self.net[-1].bias[:70].zero_()
        self.contact_head = mlp(hidden[-1], [contact_hidden], len(CHANNELS), act=nn.SiLU)
        self.shape_encoder = ShapeEncoder()
        self.in_norm, self.out_norm = RunningNorm(133), RunningNorm(70, eps=1e-4)
        self.object_in_norm = RunningNorm(27)
        self.object_out_norm = RunningNorm(12, eps=1e-4)
        self.register_buffer("robot_error_floor", torch.cat((torch.full((3,), .003), torch.full((3,), .005),
                              torch.full((3,), .02), torch.full((3,), .05), torch.full((29,), .003), torch.full((29,), .05))))
        self.register_buffer("object_error_floor", torch.tensor([.003] * 3 + [.005] * 3 + [.02] * 3 + [.05] * 3))

    def raw_inputs(self, S, O, a, xi_n, xi_o):
        _dim(a, 29, "action")
        _dim(xi_n, 33, "normalized robot context")
        _dim(xi_o, 12, "object physical context")
        return torch.cat((invariant_features(S), a, xi_n), -1), torch.cat((object_features(S, O), xi_o), -1)

    def inputs(self, S, O, a, xi_n, xi_o, points):
        old, new = self.raw_inputs(S, O, a, xi_n, xi_o)
        new = self.object_in_norm(new)
        shape = self.shape_encoder(points)
        if not self.geometry_input_enabled:
            shape = shape * 0
        return torch.cat((self.in_norm(old), new[..., :15], shape, new[..., 15:]), -1)

    @staticmethod
    def targets(S, O, S1, O1):
        return torch.cat((RobotWorldModel.targets(S, S1), object_targets(S, O, O1)), -1)

    @staticmethod
    def state_error(S_hat, O_hat, S, O):
        return torch.cat((RobotWorldModel.state_error(S_hat, S), object_state_error(S, O_hat, O)), -1)

    def step(self, S, O, a, xi_n, xi_o, points):
        hidden = run_mlp(self.net[:-1], self.inputs(S, O, a, xi_n, xi_o, points))
        y = run_mlp(self.net[-1], hidden)
        dr = y[..., :70] * self.out_norm.std() + self.out_norm.mean
        do = y[..., 70:] * self.object_out_norm.std() + self.object_out_norm.mean
        p, R, v, w, q, qd = S
        dp, dth, dv, dw, dq, dqd = dr.split((3, 3, 3, 3, 29, 29), -1)
        rot = lambda m, x: (m @ x.unsqueeze(-1)).squeeze(-1)
        S1 = (p + rot(yaw_mat(R), dp), R @ so3_exp(dth), v + rot(R, dv), w + rot(R, dw), q + dq, qd + dqd)
        return S1, integrate_object(S, O, do), run_mlp(self.contact_head, hidden)

    @torch.no_grad()
    def update_normalizers(self, S, O, a, xi_n, xi_o, points, S1, O1):
        # Deliberately normalize only raw data, never a changing shape embedding.
        old, new = self.raw_inputs(S, O, a, xi_n, xi_o)
        target = self.targets(S, O, S1, O1)
        self.in_norm.update(old)
        self.object_in_norm.update(new)
        self.out_norm.update(target[..., :70])
        self.object_out_norm.update(target[..., 70:])

    def loss(self, S_hat, O_hat, S, O, contact_logits, contact_actual, contact_valid, mask,
             pos_weight=None, lambda_object=1., lambda_contact=.1):
        """Original masked robot loss plus equally weighted object groups and pair BCE.

        ``mask`` contains the terminal transition and excludes only later reset
        transitions. A (B,K) window is summed over K, divided by B as upstream.
        """
        error = self.state_error(S_hat, O_hat, S, O)
        robot = ((error[..., :70] / torch.maximum(self.out_norm.std(), self.robot_error_floor)) ** 2).mean(-1)
        scaled = (error[..., 70:] / torch.maximum(self.object_out_norm.std(), self.object_error_floor)) ** 2
        obj = scaled.reshape(scaled.shape[:-1] + (4, 3)).mean(-1).mean(-1)
        for x, name in ((contact_logits, "contact logits"), (contact_actual, "actual pair labels"), (contact_valid, "pair validity")):
            _dim(x, len(CHANNELS), name)
        bce = F.binary_cross_entropy_with_logits(contact_logits, contact_actual.float(), reduction="none", pos_weight=pos_weight)
        valid = contact_valid.to(bce.dtype)
        contact = (bce * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        reduce = lambda x: (x * mask).sum() / max(1, mask.shape[0])
        r, o, c = reduce(robot), reduce(obj), reduce(contact)
        return {"total": r + lambda_object * o + lambda_contact * c, "robot": r, "object": o, "contact": c}


class Critic(nn.Module):
    def __init__(self, hidden=(256, 128), *, geometry_input_enabled=True):
        super().__init__()
        if type(geometry_input_enabled) is not bool:
            raise ValueError("geometry_input_enabled must be boolean")
        self.geometry_input_enabled = geometry_input_enabled
        self.net = mlp(1214, list(hidden), 1)
        self.norm, self.object_norm = RunningNorm(914), RunningNorm(236)
        self.shape_encoder = ShapeEncoder()

    @staticmethod
    def raw_object_features(object_goal, object_feat, errors, xi_o, expected):
        values = ((object_goal, 190), (object_feat, 15), (errors, 15), (xi_o, 12), (expected, 4))
        for x, d in values:
            _dim(x, d, "critic object feature")
        return torch.cat(tuple(x for x, _ in values), -1)

    def forward(self, legacy, object_goal, object_feat, errors, xi_o, expected, points):
        _dim(legacy, 914, "legacy critic features")
        raw = self.raw_object_features(object_goal, object_feat, errors, xi_o, expected)
        # Must remain in forward, including value-fitting minibatches.
        shape = self.shape_encoder(points)
        if not self.geometry_input_enabled:
            shape = shape * 0
        x = torch.cat((self.norm(legacy), self.object_norm(raw), shape), -1)
        return run_mlp(self.net, x).squeeze(-1)

    @torch.no_grad()
    def update_normalizers(self, legacy, object_goal, object_feat, errors, xi_o, expected):
        self.norm.update(legacy)
        self.object_norm.update(self.raw_object_features(object_goal, object_feat, errors, xi_o, expected))


@contextmanager
def frozen_parameters(*modules):
    """Freeze parameters without disabling action/state/pointcloud Jacobians."""
    parameters = [(p, p.requires_grad) for module in modules for p in module.parameters()]
    try:
        for p, _ in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, enabled in parameters:
            p.requires_grad_(enabled)
