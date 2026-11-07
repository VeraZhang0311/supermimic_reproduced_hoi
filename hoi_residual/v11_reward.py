"""Frozen v1.1 HOI objective on the current robot, physics and 50 Hz references.

Ported from hoi/v1.1-observation-matching@66d9184, hoi/v1/{mdp,geometry}.py
and supermimic/mdp.py. Only the prepared reference hand-pair schedules and
collision geometry are read from v1.1. Actual contacts come from the simulator.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from supermimic.motion import CONTROL_DT
from supermimic.rotations import rot_angle_sq, yaw_mat
from .assets import _digest_json
from .shape import point_sha256


MAX_STEP_REWARD = 0.15
TERM_NAMES = (
    "body_position", "body_orientation", "body_linear_velocity", "body_angular_velocity",
    "root_position", "root_orientation", "action_rate", "joint_limits",
    "object_position", "object_orientation", "object_linear_velocity", "object_angular_velocity",
    "hand_approach", "contact_match",
)
LABEL_SCHEMA = "reference_hand_object_geometric_intent_v2_surface_intersection_unknown"
_MOUNTS = {"left": [.0415, .003, 0.], "right": [.0415, -.003, 0.]}


def _masked_mean(values, mask):
    return (values * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def reward_terms(mdp, S, O, frame, action, previous_action, expected, valid,
                 actual_contact, contact_valid, hand_distances, *, bodies=None):
    """Return signed, weighted contributions per control step for each environment."""
    p, R, _, _, q, _ = S
    bp, bR, bv, bw = mdp.kin.forward(*S) if bodies is None else bodies
    motion = mdp.motion
    ref_p, ref_R = motion.root_p[frame], motion.root_R[frame]
    ref_bp, ref_bR, ref_bv, ref_bw = motion.reference_bodies(frame)
    # v1.1 takes yaw *after* composing the two rotations, including their tilt.
    dR = yaw_mat(R @ ref_R.transpose(-1, -2))
    anchor = torch.cat((p[..., :2], ref_p[..., 2:3]), -1)
    ref_bp = anchor.unsqueeze(-2) + (dR.unsqueeze(-3) @ (ref_bp - ref_p.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)
    ref_bR = dR.unsqueeze(-3) @ ref_bR
    op, oR, ov, ow = motion.object_ref(frame)
    # SDF distances have shape [environments, left/right hand, palm probes].
    squared = hand_distances.square()
    approach = torch.exp(-(torch.softmax(-squared / .02**2, -1) * squared).sum(-1) / .05**2)
    expected = expected.to(q)
    valid = valid.to(q)
    actual = actual_contact.to(q)
    contact = expected * actual + (1 - expected) * (1 - actual)
    terms = {
        "body_position": torch.exp(-(ref_bp - bp).square().sum(-1).mean(-1) / .3**2),
        "body_orientation": torch.exp(-rot_angle_sq(ref_bR, bR).mean(-1) / .4**2),
        "body_linear_velocity": torch.exp(-(ref_bv - bv).square().sum(-1).mean(-1) / 1.0**2),
        "body_angular_velocity": torch.exp(-(ref_bw - bw).square().sum(-1).mean(-1) / 3.14**2),
        "root_position": .5 * torch.exp(-(ref_p - p).square().sum(-1) / .3**2),
        "root_orientation": .5 * torch.exp(-rot_angle_sq(ref_R, R) / .4**2),
        "action_rate": -.1 * (action - previous_action).square().sum(-1),
        "joint_limits": -2 * (torch.relu(q - mdp.soft_hi) + torch.relu(mdp.soft_lo - q)).sum(-1),
        "object_position": torch.exp(-(O[0] - op).square().sum(-1) / .10**2),
        "object_orientation": .5 * torch.exp(-rot_angle_sq(O[1], oR) / .30**2),
        "object_linear_velocity": .25 * torch.exp(-(O[2] - ov).square().sum(-1) / .50**2),
        "object_angular_velocity": .25 * torch.exp(-(O[3] - ow).square().sum(-1) / 1.0**2),
        "hand_approach": .25 * _masked_mean(approach, expected * valid),
        "contact_match": .25 * _masked_mean(contact, valid * contact_valid.to(q)),
    }
    return {name: term * CONTROL_DT for name, term in terms.items()}


def _bound_cache(parent, binding):
    path = parent / binding["path"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError(f"v1.1 reward cache SHA256 mismatch: {path}")
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)


class V11Reward:
    def __init__(self, motion, asset, bundle_path, *, expected_bundle_content_sha256):
        path = Path(bundle_path)
        raw = path.read_bytes()
        bundle = json.loads(raw)
        digest = bundle.pop("bundle_content_sha256")
        if digest != expected_bundle_content_sha256 or digest != _digest_json(bundle):
            raise ValueError("v1.1 reward bundle content SHA256 mismatch")
        if (bundle["schema_version"] != "supermimic_hoi_v1_bundle_v1" or
                motion.partition != "train" or not motion.for_training or bundle["partition"] != "train"):
            raise ValueError("v1.1 training rewards require the train partition of both motion libraries")
        assets = {entry["object_asset"]: entry for entry in bundle["assets"]}
        source = assets[asset["object_asset"]]
        if any(source[key] != asset[key] for key in ("exact_scale", "collision_hash", "collision_meshes")):
            raise ValueError("v1.1 reward geometry differs from the training asset")
        geo = _bound_cache(path.parent, source["geometry_cache"])
        if geo["metadata"]["collision_hash"] != source["collision_hash"] or geo["metadata"]["grid_axis_order"] != "zyx":
            raise ValueError("v1.1 SDF grid identity or coordinate contract differs")
        self.grid, self.low, self.high = (geo[key].to(device=motion.device, dtype=torch.float32)
                                          for key in ("sdf_grid", "grid_min", "grid_max"))
        hands_binding = bundle["hands"]["geometry_cache"]
        hands = _bound_cache(path.parent, hands_binding)
        palms = []
        tracked = [motion.kin.body_names[index] for index in motion.kin.tracked_idx]
        self.wrists = [tracked.index(side + "_wrist_yaw_link") for side in _MOUNTS]
        for side, mount in _MOUNTS.items():
            hand = hands[side]
            points = hand["palm_points"]
            if hand["mount"] != mount or point_sha256(points) != hand["palm_points_sha256"]:
                raise ValueError(f"v1.1 {side} palm probe/mount mismatch")
            palms.append(points + points.new_tensor(mount))
        self.palms = torch.stack(palms).to(motion.device)

        by_hash = {c["source_motion_sha256"]: c for c in bundle["clips"]}
        if len(by_hash) != len(bundle["clips"]):
            raise ValueError("duplicate motion in v1.1 contact schedules")
        expected, valid = [], []
        for entry, length in zip(motion.entries, motion.clip_len.tolist(), strict=True):
            clip = by_hash.get(entry["motion_sha256"])
            if (clip is None or clip["source_group"] != entry["source_group"] or
                    entry["object_asset"] != asset["object_asset"] or clip["asset_id"] != source["asset_id"] or
                    clip["state_frames"] != length):
                raise ValueError("v1.1 reference contact schedule does not match training motion, asset or frame grid")
            pair = _bound_cache(path.parent, clip["contact_cache"])
            e, m = pair["expected_contact"], pair["expected_valid"]
            if (pair["metadata"]["label_schema"] != LABEL_SCHEMA or e.shape != (length, 2) or m.shape != e.shape or
                    not (((e == 0) | (e == 1)) & ((m == 0) | (m == 1))).all()):
                raise ValueError("invalid v1.1 left/right reference schedule or validity mask")
            expected.append(e.bool() & m.bool())
            valid.append(m.bool())
        self.expected = torch.cat(expected).to(motion.device)
        self.valid = torch.cat(valid).to(motion.device)
        self.contract = {
            "source_commit": "66d918488ec085377bb0d97c4c029ded9d729d86",
            "bundle_content_sha256": digest, "bundle_sha256": hashlib.sha256(raw).hexdigest(),
            "partition": bundle["partition"], "reference_contact_schema": LABEL_SCHEMA,
            "hands_sha256": hands_binding["sha256"],
            "asset_sdf_sha256": {name: value["geometry_cache"]["sha256"] for name, value in assets.items()},
            "term_names": list(TERM_NAMES), "max_step_reward": MAX_STEP_REWARD,
        }

    def hand_distances(self, bodies, O):
        bp, bR = bodies[:2]
        # Both wrists are in GMT's tracked set, so reuse the same FK as the reward.
        world = bp[:, self.wrists, None, :] + (bR[:, self.wrists, None] @ self.palms.unsqueeze(-1)).squeeze(-1)
        local = (O[1].transpose(-1, -2)[:, None, None] @ (world - O[0][:, None, None]).unsqueeze(-1)).squeeze(-1)
        clamped = local.maximum(self.low).minimum(self.high)
        query = (2 * (clamped - self.low) / (self.high - self.low) - 1).reshape(1, 1, 1, -1, 3)
        sampled = F.grid_sample(self.grid[None, None], query, mode="bilinear", padding_mode="border",
                                align_corners=True).reshape(local.shape[:-1])
        return sampled + (local - clamped).norm(dim=-1)

    def __call__(self, mdp, S, O, frame, action, previous_action, actual_contact, contact_valid):
        bodies = mdp.kin.forward(*S)
        expected = self.expected[frame]
        if actual_contact.shape != expected.shape or contact_valid.shape != expected.shape:
            raise ValueError("native contacts and validity masks must be [environments, left/right]")
        terms = reward_terms(mdp, S, O, frame, action, previous_action, expected, self.valid[frame],
                             actual_contact, contact_valid, self.hand_distances(bodies, O), bodies=bodies)
        return sum(terms.values()), terms
