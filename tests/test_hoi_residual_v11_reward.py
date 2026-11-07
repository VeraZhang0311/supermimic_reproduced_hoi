import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from hoi_residual.assets import _digest_json
from hoi_residual.shape import point_sha256
from hoi_residual.v11_reward import LABEL_SCHEMA, TERM_NAMES, V11Reward, reward_terms
from supermimic.rotations import rot_angle_sq, so3_exp, yaw_mat


def reference(batch):
    zeros = torch.zeros(batch, 3)
    eye = torch.eye(3).expand(batch, 3, 3)
    q = torch.zeros(batch, 29)
    S = (zeros, eye, zeros, zeros, q, q)
    O = S[:4]
    bodies = tuple(value.unsqueeze(1).expand(-1, 14, *value.shape[1:]) for value in O)
    kin = SimpleNamespace(forward=lambda *args: bodies)
    motion = SimpleNamespace(root_p=zeros, root_R=eye, reference_bodies=lambda f: bodies, object_ref=lambda f: O)
    mdp = SimpleNamespace(kin=kin, motion=motion, soft_lo=-torch.ones(29), soft_hi=torch.ones(29))
    return mdp, S, O


class V11RewardTests(unittest.TestCase):
    def test_exact_weighting_masks_and_no_source_flags(self):
        mdp, S, O = reference(3)
        frame = torch.arange(3)
        expected = torch.tensor([[1, 0], [1, 1], [0, 0]])
        valid = torch.tensor([[1, 1], [0, 0], [1, 1]])
        actual = torch.tensor([[1, 1], [0, 0], [1, 0]])
        measured = torch.tensor([[1, 1], [1, 1], [0, 1]])
        terms = reward_terms(mdp, S, O, frame, S[4] + .5, S[4], expected, valid, actual, measured,
                             torch.zeros(3, 2, 8))
        self.assertEqual(tuple(terms), TERM_NAMES)
        torch.testing.assert_close(terms["hand_approach"], torch.tensor([.005, 0, 0]))
        torch.testing.assert_close(terms["contact_match"], torch.tensor([.0025, 0., .005]))
        torch.testing.assert_close(terms["action_rate"], torch.full((3,), -.1 * 29 * .5**2 * .02))
        self.assertTrue(torch.all(terms["joint_limits"] == 0))
        for name, peak in (("body_position", .02), ("body_linear_velocity", .02), ("body_angular_velocity", .02),
                           ("root_position", .01), ("object_position", .02), ("object_linear_velocity", .005),
                           ("object_angular_velocity", .005)):
            torch.testing.assert_close(terms[name], torch.full((3,), peak))
        # The original geodesic angle helper clamps acos near identical rotations.
        zero_angle = rot_angle_sq(O[1], O[1])
        for key, weight, std in (("body_orientation", 1, .4), ("root_orientation", .5, .4),
                                 ("object_orientation", .5, .3)):
            torch.testing.assert_close(terms[key], weight * .02 * torch.exp(-zero_angle / std**2))

    def test_tilted_root_uses_v11_relative_rotation_before_yaw(self):
        mdp, S, O = reference(1)
        ref_R = so3_exp(torch.tensor([[.8, -.6, .4]]))
        robot_R = so3_exp(torch.tensor([[-.4, .9, .3]]))
        mdp.motion.root_R = ref_R
        wrong = yaw_mat(robot_R) @ yaw_mat(ref_R).mT
        correct = yaw_mat(robot_R @ ref_R.mT)
        self.assertGreater(float((wrong - correct).abs().max()), .08)
        ref = list(mdp.motion.reference_bodies(None))
        actual = list(ref)
        ref[0] = torch.ones(1, 14, 3)
        actual[0] = (correct[:, None] @ ref[0].unsqueeze(-1)).squeeze(-1)
        actual[1] = correct[:, None] @ ref[1]
        mdp.motion.reference_bodies = lambda f: ref
        mdp.kin.forward = lambda *args: actual
        S = (S[0], robot_R, *S[2:])
        mask = torch.ones(1, 2)
        terms = reward_terms(mdp, S, O, torch.arange(1), S[4], S[4], mask, mask, mask, mask, mask[..., None])
        torch.testing.assert_close(terms["body_position"], torch.tensor([.02]))
        self.assertGreater(float(terms["body_orientation"]), .01999)

    def test_contact_cache_binds_by_hash_asset_name_and_motion_order(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)

            def save(filename, payload):
                path = directory / filename
                torch.save(payload, path)
                return {"path": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

            mounts = {"left": [.0415, .003, 0.], "right": [.0415, -.003, 0.]}
            points = torch.zeros(5, 3)
            hands = {side: {"mount": mount, "palm_points": points, "palm_points_sha256": point_sha256(points)}
                     for side, mount in mounts.items()}
            geom = dict(sdf_grid=torch.ones(3, 3, 3), grid_min=-torch.ones(3), grid_max=torch.ones(3),
                        metadata={"collision_hash": "a", "grid_axis_order": "zyx"})
            asset = {"object_asset": "asset", "exact_scale": [1.] * 3, "collision_hash": "a",
                     "collision_meshes": [], "asset_id": 3, "geometry_cache": save("geom.pt", geom)}
            clips, entries = [], []
            for i in range(2):
                ht = torch.full((2, 2), i)
                rec = save(f"labels{i}.pt", dict(expected_contact=ht, expected_valid=torch.ones_like(ht),
                                                 metadata={"label_schema": LABEL_SCHEMA}))
                clips.append(dict(source_motion_sha256=str(i), source_group=str(i), asset_id=3,
                                  state_frames=2, contact_cache=rec))
                entries.append(dict(motion_sha256=str(i), source_group=str(i), object_asset="asset"))
            bundle = dict(schema_version="supermimic_hoi_v1_bundle_v1", partition="train", clips=clips,
                          assets=[asset], hands={"geometry_cache": save("hands.pt", hands)})
            digest = _digest_json(bundle)
            path = directory / "bundle.json"
            path.write_text(json.dumps(dict(bundle, bundle_content_sha256=digest)))
            kin = SimpleNamespace(body_names=["right_wrist_yaw_link", "left_wrist_yaw_link"], tracked_idx=[0, 1])
            motion = SimpleNamespace(kin=kin, device="cpu", for_training=True, partition="train",
                                     entries=entries[::-1], clip_len=torch.tensor([2, 2]))
            current_asset = {**asset, "asset_id": 7}
            reward = V11Reward(motion, current_asset, path, expected_bundle_content_sha256=digest)
            self.assertEqual(reward.expected[:, 0].tolist(), [True, True, False, False])
            self.assertEqual(reward.wrists, [1, 0])
            (directory / "labels0.pt").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "cache SHA256"):
                V11Reward(motion, current_asset, path, expected_bundle_content_sha256=digest)
