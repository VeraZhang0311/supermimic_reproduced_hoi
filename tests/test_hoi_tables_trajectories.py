"""Native-state recording, full629-step coverage and artifact integrity."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hoi_tables import evaluate_hoi80 as evaluation
from hoi_tables.trajectories import (SCHEMA, DT, STATE_SHAPES, CONTROL_SHAPES,
                                     NativeTrajectory, _write_payload, validate_trajectory)
from supermimic.robot_metadata import G1_BODY_NAMES
from tests.test_hoi_tables_evaluate import Motion, Environment, FrozenPolicy


def fixture_trajectory(worker_root, clip, identity):
    """Real compressed files for the existing full80 report-contract fixtures."""
    worker_root = Path(worker_root)
    worker_root.mkdir(parents=True, exist_ok=True)
    n, steps = clip['state_frames'], clip['control_steps']
    arrays = {name: np.zeros((n, *shape), np.float32) for name, shape in STATE_SHAPES.items()}
    arrays.update({name: np.zeros((steps, *shape), np.float32) for name, shape in CONTROL_SHAPES.items()})
    for name in ('actual_quat', 'reference_quat', 'actual_object_quat', 'reference_object_quat'):
        arrays[name][..., 0] = 1
    for name in ('robot_state', 'reference_robot_state', 'object_state', 'reference_object_state'):
        arrays[name][:, 3] = 1
    arrays['reference_frame'] = np.arange(clip['reference_start_frame'], clip['reference_end_frame'] + 1)
    arrays.update(time=np.arange(n, dtype=np.float64) * DT, names=np.array(G1_BODY_NAMES),
                  canonical_reference=np.zeros((n, 630), np.float32))
    context = {key: identity[key] for key in ('iteration', 'parent_run_id', 'checkpoint_sha256', 'model_state_sha256')}
    return _write_payload(worker_root, arrays, clip, context, G1_BODY_NAMES)


class NativeData:
    def __init__(self, env): self.env = env

    @property
    def body_link_pose_w(self):
        pose = torch.zeros(1, 38, 7)
        pose[..., 6] = 1.  # native xyzw
        pose[..., :3] = self.env.state()[0][:, None] + self.env.scene.env_origins[:, None]
        # Deliberately different from reference FK; a renderer must retain it.
        pose[:, 1:, 0] += .123 + float(self.env.f.item()) * .0001
        return SimpleNamespace(torch=pose)


class NativeEnvironment(Environment):
    def __init__(self, motion):
        super().__init__(motion)
        self.robot = SimpleNamespace(body_names=list(G1_BODY_NAMES), data=NativeData(self))
        self.scene = SimpleNamespace(env_origins=torch.tensor([[12., 5., 0.]]))
        self.history = SimpleNamespace(reset=lambda ids, obs: None)
        self.initialized = False

    def object_state(self):
        p, R, v, w = super().object_state()
        return p, R, v + (torch.tensor([[1., 2., 3.]]) if self.initialized else 0), w

    def step(self, action):
        result = super().step(action)
        result.update(contact=torch.ones(1, 5), contact_valid=torch.ones(1, 5, dtype=torch.bool))
        return result


def teacher_kin():
    def forward(state):
        return tuple(v[:, None].expand(-1, 38, *v.shape[1:]) for v in state[:4])
    return SimpleNamespace(body_names=list(G1_BODY_NAMES), forward=forward)


class TrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def run_clip(self, steps, recording):
        motion = Motion(steps)
        motion.entries[0]['clip_id'] = 'ultra/canonical-test'
        env, kin = NativeEnvironment(motion), teacher_kin()
        policy = FrozenPolicy().eval().requires_grad_(False)
        mdp_kin = SimpleNamespace(forward=lambda *S: tuple(v[:, None].expand(-1, 14, *v.shape[1:]) for v in S[:4]))
        mdp = SimpleNamespace(kin=mdp_kin, termination=lambda S, f: f == 2)
        asset = {'physical_profile': {'mass': 6., 'com': [0, 0, 0], 'inertia': torch.eye(3).tolist(), 'friction': .9}}
        native = ModuleType('hoi_zero_shot.native')
        def initialize(e, reference):
            e.initialized = True
            return {'physics_steps': 0}
        native.initialize_canonical_object = initialize
        trajectory = NativeTrajectory(motion, kin) if recording else None
        reference = np.zeros((steps + 1, 630), np.float32)
        recorder = evaluation.CanonicalRecorder(env, reference, kin, trajectory)
        before = deepcopy(policy.state_dict()); rng = torch.random.get_rng_state().clone()
        with patch.dict('sys.modules', {'hoi_zero_shot.native': native}), \
             patch('hoi_residual.features.torso_pose', side_effect=lambda k, S: (S[0], S[1])):
            report = evaluation.rollout_clip(recorder, motion, mdp, lambda *a: torch.zeros(1, 3371),
                                               policy, 0, kin, torch.zeros(1, 128, 3), asset)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(all(torch.equal(before[k], policy.state_dict()[k]) for k in before))
        report.update(canonical_index=0, canonical_case_id='canonical_0')
        return report, env, trajectory, reference

    def test629_recording_is_readonly_and_retains_native_frames_after_failure(self):
        baseline, base_env, _, _ = self.run_clip(629, False)
        result, env, trajectory, reference = self.run_clip(629, True)
        self.assertEqual(result, baseline)
        self.assertEqual((env.calls, env.reset_count), (base_env.calls, base_env.reset_count))
        self.assertEqual((env.calls, env.reset_count), (629, 1))
        self.assertEqual((len(trajectory.frames), len(trajectory.controls)), (630, 629))
        self.assertEqual(result['first_failure_step'], 2)
        context = {'iteration': 2000, 'parent_run_id': 'owner', 'checkpoint_sha256': {'rank_00': 'a'*64},
                   'model_state_sha256': {'policy': 'b'*64}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); lane = root/'asset_00'; lane.mkdir()
            result['trajectory'] = trajectory.save(lane, result, context, reference)
            validate_trajectory(root, result, context, verify_payload=True)
            with np.load(root/result['trajectory']['path'], allow_pickle=False) as data:
                self.assertEqual(data['actual_pos'].shape, (630, 38, 3))
                self.assertEqual(data['action'].shape, (629, 29))
                self.assertEqual(data['canonical_reference'].shape, (630, 630))
                np.testing.assert_allclose(data['actual_pos'][:, 1, 0], .123 + np.arange(630)*.0001, atol=2e-6)
                np.testing.assert_array_equal(data['reference_pos'][:, 1, 0], 0)
                np.testing.assert_array_equal(data['object_state'][0, 7:10], [1., 2., 3.])
                np.testing.assert_array_equal(data['actual_quat'][0, 0], [1., 0., 0., 0.])
                self.assertEqual(int(data['reference_frame'][-1]), 629)
                self.assertEqual(int(data['failed'].sum()), 1)
            with self.assertRaises(FileExistsError): trajectory.save(lane, baseline, context, reference)

    def test_missing_tampered_swapped_and_cropped_artifacts_are_rejected(self):
        result, _, trajectory, reference = self.run_clip(5, True)
        context = {'iteration': 2000, 'parent_run_id': 'owner', 'checkpoint_sha256': {'rank_00': 'a'*64},
                   'model_state_sha256': {'policy': 'b'*64}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); lane = root/'asset_00'; lane.mkdir()
            result['trajectory'] = trajectory.save(lane, result, context, reference)
            with self.assertRaisesRegex(ValueError, 'checkpoint/run'):
                validate_trajectory(root, result, {**context, 'iteration': 4000})
            changed = deepcopy(result); changed['control_steps'] -= 1
            with self.assertRaisesRegex(ValueError, 'duration'):
                validate_trajectory(root, changed, context)
            changed = deepcopy(result); changed['clip_id'] = 'ultra/another-clip'
            with self.assertRaisesRegex(ValueError, 'identity'):
                validate_trajectory(root, changed, context)
            target = root/result['trajectory']['path']; original = target.read_bytes()
            target.write_bytes(original[:-8])
            with self.assertRaisesRegex(ValueError, 'checksum'): validate_trajectory(root, result, context)
            target.unlink()
            with self.assertRaisesRegex(ValueError, 'missing'): validate_trajectory(root, result, context)
            changed = deepcopy(result); changed['trajectory']['path'] = '../outside.npz'
            with self.assertRaisesRegex(ValueError, 'escapes'): validate_trajectory(root, changed, context)

    def test_nonfinite_and_incomplete_recordings_fail_before_publication(self):
        result, _, trajectory, reference = self.run_clip(5, True)
        context = {'iteration': 2000, 'parent_run_id': 'owner', 'checkpoint_sha256': {}, 'model_state_sha256': {}}
        with tempfile.TemporaryDirectory() as directory:
            lane = Path(directory)/'asset_00'; lane.mkdir()
            trajectory.frames[-1]['actual_pos'][0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, 'nonfinite'): trajectory.save(lane, result, context, reference)
            self.assertFalse((lane/'trajectories').exists())
            trajectory.frames.pop()
            with self.assertRaisesRegex(ValueError, 'complete scored clip'): trajectory.save(lane, result, context, reference)


if __name__ == '__main__': unittest.main()
