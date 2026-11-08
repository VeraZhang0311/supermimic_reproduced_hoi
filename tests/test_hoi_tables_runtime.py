"""CPU contracts for runtime wiring; these do not certify native simulation."""
import ast
import hashlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hoi_tables.contacts import CHANNELS, PairContacts, _kernel, geometry_roles
from hoi_tables.reward import TablesReward, current_hands
from hoi_residual.features import ObjectHistory
from hoi_residual.v11_reward import TERM_NAMES, reward_terms
from supermimic.mdp import flatten_state
from supermimic.rotations import mat_to_quat, rot_angle_sq


def robot_state(n):
    return (torch.zeros(n, 3), torch.eye(3).expand(n, 3, 3), torch.zeros(n, 3),
            torch.zeros(n, 3), torch.zeros(n, 29), torch.zeros(n, 29))


def reward_fixture(n=2):
    reference = robot_state(n)
    O = reference[:4]
    bodies = tuple(x[:, None].expand(-1, 14, *x.shape[1:]) for x in O)
    kin = SimpleNamespace(forward=lambda *args: bodies)
    motion = SimpleNamespace(root_p=reference[0], root_R=reference[1],
                             reference_bodies=lambda f: bodies, object_ref=lambda f: O,
                             clip_asset_ids=torch.tensor([0, 1]), clip_id=torch.arange(n))
    mdp = SimpleNamespace(kin=kin, motion=motion, soft_lo=-torch.ones(29), soft_hi=torch.ones(29),
                          termination=lambda S, f: S[0][:, 2] < -.25,
                          soft_termination=lambda S, f: torch.sigmoid((-S[0][:, 2]-.25)/.02))
    reward = object.__new__(TablesReward)
    reward.mdp, reward.motion, reward.kin = mdp, motion, kin
    reward.geometry = {0: {"bbox_diagonal": torch.tensor(.4)}, 1: {"bbox_diagonal": torch.tensor(.8)}}
    expected = torch.tensor([[1., 0.]]).expand(n, -1)
    valid = torch.ones(n, 2)
    reward._expected = lambda f: (expected, valid)
    reward.hand_distances = lambda B, O, f: torch.zeros(n, 2, 8)
    return reward, reference, O, expected, valid


class _WarpPython:
    """Execute the exact recorder kernel body as scalar Python, without Warp."""
    index = 0
    kernel = staticmethod(lambda fn: fn)
    tid = lambda self: self.index
    max = staticmethod(max)

    @staticmethod
    def atomic_add(array, world, channel, value):
        array[world, channel] += value


def table_class_without_native_imports():
    # Compile only the unchanged TableSim class body from the actual source.
    # Native dependencies are replaced by explicit CPU test doubles.  No GPU
    # backend import or real physics claim is made by these bookkeeping tests.
    path = Path(__file__).resolve().parents[1]/"hoi_tables/env.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TableSim")
    namespace = {"torch": torch, "HOISim": object, "flatten_state": flatten_state,
                 "mat_to_quat": mat_to_quat, "rot_angle_sq": rot_angle_sq,
                 "torso_pose": lambda kin, S: (S[0], S[1]),
                 "object_observation": lambda torso, O: torch.cat((O[0]-torso[0], O[1][..., :, 0], O[1][..., :, 2]), -1),
                 "audit_object_solver": lambda env: {"passed": True}}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["TableSim"]


class TestFiveChannelContacts(unittest.TestCase):
    def test_exact_order_and_no_ground_self_or_other_body_leak(self):
        self.assertEqual(CHANNELS, ("left_hand_object", "right_hand_object", "object_ground", "left_foot_object", "right_foot_object"))
        wp = _WarpPython()
        kernel = _kernel(wp)
        roles = np.array([0, 1, 2, 3, 4, 5, -1], dtype=np.int32)
        # Both pair orientations, two worlds, foot-ground, hand-ground,
        # self-contact, arbitrary robot-object and inactive contact slots.
        pairs = np.array([[0, 2], [2, 1], [5, 2], [2, 3], [4, 2], [3, 5],
                          [0, 5], [0, 1], [6, 2], [-1, 2], [2, 0]])
        worlds = np.array([0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0])
        wrench = np.zeros((len(pairs), 6))
        wrench[:, 0] = 8
        impulse = np.zeros((2, 5))
        for i in range(len(pairs)):
            wp.index = i
            kernel(np.array([10]), pairs, worlds, roles, wrench, .005, impulse)
        np.testing.assert_allclose(impulse, [[.04, 0, .04, 0, .04], [0, .04, 0, .04, 0]])
        # Negative signed solver force never becomes a positive contact label.
        wp.index = 0
        wrench[0, 0] = -30
        kernel(np.array([10]), pairs, worlds, roles, wrench, .005, impulse)
        self.assertAlmostEqual(impulse[0, 0], .04)

    def test_foot_and_plane_identity_require_active_colliders(self):
        import mujoco
        roles = np.array([0, 0, 0, 1, 1, 1, 2, -1, -1, -1, -1])
        model = SimpleNamespace(geom_bodyid=np.array([1, 1, 1, 2, 2, 2, 3, 4, 5, 0, 6]),
                                geom_type=np.array([2]*9+[int(mujoco.mjtGeom.mjGEOM_PLANE), 2]),
                                geom_contype=np.ones(11, dtype=int), geom_conaffinity=np.zeros(11, dtype=int))
        with patch("hoi_tables.contacts._geom_roles", side_effect=lambda *args: roles.copy()), \
             patch("hoi_tables.contacts._entity_mapping", return_value=({"left_ankle_roll_link": 4, "right_ankle_roll_link": 5}, {})):
            actual = geometry_roles(model, None, None, 1)
            self.assertEqual(actual.tolist(), [0, 0, 0, 1, 1, 1, 2, 3, 4, 5, -1])
            for index in (7, 8, 9):
                model.geom_contype[index] = 0
                with self.assertRaisesRegex(RuntimeError, "Inactive"):
                    geometry_roles(model, None, None, 1)
                model.geom_contype[index] = 1

    def test_interval_threshold_reset_and_substep_count(self):
        class Impulse:
            value = torch.zeros(2, 5)

            def zero_(self):
                self.value.zero_()

        recorder = object.__new__(PairContacts)
        recorder.impulse = Impulse()
        recorder.wp = SimpleNamespace(to_torch=lambda value: value.value)
        first_interval = torch.tensor([[.02, .0201, 0, 0, 0], [0, 0, .01, 0, 0]])
        wanted = first_interval.clone()

        def accumulate(dt):
            recorder.impulse.value.add_(wanted/4)
            recorder.substeps += 1

        recorder._after_substep = accumulate
        recorder.physics = SimpleNamespace(step_target=lambda target, after_substep: [after_substep(.005) for _ in range(4)])
        labels, valid = recorder.step(None)
        self.assertFalse(bool(labels[0, 0]))
        self.assertTrue(bool(labels[0, 1]))
        self.assertFalse(bool(labels[1, 2]))
        self.assertTrue(bool(valid.all()))
        saved = recorder.last_force
        wanted.zero_()
        self.assertFalse(bool(recorder.step(None)[0].any()))
        self.assertGreater(saved[0, 1].item(), 1)
        recorder.physics.step_target = lambda target, after_substep: after_substep(.005)
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            recorder.step(None)
        with self.assertRaisesRegex(RuntimeError, "timebase"):
            PairContacts._after_substep(recorder, .01)

    def test_native_world_axis_must_match_environment_count(self):
        data = SimpleNamespace(efc=SimpleNamespace(force=SimpleNamespace(shape=(1, 64))))
        solver = SimpleNamespace(mjw_data=data)
        env = SimpleNamespace(physics=SimpleNamespace(solver=solver), N=2)
        with patch.dict(sys.modules, {"mujoco_warp": SimpleNamespace(), "warp": SimpleNamespace()}):
            with self.assertRaisesRegex(RuntimeError, "one solver world"):
                PairContacts(env)


class TestV11RewardAdapter(unittest.TestCase):
    def test_source_stl_duplicate_vertices_are_welded_without_asset_changes(self):
        import trimesh
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            meshes = root/"source/meshes/g1"
            meshes.mkdir(parents=True)
            path = meshes/"rubber_hand_palm_hull.stl"
            trimesh.creation.box(extents=[.04]*3).export(path)
            source = path.read_bytes()
            self.assertGreater(len(trimesh.load_mesh(path, process=False).vertices), 8)
            colliders = ''.join(f'<collision><origin xyz="{x} 0 0" rpy="0 0 0"/>'
                                '<geometry><mesh filename="rubber_hand_palm_hull.stl"/></geometry></collision>'
                                for x in (0., .05, .1))
            urdf = root/"robot.urdf"
            urdf.write_text('<robot name="mock">'+''.join(f'<link name="{side}_wrist_yaw_link">{colliders}</link>'
                                                           for side in ("left", "right"))+'</robot>')
            calibration = {side: {"palm_points": torch.tensor([[.02, 0., 0.], [0., .02, 0.]]), "mount": [0., 0., 0.]}
                           for side in ("left", "right")}
            binding_path = root/"calibration.pt"
            torch.save(calibration, binding_path)
            binding = {"path": str(binding_path), "sha256": hashlib.sha256(binding_path.read_bytes()).hexdigest()}
            with patch("hoi_tables.reward.URDF_PATH", str(urdf)), patch("supermimic.g1_physics.ASSET_DIR", root):
                hands, evidence = current_hands(binding)
            for side in ("left", "right"):
                self.assertEqual(len(hands[side]["collision_meshes"]), 3)
                self.assertTrue(all(len(m["vertices"]) == 8 for m in hands[side]["collision_meshes"]))
                self.assertLess(evidence[side]["calibrated_to_current_projection_max_m"], 1e-6)
            self.assertEqual(source, path.read_bytes())

    def test_all_fourteen_terms_equal_frozen_v11_and_only_hands_rewarded(self):
        reward, S, O, expected, mask = reward_fixture()
        f = torch.arange(2)
        a, a_prev = torch.ones(2, 29)*.1, torch.zeros(2, 29)
        contacts = torch.tensor([[1., 0., 0., 0., 0.], [1., 1., 1., 1., 1.]])
        valid = torch.ones_like(contacts)
        actual = reward.terms(S, O, f, a, a_prev, contacts, valid)
        baseline = reward_terms(reward.mdp, S, O, f, a, a_prev, expected, mask,
                                contacts[:, :2], valid[:, :2], torch.zeros(2, 2, 8))
        self.assertEqual(tuple(actual), TERM_NAMES)
        for key in TERM_NAMES:
            torch.testing.assert_close(actual[key], baseline[key], rtol=0, atol=0)
        contacts[:, 2:] = 1-contacts[:, 2:]
        valid[:, 2:] = 0
        changed = reward.terms(S, O, f, a, a_prev, contacts, valid)
        for key in TERM_NAMES:
            torch.testing.assert_close(actual[key], changed[key], rtol=0, atol=0)

    def test_termination_uses_v11_bbox_threshold_and_smooth_object_gradient(self):
        reward, S, O, _, _ = reward_fixture()
        f = torch.arange(2)
        torch.testing.assert_close(reward.object_failure_threshold(f), torch.tensor([1., 1.6]))
        position = torch.tensor([[1.001, 0., 0.], [1.599, 0., 0.]], requires_grad=True)
        obj = (position, *O[1:])
        self.assertEqual(reward.termination(S, obj, f).tolist(), [True, False])
        reward.soft_termination(S, obj, f).sum().backward()
        self.assertTrue(bool(torch.isfinite(position.grad).all()))
        self.assertTrue(bool((position.grad[:, 0] > 0).all()))


class TestNativeBookkeepingContract(unittest.TestCase):
    def make_env(self):
        Sim = table_class_without_native_imports()
        env = object.__new__(Sim)
        env.N, env.device, env.eval_mode, env.gamma = 2, "cpu", False, .99
        env.teacher_kin = None
        env._S, env._O = robot_state(2), robot_state(2)[:4]
        env.state, env.object_state = lambda: env._S, lambda: env._O
        env.xi = torch.zeros(2, 33)
        env.object_context = lambda: torch.zeros(2, 12)
        env.actuator_state = lambda: torch.full((2, 88), 4.)
        env.robot = SimpleNamespace(data=SimpleNamespace(body_mass=SimpleNamespace(torch=torch.tensor([[3.], [3.]]))))
        env.torso_id, env._nominal_torso_mass = 0, torch.tensor([3., 3.])
        env.f, env.a_prev = torch.tensor([0, 0]), torch.zeros(2, 29)
        env.q_prev, env.qd_prev = torch.ones(2, 29), torch.ones(2, 29)*2
        env.history = ObjectHistory(torch.zeros(2, 9))
        env.episode_id, env.start_bin, env.ep_step = torch.tensor([0, 1]), torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long)
        env.asset_ids = torch.zeros(2, dtype=torch.long)
        return env

    def test_snapshot_is_owned_and_partition_marked(self):
        env = self.make_env()
        snapshot = env.snapshot()
        env.xi.add_(5)
        env.q_prev.zero_()
        env.history.values.add_(1)
        env.f.add_(3)
        env.episode_id.add_(2)
        self.assertEqual(snapshot["xi"].sum().item(), 0)
        self.assertEqual(snapshot["q_prev"].min().item(), 1)
        self.assertEqual(snapshot["object_history"].sum().item(), 0)
        self.assertEqual(snapshot["f"].tolist(), [0, 0])
        self.assertEqual(snapshot["episode_id"].tolist(), [0, 1])
        self.assertEqual(snapshot["partition"], "train")
        env.eval_mode = True
        self.assertEqual(env.snapshot()["partition"], "protected_eval")

    def test_physical_step_labels_next_state_before_any_reset(self):
        env = self.make_env()
        env.default_q, env.alpha = torch.zeros(29), torch.ones(29)
        env._check_object_after_step, env.object_solver_checks = False, 0
        env.motion = SimpleNamespace(clip_id=torch.zeros(2, dtype=torch.long), clip_end=torch.tensor([1]),
                                     object_ref=lambda f: env._O, reference_bodies=lambda f: (env._S[0][:, None],))
        env.mdp = SimpleNamespace(kin=SimpleNamespace(forward=lambda *args: (args[0][:, None],)))
        env.max_episode_steps = 100
        env.quality_successes, env.ep_disc_ret, env.ep_ret = torch.zeros(2), torch.zeros(2), torch.zeros(2)
        env.reward_model = SimpleNamespace(terms=lambda S, O, f, *args: {"body_position": f.float()},
                                            termination=lambda *args: torch.tensor([False, True]))
        env.contacts = SimpleNamespace(step=lambda target: (torch.zeros(2, 5), torch.ones(2, 5)))
        captured = {}

        def update(bins, **kwargs):
            captured.update(kwargs)

        env.sampler = SimpleNamespace(update=update)
        result = env.step(torch.ones(2, 29)*.4)
        self.assertEqual(result["done"].tolist(), [True, True])
        self.assertEqual(env.f.tolist(), [1, 1])
        self.assertEqual(env.episode_id.tolist(), [0, 1])
        self.assertEqual(captured["failure_frames"].tolist(), [-1, 1])
        self.assertEqual(captured["success"].tolist(), [True, False])
        self.assertEqual(result["reward"].tolist(), [1., 1.])
        self.assertEqual(env.ep_step.tolist(), [1, 1])
        self.assertTrue(torch.equal(env.a_prev, torch.full((2, 29), .4)))
        # There is no reset in step: the collector can still snapshot both
        # terminal states and actuator contexts before invoking reset(ids).
        self.assertEqual(env.snapshot()["f"].tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()
