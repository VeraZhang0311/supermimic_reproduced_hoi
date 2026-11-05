"""Numerical protocol tests; no Isaac Lab, GPU, or external motion files needed."""
import ast
import csv
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from supermimic.boneseed import BoneseedMotionLib, load_csv, prepare
from supermimic.buffer import ReplayBuffer
from supermimic.collision import audit_self_collision
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, OBS_DIM, flatten_state, unflatten_state
from supermimic.motion import load_reference
from supermimic.networks import Policy, WorldModel
from supermimic.privileged import PRIVILEGED_DIM, PrivilegedFeatures, RobotKinematics39, validate_checkpoint
from supermimic.robot_metadata import G1_JOINT_NAMES, TRACKED_BODIES
from supermimic.rotations import so3_exp


def write_urdf(path):
    names = ["pelvis", *[n for n in TRACKED_BODIES if n != "pelvis"]]
    names += [f"link_{i}" for i in range(39-len(names))]
    root = ET.Element("robot", name="test_g1")
    mj = ET.SubElement(root, "mujoco")
    ET.SubElement(mj, "compiler", fusestatic="false", discardvisual="true")
    for name in names:
        link = ET.SubElement(root, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz="0.003 -0.002 0.001")
        ET.SubElement(inertial, "mass", value="1")
        ET.SubElement(inertial, "inertia", ixx=".01", iyy=".01", izz=".01", ixy="0", ixz="0", iyz="0")
    for i in range(1, 39):
        j = ET.SubElement(root, "joint", name=G1_JOINT_NAMES[i-1] if i <= 29 else f"fixed_{i}",
                          type="revolute" if i <= 29 else "fixed")
        ET.SubElement(j, "parent", link=names[i-1])
        ET.SubElement(j, "child", link=names[i])
        ET.SubElement(j, "origin", xyz=".01 .02 .03", rpy=".02 -.01 .03")
        ET.SubElement(j, "axis", xyz="0 1 0")
        ET.SubElement(j, "limit", lower="-3", upper="3", effort="100", velocity="30")
    ET.ElementTree(root).write(path)
    return names


def write_csv(path, shift=0., reorder=False):
    header = ["Frame", *["root_translate"+a for a in "XYZ"], *["root_rotate"+a for a in "XYZ"],
              *[n+"_dof" for n in G1_JOINT_NAMES]]
    data = np.zeros((121, len(header)))
    data[:, 0] = np.arange(121)
    data[:, 1] = np.arange(121) / 120 * 100 + shift
    data[:, 3] = 80
    data[:, 6] = 90
    data[:, 7:] = 30
    order = list(reversed(range(len(header)))) if reorder else list(range(len(header)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([header[i] for i in order])
        w.writerows(data[:, order])


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(3)
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        cls.tmp = tempfile.TemporaryDirectory(dir=cache)
        cls.root = Path(cls.tmp.name)
        cls.urdf = cls.root / "robot.urdf"
        cls.body_names = write_urdf(cls.urdf)
        # Independent MuJoCo URDF importer, preserving fixed link frames.
        model = mujoco.MjModel.from_xml_path(str(cls.urdf))
        xml = cls.root / "robot.xml"
        mujoco.mj_saveLastXML(str(xml), model)
        tree = ET.parse(xml)
        pelvis = tree.find("worldbody/body")
        ET.SubElement(pelvis, "freejoint", name="floating_base_joint")
        tree.write(xml)
        cls.mj = mujoco.MjModel.from_xml_path(str(xml))
        with patch("supermimic.kinematics.MJCF_PATH", str(xml)):
            cls.kin = G1Kinematics("cpu")
        cls.full = RobotKinematics39(cls.urdf)
        cls.source = cls.root / "source"
        write_csv(cls.source / "day1/a.csv")
        write_csv(cls.source / "day2/a.csv", shift=50., reorder=True)
        cls.library_path = cls.root / "library"
        prepare(cls.source, cls.library_path)
        cls.motion = BoneseedMotionLib(cls.library_path, cls.kin, "cpu")

    @classmethod
    def tearDownClass(cls):
        # Close memmap before Windows removes the fixture directory.
        del cls.motion.states, cls.motion._mapped
        del cls.motion
        import gc
        gc.collect()
        cls.tmp.cleanup()

    def test_native_units_header_mapping_and_time(self):
        s = load_csv(self.source / "day1/a.csv")
        s_reordered = load_csv(self.source / "day2/a.csv")
        self.assertEqual(s.shape, (51, 71))
        torch.testing.assert_close(s[:, 7], torch.ones(51), atol=4e-6, rtol=0)
        torch.testing.assert_close(s[0, 3:7], torch.tensor([2**-.5, 0., 0., 2**-.5]))
        torch.testing.assert_close(s[:, 13:42], torch.full((51, 29), torch.pi/6))
        torch.testing.assert_close(s_reordered[:, 1:], s[:, 1:])
        torch.testing.assert_close(s_reordered[:, 0]-s[:, 0], torch.full((51,), .5))
        self.assertAlmostEqual(float(s[-1, 0]), 1.)

    def test_lafan_fullset_requires_40_and_binds_every_clip(self):
        source = self.root / "lafan"
        source.mkdir()
        raw = np.zeros((31, 36))
        raw[:, 0] = np.linspace(0, 1, 31)
        raw[:, 2] = .8
        raw[:, 6] = 1  # xyzw identity
        for i in range(39):
            np.savetxt(source / f"motion_{i:02}.csv", raw, delimiter=",")
        with self.assertRaisesRegex(ValueError, "requires 40 CSV clips; found 39"):
            load_reference("lafan1", source, self.kin, "cpu")
        last = source / "motion_39.csv"
        np.savetxt(last, raw, delimiter=",")
        motion = load_reference("lafan1", source, self.kin, "cpu")
        self.assertEqual(motion.dataset, "lafan1")
        self.assertEqual(len(motion.names), 40)
        self.assertEqual(motion.clip_len.tolist(), [51]*40)
        torch.testing.assert_close(motion.root_p[motion.clip_end, 0], torch.ones(40))
        self.assertTrue((motion.bin_start < motion.clip_end[motion.bin_clip]).all())
        self.assertEqual(len(motion.bin_start), 40)
        f = motion.clip_start
        output = PrivilegedFeatures(self.full, motion)(motion.ref_state(f), f, torch.zeros(40, 29),
                                                      torch.zeros(40, 33), motion.q[f], motion.qd[f])
        self.assertEqual(output.shape, (40, PRIVILEGED_DIM))
        old_hash = motion.manifest_sha256
        raw[:, 0] += .1
        np.savetxt(last, raw, delimiter=",")
        changed = load_reference("lafan1", source, self.kin, "cpu")
        self.assertNotEqual(changed.manifest_sha256, old_hash)

    def test_reject_bad_frames_nonfinite_and_lafan_format(self):
        for kind in ("gap", "nan", "lafan"):
            path = self.root / f"bad_{kind}.csv"
            write_csv(path)
            with path.open() as stream:
                rows = list(csv.reader(stream))
            if kind == "gap":
                rows[5][0] = "88"
            elif kind == "nan":
                rows[4][2] = "nan"
            else:
                rows = rows[1:]
            with path.open("w", newline="") as f:
                csv.writer(f).writerows(rows)
            with self.assertRaises(ValueError):
                load_csv(path)

    def test_all_39_fk_poses_and_com_velocities_against_mujoco(self):
        S = list(self.motion.ref_state(torch.tensor([8])))
        S[1] = so3_exp(torch.tensor([[.1, -.2, .3]]))
        S[3] = torch.tensor([[.3, .5, -.2]])
        S[5] = torch.linspace(-.4, .6, 29).unsqueeze(0)
        p, R, v, w = self.full.forward(S)
        from supermimic.rotations import mat_to_quat
        data = mujoco.MjData(self.mj)
        data.qpos[:7] = torch.cat((S[0], mat_to_quat(S[1])), -1)[0].numpy()
        data.qvel[:3] = S[2][0].numpy()
        data.qvel[3:6] = (S[1].transpose(-1, -2) @ S[3].unsqueeze(-1))[0, :, 0].numpy()
        for j, name in enumerate(G1_JOINT_NAMES):
            joint = self.mj.joint(name)
            data.qpos[joint.qposadr[0]] = S[4][0, j]
            data.qvel[joint.dofadr[0]] = S[5][0, j]
        mujoco.mj_forward(self.mj, data)
        for i, name in enumerate(self.full.body_names):
            b = self.mj.body(name).id
            np.testing.assert_allclose(p[0, i].numpy(), data.xpos[b], atol=2e-6)
            np.testing.assert_allclose(R[0, i].numpy(), data.xmat[b].reshape(3, 3), atol=2e-6)
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.mj, data, mujoco.mjtObj.mjOBJ_BODY, b, velocity, 0)
            np.testing.assert_allclose(v[0, i].numpy(), velocity[3:], atol=3e-6)
            np.testing.assert_allclose(w[0, i].numpy(), velocity[:3], atol=3e-6)

    def test_library_clip_boundaries_goals_and_reward_parity(self):
        m = self.motion
        self.assertEqual(m.names, ["day1/a.csv", "day2/a.csv"])
        f = torch.tensor([49, 50, 51, 101])
        torch.testing.assert_close(m.future_frames(f, 13)[:, -1], torch.tensor([50, 50, 64, 101]))
        self.assertEqual(m.goal(f, 10).shape, (4, 730))
        torch.testing.assert_close(m.frame_bin[torch.tensor([0, 50, 51, 101])], torch.tensor([0, 0, 1, 1]))
        self.assertTrue((m.bin_start < m.clip_end[m.bin_clip]).all())
        mdp = MDP(self.kin, m, torch.zeros(29))
        S = m.ref_state(f)
        zero = torch.zeros(4, 29)
        # Remove only the known joint-limit penalty from this synthetic chain.
        reward, terms = mdp.reward(S, f, zero, zero, terms=True)
        torch.testing.assert_close(terms["track"], torch.full((4,), .1), atol=2e-6, rtol=0)
        self.assertTrue(torch.isfinite(reward).all())

    def test_privileged_shape_yaw_invariance_history_and_gradients(self):
        f = torch.tensor([8, 70])
        S = list(self.motion.ref_state(f))
        S[0] = S[0].clone().requires_grad_()
        S[4] = S[4].clone().requires_grad_()
        feature = PrivilegedFeatures(self.full, self.motion)
        a, xi = torch.randn(2, 29), torch.randn(2, 33)
        qprev, qdprev = S[4].detach()-.1, S[5].detach()+.2
        out = feature(S, f, a, xi, qprev, qdprev)
        self.assertEqual(out.shape, (2, PRIVILEGED_DIM))
        torch.testing.assert_close(out[:, -33:], xi)
        # q_prev appears after 583 current + 936 target/errors + a,q,qd.
        torch.testing.assert_close(out[:, 1606:1635], qprev)
        grad = torch.autograd.grad(out.square().mean(), S[4], retain_graph=True)[0]
        self.assertTrue(torch.isfinite(grad).all())
        eps = 1e-3
        plus, minus = list(S), list(S)
        delta = torch.zeros_like(S[4]); delta[0, 3] = eps
        plus[4], minus[4] = S[4]+delta, S[4]-delta
        finite = (feature(plus, f, a, xi, qprev, qdprev).square().mean() -
                  feature(minus, f, a, xi, qprev, qdprev).square().mean())/(2*eps)
        torch.testing.assert_close(grad[0, 3], finite, atol=2e-5, rtol=.02)
        # A joint world yaw/xy translation must leave the teacher invariant.
        H = so3_exp(torch.tensor([[0., 0., .7]])).expand(2, -1, -1)
        shift = torch.tensor([2., -3., 0.])
        def transform(state):
            p, R, v, w, q, qd = state
            rot = lambda x: (H @ x.unsqueeze(-1)).squeeze(-1)
            return rot(p)+shift, H@R, rot(v), rot(w), q, qd
        transformed_motion = SimpleNamespace(clip_end=self.motion.clip_end, clip_id=self.motion.clip_id,
                                             ref_state=lambda frames: transform(self.motion.ref_state(frames)))
        transformed = PrivilegedFeatures(self.full, transformed_motion)(transform(S), f, a, xi, qprev, qdprev)
        torch.testing.assert_close(out, transformed, atol=2e-5, rtol=2e-5)

    def test_h32_imagination_backward_with_robot_features(self):
        f = torch.tensor([0])
        S = self.motion.ref_state(f)
        feature = PrivilegedFeatures(self.full, self.motion)
        pi, wm = Policy(OBS_DIM, 730, privileged_dim=PRIVILEGED_DIM), WorldModel()
        for parameter in wm.parameters():
            parameter.requires_grad_(False)
        q_prev, qd_prev, a_prev = S[4], S[5], torch.zeros(1, 29)
        history, xi = torch.zeros(1, 10, OBS_DIM), torch.zeros(1, 33)
        actions = []
        for _ in range(32):
            a = pi(history, self.motion.goal(f, 10), feature(S, f, a_prev, xi, q_prev, qd_prev))
            q_prev, qd_prev = S[4], S[5]
            S = wm.step(S, a, xi)
            a_prev, f = a, f+1
            actions.append(a)
        (S[4].square().mean()+torch.stack(actions).square().mean()).backward()
        grads = [p.grad for p in pi.parameters() if p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))

    def test_batched_evaluation_covers_each_clip_and_counts_failure(self):
        # Exercise the production evaluator without importing/launching Isaac Lab.
        tree = ast.parse((Path(__file__).resolve().parents[1] / "supermimic/evaluate.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
        class FakePolicy:
            def __init__(self, *args, **kwargs): pass
            def to(self, dev): return self
            def eval(self): pass
            def load_state_dict(self, value): pass
            def __call__(self, hist, goal, direct): return torch.zeros(len(hist), 29)
        motion = self.motion
        class FakeEnv:
            N = 1
            def __init__(self):
                self.done_eval = torch.zeros(1, dtype=torch.bool)
                self.failed_eval = self.done_eval.clone()
                self.a_prev, self.xi = torch.zeros(1, 29), torch.zeros(1, 33)
                self.starts = []
            def reset(self, ids, frames):
                self.f = frames
                self.starts.extend(frames.tolist())
            def refresh(self): pass
            def state(self): return motion.ref_state(self.f)
            def step(self, action):
                self.f += 1
                if int(self.f[0]) == 53:  # clip 2 fails on its second transition
                    self.failed_eval[:] = True
                    self.done_eval[:] = True
                self.done_eval |= self.f >= motion.clip_end[motion.clip_id[self.f]]
                return {"r": torch.full((1,), .1)}
        env = FakeEnv()
        namespace = dict(torch=torch, TrackingSim=FakeEnv, dev="cpu", networks=SimpleNamespace(AMP_DTYPE=None),
                         Policy=FakePolicy, OBS_DIM=OBS_DIM, PRIVILEGED_DIM=PRIVILEGED_DIM,
                         validate_checkpoint=lambda *a: None, gmt_contract={}, motion=motion, C=2,
                         mdp=MDP(self.kin, motion, torch.zeros(29)), MAX_STEP_REWARD=.1,
                         privileged=PrivilegedFeatures(self.full, motion))
        exec(compile(ast.Module(body=[node], type_ignores=[]), "evaluate.py", "exec"), namespace)
        result = namespace["evaluate"](env, {"args": {"goal_frames": 10, "hist": 10}, "pi": {}})
        self.assertEqual(env.starts, [0, 51])
        self.assertEqual(result["n_failed"], 1)
        self.assertEqual(result["per_clip"], {"day1/a.csv": 1., "day2/a.csv": .04})

    def test_replay_retains_previous_joints_across_resets(self):
        buffer = ReplayBuffer(4, 2, "cpu")
        from supermimic.buffer import FIELDS
        for t in range(4):
            tr = {k: torch.zeros_like(v[0]) for k, v in buffer.data.items()}
            tr["q_prev"].fill_(t+10)
            tr["qd_prev"].fill_(t+20)
            tr["ep_step"].fill_(t % 2)
            buffer.add(tr)
        s, h = buffer.sample_starts(32, 10)
        self.assertTrue(((s["q_prev"] >= 10) & (s["q_prev"] <= 13)).all())
        torch.testing.assert_close(s["qd_prev"]-s["q_prev"], torch.full((32, 29), 10.))

    def test_policy_backward_and_contract_rejects_legacy(self):
        f = torch.tensor([8, 70])
        S = self.motion.ref_state(f)
        direct = PrivilegedFeatures(self.full, self.motion)(S, f, torch.zeros(2, 29), torch.zeros(2, 33), S[4], S[5])
        pi = Policy(OBS_DIM, 730, privileged_dim=PRIVILEGED_DIM)
        pi.privileged_norm.update(direct)
        count = pi.privileged_norm.count.clone()
        direct = direct.detach().requires_grad_()
        pi(torch.zeros(2, 10, OBS_DIM), self.motion.goal(f, 10), direct).square().sum().backward()
        self.assertTrue(torch.isfinite(direct.grad).all())
        torch.testing.assert_close(count, pi.privileged_norm.count)
        with self.assertRaises(ValueError):
            validate_checkpoint({"pi": {}}, {"version": "new"})
        validate_checkpoint({"gmt_contract": {"version": "new"}}, {"version": "new"})

    def test_self_collision_audit_preserves_parent_filter(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="root"><freejoint/>
            <geom size=".1"/><body name="adjacent" pos="0 0 .2"><joint/><geom size=".1"/>
            <body name="nonadjacent" pos="0 0 .2"><joint/><geom size=".1"/></body>
            </body></body></worldbody></mujoco>''')
        self.assertEqual(audit_self_collision(model)["allowed_nonadjacent_geom_pairs"], 1)
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        with self.assertRaises(RuntimeError):
            audit_self_collision(model)


if __name__ == "__main__":
    unittest.main()
