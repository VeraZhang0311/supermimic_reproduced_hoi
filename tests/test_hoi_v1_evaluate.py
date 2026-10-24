"""Frozen full-duration evaluation contracts with synthetic CPU worker states."""
from copy import deepcopy
import ast
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from hoi.data.manifest import sha256_file
from hoi.v1.evaluate import evaluate_group, validate_evaluation_identity, _load_fk39, _validate_contact_contract
from hoi.v1.motion import TrainingMotionBank
from hoi.v1.mdp import HOIMDP
from hoi.v1.networks import Policy
from hoi.v1.state import flatten_state, flatten_object, unflatten_state, unflatten_object
from tests.test_hoi_v1_learner import SyntheticKinematics, SyntheticSphereShapes


class Synthetic39Kinematics:
    """Explicit 39 bodies, distinct from the reward fixture's 14 bodies."""
    body_names = [f"link_{i}" for i in range(39)]
    dof_names = [f"joint_{i}" for i in range(29)]

    def forward(self, root, q, qd, root_velocity_is_com=False):
        assert root_velocity_is_com is False
        n = len(root)
        offset = torch.arange(39, dtype=root.dtype, device=root.device)[None, :, None] * root.new_tensor([.005, 0., 0.])
        return {"body_pos": root[:, None, :3] + offset,
                "body_rot": root[:, None, 3:7].expand(n, 39, 4).clone(),
                "body_vel": root[:, None, 7:10].expand(n, 39, 3).clone(),
                "body_ang_vel": root[:, None, 10:13].expand(n, 39, 3).clone()}


def fixture():
    kin, shapes, fk = SyntheticKinematics(), SyntheticSphereShapes(), Synthetic39Kinematics()
    clips = []
    # Nsource=5 -> T=3; Nsource=9 -> T=6, exercising unequal batch lengths.
    for clip_id, n in enumerate((4, 7)):
        p = torch.zeros(n, 3)
        p[:, 2] = .85
        p[:, 0] = torch.arange(n) * .001 + clip_id
        R = torch.eye(3).repeat(n, 1, 1)
        v, w, q = torch.zeros(n, 3), torch.zeros(n, 3), torch.zeros(n, 29)
        robot = {"root_pos": p, "root_rot": R, "derived_ref_v": v, "derived_ref_omega": w,
                 "joint_pos": q, "derived_ref_joint_qd": q.clone()}
        op = p + torch.tensor([.3, 0., .1])
        obj = {"pos": op, "rot": R.clone(), "derived_ref_v": v.clone(), "derived_ref_omega": w.clone()}
        root = torch.cat((p, torch.tensor([0., 0., 0., 1.]).repeat(n, 1), v, w), -1)
        body = fk.forward(root, q, q)
        source = torch.zeros(n, 630)
        source[:, :3], source[:, 6] = p, 1
        source[:, 71:74], source[:, 77] = op, 1
        source[:, 84:201] = body["body_pos"].flatten(-2)
        source[:, 201:357] = body["body_rot"].flatten(-2)
        clips.append({"name": f"synthetic_{clip_id}", "asset_id": 0, "reference": {"robot": robot, "object": obj, "source_sampled": source},
                      "expected_contact": torch.ones(n, 2), "expected_valid": torch.ones(n, 2),
                      "provenance": {"scope": "synthetic_CPU_test"}})
    motion = TrainingMotionBank(clips, kin, partition="eval80")
    mdp = HOIMDP(kin, motion, torch.zeros(29), shapes)
    policy = Policy(hidden=(32, 16), object_hidden=(16,))
    metadata = {"clips": [{"source_frames": 5}, {"source_frames": 9}]}
    return policy, motion, mdp, shapes, fk, metadata


class SyntheticWorkers:
    """No physics claims; inject failure at tick one and keep stepping."""
    def __init__(self, motion, mdp, *, reset_after_failure=False, robot_drop=.6, object_shift=.2):
        self.motion, self.mdp, self.N = motion, mdp, 2
        self.reset_count = self.steps = 0
        self.reset_after_failure = reset_after_failure
        self.robot_drop, self.object_shift = robot_drop, object_shift

    def reset(self, starts):
        self.reset_count += 1
        self.starts = starts.clone()
        self.ends = self.motion.clip_end[self.motion.clip_id[starts]]
        xi, xio = torch.zeros(2, 33), torch.zeros(2, 12)
        xi[:, 0], xio[:, 0] = .75, 1.
        self.value = {"s": flatten_state(self.motion.ref_state(starts)), "o": flatten_object(self.motion.ref_object(starts)),
                      "f": starts.clone(), "a_prev": torch.zeros(2, 29), "xi": xi, "xi_o": xio,
                      "ep_step": torch.zeros(2, dtype=torch.long), "asset_id": torch.zeros(2, dtype=torch.long),
                      "contact_missing_counter": torch.zeros(2, 2, dtype=torch.long)}
        return self.snapshot()

    def snapshot(self):
        return {key: value.clone() for key, value in self.value.items()}

    def step(self, action):
        before = self.snapshot()
        self.steps += 1
        f1 = torch.minimum(before["f"] + 1, self.ends)
        S1 = tuple(x.clone() for x in self.motion.ref_state(f1))
        O1 = tuple(x.clone() for x in self.motion.ref_object(f1))
        S1[0][:, 2] -= self.robot_drop
        O1[0][:, 0] += self.object_shift
        contacts = torch.zeros(2, 3)
        valid = torch.ones(2, 3, dtype=torch.bool)
        reward = self.mdp.reward(S1, O1, f1, action, before["a_prev"], contacts, valid)
        result = {"s": before["s"], "o": before["o"], "s1": flatten_state(S1), "o1": flatten_object(O1),
                  "a": action.clone(), "f": before["f"], "r": reward,
                  "term": self.mdp.termination(S1, O1, f1), "contact": contacts, "contact_valid": valid,
                  "impulse": torch.zeros(2, 3), "contact_force": torch.zeros(2, 3),
                  "contact_missing_counter": before["contact_missing_counter"] + 1}
        self.value.update(s=result["s1"], o=result["o1"], f=f1, a_prev=action.clone(),
                          ep_step=before["ep_step"] + 1, contact_missing_counter=result["contact_missing_counter"])
        if self.reset_after_failure:
            self.value["f"] = self.starts.clone()
            self.value["s"] = flatten_state(self.motion.ref_state(self.starts))
        return result


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)

    def test_evaluation_snapshot_binds_decoding_and_hash_to_same_bytes_and_distinguishes_sources(self):
        from hoi.v1.evaluate import _load_evaluation_snapshot, _evaluation_provenance, _verify_evaluation_source
        evaluation_source={"files":{"hoi/v1/evaluate.py":"e"*64},"content_sha256":"a"*64}
        training_source={"files":{"hoi/v1/evaluate.py":"f"*64},"content_sha256":"b"*64}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"checkpoint.pt"
            torch.save({"iteration":4,"extra":{"source_identity":training_source,"wm_updates_this_run":20,
                       "policy_updates_this_run":0}},path)
            original_bytes=path.read_bytes()
            original_load=torch.load
            def replace_path_after_snapshot(stream,**kwargs):
                self.assertTrue(hasattr(stream,"read"))
                torch.save({"iteration":99},path)
                return original_load(stream,**kwargs)
            with patch("hoi.v1.evaluate.source_identity",return_value=deepcopy(evaluation_source)), \
                    patch("hoi.v1.train.torch.load",side_effect=replace_path_after_snapshot):
                checkpoint,digest,provenance=_load_evaluation_snapshot(path)
            self.assertEqual(checkpoint["iteration"],4)
            self.assertEqual(digest,hashlib.sha256(original_bytes).hexdigest())
            self.assertNotEqual(digest,sha256_file(path))
            self.assertEqual(provenance["evaluation_source_identity"],evaluation_source)
            self.assertEqual(provenance["checkpoint_training_source_identity"],training_source)
            self.assertEqual(provenance["checkpoint_training_source_identity_status"],"recorded")
            self.assertEqual(provenance["checkpoint_training_progress"]["stage"],"wm_only_this_run")
            self.assertEqual(provenance["checkpoint_training_progress"]["wm_updates_this_run"],20)
            self.assertEqual(provenance["checkpoint_training_progress"]["policy_updates_this_run"],0)
            checkpoint["extra"]["source_identity"]["content_sha256"]="c"*64
            self.assertEqual(provenance["checkpoint_training_source_identity"]["content_sha256"],"b"*64)
        with patch("hoi.v1.evaluate.source_identity",return_value=deepcopy(evaluation_source)):
            unknown=_evaluation_provenance({})
            self.assertEqual(unknown["checkpoint_training_source_identity_status"],"unknown")
            self.assertNotIn("content_sha256",unknown["checkpoint_training_source_identity"])
            self.assertEqual(unknown["checkpoint_training_progress"]["stage"],"unknown")
            _verify_evaluation_source(evaluation_source)
        with patch("hoi.v1.evaluate.source_identity",return_value=training_source):
            with self.assertRaisesRegex(RuntimeError,"source changed"):
                _verify_evaluation_source(evaluation_source)

    def test_no_contact_objective_uses_its_own_bound_and_common_full_HOI_scale(self):
        from hoi.v1.variants import reward_config
        policy,motion,mdp,shapes,fk,metadata=fixture()
        mdp.cfg=reward_config("no_contact_objective")
        with tempfile.TemporaryDirectory() as temp:
            cases=evaluate_group(policy,motion,mdp,shapes,SyntheticWorkers(motion,mdp),[0,1],
                fk39=fk,joint_names=fk.dof_names,output=Path(temp)/"no_contact",
                metadata=metadata,verify_tracked_fk=False,variant="no_contact_objective")
            for case in cases:
                T=case["control_steps"]
                self.assertAlmostEqual(case["objective_max_step_reward"],.145)
                self.assertAlmostEqual(case["hoi_reference_mask_reward_upper_bound"],T*.145,places=6)
                self.assertAlmostEqual(case["hoi_normalized_return"],case["hoi_full_duration_reward"]/(T*.15),places=6)
                self.assertAlmostEqual(case["objective_normalized_return"],case["hoi_full_duration_reward"]/(T*.145),places=6)

    def test_full_duration_after_failure_and_exact_Tplus1_alignment(self):
        policy, motion, mdp, shapes, fk, metadata = fixture()
        before = deepcopy(policy.state_dict())
        workers = SyntheticWorkers(motion, mdp)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "eval"
            cases = evaluate_group(policy, motion, mdp, shapes, workers, [0, 1], fk39=fk, joint_names=fk.dof_names,
                                   output=output, metadata=metadata, verify_tracked_fk=False)
            self.assertEqual(workers.reset_count, 1)
            self.assertEqual(workers.steps, 6)
            for index, T in enumerate((3, 6)):
                self.assertEqual(cases[index]["control_steps"], T)
                self.assertTrue(cases[index]["full_reference_completed"])
                self.assertEqual(cases[index]["first_supermimic_failure_step"], 1)
                raw = torch.load(cases[index]["rollout"], weights_only=True)
                self.assertEqual(raw["states"]["s"].shape, (T + 1, 71))
                self.assertEqual(raw["states"]["o"].shape, (T + 1, 13))
                self.assertEqual(raw["states"]["body_pos"].shape, (T + 1, 39, 3))
                self.assertEqual(raw["states"]["body_rot"].shape, (T + 1, 39, 4))
                self.assertEqual(raw["states"]["reference_source630"].shape, (T + 1, 630))
                self.assertEqual(raw["control"]["action"].shape, (T, 29))
                self.assertEqual(raw["control"]["robot_observation"].shape, (T, 99))
                self.assertEqual(raw["control"]["object_goal"].shape, (T, 190))
                self.assertEqual(raw["time"].dtype, torch.float64)
                self.assertAlmostEqual(float(raw["time"][-1]), T / 50)
                self.assertEqual(raw["metadata"]["optimizer_updates"], 0)
                self.assertEqual(raw["metadata"]["normalization_updates"], 0)
                self.assertEqual(len(raw["metadata"]["evaluation_source_content_sha256"]),64)
                self.assertIsNone(raw["metadata"]["checkpoint_training_source_content_sha256"])
                self.assertEqual(raw["metadata"]["checkpoint_training_source_identity_status"],"unknown")
                self.assertEqual(raw["metadata"]["checkpoint_training_progress"]["stage"],"unknown")
                self.assertEqual(cases[index]["contact"]["missing_contact_intervals"], [T, T])
                self.assertEqual(cases[index]["contact"]["max_missing_streak_steps"], [T, T])
                human = raw["control"]["human_reward"]
                self.assertAlmostEqual(cases[index]["supermimic_norm_return"], float(human[:1].sum() / (.1 * T)), places=6)
            self.assertTrue((output / "evaluation.json").is_file())
        self.assertTrue(policy.training)
        for key, value in policy.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_failure_reset_is_rejected_instead_of_scored(self):
        policy, motion, mdp, shapes, fk, metadata = fixture()
        workers = SyntheticWorkers(motion, mdp, reset_after_failure=True)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "invalid"
            with self.assertRaisesRegex(RuntimeError, "reset|alignment"):
                evaluate_group(policy, motion, mdp, shapes, workers, [0, 1], fk39=fk, joint_names=fk.dof_names,
                               output=output, metadata=metadata, verify_tracked_fk=False)
            self.assertFalse((output / "evaluation.json").exists())

    def test_object_failure_does_not_truncate_author_human_return(self):
        policy, motion, mdp, shapes, fk, metadata = fixture()
        workers = SyntheticWorkers(motion, mdp, robot_drop=0., object_shift=2.)
        with tempfile.TemporaryDirectory() as temp:
            cases = evaluate_group(policy, motion, mdp, shapes, workers, [0, 1], fk39=fk, joint_names=fk.dof_names,
                                   output=Path(temp) / "object_failure", metadata=metadata, verify_tracked_fk=False)
            for case in cases:
                self.assertEqual(case["first_supermimic_failure_step"], -1)
                self.assertEqual(case["first_hoi_failure_step"], 1)
                raw = torch.load(case["rollout"], weights_only=True)
                expected = float(raw["control"]["human_reward"].sum() / (.1 * case["control_steps"]))
                self.assertAlmostEqual(case["supermimic_norm_return"], expected, places=6)

    def test_missing_expected_labels_reduce_only_masked_reward_upper_bound(self):
        policy, motion, mdp, shapes, fk, metadata = fixture()
        motion.expected_valid.zero_()
        workers = SyntheticWorkers(motion, mdp, robot_drop=0., object_shift=.02)
        with tempfile.TemporaryDirectory() as temp:
            cases = evaluate_group(policy, motion, mdp, shapes, workers, [0, 1], fk39=fk, joint_names=fk.dof_names,
                                   output=Path(temp) / "unknown_labels", metadata=metadata, verify_tracked_fk=False)
            for case in cases:
                T = case["control_steps"]
                self.assertAlmostEqual(case["hoi_reference_mask_reward_upper_bound"], T * .14, places=6)
                self.assertAlmostEqual(case["hoi_normalized_return"], case["hoi_full_duration_reward"] / (T * .15), places=6)
                self.assertAlmostEqual(case["hoi_reference_mask_normalized_return"], case["hoi_full_duration_reward"] / (T * .14), places=6)
                self.assertEqual(case["human_normalized_return"], case["supermimic_norm_return"])
                self.assertAlmostEqual(case["hoi_extra_full_duration_reward"], case["hoi_full_duration_reward"] - case["human_full_duration_reward"], places=6)
                self.assertIsNone(case["contact"]["expected_contact_coverage"])

    def test_existing_results_are_not_overwritten(self):
        policy, motion, mdp, shapes, fk, metadata = fixture()
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):
                evaluate_group(policy, motion, mdp, shapes, SyntheticWorkers(motion, mdp), [0, 1], fk39=fk,
                               joint_names=fk.dof_names, output=temp, metadata=metadata, verify_tracked_fk=False)

    def test_fourteen_body_fk_cannot_be_used_as_PHC39(self):
        policy, motion, mdp, shapes, fk, metadata = fixture()
        fk.body_names = fk.body_names[:14]
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "39-body"):
                evaluate_group(policy, motion, mdp, shapes, SyntheticWorkers(motion, mdp), [0, 1], fk39=fk,
                               joint_names=fk.dof_names, output=Path(temp) / "invalid", verify_tracked_fk=False)

    def test_cross_partition_identity_allows_split_but_rejects_contract_change(self):
        train = {"partition": "train", "manifest": {"sha256": "m"}, "timebase": {"source": 60, "control": 50},
                 "recipes": {"sha256": "r"}, "robot_fk": {"sha256": "g"}, "reference_kinematics": {"sha256": "u"},
                 "assets": [{"asset_id": 0, "object_asset": "sphere", "exact_scale": [1, 1, 1], "collision_hash": "c",
                             "native_asset_sha256": "n", "physical_profile": {"mass": 1.}, "geometry_cache": {"sha256": "s"}}],
                 "hands": {"geometry_cache": {"sha256": "h"}}}
        evaluation = deepcopy(train)
        evaluation["partition"] = "eval80"
        self.assertTrue(validate_evaluation_identity(train, evaluation))
        for key in ("manifest", "timebase", "recipes", "robot_fk", "reference_kinematics"):
            changed = deepcopy(evaluation)
            changed[key] = {"changed": True}
            with self.assertRaises(ValueError):
                validate_evaluation_identity(train, changed)
        changed = deepcopy(evaluation)
        changed["assets"][0]["physical_profile"]["mass"] = 2.
        with self.assertRaises(ValueError):
            validate_evaluation_identity(train, changed)
        changed = deepcopy(evaluation)
        changed["assets"][0]["geometry_cache"]["sha256"] = "different_shape_samples"
        with self.assertRaises(ValueError):
            validate_evaluation_identity(train, changed)

    def test_bound_real_ULTRA_kinematics_URDF_hash_and_joint_names(self):
        joint_names = [f"j{i}" for i in range(29)]
        body_names = [f"b{i}" for i in range(39)]
        links = "".join(f'<link name="{name}"/>' for name in body_names)
        joints = "".join(f'<joint name="{joint_names[i-1] if i <= 29 else f"fixed{i}"}" type="{"revolute" if i <=29 else "fixed"}">'
                         f'<parent link="b0"/><child link="b{i}"/><origin xyz="{i*.001} 0 0"/><axis xyz="0 0 1"/></joint>'
                         for i in range(1, 39))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ultra.urdf"
            path.write_text(f'<robot name="cpu_contract">{links}{joints}</robot>')
            bundle = {"reference_kinematics": {"urdf": {"path": str(path), "sha256": sha256_file(path)},
                                                "body_names": body_names, "dof_names": joint_names}}
            fk = _load_fk39(bundle, joint_names[::-1], "cpu")
            self.assertEqual(len(fk.body_names), 39)
            root = torch.tensor([[0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]])
            self.assertEqual(fk.forward(root, torch.zeros(1, 29), torch.zeros(1, 29))["body_pos"].shape, (1, 39, 3))
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                _load_fk39(bundle, joint_names, "cpu")

    def test_cpu_default_joint_positions_equal_original_function(self):
        from supermimic.robot_metadata import G1_JOINT_NAMES, DEFAULT_JOINT_POS, default_joint_pos_list
        source = Path(__file__).resolve().parents[1] / "supermimic" / "robot.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        original = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "default_joint_pos_list")
        # Execute only this trusted existing pure function, never the native
        # module imports/configuration, to check exact regex precedence parity.
        namespace = {"G1_JOINT_NAMES": G1_JOINT_NAMES, "DEFAULT_JOINT_POS": DEFAULT_JOINT_POS}
        exec(compile(ast.Module(body=[original], type_ignores=[]), str(source), "exec"), namespace)
        self.assertEqual(default_joint_pos_list(), namespace["default_joint_pos_list"]())
        self.assertEqual(len(default_joint_pos_list()), 29)

    def test_actual_contact_label_contract_cannot_silently_change(self):
        report = {"schema": "hoi_v1_pair_normal_impulse_v1", "threshold_N": 1., "num_substeps": 4,
                  "substep_dt": .005, "interval": "(t,t+1]",
                  "channels": ["left_hand_object", "right_hand_object", "object_ground"]}
        _validate_contact_contract(report)
        for changed in ({"threshold_N": .1}, {"num_substeps": 1}, {"interval": "last_substep"},
                        {"channels": ["left_wrist_net", "right_wrist_net", "support"]}):
            with self.assertRaises(ValueError):
                _validate_contact_contract({**report, **changed})


if __name__ == "__main__":
    unittest.main()
