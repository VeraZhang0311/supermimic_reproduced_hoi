"""CPU protocol/AST checks; none of these certify a native GPU evaluation."""
import ast
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import random
from types import SimpleNamespace
import unittest

import torch

from tools import gmt_eval_worker as worker


class EvalWorkerTests(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=cache)
        self.root = Path(self.temp.name)
        self.source, self.run = self.root / "source", self.root / "run"
        self.source.mkdir(); self.run.mkdir()
        self.checkpoint = self.run / "ckpt_002000.pt"
        self.checkpoint.write_bytes(b"CPU fixture only; not a torch checkpoint")
        self.output = self.root / "results/002000.json"
        self.args = SimpleNamespace(source_root=self.source, run_dir=self.run,
                                    checkpoint=self.checkpoint, output=self.output, eval_envs=40)

    def tearDown(self):
        self.temp.cleanup()

    def run_fake(self, callback):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return worker.run_worker(self.args, native=callback)

    def test_filename_and_metadata_contract_are_strict(self):
        self.assertEqual(worker.checkpoint_iteration("ckpt_002000.pt"), 2000)
        for name in ("last.pt", "ckpt_002000.pt.tmp", "ckpt_2000.pt", "ckpt_002001.pt", "ckpt_000000.pt"):
            with self.assertRaises(ValueError):
                worker.checkpoint_iteration(name)
        contract = {"schema": "v3"}
        checkpoint = {"iteration": 2000, "gmt_contract": contract, "args": {"dataset": "lafan1"}}
        worker.validate_checkpoint_identity(checkpoint, 2000, contract)
        for changed in ({**checkpoint, "iteration": 4000}, {**checkpoint, "gmt_contract": {}},
                        {**checkpoint, "args": {"dataset": "boneseed"}}):
            with self.assertRaises(ValueError):
                worker.validate_checkpoint_identity(changed, 2000, contract)

    def test_ast_executes_only_evaluate_not_module_top_level(self):
        source = self.source / "evaluate.py"
        source.write_text("raise RuntimeError('watcher executed')\nimport missing_dependency\n"
                          "@torch.no_grad()\ndef evaluate(value):\n    return value+constant\n")
        evaluate = worker.load_evaluate_function(source, {"torch": torch, "constant": 7})
        self.assertEqual(evaluate(3), 10)
        source.write_text("def another_function(): pass\n")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            worker.load_evaluate_function(source, {"torch": torch})

    def test_original_evaluate_covers_40_and_matches_main_return_failure_semantics(self):
        names = [f"clip_{i:02}" for i in range(40)]
        starts = torch.arange(40)*3
        motion = SimpleNamespace(names=names, clip_start=starts, clip_end=starts+2,
                                 clip_len=torch.full((40,), 3), goal=lambda f, k: torch.zeros(len(f), 73*k))
        policy_instances = []
        class FakePolicy:
            def __init__(self, *a, **kw):
                self.training = True
                policy_instances.append(self)
            def to(self, dev): return self
            def load_state_dict(self, state): pass
            def eval(self): self.training = False
            def __call__(self, hist, goal, direct):
                if torch.is_grad_enabled() or self.training:
                    raise AssertionError("evaluation enabled gradients/training")
                if direct.shape != (40, 3371):
                    raise AssertionError("wrong privileged schema")
                return torch.zeros(40, 29)
        class FakeEnv:
            N = 40
            def __init__(self):
                self.done_eval = torch.zeros(40, dtype=torch.bool)
                self.failed_eval = self.done_eval.clone()
                self.a_prev, self.xi = torch.zeros(40, 29), torch.zeros(40, 33)
            def reset(self, ids, frames): self.f = frames.clone()
            def refresh(self): pass
            def state(self):
                z = torch.zeros(40, 3)
                return z, torch.eye(3).expand(40, -1, -1), z, z, torch.zeros(40, 29), torch.zeros(40, 29)
            def actuator_state(self): return torch.zeros(40, 88)
            def step(self, action):
                self.f += 1
                self.failed_eval[1] = True
                self.done_eval[1] = True
                self.done_eval |= self.f >= motion.clip_end
                return {"r": torch.full((40,), .1)}
        namespace = dict(torch=torch, Policy=FakePolicy, dev="cpu", networks=SimpleNamespace(AMP_DTYPE=None),
                         OBS_DIM=99, PRIVILEGED_DIM=3371, validate_checkpoint=lambda *a: None,
                         gmt_contract={}, motion=motion, C=40, MAX_STEP_REWARD=.1,
                         mdp=SimpleNamespace(obs=lambda *a: torch.zeros(40, 99)),
                         privileged=lambda *a: torch.zeros(40, 3371), clamp_action=lambda a: a)
        source = Path(__file__).resolve().parents[1] / "supermimic/evaluate.py"
        evaluate = worker.load_evaluate_function(source, namespace)
        result = evaluate(FakeEnv(), {"args": {"goal_frames": 10, "hist": 10, "amp": "none"}, "pi": {}})
        worker.validate_metrics(result, names)
        self.assertEqual(result["n_failed"], 1)
        self.assertEqual(result["failed"], ["clip_01"])
        self.assertEqual(result["per_clip"]["clip_01"], .5)
        self.assertEqual(result["per_clip"]["clip_00"], 1.)
        self.assertAlmostEqual(result["norm_return"], .9875, places=6)
        self.assertAlmostEqual(result["norm_return_framew"], .9875, places=6)
        self.assertEqual(len(policy_instances), 1)

    def test_metrics_require_every_clip_finite_values_and_consistent_failure_ids(self):
        names = [str(i) for i in range(40)]
        metrics = {"norm_return": .5, "norm_return_framew": .5, "n_failed": 0,
                   "failed": [], "per_clip": dict.fromkeys(names, .5)}
        worker.validate_metrics(metrics, names)
        for changed in ({**metrics, "norm_return": float("nan")}, {**metrics, "failed": ["0"]},
                        {**metrics, "n_failed": 2, "failed": ["0", "0"]}, {**metrics, "per_clip": {}}):
            with self.assertRaises(ValueError):
                worker.validate_metrics(changed, names)

    def test_success_is_atomic_exclusive_and_binds_immutable_checkpoint(self):
        digest = worker.sha256(self.checkpoint)
        callback = lambda args: {"checkpoint_iteration": 2000, "scope": {"cpu_test_double": True}}
        self.assertEqual(self.run_fake(callback), 0)
        result = json.loads(self.output.read_text())
        self.assertTrue(result["passed"])
        self.assertEqual(result["checkpoint_sha256"], digest)
        before = self.output.read_bytes()
        self.assertEqual(self.run_fake(callback), 1)
        self.assertEqual(self.output.read_bytes(), before)
        self.assertFalse(list(self.output.parent.glob("*.tmp")))

    def test_changed_checkpoint_never_publishes_success(self):
        def mutate(args):
            self.checkpoint.write_bytes(b"changed checkpoint")
            return {"checkpoint_iteration": 2000}
        self.assertEqual(self.run_fake(mutate), 1)
        self.assertFalse(self.output.exists())
        error = json.loads(Path(str(self.output)+".error.json").read_text())
        self.assertFalse(error["passed"])
        self.assertEqual(error["phase"], "immutability")

    def test_native_exception_publishes_only_redacted_failure(self):
        def fail(args):
            raise RuntimeError("api_key=secret-do-not-echo /private/path")
        self.assertEqual(self.run_fake(fail), 1)
        self.assertFalse(self.output.exists())
        text = Path(str(self.output)+".error.json").read_text()
        self.assertNotIn("secret-do-not-echo", text)
        self.assertNotIn("api_key", text)
        self.assertNotIn("/private/path", text)
        self.assertEqual(json.loads(text)["error_type"], "RuntimeError")

    def test_readonly_source_is_never_an_output_or_error_destination(self):
        self.args.output = self.source / "forbidden.json"
        self.assertEqual(self.run_fake(lambda args: {}), 1)
        self.assertEqual(list(self.source.iterdir()), [])

    def test_state_signature_detects_parameters_and_normalizer_buffer_updates(self):
        policy = torch.nn.Linear(2, 2)
        policy.register_buffer("count", torch.zeros(()))
        initial = worker.state_signature(policy)
        with torch.no_grad():
            policy.weight.add_(1)
        self.assertNotEqual(worker.state_signature(policy), initial)
        initial = worker.state_signature(policy)
        policy.count = torch.ones(())
        self.assertNotEqual(worker.state_signature(policy), initial)

    def test_namespace_package_accepts_frozen_source_without_init_file(self):
        package = SimpleNamespace(__file__=None, __path__=[str(self.source / "supermimic")])
        worker.validate_source_package(package, self.source.resolve())
        package.__path__.append(str(self.root / "other/supermimic"))
        with self.assertRaises(ValueError):
            worker.validate_source_package(package, self.source.resolve())

    def test_native_path_uses_safe_loader_and_frozen_nominal_environment(self):
        tree = ast.parse(Path(worker.__file__).read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_native")
        calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)]
        loads = [c for c in calls if isinstance(c.func, ast.Attribute) and isinstance(c.func.value, ast.Name)
                 and c.func.value.id == "torch" and c.func.attr == "load"]
        self.assertEqual(len(loads), 1)
        self.assertIs(next(k.value.value for k in loads[0].keywords if k.arg == "weights_only"), True)
        environments = [c for c in calls if isinstance(c.func, ast.Name) and c.func.id == "TrackingSim"]
        self.assertIs(next(k.value.value for k in environments[0].keywords if k.arg == "eval_mode"), True)
        self.assertFalse(any(isinstance(c.func, ast.Attribute) and c.func.attr in ("backward", "step", "update")
                             for c in calls))

    def bones_fixture(self):
        full_dir, eval_dir = self.root / "full", self.root / "eval512"
        full_dir.mkdir(); eval_dir.mkdir()
        full_clips, offset = [], 0
        for i in range(600):
            length = 3 + i % 3
            full_clips.append({"name": f"clip_{599-i:04}", "frames": length, "offset": offset,
                               "source_offset": offset, "fps": 50., "duration": (length-1)/50,
                               "npz_sha256": "a"*64, "source_sha256": "b"*64})
            offset += length
        full = {"format": "bones_seed_g1_float32_50hz_v1", "selection": "all_original_filtered_clips",
                "frames": offset, "clips": full_clips, "state_layout": "p3,quat_wxyz4,v3,w3,q29,qd29",
                "joint_names": [f"joint_{i}" for i in range(29)], "dtype": "<f4", "control_dt": .02,
                "source_fps": 50., "lineage": {"source_index_sha256": "c"*64}}
        (full_dir / "manifest.json").write_text(json.dumps(full))
        full_hash = worker.sha256(full_dir / "manifest.json")
        sorted_entries = sorted(enumerate(full_clips), key=lambda x: x[1]["name"])
        sampled = random.Random(0).sample(range(600), 512)
        entries, offset = [], 0
        for i in sampled:
            parent_index, parent = sorted_entries[i]
            entries.append(dict(parent, offset=offset, parent_clip_index=parent_index, parent_offset=parent["offset"]))
            offset += parent["frames"]
        selection = {"kind": "eval_training_subset", "held_out": False, "training_excludes_selection": False,
                     "seed": 0, "population_clips": 600, "sampled_sorted_indices": sampled,
                     "names": [entry["name"] for entry in entries], "parent_manifest_sha256": full_hash}
        (eval_dir / "selection.json").write_text(json.dumps(selection))
        evaluation = {**full, "selection": "eval_training_subset", "held_out": False,
                      "training_excludes_selection": False, "clips": entries, "frames": offset,
                      "parent_manifest_sha256": full_hash,
                      "selection_sha256": worker.sha256(eval_dir / "selection.json")}
        (eval_dir / "manifest.json").write_text(json.dumps(evaluation))
        contract = {"dataset": "boneseed", "motion_manifest_sha256": full_hash}
        # No full states.bin exists: lineage validation must never load or scan it.
        return full_dir, eval_dir, contract, evaluation, selection

    def test_bones512_validates_complete_fixed_selection_without_opening_full_states(self):
        full, evaluation, contract, manifest, selection = self.bones_fixture()
        identity = worker.validate_boneseed_eval_library(full, evaluation, contract)
        self.assertEqual(len(identity["names"]), 512)
        self.assertEqual(identity["names"], selection["names"])
        self.assertEqual(identity["eval_manifest_sha256"], worker.sha256(evaluation / "manifest.json"))
        self.assertEqual(identity["selection_sha256"], worker.sha256(evaluation / "selection.json"))
        self.assertFalse((full / "states.bin").exists())
        with self.assertRaises(ValueError):
            worker.validate_boneseed_eval_library(full, evaluation, {**contract, "dataset": "lafan1"})
        with self.assertRaises(ValueError):
            worker.validate_boneseed_eval_library(full, evaluation, {**contract, "motion_manifest_sha256": "0"*64})

    def test_bones512_rejects_bad_lineage_duplicates_crops_and_selection_rewrites(self):
        full, directory, contract, original, original_selection = self.bones_fixture()
        mutations = [lambda m, s: m.update(parent_manifest_sha256="0"*64),
                     lambda m, s: m.update(selection_sha256="0"*64),
                     lambda m, s: m.update(held_out=True),
                     lambda m, s: m.update(training_excludes_selection=True),
                     lambda m, s: m["clips"][0].update(frames=m["clips"][0]["frames"]-1),
                     lambda m, s: m["clips"][0].update(parent_offset=999),
                     lambda m, s: m["clips"][0].update(source_sha256="0"*64),
                     lambda m, s: m["clips"][0].update(name=m["clips"][1]["name"]),
                     lambda m, s: s["names"].reverse(),
                     lambda m, s: s.update(seed=1),
                     lambda m, s: s.update(held_out=True),
                     lambda m, s: s.update(parent_manifest_sha256="0"*64)]
        for i, mutate in enumerate(mutations):
            with self.subTest(i=i):
                manifest, selection = copy.deepcopy(original), copy.deepcopy(original_selection)
                mutate(manifest, selection)
                (directory / "selection.json").write_text(json.dumps(selection))
                if selection != original_selection:
                    # A rewritten hash alone must not permit a changed selection.
                    manifest["selection_sha256"] = worker.sha256(directory / "selection.json")
                (directory / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    worker.validate_boneseed_eval_library(full, directory, contract)

    def test_original_evaluate_runs_all512_in_two256_batches_with_unchanged_scoring(self):
        count, lanes = 512, 256
        names = [f"bones_{i:04}" for i in range(count)]
        starts = torch.arange(count) * 3
        motion = SimpleNamespace(names=names, clip_start=starts, clip_end=starts+2,
                                 clip_len=torch.full((count,), 3), goal=lambda f, k: torch.zeros(len(f), 73*k))
        run_contract = {"verified_training_contract": "full_manifest"}
        checks, resets = [], []

        def check(ck, contract):
            self.assertIs(contract, run_contract)
            self.assertEqual(ck["gmt_contract"], contract)
            checks.append(True)

        class FakePolicy:
            def __init__(self, *a, **kw): self.training = True
            def to(self, dev): return self
            def load_state_dict(self, state): pass
            def eval(self): self.training = False
            def __call__(self, hist, goal, direct):
                if torch.is_grad_enabled() or self.training or direct.shape != (lanes, 3371):
                    raise AssertionError("invalid fixed evaluation policy input/mode")
                return torch.zeros(lanes, 29)

        class FakeEnv:
            N = lanes
            def __init__(self):
                self.done_eval = torch.zeros(lanes, dtype=torch.bool)
                self.failed_eval = self.done_eval.clone()
                self.a_prev, self.xi = torch.zeros(lanes, 29), torch.zeros(lanes, 33)
            def reset(self, ids, frames):
                self.f = frames.clone()
                self.clips = frames // 3
                resets.append(self.clips.tolist())
            def refresh(self): pass
            def state(self):
                z = torch.zeros(lanes, 3)
                return z, torch.eye(3).expand(lanes, -1, -1), z, z, torch.zeros(lanes, 29), torch.zeros(lanes, 29)
            def actuator_state(self): return torch.zeros(lanes, 88)
            def step(self, action):
                self.f += 1
                fail = (self.clips == 1) | (self.clips == 257)
                self.failed_eval |= fail
                self.done_eval |= fail | (self.f >= motion.clip_end[self.clips])
                return {"r": torch.full((lanes,), .1)}

        namespace = dict(torch=torch, Policy=FakePolicy, dev="cpu", networks=SimpleNamespace(AMP_DTYPE=None),
                         OBS_DIM=99, PRIVILEGED_DIM=3371, validate_checkpoint=check,
                         gmt_contract=run_contract, motion=motion, C=count, MAX_STEP_REWARD=.1,
                         mdp=SimpleNamespace(obs=lambda *a: torch.zeros(lanes, 99)),
                         privileged=lambda *a: torch.zeros(lanes, 3371), clamp_action=lambda a: a)
        evaluate = worker.load_evaluate_function(Path(__file__).resolve().parents[1] / "supermimic/evaluate.py", namespace)
        result = evaluate(FakeEnv(), {"args": {"goal_frames": 10, "hist": 10, "amp": "none"},
                                      "gmt_contract": run_contract, "pi": {}})
        worker.validate_metrics(result, names, expected_count=512)
        self.assertEqual(checks, [True])
        self.assertEqual(resets, [list(range(256)), list(range(256, 512))])
        self.assertEqual(result["failed"], [names[1], names[257]])
        self.assertEqual(result["n_failed"], 2)
        self.assertEqual(result["per_clip"][names[257]], .5)
        self.assertEqual(result["per_clip"][names[511]], 1.)
        self.assertAlmostEqual(result["norm_return"], 511/512, places=6)
        self.assertAlmostEqual(result["norm_return_framew"], 511/512, places=6)

    def test_transfer_validation_allows_subset_manifest_only_with_identical_physics_and_heads(self):
        from supermimic.gmt_transfer import validate_contract_transfer
        from supermimic.privileged import SCHEMA
        from supermimic.robot_metadata import G1_BODY_NAMES, G1_JOINT_NAMES
        source = {**copy.deepcopy(SCHEMA), "body_names": G1_BODY_NAMES, "joint_names": G1_JOINT_NAMES,
                  "physics": {"motor": "same"}, "analytic_heads": {"reward": "same"},
                  "dataset": "boneseed", "motion_manifest_sha256": "a"*64}
        target = {**source, "motion_manifest_sha256": "b"*64}
        validate_contract_transfer(source, target)
        for key in ("physics", "analytic_heads", "body_names", "actuator_state"):
            with self.subTest(key=key):
                changed = {**target, key: {"changed": True}}
                with self.assertRaises(ValueError):
                    validate_contract_transfer(source, changed)

    def test_worker_allows256_envs_for512_and_keeps40_default_and_dataset_identity(self):
        self.args.expected_clips, self.args.eval_envs = 512, 256
        self.assertEqual(self.run_fake(lambda args: {"checkpoint_iteration": 2000, "n_total": 512}), 0)
        self.assertEqual(json.loads(self.output.read_text())["n_total"], 512)
        ck = {"iteration": 2000, "gmt_contract": {}, "args": {"dataset": "boneseed"}}
        worker.validate_checkpoint_identity(ck, 2000, {}, dataset="boneseed")
        with self.assertRaises(ValueError):
            worker.validate_checkpoint_identity(ck, 2000, {})


if __name__ == "__main__":
    unittest.main()
