"""Real CPU model/RMS/Adam transfer tests; never imports or starts training."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from supermimic.gmt_transfer import (load_transfer_checkpoint, normalize_transfer_launcher_args,
                                    validate_contract_transfer, validate_transfer_checkpoint)
from supermimic.networks import RunningNorm
from supermimic.privileged import SCHEMA
from supermimic.robot_metadata import G1_BODY_NAMES, G1_JOINT_NAMES


class TinyModel(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.rms = RunningNorm(inputs)
        self.net = nn.Sequential(nn.Linear(inputs, 8), nn.ELU(), nn.Linear(8, outputs))
    def forward(self, x):
        return self.net(self.rms(x))


def models_and_optimizers():
    models = [TinyModel(8, 6), TinyModel(9, 5), TinyModel(7, 1)]
    optimizers = [torch.optim.Adam(model.parameters(), lr=2e-4,
                                   betas=(.9, .95) if i == 1 else (.9, .999)) for i, model in enumerate(models)]
    return models, optimizers


def contract(dataset, manifest):
    return {**copy.deepcopy(SCHEMA), "body_names": G1_BODY_NAMES.copy(), "joint_names": G1_JOINT_NAMES.copy(),
            "physics": {"profile": "fengqiao_hw3_mujoco_v1", "implementation_sha256": {"env.py": "c"*64}},
            "analytic_heads": {"observation_dim": 99, "control_dt": .02,
                               "implementation_sha256": {"mdp.py": "d"*64}},
            "urdf_sha256": "e"*64, "dataset": dataset, "motion_manifest_sha256": manifest*64}


class TransferTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=cache)
        self.root = Path(self.temp.name)
        self.parent_dir, self.new_dir, self.snapshot_dir = (self.root / name for name in ("original_run", "new_run", "input"))
        self.snapshot_dir.mkdir()
        self.path = self.snapshot_dir / "parent_026000_full.pt"
        self.parent_contract, self.target_contract = contract("lafan1", "a"), contract("boneseed", "b")
        self.parent_args = {"dataset": "lafan1", "motion_dir": "old_motion", "log_dir": str(self.parent_dir),
                            "iterations": 30000, "wm_only_iters": 500, "goal_frames": 10, "hist": 10,
                            "horizon": 32, "amp": "bf16", "num_envs": 4096, "buffer_steps": 1000,
                            "lr": 2e-4, "gamma": .99, "lam": .95, "utd": .4, "wm_batch": 2048,
                            "pi_batch": 4096, "seed": 0, "resume": None, "no_critic": False}
        self.target_args = {**self.parent_args, "dataset": "boneseed", "motion_dir": "filtered_motion",
                            "log_dir": str(self.new_dir), "iterations": 20000, "wm_only_iters": 0,
                            "transfer_from": self.path, "transfer_parent_iteration": 26000}
        self.models, self.optimizers = models_and_optimizers()
        for _ in range(3):
            for model, optimizer in zip(self.models, self.optimizers):
                x = torch.randn(12, model.rms.mean.numel())
                model.rms.update(x)
                optimizer.zero_grad(set_to_none=True)
                model(x).square().mean().backward()
                optimizer.step()
        self.checkpoint = {"wm": self.models[0].state_dict(), "pi": self.models[1].state_dict(),
                           "critic": self.models[2].state_dict(), "opt_wm": self.optimizers[0].state_dict(),
                           "opt_pi": self.optimizers[1].state_dict(), "opt_v": self.optimizers[2].state_dict(),
                           "iteration": 26000, "wall_time": 12345., "sim_steps": 987654,
                           "args": self.parent_args, "gmt_contract": self.parent_contract,
                           "saved_at": "2026-11-06 04:00:00"}
        self.checkpoint = copy.deepcopy(self.checkpoint)
        torch.save(self.checkpoint, self.path)

    def tearDown(self):
        self.temp.cleanup()

    def assert_state_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for key in a:
                self.assert_state_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_state_equal(x, y)
        else:
            self.assertEqual(a, b)

    def validate(self, checkpoint=None, target_args=None, **kwargs):
        return validate_transfer_checkpoint(checkpoint or self.checkpoint, self.target_contract,
            target_args or self.target_args, source_checkpoint=self.path, run_dir=self.new_dir,
            expected_parent_iteration=26000, **kwargs)

    def load(self, destination=None, **kwargs):
        models, optimizers = destination or models_and_optimizers()
        result = load_transfer_checkpoint(self.path, self.target_contract, self.target_args,
            wm=models[0], pi=models[1], critic=models[2], opt_wm=optimizers[0], opt_pi=optimizers[1],
            opt_v=optimizers[2], run_dir=self.new_dir, expected_parent_iteration=26000, **kwargs)
        return result, models, optimizers

    def test_full_models_rms_and_three_adam_states_restore_exactly(self):
        before = self.path.read_bytes()
        result, models, optimizers = self.load()
        for name, model in zip(("wm", "pi", "critic"), models):
            self.assert_state_equal(model.state_dict(), self.checkpoint[name])
            self.assertEqual(float(model.rms.count), 36.)
        for name, optimizer in zip(("opt_wm", "opt_pi", "opt_v"), optimizers):
            self.assert_state_equal(optimizer.state_dict(), self.checkpoint[name])
        self.assertEqual(result["transfer"]["parent_iteration"], 26000)
        self.assertEqual(result["transfer"]["source_dataset"], "lafan1")
        self.assertEqual(result["transfer"]["target_dataset"], "boneseed")
        self.assertEqual(len(result["transfer"]["parent_checkpoint_sha256"]), 64)
        self.assertEqual(self.path.read_bytes(), before)

    def test_new_twenty_thousand_update_budget_keeps_parent_lr_offset_and_adam_age(self):
        result, models, optimizers = self.load()
        self.assertEqual((result["iteration"], result["wall_time"], result["sim_steps"]), (0, 0., 0))
        iterations = range(result["iteration"], result["target_iterations"])
        self.assertEqual((len(iterations), iterations[0], iterations[-1]), (20000, 0, 19999))
        self.assertEqual(result["learning_rate_iteration_offset"], 26000)
        self.assertEqual(iterations[-1]+result["learning_rate_iteration_offset"], 45999)
        model, optimizer = models[0], optimizers[0]
        old_steps = [float(s["step"]) for s in optimizer.state.values()]
        optimizer.zero_grad(set_to_none=True)
        model(torch.randn(4, 8)).square().mean().backward()
        optimizer.step()
        self.assertEqual([float(s["step"]) for s in optimizer.state.values()], [n+1 for n in old_steps])

    def test_pure_contract_check_allows_fixed_subset_without_full_checkpoint(self):
        full = copy.deepcopy(self.target_contract)
        subset = copy.deepcopy(full)
        subset["motion_manifest_sha256"] = "f"*64
        validate_contract_transfer(full, subset)
        validate_contract_transfer(full, full)
        validate_contract_transfer(self.parent_contract, self.target_contract)

    def test_any_other_contract_change_or_v2_is_rejected(self):
        for key, value in (("version", "robot_privileged_gmt_hw3_38_v2"), ("dim", 3283),
                           ("body_names", list(reversed(G1_BODY_NAMES))),
                           ("physics", {"implementation_sha256": {"env.py": "changed"}}),
                           ("analytic_heads", {"implementation_sha256": {"mdp.py": "changed"}})):
            changed = copy.deepcopy(self.target_contract)
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_contract_transfer(self.parent_contract, changed)
        changed = copy.deepcopy(self.target_contract)
        changed["unexpected_contract_field"] = True
        with self.assertRaisesRegex(ValueError, "contract change"):
            validate_contract_transfer(self.parent_contract, changed)

    def test_architecture_amp_and_other_existing_hyperparameters_must_match(self):
        for key, value in (("goal_frames", 13), ("hist", 8), ("horizon", 16), ("amp", "none"),
                           ("num_envs", 2048), ("lr", 1e-4), ("gamma", .95)):
            with self.assertRaisesRegex(ValueError, "matching parent argument"):
                self.validate(target_args={**self.target_args, key: value})
        self.validate()  # New transfer CLI fields are permitted.
        with self.assertRaisesRegex(ValueError, "matching parent argument: resume"):
            self.validate(target_args={**self.target_args, "resume": "other_resume.pt"})

    def test_reject_missing_models_optimizers_and_wrong_selected_parent(self):
        for key in ("critic", "opt_wm", "opt_pi", "opt_v"):
            changed = self.checkpoint.copy()
            changed.pop(key)
            with self.assertRaisesRegex(ValueError, "full models and optimizer"):
                self.validate(checkpoint=changed)
        changed = {**self.checkpoint, "iteration": 28000}
        with self.assertRaisesRegex(ValueError, "explicitly selected iteration"):
            self.validate(checkpoint=changed)

    def test_new_budget_and_warmup_must_be_explicit(self):
        for key, value in (("iterations", 46000), ("iterations", 30000), ("wm_only_iters", 500)):
            with self.assertRaises(ValueError):
                self.validate(target_args={**self.target_args, key: value})

    def test_original_output_and_source_snapshot_directory_are_protected(self):
        for output in (self.parent_dir, self.parent_dir / "nested", self.path.parent):
            args = {**self.target_args, "log_dir": str(output)}
            with self.assertRaises(ValueError):
                validate_transfer_checkpoint(self.checkpoint, self.target_contract, args,
                    source_checkpoint=self.path, run_dir=output, expected_parent_iteration=26000)

    def test_validate_all_destinations_before_mutating_any_model(self):
        changed = copy.deepcopy(self.checkpoint)
        changed["opt_v"]["state"][0]["exp_avg"] = torch.zeros(123)
        torch.save(changed, self.path)
        destination = models_and_optimizers()
        before = [copy.deepcopy(m.state_dict()) for m in destination[0]]
        with self.assertRaisesRegex(ValueError, "Adam moment dimensions"):
            self.load(destination)
        for model, expected in zip(destination[0], before):
            self.assert_state_equal(model.state_dict(), expected)

    def test_wrong_model_dimensions_and_optimizer_parameter_order_are_rejected(self):
        destination = models_and_optimizers()
        destination[0][1] = TinyModel(10, 5)
        with self.assertRaisesRegex(ValueError, "model tensor dimensions"):
            self.load(destination)
        destination = models_and_optimizers()
        destination[1][1] = torch.optim.Adam(reversed(list(destination[0][1].parameters())), lr=2e-4)
        with self.assertRaisesRegex(ValueError, "order/ownership"):
            self.load(destination)

    def test_missing_populated_adam_moments_or_group_options_are_rejected(self):
        for mutation in ("state", "lr"):
            changed = copy.deepcopy(self.checkpoint)
            if mutation == "state":
                changed["opt_pi"]["state"] = {}
            else:
                changed["opt_pi"]["param_groups"][0].pop("lr")
            torch.save(changed, self.path)
            with self.assertRaises(ValueError):
                self.load()

    def test_changing_parent_file_during_safe_load_is_rejected(self):
        original_load = torch.load
        def change_after_load(*args, **kwargs):
            self.assertIs(kwargs.get("weights_only"), True)
            result = original_load(*args, **kwargs)
            with self.path.open("ab") as stream:
                stream.write(b"changed")
            return result
        with patch("supermimic.gmt_transfer.torch.load", change_after_load):
            with self.assertRaisesRegex(ValueError, "changed while loading"):
                self.load()

    def test_launcher_defaults_reproduce_strict_failure_then_restore_full_state_after_normalization(self):
        self.checkpoint["args"].update(livestream=0, enable_cameras=False, kit_visualizer=False)
        torch.save(self.checkpoint, self.path)
        self.target_args.update(livestream=-1, enable_cameras=None, kit_visualizer=None)
        original = copy.deepcopy(self.target_args)
        with self.assertRaisesRegex(ValueError, "matching parent argument: livestream"):
            self.load()
        normalized = normalize_transfer_launcher_args(self.target_args)
        self.assertEqual(self.target_args, original)
        self.assertIsNot(normalized, self.target_args)
        self.assertEqual(normalized["livestream"], 0)
        self.assertIs(normalized["enable_cameras"], False)
        self.assertIs(normalized["kit_visualizer"], False)
        self.target_args = normalized
        result, models, optimizers = self.load()
        self.assertEqual(result["transfer"]["parent_iteration"], 26000)
        for key, model in zip(("wm", "pi", "critic"), models):
            self.assert_state_equal(model.state_dict(), self.checkpoint[key])
        for key, optimizer in zip(("opt_wm", "opt_pi", "opt_v"), optimizers):
            self.assert_state_equal(optimizer.state_dict(), self.checkpoint[key])

    def test_launcher_normalization_preserves_explicit_requests_and_all_other_arguments(self):
        for stream in (-1, None):
            source = {"livestream": stream, "enable_cameras": None, "kit_visualizer": None,
                      "horizon": 32, "resume": None, "other": [1, 2]}
            normalized = normalize_transfer_launcher_args(source)
            self.assertEqual(normalized, {**source, "livestream": 0, "enable_cameras": False, "kit_visualizer": False})
            self.assertEqual(source["livestream"], stream)
            self.assertIsNone(source["enable_cameras"])
        explicit = {"livestream": 2, "enable_cameras": True, "kit_visualizer": True, "horizon": 32}
        self.assertEqual(normalize_transfer_launcher_args(explicit), explicit)
        self.assertEqual(normalize_transfer_launcher_args({"horizon": 32}),
                         {"horizon": 32, "livestream": 0, "enable_cameras": False, "kit_visualizer": False})

    def test_missing_prelaunch_display_keys_match_parent_postlauncher_state(self):
        self.checkpoint["args"].update(livestream=0, enable_cameras=False, kit_visualizer=False)
        torch.save(self.checkpoint, self.path)
        # Pinned add_launcher_args only adds livestream; these two booleans
        # are first inserted by launch_simulation, after transfer must load.
        self.target_args["livestream"] = -1
        self.target_args.pop("enable_cameras", None)
        self.target_args.pop("kit_visualizer", None)
        original = copy.deepcopy(self.target_args)
        with self.assertRaisesRegex(ValueError, "matching parent argument"):
            self.load()
        normalized = normalize_transfer_launcher_args(self.target_args)
        self.assertEqual(self.target_args, original)
        self.assertNotIn("enable_cameras", self.target_args)
        self.assertNotIn("kit_visualizer", self.target_args)
        self.target_args = normalized
        result, models, optimizers = self.load()
        self.assertEqual(result["transfer"]["parent_iteration"], 26000)
        for key, model in zip(("wm", "pi", "critic"), models):
            self.assert_state_equal(model.state_dict(), self.checkpoint[key])
        for key, optimizer in zip(("opt_wm", "opt_pi", "opt_v"), optimizers):
            self.assert_state_equal(optimizer.state_dict(), self.checkpoint[key])

    def test_parent_enabled_launcher_features_are_not_waived_by_normalization(self):
        for key, value in (("livestream", 1), ("enable_cameras", True), ("kit_visualizer", True)):
            checkpoint = copy.deepcopy(self.checkpoint)
            checkpoint["args"].update(livestream=0, enable_cameras=False, kit_visualizer=False)
            checkpoint["args"][key] = value
            target = {**self.target_args, "livestream": -1, "enable_cameras": None, "kit_visualizer": None}
            normalized = normalize_transfer_launcher_args(target)
            with self.assertRaisesRegex(ValueError, "matching parent argument: "+key):
                self.validate(checkpoint=checkpoint, target_args=normalized)


if __name__ == "__main__":
    unittest.main()
