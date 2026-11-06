"""Startup contract tests; real R525 CUDA acceptance belongs to native smoke."""
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hoi_residual.runtime import NEWTON_MUJOCO_MODULE, configure_warp_forward_only


class WarpRuntimeTest(unittest.TestCase):
    def fake_warp(self):
        # Match Warp's module-registration semantics: snapshot global defaults.
        modules = {}
        wp = SimpleNamespace(config=SimpleNamespace(enable_backward=True), __version__="1.17.0")

        def options(module):
            return modules.setdefault(module, {"enable_backward": wp.config.enable_backward,
                                               "fast_math": False, "fuse_fp": True})

        def set_options(value, module):
            options(module).update(value)

        wp.set_module_options, wp.get_module_options = set_options, options
        return wp

    def test_existing_and_future_modules_are_forward_only(self):
        wp = self.fake_warp()
        original = wp.get_module_options(NEWTON_MUJOCO_MODULE)
        self.assertTrue(original["enable_backward"])
        with patch.dict(sys.modules, {"warp": wp}):
            receipt = configure_warp_forward_only()
            self.assertEqual(configure_warp_forward_only(), receipt)
        self.assertFalse(original["enable_backward"])
        self.assertFalse(wp.get_module_options("later_simulator_module")["enable_backward"])
        self.assertFalse(original["fast_math"])
        self.assertTrue(original["fuse_fp"])
        self.assertEqual(receipt["module_overrides"][NEWTON_MUJOCO_MODULE], {"enable_backward": False})

    def test_policy_autograd_is_unchanged(self):
        wp = self.fake_warp()
        with torch.enable_grad(), patch.dict(sys.modules, {"warp": wp}):
            configure_warp_forward_only()
            self.assertTrue(torch.is_grad_enabled())
            policy = torch.nn.Linear(3, 1)
            optimizer = torch.optim.Adam(policy.parameters(), lr=2e-4)
            initial = policy.weight.detach().clone()
            policy(torch.ones(2, 3)).square().mean().backward()
            self.assertTrue(torch.isfinite(policy.weight.grad).all())
            optimizer.step()
            self.assertFalse(torch.equal(initial, policy.weight))
        with torch.no_grad(), patch.dict(sys.modules, {"warp": wp}):
            configure_warp_forward_only()
            self.assertFalse(torch.is_grad_enabled())

    def test_rejected_setting_fails_explicitly(self):
        wp = self.fake_warp()
        wp.get_module_options(NEWTON_MUJOCO_MODULE)
        wp.set_module_options = lambda options, module: None
        with patch.dict(sys.modules, {"warp": wp}), self.assertRaisesRegex(RuntimeError, "did not apply"):
            configure_warp_forward_only()


if __name__ == "__main__":
    unittest.main()
