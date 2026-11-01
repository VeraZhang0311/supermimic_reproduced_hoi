"""CPU checks of the official option's exact scope; not GPU compatibility evidence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from hoi.v1.native_runtime import MODULE, configure_native_forward_compat, forward_entry_hashes, require_native_physics_driver


class NativeWarpCompatibilityTests(unittest.TestCase):
    def test_full_physics_rejects_old_or_missing_driver_before_runtime_mutation(self):
        for version in ((11,8), (12,0), None):
            wp = SimpleNamespace(get_cuda_driver_version=lambda: version)
            with self.assertRaisesRegex(RuntimeError, "CCD forward kernel has 4296"):
                require_native_physics_driver(wp)
        for version in ((12,1), (12,2), (13,0)):
            require_native_physics_driver(SimpleNamespace(get_cuda_driver_version=lambda: version))

    def test_geometry_only_cooking_bypasses_full_step_requirement(self):
        # No driver/module option API is accessed by this guard on cooking.
        require_native_physics_driver(SimpleNamespace(), cooking_only=True)

    def test_old_driver_changes_only_target_module_backward_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernels.py"
            path.write_text("original forward kernel source")
            module = SimpleNamespace(__file__=str(path))
            global_options = {"enable_backward": True, "fast_math": False, "fuse_fp": True}
            target = {**global_options, "block_dim": 256, "cuda_output": None}
            original = deepcopy(target)
            calls = []
            def set_options(options, *, module):
                calls.append((options, module))
                target.update(options)
            wp = SimpleNamespace(__version__="1.17.0", get_cuda_driver_version=lambda: (12,0),
                get_module_options=lambda arg: dict(target), set_module_options=set_options)
            with patch("hoi.v1.native_runtime.importlib.import_module", return_value=module) as load:
                report = configure_native_forward_compat(wp)
            load.assert_called_once_with(MODULE)
            self.assertEqual(calls, [({"enable_backward": False}, module)])
            self.assertEqual(target, {**original, "enable_backward": False})
            self.assertEqual(global_options, {"enable_backward": True, "fast_math": False, "fuse_fp": True})
            self.assertEqual(path.read_text(), "original forward kernel source")
            self.assertEqual(report["changed_options"], {"enable_backward": {"before": True, "after": False}})
            self.assertEqual(len(report["module_source_sha256"]), 64)
            self.assertTrue(report["applied"])

    def test_modern_or_no_cuda_does_not_import_or_change_physics_module(self):
        for version in ((12,1), (12,2), (13,0), None):
            wp = SimpleNamespace(__version__="1.17.0", get_cuda_driver_version=lambda: version)
            with patch("hoi.v1.native_runtime.importlib.import_module") as load:
                report = configure_native_forward_compat(wp)
            load.assert_not_called()
            self.assertFalse(report["applied"])
            self.assertEqual(report["changed_options"], {})

    def test_native_autograd_request_cannot_silently_disable_backward(self):
        with self.assertRaisesRegex(ValueError, "physics autograd"):
            configure_native_forward_compat(physics_autograd=True)

    def test_forward_comparison_ignores_only_symbol_hash_and_checks_math(self):
        source = 'extern "C" __global__ void move_123abcdf_cuda_kernel_forward(int x) { if (x) { y=x+1; } }'
        hashes = forward_entry_hashes(source)
        self.assertEqual(hashes, forward_entry_hashes(source.replace("123abcdf", "00000000")))
        self.assertNotEqual(hashes, forward_entry_hashes(source.replace("x+1", "x+2")))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            forward_entry_hashes(source[:-1])


if __name__ == "__main__":
    unittest.main()
