"""Isolated-conversion cache/error protocol tests; no USD/importer dependency."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from supermimic import g1_usd


class IsolatedUsdTests(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=cache)
        self.asset = Path(self.temporary.name)
        (self.asset / "g1_hw3.xml").write_text("<mujoco/>", encoding="utf-8")
        self.verify = patch("supermimic.g1_assets.verify_assets", return_value={"identity": "verified-source"})
        self.verify_mock = self.verify.start()
        self.addCleanup(self.verify.stop)
        self.addCleanup(self.temporary.cleanup)

    def fake_success(self, command, **kwargs):
        self.assertEqual(command[:4], [g1_usd.sys.executable, "-m", "supermimic.g1_usd", "--worker"])
        self.assertTrue(kwargs["check"] is False)
        output = Path(command[command.index("--output-dir") + 1])
        usd = output / "robot" / "robot.usda"
        usd.parent.mkdir()
        usd.write_text("#usda 1.0\n", encoding="utf-8")
        payload = usd.parent / "payloads" / "physics.usda"
        payload.parent.mkdir()
        payload.write_text("#usda 1.0\n# payload", encoding="utf-8")
        source = g1_usd._source_identity(self.asset)
        hulls = {**g1_usd.HULL_POLICY, "collider_paths": {f"shape_{i}": f"/Robot/shape_{i}" for i in range(33)}, "schema_applied": True}
        g1_usd._write_receipt(output, usd, source, hulls)
        return SimpleNamespace(returncode=0)

    def test_converts_once_and_reuses_only_verified_receipt(self):
        with patch("supermimic.g1_usd.subprocess.run", side_effect=self.fake_success) as run:
            first = g1_usd.ensure_usd(self.asset)
            self.assertTrue(Path(first).is_file())
            self.assertEqual(g1_usd.ensure_usd(self.asset), first)
            self.assertEqual(run.call_count, 1)
            self.assertGreaterEqual(self.verify_mock.call_count, 3)

    def test_payload_tamper_is_rejected_without_fallback(self):
        with patch("supermimic.g1_usd.subprocess.run", side_effect=self.fake_success) as run:
            usd = Path(g1_usd.ensure_usd(self.asset))
            (usd.parent / "payloads/physics.usda").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                g1_usd.ensure_usd(self.asset)
            self.assertEqual(run.call_count, 1)

    def test_changed_xml_uses_new_cache_and_never_old_usd(self):
        with patch("supermimic.g1_usd.subprocess.run", side_effect=self.fake_success) as run:
            first = g1_usd.ensure_usd(self.asset)
            (self.asset / "g1_hw3.xml").write_text("<mujoco model='new'/>", encoding="utf-8")
            second = g1_usd.ensure_usd(self.asset)
            self.assertNotEqual(first, second)
            self.assertEqual(run.call_count, 2)

    def test_failed_worker_is_not_accepted_and_leaves_log(self):
        def failure(command, **kwargs):
            kwargs["stdout"].write("ApplyAPI Cannot find schema PhysxJointAPI\n")
            return SimpleNamespace(returncode=2)
        with patch("supermimic.g1_usd.subprocess.run", side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, "exited 2; inspect"):
                g1_usd.ensure_usd(self.asset)
        self.assertEqual(len(list((self.asset/"usd").glob("*.log"))), 1)
        with patch("supermimic.g1_usd.subprocess.run") as run:
            with self.assertRaisesRegex(RuntimeError, "completion receipt"):
                g1_usd.ensure_usd(self.asset)
            run.assert_not_called()

    def test_successful_exit_without_receipt_is_error(self):
        with patch("supermimic.g1_usd.subprocess.run", return_value=SimpleNamespace(returncode=0)):
            with self.assertRaisesRegex(RuntimeError, "completion receipt"):
                g1_usd.ensure_usd(self.asset)

    def test_unverified_inputs_never_start_converter(self):
        self.verify_mock.side_effect = ValueError("asset identity changed")
        with patch("supermimic.g1_usd.subprocess.run") as run:
            with self.assertRaisesRegex(ValueError, "asset identity changed"):
                g1_usd.ensure_usd(self.asset)
            run.assert_not_called()

    def test_cache_key_and_receipt_bind_unlimited_robot_mesh_hulls(self):
        import json
        with patch("supermimic.g1_usd.subprocess.run", side_effect=self.fake_success) as run:
            usd = Path(g1_usd.ensure_usd(self.asset))
            source = g1_usd._source_identity(self.asset)
            self.assertEqual(source["mesh_hull_policy"]["attribute"], "newton:maxHullVertices")
            self.assertEqual(source["mesh_hull_policy"]["value"], -1)
            self.assertEqual(source["mesh_hull_policy"]["mesh_collider_count"], 33)
            receipt_path = usd.parent.parent / g1_usd.RECEIPT_NAME
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["mesh_hull_policy"]["value"] = 64
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unlimited robot mesh-hull policy"):
                g1_usd.ensure_usd(self.asset)
            self.assertEqual(run.call_count, 1)

    def test_receipt_cannot_point_outside_cache(self):
        import json
        with patch("supermimic.g1_usd.subprocess.run", side_effect=self.fake_success):
            usd = Path(g1_usd.ensure_usd(self.asset))
        receipt_path = usd.parent.parent / g1_usd.RECEIPT_NAME
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["generated_usd"] = "../../../g1_hw3.xml"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "escapes its cache"):
            g1_usd.ensure_usd(self.asset)


if __name__ == "__main__":
    unittest.main()
