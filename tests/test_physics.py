"""Collision profiles, immutable asset reuse and actual MuJoCo filter checks."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from supermimic.physics import (PhysicsSettings, add_physics_arguments, resolve_arguments,
    validate_resume, configure_robot, load_hand_asset, compiled_self_collision_audit)
from hoi.hand_collision import HAND_SHA, RECIPE


def parse(flags=(), saved=None, legacy=False):
    parser = argparse.ArgumentParser()
    add_physics_arguments(parser)
    args = parser.parse_args(flags)
    return resolve_arguments(args, saved, legacy)


class PhysicsTests(unittest.TestCase):
    def test_explicit_foot_overlay_validates_original_hand_asset_without_relabelling_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            base.mkdir()
            source = base / "source.usda"
            source.write_text("original robot source")
            original = base / "hands.usda"
            original.write_text("original frozen hand overlay")
            h = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            report = {"source_usd": str(source), "source_usd_sha256": h(source),
                      "usd": str(original), "usd_sha256": h(original), "recipe": RECIPE,
                      "original_authored_values_unchanged": True,
                      "hands": {side: {"stl_sha256": digest} for side, digest in HAND_SHA.items()}}
            receipt = json.dumps(report).encode()
            (base / "hand-asset.json").write_bytes(receipt)
            extended = root / "extended"
            extended.mkdir()
            overlay = extended / "feet.usda"
            overlay.write_text("explicit separate foot overlay")
            (extended / "hand-asset.json").write_bytes(receipt)
            (extended / "foot-asset.json").write_text("synthetic explicit foot receipt")
            foot = {"base_robot_asset": {"path": str(original), "sha256": h(original)}}
            with patch("hoi.v1.foot_collision.validate_overlay", return_value=foot):
                actual = load_hand_asset(overlay)
                self.assertEqual(actual["hands"], report["hands"])
                self.assertEqual(actual["usd_sha256"], h(overlay))
                self.assertFalse(actual["original_authored_values_unchanged"])
                self.assertEqual(actual["foot_asset"], foot)
                self.assertEqual((extended / "hand-asset.json").read_bytes(), receipt)
                source.write_text("tampered original source")
                with self.assertRaisesRegex(ValueError, "source USD changed"):
                    load_hand_asset(overlay)

    def test_new_defaults_and_all_independent_combinations(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(asdict(parse()[0]), {'hand_collision': True, 'self_collision': True})
            for hands in (False, True):
                for own in (False, True):
                    desired = PhysicsSettings(hands, own)
                    actual, explicit = parse(desired.flags())
                    self.assertEqual(actual, desired)
                    self.assertEqual(PhysicsSettings.environment(), desired)
                    self.assertEqual(explicit, {'hand_collision', 'self_collision'})

    def test_new_resume_inherits_saved_switches_and_rejects_changes(self):
        with patch.dict(os.environ, {}, clear=True):
            saved = {'hand_collision': False, 'self_collision': True}
            settings, explicit = parse(saved=saved, legacy=True)
            self.assertEqual(settings, PhysicsSettings(False, True))
            validate_resume(saved, settings, explicit)
            with self.assertRaises(ValueError):
                validate_resume(saved, PhysicsSettings(True, True), set(saved))

    def test_old_checkpoint_requires_both_explicit_switches(self):
        with patch.dict(os.environ, {}, clear=True):
            for flags in ([], ['--no-hand-collision'], ['--self-collision']):
                with self.assertRaises(ValueError):
                    parse(flags, {'iteration': 14000}, legacy=True)
            settings, explicit = parse(['--no-hand-collision', '--self-collision'], {}, legacy=True)
            validate_resume({}, settings, explicit)

    def test_invalid_saved_and_environment_values_rejected(self):
        with patch.dict(os.environ, {'SUPERMIMIC_SELF_COLLISION': 'maybe'}, clear=True):
            with self.assertRaises(ValueError):
                PhysicsSettings.environment()
        with self.assertRaises(ValueError):
            parse(saved={'hand_collision': 'False'})

    def test_disabling_hands_does_not_create_assets_or_modify_shared_usd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'data/g1_usd/g1_29dof/g1_29dof.usda'
            source.parent.mkdir(parents=True)
            source.write_text('immutable source')
            robot = SimpleNamespace(spawn=SimpleNamespace(usd_path=str(source),
                articulation_props=SimpleNamespace(enabled_self_collisions=True)))
            report = configure_robot(robot, root, root / 'unused', PhysicsSettings(False, False))
            self.assertFalse(report['self_collision'])
            self.assertFalse(robot.spawn.articulation_props.enabled_self_collisions)
            self.assertEqual(source.read_text(), 'immutable source')
            self.assertFalse((root / 'unused').exists())

    def test_existing_hand_overlay_is_reused_and_modified_asset_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.usda'
            overlay = root / 'g1_rubber_hand_collision.usda'
            source.write_text('original USD')
            overlay.write_text('independent overlay')
            h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            report = {'source_usd': str(source), 'source_usd_sha256': h(source),
                      'usd_sha256': h(overlay), 'recipe': RECIPE,
                      'original_authored_values_unchanged': True,
                      'hands': {side: {'stl_sha256': digest} for side, digest in HAND_SHA.items()}}
            (root / 'hand-asset.json').write_text(json.dumps(report))
            robot = SimpleNamespace(spawn=SimpleNamespace(usd_path=str(overlay),
                articulation_props=SimpleNamespace(enabled_self_collisions=False)))
            actual = configure_robot(robot, root, settings=PhysicsSettings(True, True))
            self.assertTrue(actual['self_collision'])
            self.assertEqual(actual['hand_asset'], report)
            with self.assertRaises(ValueError):
                configure_robot(robot, root, settings=PhysicsSettings(False, True))
            source.write_text('unexpected overwrite')
            with self.assertRaises(ValueError):
                load_hand_asset(overlay)


class CompiledPhysicsTests(unittest.TestCase):
    def test_real_mujoco_exclusion_and_object_filtering(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest('mujoco not installed in this interpreter')
        xml = '''<mujoco><worldbody>
        <body name="left"><freejoint/><geom type="sphere" size=".1"/></body>
        <body name="right"><freejoint/><geom type="sphere" size=".1"/></body>
        <body name="scene_Object"><freejoint/><geom type="sphere" size=".1"/></body>
        </worldbody>{contact}</mujoco>'''
        on = mujoco.MjModel.from_xml_string(xml.format(contact=''))
        self.assertEqual(compiled_self_collision_audit(on, True)['allowed_nonadjacent_robot_geom_pairs'], 1)
        with self.assertRaises(ValueError):
            compiled_self_collision_audit(on, False)
        off = mujoco.MjModel.from_xml_string(xml.format(contact='<contact><exclude body1="left" body2="right"/></contact>'))
        self.assertEqual(compiled_self_collision_audit(off, False)['allowed_nonadjacent_robot_geom_pairs'], 0)


if __name__ == '__main__':
    unittest.main()
