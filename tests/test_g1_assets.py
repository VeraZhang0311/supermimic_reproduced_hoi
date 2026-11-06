"""Real prepared-asset identity and inertial preservation checks."""
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from supermimic.g1_assets import load_model, prepare, verify_assets
from supermimic.g1_physics import ASSET_DIR, CONTACT_CONE, CONTACT_IMPRATIO
from supermimic.robot_metadata import G1_BODY_NAMES, G1_JOINT_NAMES


class PreparationTests(unittest.TestCase):
    def test_wrong_archive_rejected_without_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "wrong.tar.gz"
            archive.write_bytes(b"not Feng Qiao's handoff")
            output = Path(tmp) / "out"
            with self.assertRaisesRegex(ValueError, "MD5"):
                prepare(archive, output)
            self.assertFalse(output.exists())


@unittest.skipUnless((ASSET_DIR / "manifest.json").is_file(), "Prepare verified hw3 assets")
class PreparedAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = verify_assets()
        cls.model = load_model(ASSET_DIR / "g1_hw3.xml")
        cls.source = ET.parse(ASSET_DIR / "source/urdf/g1/main_rubber_hand_meshcol_handcol_softfinger.urdf").getroot()

    def test_exact_source_topology_and_mass_preservation(self):
        self.assertEqual([x.get("name") for x in self.source.findall("link")], G1_BODY_NAMES)
        self.assertEqual(len(G1_BODY_NAMES), 38)
        moving = [x.get("name") for x in self.source.findall("joint") if x.get("type") == "revolute"]
        self.assertEqual(set(moving), set(G1_JOINT_NAMES))
        self.assertEqual((self.model.nq, self.model.nv), (36, 35))
        mass = 0.
        for link in self.source.findall("link"):
            value = link.find("inertial/mass")
            expected = float(value.get("value")) if value is not None else 0.
            actual = self.model.body(link.get("name")).mass[0]
            self.assertAlmostEqual(actual, expected, places=13)
            mass += expected
        self.assertAlmostEqual(self.model.body_mass.sum(), mass, places=12)
        for name in ("imu_in_torso", "imu_in_pelvis", "mid360_link"):
            self.assertEqual(self.model.body(name).mass[0], 0.)

    def test_all_47_named_colliders_and_soft_hard_partition(self):
        roles = self.receipt["collision_roles"]
        self.assertEqual(len(roles), 47)
        self.assertEqual(sum(role == "finger" for role in roles.values()), 4)
        for name, role in roles.items():
            geom = self.model.geom(name)
            self.assertEqual(geom.contype[0], 1)
            self.assertEqual(geom.conaffinity[0], 1)
            self.assertEqual(geom.condim[0], 3)
            if role == "finger":
                self.assertTrue((geom.solref < 0).all())
                self.assertEqual(geom.priority[0], 1)
            elif role == "palm":
                self.assertTrue((geom.solref > 0).all())
        np.testing.assert_allclose(self.model.dof_damping[6:], 0)
        self.assertEqual(CONTACT_CONE, "elliptic")
        self.assertAlmostEqual(self.model.opt.impratio, CONTACT_IMPRATIO)

    def test_manifest_detects_edited_asset_before_execution(self):
        # A small manifest copy is enough to exercise integrity failure without
        # modifying the real, immutable source assets.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "manifest.json").write_text((ASSET_DIR / "manifest.json").read_text())
            first = next(iter(self.receipt["files"]))
            path = directory / first
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"edited")
            with self.assertRaises((ValueError, FileNotFoundError)):
                verify_assets(directory)


if __name__ == "__main__":
    unittest.main()
