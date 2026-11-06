"""Real MuJoCo Warp contact-material tests, CPU or explicitly requested GPU.

Run with the matching MuJoCo/MJWarp 3.15 and Warp 1.17 dependencies.
Set G1_CONTACT_TEST_DEVICE=cuda:0 for the same contact cases plus CUDA graph
capture/replay. A requested unavailable CUDA device is a test error.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np

HAVE_WARP = all(importlib.util.find_spec(name) for name in ("mujoco_warp", "warp"))
DEVICE = os.environ.get("G1_CONTACT_TEST_DEVICE", "cpu")


@unittest.skipUnless(HAVE_WARP, "Matching MuJoCo Warp dependencies are needed for real contact-mix tests")
class ContactMixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mujoco
        import mujoco_warp as mjw
        import warp as wp
        from supermimic.g1_contactmix import _backend_contract
        _backend_contract()  # Unknown versions/order are failures, not skips.
        wp.init()
        if DEVICE.startswith("cuda") and not wp.is_cuda_available():
            raise RuntimeError(f"Requested real contact test device is unavailable: {DEVICE}")
        # Windows' compiler path limit needs an explicitly shorter cache when
        # the checkout itself has a long/Unicode prefix. This changes only
        # generated kernel storage, never simulation parameters or test scope.
        wp.config.kernel_cache_dir = os.environ.get("G1_CONTACT_TEST_CACHE", str(Path(__file__).resolve().parents[1] / ".cache/contactmix_kernels"))
        cls.mj, cls.mjw, cls.wp = mujoco, mjw, wp

    def setUp(self):
        mj, mjw, wp = self.mj, self.mjw, self.wp
        self.scope = wp.ScopedDevice(DEVICE)
        self.scope.__enter__()
        # Separate real contact groups: hand/object .2, hand/object .8,
        # hand/ground, and unrelated object/ground plus object/object.
        xml = '''<mujoco><option gravity="0 0 0" timestep=".005" cone="elliptic" impratio="10"/>
        <worldbody><geom name="ground" type="plane" size="20 20 .1" friction=".3 0 0"/>
        <body name="hand_a" pos="0 0 .5"><geom name="hand_a" size=".1" priority="1" friction="1 0 0" solref="-18 -8" solimp=".9 .9 .001 .5 2"/></body>
        <body name="object_a" pos=".15 0 .5"><freejoint/><geom name="object_a" size=".1" mass="1" friction=".2 0 0"/></body>
        <body name="hand_b" pos="1 0 .5"><geom name="hand_b" size=".1" priority="1" friction="1 0 0" solref="-18 -8" solimp=".9 .9 .001 .5 2"/></body>
        <body name="object_b" pos="1.15 0 .5"><freejoint/><geom name="object_b" size=".1" mass="1" friction=".8 0 0"/></body>
        <body name="ordinary_a" pos="3 0 .05"><freejoint/><geom name="ordinary_a" size=".1" mass="1" friction=".2 0 0"/></body>
        <body name="ordinary_b" pos="3.15 0 .05"><freejoint/><geom name="ordinary_b" size=".1" mass="1" friction=".8 0 0"/></body>
        <body name="hand_ground" pos="5 0 .05"><freejoint/><geom name="hand_ground" size=".1" mass="1" priority="1" friction="1 0 0" solref="-18 -8" solimp=".9 .9 .001 .5 2"/></body>
        </worldbody></mujoco>'''
        self.cpu = mj.MjModel.from_xml_string(xml)
        d = mj.MjData(self.cpu)
        self.model = mjw.put_model(self.cpu, batch_sizes={"geom_friction": 2})
        self.data = mjw.put_data(self.cpu, d, nworld=2, nconmax=64, njmax=200)
        self.ids = {self.cpu.geom(i).name: i for i in range(self.cpu.ngeom)}
        self.hands = [self.ids[name] for name in ("hand_a", "hand_b", "hand_ground")]
        self.grounds = [self.ids["ground"]]
        self.prior_calls = []
        def previous(model, data):
            self.prior_calls.append((model, data))
        self.model.callback.contactfilter = previous
        # Independent per-world material values expose incorrect broadcasting.
        self.mu = self.model.geom_friction.numpy()
        self.mu[1, self.ids["object_a"], 0] = .6
        self.mu[1, self.ids["object_b"], 0] = .4
        self.model.geom_friction.assign(self.mu)
        mjw.forward(self.model, self.data)
        self.baseline = self.snapshot()
        from supermimic.g1_contactmix import install_hand_contact_multiply
        self.owner = install_hand_contact_multiply(self.model, self.hands, ground_geom_ids=self.grounds)

    def tearDown(self):
        if getattr(self, "owner", None) is not None and self.model.callback.contactfilter is self.owner:
            self.owner.close()
        self.scope.__exit__(None, None, None)

    def snapshot(self):
        n = int(self.data.nacon.numpy()[0])
        self.assertGreater(n, 0, "Test must generate real contacts")
        self.assertLessEqual(n, self.data.naconmax)
        result = {}
        geoms = self.data.contact.geom.numpy()[:n]
        world = self.data.contact.worldid.numpy()[:n]
        friction = self.data.contact.friction.numpy()[:n]
        solref = self.data.contact.solref.numpy()[:n]
        solimp = self.data.contact.solimp.numpy()[:n]
        dims = self.data.contact.dim.numpy()[:n]
        for i, ((a, b), w) in enumerate(zip(geoms, world)):
            key = (int(w), *sorted((int(a), int(b))))
            self.assertNotIn(key, result, "Sphere fixture should have one contact per geom pair/world")
            result[key] = {"friction": friction[i].copy(), "solref": solref[i].copy(), "solimp": solimp[i].copy(), "dim": int(dims[i])}
        return result

    def assert_mixed(self):
        after = self.snapshot()
        self.assertEqual(set(after), set(self.baseline))
        mixed_count, preserved_count = 0, 0
        for (w, a, b), row in after.items():
            before = self.baseline[(w, a, b)]
            np.testing.assert_allclose(row["solref"], before["solref"], atol=0, rtol=0)
            np.testing.assert_allclose(row["solimp"], before["solimp"], atol=0, rtol=0)
            if (a in self.hands or b in self.hands) and a not in self.grounds and b not in self.grounds:
                expected_mu = float(self.mu[w, a, 0]*self.mu[w, b, 0])
                np.testing.assert_allclose(row["friction"][:2], expected_mu, atol=1e-7, rtol=1e-6)
                np.testing.assert_array_equal(row["friction"][2:], 0)
                mixed_count += 1
            else:
                np.testing.assert_allclose(row["friction"], before["friction"], atol=0, rtol=0)
                preserved_count += 1
        self.assertGreaterEqual(mixed_count, 4)
        self.assertGreaterEqual(preserved_count, 4)
        self.assertTrue(np.isfinite(self.data.qfrc_constraint.numpy()).all())
        self.assertGreater(np.abs(self.data.qfrc_constraint.numpy()).max(), 0)
        return after

    def test_real_contacts_use_products_and_preserve_soft_normal_and_external_pairs(self):
        self.mjw.forward(self.model, self.data)
        self.assert_mixed()
        self.assertGreaterEqual(len(self.prior_calls), 2)
        self.owner.ensure_installed()

    def test_material_randomization_is_read_at_each_collision(self):
        self.mjw.forward(self.model, self.data)
        self.assert_mixed()
        self.mu[0, self.ids["object_a"], 0] = .17
        self.mu[1, self.ids["object_b"], 0] = .73
        self.model.geom_friction.assign(self.mu)
        self.mjw.forward(self.model, self.data)
        self.assert_mixed()

    def test_unrelated_model_and_close_keep_callbacks_local(self):
        other = self.mjw.put_model(self.cpu)
        self.assertIsNone(other.callback.contactfilter)
        previous = self.owner.previous
        self.owner.close()
        self.assertIs(self.model.callback.contactfilter, previous)
        self.owner = None

    def test_duplicate_or_replaced_hook_fails_loudly(self):
        from supermimic.g1_contactmix import install_hand_contact_multiply
        with self.assertRaisesRegex(RuntimeError, "already installed"):
            install_hand_contact_multiply(self.model, self.hands)
        self.model.callback.contactfilter = None
        with self.assertRaisesRegex(RuntimeError, "replaced"):
            self.owner.ensure_installed()

    def test_zero_friction_is_frictionless_and_finite(self):
        self.mu[:, self.ids["object_a"], 0] = 0
        self.model.geom_friction.assign(self.mu)
        self.mjw.forward(self.model, self.data)
        after = self.assert_mixed()
        for key, row in after.items():
            if key[1:] == tuple(sorted((self.ids["hand_a"], self.ids["object_a"]))):
                self.assertEqual(row["dim"], 1)

    @unittest.skipUnless(DEVICE.startswith("cuda"), "CUDA graph capture is tested only on explicitly requested CUDA")
    def test_cuda_graph_replay_recomputes_contact_friction(self):
        self.mjw.forward(self.model, self.data)  # compile/warm before capture
        with self.wp.ScopedCapture(device=DEVICE) as capture:
            self.mjw.forward(self.model, self.data)
        self.mu[0, self.ids["object_a"], 0] = .31
        self.mu[1, self.ids["object_b"], 0] = .67
        self.model.geom_friction.assign(self.mu)
        self.wp.capture_launch(capture.graph)
        self.assert_mixed()


if __name__ == "__main__":
    unittest.main()
