"""Read real MuJoCo contact forces through the pinned MJWarp contact recorder."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("mujoco_warp"), "native MJWarp runtime required")
class V11HandContactTests(unittest.TestCase):
    def test_force_matches_mujoco_and_excludes_wrist_ground_and_other_hands(self):
        import mujoco
        import mujoco_warp as mjw
        import warp as wp
        from hoi_residual.v11_contacts import HandObjectContacts

        wp.config.kernel_cache_dir = os.environ.get("G1_CONTACT_TEST_CACHE", str(Path(".cache/contactmix_kernels").resolve()))
        wp.init()
        xml = ['<mujoco><option gravity="0 0 0" timestep=".005" cone="elliptic"/><worldbody>']
        robot_geoms = []
        for x, side in enumerate(("left", "right")):
            for y, part in enumerate(("palm", "fingers", "thumb")):
                name = f"hw3_{side}_wrist_yaw_link_{y}_{side}_rubber_hand_{part}_hull"
                xml.append(f'<geom name="{name}" pos="{x} {y} .2" size=".1"/>')
                robot_geoms.append(name)
        xml += ['<geom name="wrist" pos="2 0 .2" size=".1"/>',
                '<geom name="floor" type="plane" size="10 10 .1"/>',
                '<body name="Object"><freejoint/><geom name="box" size=".1" mass="1"/></body>',
                '</worldbody></mujoco>']
        cpu = mujoco.MjModel.from_xml_string(''.join(xml))
        geoms = {name: cpu.geom(name).id for name in robot_geoms + ["wrist"]}
        manager = SimpleNamespace(get_model=lambda: SimpleNamespace(body_label=[]))
        with wp.ScopedDevice("cpu"):
            model = mjw.put_model(cpu)
            # Only the first of four substeps is allowed to generate force.
            # The object remains at the same location for the reference calculation.
            for x, y, z, channel in ((.15, 0., .2, 0), (1.15, 2., .2, 1),
                                      (2.15, 0., .2, None), (3., 0., .06, None)):
                data = mujoco.MjData(cpu)
                data.qpos[:3] = (x, y, z)
                mujoco.mj_forward(cpu, data)
                self.assertGreater(data.ncon, 0)
                force = np.zeros(6)
                mujoco.mj_contactForce(cpu, data, 0, force)
                self.assertGreater(force[0] / 4, 1.)
                native = mjw.put_data(cpu, data, nworld=2, nconmax=16, njmax=64)
                solver = SimpleNamespace(mj_model=cpu, mjw_model=model, mjw_data=native)

                def step_target(target, after_substep):
                    after_substep(.005)
                    native.nacon.zero_()
                    for _ in range(3):
                        after_substep(.005)

                physics = SimpleNamespace(solver=solver, manager=manager, geom_ids=geoms, step_target=step_target)
                recorder = HandObjectContacts(SimpleNamespace(physics=physics, N=2, asset={"collision_meshes": ["box"]}))
                contact, valid = recorder.step(None)
                expected = np.zeros((2, 2))
                if channel is not None:
                    expected[:, channel] = force[0] * .005
                np.testing.assert_allclose(recorder.impulse.numpy(), expected, rtol=1e-5, atol=1e-5)
                np.testing.assert_array_equal(contact.numpy(), expected > .02)
                self.assertTrue(valid.all())
                # A new interval without any contacts must discard the prior impulse.
                self.assertFalse(recorder.step(None)[0].any())
