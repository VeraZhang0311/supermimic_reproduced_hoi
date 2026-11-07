"""Local native EPA edge limit, real scratch allocation, and legacy checkpoint contract."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

from hoi_residual.capacity import capacity_profile, capacity_readback, checkpoint_capacity, configure_capacity, overflow_detail


def sim_cfg():
    return SimpleNamespace(physics=SimpleNamespace(solver_cfg=SimpleNamespace()))


class Profiles(unittest.TestCase):
    def test_legacy_capacity_and_constraint_bounds(self):
        self.assertEqual(capacity_profile(), {"nconmax": 512, "njmax": 2048})
        self.assertEqual(checkpoint_capacity(capacity_profile(epa_horizon=64))["epa_horizon"], 64)
        for bad in (None, {}, {"epa_horizon": 64}, {"nconmax": 512, "njmax": 2048, "bad": 1},
                    {"nconmax": 511, "njmax": 2048}, {"nconmax": 512, "njmax": 2048, "epa_horizon": True}):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                checkpoint_capacity(bad)


@unittest.skipUnless(importlib.util.find_spec("mujoco_warp"), "native MJWarp runtime required")
class NativeCapacity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import warp as wp
        wp.config.kernel_cache_dir = os.environ.get("G1_CONTACT_TEST_CACHE", str(Path(".cache/contactmix_kernels").resolve()))
        wp.init()

    def test_edge_25_rejected_by_24_and_retained_by_64(self):
        import warp as wp
        from mujoco_warp._src.collision_gjk import Polytope, _add_edge

        @wp.kernel
        def boundary(horizon: wp.array(dtype=wp.int32), result: wp.array(dtype=wp.int32)):
            p = Polytope()
            p.horizon = horizon
            p.nhorizon = 0
            for edge in range(25):
                p.nhorizon = _add_edge(p, edge, edge + 1)
            result[0] = p.nhorizon

        with wp.ScopedDevice("cpu"):
            for size, expected in ((24, -1), (64, 25)):
                result = wp.zeros(1, dtype=wp.int32)
                wp.launch(boundary, 1, inputs=[wp.zeros(size, dtype=wp.int32), result])
                self.assertEqual(result.numpy()[0], expected)

    def test_real_rigid_convex_allocator_and_overflow_report(self):
        import mujoco
        import mujoco_warp as mjw
        from mujoco_warp._src import collision_convex, collision_driver
        from mujoco_warp._src.types import OverflowType
        import warp as wp

        original_dispatch = collision_driver.convex_narrowphase
        cfg = sim_cfg()
        try:
            with wp.ScopedDevice("cpu"):
                configure_capacity(cfg, capacity_profile(epa_horizon=64))
                cpu = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
                    <geom type="ellipsoid" size=".1 .1 .2"/>
                    <body pos=".12 0 0"><freejoint/><geom type="ellipsoid" size=".1 .2 .1" mass="1"/></body>
                    </worldbody></mujoco>""")
                state = mujoco.MjData(cpu)
                mujoco.mj_forward(cpu, state)
                model = mjw.put_model(cpu)
                data = mjw.put_data(cpu, state, nworld=2, nconmax=16, njmax=64)
                physics = SimpleNamespace(solver=SimpleNamespace(mjw_data=data))
                mjw.collision(model, data)
                observed = capacity_readback(physics)
                self.assertEqual(observed["epa_horizon_shape"], [32, 64])
                self.assertEqual(observed["additional_epa_bytes"], 32 * 40 * 4)
                self.assertIs(collision_driver.convex_narrowphase, original_dispatch)
                self.assertGreater(data.nacon.numpy()[0], 0)
                self.assertEqual(overflow_detail(physics)["overflow_world_count"], 0)
                data.overflow.assign([int(OverflowType.EPA_HORIZON | OverflowType.NEFC), 0])
                detail = overflow_detail(physics)
                self.assertEqual(detail["overflow_bit_counts"], {"NEFC": 1, "EPA_HORIZON": 1})
                self.assertEqual(detail["overflow_worlds"][0]["world"], 0)
                self.assertEqual(detail["epa_required_edges_at_least"], 65)
        finally:
            configure_capacity(cfg, capacity_profile())
        self.assertEqual(collision_convex.MJ_MAX_EPAHORIZON, 24)
