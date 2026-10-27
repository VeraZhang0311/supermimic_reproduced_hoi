"""Native-friction storage regressions using explicit CPU selection mocks.

These test model writes and fresh readback for different getter layouts. They
provide no native GPU physics evidence or validation of a simulator ABI.
"""
from enum import IntFlag
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hoi.v1.native import physical_context
from hoi.v1.randomization import NativeRandomization


class Flags(IntFlag):
    SHAPE_PROPERTIES = 1
    BODY_PROPERTIES = 2
    BODY_INERTIAL_PROPERTIES = 4


class MockBody:
    def __init__(self, mass, material_layout="contiguous"):
        n, bodies = mass.shape
        self.data = SimpleNamespace(
            body_mass=SimpleNamespace(torch=mass.clone()),
            body_inertia=SimpleNamespace(torch=torch.eye(3).reshape(1, 1, 9).expand(n, bodies, 9).clone()),
            body_com_pose_b=SimpleNamespace(torch=torch.zeros(n, bodies, 7)))
        # Object selection can lie between robot/ground shape storage. The
        # setter must target this selection, never flatten the whole model.
        self.material_storage = torch.full((n, 5, 2), .27)
        self.material = (torch.full((n, 2, 2), .6) if material_layout == "contiguous"
                         else self.material_storage[:, 1:3])
        self.material.fill_(.6)
        self.layout = material_layout
        self.setter_calls = []
        self._root_view = SimpleNamespace(get_attribute=self.get_attribute, set_attribute=self.set_attribute)

    def get_attribute(self, name, model):
        assert name == "shape_material_mu"
        return self.material.clone() if self.layout == "gathered" else self.material

    def set_attribute(self, name, model, values):
        assert name == "shape_material_mu"
        self.setter_calls.append(values.clone())
        self.material.copy_(values)

    def set_masses_index(self, *, masses):
        self.data.body_mass.torch.copy_(masses)

    def set_inertias_index(self, *, inertias):
        self.data.body_inertia.torch.copy_(inertias)

    def set_coms_index(self, *, coms):
        self.data.body_com_pose_b.torch[..., :3].copy_(coms)


def make_host(layout):
    robot = MockBody(torch.tensor([[2., 8., 1.]]).expand(3, -1))
    obj = MockBody(torch.full((3, 1), .5), layout)
    model = object()
    manager = SimpleNamespace(get_model=lambda: model, add_model_change=lambda flags: None)
    host = SimpleNamespace(N=3, env=SimpleNamespace(robot=robot, torso_id=1,
                               sim=SimpleNamespace(backend_manager=manager)), object=obj)
    host.xi_o = physical_context(torch.full((3,), .5), torch.zeros(3, 3), torch.eye(3).expand(3, 3, 3),
                                 torch.full((3,), .6), torch.full((3,), .3))
    return host


def reset_with_friction(dr, ids, friction):
    values = (dr.nominal_object_mass[ids] * 1.1, dr.nominal_object_com[ids] + .003,
              dr.nominal_object_inertia[ids] * 1.1, torch.tensor(friction))
    with patch("hoi.v1.randomization.sample_object_properties", return_value=values):
        dr.reset(ids)


class FrictionSelectionWriteTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(126)
        self.runtime = patch.dict(sys.modules, {
            "newton": SimpleNamespace(ModelFlags=Flags),
            "warp": SimpleNamespace(to_torch=lambda value: value, from_torch=lambda value: value)})
        self.runtime.start()
        self.addCleanup(self.runtime.stop)

    def test_all_object_shapes_write_to_model_for_contiguous_strided_and_gathered_getters(self):
        for layout in ("contiguous", "strided", "gathered"):
            with self.subTest(layout=layout):
                host = make_host(layout)
                obj, dr = host.object, NativeRandomization(host)
                if layout == "strided":
                    self.assertFalse(obj.material.is_contiguous())
                before_material, before_context = obj.material.clone(), host.xi_o.clone()
                ids = torch.tensor([0, 2])
                reset_with_friction(dr, ids, [.42, .78])
                expected = torch.tensor([.42, .6, .78])
                torch.testing.assert_close(obj.material, expected[:, None, None].expand_as(obj.material), atol=0, rtol=0)
                self.assertEqual(len(obj.setter_calls), 1)
                self.assertTrue(obj.setter_calls[0].is_contiguous())
                torch.testing.assert_close(host.xi_o[:, 10], expected, atol=0, rtol=0)
                torch.testing.assert_close(obj.material[1], before_material[1], atol=0, rtol=0)
                torch.testing.assert_close(host.xi_o[1], before_context[1], atol=0, rtol=0)
                torch.testing.assert_close(obj.material_storage[:, [0, 3, 4]], torch.full((3, 3, 2), .27), atol=0, rtol=0)
                torch.testing.assert_close(host.env.robot.data.body_mass.torch[:, [0, 2]],
                                           torch.tensor([[2., 1.]]).expand(3, -1), atol=0, rtol=0)
                # A later reset still preserves every unselected environment.
                before = obj.material.clone()
                reset_with_friction(dr, torch.tensor([1]), [.51])
                torch.testing.assert_close(obj.material[[0, 2]], before[[0, 2]], atol=0, rtol=0)

    def test_lost_native_setter_write_fails_fresh_model_readback(self):
        for layout in ("contiguous", "strided", "gathered"):
            with self.subTest(layout=layout):
                host = make_host(layout)
                dr = NativeRandomization(host)
                with patch.object(host.object._root_view, "set_attribute", return_value=None):
                    with self.assertRaisesRegex(RuntimeError, "object friction.*readback"):
                        reset_with_friction(dr, torch.tensor([0, 2]), [.42, .78])
                torch.testing.assert_close(host.object.material, torch.full_like(host.object.material, .6), atol=0, rtol=0)

    def test_invalid_environment_axis_or_empty_selection_is_rejected_before_setter(self):
        for shape in ((3,), (4, 2), (3, 0)):
            with self.subTest(shape=shape):
                host = make_host("contiguous")
                dr = NativeRandomization(host)
                with patch.object(host.object._root_view, "get_attribute", return_value=torch.empty(shape)), patch.object(
                        host.object._root_view, "set_attribute") as setter:
                    with self.assertRaisesRegex(RuntimeError, "environment axis"):
                        reset_with_friction(dr, torch.tensor([0, 2]), [.42, .78])
                setter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
