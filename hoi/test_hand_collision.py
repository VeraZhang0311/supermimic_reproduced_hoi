"""Check original hand definitions and strict OFF/ON inertial validation on CPU."""
from pathlib import Path
import copy
import unittest
import tempfile
from hoi.hand_collision import original_definition, assert_same_physics, ORIGIN, RECIPE


class HandCollisionTests(unittest.TestCase):
    def test_frozen_original_hoi_hand_definition(self):
        xml='<robot name="G1">'
        for side,y in [('left',.003),('right',-.003)]:
            xml+=f'''<link name="{side}_rubber_hand"><collision><origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry><mesh filename="meshes/{side}_rubber_hand.STL"/></geometry></collision></link>
            <joint name="{side}_mount" type="fixed"><parent link="{side}_wrist_yaw_link"/>
            <child link="{side}_rubber_hand"/><origin xyz=".0415 {y} 0" rpy="0 0 0"/></joint>'''
        xml+='</robot>'
        with tempfile.TemporaryDirectory() as tmp:
            urdf=Path(tmp)/'g1.urdf';urdf.write_text(xml)
            definition=original_definition(urdf)
        for side in ('left','right'):
            self.assertEqual(definition[side]['fixed_joint_origin'],ORIGIN[side])
            self.assertEqual(definition[side]['collision_origin'],[0.,0.,0.])

    def test_only_original_five_hull_sixteen_vertex_caps(self):
        self.assertEqual((RECIPE['max_convex_hull'],RECIPE['max_ch_vertex']),(5,16))
        self.assertEqual(RECIPE['seed'],0)

    def test_mass_joint_mutation_rejected(self):
        before={'enabled':False,'robot_bodies':{'wrist':{'mass':.17,'inertia':[.01,.02,.03]}},
                'joints':{'wrist_yaw':{'range':[-1.61,1.61]}}}
        after=copy.deepcopy(before);after['enabled']=True
        assert_same_physics(before,after)
        after['robot_bodies']['wrist']['mass']=.2
        with self.assertRaises(AssertionError):assert_same_physics(before,after)
        after=copy.deepcopy(before);after['enabled']=True;after['joints']['wrist_yaw']['range'][0]=-2
        with self.assertRaises(AssertionError):assert_same_physics(before,after)


if __name__=='__main__':unittest.main(verbosity=2)
