"""CPU geometry only: no simulator dynamics or GPU acceptance is inferred."""
import unittest
import numpy as np

from hoi.v1.contacts import BODY_ROLES
from hoi.v1.probe_pose import CompiledProbePlanner, rotation_between


def radial_robot():
    import mujoco
    count = len(BODY_ROLES)
    k = np.arange(count)
    z = 1 - 2*(k+.5)/count
    phi = k*np.pi*(3-np.sqrt(5))
    p = np.column_stack((np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z))
    geoms = "".join(f'<geom name="region_{i}" type="sphere" size=".025" pos="{x} {y} {z}"/>' for i,(x,y,z) in enumerate(p))
    xml = f'<mujoco><worldbody><body name="pelvis"><freejoint/><inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/><body name="links"><joint name="test_joint" range="-.5 .5"/>{geoms}</body></body></worldbody></mujoco>'
    return mujoco.MjModel.from_xml_string(xml)


class ProbePoseTests(unittest.TestCase):
    def test_rotation_handles_all_directions_including_opposites(self):
        for a,b in (([0,0,1],[0,0,-1]),([1,2,3],[-2,3,1]),([1,0,0],[1,0,0])):
            R=rotation_between(a,b)
            np.testing.assert_allclose(R @ (np.array(a)/np.linalg.norm(a)),np.array(b)/np.linalg.norm(b),atol=1e-12)
            np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(R),1.)

    def test_every_region_exposed_without_burying_other_geometry(self):
        model=radial_robot()
        planner=CompiledProbePlanner(model,list(BODY_ROLES),joint_names=["test_joint"])
        planner.search([0.],max_candidates=4)
        for role in BODY_ROLES:
            for negative in (False,True):
                root,R,q,surface,normal,audit=planner.placement(role,ground=negative)
                self.assertLessEqual(audit["maximum_non_target_ground_penetration_m"],.002)
                self.assertEqual(audit["search"]["physics_steps"],0)
                self.assertFalse(audit["search"]["collisions_disabled"])
                if negative:
                    self.assertAlmostEqual(audit["target_minimum_world_z_m"],-.002)
                    self.assertAlmostEqual(surface[2],-.002)
                else:
                    self.assertGreaterEqual(audit["robot_minimum_world_z_m"],.002-1e-12)
                    self.assertGreater(surface[2],1.)
                np.testing.assert_allclose(normal,[0,0,-1 if negative else 1])

    def test_hidden_region_cannot_be_faked_by_lowering_whole_robot(self):
        model=radial_robot()
        model.geom_pos[0]=0.
        planner=CompiledProbePlanner(model,list(BODY_ROLES),joint_names=["test_joint"])
        with self.assertRaisesRegex(RuntimeError,"No shallow exposed"):
            planner.search([0.],max_candidates=4)

    def test_narrow_legal_subchain_seed_preserves_physics_and_penetration_limit(self):
        import mujoco
        geoms='<geom name="ancestor" type="sphere" size=".1"/>'
        for i in range(1,len(BODY_ROLES)):
            geoms+=f'<geom name="region_{i}" type="sphere" size=".02" pos="{i} 2 0"/>'
        xml=f"""<mujoco><compiler angle="radian"/><worldbody><body name="pelvis"><freejoint/>
            {geoms}<body name="middle"><inertial pos="0 0 0" mass=".1" diaginertia=".01 .01 .01"/>
            <joint name="slide_x" type="slide" axis="1 0 0" range="0 .04"/>
            <body name="distal" pos=".11 0 0"><joint name="slide_y" type="slide" axis="0 1 0" range="-.005 .005"/>
            <geom name="distal_collision" type="sphere" size=".05"/></body></body></body></worldbody></mujoco>"""
        model=mujoco.MjModel.from_xml_string(xml)
        roles=[*BODY_ROLES,BODY_ROLES[12]]
        planner=CompiledProbePlanner(model,roles,joint_names=["slide_x","slide_y"])
        preserved={key:np.array(getattr(model,key),copy=True) for key in
                   ("jnt_range","geom_contype","geom_conaffinity","geom_pos","geom_size","exclude_signature")}
        flags=int(model.opt.disableflags)
        planner._set([0.,0.]);mujoco.mj_collision(model,planner.data)
        self.assertGreater(-planner._self_contacts(planner.data)[0]["signed_distance_m"],.03)
        seed,fixed,evidence=planner._local_clearance_seed(np.array([0.,0.]),.002)
        self.assertEqual(fixed,[0,1])
        self.assertLessEqual(evidence["seed_maximum_self_penetration_m"],.002)
        self.assertEqual(evidence["maximum_allowed_self_penetration_m"],.002)
        self.assertFalse(evidence["model_or_filter_changes"])
        self.assertTrue(np.all(seed>=model.jnt_range[planner.joints,0]))
        self.assertTrue(np.all(seed<=model.jnt_range[planner.joints,1]))
        for key,value in preserved.items():
            np.testing.assert_array_equal(getattr(model,key),value,err_msg=key)
        self.assertEqual(int(model.opt.disableflags),flags)

    def test_actual_compiled_box_capsule_ellipsoid_and_cylinder_support(self):
        import mujoco
        model=radial_robot()
        planner=CompiledProbePlanner(model,list(BODY_ROLES),joint_names=["test_joint"])
        planner._set([0.],position=[0,0,0])
        for kind,size,expected in (
            (mujoco.mjtGeom.mjGEOM_BOX,[.1,.2,.3],.1),
            (mujoco.mjtGeom.mjGEOM_CAPSULE,[.1,.2,0],.1),
            (mujoco.mjtGeom.mjGEOM_ELLIPSOID,[.1,.2,.3],.1),
            (mujoco.mjtGeom.mjGEOM_CYLINDER,[.1,.2,0],.1)):
            model.geom_type[0]=kind; model.geom_size[0]=size
            x=planner.support([[1,0,0]])[0,0]-planner.data.geom_xpos[0,0]
            self.assertAlmostEqual(x,expected)


if __name__ == "__main__":
    unittest.main()




class ObjectSupportFacetTests(unittest.TestCase):
    def test_separated_hulls_choose_real_single_component_surface_not_air(self):
        import numpy as np
        import trimesh
        from hoi.v1.probe_pose import object_support_facet
        meshes = []
        for x in (-2., 2.):
            box = trimesh.creation.box(extents=[1., 1., .2])
            meshes.append({"vertices": np.asarray(box.vertices) + [x, 0., 0.],
                           "faces": np.asarray(box.faces).copy()})
        before = [mesh["vertices"].copy() for mesh in meshes]
        rotation, support, report = object_support_facet(meshes)
        point = np.asarray(report["point_local"])
        # Both boxes share the lowest plane, but their union midpoint is air.
        self.assertGreater(abs(point[0]), 1.49)
        vertices = meshes[report["mesh_index"]]["vertices"]
        ids = report["coplanar_component_vertex_ids"]
        np.testing.assert_allclose(point, vertices[ids].mean(0), atol=1e-12)
        normal = np.asarray(report["normal_local"])
        union = np.concatenate([mesh["vertices"] for mesh in meshes])
        self.assertLessEqual(float(((union - point) @ normal).max()), 2e-6)
        np.testing.assert_allclose(rotation @ normal, [0,0,-1], atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.)
        self.assertAlmostEqual(support[2], (union @ rotation.T)[:,2].min())
        for actual, original in zip(meshes, before):
            np.testing.assert_array_equal(actual["vertices"], original)

    def test_degenerate_surface_cannot_fabricate_support(self):
        from hoi.v1.probe_pose import object_support_facet
        with self.assertRaisesRegex(RuntimeError, "no exposed"):
            object_support_facet([{"vertices": [[0,0,0],[1,0,0],[2,0,0]], "faces": [[0,1,2]]}])
