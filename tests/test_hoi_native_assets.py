import copy
import tempfile
import unittest
from pathlib import Path
from hoi.data.native_assets import audit_geometry, canonical, digest, exact_surface, physics_group_key, source_usd_physics, uniform_density_profile, source_physical_inputs, physical_policy


class NativeAssetAuditTests(unittest.TestCase):
    def test_compact_recipe_preserves_2448_aliases_without_copying_meshes(self):
        from hoi.data.curated_bundle import expand_native_recipe_assets,NATIVE_ALIAS_SCHEMA
        shared={'object_geometry_sha256':digest('label0'),'physical_profile':{'mass':1.},
                'collision_meshes':[{'vertices':[[0.,0.,0.]],'faces':[]}]}
        aliases=[]
        for i in range(2448):
            overrides={'object_geometry_sha256':digest('label'+str(i)),'object_asset':'label'+str(i)}
            aliases.append({'shared_asset_index':0,'overrides':overrides,
                            'expanded_recipe_entry_sha256':digest({**shared,**overrides})})
        recipe={'native_alias_schema':NATIVE_ALIAS_SCHEMA,'assets':[shared],'object_geometry_aliases':aliases}
        expanded=expand_native_recipe_assets(recipe)
        self.assertEqual(len(expanded),2448)
        self.assertEqual(len({x['object_geometry_sha256'] for x in expanded}),2448)
        self.assertTrue(all(x['collision_meshes'] is shared['collision_meshes'] for x in expanded))
        bad=copy.deepcopy(recipe);bad['object_geometry_aliases'][1]['overrides']['physical_profile']={'mass':2.}
        with self.assertRaisesRegex(ValueError,'may not override'):
            expand_native_recipe_assets(bad)
        bad=copy.deepcopy(recipe);bad['assets'][0]['physical_profile']['mass']=2.
        with self.assertRaisesRegex(ValueError,'expanded recipe hash'):
            expand_native_recipe_assets(bad)

    def test_exact_surface_aliases_ignore_provenance_but_not_metric_shape(self):
        a=[{'kind':'cube','size':[.2,.3,.4],'name':'A','approximation':'none'}]
        b=[{'kind':'cube','size':[.2,.3,.4],'name':'B','approximation':'convexHull'}]
        self.assertEqual(digest(exact_surface(a)),digest(exact_surface(b)))
        b[0]['size'][0]+=.000001
        self.assertNotEqual(digest(exact_surface(a)),digest(exact_surface(b)))

    def test_uniform_density_moments_and_open_surface_rejection(self):
        g={'shapes':[{'kind':'cube','size':[2.,4.,6.]}]}
        profile,evidence=uniform_density_profile(g,mass=3.,friction=.9,ground_friction=.3)
        self.assertEqual(profile['com'],[0.,0.,0.])
        self.assertEqual([profile['inertia'][i][i] for i in range(3)],[13.,10.,5.])
        self.assertEqual(evidence['summed_component_volume_m3'],48.)
        g={'shapes':[{'kind':'mesh','vertices':[[0,0,0],[1,0,0],[0,1,0]],'faces':[[0,1,2]]}]}
        with self.assertRaisesRegex(ValueError,'closed consistently'):
            uniform_density_profile(g,mass=1.,friction=.6,ground_friction=.3)

    @staticmethod
    def tetrahedron(vertices):
        return {'kind':'mesh','vertices':vertices,'faces':[[0,2,1],[0,1,3],[0,3,2],[1,2,3]]}

    @staticmethod
    def tetrahedron_moments(vertices,mass):
        import numpy as np
        vertices=np.asarray(vertices,dtype=np.float64)
        center=vertices.mean(0);relative=vertices-center
        covariance=relative.T@relative/20.
        return center,mass*(np.trace(covariance)*np.eye(3)-covariance)

    def test_small_valid_tetrahedron_uses_local_dimensionless_moments(self):
        import numpy as np
        mass=2.;offset=np.array([.4,-.2,.8])
        vertices=np.array([[0,0,0],[1.1,0,0],[0,1.8,0],[0,0,2.2]])*1e-5+offset
        g={'shapes':[self.tetrahedron(vertices.tolist())]}
        profile,evidence=uniform_density_profile(g,mass=mass,friction=.9,ground_friction=.3)
        expected_center,expected_inertia=self.tetrahedron_moments(vertices,mass)
        expected_volume=abs(np.linalg.det((vertices[1:]-vertices[0]).T))/6.
        self.assertLess(expected_volume,1e-15)
        self.assertAlmostEqual(evidence['summed_component_volume_m3']/expected_volume,1.,places=12)
        np.testing.assert_allclose(profile['com'],expected_center,atol=2*np.finfo(np.float64).eps,rtol=0)
        np.testing.assert_allclose(profile['inertia'],expected_inertia,atol=1e-24,rtol=2e-11)
        self.assertTrue((np.linalg.eigvalsh(profile['inertia'])>0).all())
        self.assertEqual(evidence['numerical_integration']['normalized_volume_lower_bound'],1e-15)
        self.assertEqual(evidence['overlap_policy'],'component_volume_sum_not_boolean_union')

    def test_mass_properties_preserve_scale_translation_and_component_weighting(self):
        import numpy as np
        shapes=[np.array([[0,0,0],[1.2,0,0],[0,.8,0],[0,0,1.6]]),
                np.array([[0,0,0],[.6,0,0],[0,.4,0],[0,0,.8]])+np.array([2.,-.3,.5])]
        original,_=uniform_density_profile({'shapes':[self.tetrahedron(v.tolist()) for v in shapes]},mass=3.,friction=.6,ground_friction=.3)
        for scale,offset in ((1e-5,np.array([.37,-.53,1.2])),(1.,np.array([1e6,-2e6,3e6])),(1e4,np.array([-7.,11.,-3.]))):
            with self.subTest(scale=scale):
                profile,evidence=uniform_density_profile({'shapes':[self.tetrahedron((v*scale+offset).tolist()) for v in shapes]},mass=3.,friction=.6,ground_friction=.3)
                np.testing.assert_allclose(profile['com'],np.asarray(original['com'])*scale+offset,atol=5e-10,rtol=0)
                np.testing.assert_allclose(profile['inertia'],np.asarray(original['inertia'])*scale**2,atol=1e-24,rtol=2e-9)
                self.assertEqual(evidence['component_count'],2)
                self.assertEqual(profile['mass'],3.)
        tiny_cube,evidence=uniform_density_profile({'shapes':[{'kind':'cube','size':[1e-6,2e-6,3e-6]}]},mass=2.,friction=.6,ground_friction=.3)
        np.testing.assert_allclose(np.diag(tiny_cube['inertia']),np.array([13.,10.,5.])*1e-12/6,rtol=1e-15)

    def test_zero_volume_and_relative_degeneracy_remain_rejected_at_every_scale(self):
        import numpy as np
        for height in (0.,1e-16):
            for scale in (1e-5,1.,1e5):
                with self.subTest(height=height,scale=scale):
                    v=(np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,height]])*scale).tolist()
                    with self.assertRaisesRegex(ValueError,'degenerate'):
                        uniform_density_profile({'shapes':[self.tetrahedron(v)]},mass=1.,friction=.6,ground_friction=.3)

    def test_user_physics_policy_preserves_source_and_explicit_omni_tasks(self):
        self.assertEqual(len(physical_policy()['hoi_retarget_mass_kg']),13)
        profile,_=source_physical_inputs('hoi_retarget',{'source':{'asset_id':'omomo/woodchair'}})
        self.assertEqual(profile,{'mass':4.,'friction':.9,'ground_friction':.3})
        for task,mass,mu in [('box/case1_carry',2.,.45),('box/case3_slide',2.,.2),('soccer/case4_carry',.5,.55)]:
            profile,_=source_physical_inputs('omnicontact',{},task=task)
            self.assertEqual((profile['mass'],profile['friction'],profile['ground_friction']),(mass,mu,1.))
        grail,_=source_physical_inputs('grail_sitting',{},source_mass=1.)
        self.assertEqual((grail['friction'],grail['ground_friction']),(1.,1.))
        with self.assertRaisesRegex(ValueError,'explicit source task'):
            source_physical_inputs('omnicontact',{})

    def test_geometry_hash_guard_and_source_physics_is_not_inferred(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'geometry.json';g={'basis':'test','shapes':[{'kind':'cube','size':[1,2,3]}]}
            p.write_bytes(canonical(g));r={'dataset':'omnicontact','object_geometry_path':str(p),'object_geometry_sha256':digest(g)}
            result=audit_geometry(r)
            self.assertEqual(result['physical_profile_status'],'unresolved')
            self.assertFalse(result['native_physics_verified'])
            g['shapes'][0]['size'][0]=2;p.write_bytes(canonical(g))
            with self.assertRaisesRegex(ValueError,'geometry SHA256'):
                audit_geometry(r)

    def test_physics_group_requires_exact_material_scene_and_inertia(self):
        meshes=[{'vertices':[[0,0,0],[1,0,0],[0,1,0]],'faces':[[0,1,2]]}]
        profile={'mass':1,'com':[0,0,0],'inertia':[1,1,1],'friction':.6,'ground_friction':.3}
        kw=dict(collision_meshes=meshes,physical_profile=profile,material_contract={'friction_combine':'native'},scene_contract={'supports':[]},cooking_contract={'recipe':'explicit_hulls'})
        base=physics_group_key(**kw)
        changed=copy.deepcopy(kw);changed['scene_contract']['supports']=['table1']
        self.assertNotEqual(base,physics_group_key(**changed))
        changed=copy.deepcopy(kw);changed['physical_profile']['mass']=2
        self.assertNotEqual(base,physics_group_key(**changed))
        changed=copy.deepcopy(kw);del changed['physical_profile']['ground_friction']
        with self.assertRaisesRegex(ValueError,'complete explicit'):
            physics_group_key(**changed)


if __name__=='__main__':
    unittest.main()
