"""Cooking fixture identity guard tests; no fake native physics success."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
from hoi.data.curated import digest, sha
from hoi.data.native_assets import BUILD_SCHEMA, exact_surface
from hoi.data.manifest import robot_joint_names
from hoi.v1.native_fixture import FIXTURE_SCHEMA, read_curated_cooking_fixture


class CookingFixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);root=Path(self.tmp.name)
        source=root/'object.usda';source.write_text('test source bytes; never opened as a native asset')
        urdf=root/'robot.urdf';urdf.write_text('test binding bytes')
        profile={'mass':2.,'com':[0.,0.,0.],'inertia':[[.1,0,0],[0,.1,0],[0,0,.1]],'friction':1.,'ground_friction':.3}
        geometry={'shapes':[{'kind':'cube','size':[.1,.2,.3]}]}
        build={'schema':BUILD_SCHEMA,'native_asset':str(source),'native_asset_sha256':sha(source),'physical_profile':profile,
               'object_geometry_sha256':digest(geometry),'exact_surface_sha256':digest(exact_surface(geometry['shapes'])),
               'collision_approximation':'convexHull','physical_evidence':{'fixture':'CPU unit test only'},
               'status':'native_cooking_and_geometry_validation_required','training_ready':False}
        build_path=root/'build.json';build_path.write_text(json.dumps(build))
        robot={'path':str(urdf),'sha256':sha(urdf),'foot_asset_manifest':{'path':'fake','sha256':'a'*64}}
        asset={k:build[k] for k in ('native_asset','native_asset_sha256','physical_profile','object_geometry_sha256')}
        asset.update(asset_id=0,object_asset='test',exact_scale=[1.,1.,1.],collision_hash='c'*64,
                     build_receipt={'path':str(build_path),'sha256':sha(build_path)})
        self.fixture={'schema':FIXTURE_SCHEMA,'asset':asset,'source_record':{'object_geometry_sha256':digest(geometry)},
            'native_robot_asset':robot,'robot_geometry_sha256':'r'*64,
            'reference_kinematics':{'urdf':{'path':str(urdf),'sha256':sha(urdf)},
                                  'body_names':['body'+str(i) for i in range(39)],'dof_names':robot_joint_names()}}
        robot_recipe=root/'robot_recipe.json'
        robot_recipe.write_text(json.dumps({k:self.fixture[k] for k in ('native_robot_asset','reference_kinematics','robot_geometry_sha256')}))
        self.fixture['robot_recipe']={'path':str(robot_recipe),'sha256':sha(robot_recipe)}
        self.clip={'label_metadata':{'robot_geometry_sha256':'r'*64},'object_geometry':geometry}
        self.path=root/'fixture.json'
        self.binding=patch('hoi.v1.prepare.native_robot_asset_binding',return_value=robot);self.binding.start();self.addCleanup(self.binding.stop)
        self.clip_loader=patch('hoi.data.curated.load_curated_clip',return_value=self.clip);self.clip_loader.start();self.addCleanup(self.clip_loader.stop)

    def write(self):
        self.fixture['content_sha256']=digest({k:v for k,v in self.fixture.items() if k!='content_sha256'})
        self.path.write_text(json.dumps(self.fixture))

    def test_exact_source_build_profile_and_robot_binding_accepted_as_unverified_fixture(self):
        self.write();fixture,build,clip=read_curated_cooking_fixture(self.path)
        self.assertFalse(build['training_ready']);self.assertEqual(fixture['schema'],FIXTURE_SCHEMA)
        self.assertEqual(build['exact_surface_sha256'],digest(exact_surface(clip['object_geometry']['shapes'])))

    def test_rehashed_asset_profile_cannot_override_build(self):
        self.fixture['asset']['physical_profile']=deepcopy(self.fixture['asset']['physical_profile'])
        self.fixture['asset']['physical_profile']['mass']=3.;self.write()
        with self.assertRaisesRegex(ValueError,'build mismatch: physical_profile'):read_curated_cooking_fixture(self.path)

    def test_changed_source_surface_cannot_be_called_compiled_geometry(self):
        self.write();self.clip['object_geometry']['shapes'][0]['size']=[.2,.2,.3]
        with self.assertRaisesRegex(ValueError,'source surface'):read_curated_cooking_fixture(self.path)

    def test_changed_label_identity_and_source_bytes_rejected(self):
        self.fixture['source_record']['object_geometry_sha256']='a'*64;self.write()
        with self.assertRaisesRegex(ValueError,'source label/native'):read_curated_cooking_fixture(self.path)
        self.fixture['source_record']['object_geometry_sha256']=self.fixture['asset']['object_geometry_sha256'];self.write()
        Path(self.fixture['asset']['native_asset']).write_text('tampered')
        with self.assertRaisesRegex(ValueError,'input bytes'):read_curated_cooking_fixture(self.path)

    def test_deferred_inertia_is_mass_only_and_never_a_physics_rollout(self):
        from hoi.v1.native_fixture import DEFERRED_INERTIA_MODE, validate_cooking_mass_profile
        from hoi.v1.native import HOITrackingSim
        profile={'mass':1.,'friction':.9,'ground_friction':.3}
        validate_cooking_mass_profile(profile)
        for extra in ({'com':[0,0,0]},{'inertia':[[1,0,0],[0,1,0],[0,0,1]]}):
            with self.assertRaisesRegex(ValueError,'placeholder'):
                validate_cooking_mass_profile({**profile,**extra})
        build_path=Path(self.fixture['asset']['build_receipt']['path'])
        build=json.loads(build_path.read_text());build.update(inertia_mode=DEFERRED_INERTIA_MODE,physical_profile=profile)
        build_path.write_text(json.dumps(build))
        self.fixture['asset'].update(inertia_mode=DEFERRED_INERTIA_MODE,physical_profile=profile)
        self.fixture['asset']['build_receipt']['sha256']=sha(build_path);self.write()
        read_curated_cooking_fixture(self.path)
        env=HOITrackingSim.__new__(HOITrackingSim);env.asset={'inertia_mode':DEFERRED_INERTIA_MODE}
        with self.assertRaisesRegex(RuntimeError,'geometry-only'):env._advance(None)
        with self.assertRaisesRegex(ValueError,'training or evaluation'):
            HOITrackingSim(None,1,None,None,{'inertia_mode':DEFERRED_INERTIA_MODE},None)

    def test_all_curated_cooking_profiles_end_after_export_without_physics_or_gate_file(self):
        from types import SimpleNamespace
        from hoi.v1.native import HOITrackingSim
        from hoi.v1.native_fixture import finish_geometry_only_cooking,DEFERRED_INERTIA_MODE
        for mode in (None,DEFERRED_INERTIA_MODE):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root=Path(temp);(root/"compiled_geometry").mkdir()
                (root/"compiled_geometry/assets.json").write_text(json.dumps({"schema":"hoi_v1_compiled_geometry_recipe_v1",
                    "assets":[{"native_compiled_surface_sha256":"n"*64,"source_exact_surface_sha256":"s"*64}]}))
                events=[]
                def forbidden(*args,**kwargs):
                    raise AssertionError("cooking stage must not step or calibrate physics")
                env=SimpleNamespace(cooking_only=True,contacts=object(),physics_report={},compiled_foot_audit={},N=1,
                    close=lambda:events.append("closed"),_advance=forbidden,step=forbidden,calibrate_pair_probes=forbidden,
                    calibrate_support_probes=forbidden)
                source={"scope":"curated_native_cooking_fixture_not_training","inertia_mode":mode}
                with patch("hoi.v1.native_acceptance.capture_native_runtime",return_value={"scope":"unit test metadata only"}):
                    report=finish_geometry_only_cooking(env,source,root,hand_audit={},self_audit={})
                self.assertEqual(events,["closed"]);self.assertEqual(report['physics_steps'],0)
                self.assertFalse(report['native_pair_calibration_complete']);self.assertFalse(report['zero_action_smoke_completed'])
                self.assertTrue((root/'cooking.json').is_file());self.assertFalse((root/'native.json').exists())
                native=HOITrackingSim.__new__(HOITrackingSim);native.cooking_only=True;native.asset={}
                with self.assertRaisesRegex(RuntimeError,'geometry-only'):native._advance(None)

    def test_content_hash_cannot_be_omitted(self):
        self.path.write_text(json.dumps(self.fixture))
        with self.assertRaisesRegex(ValueError,'content identity'):read_curated_cooking_fixture(self.path)


if __name__=='__main__':unittest.main()
