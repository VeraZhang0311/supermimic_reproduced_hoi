import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import trimesh
from hoi.data.native_assets import canonical,digest,sha
from hoi.data.native_cooking import (make_plan,run_task,collect_exports,read_plan,
                                    surface_distance_report,unsigned_distances,plan_fixtures,
                                    adopt_existing_output,collect_registry_exports,
                                    _planning_identity_cache,_identity,_PLAN_IDENTITY_CACHE)


class CookingBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.old_threads=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,self.old_threads)
        source=self.root/'source';(source/'hoi/v1').mkdir(parents=True);(source/'hoi/v1/native.py').write_text('# synthetic source binding')
        self.source=source
        box=trimesh.creation.box(extents=[1.,1.,1.]);self.mesh={'vertices':box.vertices.tolist(),'faces':box.faces.tolist()}
        geometry={'shapes':[dict(self.mesh,kind='mesh')]};g=self.root/'geometry.json';g.write_bytes(canonical(geometry))
        native=self.root/'native.usda';native.write_text('synthetic hash fixture; no native execution')
        asset={'object_geometry_sha256':digest(geometry),'native_asset':str(native),'native_asset_sha256':sha(native),'collision_meshes':[self.mesh]}
        registry={'schema':'hoi_curated_native_source_registry_v1','selected_geometry_identities':1,'object_cooking_groups':[],
                  'bindings':[{'dataset':'ultra','object_geometry_sha256':digest(geometry),'asset_recipe':asset,
                               'source_record':{'object_geometry_path':str(g),'object_geometry_sha256':digest(geometry)}}]}
        registry['content_sha256']=digest(registry);self.registry=self.root/'registry.json';self.registry.write_bytes(canonical(registry))

    def test_resume_checks_hashes_and_allocates_fresh_attempt_without_gate_claim(self):
        output=self.root/'batch';plan=make_plan(self.registry,self.source,sys.executable,['--hand-collision','--self-collision'],output,surface_samples=64)
        path=output/'plan.json'
        with contextlib.redirect_stdout(io.StringIO()):
            first=run_task(path,0);second=run_task(path,0)
        self.assertEqual(first,second);self.assertFalse(first['native_acceptance_pass'])
        task=output/'tasks'/plan['tasks'][0]['task_id'];self.assertEqual(len(list(task.glob('attempt_*'))),1)
        (task/'attempt_0001/bootstrap.json').write_bytes(b'corrupt')
        run_task(path,0)
        self.assertTrue((task/'attempt_0001/bootstrap.json').exists())
        self.assertTrue((task/'attempt_0002/SUCCESS.json').exists())
        self.assertEqual(collect_exports(path,self.root/'collection'),{})
        (self.source/'hoi/v1/native.py').write_text('# changed executable source')
        with self.assertRaisesRegex(ValueError,'executable source changed'):
            read_plan(path)

    def test_missing_tasks_cannot_produce_finalize_mapping(self):
        output=self.root/'pending';make_plan(self.registry,self.source,sys.executable,['--hand-collision','--self-collision'],output,surface_samples=64)
        with self.assertRaisesRegex(ValueError,'incomplete/unverified'):
            collect_exports(output/'plan.json',self.root/'bad_collection')
        with self.assertRaisesRegex(ValueError,'owns native input'):
            make_plan(self.registry,self.source,sys.executable,['--hand-collision','--self-collision','--output=x'],self.root/'bad')

    def test_planning_cache_reuses_only_bound_unchanged_identity(self):
        path=self.root/'cache_input.bin';path.write_bytes(b'original')
        expected=sha(path)
        with patch('hoi.data.native_cooking.sha',wraps=sha) as counter:
            with _planning_identity_cache():
                first=_identity(path,expected)
                self.assertEqual(_identity(path.parent/'.'/path.name,expected),first)
                self.assertEqual(counter.call_count,1)
                _identity(path);_identity(path)
                self.assertEqual(counter.call_count,3)  # Unbound requests are never cached.
                stat=path.stat();os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1_000_000))
                self.assertEqual(_identity(path,expected),first)
                self.assertEqual(counter.call_count,4)  # Same content, changed stat: rehash.
                with self.assertRaisesRegex(ValueError,'input identity changed'):
                    _identity(path,'0'*64)
                self.assertEqual(counter.call_count,5)  # Different expected digest cannot hit.
                path.write_bytes(b'changed bytes with different size')
                with self.assertRaisesRegex(ValueError,'input identity changed'):
                    _identity(path,expected)
                self.assertEqual(counter.call_count,6)
            self.assertIsNone(_PLAN_IDENTITY_CACHE.get())
            _identity(path,sha(path));_identity(path,sha(path))
            self.assertEqual(counter.call_count,8)  # Runtime stages never retain plan cache.

    def test_planning_cache_keeps_byte_and_canonical_json_kinds_separate(self):
        path=self.root/'cache_json.json';path.write_bytes(canonical({'x':1}))
        expected=sha(path)
        from hoi.data.native_cooking import _json
        with _planning_identity_cache(),patch('hoi.data.native_cooking.sha',wraps=sha) as byte_reader,patch('hoi.data.native_cooking._json',wraps=_json) as json_reader:
            _identity(path,expected,'bytes');_identity(path,expected,'canonical_json')
            _identity(path,expected,'bytes');_identity(path,expected,'canonical_json')
            self.assertEqual(byte_reader.call_count,1);self.assertEqual(json_reader.call_count,1)
    def test_plan_scope_cache_clears_on_failure_and_rejects_midread_mutation(self):
        path=self.root/'midread';path.write_bytes(b'before');expected=sha(path)
        def mutate_after_read(p):
            value=sha(p);Path(p).write_bytes(b'after with another size');return value
        with self.assertRaisesRegex(ValueError,'input changed while hashing'):
            with _planning_identity_cache(),patch('hoi.data.native_cooking.sha',side_effect=mutate_after_read):
                _identity(path,expected)
        self.assertIsNone(_PLAN_IDENTITY_CACHE.get())
        with self.assertRaisesRegex(ValueError,'input identity changed'):
            _identity(path,expected)

    def test_plan_preserves_python_entry_symlink_while_hashing_real_binary(self):
        entry=self.root/'venv/bin/python';entry.parent.mkdir(parents=True)
        try:
            entry.symlink_to(sys.executable)
        except OSError as error:
            self.skipTest('OS does not permit creating a test executable symlink: '+str(error))
        plan=make_plan(self.registry,self.source,entry,['--hand-collision','--self-collision'],
                       self.root/'venv_entry_plan',surface_samples=64)
        self.assertEqual(plan['execution']['python'],str(entry.absolute()))
        self.assertNotEqual(Path(plan['execution']['python']),entry.resolve())
        self.assertEqual(plan['execution']['python_identity']['path'],str(entry.resolve()))
        self.assertEqual(plan['execution']['python_identity']['sha256'],sha(sys.executable))
        self.assertEqual(read_plan(self.root/'venv_entry_plan/plan.json')['execution']['python'],str(entry.absolute()))
    def test_make_plan_deduplicates_shared_binding_but_run_rehashes_it(self):
        registry=json.loads(self.registry.read_bytes())
        first=registry['bindings'][0]
        second=json.loads(json.dumps(first));geometry=json.loads((self.root/'geometry.json').read_bytes())
        geometry['distinct_label_metadata']='second identity, same collision surface'
        geometry_path=self.root/'geometry_second.json';geometry_path.write_bytes(canonical(geometry))
        second['object_geometry_sha256']=digest(geometry)
        second['asset_recipe']['object_geometry_sha256']=digest(geometry)
        second['source_record']={'object_geometry_path':str(geometry_path),'object_geometry_sha256':digest(geometry)}
        registry['bindings'].append(second);registry['selected_geometry_identities']=2
        registry['content_sha256']=digest({k:v for k,v in registry.items() if k!='content_sha256'})
        self.registry.write_bytes(canonical(registry))
        native=(self.root/'native.usda').resolve()
        with patch('hoi.data.native_cooking.sha',wraps=sha) as counter:
            output=self.root/'shared_plan'
            make_plan(self.registry,self.source,sys.executable,['--hand-collision','--self-collision'],output,surface_samples=64)
            native_reads=lambda:sum(Path(call.args[0]).resolve()==native for call in counter.call_args_list)
            self.assertEqual(native_reads(),1)
            self.assertIsNone(_PLAN_IDENTITY_CACHE.get())
            with contextlib.redirect_stdout(io.StringIO()):run_task(output/'plan.json',0)
            after_first=native_reads();self.assertGreater(after_first,1)
            with contextlib.redirect_stdout(io.StringIO()):run_task(output/'plan.json',0)
            self.assertGreater(native_reads(),after_first)
    def test_module_and_direct_script_cli_bootstrap_paths_are_equivalent(self):
        import subprocess
        from hoi.data import native_cooking
        repo=Path(native_cooking.__file__).resolve().parents[2]
        args_file=self.root/'args.json';args_file.write_bytes(canonical(['--hand-collision','--self-collision']))
        env={**os.environ,'PYTHONPATH':str(repo)+os.pathsep+os.environ.get('PYTHONPATH','')}
        plans=[]
        for name,entry in (('module',['-m','hoi.data.native_cooking']),('script',[str(Path(native_cooking.__file__).resolve())])):
            output=self.root/name
            commands=[['plan','--registry',str(self.registry),'--source',str(self.source),'--python',sys.executable,
                       '--native-args',str(args_file),'--output',str(output),'--surface-samples','64'],
                      ['run','--plan',str(output/'plan.json'),'--task-index','0'],
                      ['collect','--plan',str(output/'plan.json'),'--output',str(self.root/(name+'_collection'))]]
            for command in commands:
                result=subprocess.run([sys.executable,*entry,*command],cwd=repo,env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            plan=json.loads((output/'plan.json').read_bytes());plans.append(plan)
            receipt=next(output.glob('tasks/*/attempt_*/SUCCESS.json'))
            self.assertFalse(json.loads(receipt.read_bytes())['native_acceptance_pass'])
            self.assertEqual(json.loads((self.root/(name+'_collection/exports.json')).read_bytes()),{})
        self.assertEqual(plans[0]['tasks'],plans[1]['tasks'])
        self.assertEqual(plans[0]['execution_sha256'],plans[1]['execution_sha256'])

    def test_repair_data_clis_support_module_and_script_help(self):
        import subprocess
        from hoi.data import native_assets,native_repairs
        repo=Path(native_assets.__file__).resolve().parents[2]
        env={**os.environ,'PYTHONPATH':str(repo)+os.pathsep+os.environ.get('PYTHONPATH','')}
        for module in (native_assets,native_repairs):
            for entry in (['-m',module.__name__],[str(Path(module.__file__).resolve())]):
                result=subprocess.run([sys.executable,*entry,'--help'],cwd=repo,env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def _synthetic_cooking_fixture(self):
        # Serialized contract fixture only: no GPU execution or real acceptance evidence.
        geometry=json.loads((self.root/'geometry.json').read_bytes());label=digest(geometry)
        output=self.root/'source_labels';output.mkdir()
        (output/'metadata.json').write_text('{}');(output/'labels.npz').write_bytes(b'test bytes only')
        robot=self.root/'robot.usda';robot.write_text('test robot')
        urdf=self.root/'robot.urdf';urdf.write_text('test URDF')
        binding=lambda p:{'path':str(p),'sha256':sha(p)}
        profile={'mass':1.,'friction':.6,'ground_friction':.3}
        build={'object_geometry_sha256':label,'physical_profile':profile,
               'exact_surface_sha256':digest([dict(self.mesh,kind='mesh')]),
               'inertia_mode':'native_cooking_only_uniform_density','collision_approximation':'convexDecomposition'}
        build_path=self.root/'build.json';build_path.write_bytes(canonical(build))
        native=self.root/'native.usda'
        fixture={'schema':'hoi_curated_native_cooking_fixture_v1',
                 'asset':{'object_geometry_sha256':label,'native_asset':str(native),'native_asset_sha256':sha(native),
                          'physical_profile':profile,'build_receipt':binding(build_path)},
                 'native_robot_asset':binding(robot),'reference_kinematics':{'urdf':binding(urdf)},
                 'source_record':{'dataset':'omnicontact','output':str(output),'metadata_sha256':sha(output/'metadata.json'),
                                  'labels_sha256':sha(output/'labels.npz'),'object_geometry_path':str(self.root/'geometry.json'),
                                  'object_geometry_sha256':label}}
        fixture['content_sha256']=digest(fixture)
        fixture_path=self.root/'fixture.json';fixture_path.write_bytes(canonical(fixture));return fixture_path

    def _synthetic_cooking_output(self,plan):
        task=plan['tasks'][0];fixture=json.loads(Path(task['fixture']['path']).read_bytes())
        output=self.root/'old_source_cooking';output.mkdir()
        native=self.root/'cooked.usda';native.write_text('synthetic cooked output')
        asset={'source_native_asset_sha256':fixture['asset']['native_asset_sha256'],
               'native_asset':str(native),'native_asset_sha256':sha(native),'native_convex_evidence':{'test_fixture':True},
               'collision_meshes':[self.mesh],'object_geometry_sha256':task['source_geometry']['sha256'],
               'source_exact_surface_sha256':digest([dict(self.mesh,kind='mesh')]),
               'native_compiled_surface_sha256':digest([self.mesh]),'physical_profile':fixture['asset']['physical_profile']}
        export=output/'assets.json';export.write_bytes(canonical({'schema':'hoi_v1_compiled_geometry_recipe_v1','assets':[asset]}))
        runtime={'schema':'hoi_native_cuda_execution_v1','backend':'Newton/MuJoCo-Warp',
                 'buffer_devices':{name:'cuda:0' for name in ('contact_geom','normal_force','qpos','interval_impulse')},
                 'verified_from_live_buffers':True,'gpu_name':'synthetic CPU test fixture; not executed',
                 'versions':plan['execution']['runtime_versions']}
        report={'schema':'hoi_native_geometry_cooking_report_v1','physics_steps':0,'formal_training_started':False,
                'training_ready':False,'native_pair_calibration_complete':False,'num_envs':1,'seed':123,
                'source':{'fixture_manifest_sha256':task['fixture']['sha256'],'scope':'curated_native_cooking_fixture_not_training'},
                'native_runtime':runtime,'compiled_geometry_export':str(export),'compiled_geometry_export_sha256':sha(export),
                'source_exact_surface_sha256':asset['source_exact_surface_sha256'],
                'native_compiled_surface_sha256':asset['native_compiled_surface_sha256']}
        (output/'cooking.json').write_bytes(canonical(report));(output/'source.json').write_bytes(canonical(plan['execution']['source_files']))
        return output

    def test_partial_plan_adoption_keeps_producer_identity_and_needs_full_coverage(self):
        fixture=self._synthetic_cooking_fixture()
        versions={key:'synthetic-test' for key in ('torch','warp','mujoco','mujoco_warp','newton')}
        with patch('hoi.data.native_cooking._runtime_versions',return_value=versions):
            batch=self.root/'partial'
            plan=plan_fixtures([fixture],self.source,sys.executable,['--hand-collision','--self-collision'],
                               batch,surface_samples=64,defer_surface_report=True)
            self.assertFalse(plan['full_selection']);self.assertEqual(plan['geometry_identity_count'],1)
            native=self._synthetic_cooking_output(plan);report_bytes=(native/'cooking.json').read_bytes()
            receipt=adopt_existing_output(batch/'plan.json',0,native)
            self.assertTrue(receipt['adopted_existing_output']);self.assertFalse(receipt['native_acceptance_pass'])
            self.assertEqual((native/'cooking.json').read_bytes(),report_bytes)
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(run_task(batch/'plan.json',0),receipt)
            attempt=batch/'tasks'/plan['tasks'][0]['task_id']/'attempt_0001'
            adoption=json.loads((attempt/'adoption.json').read_bytes())
            self.assertEqual(adoption['producer_seed'],123);self.assertEqual(adoption['producer_source'],str(self.source))
            self.assertFalse((attempt/'command.json').exists())
            exports=collect_exports(batch/'plan.json',self.root/'partial_collection');self.assertEqual(len(exports),1)
            full=json.loads(self.registry.read_bytes());full['selected_geometry_identities']=2
            full['object_cooking_groups']=[{'object_build_group_sha256':plan['tasks'][0]['object_build_group_sha256'],
                'members':plan['tasks'][0]['covered_geometry_identities']}]
            full['content_sha256']=digest({k:v for k,v in full.items() if k!='content_sha256'})
            full_path=self.root/'full_registry.json';full_path.write_bytes(canonical(full))
            with self.assertRaisesRegex(ValueError,'missing 0 cooked groups and 1 bootstrap'):
                collect_registry_exports(full_path,[batch/'plan.json'],self.root/'incomplete_full')
            (native/'source.json').write_bytes(canonical({'changed_source.py':'bad'}))
            with self.assertRaisesRegex(ValueError,'different sources'):
                adopt_existing_output(batch/'plan.json',0,native)
            with self.assertRaisesRegex(ValueError,'incomplete/unverified'):
                collect_exports(batch/'plan.json',self.root/'corrupted_collection')

    def test_adoption_rejects_rebinding_existing_cook_to_new_source(self):
        fixture=self._synthetic_cooking_fixture()
        versions={key:'synthetic-test' for key in ('torch','warp','mujoco','mujoco_warp','newton')}
        with patch('hoi.data.native_cooking._runtime_versions',return_value=versions):
            plan=plan_fixtures([fixture],self.source,sys.executable,['--hand-collision','--self-collision'],
                               self.root/'producer_plan',surface_samples=64,defer_surface_report=True)
            native=self._synthetic_cooking_output(plan)
            other=self.root/'new_source';(other/'hoi/v1').mkdir(parents=True)
            (other/'hoi/v1/native.py').write_text('# different source release')
            plan_fixtures([fixture],other,sys.executable,['--hand-collision','--self-collision'],
                          self.root/'rebind_plan',surface_samples=64,defer_surface_report=True)
            with self.assertRaisesRegex(ValueError,'different sources'):
                adopt_existing_output(self.root/'rebind_plan/plan.json',0,native)
            self.assertFalse(list((self.root/'rebind_plan').rglob('SUCCESS.json')))
    def test_unsigned_open_triangle_distance_and_sampled_geometry_report(self):
        tri={'vertices':np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),'faces':np.asarray([[0,1,2]])}
        result=unsigned_distances([[.2,.2,2.],[2.,0.,0.],[-.1,-.1,0.]],[tri])
        np.testing.assert_allclose(result,[2.,1.,np.sqrt(.02)],rtol=0,atol=1e-12)
        geometry={'shapes':[dict(self.mesh,kind='mesh')]};asset={'collision_meshes':[self.mesh]}
        report=surface_distance_report(geometry,asset,samples=64,seed=4)
        self.assertLess(report['source_to_cooked_component_surfaces']['sampled_max_m'],1e-12)
        self.assertLess(report['cooked_component_surfaces_to_source']['sampled_max_m'],1e-12)
        self.assertIsNone(report['threshold']);self.assertFalse(report['native_acceptance_pass'])
        shifted={'vertices':(np.asarray(self.mesh['vertices'])+[2.,0.,0.]).tolist(),'faces':self.mesh['faces']}
        moved=surface_distance_report(geometry,{'collision_meshes':[shifted]},samples=64,seed=4)
        self.assertGreater(moved['source_to_cooked_component_surfaces']['mean_m'],1.)
        self.assertGreater(moved['cooked_component_surfaces_to_source']['mean_m'],1.)


if __name__=='__main__':unittest.main()
