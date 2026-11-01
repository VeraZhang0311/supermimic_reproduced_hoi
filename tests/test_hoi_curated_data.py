import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from hoi.data.curated import (canonical_bytes, digest, sha, group_labels, interval_schedule, reconcile_records,
                             stratified_split, write_split, read_split, load_curated_clip, collect_metadata)
from hoi.data.curated_bundle import prepare_curated_bundle, read_curated_bundle_header
from hoi.data.manifest import robot_joint_names
from hoi.v1.contacts import BODY_GROUP_LINKS, CHANNELS
from hoi.v1.prepare import load_bundle
from tests.test_hoi_v1_geometry import cube, BundleKinematics

BODY_NAMES = [x for group in BODY_GROUP_LINKS for x in group] + ['pelvis','imu_in_pelvis','waist_yaw_link','waist_roll_link','d435_link','imu_in_torso','mid360_link']
SOURCE_CHANNELS = [x+'__object' for x in BODY_NAMES] + ['object__ground']
ACTIVE = np.asarray([x not in ('pelvis','imu_in_pelvis','waist_yaw_link','waist_roll_link','d435_link','imu_in_torso','mid360_link') for x in BODY_NAMES]+[True])


def labels(n):
    a=np.zeros((n,40),dtype=np.int8);a[:,~ACTIVE]=-1;return a


class RegionIntentTests(unittest.TestCase):
    def test_positive_substitution_and_conservative_unknown(self):
        a=labels(4);hand=SOURCE_CHANNELS.index('right_rubber_hand__object');wrist=SOURCE_CHANNELS.index('right_wrist_pitch_link__object')
        a[0,wrist]=1; a[1,hand]=-1; a[2,hand]=-1;a[2,wrist]=1
        result=group_labels(a,SOURCE_CHANNELS,ACTIVE)
        self.assertEqual(result[:,1].tolist(),[1,-1,1,0])
        self.assertEqual(result.shape,(4,16))
        active=ACTIVE.copy();active[[SOURCE_CHANNELS.index(x+'__object') for x in BODY_GROUP_LINKS[1]]]=False
        a[:,~active]=-1
        self.assertTrue(np.all(group_labels(a,SOURCE_CHANNELS,active)[:,1]==-1))

    def test_short_native_event_survives_between_target_samples(self):
        target=labels(2);source=labels(5);foot=SOURCE_CHANNELS.index('right_ankle_roll_link__object');source[2,foot]=1
        out=interval_schedule({'timestamps_s':np.array([0.,.02]),'target_labels':target,'source_timestamps_s':np.arange(5)*.005,'source_labels':source},
                              {'channels':SOURCE_CHANNELS,'active_geometry_channels':ACTIVE})
        self.assertFalse(out['expected_valid'][0].any())
        self.assertEqual(out['region_labels'][1,13],1)
        self.assertEqual(out['region_labels'][1,0],0)

    def test_unknown_subsample_and_inactive_channel_never_become_negative(self):
        source=labels(5);source[2,SOURCE_CHANNELS.index('left_wrist_pitch_link__object')]=-1
        out=interval_schedule({'timestamps_s':np.array([0.,.02]),'target_labels':labels(2),'source_timestamps_s':np.arange(5)*.005,'source_labels':source},
                              {'channels':SOURCE_CHANNELS,'active_geometry_channels':ACTIVE})
        self.assertEqual(out['region_labels'][1,0],-1)
        source[0,np.flatnonzero(~ACTIVE)[0]]=0
        with self.assertRaisesRegex(ValueError,'inactive'):
            group_labels(source,SOURCE_CHANNELS,ACTIVE)


class PartitionTests(unittest.TestCase):
    def test_stratification_protected_groups_and_determinism(self):
        records=[{'dataset':d,'clip_id':str(i),'source_group':f'{d}:{i}','task':task,'mandatory_protected':False}
                 for d,task in [('a','sit'),('b','kick')] for i in range(100)]
        records[0]['mandatory_protected']=True
        parts,report=stratified_split(records,.1,12)
        self.assertEqual(report['counts'],{'train':180,'protected_eval':20})
        self.assertEqual([r['protected_eval'] for r in report['task_strata']],[10,10])
        self.assertEqual(parts,stratified_split(list(reversed(records)),.1,12)[0])
        self.assertTrue(any(x['mandatory_protected'] for x in parts['protected_eval']))

    def test_cross_source_capture_cannot_leak_and_forced_eval_can_exceed_target(self):
        records=[{'dataset':d,'clip_id':str(i),'source_group':str(i),'task':'unspecified','mandatory_protected':d=='a' and i<3}
                 for d in ('a','b') for i in range(10)]
        parts,report=stratified_split(records,.1)
        self.assertEqual(report['counts']['protected_eval'],6)
        self.assertFalse({r['source_group'] for r in parts['train']} & {r['source_group'] for r in parts['protected_eval']})

    def test_exact_motion_hash_unions_assets_and_propagates_protection_not_parquet_hash(self):
        rows=[];metadata=[]
        for dataset,count in [('grail_sitting',2),('hoi_retarget',2)]:
            for i in range(count):
                row={'dataset':dataset,'clip_id':str(i),'recipe_id':'r','output':f'/{dataset}/{i}'}
                if dataset=='grail_sitting':
                    provenance={'source_motion_key':f'chair{i}_actor_rand00001','subset':'sitting','files':{'robot':{'sha256':'a'*64}},'split':'validation' if i else 'train'}
                else:
                    provenance={'source_group':f'sub{i}_box_000','source_sha256':'b'*64,'split':'unassigned'}
                rows.append(row);metadata.append({**row,'metadata_sha256':'c'*64,'labels_sha256':'d'*64,
                  'object_geometry_sha256':'e'*64,'object_geometry_path':'geometry','provenance':provenance})
        reconciled=reconcile_records(rows,metadata)
        grail=[x for x in reconciled if x['dataset']=='grail_sitting']
        hoi=[x for x in reconciled if x['dataset']=='hoi_retarget']
        self.assertEqual(grail[0]['source_group'],grail[1]['source_group'])
        self.assertIn('exact_source_motion_bytes_union',grail[0]['group_basis'])
        self.assertNotEqual(hoi[0]['source_group'],hoi[1]['source_group'])
        partitions,_=stratified_split(reconciled)
        self.assertEqual(len([x for x in partitions['protected_eval'] if x['dataset']=='grail_sitting']),2)
        self.assertTrue(all(x['task']=='unspecified' for x in hoi))

    def test_unknown_task_is_explicit_and_grail_variants_stay_together(self):
        rows=[{'dataset':'grail_sitting','clip_id':str(i),'recipe_id':'r','output':str(i)} for i in range(2)]
        meta=[{**r,'metadata_sha256':'a'*64,'labels_sha256':'b'*64,'object_geometry_sha256':'c'*64,'object_geometry_path':'geometry',
               'provenance':{'source_motion_key':'chair_actor_motion_rand0000'+str(i),'subset':'sitting'}} for i,r in enumerate(rows)]
        out=reconcile_records(rows,meta)
        self.assertEqual(out[0]['source_group'],out[1]['source_group'])
        self.assertEqual(out[0]['task'],'sitting')


class CuratedBundleTests(unittest.TestCase):
    def setUp(self):
        self.old_threads=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,self.old_threads)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        selection=self.root/'selection';selection.mkdir();self.selection=selection
        geo={'shapes':[{'kind':'cube','size':[.2,.2,.2]}],'basis':'synthetic_geometry_only'}
        geometry=self.root/'geometry.json';geometry.write_bytes(canonical_bytes(geo));self.geometry_sha=digest(geo)
        rows=[];metarows=[]
        for i in range(2):
            out=self.root/f'clip{i}';out.mkdir();n=3;t=np.arange(n)*.02
            a={'timestamps_s':t,'source_timestamps_s':t,'root_pos':np.column_stack([t,np.zeros(n),np.ones(n)]),
               'root_quat':np.tile([1.,0,0,0],(n,1)),'joint_pos':np.zeros((n,29)),
               'object_pos':np.tile([1.,0,.2],(n,1)),'object_quat':np.tile([1.,0,0,0],(n,1)),
               'target_labels':labels(n),'source_labels':labels(n)}
            a['target_labels'][1,SOURCE_CHANNELS.index('right_wrist_pitch_link__object')]=1
            a['source_labels'][1,SOURCE_CHANNELS.index('right_wrist_pitch_link__object')]=1
            np.savez_compressed(out/'labels.npz',**a)
            provenance={'source_group':f'sub{i}_box_000','source_joint_names':robot_joint_names(),'split':'validation' if i else 'train','quaternion_order':'wxyz'}
            metadata={'schema':'g1_expected_surface_contact_v1','clip_id':f'clip{i}','recipe_id':'recipe','channels':SOURCE_CHANNELS,
                      'robot_geometry_sha256':'f'*64,'active_geometry_channels':ACTIVE.tolist(),'provenance':provenance,'semantics':{'not_force_ground_truth':True},
                      'array_file':{'path':'labels.npz','sha256':sha(out/'labels.npz')},'object_geometry_path':str(geometry),
                      'object_geometry_sha256':self.geometry_sha,'object_asset':{'kind':'cube'},'scene_supports':[],
                      'statistics':{'source_frames':n},'timebase':{'target_hz':50.,'state_count':n,'interval_count':n-1}}
            (out/'metadata.json').write_bytes(canonical_bytes(metadata))
            success={'metadata_sha256':sha(out/'metadata.json'),'array_sha256':sha(out/'labels.npz')};(out/'SUCCESS.json').write_bytes(canonical_bytes(success))
            row={'dataset':'ultra','clip_id':f'clip{i}','output':str(out),'recipe_id':'recipe','selection_status':'included','eligible_for_privileged_tracking':True}
            rows.append(row);metarows.append({**row,'metadata_sha256':success['metadata_sha256'],'labels_sha256':success['array_sha256'],
                'provenance':provenance,'object_geometry_sha256':self.geometry_sha,'object_geometry_path':str(geometry)})
        (selection/'included.jsonl').write_text(''.join(canonical_bytes(r).decode()+'\n' for r in rows),encoding='utf-8')
        (selection/'excluded.jsonl').write_text('',encoding='utf-8');(selection/'policy.json').write_text('{}',encoding='utf-8')
        manifest={'schema':'hoi_corpus_selection_v1','selection_id':'synthetic','included_count':2,'excluded_count':0,'included_by_dataset':{'ultra':2},
                  'files':{name:{'sha256':sha(selection/name)} for name in ('included.jsonl','excluded.jsonl','policy.json')}}
        (selection/'manifest.json').write_bytes(canonical_bytes(manifest));(selection/'READY.json').write_bytes(canonical_bytes({'manifest_sha256':sha(selection/'manifest.json')}))
        metadata_path=self.root/'metadata.json';metadata_path.write_bytes(canonical_bytes(metarows))
        self.split=self.root/'split/split.json';write_split(selection,metadata_path,self.split.parent)
        mesh=cube(.1);hand=cube(.01)
        serial=lambda m:{k:v.tolist() for k,v in m.items()}
        native=self.root/'robot.usda';native.write_text('synthetic CPU binding, no native simulation')
        (self.root/'hand-asset.json').write_text('{}')
        urdf=self.root/'robot.urdf';urdf.write_text('synthetic hash binding')
        self.recipe=self.root/'recipe.json'
        recipe={'profile':'custom','robot_geometry_sha256':'f'*64,'native_robot_asset':{'path':str(native),'sha256':sha(native)},
                'reference_kinematics':{'body_names':BODY_NAMES,'dof_names':robot_joint_names(),'urdf':{'path':str(urdf),'sha256':sha(urdf)}},
                'hands':{s:{'collision_meshes':[serial(hand)],'palm_points':[[.01,0,0]]} for s in ('left','right')},
                'assets':[{'object_asset':'synthetic','object_geometry_sha256':self.geometry_sha,'exact_scale':[1,1,1],
                           'mesh_to_object':torch.eye(4).tolist(),'native_asset':str(native),'native_asset_sha256':sha(native),
                           'physical_profile':{'mass':1,'com':[0,0,0],'inertia':torch.eye(3).tolist(),'friction':.5},
                           'collision_meshes':[serial(mesh)],'distance_recipe':{'grid_resolution':12,'candidate_count':128,'max_grid_error':.05}}]}
        self.recipe.write_bytes(canonical_bytes(recipe));self.kin=BundleKinematics()

    def test_real_interface_roundtrip_and_train_eval_protection(self):
        output=self.root/'bundle'
        bundle=prepare_curated_bundle(self.split,self.recipe,output,self.kin)
        bank,shapes,loaded=load_bundle(output/'bundle.json',self.kin)
        self.assertEqual(loaded,bundle);self.assertEqual(bank.expected_contact.shape,(3,16))
        self.assertIsNone(bank.source_reference);self.assertEqual(shapes.points(torch.tensor([0])).shape,(1,128,3))
        self.assertFalse(bank.expected_valid[0].any());self.assertEqual(float(bank.expected_contact[1,1]),1.)
        self.assertEqual(float(bank.root_v[1,0]),1.)
        evaluation=self.root/'eval';prepare_curated_bundle(self.split,self.recipe,evaluation,self.kin,partition='protected_eval')
        with self.assertRaisesRegex(ValueError,'only train'):
            load_bundle(evaluation/'bundle.json',self.kin)
        eb,_,_=load_bundle(evaluation/'bundle.json',self.kin,require_train=False)
        self.assertEqual(eb.partition,'protected_eval');self.assertIsNone(eb.source_reference)
        cache=Path(bundle['clips'][0]['contact_cache']['path']);cache.write_bytes(cache.read_bytes()+b'x')
        with self.assertRaisesRegex(ValueError,'cache SHA256'):
            load_bundle(output/'bundle.json',self.kin)

    def test_compact_recipe_roundtrips_through_bundle_header(self):
        from hoi.data.curated_bundle import NATIVE_ALIAS_SCHEMA
        recipe=json.loads(self.recipe.read_text(encoding='utf-8'));entry=recipe['assets'][0]
        overrides={'object_geometry_sha256':entry['object_geometry_sha256'],'object_asset':entry['object_asset']}
        recipe['native_alias_schema']=NATIVE_ALIAS_SCHEMA
        recipe['object_geometry_aliases']=[{'shared_asset_index':0,'overrides':overrides,'expanded_recipe_entry_sha256':digest(entry)}]
        self.recipe.write_bytes(canonical_bytes(recipe))
        output=self.root/'compact';bundle=prepare_curated_bundle(self.split,self.recipe,output,self.kin)
        read_curated_bundle_header(output/'bundle.json')
        bank,_,_=load_bundle(output/'bundle.json',self.kin)
        self.assertEqual(bank.num_frames,3)
        self.assertEqual(len(bundle['object_geometry_bindings']),1)

    def test_support_trajectories_live_only_in_bound_cache_and_missing_cache_fails(self):
        from tests.test_hoi_v1_scene_supports import source as table_source
        source=self.root/'clip0';metadata=json.loads((source/'metadata.json').read_text(encoding='utf-8'))
        metadata['scene_supports']=[table_source(positions=[[1.,2.,.3]]*3)]
        (source/'metadata.json').write_bytes(canonical_bytes(metadata))
        success=json.loads((source/'SUCCESS.json').read_text(encoding='utf-8'));success['metadata_sha256']=sha(source/'metadata.json')
        (source/'SUCCESS.json').write_bytes(canonical_bytes(success))
        metarows=json.loads((self.root/'metadata.json').read_text(encoding='utf-8'));metarows[0]['metadata_sha256']=success['metadata_sha256']
        metadata_path=self.root/'support_metadata.json';metadata_path.write_bytes(canonical_bytes(metarows))
        split=self.root/'support_split/split.json';write_split(self.selection,metadata_path,split.parent)
        output=self.root/'supports';bundle=prepare_curated_bundle(split,self.recipe,output,self.kin)
        summary=bundle['clips'][0]['scene_supports']
        self.assertEqual(summary['count'],1)
        self.assertEqual(summary['storage'],'reference_cache.source_scene_supports')
        self.assertNotIn('position',str(summary))
        bank,_,_=load_bundle(output/'bundle.json',self.kin)
        self.assertEqual(len(bank.scene_supports[0]),1)
        self.assertEqual(tuple(bank.bound_scene_supports[0][0]['position'].shape),(3,3))
        cache=bundle['clips'][0]['reference_cache'];reference=torch.load(cache['path'],weights_only=True)
        del reference['source_scene_supports'];torch.save(reference,cache['path'])
        with self.assertRaisesRegex(ValueError,'cache SHA256'):
            load_bundle(output/'bundle.json',self.kin)
        cache['sha256']=sha(cache['path'])
        bundle['bundle_content_sha256']=digest({k:v for k,v in bundle.items() if k!='bundle_content_sha256'})
        (output/'bundle.json').write_bytes(canonical_bytes(bundle))
        with self.assertRaisesRegex(ValueError,'raw scene-support cache is missing'):
            load_bundle(output/'bundle.json',self.kin)
        Path(cache['path']).unlink()
        with self.assertRaises(FileNotFoundError):
            load_bundle(output/'bundle.json',self.kin)

    def test_exact_native_aliases_keep_all_label_ids_and_reject_changed_physics(self):
        import copy
        second={'shapes':[{'kind':'cube','size':[.2,.2,.2]}],'basis':'second_source_identity_same_native_surface'}
        geometry=self.root/'second_geometry.json';geometry.write_bytes(canonical_bytes(second));label_hash=digest(second)
        source=self.root/'clip1';metadata=json.loads((source/'metadata.json').read_text(encoding='utf-8'))
        metadata['object_geometry_path']=str(geometry);metadata['object_geometry_sha256']=label_hash
        (source/'metadata.json').write_bytes(canonical_bytes(metadata))
        success=json.loads((source/'SUCCESS.json').read_text(encoding='utf-8'));success['metadata_sha256']=sha(source/'metadata.json')
        (source/'SUCCESS.json').write_bytes(canonical_bytes(success))
        metarows=json.loads((self.root/'metadata.json').read_text(encoding='utf-8'))
        metarows[1].update(object_geometry_path=str(geometry),object_geometry_sha256=label_hash,metadata_sha256=success['metadata_sha256'])
        metadata_path=self.root/'alias_metadata.json';metadata_path.write_bytes(canonical_bytes(metarows))
        split=self.root/'alias_split/split.json';write_split(self.selection,metadata_path,split.parent)
        recipe=json.loads(self.recipe.read_text(encoding='utf-8'));alias=copy.deepcopy(recipe['assets'][0]);alias['object_geometry_sha256']=label_hash
        alias['object_asset']='second_label_alias';recipe['assets'].append(alias);self.recipe.write_bytes(canonical_bytes(recipe))
        output=self.root/'aliases';bundle=prepare_curated_bundle(split,self.recipe,output,self.kin)
        self.assertEqual(len(bundle['assets']),1)
        self.assertEqual(len(bundle['object_geometry_bindings']),2)
        self.assertEqual(set(bundle['assets'][0]['object_geometry_aliases']),{self.geometry_sha,label_hash})
        evaluation=self.root/'alias_eval';eval_bundle=prepare_curated_bundle(split,self.recipe,evaluation,self.kin,partition='protected_eval')
        self.assertEqual(eval_bundle['clips'][0]['object_geometry_sha256'],label_hash)
        self.assertEqual(eval_bundle['clips'][0]['asset_id'],0)
        read_curated_bundle_header(evaluation/'bundle.json')
        original=copy.deepcopy(bundle);bundle['object_geometry_bindings'][1]['recipe_entry_sha256']='0'*64
        bundle['bundle_content_sha256']=digest({k:v for k,v in bundle.items() if k!='bundle_content_sha256'})
        (output/'bundle.json').write_bytes(canonical_bytes(bundle))
        with self.assertRaisesRegex(ValueError,'alias physics/geometry'):
            read_curated_bundle_header(output/'bundle.json')
        recipe['assets'][1]['physical_profile']['mass']=2.;self.recipe.write_bytes(canonical_bytes(recipe))
        original['recipes']['sha256']=sha(self.recipe)
        original['object_geometry_bindings'][1]['recipe_entry_sha256']=digest(recipe['assets'][1])
        original['bundle_content_sha256']=digest({k:v for k,v in original.items() if k!='bundle_content_sha256'})
        (output/'bundle.json').write_bytes(canonical_bytes(original))
        with self.assertRaisesRegex(ValueError,'alias physics/geometry'):
            read_curated_bundle_header(output/'bundle.json')
        separate=prepare_curated_bundle(split,self.recipe,self.root/'separate_physics',self.kin)
        self.assertEqual(len(separate['assets']),2)
        self.assertEqual(len({b['asset_id'] for b in separate['object_geometry_bindings']}),2)

    def test_native_validation_fixture_cannot_supply_training(self):
        _, parts = read_split(self.split)
        selected = [parts['train'][0]['clip_id']]
        output = self.root/'native_fixture'
        bundle = prepare_curated_bundle(self.split, self.recipe, output, self.kin, clip_ids=selected)
        self.assertEqual(bundle['scope']['purpose'], 'native_validation')
        read_curated_bundle_header(output/'bundle.json')
        with self.assertRaisesRegex(ValueError, 'only train partition'):
            load_bundle(output/'bundle.json', self.kin, require_train=True)
        bank, _, _ = load_bundle(output/'bundle.json', self.kin, require_train=False)
        self.assertEqual(bank.bound_scene_supports, [[]])
        with self.assertRaisesRegex(ValueError, 'declared partition'):
            prepare_curated_bundle(self.split, self.recipe, self.root/'bad_fixture', self.kin, clip_ids=['missing'])

    def test_portable_split_relocates_and_binds_exact_source_membership(self):
        import shutil
        relocated=self.root/'relocated'
        shutil.copytree(self.split.parent,relocated)
        header,parts=read_split(relocated/'split.json')
        self.assertEqual(len(parts['train']),1)
        self.assertEqual(len(parts['protected_eval']),1)
        # A self-consistent edited partition/hash cannot drop an included clip.
        partition=relocated/header['partitions']['protected_eval']['path']
        partition.write_text('',encoding='utf-8')
        header['partitions']['protected_eval']['count']=0
        header['partitions']['protected_eval']['sha256']=sha(partition)
        header['content_sha256']=digest({k:v for k,v in header.items() if k!='content_sha256'})
        (relocated/'split.json').write_bytes(canonical_bytes(header))
        with self.assertRaisesRegex(ValueError,'exact included selection'):
            read_split(relocated/'split.json')
        snapshot=self.split.parent/'source_metadata.json'
        snapshot.write_bytes(snapshot.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'metadata snapshot changed'):
            read_split(self.split)

    def test_explicit_source_path_maps_roundtrip_without_changing_source_identity(self):
        rows=[json.loads(x) for x in (self.selection/'included.jsonl').read_text(encoding='utf-8').splitlines()]
        meta_path=self.root/'metadata.json';metadata=json.loads(meta_path.read_text(encoding='utf-8'))
        for row in rows: row['output']='/remote_reference/'+Path(row['output']).name
        for row in metadata: row['output']='/remote_reference/'+Path(row['output']).name
        (self.selection/'included.jsonl').write_text(''.join(canonical_bytes(r).decode()+'\n' for r in rows),encoding='utf-8')
        manifest=json.loads((self.selection/'manifest.json').read_text(encoding='utf-8'))
        manifest['files']['included.jsonl']['sha256']=sha(self.selection/'included.jsonl')
        (self.selection/'manifest.json').write_bytes(canonical_bytes(manifest))
        (self.selection/'READY.json').write_bytes(canonical_bytes({'manifest_sha256':sha(self.selection/'manifest.json')}))
        meta_path.write_bytes(canonical_bytes(metadata))
        split=self.root/'mapped_split';write_split(self.selection,meta_path,split)
        output=self.root/'mapped_bundle'
        prepare_curated_bundle(split/'split.json',self.recipe,output,self.kin,path_maps=[('/remote_reference',self.root)])
        bank,_,bundle=load_bundle(output/'bundle.json',self.kin)
        self.assertEqual(bank.num_frames,3)
        self.assertEqual(bundle['clips'][0]['output'],'/remote_reference/clip0')
        self.assertEqual(Path(bundle['clips'][0]['resolved_source_output']),self.root/'clip0')

    def test_bound_source_integrity_and_schema(self):
        scanned=collect_metadata(self.selection,workers=2)
        self.assertEqual(len(scanned),2)
        self.assertTrue(all(r['robot_geometry_sha256']=='f'*64 for r in scanned))
        _,parts=read_split(self.split);record=parts['train'][0]
        clip=load_curated_clip(record)
        self.assertEqual(clip['reference']['metadata']['state_frames'],3)
        meta=Path(record['output'])/'metadata.json';meta.write_bytes(meta.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'metadata hash'):
            load_curated_clip(record)
        with self.assertRaisesRegex(ValueError,'metadata SHA'):
            collect_metadata(self.selection,workers=2)

    def test_recipe_binding_and_partition_hash_guard(self):
        recipe=json.loads(self.recipe.read_text(encoding="utf-8"));recipe['assets'][0]['object_geometry_sha256']='0'*64;self.recipe.write_bytes(canonical_bytes(recipe))
        with self.assertRaisesRegex(ValueError,'unambiguous native asset'):
            prepare_curated_bundle(self.split,self.recipe,self.root/'bad',self.kin)
        value,_=read_split(self.split);p=self.split.parent/value['partitions']['train']['path'];p.write_bytes(p.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'partition bytes'):
            read_split(self.split)


if __name__ == '__main__':
    unittest.main()
