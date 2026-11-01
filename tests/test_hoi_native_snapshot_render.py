"""Offline native snapshot software checks; fixture uses CPU MuJoCo only."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from hoi.v1.native_model_debug import export_actual_model
from hoi.v1.native_snapshot_render import load_snapshot, populate_scene, render_snapshot, sha
from hoi.v1.scene_context import empty_scene_state
from hoi.v1.contacts import GROUND, OBJECT, LEFT


def export_fixture(root):
    import mujoco
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    mesh=root/'tetra.obj';mesh.write_text('v 0 0 0\nv .1 0 0\nv 0 .2 0\nv 0 0 .3\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n')
    xml='<mujoco><asset><mesh name="mesh" file="'+mesh.as_posix()+'"/></asset><worldbody><geom name="ground" type="plane" size="4 4 .1"/><body name="robot" pos="0 0 .8"><freejoint/><geom name="robot_geom" type="mesh" mesh="mesh"/></body><body name="Object" pos=".3 0 .85"><freejoint/><geom name="object_geom" type="box" size=".1 .1 .1"/></body><body name="Support_table1" mocap="true" pos=".3 0 .74"><geom name="table_geom" type="box" size=".15 .125 .005"/></body></worldbody></mujoco>'
    spec=mujoco.MjSpec.from_string(xml);model=spec.compile();data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)  # Fixture preparation only; renderer is forbidden to call it.
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(model.ngeom)]
    mapping={'topology':'one_mujoco_world_per_environment','geom_names':names,'geom_roles':[GROUND,LEFT,OBJECT,20]}
    def field(x):return SimpleNamespace(numpy=lambda:np.asarray(x))
    saved_data=SimpleNamespace(geom_xpos=field(data.geom_xpos[None]),geom_xmat=field(data.geom_xmat.reshape(1,-1,3,3)),qpos=field(data.qpos[None]),qvel=field(data.qvel[None]))
    recorder=SimpleNamespace(solver=SimpleNamespace(mjw_data=saved_data,mjw_model=SimpleNamespace()),geom_envs=[-1,0,0,0],manifest=lambda:deepcopy(mapping))
    scene=empty_scene_state((1,));scene['support_present'][0,0]=True;scene['support_pose_xyzw'][0,0,:3]=torch.tensor([.3,0,.74])
    scene.update(reference_support_pose_xyzw=scene['support_pose_xyzw'].clone(),environment_origins=torch.zeros(1,3),reference_frames=torch.tensor([23]),declared_support_ids=['table1'])
    generated=root/'compiled_model.generated.xml';generated.write_text(spec.to_xml())
    export_actual_model(model,root,generated,recorder=recorder,scene_snapshot=scene)
    return root/'compiled_model_manifest.json'


class NativeSnapshotTests(unittest.TestCase):
    def temporary(self):
        temp=tempfile.TemporaryDirectory(prefix='hoi-native-preview-',dir='C:/Windows/Temp' if os.name=='nt' else None)
        self.addCleanup(temp.cleanup)
        return Path(temp.name)

    def test_smoke_capture_runs_once_after_existing_interval_and_before_autoreset(self):
        from hoi.v1.native import HOITrackingSim
        from hoi.v1.native_model_debug import enable_smoke_visual_capture
        events=[];env=HOITrackingSim.__new__(HOITrackingSim);env.asset={};env.cooking_only=False;env.active_pair_probe=None
        env.supports={};env.motion=SimpleNamespace(clip_end=torch.tensor([100]),clip_id=torch.zeros(101,dtype=torch.long))
        robot=SimpleNamespace(set_joint_position_target_index=lambda **kwargs:None)
        scene=SimpleNamespace(env_origins=torch.zeros(1,3),write_data_to_sim=lambda:events.append('write'),update=lambda dt:events.append('update'))
        env.env=SimpleNamespace(default_q=torch.zeros(1,29),xi=torch.zeros(1,33),alpha=.5,robot=robot,scene=scene,
            sim=SimpleNamespace(step=lambda **kwargs:events.append('existing_step')),f=torch.tensor([12]))
        env._overflow_gate=lambda:None;env._check_support_alignment=lambda:None
        S=(torch.zeros(1,3),torch.eye(3)[None],torch.zeros(1,3),torch.zeros(1,3),torch.zeros(1,29),torch.zeros(1,29))
        env.state=lambda:(S,S[:4]);record=empty_scene_state((1,));record['reference_support_pose_xyzw']=record['support_pose_xyzw'].clone()
        env.support_state=lambda:record
        def end():events.append('end_interval');return {'recorded':True}
        env.contacts=SimpleNamespace(solver=SimpleNamespace(mj_model=object()),begin_interval=lambda:events.append('begin'),end_interval=end)
        output=self.temporary()/'visual';saved=[]
        def export(model,destination,xml,**kwargs):
            events.append('capture');saved.append(kwargs['scene_snapshot'])
            (Path(destination)/'compiled_model_manifest.json').write_text('{}')
        status=enable_smoke_visual_capture(env,output,'generated.xml')
        self.assertFalse(status['captured']);self.assertEqual(events,[])
        with patch('hoi.v1.native_model_debug.export_actual_model',side_effect=export):
            self.assertEqual(env._advance(torch.zeros(1,29)),{'recorded':True})
            events.append('autoreset')
            env._advance(torch.zeros(1,29))
        self.assertEqual(events.count('existing_step'),2)
        self.assertEqual(events.count('capture'),1)
        self.assertLess(events.index('end_interval'),events.index('capture'))
        self.assertLess(events.index('capture'),events.index('autoreset'))
        self.assertEqual(saved[0]['reference_frames'].tolist(),[13])
        self.assertEqual(status['extra_physics_steps'],0);self.assertTrue(status['captured'])
        self.assertNotIn('_post_advance_capture',env.__dict__)

    def test_smoke_capture_rejects_cooking_probe_phase_and_overwrite(self):
        from hoi.v1.native_model_debug import enable_smoke_visual_capture
        for state in ({'cooking_only':True},{'asset':{'inertia_mode':'native_cooking_only_uniform_density'}},{'active_pair_probe':'hand_object'}):
            env=SimpleNamespace(asset={},cooking_only=False,active_pair_probe=None);env.__dict__.update(state)
            with self.assertRaises(ValueError):enable_smoke_visual_capture(env,self.temporary()/'visual','generated.xml')
        env=SimpleNamespace(asset={},cooking_only=False,active_pair_probe=None)
        with self.assertRaises(FileExistsError):enable_smoke_visual_capture(env,self.temporary(),'generated.xml')

    def test_snapshot_binds_support_pose_reference_presence_and_same_live_arrays(self):
        manifest=export_fixture(self.temporary()/'dump');snapshot=load_snapshot(manifest)
        self.assertEqual(snapshot['frame'],23);self.assertEqual(snapshot['present'].tolist(),[True,False])
        np.testing.assert_allclose(snapshot['reference_support_pose'][0,:3],[.3,0,.74],atol=1e-7)
        self.assertLess(snapshot['support_errors']['table1']['position_error_m'],1e-7)
        self.assertEqual(snapshot['manifest']['scene_snapshot']['reference_pose_source'],'saved sampled reset-frame support pose held for episode')
        self.assertIn('scene.reference_support_pose_xyzw',snapshot['manifest']['live_gpu_arrays']['fields'])

    def test_render_scene_uses_saved_geom_transforms_and_never_fk_or_physics(self):
        import mujoco
        snapshot=load_snapshot(export_fixture(self.temporary()/'dump'))
        # Editing only saved geom transforms changes the displayed robot directly;
        # the original compiled qpos0 remains elsewhere and must never be used.
        snapshot['positions'][1,0]+=.8
        scene=mujoco.MjvScene(snapshot['model'],maxgeom=64)
        with patch.object(mujoco,'mj_forward',side_effect=AssertionError('FK forbidden')), \
                patch.object(mujoco,'mj_kinematics',side_effect=AssertionError('FK forbidden')), \
                patch.object(mujoco,'mj_step',side_effect=AssertionError('physics forbidden')):
            populate_scene(snapshot,scene,mujoco.MjvCamera())
        np.testing.assert_allclose(scene.geoms[1].pos,snapshot['positions'][1],atol=1e-7)
        np.testing.assert_allclose(scene.geoms[1].mat,snapshot['rotations'][1],atol=1e-7)
        self.assertEqual(scene.geoms[1].dataid,2*int(snapshot['model'].geom_dataid[1]))
        self.assertEqual(scene.ngeom,snapshot['model'].ngeom+13)

    def test_stale_geom_cache_is_rejected_even_when_payload_hash_is_updated(self):
        manifest=export_fixture(self.temporary()/'dump');record=json.loads(manifest.read_text())
        arrays_path=manifest.parent/'compiled_model_live_arrays.pt';arrays=torch.load(arrays_path,weights_only=True)
        arrays['scene.support_pose_xyzw'][0,0,0]+=.01
        torch.save(arrays,arrays_path);record['live_gpu_arrays']['sha256']=sha(arrays_path)
        manifest.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'Stale native geom cache'):load_snapshot(manifest)

    def test_bound_binary_and_array_bytes_cannot_be_replaced(self):
        manifest=export_fixture(self.temporary()/'dump');binary=manifest.parent/'compiled_model.mjb'
        with binary.open('ab') as stream:stream.write(b'changed')
        with self.assertRaisesRegex(ValueError,'hash differs'):load_snapshot(manifest)

    def test_no_support_dump_is_explicit_and_missing_scene_is_not_guessed(self):
        manifest=export_fixture(self.temporary()/'dump');record=json.loads(manifest.read_text());record.pop('scene_snapshot')
        manifest.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'bound observed/reset-held'):load_snapshot(manifest)


if __name__=='__main__':unittest.main()
