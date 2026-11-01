"""Read-only collision-geometry preview of one actual native model dump.

No FK, physics, policy, pose fitting or reference resampling is performed.
The model supplies compiled geometry; all world transforms come from the dump.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bound_file(root, record):
    # Native outputs may be relocated as a complete diagnostic directory.
    path=root/Path(record['path']).name
    if sha(path)!=record['sha256']:
        raise ValueError('Native snapshot bound file hash differs: '+path.name)
    return path


def rotation_xyzw(q):
    import mujoco
    result=np.empty(9,dtype=np.float64)
    mujoco.mju_quat2Mat(result,np.asarray(q,dtype=np.float64)[[3,0,1,2]])
    return result.reshape(3,3)


def load_snapshot(path, environment=0):
    import mujoco
    import torch
    from .contacts import GROUND
    from .scene_context import validate_scene_state
    from .scene_supports import SUPPORT_IDS, SUPPORT_ROLES, omni_support_recipe, digest
    path=Path(path);root=path.parent;manifest=json.loads(path.read_text(encoding='utf8'))
    if manifest.get('schema')!='hoi_actual_native_model_dump_v1':
        raise ValueError('Expected an actual native model dump')
    scene=manifest.get('scene_snapshot',{})
    if (scene.get('schema')!='hoi_actual_native_scene_snapshot_v1' or scene.get('support_ids')!=list(SUPPORT_IDS) or
            scene.get('geometry_recipe')!=omni_support_recipe() or scene.get('geometry_recipe_sha256')!=digest(omni_support_recipe())):
        raise ValueError('Native dump lacks its bound observed/reset-held scene snapshot')
    binary=_bound_file(root,manifest['binary']);live=_bound_file(root,manifest['live_gpu_arrays'])
    model=mujoco.MjModel.from_binary_path(str(binary))
    arrays=torch.load(live,map_location='cpu',weights_only=True)
    actual=arrays['scene.support_pose_xyzw'];reference=arrays['scene.reference_support_pose_xyzw'];present=arrays['scene.support_present']
    origins=arrays['scene.environment_origins'];frames=arrays['scene.reference_frames']
    validate_scene_state(actual,present);validate_scene_state(reference,present)
    n=scene['num_envs']
    if (actual.shape!=(n,2,7) or origins.shape!=(n,3) or frames.shape!=(n,) or
            not bool(torch.isfinite(origins).all()) or not 0<=environment<n):
        raise ValueError('Saved environment/scene alignment differs')
    mapping=manifest['contact_mapping'];roles=np.asarray(mapping['geom_roles']);envs=np.asarray(mapping['geom_environment_ids'])
    if roles.shape!=(model.ngeom,) or envs.shape!=roles.shape or len(mapping['geom_names'])!=model.ngeom:
        raise ValueError('Compiled geom identity and contact mapping differ')
    for g,name in enumerate(mapping['geom_names']):
        # Unnamed geoms retain the recorder display fallback.
        actual_name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g)
        if actual_name is not None and actual_name!=name:
            raise ValueError('Actual compiled geom name differs from recorder mapping')
    pos=arrays['data.geom_xpos'].numpy();mat=arrays['data.geom_xmat'].numpy()
    if mat.shape[-1:]==(9,):mat=mat.reshape(mat.shape[:-1]+(3,3))
    topology=mapping['topology']
    if topology=='one_mujoco_world_per_environment':
        world=environment;selected=np.ones(model.ngeom,dtype=bool)
        worlds=n
    elif topology=='one_world_with_named_environments':
        world=0;selected=(envs==environment)|(roles==GROUND);worlds=1
    else:raise ValueError('Unknown native world topology')
    if pos.shape!=(worlds,model.ngeom,3) or mat.shape!=(worlds,model.ngeom,3,3):
        raise ValueError('Saved native geom transform array shapes differ')
    pos=pos[world];mat=mat[world]
    if not np.isfinite(pos).all() or not np.isfinite(mat).all() or not np.allclose(mat@mat.transpose(0,2,1),np.eye(3),atol=2e-4,rtol=0):
        raise ValueError('Saved native geom transforms are not finite rigid rotations')
    actual=actual[environment].numpy().copy();reference=reference[environment].numpy().copy();mask=present[environment].numpy().astype(bool)
    actual[:,:3]+=origins[environment].numpy();reference[:,:3]+=origins[environment].numpy()
    support_errors={}
    for slot,(sid,role) in enumerate(zip(SUPPORT_IDS,SUPPORT_ROLES)):
        ids=np.flatnonzero(selected & (roles==role))
        if sid in scene['declared_support_ids']:
            if len(ids)!=1:raise ValueError('Each declared support must map to one exact compiled geom')
            g=int(ids[0]);R=rotation_xyzw(actual[slot,3:]);local=rotation_xyzw(np.asarray(model.geom_quat[g])[[1,2,3,0]])
            ep=float(np.max(np.abs(pos[g]-(actual[slot,:3]+R@model.geom_pos[g]))))
            er=float(np.max(np.abs(mat[g]-R@local)))
            if ep>2e-5 or er>2e-5:
                raise ValueError(f'Stale native geom cache or support pose mismatch for {sid}: position={ep}, rotation={er}; no FK fallback')
            if int(model.geom_type[g])!=int(mujoco.mjtGeom.mjGEOM_BOX) or not np.allclose(model.geom_size[g]*2,scene['geometry_recipe']['size_m'],atol=1e-7,rtol=0):
                raise ValueError('Compiled support size/type differs from saved recipe')
            support_errors[sid]={'position_error_m':ep,'rotation_matrix_error':er}
        elif len(ids) or mask[slot]:raise ValueError('Undeclared support collider/presence in snapshot')
        if not mask[slot]:selected[ids]=False
    return {'model':model,'positions':pos,'rotations':mat,'selected':selected,'roles':roles,
        'reference_support_pose':reference,'present':mask,'support_ids':list(SUPPORT_IDS),
        'support_size':np.asarray(scene['geometry_recipe']['size_m']), 'support_errors':support_errors,
        'manifest':manifest,'manifest_sha256':sha(path),'binary_sha256':sha(binary),'live_arrays_sha256':sha(live),
        'environment':environment,'frame':int(frames[environment])}


def populate_scene(snapshot, scene, camera):
    """Fill the visual scene directly from saved transforms, never from qpos."""
    import mujoco
    from .contacts import GROUND, OBJECT
    from .scene_supports import SUPPORT_ROLES
    model=snapshot['model'];scene.ngeom=0
    def add(kind,size,pos,mat,rgba,dataid=-1,geom_id=-1):
        if scene.ngeom>=scene.maxgeom:raise ValueError('Native preview visual buffer is too small')
        g=scene.geoms[scene.ngeom];scene.ngeom+=1
        mujoco.mjv_initGeom(g,int(kind),np.asarray(size,dtype=np.float64),np.asarray(pos,dtype=np.float64),np.asarray(mat,dtype=np.float64).reshape(9),np.asarray(rgba,dtype=np.float32))
        g.dataid=dataid;g.objid=geom_id;g.objtype=int(mujoco.mjtObj.mjOBJ_GEOM) if geom_id>=0 else int(mujoco.mjtObj.mjOBJ_UNKNOWN)
        g.category=int(mujoco.mjtCatBit.mjCAT_DYNAMIC);g.segid=scene.ngeom-1;g.emission=.05;g.specular=.1
        return g
    bounds=[]
    for i in np.flatnonzero(snapshot['selected']):
        role=snapshot['roles'][i];kind=int(model.geom_type[i])
        color=(.7,.73,.78,1) if role==GROUND else (.94,.48,.12,1) if role==OBJECT else (.18,.65,.4,1) if role in SUPPORT_ROLES else (.4,.44,.5,1)
        dataid=int(model.geom_dataid[i]);dataid=2*dataid if kind==int(mujoco.mjtGeom.mjGEOM_MESH) else dataid
        add(kind,model.geom_size[i],snapshot['positions'][i],snapshot['rotations'][i],color,dataid,int(i))
        if role!=GROUND:
            radius=max(float(model.geom_rbound[i]),.02)
            bounds.extend((snapshot['positions'][i]-radius,snapshot['positions'][i]+radius))
    corners=np.asarray(list(itertools.product((-1.,1.),repeat=3)))
    for slot in np.flatnonzero(snapshot['present']):
        pose=snapshot['reference_support_pose'][slot];R=rotation_xyzw(pose[3:]);half=snapshot['support_size']/2
        add(mujoco.mjtGeom.mjGEOM_BOX,half,pose[:3],R,(.12,.48,.98,.18))
        vertices=(corners*half)@R.T+pose[:3];bounds.extend(vertices)
        for a,b in itertools.combinations(range(8),2):
            if np.count_nonzero(corners[a]!=corners[b])==1:
                g=add(mujoco.mjtGeom.mjGEOM_CAPSULE,(.0015,0.,0.),pose[:3],np.eye(3),(.15,.55,1.,.8))
                mujoco.mjv_connector(g,int(mujoco.mjtGeom.mjGEOM_CAPSULE),.0015,vertices[a],vertices[b])
    if not bounds:raise ValueError('No saved robot/object/support geometry to render')
    bounds=np.asarray(bounds);lo,hi=bounds.min(0),bounds.max(0)
    camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.lookat[:]=(lo+hi)/2
    camera.distance=max(1.2,float(np.linalg.norm(hi-lo))*1.65);camera.azimuth=125.;camera.elevation=-22.
    # Camera-only visualization does not evaluate kinematics or integration.
    mujoco.mjv_updateCamera(model,mujoco.MjData(model),camera,scene)
    scene.nlight=2
    for index,direction in enumerate(((-.3,-.5,-1.),(.5,.2,-1.))):
        light=scene.lights[index];light.pos[:]=camera.lookat+np.array([0.,0.,3.]);light.dir[:]=direction
        light.ambient[:]=.25;light.diffuse[:]=.65;light.specular[:]=.1;light.castshadow=False
        if hasattr(light,'type'):light.type=int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        else:light.directional=True
        light.attenuation[:]=[1.,0.,0.]
    scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False;scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
    return {'lookat':camera.lookat.tolist(),'distance':float(camera.distance),'azimuth':float(camera.azimuth),'elevation':float(camera.elevation)}


def render_snapshot(manifest, output, *, environment=0, width=1280, height=720):
    os.environ.setdefault('MUJOCO_GL','glfw' if os.name=='nt' else 'egl')
    import mujoco
    from PIL import Image,ImageDraw
    snapshot=load_snapshot(manifest,environment);model=snapshot['model'];output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    model.vis.global_.offwidth=max(model.vis.global_.offwidth,width);model.vis.global_.offheight=max(model.vis.global_.offheight,height)
    renderer=mujoco.Renderer(model,height,width,max_geom=model.ngeom+32)
    try:
        camera=populate_scene(snapshot,renderer.scene,mujoco.MjvCamera())
        image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
        draw.rectangle((0,0,width,45),fill=(243,246,251))
        draw.text((12,6),'Saved native collision geometry | robot: gray | object: orange | table: green | reset-held reference: blue',fill=(20,35,55))
        draw.text((12,24),f'Environment {environment} | saved frame {snapshot["frame"]} | 0 FK / 0 physics / 0 policy | shared world camera',fill=(20,35,55))
        image.save(output/'preview.png')
    finally:renderer.close()
    report={'schema':'hoi_actual_native_snapshot_render_v1','manifest_sha256':snapshot['manifest_sha256'],
        'binary_sha256':snapshot['binary_sha256'],'live_arrays_sha256':snapshot['live_arrays_sha256'],
        'preview_sha256':sha(output/'preview.png'),'renderer_sha256':sha(__file__),'environment':environment,
        'saved_frame':snapshot['frame'],'visible_support_ids':[sid for sid,visible in zip(snapshot['support_ids'],snapshot['present']) if visible],
        'support_geom_pose_errors':snapshot['support_errors'],'camera':camera,'physics_steps':0,'fk_calls':0,'policy_calls':0,
        'source':'actual saved compiled collision meshes/primitives and GPU geom_xpos/xmat; reference supports only from same saved reset-held poses',
        'scope':'native collision geometry visualization; no task-success or contact-force claim'}
    (output/'rendering.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--environment',type=int,default=0);parser.add_argument('--width',type=int,default=1280);parser.add_argument('--height',type=int,default=720)
    args=parser.parse_args();print(json.dumps(render_snapshot(args.manifest,args.output,environment=args.environment,width=args.width,height=args.height),indent=2))


if __name__=='__main__':main()
