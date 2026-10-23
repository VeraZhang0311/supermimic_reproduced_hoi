"""Render actual recorded policy states beside synchronized references; no simulation steps."""
from pathlib import Path
import argparse
import ast
import functools
import hashlib
import json
import os
import subprocess
import xml.etree.ElementTree as ET

os.environ['MUJOCO_GL']='glfw'
import mujoco
import numpy as np
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
@functools.lru_cache(maxsize=16)
def font(size): return ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size)

def make_model(assets):
    root=ET.parse(assets/'g1_29dof.xml').getroot()
    root.find('compiler').set('meshdir',assets.resolve().as_posix())
    asset=root.find('asset')
    ET.SubElement(asset,'texture',name='video_floor_tex',type='2d',builtin='checker',rgb1='.88 .90 .93',rgb2='.73 .77 .82',width='512',height='512')
    ET.SubElement(asset,'material',name='video_floor_mat',texture='video_floor_tex',texrepeat='3 3',reflectance='.05')
    ET.SubElement(asset,'texture',name='video_sky',type='skybox',builtin='gradient',rgb1='.96 .97 .99',rgb2='.77 .82 .89',width='256',height='256')
    world=root.find('worldbody')
    ET.SubElement(world,'geom',name='video_floor',type='plane',size='0 0 .05',material='video_floor_mat',contype='0',conaffinity='0',group='2')
    ET.SubElement(world,'light',name='video_light',pos='0 -3 5',dir='0 0 -1',diffuse='.8 .8 .8',ambient='.3 .3 .3',directional='true')
    visual=root.find('visual')
    if visual is None: visual=ET.SubElement(root,'visual')
    old=visual.find('global')
    if old is not None: visual.remove(old)
    ET.SubElement(visual,'global',offwidth='640',offheight='480')
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    assert model.nq==36
    model.vis.headlight.ambient[:]=.5
    model.vis.headlight.diffuse[:]=.6
    return model

def joint_names(source):
    tree=ast.parse(source.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='G1_JOINT_NAMES' for t in n.targets))
    return ast.literal_eval(node.value)

def encode_frame(renderer,model,data,camera,option,p,q,quat,joint_addresses):
    data.qpos[:3]=p;data.qpos[3:7]=quat;data.qpos[joint_addresses]=q
    mujoco.mj_kinematics(model,data)
    camera.lookat[:]=[float(p[0]),float(p[1]),.72]
    renderer.update_scene(data,camera=camera,scene_option=option)
    return Image.fromarray(renderer.render())

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=Path('D:/supermimic_reproduced_hoi/supermimic/robot.py'))
    parser.add_argument('--preview-only',action='store_true')
    args=parser.parse_args();work=args.work
    assets=work/'data/g1_description'
    model=make_model(assets);data=mujoco.MjData(model)
    names=joint_names(args.source)
    ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]
    assert len(set(ids))==29 and min(ids)>=0
    addresses=model.jnt_qposadr[ids]
    renderer=mujoco.Renderer(model,480,640)
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    camera.distance=3.25;camera.azimuth=135;camera.elevation=-13
    option=mujoco.MjvOption()
    if np.any(model.geom_group==1): option.geomgroup[0]=0
    option.geomgroup[1]=1;option.geomgroup[2]=1
    out=work/'videos';out.mkdir(exist_ok=True)
    if args.preview_only:
        q=np.zeros(29)
        for i,n in enumerate(names):
            if 'hip_pitch' in n:q[i]=-.312
            if 'knee' in n:q[i]=.669
            if 'ankle_pitch' in n:q[i]=-.363
            if 'elbow' in n:q[i]=.6
        encode_frame(renderer,model,data,camera,option,[0,0,.76],q,[1,0,0,0],addresses).save(work/'renderer_test.png')
        print(json.dumps({'mujoco':mujoco.__version__,'joint_mapping_valid':True,'geom_groups':{int(g):int((model.geom_group==g).sum()) for g in np.unique(model.geom_group)},'preview':str(work/'renderer_test.png')}))
        renderer.close();return
    record=json.loads((work/'recording.json').read_text())
    assert record['status']=='actual_policy_rollouts_recorded' and record['optimizer_updates']==0
    assert sha(work/'selected5_rollout.npz')==record['files_sha256']['selected5_rollout.npz']
    with np.load(work/'selected5_rollout.npz',allow_pickle=False) as z: d={k:z[k] for k in z.files}
    clip_names=d['names'].tolist()
    assert len(clip_names)==5
    assert all(np.isfinite(d[k]).all() for k in ('root_p','root_quat','q','ref_root_p','ref_root_R','ref_q'))
    assert np.allclose(np.linalg.norm(d['root_quat'],axis=-1),1,atol=1e-3)
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    labels={'dance1_subject1':'Dance','jumps1_subject2':'Jumps','fight1_subject3':'Fight','walk3_subject1':'Walk','fallAndGetUp1_subject4':'Fall and get up'}
    rendered=[]
    try:
        for c,name in enumerate(clip_names):
            meta=next(v for v in record['clips'] if v['name']==name)
            live=d['alive'][:,c].astype(bool)
            count=int(live.sum())
            assert count>0 and live[:count].all() and not live[count:].any()
            failed=meta['failed_in_recording']
            end=min(len(live),count+50) if failed else min(count,1000)
            start=max(0,end-1000) if failed else 0
            indices=np.arange(start,end,2)
            video=out/f'{c+1:02d}_{name}.mp4'
            assert not video.exists(),'Preserve existing video'
            process=subprocess.Popen([ffmpeg,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x640','-r','25','-i','-',
                '-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
            sheet=[]
            try:
                for frame_number,t in enumerate(indices):
                    robot_t=min(int(t),count) if count<len(live) else int(t)
                    f=int(d['f'][t,c])
                    ref_quat=np.empty(4)
                    mujoco.mju_mat2Quat(ref_quat,d['ref_root_R'][f].astype(np.float64).reshape(-1))
                    canvas=Image.new('RGB',(1280,640),(245,247,251))
                    canvas.paste(encode_frame(renderer,model,data,camera,option,d['ref_root_p'][f],d['ref_q'][f],ref_quat,addresses),(0,88))
                    canvas.paste(encode_frame(renderer,model,data,camera,option,d['root_p'][robot_t,c],d['q'][robot_t,c],d['root_quat'][robot_t,c],addresses),(640,88))
                    draw=ImageDraw.Draw(canvas)
                    draw.text((18,8),'SuperMimic baseline 1  |  checkpoint 8000  |  self-collision OFF',font=font(23),fill=(30,43,65))
                    draw.text((18,42),'REFERENCE',font=font(24),fill=(29,112,190))
                    draw.text((658,42),'ACTUAL POLICY ROLLOUT',font=font(24),fill=(172,92,22))
                    sec=(f-int(d['clip_start'][c]))*.02
                    draw.text((200,46),f'{labels[name]}  /  {name}',font=font(17),fill=(51,65,85))
                    draw.text((1020,46),f't = {sec:.2f} s',font=font(21),fill=(51,65,85))
                    draw.line((640,88,640,568),fill=(205,212,223),width=2)
                    xy=float(np.linalg.norm(d['root_p'][robot_t,c,:2]-d['ref_root_p'][f,:2]))
                    draw.text((18,578),f'Excerpt {start*.02:.1f}-{(end-1)*.02:.1f}s  /  reference {meta["reference_seconds"]:.1f}s',font=font(19),fill=(51,65,85))
                    draw.text((658,578),f'Root XY error: {xy:.3f} m  |  original clip score: {meta["original_norm_return"]:.4f}',font=font(19),fill=(51,65,85))
                    draw.text((18,608),'Recorded at 50 Hz; playback 25 fps, real time. Root-follow cameras; XY drift is shown numerically.',font=font(17),fill=(80,95,115))
                    if t>=count:
                        draw.rectangle((650,94,1270,132),fill=(160,40,44))
                        draw.text((662,101),'TERMINATED - holding last recorded pose',font=font(21),fill='white')
                    if frame_number in {0,len(indices)//2,len(indices)-1}:
                        sheet.append(canvas.copy())
                    process.stdin.write(np.asarray(canvas).tobytes())
                process.stdin.close()
                assert process.wait()==0,'ffmpeg encode failed'
            finally:
                if process.poll() is None:process.kill();process.wait()
            # Decode the whole video and compare the frame count, not just the container header.
            verify=subprocess.run([ffmpeg,'-v','error','-i',str(video),'-f','null','-','-progress','pipe:1','-nostats'],capture_output=True,text=True,check=True)
            counts=[int(line.split('=')[1]) for line in verify.stdout.splitlines() if line.startswith('frame=')]
            assert counts and counts[-1]==len(indices)
            contact=Image.new('RGB',(1280,640*len(sheet)))
            for i,picture in enumerate(sheet):contact.paste(picture,(0,i*640))
            contact.save(out/f'{c+1:02d}_{name}_contact.png')
            sheet[0].save(out/f'{c+1:02d}_{name}_preview.png')
            entry={'clip':name,'file':video.name,'frames':len(indices),'fps':25,'seconds':len(indices)/25,
                'start_seconds':start*.02,'end_seconds':(end-1)*.02,'failed_in_original_eval':meta['failed_in_original_eval'],
                'failed_in_recording':failed,'original_clip_score':meta['original_norm_return'],
                'replayed_clip_score':meta['replayed_norm_return'],'sha256':sha(video),'decoded_frames':counts[-1]}
            rendered.append(entry)
            print(json.dumps(entry),flush=True)
    finally:renderer.close()
    playlist=out/'concat.txt'
    playlist.write_text(''.join("file '"+str((out/x['file']).resolve()).replace('\\','/')+"'\n" for x in rendered))
    montage=out/'baseline1_eval8000_five_actions.mp4'
    subprocess.run([ffmpeg,'-y','-loglevel','error','-f','concat','-safe','0','-i',str(playlist),'-c','copy','-movflags','+faststart',str(montage)],check=True)
    report={'checkpoint_iteration':8000,'checkpoint_sha256':record['checkpoint_sha256'],
        'recorded_rollout_sha256':record['files_sha256']['selected5_rollout.npz'],
        'rendering_only':'MuJoCo mj_kinematics; no mj_step; all policy poses come from recorded simulation',
        'camera':'Each panel follows its own root XY, same angles/distance; global XY difference reported numerically',
        'original_full40':record['original_full40'],'replayed_full40':record['replayed_full40'],
        'renderer_mujoco_version':mujoco.__version__,'clips':rendered,
        'montage':montage.name,'montage_sha256':sha(montage),'optimizer_updates':0}
    (out/'videos.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'rendered_and_decode_verified','montage':str(montage),'clips':len(rendered)}))

if __name__=='__main__':main()
