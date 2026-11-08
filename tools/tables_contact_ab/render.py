"""Paired full-body and foot/table close-up movies from recorded poses only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import os
os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import display_library as display
from camera_library import camera_basis,object_radius


def fit(points,radii,width,height):
    p=np.concatenate(points);center=(p.min(0)+p.max(0))/2
    backward,right,up,tanv=camera_basis();tanh=tanv*width/height
    distance=1.
    for group,r in zip(points,radii):
        delta=group-center;v=delta@up
        distance=max(distance,float((delta@backward+r+np.maximum.reduce([
            (np.abs(delta@right)+r)/(tanh*.94),(v+r)/(tanv*.70),(-v+r)/(tanv*.94)])).max())*1.04)
    return {'lookat':center.tolist(),'distance':distance,'azimuth':135.,'elevation':-17.}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--campaign',type=Path,required=True);args=parser.parse_args()
    root=args.root;output=root/'videos';output.mkdir(exist_ok=False)
    result=json.loads((root/'comparison.json').read_text());assert result['passed']
    catalog=json.loads((args.campaign/'input/assets.json').read_text())['assets'][1]
    values={v:dict(np.load(root/v/'rollout.npz',allow_pickle=False)) for v in ('A','B')}
    depths={v:np.load(root/v/'penetration.npz')['control_depth_m'].max(1)*1000 for v in ('A','B')}
    for v,data in values.items():
        data['names']=list(map(str,data['names']))
        assert len(data['time'])==256 and data['actual_pos'].shape==(256,38,3)
    meshes,geometry=display.object_display(catalog,output/'object')
    urdf=args.campaign/'source_eval01/data/g1_hw3/converted.urdf';meshroot=urdf.parent
    models={};width,height=960,720
    for v in ('A','B'):
        m,maps,robot=display.display_model(urdf,meshes,values[v]['names'],width=width,height=height,verify_hands=False,mesh_root=meshroot)
        assert m.nq==m.nv==0
        models[v]=(m,mujoco.MjData(m),maps,mujoco.Renderer(m,height,width))
    radius=object_radius(meshes)
    body=np.concatenate([values[v][p+'_pos'].reshape(-1,3) for v in ('A','B') for p in ('actual','reference')])
    objects=np.concatenate([values[v][p+'_object_pos'] for v in ('A','B') for p in ('actual','reference')])
    whole=fit([body,objects],[.22,radius+.03],width,height)
    foot_ids=[values['A']['names'].index(n) for n in ('left_ankle_roll_link','right_ankle_roll_link')]
    foot=np.concatenate([values[v]['actual_pos'][:,foot_ids].reshape(-1,3) for v in ('A','B')])
    actual_objects=np.concatenate([values[v]['actual_object_pos'] for v in ('A','B')])
    close=fit([foot,actual_objects],[.17,radius+.025],width,height)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    reports=[]
    for view,params in (('full_body',whole),('foot_table_closeup',close)):
        path=output/f'{view}_AB.mp4'
        encoder=subprocess.Popen(['ffmpeg','-n','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x720','-r','50','-i','-',
            '-an','-c:v','libx264','-threads','3','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
        camera=mujoco.MjvCamera()
        camera.lookat[:]=params['lookat'];camera.distance=params['distance'];camera.azimuth=params['azimuth'];camera.elevation=params['elevation']
        for frame in range(256):
            canvas=Image.new('RGB',(1920,720))
            for col,v in enumerate(('A','B')):
                m,d,maps,renderer=models[v]
                display.apply_saved_frame(m,d,maps,values[v],frame)
                renderer.update_scene(d,camera=camera)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
                im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,width,92),fill=(245,247,251))
                tc=20 if v=='A' else 10
                draw.text((12,6),f'{v}: foot/table {tc}ms | checkpoint6000 | sub3_smalltable_027',font=font,fill=(20,40,65))
                draw.text((12,36),f't={frame*.02:.2f}s | frame depth={depths[v][frame]:.2f}mm (50Hz) | frame max={depths[v][:frame+1].max():.2f}mm',font=small,fill=(20,40,65))
                peak=result['variants'][v]['penetration']['physics_200hz']['max_depth_mm']
                draw.text((12,62),f'Original colours: actual; blue: reference | physics peak: {peak:.2f}mm (200Hz)',font=small,fill=(20,40,65))
                canvas.paste(im,(col*width,0))
            encoder.stdin.write(np.asarray(canvas).tobytes())
            if frame in (0,98,128,156,255):canvas.save(output/f'{view}_frame_{frame:04d}.png')
        encoder.stdin.close();assert encoder.wait()==0
        display.verify_decoded_frames('ffmpeg',path,256)
        reports.append({'path':path.name,'frames':256,'fps':50,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'camera':params})
    for m,d,maps,renderer in models.values():renderer.close()
    slow=output/'foot_table_closeup_AB_quarter_speed.mp4'
    subprocess.run(['ffmpeg','-n','-v','error','-i',str(output/'foot_table_closeup_AB.mp4'),'-vf','setpts=4*PTS','-r','50',
        '-an','-c:v','libx264','-threads','3','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(slow)],check=True)
    subprocess.run(['ffmpeg','-v','error','-i',str(slow),'-f','null','-'],check=True)
    (output/'rendering.json').write_text(json.dumps({'passed':True,'videos':reports,'quarter_speed':slow.name,
        'quarter_speed_sha256':hashlib.sha256(slow.read_bytes()).hexdigest(),'same_fixed_world_camera_per_view':True,
        'offline_policy_inferences':0,'offline_physics_steps':0,'actual_pose_fk_reconstruction':False,
        'robot':robot,'object':geometry},indent=2))
    print('CONTACT_AB_VIDEOS_PASS',flush=True)


if __name__=='__main__':main()
