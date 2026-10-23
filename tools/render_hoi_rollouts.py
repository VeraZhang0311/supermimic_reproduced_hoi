"""Offline rendering of frozen HOI rollouts. No inference, FK, or physics steps."""
from pathlib import Path
import argparse
import functools
import hashlib
import html
import json
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET

os.environ['MUJOCO_GL'] = 'glfw'
import mujoco
import numpy as np
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

KINDS = ('largebox', 'plasticbox', 'smallbox', 'suitcase')
IDS = (80, 84, 89, 94, 99)
BLUE = '.08 .43 1 .35'
W, H, FPS = 1280, 720, 50

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')

@functools.lru_cache(None)
def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/arial.ttf', size)

def object_mesh(source, destination):
    """Strict reader for the identity-transform flattened mesh snapshots in this run."""
    text = source.read_text(encoding='utf-8')
    assert len(re.findall(r'def Mesh\s', text)) == 1
    assert 'metersPerUnit = 1' in text and 'upAxis = "Z"' in text
    transforms = re.findall(r'^\s*(?:quatd|double3) (xformOp:\w+) = (\([^\n]+\))', text, re.M)
    assert len(transforms) == 3, transforms
    expected = {'xformOp:orient': (1, 0, 0, 0), 'xformOp:scale': (1, 1, 1), 'xformOp:translate': (0, 0, 0)}
    for name, value in transforms:
        assert tuple(float(x) for x in value.strip('()').split(',')) == expected[name]
    assert 'xformOp:transform' not in text and 'visibility = "invisible"' not in text
    def array(name, dtype):
        matches = re.findall(r'\b' + name + r' = \[([^\]]+)\]', text)
        assert len(matches) == 1, name
        return np.fromstring(re.sub(r'[(),]', ' ', matches[0]), sep=' ', dtype=dtype)
    points = array('points', np.float64).reshape(-1, 3)
    counts = array('faceVertexCounts', np.int64)
    indices = array('faceVertexIndices', np.int64).reshape(-1, 3)
    assert (counts == 3).all() and len(counts) == len(indices)
    assert np.isfinite(points).all() and indices.min() >= 0 and indices.max() < len(points)
    # Binary STL retains all triangles, without approximate boxes or hull substitutions.
    triangles = points[indices].astype(np.float32)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    entries = np.zeros(len(indices), dtype=[('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attr', '<u2')])
    entries['normal'] = normals
    entries['vertices'] = triangles
    with destination.open('wb') as stream:
        stream.write(b'Exact triangles from frozen object.usda'.ljust(80, b'\0'))
        stream.write(np.array([len(entries)], dtype='<u4').tobytes())
        stream.write(entries.tobytes())
    return {'source': str(source), 'source_sha256': sha(source), 'mesh_sha256': sha(destination),
            'vertices': len(points), 'triangles': len(indices), 'identity_transform_verified': True,
            'bounds_m': [points.min(0).tolist(), points.max(0).tolist()]}

def model_for(urdf, object_path, names):
    root = ET.Element('mujoco', model='recorded_hoi_overlay')
    ET.SubElement(root, 'compiler', angle='radian')
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'texture', name='sky', type='skybox', builtin='gradient', rgb1='.97 .98 1', rgb2='.78 .84 .92', width='256', height='256')
    ET.SubElement(asset, 'texture', name='floor', type='2d', builtin='checker', rgb1='.89 .91 .94', rgb2='.75 .79 .84', width='512', height='512')
    ET.SubElement(asset, 'material', name='floor_mat', texture='floor', texrepeat='2 2', reflectance='0')
    ET.SubElement(asset, 'mesh', name='object_mesh', file=object_path.resolve().as_posix())
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='0 0 .05', material='floor_mat', contype='0', conaffinity='0')
    ET.SubElement(world, 'light', pos='-3 -4 7', dir='.4 .5 -1', directional='true', diffuse='.75 .75 .75', ambient='.4 .4 .4')
    ET.SubElement(world, 'light', pos='3 4 5', dir='-.4 -.5 -1', directional='true', diffuse='.35 .35 .35')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth=str(W), offheight=str(H))
    ET.SubElement(visual, 'quality', shadowsize='1024', offsamples='1')
    source = ET.parse(urdf).getroot()
    visuals = []
    for link in source.findall('link'):
        name = link.get('name')
        for i, v in enumerate(link.findall('visual')):
            mesh = v.find('geometry/mesh')
            assert mesh is not None
            origin = v.find('origin')
            assert origin is not None and np.allclose(np.fromstring(origin.get('rpy'), sep=' '), 0)
            assert name in names
            mesh_name = 'mesh_' + name + '_' + str(i)
            file = (urdf.parent / mesh.get('filename')).resolve()
            ET.SubElement(asset, 'mesh', name=mesh_name, file=file.as_posix(), scale=mesh.get('scale', '1 1 1'))
            color = v.find('material/color')
            visuals.append((name, mesh_name, origin.get('xyz', '0 0 0'), color.get('rgba') if color is not None else '.65 .65 .65 1'))
    for prefix in ('actual', 'reference'):
        for name in names:
            body = ET.SubElement(world, 'body', name=prefix + '/' + name, mocap='true')
            for link, mesh_name, pos, color in visuals:
                if link == name:
                    ET.SubElement(body, 'geom', type='mesh', mesh=mesh_name, pos=pos,
                                  rgba=BLUE if prefix == 'reference' else color,
                                  contype='0', conaffinity='0')
        body = ET.SubElement(world, 'body', name=prefix + '/object', mocap='true')
        ET.SubElement(body, 'geom', type='mesh', mesh='object_mesh', rgba=BLUE if prefix == 'reference' else '.72 .36 .12 1', contype='0', conaffinity='0')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    model.vis.headlight.ambient[:] = .45
    model.vis.headlight.diffuse[:] = .55
    maps = {}
    for prefix in ('actual', 'reference'):
        maps[prefix] = np.array([model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + '/' + name)] for name in names])
        maps[prefix + '_object'] = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + '/object')]
    assert model.nq == 0 and model.nmocap == 2 * (len(names) + 1)
    return model, maps

def xyzw_to_wxyz(quat):
    return quat[..., [3, 0, 1, 2]]

def prepare(path, case):
    assert sha(path) == case['rollout_sha256']
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in ('physical_body_pose', 'physical_body_names', 'body_names', 'body_pos', 'root_p', 'root_quat', 'object_state', 'reference', 'time')}
    n = len(d['time'])
    assert n == case['control_steps'] + 1 and case['full_reference_completed']
    assert np.allclose(np.diff(d['time']), 1/FPS)
    names = d['physical_body_names'].tolist()
    order = [d['body_names'].tolist().index(x) for x in names]
    assert np.max(np.abs(d['body_pos'][:, order] - d['physical_body_pose'][:, :, :3])) < 2e-5
    d['actual_pos'] = d['physical_body_pose'][:, :, :3]
    d['actual_quat'] = xyzw_to_wxyz(d['physical_body_pose'][:, :, 3:7])
    d['reference_pos'] = d['reference'][:, 84:201].reshape(n, 39, 3)[:, order]
    d['reference_quat'] = xyzw_to_wxyz(d['reference'][:, 201:357].reshape(n, 39, 4)[:, order])
    d['reference_object'] = d['reference'][:, 71:78]
    for prefix in ('actual', 'reference'):
        assert np.isfinite(d[prefix + '_pos']).all()
        assert np.allclose(np.linalg.norm(d[prefix + '_quat'], axis=-1), 1, atol=1e-3)
    return d, names

def camera_for(d):
    positions = np.concatenate([d['actual_pos'].reshape(-1, 3), d['reference_pos'].reshape(-1, 3),
                                d['object_state'][:, :3], d['reference_object'][:, :3]])
    lo, hi = positions.min(0), positions.max(0)
    center = (lo + hi) / 2
    radius = np.linalg.norm((hi - lo) / 2) + .3
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = center
    cam.distance = max(3.7, radius / np.sin(np.deg2rad(21)) * 1.1)
    cam.azimuth = 135
    cam.elevation = -17
    return cam

def render_frame(renderer, model, data, maps, d, t, camera, case):
    for prefix in ('actual', 'reference'):
        data.mocap_pos[maps[prefix]] = d[prefix + '_pos'][t]
        data.mocap_quat[maps[prefix]] = d[prefix + '_quat'][t]
        obj = d['object_state'][t, :7] if prefix == 'actual' else d['reference_object'][t]
        data.mocap_pos[maps[prefix + '_object']] = obj[:3]
        data.mocap_quat[maps[prefix + '_object']] = xyzw_to_wxyz(obj[3:7])
    mujoco.mj_kinematics(model, data)
    renderer.update_scene(data, camera=camera)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    result = Image.fromarray(renderer.render())
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, W, 86), fill=(245, 248, 252))
    draw.text((22, 9), f'HOI zero-shot  |  {case["id"]}  |  rubber-hand collisions ON  |  checkpoint 14000', font=font(23), fill=(29, 43, 65))
    draw.rectangle((22, 51, 38, 67), fill=(95, 98, 103))
    draw.text((47, 46), 'ACTUAL', font=font(19), fill=(45, 52, 64))
    draw.rectangle((170, 51, 186, 67), fill=(45, 133, 255))
    draw.text((195, 46), 'REFERENCE - translucent blue', font=font(19), fill=(25, 105, 206))
    draw.text((965, 46), f'{d["time"][t]:.2f} / {d["time"][-1]:.2f} s', font=font(20), fill=(45, 52, 64))
    draw.rectangle((0, H-65, W, H), fill=(245, 248, 252))
    err = float(np.linalg.norm(d['object_state'][t, :3] - d['reference_object'][t, :3]))
    draw.text((22, H-57), f'Return {case["supermimic_norm_return"]:.4f}   |   PHC {"PASS" if case["phc_pass"] else "FAIL"}   |   Object pose {"PASS" if case["object_pose_pass"] else "FAIL"}', font=font(20), fill=(35, 49, 68))
    draw.text((820, H-57), f'Object position error: {err:.3f} m', font=font(20), fill=(35, 49, 68))
    draw.text((22, H-27), 'Full recorded trajectory, real-time 50 fps  |  Shared world coordinates and fixed camera  |  Solid orange = actual object', font=font(16), fill=(84, 100, 119))
    return result

def verify_video(ffmpeg, path, expected):
    result = subprocess.run([ffmpeg, '-v', 'error', '-i', str(path), '-f', 'null', '-', '-progress', 'pipe:1', '-nostats'], capture_output=True, text=True, check=True)
    counts = [int(x.split('=')[1]) for x in result.stdout.splitlines() if x.startswith('frame=')]
    assert counts and counts[-1] == expected, (path, counts[-1:] , expected)
    assert not result.stderr.strip(), result.stderr
    return counts[-1]

def gallery(out, clips):
    cards = []
    for c in clips:
        cards.append(f'<article><h2>{html.escape(c["id"])}</h2><video controls preload="none" poster="{c["preview"]}" src="{c["file"]}"></video><p>{c["frames"]} frames · {c["seconds"]:.2f}s · return {c["return"]:.4f} · PHC {"pass" if c["phc_pass"] else "fail"} · object {"pass" if c["object_pose_pass"] else "fail"}</p></article>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>HOI20 rubber-hand collision rollouts</title><style>body{font:16px system-ui;background:#eef2f7;color:#203049;margin:32px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:24px}article{background:white;padding:18px;border-radius:14px}video{width:100%}h1{margin-bottom:8px}p{color:#53637a}a{color:#1768cf}</style><h1>HOI20 · rubber-hand collisions ON · checkpoint 14000</h1><p>四类物体各5条：080 / 084 / 089 / 094 / 099。灰色机器人和橙色物体为实际轨迹；半透明蓝色为reference。共享世界坐标，完整50fps轨迹。</p><p><a href="all20.mp4">播放全部20条合集</a> · <a href="rendering.json">渲染记录与校验</a></p><main>' + ''.join(cards) + '</main>', encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out/'assets').mkdir(exist_ok=True)
    report = json.loads((args.input/'evaluation.json').read_text())
    selection = [next(c for c in report['cases'] if c['id'] == f'{kind}_{i:03d}') for kind in KINDS for i in IDS]
    dump(out/'selection.json', {'input': str(args.input), 'input_evaluation_sha256': sha(args.input/'evaluation.json'), 'checkpoint_sha256': report['checkpoint_sha256'], 'selection_rule': 'Uniformly spaced IDs over 080..099, one endpoint-inclusive five-point set per object type; no performance-based selection.', 'cases': selection})
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    urdf = args.assets/'g1_29dof_rev_1_0.urdf'
    assets, clips, previews = {}, [], []
    selected = [selection[i] for i in (0, 5, 10, 15)] if args.preview_only else selection
    for case in selected:
        source = args.input/case['rollout']
        mesh_path = out/'assets'/f'{case["object_asset"]}.stl'
        if case['object_asset'] not in assets:
            assets[case['object_asset']] = object_mesh(source.parent/'object.usda', mesh_path)
        d, names = prepare(source, case)
        model, maps = model_for(urdf, mesh_path, names)
        data, camera = mujoco.MjData(model), camera_for(d)
        renderer = mujoco.Renderer(model, H, W)
        try:
            n = len(d['time'])
            if args.preview_only:
                for t in (0, n//2, n-1):
                    img = render_frame(renderer, model, data, maps, d, t, camera, case)
                    previews.append(img.resize((640, 360)))
                print(json.dumps({'preview': case['id'], 'frames': n, 'camera_distance': camera.distance}), flush=True)
                continue
            target = out/f'{case["id"]}.mp4'
            assert not target.exists(), 'Existing videos must be preserved'
            process = subprocess.Popen([ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264', '-threads', '4', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)], stdin=subprocess.PIPE)
            started = time.time()
            pictures = []
            try:
                for t in range(n):
                    img = render_frame(renderer, model, data, maps, d, t, camera, case)
                    if t in (0, n//2, n-1):
                        pictures.append(img.copy())
                    process.stdin.write(np.asarray(img).tobytes())
                process.stdin.close()
                assert process.wait() == 0
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
            decoded = verify_video(ffmpeg, target, n)
            preview = out/f'{case["id"]}.jpg'
            pictures[1].save(preview, quality=92)
            contact = Image.new('RGB', (W, H*3))
            for i, img in enumerate(pictures):
                contact.paste(img, (0, i*H))
            contact.resize((960, 1620)).save(out/f'{case["id"]}_contact.jpg', quality=90)
            clips.append({'id': case['id'], 'file': target.name, 'preview': preview.name, 'frames': n, 'decoded_frames': decoded, 'fps': FPS, 'seconds': n/FPS, 'last_recorded_time_s': float(d['time'][-1]), 'full_rollout': True, 'return': case['supermimic_norm_return'], 'phc_pass': case['phc_pass'], 'object_pose_pass': case['object_pose_pass'], 'rollout_sha256': sha(source), 'video_sha256': sha(target), 'render_seconds': time.time()-started})
            dump(out/'progress.json', {'completed': len(clips), 'total': len(selection), 'clips': clips})
            print(json.dumps(clips[-1]), flush=True)
        finally:
            renderer.close()
    if args.preview_only:
        sheet = Image.new('RGB', (1920, 1440))
        for i, img in enumerate(previews):
            sheet.paste(img, ((i%3)*640, (i//3)*360))
        sheet.save(out/'preview_four_categories.jpg', quality=93)
        dump(out/'preview.json', {'assets': assets, 'render_method': 'Exact recorded physical_body_pose and raw reference body transforms, mocap visual meshes, mj_kinematics only; zero simulation steps.'})
        return
    playlist = out/'concat.txt'
    playlist.write_text(''.join("file '" + c['file'] + "'\n" for c in clips), encoding='utf-8')
    montage = out/'all20.mp4'
    assert not montage.exists()
    subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', str(playlist), '-c', 'copy', '-movflags', '+faststart', str(montage)], check=True)
    total_frames = sum(c['frames'] for c in clips)
    decoded = verify_video(ffmpeg, montage, total_frames)
    contact = Image.new('RGB', (1280, 180*5))
    for i, c in enumerate(clips):
        with Image.open(out/c['preview']) as img:
            contact.paste(img.resize((320, 180)), ((i//5)*320, (i%5)*180))
    contact.save(out/'contact_sheet.jpg', quality=94)
    result = {'status': 'rendered_and_fully_decode_verified', 'checkpoint_iteration': 14000, 'checkpoint_sha256': report['checkpoint_sha256'], 'evaluation_job': report['slurm_job_id'], 'renderer_version': mujoco.__version__, 'script_sha256': sha(__file__), 'robot_urdf_sha256': sha(urdf), 'selection': [c['id'] for c in clips], 'reference_rgba': [0.08, .43, 1, .35], 'actual_robot': 'original URDF visual colors', 'actual_object': 'solid orange', 'coordinates': 'Shared original world coordinates; fixed common camera per entire clip; no independent alignment or normalization.', 'state_source': 'Actual: saved physical_body_pose (xyzw). Reference: saved reference columns 84:201/201:357 in saved body_names order (xyzw). Object: saved object_state / reference 71:78. Quaternion conversion only.', 'replay': 'Mocap visual bodies; mj_kinematics only; no mj_step, policy inference or optimization. All T+1 state frames retained at 50fps.', 'object_meshes': assets, 'clips': clips, 'all20': {'file': montage.name, 'decoded_frames': decoded, 'seconds': total_frames/FPS, 'sha256': sha(montage)}}
    result['render_quality'] = {'width': W, 'height': H, 'samples': 1, 'shadows': False, 'floor_reflections': False, 'reason': 'Visual-only optimization for local Intel Arc rendering; exact meshes and all recorded frames retained.'}
    dump(out/'rendering.json', result)
    gallery(out, clips)
    print(json.dumps({'status': result['status'], 'clips': len(clips), 'seconds': total_frames/FPS, 'output': str(out)}), flush=True)

if __name__ == '__main__':
    main()
