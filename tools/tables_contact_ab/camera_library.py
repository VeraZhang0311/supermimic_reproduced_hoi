"""Offline videos of saved native hw3 link/object transforms, no rollout."""
import argparse
from functools import lru_cache
import hashlib
import html
import json
import math
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import display_library as display


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_radius(meshes):
    radius = 0.
    for mesh in meshes:
        raw = Path(mesh).read_bytes()
        triangles = np.frombuffer(raw, dtype=np.dtype([('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attr', '<u2')]), offset=84)
        radius = max(radius, float(np.linalg.norm(triangles['vertices'], axis=-1).max()))
    return radius


@lru_cache(maxsize=1)
def camera_basis():
    # Query the renderer's camera convention instead of reimplementing it.
    # The empty display scene has no physical or articulated coordinates.
    model = mujoco.MjModel.from_xml_string('<mujoco/>')
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 135., -17., 1.
    camera.lookat[:] = 0
    scene = mujoco.MjvScene(model, maxgeom=1)
    mujoco.mjv_updateScene(model, mujoco.MjData(model), mujoco.MjvOption(), None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    forward = np.array(scene.camera[0].forward, dtype=np.float64, copy=True)
    up = np.array(scene.camera[0].up, dtype=np.float64, copy=True)
    forward /= np.linalg.norm(forward)
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    tan_vertical = float(scene.camera[0].frustum_top / scene.camera[0].frustum_near)
    return -forward, right, up, tan_vertical


def fit_camera(data, radius):
    """Fit full trajectories to the 16:9 viewport, retaining one world camera."""
    bodies = np.concatenate([data[k].reshape(-1, 3) for k in ('actual_pos', 'reference_pos')])
    objects = np.concatenate([data[k] for k in ('actual_object_pos', 'reference_object_pos')])
    positions = np.concatenate([bodies, objects])
    center = (positions.min(0) + positions.max(0)) / 2
    backward, right, up, tan_vertical = camera_basis()
    tan_horizontal = tan_vertical * 1280 / 720
    distance = 2.7
    bounds = []
    for points, margin in ((bodies, .22), (objects, radius + .03)):
        relative = points - center
        vertical = relative @ up
        required = relative @ backward + margin + np.maximum.reduce([
            (np.abs(relative @ right) + margin) / (tan_horizontal * .97),
            (vertical + margin) / (tan_vertical * .72),  # keep the 92px header clear
            (-vertical + margin) / (tan_vertical * .97)])
        distance = max(distance, float(required.max()) * 1.03)
        bounds.append((relative, margin))
    for relative, margin in bounds:
        depth = distance - relative @ backward - margin
        assert (depth > 0).all()
        assert ((np.abs(relative @ right) + margin) / depth / tan_horizontal < .98).all()
        assert ((relative @ up + margin) / depth / tan_vertical < .74).all()
        assert ((-relative @ up + margin) / depth / tan_vertical < .98).all()
    return {'lookat': center.tolist(), 'distance': distance, 'azimuth': 135., 'elevation': -17.,
            'shared_world': True, 'trajectory_alignment': 'none', 'fit': 'native_mjv_camera_viewport_extents_v4',
            'object_display_radius_m': radius, 'all_frame_projection_bounds_passed': True,
            'camera_basis_source': 'mujoco.mjv_updateScene', 'header_pixels_reserved': 92,
            'native_camera_backward': backward.tolist(), 'native_camera_up': up.tolist()}


def concat(paths, target):
    listing = target.with_suffix('.concat.txt')
    listing.write_text(''.join("file '" + str(p.resolve()) + "'\n" for p in paths))
    subprocess.run(['ffmpeg', '-n', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', str(listing),
                    '-c', 'copy', '-movflags', '+faststart', str(target)], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--mesh-root', type=Path, required=True)
    parser.add_argument('--video-subdir', default='videos')
    args = parser.parse_args()
    output = args.root / args.video_subdir
    output.mkdir(exist_ok=False)
    catalog = json.loads(args.assets.read_text())['assets']
    complete = json.loads((args.root / 'result.json').read_text())
    assert complete['clip_count'] == 22 and complete['control_steps'] == 6482
    original_path = args.root.parent / 'protected_eval_2000_01/iter_006000/evaluation/result.json'
    original = json.loads(original_path.read_text())
    assert complete['iteration'] == original['iteration'] == 6000
    for key in ('checkpoint_sha256', 'model_state_sha256', 'input_sha256', 'expected_clip_ids', 'source_identity'):
        assert complete[key] == original[key], key
    paths = sorted((args.root / 'recordings').glob('asset_*/*.npz'))
    assert len(paths) == 22
    reports, models = [], {}
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 20)
    small = ImageFont.truetype(font_path, 17)
    expected_ids = set(complete['expected_clip_ids'])
    assert len({Path(clip).name for clip in expected_ids}) == 22
    assert {p.stem for p in paths} == {Path(clip).name for clip in expected_ids}
    assert {json.loads(p.with_suffix('.json').read_text())['metrics']['clip_id'] for p in paths} == expected_ids
    for index, path in enumerate(paths):
        started = time.monotonic()
        record = json.loads(path.with_suffix('.json').read_text())
        assert record['iteration'] == 6000
        assert record['file_sha256'] == sha(path)
        data = dict(np.load(path, allow_pickle=False))
        data['names'] = list(map(str, data['names']))
        n, T = len(data['time']), len(data['action'])
        assert n == T + 1 == record['frames']
        assert np.allclose(data['time'], np.arange(n) / 50, atol=1e-10, rtol=0)
        assert data['actual_pos'].shape == (n, len(data['names']), 3)
        for key in ('actual_quat', 'reference_quat', 'actual_object_quat', 'reference_object_quat'):
            assert np.allclose(np.linalg.norm(data[key], axis=-1), 1, atol=1e-3, rtol=0)
        asset_index = int(path.parent.name[-1])
        if asset_index not in models:
            meshes, geometry_record = display.object_display(catalog[asset_index], output / f'object_{asset_index}')
            model, maps, robot_record = display.display_model(args.urdf, meshes, data['names'], width=1280, height=720,
                                                             verify_hands=False, mesh_root=args.mesh_root)
            assert model.nq == model.nv == 0
            models[asset_index] = model, maps, robot_record, geometry_record, object_radius(meshes)
        model, maps, robot_record, geometry_record, radius = models[asset_index]
        d, camera = mujoco.MjData(model), mujoco.MjvCamera()
        parameters = fit_camera(data, radius)
        camera.lookat[:] = parameters['lookat']
        camera.distance = parameters['distance']
        camera.azimuth, camera.elevation = parameters['azimuth'], parameters['elevation']
        renderer = mujoco.Renderer(model, 720, 1280)
        target = output / (path.stem + '.mp4')
        encoder = subprocess.Popen(['ffmpeg', '-n', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                                    '-s', '1280x720', '-r', '50', '-i', '-', '-an', '-c:v', 'libx264',
                                    '-threads', '3', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
                                    '-movflags', '+faststart', str(target)], stdin=subprocess.PIPE)
        chosen, previews = {0, n // 2, n - 1}, []
        failed_at = record['metrics']['first_failure_step']
        try:
            for frame in range(n):
                display.apply_saved_frame(model, d, maps, data, frame)
                renderer.update_scene(d, camera=camera)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 1280, 92), fill=(244, 247, 252))
                draw.text((14, 6), f'TABLES | checkpoint 6000 | {index+1:02d}/22 | {path.stem}', font=font, fill=(20, 40, 65))
                draw.text((14, 34), 'ACTUAL: robot original colours / orange table    REFERENCE: translucent blue', font=small, fill=(20, 40, 65))
                status = 'No failure yet' if failed_at is None or frame < failed_at else f'Failure triggered at {failed_at * .02:.2f}s; rollout continues'
                draw.text((14, 61), f't={data["time"][frame]:.2f}/{data["time"][-1]:.2f}s | {status} | fixed world camera, real time',
                          font=small, fill=(145, 35, 35) if failed_at is not None and frame >= failed_at else (20, 70, 50))
                if frame in chosen:
                    preview = output / (path.stem + f'_frame_{frame:04d}.png')
                    image.save(preview)
                    previews.append(preview.name)
                encoder.stdin.write(np.asarray(image).tobytes())
            encoder.stdin.close()
            assert encoder.wait() == 0
        finally:
            renderer.close()
            if encoder.poll() is None:
                encoder.kill()
                encoder.wait()
        decoded = display.verify_decoded_frames('ffmpeg', target, n)
        report = {'clip_id': record['metrics']['clip_id'], 'asset': catalog[asset_index]['object_asset'], 'video': target.name,
                  'video_sha256': sha(target), 'rollout_sha256': record['file_sha256'],
                  'state_frames': n, 'control_steps': T, 'decoded_frames': decoded,
                  'fps': 50, 'video_seconds': n / 50, 'metrics': record['metrics'],
                  'previews': previews, 'camera': parameters, 'robot': robot_record, 'object': geometry_record,
                  'physics_steps': 0, 'policy_inferences': 0, 'actual_pose_fk_reconstructions': 0,
                  'render_seconds': time.monotonic() - started}
        reports.append(report)
        print(json.dumps({'phase': 'video_complete', 'clip': path.stem, 'frames': decoded,
                          'render_seconds': report['render_seconds']}), flush=True)
    for name, clips in [('all22', reports), ('largetable', reports[:11]), ('smalltable', reports[11:])]:
        target = output / ('tables_ckpt6000_' + name + '.mp4')
        concat([output / clip['video'] for clip in clips], target)
        display.verify_decoded_frames('ffmpeg', target, sum(clip['decoded_frames'] for clip in clips))
    assert sum(r['state_frames'] for r in reports) == 6504
    assert sum(r['control_steps'] for r in reports) == 6482
    original_path = args.root.parent / 'protected_eval_2000_01/iter_006000/evaluation/result.json'
    original = json.loads(original_path.read_text())
    final = {'schema': 'tables_ckpt6000_all22_native_videos_v1', 'iteration': 6000, 'clip_count': 22,
             'state_frames': 6504, 'control_steps': 6482, 'fps': 50, 'all_videos_full_decode_verified': True,
             'recording_evaluation_sha256': sha(args.root / 'result.json'), 'original_evaluation_sha256': sha(original_path),
             'checkpoint_sha256': complete['checkpoint_sha256'], 'original_metrics': original['metrics'],
             'recording_metrics': complete['metrics'], 'clips': reports,
             'display_source_sha256': sha(display.__file__), 'render_script_sha256': sha(__file__),
             'protocol': 'same frozen full22 evaluator, seed0, deterministic policy, nominal physics; separate visualization rerun',
             'training_modified': False, 'wandb_uploads': False}
    (output / 'rendering.json').write_text(json.dumps(final, indent=2))
    cards = []
    for report in reports:
        clip = html.escape(report['clip_id'])
        cards.append(f'<article><h2>{clip}</h2><video controls preload="none" width="960" poster="{report["previews"][1]}" src="{report["video"]}"></video></article>')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Tables checkpoint 6000 — 22 protected clips</title><style>body{background:#eef1f5;color:#142841;font-family:sans-serif;max-width:1100px;margin:32px auto}video{max-width:100%}article{padding:20px;background:white;margin:20px 0;border-radius:12px}</style><h1>Tables checkpoint 6000 · all 22 protected clips</h1><p>50 fps, real time, complete trajectories. Actual: original robot / orange table. Reference: translucent blue. Fixed shared world camera.</p><p>Visualization rerun of the frozen checkpoint using the original evaluation protocol. Original scheduled metrics are preserved.</p><video controls width="1080" src="tables_ckpt6000_all22.mp4"></video>' + ''.join(cards), encoding='utf-8')
    print('TABLES_ALL22_VIDEOS_FULL_DECODE_PASS', flush=True)


if __name__ == '__main__':
    main()
