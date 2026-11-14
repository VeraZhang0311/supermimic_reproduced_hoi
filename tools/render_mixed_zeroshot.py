"""Render every native 50 Hz frame of a standalone protected tables20 evaluation."""
import argparse
import html
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import display_library as display
from render_tables import concat, fit_camera, object_radius
from hoi_tables.mixed_zeroshot import validate_result
from hoi_tables.trajectories import sha256


def render_clip(path, clip, output, model, maps, radius, number, fonts, title):
    if sha256(path) != clip["trajectory_sha256"]:
        raise ValueError("Table recording checksum differs")
    with np.load(path, allow_pickle=False) as recording:
        data = dict(recording)
    data["names"] = list(map(str, data["names"]))
    frames = clip["state_frames"]
    assert frames == len(data["time"]) == len(data["action"]) + 1
    np.testing.assert_array_equal(data["reference_frame"], np.arange(clip["start_frame"], clip["final_frame"] + 1))
    np.testing.assert_allclose(data["time"], np.arange(frames) / 50, atol=1e-10, rtol=0)
    for key in ("actual_quat", "reference_quat", "actual_object_quat", "reference_object_quat"):
        np.testing.assert_allclose(np.linalg.norm(data[key], axis=-1), 1, atol=1e-3, rtol=0)
    settings = fit_camera(data, radius)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = settings["lookat"]
    camera.distance, camera.azimuth, camera.elevation = (settings[key] for key in ("distance", "azimuth", "elevation"))
    scene = mujoco.MjData(model)
    name = Path(clip["clip_id"]).name
    target, poster = output / f"{name}.mp4", output / f"{name}.jpg"
    encoder = subprocess.Popen(["ffmpeg", "-n", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                "-s", "1280x720", "-r", "50", "-i", "-", "-an", "-c:v", "libx264",
                                "-threads", "3", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                                "-movflags", "+faststart", str(target)], stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(model, 720, 1280) as renderer:
            for frame in range(frames):
                display.apply_saved_frame(model, scene, maps, data, frame)
                renderer.update_scene(scene, camera=camera)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 1280, 92), fill=(244, 247, 252))
                draw.text((14, 6), f"{title} | {number:02d}/20 | {name} | {clip['action']}",
                          font=fonts[0], fill=(20, 40, 65))
                draw.text((14, 34), "POLICY: robot / orange table    REFERENCE: translucent blue",
                          font=fonts[1], fill=(20, 40, 65))
                first = clip["first_failure_step"]
                failed = first is not None and frame >= first
                status = f"Failure at {first/50:.2f}s" if failed else "No failure trigger yet"
                draw.text((14, 61), f"t={frame/50:.2f}/{(frames-1)/50:.2f}s | {status} | 1x",
                          font=fonts[1], fill=(145, 35, 35) if failed else (20, 70, 50))
                if frame == frames // 2:
                    image.save(poster, quality=92)
                encoder.stdin.write(image.tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError(f"Video encoding failed for {name}")
    finally:
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
    decoded = display.verify_decoded_frames("ffmpeg", target, frames)
    return {"clip_id": clip["clip_id"], "asset": clip["asset"], "action": clip["action"], "video": target.name,
            "poster": poster.name, "video_sha256": sha256(target), "frames": decoded, "fps": 50,
            "original_failure": None, "replay_failure": clip["failed"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("original", "replay", "output", "assets", "urdf"):
        parser.add_argument("--" + argument, type=Path, required=True)
    args = parser.parse_args()
    original, replay = (json.loads((root / "request.json").read_text()) for root in (args.original, args.replay))
    original_path, replay_path = (Path(q["output"]) / "result.json" for q in (original, replay))
    validate_result(original_path, original)
    validate_result(replay_path, replay)
    assert original["checkpoint_sha256"] == replay["checkpoint_sha256"]
    assert original["table_manifest_sha256"] == replay["table_manifest_sha256"]
    assert original["iteration"] == replay["iteration"]
    iteration = original["iteration"]
    title = "Ultra 8000 zero-shot" if iteration == 8000 else f"Ultra+Tables {iteration}"
    originals, clips = (json.loads(p.read_text())["clips"] for p in (original_path, replay_path))
    by_id = {row["clip_id"]: row for row in originals}
    catalog = {row["object_asset"]: row for row in json.loads(args.assets.read_text())["assets"]}
    fonts = tuple(ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size) for size in (20, 17))
    args.output.mkdir(exist_ok=False)
    reports, models = [], {}
    for index, clip in enumerate(clips, 1):
        prior = by_id[clip["clip_id"]]
        assert all(prior[k] == clip[k] for k in ("asset", "action", "control_steps", "state_frames"))
        asset = clip["asset"]
        path = replay_path.parent / "tables20" / asset / (Path(clip["clip_id"]).name + ".npz")
        if asset not in models:
            meshes, _ = display.object_display(catalog[asset], args.output / f"object_{asset}")
            with np.load(path, allow_pickle=False) as recording:
                names = list(map(str, recording["names"]))
            model, maps, _ = display.display_model(args.urdf, meshes, names, width=1280, height=720,
                                                   verify_hands=False, mesh_root=args.urdf.parent)
            assert model.nq == model.nv == 0
            models[asset] = model, maps, object_radius(meshes)
        report = render_clip(path, clip, args.output, *models[asset], index, fonts, title)
        report["original_failure"] = prior["failed"]
        report["return_delta"] = clip["normalized_return"] - prior["normalized_return"]
        reports.append(report)
        print(json.dumps(report), flush=True)
    total_frames = sum(row["frames"] for row in reports)
    assert len(reports) == 20 and total_frames == sum(row["state_frames"] for row in originals)
    name = "ultra8000_zeroshot" if iteration == 8000 else f"ultra_tables{iteration}"
    combined = args.output / f"{name}_tables20.mp4"
    concat([args.output / row["video"] for row in reports], combined)
    display.verify_decoded_frames("ffmpeg", combined, total_frames)
    summary = {"clips": reports, "clip_count": len(reports), "frames": total_frames, "combined_sha256": sha256(combined),
               "original_result_sha256": sha256(original_path), "replay_result_sha256": sha256(replay_path)}
    (args.output / "rendering.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cards = []
    for asset in catalog:
        cards.append(f"<h2>{html.escape(asset)}</h2>")
        for row in (item for item in reports if item["asset"] == asset):
            cards.append(f'<article><h3>{html.escape(row["clip_id"])} · {html.escape(row["action"])}</h3>'
                         f'<video controls preload="none" poster="{row["poster"]}" src="{row["video"]}"></video></article>')
    (args.output / "index.html").write_text(
        f'<!doctype html><html lang="en"><meta charset="utf-8"><title>{title} tables20</title>'
        '<style>body{margin:2rem auto;max-width:70rem;padding:0 1rem;font-family:system-ui}video{width:100%}</style>'
        f'<h1>{title} tables20</h1><p>Policy and reference overlay · 50 fps</p>'
        f'<video controls preload="none" src="{combined.name}"></video>' + "".join(cards), encoding="utf-8")


if __name__ == "__main__":
    main()
