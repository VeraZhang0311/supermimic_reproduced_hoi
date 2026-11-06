"""Losslessly project the reviewed 253D packed BONES library to 71D at 50 Hz.

No filtering, resampling, finite differences, quaternion normalization, or FK.
The old producer's explicit world-root fields are copied; redundant body fields
are discarded. Source data are read only. A final manifest is the commit marker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys

import numpy as np

# Running this file directly does not otherwise put the project on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.robot_metadata import G1_JOINT_NAMES

FORMAT = "bones_seed_g1_float32_50hz_v1"
STATE_LAYOUT = "p3,quat_wxyz4,v3,w3,q29,qd29"
SOURCE_LAYOUT = "q29,qd29,p_world3,quat_world_wxyz4,v_world3,w_world3,body_p42,body_quat56,body_v42,body_w42"
PRODUCER_ROOT = "/mnt/projects/ll/tianzizh/SuperMimic_HOI/code/src/supermimic"
# Inspected on the actual cluster before implementing this conversion.
SCHEMA_EVIDENCE = {
    "motion.py": {"path": PRODUCER_ROOT + "/data/motion.py",
                  "sha256": "70ed6e3f7c88968a7708fed2472bbfda2e00801869b42c998c3c2afdb1cdf911",
                  "definition": "MotionClip.states(), lines 123-132: explicit world root before redundant bodies; duration=(frames-1)/fps"},
    "mapped.py": {"path": PRODUCER_ROOT + "/data/mapped.py",
                  "sha256": "b2df64183897edb493764e31831d9d2abaa25ff9658446fdcddb8acf2e88e375",
                  "definition": "build_packed_library(), lines 23-81: contiguous clip.states() and source hashes"},
    "tracking.py": {"path": PRODUCER_ROOT + "/tracking.py",
                    "sha256": "01578471abf0888c9236f59ce8e8c40d6ca156bae60e9bef8f99d063d83d545b",
                    "definition": "StateLayout.split(): widths (29,29,3,4,3,3,42,56,42,42)"},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path, value):
    partial = path.with_name(path.name + ".partial")
    with partial.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


def _new_destination(source, output):
    output = Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError("output must be outside the read-only source directory")
    if output.exists():
        raise FileExistsError(f"destination exists; use a new directory: {output}")
    return output


def _chunk_size(value):
    if type(value) is not int or value < 1:
        raise ValueError("chunk_frames must be a positive integer")


def _hash_value(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"invalid {label} SHA256")


def _index_validate(index):
    if (index.get("schema_version") != 1 or index.get("state_file") != "states.npy"
            or index.get("dtype") != "float32" or index.get("quaternion_order") != "wxyz"
            or index.get("state_dim") != 253 or index.get("joint_names") != G1_JOINT_NAMES
            or index.get("root_body") != "pelvis"):
        raise ValueError("incompatible reviewed packed BONES schema/joint order")
    bodies = index.get("body_names")
    if not isinstance(bodies, list) or len(bodies) != 14 or len(set(bodies)) != 14 or "pelvis" not in bodies:
        raise ValueError("invalid 14-body source metadata")
    entries = index.get("clips")
    if not isinstance(entries, list) or not entries:
        raise ValueError("source clips must be nonempty")
    names, total = set(), 0
    for entry in entries:
        name, frames = entry.get("name"), entry.get("frames")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("source clip names must be nonempty and unique")
        if type(frames) is not int or frames < 2 or type(entry.get("offset")) is not int or entry["offset"] != total:
            raise ValueError("source clip frames/contiguous offsets invalid")
        if (entry.get("fps") != 50.0 or not isinstance(entry.get("duration"), (int, float))
                or not math.isclose(entry["duration"], (frames - 1) / 50.0, rel_tol=1e-12, abs_tol=1e-10)):
            raise ValueError("source clips must retain exact 50Hz inclusive-last-frame timing")
        _hash_value(entry.get("npz_sha256"), "clip npz")
        if entry.get("source_sha256") is not None:
            _hash_value(entry["source_sha256"], "clip source")
        names.add(name)
        total += frames
    if type(index.get("total_frames")) is not int or index["total_frames"] != total:
        raise ValueError("source total_frames disagrees with complete clips")
    _hash_value(index.get("state_file_sha256"), "source states")
    return total


def _signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


def _project(block):
    """Copy source root [58:71], then q/qd [0:58], without changing any value."""
    result = np.empty((len(block), 71), dtype="<f4")
    result[:, :13] = block[:, 58:71]
    result[:, 13:] = block[:, :58]
    if not np.isfinite(result).all():
        raise ValueError("nonfinite retained generalized state")
    if np.any(np.abs(np.linalg.norm(result[:, 3:7], axis=1) - 1.0) > 1e-3):
        raise ValueError("source root quaternion is not unit wxyz (tolerance 1e-3)")
    return result


def _base_manifest(clips, frames, digest):
    return {"format": FORMAT, "dataset": "bones-studio/seed", "control_dt": .02,
            "source_fps": 50.0, "joint_names": G1_JOINT_NAMES,
            "state_layout": STATE_LAYOUT, "dtype": "<f4", "frames": frames,
            "clips": clips, "states_sha256": digest,
            "exact_paper_selection": False}


def prepare(source, output, *, chunk_frames=65536, eval_output=None, eval_clips=512, eval_seed=0):
    """Convert every original clip; optionally also produce a small eval subset.

    Memory is O(chunk_frames * 253 + index metadata), independent of frame count.
    The complete source NPY bytes are hashed in the same sequential pass used for
    projection, including its header. An incomplete output never has a manifest.
    """
    _chunk_size(chunk_frames)
    source = Path(source).resolve()
    output = _new_destination(source, output)
    index_path, state_path = source / "index.json", source / "states.npy"
    index_hash = sha256(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    total = _index_validate(index)
    if eval_output is not None:
        eval_output = _new_destination(source, eval_output)
        if eval_output == output or output in eval_output.parents or eval_output in output.parents:
            raise ValueError("full and eval destinations must be separate new directories")
        _selection_indices(index["clips"], eval_clips, eval_seed)
    before = _signature(state_path)
    states = np.load(state_path, mmap_mode="r", allow_pickle=False)
    try:
        if states.dtype != np.dtype("<f4") or states.shape != (total, 253) or not states.flags.c_contiguous:
            raise ValueError("source NPY must be C-contiguous little-endian float32 [frames,253]")
        if before[0] != index.get("state_file_bytes") or before[0] != states.offset + states.nbytes:
            raise ValueError("source NPY file length/header/index mismatch")
        data_offset = states.offset
        # Validate with NumPy's safe NPY reader, but do not retain a huge mapping:
        # resident source pages must not accumulate over the full 49 GB pass.
        states._mmap.close()
        states = None
        output.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(output.parent).free < total * 71 * 4 + 1024 * 1024:
            raise OSError("insufficient free space for the complete new 71D library")
        output.mkdir(exist_ok=False)
        source_digest, result_digest = hashlib.sha256(), hashlib.sha256()
        partial = output / "states.bin.partial"
        with state_path.open("rb", buffering=0) as source_stream, partial.open("xb") as stream:
            source_digest.update(source_stream.read(data_offset))
            for start in range(0, total, chunk_frames):
                rows = min(chunk_frames, total - start)
                values = np.fromfile(source_stream, dtype="<f4", count=rows * 253)
                if values.size != rows * 253:
                    raise ValueError("source NPY truncated during conversion")
                block = values.reshape(rows, 253)
                source_digest.update(memoryview(block).cast("B"))
                result = _project(block)
                payload = memoryview(result).cast("B")
                stream.write(payload)
                result_digest.update(payload)
                if start // chunk_frames % 100 == 0:
                    print(f"projected {start + len(block)}/{total} frames", flush=True)
            stream.flush()
            os.fsync(stream.fileno())
        if source_digest.hexdigest() != index["state_file_sha256"]:
            raise ValueError("source NPY SHA256 mismatch; no manifest published")
        if _signature(state_path) != before or sha256(index_path) != index_hash:
            raise ValueError("source changed during conversion; no manifest published")
        partial.replace(output / "states.bin")
        clips = [dict(entry, source_offset=entry["offset"]) for entry in index["clips"]]
        manifest = _base_manifest(clips, total, result_digest.hexdigest())
        manifest.update(selection="all_original_filtered_clips", transformation="lossless_column_projection_only",
                        lineage={"source_directory": str(source), "source_index_sha256": index_hash,
                                 "source_state_file_sha256": source_digest.hexdigest(),
                                 "source_state_file_bytes": before[0], "source_shape": [total, 253],
                                 "source_clip_count": len(clips), "source_layout": SOURCE_LAYOUT,
                                 "column_projection": [[58, 71], [0, 58]],
                                 "position_velocity_frames": "world", "joint_units": "rad,rad/s",
                                 "source_file_hash_verified": True, "schema_evidence": SCHEMA_EVIDENCE,
                                 "conversion_script_sha256": sha256(__file__)})
        _json_write(output / "manifest.json", manifest)
    finally:
        if isinstance(states, np.memmap):
            states._mmap.close()
    if eval_output is not None:
        prepare_eval(output, eval_output, clips=eval_clips, seed=eval_seed, chunk_frames=chunk_frames)
    return manifest


def _selection_indices(entries, clips, seed):
    if type(clips) is not int or not 1 <= clips <= len(entries) or type(seed) is not int:
        raise ValueError("eval_clips must be within the original clip count; eval_seed must be an integer")
    order = sorted(range(len(entries)), key=lambda i: entries[i]["name"])
    sampled = random.Random(seed).sample(range(len(order)), clips)
    return [order[i] for i in sampled], sampled


def prepare_eval(full_library, output, *, clips=512, seed=0, chunk_frames=65536):
    """Copy fixed complete training clips into a small evaluation library.

    This is an eval_training_subset, NOT a held-out set. No clips are removed
    from the full training library. Sample order is Python Random.sample order
    over indices in the lexicographically name-sorted original clip list.
    """
    _chunk_size(chunk_frames)
    full_library = Path(full_library).resolve()
    output = _new_destination(full_library, output)
    manifest_path = full_library / "manifest.json"
    parent_hash = sha256(manifest_path)
    full = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (full.get("format") != FORMAT or full.get("state_layout") != STATE_LAYOUT
            or full.get("joint_names") != G1_JOINT_NAMES or full.get("dtype") != "<f4"
            or full.get("control_dt") != .02 or full.get("selection") != "all_original_filtered_clips"):
        raise ValueError("eval parent must be the complete converted filtered library")
    chosen, sampled = _selection_indices(full["clips"], clips, seed)
    state_path = full_library / "states.bin"
    before = _signature(state_path)
    if before[0] != full["frames"] * 71 * 4 or sha256(state_path) != full["states_sha256"]:
        raise ValueError("full library state size/SHA256 mismatch")
    count = sum(full["clips"][i]["frames"] for i in chosen)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < count * 71 * 4 + 1024 * 1024:
        raise OSError("insufficient free space for eval subset")
    output.mkdir(exist_ok=False)
    digest, entries, offset = hashlib.sha256(), [], 0
    with state_path.open("rb", buffering=0) as source_stream, (output / "states.bin.partial").open("xb") as stream:
        for i in chosen:
            original = full["clips"][i]
            start, length = original["offset"], original["frames"]
            source_stream.seek(start * 71 * 4)
            for first in range(0, length, chunk_frames):
                rows = min(chunk_frames, length - first)
                block = np.fromfile(source_stream, dtype="<f4", count=rows * 71)
                if block.size != rows * 71:
                    raise ValueError("full library truncated while copying eval subset")
                payload = memoryview(block).cast("B")
                stream.write(payload)
                digest.update(payload)
            entries.append(dict(original, offset=offset, parent_clip_index=i, parent_offset=start))
            offset += length
        stream.flush()
        os.fsync(stream.fileno())
    if _signature(state_path) != before or sha256(manifest_path) != parent_hash:
        raise ValueError("full library changed while copying eval subset")
    (output / "states.bin.partial").replace(output / "states.bin")
    selection = {"kind": "eval_training_subset", "held_out": False, "training_excludes_selection": False,
                 "algorithm": "random.Random(seed).sample(range(N), clips) over lexicographically sorted clip names; preserve sample order",
                 "seed": seed, "population_clips": len(full["clips"]), "sampled_sorted_indices": sampled,
                 "names": [entry["name"] for entry in entries],
                 "parent_manifest_sha256": parent_hash, "parent_library": str(full_library)}
    _json_write(output / "selection.json", selection)
    result = _base_manifest(entries, count, digest.hexdigest())
    result.update(selection="eval_training_subset", held_out=False, training_excludes_selection=False,
                  selection_sha256=sha256(output / "selection.json"),
                  parent_manifest_sha256=parent_hash, parent_library=str(full_library),
                  lineage=full["lineage"], transformation="copy_complete_training_clips_without_resampling")
    _json_write(output / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="existing read-only 253D packed directory")
    parser.add_argument("--output", required=True, help="new full 71D library directory")
    parser.add_argument("--chunk-frames", type=int, default=65536)
    parser.add_argument("--eval-output", help="optional separate new small evaluation library")
    parser.add_argument("--eval-clips", type=int, default=512)
    parser.add_argument("--eval-seed", type=int, default=0)
    args = parser.parse_args()
    result = prepare(args.source, args.output, chunk_frames=args.chunk_frames,
                     eval_output=args.eval_output, eval_clips=args.eval_clips, eval_seed=args.eval_seed)
    print(f"complete: {len(result['clips'])} clips, {result['frames']} frames, SHA256 {result['states_sha256']}", flush=True)


if __name__ == "__main__":
    main()
