"""Prepare independent object caches in bounded processes for both partitions.

The existing cache algorithm and deterministic seeds are unchanged. This is
CPU data preparation, never a replacement for native physics acceptance.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import time

from .curated import canonical_bytes, digest, load_json, sha

SCHEMA = 'hoi_curated_prepared_geometry_v1'


def implementation_identity():
    root = Path(__file__).resolve().parents[2]
    return {name:sha(root/name) for name in (
        'hoi/v1/geometry.py', 'hoi/v1/prepare.py',
        'hoi/data/curated_bundle.py', 'hoi/data/curated_geometry.py')}


def geometry_key(collision_hash, distance_recipe):
    return digest({'collision_hash':collision_hash, 'distance_recipe':distance_recipe})


def _initialize_worker(threads):
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def _task(index, entry, root, output):
    from .curated_bundle import _prepare_asset, native_asset_equivalence
    base = Path(output)/f'asset_{index:04d}'
    base.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (base/f'attempt_{attempt:04d}').exists():
        attempt += 1
    folder = base/f'attempt_{attempt:04d}'
    folder.mkdir()
    start = time.perf_counter()
    asset = _prepare_asset(entry, index, Path(root), folder)
    row = {'asset_index':index, 'native_equivalence_sha256':native_asset_equivalence(entry, Path(root)),
           'cache_key':geometry_key(asset['collision_hash'], entry.get('distance_recipe', {})),
           'collision_hash':asset['collision_hash'], 'distance_recipe':entry.get('distance_recipe', {}),
           'geometry_cache':asset['geometry_cache'], 'elapsed_s':time.perf_counter()-start}
    row['receipt_content_sha256'] = digest(row)
    with (base/'SUCCESS.json').open('xb') as stream:
        stream.write(canonical_bytes(row)+b'\n')
    return row


def prepare_geometry_manifest(recipes_path, output, *, workers=1, threads=3, resume=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=resume)
    claim = output/'PREPARING.claim'
    with claim.open('x', encoding='utf-8') as f:
        json.dump({'pid':os.getpid(), 'purpose':'CPU geometry preparation; no training'}, f)
    try:
        return _prepare_locked(recipes_path, output, workers=workers, threads=threads, resume=resume)
    finally:
        claim.unlink()


def _prepare_locked(recipes_path, output, *, workers, threads, resume):
    from .curated_bundle import expand_native_recipe_assets, native_asset_equivalence
    if type(workers) is not int or workers < 1 or type(threads) is not int or threads < 1:
        raise ValueError('positive bounded worker and thread counts required')
    recipes_path, output = Path(recipes_path).resolve(), Path(output).resolve()
    recipe_sha = sha(recipes_path)
    recipe = load_json(recipes_path)
    unique = {}
    for entry in expand_native_recipe_assets(recipe):
        unique.setdefault(native_asset_equivalence(entry, recipes_path.parent), entry)
    if not unique:
        raise ValueError('geometry preparation requires at least one native asset')
    code = implementation_identity()
    plan = {'recipes':{'path':str(recipes_path), 'sha256':recipe_sha},
            'preparation_code':code, 'native_equivalences':list(unique)}
    plan_path = output/'preparation_plan.json'
    if plan_path.exists():
        if not resume or load_json(plan_path) != plan:
            raise ValueError('geometry resume plan differs from the bound recipe or algorithm')
    else:
        plan_path.write_bytes(canonical_bytes(plan)+b'\n')
    if (output/'geometry_manifest.json').exists():
        load_geometry_manifest(output/'geometry_manifest.json',recipes_path)
        return load_json(output/'geometry_manifest.json')
    rows = []
    pending = []
    from hoi.v1.prepare import _bound_load
    for i,(equivalence,entry) in enumerate(unique.items()):
        receipt = output/f'asset_{i:04d}/SUCCESS.json'
        if receipt.exists():
            row = load_json(receipt)
            if (row.get('receipt_content_sha256') != digest({k:v for k,v in row.items() if k != 'receipt_content_sha256'}) or
                    row.get('asset_index') != i or row.get('native_equivalence_sha256') != equivalence):
                raise ValueError('geometry success receipt identity changed')
            _bound_load(row['geometry_cache'])
            rows.append(row)
        else:
            pending.append((i,entry))
    # Spawn gives each process independent Torch thread pools and RNG state.
    # Surface sampling itself uses the existing private, fixed-seed Generator.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=_initialize_worker, initargs=(threads,)) as pool:
        futures = [pool.submit(_task, i, entry, str(recipes_path.parent), str(output))
                   for i,entry in pending]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps({'prepared_assets':len(rows), 'total_assets':len(unique),
                              'asset_index':row['asset_index'], 'elapsed_s':row['elapsed_s']}), flush=True)
    if sha(recipes_path) != recipe_sha or implementation_identity() != code:
        raise ValueError('recipe or preparation code changed during cache preparation')
    rows.sort(key=lambda row:row['asset_index'])
    manifest = {'schema':SCHEMA, 'recipes':{'path':str(recipes_path), 'sha256':recipe_sha},
                'preparation_code':code, 'workers':workers, 'threads_per_worker':threads,
                'assets':rows, 'training_started':False,
                'scope':'deterministic CPU geometry cache only; native acceptance remains required'}
    manifest['content_sha256'] = digest(manifest)
    (output/'geometry_manifest.json').write_bytes(canonical_bytes(manifest)+b'\n')
    return manifest


def load_geometry_manifest(path, recipes_path):
    manifest = load_json(path)
    if manifest.get('schema') != SCHEMA or manifest.get('content_sha256') != digest({k:v for k,v in manifest.items() if k != 'content_sha256'}):
        raise ValueError('prepared geometry manifest identity changed')
    if manifest['recipes']['sha256'] != sha(recipes_path):
        raise ValueError('prepared geometry belongs to a different native recipe')
    if manifest['preparation_code'] != implementation_identity():
        raise ValueError('prepared geometry algorithm identity changed')
    caches, equivalents = {}, set()
    from hoi.v1.prepare import _bound_load
    for row in manifest['assets']:
        key = geometry_key(row['collision_hash'], row['distance_recipe'])
        if key != row['cache_key'] or row['native_equivalence_sha256'] in equivalents:
            raise ValueError('prepared geometry key or asset identity changed')
        value = _bound_load(row['geometry_cache'])
        if value['metadata'].get('collision_hash') != row['collision_hash'] or value['metadata'].get('distance_recipe') != row['distance_recipe']:
            raise ValueError('prepared geometry cache metadata differs from its manifest')
        # Do not collapse a conflicting producer of the same immutable geometry.
        if key in caches and caches[key]['sha256'] != row['geometry_cache']['sha256']:
            from hoi.v1.geometry import tensor_sha256
            left, right = _bound_load(caches[key]), _bound_load(row['geometry_cache'])
            if left['metadata'] != right['metadata'] or any(tensor_sha256(left[k]) != tensor_sha256(right[k])
                    for k in ('points','sdf_grid','grid_min','grid_max','bbox_diagonal')):
                raise ValueError('prepared geometry producers disagree')
        caches.setdefault(key, row['geometry_cache'])
        equivalents.add(row['native_equivalence_sha256'])
    from .curated_bundle import expand_native_recipe_assets, native_asset_equivalence
    root = Path(recipes_path).resolve().parent
    expected = {native_asset_equivalence(entry,root) for entry in expand_native_recipe_assets(load_json(recipes_path))}
    if equivalents != expected:
        raise ValueError('prepared geometry does not cover the complete native recipe')
    return caches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--resume', action='store_true', help='reuse only hash-verified successes from this exact preparation plan')
    args = parser.parse_args(argv)
    manifest = prepare_geometry_manifest(args.recipes,args.output,workers=args.workers,threads=args.threads,resume=args.resume)
    print(json.dumps({'manifest':str(args.output/'geometry_manifest.json'), 'assets':len(manifest['assets'])}))


if __name__ == '__main__':
    main()
