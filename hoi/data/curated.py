"""Frozen curated references and deterministic, leakage-aware teacher partitions.

Geometric intent is never simulator contact truth. Source labels/assets remain
immutable; every emitted partition is derived from the curated included list.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

SELECTION_SCHEMA = 'hoi_corpus_selection_v1'
SPLIT_SCHEMA = 'hoi_curated_task_group_split_v1'
REFERENCE_SCHEMA = 'hoi_curated_reference_50hz_regions16_v1'
EXPECTED_SCHEMA = 'hoi_reference_region_interval_intent_v1'


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_selection(directory):
    directory = Path(directory)
    manifest, ready = load_json(directory / 'manifest.json'), load_json(directory / 'READY.json')
    if manifest.get('schema') != SELECTION_SCHEMA or ready['manifest_sha256'] != sha(directory / 'manifest.json'):
        raise ValueError('curated selection manifest/READY identity mismatch')
    for name in ('included.jsonl', 'excluded.jsonl', 'policy.json'):
        if sha(directory / name) != manifest['files'][name]['sha256']:
            raise ValueError('curated selection file hash mismatch: ' + name)
    rows = [json.loads(line) for line in (directory / 'included.jsonl').read_text(encoding='utf-8-sig').splitlines() if line]
    if len(rows) != manifest['included_count'] or len({(x['dataset'], x['clip_id']) for x in rows}) != len(rows):
        raise ValueError('curated included count/identities mismatch')
    if any(x.get('selection_status') != 'included' or x.get('eligible_for_privileged_tracking') is not True for x in rows):
        raise ValueError('only explicitly included privileged-tracking candidates may be consumed')
    if dict(Counter(x['dataset'] for x in rows)) != manifest['included_by_dataset']:
        raise ValueError('curated dataset counts mismatch')
    return rows, manifest


def _resolved(path, path_maps=()):
    if not path_maps and Path(path).is_absolute():
        return Path(path).resolve()
    from .manifest import resolve_motion_path
    return resolve_motion_path(str(path), path_maps)


def collect_metadata(selection, path_maps=(), workers=8):
    """Read hash-verified metadata/SUCCESS only, using bounded I/O concurrency."""
    from concurrent.futures import ThreadPoolExecutor
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError('metadata workers must be between 1 and 32')
    rows, _ = read_selection(selection)
    def read(row):
        root = _resolved(row['output'], path_maps)
        raw = (root/'metadata.json').read_bytes()
        metadata, success = json.loads(raw), load_json(root/'SUCCESS.json')
        if hashlib.sha256(raw).hexdigest() != success['metadata_sha256']:
            raise ValueError('metadata SHA mismatch: ' + row['clip_id'])
        if metadata['clip_id'] != row['clip_id'] or metadata['recipe_id'] != row['recipe_id']:
            raise ValueError('selection/label identity mismatch')
        if metadata['array_file']['sha256'] != success['array_sha256']:
            raise ValueError('metadata/SUCCESS array identity mismatch')
        return {**row, 'metadata_sha256': success['metadata_sha256'], 'labels_sha256': success['array_sha256'],
                'provenance': metadata['provenance'], 'object_geometry_sha256': metadata['object_geometry_sha256'],
                'object_geometry_path': metadata['object_geometry_path'], 'robot_geometry_sha256':metadata['robot_geometry_sha256']}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(read, rows))


def reconcile_records(rows, metadata_rows, task_annotations=None):
    """Keep explicit task evidence and conservatively join shared capture IDs.

    ``task_annotations`` are source-backed (dataset, clip_id) records, not labels
    guessed from filenames. Missing action annotations remain ``unspecified``.
    """
    index = {(r['dataset'], r['clip_id']): r for r in metadata_rows}
    if len(index) != len(metadata_rows) or set(index) != {(r['dataset'], r['clip_id']) for r in rows}:
        raise ValueError('metadata must cover the curated selection exactly once')
    annotations = {(r['dataset'], r['clip_id']): r for r in (task_annotations or [])}
    result = []
    for row in rows:
        key = row['dataset'], row['clip_id']
        record, dataset = index[key], row['dataset']
        if record['output'] != row['output'] or record['recipe_id'] != row['recipe_id']:
            raise ValueError('metadata row refers to a different selected label output')
        p = record['provenance']
        task = annotations.get(key, {}).get('task') or p.get('task') or (p.get('subset') if dataset == 'grail_sitting' else None)
        task_evidence = annotations.get(key, {}).get('evidence') or ('label_metadata.provenance.task' if p.get('task') else
                        'label_metadata.provenance.subset' if dataset == 'grail_sitting' else 'source_has_no_action_annotation')
        protected = bool(p.get('protected_eval80'))
        original_split = p.get('split', p.get('partition'))
        if original_split in ('test', 'validation', 'val', 'eval', 'eval80', 'protected_eval'):
            protected = True
        if dataset in ('ultra', 'hoi_retarget'):
            capture = p.get('source_group')
            if not capture:
                raise ValueError('OMOMO capture group missing')
            group, basis = 'omomo:' + capture, 'shared_OMOMO_capture_ID_conservative_cross_source_union'
        elif dataset == 'omnicontact':
            capture = p.get('capture_id') or p.get('source_group_id')
            if not capture or p.get('source_group_verified') is not True:
                raise ValueError('unverified OmniContact capture identity')
            group, basis = 'omnicontact:' + capture, 'official_capture_id'
        elif dataset == 'grail_sitting':
            capture = p.get('source_motion_key')
            if not capture:
                raise ValueError('GRAIL source motion key missing')
            # Randomized exports of one source motion stay together. Asset-name
            # prefixes are retained until stronger cross-asset identity evidence.
            capture = re.sub(r'_rand\d+$', '', capture)
            group, basis = 'grail_sitting:' + capture, 'source_motion_key_randomization_suffix_removed'
        else:
            raise ValueError('unsupported curated dataset: ' + dataset)
        result.append({**row, 'metadata_sha256': record['metadata_sha256'], 'labels_sha256': record['labels_sha256'],
                       'source_group': group, 'group_basis': basis, 'task': task or 'unspecified', 'task_evidence': task_evidence,
                       'source_split': original_split, 'mandatory_protected': protected,
                       'source_provenance': p, 'object_geometry_sha256': record['object_geometry_sha256'],
                       'object_geometry_path': record['object_geometry_path'], 'robot_geometry_sha256': record.get('robot_geometry_sha256')})
    # Exact source motion bytes are additional, auditable same-motion edges.
    # Do not use HOI-Retarget's whole-Parquet SHA: unrelated rows share it.
    parent = {r['source_group']:r['source_group'] for r in result}
    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g
    aliases = {}
    for r in result:
        p = r['source_provenance']
        if r['dataset'] == 'ultra':
            identity = p.get('motion_sha256')
        elif r['dataset'] == 'grail_sitting':
            identity = p.get('files', {}).get('robot', {}).get('sha256')
        elif r['dataset'] == 'omnicontact':
            identity = p.get('sha256')
        else:
            identity = None
        if identity:
            token = (r['dataset'], identity)
            if token in aliases:
                a,b = find(r['source_group']),find(aliases[token])
                parent[max(a,b)] = min(a,b)
            aliases[token] = r['source_group']
    sizes = Counter(find(g) for g in parent)
    for r in result:
        original = r['source_group']
        r['source_group_before_identity_union'] = original
        r['source_group'] = find(original)
        if sizes[r['source_group']] > 1:
            r['group_basis'] += ';exact_source_motion_bytes_union'
    return result


def stratified_split(records, eval_fraction=.1, seed=20261031):
    """Atomic source groups; deterministic task/dataset clip-count balancing.

    Existing protected groups are mandatory even when their count exceeds a
    stratum's target. Greedy additions minimize weighted per-stratum discrepancy.
    """
    if not (0 < eval_fraction < 1):
        raise ValueError('eval_fraction must be strictly between zero and one')
    if not records or len({(r['dataset'], r['clip_id']) for r in records}) != len(records):
        raise ValueError('records must be nonempty with unique clip identities')
    groups = defaultdict(list)
    for r in records:
        groups[r['source_group']].append(r)
    stratum = lambda r: (r['dataset'], r['task'])
    total = Counter(stratum(r) for r in records)
    target = {k: n * eval_fraction for k, n in total.items()}
    counts = {g: Counter(stratum(r) for r in rs) for g, rs in groups.items()}
    mandatory = {g for g, rs in groups.items() if any(r['mandatory_protected'] for r in rs)}
    forced_count = Counter(stratum(r) for g in mandatory for r in groups[g])
    selected = set(mandatory)
    current = Counter(stratum(r) for g in selected for r in groups[g])
    tie = lambda g: hashlib.sha256((str(seed) + '\0' + g).encode()).hexdigest()
    remaining = set(groups) - selected
    # A mixed-dataset group receives one indivisible decision. The objective
    # balances absolute clip counts within each dataset/task stratum.
    while remaining:
        candidates = []
        for g in remaining:
            delta = sum(((current[k] + n - target[k]) ** 2 - (current[k] - target[k]) ** 2) / max(1., target[k])
                        for k, n in counts[g].items())
            candidates.append((delta, tie(g), g))
        delta, _, group = min(candidates)
        if delta >= -1e-12:
            break
        selected.add(group)
        current.update(counts[group])
        remaining.remove(group)
    partitions = {name: [] for name in ('train', 'protected_eval')}
    for r in sorted(records, key=lambda x: (x['dataset'], x['clip_id'])):
        partition = 'protected_eval' if r['source_group'] in selected else 'train'
        partitions[partition].append({**r, 'partition': partition})
    if not partitions['train'] or not partitions['protected_eval']:
        raise ValueError('group constraints produced an empty partition')
    report = []
    for (dataset, task), n in sorted(total.items()):
        ne = current[(dataset, task)]
        report.append({'dataset': dataset, 'task': task, 'total': n, 'train': n-ne, 'protected_eval': ne,
                       'eval_fraction': ne/n, 'target_eval_count': target[(dataset, task)],
                       'total_fraction': n/len(records),
                       'train_partition_fraction': (n-ne)/len(partitions['train']),
                       'protected_eval_partition_fraction': ne/len(partitions['protected_eval'])})
    return partitions, {'requested_eval_fraction': eval_fraction, 'seed': seed,
                       'counts': {k: len(v) for k, v in partitions.items()},
                       'source_groups': {k: len({r['source_group'] for r in v}) for k, v in partitions.items()},
                       'mandatory_protected_groups': len({r['source_group'] for r in records if r['mandatory_protected']}),
                       'actual_eval_fraction': len(partitions['protected_eval'])/len(records),
                       'task_strata': report, 'source_group_overlap': [],
                       'strata_without_train': [{'dataset':r['dataset'],'task':r['task']} for r in report if not r['train']],
                       'strata_without_protected_eval': [{'dataset':r['dataset'],'task':r['task']} for r in report if not r['protected_eval']],
                       'mandatory_eval_exceeds_target_strata': [{'dataset':d,'task':t,'mandatory_clips':forced_count[(d,t)]} for d,t in total if forced_count[(d,t)] > target[(d,t)]],
                       'group_policy': 'atomic_capture_groups; mandatory_original_eval; deterministic_greedy_dataset_task_clip_balance',
                       'task_missing_policy': 'unspecified remains explicit; never inferred from object class',
                       'split_group_basis': dict(Counter(r.get('group_basis','caller_declared_group') for r in records)),
                       'group_identity_limitations': ['GRAIL groups published object/motion families across randomization variants; shared underlying human template independence is not established', 'ULTRA and HOI-Retarget matching OMOMO capture IDs are conservatively grouped across datasets'],
                       'train_semantics': 'privileged_tracking_candidates; selection inclusion is not physical tracking success',
                       'physical_tracking_pass': False, 'training_started': False}


def write_split(selection, metadata_path, output, eval_fraction=.1, seed=20261031, task_annotations_path=None):
    rows, manifest = read_selection(selection)
    metadata_rows = load_json(metadata_path)
    annotations = load_json(task_annotations_path) if task_annotations_path else None
    records = reconcile_records(rows, metadata_rows, annotations)
    partitions, report = stratified_split(records, eval_fraction, seed)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_selection = output/'source_selection'
    source_selection.mkdir()
    for name in ('manifest.json','READY.json','included.jsonl','excluded.jsonl','policy.json'):
        (source_selection/name).write_bytes((Path(selection)/name).read_bytes())
    (output/'source_metadata.json').write_bytes(Path(metadata_path).read_bytes())
    if task_annotations_path:
        (output/'source_tasks.json').write_bytes(Path(task_annotations_path).read_bytes())
    files = {}
    for partition, entries in partitions.items():
        path = output / (partition + '.jsonl')
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            for r in entries:
                stream.write(canonical_bytes(r).decode() + '\n')
        files[partition] = {'path': path.name, 'sha256': sha(path), 'count': len(entries)}
    result = {'schema': SPLIT_SCHEMA, 'selection_id': manifest['selection_id'],
              'selection': {'path': 'source_selection', 'original_path':str(Path(selection).resolve()), 'manifest_sha256': sha(Path(selection)/'manifest.json')},
              'source_metadata': {'path': 'source_metadata.json', 'original_path':str(Path(metadata_path).resolve()), 'sha256': sha(metadata_path)},
              'task_annotations': {'path': 'source_tasks.json', 'original_path':str(Path(task_annotations_path).resolve()), 'sha256': sha(task_annotations_path)} if task_annotations_path else None,
              'partitions': files, 'report': report}
    result['content_sha256'] = digest(result)
    (output/'split.json').write_bytes(canonical_bytes(result) + b'\n')
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    import csv
    with (output/'task_distribution.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(report['task_strata'][0]))
        writer.writeheader(); writer.writerows(report['task_strata'])
    return result


def read_split(path):
    value = load_json(path)
    if value.get('schema') == 'hoi_compact_teacher_split_v1':
        from .compact_teacher import read_compact_split
        return read_compact_split(path)
    root = Path(path).resolve().parent
    def bound(name):
        candidate=Path(name)
        if candidate.is_absolute() or '..' in candidate.parts:
            raise ValueError('split dependency paths must be relative within the portable split directory')
        result=(root/candidate).resolve()
        if not result.is_relative_to(root):
            raise ValueError('split dependency escapes its directory')
        return result
    payload = {k: v for k, v in value.items() if k != 'content_sha256'}
    if value.get('schema') != SPLIT_SCHEMA or value.get('content_sha256') != digest(payload):
        raise ValueError('curated split schema/hash mismatch')
    if set(value['partitions']) != {'train', 'protected_eval'}:
        raise ValueError('curated split requires train and protected_eval')
    selection_dir = bound(value['selection']['path'])
    selected, source_manifest = read_selection(selection_dir)
    if sha(selection_dir/'manifest.json') != value['selection']['manifest_sha256']:
        raise ValueError('curated source selection identity changed')
    meta_spec = value['source_metadata']
    if sha(bound(meta_spec['path'])) != meta_spec['sha256']:
        raise ValueError('curated source metadata snapshot changed')
    annotation_spec = value.get('task_annotations')
    if annotation_spec is not None and sha(bound(annotation_spec['path'])) != annotation_spec['sha256']:
        raise ValueError('curated task annotation evidence changed')
    expected_rows = reconcile_records(selected, load_json(bound(meta_spec['path'])), load_json(bound(annotation_spec['path'])) if annotation_spec else None)
    expected = {(r['dataset'],r['clip_id']): r for r in expected_rows}
    partitions, seen, groups = {}, set(), {}
    for partition, spec in value['partitions'].items():
        if sha(bound(spec['path'])) != spec['sha256']:
            raise ValueError('partition bytes changed')
        entries = [json.loads(line) for line in bound(spec['path']).read_text(encoding='utf-8').splitlines()]
        if len(entries) != spec['count']:
            raise ValueError('partition count mismatch')
        for r in entries:
            key = r['dataset'], r['clip_id']
            if key in seen or r['partition'] != partition:
                raise ValueError('duplicate/wrong-partition curated clip')
            if key not in expected or {k:v for k,v in r.items() if k != 'partition'} != expected[key]:
                raise ValueError('curated row differs from frozen selection/source metadata')
            if r['source_group'] in groups and groups[r['source_group']] != partition:
                raise ValueError('source-group leakage')
            if r['mandatory_protected'] and partition != 'protected_eval':
                raise ValueError('protected source was placed in training')
            seen.add(key); groups[r['source_group']] = partition
        partitions[partition] = entries
    if seen != set(expected):
        raise ValueError('curated split must cover the exact included selection')
    return value, partitions


def group_labels(labels, channels, active):
    """Spatial tri-state OR: any positive, all active negative, else unknown."""
    import numpy as np
    from hoi.v1.contacts import BODY_GROUP_LINKS, CHANNELS
    labels, active = np.asarray(labels), np.asarray(active, dtype=bool)
    if labels.ndim != 2 or labels.shape[1] != len(channels) or active.shape != (len(channels),):
        raise ValueError('label/channel/active shape mismatch')
    if len(set(channels)) != len(channels) or not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError('labels require unique channels and tri-state values')
    if np.any(labels[:, ~active] != -1):
        raise ValueError('inactive geometry must remain unknown')
    out = np.full((len(labels), len(CHANNELS)), -1, dtype=np.int8)
    groups = [tuple(link + '__object' for link in links) for links in BODY_GROUP_LINKS] + [('object__ground',)]
    index = {name: i for i, name in enumerate(channels)}
    for g, names in enumerate(groups):
        if any(name not in index for name in names):
            raise ValueError('missing canonical channel for region ' + CHANNELS[g])
        ids = [index[name] for name in names if active[index[name]]]
        if not ids:
            continue
        values = labels[:, ids]
        out[np.all(values == 0, axis=1), g] = 0
        out[np.any(values == 1, axis=1), g] = 1
    return out


def _interval_labels(times, labels, target_times):
    import numpy as np
    times, target_times = np.asarray(times, dtype=np.float64), np.asarray(target_times, dtype=np.float64)
    if times.ndim != 1 or len(times) != len(labels) or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('invalid source label times')
    edges = np.r_[times[0], (times[:-1] + times[1:])*.5, times[-1]]
    out = np.full((len(target_times)-1, labels.shape[1]), -1, dtype=np.int8)
    for i, (start, end) in enumerate(zip(target_times[:-1], target_times[1:])):
        if start < times[0]-1e-9 or end > times[-1]+1e-9:
            raise ValueError('control interval outside observed source duration')
        left = max(0, np.searchsorted(edges, start, side='right')-1)
        right = min(len(times), np.searchsorted(edges, end, side='left')+1)
        overlap = np.minimum(end, edges[left+1:right+1])-np.maximum(start, edges[left:right])
        values = labels[left:right][overlap > 1e-12]
        if len(values):
            out[i, np.all(values == 0, axis=0)] = 0
            out[i, np.any(values == 1, axis=0)] = 1
    return out


def interval_schedule(arrays, metadata):
    """Combine source and target sample-support evidence without interpolation.

    Spatial aggregation precedes time aggregation. Any observed positive wins;
    negatives require every supported sample on both grids explicitly negative.
    Row i is the completed (i-1,i] control interval; row zero has no predecessor.
    """
    import numpy as np
    active = metadata['active_geometry_channels']
    active = arrays[active['npz_key']] if isinstance(active, dict) else active
    target_times = np.asarray(arrays['timestamps_s'], dtype=np.float64)
    if len(target_times) < 2 or not np.isfinite(target_times).all() or not np.allclose(np.diff(target_times), .02, atol=1e-9, rtol=0):
        raise ValueError('curated reference must already be on its unchanged 50 Hz grid')
    grouped = group_labels(arrays['target_labels'], metadata['channels'], active)
    source = group_labels(arrays['source_labels'], metadata['channels'], active)
    target_interval = _interval_labels(target_times, grouped, target_times)
    source_interval = _interval_labels(arrays['source_timestamps_s'], source, target_times)
    labels = np.full(grouped.shape, -1, dtype=np.int8)
    labels[1:][(target_interval == 0) & (source_interval == 0)] = 0
    labels[1:][(target_interval == 1) | (source_interval == 1)] = 1
    return {'expected_contact': (labels == 1).astype(np.float32), 'expected_valid': (labels >= 0).astype(np.float32),
            'region_labels': labels, 'metadata': {'schema': EXPECTED_SCHEMA, 'alignment': 'row_i_interval_(i-1,i];row_0_unknown',
            'spatial_aggregation': 'any_positive;all_active_negative;otherwise_unknown',
            'temporal_aggregation': 'source_and_target_nearest_sample_support;any_positive;all_negative_on_both_grids',
            'fractions_are_probabilities': False}}


def _restore(value, arrays):
    if isinstance(value, dict):
        if set(value) == {'npz_key'}:
            return arrays[value['npz_key']].tolist()
        return {k: _restore(v, arrays) for k,v in value.items()}
    if isinstance(value, list):
        return [_restore(x, arrays) for x in value]
    return value


def load_curated_clip(record, path_maps=()):
    """Read hash-bound 50 Hz labels into the common reference, without resampling."""
    import numpy as np
    import torch
    from .manifest import robot_joint_names
    from .reference import _gradient, _world_angular_velocity
    from supermimic.rotations import quat_to_mat
    from hoi.v1.contacts import CHANNELS
    root = _resolved(record['output'], path_maps)
    metadata, success = load_json(root/'metadata.json'), load_json(root/'SUCCESS.json')
    if sha(root/'metadata.json') != success['metadata_sha256'] or success['metadata_sha256'] != record['metadata_sha256']:
        raise ValueError('selected metadata hash mismatch')
    if sha(root/'labels.npz') != success['array_sha256'] or success['array_sha256'] != record['labels_sha256'] or metadata['array_file']['sha256'] != success['array_sha256']:
        raise ValueError('selected label array hash mismatch')
    if metadata.get('schema') != 'g1_expected_surface_contact_v1' or metadata['clip_id'] != record['clip_id'] or metadata['recipe_id'] != record['recipe_id']:
        raise ValueError('selected label schema/recipe/clip mismatch')
    if record.get('robot_geometry_sha256') is not None and metadata.get('robot_geometry_sha256') != record['robot_geometry_sha256']:
        raise ValueError('selected robot geometry identity mismatch')
    if metadata['semantics'].get('not_force_ground_truth') is not True:
        raise ValueError('reference intent contract missing')
    with np.load(root/'labels.npz', allow_pickle=False) as z:
        arrays = {name: z[name] for name in z.files}
    contact = interval_schedule(arrays, metadata)
    restored = _restore(metadata, arrays)
    times = np.asarray(arrays['timestamps_s'], dtype=np.float64); n = len(times)
    p = restored['provenance']
    # Every pipeline adapter exports the verified canonical joint order.
    source_names = p.get('source_joint_names')
    if source_names is not None and set(source_names) != set(robot_joint_names()):
        raise ValueError('source joint-name evidence differs from canonical G1')
    def finite(name, width):
        a = np.asarray(arrays[name])
        if a.shape != (n, width) or not np.isfinite(a).all():
            raise ValueError('invalid curated pose array: ' + name)
        return torch.as_tensor(a.copy(), dtype=torch.float32)
    rp, rq, q, op, oq = [finite(k,w) for k,w in [('root_pos',3),('root_quat',4),('joint_pos',29),('object_pos',3),('object_quat',4)]]
    for quaternion in (rq, oq):
        if bool((torch.linalg.vector_norm(quaternion, dim=-1)-1).abs().gt(1e-3).any()):
            raise ValueError('nonunit curated wxyz quaternion')
    rr, ro = quat_to_mat(rq), quat_to_mat(oq)
    reference = {'timestamps_s': torch.as_tensor(times.copy()),
                 'robot': {'root_pos': rp, 'root_rot': rr, 'joint_pos': q,
                           'root_quat_xyzw': rq[:, [1,2,3,0]], 'derived_ref_v': _gradient(rp,.02),
                           'derived_ref_omega': _world_angular_velocity(rr,.02), 'derived_ref_joint_qd': _gradient(q,.02)},
                 'object': {'pos': op, 'rot': ro, 'quat_xyzw': oq[:, [1,2,3,0]],
                            'derived_ref_v': _gradient(op,.02), 'derived_ref_omega': _world_angular_velocity(ro,.02)},
                 'metadata': {'schema_version': REFERENCE_SCHEMA, 'state_frames': n, 'control_steps': n-1,
                              'control_hz': 50., 'control_dt_s': .02, 'source_frames': metadata['statistics']['source_frames'],
                              'target_joint_names': robot_joint_names(), 'quaternion_source_order': 'wxyz',
                              'world_frame': 'source_world_z_up_no_recenter', 'timebase': metadata['timebase'],
                              'source_sampled': 'not_fabricated_for_non_ULTRA_sources'},
                 'provenance': {'dataset': record['dataset'], 'clip_id': record['clip_id'], 'source_group': record['source_group'],
                                'source_motion': record['output'], 'source_motion_sha256': record['labels_sha256'],
                                'label_metadata_sha256': record['metadata_sha256'], 'source_provenance': p}}
    from hoi.v1.scene_supports import bind_scene_supports
    bound_supports = bind_scene_supports(restored.get('scene_supports', []), arrays['source_timestamps_s'], times)
    reference['scene_supports'] = bound_supports
    contact = {**contact, **{k: torch.as_tensor(contact[k]) for k in ('expected_contact','expected_valid','region_labels')}}
    contact['metadata']['channels'] = list(CHANNELS)
    geometry_path = _resolved(restored['object_geometry_path'], path_maps)
    geometry = load_json(geometry_path)
    if digest(geometry) != restored['object_geometry_sha256'] or restored['object_geometry_sha256'] != record['object_geometry_sha256']:
        raise ValueError('source object geometry identity mismatch')
    return {'name': record['dataset'] + ':' + record['clip_id'], 'source_group': record['source_group'],
            'reference': reference, **{k:contact[k] for k in ('expected_contact','expected_valid')},
            'contact': contact, 'object_geometry': geometry, 'object_asset': restored['object_asset'],
            'scene_supports': restored.get('scene_supports', []), 'bound_scene_supports': bound_supports, 'label_metadata': restored}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    metadata = sub.add_parser('metadata', help='read metadata/SUCCESS only; no motion loading or training')
    metadata.add_argument('--selection', required=True); metadata.add_argument('--output', required=True)
    metadata.add_argument('--workers', type=int, default=8)
    split = sub.add_parser('split')
    split.add_argument('--selection', required=True); split.add_argument('--metadata', required=True)
    split.add_argument('--output', required=True); split.add_argument('--eval-fraction', type=float, default=.1)
    split.add_argument('--seed', type=int, default=20261031); split.add_argument('--task-annotations')
    args = parser.parse_args(argv)
    if args.command == 'metadata':
        result = collect_metadata(args.selection, workers=args.workers)
        with Path(args.output).open('xb') as stream:
            stream.write(canonical_bytes(result)+b'\n')
        print(json.dumps({'clips':len(result),'metadata_sha256':sha(args.output)}))
    else:
        result = write_split(args.selection, args.metadata, args.output, args.eval_fraction, args.seed, args.task_annotations)
        print(json.dumps(result['report'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
