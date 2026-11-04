"""Authenticated development selections and bounded mixed-asset evaluation batches."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from hoi.data.protected_eval_subset import digest


def load_selection(path, training, parent):
    path = Path(path)
    selection = json.loads(path.read_text(encoding='utf-8'))
    expected = selection.pop('content_sha256', None)
    if digest(selection) != expected or selection.get('schema_version') != 'hoi_protected_eval_subset_v1':
        raise ValueError('development selection schema/hash changed')
    if selection.get('training_allowed') is not False or selection.get('full_duration') is not True:
        raise ValueError('development selection must remain evaluation-only and full-duration')
    if parent['partition'] != 'protected_eval' or training['partition'] != 'train':
        raise ValueError('development selection requires protected evaluation')
    for field, bundle in [('parent_bundle', parent), ('training_bundle', training)]:
        binding = selection[field]
        if (binding['bundle_content_sha256'] != bundle['bundle_content_sha256'] or
                hashlib.sha256(Path(binding['path']).read_bytes()).hexdigest() != binding['sha256']):
            raise ValueError('selection parent/training bundle identity differs')
    if selection['selected_clips']['path'] != 'clips.jsonl':
        raise ValueError('selected rows must use the sibling clips.jsonl')
    raw = (path.parent/'clips.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest() != selection['selected_clips']['sha256']:
        raise ValueError('selected rows bytes changed')
    rows = [json.loads(line) for line in raw.splitlines()]
    cases = selection['cases']
    ids = [c['parent_clip_index'] for c in cases]
    if (len(rows) != selection['selected_count'] or len(cases) != len(rows) or
            len(set(ids)) != len(ids) or selection['parent_count'] != len(parent['clips'])):
        raise ValueError('selection count/uniqueness differs')
    train_keys = {(c['dataset'], c['clip_id']) for c in training['clips']}
    train_groups = {c['source_group'] for c in training['clips']}
    for case, row in zip(cases, rows):
        i = case['parent_clip_index']
        if type(i) is not int or not 0 <= i < len(parent['clips']):
            raise ValueError('selection index outside parent')
        if (row != parent['clips'][i] or digest(row) != case['parent_row_sha256'] or
                (row['dataset'], row['clip_id']) != (case['dataset'], case['clip_id'])):
            raise ValueError('selected row/identity differs from parent')
        if row['partition'] != 'protected_eval' or (row['dataset'], row['clip_id']) in train_keys or row['source_group'] in train_groups:
            raise ValueError('selected evaluation clip/source group leaked into training')
    selection['content_sha256'] = expected
    return set(ids), {**selection, 'selection_manifest_path': str(path.resolve()),
                      'selection_manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'is_subset': True}


def asset_batches(bundle, selected_ids, batch_size):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('asset batch size must be a positive integer')
    grouped = defaultdict(list)
    for i, clip in enumerate(bundle['clips']):
        if selected_ids is None or i in selected_ids:
            grouped[int(clip['asset_id'])].append(i)
    groups = sorted(grouped.items())
    if batch_size > 1:
        groups.sort(key=lambda item: (max(bundle['clips'][i]['state_frames'] for i in item[1]), item[0]))
    return [groups[i:i+batch_size] for i in range(0, len(groups), batch_size)]


def stratified_results(bundle, cases):
    from .evaluate import aggregate_evaluation_cases
    assets = {int(a['asset_id']): a for a in bundle['assets']}
    output = {}
    for mode in ('dataset', 'task', 'asset'):
        groups = defaultdict(list)
        for case in cases:
            row = bundle['clips'][case['clip_id']]
            key = (row['dataset'],)
            if mode == 'task': key += (row.get('task', 'unspecified'),)
            if mode == 'asset': key += (str(row['asset_id']), assets[row['asset_id']]['object_asset'])
            groups[key].append(case)
        output['by_'+mode] = [dict(group=list(key), count=len(values),
                                  aggregate=aggregate_evaluation_cases(values)) for key, values in sorted(groups.items())]
    return output
