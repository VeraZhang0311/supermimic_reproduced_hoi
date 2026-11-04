"""Freeze a development-only clip selection without changing either partition.

This is a selection manifest, not a new training/evaluation bundle. Consumers
must use its bound original bundle and preserve complete selected trajectories.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_parent(path, partition):
    path = Path(path)
    raw = path.read_bytes()
    value = json.loads(raw)
    expected = value.pop('bundle_content_sha256')
    if digest(value) != expected or value['partition'] != partition or value['scope']['is_subset']:
        raise ValueError('parent must be a complete, intact declared partition')
    value['bundle_content_sha256'] = expected
    return value, dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                       bundle_content_sha256=expected)


def choose_clips(training, parent, *, seed=20261104, fraction_numerator=1, fraction_denominator=10):
    if training['partition'] != 'train' or parent['partition'] != 'protected_eval':
        raise ValueError('development selection must come from protected_eval')
    if training['manifest'] != parent['manifest']:
        raise ValueError('train/protected_eval must share the frozen split')
    clips = parent['clips']
    if not clips or not 0 < fraction_numerator <= fraction_denominator:
        raise ValueError('invalid selection size')
    keys = [(c['dataset'], c['clip_id']) for c in clips]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate protected clips')
    train_keys = {(c['dataset'], c['clip_id']) for c in training['clips']}
    train_groups = {c['source_group'] for c in training['clips']}
    if set(keys) & train_keys or {c['source_group'] for c in clips} & train_groups:
        raise ValueError('train/protected_eval clip or source-group overlap')
    if any(c['partition'] != 'protected_eval' for c in clips):
        raise ValueError('wrong partition in parent rows')
    # Round half up, then Hamilton largest remainders across dataset/task strata.
    target = max(1, (2 * len(clips) * fraction_numerator + fraction_denominator) // (2 * fraction_denominator))
    strata = [(c['dataset'], c.get('task', 'unspecified')) for c in clips]
    counts = Counter(strata)
    quotas = {s: counts[s] * target // len(clips) for s in sorted(counts)}
    remaining = target - sum(quotas.values())
    for s in sorted(counts, key=lambda s: (-(counts[s] * target % len(clips)), s))[:remaining]:
        quotas[s] += 1
    if any(q == 0 for q in quotas.values()):
        raise ValueError('selection too small to cover every task stratum')
    assets, groups = Counter(), Counter()
    selected = []
    for s, quota in sorted(quotas.items()):
        candidates = [i for i, value in enumerate(strata) if value == s]
        for _ in range(quota):
            # Maximize asset coverage first; prefer less-repeated source groups.
            i = min(candidates, key=lambda i: (
                assets[(s, clips[i]['asset_id'])], groups[clips[i]['source_group']],
                digest([seed, *keys[i]]), keys[i]))
            selected.append(i)
            candidates.remove(i)
            assets[(s, clips[i]['asset_id'])] += 1
            groups[clips[i]['source_group']] += 1
    selected.sort()  # Preserve original motion-bank order for later consumers.
    return selected, [dict(dataset=s[0], task=s[1], parent_clips=counts[s], selected_clips=quotas[s])
                      for s in sorted(counts)]


def create_subset(training_path, parent_path, output, *, seed=20261104):
    training, training_binding = read_parent(training_path, 'train')
    parent, parent_binding = read_parent(parent_path, 'protected_eval')
    indices, strata = choose_clips(training, parent, seed=seed)
    rows = [parent['clips'][i] for i in indices]
    raw = ''.join(json.dumps(c, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n' for c in rows).encode()
    result = dict(schema_version='hoi_protected_eval_subset_v1', name='protect_eval_sm',
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  purpose='development_only; not a substitute for full protected evaluation',
                  parent_bundle=parent_binding, training_bundle=training_binding,
                  seed=seed, target_fraction=0.1,
                  selection_rule='round_half_up; dataset/task largest remainders; asset coverage then source-group diversity; SHA256 seeded tie-break',
                  parent_count=len(parent['clips']), selected_count=len(rows),
                  actual_fraction=len(rows)/len(parent['clips']), strata=strata,
                  selected_assets=len({c['asset_id'] for c in rows}),
                  selected_source_groups=len({c['source_group'] for c in rows}),
                  control_intervals=sum(c['state_frames']-1 for c in rows),
                  full_duration=True, training_allowed=False, original_partition='protected_eval',
                  selected_clips=dict(path='clips.jsonl', sha256=hashlib.sha256(raw).hexdigest()),
                  cases=[dict(parent_clip_index=i, dataset=c['dataset'], clip_id=c['clip_id'],
                              parent_row_sha256=digest(c)) for i, c in zip(indices, rows)])
    result['content_sha256'] = digest(result)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'clips.jsonl').write_bytes(raw)
    (output/'selection.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['# protect_eval_sm', '', f"开发小测固定子集：{len(rows)} / {len(parent['clips'])} 条（{100*result['actual_fraction']:.2f}%），seed={seed}。", '',
             '按 dataset/task 分层分配名额，层内优先覆盖资产、减少重复 source group。未使用 checkpoint 成绩选样。',
             'clips.jsonl 保留原始完整条目及缓存引用；selection.json 绑定原 bundle 文件哈希、行哈希和原 motion-bank 索引。',
             '这是可复用的选择名单，不是独立 bundle；未来评估应加载绑定的原 bundle，再按名单选择完整轨迹。',
             '全部条目仍属于 protected_eval，禁止进入训练。其余保护条目也继续保持保护，不改完整评估日程。',
             '本次只生成名单，未启动训练或评估。开发子集结果不能当作完整 protected_eval 结果。', '',
             '| 数据集 | 原 task | 原条数 | 子集条数 |', '|---|---|---:|---:|']
    lines += [f"| {s['dataset']} | {s['task']} | {s['parent_clips']} | {s['selected_clips']} |" for s in strata]
    lines += ['', f"资产数：{result['selected_assets']}；source group 数：{result['selected_source_groups']}。",
              f"参考时长：{result['control_intervals']/50/60:.2f} 分钟（完整 50 Hz 控制区间）。", '',
              '父 protected_eval：`'+parent_binding['path']+'`', '', '内容 SHA256：`'+result['content_sha256']+'`', '']
    (output/'README.zh-CN.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-bundle', type=Path, required=True)
    p.add_argument('--parent-bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261104)
    a = p.parse_args()
    result = create_subset(a.training_bundle, a.parent_bundle, a.output, seed=a.seed)
    print(json.dumps({k: result[k] for k in ('name', 'parent_count', 'selected_count', 'actual_fraction',
                                           'selected_assets', 'selected_source_groups', 'content_sha256', 'strata')}))


if __name__ == '__main__':
    main()
