"""Append continuation CSV metrics to the existing ON cloud run, never upload artifacts."""
import csv
import json
import math
import os
import time


def rows(path):
    if not path.exists():
        return []
    text = path.read_text()
    if not text.endswith('\n'):
        text = text.rsplit('\n', 1)[0] + '\n'
    return list(csv.DictReader(text.splitlines()))


def main():
    import wandb
    from run_pipeline import write_json, stamp
    from continue_pipeline import RUN, START, RUN_ID, CLOUD_PATH
    os.environ['WANDB_BASE_URL'] = 'https://api.forge.coreweave.com'
    # Require the existing run. A typo or missing run must never create a new one.
    cloud = wandb.Api(timeout=30).run(CLOUD_PATH)
    assert cloud.id == RUN_ID
    assert int(cloud.summary['latest_full40_iteration']) >= START
    run = wandb.init(entity='tianzizh-university-of-pennsylvania',
        project='SuperMimic_Upstream_Reproduction', id=RUN_ID, resume='must',
        save_code=False, mode='online', settings=wandb.Settings(disable_code=True),
        dir=str(RUN))
    assert run.id == RUN_ID and run.resumed
    manifest = json.loads((RUN / 'manifest.json').read_text())
    run.config.update({'continuation_14000_to_24000': {
        key: manifest[key] for key in ('continuation', 'effective_parameters', 'resume_limitations', 'hardware')}},
        allow_val_change=True)
    run.summary['continuation/status'] = 'running'
    run.summary['pipeline_status'] = 'running'
    run.define_metric('train/iteration')
    run.define_metric('train/*', step_metric='train/iteration')
    run.define_metric('eval/iteration')
    run.define_metric('eval/*', step_metric='eval/iteration')
    state = RUN / 'wandb-cursor.json'
    cursor = json.loads(state.read_text()) if state.exists() else {'train': START, 'eval': START}
    write_json(RUN / 'wandb-resumed.json', {'utc': stamp(), 'id': run.id, 'url': run.url,
        'resumed': bool(run.resumed), 'existing_cloud_iteration': int(cloud.summary.get('train/iteration', START))})

    def once():
        for row in rows(RUN / 'train_log.csv'):
            iteration = int(row['iteration'])
            if iteration <= cursor['train']:
                continue
            values = {f'train/{k}': float(v) for k, v in row.items()}
            assert all(math.isfinite(v) for v in values.values()), 'Nonfinite training metric'
            values['train/seconds'] = values['train/wall_h'] * 3600
            run.log(values)
            cursor['train'] = iteration
        for row in rows(RUN / 'eval.csv'):
            iteration = int(row['iteration'])
            if iteration <= cursor['eval']:
                continue
            values = {f'eval/{k}': float(row[k]) for k in
                ('iteration', 'train_wall_h', 'norm_return', 'norm_return_framew', 'n_failed')}
            assert all(math.isfinite(v) for v in values.values())
            values['eval/completion_fraction'] = 1 - values['eval/n_failed'] / 40
            values['eval/target_reproduced'] = values['eval/norm_return'] >= .82 and values['eval/n_failed'] == 0
            values['eval/training_seconds'] = values['eval/train_wall_h'] * 3600
            run.log(values)
            run.summary['latest_full40_iteration'] = iteration
            run.summary['latest_full40_norm_return'] = values['eval/norm_return']
            run.summary['latest_full40_failed_clips'] = int(values['eval/n_failed'])
            cursor['eval'] = iteration
        write_json(state, cursor)

    while True:
        once()
        if (RUN / 'pipeline.json').exists():
            break
        time.sleep(10)
    once()
    outcome = json.loads((RUN / 'pipeline.json').read_text())
    run.summary['continuation/status'] = outcome['status']
    run.summary['pipeline_status'] = outcome['status']
    if (RUN / 'acceptance.json').exists():
        for key, value in json.loads((RUN / 'acceptance.json').read_text()).items():
            if isinstance(value, (bool, int, float)):
                run.summary['acceptance/' + key] = value
    run.finish(exit_code=0 if outcome['status'] == 'completed' else 1)
    cloud = wandb.Api(timeout=30).run(CLOUD_PATH)
    assert int(cloud.summary.get('train/iteration', 0)) == cursor['train']
    assert int(cloud.summary['latest_full40_iteration']) == cursor['eval']
    write_json(RUN / 'wandb-sync-complete.json', {'utc': stamp(), 'cursor': cursor,
        'url': cloud.url, 'cloud_readback': True})


if __name__ == '__main__':
    main()
