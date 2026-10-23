"""Resume upstream training with a fresh replay buffer, without editing scientific sources."""
from pathlib import Path
import ast
import json
import sys


def transform(source):
    tree = ast.parse(source)
    restore = [n for n in tree.body if isinstance(n, ast.If)
               and ast.unparse(n.test) == 'args.resume']
    assert len(restore) == 1, 'Upstream restore block changed'
    at = tree.body.index(restore[0]) + 1
    tree.body[at:at] = ast.parse('_resume_audit(globals())').body
    warmups = headers = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or ast.unparse(node.test) != 'start_it == 0':
            continue
        statement = ast.unparse(node.body[0])
        if statement == 'collect(args.warmup_steps, stats)':
            node.test = ast.parse('start_it == 0 or bool(args.resume)', mode='eval').body
            node.body += ast.parse('_prefill_audit(globals())').body
            warmups += 1
        elif statement.startswith('log_f.write('):
            node.test = ast.parse('log_f.tell() == 0', mode='eval').body
            headers += 1
    assert (warmups, headers) == (1, 1), (warmups, headers)
    return ast.fix_missing_locations(tree)


def assert_same(actual, expected, torch, path='state'):
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, path
        assert torch.equal(actual.detach().cpu(), expected.detach().cpu()), path
        if torch.is_floating_point(actual):
            assert torch.isfinite(actual).all().item(), path
    elif isinstance(expected, dict):
        assert set(actual) == set(expected), path
        for key in expected:
            assert_same(actual[key], expected[key], torch, f'{path}.{key}')
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for index, value in enumerate(expected):
            assert_same(actual[index], value, torch, f'{path}[{index}]')
    else:
        assert actual == expected, (path, actual, expected)


def optimizer_steps(namespace):
    return {key: sorted({float(s['step']) for s in namespace[key].state.values()})
            for key in ('opt_wm', 'opt_pi', 'opt_v')}


def main():
    from run_pipeline import ROOT, write_json, stamp, sha
    from continue_pipeline import RUN, SNAPSHOT, START, END, EXPECTED_SHA
    source = ROOT / 'supermimic/train.py'
    manifest = json.loads((RUN / 'manifest.json').read_text())
    expected_args = manifest['effective_parameters']
    assert not (RUN / 'train_log.csv').exists(), 'Existing training logs must be preserved'
    assert sha(SNAPSHOT) == EXPECTED_SHA

    def audit(ns):
        torch = ns['torch']
        assert ns['args'].resume == str(SNAPSHOT)
        assert (ns['start_it'], ns['args'].iterations) == (START, END)
        assert vars(ns['args']) == expected_args
        checkpoint = ns['ck']
        for key in ('wm', 'pi', 'critic', 'opt_wm', 'opt_pi', 'opt_v'):
            assert_same(ns[key].state_dict(), checkpoint[key], torch, key)
        receipt = {'utc': stamp(), 'iteration': ns['start_it'],
                   'checkpoint_sha256': EXPECTED_SHA, 'exact_models_and_optimizers': True,
                   'optimizer_steps': optimizer_steps(ns), 'sim_steps': ns['sim_steps'],
                   'wall_time': ns['wall_offset'], 'scientific_source_sha256': sha(source),
                   'replay_simulator_rng_restored': False,
                   'next_learning_rate': expected_args['lr'] * (.03 + .97 * .5 *
                       (1 + __import__('math').cos(__import__('math').pi * START / expected_args['lr_decay_iters'])))}
        write_json(RUN / 'restore-audit.json', receipt)
        print('[resume] exact model/optimizer restore verified at iteration 14000', flush=True)
        # Upstream retains the loaded CUDA checkpoint indefinitely. Drop only this
        # redundant copy after equality checks; model and optimizer tensors remain.
        del ns['ck']

    def prefill(ns):
        assert ns['buffer'].size == expected_args['warmup_steps'] == 64
        restore = json.loads((RUN / 'restore-audit.json').read_text())
        assert optimizer_steps(ns) == restore['optimizer_steps'], 'An optimizer updated during replay fill'
        assert ns['sim_steps'] == restore['sim_steps'] + 64 * 4096
        write_json(RUN / 'prefill-audit.json', {
            'utc': stamp(), 'vector_steps': 64, 'transitions': 64 * 4096,
            'optimizer_steps_unchanged': True, 'buffer_size': ns['buffer'].size,
            'normalizers_updated_by_original_collect': True,
            'next_iteration': START + 1, 'repeat_wm_pretraining': False})
        print('[resume] fresh replay filled; continue at iteration 14001', flush=True)

    sys.argv = [str(source), *sys.argv[1:]]
    namespace = {'__name__': '__main__', '__file__': str(source),
                 '_resume_audit': audit, '_prefill_audit': prefill}
    exec(compile(transform(source.read_text()), str(source), 'exec'), namespace)


if __name__ == '__main__':
    main()
