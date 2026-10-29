"""Exercise real native snapshot freeze/replay on tiny mocked-price banks."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'src'))
from sbi4finance.calibration import bank_collection as bc, tasks


def module(name):
    p = Path(__file__).with_name(name + '.py')
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def exercise(name, joint):
    m = module(name)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary); collection = root / 'collection'; output = root / 'phase'; output.mkdir()
        task = tasks.joint_rough_task() if joint else tasks.rough_task()
        config = {'task': task, 'root_seed': 777, 'splits': {'train': 2, 'validation': 1, 'test': 1}}
        record = bc.create_collection(collection, config)
        def cheap_price(task, parameters, context, seed):
            n = len(task['context']['contracts'])
            return np.full(n, .01), np.zeros(n), np.zeros((n, n)), np.zeros(n), np.full(3, task['numerics']['paths_ladder'][0])
        with patch.object(tasks, 'price', side_effect=cheap_price):
            for job in bc.plan_extension(collection, 2, 1)['jobs']:
                bc.run_job(collection, job)
        _, current = bc._read_collection(collection, current=True)
        parts = bc._inventory(collection, current)
        inventory = [{'split': q['split'], 'start': q['start'], 'stop': q['stop'], 'path': str(q['path']),
                      'manifest_sha256': q['manifest_sha256'], 'array_sha256': q['array_sha256']} for q in parts]
        (output / 'preserved-parts.json').write_text(json.dumps(inventory[:1]))
        (output / 'jobs.json').write_text('{}')
        (output / 'preflight.json').write_text(json.dumps({'status': 'passed_preflight', 'job_map_sha256': m.sha(output / 'jobs.json')}))
        phase = {'root': str(output), 'repo': str(REPO), 'collection': str(collection), 'revision': 'synthetic',
                 'collection_id': record['collection_id'], 'collection_sha256': m.sha(collection / 'collection.json'),
                 'root_seed': 777, 'train_target': 2, 'splits': config['splits'], 'snapshot': str(root / 'snapshot'),
                 'numerical_scope': 'Synthetic mocked prices only; no financial simulator calls.'}
        plan = root / 'plan.json'; label = 'joint' if joint else 'known'
        plan.write_text(json.dumps({'phases': {label: phase}}))
        freezes = []
        def freeze(argv, **kwargs):
            assert argv[2] == 'freeze'
            result = bc.freeze_snapshot(Path(argv[argv.index('--collection') + 1]),
                                        int(argv[argv.index('--train') + 1]), Path(argv[argv.index('--out') + 1]))
            freezes.append(result)
        with patch.object(sys, 'argv', ['check', 'final', '--plan', str(plan), '--phase', label]), \
             patch.object(m.subprocess, 'check_output', side_effect=lambda argv, **kw: 'synthetic\n' if 'rev-parse' in argv else ''), \
             patch.object(m.subprocess, 'run', side_effect=freeze):
            try:
                m.main()
            except ValueError as error:
                if name == 'bank_delivery_stage' and joint and 'joint curve is derived' in str(error):
                    assert len(freezes) == 1
                    return {'expected_before_fix_failure': str(error), 'real_native_freeze_exercised': True}
                raise
        result = json.loads((output / 'final.json').read_text())
        assert result['status'] == 'completed_checked_snapshot' and result['split_counts'] == config['splits']
        assert result['exact_prior_context_noise_replay'] and len(freezes) == 1
        return {'status': result['status'], 'real_native_freeze_exercised': True, 'no_financial_price_calls': True}


if __name__ == '__main__':
    results = {'before_fix_joint': exercise('bank_delivery_stage', True),
               'after_fix_known': exercise('bank_delivery_stage_v2', False),
               'after_fix_joint': exercise('bank_delivery_stage_v2', True)}
    Path(__file__).with_name('finalizer-checks.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results))
