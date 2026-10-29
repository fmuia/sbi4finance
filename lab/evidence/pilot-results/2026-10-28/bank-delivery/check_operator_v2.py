"""No scheduler calls: exercise prospective budgets and atomic submission receipts."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import patch

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('controller_v2', HERE / 'bank_delivery_controller_v2.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
original = json.loads((HERE / 'plan.json').read_text())
checks = {}
for name, phase in original['phases'].items():
    b = m.exposure(phase, [], {})
    cost = b['exposure_cpu_seconds'] + phase['max_workers'] * (phase['worker_wall_seconds'] + 240)
    assert cost < phase['cpu_hour_ceiling'] * 3600
    checks[name] = {'prospective_full_wave_cpu_hours': cost / 3600, 'cap_cpu_hours': phase['cpu_hour_ceiling']}
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary); plan = json.loads(json.dumps(original)); plan['preflight_plan_sha256'] = 'old-frozen-plan'
    for name, phase in plan['phases'].items():
        phase['root'] = str(root / name); Path(phase['root']).mkdir()
        (Path(phase['root']) / 'jobs.json').write_text('{}')
        (Path(phase['root']) / 'preflight.json').write_text(json.dumps({'status': 'passed_preflight',
            'plan_sha256': 'old-frozen-plan', 'job_map_sha256': m.sha(Path(phase['root']) / 'jobs.json')}))
    ledger = {'plan_sha256': 'new-reviewed-plan', 'phases': {name: {'jobs': []} for name in plan['phases']}}
    calls = []
    def fake_run(argv, **kwargs):
        receipt = json.loads((root / 'execution-v2.json').read_text())
        assert receipt['phases']['known']['jobs'][-1]['state'] == 'submission_attempted'
        assert kwargs['cwd'] == plan['phases']['known']['repo']
        assert '--array=0,1%2' in argv
        calls.append(argv)
        return SimpleNamespace(stdout='12345\n', stderr='', returncode=0)
    with patch.object(m, 'accounting', return_value={}), patch.object(m.subprocess, 'run', side_effect=fake_run):
        m.submit(root, root / 'plan-v2.json', plan, ledger, 'known', 'workers', [0, 1])
    assert len(calls) == 1 and ledger['phases']['known']['jobs'][0]['job_id'] == '12345'
    assert ledger['phases']['known']['jobs'][0]['state'] == 'submitted'
    try:
        with patch.object(m, 'accounting', return_value={}):
            m.submit(root, root / 'plan-v2.json', plan, ledger, 'known', 'workers', [0, 1])
    except RuntimeError as error:
        assert 'Duplicate/recovery' in str(error)
    else:
        raise AssertionError('Duplicate submission must fail')
    known_pre = root / 'known/preflight.json'
    changed = json.loads(known_pre.read_text()); changed['status'] = 'running'; known_pre.write_text(json.dumps(changed))
    def joint_run(argv, **kwargs):
        assert kwargs['cwd'] == plan['phases']['joint']['repo']
        assert '--array=397,398%2' in argv
        return SimpleNamespace(stdout='23456\n', stderr='', returncode=0)
    with patch.object(m, 'accounting', return_value={}), patch.object(m.subprocess, 'run', side_effect=joint_run):
        m.submit(root, root / 'plan-v2.json', plan, ledger, 'joint', 'workers', [397, 398])
        try:
            m.submit(root, root / 'plan-v2.json', plan, ledger, 'known', 'workers', [2])
        except RuntimeError as error:
            assert 'This phase preflight' in str(error)
        else:
            raise AssertionError('Known must wait for its own preflight')
    phase = plan['phases']['known']
    job = {'job_id': '10', 'state': 'submitted', 'role': 'workers', 'indices': [0, 1], 'wall_seconds': 2400}
    b = m.exposure(phase, [job], {'10_0': {'state': 'COMPLETED', 'exit_code': '0:0', 'elapsed_seconds': 200, 'cpus': 1}})
    assert b['actual_allocated_cpu_seconds'] == 200 and b['outstanding_full_cpu_seconds'] == 2640
    for jobs, rows in [([{**job, 'state': 'unknown_submission'}], {}),
                       ([job], {'10_0': {'state': 'TIMEOUT', 'exit_code': '0:0', 'elapsed_seconds': 2400, 'cpus': 1}})]:
        try:
            m.exposure(phase, jobs, rows)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Unknown/failed state must stop')
checks['checks'] = ['full wave exposure within both caps', 'receipt persisted before fake sbatch',
                    'submission cwd is frozen producer', 'no real scheduler call', 'duplicate indices rejected',
                    'joint PASS advances independently while known RUNNING stays gated',
                    'completed actual counted once; unknown reserves full wall+cleanup',
                    'ambiguous submission and timeout fail closed']
(HERE / 'operator-v2-checks.json').write_text(json.dumps(checks, indent=2) + '\n')
print(json.dumps(checks))
