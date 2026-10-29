"""Continue the original bounded training pool using its unchanged frozen worker.

Metadata only: verify native bank completion, bind a new one-route declaration,
submit once, and detach the existing watcher. No producer or worker is modified.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

BASE = Path('/oscar/scratch/afengler')
OLD = BASE / 'sbi4finance-post-bank-baselines-recovery-v2-20261028'
ROOT = BASE / 'sbi4finance-independent-work-20261028/known100k'
CAMPAIGN = BASE / 'sbi4finance-overnight-20261028/followons/known-100k/execution.json'


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value):
    path = Path(path)
    with path.open('x') as output:
        output.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    original_plan, previous = read(OLD / 'plan.json'), read(OLD / 'execution.json')
    if previous['state'] != 'finished' or any(q['state'] != 'completed' for q in previous['routes'].values()):
        raise ValueError('Original training allocations must be reconciled first')
    cpu = sum(q['allocated_cpu_seconds'] for q in previous['routes'].values())
    gpu = sum(q['allocated_gpu_seconds'] for q in previous['routes'].values())
    # One hour, four CPUs, one GPU, plus the original cleanup reservation.
    if cpu + 4 * 3840 > 9 * 3600 or gpu + 3840 > 2.25 * 3600:
        raise ValueError('One full additional allocation exceeds the original pool')
    bank_run = read(CAMPAIGN)
    bank = Path(bank_run['snapshot']); manifest = read(bank / 'manifest.json')
    if (bank_run['state'] != 'completed_checked_snapshot'
            or sha(bank / 'manifest.json') != bank_run['snapshot_manifest_sha256']
            or manifest['configuration']['splits'] != {'train': 100000, 'validation': 5000, 'test': 7000}
            or bank_run['budget']['outstanding_cpu_seconds'] != 0):
        raise ValueError('The declared known100k bank is not checked and settled')
    attempts = bank_run['attempts']
    result = subprocess.run(['sacct', '-S', '2026-10-27', '-j', ','.join(a['job_id'] for a in attempts),
        '-n', '-P', '--format=JobID%64,State%40,ExitCode,ElapsedRaw,AllocCPUS'],
        capture_output=True, text=True, timeout=45, check=True)
    rows = {r[0]: r for line in result.stdout.splitlines() if len(r := line.split('|')) >= 5 and '.' not in r[0]}
    actual = 0
    for attempt in attempts:
        if attempt['state'] != 'submitted': raise ValueError('Ambiguous bank allocation')
        ids = ([attempt['job_id'] + '_' + str(i - attempt['array_offset']) for i in attempt['indices']]
               if attempt['role'] == 'workers' else [attempt['job_id']])
        for identifier in ids:
            row = rows[identifier]
            if row[1].strip().split()[0] != 'COMPLETED' or row[2] != '0:0':
                raise ValueError('Unsettled bank allocation: ' + identifier)
            actual += int(row[3]) * int(row[4])
    if actual != bank_run['budget']['actual_cpu_seconds']:
        raise ValueError('Native bank accounting differs from declared completion')
    repo, python = Path(original_plan['repo']), original_plan['python']
    operator = repo / 'cluster/submit_full_surface_backbones_20261028.py'
    spec = importlib.util.spec_from_file_location('unchanged_post_bank_worker', operator)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.source_check(original_plan)
    ROOT.mkdir(parents=True, exist_ok=False); (ROOT / 'logs').mkdir()
    readiness = {'status': 'completed_checked_snapshot', 'snapshot': str(bank),
        'snapshot_manifest_sha256': sha(bank / 'manifest.json'),
        'split_counts': manifest['configuration']['splits'], 'collection_id': manifest['collection_id'],
        'source_revision': bank_run['producer_git_revision'], 'source_sha256': manifest['source_sha256'],
        'task_fingerprint': manifest['task_fingerprint'],
        'evidence_kind': 'native collection freeze and successful exact allocation accounting',
        'native_execution': str(CAMPAIGN), 'native_execution_sha256': sha(CAMPAIGN),
        'bank_allocated_cpu_seconds': actual, 'scientific_acceptance': False}
    save(ROOT / 'native-readiness.json', readiness)
    route = {**original_plan['routes']['known'], 'bank': str(bank),
             'ready': str(ROOT / 'native-readiness.json'), 'splits': manifest['configuration']['splits']}
    plan = {**original_plan, 'prepared_utc': datetime.now(timezone.utc).isoformat(),
        'root': str(ROOT), 'routes': {'known': route},
        'continuation': {'original_phase': str(OLD), 'original_execution_sha256': sha(OLD / 'execution.json'),
            'prior_cpu_seconds': cpu, 'prior_gpu_seconds': gpu,
            'original_cpu_seconds_ceiling': 9 * 3600, 'original_gpu_seconds_ceiling': 2.25 * 3600,
            'one_new_full_cpu_seconds': 4 * 3840, 'one_new_full_gpu_seconds': 3840,
            'no_additional_training_allowance': True},
        'operator_note': 'Reuse the unchanged c96a785 worker, sampler and 500-epoch final selection; one route only.'}
    save(ROOT / 'plan.json', plan)
    ledger = {'format': 'full_surface_post_bank_execution_v1', 'state': 'waiting_for_banks',
        'plan_sha256': sha(ROOT / 'plan.json'), 'routes': {'known': {'state': 'waiting'}},
        'scientific_acceptance': False, 'reserved_test_evaluated': False}
    save(ROOT / 'execution.json', ledger)
    save(ROOT / 'launch-attempt.json', {'source_sha256': sha(__file__), 'operator_sha256': sha(operator),
        'plan_sha256': sha(ROOT / 'plan.json'), 'automatic_retry': False})
    module.submit(plan, ledger, 'known')
    if ledger['routes']['known']['state'] != 'submitted':
        raise RuntimeError('Submission not confirmed; reconcile receipt before any retry')
    with (ROOT / 'watcher.log').open('ab') as log:
        watcher = subprocess.Popen([python, '-u', str(operator), 'watch', '--root', str(ROOT)],
            cwd=repo, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {'job_id': ledger['routes']['known']['job_id'], 'watcher_pid': watcher.pid,
        'root': str(ROOT), 'prior_cpu_seconds': cpu, 'prior_gpu_seconds': gpu,
        'full_cumulative_cpu_seconds': cpu + 4 * 3840, 'full_cumulative_gpu_seconds': gpu + 3840}
    save(ROOT / 'launch.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__': main()
