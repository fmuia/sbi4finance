"""Read-only diagnosis of retained joint-bank attempts and current coverage."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/oscar/scratch/afengler/sbi4finance-bank-delivery-20261028')
OUT = ROOT / 'morning-recovery-20261028'
OUT.mkdir(exist_ok=True)
ledger = json.loads((ROOT / 'execution-v2.json').read_text())
plan = json.loads((ROOT / 'plan-v2.json').read_text())
phase = plan['phases']['joint']; jobs = ledger['phases']['joint']['jobs']
identifiers = [j['job_id'] for j in jobs]
r = subprocess.run(['sacct', '-S', '2026-10-27', '-j', ','.join(identifiers), '-n', '-P',
                    '--format=JobID%64,State%40,ElapsedRaw,AllocCPUS,ExitCode,NodeList%80,Start,End,TimelimitRaw,MaxRSS'],
                   text=True, capture_output=True, check=True, timeout=60)
(OUT / 'submitted-accounting.psv').write_text(r.stdout)
rows, steps = {}, {}
for line in r.stdout.splitlines():
    a = line.split('|')
    if len(a) < 10: continue
    row = dict(zip(('job_id', 'state', 'elapsed_seconds', 'cpus', 'exit_code', 'nodes', 'start', 'end',
                    'time_limit_minutes', 'max_rss'), a[:10]))
    (steps if '.' in a[0] else rows)[a[0]] = row
parts = []
for p in sorted((Path(phase['collection']) / 'parts').glob('*/manifest.json')):
    record = json.loads(p.read_text())
    parts.append({'path': str(p.parent), 'split': record['split'], 'start': record['start'], 'stop': record['stop'],
                  'manifest_sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'array_sha256': record['sha256']})
present = {(p['split'], p['start'], p['stop']) for p in parts}
mapping = json.loads((ROOT / 'joint/jobs.json').read_text())['jobs']
missing = [i for i, p in enumerate(mapping) if (p['split'], p['start'], p['stop']) not in present]
attempts, failures = [], []
spent = 0; outstanding = 0
for job in jobs:
    for i in job.get('indices', [None]):
        key = job['job_id'] + ('_' + str(i) if i is not None else '')
        row = rows.get(key)
        if row:
            record = {**row, 'role': job['role'], 'map_index': i}
            if i is not None: record['range'] = mapping[i]
            attempts.append(record)
            state = row['state'].split()[0]
            if state in ('COMPLETED', 'TIMEOUT', 'FAILED', 'CANCELLED', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED'):
                spent += int(row['elapsed_seconds']) * int(row['cpus'])
            else: outstanding += job['wall_seconds'] + 240
            if state != 'COMPLETED' or row['exit_code'] != '0:0':
                if state not in ('RUNNING', 'PENDING', 'COMPLETING', 'CONFIGURING'):
                    failures.append(record)
        else:
            outstanding += job['wall_seconds'] + 240
            attempts.append({'job_id': key, 'state': 'UNKNOWN', 'role': job['role'], 'map_index': i})
for failure in failures:
    i = failure['map_index']; job_id = failure['job_id']
    if i is None: continue
    parent = job_id.rsplit('_', 1)[0]
    log = ROOT / 'joint/logs' / f'workers-{parent}_{i}.out'
    if log.exists():
        failure['log_path'] = str(log); failure['log_sha256'] = hashlib.sha256(log.read_bytes()).hexdigest()
        failure['log_tail'] = log.read_text(errors='replace').splitlines()[-25:]
    job = mapping[i]; name = f"{job['split']}-{job['start']:010d}-{job['stop']:010d}"
    retained = []
    for location in ('.tmp', 'failures'):
        for p in (Path(phase['collection']) / location).glob(name + '*'):
            contents = [{'path': str(q.relative_to(p)), 'bytes': q.stat().st_size,
                         'sha256': hashlib.sha256(q.read_bytes()).hexdigest() if q.is_file() else None}
                        for q in p.rglob('*') if q.is_file()]
            retained.append({'path': str(p), 'files': contents})
    failure['retained_partial_work'] = retained
    failure['step_accounting'] = [v for k, v in steps.items() if k.startswith(job_id + '.')]
summary = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'read_only': True,
           'controller_status': ledger['status'], 'controller_error': ledger.get('error'),
           'published_parts': len(parts), 'published_surfaces_by_split': dict(Counter({s: sum(p['stop'] - p['start'] for p in parts if p['split'] == s)
                                                                                   for s in ('train', 'validation', 'test')})),
           'submitted_attempts': len(attempts), 'states': dict(Counter(p['state'] for p in attempts)),
           'failed_attempts': failures, 'missing_original_map_indices': missing,
           'new_completed_or_failed_allocated_cpu_seconds': spent,
           'prior_allocated_cpu_seconds': phase['prior_allocated_cpu_seconds'],
           'cumulative_allocated_cpu_seconds': spent + phase['prior_allocated_cpu_seconds'],
           'outstanding_full_wall_cleanup_cpu_seconds': outstanding,
           'cap_cpu_seconds': 1500 * 3600, 'final_checks_reserved_cpu_seconds': 7200,
           'remaining_after_outstanding_and_checks_cpu_seconds': 1500 * 3600 - spent - phase['prior_allocated_cpu_seconds'] - outstanding - 7200,
           'attempts': attempts}
(OUT / 'inspection.json').write_text(json.dumps(summary, indent=2) + '\n')
(OUT / 'preserved-parts.json').write_text(json.dumps(parts, indent=2) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k not in ('attempts', 'failed_attempts')}))
print(json.dumps({'failures': [{k: f.get(k) for k in ('job_id', 'map_index', 'state', 'nodes', 'elapsed_seconds', 'log_tail', 'retained_partial_work')}
                               for f in failures]}))
