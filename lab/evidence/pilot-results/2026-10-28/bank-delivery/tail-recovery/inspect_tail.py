"""Read-only reconciliation for the two reviewed original joint ranges."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BASE = Path('/oscar/scratch/afengler/sbi4finance-bank-delivery-20261028')
MORNING = BASE / 'morning-recovery-20261028'
PRODUCER = Path('/oscar/scratch/afengler/sbi4finance-native-bank-20261027/repo')
COLLECTION = PRODUCER.parent / 'collection'
sys.path.insert(0, str(PRODUCER / 'src'))
from sbi4finance.calibration import bank_collection as bc


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


execution = json.loads((MORNING / 'execution.json').read_text())
plan = json.loads((MORNING / 'recovery-plan.json').read_text())
assert execution['status'] == 'stopped_for_review'
assert sha(BASE / 'execution-v2.json') == plan['original_execution_sha256']
assert sha(MORNING / 'recovery-plan.json') == execution['plan_sha256']
jobs = execution['phases']['joint']['jobs']
assert all(j['state'] == 'submitted' for j in jobs)
expected = {j['job_id'] + ('_' + str(i) if i is not None else ''): i
            for j in jobs for i in j.get('indices', [None])}
result = subprocess.run(['sacct', '-S', '2026-10-27', '-j', ','.join(j['job_id'] for j in jobs),
                         '-n', '-P', '--format=JobID%64,State%40,ElapsedRaw,AllocCPUS,ExitCode,NodeList%80'],
                        capture_output=True, text=True, timeout=45, check=True)
rows = {}
for line in result.stdout.splitlines():
    fields = line.split('|')
    if len(fields) >= 6 and fields[0] in expected:
        rows[fields[0]] = dict(job_id=fields[0], state=fields[1].split()[0],
                               elapsed_seconds=int(fields[2]), cpus=int(fields[3]),
                               exit_code=fields[4], node=fields[5], original_map_index=expected[fields[0]])
assert set(rows) == set(expected), sorted(set(expected) - set(rows))
failed = [r for r in rows.values() if r['state'] != 'COMPLETED' or r['exit_code'] != '0:0']
assert sorted(r['original_map_index'] for r in failed) == [997, 1000], failed
assert all(r['state'] == 'TIMEOUT' and r['node'] == 'node2346' and r['cpus'] == 1 for r in failed)
assert len(rows) == 310
original_map = json.loads(Path(plan['phases']['joint']['original_map']).read_text())
assert sha(plan['phases']['joint']['original_map']) == plan['original_job_map_sha256']
native = bc.plan_extension(COLLECTION, 20000, 25)
assert len(native['jobs']) == 2
selected = [original_map['jobs'][i] for i in [997, 1000]]
for got, old in zip(native['jobs'], selected):
    assert all(got[k] == old[k] for k in ('split', 'start', 'stop', 'job_sha256')), (got, old)
record = json.loads((COLLECTION / 'collection.json').read_text())
inventory = []
for part in sorted((COLLECTION / 'parts').iterdir()):
    manifest = json.loads((part / 'manifest.json').read_text())
    inventory.append(dict(path=str(part), split=manifest['split'], start=manifest['start'],
                          stop=manifest['stop'], manifest_sha256=sha(part / 'manifest.json'),
                          array_sha256=manifest['sha256']))
assert len(inventory) == 1278
counts = Counter()
for part in inventory:
    counts[part['split']] += part['stop'] - part['start']
prior = plan['phases']['joint']['prior_allocated_cpu_seconds']
current = sum(r['elapsed_seconds'] * r['cpus'] for r in rows.values())
assert prior == 2384586
output = dict(format='joint_two_range_reconciliation_v1', status='passed',
              checked_utc=datetime.now(timezone.utc).isoformat(), read_only=True,
              original_execution_sha256=plan['original_execution_sha256'],
              morning_execution=str(MORNING / 'execution.json'), morning_execution_sha256=sha(MORNING / 'execution.json'),
              morning_plan_sha256=sha(MORNING / 'recovery-plan.json'),
              original_job_map_sha256=plan['original_job_map_sha256'],
              prior_original_allocated_cpu_seconds=prior, morning_allocated_cpu_seconds=current,
              cumulative_allocated_cpu_seconds=prior + current, outstanding_allocations=0,
              states=dict(Counter(r['state'] for r in rows.values())), failed_attempts=failed,
              terminal_accounting=list(rows.values()), collection_id=record['collection_id'],
              collection_sha256=sha(COLLECTION / 'collection.json'), runtime=record['versions'],
              producer_source_sha256=record['source_sha256'], preserved_parts=len(inventory),
              preserved_split_counts=dict(counts), preserved_inventory=inventory,
              native_plan=native, original_map_indices=[997, 1000],
              snapshot_exists=Path(plan['phases']['joint']['snapshot']).exists())
assert not output['snapshot_exists']
print(json.dumps(output, indent=2, allow_nan=False))
