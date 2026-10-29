"""No new financial audit: preserve existing parts, then reuse the tested finalizer."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('preflight', 'final'))
    p.add_argument('--plan', type=Path, required=True); p.add_argument('--phase', choices=('joint',), required=True)
    a = p.parse_args(); plan = json.loads(a.plan.read_text()); phase = plan['phases']['joint']; root = a.plan.parent
    if a.stage == 'final':
        if sha(plan['tested_finalizer']) != plan['tested_finalizer_sha256']:
            raise ValueError('Previously tested finalizer changed')
        spec = importlib.util.spec_from_file_location('tested_finalizer', plan['tested_finalizer'])
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); m.main(); return
    repo = Path(phase['repo']); sys.path.insert(0, str(repo / 'src'))
    from sbi4finance.calibration import bank, bank_collection as bc
    if (subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != phase['revision']
            or subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True)):
        raise ValueError('Frozen numerical producer changed')
    collection, record = bc._read_collection(phase['collection'], current=True)
    if (record['collection_id'] != phase['collection_id'] or sha(collection / 'collection.json') != phase['collection_sha256']
            or sha(plan['original_execution']) != plan['original_execution_sha256']):
        raise ValueError('Collection or retained stopped journal changed')
    if sha(phase['original_map']) != plan['original_job_map_sha256']:
        raise ValueError('Original map changed')
    original_pre = json.loads(Path(plan['original_preflight']).read_text())
    if original_pre['status'] != 'passed_preflight' or original_pre['plan_sha256'] != plan['preflight_plan_sha256']:
        raise ValueError('Original source/RNG certificate not valid')
    parts = bc._inventory(collection, record, full=False)  # Hash actual payload bytes without redoing prices/history.
    current = [{'path': str(q['path']), 'split': q['split'], 'start': q['start'], 'stop': q['stop'],
                'manifest_sha256': q['manifest_sha256'], 'array_sha256': q['array_sha256']} for q in parts]
    preserved = json.loads((root / 'preserved-parts.json').read_text())
    if current != preserved or len(current) != 971:
        raise ValueError('Preserved part inventory changed')
    mapping = json.loads(Path(phase['original_map']).read_text())['jobs']
    present = {(q['split'], q['start'], q['stop']) for q in parts}
    missing = [i for i, q in enumerate(mapping) if (q['split'], q['start'], q['stop']) not in present]
    if missing != phase['worker_indices']:
        raise ValueError('Missing original IDs changed; do not duplicate active/completed work')
    for i in missing:
        q = mapping[i]; lock = collection / '.locks' / f"{q['split']}-{q['start']:010d}-{q['stop']:010d}.lock"
        if lock.exists():
            with lock.open('r') as f:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
    destination = root / 'jobs.json'
    with destination.open('xb') as f:
        f.write(Path(phase['original_map']).read_bytes())
    report = {'status': 'passed_preflight', 'plan_sha256': plan['preflight_plan_sha256'],
              'recovery_plan_sha256': sha(a.plan), 'job_map_sha256': sha(destination),
              'collection_id': record['collection_id'], 'source_revision': phase['revision'],
              'source_sha256': bank.source_hashes(record['configuration']['task']), 'runtime': bank.versions(),
              'preserved_parts': 971, 'preserved_inventory_sha256': sha(root / 'preserved-parts.json'),
              'missing_original_indices': missing, 'original_failed_indices': list(range(404, 425)),
              'no_active_overlap_locks': True, 'no_new_historical_seed_scan': True,
              'no_financial_pricing_calls': True, 'no_posterior_evaluation': True}
    temporary = root / 'preflight.json.tmp'; temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(root / 'preflight.json'); print(json.dumps({'status': report['status'], 'missing': len(missing)}), flush=True)


if __name__ == '__main__':
    main()
