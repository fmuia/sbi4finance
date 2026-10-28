"""Collect terminal metadata only; never retry a job or evaluate model/test data."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/oscar/scratch/afengler/sbi4finance-heston-acceptance-20261028')
JOB = '8032275'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    accounting = subprocess.check_output(['sacct', '-j', JOB, '--parsable2', '--noheader',
        '--format=JobIDRaw,State,ElapsedRaw,AllocCPUS,AllocTRES,ExitCode'], text=True)
    rows = [line.split('|') for line in accounting.splitlines() if line]
    row = next(row for row in rows if row[0] == JOB)
    state = row[1].split()[0]
    if state in ('RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING'):
        raise ValueError('Job is not terminal; no ledger reconciliation yet')
    (ROOT / 'accounting.psv').write_text('JobIDRaw|State|ElapsedRaw|AllocCPUS|AllocTRES|ExitCode\n' + accounting)
    ledger = json.loads((ROOT / 'execution.json').read_text())
    if ledger['job_id'] != JOB:
        raise ValueError('Different execution ledger job')
    if not (ROOT / 'execution-submitted.json').exists():
        (ROOT / 'execution-submitted.json').write_bytes((ROOT / 'execution.json').read_bytes())
    elapsed, cpus = int(row[2]), int(row[3])
    if cpus != 4 or 'gres/gpu=1' not in row[4]:
        raise ValueError('Actual allocation differs from the one-GPU/four-CPU declaration')
    ledger.update(state=state.lower(), terminal_utc=datetime.now(timezone.utc).isoformat(),
                  slurm_state=state, slurm_exit_code=row[5], actual_elapsed_seconds=elapsed,
                  actual_allocated_cpu_seconds=elapsed * cpus, actual_gpu_seconds=elapsed)
    write(ROOT / 'execution.json', ledger)
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    verified = {name: sha(Path(name)) == expected for name, expected in manifest['frozen_sha256'].items()}
    if not all(verified.values()) or sha(ROOT / 'manifest.json') != ledger['manifest_sha256']:
        raise ValueError('Frozen declaration or input bytes changed')
    small = [p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.json', '.py', '.psv', '.out')
             and p.name not in ('remote-artifacts.json', 'summary.json')
             and 'cache' not in p.relative_to(ROOT).parts]
    large = [p for p in (ROOT / 'results').rglob('*.npz')]
    artifacts = {'small_files': {str(p.relative_to(ROOT)): {'path': str(p), 'bytes': p.stat().st_size,
                 'sha256': sha(p)} for p in sorted(small)},
                 'large_payloads_not_downloaded': {str(p.relative_to(ROOT)): {'path': str(p),
                 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(large)},
                 'frozen_inputs_rechecked': verified}
    write(ROOT / 'remote-artifacts.json', artifacts)
    evaluation_path = ROOT / 'results/acceptance/evaluation.json'
    product_path = ROOT / 'results/products/products.json'
    evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else None
    products = json.loads(product_path.read_text()) if product_path.exists() else None
    statuses = [] if products is None else [quantile['status'] for case in products['cases']
        for product in case['products'].values() for quantile in product['quantiles']]
    summary = {'job_id': JOB, 'slurm_state': state, 'exit_code': row[5],
               'actual_allocated_cpu_seconds': elapsed * cpus, 'actual_gpu_seconds': elapsed,
               'sbc': None if evaluation is None else evaluation['sbc'],
               'coverage': None if evaluation is None else evaluation['coverage'],
               'rank_screen_passed': None if evaluation is None else evaluation['rank_screen_passed'],
               'product_comparisons': {name: statuses.count(name) for name in sorted(set(statuses))},
               'frozen_input_hashes_verified': len(verified), 'scientific_acceptance': False}
    write(ROOT / 'summary.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
