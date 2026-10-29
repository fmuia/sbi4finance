"""Metadata-only freeze and one-shot submission; never opens bank arrays."""
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path('/oscar/scratch/afengler/sbi4finance-heston-acceptance-20261028')
REPO = Path('/oscar/scratch/afengler/sbi4finance-rough-known-reference-20261027/repo')
CHECKPOINT = Path('/oscar/scratch/afengler/sbi4finance-heston-100k-fit-20261027/runs/heston-100k-native/checkpoint')
BANK = Path('/oscar/scratch/afengler/sbi4finance-heston-backbone-20261027/snapshot-100k')
ENV = Path('/oscar/scratch/afengler/sbi4finance-full-core-20261026-wy2_j3o4/repo/.pixi/envs')
REVISION = '56dcb13e1d2bf5b6dc96e573dfcaa6240a7104a9'
CANDIDATE_SHA = '8d4a243ba099524cf4cda40e4f0616e895879f053829a7f60dadbe65b98301db'
BANK_SHA = '4b5929b0d0b23698c340eec6defabfd629fe23eca2ed78fe9bd46e4de35c244e'
NEURAL_SHA = ('328e620550243043109bd500329f59208b1ea7fb2f33258a6ae86c518f80812c',
              '1c2d6dd261227b29d9714aea504bc623b6d7c0bff9ebee4e93e1c4f9267654b6',
              'b627a304c4cfdaffe08bdcf3f95402041ef279246ac6f6b9adc7c25818253fe4')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    driver = ROOT / 'check_heston_candidate_20261028.py'
    source_revision = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if source_revision != REVISION or dirty:
        raise ValueError('Frozen source revision differs or is modified')
    resume = json.loads((CHECKPOINT / 'resume.json').read_text())
    training = json.loads((CHECKPOINT / 'training.json').read_text())
    bank_manifest = json.loads((BANK / 'manifest.json').read_text())
    task = json.loads((CHECKPOINT / 'task.json').read_text())
    candidate = CHECKPOINT / resume['checkpoint_files']['last']
    if (resume['epochs_done'] != 500 or sha(candidate) != CANDIDATE_SHA or sha(BANK / 'manifest.json') != BANK_SHA
            or training['run_contract']['dataset']['manifest_sha256'] != BANK_SHA
            or bank_manifest['configuration']['task'] != task
            or bank_manifest['configuration']['root_seed'] != 271028):
        raise ValueError('Candidate, bank, source or task identity differs')
    journals = []
    journal_roots = [str(path) for path in sorted(Path(
        '/oscar/scratch/afengler/sbi4finance-full-core-20261026-wy2_j3o4/runs').glob('heston-*'))] + [
        '/oscar/scratch/afengler/sbi4finance-backbone-fit-20261027/runs/heston-50k-native',
        '/oscar/scratch/afengler/sbi4finance-heston-100k-fit-20261027/runs/heston-100k-native',
        '/oscar/scratch/afengler/sbi4finance-reviewed-unit-20261027-9b495cf/runs/heston',
        '/oscar/scratch/afengler/sbi4finance-heston-audit-20261027/results',
    ]
    for directory in journal_roots:
        for path in sorted(Path(directory).rglob('*.json')):
            if path.name not in ('run.json', 'evaluation.json', 'reference.json'):
                continue
            record = json.loads(path.read_text())
            ids = record.get('example_ids', [])
            reserved = record.get('reserved_test_evaluated') is True
            test_split = record.get('evaluated_split') == 'test'
            test_ids = any(str(x).startswith('test:') for x in ids)
            surface_id = str(record.get('surface_id', ''))
            if reserved or test_split or test_ids or surface_id.startswith('test:'):
                raise ValueError('Retained journal records prior test performance access: ' + str(path))
            journals.append({'path': str(path), 'sha256': sha(path),
                             'evaluated_split': record.get('evaluated_split'),
                             'reserved_test_evaluated': record.get('reserved_test_evaluated'),
                             'surface_id': record.get('surface_id'), 'example_count': len(ids)})
    if not journals:
        raise ValueError('No retained journals found for data-use audit')
    development_cases = []
    cached_run = Path('/oscar/scratch/afengler/sbi4finance-heston-audit-20261027/results/new/run.json')
    cached_record = json.loads(cached_run.read_text())
    if (cached_record.get('status') != 'completed' or len([row for row in cached_record['cases']
            if row['fit'] == '100k' and row['final_checkpoint_sha256'] == CANDIDATE_SHA]) != 3):
        raise ValueError('Cached final neural draws do not bind to frozen candidate')
    frozen_paths = [candidate, BANK / 'manifest.json', driver, Path(__file__),
                    cached_run,
                    *[CHECKPOINT / n for n in ('task.json', 'training.json', 'resume.json')],
                    REPO / 'pixi.lock', REPO / 'pyproject.toml']
    for name in ('full_surface.py', 'inference.py', 'validation.py', 'variance_functionals.py',
                 'bank.py', 'bank_collection.py', 'tasks.py', 'pricing.py', 'surface.py'):
        frozen_paths.append(REPO / 'src/sbi4finance/calibration' / name)
    for case in range(3):
        reference = Path(f'/oscar/scratch/afengler/sbi4finance-reviewed-unit-20261027-9b495cf/runs/heston/case-{case}/reference')
        neural = Path(f'/oscar/scratch/afengler/sbi4finance-heston-audit-20261027/results/new/100k-case-{case}.npz')
        if sha(neural) != NEURAL_SHA[case]:
            raise ValueError('Previously audited cached neural draws changed')
        development_cases.append({'case': case, 'reference': str(reference), 'neural_draws': str(neural)})
        frozen_paths.extend([neural, *[reference / n for n in ('reference.json', 'diagnostics.json', 'chains.npz')]])
    manifest = {'format': 'heston_candidate_acceptance_declaration_v1', 'prepared_utc': datetime.now(timezone.utc).isoformat(),
                'root': str(ROOT), 'repo': str(REPO), 'source_revision': REVISION,
                'candidate': str(candidate), 'candidate_sha256': CANDIDATE_SHA,
                'checkpoint_directory': str(CHECKPOINT), 'bank': str(BANK), 'bank_sha256': BANK_SHA, 'task': task,
                'selection_rule': 'Use the already selected final500 development candidate once, without acceptance-based model selection.',
                'scientific_acceptance': False, 'three_seed_confirmation': False,
                'sampler': {'method': 'tsit5', 'steps': 'adaptive'},
                'acceptance': {'split': 'test', 'example_ids': [f'test:{i}' for i in range(2000)],
                               'draws_per_surface': 1000, 'batch_size': 100, 'batch_seed': '280028 + first row index',
                               'rank_randomization_seed': 280028, 'family_alpha': .05,
                               'rank_dkw_bound': math.sqrt(math.log(8 / .05) / 4000),
                               'primary_gate': 'All four parameter marginal KS statistics at most the simultaneous DKW bound.',
                               'coverage': '50/80/90/95 percent with pointwise Wilson intervals, descriptive not extra pass/fail gates.',
                               'secondary_metrics': ['logit contraction', 'median recovery correlation'],
                               'partial_results': 'Commit every100-case batch; preserve failures and no automatic retries.'},
                'products': {'names': ['variance_swap_1y', 'forward_variance_0_5_to_1y'],
                             'epsilon_annual_variance': .0001, 'probabilities': [.1, .9],
                             'tolerance': 'T=max(.10*(referenceq95-referenceq05),epsilon)',
                             'uncertainty': '1.95996398454*hypot(chain-aware reference quantile MCSE, neural quantile MCSE)',
                             'within': 'absolute error+uncertainty<=T and uncertainty<=T/2',
                             'resolved_excess': 'absolute error-uncertainty>T',
                             'otherwise': 'inconclusive', 'role': 'Pointwise analytic development curve-product check, not independent conditional acceptance.'},
                'development_cases': development_cases, 'reference_python': str(ENV / 'core-reference/bin/python'),
                'data_use_audit': {'journal_roots': journal_roots, 'journals': journals,
                                   'finding': 'No prior test performance evaluation recorded in these retained new-root run journals. '
                                              'Generation/integrity checks did access test arrays; historical old-root test outcomes are separate.',
                                   'test_arrays_opened_by_preflight': False},
                'expected_gpu_runtime': {name: metadata.version(name) for name in ('bayesflow', 'keras', 'jax', 'jaxlib', 'numpy')},
                'frozen_sha256': {str(p): sha(p) for p in frozen_paths}}
    create(ROOT / 'manifest.json', manifest)
    command = ('env PYTHONPATH=' + str(REPO / 'src') + ' KERAS_BACKEND=jax JAX_PLATFORMS=cuda,cpu '
               'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 '
               'XLA_PYTHON_CLIENT_PREALLOCATE=false MPLCONFIGDIR=' + str(ROOT / 'cache/mpl') +
               ' XDG_CACHE_HOME=' + str(ROOT / 'cache') + ' ' + str(ENV / 'core-gpu/bin/python') + ' ' +
               str(driver) + ' --manifest ' + str(ROOT / 'manifest.json'))
    argv = ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '--no-requeue', '--account=default',
            '--partition=gpu-debug', '--qos=gpu-debug', '--cpus-per-task=4', '--gres=gpu:l40s:1',
            '--mem=24G', '--time=00:25:00', '--chdir=' + str(REPO), '--job-name=sbi4f-heston-acceptance',
            '--output=' + str(ROOT / 'logs/slurm-%A.out'), '--wrap=' + command]
    ledger = {'format': 'heston_candidate_acceptance_execution_v1', 'state': 'prospective',
              'prepared_utc': datetime.now(timezone.utc).isoformat(), 'root': str(ROOT), 'argv': argv,
              'manifest_sha256': sha(ROOT / 'manifest.json'), 'driver_sha256': sha(driver),
              'authorization': 'User requests continuing Heston acceptance and rough pricing tracks; one frozen candidate only.',
              'resources': {'cpus': 4, 'gpus': 1, 'wall_seconds': 1500, 'cleanup_seconds': 240,
                            'full_exposure_cpu_seconds': 6960, 'full_exposure_gpu_seconds': 1740,
                            'separate_cpu_hour_ceiling': 2, 'separate_gpu_hour_ceiling': .5},
              'automatic_retry': False, 'scientific_acceptance': False}
    create(ROOT / 'execution-prospective.json', ledger)
    # Mark submission attempted before calling the scheduler; an ambiguous response must not be retried.
    ledger['state'] = 'submission_attempted'
    create(ROOT / 'execution.json', ledger)
    result = subprocess.run(argv, capture_output=True, text=True)
    ledger.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
    if result.returncode != 0 or not result.stdout.strip().split(';')[0].isdigit():
        ledger['state'] = 'unknown_or_failed_submission'
    else:
        ledger.update(state='submitted', job_id=result.stdout.strip().split(';')[0])
    temporary = ROOT / 'execution.json.tmp'
    temporary.write_text(json.dumps(ledger, indent=2) + '\n')
    temporary.replace(ROOT / 'execution.json')
    print(json.dumps(ledger))
    if ledger['state'] != 'submitted':
        sys.exit(1)


if __name__ == '__main__':
    main()
