"""Deploy and submit the one authorized read-only MC-loss audit; no retries."""
import hashlib, json, shlex, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = '/oscar/scratch/afengler/sbi4finance-heston-audit-20261027'
REPO = '/oscar/scratch/afengler/sbi4finance-rough-known-reference-20261027/repo'
PYTHON = '/oscar/scratch/afengler/sbi4finance-full-core-20261026-wy2_j3o4/repo/.pixi/envs/core-gpu/bin/python'
SCRIPT = Path('/private/tmp/heston_validation_loss_mc_20261028.py')
LEDGER = Path('/private/tmp/heston-validation-loss-mc-execution-20261028.json')
REMOTE_LEDGER = ROOT + '/validation-loss-mc-execution.json'
REMOTE_SCRIPT = ROOT + '/' + SCRIPT.name
EXPECTED = 'f039389d8d118caa755a6bfefc0da8671034dff9bbdcacda268e659c22d65f09'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def remote(argv, **kwargs):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', 'oscar', shlex.join(argv)],
                          text=True, capture_output=True, check=True, timeout=45, **kwargs)
def save():
    p = LEDGER.with_suffix('.json.tmp'); p.write_text(json.dumps(d, indent=2) + '\n'); p.replace(LEDGER)
def upload(p, destination):
    # Exclusive creation prevents a second deployment or accidental overwrite.
    code = 'import sys; from pathlib import Path; p=Path(sys.argv[1]); p.write_bytes(sys.stdin.buffer.read()) if not p.exists() else sys.exit("Destination exists")'
    remote(['python3', '-c', code, destination], input=Path(p).read_text())

if LEDGER.exists(): raise RuntimeError('Existing ledger: reconcile outcome; never retry submission')
if sha(SCRIPT) != EXPECTED: raise RuntimeError('Reviewed driver changed')
prior_path = Path('/private/tmp/heston-audit-execution-20261027.json')
prior = json.loads(prior_path.read_text())
if (prior['state'] != 'completed' or prior['job_id'] != '8031924' or prior['actual_elapsed_seconds'] != 168
        or prior['actual_allocated_cpu_seconds'] != 672 or prior['actual_gpu_seconds'] != 168):
    raise RuntimeError('Initial-controls budget reconciliation changed')
cmd = ['env', 'PYTHONPATH=' + REPO + '/src', 'KERAS_BACKEND=jax', 'JAX_PLATFORMS=cuda,cpu',
       'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'NUMEXPR_NUM_THREADS=1',
       'XLA_PYTHON_CLIENT_PREALLOCATE=false', 'MPLCONFIGDIR=' + ROOT + '/cache/mpl',
       'XDG_CACHE_HOME=' + ROOT + '/cache', PYTHON, REMOTE_SCRIPT, '--repo', REPO,
       '--checkpoint', '/oscar/scratch/afengler/sbi4finance-heston-100k-fit-20261027/runs/heston-100k-native/checkpoint',
       '--bank', '/oscar/scratch/afengler/sbi4finance-heston-backbone-20261027/snapshot-100k',
       '--out', ROOT + '/results/validation-loss-mc']
argv = ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '--no-requeue', '--account=default',
        '--partition=gpu-debug', '--qos=gpu-debug', '--cpus-per-task=4', '--gres=gpu:l40s:1',
        '--mem=24G', '--time=00:05:00', '--chdir=' + REPO, '--job-name=sbi4f-heston-loss-mc',
        '--output=' + ROOT + '/logs/validation-loss-mc-%A.out', '--wrap=' + shlex.join(cmd)]
d = {'format': 'heston_validation_loss_mc_execution_v1', 'state': 'preparing', 'root': ROOT,
     'created_utc': datetime.now(timezone.utc).isoformat(),
     'authorization': 'Parent explicitly authorizes exactly one read-only paired validation-loss audit after reviewing driver SHA f039389d; no training, gradients, optimizer updates, pricing, MCMC, posterior sampling, reserved test, retries, or other jobs.',
     'source_sha256': {SCRIPT.name: EXPECTED, Path(__file__).name: sha(__file__)}, 'argv': argv,
     'preflight_verified': {'source_revision': '56dcb13e1d2bf5b6dc96e573dfcaa6240a7104a9',
         'source_tracked_files_clean': True,
         'best402_sha256': '7f037dee0fde78cd5a33f1107e13031ff01fdcb5481dab76fc6c8f7d07598f39',
         'final500_sha256': '8d4a243ba099524cf4cda40e4f0616e895879f053829a7f60dadbe65b98301db',
         'bank_manifest_sha256': '4b5929b0d0b23698c340eec6defabfd629fe23eca2ed78fe9bd46e4de35c244e',
         'sacct_prior_job': '8031924|COMPLETED|168|4|billing=110,cpu=4,gres/gpu=1,mem=24G,node=1|0:0'},
     'resources': {'cpus': 4, 'gpus': 1, 'wall_seconds': 300, 'internal_deadline_seconds': 270,
         'cleanup_seconds': 240, 'full_exposure_cpu_seconds': 2160, 'full_exposure_gpu_seconds': 540,
         'pool_cpu_seconds_ceiling': 7200, 'pool_gpu_seconds_ceiling': 900,
         'prior_actual_cpu_seconds': 672, 'prior_actual_gpu_seconds': 168,
         'combined_prospective_cpu_seconds': 2832, 'combined_prospective_gpu_seconds': 708,
         'prior_job_id': '8031924', 'prior_execution_ledger_sha256': sha(prior_path),
         'separate_excluded_final_panel_job_id': '8032002'},
     'seeds': list(range(281001, 281033)), 'automatic_retry': False, 'scientific_acceptance': False}
save()
try:
    upload(SCRIPT, REMOTE_SCRIPT)
    check = remote(['sha256sum', REMOTE_SCRIPT]).stdout.split()[0]
    if check != EXPECTED: raise RuntimeError('Deployed script differs')
    d['state'] = 'submitting'; d['attempted_utc'] = datetime.now(timezone.utc).isoformat(); save()
    upload(LEDGER, REMOTE_LEDGER)  # Prospective execution ledger exists remotely before sbatch.
    r = remote(argv)
except BaseException as e:
    d.update(state='unknown_or_failed_submission', error_type=type(e).__name__, error=str(e)); save(); raise
job = r.stdout.strip().split(';')[0]
d.update(stdout=r.stdout, stderr=r.stderr)
if not job.isdigit():
    d['state'] = 'unknown_submission'; save(); raise RuntimeError('Ambiguous sbatch response; do not retry')
d.update(state='submitted', job_id=job, submitted_utc=datetime.now(timezone.utc).isoformat()); save()
update = 'import sys;from pathlib import Path;p=Path(sys.argv[1]);q=p.with_suffix(".tmp");q.write_text(sys.stdin.read());q.replace(p)'
remote(['python3', '-c', update, REMOTE_LEDGER], input=LEDGER.read_text())
print(json.dumps({'job_id': job, 'execution_ledger': str(LEDGER), 'remote_execution_ledger': REMOTE_LEDGER,
                  'combined_cpu_seconds': 2832, 'combined_gpu_seconds': 708}))
