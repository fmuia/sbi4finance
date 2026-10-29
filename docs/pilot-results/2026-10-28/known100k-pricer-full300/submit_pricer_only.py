"""One explicitly bounded optimization diagnostic; never retry submission."""
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import importlib.util
import json
from pathlib import Path
import platform
import shlex
import subprocess

BASE = Path('/oscar/scratch/afengler')
ROOT = BASE / 'sbi4finance-independent-work-20261028/known100k-pricer-full300'
OLD = BASE / 'sbi4finance-post-bank-baselines-recovery-v2-20261028'
KNOWN = BASE / 'sbi4finance-independent-work-20261028/known100k'


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def create(p, x):
    with Path(p).open('x') as stream:
        stream.write(json.dumps(x, indent=2, allow_nan=False) + '\n')


def main():
    if ROOT.exists():
        raise FileExistsError('Existing diagnostic requires reconciliation; never resubmit')
    previous, completed = read(OLD/'execution.json'), read(KNOWN/'execution.json')
    original_plan, known_plan = read(OLD/'plan.json'), read(KNOWN/'plan.json')
    assert previous['state'] == completed['state'] == 'finished'
    assert sha(OLD/'plan.json') == previous['plan_sha256']
    assert sha(KNOWN/'plan.json') == completed['plan_sha256']
    assert all(x['state'] == 'completed' for x in previous['routes'].values())
    assert completed['routes']['known']['state'] == 'completed'
    expected = {'8046588': 1209, '8053365': 1088, '8059257': 1478}
    actual = subprocess.run(['sacct', '-S', '2026-10-27', '-j', ','.join(expected), '-n', '-P',
        '--format=JobID%64,State%40,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES%120'],
        capture_output=True, text=True, timeout=45, check=True)
    rows = {r[0]: r for line in actual.stdout.splitlines()
            if len(r := line.split('|')) >= 6 and r[0] in expected}
    assert set(rows) == set(expected)
    for identifier, seconds in expected.items():
        row = rows[identifier]
        assert row[1].strip().split()[0] == 'COMPLETED' and row[2] == '0:0'
        assert int(row[3]) == seconds and int(row[4]) == 4
        assert dict(p.split('=', 1) for p in row[5].split(','))['gres/gpu'] == '1'
    prior_cpu = sum(int(r[3])*int(r[4]) for r in rows.values())
    prior_gpu = sum(int(r[3]) for r in rows.values())
    assert (prior_cpu, prior_gpu) == (15100, 3775)
    assert prior_cpu + 3360 <= 32400 and prior_gpu + 840 <= 8100
    assert sum(r['allocated_cpu_seconds'] for r in previous['routes'].values()) == 9188
    assert sum(r['allocated_gpu_seconds'] for r in previous['routes'].values()) == 2297
    assert completed['routes']['known']['allocated_cpu_seconds'] == 5912
    assert completed['routes']['known']['allocated_gpu_seconds'] == 1478

    repo, python = Path(known_plan['repo']), known_plan['python']
    assert known_plan['source_revision'] == 'c96a785fa1cc1a3e1ef234b8ddc81322776e5f4a'
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == known_plan['source_revision']
    operator = repo/'cluster/submit_full_surface_backbones_20261028.py'
    spec = importlib.util.spec_from_file_location('unchanged_worker_source_check', operator)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.source_check(known_plan)
    assert {k: metadata.version(k) for k in known_plan['runtime']} == known_plan['runtime']
    bank = Path(known_plan['routes']['known']['bank'])
    assert sha(bank/'manifest.json') == 'f1bfa42f7bf2acba3daaffb263b5bdeb60b5b12ebdfc267c5c5cd6eb7e08c313'
    manifest = read(bank/'manifest.json'); task = manifest['configuration']['task']
    assert manifest['configuration']['splits'] == {'train':100000,'validation':5000,'test':7000}
    assert manifest['configuration']['root_seed'] == 271300
    assert manifest['task_fingerprint'] == 'c99b0dd131896d89d8483a1668a3fde0fb7a0f28794c3db7fe14635cbbd52b65'
    assert task['task_id'] == 'rough_calibration_v1' and task['parameter_names'] == ['hurst','eta','rho']
    prior_pricer = read(KNOWN/'known/pricer/checkpoint/pricer.json')
    assert platform.python_version() == prior_pricer['versions']['python']
    assert sha(KNOWN/'known/posterior/model.keras') == '39f1a81986a38994dc6eacc520ad95254b34346abc02e758c08779c7c5916fc4'
    assert sha(KNOWN/'known/pricer/checkpoint/pricer.npz') == '9f0f695b0c369f6eaedb0c5559a46c254459047bd806fc14982377416a61a546'
    settings = {**prior_pricer['settings'], 'patience':300}
    assert settings == {'widths':[256,256,256,256],'epochs':300,'batch_size':256,'learning_rate':0.001,'patience':300,'seed':0}
    assert prior_pricer['settings']['patience'] == 30
    assert sha(repo/'scripts/train_full_surface_pricer.py') == '48d166911cc1ac10c9c938e1a8b358ef8c3159105c80afc08b1152d63db094fd'
    ROOT.mkdir(parents=True, exist_ok=False); (ROOT/'logs').mkdir()
    env = {'PYTHONPATH':str(repo/'src'),'KERAS_BACKEND':'jax','JAX_PLATFORMS':'cuda,cpu',
        'XLA_PYTHON_CLIENT_PREALLOCATE':'false','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
        'MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','MPLCONFIGDIR':str(ROOT/'cache/mpl'),'XDG_CACHE_HOME':str(ROOT/'cache')}
    command = ['env', *[k+'='+v for k,v in env.items()], python, str(repo/'scripts/train_full_surface_pricer.py'),
        '--bank',str(bank),'--out',str(ROOT/'pricer'),'--widths','256,256,256,256','--epochs','300',
        '--batch-size','256','--learning-rate','0.001','--patience','300','--seed','0']
    argv = ['sbatch','--parsable','--nodes=1','--ntasks=1','--no-requeue','--account=default',
        '--partition=gpu-debug','--qos=gpu-debug','--cpus-per-task=4','--gres=gpu:l40s:1','--mem=32G',
        '--time=00:10:00','--job-name=sbi4f-pricer-full300','--chdir='+str(repo),
        '--output='+str(ROOT/'logs/pricer-%A.out'),'--wrap='+shlex.join(command)]
    plan = {'format':'known100k_pricer_full300_diagnostic_v1','prepared_utc':now(),
        'source_revision':known_plan['source_revision'],'source_sha256':known_plan['source_sha256'],
        'runtime':known_plan['runtime'],'driver_sha256':sha(__file__),'bank':str(bank),
        'bank_manifest_sha256':sha(bank/'manifest.json'),'task_fingerprint':manifest['task_fingerprint'],
        'settings':settings,'only_setting_change':'patience 30 to 300; same selected-minimum validation rule',
        'preserved_posterior_sha256':sha(KNOWN/'known/posterior/model.keras'),
        'preserved_pricer_sha256':sha(KNOWN/'known/pricer/checkpoint/pricer.npz'),
        'prior_phase_execution_sha256':sha(OLD/'execution.json'),'known100k_execution_sha256':sha(KNOWN/'execution.json'),
        'resources':{'cpus':4,'gpus':1,'gpu_type':'l40s','memory_gib':32,'wall_seconds':600,'cleanup_seconds':240,
            'prior_cpu_seconds':prior_cpu,'prior_gpu_seconds':prior_gpu,'full_new_cpu_seconds':3360,'full_new_gpu_seconds':840,
            'full_cumulative_cpu_seconds':18460,'full_cumulative_gpu_seconds':4615,'original_cpu_seconds_ceiling':32400,
            'original_gpu_seconds_ceiling':8100,'no_additional_training_allowance':True},
        'command':command,'scientific_acceptance':False,'reserved_test_evaluated':False,'promotion':False,
        'scope':'One pricer-only optimization diagnostic. No posterior training/evaluation, MCMC, new data or artifact promotion.'}
    create(ROOT/'plan.json',plan)
    create(ROOT/'prior-accounting.json',{'checked_utc':now(),'rows':rows,'raw_sacct':actual.stdout,
        'prior_cpu_seconds':prior_cpu,'prior_gpu_seconds':prior_gpu,'all_terminal_success':True})
    attempt = {'state':'submission_attempted','utc':now(),'plan_sha256':sha(ROOT/'plan.json'),
        'driver_sha256':sha(__file__),'argv':argv,'automatic_retry':False}
    create(ROOT/'launch-attempt.json',attempt)  # Always before sbatch.
    try:
        result = subprocess.run(argv,capture_output=True,text=True,timeout=45)
        identifier = result.stdout.strip().split(';')[0]
        receipt = {**attempt,'stdout':result.stdout,'stderr':result.stderr,'returncode':result.returncode}
        if result.returncode or not identifier.isdigit():
            raise RuntimeError('Unconfirmed sbatch response; reconcile before any further action')
        receipt.update(state='submitted',job_id=identifier,accepted_utc=now())
    except BaseException as error:
        create(ROOT/'launch.json',{**attempt,'state':'unknown_submission','error_type':type(error).__name__,'error':str(error)})
        raise
    create(ROOT/'launch.json',receipt)
    print(json.dumps(receipt))


if __name__ == '__main__': main()
