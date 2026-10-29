"""One bounded six-variant development screen on an existing Heston bank.

This is a feasibility/learning pilot, not a full architecture study or acceptance
test. It uses the unchanged frozen BF2.1 trainer, final-epoch selection and native
samplers; no new data, pricers, notebooks or reserved-test outcomes are used.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

BASE = Path('/oscar/scratch/afengler')
OLD = BASE / 'sbi4finance-post-bank-baselines-recovery-v2-20261028'
ROOT = BASE / 'sbi4finance-independent-work-20261028/architecture-screen'
BANK = BASE / 'sbi4finance-full-core-20261026-wy2_j3o4/runs/heston-development/snapshot-20k'
VARIANTS = [('flow_matching', 'vector'), ('affine', 'vector'), ('spline', 'vector'),
            ('mean_flow', 'vector'), ('flow_matching', 'deepset'), ('flow_matching', 'set_transformer')]


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value):
    path = Path(path); temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temporary.replace(path)


def source_check(plan):
    original = read(OLD / 'plan.json')
    operator = Path(original['repo']) / 'cluster/submit_full_surface_backbones_20261028.py'
    spec = importlib.util.spec_from_file_location('frozen_source_check', operator)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.source_check(original)
    if sha(__file__) != plan['driver_sha256'] or sha(BANK / 'manifest.json') != plan['bank_manifest_sha256']:
        raise ValueError('Frozen driver or bank changed')


def submit():
    original = read(OLD / 'plan.json'); manifest = read(BANK / 'manifest.json')
    if manifest['configuration']['splits']['train'] != 20000:
        raise ValueError('Screen requires the existing common 20k Heston bank')
    ROOT.mkdir(parents=True, exist_ok=False); (ROOT / 'logs').mkdir()
    plan = {'format': 'tutorial_architecture_screen_v1', 'prepared_utc': datetime.now(timezone.utc).isoformat(),
        'repo': original['repo'], 'source_revision': original['source_revision'], 'python': original['python'],
        'driver_sha256': sha(__file__), 'bank': str(BANK), 'bank_manifest_sha256': sha(BANK / 'manifest.json'),
        'variants': [{'family': family, 'encoder': encoder} for family, encoder in VARIANTS],
        'recipe': {'epochs': 200, 'batch_size': 256, 'chunk_epochs': 50, 'seed': 0,
            'widths': [256,256,256,256], 'summary_dim': 32, 'checkpoint_selection': 'final',
            'validation_first': 0, 'validation_surfaces': 200, 'draws_per_surface': 256},
        'resources': {'wall_seconds': 3600, 'cleanup_seconds': 240, 'cpus': 4, 'gpus': 1,
            'full_allocated_cpu_seconds_ceiling': 15360, 'full_allocated_gpu_seconds_ceiling': 3840,
            'scope': 'separate one-job pilot allowance; original bank and baseline pools unchanged'},
        'scientific_acceptance': False, 'reserved_test_evaluated': False,
        'scope': 'one-factor family and encoder feasibility screen, equal 20k data/200 epochs; no winner or loss cross-family ranking'}
    source_check(plan); save(ROOT / 'plan.json', plan)
    env = {'KERAS_BACKEND':'jax','JAX_PLATFORMS':'cuda,cpu','XLA_PYTHON_CLIENT_PREALLOCATE':'false',
        'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1',
        'MPLCONFIGDIR':str(ROOT/'cache/mpl'),'XDG_CACHE_HOME':str(ROOT/'cache')}
    command = ['env', *[k+'='+v for k,v in env.items()], plan['python'], str(Path(__file__).resolve()), 'worker']
    argv = ['sbatch','--parsable','--nodes=1','--ntasks=1','--no-requeue','--account=default',
        '--partition=gpu-debug','--qos=gpu-debug','--cpus-per-task=4','--gres=gpu:l40s:1','--mem=32G',
        '--time=01:00:00','--job-name=sbi4f-architecture-screen','--chdir='+plan['repo'],
        '--output='+str(ROOT/'logs/screen-%A.out'),'--wrap='+shlex.join(command)]
    ledger = {'state':'submission_attempted','plan_sha256':sha(ROOT/'plan.json'),'argv':argv,
        'attempted_utc':datetime.now(timezone.utc).isoformat(),'automatic_retry':False}
    save(ROOT/'execution.json',ledger)
    try:
        result = subprocess.run(argv,capture_output=True,text=True,timeout=45)
        ledger.update(stdout=result.stdout,stderr=result.stderr,returncode=result.returncode)
        identifier = result.stdout.strip().split(';')[0]
        if result.returncode or not identifier.isdigit(): raise RuntimeError('Unconfirmed submission')
        ledger.update(state='submitted',job_id=identifier)
    except BaseException:
        ledger['state']='unknown_submission';save(ROOT/'execution.json',ledger);raise
    save(ROOT/'execution.json',ledger);print(json.dumps(ledger))


def worker():
    plan=read(ROOT/'plan.json');execution=read(ROOT/'execution.json')
    if execution['plan_sha256']!=sha(ROOT/'plan.json'):raise ValueError('Declaration changed')
    source_check(plan)
    if os.environ.get('SLURM_CPUS_PER_TASK')!='4':raise ValueError('Four CPUs required')
    import jax
    if jax.default_backend()!='gpu':raise ValueError('Native GPU required')
    repo=Path(plan['repo']);started=time.monotonic();deadline=started+3500
    record={'state':'running','plan_sha256':sha(ROOT/'plan.json'),'variants':[],
        'scientific_acceptance':False,'reserved_test_evaluated':False}
    save(ROOT/'run.json',record)
    for settings in plan['variants']:
        name=settings['family']+'-'+settings['encoder'];directory=ROOT/name;directory.mkdir()
        result={**settings,'name':name,'state':'running'};record['variants'].append(result);save(ROOT/'run.json',record)
        commands=[('training',[repo/'scripts/train_full_surface.py','--bank',BANK,'--out',directory/'posterior',
            '--family',settings['family'],'--encoder',settings['encoder'],'--widths','256,256,256,256',
            '--epochs','200','--batch-size','256','--chunk-epochs','50','--seed','0','--checkpoint-selection','final'],500),
            ('validation',[repo/'scripts/evaluate_full_surface.py','--bank',BANK,'--checkpoint',directory/'posterior',
            '--out',directory/'validation','--split','validation','--first','0','--surfaces','200',
            '--draws','256','--batch-size','50','--seed','0'],65)]
        try:
            for label,arguments,limit in commands:
                remaining=min(limit,deadline-time.monotonic())
                if remaining<=0:raise TimeoutError('Pilot allocation exhausted')
                before=time.monotonic()
                with (directory/(label+'.log')).open('w') as log:
                    subprocess.run([plan['python'],*map(str,arguments)],cwd=repo,stdout=log,
                        stderr=subprocess.STDOUT,check=True,timeout=remaining)
                result[label+'_seconds']=time.monotonic()-before
            source_check(plan);result['state']='completed'
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError,TimeoutError) as error:
            result.update(state='incomplete',error_type=type(error).__name__,error=str(error))
        save(ROOT/'run.json',record)
    record.update(state='completed_screen' if all(r['state']=='completed' for r in record['variants']) else 'partial_screen',
        seconds=time.monotonic()-started)
    save(ROOT/'run.json',record)
    if record['state']!='completed_screen':raise RuntimeError('Some pilot variants incomplete; preserve and review')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('submit','worker'));args=parser.parse_args()
    submit() if args.mode=='submit' else worker()
