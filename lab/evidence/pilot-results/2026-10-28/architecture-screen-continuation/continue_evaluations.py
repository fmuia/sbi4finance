"""One bounded evaluation-only continuation of two saved pilot checkpoints."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time

BASE = Path('/oscar/scratch/afengler')
OLD = BASE/'sbi4finance-independent-work-20261028/architecture-screen'
ROOT = BASE/'sbi4finance-independent-work-20261028/architecture-screen-continuation'
EXPECTED = Path(__file__).with_name('architecture_screen_continuation_preflight.json')
NAMES = ['flow_matching-deepset', 'flow_matching-set_transformer']


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def create(p, x):
    with Path(p).open('x') as output:
        output.write(json.dumps(x, indent=2, allow_nan=False)+'\n')
def save(p, x):
    p=Path(p); temporary=p.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n'); temporary.replace(p)


def source_check(pilot):
    original=read(BASE/'sbi4finance-post-bank-baselines-recovery-v2-20261028/plan.json')
    path=Path(original['repo'])/'cluster/submit_full_surface_backbones_20261028.py'
    spec=importlib.util.spec_from_file_location('frozen_c96_source_check',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.source_check(original)
    assert original['source_revision']==pilot['source_revision']=='c96a785fa1cc1a3e1ef234b8ddc81322776e5f4a'


def submit():
    if ROOT.exists(): raise FileExistsError('Continuation already has artifacts; reconcile, never resubmit')
    expected=read(EXPECTED); pilot=read(OLD/'plan.json'); original=read(OLD/'run.json')
    assert original['state']=='partial_screen'
    assert sha(OLD/'plan.json')==expected['old_plan_sha256']=='8f2ec3d0059374bfc23f3d2c0f46a6c3dc728cc7a05bb3f90e16da5063b88855'
    assert sha(OLD/'run.json')==expected['old_run_sha256']
    assert sha(Path(pilot['bank'])/'manifest.json')==expected['bank_manifest_sha256']
    assert pilot['recipe']['validation_surfaces']==200 and pilot['recipe']['draws_per_surface']==256
    assert pilot['recipe']['seed']==0 and pilot['recipe']['validation_first']==0
    source_check(pilot)
    q=subprocess.run(['sacct','-S','2026-10-27','-j','8059371','-n','-P',
        '--format=JobID%64,State%40,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES%120'],capture_output=True,text=True,timeout=45,check=True)
    rows=[line.split('|') for line in q.stdout.splitlines() if line.startswith('8059371|')]
    assert len(rows)==1
    row=rows[0]
    assert row[1]=='FAILED' and row[2]=='1:0' and int(row[3])==1231 and int(row[4])==4
    assert dict(p.split('=',1) for p in row[5].split(','))['gres/gpu']=='1'
    assert 4924+3360==8284<=15360 and 1231+840==2071<=3840
    assert pilot['resources']['full_allocated_cpu_seconds_ceiling']==15360
    assert pilot['resources']['full_allocated_gpu_seconds_ceiling']==3840
    frozen={str(p):sha(p) for p in [Path(__file__),EXPECTED,OLD/'plan.json',OLD/'run.json',
        Path(pilot['bank'])/'manifest.json',Path(pilot['repo'])/'scripts/evaluate_full_surface.py']}
    checkpoints={}
    for settings in pilot['variants']:
        name=settings['family']+'-'+settings['encoder'];d=OLD/name/'posterior';t=read(d/'training.json')
        assert t['selected_epoch']==200 and t['checkpoint_selection']=='final' and t['same_seed_reload_exact']
        assert sha(d/'model.keras')==t['checkpoint_sha256']==expected['checkpoints'][name]['model_sha256']
        assert sha(d/'training.json')==expected['checkpoints'][name]['training_json_sha256']
        checkpoints[name]=expected['checkpoints'][name]
        if name in NAMES:
            assert t['flow_matching_integrate_kwargs']=={'method':'tsit5','steps':'adaptive'}
            for p in [d/'model.keras',d/'training.json']:frozen[str(p)]=sha(p)
    assert [v['name'] for v in original['variants'] if v['state']=='incomplete']==NAMES
    ROOT.mkdir(parents=True,exist_ok=False);(ROOT/'logs').mkdir();(ROOT/'original').mkdir()
    for p in [OLD/'plan.json',OLD/'execution.json',OLD/'run.json']:
        (ROOT/'original'/p.name).write_bytes(p.read_bytes())
    for v in original['variants']:
        if v['state']=='completed':
            d=ROOT/'original'/v['name'];d.mkdir()
            (d/'evaluation.json').write_bytes((OLD/v['name']/'validation/evaluation.json').read_bytes())
    create(ROOT/'checkpoint-identities.json',checkpoints)
    create(ROOT/'prior-accounting.json',{'checked_utc':now(),'raw_sacct':q.stdout,'prior_cpu_seconds':4924,
        'prior_gpu_seconds':1231,'reviewed_failed_job':'8059371','historical_failure_preserved':True})
    plan={'format':'architecture_evaluation_continuation_v1','prepared_utc':now(),'repo':pilot['repo'],
        'python':pilot['python'],'bank':pilot['bank'],'variants':NAMES,'frozen_sha256':frozen,
        'settings':{'split':'validation','first':0,'surfaces':200,'draws':256,'batch_size':50,'seed':0,
            'per_evaluation_timeout_seconds':240,'overall_worker_deadline_seconds':550},
        'resources':{'cpus':4,'gpus':1,'memory_gib':32,'wall_seconds':600,'cleanup_seconds':240,
            'prior_cpu_seconds':4924,'prior_gpu_seconds':1231,'full_new_cpu_seconds':3360,'full_new_gpu_seconds':840,
            'full_cumulative_cpu_seconds':8284,'full_cumulative_gpu_seconds':2071,
            'original_cpu_seconds_ceiling':15360,'original_gpu_seconds_ceiling':3840},
        'scope':'Evaluate only two saved final200 checkpoints. No retraining, original-pilot mutation or promotion.',
        'scientific_acceptance':False,'reserved_test_evaluated':False,'automatic_retry':False}
    create(ROOT/'plan.json',plan)
    env={'KERAS_BACKEND':'jax','JAX_PLATFORMS':'cuda,cpu','XLA_PYTHON_CLIENT_PREALLOCATE':'false',
        'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1',
        'MPLCONFIGDIR':str(ROOT/'cache/mpl'),'XDG_CACHE_HOME':str(ROOT/'cache')}
    command=['env',*[k+'='+v for k,v in env.items()],pilot['python'],str(Path(__file__).resolve()),'worker']
    argv=['sbatch','--parsable','--nodes=1','--ntasks=1','--no-requeue','--account=default','--partition=gpu-debug',
        '--qos=gpu-debug','--cpus-per-task=4','--gres=gpu:l40s:1','--mem=32G','--time=00:10:00',
        '--job-name=sbi4f-evaluation-continuation','--chdir='+pilot['repo'],
        '--output='+str(ROOT/'logs/evaluation-%A.out'),'--wrap='+shlex.join(command)]
    receipt={'state':'submission_attempted','utc':now(),'plan_sha256':sha(ROOT/'plan.json'),
        'driver_sha256':sha(__file__),'argv':argv,'automatic_retry':False}
    create(ROOT/'launch-attempt.json',receipt)
    try:
        r=subprocess.run(argv,capture_output=True,text=True,timeout=45)
        receipt.update(stdout=r.stdout,stderr=r.stderr,returncode=r.returncode)
        identifier=r.stdout.strip().split(';')[0]
        if r.returncode or not identifier.isdigit():raise RuntimeError('Unconfirmed submission; never retry')
        receipt.update(state='submitted',job_id=identifier,accepted_utc=now())
    except BaseException as error:
        receipt.update(state='unknown_submission',error_type=type(error).__name__,error=str(error))
        create(ROOT/'launch.json',receipt);raise
    create(ROOT/'launch.json',receipt);print(json.dumps(receipt))


def worker():
    plan=read(ROOT/'plan.json');receipt=read(ROOT/'launch-attempt.json')
    assert sha(ROOT/'plan.json')==receipt['plan_sha256']
    assert all(sha(p)==h for p,h in plan['frozen_sha256'].items())
    source_check(read(OLD/'plan.json'))
    started=time.monotonic();run={'state':'running','started_utc':now(),'plan_sha256':receipt['plan_sha256'],
        'variants':[],'scientific_acceptance':False,'reserved_test_evaluated':False}
    save(ROOT/'run.json',run)
    for name in plan['variants']:
        d=ROOT/name;d.mkdir();v={'name':name,'state':'running'};run['variants'].append(v);save(ROOT/'run.json',run)
        remaining=min(240,550-(time.monotonic()-started))
        try:
            if remaining<=0:raise TimeoutError('Bounded evaluation allocation exhausted')
            argv=[plan['python'],str(Path(plan['repo'])/'scripts/evaluate_full_surface.py'),'--bank',plan['bank'],
                '--checkpoint',str(OLD/name/'posterior'),'--out',str(d/'validation'),'--split','validation',
                '--first','0','--surfaces','200','--draws','256','--batch-size','50','--seed','0']
            before=time.monotonic()
            with (d/'evaluation.log').open('w') as log:subprocess.run(argv,cwd=plan['repo'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining)
            assert all(sha(p)==h for p,h in plan['frozen_sha256'].items())
            v.update(state='completed',seconds=time.monotonic()-before)
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError,TimeoutError) as error:
            v.update(state='incomplete',error_type=type(error).__name__,error=str(error))
        save(ROOT/'run.json',run)
    run.update(state='completed_continuation' if all(v['state']=='completed' for v in run['variants']) else 'partial_continuation',
        completed_utc=now(),seconds=time.monotonic()-started)
    save(ROOT/'run.json',run)
    if run['state']!='completed_continuation':raise RuntimeError('Continuation incomplete; no automatic retry')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['submit','worker']);a=p.parse_args()
    submit() if a.mode=='submit' else worker()
