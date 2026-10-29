"""One bounded continuation of the existing known-curve PyMC reference pool.

Reuse the c96 consumer CLI and existing comparison statistics. Never price the
financial simulator or resample the neural network. These references target the
fixed current mean emulator, with its approximation limitations retained.
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
import sys
import time

BASE = Path('/oscar/scratch/afengler')
ROOT = BASE / 'sbi4finance-scientific-validation-20261028/known50k-references'
REPO = BASE / 'sbi4finance-training-recovery-v2-20261028/repo'
BANK = BASE / 'sbi4finance-rough-known-backbone-20261027/snapshot-50k'
FIT = BASE / 'sbi4finance-post-bank-baselines-recovery-v2-20261028/known'
OLD = BASE / 'sbi4finance-rough-known-reference-20261027'
PYTHON = BASE / 'sbi4finance-full-core-20261026-wy2_j3o4/repo-conditioned-reference/.pixi/envs/core-reference/bin/python'


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value):
    path = Path(path); tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)
def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result); return result


def frozen(plan):
    if subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()!=plan['source_revision']:
        raise ValueError('Frozen consumer source changed')
    for path,digest in plan['frozen_sha256'].items():
        if sha(path)!=digest:raise ValueError('Frozen input changed: '+path)


def prepare():
    import numpy as np
    sys.path.insert(0,str(REPO/'src'))
    from sbi4finance.calibration import bank_collection,reference
    prior=read(OLD/'execution.json')
    result=subprocess.run(['sacct','-S','2026-10-27','-j',prior['job_id'],'-n','-P',
        '--format=JobID%64,State%40,ExitCode,ElapsedRaw,AllocCPUS'],capture_output=True,text=True,timeout=30,check=True)
    rows={r[0]:r for line in result.stdout.splitlines() if len(r:=line.split('|'))>=5 and '.' not in r[0]}
    old_cost=0
    for index in range(3):
        r=rows[prior['job_id']+'_'+str(index)]
        if r[1].strip()!='COMPLETED' or r[2]!='0:0':raise ValueError('Prior reference allocation is unsettled')
        old_cost+=int(r[3])*int(r[4])
    if old_cost!=788 or old_cost+4*(3600+240)>21600:raise ValueError('Original six CPU-hour pool differs')
    manifest,data=bank_collection.load_bank(BANK,splits=['validation'],roles=['observations','context_values','example_id'])
    task=manifest['configuration']['task'];pricer,identity=reference.load_forward_pricer(FIT/'pricer/checkpoint',task)
    evaluation=read(FIT/'validation/evaluation.json');training=read(FIT/'posterior/training.json')
    if evaluation['checkpoint_sha256']!=training['checkpoint_sha256'] or evaluation['bank_manifest_sha256']!=sha(BANK/'manifest.json'):
        raise ValueError('Saved neural panel does not identify the current 50k model/bank')
    with np.load(FIT/'validation/draws.npz',allow_pickle=False) as arrays:
        selected=arrays['draws'][:3].copy()
    if selected.shape!=(3,1000,3):raise ValueError('Expected three saved 1000-draw posterior panels')
    cases=[]
    for index in range(3):
        old=read(OLD/f'runs/case-{index}/reference.json')
        quotes=data['validation']['observations'][index,:,0];curve=data['validation']['context_values'][index]
        if (old['task']!=task or not np.array_equal(quotes,old['observed_prices'])
                or not np.array_equal(curve,old['known_curve_levels'])
                or evaluation['example_ids'][index]!=f'validation:{index}'):
            raise ValueError('The fixed old/current observation or known curve differs')
        cases.append({'index':index,'surface_id':f'validation:{index}','seed':271400+index,
            'observed_prices':quotes.tolist(),'known_curve_levels':curve.tolist(),
            'old_reference_sha256':sha(OLD/f'runs/case-{index}/reference.json')})
    paths=[Path(__file__).resolve(),BANK/'manifest.json',OLD/'execution.json',
        FIT/'posterior/training.json',FIT/'posterior/task.json',FIT/'posterior/model.keras',
        FIT/'validation/evaluation.json',FIT/'validation/draws.npz',
        *[FIT/'pricer/checkpoint'/name for name in ('pricer.npz','pricer.json','task.json')],
        REPO/'scripts/calibration_reference.py',REPO/'scripts/compare_full_surface_reference.py',
        *[REPO/'src/sbi4finance/calibration'/name for name in ('reference.py','pymc_reference_op.py',
            'reference_initialization.py','pricer_net.py','validation.py','full_surface.py')],
        *[OLD/f'runs/case-{i}/reference.json' for i in range(3)]]
    plan={'format':'known50k_current_emulator_reference_continuation_v1','root':str(ROOT),
        'prepared_utc':datetime.now(timezone.utc).isoformat(),'repo':str(REPO),'python':str(PYTHON),
        'driver':str(Path(__file__).resolve()),'source_revision':subprocess.check_output(
            ['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
        'bank':str(BANK),'bank_manifest_sha256':sha(BANK/'manifest.json'),'cases':cases,
        'posterior_checkpoint_sha256':training['checkpoint_sha256'],'pricer':identity,
        'frozen_sha256':{str(p):sha(p) for p in paths},
        'recipe':{'chains':4,'cores':4,'draws':8000,'tune':4000,'initialization':'data-fit',
            'fit_starts':4,'fit_budget':2000,'mp_context':'spawn','checkpoint_every':100},
        'resources':{'cpus':4,'gpus':0,'wall_seconds':3600,'cleanup_seconds':240,
            'prior_allocated_cpu_seconds':old_cost,'full_new_exposure_cpu_seconds':15360,
            'cumulative_full_exposure_cpu_seconds':old_cost+15360,'original_cpu_ceiling_seconds':21600},
        'neural_draws_per_case':1000,'new_neural_sampling':False,'new_financial_pricing':False,
        'scientific_acceptance':False,'reserved_test_evaluated':False,'automatic_retry':False,
        'target_scope':'Current fixed conditional-mean emulator with independent Gaussian quote noise and physical uniform prior/logit Jacobian. MC label randomness/covariance and surrogate/grid bias omitted; approximate development comparison only.'}
    frozen(plan);ROOT.mkdir(parents=True,exist_ok=False);(ROOT/'logs').mkdir()
    save(ROOT/'plan.json',plan);np.savez_compressed(ROOT/'saved-neural-draws.npz',draws=selected)
    (ROOT/'prior-accounting.psv').write_text(result.stdout)
    print(json.dumps({'root':str(ROOT),'plan_sha256':sha(ROOT/'plan.json'),'cases':[c['surface_id'] for c in cases],
                      'pricer_sha256':sha(FIT/'pricer/checkpoint/pricer.npz'),
                      'resources':plan['resources']}))


def submit():
    plan=read(ROOT/'plan.json');frozen(plan)
    if (ROOT/'execution.json').exists():raise FileExistsError('Prior attempt must be reconciled; never retry automatically')
    env={'PYTHONPATH':str(REPO/'src'),'KERAS_BACKEND':'jax','JAX_PLATFORMS':'cpu',
        'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1',
        'MPLCONFIGDIR':str(ROOT/'cache/mpl'),'XDG_CACHE_HOME':str(ROOT/'cache'),
        'PYTENSOR_FLAGS':'compiledir='+str(ROOT/'cache/pytensor')}
    command=['env',*[k+'='+v for k,v in env.items()],str(PYTHON),plan['driver'],'worker']
    argv=['sbatch','--parsable','--nodes=1','--ntasks=1','--no-requeue','--account=carney-mjfrank-condo2',
        '--partition=batch','--cpus-per-task=4','--mem=16G','--time=01:00:00',
        '--exclude=node2338,node2339,node2336,node2344,node2346','--job-name=sbi4f-known50k-pymc',
        '--chdir='+str(REPO),'--output='+str(ROOT/'logs/reference-%A.out'),'--wrap='+shlex.join(command)]
    record={'state':'submission_attempted','plan_sha256':sha(ROOT/'plan.json'),'argv':argv,
            'attempted_utc':datetime.now(timezone.utc).isoformat(),'automatic_retry':False}
    save(ROOT/'execution.json',record)
    try:
        result=subprocess.run(argv,capture_output=True,text=True,timeout=45)
        record.update(stdout=result.stdout,stderr=result.stderr,returncode=result.returncode)
        identifier=result.stdout.strip().split(';')[0]
        if result.returncode or not identifier.isdigit():raise RuntimeError('Unconfirmed scheduler submission')
        record.update(state='submitted',job_id=identifier)
    except BaseException:
        record['state']='unknown_submission';save(ROOT/'execution.json',record);raise
    save(ROOT/'execution.json',record);print(json.dumps(record))


def compare_saved(plan,index):
    import numpy as np
    comparison=module(REPO/'scripts/compare_full_surface_reference.py','frozen_comparison')
    root=ROOT/f'case-{index}';record=read(root/'reference.json');diagnostics=read(root/'diagnostics.json')
    case=plan['cases'][index];task=record['task'];names=task['parameter_names']
    if (record['surface_id']!=case['surface_id'] or record['settings']['seed']!=case['seed']
            or record['observed_prices']!=case['observed_prices']
            or record['known_curve_levels']!=case['known_curve_levels']):
        raise ValueError('Current reference observation/context/seed changed')
    comparison._emulator_binding(record,task,record['task_fingerprint'])
    with np.load(root/'chains.npz',allow_pickle=False) as arrays:chains=arrays['parameters']
    with np.load(ROOT/'saved-neural-draws.npz',allow_pickle=False) as arrays:neural=arrays['draws'][index]
    if chains.shape!=(4,8000,3):raise ValueError('Reference chain axes or draw counts changed')
    issues=comparison._reference_issues(diagnostics,names)
    issues+=['rho_eta: '+s for s in comparison._reference_issues(diagnostics.get('derived_quantities'),['rho_eta'])]
    result={'status':'inconclusive_reference' if issues else 'completed_approximate_emulator_comparison',
        'surface_id':case['surface_id'],'reference_issues':issues,'reference_chain_shape':list(chains.shape),
        'neural_draws':len(neural),'neural_source':'saved completed known50k validation panel; no new neural sampling',
        'saved_neural_sha256':sha(ROOT/'saved-neural-draws.npz'),
        'checkpoint_sha256':plan['posterior_checkpoint_sha256'],'reference_sha256':sha(root/'reference.json'),
        'diagnostics_sha256':sha(root/'diagnostics.json'),'chains_sha256':sha(root/'chains.npz'),
        'scientific_acceptance':False,'reserved_test_evaluated':False,'target_scope':plan['target_scope'],
        'old_neural_draw_count':10000,'draw_count_qualification':'Current comparison reuses1000draws versus10000in the old20kcomparison; retain neural MonteCarlo uncertainty, not a strict paired sample comparison.'}
    if not issues:
        agreement=comparison.validation.agreement(neural,chains.reshape(-1,3))
        result['proposed_parameter_agreement']={'parameters':{name:{
            'standardized_mean_error':agreement['standardized_mean_error'][j],
            'width_ratio_90':agreement['width_ratio'][j],
            'mean_error_within_0_3_reference_sd':agreement['standardized_mean_error'][j]<=.3,
            'relative_width_error_within_25_percent':abs(agreement['width_ratio'][j]-1)<=.25}
            for j,name in enumerate(names)}}
        result['proposed_parameter_agreement']['all_checks_passed']=all(
            p['mean_error_within_0_3_reference_sd'] and p['relative_width_error_within_25_percent']
            for p in result['proposed_parameter_agreement']['parameters'].values())
        result['derived_parameter_comparison']=comparison._rough_derived_comparison(task,neural,chains,diagnostics)
    save(root/'saved-neural-comparison.json',result);return result


def worker():
    plan=read(ROOT/'plan.json');execution=read(ROOT/'execution.json');frozen(plan)
    if execution['plan_sha256']!=sha(ROOT/'plan.json') or os.environ.get('SLURM_CPUS_PER_TASK')!='4':
        raise ValueError('Declaration or four-CPU allocation differs')
    started=time.monotonic();deadline=started+3500
    record={'state':'running','job_id':os.environ['SLURM_JOB_ID'],'plan_sha256':sha(ROOT/'plan.json'),
        'cases':[],'scientific_acceptance':False,'reserved_test_evaluated':False}
    save(ROOT/'run.json',record)
    try:
        for case in plan['cases']:
            index=case['index'];out=ROOT/f'case-{index}';entry={'surface_id':case['surface_id'],'state':'sampling'}
            record['cases'].append(entry);save(ROOT/'run.json',record)
            args=[str(PYTHON),str(REPO/'scripts/calibration_reference.py'),'--bank',str(BANK),'--split','validation',
                '--index',str(index),'--out',str(out),'--pricer',str(FIT/'pricer/checkpoint'),
                '--draws','8000','--tune','4000','--chains','4','--cores','4','--mp-context','spawn',
                '--initialization','data-fit','--fit-starts','4','--fit-budget','2000','--seed',str(case['seed']),
                '--checkpoint-every','100','--no-progress']
            with (ROOT/'logs'/f'case-{index}.log').open('w') as log:
                subprocess.run(args,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True,
                    timeout=max(1,min(1000,deadline-time.monotonic())))
            comparison=compare_saved(plan,index);entry.update(state='completed',comparison_status=comparison['status'],
                proposed_parameter_agreement=comparison.get('proposed_parameter_agreement',{}).get('all_checks_passed'))
            frozen(plan);save(ROOT/'run.json',record)
        record.update(state='completed',seconds=time.monotonic()-started)
    except BaseException as error:
        record.update(state='failed',error_type=type(error).__name__,error=str(error),seconds=time.monotonic()-started)
        save(ROOT/'run.json',record);raise
    save(ROOT/'run.json',record)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','submit','worker'));a=p.parse_args()
    {'prepare':prepare,'submit':submit,'worker':worker}[a.mode]()
