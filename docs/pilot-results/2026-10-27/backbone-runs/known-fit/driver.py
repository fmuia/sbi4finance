"""One bounded known-curve backbone fit, composed from existing commands.

No pricing, MCMC, test evaluation, automatic retry or acceptance claim.
"""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path

TASK='c99b0dd131896d89d8483a1668a3fde0fb7a0f28794c3db7fe14635cbbd52b65'
SPLITS={'train':20000,'validation':5000,'test':7000}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temporary.replace(path)
def git(repo,*args):return subprocess.check_output(['git',*args],cwd=repo,text=True).strip()
def commands(repo,bank,out):
    py=sys.executable
    return [
      ('environment',[py,str(repo/'scripts/check_full_surface_environment.py'),'--require-gpu'],60),
      ('posterior',[py,str(repo/'scripts/train_full_surface.py'),'--bank',str(bank),'--out',str(out/'checkpoint'),
         '--family','flow_matching','--encoder','vector','--widths','256,256,256,256','--epochs','300',
         '--batch-size','256','--chunk-epochs','25','--seed','0'],600),
      ('pricer',[py,str(repo/'scripts/train_full_surface_pricer.py'),'--bank',str(bank),'--out',str(out/'pricer'),
         '--widths','256,256,256,256','--epochs','300','--batch-size','512','--patience','30','--seed','0'],600),
      ('validation',[py,str(repo/'scripts/evaluate_full_surface.py'),'--bank',str(bank),'--checkpoint',str(out/'checkpoint'),
         '--out',str(out/'validation'),'--split','validation','--first','0','--surfaces','2000',
         '--draws','1000','--batch-size','100','--seed','0'],1050)]
def run(args):
    if not os.environ.get('SLURM_JOB_ID') or int(os.environ.get('SLURM_CPUS_PER_TASK','0'))!=4:
        raise ValueError('one4-CPU GPU Slurm allocation is required')
    if any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')):
        raise ValueError('all numerical-library thread caps must equal1')
    if git(args.repo,'rev-parse','HEAD')!=args.source_revision or git(args.repo,'status','--porcelain','--untracked-files=no'):
        raise ValueError('the exact clean frozen source revision is required')
    started=time.perf_counter();deadline=started+2370
    args.out.mkdir(parents=True,exist_ok=False)
    files=[args.repo/'scripts'/name for name in ('check_full_surface_environment.py','train_full_surface.py','train_full_surface_pricer.py','evaluate_full_surface.py')]
    files += [*sorted((args.repo/'src/sbi4finance/calibration').glob('*.py')),args.repo/'src/sbi4finance/gbm_bank.py',args.repo/'tutorial_final/studies/train_calibration.py']
    state={'format':'known_curve_backbone_fit_run_v1','status':'preflight','source_revision':args.source_revision,
      'source_sha256':{str(p.relative_to(args.repo)):sha(p) for p in files},'driver_sha256':sha(__file__),
      'environment_sha256':{n:sha(args.repo/n) for n in ('pyproject.toml','pixi.lock')},
      'bank_manifest_sha256':args.bank_sha,'task_fingerprint':TASK,'scientific_acceptance':False,
      'reserved_test_evaluated':False,'automatic_retry':False,'no_new_pricing_or_mcmc':True,
      'resources':{'cpus':4,'gpus':1,'memory_gib':24,'wall_seconds':2400,'cleanup_seconds':240,
                   'full_limit_plus_cleanup_cpu_seconds':10560,'full_limit_plus_cleanup_gpu_seconds':2640,
                   'allocated_cpu_hour_ceiling':3,'gpu_hour_ceiling':.75,'internal_deadline_seconds':2370},
      'approximation_scope':'Forward emulator learns numerical bank prices; its Gaussian likelihood would omit label MC randomness/covariance. No independent-reference or like-for-like posterior claim follows from this fit.',
      'stages':[{'name':n,'command':c,'timeout_cap_seconds':cap,'status':'pending'} for n,c,cap in commands(args.repo,args.bank,args.out)]}
    path=args.out/'run.json';write(path,state)
    def interrupted(signum,frame):raise SystemExit(f'signal{signum}; retain partial outputs, no retry')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        manifest_path=args.bank/'manifest.json'
        if sha(manifest_path)!=args.bank_sha:raise ValueError('bank manifest checksum differs')
        manifest=json.loads(manifest_path.read_text());config=manifest['configuration']
        declaration=args.repo/'configs/calibration_rough_known_backbone_20k.json'
        if (manifest['task_fingerprint']!=TASK or config['splits']!=SPLITS
            or config!=json.loads(declaration.read_text()) or config['task']['task_id']!='rough_calibration_v1'
            or config['task']['parameter_names']!=['hurst','eta','rho']):
            raise ValueError('the declared20k/5k/7k known-curve task/configuration is required')
        state['declared_config_sha256']=sha(declaration)
        state['observation_selection']={'split':'validation','first':0,'surfaces':2000,'draws':1000,'seed':0,
          'scope':'development IDs only; no reserved acceptance evaluation'}
        write(path,state)
        for row in state['stages']:
            remaining=deadline-time.perf_counter()
            if remaining<=0:raise TimeoutError('whole-run deadline exhausted')
            row.update(status='running',effective_timeout_seconds=min(remaining,row['timeout_cap_seconds']))
            state['status']=row['name'];write(path,state);before=time.perf_counter()
            try:
                with (args.out/f"{row['name']}.log").open('w') as log:
                    result=subprocess.run(row['command'],cwd=args.repo,stdout=log,stderr=subprocess.STDOUT,
                                          timeout=row['effective_timeout_seconds'],check=False)
                row.update(returncode=result.returncode,status='completed' if result.returncode==0 else 'failed')
                if result.returncode:raise RuntimeError(row['name']+' failed; inspect retained outputs')
            except subprocess.TimeoutExpired:row['status']='timeout';raise
            finally:row['seconds']=time.perf_counter()-before;write(path,state)
        posterior=json.loads((args.out/'checkpoint/training.json').read_text())
        pricer_dir=args.out/'pricer/checkpoint';pricer=json.loads((pricer_dir/'pricer.json').read_text())
        evaluation=json.loads((args.out/'validation/evaluation.json').read_text())
        if (posterior['task_fingerprint']!=TASK or posterior['training_surfaces']!=20000
            or posterior['validation_surfaces']!=5000 or posterior['same_seed_reload_exact'] is not True
            or posterior['reserved_test_evaluated'] is not False
            or posterior['run_contract']['dataset']['manifest_sha256']!=args.bank_sha
            or posterior['flow_matching_integrate_kwargs']!={'method':'tsit5','steps':'adaptive'}
            or posterior['network']['flow_steps'] is not None
            or 'supplied known curve' not in posterior['conditioning']):
            raise ValueError('posterior output does not match known-curve/native/reload contract')
        if (pricer['task_fingerprint']!=TASK or pricer['pricer_sha256']!=sha(pricer_dir/'pricer.npz')
            or pricer['bank_manifest_sha256']!=args.bank_sha or pricer['same_reload_exact'] is not True
            or json.loads((pricer_dir/'task.json').read_text())!=config['task']
            or pricer['training_surfaces']!=20000 or pricer['validation_surfaces']!=5000
            or pricer['reserved_test_evaluated'] is not False):
            raise ValueError('pricer checkpoint/task/data binding or reload differs')
        family=evaluation.get('diagnostics_with_derived_targets',{})
        if (evaluation['task_fingerprint']!=TASK or evaluation['bank_manifest_sha256']!=args.bank_sha
            or evaluation['checkpoint_sha256']!=posterior['checkpoint_sha256']
            or evaluation['example_ids']!=[f'validation:{i}' for i in range(2000)]
            or evaluation['evaluated_split']!='validation' or evaluation['reserved_test_evaluated'] is not False
            or family.get('targets')!=['hurst','eta','rho','rho_eta']):
            raise ValueError('evaluation must retain all2000 declared cases and the4-target diagnostic family')
        if (sha(manifest_path)!=args.bank_sha or sha(declaration)!=state['declared_config_sha256']
            or sha(__file__)!=state['driver_sha256']
            or any(sha(args.repo/n)!=h for n,h in state['source_sha256'].items())
            or any(sha(args.repo/n)!=h for n,h in state['environment_sha256'].items())
            or git(args.repo,'rev-parse','HEAD')!=args.source_revision):
            raise ValueError('frozen bank/source/environment/driver changed during unit')
        state.update(status='completed_development_checks',seconds=time.perf_counter()-started,
          selected_posterior_epoch=posterior['selected_epoch'],selected_pricer_epoch=pricer['fit']['selected_epoch'],
          posterior_checkpoint_sha256=posterior['checkpoint_sha256'],pricer_sha256=pricer['pricer_sha256'],
          calibration_with_derived_targets=family,pricer_validation_errors=pricer['validation_errors'],
          interpretation='Selected development fits and numerical-label comparisons; retain failed scientific diagnostics. No conditional reference/product or tutorial acceptance claim.')
        write(path,state);return state
    except BaseException as error:
        state.update(status='incomplete' if isinstance(error,(TimeoutError,subprocess.TimeoutExpired,SystemExit)) else 'failed',
                     seconds=time.perf_counter()-started,failure={'type':type(error).__name__,'error':str(error)})
        write(path,state);raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','bank','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--bank-sha',required=True);p.add_argument('--source-revision',required=True)
    args=p.parse_args()
    for name in ('repo','bank','out'):setattr(args,name,getattr(args,name).resolve())
    state=run(args);print(json.dumps({'status':state['status'],'scientific_acceptance':False}))
if __name__=='__main__':main()
