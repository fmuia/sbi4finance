"""One bounded allocation: three approximate-emulator development comparisons."""
import hashlib,json,os,pathlib,signal,subprocess,sys,time
ROOT=pathlib.Path('/oscar/scratch/afengler/sbi4finance-rough-known-reference-20261027')
PLAN=ROOT/'manifests/known_emulator_comparison_plan_20261027.json'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,d):
 t=p.with_suffix('.json.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def check(plan):
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'repo',text=True).strip()!=plan['source_revision']:raise ValueError('source revision differs')
 if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT/'repo',text=True):raise ValueError('tracked source is dirty')
 for p,h in plan['frozen_sha256'].items():
  if sha(p)!=h:raise ValueError('frozen input differs: '+p)
 for i in range(3):
  if (ROOT/f'runs/case-{i}/failure.json').exists():raise ValueError('retained reference failure')

def main():
 if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK')!='4':raise ValueError('one4CPU GPU allocation required')
 if any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']):raise ValueError('thread caps must all equal1')
 started=time.perf_counter();deadline=started+570;plan_bytes=PLAN.read_bytes();plan=json.loads(plan_bytes);check(plan)
 if sha(__file__)!=plan['driver_sha256']:raise ValueError('driver differs')
 out=ROOT/'runs/comparisons';out.mkdir(parents=True,exist_ok=False)
 run={'format':'known_emulator_comparison_run_v1','status':'started','job_id':os.environ['SLURM_JOB_ID'],'plan_sha256':sha(PLAN),'driver_sha256':sha(__file__),'source_revision':plan['source_revision'],'frozen_sha256':plan['frozen_sha256'],'scientific_acceptance':False,'independent_pricing_assessed':False,'reserved_test_evaluated':False,'no_new_fits_prices_or_mcmc':True,'cases':[],'scope':'Approximate-emulator agreement only; stochastic simulator target and independent numerical accuracy are not established.'};write(out/'run.json',run)
 def interrupted(signum,frame):raise SystemExit('signal'+str(signum)+'; retain outputs, no retry')
 signal.signal(signal.SIGTERM,interrupted)
 try:
  for i in range(3):
   remaining=deadline-time.perf_counter()
   if remaining<=0:raise TimeoutError('whole570second deadline exhausted')
   command=[sys.executable,str(ROOT/'repo/scripts/compare_full_surface_reference.py'),'--checkpoint',plan['checkpoint'],'--reference',str(ROOT/f'runs/case-{i}'),'--out',str(out/f'case-{i}'),'--draws','10000','--seed',str(271500+i)]
   row={'index':i,'seed':271500+i,'command':command,'status':'running','timeout_cap_seconds':180,'effective_timeout_seconds':min(180,remaining)};run['cases'].append(row);write(out/'run.json',run);before=time.perf_counter()
   try:
    with (out/f'case-{i}.log').open('w') as f:result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=row['effective_timeout_seconds'],check=False,cwd=ROOT/'repo')
    row.update(returncode=result.returncode,status='completed' if result.returncode==0 else 'failed')
    if result.returncode:raise RuntimeError('case'+str(i)+' command failed')
   except subprocess.TimeoutExpired:row['status']='timeout';raise
   finally:row['seconds']=time.perf_counter()-before;write(out/'run.json',run)
   report=json.loads((out/f'case-{i}/comparison.json').read_text())
   if report['status']!='completed_approximate_emulator_comparison' or report['scientific_acceptance'] is not False or report['independent_pricing_assessed'] is not False or report['reference_issues'] or report['sampling_seed']!=271500+i or report['neural_draws_requested']!=10000 or report['surface_id']!=f'validation:{i}' or report['checkpoint_sha256']!=plan['checkpoint_sha256']:raise ValueError('comparison contract differs or source reference is inconclusive')
   row['proposed_parameter_agreement_all_checks_passed']=report['proposed_parameter_agreement']['all_checks_passed'];write(out/'run.json',run);check(plan)
  if PLAN.read_bytes()!=plan_bytes or sha(__file__)!=plan['driver_sha256']:raise ValueError('plan/driver changed')
  run.update(status='completed_approximate_emulator_comparisons',seconds=time.perf_counter()-started);write(out/'run.json',run)
 except BaseException as error:
  run.update(status='incomplete' if isinstance(error,(TimeoutError,subprocess.TimeoutExpired,SystemExit)) else 'failed',seconds=time.perf_counter()-started,failure={'type':type(error).__name__,'error':str(error)});write(out/'run.json',run);raise
 print(json.dumps({'status':run['status'],'scientific_acceptance':False,'case_parameter_checks':[c['proposed_parameter_agreement_all_checks_passed'] for c in run['cases']]}))
if __name__=='__main__':main()
