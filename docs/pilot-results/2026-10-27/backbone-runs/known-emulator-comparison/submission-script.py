"""Freeze and submit exactly one authorized bounded comparison allocation."""
import datetime,hashlib,json,math,pathlib,shlex,shutil,subprocess
REPO=pathlib.Path('/Users/afengler/Projects/proj_sbi4finance/sbi4finance-full-surface-core')
DEST=REPO/'docs/pilot-results/2026-10-27/backbone-runs/known-emulator-comparison'
P=pathlib.Path('/private/tmp/known-emulator-comparison-execution-20261027.json')
ROOT='/oscar/scratch/afengler/sbi4finance-rough-known-reference-20261027'
FIT='/oscar/scratch/afengler/sbi4finance-rough-known-fit-20261027'
REV='56dcb13e1d2bf5b6dc96e573dfcaa6240a7104a9'
PY='/oscar/scratch/afengler/sbi4finance-full-core-20261026-wy2_j3o4/repo/.pixi/envs/core-gpu/bin/python'
DRIVER=pathlib.Path('/private/tmp/run_known_emulator_comparison_20261027.py')
REMOTE_DRIVER=ROOT+'/manifests/run_known_emulator_comparison_20261027.py'
REMOTE_PLAN=ROOT+'/manifests/known_emulator_comparison_plan_20261027.json'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(b):return hashlib.sha256(b).hexdigest()
def ssh(args):
 r=subprocess.run(['ssh','oscar',shlex.join(args)],capture_output=True,text=True,timeout=90)
 if r.returncode:raise RuntimeError(r.stderr)
 return r.stdout
def remote(script):return json.loads(ssh(['python3','-c','exec(bytes.fromhex('+repr(script.encode().hex())+').decode())']))
def save(d):
 t=P.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(P);shutil.copyfile(P,DEST/'execution.json')
def main():
 if P.exists():raise RuntimeError('comparison ledger already exists: reconcile, never resubmit')
 DEST.mkdir(parents=True,exist_ok=True)
 fit=json.loads(pathlib.Path('/private/tmp/known-backbone-fit-execution-20261027.json').read_text());refs=json.loads(pathlib.Path('/private/tmp/known-emulator-references-execution-20261027.json').read_text())
 assert fit['state']=='completed_development_checks' and fit['actuals']['allocated_cpu_seconds']==4032 and fit['actuals']['gpu_seconds']==1008
 assert refs['state']=='completed_approximate_development_references' and refs['terminal_actuals']['allocated_worker_cpu_seconds_counted_once']==788
 frozen=dict(refs['frozen_sha256'])
 source_files=subprocess.check_output(['git','ls-tree','-r','--name-only',REV,'scripts/compare_full_surface_reference.py','src/sbi4finance/calibration','src/sbi4finance/gbm_bank.py','pyproject.toml','pixi.lock'],cwd=REPO,text=True).splitlines()
 for name in source_files:frozen[ROOT+'/repo/'+name]=digest(subprocess.check_output(['git','show',REV+':'+name],cwd=REPO))
 training=REPO/'docs/pilot-results/2026-10-27/backbone-runs/known-fit'
 inv=json.loads((training/'artifact-inventory.json').read_text())['files']
 for name in ['checkpoint/model.keras','checkpoint/training.json','checkpoint/task.json']:frozen[FIT+'/runs/known-20k-native/'+name]=inv[name]['sha256']
 assert frozen[FIT+'/runs/known-20k-native/checkpoint/model.keras']=='9d6cbd534a49bd0862316c9e38a02ed4e9da818986951d1b020162fa0ef9573e'
 refdir=REPO/'docs/pilot-results/2026-10-27/backbone-runs/known-emulator-references';art=json.loads((refdir/'remote-artifacts.json').read_text())
 for i in range(3):
  for name in ['reference.json','diagnostics.json']:frozen[ROOT+f'/runs/case-{i}/'+name]=digest((refdir/f'case-{i}'/name).read_bytes())
  frozen[ROOT+f'/runs/case-{i}/chains.npz']=art[ROOT+f'/runs/case-{i}/chains.npz']['sha256']
 frozen[REMOTE_DRIVER]=digest(DRIVER.read_bytes())
 plan={'format':'known_emulator_comparison_plan_v1','created_utc':now(),'source_revision':REV,'driver_sha256':frozen[REMOTE_DRIVER],'checkpoint':FIT+'/runs/known-20k-native/checkpoint','checkpoint_sha256':frozen[FIT+'/runs/known-20k-native/checkpoint/model.keras'],'frozen_sha256':frozen,'cases':[{'index':i,'surface_id':f'validation:{i}','neural_seed':271500+i,'neural_draws':10000} for i in range(3)],'resources':{'cpus':4,'gpus':1,'gpu_kind':'L40S','memory_gib':24,'wall_seconds':600,'cleanup_seconds':240,'driver_deadline_seconds':570,'per_case_timeout_seconds':180},'pool':{'allocated_cpu_ceiling_seconds':10800,'allocated_gpu_ceiling_seconds':2700,'completed_fit_cpu_seconds':4032,'completed_fit_gpu_seconds':1008,'new_full_exposure_cpu_seconds':3360,'new_full_exposure_gpu_seconds':840,'combined_full_exposure_cpu_seconds':7392,'combined_full_exposure_gpu_seconds':1848,'cpu_reference_788_seconds_in_separate_pool':True},'scientific_acceptance':False,'reserved_test_evaluated':False,'no_new_fits_prices_or_mcmc':True,'scope':'Approximate-emulator conditional comparisons only; no model-price or like-for-like simulator posterior validation.'}
 plan_path=DEST/'plan.json';plan_path.write_text(json.dumps(plan,indent=2)+'\n');shutil.copyfile(DRIVER,DEST/'driver.py');shutil.copyfile(__file__,DEST/'submission-script.py')
 subprocess.run(['scp',str(DRIVER),'oscar:'+REMOTE_DRIVER],check=True,capture_output=True,text=True)
 subprocess.run(['scp',str(plan_path),'oscar:'+REMOTE_PLAN],check=True,capture_output=True,text=True)
 script="""import hashlib,json,pathlib,subprocess,math
root=pathlib.Path(ROOT_VALUE);plan=json.loads((root/'manifests/known_emulator_comparison_plan_20261027.json').read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root/'repo').decode().strip()==plan['source_revision']
assert subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root/'repo').decode()==''
assert all(sha(p)==h for p,h in plan['frozen_sha256'].items())
assert not (root/'runs/comparisons').exists() and not (root/'comparison-execution.json').exists()
deployment=json.loads((root/'manifests/deployment.json').read_text());assert deployment['source_revision']==plan['source_revision'] and deployment['locked_environment_files_match_50k_run'] is True
cases=[]
for i in range(3):
 directory=root/('runs/case-'+str(i));reference=json.loads((directory/'reference.json').read_text());d=json.loads((directory/'diagnostics.json').read_text())
 assert not (directory/'failure.json').exists() and reference['surface_id']=='validation:'+str(i) and reference['split']=='validation' and reference['scientific_acceptance'] is False
 assert reference['likelihood_kind']=='forward-pricer-emulator' and reference['task_fingerprint']=='c99b0dd131896d89d8483a1668a3fde0fb7a0f28794c3db7fe14635cbbd52b65'
 assert d['mcmc_diagnostics_passed'] and d['parameter_mcmc_diagnostics_passed'] and d['derived_quantities']['mcmc_diagnostics_passed']
 rows=[d['parameters']['parameters['+n+']'] for n in ['hurst','eta','rho']]+[d['derived_quantities']['parameters']['parameters[rho_eta]']]
 assert all(math.isfinite(row[k]) and (row[k]<=bound if upper else row[k]>=bound) for row in rows for k,bound,upper in [('r_hat',1.01,True),('ess_bulk',400,False),('ess_tail',400,False)])
 assert all(sha(path)==value for path,value in reference['emulator']['input_sha256'].items())
 cases.append({'index':i,'reference_diagnostics_passed':True,'frozen_reference_sha256':sha(directory/'reference.json'),'frozen_chains_sha256':sha(directory/'chains.npz')})
accounting=subprocess.check_output(['sacct','-S2026-10-27','-j8030892','-n','-P','--format=JobID%64,State%40,ElapsedRaw,AllocCPUS,ExitCode']).decode()
rows=[r.split('|') for r in accounting.splitlines() if r and '.' not in r.split('|')[0]]
assert sorted(r[0] for r in rows)==['8030892_0','8030892_1','8030892_2'] and all(r[1]=='COMPLETED' and r[4]=='0:0' for r in rows)
print(json.dumps({'status':'passed','source_revision':plan['source_revision'],'frozen_input_count':len(plan['frozen_sha256']),'all_frozen_hashes_match':True,'reference_accounting':accounting,'cases':cases,'deployment':deployment,'remote_plan_sha256':sha(root/'manifests/known_emulator_comparison_plan_20261027.json')}))
""".replace('ROOT_VALUE',repr(ROOT))
 preflight=remote(script);assert preflight['remote_plan_sha256']==digest(plan_path.read_bytes());(DEST/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
 command=['env','PYTHONPATH='+ROOT+'/repo/src','KERAS_BACKEND=jax','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','NUMEXPR_NUM_THREADS=1','XLA_PYTHON_CLIENT_PREALLOCATE=false','MPLCONFIGDIR='+ROOT+'/cache/mpl','XDG_CACHE_HOME='+ROOT+'/cache',PY,REMOTE_DRIVER]
 argv=['sbatch','--parsable','--nodes=1','--ntasks=1','--no-requeue','--account=default','--partition=gpu-debug','--qos=gpu-debug','--cpus-per-task=4','--gres=gpu:l40s:1','--mem=24G','--time=00:10:00','--chdir='+ROOT+'/repo','--job-name=sbi4f-known-emulator-compare','--output='+ROOT+'/logs/comparison-%A.out','--wrap='+shlex.join(command)]
 d={'format':'known_emulator_comparison_execution_v1','state':'planned','created_utc':now(),'authorization':'Parent explicitly authorizes exactly one ten-minute4CPU1L40S comparison allocation; shared known-fit3CPUh/.75GPUh pool. No retries, fits, prices, MCMC or test evaluation.','root':ROOT,'remote_execution_ledger':ROOT+'/comparison-execution.json','plan_sha256':digest(plan_path.read_bytes()),'driver_sha256':plan['driver_sha256'],'source_revision':REV,'operator_sha256':digest(pathlib.Path(__file__).read_bytes()),'argv':argv,'allocation_ceiling':plan['resources'],'shared_pool':plan['pool'],'scientific_acceptance':False,'independent_pricing_assessed':False,'reserved_test_evaluated':False,'automatic_retry':False,'preflight_sha256':digest((DEST/'preflight.json').read_bytes())};save(d)
 subprocess.run(['scp',str(P),'oscar:'+d['remote_execution_ledger']],check=True,capture_output=True,text=True)
 d.update(state='submitting',attempted_utc=now());save(d)
 try:stdout=ssh(argv)
 except BaseException:d.update(state='unknown_submission');save(d);raise
 job=stdout.strip().split(';')[0]
 if not job.isdigit():d.update(state='unknown_submission',stdout=stdout);save(d);raise RuntimeError('ambiguous submission; do not retry')
 d.update(state='submitted',job_id=job,stdout=stdout,submitted_utc=now());save(d);print(json.dumps({'job_id':job,'shared_pool_full_cpu_seconds':7392,'shared_pool_full_gpu_seconds':1848}),flush=True)
 subprocess.run(['scp',str(P),'oscar:'+d['remote_execution_ledger']],check=True,capture_output=True,text=True)
if __name__=='__main__':main()
