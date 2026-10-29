import hashlib,json,shlex,subprocess
from datetime import datetime,timezone
from pathlib import Path
P=Path('/private/tmp/heston-100k-validation-execution-20261027.json')
ROOT='/oscar/scratch/afengler/sbi4finance-heston-100k-fit-20261027'
def now():return datetime.now(timezone.utc).isoformat()
def remote(argv):return subprocess.run(['ssh','-o','BatchMode=yes','oscar',shlex.join(argv)],capture_output=True,text=True,check=True)
def save(d):
 q=P.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(P)
def main():
 if P.exists():raise RuntimeError('Existing submission ledger: reconcile, no repeat')
 expected={'8028749':('COMPLETED','0:0',1),'8028804':('CANCELLED by 140010387','0:0',0),'8028941':('FAILED','1:0',1),'8029251':('COMPLETED','0:0',1),'8030120':('FAILED','1:0',1)}
 r=remote(['sacct','-S','2026-10-27','-j',','.join(expected),'-X','-n','-P','--format=JobID%64,State%40,ElapsedRaw,AllocCPUS,ExitCode'])
 rows=[x.split('|') for x in r.stdout.splitlines() if x.strip()]
 if len(rows)!=5 or {x[0] for x in rows}!=set(expected):raise RuntimeError('Missing terminal accounting')
 cpu=gpu=0
 for row in rows:
  state,code,gpus=expected[row[0]]
  if (row[1],row[4],row[3])!=(state,code,'4'):raise RuntimeError('Unreviewed job state')
  cpu+=int(row[2])*4;gpu+=int(row[2])*gpus
 if cpu+5760>21600 or gpu+1440>4500:raise RuntimeError('20minute continuation plus240s cleanup would exceed pool')
 runpath=ROOT+'/runs/heston-100k-native';old=json.loads(remote(['cat',runpath+'/run.json']).stdout)
 if old['status']!='incomplete' or old['failure']['type']!='TimeoutExpired' or old['stages'][-1]['status']!='timeout' or any(x['status']!='completed' for x in old['stages'][:-1]):raise RuntimeError('Failure differs from reviewed validation-only timeout')
 checkpoint=runpath+'/checkpoint';training=json.loads(remote(['cat',checkpoint+'/training.json']).stdout)
 if training['selected_epoch']!=402 or not training['same_seed_reload_exact'] or training['training_surfaces']!=100000 or training['flow_matching_integrate_kwargs']!={'method':'tsit5','steps':'adaptive'}:raise RuntimeError('Selected completed100kfit differs')
 if remote(['sha256sum',checkpoint+'/model.keras']).stdout.split()[0]!=training['checkpoint_sha256']:raise RuntimeError('Selected model bytes differ')
 command=['env','PYTHONPATH='+ROOT+'/repo/src','KERAS_BACKEND=jax','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','NUMEXPR_NUM_THREADS=1','XLA_PYTHON_CLIENT_PREALLOCATE=false','MPLCONFIGDIR='+ROOT+'/cache/mpl','XDG_CACHE_HOME='+ROOT+'/cache','/oscar/scratch/afengler/sbi4finance-full-core-20261026-wy2_j3o4/repo/.pixi/envs/core-gpu/bin/python',ROOT+'/repo/scripts/evaluate_full_surface.py','--bank','/oscar/scratch/afengler/sbi4finance-heston-backbone-20261027/snapshot-100k','--checkpoint',checkpoint,'--out',ROOT+'/runs/heston-100k-validation-continuation/validation','--split','validation','--first','0','--surfaces','2000','--draws','1000','--batch-size','100','--seed','0']
 argv=['sbatch','--parsable','--nodes=1','--ntasks=1','--no-requeue','--account=default','--partition=gpu-debug','--qos=gpu-debug','--cpus-per-task=4','--gres=gpu:l40s:1','--mem=24G','--time=00:20:00','--chdir='+ROOT+'/repo','--job-name=sbi4f-heston-100k-val','--output='+ROOT+'/logs/heston-100k-validation-%A.out','--wrap='+shlex.join(command)]
 d={'format':'heston_100k_validation_continuation_v1','state':'submitting','attempted_utc':now(),'authorization':'Explicit reviewed evaluation-only continuation after900second internal timeout. No retraining, new data, MCMC, changed panel/seeds/solver or automatic retry.','root':ROOT,'original_failed_job':'8030120','original_failure':old['failure'],'original_run_sha256':hashlib.sha256(remote(['cat',runpath+'/run.json']).stdout.encode()).hexdigest(),'checkpoint_sha256':training['checkpoint_sha256'],'source_revision':old['source_commit'],'operator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'argv':argv,'allocation_ceiling':{'cpus':4,'gpus':1,'wall_seconds':1200,'cleanup_seconds':240,'full_limit_plus_cleanup_cpu_seconds':5760,'full_limit_plus_cleanup_gpu_seconds':1440},'pooled_budget_check':{'previous_actual_cpu_seconds':cpu,'previous_actual_gpu_seconds':gpu,'after_submission_cpu_seconds':cpu+5760,'after_submission_gpu_seconds':gpu+1440,'cpu_ceiling_seconds':21600,'gpu_ceiling_seconds':4500,'accounting':r.stdout},'scientific_acceptance':False,'reserved_test_evaluated':False,'automatic_retry':False}
 save(d)
 try:r=remote(argv)
 except BaseException:d['state']='unknown_submission';save(d);raise
 job=r.stdout.strip().split(';')[0];d.update(stdout=r.stdout,stderr=r.stderr)
 if not job.isdigit():d['state']='unknown_submission';save(d);raise RuntimeError('Ambiguous submission')
 d.update(state='submitted',job_id=job,submitted_utc=now());save(d)
 subprocess.run(['scp',str(P),'oscar:'+ROOT+'/validation-continuation-execution.json'],check=True)
 print(json.dumps({'job_id':job,'cpu_exposure':cpu+5760,'gpu_exposure':gpu+1440}))
if __name__=='__main__':main()
