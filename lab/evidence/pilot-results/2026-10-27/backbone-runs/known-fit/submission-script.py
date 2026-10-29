import hashlib,json,shlex,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
P=Path('/private/tmp/known-backbone-fit-execution-20261027.json')
ROOT='/oscar/scratch/afengler/sbi4finance-rough-known-fit-20261027'
BANKROOT='/oscar/scratch/afengler/sbi4finance-rough-known-backbone-20261027'
REV='845c458b849c472271527cccf815ea6714f02206'
DRIVER='e19aa4f1f8011ebe9249113dc2058b0cee0e2aa9a530f39307a375045ff96945'
def now():return datetime.now(timezone.utc).isoformat()
def remote(argv):return subprocess.run(['ssh','-o','BatchMode=yes','oscar',shlex.join(argv)],capture_output=True,text=True,check=True)
def save(d):
 t=P.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(P)
def main():
 if P.exists():raise RuntimeError('An attempted submission already has a ledger; reconcile, no retry')
 freeze_job=sys.argv[1]
 if not freeze_job.isdigit():raise RuntimeError('Expected final snapshot Slurm ID')
 a=remote(['sacct','-S','2026-10-27','-j',freeze_job,'-X','-n','-P','--format=JobID%64,State%40,ElapsedRaw,AllocCPUS,ExitCode'])
 rows=[x.split('|') for x in a.stdout.splitlines() if x.split('|')[0]==freeze_job]
 if len(rows)!=1 or rows[0][1]!='COMPLETED' or rows[0][4]!='0:0':raise RuntimeError('Final integrity allocation must be completed0:0')
 report=json.loads(remote(['cat',BANKROOT+'/final-check.json']).stdout)
 if (report['status']!='completed_checked_snapshot_and_integrity' or report['verified_examples']!=32000 or report['verified_parts']!=1280 or report['configuration_splits']!={'train':20000,'validation':5000,'test':7000} or not report['all_prior_curve_quote_noise_reconstructed_exactly'] or not report['initial100_payload_manifest_hashes_preserved'] or report['hard_cap_exceptions']!=0):raise RuntimeError('Full bank integrity not complete')
 snapshot=BANKROOT+'/snapshot-20k';banksha=report['snapshot_manifest_sha256']
 if remote(['sha256sum',snapshot+'/manifest.json']).stdout.split()[0]!=banksha:raise RuntimeError('Snapshot changed since final check')
 if remote(['git','-C',ROOT+'/repo','rev-parse','HEAD']).stdout.strip()!=REV:raise RuntimeError('Frozen source differs')
 if remote(['sha256sum',ROOT+'/manifests/run_known_backbone_fit_20261027.py']).stdout.split()[0]!=DRIVER:raise RuntimeError('Driver differs')
 command=['env','PYTHONPATH='+ROOT+'/repo/src','KERAS_BACKEND=jax','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','NUMEXPR_NUM_THREADS=1','XLA_PYTHON_CLIENT_PREALLOCATE=false','MPLCONFIGDIR='+ROOT+'/cache/mpl','XDG_CACHE_HOME='+ROOT+'/cache','/oscar/scratch/afengler/sbi4finance-full-core-20261026-wy2_j3o4/repo/.pixi/envs/core-gpu/bin/python',ROOT+'/manifests/run_known_backbone_fit_20261027.py','--repo',ROOT+'/repo','--bank',snapshot,'--bank-sha',banksha,'--out',ROOT+'/runs/known-20k-native','--source-revision',REV]
 argv=['sbatch','--parsable','--nodes=1','--ntasks=1','--no-requeue','--account=default','--partition=gpu-debug','--qos=gpu-debug','--cpus-per-task=4','--gres=gpu:l40s:1','--mem=24G','--time=00:40:00','--chdir='+ROOT+'/repo','--job-name=sbi4f-known-20k-fit','--output='+ROOT+'/logs/known-20k-fit-%A.out','--wrap='+shlex.join(command)]
 d={'format':'known_curve_backbone_fit_execution_v1','authorization':'User continue the tutorial backbone. First matched known-curve20kposterior+forwardpricer and2000developmentvalidation; no independent/reference acceptance yet.','state':'submitting','attempted_utc':now(),'root':ROOT,'source_revision':REV,'driver_sha256':DRIVER,'operator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'bank_manifest_sha256':banksha,'completed_bank_final_job':freeze_job,'final_integrity_report_sha256':hashlib.sha256(remote(['cat',BANKROOT+'/final-check.json']).stdout.encode()).hexdigest(),'argv':argv,'allocation_ceiling':{'cpus':4,'gpus':1,'gpu_kind':'L40S','memory_gib':24,'wall_seconds':2400,'cleanup_seconds':240,'allocated_cpu_hours':3,'gpu_hours':.75,'full_limit_plus_cleanup_cpu_seconds':10560,'full_limit_plus_cleanup_gpu_seconds':2640},'budget_scope':'Separate firstknown-curve fit ceiling, not charged to its150CPU-hour generation ceiling or Heston6CPUh/1.25GPUh pool.','new_pricing_or_mcmc':False,'reserved_test_evaluated':False,'scientific_acceptance':False,'automatic_retry':False}
 save(d)
 try:r=remote(argv)
 except BaseException:d['state']='unknown_submission';save(d);raise
 job=r.stdout.strip().split(';')[0];d.update(stdout=r.stdout,stderr=r.stderr)
 if not job.isdigit():d['state']='unknown_submission';save(d);raise RuntimeError('Ambiguous response, reconcile without retry')
 d.update(state='submitted',job_id=job,submitted_utc=now());save(d)
 subprocess.run(['scp',str(P),'oscar:'+ROOT+'/execution.json'],check=True)
 print(json.dumps({'job_id':job,'snapshot_sha256':banksha,'cpu_hour_ceiling':3,'gpu_hour_ceiling':.75}))
if __name__=='__main__':main()
