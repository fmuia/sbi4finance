"""Single-unit fail-closed submission journal; no automatic retries."""
import argparse,hashlib,json,shlex,subprocess,time
from pathlib import Path

ROOT=Path('/oscar/scratch/afengler/sbi4finance-known-pricing-validation-20261028')
REPO=Path('/oscar/scratch/afengler/sbi4finance-rough-known-reference-20261027/repo')
PYTHON='/oscar/scratch/afengler/sbi4finance-full-core-20261026-wy2_j3o4/repo-conditioned-reference/.pixi/envs/core-reference/bin/python'
DRIVER=Path('/oscar/scratch/afengler/known_rough_pricing_check_20261028.py')
AUDIT=Path('/oscar/scratch/afengler/audit_known_pricing_streams_20261028.py')
PLAN_SHA='759b3df88ac650aab83f2bf00fcd04aa2a2ebc669d03a3b6634661f68fdb8654'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(value):
 p=ROOT/'execution.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(value,indent=2)+'\n');q.replace(p)
def terminal(job):
 text=subprocess.check_output(['sacct','-X','-n','-P','-j',job,'--format=JobID,State,ElapsedRaw,AllocCPUS,ExitCode'],text=True)
 rows=[r.split('|') for r in text.splitlines() if r.strip()]
 found=[r for r in rows if r[0]==job]
 if len(found)!=1:raise ValueError('Unambiguous terminaltop-levelaccountingrequired: '+job)
 row=found[0]
 if row[1]!='COMPLETED' or row[4]!='0:0':raise ValueError('PriorjobnotCOMPLETED0:0: '+str(row))
 return {'job_id':job,'state':row[1],'elapsed_seconds':int(row[2]),'allocated_cpus':int(row[3]),'exit_code':row[4],
         'allocated_cpu_seconds':int(row[2])*int(row[3])}
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['metadata','pilot','remaining']);args=p.parse_args()
 if sha(ROOT/'plan.json')!=PLAN_SHA:raise ValueError('FrozenplanSHAchanged')
 plan=json.loads((ROOT/'plan.json').read_text())
 for name,digest in plan['frozen_sha256'].items():
  if sha(name)!=digest:raise ValueError('Frozeninputchanged: '+name)
 if args.mode=='metadata':
  if (ROOT/'execution.json').exists():raise ValueError('Journalalreadyexists; noretry')
  ledger={'format':'known_rough_pricing_execution_v1','state':'prepared','plan_sha256':PLAN_SHA,'attempts':[],
          'authorization':'Usercontinuesknown-curvepricingdiagnosis; rootapprovedsixpoints,eight65536-total-antithetic-pathreplicates,bothgrids,24CPUh,noGPU.',
          'resources':{'cpu_ceiling_seconds':86400,'gpu_ceiling_seconds':0,'cpus_per_worker':1,'gpus':0,'worker_wall_seconds':1200,
                       'cleanup_seconds_per_job':240,'max_concurrent_workers':16,'memory_gib_per_worker':16,
                       'all48_worker_full_exposure_cpu_seconds':48*1440,'metadata_full_exposure_cpu_seconds':540,
                       'final_integrity_reserved_cpu_seconds':3600,'prospective_total_cpu_seconds':48*1440+540+3600},
          'frozen_submission_sources':{str(x):sha(x) for x in [DRIVER,AUDIT,Path(__file__)]},'automatic_retry':False,
          'scientific_acceptance':False,'reserved_test_evaluated':False}
 else:
  ledger=json.loads((ROOT/'execution.json').read_text())
  for name,digest in ledger['frozen_submission_sources'].items():
   if sha(name)!=digest:raise ValueError('Submission/auditdriverchanged')
  expected=['metadata'] if args.mode=='pilot' else ['metadata','pilot']
  if [a['role'] for a in ledger['attempts']]!=expected or any(a['state']!='submitted' for a in ledger['attempts']):
   raise ValueError('Unexpected/uncertainsubmissionhistory; failclosed')
  for attempt in ledger['attempts']:attempt['terminal']=terminal(attempt['job_id'])
  audit=json.loads((ROOT/'stream-audit.json').read_text())
  if audit['status']!='passed' or audit['plan_sha256']!=PLAN_SHA:raise ValueError('Missing/prematurestreamaudit')
  ledger['stream_audit_sha256']=sha(ROOT/'stream-audit.json')
  if args.mode=='remaining':
   pilot=json.loads((ROOT/'results/batch-00/run.json').read_text())
   if pilot['status']!='completed' or not pilot['frozen_inputs_verified']:raise ValueError('Firstbatchdidnotcompleteimmutably')
   ledger['pilot_elapsed_seconds']=pilot['elapsed_seconds']
 if ledger['resources']['prospective_total_cpu_seconds']>ledger['resources']['cpu_ceiling_seconds']:
  raise ValueError('Fullworst-caseexposureexceedsceiling')
 env=['env',f'PYTHONPATH={REPO}/src','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','NUMEXPR_NUM_THREADS=1']
 if args.mode=='metadata':command=env+[PYTHON,str(AUDIT),str(REPO),str(ROOT)];wall='00:05:00';memory='4G'
 else:command=env+[PYTHON,str(DRIVER),'worker','--repo',str(REPO),'--root',str(ROOT),'--index','0'];wall='00:20:00';memory='16G'
 argv=['sbatch','--parsable','--nodes=1','--ntasks=1','--no-requeue','--account=carney-mjfrank-condo2','--partition=batch',
       '--cpus-per-task=1','--mem='+memory,'--time='+wall,'--chdir='+str(REPO),'--job-name=known-price-'+args.mode,
       '--output='+str(ROOT/'logs'/f'{args.mode}-%A_%a.out')]
 if args.mode=='remaining':argv+=['--array=1-47%16']
 argv+=['--wrap='+shlex.join(command)]
 attempt={'role':args.mode,'state':'submitting','attempted_unix':time.time(),'argv':argv}
 ledger['attempts'].append(attempt);ledger['state']='submission_in_progress';save(ledger)
 (ROOT/f'execution-before-{args.mode}.json').write_bytes((ROOT/'execution.json').read_bytes())
 try:
  result=subprocess.run(argv,text=True,capture_output=True,timeout=45)
  attempt.update(stdout=result.stdout,stderr=result.stderr,returncode=result.returncode)
  if result.returncode or not result.stdout.strip().split(';')[0].isdigit():
   attempt['state']='submission_failed_or_unknown';ledger['state']='stopped_no_retry';save(ledger)
   raise ValueError('Cannotestablishsubmissionoutcome; noretry')
  attempt.update(state='submitted',job_id=result.stdout.strip().split(';')[0]);ledger['state']='submitted_'+args.mode
  save(ledger);print(json.dumps({'role':args.mode,'job_id':attempt['job_id'],'ledger':str(ROOT/'execution.json')}))
 except BaseException:
  if attempt['state']=='submitting':attempt['state']='submission_uncertain';ledger['state']='stopped_no_retry';save(ledger)
  raise
if __name__=='__main__':main()
