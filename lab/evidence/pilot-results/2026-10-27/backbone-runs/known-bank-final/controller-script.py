"""One-off finite wave loop; ONLY the existing guarded helper submits jobs."""
import hashlib,importlib.util,json,math,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
HELPER=Path('/private/tmp/submit_known_backbone_20261027.py')
EXPECTED_HELPER='1d0421569471262af6c05ffe7ed5fce346e6f519cbf170ec2b120118a7955759'
BASE=Path('/Users/afengler/Projects/proj_sbi4finance/sbi4finance-full-surface-core/docs/pilot-results/2026-10-27/backbone-runs/known-bank-final');BASE.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('guarded_known_submitter',HELPER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
START=time.monotonic()
def now():return datetime.now(timezone.utc).isoformat()
def write(report):
 report.update(controller_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest(),elapsed_controller_wall_seconds=time.monotonic()-START,controller_pid=os.getpid())
 temp=BASE/'controller-state.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(BASE/'controller-state.json')
 with (BASE/'controller-progress.jsonl').open('a') as f:f.write(json.dumps(report,separators=(',',':'))+'\n')
def submit(role,indices=None):
 cmd=[sys.executable,str(HELPER),role]
 if indices is not None:cmd+=['--indices',','.join(map(str,indices))]
 r=subprocess.run(cmd,capture_output=True,text=True)
 print('SUBMISSION',role,r.returncode,r.stdout.strip(),r.stderr.strip(),flush=True)
 if r.returncode:raise RuntimeError('Guarded helper refused/stopped submission: '+r.stdout+r.stderr)
 return r.stdout.strip()
try:
 for poll in range(480):
  assert hashlib.sha256(HELPER.read_bytes()).hexdigest()==EXPECTED_HELPER,'Guarded helper changed; owner review required'
  d=json.loads(m.PATH.read_text());assert d['source_revision']=='616352035be4b9d271151090aa5d2d9539da8b1c' and d['allocated_CPU_hour_ceiling']==150 and d['maximum_known_curve_nonterminal_workers']==192
  assert d['configuration']['root_seed']==271300 and d['configuration']['splits']=={'train':20000,'validation':5000,'test':7000}
  rows=m.accounting(d)
  for row in rows.values():
   if row[1] not in ('COMPLETED','RUNNING','PENDING','COMPLETING','CONFIGURING','SUSPENDED','RESIZING','REQUEUED','REQUEUE_FED','REQUEUE_HOLD','SIGNALING','STAGE_OUT'):raise RuntimeError('Terminal or unrecognized scheduler state requires review: '+row[0]+' '+row[1])
  b=m.budget(d,rows,{'role':'later-workers','tasks':0,'wall_seconds_per_task':1800})
  worker_jobs=[j for j in d['jobs'] if j['role'] in ('initial-workers','later-workers')];submitted={i for j in worker_jobs for i in j['indices']};missing=[i for i in range(1280) if i not in submitted]
  done=0
  for j in worker_jobs:
   for i in j['indices']:
    r=rows.get(j['job_id']+f'_{i}')
    if r and r[1]=='COMPLETED' and r[4]=='0:0':done+=1
  free=192-b['existing_nonterminal_workers'];affordable=max(0,math.floor((150*3600-b['after_submission_exposure_cpu_seconds'])/2040));n=min(free,affordable,len(missing))
  report={'utc':now(),'poll':poll,'status':'advancing','submitted_worker_tasks':len(submitted),'completed_worker_tasks':done,'published_surface_count_implied_by_completed_tasks':done*25,'missing_worker_tasks':len(missing),'known_nonterminal_or_unknown_workers':b['existing_nonterminal_workers'],'free_known_worker_slots':free,'budget_affordable_new_workers':affordable,'budget':b}
  if n:
   selected=missing[:n];result=submit('later-workers',selected);report.update(action='submitted_refill',new_original_indices=selected,new_submission=result);write(report)
  elif done==1280:
   final=[j for j in d['jobs'] if j['role']=='final-snapshot']
   if not final:
    f=d['files']['final_summary_script'];r=m.remote(['sha256sum',f['target_path']]);assert r.returncode==0 and r.stdout.split()[0]==f['sha256'],'Final summary script changed'
    result=submit('final-snapshot');report.update(action='submitted_final_snapshot',new_submission=result);write(report)
   else:
    r=rows.get(final[0]['job_id'])
    if r and r[1]=='COMPLETED' and r[4]=='0:0':
     result=m.remote(['cat',d['target_root']+'/final-check.json']);assert result.returncode==0;check=json.loads(result.stdout)
     assert check['status']=='completed_checked_snapshot_and_integrity' and check['verified_examples']==32000 and check['verified_parts']==1280
     (BASE/'final-check.json').write_text(result.stdout);report.update(status='completed_full_checked_bank',action='finished',snapshot_manifest_sha256=check['snapshot_manifest_sha256'],nominal_target_exceptions=check['total_nominal_target_exception_cases'],hard_cap_exceptions=check['hard_cap_exceptions']);write(report);print('COMPLETED',json.dumps(report),flush=True);break
    report['action']='waiting_final_snapshot';write(report)
  elif not missing:
   report['action']='waiting_submitted_workers';write(report)
  elif b['existing_nonterminal_workers']==0 and affordable==0:
   report.update(status='stopped_budget_insufficient',action='no_cap_increase');write(report);raise RuntimeError('Remaining budget cannot reserve another original task plus final integrity')
  else:report['action']='waiting_for_capacity_or_budget_release';write(report)
  if poll%4==0 or report.get('action') in ('submitted_refill','submitted_final_snapshot'):print('PROGRESS',json.dumps({k:v for k,v in report.items() if k not in ('new_original_indices',)}),flush=True)
  time.sleep(45)
 else:raise TimeoutError('Bounded480-poll observer allowance reached; retain jobs and ledger for review')
except BaseException as e:
 report={'utc':now(),'status':'stopped_requires_review','error_type':type(e).__name__,'error':str(e)};write(report);print('STOP',json.dumps(report),flush=True);raise
