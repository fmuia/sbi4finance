"""One reviewed recovery, one submitting owner; no automatic retry/refill."""
import json,subprocess,shlex,hashlib,sys
from pathlib import Path
from datetime import datetime,timezone
LOCAL=Path('/private/tmp/joint-timeout-recovery-execution-20261027.json')
PARENT=Path('/private/tmp/sbi4finance-native-bank-execution-20261027.json')
ROOT='/oscar/scratch/afengler/sbi4finance-native-bank-20261027'
UNIT=ROOT+'/recovery-378-380-20261027'
def now():return datetime.now(timezone.utc).isoformat()
def remote(args,cwd=None):
 cmd=shlex.join(args)
 if cwd:cmd='cd '+shlex.quote(cwd)+' && '+cmd
 return subprocess.run(['ssh','-o','BatchMode=yes','oscar',cmd],capture_output=True,text=True)
def save(d):
 tmp=LOCAL.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(LOCAL)
 parent=json.loads(PARENT.read_text());bank=parent['bank'];bank['manual_recovery']={'unit':'three_timeout_ranges_378_380','status':d['status'],'approved_scope':d['approved_scope'],'unit_cpu_hour_ceiling':9.5,'unit_execution':UNIT+'/execution.json','stages':d['stages'],'old_timeouts_retained':True,'no_wider_refill':True}
 for s in d['stages']:
  if s['state']=='planned_not_submitted':continue
  key=s['parent_attempt_key'];entry=next((x for x in bank['jobs'] if x.get('manual_recovery_attempt_key')==key),None)
  row={'role':'worker_recovery' if s['role']=='workers' else 'recovery_integrity','manual_recovery_attempt_key':key,'manual_review_exception':'Original8019485_378/379/380 timedout; explicitly approved same-ID recovery excludingnode2338. No automatic duplicate-index retry.','source_revision':d['source_revision'],'state':s['state'],'indices':s['indices']} if s['indices'] is not None else {'role':'recovery_integrity','manual_recovery_attempt_key':key,'state':s['state'],'source_revision':d['source_revision']}
  row.update(cpu_seconds_per_task_cap=s['wall_seconds_per_task'],arguments=s['argv'],submission_cwd=s['submission_cwd'])
  for field in ('job_id','budget_check','attempted_utc','submitted_utc','stdout','stderr','returncode','operator_sha256'):
   if field in s:row[field]=s[field]
  if entry is None:bank['jobs'].append(row)
  else:entry.update(row)
 bank['status']='manual_recovery_'+d['status'];tmp=PARENT.with_suffix('.tmp');tmp.write_text(json.dumps(parent,indent=2)+'\n');tmp.replace(PARENT)
def sync():
 subprocess.run(['scp',str(LOCAL),'oscar:'+UNIT+'/execution.json'],check=True,capture_output=True)
 subprocess.run(['scp',str(PARENT),'oscar:'+ROOT+'/execution.json'],check=True,capture_output=True)
def rows_for(ids):
 if not ids:return {}
 r=remote(['sacct','-S','2026-10-27','-j',','.join(ids),'-n','-P','--format=JobID%64,State%40,ElapsedRaw,AllocCPUS,ExitCode'])
 if r.returncode:raise RuntimeError('Accounting unavailable; no submission')
 return {a[0]:a for line in r.stdout.splitlines() if len(a:=line.split('|'))>=5 and '.' not in a[0]}
def budget(d):
 if any(s['state'] in ('submitting','unknown_submission') for s in d['stages']):raise RuntimeError('Reconcile ambiguous previous submission')
 rows=rows_for([s['job_id'] for s in d['stages'] if s.get('job_id')]);spent=reserved=0
 for s in d['stages']:
  if not s.get('job_id'):reserved+=s['full_cpu_seconds_including_cleanup'];continue
  for i in (s['indices'] if s['indices'] is not None else [None]):
   key=s['job_id']+(f'_{i}' if i is not None else '');row=rows.get(key);state=row[1] if row else 'UNKNOWN'
   if state=='COMPLETED':
    if row[4]!='0:0':raise RuntimeError('Nonzero completed recovery task')
    spent+=int(row[2])*int(row[3])
   elif any(state.startswith(x) for x in ('FAILED','TIMEOUT','CANCELLED','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL')):raise RuntimeError(key+' '+state+'; stop this one-shot recovery')
   else:reserved+=s['wall_seconds_per_task']+240
 unit=spent+reserved;global_exposure=d['baseline_bank_allocated_cpu_seconds']+unit+d['final_snapshot_cpu_seconds_reserved']
 if unit>34200 or global_exposure>1500*3600:raise RuntimeError('Recovery/bank spending ceiling exceeded')
 return {'checked_utc':now(),'completed_unit_allocated_cpu_seconds':spent,'all_outstanding_unsubmitted_full_caps_cleanup_cpu_seconds':reserved,'unit_exposure_cpu_seconds':unit,'bank_plus_final_snapshot_exposure_cpu_seconds':global_exposure}
def main():
 role=sys.argv[1];d=json.loads(LOCAL.read_text());stage=next(s for s in d['stages'] if s['role']==role)
 if stage['state']!='planned_not_submitted':raise RuntimeError('Stage already attempted; no automatic retry')
 if role!='preflight':
  pre=next(s for s in d['stages'] if s['role']=='preflight');rows=rows_for([pre['job_id']]);r=rows.get(pre['job_id'])
  if r is None or r[1]!='COMPLETED' or r[4]!='0:0':raise RuntimeError('Successful terminal allocated preflight required')
  result=remote(['cat',UNIT+'/preflight.json']);checked=json.loads(result.stdout) if result.returncode==0 else {}
  if checked.get('status')!='completed_integrity_preflight' or checked.get('source_revision')!=d['source_revision'] or checked.get('job_map_sha256')!=d['job_map_sha256']:raise RuntimeError('Recovery preflight identity mismatch')
  result=remote(['sha256sum',ROOT+'/jobs-20k.json',ROOT+'/collection/collection.json'])
  if result.returncode or [x.split()[0] for x in result.stdout.splitlines()]!=[d['job_map_sha256'],d['collection_sha256']]:raise RuntimeError('Frozen map/collection changed')
 if role=='postcheck':
  workers=next(s for s in d['stages'] if s['role']=='workers');rows=rows_for([workers['job_id']])
  if any((r:=rows.get(workers['job_id']+f'_{i}')) is None or r[1]!='COMPLETED' or r[4]!='0:0' for i in workers['indices']):raise RuntimeError('All three original ranges must complete successfully')
 stage['budget_check']=budget(d);stage.update(state='submitting',attempted_utc=now(),operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());d['status']=role+'_submitting';save(d);sync()
 try:r=remote(stage['argv'],stage['submission_cwd'])
 except BaseException:stage['state']='unknown_submission';d['status']='unknown_submission';save(d);sync();raise
 stage.update(stdout=r.stdout,stderr=r.stderr,returncode=r.returncode);identifier=r.stdout.strip().split(';')[0]
 if r.returncode or not identifier.isdigit():stage['state']='unknown_submission';d['status']='unknown_submission';save(d);sync();raise RuntimeError('Ambiguous submission; stop, reconcile, never blind retry')
 stage.update(state='submitted',job_id=identifier,submitted_utc=now());d['status']=role+'_submitted';save(d);sync();print(role,identifier,flush=True)
if __name__=='__main__':main()
