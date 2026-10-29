"""Reviewed one-time recovery of an unsubmitted blocked training waiter."""
import argparse, fcntl, hashlib, json, os, signal, socket, subprocess, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def save(p,x):
 q=Path(p).with_suffix('.json.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)
def no_attempts(root, ledger):
 if set(ledger['routes']) != {'known','joint'}: raise ValueError('Unexpected original routes')
 for name,row in ledger['routes'].items():
  if row.get('state') not in ('waiting','blocked'): raise ValueError('Original route may have launched work')
  if any(k in row for k in ('attempted_utc','argv','job_id','bank','stdout')) or (root/name).exists():
   raise ValueError('Original route has a recorded or possible scheduler attempt')

def main():
 p=argparse.ArgumentParser();p.add_argument('--old-root',type=Path,required=True);p.add_argument('--new-root',type=Path,required=True)
 p.add_argument('--repo',type=Path,required=True);p.add_argument('--revision',required=True);p.add_argument('--python',type=Path,required=True)
 p.add_argument('--old-pid',type=int,required=True);a=p.parse_args()
 receipt_root=a.new_root.with_name(a.new_root.name+'-receipt');receipt_root.mkdir(parents=True,exist_ok=False)
 receipt={'format':'post_bank_waiter_recovery_v2','status':'preflight','host':socket.gethostname(),
  'old_root':str(a.old_root),'new_root':str(a.new_root),'new_source_revision':a.revision,'old_pid':a.old_pid,
  'prepared_utc':datetime.now(timezone.utc).isoformat(),'same_original_budget_pool':True,
  'cpu_hour_ceiling':9,'gpu_hour_ceiling':2.25,'scientific_acceptance':False,'automatic_retry':False}
 target=receipt_root/'recovery.json';save(target,receipt)
 try:
  old_plan=read(a.old_root/'plan.json');old=read(a.old_root/'execution.json');no_attempts(a.old_root,old)
  if old_plan['resources']['cpu_hour_ceiling']!=9 or old_plan['resources']['gpu_hour_ceiling']!=2.25:
   raise ValueError('Original approved resource pool differs')
  for name in ('plan.json','execution.json'):
   (receipt_root/('original-'+name)).write_bytes((a.old_root/name).read_bytes())
  receipt['old_plan_sha256']=sha(a.old_root/'plan.json');receipt['old_execution_before_sha256']=sha(a.old_root/'execution.json')
  command=Path(f'/proc/{a.old_pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
  expected=str(Path(old_plan['repo'])/'cluster/submit_full_surface_backbones_20261028.py')
  if expected not in command or 'watch --root '+str(a.old_root) not in command:
   raise ValueError('Old PID is not the exact declared watcher on this host')
  receipt['old_command']=command
  revision=subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'],text=True).strip()
  dirty=subprocess.check_output(['git','-C',str(a.repo),'status','--porcelain','--untracked-files=no'],text=True)
  if revision!=a.revision or dirty:raise ValueError('New source is not the reviewed clean revision')
  since=(datetime.fromisoformat(old_plan['prepared_utc'])-timedelta(days=1)).strftime('%Y-%m-%dT00:00:00')
  query=subprocess.run(['sacct','-u','afengler','-S',since,
   '--name=sbi4f-known-baseline,sbi4f-joint-baseline','-n','-P','--format=JobIDRaw,JobName,State,ElapsedRaw,AllocCPUS'],
   capture_output=True,text=True,timeout=25)
  receipt['original_job_name_accounting_query']={'returncode':query.returncode,'stdout':query.stdout,'stderr':query.stderr}
  if query.returncode or query.stdout.strip():raise ValueError('Original job-name accounting is not unambiguously empty')
  receipt.update(status='stop_old_watcher_authorized',new_operator_sha256=sha(a.repo/'cluster/submit_full_surface_backbones_20261028.py'))
  save(target,receipt);os.kill(a.old_pid,signal.SIGTERM)
  for _ in range(100):
   proc=Path(f'/proc/{a.old_pid}/stat')
   if not proc.exists() or proc.read_text().split()[2]=='Z':break
   time.sleep(.1)
  else:raise RuntimeError('Old metadata watcher did not stop')
  with (a.old_root/'operator.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   stopped=read(a.old_root/'execution.json');no_attempts(a.old_root,stopped)
   (receipt_root/'original-execution-stopped.json').write_bytes((a.old_root/'execution.json').read_bytes())
   receipt.update(status='old_watcher_stopped_no_jobs',old_execution_stopped_sha256=sha(a.old_root/'execution.json'))
   save(target,receipt)
   script=a.repo/'cluster/submit_full_surface_backbones_20261028.py'
   subprocess.run([str(a.python),str(script),'initialize','--root',str(a.new_root),'--repo',str(a.repo),
                  '--revision',a.revision,'--python',str(a.python)],check=True,timeout=90)
   plan=read(a.new_root/'plan.json')
   plan['continuation_of']={'original_root':str(a.old_root),'original_plan_sha256':receipt['old_plan_sha256'],
    'original_allocations':[],'original_allocated_cpu_seconds':0,'original_allocated_gpu_seconds':0,
    'same_resource_pool':True,'receipt':str(target)}
   save(a.new_root/'plan.json',plan);ledger=read(a.new_root/'execution.json')
   ledger['plan_sha256']=sha(a.new_root/'plan.json');save(a.new_root/'execution.json',ledger)
   ready=read(plan['routes']['known']['ready'])
   expected_bank='412f3cf437895b4764fa981bedbc1f8ad0f532d6f5b18e1b1fb957297cb627c4'
   if ready.get('snapshot_manifest_sha256')!=expected_bank or sha(Path(plan['routes']['known']['bank'])/'manifest.json')!=expected_bank:
    raise ValueError('Known50k snapshot differs from reviewed completed bank')
   receipt.update(status='new_phase_initialized_before_submission',new_plan_sha256=sha(a.new_root/'plan.json'))
   save(target,receipt)
   subprocess.run([str(a.python),str(script),'once','--root',str(a.new_root)],check=True,timeout=120)
   ledger=read(a.new_root/'execution.json');known=ledger['routes']['known']
   receipt['new_execution_after_once']=ledger
   if known['state']!='submitted' or not known.get('job_id'):raise RuntimeError('Known training submission not confirmed; do not retry')
   with (a.new_root/'watcher.log').open('a') as log:
    process=subprocess.Popen([str(a.python),str(script),'watch','--root',str(a.new_root)],stdin=subprocess.DEVNULL,
                             stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   receipt.update(status='known_training_submitted_and_watcher_running',known_job_id=known['job_id'],new_watcher_pid=process.pid)
   save(target,receipt)
   save(a.old_root/'retired-by-recovery-v2.json',{'receipt':str(target),'old_watcher_stopped':True,
        'replacement_root':str(a.new_root),'new_watcher_pid':process.pid,'known_job_id':known['job_id']})
   time.sleep(1)
   if process.poll() is not None:raise RuntimeError('New metadata watcher exited; submitted job is retained')
  print(json.dumps({'status':receipt['status'],'known_job_id':receipt['known_job_id'],
                    'new_watcher_pid':receipt['new_watcher_pid'],'host':receipt['host'],'receipt':str(target)}))
 except BaseException as error:
  receipt.update(status='recovery_stopped_for_review',error_type=type(error).__name__,error=str(error));save(target,receipt);raise
if __name__=='__main__':main()
