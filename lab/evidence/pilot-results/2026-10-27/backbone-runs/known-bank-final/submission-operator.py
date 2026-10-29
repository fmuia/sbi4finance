"""One-off serialized, budget-checked known-curve collection submissions."""
import argparse,hashlib,json,shlex,subprocess
from pathlib import Path
from datetime import datetime,timezone
PATH=Path('/private/tmp/rough-known-backbone-execution-20261027.json')
def now():return datetime.now(timezone.utc).isoformat()
def save(d):
 p=PATH.with_suffix('.tmp');p.write_text(json.dumps(d,indent=2)+'\n');p.replace(PATH)
def remote(argv,cwd=None):
 cmd=shlex.join(argv)
 if cwd:cmd='cd '+shlex.quote(cwd)+' && '+cmd
 return subprocess.run(['ssh','-o','BatchMode=yes','oscar',cmd],capture_output=True,text=True)
def accounting(d):
 ids=[j['job_id'] for j in d['jobs'] if j.get('job_id')];rows={}
 if ids:
  r=remote(['sacct','-S','2026-10-27','-j',','.join(ids),'-n','-P','--format=JobID%64,State%40,ElapsedRaw,AllocCPUS,ExitCode'])
  if r.returncode:raise RuntimeError('Accounting unavailable')
  for line in r.stdout.splitlines():
   a=line.split('|')
   if len(a)>=5 and '.' not in a[0]:rows[a[0]]=a
 return rows
def budget(d,rows,new):
 if any(j['state'] in ('submitting','unknown_submission') for j in d['jobs']):raise RuntimeError('Reconcile ambiguous submission first')
 spent=outstanding=workers=0
 for j in d['jobs']:
  for index in j.get('indices',[None]):
   key=j['job_id']+(f'_{index}' if index is not None else '');r=rows.get(key);state=r[1] if r else 'UNKNOWN'
   if state=='COMPLETED':
    if r[4]!='0:0':raise RuntimeError('Nonzero completed task')
    spent+=int(r[2])*int(r[3])
   elif any(state.startswith(x) for x in ('FAILED','TIMEOUT','CANCELLED','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE')):raise RuntimeError(key+' '+state+'; retain range/diagnostics and stop expansion')
   else:
    outstanding+=j['wall_seconds_per_task']+240
    if j['role'] in ('initial-workers','later-workers'):workers+=1
 reserved=(0 if (new['role']=='final-snapshot' or any(j['role']=='final-snapshot' for j in d['jobs'])) else 3960)+(0 if (new['role']=='initial-integrity' or any(j['role']=='initial-integrity' for j in d['jobs'])) else 840)
 exposure=spent+outstanding+reserved+new['tasks']*(new['wall_seconds_per_task']+240)
 if exposure>150*3600:raise RuntimeError('150CPUh ceiling exceeded')
 if new['role'] in ('initial-workers','later-workers') and workers+new['tasks']>192:raise RuntimeError('Global192-worker cap exceeded')
 return {'checked_utc':now(),'completed_allocated_cpu_seconds':spent,'outstanding_reserved_cpu_seconds':outstanding,'unsubmitted_integrity_reserved_cpu_seconds':reserved,'after_submission_exposure_cpu_seconds':exposure,'existing_nonterminal_workers':workers}
def main():
 p=argparse.ArgumentParser();p.add_argument('role');p.add_argument('--indices');a=p.parse_args();d=json.loads(PATH.read_text());root=d['target_root'];repo=d['producer_repo'];row=json.loads(json.dumps(next(s for s in d['stages'] if s['role']==a.role)));rows=accounting(d)
 if a.role!='later-workers' and any(j['role']==a.role for j in d['jobs']):raise RuntimeError('Stage already attempted; no automatic retry')
 if a.role!='preflight':
  r=remote(['cat',root+'/preflight.json']);pre=json.loads(r.stdout) if r.returncode==0 else {}
  if pre.get('status')!='completed_metadata_only' or pre.get('source_revision')!=d['source_revision']:raise RuntimeError('Successful frozen preflight required')
  r=remote(['sha256sum',root+'/jobs-20k.json'])
  if r.returncode or r.stdout.split()[0]!=pre['job_map_sha256']:raise RuntimeError('Changed map')
  pj=next(j for j in d['jobs'] if j['role']=='preflight');pr=rows.get(pj['job_id'])
  if not pr or pr[1]!='COMPLETED' or pr[4]!='0:0':raise RuntimeError('Preflight allocation must complete successfully')
 if a.role in ('initial-integrity','final-snapshot'):
  expected=set(range(4 if a.role=='initial-integrity' else 1280));submitted={i:(j['job_id'],i) for j in d['jobs'] for i in j.get('indices',[])}
  if not expected<=submitted.keys() or any(rows.get(submitted[i][0]+f'_{i}',[None,None,None,None,None])[1:5:3]!=['COMPLETED','0:0'] for i in expected):raise RuntimeError('Required worker ranges must all complete successfully')
 if a.role=='later-workers':
  r=remote(['cat',root+'/initial-check.json']);gate=json.loads(r.stdout) if r.returncode==0 else {}
  if (gate.get('status')!='completed_initial_gate' or gate.get('surfaces_above_nominal_target')!=0
      or gate.get('collection_id')!=pre.get('collection_id')
      or gate.get('config_sha256')!=d['files']['config']['sha256']
      or gate.get('audit_sha256')!=d['files']['seed_audit']['sha256']
      or gate.get('source_revision')!=d['source_revision']):raise RuntimeError('Exact initial100 integrity/precision gate required before expansion')
  gate_job=next((j for j in d['jobs'] if j['role']=='initial-integrity'),None)
  gr=rows.get(gate_job['job_id']) if gate_job else None
  if not gr or gr[1]!='COMPLETED' or gr[4]!='0:0':raise RuntimeError('Initial integrity allocation must complete successfully')
  indices=[int(v) for v in a.indices.split(',')] if a.indices else []
  if not indices or len(set(indices))!=len(indices) or min(indices)<0 or max(indices)>=1280:raise ValueError('Distinct original indices0..1279 required')
  submitted={i for j in d['jobs'] for i in j.get('indices',[])}
  if submitted.intersection(indices):raise RuntimeError('Duplicate/recovery indices require review')
  row.update(indices=indices,tasks=len(indices));row['argv']=[x.replace('__ORIGINAL_INDICES__',','.join(map(str,indices))).replace('__THROTTLE_LE_192__',str(min(192,len(indices)))) for x in row['argv']]
 row['budget_check']=budget(d,rows,row);row.update(state='submitting',attempted_utc=now(),operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());d['jobs'].append(row);save(d)
 try:r=remote(row['argv'],repo)
 except BaseException:row['state']='unknown_submission';save(d);raise
 row.update(stdout=r.stdout,stderr=r.stderr,returncode=r.returncode);identifier=r.stdout.strip().split(';')[0]
 if r.returncode or not identifier.isdigit():row['state']='unknown_submission';save(d);raise RuntimeError('Ambiguous submission: reconcile before retry')
 row.update(state='submitted',job_id=identifier,submitted_utc=now());save(d);subprocess.run(['scp',str(PATH),'oscar:'+root+'/execution.json'],check=True);print(a.role,identifier,flush=True)
if __name__=='__main__':main()
