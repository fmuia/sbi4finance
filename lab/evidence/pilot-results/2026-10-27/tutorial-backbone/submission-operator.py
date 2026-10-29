"""One-off serialized stage submissions; existing frozen producer does the work."""
import hashlib,json,shlex,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
LEDGER=Path('/private/tmp/heston-backbone-execution-20261027.json')
def now():return datetime.now(timezone.utc).isoformat()
def save(d):
    p=LEDGER.with_suffix('.tmp');p.write_text(json.dumps(d,indent=2)+'\n');p.replace(LEDGER)
def remote(argv,cwd=None):
    command=shlex.join(argv)
    if cwd:command='cd '+shlex.quote(cwd)+' && '+command
    return subprocess.run(['ssh','-o','BatchMode=yes','oscar',command],text=True,capture_output=True)
def budget(d):
    stages=d['stages'];ids=[s['job_id'] for s in stages if s.get('job_id')]
    if any(s['state'] in ('submitting','unknown_submission') for s in stages):raise RuntimeError('Reconcile ambiguous submission first')
    rows={}
    if ids:
        r=remote(['sacct','-S','2026-10-27','-j',','.join(ids),'-P','-n','--format=JobID%64,State%40,ElapsedRaw,AllocCPUS,ExitCode'])
        if r.returncode:raise RuntimeError('Accounting unavailable')
        for line in r.stdout.splitlines():
            a=line.split('|')
            if len(a)>=5 and '.' not in a[0]:rows[a[0]]=a
    spent=reserved=0
    for s in stages:
        if not s.get('job_id'):reserved+=s['full_cpu_seconds_including_cleanup'];continue
        for i in (s['indices'] if s['indices'] is not None else [None]):
            key=s['job_id']+(f'_{i}' if i is not None else '');a=rows.get(key);state=a[1] if a else 'UNKNOWN'
            if state=='COMPLETED':
                if a[4]!='0:0':raise RuntimeError('Nonzero completed task')
                spent+=int(a[2])*int(a[3])
            elif any(state.startswith(x) for x in ('FAILED','TIMEOUT','CANCELLED','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL')):raise RuntimeError(key+' '+state+'; review exact range before recovery')
            else:reserved+=s['cpus_per_task']*(s['wall_seconds_per_task']+s['cleanup_seconds_per_task'])
    if spent+reserved>d['allocated_CPU_hour_ceiling']*3600:raise RuntimeError('6CPU-hour ceiling exceeded')
    return {'checked_utc':now(),'completed_allocated_cpu_seconds':spent,'all_outstanding_and_unsubmitted_reserved_cpu_seconds':reserved,'total_exposure_cpu_seconds':spent+reserved}
def main():
    role=sys.argv[1];d=json.loads(LEDGER.read_text());stage=next(s for s in d['stages'] if s['role']==role)
    if stage['state']!='planned_not_submitted':raise RuntimeError('Stage already attempted; no automatic resubmission')
    if role!='preflight':
        r=remote(['cat',d['target_root']+'/preflight.json']);p=json.loads(r.stdout) if r.returncode==0 else {}
        if p.get('status')!='completed_metadata_only':raise RuntimeError('Successful preflight required')
        r=remote(['sha256sum',d['target_root']+'/jobs-100k.json'])
        if r.returncode or r.stdout.split()[0]!=p['job_map_sha256']:raise RuntimeError('Changed job map')
        pre=next(s for s in d['stages'] if s['role']=='preflight');r=remote(['sacct','-S','2026-10-27','-j',pre['job_id'],'-n','-P','--format=JobID%64,State%40,ExitCode'])
        a=[x.split('|') for x in r.stdout.splitlines() if x.split('|')[0]==pre['job_id']]
        if r.returncode or len(a)!=1 or a[0][1:3]!=['COMPLETED','0:0']:raise RuntimeError('Preflight allocation not successfully terminal')
    if role.startswith('freeze'):
        # Require all32 generation tasks terminal and successful, not just a
        # published directory or an afterok dependency that hides a failure.
        workers=next(s for s in d['stages'] if s['role']=='workers')
        if not workers.get('job_id'):raise RuntimeError('Workers not submitted')
        r=remote(['sacct','-S','2026-10-27','-j',workers['job_id'],'-n','-P','--format=JobID%64,State%40,ExitCode'])
        rows={a[0]:a[1:3] for line in r.stdout.splitlines() if len(a:=line.split('|'))>=3 and '.' not in a[0]}
        if r.returncode or any(rows.get(workers['job_id']+f'_{i}')!=['COMPLETED','0:0'] for i in workers['indices']):raise RuntimeError('All32 workers must complete before freeze')
    stage['budget_check']=budget(d);stage.update(state='submitting',attempted_utc=now(),operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());save(d)
    try:r=remote(stage['argv'],stage['submission_cwd'])
    except BaseException:stage['state']='unknown_submission';save(d);raise
    stage.update(stdout=r.stdout,stderr=r.stderr,returncode=r.returncode)
    identifier=r.stdout.strip().split(';')[0]
    if r.returncode or not identifier.isdigit():stage['state']='unknown_submission';save(d);raise RuntimeError('Ambiguous submission; reconcile, do not retry')
    stage.update(state='submitted',job_id=identifier,submitted_utc=now());save(d)
    subprocess.run(['scp',str(LEDGER),'oscar:'+d['target_root']+'/execution.json'],check=True)
    print(role,identifier,flush=True)
if __name__=='__main__':main()
