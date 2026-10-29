"""Bounded read-only observation of the single known-curve fit allocation."""
import datetime,json,pathlib,shlex,subprocess,time

OUT=pathlib.Path('/Users/afengler/Projects/proj_sbi4finance/sbi4finance-full-surface-core/docs/pilot-results/2026-10-27/backbone-runs/known-fit')
REMOTE="""import json,pathlib,subprocess
p=pathlib.Path('/oscar/scratch/afengler/sbi4finance-rough-known-fit-20261027/runs/known-20k-native/run.json')
r={'accounting':subprocess.check_output(['sacct','-S2026-10-27','-j8030774','-n','-P','--format=JobID%64,State%40,ElapsedRaw,AllocCPUS,ExitCode']).decode()}
if p.exists():
 d=json.loads(p.read_text());r['run']={k:d.get(k) for k in ['status','seconds','failure','selected_posterior_epoch','selected_pricer_epoch']}
 r['stages']=[{k:s.get(k) for k in ['name','status','seconds','returncode','timeout_cap_seconds']} for s in d['stages']]
print(json.dumps(r))
"""

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def observe():
 r=subprocess.run(['ssh','oscar',shlex.join(['python3','-c','exec('+repr(REMOTE)+')'])],capture_output=True,text=True,timeout=30)
 if r.returncode:raise RuntimeError('read-only SSH observation failed: '+r.stderr)
 data=json.loads(r.stdout);data['observed_utc']=now()
 rows=[line.split('|') for line in data['accounting'].splitlines() if line.strip()]
 top=[r for r in rows if r[0]=='8030774']
 if len(top)!=1:raise RuntimeError('ambiguous top-level Slurm accounting')
 row=top[0];data['state']=row[1];data['elapsed_seconds']=int(row[2]);data['allocated_cpus']=int(row[3]);data['exit_code']=row[4]
 return data

OUT.mkdir(parents=True,exist_ok=True)
for poll in range(32):
 data=observe();data['poll']=poll
 with (OUT/'monitor-progress.jsonl').open('a') as f:f.write(json.dumps(data,sort_keys=True)+'\n')
 (OUT/'monitor-state.json').write_text(json.dumps(data,indent=2)+'\n')
 print(json.dumps(data),flush=True)
 if data['state'] not in ('PENDING','RUNNING','COMPLETING','CONFIGURING'):
  if data['state']!='COMPLETED' or data['exit_code']!='0:0' or data.get('run',{}).get('status')!='completed_development_checks':
   raise RuntimeError('terminal fit is unsuccessful; retain evidence, no retry')
  break
 time.sleep(45)
else:raise RuntimeError('bounded observer ended; no job changes made')
