"""Read-only initial-shard metadata observer; no submissions or pricing."""
import datetime,gzip,hashlib,json,shlex,subprocess,time
from pathlib import Path
BASE=Path('/Users/afengler/Projects/proj_sbi4finance/sbi4finance-full-surface-core/docs/pilot-results/2026-10-27/backbone-runs/known-initial')
BASE.mkdir(parents=True,exist_ok=True)
SCRIPT=r'''
import datetime,hashlib,json,subprocess
from pathlib import Path
root=Path('/oscar/scratch/afengler/sbi4finance-rough-known-backbone-20261027')
r=subprocess.run(['sacct','-S','2026-10-27','-j','8028825','-n','-P','--format=JobID%64,State%40,ElapsedRaw,AllocCPUS,ExitCode,TotalCPU,NodeList'],capture_output=True,text=True,check=True)
parts=[]
for p in sorted((root/'collection/parts').glob('*/manifest.json')):
 raw=p.read_bytes();parts.append({'path':str(p),'manifest_sha256':hashlib.sha256(raw).hexdigest(),'record':json.loads(raw)})
failures=[]
for p in sorted((root/'collection/failures').glob('*/failure.json')):
 failures.append({'path':str(p),'text':p.read_text()})
logs=[]
for p in sorted((root/'logs').glob('workers-8028825_*.out')):
 raw=p.read_bytes();logs.append({'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode() if len(raw)<65536 else raw[-8192:].decode(errors='replace')})
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accounting_psv':r.stdout,'parts':parts,'failures':failures,'logs':logs,'staging':[p.name for p in sorted((root/'collection/.tmp').iterdir())]}))
'''
previous=None;polls=0
while True:
 r=subprocess.run(['ssh','-o','BatchMode=yes','oscar',shlex.join(['python3','-c','exec('+repr(SCRIPT)+')'])],capture_output=True,text=True)
 if r.returncode:print('READ_FAILURE',r.stderr,flush=True);raise SystemExit(2)
 d=json.loads(r.stdout);polls+=1
 with (BASE/'progress.jsonl').open('a') as f:f.write(json.dumps(d,separators=(',',':'))+'\n')
 (BASE/'latest.json').write_text(json.dumps(d,indent=2)+'\n')
 rows={}
 for line in d['accounting_psv'].splitlines():
  a=line.split('|')
  if len(a)>=5 and a[0] in {f'8028825_{i}' for i in range(4)}:rows[a[0]]=a
 states={k:v[1] for k,v in rows.items()};key=json.dumps(states,sort_keys=True)
 if key!=previous:print(json.dumps({'utc':d['utc'],'states':states,'published_shards':len(d['parts']),'failures':len(d['failures'])}),flush=True);previous=key
 bad=[v for v in rows.values() if any(v[1].startswith(s) for s in ('FAILED','TIMEOUT','CANCELLED','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE')) or (v[1]=='COMPLETED' and v[4]!='0:0')]
 done=len(rows)==4 and all(v[1]=='COMPLETED' and v[4]=='0:0' for v in rows.values())
 if bad or d['failures'] or done:
  (BASE/'accounting.psv').write_text(d['accounting_psv'])
  manifest_dir=BASE/'manifests';manifest_dir.mkdir(exist_ok=True)
  for p in d['parts']:
   m=p['record'];(manifest_dir/f"{m['split']}-{m['start']}-{m['stop']}.json").write_text(json.dumps(m,indent=2)+'\n')
  raw=(BASE/'progress.jsonl').read_bytes();(BASE/'progress.jsonl.gz').write_bytes(gzip.compress(raw,mtime=0));backup=Path('/private/tmp/known-initial-progress-20261027.jsonl');backup.write_bytes(raw);(BASE/'progress.jsonl').unlink()
  summary={'status':'all_four_completed_metadata_only' if done and not bad and not d['failures'] else 'failure_requires_review','polls':polls,'utc':d['utc'],'states':states,'checked_payloads':False,'no_pricing_calls':True,'no_neural_evaluation':True,'published_training_surfaces':sum(p['record']['stop']-p['record']['start'] for p in d['parts']),'allocated_cpu_seconds':sum(int(v[2])*int(v[3]) for v in rows.values() if v[1]=='COMPLETED'),'parts':[{k:v for k,v in p.items() if k!='record'} for p in d['parts']],'failed_rows':bad,'failure_records':d['failures'],'staging':d['staging'],'progress_uncompressed_sha256':hashlib.sha256(raw).hexdigest()}
  (BASE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
  print('TERMINAL',json.dumps(summary),flush=True);break
 time.sleep(35)
