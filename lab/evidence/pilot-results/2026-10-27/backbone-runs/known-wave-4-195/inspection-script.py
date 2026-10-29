"""Read-only wave observer. No scheduling, financial pricing or payload reads."""
import gzip,hashlib,json,shlex,subprocess,time
from pathlib import Path
BASE=Path('/Users/afengler/Projects/proj_sbi4finance/sbi4finance-full-surface-core/docs/pilot-results/2026-10-27/backbone-runs/known-wave-4-195');BASE.mkdir(parents=True,exist_ok=True)
SCRIPT=r'''
import datetime,hashlib,json,subprocess
from pathlib import Path
root=Path('/oscar/scratch/afengler/sbi4finance-rough-known-backbone-20261027')
fields='JobID%64,State%40,ElapsedRaw,AllocCPUS,ExitCode,TotalCPU,NodeList,ReqMem,TimelimitRaw'
r=subprocess.run(['sacct','-S','2026-10-27','-j','8028945','-n','-P','--format='+fields],capture_output=True,text=True,check=True)
failures=[{'path':str(p),'text':p.read_text()} for p in sorted((root/'collection/failures').glob('*/failure.json'))]
parts=[p.name for p in sorted((root/'collection/parts').iterdir())]
logs=[]
for p in sorted((root/'logs').glob('workers-8028945_*.out')):
 raw=p.read_bytes();text=raw.decode(errors='replace');errors=[x for x in text.splitlines() if any(s in x for s in ('Traceback','Error:','Exception:','CANCELLED','TIME LIMIT','Killed'))]
 logs.append({'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'errors':errors,'tail':text[-4096:] if errors else None})
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accounting_psv':r.stdout,'published_part_names':parts,'failures':failures,'logs':logs,'staging':[p.name for p in sorted((root/'collection/.tmp').iterdir())]}))
'''
SOURCE=r'''
import hashlib,json,subprocess
from pathlib import Path
r=Path('/oscar/scratch/afengler/sbi4finance-rough-known-backbone-20261027');repo=Path('/oscar/scratch/afengler/sbi4finance-backbone-fit-20261027/repo')
print(json.dumps({'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'source_status':subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True),'collection_sha256':hashlib.sha256((r/'collection/collection.json').read_bytes()).hexdigest(),'map_sha256':hashlib.sha256((r/'jobs-20k.json').read_bytes()).hexdigest(),'preflight':json.loads((r/'preflight.json').read_text()),'first_task_scontrol':subprocess.check_output(['scontrol','show','job','8028945_4','-o'],text=True)}))
'''
def read(script):
 r=subprocess.run(['ssh','-o','BatchMode=yes','oscar',shlex.join(['python3','-c','exec('+repr(script)+')'])],capture_output=True,text=True)
 if r.returncode:raise RuntimeError(r.stderr)
 return json.loads(r.stdout)
source=read(SOURCE)
assert source['source_revision']=='616352035be4b9d271151090aa5d2d9539da8b1c' and source['source_status']==''
assert source['collection_sha256']=='9e9e0ed2f087febd84bbd42ddc30aa24840d065c12a186b410851a68b5c83e13'
assert source['map_sha256']=='85382da5a542d0f9466124a936b3025e62b76a6150b4ea2fffedc66326cf6974'
(BASE/'source-resource-check.json').write_text(json.dumps(source,indent=2)+'\n')
print('SOURCE_PASS',source['source_revision'],source['first_task_scontrol'],flush=True)
last_report=-1;polls=0
while True:
 d=read(SCRIPT);polls+=1
 with (BASE/'progress.jsonl').open('a') as f:f.write(json.dumps(d,separators=(',',':'))+'\n')
 (BASE/'latest.json').write_text(json.dumps(d,indent=2)+'\n')
 rows={}
 for line in d['accounting_psv'].splitlines():
  a=line.split('|')
  if len(a)>=9 and a[0] in {f'8028945_{i}' for i in range(4,196)}:rows[a[0]]=a
 assert all(a[3]=='1' and a[7]=='8G' and a[8]=='30' for a in rows.values())
 completed=[a for a in rows.values() if a[1]=='COMPLETED' and a[4]=='0:0'];bad=[a for a in rows.values() if any(a[1].startswith(x) for x in ('FAILED','TIMEOUT','CANCELLED','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE')) or (a[1]=='COMPLETED' and a[4]!='0:0')]
 counts={}
 for a in rows.values():counts[a[1]]=counts.get(a[1],0)+1
 report={'utc':d['utc'],'states':counts,'completed':len(completed),'known_wave_nonterminal_or_unknown':192-len(completed)-len(bad),'freed_slots_this_wave':len(completed),'published_training_surfaces':len(d['published_part_names'])*25,'completed_allocated_cpu_seconds':sum(int(a[2])*int(a[3]) for a in completed),'hard_failures':len(bad),'durable_failures':len(d['failures']),'log_errors':sum(bool(x['errors']) for x in d['logs'])}
 if last_report<0 or len(completed)//24>last_report or bad or d['failures']:
  print('PROGRESS',json.dumps(report),flush=True);last_report=len(completed)//24
 done=len(completed)==192
 if done or bad or d['failures']:
  (BASE/'accounting.psv').write_text(d['accounting_psv']);(BASE/'log-inventory.json').write_text(json.dumps(d['logs'],indent=2)+'\n')
  raw=(BASE/'progress.jsonl').read_bytes();(BASE/'progress.jsonl.gz').write_bytes(gzip.compress(raw,mtime=0));Path('/private/tmp/known-wave-4-195-progress-20261027.jsonl').write_bytes(raw);(BASE/'progress.jsonl').unlink()
  report.update(status='all192_completed_metadata_only' if done and not bad and not d['failures'] else 'failure_requires_review',polls=polls,failed_rows=bad,failure_records=d['failures'],checked_payloads=False,no_financial_pricing=True,no_neural_evaluation=True,progress_uncompressed_sha256=hashlib.sha256(raw).hexdigest())
  (BASE/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print('TERMINAL',json.dumps(report),flush=True);break
 time.sleep(45)
