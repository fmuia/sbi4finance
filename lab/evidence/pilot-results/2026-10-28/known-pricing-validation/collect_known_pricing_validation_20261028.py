"""Archive terminal accounting, immutable inputs and small diagnostic records."""
import gzip,hashlib,json,subprocess,time
from pathlib import Path

ROOT=Path('/oscar/scratch/afengler/sbi4finance-known-pricing-validation-20261028')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
started=time.perf_counter();ledger=json.loads((ROOT/'execution.json').read_text());plan=json.loads((ROOT/'plan.json').read_text())
for name,digest in {**plan['frozen_sha256'],**ledger['frozen_submission_sources']}.items():
 if sha(name)!=digest:raise ValueError('Frozeninputorsourcechanged: '+name)
if sha(ROOT/'plan.json')!=ledger['plan_sha256']:raise ValueError('Planbyteschanged')
summary=json.loads((ROOT/'summary.json').read_text())
if summary['status']!='complete' or summary['plan_sha256']!=ledger['plan_sha256']:raise ValueError('Unboundsummary')
jobs={a['role']:a['job_id'] for a in ledger['attempts']}
texts=[]
for job in jobs.values():
 r=subprocess.run(['sacct','-X','-n','-P','-j',job,'--format=JobID,State,ElapsedRaw,AllocCPUS,ExitCode'],
                  text=True,capture_output=True,timeout=25)
 if r.returncode:raise ValueError('Accountingqueryfailed: '+r.stderr)
 texts.append(r.stdout)
text=''.join(texts);(ROOT/'accounting.psv').write_text(text)
rows={r[0]:r for r in [line.split('|') for line in text.splitlines() if line.strip()]}
expected=[jobs['metadata'],jobs['pilot']]+[jobs['remaining']+'_'+str(i) for i in range(1,48)]
if any(j not in rows for j in expected):raise ValueError('Terminalaccountingpending; nocostclaimyet')
terminal=[]
for job in expected:
 r=rows[job]
 if r[1] not in ['COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL']:
  raise ValueError('Nonterminalallocation; waitforaccounting: '+job+' '+r[1])
 terminal.append({'job_id':job,'state':r[1],'elapsed_seconds':int(r[2]),'allocated_cpus':int(r[3]),
                  'exit_code':r[4],'allocated_cpu_seconds':int(r[2])*int(r[3])})
logs=[]
for p in sorted((ROOT/'logs').glob('*.out')):
 logs.append({'file':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'raw_text':p.read_text()})
with gzip.open(ROOT/'worker-logs.json.gz','wt') as f:json.dump(logs,f)
reports=[];payloads=[]
for job in plan['jobs']:
 folder=ROOT/'results'/f"batch-{job['index']:02d}"
 r=json.loads((folder/'run.json').read_text())
 if r['status']!='completed' or not r['frozen_inputs_verified']:raise ValueError('Unfinisheddriveroutput')
 reports.append(r)
 for p in sorted(folder.glob('*')):
  payloads.append({'file':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
with gzip.open(ROOT/'worker-reports.json.gz','wt') as f:json.dump(reports,f)
save(ROOT/'remote-artifacts.json',{'artifacts':payloads,'scope':'Allperbatchcovariances,momentsandpartialblockpayloadsremainonOscar;compactmeansdownloadedseparately.'})
actual=sum(r['allocated_cpu_seconds'] for r in terminal)
ledger.update(state='completed_pricing_diagnostic' if all(r['state']=='COMPLETED' and r['exit_code']=='0:0' for r in terminal) else 'driver_outputs_complete_with_allocation_failures',
              terminal_accounting=terminal,total_allocated_cpu_seconds=actual,total_allocated_cpu_hours=actual/3600,
              allocated_gpu_seconds=0,completed_pricing_batches=48,frozen_inputs_verified=True,
              summary_sha256=sha(ROOT/'summary.json'),summary_driver_sha256=summary['summary_driver_sha256'],
              collection_driver_sha256=sha(__file__),scientific_acceptance=False)
if actual>ledger['resources']['cpu_ceiling_seconds']:raise ValueError('CPUceilingexceeded')
save(ROOT/'execution.json',ledger)
save(ROOT/'collection.json',{'status':'completed','source_checks_passed':True,'completed_pricing_batches':48,
                           'actual_allocated_cpu_seconds':actual,'actual_allocated_gpu_seconds':0,
                           'worker_driver_elapsed_seconds_sum':sum(r['elapsed_seconds'] for r in reports),
                           'collection_elapsed_seconds':time.perf_counter()-started,'allocation_failures':[
                             r for r in terminal if r['state']!='COMPLETED' or r['exit_code']!='0:0']})
print(json.dumps({'state':ledger['state'],'cpu_seconds':actual,'cpu_hours':actual/3600,'pricing_batches':48}))
