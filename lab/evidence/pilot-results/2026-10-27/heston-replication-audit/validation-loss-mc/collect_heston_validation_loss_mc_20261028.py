"""Download only this one audit's small files and reconcile actual allocation resources."""
import argparse, base64, hashlib, json, shlex, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = '/oscar/scratch/afengler/sbi4finance-heston-audit-20261027'
LEDGER = Path('/private/tmp/heston-validation-loss-mc-execution-20261028.json')
DEST = Path('/Users/afengler/Projects/proj_sbi4finance/sbi4finance-heston-audit/docs/pilot-results/2026-10-27/heston-replication-audit/validation-loss-mc')
CODE = '''import base64,hashlib,json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);job=sys.argv[2]
accounting=subprocess.check_output(['sacct','-j',job,'--parsable2','--format=JobIDRaw,State,ElapsedRaw,AllocCPUS,AllocTRES,ExitCode,CPUTimeRAW,TotalCPU,MaxRSS,Start,End,NodeList'],universal_newlines=True)
files={}
for name in ['heston_validation_loss_mc_20261028.py','validation-loss-mc-execution.json','results/validation-loss-mc/run.json','logs/validation-loss-mc-'+job+'.out']:
 p=root/name
 if p.exists():
  if p.stat().st_size>5000000:raise ValueError('Unexpected large artifact')
  b=p.read_bytes();files[name]={'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()}
print(json.dumps({'accounting':accounting,'files':files}))
'''
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, x):
    q=p.with_suffix('.json.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)
def remote(argv, **kwargs):
    r=subprocess.run(['ssh','-o','BatchMode=yes','oscar',shlex.join(argv)],capture_output=True,text=True,timeout=45,**kwargs)
    if r.returncode:raise RuntimeError(r.stderr)
    return r
p=argparse.ArgumentParser();p.add_argument('--archive',action='store_true');a=p.parse_args()
execution=json.loads(LEDGER.read_text());job=execution['job_id']
if job!='8032144':raise ValueError('Unexpected job identity')
encoded=base64.b64encode(CODE.encode()).decode()
bundle=json.loads(remote(['python3','-c','exec(__import__("base64").b64decode("'+encoded+'"))',ROOT,job]).stdout)
lines=bundle['accounting'].splitlines();columns=lines[0].split('|')
rows=[dict(zip(columns,line.split('|'),strict=True)) for line in lines[1:] if line]
allocation=next(r for r in rows if r['JobIDRaw']==job)
run_name='results/validation-loss-mc/run.json'
report=json.loads(base64.b64decode(bundle['files'][run_name]['base64'])) if run_name in bundle['files'] else {}
summary={'job_id':job,'slurm_state':allocation['State'],'elapsed_seconds':allocation['ElapsedRaw'],
         'run_status':report.get('status'),'completed_pairs':len(report.get('completed_pairs',[])),
         'summaries':report.get('summaries'),'paired_difference':report.get('paired_difference'),
         'error':report.get('error')}
print(json.dumps(summary))
if not a.archive:raise SystemExit(0)
if allocation['State'].split()[0] not in ('COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'):
    raise ValueError('Non-terminal job; do not archive an assumed outcome')
DEST.mkdir(parents=True,exist_ok=False)
downloaded={}
for name,item in bundle['files'].items():
    data=base64.b64decode(item['base64'])
    if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Download checksum mismatch')
    q=DEST/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data);downloaded[name]=item['sha256']
for name in ['submit_heston_validation_loss_mc_20261028.py','collect_heston_validation_loss_mc_20261028.py']:
    (DEST/name).write_bytes((Path('/private/tmp')/name).read_bytes())
(DEST/'accounting.psv').write_text(bundle['accounting'])
save(DEST/'execution-submitted.json',execution)
elapsed=int(allocation['ElapsedRaw']);cpus=int(allocation['AllocCPUS'])
if cpus!=4 or 'gres/gpu=1' not in allocation['AllocTRES']:raise ValueError('Unexpected actual allocation')
execution.update(state='completed' if allocation['State']=='COMPLETED' and report.get('status')=='completed' else 'failed',
    slurm_state=allocation['State'],exit_code=allocation['ExitCode'],actual_elapsed_seconds=elapsed,
    actual_allocated_cpu_seconds=cpus*elapsed,actual_gpu_seconds=elapsed,
    combined_pool_actual_cpu_seconds=672+cpus*elapsed,combined_pool_actual_gpu_seconds=168+elapsed,
    reconciled_utc=datetime.now(timezone.utc).isoformat(),cleanup_reserve_consumed_seconds=0,
    no_retry_or_further_jobs=True)
save(LEDGER,execution);save(DEST/'execution.json',execution)
update='import sys;from pathlib import Path;p=Path(sys.argv[1]);q=p.with_suffix(".tmp");q.write_text(sys.stdin.read());q.replace(p)'
remote(['python3','-c',update,ROOT+'/validation-loss-mc-execution.json'],input=LEDGER.read_text())
save(DEST/'summary.json',summary)
save(DEST/'downloaded-files.json',{'root':ROOT,'downloaded_utc':datetime.now(timezone.utc).isoformat(),'sha256':downloaded})
print(json.dumps({'archive':str(DEST),'state':execution['state'],'actual_cpu_seconds':cpus*elapsed,'actual_gpu_seconds':elapsed,
                  'combined_pool_actual_cpu_seconds':execution['combined_pool_actual_cpu_seconds'],
                  'combined_pool_actual_gpu_seconds':execution['combined_pool_actual_gpu_seconds']}))
