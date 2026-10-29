"""Collect terminal provenance for the single declared case; never submit work."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('/oscar/scratch/afengler/sbi4finance-interior-corners-20261028')
JOB='8032768'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def main():
    plan=json.loads((ROOT/'manifest.json').read_text())
    for name,digest in plan['frozen_sha256'].items():
        if sha(Path(name))!=digest:raise ValueError('Frozen input changed: '+name)
    run=json.loads((ROOT/'run.json').read_text())
    result=subprocess.run(['sacct','-n','-P','-j',JOB,'--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES'],
                          capture_output=True,text=True,check=True,timeout=30)
    (ROOT/'accounting.psv').write_text('JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|AllocTRES\n'+result.stdout)
    rows=[line.split('|') for line in result.stdout.splitlines() if line.strip()]
    row=next((r for r in rows if r[0]==JOB),None)
    if row is None or row[1] not in ('COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY'):
        raise ValueError('Allocation has no known terminal state; do not resubmit')
    original=ROOT/'execution-submitted.json'
    if not original.exists():original.write_bytes((ROOT/'execution.json').read_bytes())
    execution=json.loads(original.read_text())
    if execution['job_id']!=JOB:raise ValueError('Unexpected allocation ID')
    execution.update(state='completed' if row[1]=='COMPLETED' and row[2]=='0:0' and run['status']=='completed_development_case' else 'failed_or_inconclusive',
                     terminal_utc=datetime.now(timezone.utc).isoformat(),slurm_state=row[1],slurm_exit_code=row[2],
                     actual_elapsed_seconds=int(row[3]),actual_allocated_cpu_seconds=int(row[3])*int(row[4]),
                     actual_gpu_seconds=0,run_status=run['status'])
    save(ROOT/'execution.json',execution)
    artifact=[]
    for p in sorted(ROOT.rglob('*.npz')):
        artifact.append({'path':str(p),'relative_path':str(p.relative_to(ROOT)),
                         'bytes':p.stat().st_size,'sha256':sha(p),'payload_in_git':False})
    save(ROOT/'remote-artifacts.json',{'artifacts':artifact,'scope':'Full draw payloads stay on Oscar/private temporary storage.'})
    print(json.dumps({'state':execution['state'],'elapsed_seconds':int(row[3]),'allocated_cpu_seconds':execution['actual_allocated_cpu_seconds'],
                      'run_status':run['status'],'payloads':len(artifact),'frozen_input_hashes_verified':len(plan['frozen_sha256'])}))

if __name__=='__main__':main()
