"""Read saved pilot blocks and scheduler telemetry; never invoke a pricer."""
from pathlib import Path
import json,subprocess,time,hashlib
from datetime import datetime,timezone
import numpy as np
r=Path('/oscar/scratch/afengler/sbi4finance-rough-actual-authorized-20261111');work=json.loads((r/'preparation/work-order.json').read_text())
rows=[]
for p in sorted((r/'payload').glob('component-*/run.json')):
 try:a=json.loads(p.read_text())
 except json.JSONDecodeError:continue
 comp=a['component'];task=work['tasks'][a['curve_role']];strict=.5*np.asarray(task['design']['half_spread_forward']);bs=[]
 for b in a.get('block_payloads',[]):
  f=Path(b['path'])
  if hashlib.sha256(f.read_bytes()).hexdigest()!=b['sha256']:raise ValueError('completed block hash differs')
  chosen=b['contracts']
  with np.load(f,allow_pickle=False) as v:
   cov=v['bank_covariance'];cov=cov[None] if cov.ndim==2 else cov
   mc=[float(np.max(np.sqrt(np.diag(c))/strict[chosen])) for c in cov]
   q={'block':b['block'],'horizon':task['blocks'][b['block']]['horizon'],'mcse_over_strict_sd':mc}
   if 'difference' in v:
    q['abs_gap_over_strict_sd']=float(np.max(np.abs(v['difference'])/strict[chosen]));q['gap_se_over_strict_sd']=float(np.max(np.sqrt(np.diag(v['difference_covariance']))/strict[chosen]))
   bs.append(q)
 rows.append({'index':comp['index'],'case_index':comp['case_index'],'role':comp['role'],'status':a['status'],'blocks_completed':a.get('blocks_completed'),'error':a.get('error'),'blocks':bs,'max_mcse_over_strict_sd':a.get('max_mcse_over_strict_sd'),'total_wall_seconds':a.get('total_wall_seconds')})
def cmd(a):return subprocess.run(a,text=True,capture_output=True,check=True).stdout
out={'asof_utc':datetime.now(timezone.utc).isoformat(),'job_id':'8820876','components':rows,'scope':'partial block provisional evidence; no full numerical acceptance','squeue':cmd(['squeue','-h','-j','8820876','-r','-o','%i|%T|%R|%C|%m']),'sstat':cmd(['sstat','-a','-j','8820876','--format=JobID,MaxRSS,AveRSS','-n','-P']),'sacct':cmd(['sacct','-j','8820876','--format=JobID,JobIDRaw,State,ElapsedRaw,TotalCPU,MaxRSS,ReqMem,AllocCPUS,ExitCode','-n','-P'])}
(r/'monitoring').mkdir(exist_ok=True);path=r/'monitoring'/('snapshot-'+str(time.time_ns())+'.json');path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out));print(str(path))
