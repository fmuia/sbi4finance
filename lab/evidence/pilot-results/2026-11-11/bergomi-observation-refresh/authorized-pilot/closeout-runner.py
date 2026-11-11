"""Read cancelled pilot outputs; record intentional user deadline stop honestly."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,time,hashlib
r=Path('/oscar/scratch/afengler/sbi4finance-rough-actual-authorized-20261111')
def cmd(args):return subprocess.run(args,capture_output=True,text=True,check=True).stdout
rows=[]
for p in sorted((r/'payload').glob('component-*/run.json')):
 try:a=json.loads(p.read_text())
 except json.JSONDecodeError:continue
 rows.append({'index':a['component']['index'],'case_index':a['component']['case_index'],'role':a['component']['role'],'metadata_status':a['status'],'blocks_completed':a.get('blocks_completed'),'total_wall_seconds':a.get('total_wall_seconds'),'pricing_cpu_seconds':a.get('pricing_cpu_seconds'),'max_mcse_over_strict_sd':a.get('max_mcse_over_strict_sd'),'strict_mc_target_met':a.get('strict_mc_target_met'),'block_payloads':a.get('block_payloads',[]),'payload':a.get('payload'),'record_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
queue=cmd(['squeue','-h','-j','8820876','-r','-o','%i|%T|%R|%C|%m'])
acct=cmd(['sacct','-j','8820876','--format=JobID,JobIDRaw,State,Start,End,ElapsedRaw,TotalCPU,MaxRSS,ReqMem,AllocCPUS,ExitCode','-n','-P'])
a={'format':'rough_actual_layout_intentional_pilot_stop_v1','asof_utc':datetime.now(timezone.utc).isoformat(),'job_id':'8820876','reason':'User explicitly requested a result within a few minutes and scale-down of a longer pilot; remaining array tasks cancelled once representative outputs were available.','parent_confirmed_user_instruction':True,'cancellation_command':['scancel','8820876'],'automatic_retry':False,'new_tasks_submitted_after_stop':False,'components_created':len(rows),'components_complete':sum(v['metadata_status']=='complete' for v in rows),'components_with_partial_blocks':sum(v['metadata_status']!='complete' and (v['blocks_completed'] or 0)>0 for v in rows),'components_not_started':48-len(rows),'records':rows,'queue_after_cancel':queue,'sacct':acct,'scope':'Intentional administrative stop, not a numerical acceptance or execution-failure claim. Running metadata remains preserved as written; Slurm cancellation is authoritative.','numerical_target_established':False,'scientific_acceptance':False}
path=r/'intentional-stop.json'
with path.open('x') as f:json.dump(a,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(a));print('record',str(path))
