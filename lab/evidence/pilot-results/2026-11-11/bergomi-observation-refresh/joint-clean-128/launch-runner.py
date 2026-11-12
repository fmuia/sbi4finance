"""Hash-check the authorized first128 joint rows and submit exactly once."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,tarfile,time,socket
S=Path('/oscar/scratch/afengler/sbi4finance-rough-main-bank-20261111');R=S/'repo';P=S/'preparation';O=Path('/oscar/scratch/afengler/rough-main128-e246fb3c0234.tar')
PY=Path('/oscar/scratch/afengler/sbi4finance-full-core-20261026-wy2_j3o4/repo-conditioned-reference/.pixi/envs/core-reference/bin/python')
SHA='e246fb3c0234b8fa6973bb50a99a99cbb3cfcc9d144f17b820eef876c104a178';PLAN_SHA='08ed9816e3ecf42dbaa4af03a3e9dfec6a7e523fd9103fae61c036a3203e662e'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(p),'sha256':digest(p)}
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def cmd(args):return subprocess.run(args,text=True,capture_output=True,check=True).stdout
if not (S/'source-stage.json').is_file() or (S/'submission-claim.json').exists():raise ValueError('fresh prestaged source and unclaimed submission required')
if digest(O)!=SHA:raise ValueError('bank overlay hash differs')
with tarfile.open(O) as t:
 if any(m.name.startswith('/') or '..' in Path(m.name).parts or not m.isfile() for m in t.getmembers()):raise ValueError('unsafe overlay')
 t.extractall(S,filter='data')
plan_path=P/'plan.json';review=P/'pilot-readout.json'
if digest(plan_path)!=PLAN_SHA:raise ValueError('root-approved bank plan hash differs')
plan=json.loads(plan_path.read_text())
if len(plan['jobs'])!=128 or plan['limits']['concurrent_workers']!=128 or plan['limits']['memory_gib_per_worker']!=8 or plan['limits']['shard_wall_seconds']!=1800:raise ValueError('root-approved active bounds differ')
if digest(R/'lab/src/sbi4finance/calibration/rough_actual_layout_bank.py')!='6bb09a3d2a5c5bbd9cce66a90c16695cbb293ceabb134e6bdb1b5ae7d8f3208e':raise ValueError('bank producer changed')
if digest(R/'lab/scripts/rough_actual_layout_bank.py')!='5b8ae33fff4503e2a9dc6048f2c112b698b67a9d79520bceb76a42dd0361681c':raise ValueError('bank CLI changed')
admission={'format':'rough_actual_layout_bank_execution_authorization_v1','created_utc':datetime.now(timezone.utc).isoformat(),'bank_plan_sha256':digest(plan_path),'generation_authorized':True,'mode':'candidate_only','label_policy_authorized':True,'pilot_execution_reviewed':True,'pilot_numerical_review_passed':False,'scientific_acceptance':False,'numerical_policy':plan['numerical_policy'],'numerical_review':bind(review),'direct_user_request':'Start Bergomi simulations, parallelize on Oscar, prioritize jointly inferred c; root approved this finite first128-row tranche before larger300k target.','heston_diagnostics_passed':False,'supersedes_previous_Heston_success_execution_gate':True,'root_reviewed_exact_plan_sha256':PLAN_SHA,'scope':'only128joint training clean-price rows; future300k planned, not allocated; retained failures and MC diagnostics; no scientific precision pass inferred'}
admission_path=P/'admission.json';write(admission_path,admission)
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(R/'lab/src')+':'+str(R/'tutorial/src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLCONFIGDIR=str(S/'preflight/matplotlib'))
check=[str(PY),str(R/'lab/scripts/rough_actual_layout_bank.py'),'check','--plan',str(plan_path)]
p=subprocess.run(check,env=env,cwd='/tmp',text=True,capture_output=True)
(S/'preflight/source-check.stdout.txt').write_text(p.stdout);(S/'preflight/source-check.stderr.txt').write_text(p.stderr)
if p.returncode:raise ValueError('bank source/plan check failed; no submission')
probe='import sys;sys.path.insert(0,'+repr(str(R/'lab'))+');from sbi4finance.calibration import rough_actual_layout_bank as b;b.validate_admission('+repr(str(plan_path))+',b.pilot.read('+repr(str(plan_path))+'),'+repr(str(admission_path))+');print("admission validated")'
subprocess.run([str(PY),'-c',probe],env=env,cwd='/tmp',check=True,capture_output=True,text=True)
qos=cmd(['sacctmgr','-n','-P','show','qos','where','name=carney-condo2','format=Name,MaxTRESPU,MaxTRESPA,GrpTRES,MaxJobsPU,MaxJobsPA,MaxSubmitPU,MaxSubmitPA,MaxWall'])
userq=cmd(['squeue','-h','-u','afengler','-o','%i|%q|%T|%C|%m|%D'])
shared=cmd(['squeue','-h','-A','carney-mjfrank-condo2','-o','%i|%q|%T|%C|%m|%D'])
config=cmd(['scontrol','show','config']);limits='\n'.join(l for l in config.splitlines() if any(x in l for x in ['MaxArraySize','MaxJobCount','AccountingStorageEnforce']))
resources={'asof_utc':datetime.now(timezone.utc).isoformat(),'qos':qos,'user_queue_before_submission':userq,'shared_account_queue':shared,'batch':cmd(['sinfo','-h','-p','batch','-o','%P|%C']),'scheduler_limits':limits,'requested':{'CPU':128,'memory_GiB':1024,'arraytasks':128,'total_allocated_CPU_seconds':230400}}
write(S/'preflight/resources.json',resources)
if 'carney-condo2|cpu=208,mem=2500G' not in qos:raise ValueError('resource limits changed; inspect before launch')
if userq.strip():raise ValueError('user queue is no longer empty; inspect combined headroom before launch')
script='''#!/bin/bash
#SBATCH --job-name=rough-main-jc128
#SBATCH --account=carney-mjfrank-condo2
#SBATCH --partition=batch
#SBATCH --qos=carney-condo2
#SBATCH --array=0-127%128
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROUGH_REPO_ROOT/lab/src:$ROUGH_REPO_ROOT/tutorial/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rough-bank-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
exec "$ROUGH_PYTHON" "$ROUGH_REPO_ROOT/lab/scripts/rough_actual_layout_bank.py" worker --plan "$ROUGH_BANK_PLAN" --admission "$ROUGH_BANK_ADMISSION" --index "$SLURM_ARRAY_TASK_ID" --out "$ROUGH_PAYLOAD_ROOT/shard-$(printf '%06d' "$SLURM_ARRAY_TASK_ID")"
'''
script_path=P/'bank.sbatch'
with script_path.open('x') as f:f.write(script)
export='ALL,ROUGH_REPO_ROOT='+str(R)+',ROUGH_PYTHON='+str(PY)+',ROUGH_BANK_PLAN='+str(plan_path)+',ROUGH_BANK_ADMISSION='+str(admission_path)+',ROUGH_PAYLOAD_ROOT='+str(S/'payload')
submit=['sbatch','--parsable','--export='+export,'--output='+str(S/'logs/%A_%a.out'),'--error='+str(S/'logs/%A_%a.err'),'--chdir='+str(S),str(script_path)]
record={'format':'rough_actual_layout_joint_main128_submission_v1','created_utc':datetime.now(timezone.utc).isoformat(),'host':socket.gethostname(),'stage_root':str(S),'source_overlay':bind(O),'plan':bind(plan_path),'admission':bind(admission_path),'script':bind(script_path),'source_bindings_verified':len(plan['source_bindings']),'preflight_exit_code':p.returncode,'resource_evidence':bind(S/'preflight/resources.json'),'command':submit,'training_target_planned':300000,'active_training_rows':128,'inference_parameter_names':plan['inference_blueprint']['parameter_names'],'SBI_training_readiness':False,'scientific_acceptance':False,'automatic_retry':False}
write(S/'preflight/preflight.json',record);write(S/'submission-claim.json',{'plan':bind(plan_path),'admission':bind(admission_path),'command':submit})
result=subprocess.run(submit,text=True,capture_output=True)
record.update(sbatch_stdout=result.stdout,sbatch_stderr=result.stderr,sbatch_returncode=result.returncode)
if result.returncode:record['status']='submission_failed';write(S/'submission.json',record);raise ValueError('submission failed; no automatic retry')
job=result.stdout.strip().split(';')[0]
if not job.isdigit():raise ValueError('unexpected job ID')
record.update(status='submitted',job_id=job,payload_root=str(S/'payload'));write(S/'submission.json',record)
time.sleep(3)
q=cmd(['squeue','-h','-j',job,'-r','-o','%i|%T|%R|%C|%m']);(S/'queue-after-submission.txt').write_text(q)
counts={}
for line in q.splitlines():
 parts=line.split('|');counts[parts[1]]=counts.get(parts[1],0)+1
print(json.dumps({'job_id':job,'counts':counts,'queue':q,'record':bind(S/'submission.json'),'payload_root':str(S/'payload')}))
