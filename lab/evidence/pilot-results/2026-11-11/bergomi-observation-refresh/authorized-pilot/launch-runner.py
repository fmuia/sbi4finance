"""Create a fresh frozen execution root, verify permissions/source, submit once."""
from pathlib import Path
import getpass,hashlib,json,os,shutil,socket,subprocess,tarfile,time
from datetime import datetime,timezone
STAGE=Path('/oscar/scratch/afengler/sbi4finance-rough-actual-authorized-20261111')
BASE=Path('/oscar/scratch/afengler/sbi4finance-rough-actual-layout-portable-20261110')
OVERLAY=Path('/oscar/scratch/afengler/rough-authorized-a2052392d428.tar')
OVERLAY_SHA='a2052392d428929ce3b5329bd2ca18320289bffc0c45b6d4b2639e472310acc5'
PYTHON=Path('/oscar/scratch/afengler/sbi4finance-full-core-20261026-wy2_j3o4/repo-conditioned-reference/.pixi/envs/core-reference/bin/python')
WORK_SHA='38455e9565681ea502cac33116f92c27f2d3396eb3ff06c0248b251e9f9edb45'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(p),'sha256':digest(p)}
def write(p,v):
 with p.open('x') as s:json.dump(v,s,indent=2,allow_nan=False);s.write('\n')
def cmd(args):
 p=subprocess.run(args,text=True,capture_output=True,check=True);return p.stdout
if STAGE.exists():raise FileExistsError('new immutable execution root already exists; do not resubmit')
if digest(OVERLAY)!=OVERLAY_SHA:raise ValueError('overlay transfer hash differs')
STAGE.mkdir(mode=0o700);repo=STAGE/'repo';prep=STAGE/'preparation';logs=STAGE/'logs';payload=STAGE/'payload';proof=STAGE/'preflight'
for p in [logs,payload,proof]:p.mkdir()
shutil.copytree(BASE/'repo',repo)
with tarfile.open(OVERLAY) as t:
 if any(m.name.startswith('/') or '..' in Path(m.name).parts or not m.isfile() for m in t.getmembers()):raise ValueError('unsafe overlay member')
 t.extractall(STAGE,filter='data')
shutil.copyfile(BASE/'preparation/work-order.json',prep/'work-order.json')
work=prep/'work-order.json';auth=prep/'authorization.json';script=prep/'pilot.sbatch'
if digest(work)!=WORK_SHA:raise ValueError('unchanged work order transfer differs')
a=json.loads(auth.read_text())
if digest(script)!=a['sbatch_sha256']:raise ValueError('scheduler declaration differs')
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(repo/'lab/src')+':'+str(repo/'tutorial/src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLCONFIGDIR=str(proof/'matplotlib-cache'))
check=[str(PYTHON),str(repo/'lab/scripts/run_rough_actual_layout_authorized.py'),'check','--work-order',str(work),'--authorization',str(auth)]
p=subprocess.run(check,env=env,cwd='/tmp',text=True,capture_output=True)
(proof/'source-check.stdout.txt').write_text(p.stdout);(proof/'source-check.stderr.txt').write_text(p.stderr)
resources={
 'associations':cmd(['sacctmgr','-n','-P','show','assoc','where','user=afengler','format=Account,Partition,QOS,GrpTRES,MaxTRES']),
 'condo_qos':cmd(['sacctmgr','-n','-P','show','qos','where','name=carney-condo2','format=Name,MaxTRESPU,MaxTRESPA,GrpTRES,MaxJobsPU,MaxJobsPA,MaxSubmitPU,MaxSubmitPA,MaxWall']),
 'normal_qos':cmd(['sacctmgr','-n','-P','show','qos','where','name=normal','format=Name,MaxTRESPU,GrpTRES']),
 'user_queue_before_submission':cmd(['squeue','-h','-u',getpass.getuser(),'-o','%i|%P|%a|%q|%T|%C|%m']),
 'shared_account_queue':cmd(['squeue','-h','-A','carney-mjfrank-condo2','-o','%i|%q|%T|%C|%m|%D']),
 'batch_inventory':cmd(['sinfo','-h','-p','batch','-o','%P|%C']),
 'batch_partition':cmd(['scontrol','show','partition','batch'])}
config=cmd(['scontrol','show','config']);resources['array_scheduler_limits']='\n'.join(v for v in config.splitlines() if any(k in v for k in ['MaxArraySize','MaxJobCount','MaxStepCount','SchedulerParameters','AccountingStorageEnforce']))
resource_path=proof/'resources.json';write(resource_path,resources)
if p.returncode:raise RuntimeError('source/authorization preflight failed; see preserved stderr')
if not all(digest(repo/s['relative_path'])==s['sha256'] for s in a['source_bindings'].values()):raise ValueError('25 source bindings differ')
if 'carney-condo2|cpu=208,mem=2500G' not in resources['condo_qos']:raise ValueError('resource ceilings changed; inspect before submission')
probe=cmd([str(PYTHON),'-c','import sys,json;from importlib import metadata;print(json.dumps({"executable":sys.executable,"python":sys.version,"numpy":metadata.version("numpy"),"scipy":metadata.version("scipy")}))'])
record={'format':'rough_actual_layout_authorized_submission_v1','created_utc':datetime.now(timezone.utc).isoformat(),'host':socket.gethostname(),'stage_root':str(STAGE),'new_repo':str(repo),'base_verified_source_root':str(BASE),'base_source_revision':'3e979d94bc1966f62b917236a9e9963cc37029b7','overlay':bind(OVERLAY),'work_order':bind(work),'execution_authorization':bind(auth),'sbatch':bind(script),'resource_evidence':bind(resource_path),'source_bindings_verified':25,'preflight_command':check,'preflight_exit_code':p.returncode,'environment':json.loads(probe),'requested_parallelism':{'array_components':48,'concurrent_workers':16,'CPU_per_worker':1,'GiB_per_worker':32,'peak_CPU':16,'peak_memory_GiB':512,'allocated_CPU_seconds_ceiling':28800},'heston_diagnostics_passed':False,'user_direct_authorization_supersedes_execution_condition':True,'scientific_acceptance':False,'numerical_target_established':False,'status':'preflight_passed_not_yet_submitted'}
write(proof/'preflight.json',record)
export='ALL,ROUGH_REPO_ROOT='+str(repo)+',ROUGH_PYTHON='+str(PYTHON)+',ROUGH_WORK_ORDER='+str(work)+',ROUGH_AUTHORIZATION='+str(auth)+',ROUGH_PAYLOAD_ROOT='+str(payload)
submit=['sbatch','--parsable','--export='+export,'--output='+str(logs/'%A_%a.out'),'--error='+str(logs/'%A_%a.err'),'--chdir='+str(STAGE),str(script)]
write(STAGE/'submission-claim.json',{'command':submit,'authorization':bind(auth),'automatic_retry':False})
submitted=subprocess.run(submit,text=True,capture_output=True)
record.update(command=submit,sbatch_returncode=submitted.returncode,sbatch_stdout=submitted.stdout,sbatch_stderr=submitted.stderr)
if submitted.returncode:
 record['status']='submission_failed';write(STAGE/'submission.json',record);raise RuntimeError('sbatch failed; no retry')
job_id=submitted.stdout.strip().split(';')[0]
if not job_id.isdigit():raise ValueError('unexpected sbatch job identity')
record.update(status='submitted',job_id=job_id,payload_root=str(payload),logs_root=str(logs),automatic_retry=False)
write(STAGE/'submission.json',record)
time.sleep(2)
queue=cmd(['squeue','-h','-j',job_id,'-r','-o','%i|%T|%R|%C|%m'])
(STAGE/'queue-after-submission.txt').write_text(queue)
print(json.dumps({'job_id':job_id,'status':'submitted','record':bind(STAGE/'submission.json'),'queue':queue,'payload_root':str(payload)}))
