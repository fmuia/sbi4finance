import hashlib,json,os,shutil,signal,socket,subprocess,time
from datetime import datetime,timezone
from pathlib import Path

if socket.gethostname()!='login010':
 print(json.dumps({'state':'wrong_login_host_no_action','host':socket.gethostname()}));raise SystemExit(3)
base=Path('/oscar/scratch/afengler/sbi4finance-overnight-20261028/followons')
root=base/'accounting-recovery-20261028';root.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def create(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def stamp():return datetime.now(timezone.utc).isoformat()
original=json.loads((base/'waiter/execution.json').read_text())
assert original['pid']==3792953 and original['state']=='armed'
assert all(not x.get('attempt') for x in original['routes'].values())
plan=json.loads((base/'waiter/plan.json').read_text())
for route in plan['routes'].values():
 c=json.loads(Path(route['campaign_config']).read_text())
 assert not Path(c['campaign_root']).exists(), 'A campaign artifact exists; reconcile first'
cmd=Path('/proc/3792953/cmdline').read_bytes().replace(b'\0',b' ').decode()
assert str(base/'arm_bank_extensions.py') in cmd and '--run' in cmd
for src,name in [(base/'waiter/execution.json','old-waiter-execution.json'),(base/'waiter/launch.json','old-waiter-launch.json'),
                 (base/'waiter/plan.json','old-waiter-plan.json'),(base/'arm_bank_extensions.py','old-arm_bank_extensions.py')]:
 target=root/name
 with target.open('xb') as f:f.write(src.read_bytes())
prospective={'state':'reviewed_recovery_prepared','utc':stamp(),'old_pid':3792953,'host':socket.gethostname(),
 'old_command':cmd,'old_execution_sha256':sha(root/'old-waiter-execution.json'),
 'root_review':'Exact date-filter patch and16 tests passed; user100k expansion remains authorized.',
 'budget_change':False,'new_simulations_or_repricing':False,
 'patched_launcher_sha256':sha(root/'run_bank_campaign.py'),'waiter_sha256':sha(root/'arm_bank_extensions.py')}
create(root/'recovery-prospective.json',prospective)
os.kill(3792953,signal.SIGTERM)
for _ in range(40):
 p=Path('/proc/3792953/stat')
 if not p.exists() or p.read_text().split()[2]=='Z':break
 time.sleep(.1)
else:raise RuntimeError('Original waiter still active; no replacement launch')
create(root/'old-waiter-stop.json',{'state':'old_waiter_stopped_without_campaign_attempts','utc':stamp(),
 'pid':3792953,'host':socket.gethostname(),'original_record_files_unchanged':True})
plan['root']=str(root/'waiter');plan['launcher']=str(root/'run_bank_campaign.py')
plan['recovery_of']={'old_pid':3792953,'old_execution_sha256':sha(root/'old-waiter-execution.json'),
 'cause':'UTC date was interpreted by sacct in EDT and excluded prior-local-evening jobs.'}
create(root/'plan.json',plan)
argv=[plan['python'],str(root/'arm_bank_extensions.py'),'--plan',str(root/'plan.json'),'--detach']
attempt={'state':'launch_attempted','utc':stamp(),'argv':argv,'automatic_retry':False,
 'plan_sha256':sha(root/'plan.json'),'launcher_sha256':sha(root/'run_bank_campaign.py'),'waiter_sha256':sha(root/'arm_bank_extensions.py')}
create(root/'recovery-launch-prospective.json',attempt)
try:
 r=subprocess.run(argv,capture_output=True,text=True,timeout=60)
 attempt.update(stdout=r.stdout,stderr=r.stderr,returncode=r.returncode)
 if r.returncode:raise RuntimeError('Replacement launch failed; no automatic retry')
 receipt=json.loads((root/'waiter/launch.json').read_text());os.kill(receipt['pid'],0)
 attempt.update(state='replacement_armed_verified_alive',pid=receipt['pid'],host=socket.gethostname(),
                launch_receipt_sha256=sha(root/'waiter/launch.json'),verified_utc=stamp())
except BaseException as error:
 attempt.update(state='unknown_or_failed_recovery_launch',error_type=type(error).__name__,error=str(error));create(root/'recovery-launch.json',attempt);raise
create(root/'recovery-launch.json',attempt)
print(json.dumps(attempt))
