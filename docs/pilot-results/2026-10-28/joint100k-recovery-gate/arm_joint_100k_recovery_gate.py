import hashlib,json,os,signal,socket,subprocess,time
from datetime import datetime,timezone
from pathlib import Path

if socket.gethostname()!='login010':
 print(json.dumps({'state':'wrong_login_host_no_action','host':socket.gethostname()}));raise SystemExit(3)
base=Path('/oscar/scratch/afengler/sbi4finance-overnight-20261028/followons')
root=base/'joint-recovery-gate-20261028'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def create(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def now():return datetime.now(timezone.utc).isoformat()
plan_path=root/'joint_100k_recovery_gate_20261028.json';plan=json.loads(plan_path.read_text())
route=plan['routes']['joint'];cfg=json.loads(Path(route['campaign_config']).read_text())
assert set(plan['routes'])=={'joint'} and 'after_route' not in route
assert cfg['cpu_hours']==3000 and cfg['max_workers']==200 and cfg['train']==100000 and cfg['prior_cpu_seconds']==0
assert cfg['exclude_nodes']==['node2338','node2339','node2336','node2344']
assert not Path(cfg['campaign_root']).exists(),'Joint100k already has artifacts; reconcile before any launch'
recovery=json.loads(Path(route['recovery_plan']).read_text());binding=route['reviewed_recovery']
assert recovery['original_execution_sha256']==binding['original_execution_sha256']
assert recovery['preserved_inventory_sha256']==binding['preserved_inventory_sha256']
assert recovery['original_job_map_sha256']==binding['original_job_map_sha256']
assert recovery['phases']['joint']['worker_indices']==binding['new_worker_indices']
assert len(binding['new_worker_indices'])==309 and binding['preserved_parts']==971
oldroot=base/'accounting-recovery-20261028/waiter';old=json.loads((oldroot/'execution.json').read_text())
assert old['pid']==2533826 and old['routes']['known']['campaign_pid']==2533833
assert old['routes']['known']['state']=='launched' and not old['routes']['joint'].get('attempt')
cmd=Path('/proc/2533826/cmdline').read_bytes().replace(b'\0',b' ').decode()
assert str(base/'accounting-recovery-20261028/arm_bank_extensions.py') in cmd
known_config=Path(route['after_campaign']['config']);known=json.loads(known_config.read_text())
known_root=Path(known['campaign_root']);known_receipt=json.loads((known_root/'launch.json').read_text())
assert known_receipt['pid']==2533833
untouched={str(p):sha(p) for p in [known_config,base/'accounting-recovery-20261028/run_bank_campaign.py',known_root/'campaign.json']}
for p,name in [(oldroot/'execution.json','replaced-metadata-waiter-execution.json'),(oldroot/'launch.json','replaced-metadata-waiter-launch.json'),(oldroot/'plan.json','replaced-metadata-waiter-plan.json')]:
 with (root/name).open('xb') as f:f.write(p.read_bytes())
record={'state':'joint_only_gate_launch_prepared','utc':now(),'old_metadata_waiter_pid':2533826,'known_campaign_pid_untouched':2533833,
 'known_immutable_files_sha256':untouched,'plan_sha256':sha(plan_path),'launcher_sha256':sha(plan['launcher']),
 'waiter_sha256':sha(root/'arm_bank_extensions.py'),'recovery_plan_sha256':sha(route['recovery_plan']),
 'budget_change':False,'historical_failure_rewrite':False,'new_numerical_jobs_by_deployer':0,'automatic_retry':False}
create(root/'deployment-prospective.json',record)
os.kill(2533826,signal.SIGTERM)
for _ in range(40):
 p=Path('/proc/2533826/stat')
 if not p.exists() or p.read_text().split()[2]=='Z':break
 time.sleep(.1)
else:raise RuntimeError('Metadata waiter still active; replacement not launched')
assert all(sha(p)==h for p,h in untouched.items())
create(root/'metadata-waiter-stop.json',{'state':'stopped_metadata_only','utc':now(),'pid':2533826,'known_campaign_pid_untouched':2533833})
argv=[plan['python'],str(root/'arm_bank_extensions.py'),'--plan',str(plan_path),'--detach']
record['argv']=argv;record['state']='launch_attempted';create(root/'launch-attempt.json',record)
try:
 r=subprocess.run(argv,capture_output=True,text=True,timeout=60);record.update(stdout=r.stdout,stderr=r.stderr,returncode=r.returncode)
 if r.returncode:raise RuntimeError('Joint-only waiter launch failed; no retry')
 receipt=json.loads((root/'waiter/launch.json').read_text());os.kill(receipt['pid'],0)
 for _ in range(30):
  p=root/'waiter/execution.json'
  if p.exists():break
  time.sleep(.25)
 state=json.loads(p.read_text())
 assert state['state']=='armed' and set(state['routes'])=={'joint'}
 assert all(sha(p)==h for p,h in untouched.items())
 record.update(state='joint_only_waiter_armed_verified_alive',pid=receipt['pid'],host=socket.gethostname(),
               routes=state['routes'],launch_receipt_sha256=sha(root/'waiter/launch.json'),verified_utc=now())
except BaseException as error:
 record.update(state='unknown_or_failed_launch',error_type=type(error).__name__,error=str(error));create(root/'deployment-launch.json',record);raise
create(root/'deployment-launch.json',record)
print(json.dumps({k:record[k] for k in ['state','pid','host','routes','known_campaign_pid_untouched','launcher_sha256']}))
