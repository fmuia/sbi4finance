import hashlib,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path

root=Path('/oscar/scratch/afengler/sbi4finance-overnight-20261028/followons')
plan_path=root/'configs/rough_followon_100k_20261028.json'
plan=json.loads(plan_path.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def create(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
repo=Path('/oscar/scratch/afengler/sbi4finance-overnight-20261028/repo')
revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
dirty=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True)
assert revision=='00faaefe946c8e8fef3b55ff3156c2d0db61ffb3' and not dirty
assert plan['launcher']==str(repo/'scripts/run_bank_campaign.py')
for name,route in plan['routes'].items():
 p=Path(route['collection'])/'collection.json';assert sha(p)==route['collection_sha256']
 record=json.loads(p.read_text());c=json.loads(Path(route['campaign_config']).read_text())
 assert record['collection_id']==route['collection_id'] and record['configuration']['root_seed']==route['root_seed']
 assert c['collection']==route['collection'] and c['train']==100000 and c['prior_cpu_seconds']==0
 assert c['cpu_hours']==(200 if name=='known' else 3000) and c['max_workers']==(64 if name=='known' else 200)
 assert c['seed_policy']=='native_collection' and c['expected_git_revision']==route['producer_revision']
 task=Path(c['task_config']);payload=json.dumps(record['configuration'],indent=2,allow_nan=False)+'\n'
 if task.exists():assert task.read_text()==payload
 else:task.write_text(payload)
assert plan['routes']['joint']['after_route']=='known'
frozen={str(p):sha(p) for p in [Path(__file__),plan_path,Path(plan['launcher']),root/'arm_bank_extensions.py',
 *[Path(r['campaign_config']) for r in plan['routes'].values()],
 *[Path(json.loads(Path(r['campaign_config']).read_text())['task_config']) for r in plan['routes'].values()]]}
argv=[plan['python'],str(root/'arm_bank_extensions.py'),'--plan',str(plan_path),'--detach']
record={'format':'rough_100k_followon_deployment_v1','state':'launch_attempted',
 'utc':datetime.now(timezone.utc).isoformat(),'authorization':'User explicitly requests100k training surfaces for EACH rough route and overnight continuation; root approves new200/3000CPU-hour tranches.',
 'frozen_launcher_revision':revision,'frozen_sha256':frozen,'argv':argv,'automatic_retry':False,
 'generation_started_by_deployer':False,'resource_gate':'joint100k waits for its own checked20k snapshot AND known100k completed/settled; known100k may overlap originaljoint120.'}
create(root/'deployment-prospective.json',record)
try:
 r=subprocess.run(argv,capture_output=True,text=True,timeout=60)
 record.update(stdout=r.stdout,stderr=r.stderr,returncode=r.returncode)
 if r.returncode:raise RuntimeError('Waiter launch failed; do not retry automatically')
 receipt=json.loads((root/'waiter/launch.json').read_text())
 os.kill(receipt['pid'],0)
 for _ in range(20):
  p=root/'waiter/execution.json'
  if p.exists():break
  time.sleep(.25)
 state=json.loads(p.read_text())
 record.update(state='armed_and_verified_alive',waiter_pid=receipt['pid'],launch_receipt_sha256=sha(root/'waiter/launch.json'),
               execution_sha256=sha(p),routes=state['routes'],alive_verified_utc=datetime.now(timezone.utc).isoformat())
except BaseException as e:
 record.update(state='unknown_or_failed_launch',error_type=type(e).__name__,error=str(e));create(root/'deployment-launch.json',record);raise
create(root/'deployment-launch.json',record)
print(json.dumps({k:record[k] for k in ['state','waiter_pid','routes','frozen_launcher_revision']}))
