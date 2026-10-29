"""One reviewed detached launch of the unchanged native campaign operator."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

BASE=Path('/oscar/scratch/afengler/sbi4finance-overnight-20261028/followons')
SOURCE=BASE/'joint-100k-recovery-source-20261028'
OPERATOR=BASE/'joint-tail-gate-20261028/run_bank_campaign.py'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def create(p,x):
    with Path(p).open('x') as stream:stream.write(json.dumps(x,indent=2,allow_nan=False)+'\n')
def now():return datetime.now(timezone.utc).isoformat()


assert socket.gethostname()=='login010','Use the declared login host; no launch attempted'
config_path=SOURCE/'recovery-config.json';config=read(config_path);preflight=read(SOURCE/'native-preflight.json')
assert sha(config_path)=='a27a7a142f160e37ce903788308744dde61f2889fb9091aa8a23b583043234a7'
assert sha(OPERATOR)==config['expected_operator_sha256']=='a59339f9f9d10f4de961a42977e41278324c76e290dc32b806094dc0542d5ee0'
assert config['prior_cpu_seconds']==3501292 and config['cpu_hours']==3000 and config['chunk']==5
assert config['max_workers']==200 and config['worker_wall_seconds']==10800 and config['poll_seconds']==30
assert config['exclude_nodes']==['node2338','node2339','node2336','node2344','node2346','node2334']
assert config['train']==100000 and config['memory_gib']==8
assert preflight['new_jobs']==7285 and preflight['seed_audit']['duplicate_pricing_seeds']==0
assert preflight['holdout_parts_unchanged_against_checked20k']==480
assert all(sha(p)==h for p,h in preflight['unchanged_inputs'].items())
assert not Path('/proc/4070959').exists(),'Original controller PID is still present; reconcile before launch'
for p in [BASE/'joint-100k/campaign.lock',Path(config['collection'])/'.locks/mutation.lock']:
    with p.open('r') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(stream,fcntl.LOCK_UN)
root=Path(config['campaign_root'])
assert not root.exists(),'Recovery campaign already has artifacts; never automatically retry'
assert 3501292+200*(10800+240)+3600+240==5713132<3000*3600
command=[config['python'],str(OPERATOR),'--config',str(config_path),'--detach']
record={'state':'launch_attempted','utc':now(),'host':socket.gethostname(),'argv':command,
    'config_sha256':sha(config_path),'operator_sha256':sha(OPERATOR),'deployer_sha256':sha(__file__),
    'preflight_sha256':sha(SOURCE/'native-preflight.json'),'preserved_inputs_sha256':preflight['unchanged_inputs'],
    'original_failed_job':'8056659_1','prior_cpu_seconds':3501292,'cpu_hour_ceiling':3000,
    'first200_full_exposure_cpu_seconds':5713132,'automatic_retry':False,'old_source_or_journal_changed':False}
create(SOURCE/'launch-attempt.json',record)  # Before a process which can submit work.
try:
    result=subprocess.run(command,capture_output=True,text=True,timeout=60)
    record.update(stdout=result.stdout,stderr=result.stderr,returncode=result.returncode)
    if result.returncode:raise RuntimeError('Unconfirmed detached launch; reconcile before further action')
    receipt=read(root/'launch.json')
    assert receipt['state']=='launched_not_yet_checked' and receipt['operator_sha256']==sha(OPERATOR)
    os.kill(receipt['pid'],0)
    assert all(sha(p)==h for p,h in preflight['unchanged_inputs'].items())
    record.update(state='detached_campaign_launched',pid=receipt['pid'],launch_receipt_sha256=sha(root/'launch.json'),
        launch_receipt=str(root/'launch.json'),verified_utc=now())
except BaseException as error:
    record.update(state='unknown_or_failed_launch',error_type=type(error).__name__,error=str(error))
    create(SOURCE/'deployment-receipt.json',record);raise
create(SOURCE/'deployment-receipt.json',record)
print(json.dumps({k:record[k] for k in ['state','pid','host','launch_receipt','config_sha256','operator_sha256']}))
