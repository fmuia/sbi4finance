"""Bounded native-bank handoff to one existing joint100k training allocation."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def save(p, x):
    p=Path(p); temporary=p.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');temporary.replace(p)
def create(p, x):
    with Path(p).open('x') as f:f.write(json.dumps(x,indent=2,allow_nan=False)+'\n')


def sources(p):
    if sha(__file__)!=p['helper_sha256'] or sha(p['operator'])!=p['operator_sha256']:
        raise ValueError('Frozen helper/operator changed')
    original_path=Path(p['original_phase'])/'plan.json'
    if sha(original_path)!=p['original_plan_sha256']:raise ValueError('Original phase declaration changed')
    original=read(original_path)
    old=Path(original['repo'])/'cluster/submit_full_surface_backbones_20261028.py'
    if sha(old)!='1ada0012f6deab62269d16bb8479b4a63e28d326bd215575096ad24dab5150dc':
        raise ValueError('Original c96 operator changed')
    if Path(p['operator']).read_bytes()!=old.read_bytes().replace(
            b'"--patience", "30"',b'"--patience", str(plan["recipe"]["pricer_patience"])',1):
        raise ValueError('Versioned operator differs beyond the one declared argument')
    spec=importlib.util.spec_from_file_location('frozen_joint_operator',p['operator'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.source_check(original)
    return original,module


def accounting(ids):
    r=subprocess.run(['sacct','-S','2026-10-27','-j',','.join(ids),'-n','-P',
        '--format=JobID%64,State%40,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES%120'],
        capture_output=True,text=True,timeout=60,check=True)
    return {x[0]:x for line in r.stdout.splitlines() if len(x:=line.split('|'))>=6 and '.' not in x[0]}


def successful(row, identifier, cpus):
    if row is None or row[1].split()[0] in ['RUNNING','PENDING','COMPLETING','CONFIGURING']:
        raise TimeoutError('Waiting for accounting: '+identifier)
    if row[1].split()[0]!='COMPLETED' or row[2]!='0:0' or int(row[4])!=cpus:
        raise ValueError('Allocation not successfully settled: '+identifier)
    return int(row[3])*cpus


def pool(p):
    rows=accounting(list(p['pool_jobs']));cpu=gpu=0
    for identifier,elapsed in p['pool_jobs'].items():
        row=rows.get(identifier);cpu+=successful(row,identifier,4)
        if int(row[3])!=elapsed or dict(v.split('=',1) for v in row[5].split(','))['gres/gpu']!='1':
            raise ValueError('Training pool cost/resources changed')
        gpu+=elapsed
    if (cpu,gpu)!=(15632,3908) or cpu+15360>32400 or gpu+3840>8100:
        raise ValueError('Original training pool cannot cover one full allocation')
    return {'prior_cpu_seconds':cpu,'prior_gpu_seconds':gpu,'full_cumulative_cpu_seconds':cpu+15360,
        'full_cumulative_gpu_seconds':gpu+3840,'cpu_seconds_ceiling':32400,'gpu_seconds_ceiling':8100}


def native_ready(p):
    run=read(p['bank_execution'])
    if run['state'].startswith('stopped_'):raise ValueError('Native bank stopped: '+run['state'])
    if run['state']!='completed_checked_snapshot':
        if run['state'] not in ['preparing','generation_active','final_publication_submitted']:
            raise ValueError('Unrecognized native bank state: '+run['state'])
        return None
    bank=Path(p['bank']);manifest=read(bank/'manifest.json');collection=read(Path(p['collection'])/'collection.json')
    if (sha(p['bank_config'])!=p['bank_config_sha256'] or sha(Path(p['collection'])/'collection.json')!=p['collection_sha256']
            or run['snapshot']!=str(bank) or sha(bank/'manifest.json')!=run['snapshot_manifest_sha256']
            or run['collection_id']!=p['collection_id'] or manifest['collection_id']!=p['collection_id']
            or manifest['configuration']['splits']!=p['splits'] or manifest['configuration']['root_seed']!=271200
            or manifest['configuration']['task']!=collection['configuration']['task']
            or manifest['task_fingerprint']!=p['task_fingerprint']
            or run['producer_git_revision']!=p['producer_revision']
            or run['source_sha256']!=manifest['source_sha256'] or manifest['source_sha256']!=collection['source_sha256']
            or run['runtime']!=manifest['versions'] or manifest['versions']!=collection['versions']
            or run['config']['prior_cpu_seconds']!=3501292 or run['config']['cpu_hours']!=3000
            or run['budget']['outstanding_cpu_seconds']!=0 or run['budget']['nonterminal_workers']!=0
            or run['planned_workers']!=7285 or run['completed_workers']!=7285
            or sha(p['previous_bank_execution'])!=p['previous_bank_execution_sha256']):
        raise ValueError('Native snapshot/source/task/runtime/budget identity differs')
    if run['config']!=read(p['bank_config']):raise ValueError('Native campaign configuration changed')
    attempts=run['attempts']
    if any(a.get('state')!='submitted' or not a.get('job_id') for a in attempts):raise ValueError('Ambiguous native attempt')
    rows=accounting([a['job_id'] for a in attempts]);ids=[];actual=0
    for a in attempts:
        targets=([a['job_id']+'_'+str(i-a['array_offset']) for i in a['indices']] if a['role']=='workers' else [a['job_id']])
        for identifier in targets:ids.append(identifier);actual+=successful(rows.get(identifier),identifier,1)
    if len(ids)!=len(set(ids)) or actual+3501292!=run['budget']['actual_cpu_seconds']:
        raise ValueError('Native failed-inclusive allocation accounting differs')
    return {'status':'completed_checked_snapshot','snapshot':str(bank),'snapshot_manifest_sha256':sha(bank/'manifest.json'),
        'split_counts':p['splits'],'collection_id':p['collection_id'],'source_revision':p['producer_revision'],
        'source_sha256':manifest['source_sha256'],'task_fingerprint':manifest['task_fingerprint'],
        'native_execution':p['bank_execution'],'native_execution_sha256':sha(p['bank_execution']),
        'bank_allocated_cpu_seconds':actual+3501292,'scientific_acceptance':False,'reserved_test_evaluated':False}


def handoff(p, ready, state, state_path):
    original,module=sources(p);cost=pool(p);root=Path(p['training_root']);root.mkdir(parents=True,exist_ok=False)
    (root/'logs').mkdir();create(root/'native-readiness.json',ready)
    route={**original['routes']['joint'],'bank':p['bank'],'ready':str(root/'native-readiness.json'),'splits':p['splits']}
    phase={**original,'prepared_utc':now(),'root':str(root),'routes':{'joint':route},
        'recipe':{**original['recipe'],'pricer_patience':300},'continuation':cost,
        'source_sha256':{**original['source_sha256'],p['operator']:p['operator_sha256'],str(Path(__file__).resolve()):p['helper_sha256']},
        'operator_note':'One joint100k fit; same c96 numerical workflow, explicit pricer patience300.'}
    create(root/'plan.json',phase)
    ledger={'format':'full_surface_post_bank_execution_v1','state':'waiting_for_banks','plan_sha256':sha(root/'plan.json'),
        'routes':{'joint':{'state':'waiting'}},'scientific_acceptance':False,'reserved_test_evaluated':False}
    create(root/'execution.json',ledger)
    state.update(state='handoff_attempted',attempt_utc=now(),phase_plan_sha256=sha(root/'plan.json'));save(state_path,state)
    create(root/'launch-attempt.json',{'utc':now(),'plan_sha256':sha(root/'plan.json'),'operator_sha256':p['operator_sha256'],
        'automatic_retry':False,'training_pool':cost})
    module.submit(phase,ledger,'joint')
    if ledger['routes']['joint']['state']!='submitted':raise RuntimeError('Unconfirmed submission; reconcile, never retry')
    state.update(job_id=ledger['routes']['joint']['job_id']);save(state_path,state)
    create(root/'watcher-launch-attempt.json',{'utc':now(),'job_id':state['job_id'],'automatic_retry':False})
    with (root/'watcher.log').open('ab') as log:
        child=subprocess.Popen([p['python'],'-u',p['operator'],'watch','--root',str(root)],cwd=phase['repo'],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    os.kill(child.pid,0);receipt={'state':'submitted_with_existing_watcher','job_id':state['job_id'],'watcher_pid':child.pid,
        'root':str(root),'utc':now(),**cost};create(root/'launch.json',receipt)
    state.update(state='handed_off',watcher_pid=child.pid,completed_utc=now());save(state_path,state)


def wait(p, plan_path):
    root=Path(p['handoff_root']);path=root/'execution.json'
    state={'state':'waiting_for_native_bank','pid':os.getpid(),'started_utc':now(),'plan_sha256':sha(plan_path),
        'scientific_acceptance':False,'automatic_retry':False};create(path,state)
    started=time.monotonic()
    try:
        sources(p)
        while time.monotonic()-started<7200:
            if sha(plan_path)!=state['plan_sha256'] or sha(__file__)!=p['helper_sha256'] or sha(p['operator'])!=p['operator_sha256']:
                raise ValueError('Frozen handoff declaration/source changed')
            try:ready=native_ready(p)
            except (subprocess.SubprocessError,OSError,TimeoutError) as error:
                ready=None;state['progress']='waiting_for_readonly_accounting:'+type(error).__name__
            if ready is not None:handoff(p,ready,state,path);return
            state['last_poll_utc']=now();state.setdefault('progress','native_generation_or_final_check_running');save(path,state);time.sleep(30)
        state.update(state='expired_without_submission',completed_utc=now());save(path,state)
    except BaseException as error:
        state.update(state='stopped_for_review',error_type=type(error).__name__,error=str(error),completed_utc=now());save(path,state);raise


def detach(p, plan_path):
    sources(p);pool(p)
    if Path(p['training_root']).exists():raise FileExistsError('Joint100k training already has artifacts')
    root=Path(p['handoff_root']);root.mkdir(parents=True,exist_ok=False);(root/'plan.json').write_bytes(Path(plan_path).read_bytes())
    create(root/'launch-attempt.json',{'utc':now(),'helper_sha256':sha(__file__),'plan_sha256':sha(root/'plan.json'),'automatic_retry':False})
    with (root/'handoff.log').open('ab') as log:
        child=subprocess.Popen([p['python'],str(Path(__file__).resolve()),'--plan',str(root/'plan.json'),'--run'],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt={'state':'waiting_process_launched','pid':child.pid,'utc':now(),'plan_sha256':sha(root/'plan.json'),
        'helper_sha256':sha(__file__),'operator_sha256':p['operator_sha256'],'wall_seconds':7200,'outside_slurm':True}
    create(root/'launch.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--plan',type=Path,required=True);g=a.add_mutually_exclusive_group(required=True)
    g.add_argument('--detach',action='store_true');g.add_argument('--run',action='store_true');args=a.parse_args();p=read(args.plan)
    detach(p,args.plan) if args.detach else wait(p,args.plan)
