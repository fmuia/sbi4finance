"""One-off bounded submissions; no model or pricing work runs locally."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

PATH=Path('/private/tmp/sbi4finance-native-bank-execution-20261027.json')
OLD='/oscar/scratch/afengler/sbi4finance-full-core-20261026-wy2_j3o4'
FOLLOWUP='/oscar/scratch/afengler/sbi4finance-followup-20261027-f4d3679'
REVIEWED='/oscar/scratch/afengler/sbi4finance-reviewed-unit-20261027-9b495cf'

def now():return datetime.now(timezone.utc).isoformat()
def save(x):
    p=PATH.with_suffix('.json.tmp');p.write_text(json.dumps(x,indent=2)+'\n');p.replace(PATH)
def remote(args,cwd=None):
    command=shlex.join(args)
    if cwd:command='cd '+shlex.quote(cwd)+' && '+command
    return subprocess.run(['ssh','-o','BatchMode=yes','oscar',command],capture_output=True,text=True)

def budget(ledger,new_count,new_seconds):
    jobs=ledger['bank']['jobs'];ids=[j['job_id'] for j in jobs if j.get('job_id')]
    if any(j['state'] in ('submitting','unknown_submission') for j in jobs):
        raise RuntimeError('Ambiguous submission must be reconciled before any further submission')
    rows={}
    if ids:
        result=remote(['sacct','-S','2026-10-27','-j',','.join(ids),'-P','-n',
                       '--format=JobID%64,State%40,ElapsedRaw,AllocCPUS,ExitCode'])
        if result.returncode:raise RuntimeError('Accounting unavailable; no expansion')
        for line in result.stdout.splitlines():
            values=line.split('|')
            if len(values)>=5 and '.' not in values[0]:rows[values[0]]=values[:5]
    spent=outstanding=0.;workers=0
    for job in jobs:
        if job['state']=='rejected':continue
        for index in job.get('indices',[None]):
            key=job['job_id']+(f'_{index}' if index is not None else '')
            row=rows.get(key);state=row[1] if row else 'UNKNOWN'
            if state=='COMPLETED':
                if row[4]!='0:0':raise RuntimeError('Nonzero completed task; no expansion')
                spent+=int(row[2])*int(row[3])
            elif any(state.startswith(x) for x in ('FAILED','TIMEOUT','CANCELLED','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE')):
                raise RuntimeError(f'{key} {state}; retain exact range and stop expansion for review')
            else:
                # Full requested cap plus a small cleanup allowance, never
                # elapsed-only charging for a still-running/queued task.
                outstanding+=job['cpu_seconds_per_task_cap']+240
                if job['role']=='worker':workers+=1
    reserved=ledger['bank']['final_integrity_cpu_hours_reserved']*3600
    exposure=spent+outstanding+reserved+new_count*(new_seconds+240)
    if exposure>ledger['bank']['allocated_cpu_hour_ceiling']*3600:
        raise RuntimeError('Spending plus outstanding full-cap exposure exceeds generation ceiling')
    if workers+new_count>ledger['bank']['maximum_nonterminal_workers']:
        raise RuntimeError('Global nonterminal worker cap exceeded')
    return {'checked_utc':now(),'completed_allocated_cpu_seconds':spent,'outstanding_reserved_cpu_seconds':outstanding,
            'final_integrity_reserved_cpu_seconds':reserved,'after_submission_exposure_seconds':exposure,
            'existing_nonterminal_workers':workers,'new_task_count':new_count}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('role',choices=['native','preflight','workers'])
    parser.add_argument('--indices',help='Explicit comma-separated original job-map indices')
    parser.add_argument('--corrective-retry',action='store_true')
    args=parser.parse_args();ledger=json.loads(PATH.read_text());root=ledger['root'];repo=root+'/repo'
    if args.role=='native':repo=ledger['native_heston'].get('repo',repo)
    common=['--nodes=1','--ntasks=1','--no-requeue','--chdir='+repo]
    env=['env','PYTHONPATH='+repo+'/src','KERAS_BACKEND=jax','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
         'MKL_NUM_THREADS=1','NUMEXPR_NUM_THREADS=1','XLA_PYTHON_CLIENT_PREALLOCATE=false']
    if args.role=='native':
        target=ledger['native_heston'];name='native-heston'
        wall_seconds=900;output=root+'/runs/heston-native-adaptive'
        if target['jobs']:
            if not args.corrective_retry or not target.get('reviewed_guard_correction'):
                raise RuntimeError('Native run already recorded; explicit reviewed correction required')
            spent=0
            for previous in target['jobs']:
                result=remote(['sacct','-S','2026-10-27','-j',previous['job_id'],'-P','-n',
                               '--format=JobID%64,State%40,ElapsedRaw,AllocCPUS'])
                rows=[v.split('|') for v in result.stdout.splitlines()]
                exact=[r for r in rows if r[0]==previous['job_id']]
                if result.returncode or len(exact)!=1 or exact[0][1]!='FAILED':
                    raise RuntimeError('Only the verified failed pre-sampling guard attempt may be corrected')
                spent+=int(exact[0][2])
            wall_seconds=600  # Future requests use whole minutes; historical630s request rounded to660s.
            if spent+wall_seconds+240>900 or (spent+wall_seconds+240)*4>3600:
                raise RuntimeError('Cumulative native allocation/cleanup allowance would exceed original budget')
            name='native-heston-corrected-guard';output=root+'/runs/heston-native-adaptive-guard-fixed'
        command=env+[OLD+'/repo/.pixi/envs/core-gpu/bin/python',repo+'/scripts/check_cached_heston_native.py',
                     '--euler-run',FOLLOWUP+'/runs/heston-flow-matching-500','--references',REVIEWED+'/runs/heston',
                     '--out',output]
        resources=['--account=default','--partition=gpu-debug','--qos=gpu-debug','--cpus-per-task=4',
                   '--gres=gpu:l40s:1','--mem=24G','--time='+('00:10:00' if wall_seconds==600 else '00:15:00')]
        suffix=['--wrap='+shlex.join(command)]
        row={'role':'native','cpu_seconds_per_task_cap':wall_seconds*4,'gpu_seconds_cap':wall_seconds,
             'output':output,'corrective_retry':bool(args.corrective_retry)}
    elif args.role=='preflight':
        target=ledger['bank'];name='bank-preflight'
        if target['jobs']:raise RuntimeError('Preflight already recorded; inspect before any recovery')
        command=env+[OLD+'/repo-conditioned-reference/.pixi/envs/core-reference/bin/python',
                     root+'/manifests/sbi4finance-bank-preflight-20261027.py','--repo',repo,'--root',root,
                     '--audit',root+'/manifests/rough-bank-seed-audit-20261027.json']
        resources=['--account=carney-mjfrank-condo2','--partition=batch','--cpus-per-task=1','--mem=8G','--time=00:15:00']
        suffix=['--wrap='+shlex.join(command)];row={'role':'preflight','cpu_seconds_per_task_cap':900}
        row['budget_check']=budget(ledger,1,900)
    else:
        target=ledger['bank']
        if not args.indices:raise ValueError('Explicit indices required')
        preflight=remote(['cat',root+'/preflight.json'])
        if preflight.returncode:raise RuntimeError('Completed deployment preflight required before workers')
        checked=json.loads(preflight.stdout)
        if (checked.get('status')!='completed_metadata_only'
                or checked.get('source_commit')!=ledger['source_revision']
                or checked.get('config_sha256')!=target['config_sha256']):
            raise RuntimeError('Preflight/source/config mismatch; no worker submission')
        actual_map=remote(['sha256sum',root+'/jobs-20k.json'])
        if actual_map.returncode or actual_map.stdout.split()[0]!=checked['job_map_sha256']:
            raise RuntimeError('Frozen job map bytes changed')
        indices=[int(v) for v in args.indices.split(',')]
        if len(indices)!=len(set(indices)) or not indices or min(indices)<0 or max(indices)>=1280:
            raise ValueError('Distinct original-map indices0..1279 required')
        submitted={i for j in target['jobs'] if j['state']!='rejected' for i in j.get('indices',[])}
        if submitted.intersection(indices):raise RuntimeError('Indices already submitted; no duplicate or automatic retry')
        name=f'bank-workers-{indices[0]}-{indices[-1]}'
        row={'role':'worker','indices':indices,'cpu_seconds_per_task_cap':10800,'budget_check':budget(ledger,len(indices),10800)}
        resources=['--account=carney-mjfrank-condo2','--partition=batch','--cpus-per-task=1','--mem=8G',
                   '--time=03:00:00','--array='+','.join(map(str,indices))+f'%{min(200,len(indices))}']
        suffix=[repo+'/cluster/calibration_collection_array.sbatch',
                OLD+'/repo-conditioned-reference/.pixi/envs/core-reference/bin/python',root+'/jobs-20k.json']
    arguments=['sbatch','--parsable','--job-name=sbi4f-'+name,*common,*resources,
               '--output='+root+'/logs/'+name+'-%A_%a.out',*suffix]
    row.update(name=name,state='submitting',created_utc=now(),arguments=arguments,submission_cwd=repo,
               source_revision=target.get('source_revision',ledger['source_revision']),
               operator_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target['jobs'].append(row);save(ledger)
    result=remote(arguments,cwd=repo);row.update(stdout=result.stdout,stderr=result.stderr,returncode=result.returncode)
    if result.returncode:
        row['state']='unknown_submission';save(ledger)
        raise RuntimeError('Submission response ambiguous: reserve exposure and reconcile before retry. '+result.stderr)
    identifier=result.stdout.strip().split(';')[0]
    if not identifier.isdigit():
        row['state']='unknown_submission';save(ledger);raise RuntimeError('Inspect nonstandard submission response before retry')
    row.update(state='submitted',job_id=identifier,submitted_utc=now());target['status']='submitted';save(ledger)
    print(name,identifier,flush=True)
    subprocess.run(['scp',str(PATH),'oscar:'+root+'/execution.json'],check=True)

if __name__=='__main__':main()
