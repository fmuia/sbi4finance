"""Source staging/preflight only: no worker, Monte Carlo or Slurm submission."""
import getpass
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone

stage=Path('/oscar/scratch/afengler/sbi4finance-rough-actual-layout-portable-20261110')
repo=stage/'repo';prep=stage/'preparation';output=stage/'preflight'
archive=stage/'source.tar.gz'
revision='3e979d94bc1966f62b917236a9e9963cc37029b7'
expected_archive='db3f4b3bff54ac507fd13cce451aa7534ee3f63ffeb904de143ca0e04dc15daa'
expected_work='38455e9565681ea502cac33116f92c27f2d3396eb3ff06c0248b251e9f9edb45'
python=Path('/oscar/scratch/afengler/sbi4finance-full-core-20261026-wy2_j3o4/repo-conditioned-reference/.pixi/envs/core-reference/bin/python')
def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024),b''):result.update(part)
    return result.hexdigest()
def bind(path):return {'path':str(path),'sha256':digest(path)}
def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
if digest(archive)!=expected_archive:raise ValueError('archive transfer hash differs')
if any(repo.iterdir()):raise ValueError('fresh source destination is not empty')
subprocess.run(['tar','-xf',str(archive),'-C',str(repo)],check=True)
source_prep=repo/'lab/evidence/pilot-results/2026-11-10/bergomi-observation-refresh/preparation-portable'
for name in ('work-order.json','pilot.sbatch'):
    with (prep/name).open('xb') as stream:stream.write((source_prep/name).read_bytes())
work_path=prep/'work-order.json';work=json.loads(work_path.read_text())
if digest(work_path)!=expected_work:raise ValueError('staged work-order differs')
if digest(prep/'pilot.sbatch')!=work['sbatch_sha256']:raise ValueError('staged submission declaration differs')
env=os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(repo/'lab/src')+':'+str(repo/'tutorial/src'),
    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
    MPLCONFIGDIR=str(output/'matplotlib-cache'))
command=[str(python),str(repo/'lab/scripts/prepare_rough_actual_layout_pilot_portable.py'),'check-sources','--work-order',str(work_path),'--repo-root',str(repo)]
checked=subprocess.run(command,env=env,text=True,capture_output=True)
(output/'source-check.stdout.txt').write_text(checked.stdout)
(output/'source-check.stderr.txt').write_text(checked.stderr)
probe='import json,sys,platform; from importlib import metadata; import sbi4f_tutorial; from sbi4finance.calibration import rough_actual_layout_pilot_portable as p; print(json.dumps({"python":sys.version,"executable":sys.executable,"numpy":metadata.version("numpy"),"scipy":metadata.version("scipy"),"tutorial_runtime":sbi4f_tutorial.__file__,"pilot_source":p.__file__}))'
probe_result=subprocess.run([str(python),'-c',probe],env=env,text=True,capture_output=True,check=True)
environment=json.loads(probe_result.stdout)
if environment['tutorial_runtime']!=str(repo/'tutorial/src/sbi4f_tutorial/__init__.py') or environment['pilot_source']!=str(repo/'lab/src/sbi4finance/calibration/rough_actual_layout_pilot_portable.py'):
    raise ValueError('execution import resolved outside the new source identity')
if {key:environment[key] for key in ('numpy','scipy')}!=work['execution_versions']:raise ValueError('execution versions differ')
source_checks={name:{**spec,'verified':digest(repo/spec['relative_path'])==spec['sha256']} for name,spec in work['source_bindings'].items()}
if not all(value['verified'] for value in source_checks.values()):raise ValueError('source hash mismatch')
queue=subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%i|%P|%j|%T|%M|%C'],capture_output=True,text=True,check=True).stdout.splitlines()
partitions=subprocess.run(['sinfo','-h','-p','batch','-o','%P|%C'],capture_output=True,text=True,check=True).stdout.splitlines()
fs=os.statvfs(stage)
record={'format':'rough_actual_layout_portable_source_deployment_preflight_v1','status':'passed' if checked.returncode==0 else 'failed',
    'created_utc':datetime.now(timezone.utc).isoformat(),'source_revision':revision,'host':socket.gethostname(),
    'new_stage_root':str(stage),'source_archive':{**bind(archive),'bytes':archive.stat().st_size,'scope':'git archive tracked commit only; ignored banks/environments excluded'},
    'new_repo_root':str(repo),'work_order':bind(work_path),'submission_declaration':bind(prep/'pilot.sbatch'),
    'authoritative_panel_sha256':work['authoritative_panel_sha256'],'original_stored_panel_preserved':True,'execution_environment':environment,'source_bindings':source_checks,'source_check':{'command':command,'exit_code':checked.returncode,'stdout':bind(output/'source-check.stdout.txt'),'stderr':bind(output/'source-check.stderr.txt')},
    'read_only_scheduler_inventory':{'batch_CPU_lines':partitions,'user_jobs':queue,'scope':'as-of observation, not reserved or guaranteed availability'},
    'scratch_filesystem':{'total_bytes':fs.f_blocks*fs.f_frsize,'available_bytes':fs.f_bavail*fs.f_frsize,'scope':'filesystem capacity, not user quota'},
    'condition_created':False,'worker_invoked':False,'rough_MC_executed':False,'pricing_executed':False,'slurm_jobs_submitted':False,
    'existing_remote_repositories_modified':False,'scientific_acceptance':False}
write(output/'deployment.json',record)
print(json.dumps({'record':bind(output/'deployment.json'),'status':record['status'],'sources_verified':len(source_checks),'environment':environment,'scheduler':record['read_only_scheduler_inventory']}))
sys.exit(checked.returncode)
