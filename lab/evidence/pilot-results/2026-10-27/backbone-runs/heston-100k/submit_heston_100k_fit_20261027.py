import hashlib,json,shlex,subprocess
from datetime import datetime,timezone
from pathlib import Path
P=Path('/private/tmp/heston-100k-fit-execution-20261027.json')
ROOT='/oscar/scratch/afengler/sbi4finance-heston-100k-fit-20261027'
REPO=ROOT+'/repo'
def remote(argv):
    return subprocess.run(['ssh','-o','BatchMode=yes','oscar',shlex.join(argv)],capture_output=True,text=True,check=True)
def save(d):
    t=P.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(P)
def main():
    if P.exists(): raise RuntimeError('Existing ledger: reconcile manually, no repeat submission')
    old=remote(['sacct','-S','2026-10-27','-j','8028749,8028804,8028941,8029251','-X','-n','-P','--format=JobID%64,State%40,ElapsedRaw,AllocCPUS,ExitCode'])
    expected={'8028749':('COMPLETED','0:0',1),'8028804':('CANCELLED by 140010387','0:0',0),'8028941':('FAILED','1:0',1),'8029251':('COMPLETED','0:0',1)}
    rows=[line.split('|') for line in old.stdout.splitlines() if line.strip()]
    if len(rows)!=4 or {r[0] for r in rows}!=set(expected):raise RuntimeError('Incomplete accounting')
    cpu=gpu=0
    for row in rows:
        state,exitcode,gpus=expected[row[0]]
        if row[1]!=state or row[4]!=exitcode or row[3]!='4':raise RuntimeError('Unreviewed state')
        cpu+=int(row[2])*int(row[3]);gpu+=int(row[2])*gpus
    if cpu+10560>21600 or gpu+2640>4500:raise RuntimeError('Pooled ceiling would be exceeded')
    if remote(['git','-C',REPO,'rev-parse','HEAD']).stdout.strip()!='eb53cea9b26283c5d7094b022af3c9c28f44a596':raise RuntimeError('Frozen source mismatch')
    command=['env','PYTHONPATH='+REPO+'/src','KERAS_BACKEND=jax','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','NUMEXPR_NUM_THREADS=1','XLA_PYTHON_CLIENT_PREALLOCATE=false','MPLCONFIGDIR='+ROOT+'/cache/mpl','XDG_CACHE_HOME='+ROOT+'/cache','/oscar/scratch/afengler/sbi4finance-full-core-20261026-wy2_j3o4/repo/.pixi/envs/core-gpu/bin/python',REPO+'/scripts/run_heston_backbone_fit.py','--recipe',REPO+'/configs/heston_backbone_100k_fit.json','--bank','/oscar/scratch/afengler/sbi4finance-heston-backbone-20261027/snapshot-100k','--references','/oscar/scratch/afengler/sbi4finance-reviewed-unit-20261027-9b495cf/runs/heston','--out',ROOT+'/runs/heston-100k-native']
    argv=['sbatch','--parsable','--nodes=1','--ntasks=1','--no-requeue','--account=default','--partition=gpu-debug','--qos=gpu-debug','--cpus-per-task=4','--gres=gpu:l40s:1','--mem=24G','--time=00:40:00','--chdir='+REPO,'--job-name=sbi4f-heston-100k','--output='+ROOT+'/logs/heston-100k-%A.out','--wrap='+shlex.join(command)]
    d={'format':'heston_100k_backbone_execution_v1','authorization':'Continuation of requested tutorial backbone; planned 100k data/update comparison after matched50k, same recipe and development panel.','source_revision':'eb53cea9b26283c5d7094b022af3c9c28f44a596','root':ROOT,'state':'submitting','attempted_utc':datetime.now(timezone.utc).isoformat(),'argv':argv,'operator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'bank_manifest_sha256':'4b5929b0d0b23698c340eec6defabfd629fe23eca2ed78fe9bd46e4de35c244e','recipe_sha256':hashlib.sha256(Path('/Users/afengler/Projects/proj_sbi4finance/sbi4finance-full-surface-core/configs/heston_backbone_100k_fit.json').read_bytes()).hexdigest(),'allocation_ceiling':{'cpu_hours':3,'gpu_hours':.75,'wall_seconds':2400,'cpus':4,'gpus':1,'cleanup_seconds':240},'pooled_budget_check':{'previous_actual_cpu_seconds':cpu,'previous_actual_gpu_seconds':gpu,'new_limit_plus_cleanup_cpu_seconds':10560,'new_limit_plus_cleanup_gpu_seconds':2640,'pooled_cpu_ceiling_seconds':21600,'pooled_gpu_ceiling_seconds':4500,'reviewed_previous_accounting':old.stdout},'new_pricing_or_MCMC':False,'reserved_test_evaluated':False,'automatic_retry':False}
    save(d)
    try:r=remote(argv)
    except BaseException:
        d['state']='unknown_submission';save(d);raise
    job=r.stdout.strip().split(';')[0]
    d.update(stdout=r.stdout,stderr=r.stderr)
    if not job.isdigit():d['state']='unknown_submission';save(d);raise RuntimeError('Ambiguous submission')
    d.update(state='submitted',job_id=job,submitted_utc=datetime.now(timezone.utc).isoformat());save(d)
    subprocess.run(['scp',str(P),'oscar:'+ROOT+'/execution.json'],check=True)
    print(json.dumps({'job_id':job,'cumulative_full_exposure_cpu_seconds':cpu+10560,'cumulative_full_exposure_gpu_seconds':gpu+2640}))
if __name__=='__main__':main()
