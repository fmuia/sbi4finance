"""Freeze one midpoint case before prices, then submit exactly once."""
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess

ROOT=Path('/oscar/scratch/afengler/sbi4finance-interior-corners-20261028')
REPO=Path('/oscar/scratch/afengler/sbi4finance-rough-known-reference-20261027/repo')
CHECKPOINT=Path('/oscar/scratch/afengler/sbi4finance-heston-100k-fit-20261027/runs/heston-100k-native/checkpoint')
PYTHON=Path('/oscar/scratch/afengler/sbi4finance-full-core-20261026-wy2_j3o4/repo/.pixi/envs/core-reference/bin/python')
CANDIDATE_SHA='8d4a243ba099524cf4cda40e4f0616e895879f053829a7f60dadbe65b98301db'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def create(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def main():
    driver=ROOT/'run_heston_interior.py'
    rev=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    if rev!='56dcb13e1d2bf5b6dc96e573dfcaa6240a7104a9' or subprocess.check_output(
        ['git','-C',str(REPO),'status','--porcelain','--untracked-files=no'],text=True):
        raise ValueError('Clean frozen source56 required')
    task=json.loads((CHECKPOINT/'task.json').read_text())
    truth=[sum(pair)/2 for pair in task['prior_bounds']]
    if truth!=[.085,.085,.8,-.525]:raise ValueError('Unexpected task bounds or midpoint')
    resume=json.loads((CHECKPOINT/'resume.json').read_text());candidate=CHECKPOINT/resume['checkpoint_files']['last']
    if sha(candidate)!=CANDIDATE_SHA:raise ValueError('Frozen final500 candidate required')
    frozen=[driver,Path(__file__),candidate,CHECKPOINT/'task.json',CHECKPOINT/'training.json',
            CHECKPOINT/'resume.json',REPO/'pixi.lock',REPO/'pyproject.toml']
    frozen += [REPO/'src/sbi4finance'/name for name in ('hkde.py','heston.py','gbm_bank.py',
        'calibration/reference.py','calibration/pymc_reference_op.py','calibration/reference_initialization.py',
        'calibration/variance_functionals.py','calibration/tasks.py','calibration/pricing.py',
        'calibration/full_surface.py','calibration/inference.py','calibration/bank.py','calibration/bank_collection.py',
        'calibration/surface.py')]
    frozen += [REPO/'scripts'/name for name in ('calibration_reference.py','check_heston_reference_precision.py')]
    plan={'format':'heston_interior_case_declaration_v1','prepared_utc':datetime.now(timezone.utc).isoformat(),
        'root':str(ROOT),'repo':str(REPO),'source_revision':rev,'task':task,'truth':truth,
        'case_selection':'One ex-ante midpoint of all four prior intervals; fixed before observation generation or inference. No posterior-based reselection.',
        'observation_entropy':[0x434F524E,20261028,0],
        'observation_law':'Checked Fourier prices plus one independent Gaussian quote-noise realization; same full63 task.',
        'mcmc':{'draws':8000,'tune':4000,'chains':4,'cores':4,'mp_context':'spawn','seed':28002802,
                'initialization':'data-fit','fit_starts':4,'fit_budget':2000,'checkpoint_every':100},
        'interior_criterion':'Each parameter central99% reference interval must lie strictly within inner90% of its prior range (unit bounds .05,.95). Retain/report failure; never replace the case.',
        'reference_criteria':{'r_hat_max':1.01,'ess_bulk_min':400,'ess_tail_min':400},
        'precision':'Existing16-point doubled Fourier ladder check, descriptive sensitivity not posterior stability certificate.',
        'checkpoint_directory':str(CHECKPOINT),'candidate':str(candidate),'candidate_sha256':CANDIDATE_SHA,
        'neural_seed':28002803,'neural_draws':10000,'scientific_acceptance':False,'reserved_test_evaluated':False,
        'runtime':{name:metadata.version(name) for name in ('numpy','scipy','pymc','pytensor','arviz-stats','bayesflow')},
        'frozen_sha256':{str(p):sha(p) for p in frozen}}
    create(ROOT/'manifest.json',plan)
    command=('env PYTHONPATH='+str(REPO/'src')+' KERAS_BACKEND=jax JAX_PLATFORMS=cpu '
        'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 '
        'MPLCONFIGDIR='+str(ROOT/'cache/mpl')+' XDG_CACHE_HOME='+str(ROOT/'cache')+' '+str(PYTHON)+' '+str(driver)+
        ' --manifest '+str(ROOT/'manifest.json'))
    argv=['sbatch','--parsable','--nodes=1','--ntasks=1','--no-requeue','--account=carney-mjfrank-condo2',
        '--partition=batch','--cpus-per-task=4','--mem=16G','--time=00:30:00','--chdir='+str(REPO),
        '--job-name=sbi4f-heston-interior','--output='+str(ROOT/'logs/slurm-%A.out'),'--wrap='+command]
    ledger={'format':'heston_interior_execution_v1','state':'prospective','root':str(ROOT),
        'manifest_sha256':sha(ROOT/'manifest.json'),'driver_sha256':sha(driver),'argv':argv,
        'resources':{'cpus':4,'gpus':0,'wall_seconds':1800,'cleanup_seconds':240,'full_exposure_cpu_seconds':8160,
                     'separate_cpu_hour_ceiling':3},'automatic_retry':False,'scientific_acceptance':False}
    create(ROOT/'execution-prospective.json',ledger)
    ledger['state']='submission_attempted';create(ROOT/'execution.json',ledger)
    result=subprocess.run(argv,capture_output=True,text=True)
    ledger.update(stdout=result.stdout,stderr=result.stderr,returncode=result.returncode)
    if result.returncode==0 and result.stdout.strip().split(';')[0].isdigit():
        ledger.update(state='submitted',job_id=result.stdout.strip().split(';')[0])
    else:ledger['state']='unknown_or_failed_submission'
    temp=ROOT/'execution.json.tmp';temp.write_text(json.dumps(ledger,indent=2)+'\n');temp.replace(ROOT/'execution.json')
    print(json.dumps(ledger))
    if ledger['state']!='submitted':raise RuntimeError('Submission failed or is ambiguous; do not retry')

if __name__=='__main__':main()
