"""Three fixed approximate-emulator PyMC cases; no simulator calls or retries."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path('/oscar/scratch/afengler/sbi4finance-rough-known-reference-20261027')
REPO=Path('/oscar/scratch/afengler/sbi4finance-rough-known-fit-20261027/repo')
BANK=Path('/oscar/scratch/afengler/sbi4finance-rough-known-backbone-20261027/snapshot-20k')
PRICER=Path('/oscar/scratch/afengler/sbi4finance-rough-known-fit-20261027/runs/known-20k-native/pricer/checkpoint')
i=int(os.environ['SLURM_ARRAY_TASK_ID'])
if i not in (0,1,2) or int(os.environ.get('SLURM_CPUS_PER_TASK','0'))!=4:raise ValueError('exact3-case map and4allocatedCPUsrequired')
manifest=json.loads((ROOT/'execution.json').read_text())
for name,expected in manifest['frozen_sha256'].items():
 if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:raise ValueError('frozen input/source changed: '+name)
os.environ['PYTENSOR_FLAGS']='base_compiledir='+str(ROOT/'cache'/f'pytensor-{i}')
command=[sys.executable,str(REPO/'scripts/calibration_reference.py'),'--bank',str(BANK),'--pricer',str(PRICER),'--split','validation','--index',str(i),'--out',str(ROOT/'runs'/f'case-{i}'),'--draws','8000','--tune','4000','--chains','4','--cores','4','--mp-context','spawn','--checkpoint-every','100','--initialization','data-fit','--fit-starts','4','--fit-budget','2000','--seed',str(271400+i),'--no-progress']
subprocess.run(command,cwd=REPO,check=True)
for name,expected in manifest['frozen_sha256'].items():
 if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:raise ValueError('frozen input/source changed during reference')
