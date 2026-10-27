import json,os,subprocess,sys,time
from pathlib import Path
root=Path('/oscar/scratch/afengler/sbi4finance-heston-audit-20261027');manifest=root/'inputs.json'
if not os.environ.get('SLURM_JOB_ID') or int(os.environ.get('SLURM_CPUS_PER_TASK','0'))!=4:raise ValueError('Require4allocatedCPUs')
started=time.perf_counter();path=root/'audit-run.json';state={'status':'started','scientific_acceptance':False,'no_training_pricing_mcmc_or_test':True,'stages':[]}
def save():
 p=path.with_suffix('.json.tmp');p.write_text(json.dumps(state,indent=2)+'\n');p.replace(path)
save()
try:
 for role,cap in [('old',180),('new',330)]:
  row={'role':role,'status':'running','seconds_cap':cap};state['stages'].append(row);save();t=time.perf_counter()
  with (root/'logs'/f'{role}.log').open('w') as output:
   result=subprocess.run([sys.executable,str(root/'audit_heston_saved_readouts_20261027.py'),'--manifest',str(manifest),'--role',role],stdout=output,stderr=subprocess.STDOUT,timeout=min(cap,570-(time.perf_counter()-started)),check=False)
  row.update(status='completed' if result.returncode==0 else 'failed',seconds=time.perf_counter()-t,returncode=result.returncode);save()
  if result.returncode:raise RuntimeError('Failed '+role+'; no retry')
 state.update(status='completed',seconds=time.perf_counter()-started);save()
except BaseException as error:
 state.update(status='failed',error_type=type(error).__name__,error=str(error),seconds=time.perf_counter()-started);save();raise
