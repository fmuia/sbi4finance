"""Post-hoc final500 checkpoint on the unchanged2000-case development panel."""
import argparse,hashlib,json,os,signal,sys,time
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 t=p.with_suffix('.json.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--bank',type=Path,required=True);a=p.parse_args()
 if not os.environ.get('SLURM_JOB_ID') or int(os.environ.get('SLURM_CPUS_PER_TASK','0'))!=4:raise ValueError('Allocated4CPU GPU required')
 m=json.loads(a.manifest.read_text());root=Path(m['root']);out=root/'results/final100k-validation';out.mkdir(parents=True,exist_ok=False);(out/'partial').mkdir()
 for name,h in m['frozen_sha256'].items():
  if sha(name)!=h:raise ValueError('Frozen audit input changed')
 sys.path.insert(0,str(Path(m['repo'])/'src'))
 import numpy as np
 import bayesflow as bf,jax
 from sbi4finance.calibration import bank_collection,full_surface,validation
 if bf.__version__!='2.1.0' or not any(d.platform=='gpu' for d in jax.devices()):raise ValueError('PinnedBF2.1 GPU required')
 selected=Path(m['new_checkpoints']['100k']);task,w,training=full_surface.load_checkpoint(selected)
 state=json.loads((selected/'resume.json').read_text());last=selected/state['checkpoint_files']['last'];lastsha=sha(last)
 if state['epochs_done']!=500 or lastsha!=state['checkpoint_sha256'][state['checkpoint_files']['last']]:raise ValueError('Final500 identity mismatch')
 w.load_estimator(last)
 if dict(w.inference_network.integrate_kwargs)!={'method':'tsit5','steps':'adaptive'}:raise ValueError('Native default changed')
 banksha=sha(a.bank/'manifest.json')
 if banksha!=training['run_contract']['dataset']['manifest_sha256']:raise ValueError('Different training/validation snapshot')
 manifest,splits=bank_collection.load_bank(a.bank,splits=('validation',),roles=('observations','parameters','example_id'))
 split=splits['validation'];ids=split['example_id'][:2000].tolist()
 if manifest['configuration']['task']!=task or ids!=[f'validation:{i}' for i in range(2000)]:raise ValueError('Task or2000case panel changed')
 report={'format':'heston_posthoc_final_checkpoint_audit_v1','status':'sampling','scientific_acceptance':False,'posthoc_development_candidate':True,'selection_scope':'Final500 was examined after observing selected-checkpoint failures. This is development evidence, not independent selection or final acceptance.','checkpoint_role':'final_epoch500','checkpoint_sha256':lastsha,'checkpoint_path':str(last),'original_selected_checkpoint_sha256':training['checkpoint_sha256'],'original_selected_epoch':training['selected_epoch'],'bank_manifest_sha256':banksha,'evaluated_split':'validation','reserved_test_evaluated':False,'example_ids':ids,'draws_per_surface':1000,'batch_size':100,'seed':0,'targets':task['parameter_names'],'native_policy':dict(w.inference_network.integrate_kwargs),'source_sha256':{'driver':sha(__file__),'full_surface':sha(full_surface.__file__),'validation':sha(validation.__file__)},'completed_surfaces':0}
 path=out/'run.json';write(path,report);start=time.perf_counter();draws=[]
 def stop(signum,frame):raise TimeoutError('Audit deadline; preserve committed batches, no automatic continuation')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(1470)
 try:
  for first in range(0,2000,100):
   x=full_surface.sample(w,task,split['observations'][first:first+100],1000,seed=first)
   np.savez_compressed(out/'partial'/f'batch-{first:04d}.npz',draws=x,truth=split['parameters'][first:first+100],example_ids=split['example_id'][first:first+100])
   draws.append(x);report.update(completed_surfaces=first+100,sampling_seconds=time.perf_counter()-start);write(path,report)
  x=np.concatenate(draws);truth=split['parameters'][:2000];sbc=validation.sbc(x,truth,seed=0)
  evaluation={**report,'status':'completed','sbc':{k:v for k,v in sbc.items() if k!='fractional_ranks'},'coverage':validation.coverage(x,truth),'contraction_logit':dict(zip(task['parameter_names'],validation.contraction(x,np.asarray(task['prior_bounds'])))),'sampling_seconds':time.perf_counter()-start}
  if sha(last)!=lastsha or sha(a.bank/'manifest.json')!=banksha:raise ValueError('Model or snapshot changed')
  for name,h in m['frozen_sha256'].items():
   if sha(name)!=h:raise ValueError('Frozen input changed during audit')
  write(out/'evaluation.json',evaluation);np.savez_compressed(out/'draws.npz',draws=x,truth=truth)
  report.update(status='completed',sbc=evaluation['sbc'],coverage=evaluation['coverage'],sampling_seconds=evaluation['sampling_seconds']);write(path,report)
  print(json.dumps({'status':'completed','sbc':evaluation['sbc'],'coverage95':evaluation['coverage']['0.95']}))
 except BaseException as e:
  report.update(status='failed',error_type=type(e).__name__,error=str(e),sampling_seconds=time.perf_counter()-start);write(path,report);raise
 finally:signal.alarm(0)
if __name__=='__main__':main()
