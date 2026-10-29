"""Read-only checkpoint controls: archive sampler and selected versus final weights."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 t=p.with_suffix('.json.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--role',choices=['old','new'],required=True);a=p.parse_args()
 manifest=json.loads(a.manifest.read_text());root=Path(manifest['root']);out=root/'results'/a.role;out.mkdir(parents=True,exist_ok=False)
 for name,h in manifest['frozen_sha256'].items():
  if sha(name)!=h:raise ValueError('Frozen input changed: '+name)
 repo=Path(manifest['repo']);sys.path.insert(0,str(repo/'src'))
 if a.role=='old':sys.path.insert(0,manifest['overlay'])
 import numpy as np
 import keras,bayesflow as bf,jax
 from sbi4finance.calibration import inference,full_surface,validation
 if bf.__version__!=('2.0.14' if a.role=='old' else '2.1.0'):raise ValueError('Wrong BF version')
 if not any(d.platform=='gpu' for d in jax.devices()):raise ValueError('Allocated GPU required')
 started=time.perf_counter();report={'role':a.role,'status':'started','scientific_acceptance':False,'no_training_pricing_mcmc_or_test':True,'draws':10000,'versions':{'bayesflow':bf.__version__,'keras':keras.__version__,'jax':jax.__version__,'numpy':np.__version__},'cases':[]}
 write(out/'run.json',report)
 refs=[]
 for i in range(3):
  q=Path(manifest['references'])/f'case-{i}'/'reference';r=json.loads((q/'reference.json').read_text());diag=json.loads((q/'diagnostics.json').read_text())
  if diag['mcmc_diagnostics_passed'] is not True:raise ValueError('Reference diagnostics failed')
  with np.load(q/'chains.npz',allow_pickle=False) as z:draws=z['parameters'].reshape(-1,4)
  refs.append((np.asarray(r['observed_prices'])[None,:],draws))
 def sample(w,task,q,seed,kwargs=None):
  cond=(inference.conditioning_data(q,None,task) if a.role=='old' else full_surface.conditioning_data(q,task))
  values=w.sample(num_samples=10000,conditions=cond,seed=seed,**(kwargs or {}))
  result=np.concatenate([np.asarray(values[n],dtype=float) for n in task['parameter_names']],axis=-1)[0]
  if result.shape!=(10000,4) or not np.isfinite(result).all():raise ValueError('Invalid draws')
  return result
 def score(x,r):
  v=validation.agreement(x,r);return {'mean_error_reference_sd':v['standardized_mean_error'],'width_ratio_90':v['width_ratio'],'mean':x.mean(0).tolist(),'sd':x.std(0).tolist(),'pass_proposed_rules':bool(np.all(np.asarray(v['standardized_mean_error'])<=.3) and np.all(np.abs(np.asarray(v['width_ratio'])-1)<=.25))}
 def pair(left,right,r):
  sd=r.std(0);return {'signed_mean_change_reference_sd':((left-right).mean(0)/sd).tolist(),'width_ratio_left_over_right':(np.diff(np.quantile(left,[.05,.95],axis=0),axis=0)[0]/np.diff(np.quantile(right,[.05,.95],axis=0),axis=0)[0]).tolist()}
 try:
  if a.role=='old':
   checkpoint=Path(manifest['archive']);task=json.loads((checkpoint/'task.json').read_text());w=inference.build_workflow(task,subnet_widths=(256,)*4,time_power_law_alpha=-.5);w.load_approximator(checkpoint/'model.keras');inference.use_sampler(w)
   for i,(q,r) in enumerate(refs):
    x=sample(w,task,q,271026+i);y=sample(w,task,q,271026+i,{'method':'tsit5','steps':'adaptive','atol':1e-7,'rtol':1e-5})
    np.savez_compressed(out/f'case-{i}.npz',euler=x,adaptive=y)
    report['cases'].append({'case':i,'seed':271026+i,'euler100_end_dense':score(x,r),'adaptive_tsit5_tight':score(y,r),'adaptive_minus_euler':pair(y,x,r)});write(out/'run.json',report)
  else:
   for label,directory in manifest['new_checkpoints'].items():
    directory=Path(directory);task,w,training=full_surface.load_checkpoint(directory);state=json.loads((directory/'resume.json').read_text());last=directory/state['checkpoint_files']['last']
    if state['epochs_done']!=500 or sha(last)!=state['checkpoint_sha256'][state['checkpoint_files']['last']]:raise ValueError('Final recovery checkpoint identity differs')
    selected=[sample(w,task,q,271026+i) for i,(q,r) in enumerate(refs)]
    w.load_estimator(last)
    if w.inference_network.integrate_kwargs!={'method':'tsit5','steps':'adaptive'}:raise ValueError('Final sampler changed')
    for i,(q,r) in enumerate(refs):
     final=sample(w,task,q,271026+i);x=selected[i];np.savez_compressed(out/f'{label}-case-{i}.npz',selected=x,final=final)
     report['cases'].append({'fit':label,'selected_epoch':training['selected_epoch'],'final_epoch':500,'final_checkpoint':str(last),'final_checkpoint_sha256':sha(last),'case':i,'seed':271026+i,'selected':score(x,r),'final':score(final,r),'final_minus_selected':pair(final,x,r)});write(out/'run.json',report)
  for name,h in manifest['frozen_sha256'].items():
   if sha(name)!=h:raise ValueError('Frozen input changed during readout')
  report.update(status='completed',seconds=time.perf_counter()-started);write(out/'run.json',report)
 except BaseException as e:
  report.update(status='failed',error_type=type(e).__name__,error=str(e),seconds=time.perf_counter()-started);write(out/'run.json',report);raise
if __name__=='__main__':main()
