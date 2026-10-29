"""Native snapshot freeze plus bounded numerical/RNG integrity summary.

Runs inside the existing one-CPU final allocation. No model fit, posterior
sampling, financial pricing calls or statistical test-outcome evaluation.
"""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--source-revision',required=True);a=p.parse_args()
 sys.path.insert(0,str(a.repo/'src'))
 import numpy as np
 from sbi4finance.calibration import bank,bank_collection as bc,tasks
 start=time.perf_counter();output=a.root/'final-check.json'
 if output.exists():raise FileExistsError('Final check already exists; review before recovery')
 report={'format':'known_curve_backbone_final_integrity_v1','status':'started','script_sha256':sha(__file__),'source_revision':a.source_revision,'no_financial_pricing_calls':True,'no_neural_training_or_evaluation':True,'test_use':'Numerical/ID/RNG integrity only; no posterior or model-selection outcomes.'}
 def save():
  temp=output.with_suffix('.json.tmp');temp.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n');temp.replace(output)
 save()
 try:
  pre=json.loads((a.root/'preflight.json').read_text());gate=json.loads((a.root/'initial-check.json').read_text())
  assert pre['status']=='completed_metadata_only' and gate['status']=='completed_initial_gate'
  assert pre['source_revision']==a.source_revision==subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,text=True).strip()
  assert not subprocess.check_output(['git','status','--porcelain'],cwd=a.repo,text=True)
  for key in ('source_revision','collection_id','config_sha256','audit_sha256'):assert gate[key]==pre[key]
  assert sha(a.root/'manifests/config.json')==pre['config_sha256']
  assert sha(a.root/'manifests/rough-known-backbone-seed-audit-20261027.json')==pre['audit_sha256']
  assert sha(a.root/'jobs-20k.json')==pre['job_map_sha256'] and sha(a.root/'collection/collection.json')==pre['collection_sha256']
  subprocess.run([sys.executable,str(a.repo/'scripts/calibration_bank_collection.py'),'freeze','--collection',str(a.root/'collection'),'--train','20000','--out',str(a.root/'snapshot-20k')],cwd=a.repo,check=True,timeout=2700)
  path,manifest,record,parts=bc._read_snapshot(a.root/'snapshot-20k');config=manifest['configuration'];task=config['task']
  assert config['root_seed']==271300 and config['splits']=={'train':20000,'validation':5000,'test':7000} and len(parts)==1280
  assert bank.task_fingerprint(config)==pre['task_fingerprint']=='c99b0dd131896d89d8483a1668a3fde0fb7a0f28794c3db7fe14635cbbd52b65'
  assert bank.source_hashes(task)==pre['source_sha256'] and bank.versions()==pre['runtime']
  references={(x['split'],x['start'],x['stop']):x for x in parts}
  for x in gate['parts']:
   ref=references[(x['split'],x['start'],x['stop'])]
   assert ref['manifest_sha256']==x['manifest_sha256'] and ref['array_sha256']==x['array_sha256']
  assert not list((a.root/'collection/failures').glob('*')) and not list((a.root/'collection/.tmp').glob('*'))
  report.update(status='snapshot_frozen_summary_running',snapshot=str(path),snapshot_manifest_sha256=sha(path/'manifest.json'),initial100_payload_manifest_hashes_preserved=True,source_sha256=pre['source_sha256'],runtime=bank.versions(),task_fingerprint=pre['task_fingerprint'],configuration_splits=config['splits'],collection_id=record['collection_id'],collection_sha256=pre['collection_sha256'],job_map_sha256=pre['job_map_sha256']);save()
  prefix=bc._seed_prefix(config);bounds=np.asarray(task['prior_bounds']);sigma=tasks.sigma(task);maturity=np.array([c['maturity'] for c in task['context']['contracts']]);previous=0.;groups=[]
  for block in task['numerics']['blocks']:
   groups.append((maturity>previous+1e-12)&(maturity<=block['horizon']+1e-12));previous=block['horizon']
  ladder=np.asarray(task['numerics']['paths_ladder']);count=0;ratio_batches={k:[] for k in bank.SPLITS};summaries={k:{'examples':0,'parts':0,'surfaces_above_nominal_target':0,'surfaces_above_final_cap':0,'maximum_reported_mcse_over_quote_sigma':0.,'path_histograms':[{}, {}, {}]} for k in bank.SPLITS};exceptions=[];worst={'maximum_reported_mcse_over_quote_sigma':0.}
  keys=['example_id','parameters','context_values','context_parameters','reference_prices','observations','pricing_standard_errors','pricing_paths']
  for number,ref in enumerate(sorted(parts,key=lambda x:(bank.SPLITS[x['split']],x['start']))):
   if time.perf_counter()-start>3300:raise TimeoutError('Bounded final integrity summary exceeded internal wall allowance')
   part=ref['path'];assert sha(part/'manifest.json')==ref['manifest_sha256']
   with np.load(part/'arrays.npz',allow_pickle=False) as z:arrays={key:z[key] for key in keys}
   split,left,right=ref['split'],ref['start'],ref['stop'];expected_ids=[f'{split}:{i}' for i in range(left,right)];assert np.array_equal(arrays['example_id'],expected_ids)
   for row,i in enumerate(range(left,right)):
    identity=[*prefix,bank.SPLITS[split],i]
    expected=np.random.default_rng(np.random.SeedSequence([*identity,0])).uniform(bounds[:,0],bounds[:,1]);assert np.array_equal(arrays['parameters'][row],expected)
    levels,coefficients=tasks.draw_context(task,np.random.default_rng(np.random.SeedSequence([*identity,3])));assert np.array_equal(arrays['context_values'][row],levels) and np.array_equal(arrays['context_parameters'][row],coefficients)
    observed=arrays['reference_prices'][row]+sigma*np.random.default_rng(np.random.SeedSequence([*identity,2])).standard_normal(len(sigma));assert np.array_equal(arrays['observations'][row,:,0],observed)
   ratios=arrays['pricing_standard_errors']/sigma;paths=arrays['pricing_paths'];assert np.isfinite(ratios).all() and np.all(ratios>=0) and np.isin(paths,ladder).all()
   nominal=np.max(ratios,axis=1)>0.25;hard=np.max(ratios,axis=1)>0.5;assert not np.any(hard)
   s=summaries[split];s['examples']+=right-left;s['parts']+=1;s['surfaces_above_nominal_target']+=int(np.count_nonzero(nominal));s['surfaces_above_final_cap']+=int(np.count_nonzero(hard));s['maximum_reported_mcse_over_quote_sigma']=max(s['maximum_reported_mcse_over_quote_sigma'],float(ratios.max()));ratio_batches[split].append(ratios.reshape(-1))
   for block in range(3):
    over=np.any(ratios[:,groups[block]]>0.25,axis=1);assert np.all(paths[over,block]==ladder[-1]),'Nominal exception before final rung'
    for n,ncount in zip(*np.unique(paths[:,block],return_counts=True)):
     key=str(int(n));s['path_histograms'][block][key]=s['path_histograms'][block].get(key,0)+int(ncount)
   for row in np.flatnonzero(nominal):exceptions.append({'example_id':expected_ids[row],'maximum_reported_mcse_over_quote_sigma':float(ratios[row].max()),'blocks_above_nominal_target':[b for b in range(3) if np.any(ratios[row,groups[b]]>0.25)],'paths_by_block':[int(n) for n in paths[row]]})
   row,col=np.unravel_index(np.argmax(ratios),ratios.shape)
   if ratios[row,col]>worst['maximum_reported_mcse_over_quote_sigma']:worst={'maximum_reported_mcse_over_quote_sigma':float(ratios[row,col]),'example_id':expected_ids[row],'contract_index':int(col)}
   count+=right-left
   if number%100==0:report.update(verified_examples=count,verified_parts=number+1,seconds=time.perf_counter()-start);save()
  assert count==32000 and all(summaries[k]['examples']==n for k,n in config['splits'].items())
  for split,batches in ratio_batches.items():summaries[split]['quote_mcse_ratio_quantiles']=dict(zip(['q50','q90','q99'],map(float,np.quantile(np.concatenate(batches),[0.5,0.9,0.99]))))
  report.update(status='completed_checked_snapshot_and_integrity',verified_examples=count,verified_parts=len(parts),all_prior_curve_quote_noise_reconstructed_exactly=True,initial100_payload_manifest_hashes_preserved=True,splits=summaries,nominal_target_exception_cases=exceptions,total_nominal_target_exception_cases=len(exceptions),hard_cap_exceptions=0,worst_reported_mcse=worst,reported_mcse_is_empirical_not_bias_certificate=True,numerical_scope='Original80/504/504 known-curve task; bounded posterior/product grid sensitivity remains required before final teachingclaims.',seconds=time.perf_counter()-start);save();print(json.dumps({'status':report['status'],'verified_examples':count,'nominal_exceptions':len(exceptions),'worst_reported_mcse':worst,'seconds':report['seconds']}),flush=True)
 except BaseException as e:
  report.update(status='failed',error_type=type(e).__name__,error=str(e),seconds=time.perf_counter()-start);save();raise
if __name__=='__main__':main()
