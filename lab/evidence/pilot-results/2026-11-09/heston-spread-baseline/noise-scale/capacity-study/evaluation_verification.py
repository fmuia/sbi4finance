import json
from collections import Counter
from pathlib import Path
import numpy as np
from scripts import evaluate_heston_spread_scale_precision as e
R=Path('/Users/afengler/Projects/proj_sbi4finance/sbi4finance-heston-spread-baseline-20261109')
E=R/'lab/evidence/pilot-results/2026-11-09/heston-spread-baseline/noise-scale/capacity-study/precision-replay'
P=R/'local-artifacts/heston-spread-baseline-20261109/noise-scale/capacity-study/precision-replay'
protocol=e.read(E/'protocol.json')
e.validate_study(protocol,protocol['joint_task'])
baseline=e.read(protocol['baseline_evaluation']['path'])['population']
protected={}
summary=[]
counts={kind:Counter() for kind in ('physical','c','products')}
for b in (128,512):
 for passes in (25,100,250,500):
  path=E/'evaluations'/f'batch-{b}'/f'pass-{passes:04d}'/'evaluation.json'
  r=e.read(path)
  assert r['status']=='complete' and r['protected_inputs_unchanged'] and r['task']==protocol['joint_task']
  assert r['actual_optimizer_updates']==60000+passes*((100000+b-1)//b)
  assert r['family_checks']==384 and r['reserved_arrays_opened'] is False and r['new_reference_sampling'] is False
  for key in ('source_sha256','study_protocol','training_protocol','precision_recovery','reference_protocol','reference_recovery','panel','conditional_draws'):
   e.saved.verify_artifacts(r[key],protected)
  checkpoint=Path(r['checkpoint'])
  for filename,key in (('checkpoint.json','checkpoint_record_sha256'),('model.keras','model_sha256'),('state.npz','state_sha256')):
   assert e.fixed.digest(checkpoint/filename)==r[key]
  assert e.fixed.digest(path.parent/'recovery.npz')==r['population']['recovery_sha256']
  assert r['population']['observations_sha256']==baseline['observations_sha256']
  assert r['population']['truth_sha256']==baseline['truth_sha256']
  assert not r['population']['support']['material_outside_value_count']
  with np.load(path.parent/'recovery.npz',allow_pickle=False) as arrays:
   assert arrays['truth'].shape==(512,6) and arrays['quantiles'].shape==(3,512,6)
   assert e.baseline.observations_digest(arrays['truth'])==baseline['truth_sha256']
   s=e.recovery_summary(arrays['truth'],arrays['quantiles'],r['task']['prior_bounds'])
  assert np.allclose(s['median_rmse'],r['population']['recovery']['median_rmse'],rtol=1e-3,atol=1e-6)
  for case,entry in r['cases'].items():
   assert not entry['neural_support']['material_outside_value_count']
   for prefix in ('reference_run','reference_diagnostics','reference_chain_payload'):
    assert e.fixed.digest(entry[prefix+'_path'])==entry[prefix+'_sha256']
   for name,stats in entry['comparison']['parameters'].items():
    kind='c' if name=='c' else 'physical' if name in r['task']['parameter_names'][:5] else 'products'
    for key in ('mean_status','width_status'):counts[kind][stats[key]]+=1
  c=r['population']['recovery']
  summary.append({'batch_size':b,'passes':passes,'c_rmse':c['median_rmse'][-1],'c_correlation':c['truth_posterior_median_correlation'][-1],'c_width90':c['median_posterior_width90'][-1],'c_width_over_prior':c['median_width90_over_prior'][-1],'sbc_passed':r['population']['rank_dkw']['passed'],'conditional_status':r['conditional_status']})
assert sum(sum(c.values()) for c in counts.values())==384
assert all(e.fixed.digest(path)==digest for path,digest in protected.items())
print(json.dumps({'all_eight_evaluations_verified':True,'conditional_metrics_count':384,'counts_by_kind':{k:dict(v) for k,v in counts.items()},'source_and_artifact_bindings_verified':len(protected),'population_identical_to_original_baseline':True,'reserved_arrays_opened':False,'new_model_loading_sampling_or_fitting_during_audit':False,'endpoints':summary},indent=2))
