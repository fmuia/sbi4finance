"""Pure local integrity/MC bookkeeping for one downloaded training shard."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path('/Users/afengler/Projects/proj_sbi4finance/sbi4finance-full-surface-core')
sys.path.insert(0,str(ROOT/'src'))
from sbi4finance.calibration import bank,tasks
from sbi4finance.gbm_bank import _json

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--part',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--native-check',type=Path,required=True);args=parser.parse_args()
evidence=ROOT/'docs/pilot-results/2026-10-27/native-bank-start/bank-initial'
collection=json.loads((evidence/'collection.json').read_text()); config=collection['configuration']; task=config['task']
manifest=json.loads((args.part/'manifest.json').read_text());split=manifest['split'];start,stop=manifest['start'],manifest['stop']
assert split=='train' and 0<=start<stop<=100 and stop-start==25
assert manifest['configuration']==config and manifest['task_fingerprint']==collection['task_fingerprint']
assert manifest['source_sha256']==collection['source_sha256'] and manifest['versions']==collection['versions']
assert bank.source_hashes(task)==collection['source_sha256']
assert hashlib.sha256((args.part/'arrays.npz').read_bytes()).hexdigest()==manifest['sha256']
with np.load(args.part/'arrays.npz',allow_pickle=False) as saved:arrays=dict(saved)
bank.check_arrays(arrays,config,split,start,stop)
sigma=tasks.sigma(task);ratio=arrays['pricing_standard_errors']/sigma
words=[int(collection['task_fingerprint'][i:i+8],16) for i in range(0,64,8)];bounds=np.asarray(task['prior_bounds'])
native=json.loads(args.native_check.read_text());assert native['passed'] and native['exact_linux_prior_and_quote_noise_reconstruction'] and native['array_sha256']==manifest['sha256'] and (native['split'],native['start'],native['stop'])==(split,start,stop)
max_prior_delta=0.;exact_prior_rows=0;exact_quote_rows=0
for row,i in enumerate(range(start,stop)):
    identity=[0x4F5054,config['root_seed'],*words,bank.SPLITS[split],i]
    prior=np.random.default_rng(np.random.SeedSequence([*identity,0]));quotes=np.random.default_rng(np.random.SeedSequence([*identity,2]))
    local_prior=prior.uniform(bounds[:,0],bounds[:,1]);max_prior_delta=max(max_prior_delta,float(np.max(np.abs(arrays['parameters'][row]-local_prior))));exact_prior_rows+=int(np.array_equal(arrays['parameters'][row],local_prior))
    observations=(arrays['reference_prices'][row]+sigma*quotes.standard_normal(sigma.shape))[:,None]
    exact_quote_rows+=int(np.array_equal(arrays['observations'][row],observations))
paths=arrays['pricing_paths'];ladder=task['numerics']['paths_ladder'];assert np.isin(paths,ladder).all()
assignment=np.full(len(sigma),-1);maturities=np.asarray([c['maturity'] for c in task['context']['contracts']])
for b,block in reversed(list(enumerate(task['numerics']['blocks']))):assignment[maturities<=block['horizon']+1e-12]=b
for b in range(3):assert np.all(paths[np.any(ratio[:,assignment==b]>.25,axis=1),b]==ladder[-1])
result={'format':'initial_rough_development_part_integrity_v1','scientific_scope':'Training-only integrity, seed/noise identity and nominalMCSE bookkeeping; no neural/test evaluation or continuous-model accuracy claim.','collection_id':collection['collection_id'],'collection_sha256':hashlib.sha256((evidence/'collection.json').read_bytes()).hexdigest(),'task_fingerprint':collection['task_fingerprint'],'root_seed':config['root_seed'],'split':split,'start':start,'stop':stop,'examples':stop-start,'array_sha256':manifest['sha256'],'manifest_sha256':hashlib.sha256((args.part/'manifest.json').read_bytes()).hexdigest(),'source_sha256':manifest['source_sha256'],'versions':manifest['versions'],'array_shapes':{k:list(v.shape) for k,v in arrays.items()},'exact_id_prior_quote_noise_reconstruction_passed':True,'authoritative_native_reconstruction':native,'portable_local_max_prior_abs_difference':max_prior_delta,'portable_local_bitexact_prior_rows':exact_prior_rows,'portable_local_bitexact_quote_rows':exact_quote_rows,'portability_note':'Exact reconstruction verified in frozen Linux runtime; local Mac float operations may differ by ULPs. No production tolerance changed.','covariance_shape_symmetry_psd_se_consistency_passed':True,'maximum_mc_se_over_quote_sigma':float(np.max(ratio)),'rows_above_nominal_target_025':int(np.count_nonzero(np.any(ratio>.25,axis=1))),'quotes_above_nominal_target_025':int(np.count_nonzero(ratio>.25)),'rows_above_final_cap_05':int(np.count_nonzero(np.any(ratio>.5,axis=1))),'max_ratio_quantiles_per_row':dict(zip(('q0','q50','q90','q100'),np.quantile(np.max(ratio,axis=1),[0,.5,.9,1]).tolist())),'paths_by_block':[{str(n):int(np.count_nonzero(paths[:,b]==n)) for n in ladder} for b in range(3)],'generation_cpu_seconds':manifest['generation_cpu_seconds'],'passed':True}
with args.out.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
print(json.dumps({k:result[k] for k in ('passed','start','stop','maximum_mc_se_over_quote_sigma','rows_above_nominal_target_025','generation_cpu_seconds')}))
