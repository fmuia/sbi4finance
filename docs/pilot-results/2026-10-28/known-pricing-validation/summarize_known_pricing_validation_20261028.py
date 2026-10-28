"""Summarize a fixed independent-pricing diagnostic without fitting a posterior."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.stats import t

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def interval(rows):
 a=np.asarray(rows);mean=a.mean(0);se=a.std(0,ddof=1)/np.sqrt(len(a));q=t.ppf(.975,len(a)-1)
 return {'mean':mean.tolist(),'mcse':se.tolist(),'pointwise_t95_lower':(mean-q*se).tolist(),
         'pointwise_t95_upper':(mean+q*se).tolist(),'degrees_of_freedom':len(a)-1}
def loglik_rows(values,within_covariance,sigma,y):
 residual=(y-values)/sigma
 # The trace correction removes plug-in squared-MC-mean bias under the
 # estimated covariance. Fitted CV bias and grid bias are not removed.
 return -.5*np.sum(residual**2,axis=-1)+.5*np.sum(within_covariance/sigma**2,axis=-1)
def main():
 started=time.perf_counter()
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
 plan=json.loads((root/'plan.json').read_text());sigma=np.asarray(plan['task']['context']['quote_noise_std'])
 for name,digest in plan['frozen_sha256'].items():
  if sha(name)!=digest:raise ValueError('Frozeninputchanged: '+name)
 report={'format':'known_rough_pricing_summary_v1','status':'complete','plan_sha256':sha(root/'plan.json'),
         'summary_driver_sha256':sha(__file__),
         'scientific_acceptance':False,'reserved_test_evaluated':False,'points':[],'case_likelihood_contrasts':[],
         'uncertainty_scope':'Pointwise Studentt7 intervals across8independentreplicates. Commonstreams acrosspointroles supportpairedcontrasts; notsimultaneousacceptancebands.',
         'likelihood_scope':'Descriptive fixed-point Gaussian-quote likelihood calculations only. TraceMCbiascorrectionapproximateforfittedCV; protocol-required Heston/tractablenoisytoy validation of a roughlikelihoodcorrection has NOT been performed. These numbers are not posterior corrections, validated reference likelihoods, acceptance gates or stochasticMCMC.',
         'covariance_scope':'Withinbatchpathcovarianceconditionalonsame-samplefittedcontrols; externalreplicatedispersionauthoritative. No inversionofrank7batchcovariance.'}
 aggregate=[];raw_aggregate=[];controlled_ll=[];raw_ll=[];within_diags=[];identities=[]
 for point_index,point in enumerate(plan['points']):
  completed=[];raw=[];bankdiag=[];rawdiag=[];difference=[];blockbank=[]
  jobs=[j for j in plan['jobs'] if j['point_index']==point_index]
  for job in jobs:
   folder=root/'results'/f"batch-{job['index']:02d}";run=json.loads((folder/'run.json').read_text())
   if run['status']!='completed' or run['plan_sha256']!=report['plan_sha256'] or not run['frozen_inputs_verified']:
    raise ValueError('Incomplete/unboundpricingbatch')
   if sha(folder/'prices.npz')!=run['prices_sha256']:raise ValueError('Pricingpayloadchanged')
   with np.load(folder/'prices.npz',allow_pickle=False) as a:
    completed.append(a['controlled']);raw.append(a['raw']);difference.append(a['grid_difference'])
    bankdiag.append(np.stack([np.diag(a['covariance'][g,:,g,:]) for g in (0,1)]))
    rawdiag.append(np.stack([np.diag(a['covariance'][g,:,g,:]) for g in (2,3)]))
   identities.append({'index':job['index'],'run_sha256':sha(folder/'run.json'),'prices_sha256':run['prices_sha256'],
                      'elapsed_seconds':run['elapsed_seconds'],'prices_bytes':(folder/'prices.npz').stat().st_size})
  a=np.asarray(completed);r=np.asarray(raw);d=np.asarray(difference)
  if len(a)!=8:raise ValueError('Exactly8replicatesrequired')
  y=np.asarray(point['observed_prices']);emu=np.asarray(point['emulator_scaled_prices'])
  plugin=-.5*np.sum(((y-a.mean(0))/sigma)**2,axis=-1)
  controlled_ll.append(loglik_rows(a,np.asarray(bankdiag),sigma,y));raw_ll.append(loglik_rows(r,np.asarray(rawdiag),sigma,y))
  aggregate.append(a);raw_aggregate.append(r);within_diags.append(np.asarray(bankdiag))
  normalized_error=emu[None,None,:]-a/sigma
  empirical_sem=a.std(0,ddof=1)/np.sqrt(8)/sigma
  nominal_sem=np.sqrt(np.mean(np.asarray(bankdiag),axis=0)/8)/sigma
  record={'case':point['case'],'role':point['role'],'theta':point['theta'],'emulator_loglik':float(-.5*np.sum((y/sigma-emu)**2)),
          'controlled_prices':interval(a),'raw_prices':interval(r),'raw_minus_controlled_over_quote_sigma':interval((r-a)/sigma),
          'emulator_minus_controlled_over_quote_sigma':interval(normalized_error),
          'fine_minus_coarse_over_quote_sigma':interval(d/sigma),
          'max_abs_emulator_mean_error_over_quote_sigma_by_grid':np.max(np.abs(normalized_error.mean(0)),axis=-1).tolist(),
          'rms_emulator_mean_error_over_quote_sigma_by_grid':np.sqrt(np.mean(normalized_error.mean(0)**2,axis=-1)).tolist(),
          'max_abs_grid_shift_over_quote_sigma':float(np.max(np.abs(d.mean(0)/sigma))),
          'rms_grid_shift_over_quote_sigma':float(np.sqrt(np.mean((d.mean(0)/sigma)**2))),
          'plugin_loglik_at_averaged_prices_by_grid':plugin.tolist(),
          'mc_trace_corrected_batch_loglik_by_grid':interval(controlled_ll[-1]),
          'raw_mc_trace_corrected_batch_loglik_by_grid':interval(raw_ll[-1]),
          'external_price_mcse_over_quote_sigma_by_grid':empirical_sem.tolist(),
          'within_batch_nominal_pooled_mcse_over_quote_sigma_by_grid':nominal_sem.tolist()}
  if point['role']=='truth':
   record['stored_bank_minus_independent_coarse_over_quote_sigma']=(
     (np.asarray(point['stored_bank_prices'])-a[:,0,:].mean(0))/sigma).tolist()
  report['points'].append(record)
 for case in (0,2):
  n=next(i for i,p in enumerate(plan['points']) if p['case']==case and p['role']=='neural_mean')
  r=next(i for i,p in enumerate(plan['points']) if p['case']==case and p['role']=='reference_mean')
  contrast=controlled_ll[r]-controlled_ll[n]
  raw_contrast=raw_ll[r]-raw_ll[n]
  report['case_likelihood_contrasts'].append({'case':case,'direction':'reference_mean-minus-neural_mean',
    'emulator_loglik_difference':report['points'][r]['emulator_loglik']-report['points'][n]['emulator_loglik'],
    'controlled_mc_corrected_loglik_difference_by_grid':interval(contrast),
    'raw_mc_corrected_loglik_difference_by_grid':interval(raw_contrast),
    'grid_effect_on_loglik_difference':interval(contrast[:,1]-contrast[:,0]),
    'scope':'Fixedpointcomparisons; neitherpointneedbemode, andthreepoints do not characterize a posterior.'})
 report['pricing_payload_inventory']=identities
 report['summary_elapsed_seconds']=time.perf_counter()-started
 save(root/'summary.json',report)
 np.savez_compressed(root/'batch-summaries.npz',controlled=np.asarray(aggregate),raw=np.asarray(raw_aggregate),
                     within_controlled_diagonals=np.asarray(within_diags),controlled_loglik=np.asarray(controlled_ll),raw_loglik=np.asarray(raw_ll))
 print(json.dumps({'status':'complete','points':len(report['points']),'batches':len(identities),
                   'case_likelihood_contrasts':report['case_likelihood_contrasts']}))
if __name__=='__main__':main()
