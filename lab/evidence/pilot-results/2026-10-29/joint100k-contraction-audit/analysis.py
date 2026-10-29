"""Descriptive audit of existing saved posterior draws; no sampling or model calls."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.optimize import brentq
from scipy.special import expi, logit


EXPECTED = {'draws.npz':'d7b6e67f74228aa15de413fd0a2802cbe0c7a586888853ce3ef4403856ee422d',
    'task.json':'29eca4e1b970c4f13b99573939aa3449c5280a3a4478f7d24beace6539a9889d',
    'training.json':'ef41e5f9fc24fd897f3bdf1162bed15765387ca1c49107d24f9593cf6b951c37',
    'evaluation.json':'395eab6088e5d3799d9b8d65904ac25daa536991ef43c23f84f161b7e82a8ac9'}


def quantiles(x):
    return dict(zip(['q10','median','q90'],map(float,np.quantile(x,[.1,.5,.9]))))


def metrics(draws, truth, mean, variance, width90=None):
    """One target; exact all-offdiagonal case re-pairing, without random permutations."""
    n, samples=draws.shape;mu=draws.mean(1);v=draws.var(1,ddof=1)
    lo,hi=np.quantile(draws,[.05,.95],axis=1);covered=(truth>=lo)&(truth<=hi)
    sq=(mu-truth)**2;baseline=(truth-mean)**2
    # Sum every offdiagonal posterior-mean/truth pairing analytically.
    mc,tc=mu-mean,truth-mean
    offmse=(n*(np.sum(mc*mc)+np.sum(tc*tc))-2*mc.sum()*tc.sum()-sq.sum())/(n*(n-1))
    sorted_truth=np.sort(truth)
    offcovered=(np.searchsorted(sorted_truth,hi,side='right')-
        np.searchsorted(sorted_truth,lo,side='left')).sum()-covered.sum()
    out={'prior_mean':float(mean),'prior_variance':float(variance),'prior_sd':float(np.sqrt(variance)),
        'physical_variance_ratio':quantiles(v/variance),'physical_sd_ratio':quantiles(np.sqrt(v/variance)),
        'mean_physical_variance_ratio':float(v.mean()/variance),'physical_contraction':float(1-v.mean()/variance),
        'posterior_width90':quantiles(hi-lo),
        'coverage90':{'matched':float(covered.mean()),'offdiagonal':float(offcovered/(n*(n-1)))},
        'mean_recovery':{'rmse':float(np.sqrt(sq.mean())),'prior_midpoint_rmse':float(np.sqrt(baseline.mean())),
            'matched_rmse_ratio':float(np.sqrt(sq.mean()/baseline.mean())),
            'offdiagonal_rmse':float(np.sqrt(offmse)),
            'offdiagonal_rmse_ratio':float(np.sqrt(offmse/baseline.mean()))},
        'bayes_consistency':{'mse_over_mean_posterior_variance':float(sq.mean()/v.mean()),
            'total_variance_over_prior_variance':float((v.mean()+mu.var())/variance),
            'total_variance_over_empirical_truth_variance':float((v.mean()+mu.var())/truth.var()),
            'posterior_mixture_mean_minus_prior_mean_in_prior_sd':float((mu.mean()-mean)/np.sqrt(variance))}}
    if width90 is not None:out.update(prior_width90=float(width90),width90_ratio=quantiles((hi-lo)/width90))
    return out


def product_positive_cdf(z,a,b,c,d):
    full=max(0.,min(b,z/d)-a);lo=max(a,z/d);hi=min(b,z/c)
    partial=(z*np.log(hi/lo)-c*(hi-lo))/(d-c) if hi>lo else 0.
    return float(np.clip((full+partial)/(b-a),0,1))


def run(panel):
    hashes={name:hashlib.sha256((panel/name).read_bytes()).hexdigest() for name in EXPECTED}
    if hashes!=EXPECTED:raise ValueError('Saved panel or metadata identity differs')
    task=json.loads((panel/'task.json').read_text());evaluation=json.loads((panel/'evaluation.json').read_text())
    training=json.loads((panel/'training.json').read_text())
    with np.load(panel/'draws.npz',allow_pickle=False) as stored:draws,truth=stored['draws'],stored['truth']
    if draws.shape!=(2000,1000,6) or truth.shape!=(2000,6) or not np.isfinite(draws).all() or not np.isfinite(truth).all():
        raise ValueError('Unexpected saved validation panel')
    if evaluation['example_ids']!=['validation:'+str(i) for i in range(2000)]:raise ValueError('Different validation IDs')
    bounds=np.asarray(task['prior_bounds']);lower,upper=bounds.T;span=upper-lower;prior_var=span**2/12
    names=task['parameter_names'];unit=(draws-lower)/span
    z=logit(np.clip(unit,1e-12,1-1e-12));existing=1-z.var(1,ddof=0).mean(0)/(np.pi**2/3)
    reported=np.array([evaluation['contraction_logit'][name] for name in names])
    if not np.allclose(existing,reported,rtol=0,atol=1e-12):raise ValueError('Saved logit contraction does not reproduce')
    rows=[]
    for j,name in enumerate(names):
        row={'name':name,**metrics(draws[:,:,j],truth[:,j],(lower[j]+upper[j])/2,prior_var[j],.9*span[j]),
            'prior_bounds':bounds[j].tolist(),'reported_logit_contraction':float(reported[j]),
            'logit_variance_ratio':quantiles(z[:,:,j].var(1,ddof=1)/(np.pi**2/3)),
            'bounds':{'outside_draws':int(((unit[:,:,j]<0)|(unit[:,:,j]>1)).sum()),
                'exact_bound_draws':int(((unit[:,:,j]==0)|(unit[:,:,j]==1)).sum()),
                'clipped_draws':int(((unit[:,:,j]<1e-12)|(unit[:,:,j]>1-1e-12)).sum()),
                'minimum_unit_draw':float(unit[:,:,j].min()),'maximum_unit_draw':float(unit[:,:,j].max())}}
        rows.append(row)
    # Truth-selected strata describe broadness; they are not conditional SBC tests.
    gap=np.abs(truth[:,3]-truth[:,4])/(.5*(truth[:,3]+truth[:,4]));strata=[]
    for name,mask in [('near_flat_relative_gap_le_0.10',gap<=.1),('intermediate_gap_0.10_to_0.50',(gap>.1)&(gap<.5)),('large_contrast_gap_ge_0.50',gap>=.5)]:
        d=draws[mask,:,5];v=d.var(1,ddof=1);q=np.quantile(d,[.05,.95],axis=1)
        strata.append({'name':name,'cases':int(mask.sum()),'relative_gap':quantiles(gap[mask]),
            'decay_physical_sd_ratio':quantiles(np.sqrt(v/prior_var[5])),
            'decay_width90_ratio':quantiles((q[1]-q[0])/(.9*span[5])),
            'scope':'Descriptive truth-selected broadness, not conditional calibration or a required exact-prior posterior.'})
    # Derived functionals: exact prior moments, no simulated prior samples.
    derived=[];rho,eta=2,1
    m=(lower+upper)/2
    variance=(prior_var[rho]+m[rho]**2)*(prior_var[eta]+m[eta]**2)-(m[rho]*m[eta])**2
    a,b=-upper[rho],-lower[rho];c,d=lower[eta],upper[eta]
    quant=lambda q:brentq(lambda x:product_positive_cdf(x,a,b,c,d)-q,a*c,b*d)
    derived.append({'name':'rho_eta','prior_method':'Exact independent-uniform product moments and analytic product CDF',
        **metrics(draws[:,:,rho]*draws[:,:,eta],truth[:,rho]*truth[:,eta],m[rho]*m[eta],variance,quant(.95)-quant(.05))})
    if not np.array_equal(bounds[3],bounds[4]):raise ValueError('This analytic curve-prior audit assumes identical level bounds')
    k0,k1=bounds[5];dv=k1-k0;level_mean=m[3];level_var=prior_var[3]
    def decay_expect(t):return (np.exp(-k0*t)-np.exp(-k1*t))/(dv*t)
    def hmean(t):return ((np.log(k1)-expi(-t*k1))-(np.log(k0)-expi(-t*k0)))/dv
    def hcross(t,s):
        def primitive(k):
            return (-1+np.exp(-t*k)+np.exp(-s*k)-np.exp(-(t+s)*k))/k+t*expi(-t*k)+s*expi(-s*k)-(t+s)*expi(-(t+s)*k)
        return (primitive(k1)-primitive(k0))/dv
    def curve(values,t):return values[...,4]+(values[...,3]-values[...,4])*np.exp(-values[...,5]*t)
    def integrated(values,t):return values[...,4]*t+(values[...,3]-values[...,4])*(-np.expm1(-values[...,5]*t))/values[...,5]
    for t in [.25,1.]:
        var=level_var*(1-2*decay_expect(t)+2*decay_expect(2*t))
        derived.append({'name':'xi_'+str(t)+'y','prior_method':'Analytic exponential moments of uniform decay and independent uniform levels',
            **metrics(curve(draws,t),curve(truth,t),level_mean,var)})
    var=level_var*(1-2*hmean(1)+2*hcross(1,1))
    derived.append({'name':'W_1y','prior_method':'Analytic exponential-integral moments; W(T)=long*T+(short-long)*(1-exp(-decay*T))/decay',
        **metrics(integrated(draws,1),integrated(truth,1),level_mean,var)})
    amean=(hmean(1)-hmean(.5))/.5;asecond=(hcross(1,1)+hcross(.5,.5)-2*hcross(1,.5))/.25
    var=level_var*(1-2*amean+2*asecond)
    derived.append({'name':'forward_variance_0.5_to_1y','prior_method':'Analytic moments of the interval-average analytic initial curve',
        **metrics((integrated(draws,1)-integrated(draws,.5))/.5,(integrated(truth,1)-integrated(truth,.5))/.5,level_mean,var)})
    return {'checked_utc':datetime.now(timezone.utc).isoformat(),'scope':'Saved-draw descriptive contraction audit; no new inference, simulations, prior MC, fitting or promotion.',
        'saved_panel':{'path':str(panel/'draws.npz'),'hashes':hashes,'shape':list(draws.shape),'truth_shape':list(truth.shape),
            'model_sha256':training['checkpoint_sha256'],'bank_sha256':evaluation['bank_manifest_sha256'],'split':'validation','cases':2000,'draws_per_case':1000},
        'prior':{'distribution':'Independent physical-coordinate uniforms; bank.generate_examples uses prior_rng.uniform(bounds[:,0],bounds[:,1]).',
            'bounds':task['prior_bounds'],'analytic_variance':'(upper-lower)^2/12','analytic_width90':'0.9*(upper-lower)',
            'analytic_logit_variance':float(np.pi**2/3),'no_prior_MC':True},
        'existing_definition_audit':{'definition':'1-mean_across_cases(Var_across_draws(logit((theta-lower)/(upper-lower)),ddof=0))/(pi^2/3)',
            'reproduced_max_absolute_difference':float(np.max(np.abs(existing-reported))),
            'population_variance_ddof0_prior_sampling_bias_at1000draws':.001,
            'existing_recovery_correlation_uses':'posterior MEDIAN, while this audit computes posterior MEAN RMSE',
            'clip_interval':[1e-12,1-1e-12],'clip_incidence_audited':True},
        'parameters':rows,'decay_truth_strata':strata,'derived_functionals':derived,
        'permutation_control':'Exact expectation over all offdiagonal saved-posterior/validation-truth pairings. Widths unchanged, no new model calls or random permutations.',
        'bayes_consistency_note':'Aggregate descriptive identities, not conditional/full-copula certification. Posterior variance uses ddof1; posterior-mean Monte Carlo adds approximately1/1000 of posterior variance to risk.',
        'strata_note':'Near-flat true curves weaken decay sensitivity; a joint nuisance-marginal decay posterior is not required to equal its prior. No conditional coverage/risk criterion is imposed on truth-selected strata.',
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
        'scientific_acceptance':False,'reserved_test_evaluated':False,'promotion':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--panel',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=run(a.panel);a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'audit':str(a.out),'parameters':[{'name':x['name'],'sd':x['physical_sd_ratio'],'width':x['width90_ratio']} for x in result['parameters']],
        'decay_truth_strata':result['decay_truth_strata'],'derived_sd':{x['name']:x['physical_sd_ratio'] for x in result['derived_functionals']}},indent=2))
