"""Exploratory within-case geometry from an existing, fixed validation panel.

No model is loaded and no random draws, pricing or inference are performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.special import ndtr
from scipy.stats import rankdata, skew

EXPECTED_PANEL_SHA = 'd7b6e67f74228aa15de413fd0a2802cbe0c7a586888853ce3ef4403856ee422d'
DECLARATION = {
    'interpretation': 'Exploratory selection and geometry of the learned candidate; no independent reference claim',
    'interior_truth_fraction': [.10, .90],
    'interior_central98_fraction': [.05, .95],
    'interior_central99_fraction': [.025, .975],
    'primary_selection': 'Among strict interior cases, greatest absolute within-case Spearman among short/long/decay pairs; lowest ID breaks ties',
    'secondary_selection': 'Among remaining strict interior cases, greatest absolute eta/rho Spearman; lowest ID breaks ties',
    'near_flat_definition': 'abs(true long_variance minus true short_variance) <= .005 variance units',
    'near_flat_selection': 'Among near-flat cases with all six truths10percent inside bounds, largest decay90percent width/prior width; not an interior showcase',
    'nonlinearity': 'Descriptive additional R2 from a residualized quadratic predictor, in either direction; not a Gaussianity test',
}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def describe(values):
    values=np.asarray(values);good=values[np.isfinite(values)]
    if not len(good): return {'finite':0}
    return {'finite':len(good),'q05':float(np.quantile(good,.05)), 'q25':float(np.quantile(good,.25)),
        'median':float(np.median(good)),'q75':float(np.quantile(good,.75)), 'q95':float(np.quantile(good,.95)),
        'median_absolute':float(np.median(np.abs(good))),
        'fraction_absolute_ge_half':float(np.mean(np.abs(good)>=.5)),
        'fraction_absolute_ge_point8':float(np.mean(np.abs(good)>=.8)),
        'fraction_absolute_ge_point95':float(np.mean(np.abs(good)>=.95))}


def correlation(values, quadratic=False):
    centered=values-values.mean(axis=1,keepdims=True)
    sd=np.sqrt(np.mean(centered*centered,axis=1))
    z=np.divide(centered,sd[:,None,:],out=np.zeros_like(centered),where=sd[:,None,:]>0)
    corr=np.einsum('nsp,nsq->npq',z,z)/values.shape[1]
    valid=sd>0
    corr=np.where(valid[:,:,None]&valid[:,None,:],np.clip(corr,-1,1),np.nan)
    if not quadratic:return corr
    # Project x² off {1,x}; its extra explanatory power is not linear correlation.
    q=z*z-1-np.mean(z*z*z,axis=1)[:,None,:]*z
    cross=np.einsum('nsp,nsq->npq',q,z)/values.shape[1]
    variance=np.mean(q*q,axis=1)
    gain=np.divide(cross*cross,variance[:,:,None],out=np.zeros_like(cross),where=variance[:,:,None]>0)
    gain=np.where(valid[:,:,None]&valid[:,None,:],np.clip(gain,0,1),np.nan)
    return corr,gain


def curve(theta, times):
    return theta[...,4,None]+(theta[...,3,None]-theta[...,4,None])*np.exp(-theta[...,5,None]*times)


def integrated(theta,times):
    return theta[...,4,None]*times+(theta[...,3,None]-theta[...,4,None])*(-np.expm1(-theta[...,5,None]*times))/theta[...,5,None]


def gaussian_cdf_distance(values):
    order=np.sort(values,axis=0);n=len(order);sd=values.std(axis=0)
    fitted=ndtr((order-values.mean(axis=0))/sd)
    return np.maximum(np.max(np.arange(1,n+1)[:,None]/n-fitted,axis=0),
                      np.max(fitted-np.arange(n)[:,None]/n,axis=0)).tolist()


def candidate(index, draws, truth, bounds, pearson, spearman, gain, names):
    values=draws[index]; fractions=(values-bounds[:,0])/(bounds[:,1]-bounds[:,0])
    quant=np.quantile(fractions,[.005,.01,.99,.995],axis=0)
    lo90,hi90=np.quantile(values,[.05,.95],axis=0)
    result={'case_id':f'validation:{index}','row':int(index),'truth':truth[index].tolist(),
        'truth_percentile_0_100':dict(zip(names,(100*np.mean(values<=truth[index],axis=0)).tolist())),
        'truth_inside_marginal90':dict(zip(names,((truth[index]>=lo90)&(truth[index]<=hi90)).tolist())),
        'all_truths_inside_marginal90':bool(np.all((truth[index]>=lo90)&(truth[index]<=hi90))),
        'coverage_not_a_selection_criterion':True,
        'truth_prior_fraction':((truth[index]-bounds[:,0])/(bounds[:,1]-bounds[:,0])).tolist(),
        'posterior_fraction_q005_q01_q99_q995':quant.tolist(),
        'pearson':pearson[index].tolist(),'spearman':spearman[index].tolist(),
        'quadratic_extra_R2_directional':gain[index].tolist(),
        'physical_marginal_skewness':skew(values,axis=0,bias=False).tolist(),
        'physical_moment_matched_gaussian_cdf_distance':gaussian_cdf_distance(values),
        'posterior_probability_rising_curve':float(np.mean(values[:,4]>values[:,3])),
        'rho_eta_quantiles':np.quantile(values[:,1]*values[:,2],[.05,.5,.95]).tolist(),
        'decay90_width_over_prior':float(np.diff(np.quantile(values[:,5],[.05,.95]))[0]/np.diff(bounds[5])[0])}
    if np.all((fractions>0)&(fractions<1)):
        logits=np.log(fractions)-np.log1p(-fractions)
        lp,lg=correlation(logits[None],quadratic=True)
        result.update(logit_pearson=lp[0].tolist(),logit_quadratic_extra_R2_directional=lg[0].tolist(),
            logit_marginal_skewness=skew(logits,axis=0,bias=False).tolist(),
            logit_moment_matched_gaussian_cdf_distance=gaussian_cdf_distance(logits))
    else:result['logit_diagnostics']='Unavailable because at least one saved draw reaches a prior endpoint; no clipping/dropping'
    return result


def export(path, **arrays):
    if path.exists():
        with np.load(path,allow_pickle=False) as old:
            if set(old.files)!=set(arrays) or any(not np.array_equal(old[k],v) for k,v in arrays.items()):
                raise ValueError('Existing candidate arrays changed: '+str(path))
        return
    temporary=path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:np.savez_compressed(handle,**arrays)
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--panel-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    write(a.out/'selection-declaration.json',DECLARATION)
    root=a.panel_dir;task=json.loads((root/'task.json').read_text());evaluation=json.loads((root/'evaluation.json').read_text());training=json.loads((root/'training.json').read_text())
    started=time.perf_counter();digest=sha(root/'draws.npz')
    if digest!=EXPECTED_PANEL_SHA:raise ValueError('Fixed shared panel changed')
    names=task['parameter_names'];bounds=np.asarray(task['prior_bounds'])
    expected=['hurst','eta','rho','short_variance','long_variance','decay']
    if names!=expected or evaluation['evaluated_split']!='validation' or evaluation['example_ids']!=[f'validation:{i}' for i in range(2000)]:raise ValueError('Wrong task/validation identities')
    if evaluation['checkpoint_sha256']!=training['checkpoint_sha256']:raise ValueError('Model identities differ')
    with np.load(root/'draws.npz',allow_pickle=False) as saved:draws=saved['draws'];truth=saved['truth']
    if draws.shape!=(2000,1000,6) or truth.shape!=(2000,6) or not np.isfinite(draws).all() or not np.isfinite(truth).all():raise ValueError('Wrong/invalid saved panel')
    if np.any(draws<bounds[:,0]) or np.any(draws>bounds[:,1]):raise ValueError('Posterior outside prior box')
    pearson,gain=correlation(draws,quadratic=True)
    spearman=correlation(rankdata(draws,axis=1,method='average'))
    tf=(truth-bounds[:,0])/(bounds[:,1]-bounds[:,0]);q=np.quantile(draws,[.005,.01,.05,.95,.99,.995],axis=1)
    qf=(q-bounds[:,0])/(bounds[:,1]-bounds[:,0])
    valid=np.isfinite(pearson).all(axis=(1,2))
    interior=valid&np.all((tf>=.1)&(tf<=.9),axis=1)&np.all(qf[1]>=.05,axis=1)&np.all(qf[4]<=.95,axis=1)&np.all(qf[0]>=.025,axis=1)&np.all(qf[5]<=.975,axis=1)
    ids=np.flatnonzero(interior)
    curve_score=np.max(np.abs(np.stack([spearman[:,3,4],spearman[:,3,5],spearman[:,4,5]],axis=-1)),axis=-1)
    selected=[]
    if len(ids):
        first=int(ids[np.argmax(curve_score[ids])]);selected.append(first)
        remaining=ids[ids!=first]
        if len(remaining):selected.append(int(remaining[np.argmax(np.abs(spearman[remaining,1,2]))]))
    difference=truth[:,4]-truth[:,3]
    strata={'rising':difference>.005,'falling':difference<-.005,'near_flat':np.abs(difference)<=.005}
    pairs=[]
    for i in range(6):
        for j in range(i+1,6):
            pairs.append({'parameters':[names[i],names[j]],'pearson':describe(pearson[:,i,j]),
                'spearman':describe(spearman[:,i,j]),
                'max_direction_quadratic_extra_R2':describe(np.maximum(gain[:,i,j],gain[:,j,i]))})
    flat=np.flatnonzero(strata['near_flat']&np.all((tf>=.1)&(tf<=.9),axis=1)&valid)
    flat_id=None if not len(flat) else int(flat[np.argmax((q[3,flat,5]-q[2,flat,5])/np.diff(bounds[5])[0])])
    chosen=[candidate(i,draws,truth,bounds,pearson,spearman,gain,names) for i in selected]
    maturities=np.unique([c['maturity'] for c in task['context']['contracts']]);times=np.unique(np.r_[np.linspace(0,1,101),maturities])
    for record,index in zip(chosen,selected):
        label=f'candidate-{index:04d}.npz';w=integrated(draws[index],maturities);delta=np.diff(np.r_[0,maturities])
        export(a.out/label,parameters=draws[index],truth=truth[index],prior_bounds=bounds,
            parameter_names=np.asarray(names),case_id=np.asarray(f'validation:{index}'),curve_times=times,
            forward_variance=curve(draws[index],times),truth_forward_variance=curve(truth[index],times),
            observed_maturities=maturities,effective_variance=w/maturities,
            interval_forward_variance=np.diff(np.column_stack([np.zeros(len(w)),w]),axis=1)/delta,
            rho_eta=draws[index,:,1]*draws[index,:,2])
        record['saved_npz']=label;record['saved_npz_sha256']=sha(a.out/label)
    weak=None if flat_id is None else candidate(flat_id,draws,truth,bounds,pearson,spearman,gain,names)
    if weak is not None:
        label=f'weak-decay-{flat_id:04d}.npz'
        export(a.out/label,parameters=draws[flat_id],truth=truth[flat_id],prior_bounds=bounds,
            parameter_names=np.asarray(names),case_id=np.asarray(f'validation:{flat_id}'),
            model_checkpoint_sha256=np.asarray(training['checkpoint_sha256']),
            bank_manifest_sha256=np.asarray(evaluation['bank_manifest_sha256']))
        weak.update(saved_npz=label,saved_npz_sha256=sha(a.out/label),not_an_interior_showcase=True)
    report={'format':'joint100k_saved_posterior_tradeoffs_v1','exploratory_selection':True,
        'scientific_acceptance':False,'independent_posterior_reference':False,'new_sampling':False,
        'saved_panel_shape':list(draws.shape),'parameter_names':names,'prior_bounds':bounds.tolist(),
        'prior':'Six independent physical Uniform distributions; source generator uses vectoruniform, no curveorientation constraint',
        'model_checkpoint_sha256':training['checkpoint_sha256'],'bank_manifest_sha256':evaluation['bank_manifest_sha256'],
        'input_sha256':{str(root/f):sha(root/f) for f in ('draws.npz','task.json','training.json','evaluation.json')},
        'analysis_source_sha256':sha(__file__),'selection':DECLARATION,'strict_interior_cases':int(interior.sum()),
        'pair_distributions':pairs,'curve_orientation_strata':{name:{'cases':int(mask.sum()),
            'long_decay_pearson':describe(pearson[mask,4,5]),'long_decay_spearman':describe(spearman[mask,4,5]),
            'short_decay_spearman':describe(spearman[mask,3,5])} for name,mask in strata.items()},
        'candidates':chosen,'separate_weak_decay_candidate':weak,
        'weak_decay_candidate_is_interior_showcase':False,
        'scope':'All correlations computed within each observed case; no pooled posterior correlation. Quadratic/skew/CDF distances are descriptive learned-candidate diagnostics, not proof of nonGaussian true posteriors.',
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'system':platform.system(),'machine':platform.machine()},
        'seconds':time.perf_counter()-started}
    if sha(root/'draws.npz')!=digest:raise ValueError('Shared panel changed during read')
    write(a.out/'tradeoffs.json',report)
    np.savez_compressed(a.out/'within-case-metrics.npz',pearson=pearson,spearman=spearman,
        quadratic_extra_R2=gain,strict_interior=interior,true_curve_difference=difference)
    print(json.dumps({'strict_interior_cases':int(interior.sum()),'candidate_ids':selected,
                     'weak_decay_candidate':flat_id,'seconds':report['seconds']}))


if __name__=='__main__':main()
