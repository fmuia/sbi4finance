"""A finite joint-curve price panel using the already-validated paired kernel.

No bank generation, training, reference sampling or covariance inversion.
"""
import argparse
import copy
import hashlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, record):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def module(path, name):
    spec=importlib.util.spec_from_file_location(name,path); result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result); return result


def inputs(plan):
    sys.path.insert(0, str(Path(plan['repo'])/'src'))
    from sbi4finance.calibration import bank, tasks
    for path, digest in plan['frozen_sha256'].items():
        if sha(path)!=digest: raise ValueError('Frozen input changed: '+path)
    if bank.source_hashes(plan['task'])!=plan['producer_source_sha256']:
        raise ValueError('Actual numerical producer modules differ from the bank')
    if {name:metadata.version(name) for name in ('numpy','scipy')}!=plan['runtime']:
        raise ValueError('Prepared numerical runtime differs')
    return tasks, module(plan['coupled_helper'], 'frozen_coupled_pricing')


def fine_curve(task, theta, fine_steps):
    """Evaluate the analytic curve on both grids, without coarse interpolation."""
    from sbi4finance.calibration import tasks
    fine=copy.deepcopy(task)
    for block, n in zip(fine['numerics']['blocks'], fine_steps): block['n_steps']=n
    return tasks.joint_curve(fine, theta)


def audit(plan, root):
    tasks, helper=inputs(plan)
    from sbi4finance.calibration import pricing
    diagnostic=module(plan['gradient_driver'],'frozen_gradient_check')
    consumer=diagnostic.load_consumer(plan['pricer_consumer'])
    pricer, identity=diagnostic.load_bound_pricer(plan['pricer'],plan['task'],consumer=consumer)
    checks=[diagnostic.check_gradient(pricer,plan['task'],p,check_pytensor=True,consumer=consumer) for p in plan['points']]
    if not all(p['value_parity_passed'] and p['finite_difference_passed'] and p['pytensor']['passed'] for p in checks):
        raise ValueError('Gradient validation failed')
    known_plan=json.loads(Path(plan['known_panel']).read_text())
    known_pricer,known_identity=diagnostic.load_bound_pricer(plan['known_pricer'],known_plan['task'],consumer=consumer)
    known_checks=[diagnostic.check_gradient(known_pricer,known_plan['task'],p,check_pytensor=True,consumer=consumer)
                  for p in known_plan['points']]
    if not all(p['value_parity_passed'] and p['finite_difference_passed'] and p['pytensor']['passed'] for p in known_checks):
        raise ValueError('Known-curve gradient validation failed')
    with np.load(plan['known_batches'],allow_pickle=False) as a:
        known_rescore=diagnostic.rescore_batches(known_pricer,known_plan,a['controlled'],consumer=consumer)
    save(root/'known50k-rescore.json',{'scientific_acceptance':False,'reserved_test_evaluated':False,
        'new_pricing_paths':0,'pricer':known_identity,'gradient_checks':known_checks,'points':known_rescore})
    for point in plan['points']:
        predicted=pricer(consumer.features(plan['task'],point['theta']))[0]
        if not np.allclose(predicted,np.asarray(point['emulator_scaled_prices']),rtol=0,atol=1e-8):
            raise ValueError('Frozen emulator prediction differs')
    # This tiny call checks the reused helper's finest marginal against the
    # actual frozen bank estimator; it is not an extra high-precision batch.
    task=plan['task']; theta=plan['points'][0]['theta']; curve=fine_curve(task,theta,plan['fine_steps'])
    block=task['numerics']['blocks'][0]; contracts=task['context']['contracts'][:2]
    kwargs=dict(horizon=block['horizon'],n_pairs=32,seed=123456789)
    parameters=tasks.model_parameters(task,theta)
    paired=helper.coupled_controlled_block(parameters,curve,contracts,
        steps=(plan['coarse_steps'][0],plan['fine_steps'][0]),**kwargs)
    native=pricing.rough_block(parameters,curve,contracts,n_steps=plan['fine_steps'][0],**kwargs)
    np.testing.assert_allclose(paired['controlled_prices'][1],native[0],rtol=0,atol=1e-14)
    report={'status':'passed','gradient_checks':checks,'pricer':identity,'tiny_finest_price_parity':True,
            'pricing_batch_count':len(plan['jobs']),'scientific_acceptance':False,'reserved_test_evaluated':False}
    save(root/'audit.json',report)


def worker(plan, root, index):
    tasks, helper=inputs(plan)
    if int(os.environ.get('SLURM_CPUS_PER_TASK','0'))!=1 or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('One allocated CPU required')
    if json.loads((root/'audit.json').read_text())['status']!='passed': raise ValueError('Audit must pass first')
    job=plan['jobs'][index]; point=plan['points'][job['point_index']]; task=plan['task']
    folder=root/'results'/f'batch-{index:02d}'; folder.mkdir(exist_ok=False)
    started=time.perf_counter()
    def stop(signum,frame): raise TimeoutError('Fixed1740second deadline; preserve partials, no retry')
    signal.signal(signal.SIGALRM,stop); signal.signal(signal.SIGTERM,stop); signal.alarm(1740)
    record={'status':'running','index':index,'job':job,'point':point,'slurm_job_id':os.environ['SLURM_JOB_ID'],
            'scientific_acceptance':False,'reserved_test_evaluated':False,'blocks_done':0}
    save(folder/'run.json',record)
    try:
        contracts=task['context']['contracts']; maturities=np.asarray([c['maturity'] for c in contracts])
        assignment=np.full(len(contracts),-1)
        for b,block in reversed(list(enumerate(task['numerics']['blocks']))): assignment[maturities<=block['horizon']+1e-12]=b
        if np.any(assignment<0): raise ValueError('Unassigned contracts')
        curve=fine_curve(task,point['theta'],plan['fine_steps']); parameters=tasks.model_parameters(task,point['theta'])
        controlled=np.zeros((2,63)); raw=controlled.copy(); covariance=np.zeros((4,63,4,63))
        difference=np.zeros(63); difference_covariance=np.zeros((63,63)); moments=[]
        for b,block in enumerate(task['numerics']['blocks']):
            chosen=np.flatnonzero(assignment==b)
            found=helper.coupled_controlled_block(parameters,curve,[contracts[i] for i in chosen],
                horizon=block['horizon'],steps=(plan['coarse_steps'][b],plan['fine_steps'][b]),
                n_pairs=plan['paths_per_batch']//2,seed=np.random.SeedSequence(job['stream_entropy_by_block'][b]))
            controlled[:,chosen]=found['controlled_prices'];raw[:,chosen]=found['raw_prices']
            for g in range(4):
                for h in range(4): covariance[g,chosen[:,None],h,chosen[None,:]]=found['joint_covariance'][g,:,h,:]
            difference[chosen]=found['difference']; difference_covariance[np.ix_(chosen,chosen)]=found['difference_covariance']
            moments.append({'block':b,'contracts':chosen.tolist(),'grids':found['moments']})
            np.savez_compressed(folder/f'block-{b}.npz',**{k:v for k,v in found.items() if k!='moments'})
            record.update(blocks_done=b+1,elapsed_seconds=time.perf_counter()-started);save(folder/'run.json',record)
        np.savez_compressed(folder/'prices.npz',controlled=controlled,raw=raw,covariance=covariance,
            grid_difference=difference,grid_difference_covariance=difference_covariance)
        inputs(plan)
        record.update(status='completed',moments=moments,prices_sha256=sha(folder/'prices.npz'),
                      elapsed_seconds=time.perf_counter()-started)
        save(folder/'run.json',record)
    except BaseException as e:
        record.update(status='failed',error=str(e),error_type=type(e).__name__,elapsed_seconds=time.perf_counter()-started)
        save(folder/'run.json',record);raise
    finally: signal.alarm(0)


def summarize(plan,root):
    tasks,_=inputs(plan); diagnostic=module(plan['gradient_driver'],'frozen_gradient_check')
    sigma=tasks.sigma(plan['task']); points=[]; covariance=[]; pooled=[]; raw_pooled=[]
    for index,point in enumerate(plan['points']):
        values=[];raw=[];diagonals=[]
        for job in (j for j in plan['jobs'] if j['point_index']==index):
            folder=root/'results'/f"batch-{job['index']:02d}"; run=json.loads((folder/'run.json').read_text())
            if run['status']!='completed' or sha(folder/'prices.npz')!=run['prices_sha256']: raise ValueError('Incomplete pricing batch')
            with np.load(folder/'prices.npz',allow_pickle=False) as a:
                values.append(a['controlled']);raw.append(a['raw'])
                diagonals.append([np.diag(a['covariance'][g,:,g,:]) for g in (0,1)])
        values=np.asarray(values);raw=np.asarray(raw)
        if values.shape!=(8,2,63): raise ValueError('Exactly8replicates required')
        error=np.asarray(point['emulator_scaled_prices'])[None,None,:]-values/sigma
        record={k:point[k] for k in ('case','role','theta')}
        record.update(emulator_minus_controlled_quote_sigma=diagnostic.interval(error),
            raw_minus_controlled_quote_sigma=diagnostic.interval((raw-values)/sigma),
            fine_minus_coarse_quote_sigma=diagnostic.interval((values[:,1]-values[:,0])/sigma),
            rms_emulator_error_by_grid=np.sqrt(np.mean(error.mean(0)**2,axis=-1)).tolist(),
            max_abs_emulator_error_by_grid=np.max(np.abs(error.mean(0)),axis=-1).tolist(),
            empirical_price_mcse_quote_sigma=(values.std(0,ddof=1)/np.sqrt(8)/sigma).tolist(),
            nominal_within_batch_mcse_quote_sigma=(np.sqrt(np.mean(diagonals,axis=0)/8)/sigma).tolist())
        points.append(record);pooled.append(values);raw_pooled.append(raw)
        covariance.append([np.cov(values[:,g,:]/sigma,rowvar=False,ddof=1)/8 for g in (0,1)])
    np.savez_compressed(root/'batch-summaries.npz',controlled=np.asarray(pooled),raw=np.asarray(raw_pooled),
                        covariance_of_mean_quote_sigma=np.asarray(covariance))
    save(root/'summary.json',{'status':'complete','points':points,'scientific_acceptance':False,
        'reserved_test_evaluated':False,'new_reference_mcmc':False,
        'scope':'Pointwise8batcht7 diagnostics, not simultaneous accuracy or posterior acceptance; rank7covariance not inverted; doubled grid is not exact truth.'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('audit','worker','summarize'))
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    plan=json.loads(a.plan.read_text());root=Path(plan['root'])
    if a.mode=='audit': audit(plan,root)
    elif a.mode=='summarize': summarize(plan,root)
    else: worker(plan,root,int(os.environ.get('SLURM_ARRAY_TASK_ID',a.index)))


if __name__=='__main__':main()
