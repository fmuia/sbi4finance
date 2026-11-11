"""Compare saved MCMC interval summaries only; never open chains or models."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def read(path): return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def checked(spec):
    if binding(spec['path']) != {key:spec[key] for key in ('path','sha256')}:
        raise ValueError('bound summary input changed')
    return Path(spec['path'])


def require(value, message):
    if not value: raise ValueError(message)


def interval(summary):
    quantiles = summary['comparison']['reference_quantiles_05_50_95']
    require(len(quantiles) == 3 and all(len(row) == 6 for row in quantiles)
        and all(math.isfinite(x) for row in quantiles for x in row)
        and all(quantiles[0][j] <= quantiles[1][j] <= quantiles[2][j] for j in range(6)),
        'finite ordered3x6 MCMC quantiles required')
    widths = [quantiles[2][j]-quantiles[0][j] for j in range(6)]
    require(all(value > 0 for value in widths),'positive reference interval widths required')
    return quantiles,widths


def compare(evidence, out):
    evidence,out = Path(evidence).resolve(),Path(out).resolve()
    campaign_path = evidence/'campaign-protocol.json'; campaign = read(campaign_path)
    protocol_path = checked(campaign['runs']['200k-300']['protocol']); protocol = read(protocol_path)
    old_path = checked(protocol['old54_control'])
    new_path = Path(protocol['evaluations']['primary']['output'])/'evaluation.json'
    old,new = read(old_path),read(new_path)
    require(old['status'] == new['status'] == 'complete' and old['endpoint'] == new['endpoint'] == 'primary_final'
        and old['native_epoch'] == new['native_epoch'] == 300 and old['training_rows'] == new['training_rows'] == 200000,
        'completed paired primary200k300 summaries required')
    names = new['task']['parameter_names']
    require(old['task']['parameter_names'] == names and len(names) == 6 and names[-1] == 'c'
        and old['task']['prior_bounds'] == new['task']['prior_bounds']
        and old['task']['quote_count'] == 54 and new['task']['quote_count'] == 63,
        'unchanged six physical/c priors and old54/new63 tasks required')
    prep_path = checked(campaign['readiness']['preparation_protocol']); prep = read(prep_path)
    truth_path = Path(prep['output'])/'conditional-truth-audit.json'; truths = read(truth_path)
    panel_path = checked(protocol['conditional_panel']); panel = read(panel_path)
    old_panel_path = checked(truths['old_panel']); old_panel = read(old_panel_path)
    require(truths['panel'] == protocol['conditional_panel'] and prep['old_panel'] == truths['old_panel']
        and panel['task'] == new['task'] and old_panel['joint_task'] == old['task'],
        'frozen conditional target/truth bindings differ')
    cases = {}
    for case_id in ('synthetic-broad','synthetic-market-like','market'):
        a,b = old['cases'][case_id],new['cases'][case_id]
        require(a['reference_ready'] is True and b['reference_ready'] is True,'ready target-matched MCMC summaries required')
        aq,aw = interval(a); bq,bw = interval(b)
        ratios = [x/y for x,y in zip(bw,aw,strict=True)]
        old_case = next(case for case in old_panel['cases'] if case['id'] == case_id)
        new_case = next(case for case in panel['cases'] if case['id'] == case_id)
        require(b['observation_sha256'] == new_case['observations_sha256'], 'new realized observations differ')
        if case_id != 'market':
            new_truth = next(case for case in truths['cases'] if case['id'] == case_id)
            require(old_case['physical_truth'] == new_truth['physical_truth'] == prep['conditional_truths'][case_id]
                and old_case['generating_c'] == new_truth['generating_c'] == 1.,'synthetic generating physical/c truth changed')
            generating_truth = [*new_truth['physical_truth'],new_truth['generating_c']]
        else: generating_truth = None
        cases[case_id] = {'old_reference_quantiles_05_50_95':aq,'new_reference_quantiles_05_50_95':bq,
            'old_width90':aw,'new_width90':bw,'new_over_old_width90':ratios,
            'physical_width_ratio_range':[min(ratios[:5]),max(ratios[:5])],
            'generating_truth':generating_truth,'same_synthetic_physical_and_c_truth':case_id != 'market',
            'old_observations_sha256':old_case['observations_sha256'],'new_observations_sha256':new_case['observations_sha256'],
            'realized_observations_differ':old_case['observations_sha256'] != new_case['observations_sha256']}
    archive = out.with_name('reference-width-comparison-consumer.py')
    require(not out.exists() and not archive.exists(),'new immutable comparison output required')
    with archive.open('xb') as stream: stream.write(Path(__file__).read_bytes())
    require(binding(archive)['sha256'] == binding(__file__)['sha256'],'archived consumer differs')
    result = {'format':'heston_pr87_grid_reference_width_comparison_v1','status':'complete',
        'created_utc':datetime.now(timezone.utc).isoformat(),'consumer':binding(__file__),'archived_consumer':binding(archive),
        'old54_evaluation':binding(old_path),'new63_evaluation':binding(new_path),'protocol':binding(protocol_path),
        'preparation_protocol':binding(prep_path),'conditional_truth_audit':binding(truth_path),
        'old_panel':binding(old_panel_path),'new_panel':binding(panel_path),'parameter_names':names,
        'prior_bounds_unchanged':True,'prior_bounds':new['task']['prior_bounds'],'cases':cases,
        'chain_arrays_opened':False,'model_loads':0,'new_pricing_or_posterior_queries':0,'reserved_arrays_opened':False,
        'scope':'Descriptive MCMC90% width change for two realized observation tasks. Geometry, measured spreads and most noise vectors differ; this is not a causal put-shift attribution, prior-wide optimality claim or evidence of neural calibration.',
        'scientific_acceptance':False}
    with out.open('x') as stream: stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',required=True,type=Path); parser.add_argument('--out',required=True,type=Path)
    args = parser.parse_args(); result = compare(args.evidence,args.out)
    print(json.dumps({case:row['new_over_old_width90'] for case,row in result['cases'].items()}))
