"""Crossmatch endpoint reference links to declared jobs using JSON only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def read(path): return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def require(value,message):
    if not value: raise ValueError(message)


def checked(spec):
    require(binding(spec['path']) == {key:spec[key] for key in ('path','sha256')},'bound JSON changed')
    return Path(spec['path'])


def audit(evidence):
    evidence = Path(evidence).resolve(); campaign_path = evidence/'campaign-protocol.json'; campaign = read(campaign_path)
    ref_path = checked(campaign['reference_protocol']); reference = read(ref_path)
    expected = {}
    for job in reference['jobs']:
        run_path = Path(job['evidence_out'])/'run.json'; diagnostic_path = Path(job['evidence_out'])/'diagnostics.json'
        run,diagnostics = read(run_path),read(diagnostic_path)
        require(run['status'] == 'completed' and run['reference_ready'] is True and diagnostics['reference_ready'] is True
            and run['case_id'] == job['id'] and run['task'] == job['task']
            and run['observation_sha256'] == job['observation_sha256'] and run['protocol'] == binding(ref_path),
            'declared target-matched ready reference required')
        expected[job['id']] = {'reference_run':binding(run_path),'reference_diagnostics':binding(diagnostic_path),
            'reference_chain_payload':{'path':str(Path(run['chain_payload_path']).resolve()),'sha256':diagnostics['chain_payload_sha256']}}
    require(list(expected) == ['synthetic-broad','synthetic-market-like','market'],'three declared reference jobs required')
    records = []
    for name,spec in campaign['runs'].items():
        protocol = read(checked(spec['protocol'])); run = read(Path(protocol['output'])/'run.json')
        require(run['status'] == 'complete','complete declared fit required')
        for role,item in protocol['evaluations'].items():
            if role == 'secondary' and run['best_validation_archive']['epoch'] == 300: continue
            path = Path(item['output'])/'evaluation.json'; result = read(path)
            require(result['status'] == 'complete' and result['reference_protocol'] == campaign['reference_protocol']
                and result['protocol'] == spec['protocol'] and set(result['cases']) == set(expected), 'complete source-bound endpoint required')
            for case_id,links in expected.items():
                case = result['cases'][case_id]
                job = next(job for job in reference['jobs'] if job['id'] == case_id)
                require(case['reference_ready'] is True and case['observation_sha256'] == job['observation_sha256']
                    and all(case[key] == value for key,value in links.items()), 'endpoint case linked to a different reference job')
            records.append({'fit':name,'endpoint':role,'epoch':result['native_epoch'],'evaluation':binding(path),
                'declared_case_links_matched':list(expected)})
    require(len(records) == 4,'all four distinct declared endpoints required for this twelve-pair audit')
    return {'format':'heston_pr87_grid_reference_link_audit_v1','status':'complete',
        'audited_utc':datetime.now(timezone.utc).isoformat(),'consumer':binding(__file__),
        'campaign_protocol':binding(campaign_path),'reference_protocol':binding(ref_path),'endpoints':records,
        'expected_endpoint_case_pairs':12,'endpoint_case_pairs_verified':3*len(records),'declared_reference_links':expected,
        'chain_arrays_opened':False,'raw_draw_arrays_opened':False,'model_loads':0,'new_pricing_or_posterior_queries':0,
        'scientific_acceptance':False,'scope':'Metadata supplement to the independent raw-draw audit; all reference payload hashes are crossmatched from JSON, with no chain arrays opened.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',required=True,type=Path); parser.add_argument('--out',required=True,type=Path)
    args = parser.parse_args(); result = audit(args.evidence)
    archive = args.out.with_name('independent-reference-link-consumer.py')
    require(not args.out.exists() and not archive.exists(),'new immutable metadata audit output required')
    with archive.open('xb') as stream:stream.write(Path(__file__).read_bytes())
    result['archived_consumer'] = binding(archive)
    require(result['consumer']['sha256'] == result['archived_consumer']['sha256'],'archived metadata consumer differs')
    with args.out.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'endpoint_case_pairs_verified':result['endpoint_case_pairs_verified']}))
