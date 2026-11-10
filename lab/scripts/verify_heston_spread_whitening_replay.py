"""Read-only identity audit of an exact ZCA checkpoint replay and saved payloads."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import replay_heston_spread_whitening as replay


def audit(protocol, preserved_catalogue):
    protocol,preserved_catalogue = Path(protocol).resolve(),Path(preserved_catalogue).resolve()
    declaration = replay.selected_contract(json.loads(protocol.read_text()))
    contract,out = declaration['contract'],Path(declaration['output'])
    run = json.loads((out/'run.json').read_text())
    training = json.loads((out/'training.json').read_text())
    original = Path(contract['original']['directory'])
    old_run = json.loads((original/'run.json').read_text())
    old_curve = json.loads((original/'learning-curve.json').read_text())
    curve = json.loads((out/'learning-curve.json').read_text())
    if (run['source'] != replay.sources() or run['training_protocol'] != contract
            or run['protocol_sha256'] != replay.fixed.digest(protocol)
            or training['status'] != 'complete' or training['completed_updates'] != 4000
            or training['completed_bank_passes'] != 100 or training['presentations'] != 1000000
            or contract['schedule_updates'] != 8000 or len(curve) != 100
            or contract['checkpoint_updates'] != [2000,3000,3320,4000]):
        raise ValueError('replay source, protocol or completed budget differs')
    replay.assert_initial_state(run['initial_state'],old_run['initial_state'])
    if (out/'initial-probes.json').read_bytes() != (original/'initial-probes.json').read_bytes():
        raise ValueError('initial probes differ')
    if (out/'preprocessing.json').read_bytes() != (original/'preprocessing.json').read_bytes():
        raise ValueError('replay refitted or changed frozen preprocessing')
    for actual,expected in zip(curve,old_curve[:100],strict=True):
        replay.assert_curve_point(actual,expected)
    with np.load(out/'exposure.npz',allow_pickle=False) as archive:
        if archive['presentations'].shape != (10000,) or not np.all(archive['presentations']==100):
            raise ValueError('full fixed-row exposure differs')
    checkpoints=[]
    for step in contract['checkpoint_updates']:
        path=out/'checkpoints'/f'update-{step:06d}'
        wrapper,record=replay.load_checkpoint(path)
        loaded=replay.fixed.state_summary(wrapper.estimator,replay.fixed.current_state(wrapper.estimator))
        replay.assert_initial_state(loaded,record['state'])
        if record['probe_metrics'] != curve[step//40-1]:
            raise ValueError('checkpoint probe differs from exact curve')
        if (path/'preprocessing.json').read_bytes() != (original/'preprocessing.json').read_bytes():
            raise ValueError('checkpoint preprocessing changed')
        schedule=record['optimizer_schedule']
        if schedule['class_name']!='CosineDecay' or schedule['config']['decay_steps']!=8000:
            raise ValueError('checkpoint cosine horizon differs')
        # Deterministic serialization probe at the training-input mean, not a
        # market observation or an additional statistical evaluation dataset.
        mean=np.asarray(wrapper.transform.mean)[None,:]
        conditions={'observations':np.repeat(mean,2,axis=0)}
        first=wrapper.sample(conditions=conditions,num_samples=16,sample_shape=(),seed=7721)
        restored,_=replay.load_checkpoint(path)
        second=restored.sample(conditions=conditions,num_samples=16,sample_shape=(),seed=7721)
        if any(not np.array_equal(first[name],second[name]) for name in record['names']):
            raise ValueError('seeded native reload draws differ')
        checkpoints.append({'updates':step,'epoch':step//40,'checkpoint_json_sha256':replay.fixed.digest(path/'checkpoint.json'),
            'all_saved_full_state_arrays_match_loaded':True,'state_array_counts':{group:len(record['state'][group]) for group in replay.fixed.GROUPS},
            'frozen_preprocessing_bytes_match_original':True,'schedule_decay_steps':8000,'native_seeded_reload_draws_exact':True})
    catalogue=json.loads(preserved_catalogue.read_text())
    changed=[]
    for value in catalogue['files']:
        path=Path(value['path'])
        if not path.is_file() or replay.fixed.digest(path)!=value['sha256']:
            changed.append(str(path))
    if changed:
        raise ValueError('preserved file changed: '+', '.join(changed))
    protected_fingerprint=hashlib.sha256(json.dumps(catalogue['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {'format':'heston_spread_whitening_checkpoint_replay_results_verification_v1','status':'passed',
        'audit_source':{'path':str(Path(__file__).resolve()),'sha256':replay.fixed.digest(__file__)},
        'protocol':{'path':str(protocol),'sha256':replay.fixed.digest(protocol)},'producer_sources':replay.sources(),
        'training':{'path':str(out/'training.json'),'sha256':replay.fixed.digest(out/'training.json')},
        'checks':{'all_four_original_initial_state_groups_hash_exact':True,'original_initial_probe_bytes_exact':True,
            'original_preprocessing_bytes_exact':True,'original_learning_curve_prefix_points_exact':100,
            'only_ignored_learning_curve_field':'seconds','all_10000_rows_seen_exactly_100_times':True,
            'schedule_horizon_8000_while_stopping_4000':True,'all_checkpoints_load_full_state_and_preserve_seeded_draws':True,
            'preserved_files_checked':len(catalogue['files']),'preserved_files_unchanged':True},
        'preserved_catalogue':{'path':str(preserved_catalogue),'sha256':replay.fixed.digest(preserved_catalogue),
            'file_count':len(catalogue['files']),'files_fingerprint':protected_fingerprint},'checkpoints':checkpoints,
        'limits':['Historical early weights were never saved; exact equality of those unavailable weights is not directly testable.',
            'All saved curve fields except elapsed seconds and all initial state array hashes agree with the original trajectory.',
            'This verifies replay identity and serialization, not posterior calibration or scientific acceptance.'],
        'reserved_arrays_opened':False,'new_pricing_labels':False,'new_reference_sampling':False,'scientific_acceptance':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--preserved-catalogue',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result=audit(args.protocol,args.preserved_catalogue)
    replay.fixed.write_json(args.out,result)
    print(json.dumps({'status':result['status'],'checks':result['checks']}))


if __name__=='__main__':
    main()
