"""One-off audit of new Heston train IDs20000:100000; no pricing or scheduling.

Reuse the frozen rough audit's inventory and stream helpers. Existing Heston
prefix/holdouts enter history; only appended IDs enter the candidate set.
"""
import argparse, hashlib, importlib.util, json, platform, sys, time
from pathlib import Path
import numpy as np

ROUGH_SCRIPT=Path('/private/tmp/audit_rough_development_seeds_20261027.py')
spec=importlib.util.spec_from_file_location('retained_rough_audit', ROUGH_SCRIPT)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
bank,collection,digest,rng_digest,read=old.bank,old.bank_collection,old.digest,old.rng_digest,old.read
ROOT=old.ROOT
CONFIG=Path('/private/tmp/heston-backbone-100k-config-20261027.json')
INVENTORY=Path('/private/tmp/heston-backbone-source-inventory-20261027.json')
PREVIOUS_AUDIT=Path('/private/tmp/rough-bank-seed-audit-20261027.json')


def main(out):
    started=time.perf_counter();cpu_started=time.process_time()
    config=json.loads(CONFIG.read_text());bank.validate_config(config)
    assert config['root_seed']==271028 and config['splits']=={'train':100000,'validation':5000,'test':5000}
    assert config['task']['task_id']=='hkde_calibration_v1'
    inventory=json.loads(INVENTORY.read_text());assert inventory['source_revision']=='9d903dbb5986f766844fbb942b322c0d55856705'
    assert inventory['source_status']=='' and inventory['runtime_pyproject_matches'] and inventory['runtime_lock_matches']
    prior=json.loads(PREVIOUS_AUDIT.read_text());assert prior['passed']
    historical_states=set();historical_seeds=set();history_configs={};inputs=[];explicit_seeds=set()
    candidate_key=(271028,digest(config['task']))
    def add_config(c,path):
        if not isinstance(c,dict) or not isinstance(c.get('task'),dict) or 'splits' not in c or 'root_seed' not in c:return
        key=(int(c['root_seed']),digest(c['task']))
        if key==candidate_key and (c['splits']['train']>20000 or c['splits']['validation']!=5000 or c['splits']['test']!=5000):
            raise ValueError('Historical declaration already reaches appended candidate IDs: '+path)
        if key not in history_configs:history_configs[key]={'configuration':json.loads(json.dumps(c)),'sources':[]}
        else:
            for split,n in c['splits'].items():history_configs[key]['configuration']['splits'][split]=max(n,history_configs[key]['configuration']['splits'].get(split,0))
        history_configs[key]['sources'].append(path)
    def scalars(x):
        if type(x) is int and x>=0:explicit_seeds.add(x)
        elif isinstance(x,(list,tuple)):
            for v in x:scalars(v)
        elif isinstance(x,dict):
            for v in x.values():scalars(v)
    def walk(x,path):
        if isinstance(x,dict):
            add_config(x,path)
            for k,v in x.items():
                if k=='pcg64_initial_state_sha256' and isinstance(v,str):historical_states.add(bytes.fromhex(v))
                elif k=='seed_sequence_entropy' and isinstance(v,list) and all(type(z) is int for z in v):historical_states.add(rng_digest(np.random.SeedSequence(v)))
                elif k=='seed_sequence_state_uint64' and isinstance(v,list) and v and type(v[0]) is int:historical_seeds.add(v[0])
                elif k in ('seed','seeds','pricing_seed','pricing_seeds','parameter_seed'):scalars(v)
                elif k=='streams' and isinstance(v,dict):scalars(v)
                walk(v,path)
        elif isinstance(x,list):
            for v in x:walk(v,path)
    # Reuse the original frozen file list, avoiding self-inclusion of this new
    # plan/audit and later status artifacts. Record current bytes explicitly.
    for item in prior['input_files']:
        p=Path(item['path']);p=p if p.is_absolute() else ROOT/p
        if not p.exists() and Path(str(p)+'.gz').exists():p=Path(str(p)+'.gz')
        raw,x=read(p);inputs.append({'path':str(p),'uncompressed_sha256':hashlib.sha256(raw).hexdigest(),'previous_audit_input':item})
        if isinstance(x,list) and p.name=='rough-bank-historical-manifests.json':
            for row in x:walk(row['record'],row['path'])
        else:walk(x,str(p))
    walk(inventory['collection_record'],'live-original-Heston-collection')
    # The first rough audit excluded its own candidate. Include all32k now,
    # including every potential rung, whether already generated or not.
    rough_path=ROOT/'configs/calibration_rough_joint_development_20k.json'
    rough=json.loads(rough_path.read_text());assert rough['root_seed']==271200
    walk(rough,str(rough_path));inputs.append({'path':str(rough_path),'uncompressed_sha256':hashlib.sha256(rough_path.read_bytes()).hexdigest()})
    for seed in explicit_seeds:
        historical_seeds.add(seed);historical_states.add(rng_digest(seed))
        for b in range(3):
            for r in range(7):
                ss=np.random.SeedSequence([seed,b,r]);historical_seeds.add(int(ss.generate_state(1,dtype=np.uint64)[0]));historical_states.add(rng_digest(ss))
    def history_state(ss):historical_states.add(rng_digest(ss))
    counts={}
    for (root,task_sha),entry in history_configs.items():
        c=entry['configuration'];task=c['task'];tid=task.get('task_id','');words=[int(task_sha[i:i+8],16) for i in range(0,64,8)]
        counts[tid]=counts.get(tid,0)+sum(c['splits'].values())
        for split,n in c['splits'].items():
            code=bank.SPLITS[split]
            for i in range(n):
                if tid in ('hkde_calibration_v1','rough_calibration_v1','rough_joint_calibration_v1') or tid.endswith('_option_panel_v1'):
                    identity=[0x4F5054,root,*words,code,i]
                    for purpose in ((0,2,3) if 'calibration' in tid else (0,2)):history_state(np.random.SeedSequence([*identity,purpose]))
                    seed=int(np.random.SeedSequence([*identity,1]).generate_state(1,dtype=np.uint64)[0]);historical_seeds.add(seed)
                    if tid.startswith('rough_') and 'calibration' in tid:
                        for b in range(len(task['numerics']['blocks'])):
                            for r in range(len(task['numerics']['paths_ladder'])):
                                ss=np.random.SeedSequence([seed,b,r]);historical_seeds.add(int(ss.generate_state(1,dtype=np.uint64)[0]));history_state(ss)
                    elif tid.endswith('_option_panel_v1'):history_state(seed)
                elif tid=='gbm_drift_known_vol_bank_v1':
                    for purpose in (0,1):history_state(np.random.SeedSequence([root,*words,code,i,purpose]))
                elif tid=='svj_daily_parameter_state_v1':
                    for purpose in range(5):history_state(np.random.SeedSequence([0x53564A31,root&0xffffffff,root>>32,*words,code,i&0xffffffff,i>>32,purpose]))
                elif tid=='heston_theta_state_v1':
                    identity=[0x48535431,root&0xffffffff,root>>32,*words,code]
                    history_state(np.random.SeedSequence([*identity,0,i&0xffffffff,i>>32]))
                    kernel=int(np.random.SeedSequence([*identity,1]).generate_state(1,dtype=np.uint64)[0]);historical_seeds.add(kernel)
                    for purpose in range(3):history_state(np.random.SeedSequence([0x48535631,kernel&0xffffffff,kernel>>32,i&0xffffffff,i>>32,0,0,purpose]))
                else:raise ValueError('Unsupported historical protocol '+tid)
        print('historical',root,tid,sum(c['splits'].values()),flush=True)
    assert history_configs[candidate_key]['configuration']['splits']=={'train':20000,'validation':5000,'test':5000}
    prefix=collection._seed_prefix(config);states=set();seeds=set();state_hash=hashlib.sha256();seed_hash=hashlib.sha256()
    for i in range(20000,100000):
        identity=[*prefix,bank.SPLITS['train'],i]
        for purpose in (0,2,3):
            value=rng_digest(np.random.SeedSequence([*identity,purpose]))
            if value in states or value in historical_states:raise ValueError(f'Initialized-state collision train:{i}/purpose{purpose}')
            states.add(value);state_hash.update(value)
        seed=collection.pricing_seed(config,'train',i)
        if seed in seeds or seed in historical_seeds:raise ValueError(f'Reduced pricing-seed collision train:{i}')
        seeds.add(seed);seed_hash.update(seed.to_bytes(8,'little'))
    full_audit=collection._audit_seeds(config)
    report={'format':'heston_backbone_extension_seed_audit_v1','passed':True,'scope':'Metadata/RNG only; no prices, model training, collection mutation or scheduling. New train IDs20000:100000 only. Existing train0:20000 and both fixed5000-row holdouts intentionally reused, included in history, excluded from candidates. Historical declared-but-ungenerated IDs conservatively included. This checks finite initialized-state/seed identities, not mathematical stream independence.','configuration_sha256':digest(config),'config_bytes_sha256':hashlib.sha256(CONFIG.read_bytes()).hexdigest(),'task_fingerprint':bank.task_fingerprint(config),'root_seed':271028,'splits':config['splits'],'new_ids':{'split':'train','start':20000,'stop':100000,'examples':80000},'candidate':{'reduced_uint64_pricing_seeds':len(seeds),'initialized_pcg64_states':len(states),'ordered_uint64_le_sha256':seed_hash.hexdigest(),'ordered_state_digest_bytes_sha256':state_hash.hexdigest(),'within_candidate_seed_collisions':0,'within_candidate_state_collisions':0,'historical_seed_intersections':0,'historical_state_intersections':0},'full_bank_reduced_seed_audit':full_audit,'historical':{'configurations':len(history_configs),'reduced_seeds':len(historical_seeds),'initialized_states':len(historical_states),'explicit_semantic_seeds':len(explicit_seeds),'declared_examples_by_task':counts,'joint_rough_32k_included':True,'original_course_raw_bank_unavailable':True},'inputs':inputs,'source_inventory_sha256':hashlib.sha256(INVENTORY.read_bytes()).hexdigest(),'reused_rough_audit_sha256':hashlib.sha256(PREVIOUS_AUDIT.read_bytes()).hexdigest(),'reused_rough_script_sha256':hashlib.sha256(ROUGH_SCRIPT.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'versions':{'numpy':np.__version__,'python':platform.python_version()},'wall_seconds':time.perf_counter()-started,'process_cpu_seconds':time.process_time()-cpu_started}
    with Path(out).open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({k:report[k] for k in ('passed','candidate','historical','wall_seconds','process_cpu_seconds')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();main(args.out)
