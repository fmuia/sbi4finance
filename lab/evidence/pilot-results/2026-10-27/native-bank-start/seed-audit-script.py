"""One-off, no-pricing audit of all declared 32k RNG identities against retained sources.

Only output is a compact JSON summary; no collection is created. Historical
configuration counts are a conservative declared superset, not completed work.
"""
import argparse, gzip, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np

ROOT = Path('/Users/afengler/Projects/proj_sbi4finance/sbi4finance-full-surface-core')
sys.path.insert(0, str(ROOT / 'src'))
from sbi4finance.calibration import bank, bank_collection
from sbi4finance.gbm_bank import _json


def digest(x): return hashlib.sha256(_json(x).encode()).hexdigest()
def rng_digest(seed): return bytes.fromhex(digest(np.random.PCG64(seed).state))
def read(path):
    raw = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return raw, json.loads(raw)


def main(out):
    start=time.perf_counter()
    config_path=ROOT/'configs/calibration_rough_joint_development_20k.json'
    config=json.loads(config_path.read_text()); bank.validate_config(config)
    assert config['root_seed']==271200 and config['splits']=={'train':20000,'validation':5000,'test':7000}
    assert [b['n_steps'] for b in config['task']['numerics']['blocks']]==[1280,2016,2016]
    historical_states=set(); historical_seeds=set(); history_configs={}; inputs=[]; explicit_seeds=set()

    def add_config(c, path):
        if not isinstance(c,dict) or not isinstance(c.get('task'),dict) or 'splits' not in c or 'root_seed' not in c:return
        if c==config:return
        if c['root_seed']==271200:raise ValueError('candidate root already present in history: '+path)
        key=(int(c['root_seed']),digest(c['task']))
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
                elif k=='seed_sequence_entropy' and isinstance(v,list) and all(type(z) is int for z in v):
                    historical_states.add(rng_digest(np.random.SeedSequence(v)))
                elif k=='seed_sequence_state_uint64' and isinstance(v,list) and v and type(v[0]) is int:historical_seeds.add(v[0])
                elif k in ('seed','seeds','pricing_seed','pricing_seeds','parameter_seed'):scalars(v)
                elif k=='streams' and isinstance(v,dict):scalars(v)
                walk(v,path)
        elif isinstance(x,list):
            for v in x:walk(v,path)

    # Retained bank/collection manifests retrieved read-only from Oscar, plus
    # checked repository evidence. Configs include earlier declared campaigns
    # conservatively even when their complete bank was never generated.
    remote_path=Path('/private/tmp/rough-bank-historical-manifests.json')
    remote=json.loads(remote_path.read_text())
    inputs.append({'path':str(remote_path),'sha256':hashlib.sha256(remote_path.read_bytes()).hexdigest()})
    for item in remote:
        walk(item['record'],item['path'])
    paths=[]
    for folder in ('configs','docs/results','docs/pilot-results','tutorial_final/results','tutorial_final/checks'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and (path.suffix=='.json' or path.name.endswith('.json.gz')):paths.append(path)
    for path in sorted(set(paths)):
        raw,x=read(path)
        if path==config_path:continue
        inputs.append({'path':str(path.relative_to(ROOT)),'uncompressed_sha256':hashlib.sha256(raw).hexdigest()})
        walk(x,str(path.relative_to(ROOT)))
    # Direct retained semantic seeds may initialize default_rng themselves;
    # include all possible production block/rung descendants conservatively.
    for seed in explicit_seeds:
        historical_seeds.add(seed); historical_states.add(rng_digest(seed))
        for b in range(3):
            for r in range(7):
                ss=np.random.SeedSequence([seed,b,r]);historical_seeds.add(int(ss.generate_state(1,dtype=np.uint64)[0]));historical_states.add(rng_digest(ss))

    def history_state(ss):historical_states.add(rng_digest(ss))
    protocol_counts={}
    for (root,task_sha),entry in history_configs.items():
        c=entry['configuration'];task=c['task']; tid=task.get('task_id','');words=[int(task_sha[i:i+8],16) for i in range(0,64,8)]
        protocol_counts[tid]=protocol_counts.get(tid,0)+sum(c['splits'].values())
        for split,n in c['splits'].items():
            code=bank.SPLITS[split]
            for i in range(n):
                if tid in ('hkde_calibration_v1','rough_calibration_v1','rough_joint_calibration_v1') or tid.endswith('_option_panel_v1'):
                    identity=[0x4F5054,root,*words,code,i]
                    for p in ((0,2,3) if 'calibration' in tid else (0,2)):history_state(np.random.SeedSequence([*identity,p]))
                    seed=int(np.random.SeedSequence([*identity,1]).generate_state(1,dtype=np.uint64)[0]);historical_seeds.add(seed)
                    if tid.startswith('rough_') and 'calibration' in tid:
                        for b in range(len(task['numerics']['blocks'])):
                            for r in range(len(task['numerics']['paths_ladder'])):
                                ss=np.random.SeedSequence([seed,b,r]);historical_seeds.add(int(ss.generate_state(1,dtype=np.uint64)[0]));history_state(ss)
                    elif tid.endswith('_option_panel_v1'):history_state(seed)
                elif tid=='gbm_drift_known_vol_bank_v1':
                    for p in (0,1):history_state(np.random.SeedSequence([root,*words,code,i,p]))
                elif tid=='svj_daily_parameter_state_v1':
                    for p in range(5):history_state(np.random.SeedSequence([0x53564A31,root&0xffffffff,root>>32,*words,code,i&0xffffffff,i>>32,p]))
                elif tid=='heston_theta_state_v1':
                    identity=[0x48535431,root&0xffffffff,root>>32,*words,code]
                    history_state(np.random.SeedSequence([*identity,0,i&0xffffffff,i>>32]))
                    kernel=int(np.random.SeedSequence([*identity,1]).generate_state(1,dtype=np.uint64)[0]);historical_seeds.add(kernel)
                    for p in range(3):history_state(np.random.SeedSequence([0x48535631,kernel&0xffffffff,kernel>>32,i&0xffffffff,i>>32,0,0,p]))
                else:raise ValueError('unsupported historical seed protocol '+tid)
        print('historical',root,tid,sum(c['splits'].values()),flush=True)

    seen_states=set();seen_seeds=set(); prior_rows=[]; seed_hash=hashlib.sha256(); state_hash=hashlib.sha256()
    prefix=bank_collection._seed_prefix(config)
    def new_state(ss,label):
        value=rng_digest(ss)
        if value in seen_states or value in historical_states:raise ValueError('PCG64 initialized-state collision '+label)
        seen_states.add(value);state_hash.update(value)
    def new_seed(value,label):
        if value in seen_seeds or value in historical_seeds:raise ValueError('reduced uint64 seed collision '+label)
        seen_seeds.add(value);seed_hash.update(value.to_bytes(8,'little'))
    task=config['task']; bounds=np.asarray(task['prior_bounds'])
    for split in bank.SPLITS:
        for i in range(config['splits'][split]):
            identity=[*prefix,bank.SPLITS[split],i];label=f'{split}:{i}'
            for p in (0,2,3):new_state(np.random.SeedSequence([*identity,p]),label+f'/purpose{p}')
            seed=bank_collection.pricing_seed(config,split,i);new_seed(seed,label+'/pricing')
            # A few frozen prior draws allow independent Linux cross-checks;
            # these contain no prices/observations and do not inspect tests.
            if i in (0,24,25,99,config['splits'][split]-1):
                values=np.random.default_rng(np.random.SeedSequence([*identity,0])).uniform(bounds[:,0],bounds[:,1])
                prior_rows.append({'example_id':label,'pricing_seed':seed,'parameters':values.tolist()})
            for b in range(3):
                for r,paths_count in enumerate(task['numerics']['paths_ladder']):
                    ss=np.random.SeedSequence([seed,b,r]);new_seed(int(ss.generate_state(1,dtype=np.uint64)[0]),label+f'/block{b}/rung{r}');new_state(ss,label+f'/block{b}/rung{r}')
    report={'format':'rough_development_seed_audit_v1','scope':'No pricing, collection creation, seed redraws or scheduling. All32k IDs, all possible block/rung streams; historical configuration counts conservatively include declared but ungenerated IDs. Exact initialized-state uniqueness is not a mathematical proof of stream independence/non-overlap.','configuration_sha256':digest(config),'config_bytes_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),'task_fingerprint':bank.task_fingerprint(config),'root_seed':config['root_seed'],'splits':config['splits'],'split_codes':bank.SPLITS,'purpose_codes':{'prior':0,'pricing_base':1,'quote_noise':2,'context':3},'candidate':{'split_qualified_ids':sum(config['splits'].values()),'unique_reduced_uint64_pricing_and_leaf_seeds':len(seen_seeds),'unique_initialized_pcg64_states':len(seen_states),'possible_block_rung_streams':sum(config['splits'].values())*21,'ordered_uint64_le_sha256':seed_hash.hexdigest(),'ordered_pcg64_state_sha256_digest_bytes_sha256':state_hash.hexdigest(),'within_candidate_seed_collisions':0,'within_candidate_state_collisions':0,'historical_seed_intersections':0,'historical_state_intersections':0},'historical':{'unique_reduced_seeds':len(historical_seeds),'unique_initialized_states':len(historical_states),'explicit_semantic_seeds':len(explicit_seeds),'declared_configurations':len(history_configs),'declared_examples_by_task':protocol_counts,'configurations':[{'root_seed':r,'task_fingerprint':t,**e} for (r,t),e in history_configs.items()]},'prior_seed_spot_checks':prior_rows,'input_files':inputs,'versions':{'numpy':np.__version__,'python':platform.python_version()},'audit_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'elapsed_wall_seconds':time.perf_counter()-start,'passed':True}
    with Path(out).open('x') as stream:json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k in ('passed','candidate','task_fingerprint','elapsed_wall_seconds')}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True);args=parser.parse_args();main(args.out)
