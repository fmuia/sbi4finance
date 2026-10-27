"""One-off metadata-only deployment check; create/plan only after RNG replay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo/'src'))
    import numpy as np
    from sbi4finance.calibration import bank, bank_collection
    config_path = args.repo/'configs/calibration_rough_joint_development_20k.json'
    config = json.loads(config_path.read_text())
    audit = json.loads(args.audit.read_text())
    assert audit['passed'] and sha(config_path) == audit['config_bytes_sha256']
    assert bank.task_fingerprint(config) == audit['task_fingerprint']
    started = time.perf_counter()
    report = {'format':'rough_bank_deployment_preflight_v1','status':'rng_replay',
              'config_sha256':sha(config_path),'audit_sha256':sha(args.audit),
              'versions':bank.versions(),'no_financial_prices_generated':True,
              'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.repo,text=True).strip()}
    def save():
        p=args.root/'preflight.json';temp=p.with_suffix('.json.tmp')
        temp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');temp.replace(p)
    save()
    try:
        prefix=bank_collection._seed_prefix(config)
        state_hash,seed_hash=hashlib.sha256(),hashlib.sha256()
        states=seeds=0
        def state(sequence):
            nonlocal states
            value=np.random.PCG64(sequence).state
            state_hash.update(hashlib.sha256(bank._json(value).encode()).digest());states+=1
        def seed(value):
            nonlocal seeds
            seed_hash.update(value.to_bytes(8,'little'));seeds+=1
        for split in bank.SPLITS:
            for index in range(config['splits'][split]):
                identity=[*prefix,bank.SPLITS[split],index]
                for purpose in (0,2,3):state(np.random.SeedSequence([*identity,purpose]))
                pricing_seed=bank_collection.pricing_seed(config,split,index);seed(pricing_seed)
                for block in range(3):
                    for rung in range(len(config['task']['numerics']['paths_ladder'])):
                        sequence=np.random.SeedSequence([pricing_seed,block,rung])
                        seed(int(sequence.generate_state(1,dtype=np.uint64)[0]));state(sequence)
        expected=audit['candidate']
        assert states==expected['unique_initialized_pcg64_states']==768000
        assert seeds==expected['unique_reduced_uint64_pricing_and_leaf_seeds']==704000
        assert state_hash.hexdigest()==expected['ordered_pcg64_state_sha256_digest_bytes_sha256']
        assert seed_hash.hexdigest()==expected['ordered_uint64_le_sha256']
        report.update(status='rng_replay_passed',candidate_states=states,candidate_seeds=seeds,
                      rng_state_digest=state_hash.hexdigest(),rng_seed_digest=seed_hash.hexdigest())
        save()
        driver_paths=['src/sbi4finance/calibration/bank_collection.py','scripts/calibration_bank_collection.py',
                      'cluster/calibration_collection_array.sbatch','configs/calibration_rough_joint_development_20k.json']
        report['driver_sha256']={name:sha(args.repo/name) for name in driver_paths}
        collection=args.root/'collection';job_map=args.root/'jobs-20k.json'
        cli=[sys.executable,str(args.repo/'scripts/calibration_bank_collection.py')]
        subprocess.run([*cli,'init','--config',str(config_path),'--out',str(collection)],check=True,cwd=args.repo)
        subprocess.run([*cli,'plan','--collection',str(collection),'--train','20000','--chunk','25','--out',str(job_map)],check=True,cwd=args.repo)
        jobs=json.loads(job_map.read_text())
        assert len(jobs['jobs'])==1280
        assert [(j['split'],j['start'],j['stop']) for j in jobs['jobs'][:4]]==[('train',i*25,(i+1)*25) for i in range(4)]
        report.update(status='completed_metadata_only',collection_id=jobs['collection_id'],
                      collection_sha256=sha(collection/'collection.json'),job_map_sha256=sha(job_map),
                      planned_shards=len(jobs['jobs']),initial_shard_indices=[0,1,2,3],seconds=time.perf_counter()-started)
        save()
        print(json.dumps({k:report[k] for k in ('status','collection_id','planned_shards','seconds')}))
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error),seconds=time.perf_counter()-started)
        save();raise

if __name__=='__main__':main()
