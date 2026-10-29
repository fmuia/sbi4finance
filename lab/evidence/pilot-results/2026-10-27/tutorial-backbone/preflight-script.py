"""One-off frozen-Heston extension preflight/freeze. No command submits jobs.

Preflight checks existing bytes/RNG identities and writes the existing producer's
job map. --freeze publishes one checked prefix snapshot after data generation.
"""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path

OLD=Path('/oscar/scratch/afengler/sbi4finance-full-core-20261026-wy2_j3o4')
REPO=OLD/'repo'
COLLECTION=OLD/'runs/heston-development/collection'
REVISION='9d903dbb5986f766844fbb942b322c0d55856705'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--freeze',type=int,choices=[50000,100000]);args=p.parse_args()
    sys.path.insert(0,str(REPO/'src'))
    import numpy as np
    from sbi4finance.calibration import bank,bank_collection as bc
    inv=json.loads(args.inventory.read_text());audit=json.loads(args.audit.read_text());config=json.loads(args.config.read_text());started=time.perf_counter()
    assert audit['passed'] and sha(args.inventory)==audit['source_inventory_sha256'] and sha(args.config)==audit['config_bytes_sha256']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==REVISION
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)
    assert all(sha(REPO/name)==expected for name,expected in inv['driver_sha256'].items())
    assert sha(COLLECTION/'collection.json')==inv['collection_sha256']
    _,record=bc._read_collection(COLLECTION,current=True)
    assert bc._target(record,100000)==config and bank.versions()==inv['collection_record']['versions']
    args.root.mkdir(parents=True,exist_ok=True)
    report_path=args.root/(f'freeze-{args.freeze}.json' if args.freeze else 'preflight.json')
    if report_path.exists():raise FileExistsError('Review the existing stage record before any recovery: '+str(report_path))
    report={'format':'heston_backbone_stage_v1','status':'started','stage':'freeze' if args.freeze else 'preflight','source_revision':REVISION,'collection_id':record['collection_id'],'collection_sha256':inv['collection_sha256'],'config_sha256':sha(args.config),'audit_sha256':sha(args.audit),'inventory_sha256':sha(args.inventory),'runtime':bank.versions(),'no_financial_pricing_calls':True,'no_model_training':True,'script_sha256':sha(__file__)}
    def save():
        tmp=report_path.with_suffix('.json.tmp');tmp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');tmp.replace(report_path)
    save()
    try:
        if args.freeze:
            previous=json.loads((args.root/'preflight.json').read_text());assert previous['status']=='completed_metadata_only'
            assert sha(args.root/'jobs-100k.json')==previous['job_map_sha256']
            output=args.root/f'snapshot-{args.freeze//1000}k'
            for original in inv['parts']:
                original_path=Path(original['path'])
                assert sha(original_path/'manifest.json')==original['manifest_sha256'] and sha(original_path/'arrays.npz')==original['manifest']['sha256']
            result=bc.freeze_snapshot(COLLECTION,args.freeze,output)
            expected={(x['manifest']['split'],x['manifest']['start'],x['manifest']['stop']):x for x in inv['parts']}
            selected={(x['split'],x['start'],x['stop']):x for x in result['parts']}
            for key,x in expected.items():
                assert key in selected and selected[key]['manifest_sha256']==x['manifest_sha256'] and selected[key]['array_sha256']==x['manifest']['sha256']
            assert result['configuration']['splits']=={'train':args.freeze,'validation':5000,'test':5000}
            report.update(status='completed_snapshot',snapshot=str(output),snapshot_manifest_sha256=sha(output/'manifest.json'),original_parts_preserved=len(expected),reserved_test_outcomes_evaluated=False)
        else:
            # No pricing: replay all new actual initialized states/reduced seeds
            # on the frozen Linux runtime before creating the extension map.
            state_hash,seed_hash=hashlib.sha256(),hashlib.sha256();prefix=bc._seed_prefix(config)
            for i in range(20000,100000):
                identity=[*prefix,bank.SPLITS['train'],i]
                for purpose in (0,2,3):state_hash.update(hashlib.sha256(bank._json(np.random.PCG64(np.random.SeedSequence([*identity,purpose])).state).encode()).digest())
                seed=bc.pricing_seed(config,'train',i);seed_hash.update(seed.to_bytes(8,'little'))
            assert state_hash.hexdigest()==audit['candidate']['ordered_state_digest_bytes_sha256']
            assert seed_hash.hexdigest()==audit['candidate']['ordered_uint64_le_sha256']
            expected={(x['manifest']['split'],x['manifest']['start'],x['manifest']['stop']):x for x in inv['parts']}
            actual={(x['split'],x['start'],x['stop']):x for x in bc._inventory(COLLECTION,record)}
            assert set(actual)==set(expected) and len(actual)==60
            for key,x in expected.items():
                assert actual[key]['manifest_sha256']==x['manifest_sha256'] and actual[key]['array_sha256']==x['manifest']['sha256']
            job_map=args.root/'jobs-100k.json'
            cli=[sys.executable,str(REPO/'scripts/calibration_bank_collection.py')]
            subprocess.run([*cli,'plan','--collection',str(COLLECTION),'--train','100000','--chunk','2500','--out',str(job_map)],cwd=REPO,check=True)
            jobs=json.loads(job_map.read_text());assert [(j['split'],j['start'],j['stop']) for j in jobs['jobs']]==[('train',i,i+2500) for i in range(20000,100000,2500)]
            assert len(jobs['jobs'])==32 and jobs['collection_sha256']==inv['collection_sha256']
            report.update(status='completed_metadata_only',original_parts_checked=60,original_train_surfaces=20000,fixed_validation_surfaces=5000,fixed_test_surfaces=5000,new_train_surfaces=80000,planned_shards=32,job_map_sha256=sha(job_map),ordered_new_state_digest=state_hash.hexdigest(),ordered_new_seed_digest=seed_hash.hexdigest(),reserved_test_outcomes_evaluated=False)
        report['seconds']=time.perf_counter()-started;save();print(json.dumps(report),flush=True)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error),seconds=time.perf_counter()-started);save();raise
if __name__=='__main__':main()
