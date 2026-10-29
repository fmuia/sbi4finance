"""Metadata-only collision audit; no pricing, posterior fitting or test outcomes."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
repo=Path(sys.argv[1]);root=Path(sys.argv[2]);sys.path.insert(0,str(repo/'src'))
from sbi4finance.calibration import bank
started=time.perf_counter();plan=json.loads((root/'plan.json').read_text())
def digest(stream):return hashlib.sha256(stream.generate_state(8).tobytes()).hexdigest()
candidate=set(plan['unique_stream_state_sha256'])
if len(candidate)!=48:raise ValueError('Expected48uniqueindependentblockstreams')
configs=['configs/calibration_rough_known_backbone_20k.json','configs/calibration_rough_joint_development_20k.json',
         'configs/calibration_heston_backbone_100k.json','configs/tasks/calibration_heston_bank.json',
         'configs/tasks/calibration_rough_pilot.json','configs/tasks/calibration_rough_bank.json']
counts=[];sources={}
for name in configs:
 p=repo/name;config=json.loads(p.read_text());sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 task=config['task'];fingerprint=bank.task_fingerprint(config)
 words=[int(fingerprint[i:i+8],16) for i in range(0,64,8)]
 checked=0;pricing_checked=0
 for split,count in config['splits'].items():
  for row in range(count):
   identity=[0x4F5054,config['root_seed'],*words,bank.SPLITS[split],row]
   for purpose in range(4):
    s=np.random.SeedSequence([*identity,purpose])
    if digest(s) in candidate:raise ValueError('Outerbankstreamcollision')
    checked+=1
    if purpose==1 and task['model']=='rough':
     seed=int(s.generate_state(1,dtype=np.uint64)[0])
     for b in range(len(task['numerics']['blocks'])):
      for rung in range(len(task['numerics']['paths_ladder'])):
       if digest(np.random.SeedSequence([seed,b,rung])) in candidate:raise ValueError('Pricingblockstreamcollision')
       pricing_checked+=1
 counts.append({'configuration':name,'root_seed':config['root_seed'],'surfaces':sum(config['splits'].values()),
                'outer_purpose_streams_checked':checked,'all_possible_pricing_block_rung_streams_checked':pricing_checked})
result={'status':'passed','format':'known_pricing_stream_audit_v1','plan_sha256':hashlib.sha256((root/'plan.json').read_bytes()).hexdigest(),
        'candidate_unique_block_streams':len(candidate),'intentional_reuse':'Eachcase/replicate/blockstreamsharedacrossthreepointroles; independentacrosscases/replicates/blocks.',
        'candidate_collision_count':0,'configurations':counts,'configuration_source_sha256':sources,
        'method':'Compare SHA256 of SeedSequence.generate_state(8,uint32) for every planned outerpurpose and every possible pricing-block/rung stream.',
        'scope':'Reconstructs recorded configuration candidates, including ungenerated planned rows; not proof of historical payload availability. No financial simulations or test outcomes.',
        'elapsed_seconds':time.perf_counter()-started}
(root/'stream-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
