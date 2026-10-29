"""Allocated integrity checks for one reviewed three-shard recovery; no pricing."""
import argparse,hashlib,json,subprocess,sys,time,fcntl
from pathlib import Path
ROOT=Path('/oscar/scratch/afengler/sbi4finance-native-bank-20261027')
REPO=ROOT/'repo';COLLECTION=ROOT/'collection';UNIT=ROOT/'recovery-378-380-20261027'
REV='c773ceb26e7273612b15c69fd63715d131778083'
COLLECTION_SHA='802a200f73bacddaa744abcfdcf2497b2d45a609c9039ed816baf1e3e9a5b12c'
MAP_SHA='a815c58087279fa7c2b5603d6bc3fee11ed92ec7965c0ac29c2827a665bbae0c'
INDICES=[378,379,380];RANGES=[(9450,9475),(9475,9500),(9500,9525)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def accounting(ids):
 raw=subprocess.check_output(['sacct','-S','2026-10-27','-j',','.join(ids),'-n','-P','--format=JobID%64,State%40,ElapsedRaw,AllocCPUS,ExitCode'],text=True)
 rows={a[0]:a for line in raw.splitlines() if len(a:=line.split('|'))>=5 and '.' not in a[0]};return raw,rows
def tree():
 return {p.name:[{'name':str(q.relative_to(p)),'kind':'directory' if q.is_dir() else 'file','bytes':q.stat().st_size} for q in sorted(p.rglob('*'))] for p in sorted((COLLECTION/'.tmp').iterdir())}
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','postcheck']);a=p.parse_args();sys.path.insert(0,str(REPO/'src'))
 from sbi4finance.calibration import bank,bank_collection as bc
 started=time.perf_counter();report_path=UNIT/(a.stage+'.json')
 if report_path.exists():raise FileExistsError('Preserve existing stage record; no automatic retry')
 report={'format':'joint_three_shard_recovery_integrity_v1','stage':a.stage,'status':'started','no_financial_pricing_calls':True,'no_neural_calls':True,'source_revision':REV,'script_sha256':sha(__file__)};write(report_path,report)
 try:
  if subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()!=REV or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise ValueError('Frozen producer changed')
  if sha(COLLECTION/'collection.json')!=COLLECTION_SHA or sha(ROOT/'jobs-20k.json')!=MAP_SHA:raise ValueError('Collection or original job map changed')
  _,record=bc._read_collection(COLLECTION,current=True);jobmap=json.loads((ROOT/'jobs-20k.json').read_text())
  if jobmap['configuration']!=record['configuration'] or jobmap['collection_id']!=record['collection_id']:raise ValueError('Task/prior/root/splits changed')
  if [(jobmap['jobs'][i]['split'],jobmap['jobs'][i]['start'],jobmap['jobs'][i]['stop']) for i in INDICES]!=[('train',first,last) for first,last in RANGES]:raise ValueError('Original recovery ID range changed')
  raw,old=accounting(['8010139','8010762','8019485']);expected={('8010139' if i<4 else '8010762' if i<204 else '8019485')+f'_{i}' for i in range(397)}
  if not expected<=set(old):raise ValueError('Incomplete old accounting')
  for name in expected:
   permitted='TIMEOUT' if name in {f'8019485_{i}' for i in INDICES} else 'COMPLETED'
   if old[name][1]!=permitted or (permitted=='COMPLETED' and old[name][4]!='0:0'):raise ValueError('Old allocation is unresolved or changed: '+name)
  (UNIT/(a.stage+'-old-accounting.psv')).write_text(raw)
  with bc._lock(COLLECTION/'.locks/mutation.lock'):
   parts=bc._inventory(COLLECTION,record)
  current={str(x['path']):{k:x[k] for k in ['split','start','stop','manifest_sha256','array_sha256']} for x in parts}
  if a.stage=='preflight':
   if len(parts)!=394 or sum(x['stop']-x['start'] for x in parts)!=9850:raise ValueError('Unexpected published baseline')
   old_tree=tree();lock_checks=[]
   if len(old_tree)!=3 or any(len(v)!=1 or v[0]['name']!='part' or v[0]['kind']!='directory' for v in old_tree.values()):raise ValueError('Unexpected retained staging; review it before recovery')
   for first,last in RANGES:
    dest=COLLECTION/'parts'/f'train-{first:010d}-{last:010d}';lock=COLLECTION/'.locks'/f'train-{first:010d}-{last:010d}.lock';before=sha(lock)
    if dest.exists():raise ValueError('Recovery range already published')
    with lock.open('r') as stream:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(stream,fcntl.LOCK_UN)
    if sha(lock)!=before:raise ValueError('Lock bytes changed')
    lock_checks.append({'range':[first,last],'unheld_lock_checked':True,'lock_sha256':before})
   write(UNIT/'preserved-parts.json',current)
   report.update(status='completed_integrity_preflight',preserved_parts=394,preserved_training_surfaces=9850,collection_id=record['collection_id'],task_fingerprint=record['task_fingerprint'],collection_sha256=COLLECTION_SHA,job_map_sha256=MAP_SHA,source_sha256=record['source_sha256'],versions=record['versions'],original_staging=old_tree,lock_checks=lock_checks,preserved_parts_sha256=sha(UNIT/'preserved-parts.json'))
  else:
   before=json.loads((UNIT/'preflight.json').read_text());baseline=json.loads((UNIT/'preserved-parts.json').read_text());ledger=json.loads((UNIT/'execution.json').read_text());workers=next(x for x in ledger['stages'] if x['role']=='workers');raw,new=accounting([workers['job_id']]);(UNIT/'postcheck-recovery-accounting.psv').write_text(raw)
   if before['status']!='completed_integrity_preflight' or sha(UNIT/'preserved-parts.json')!=before['preserved_parts_sha256']:raise ValueError('Baseline integrity declaration changed')
   for i in INDICES:
    row=new.get(workers['job_id']+f'_{i}')
    if row is None or row[1]!='COMPLETED' or row[4]!='0:0':raise ValueError('Recovery task is not successfully terminal')
   if len(current)!=397 or any(current.get(k)!=v for k,v in baseline.items()):raise ValueError('Original parts changed or recovery coverage not exact')
   expected_new={str(COLLECTION/'parts'/f'train-{first:010d}-{last:010d}') for first,last in RANGES}
   if set(current)-set(baseline)!=expected_new or tree()!=before['original_staging']:raise ValueError('Unexpected new parts or original staging changed')
   import numpy as np
   ratios=[];pricing_seeds=[];prefix=bc._seed_prefix(jobmap['configuration']);bounds=np.asarray(record['configuration']['task']['prior_bounds']);sigma=np.asarray(record['configuration']['task']['context']['quote_noise_std'])
   for path in sorted(expected_new):
    with np.load(Path(path)/'arrays.npz',allow_pickle=False) as saved:
     part=current[path];ratios.extend(np.max(saved['pricing_standard_errors']/sigma,axis=1).tolist())
     for row,i in enumerate(range(part['start'],part['stop'])):
      identity=[*prefix,bank.SPLITS['train'],i]
      prior=np.random.Generator(np.random.PCG64(np.random.SeedSequence([*identity,0]))).uniform(bounds[:,0],bounds[:,1])
      noise=np.random.Generator(np.random.PCG64(np.random.SeedSequence([*identity,2]))).standard_normal(len(sigma))
      if not np.array_equal(saved['parameters'][row],prior) or not np.array_equal(saved['observations'][row,:,0],saved['reference_prices'][row]+sigma*noise):raise ValueError('Exact frozen-runtime prior/quote reconstruction failed')
      pricing_seeds.append(bc.pricing_seed(jobmap['configuration'],'train',i))
   report.update(status='completed_integrity_postcheck',preserved_parts=394,new_original_ranges_published=3,recovered_surface_ids=75,total_published_training_surfaces=9925,total_published_parts=397,original_staging_preserved=True,new_parts={k:current[k] for k in sorted(expected_new)},maximum_reported_mcse_over_quote_sigma=max(ratios),recovered_rows_above_nominal_025=sum(x>.25 for x in ratios),recovered_rows_above_hard_05=sum(x>.5 for x in ratios),exact_prior_and_quote_noise_reconstruction=True,ordered_pricing_seed_digest_sha256=hashlib.sha256(np.asarray(pricing_seeds,dtype='<u8').tobytes()).hexdigest(),full_32000_snapshot_complete=False,reserved_test_outcomes_evaluated=False)
  report['seconds']=time.perf_counter()-started;write(report_path,report);print(json.dumps(report),flush=True)
 except BaseException as exc:report.update(status='failed',error_type=type(exc).__name__,error=str(exc),seconds=time.perf_counter()-started);write(report_path,report);raise
if __name__=='__main__':main()
