"""Read-only final bank/accounting archive; no data payload transfer or models."""
import datetime,gzip,hashlib,importlib.util,json,subprocess
from pathlib import Path
BASE=Path('/Users/afengler/Projects/proj_sbi4finance/sbi4finance-full-surface-core/docs/pilot-results/2026-10-27/backbone-runs/known-bank-final')
helper=Path('/private/tmp/submit_known_backbone_20261027.py');spec=importlib.util.spec_from_file_location('guarded_known',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
d=json.loads(m.PATH.read_text());root=d['target_root'];ids=[j['job_id'] for j in d['jobs']]
fields=['JobID%64','JobIDRaw%64','State%40','ElapsedRaw','AllocCPUS','ExitCode','TotalCPU','UserCPU','SystemCPU','NodeList','ReqMem','TimelimitRaw','Start','End','MaxRSS']
r=m.remote(['sacct','-S','2026-10-27','-j',','.join(ids),'-n','-P','--format='+','.join(fields)]);assert r.returncode==0,r.stderr
raw_accounting=r.stdout.encode();(BASE/'accounting-all-jobs.psv.gz').write_bytes(gzip.compress(raw_accounting,mtime=0));Path('/private/tmp/known-bank-final-accounting-20261027.psv').write_bytes(raw_accounting)
rows={}
for line in r.stdout.splitlines():
 a=line.split('|')
 if len(a)>=15 and '.' not in a[0]:rows[a[0]]=a
(BASE/'accounting-top-level.psv').write_text(''.join(line+'\n' for line in r.stdout.splitlines() if line.split('|')[0] in rows))
spent=0;generation=0;worker_indices=[]
for j in d['jobs']:
 total=0
 for i in j.get('indices',[None]):
  key=j['job_id']+(f'_{i}' if i is not None else '');a=rows[key];assert a[2]=='COMPLETED' and a[5]=='0:0' and a[4]=='1',(key,a)
  total+=int(a[3])*int(a[4])
  if i is not None:worker_indices.append(i)
 j['state']='completed';j['actual_allocated_cpu_seconds']=total;spent+=total
 if j['role'] in ('initial-workers','later-workers'):generation+=total
assert len(worker_indices)==len(set(worker_indices))==1280 and set(worker_indices)==set(range(1280))
assert spent==342891 and generation==342787
index={}
def store_remote(path,name,compress=False):
 local=Path('/private/tmp/known-bank-final-download-20261027')/name;local.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(['scp','oscar:'+root+'/'+path,str(local)],check=True)
 raw=local.read_bytes();packed=gzip.compress(raw,mtime=0) if compress else raw;stored=name+'.gz' if compress else name;(BASE/stored).write_bytes(packed)
 if compress:assert gzip.decompress(packed)==raw
 index[stored]={'source_path':root+'/'+path,'raw_sha256':hashlib.sha256(raw).hexdigest(),'stored_sha256':hashlib.sha256(packed).hexdigest(),'raw_bytes':len(raw),'stored_bytes':len(packed),'lossless_gzip':compress}
 return raw
final=json.loads(store_remote('final-check.json','final-check.json'));manifest_raw=store_remote('snapshot-20k/manifest.json','snapshot-manifest.json',True);collection_raw=store_remote('collection/collection.json','collection.json');store_remote('logs/freeze-8030755_4294967294.out','slurm-8030755.out')
assert hashlib.sha256(manifest_raw).hexdigest()==final['snapshot_manifest_sha256']=='7cf0314dc6371fd045ad6b314dd6dd605db2f3343dbccf4e35b4df9ddf9397eb'
assert final['status']=='completed_checked_snapshot_and_integrity' and final['verified_examples']==32000 and final['verified_parts']==1280
assert final['configuration_splits']=={'train':20000,'validation':5000,'test':7000}
assert final['all_prior_curve_quote_noise_reconstructed_exactly'] and final['initial100_payload_manifest_hashes_preserved'] and final['hard_cap_exceptions']==0
LOG_SCRIPT="""
import hashlib,json
from pathlib import Path
root=Path('/oscar/scratch/afengler/sbi4finance-rough-known-backbone-20261027')
logs=[]
for p in sorted((root/'logs').glob('workers-*.out')):
 raw=p.read_bytes();text=raw.decode('utf-8');assert text.encode('utf-8')==raw
 logs.append({'source_path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'text':text})
assert len(logs)==1280
print(json.dumps(logs,separators=(',',':')))
"""
r=m.remote(['python3','-c','exec('+repr(LOG_SCRIPT)+')']);assert r.returncode==0,r.stderr
logs_raw=r.stdout.encode();logs=json.loads(logs_raw);assert len(logs)==1280
(BASE/'worker-logs.json.gz').write_bytes(gzip.compress(logs_raw,mtime=0));index['worker-logs.json.gz']={'source_path':root+'/logs/workers-*.out','raw_sha256':hashlib.sha256(logs_raw).hexdigest(),'stored_sha256':hashlib.sha256((BASE/'worker-logs.json.gz').read_bytes()).hexdigest(),'raw_bytes':len(logs_raw),'lossless_gzip':True}
for log in logs:assert hashlib.sha256(log['text'].encode()).hexdigest()==log['sha256'] and len(log['text'].encode())==log['bytes']
index['accounting-all-jobs.psv.gz']={'source_command':'sacct -S2026-10-27 -j'+','.join(ids)+' -n -P --format='+','.join(fields),'raw_sha256':hashlib.sha256(raw_accounting).hexdigest(),'stored_sha256':hashlib.sha256((BASE/'accounting-all-jobs.psv.gz').read_bytes()).hexdigest(),'raw_bytes':len(raw_accounting),'lossless_gzip':True}
if (BASE/'controller-progress.jsonl').exists():
 raw=(BASE/'controller-progress.jsonl').read_bytes();(BASE/'controller-progress.jsonl.gz').write_bytes(gzip.compress(raw,mtime=0));Path('/private/tmp/known-bank-controller-progress-20261027.jsonl').write_bytes(raw);(BASE/'controller-progress.jsonl').unlink();index['controller-progress.jsonl.gz']={'raw_sha256':hashlib.sha256(raw).hexdigest(),'stored_sha256':hashlib.sha256((BASE/'controller-progress.jsonl.gz').read_bytes()).hexdigest(),'raw_bytes':len(raw),'lossless_gzip':True}
d.update(status='completed_checked_20k_5k_7k_snapshot',snapshot=root+'/snapshot-20k',snapshot_manifest_sha256=final['snapshot_manifest_sha256'],actual_allocated_cpu_seconds=spent,actual_allocated_cpu_hours=spent/3600,generation_allocated_cpu_seconds=generation,actual_gpu_seconds=0,initial100_preserved=True,all_prior_curve_quote_noise_exactly_reconstructed=True,no_failures_retries_replacements=True,reserved_test_outcomes_evaluated=False,no_neural_training_or_evaluation=True,completed_verified_utc='2026-10-27T22:03:49.344141+00:00',nominal_target_exception_cases=final['nominal_target_exception_cases'],hard_cap_exceptions=0,remaining_cpu_seconds_below_ceiling=150*3600-spent)
m.save(d);(BASE/'execution.json').write_bytes(m.PATH.read_bytes());subprocess.run(['scp',str(m.PATH),'oscar:'+root+'/execution.json'],check=True)
for source,name in [(Path('/private/tmp/advance_known_backbone_20261027.py'),'controller-script.py'),(Path('/private/tmp/finalize_known_backbone_20261027.py'),'final-integrity-script.py'),(helper,'submission-operator.py')]:
 (BASE/name).write_bytes(source.read_bytes())
summary={'format':'known_curve_backbone_final_summary_v1','status':'complete_checked_data_component','recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_revision':d['source_revision'],'task_fingerprint':final['task_fingerprint'],'same_task_as_Francesco_pilot':True,'root_seed':271300,'numerical_block_steps':[80,504,504],'snapshot':root+'/snapshot-20k','snapshot_manifest_sha256':final['snapshot_manifest_sha256'],'splits':final['configuration_splits'],'parts':1280,'unique_original_worker_indices':len(set(worker_indices)),'all_tasks_completed_exit_zero':True,'no_retries_failures_replacement_seeds':True,'initial100_payload_manifest_hashes_preserved':True,'all_prior_curve_quote_noise_reconstructed_exactly':True,'allocated_cpu_seconds_total':spent,'allocated_cpu_hours_total':spent/3600,'allocated_cpu_seconds_generation':generation,'allocated_cpu_hour_ceiling':150,'gpu_seconds':0,'nominal_target_exception_cases':final['nominal_target_exception_cases'],'hard_cap_exceptions':0,'no_bank_payloads_transferred':True,'no_model_training_or_neural_evaluation':True,'test_use':'Generated and numerically verified only; no statistical inference evaluation or model selection.','scope':'Matched-task development bank at original numerical resolution. The empirical MCSE policy and bounded posterior/product refinement requirements remain; this bank is not an accepted teaching estimator or a continuous-price bias certificate.','archived_worker_logs':len(logs)}
(BASE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(BASE/'compressed-artifacts.json').write_text(json.dumps(index,indent=2)+'\n')
(BASE/'downloaded-files.sha256').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(BASE)}\n' for p in sorted(BASE.rglob('*')) if p.is_file() and p.name!='downloaded-files.sha256'))
print(json.dumps({k:summary[k] for k in ['status','snapshot_manifest_sha256','splits','parts','allocated_cpu_hours_total','nominal_target_exception_cases','hard_cap_exceptions','archived_worker_logs']}),flush=True)
