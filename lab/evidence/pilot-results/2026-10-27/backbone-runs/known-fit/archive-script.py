"""Retain metadata/logs and hash-only payload provenance for fit8030774.

This script submits no jobs, evaluates no models, and reads no reserved test outcomes.
The sole remote write is the explicitly delegated terminal execution-ledger update.
"""
import base64,datetime,gzip,hashlib,json,pathlib,shlex,shutil,subprocess

DEST=pathlib.Path('/Users/afengler/Projects/proj_sbi4finance/sbi4finance-full-surface-core/docs/pilot-results/2026-10-27/backbone-runs/known-fit')
LEDGER=pathlib.Path('/private/tmp/known-backbone-fit-execution-20261027.json')
ROOT='/oscar/scratch/afengler/sbi4finance-rough-known-fit-20261027'
SOURCE='845c458b849c472271527cccf815ea6714f02206'
DRIVER='e19aa4f1f8011ebe9249113dc2058b0cee0e2aa9a530f39307a375045ff96945'
FIELDS=['JobID%64','JobIDRaw%64','State%40','ElapsedRaw','AllocCPUS','ExitCode','TotalCPU','UserCPU','SystemCPU','MaxRSS','ReqMem','TimelimitRaw','Start','End','NodeList','ReqTRES%160','AllocTRES%160']
REMOTE="""import base64,hashlib,json,pathlib,subprocess
root=pathlib.Path('/oscar/scratch/afengler/sbi4finance-rough-known-fit-20261027');out=root/'runs/known-20k-native'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
run=json.loads((out/'run.json').read_text())
artifact={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()}
names=['run.json','checkpoint/training.json','checkpoint/task.json','pricer/run.json','pricer/checkpoint/pricer.json','pricer/checkpoint/task.json','validation/evaluation.json','environment.log','posterior.log','pricer.log','validation.log']
files={name:base64.b64encode((out/name).read_bytes()).decode() for name in names if (out/name).is_file()}
log=root/'logs/known-20k-fit-8030774.out'
if log.exists():files['slurm-8030774.out']=base64.b64encode(log.read_bytes()).decode()
freeze={'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root/'repo').decode().strip(),'tracked_source_status':subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root/'repo').decode(),'source_hashes_unchanged':all(sha(root/'repo'/n)==v for n,v in run['source_sha256'].items()),'environment_hashes_unchanged':all(sha(root/'repo'/n)==v for n,v in run['environment_sha256'].items()),'driver_sha256':sha(root/'manifests/run_known_backbone_fit_20261027.py'),'bank_manifest_sha256':sha(pathlib.Path('/oscar/scratch/afengler/sbi4finance-rough-known-backbone-20261027/snapshot-20k/manifest.json'))}
print(json.dumps({'artifacts':artifact,'files':files,'freeze':freeze}))
"""

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(b):return hashlib.sha256(b).hexdigest()
def ssh(argv):
 r=subprocess.run(['ssh','oscar',shlex.join(argv)],capture_output=True,text=True,timeout=90)
 if r.returncode:raise RuntimeError(r.stderr)
 return r.stdout

DEST.mkdir(parents=True,exist_ok=True)
ledger=json.loads(LEDGER.read_text())
assert ledger['job_id']=='8030774' and ledger['source_revision']==SOURCE and ledger['driver_sha256']==DRIVER
raw=ssh(['sacct','-S2026-10-27','-j8030774','-n','-P','--format='+','.join(FIELDS)]).encode()
(DEST/'accounting.psv').write_bytes(raw)
rows=[line.split('|') for line in raw.decode().splitlines() if line.strip()]
rows=[r for r in rows if r[0]=='8030774']
assert len(rows)==1
row=rows[0];assert row[2] not in ['PENDING','RUNNING','COMPLETING','CONFIGURING']
elapsed,cpus=int(row[3]),int(row[4]);assert cpus==4
assert 'gres/gpu=1' in row[16].split(','),'Expected one allocated GPU in terminal Slurm TRES'
remote=json.loads(ssh(['python3','-c','exec('+repr(REMOTE)+')']))
freeze=remote['freeze'];assert freeze['source_revision']==SOURCE and freeze['tracked_source_status']=='' and freeze['source_hashes_unchanged'] and freeze['environment_hashes_unchanged']
assert freeze['driver_sha256']==DRIVER and freeze['bank_manifest_sha256']==ledger['bank_manifest_sha256']
artifact_index={}
for name,encoded in remote['files'].items():
 content=base64.b64decode(encoded)
 if name in remote['artifacts']:
  assert len(content)==remote['artifacts'][name]['bytes'] and digest(content)==remote['artifacts'][name]['sha256']
 stored_name=name
 if name.endswith('.log'):
  stored_name=name+'.gz';stored=gzip.compress(content,mtime=0)
 else:stored=content
 path=DEST/stored_name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(stored)
 artifact_index[stored_name]={'source_path':str(pathlib.PurePosixPath(ROOT)/'runs/known-20k-native'/name) if not name.startswith('slurm-') else ROOT+'/logs/known-20k-fit-8030774.out','raw_bytes':len(content),'raw_sha256':digest(content),'stored_bytes':len(stored),'stored_sha256':digest(stored),'lossless_gzip':name.endswith('.log')}
(DEST/'artifact-inventory.json').write_text(json.dumps({'root':ROOT+'/runs/known-20k-native','payloads_transferred':False,'files':remote['artifacts']},indent=2)+'\n')
(DEST/'source-freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
(DEST/'downloaded-artifacts.json').write_text(json.dumps(artifact_index,indent=2)+'\n')
run=json.loads((DEST/'run.json').read_text())
success=row[2]=='COMPLETED' and row[5]=='0:0' and run['status']=='completed_development_checks'
posterior=json.loads((DEST/'checkpoint/training.json').read_text()) if (DEST/'checkpoint/training.json').exists() else None
pricer=json.loads((DEST/'pricer/checkpoint/pricer.json').read_text()) if (DEST/'pricer/checkpoint/pricer.json').exists() else None
evaluation=json.loads((DEST/'validation/evaluation.json').read_text()) if (DEST/'validation/evaluation.json').exists() else None
for name in ['checkpoint/model.keras','pricer/checkpoint/pricer.npz','validation/draws.npz']:
 if success:assert name in remote['artifacts']
if posterior:assert remote['artifacts']['checkpoint/model.keras']['sha256']==posterior['checkpoint_sha256']
if pricer:assert remote['artifacts']['pricer/checkpoint/pricer.npz']['sha256']==pricer['pricer_sha256']
if evaluation:
 assert evaluation['example_ids']==['validation:'+str(i) for i in range(2000)] and evaluation['reserved_test_evaluated'] is False
 assert evaluation['checkpoint_sha256']==posterior['checkpoint_sha256']
 assert evaluation['diagnostics_with_derived_targets']['targets']==['hurst','eta','rho','rho_eta']
actual={'elapsed_seconds':elapsed,'allocated_cpus':cpus,'allocated_cpu_seconds':elapsed*cpus,'allocated_cpu_hours':elapsed*cpus/3600,'allocated_gpus':1,'gpu_seconds':elapsed,'gpu_hours':elapsed/3600,'top_level_slurm_state':row[2],'exit_code':row[5],'accounting_sha256':digest(raw),'scope':'One top-level allocation; batch/extern rows retained without double counting. Queue waiting time is excluded.'}
assert actual['allocated_cpu_seconds']<=ledger['allocation_ceiling']['full_limit_plus_cleanup_cpu_seconds'] and elapsed<=ledger['allocation_ceiling']['full_limit_plus_cleanup_gpu_seconds']
ledger.update(state='completed_development_checks' if success else 'failed_or_incomplete',terminal_observed_utc=now(),terminal_run_status=run['status'],actuals=actual,artifact_inventory_sha256=digest((DEST/'artifact-inventory.json').read_bytes()),scientific_acceptance=False)
if not success:ledger['failure']=run.get('failure',{'slurm_state':row[2],'exit_code':row[5]})
temporary=LEDGER.with_suffix('.tmp');temporary.write_text(json.dumps(ledger,indent=2)+'\n');temporary.replace(LEDGER)
shutil.copyfile(LEDGER,DEST/'execution.json')
subprocess.run(['scp',str(LEDGER),'oscar:'+ROOT+'/execution.json'],check=True,capture_output=True,text=True)
assert ssh(['sha256sum',ROOT+'/execution.json']).split()[0]==digest(LEDGER.read_bytes())
summary={'format':'known_curve_backbone_fit_summary_v1','recorded_utc':now(),'status':ledger['state'],'job_id':'8030774','source_revision':SOURCE,'driver_sha256':DRIVER,'bank_manifest_sha256':ledger['bank_manifest_sha256'],'task_fingerprint':run['task_fingerprint'],'actuals':actual,'stages':[{k:s.get(k) for k in ['name','status','seconds','returncode','timeout_cap_seconds']} for s in run['stages']],'selected_posterior_epoch':None if posterior is None else posterior['selected_epoch'],'posterior_reload_exact':None if posterior is None else posterior['same_seed_reload_exact'],'selected_pricer_epoch':None if pricer is None else pricer['fit']['selected_epoch'],'pricer_reload_exact':None if pricer is None else pricer['same_reload_exact'],'pricer_export_max_abs_difference_quote_noise':None if pricer is None else pricer['fit']['export_max_abs_difference_quote_noise'],'pricer_validation_errors':None if pricer is None else pricer['validation_errors'],'four_target_diagnostics':None if evaluation is None else evaluation['diagnostics_with_derived_targets'],'approximation_scope':run['approximation_scope'],'scientific_acceptance':False,'reserved_test_evaluated':False,'no_new_pricing_or_mcmc':True,'no_bank_model_or_draw_payloads_transferred':True}
if pricer:
 summary['pricer_selected_keras_validation_mse_quote_noise']=pricer['fit']['best_val_mse_quote_noise']
 summary['pricer_numpy_validation_mse_standardized']=pricer['fit']['validation_mse_standardized_at_selected_epoch']
 summary['pricer_export_interpretation']='Saved NumPy-pricer reload was exact. Keras-to-NumPy predictions differed by the recorded maximum; no export-parity claim. GPU arithmetic/precision is an unconfirmed possible contribution.'
(DEST/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for source,name in [('/private/tmp/observe_known_backbone_fit_20261027.py','observer-script.py'),(__file__,'archive-script.py')]:shutil.copyfile(source,DEST/name)
for name in ['driver.py','submission-script.py']:
 expected=DRIVER if name=='driver.py' else ledger['operator_sha256']
 assert digest((DEST/name).read_bytes())==expected
progress=DEST/'monitor-progress.jsonl'
if progress.exists():(DEST/'monitor-progress.jsonl.gz').write_bytes(gzip.compress(progress.read_bytes(),mtime=0));progress.unlink()
(DEST/'downloaded-files.sha256').write_text(''.join(digest(f.read_bytes())+'  '+str(f.relative_to(DEST))+'\n' for f in sorted(DEST.rglob('*')) if f.is_file() and f.name!='downloaded-files.sha256'))
print(json.dumps({'status':summary['status'],'actuals':actual,'posterior_epoch':summary['selected_posterior_epoch'],'pricer_epoch':summary['selected_pricer_epoch'],'four_target_diagnostics':summary['four_target_diagnostics']}))
