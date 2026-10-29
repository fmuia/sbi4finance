"""Archive terminal comparison metadata/logs; payloads remain on Oscar."""
import base64,datetime,gzip,hashlib,json,pathlib,shlex,shutil,subprocess
DEST=pathlib.Path('/Users/afengler/Projects/proj_sbi4finance/sbi4finance-full-surface-core/docs/pilot-results/2026-10-27/backbone-runs/known-emulator-comparison')
P=pathlib.Path('/private/tmp/known-emulator-comparison-execution-20261027.json')
ROOT='/oscar/scratch/afengler/sbi4finance-rough-known-reference-20261027'
def digest(b):return hashlib.sha256(b).hexdigest()
def ssh(argv):
 r=subprocess.run(['ssh','oscar',shlex.join(argv)],capture_output=True,text=True,timeout=90)
 if r.returncode:raise RuntimeError(r.stderr)
 return r.stdout
ledger=json.loads(P.read_text());job=ledger['job_id'];assert job=='8030954'
raw=ssh(['sacct','-S2026-10-27','-j'+job,'-n','-P','--format=JobID%64,JobIDRaw%64,State%40,ElapsedRaw,AllocCPUS,ExitCode,TotalCPU,UserCPU,SystemCPU,MaxRSS,ReqMem,TimelimitRaw,Start,End,NodeList,ReqTRES%160,AllocTRES%160']).encode()
(DEST/'accounting.psv').write_bytes(raw);rows=[r.split('|') for r in raw.decode().splitlines() if r and r.split('|')[0]==job];assert len(rows)==1;row=rows[0]
assert row[2] not in ['PENDING','RUNNING','COMPLETING','CONFIGURING'] and int(row[4])==4 and 'gres/gpu=1' in row[16].split(',')
SCRIPT="""import base64,hashlib,json,pathlib,subprocess
root=pathlib.Path('/oscar/scratch/afengler/sbi4finance-rough-known-reference-20261027');out=root/'runs/comparisons';plan=json.loads((root/'manifests/known_emulator_comparison_plan_20261027.json').read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
inventory={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()}
files={str(p.relative_to(out)):base64.b64encode(p.read_bytes()).decode() for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in ['.json','.log']}
files['slurm-8030954.out']=base64.b64encode((root/'logs/comparison-8030954.out').read_bytes()).decode()
freeze={'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root/'repo').decode().strip(),'tracked_source_status':subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root/'repo').decode(),'all_frozen_hashes_match':all(sha(p)==h for p,h in plan['frozen_sha256'].items()),'frozen_input_count':len(plan['frozen_sha256']),'plan_sha256':sha(root/'manifests/known_emulator_comparison_plan_20261027.json')}
print(json.dumps({'inventory':inventory,'files':files,'freeze':freeze}))
"""
remote=json.loads(ssh(['python3','-c','exec(bytes.fromhex('+repr(SCRIPT.encode().hex())+').decode())']))
freeze=remote['freeze'];assert freeze['source_revision']==ledger['source_revision'] and freeze['tracked_source_status']=='' and freeze['all_frozen_hashes_match'] and freeze['plan_sha256']==ledger['plan_sha256']
index={}
for name,encoded in remote['files'].items():
 content=base64.b64decode(encoded)
 if name in remote['inventory']:assert remote['inventory'][name]=={'bytes':len(content),'sha256':digest(content)}
 stored=gzip.compress(content,mtime=0) if name.endswith('.log') else content;stored_name=name+'.gz' if name.endswith('.log') else name;path=DEST/stored_name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(stored)
 index[stored_name]={'source_path':ROOT+'/logs/comparison-8030954.out' if name.startswith('slurm-') else ROOT+'/runs/comparisons/'+name,'raw_bytes':len(content),'raw_sha256':digest(content),'stored_bytes':len(stored),'stored_sha256':digest(stored),'lossless_gzip':name.endswith('.log')}
(DEST/'downloaded-artifacts.json').write_text(json.dumps(index,indent=2)+'\n');(DEST/'artifact-inventory.json').write_text(json.dumps({'root':ROOT+'/runs/comparisons','payloads_transferred':False,'files':remote['inventory']},indent=2)+'\n');(DEST/'source-freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
plan=json.loads((DEST/'plan.json').read_text());run=json.loads((DEST/'run.json').read_text());cases=[]
for i in range(3):
 path=DEST/f'case-{i}/comparison.json'
 if not path.exists():continue
 report=json.loads(path.read_text());assert report['scientific_acceptance'] is False and report['independent_pricing_assessed'] is False and report['checkpoint_sha256']==plan['checkpoint_sha256'] and report['sampling_seed']==271500+i and report['surface_id']==f'validation:{i}'
 for name,h in report['reference_input_sha256'].items():assert plan['frozen_sha256'][ROOT+f'/runs/case-{i}/'+name]==h
 assert all(plan['frozen_sha256'][p]==h for p,h in report['emulator']['input_sha256'].items())
 for name,h in report['checkpoint_metadata_sha256'].items():assert plan['frozen_sha256'][plan['checkpoint']+'/'+name]==h
 cases.append({'index':i,'surface_id':report['surface_id'],'status':report['status'],'sampling_seed':report['sampling_seed'],'neural_draws_requested':report['neural_draws_requested'],'reference_issues':report['reference_issues'],'parameter_agreement':report['proposed_parameter_agreement'],'derived_parameter_comparison':report['derived_parameter_comparison'],'sampling_and_comparison_seconds':report['sampling_and_comparison_seconds'],'neural_payload':remote['inventory'].get(f'case-{i}/neural_draws.npz')})
elapsed=int(row[3]);actual={'elapsed_seconds':elapsed,'allocated_cpus':4,'allocated_cpu_seconds':elapsed*4,'allocated_cpu_hours':elapsed*4/3600,'allocated_gpus':1,'allocated_gpu_seconds':elapsed,'allocated_gpu_hours':elapsed/3600,'slurm_state':row[2],'exit_code':row[5],'accounting_sha256':digest(raw),'scope':'Count top-level allocation once, without batch/extern or queue-time double counting.'}
pool={'fit_actual_cpu_seconds':4032,'fit_actual_gpu_seconds':1008,'comparison_actual_cpu_seconds':elapsed*4,'comparison_actual_gpu_seconds':elapsed,'combined_actual_cpu_seconds':4032+elapsed*4,'combined_actual_gpu_seconds':1008+elapsed,'combined_actual_cpu_hours':(4032+elapsed*4)/3600,'combined_actual_gpu_hours':(1008+elapsed)/3600,'ceiling_cpu_seconds':10800,'ceiling_gpu_seconds':2700,'outstanding_comparison_reservation_seconds':0,'cpu_reference_788_seconds_in_separate_pool':True}
assert elapsed<=840 and actual['allocated_cpu_seconds']<=3360 and pool['combined_actual_cpu_seconds']<=10800 and pool['combined_actual_gpu_seconds']<=2700
success=row[2]=='COMPLETED' and row[5]=='0:0' and run['status']=='completed_approximate_emulator_comparisons' and len(cases)==3
summary={'format':'known_emulator_comparison_summary_v1','job_id':job,'status':'completed_approximate_emulator_comparisons' if success else 'failed_or_incomplete','source_revision':ledger['source_revision'],'actuals':actual,'shared_pool_actuals':pool,'cases':cases,'scientific_acceptance':False,'independent_pricing_assessed':False,'reserved_test_evaluated':False,'no_new_fits_prices_or_mcmc':True,'payloads_transferred':False,'scope':'Agreement is with a fixed deterministic emulator and Gaussian quote-noise likelihood. It omits bank pricing MC randomness/covariance and emulator/grid bias; not exact-model or like-for-like simulator-posterior validation.'}
(DEST/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
ledger.update(state=summary['status'],terminal_observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),terminal_run_status=run['status'],actuals=actual,shared_pool_actuals=pool)
if not success:ledger['failure']=run.get('failure',{'slurm_state':row[2],'exit_code':row[5]})
t=P.with_suffix('.tmp');t.write_text(json.dumps(ledger,indent=2)+'\n');t.replace(P);shutil.copyfile(P,DEST/'execution.json');subprocess.run(['scp',str(P),'oscar:'+ledger['remote_execution_ledger']],check=True,capture_output=True,text=True)
assert ssh(['sha256sum',ledger['remote_execution_ledger']]).split()[0]==digest(P.read_bytes())
for source,name in [('/private/tmp/observe_known_emulator_comparison_20261027.py','observer-script.py'),(__file__,'archive-script.py')]:shutil.copyfile(source,DEST/name)
progress=DEST/'monitor-progress.jsonl'
if progress.exists():(DEST/'monitor-progress.jsonl.gz').write_bytes(gzip.compress(progress.read_bytes(),mtime=0));progress.unlink()
(DEST/'downloaded-files.sha256').write_text(''.join(digest(f.read_bytes())+'  '+str(f.relative_to(DEST))+'\n' for f in sorted(DEST.rglob('*')) if f.is_file() and f.name!='downloaded-files.sha256'))
print(json.dumps({'status':summary['status'],'actuals':actual,'shared_pool_actuals':pool,'cases':[{'index':c['index'],'all_proposed_parameter_checks_passed':c['parameter_agreement']['all_checks_passed'],'parameters':{n:{k:r[k] for k in ['standardized_mean_error','width_ratio_90','mean_error_within_0_3_reference_sd','relative_width_error_within_25_percent']} for n,r in c['parameter_agreement']['parameters'].items()},'rho_eta':{k:c['derived_parameter_comparison'][k] for k in ['standardized_mean_error','width_ratio_90']}} for c in cases]}))
