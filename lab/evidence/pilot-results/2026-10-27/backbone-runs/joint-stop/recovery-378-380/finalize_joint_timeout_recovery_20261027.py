"""Finalize only the reviewed recovery's ledger after all allocated checks pass."""
import hashlib,json,runpy
from pathlib import Path
api=runpy.run_path('/private/tmp/submit_joint_timeout_recovery_20261027.py')
d=json.loads(api['LOCAL'].read_text());ids=[s['job_id'] for s in d['stages']];rows=api['rows_for'](ids);spent=0;stages={}
for s in d['stages']:
 entries=[]
 for i in (s['indices'] if s['indices'] is not None else [None]):
  key=s['job_id']+(f'_{i}' if i is not None else '');row=rows.get(key)
  if row is None or row[1]!='COMPLETED' or row[4]!='0:0':raise RuntimeError('Recovery is not fully terminal-successful: '+key)
  allocated=int(row[2])*int(row[3]);spent+=allocated;entries.append({'job_id':key,'state':row[1],'elapsed_seconds':int(row[2]),'allocated_cpus':int(row[3]),'allocated_cpu_seconds':allocated,'exit_code':row[4]})
 s.update(state='completed',terminal_accounting=entries);stages[s['role']]=entries
result=api['remote'](['cat',api['UNIT']+'/postcheck.json'])
if result.returncode:raise RuntimeError('Postcheck unavailable')
check=json.loads(result.stdout)
if (check.get('status')!='completed_integrity_postcheck' or check.get('source_revision')!=d['source_revision'] or check.get('script_sha256')!=d['stage_driver_sha256'] or check.get('total_published_training_surfaces')!=9925 or check.get('preserved_parts')!=394 or check.get('recovered_surface_ids')!=75 or not check.get('original_staging_preserved') or not check.get('exact_prior_and_quote_noise_reconstruction') or check.get('recovered_rows_above_hard_05')!=0):raise RuntimeError('Final integrity declaration does not match approved recovery')
if spent>34200 or d['baseline_bank_allocated_cpu_seconds']+spent+d['final_snapshot_cpu_seconds_reserved']>1500*3600:raise RuntimeError('Recovery/bank spending ceiling exceeded')
d.update(status='completed_validated_original_three_ranges',completed_utc=api['now'](),final_accounting={'stages':stages,'unit_allocated_cpu_seconds':spent,'unit_allocated_cpu_hours':spent/3600,'unit_full_authorized_cpu_seconds':34200,'bank_actual_allocated_cpu_seconds':d['baseline_bank_allocated_cpu_seconds']+spent,'bank_actual_plus_original_final_snapshot_reserve_cpu_seconds':d['baseline_bank_allocated_cpu_seconds']+spent+d['final_snapshot_cpu_seconds_reserved'],'no_additional_recovery_or_refill_reserved':True},postcheck_summary=check)
api['save'](d)
parent=json.loads(api['PARENT'].read_text());bank=parent['bank'];bank['status']='paused_after_reviewed_recovery_9925_training_only';bank['manual_recovery'].update(final_accounting=d['final_accounting'],postcheck_summary=check)
for s in d['stages']:
 entry=next(x for x in bank['jobs'] if x.get('manual_recovery_attempt_key')==s['parent_attempt_key']);entry['terminal_accounting']=s['terminal_accounting']
tmp=api['PARENT'].with_suffix('.tmp');tmp.write_text(json.dumps(parent,indent=2)+'\n');tmp.replace(api['PARENT']);api['sync']();print(json.dumps(d['final_accounting'],indent=2),flush=True)
