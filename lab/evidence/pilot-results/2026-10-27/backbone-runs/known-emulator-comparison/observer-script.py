"""Bounded read-only monitor for the single comparison job8030954."""
import datetime,json,pathlib,shlex,subprocess,time
DEST=pathlib.Path('/Users/afengler/Projects/proj_sbi4finance/sbi4finance-full-surface-core/docs/pilot-results/2026-10-27/backbone-runs/known-emulator-comparison')
REMOTE="""import json,pathlib,subprocess
out=pathlib.Path('/oscar/scratch/afengler/sbi4finance-rough-known-reference-20261027/runs/comparisons')
d={'accounting':subprocess.check_output(['sacct','-S2026-10-27','-j8030954','-n','-P','--format=JobID%64,State%40,ElapsedRaw,AllocCPUS,ExitCode']).decode()}
if (out/'run.json').exists():
 r=json.loads((out/'run.json').read_text());d['run']={k:r.get(k) for k in ['status','seconds','failure']};d['cases']=[{k:c.get(k) for k in ['index','status','seconds','returncode','proposed_parameter_agreement_all_checks_passed']} for c in r['cases']]
print(json.dumps(d))
"""
for poll in range(28):
 command='exec(bytes.fromhex('+repr(REMOTE.encode().hex())+').decode())'
 r=subprocess.run(['ssh','oscar',shlex.join(['python3','-c',command])],capture_output=True,text=True,timeout=30)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout);d.update(poll=poll,observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 rows=[l.split('|') for l in d['accounting'].splitlines() if l.strip()];top=[r for r in rows if r[0]=='8030954'];assert len(top)==1
 d.update(state=top[0][1],elapsed_seconds=int(top[0][2]),exit_code=top[0][4]);(DEST/'monitor-state.json').write_text(json.dumps(d,indent=2)+'\n')
 with (DEST/'monitor-progress.jsonl').open('a') as f:f.write(json.dumps(d,sort_keys=True)+'\n')
 print(json.dumps(d),flush=True)
 if d['state'] not in ['PENDING','RUNNING','COMPLETING','CONFIGURING']:break
 if d.get('run',{}).get('status') in ['failed','incomplete']:print('FAILURE_REPORTED',flush=True)
 time.sleep(30)
else:raise RuntimeError('bounded observer ended; no job changes made')
