"""Read running bank journals and scheduler memory; no simulator calls."""
from pathlib import Path
from datetime import datetime,timezone
import json,sqlite3,subprocess
r=Path('/oscar/scratch/afengler/sbi4finance-rough-main-legacy-20261111');items=[]
for p in sorted((r/'payload').glob('shard-*/journal.sqlite')):
 a={'shard':p.parent.name}
 try:
  db=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True,timeout=.05)
  a.update(draws=db.execute('SELECT COUNT(*) FROM draws').fetchone()[0],attempts=db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0],rows=db.execute('SELECT status,COUNT(*) FROM rows GROUP BY status').fetchall());db.close()
 except sqlite3.Error as ex:a['read_status']=str(ex)
 items.append(a)
def cmd(args):return subprocess.run(args,text=True,capture_output=True,check=True).stdout
q=cmd(['squeue','-h','-j','8822042','-r','-o','%T']);states={k:q.splitlines().count(k) for k in set(q.splitlines())}
rss=cmd(['sstat','-a','-j','8822042','--format=JobID,MaxRSS,AveRSS','-n','-P']);rss_k=[]
for s in rss.splitlines():
 f=s.split('|')
 if len(f)>1 and f[1].endswith('K'):rss_k.append(int(f[1][:-1]))
errors=[]
for p in (r/'logs').glob('*.err'):
 text=p.read_text(errors='replace')
 if 'Traceback' in text or 'Error:' in text:errors.append({'file':p.name,'tail':text[-1000:]})
a={'format':'rough_actual_layout_legacy1000_early_running_check_v1','asof_utc':datetime.now(timezone.utc).isoformat(),'job_id':'8822042','states':states,'journals':items,'journals_created':len(items),'journals_with_prior_draw':sum(i.get('draws',0)>0 for i in items),'committed_pricing_block_attempts':sum(i.get('attempts',0) for i in items),'bootstrap_errors':errors,'live_MaxRSS_KiB':max(rss_k) if rss_k else None,'sstat':rss,'scope':'live partial evidence; no whole-row numerical quality or full memory certificate'}
(r/'early-running-check.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps({k:v for k,v in a.items() if k not in ['journals','sstat']}))
