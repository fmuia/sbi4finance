"""Reviewed one-shot slot transfer after known completion; never cancels a job."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/oscar/scratch/afengler/sbi4finance-bank-delivery-20261028')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, x):
    q = p.with_suffix('.json.tmp'); q.write_text(json.dumps(x, indent=2) + '\n'); q.replace(p)


def ready(ledger, plan):
    if ledger['status'] in ('stopped_for_review', 'stopped_at_budget_ceiling', 'stopped_at_controller_wall_limit'):
        raise RuntimeError('Generation is stopped; do not switch automatically')
    if ledger['phases']['known']['status'] != 'completed_checked_snapshot':
        return False
    final = json.loads((Path(plan['phases']['known']['root']) / 'final.json').read_text())
    return (final['status'] == 'completed_checked_snapshot'
            and final['snapshot_manifest_sha256'] == sha(Path(final['snapshot']) / 'manifest.json')
            and final['split_counts'] == {'train': 50000, 'validation': 5000, 'test': 7000})


def main():
    base_path = ROOT / 'plan-v2.json'; old = json.loads(base_path.read_text())
    path = ROOT / 'joint-slot-transfer.json'
    with path.open('x') as f:
        json.dump({'status': 'waiting_known_checked_snapshot', 'started_utc': datetime.now(timezone.utc).isoformat(),
                   'original_plan_sha256': sha(base_path), 'no_job_cancellation_or_retry': True}, f, indent=2)
    report = json.loads(path.read_text()); started = time.monotonic()
    try:
        while time.monotonic() - started < 24 * 3600:
            ledger_path = ROOT / 'execution-v2.json'; ledger = json.loads(ledger_path.read_text())
            if ledger['phases']['joint']['status'] == 'completed_checked_snapshot':
                report['status'] = 'completed_without_transfer_needed'; save(path, report); return
            if ready(ledger, old):
                pid = ledger['controller_pid']; proc = Path('/proc') / str(pid)
                command = (proc / 'cmdline').read_bytes()
                if (proc.stat().st_uid != os.getuid() or b'bank_delivery_controller_v2.py' not in command
                        or str(base_path).encode() not in command):
                    raise RuntimeError('Cannot verify the original controller on this host')
                # Pause only in its between-polls sleep, never across an sbatch.
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(ledger['last_poll_utc'])).total_seconds()
                if ('nanosleep' not in (proc / 'wchan').read_text()
                        or not 0 <= age <= 45
                        or any(j['state'] != 'submitted' for s in ledger['phases'].values() for j in s['jobs'])):
                    time.sleep(2); continue
                report.update(status='pause_attempted_between_polls', old_controller_pid=pid)
                save(path, report); os.kill(pid, signal.SIGINT)
                for _ in range(30):
                    paused = json.loads(ledger_path.read_text())
                    if paused['status'] == 'stopped_for_review' and paused['error_type'] == 'KeyboardInterrupt':
                        break
                    time.sleep(1)
                else:
                    raise RuntimeError('Controller pause not confirmed; no new controller launched')
                new = json.loads(json.dumps(old)); new['phases']['joint']['max_workers'] = 184
                new.update(resume_plan_sha256=sha(base_path), resume_execution_sha256=sha(ledger_path),
                           controller_sha256=sha(ROOT / 'bank_delivery_controller_v3.py'),
                           concurrency_amendment='Known checked; joint may borrow free slots up to global184, no numerical/resource ceiling changes.')
                assert new['phases']['joint']['cpu_hour_ceiling'] == old['phases']['joint']['cpu_hour_ceiling'] == 1500
                assert new['max_combined_bank_workers'] == old['max_combined_bank_workers'] == 184
                new_path = ROOT / 'plan-v3.json'
                with new_path.open('x') as f:
                    json.dump(new, f, indent=2); f.write('\n')
                with (ROOT / 'controller-v3.log').open('ab') as log:
                    child = subprocess.Popen([new['python'], '-u', str(ROOT / 'bank_delivery_controller_v3.py'),
                                              '--plan', str(new_path), '--resume-execution', str(ledger_path)],
                                             cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                report.update(status='launched_reviewed_concurrency_continuation', new_controller_pid=child.pid,
                              original_execution_sha256=new['resume_execution_sha256'], plan_v3_sha256=sha(new_path),
                              controller_v3_sha256=new['controller_sha256'])
                save(path, report); return
            time.sleep(60)
        raise TimeoutError('Known bank did not finish inside the one-day watch; leave original controller unchanged')
    except BaseException as error:
        report.update(status='stopped_transfer_for_review', error_type=type(error).__name__, error=str(error))
        save(path, report); raise


if __name__ == '__main__':
    main()
