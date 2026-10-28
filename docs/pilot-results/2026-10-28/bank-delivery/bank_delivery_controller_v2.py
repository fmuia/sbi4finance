"""One finite, durable wave loop for two existing collections; no service framework."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time


ACTIVE = {'RUNNING', 'PENDING', 'COMPLETING', 'CONFIGURING', 'SUSPENDED', 'RESIZING', 'REQUEUED', 'REQUEUE_FED', 'REQUEUE_HOLD', 'SIGNALING', 'STAGE_OUT'}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, x):
    tmp = p.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def accounting(jobs):
    if not jobs:
        return {}
    ids = [j['job_id'] for j in jobs if j.get('job_id')]
    if not ids:
        return {}
    r = subprocess.run(['sacct', '-S', '2026-10-27', '-j', ','.join(ids), '-n', '-P',
                        '--format=JobID%64,State%40,ElapsedRaw,AllocCPUS,ExitCode'],
                       capture_output=True, text=True, timeout=45, check=True)
    rows = {}
    for line in r.stdout.splitlines():
        x = line.split('|')
        if len(x) >= 5 and '.' not in x[0] and x[0]:
            rows[x[0]] = {'job_id': x[0], 'state': x[1].split()[0], 'elapsed_seconds': int(x[2]),
                          'cpus': int(x[3]), 'exit_code': x[4]}
    return rows


def task_rows(job):
    return [(job['job_id'] + ('_' + str(i) if i is not None else ''), i)
            for i in job.get('indices', [None])]


def exposure(phase, jobs, rows):
    actual = phase['prior_allocated_cpu_seconds']; outstanding = 0; workers = 0
    for job in jobs:
        if job['state'] != 'submitted':
            raise RuntimeError('Ambiguous/non-submitted attempt must be reconciled; no automatic retry')
        for key, index in task_rows(job):
            row = rows.get(key)
            if row and row['state'] == 'COMPLETED':
                if row['exit_code'] != '0:0':
                    raise RuntimeError('Completed nonzero task ' + key)
                actual += row['elapsed_seconds'] * row['cpus']
            else:
                if row and row['state'] not in ACTIVE:
                    raise RuntimeError('Terminal/unrecognized task state ' + key + ': ' + row['state'])
                outstanding += job['wall_seconds'] + 240
                workers += job['role'] == 'workers'
    # Reserve the complete checks allowance throughout generation; actual
    # metadata costs are additionally counted, a conservative small overlap.
    return {'actual_allocated_cpu_seconds': actual, 'outstanding_full_cpu_seconds': outstanding,
            'checks_reserve_cpu_seconds': 7200, 'exposure_cpu_seconds': actual + outstanding + 7200,
            'nonterminal_workers': workers}


def submit(root, plan_path, plan, ledger, name, role, indices=None):
    phase = plan['phases'][name]
    phase_jobs = ledger['phases'][name]['jobs']
    rows = accounting(phase_jobs)
    before = exposure(phase, phase_jobs, rows)
    wall = phase['worker_wall_seconds'] if role == 'workers' else (900 if role == 'preflight' else 3600)
    tasks = len(indices) if indices is not None else 1
    if before['exposure_cpu_seconds'] + tasks * (wall + 240) > phase['cpu_hour_ceiling'] * 3600:
        raise RuntimeError('Prospective allocation exceeds phase CPU-hour ceiling')
    if role == 'workers' and before['nonterminal_workers'] + tasks > phase['max_workers']:
        raise RuntimeError('Phase worker ceiling exceeded')
    if role == 'workers':
        # The two independent banks advance after their own validated gates.
        q = json.loads((Path(phase['root']) / 'preflight.json').read_text())
        if q['status'] != 'passed_preflight' or q['plan_sha256'] != plan['preflight_plan_sha256']:
            raise RuntimeError('This phase preflight must pass before pricing')
        if sha(Path(phase['root']) / 'jobs.json') != q['job_map_sha256']:
            raise RuntimeError('Frozen job map changed')
        previous = {i for j in phase_jobs for i in j.get('indices', [])}
        if previous.intersection(indices):
            raise RuntimeError('Duplicate/recovery worker indices require review')
    elif any(j['role'] == role for j in phase_jobs):
        raise RuntimeError('Metadata stage already attempted')
    env = ['env', 'PYTHONPATH=' + phase['repo'] + '/src', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
           'MKL_NUM_THREADS=1', 'NUMEXPR_NUM_THREADS=1']
    argv = ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '--cpus-per-task=1', '--mem=8G',
            '--no-requeue', '--account=carney-mjfrank-condo2', '--partition=batch',
            '--time=' + str(wall // 60), '--chdir=' + phase['repo'],
            '--job-name=sbi4f-' + name + '-' + role,
            '--output=' + str(Path(phase['root']) / 'logs' / (role + '-%A_%a.out'))]
    if name == 'joint':
        argv.append('--exclude=node2338')
    if role == 'workers':
        argv += ['--array=' + ','.join(map(str, indices)) + '%' + str(min(phase['max_workers'], len(indices))),
                 phase['repo'] + '/cluster/calibration_collection_array.sbatch', plan['python'],
                 str(Path(phase['root']) / 'jobs.json')]
    else:
        command = env + [plan['python'], plan['stage_driver'], 'preflight' if role == 'preflight' else 'final',
                         '--plan', str(plan_path), '--phase', name]
        argv += ['--wrap=' + shlex.join(command)]
    attempt = {'role': role, 'state': 'submission_attempted', 'attempted_utc': now(), 'wall_seconds': wall,
               'argv': argv, 'prospective_budget': before}
    if indices is not None:
        attempt['indices'] = indices
    phase_jobs.append(attempt)
    save(root / 'execution-v2.json', ledger)
    try:
        result = subprocess.run(argv, cwd=phase['repo'], capture_output=True, text=True, timeout=60)
    except BaseException:
        attempt['state'] = 'unknown_submission'
        save(root / 'execution-v2.json', ledger)
        raise
    attempt.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
    identifier = result.stdout.strip().split(';')[0]
    if result.returncode or not identifier.isdigit():
        attempt['state'] = 'unknown_submission'
        save(root / 'execution-v2.json', ledger)
        raise RuntimeError('Unknown submission: reconcile before any retry')
    attempt.update(state='submitted', job_id=identifier, submitted_utc=now())
    save(root / 'execution-v2.json', ledger)
    print(json.dumps({'phase': name, 'role': role, 'job_id': identifier, 'tasks': tasks}), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--resume-execution', type=Path, required=True)
    a = p.parse_args(); root = a.plan.parent
    lock = (root / 'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads(a.plan.read_text())
    if sha(plan['stage_driver']) != plan['stage_driver_sha256'] or sha(__file__) != plan['controller_sha256']:
        raise ValueError('Frozen operator sources changed')
    path = root / 'execution-v2.json'
    if path.exists():
        raise FileExistsError('Existing execution requires explicit reconciliation, not automatic restart')
    previous = json.loads(a.resume_execution.read_text())
    if (sha(a.resume_execution) != plan['resume_execution_sha256']
            or previous['status'] != 'stopped_for_review' or previous['error_type'] != 'KeyboardInterrupt'
            or previous['plan_sha256'] != plan['preflight_plan_sha256']
            or any(len(s['jobs']) != 1 or s['jobs'][0]['role'] != 'preflight'
                   or s['jobs'][0]['state'] != 'submitted' for s in previous['phases'].values())):
        raise ValueError('Resume requires the exact manually reviewed metadata-only interruption')
    ledger = previous
    ledger.update(status='resumed_after_review', resumed_utc=now(), resume_execution_sha256=sha(a.resume_execution),
                  prior_plan_sha256=previous['plan_sha256'], plan_sha256=sha(a.plan), controller_pid=os.getpid())
    save(path, ledger)
    started = time.monotonic()
    try:
        for poll in range(1440):
            if (sha(a.plan) != ledger['plan_sha256'] or sha(plan['stage_driver']) != plan['stage_driver_sha256']
                    or sha(__file__) != plan['controller_sha256']):
                raise RuntimeError('Frozen plan/stage driver changed')
            all_rows = {n: accounting(ledger['phases'][n]['jobs']) for n in plan['phases']}
            global_workers = 0; prepared = {}
            for name, phase in plan['phases'].items():
                state = ledger['phases'][name]; rows = all_rows[name]
                state['budget'] = exposure(phase, state['jobs'], rows)
                if state['budget']['exposure_cpu_seconds'] > phase['cpu_hour_ceiling'] * 3600:
                    raise RuntimeError('Existing exposure exceeds CPU ceiling')
                global_workers += state['budget']['nonterminal_workers']
                pre_job = next(j for j in state['jobs'] if j['role'] == 'preflight')
                row = rows.get(pre_job['job_id'])
                prepared[name] = bool(row and row['state'] == 'COMPLETED' and row['exit_code'] == '0:0')
            if global_workers > 184:
                raise RuntimeError('Combined184-worker ceiling exceeded')
            for name, phase in plan['phases'].items():
                state = ledger['phases'][name]; rows = all_rows[name]; b = state['budget']
                submitted = {i for j in state['jobs'] for i in j.get('indices', [])}
                done = {i for j in state['jobs'] for key, i in task_rows(j)
                        if i is not None and rows.get(key, {}).get('state') == 'COMPLETED'
                        and rows[key]['exit_code'] == '0:0'}
                missing = [i for i in phase['worker_indices'] if i not in submitted]
                state.update(submitted_workers=len(submitted), completed_workers=len(done), missing_workers=len(missing))
                if prepared[name] and missing:
                    affordable = max(0, math.floor((phase['cpu_hour_ceiling'] * 3600 - b['exposure_cpu_seconds']) /
                                                  (phase['worker_wall_seconds'] + 240)))
                    n = min(len(missing), phase['max_workers'] - b['nonterminal_workers'],
                            184 - global_workers, affordable)
                    if n:
                        submit(root, a.plan, plan, ledger, name, 'workers', missing[:n])
                        global_workers += n
                        state['status'] = 'generation_active'
                    elif b['nonterminal_workers'] == 0 and affordable == 0:
                        state['status'] = 'stopped_at_budget_ceiling'
                if len(done) == len(phase['worker_indices']):
                    final_jobs = [j for j in state['jobs'] if j['role'] == 'final']
                    if not final_jobs:
                        submit(root, a.plan, plan, ledger, name, 'final')
                        state['status'] = 'final_integrity_submitted'
                    else:
                        row = rows.get(final_jobs[0]['job_id'])
                        if row and row['state'] == 'COMPLETED' and row['exit_code'] == '0:0':
                            report = json.loads((Path(phase['root']) / 'final.json').read_text())
                            if report['status'] != 'completed_checked_snapshot':
                                raise RuntimeError('Final snapshot check did not pass')
                            state.update(status='completed_checked_snapshot', snapshot=phase['snapshot'],
                                         snapshot_manifest_sha256=report['snapshot_manifest_sha256'])
            ledger.update(last_poll_utc=now(), poll=poll, controller_elapsed_seconds=time.monotonic() - started,
                          global_nonterminal_bank_workers=global_workers)
            if all(s['status'] in ('completed_checked_snapshot', 'stopped_at_budget_ceiling') for s in ledger['phases'].values()):
                ledger['status'] = ('completed_both_snapshots' if all(s['status'] == 'completed_checked_snapshot'
                                      for s in ledger['phases'].values()) else 'stopped_at_budget_ceiling')
                save(path, ledger); break
            save(path, ledger)
            with (root / 'progress.jsonl').open('a') as f:
                f.write(json.dumps({'utc': now(), 'poll': poll, 'phases': {n: {k: v for k, v in s.items() if k != 'jobs'}
                                                                           for n, s in ledger['phases'].items()}}) + '\n')
            time.sleep(60)
        else:
            ledger.update(status='stopped_at_controller_wall_limit', stopped_utc=now())
            save(path, ledger)
    except BaseException as e:
        ledger.update(status='stopped_for_review', stopped_utc=now(), error_type=type(e).__name__, error=str(e))
        save(path, ledger)
        raise


if __name__ == '__main__':
    main()
