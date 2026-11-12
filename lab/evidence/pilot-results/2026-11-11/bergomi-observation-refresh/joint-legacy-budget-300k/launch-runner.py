"""Deploy the reviewed overnight source overlay and queue one packed array once.

Invoked on Oscar with overlay path/hash, approved plan hash, producer hash and CLI
hash. Existing financial sources are hard-linked read-only; only new source names
may be extracted. A campaign claim and per-array records prevent blind retries.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import socket
import subprocess
import sys
import tarfile
import time

sys.dont_write_bytecode = True
S = Path('/oscar/scratch/afengler/sbi4finance-rough-main-full-20261111')
R = S / 'repo'
P = S / 'preparation'
PY = Path('/oscar/scratch/afengler/sbi4finance-full-core-20261026-wy2_j3o4/repo-conditioned-reference/.pixi/envs/core-reference/bin/python')
overlay, overlay_sha, plan_sha, producer_sha, cli_sha = sys.argv[1:]
O = Path(overlay)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    return {'path': str(path), 'sha256': digest(path)}


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def cmd(args):
    result = subprocess.run(args, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError({'command': args, 'code': result.returncode,
                            'stdout': result.stdout, 'stderr': result.stderr})
    return result.stdout


if (S / 'submission-claim.json').exists():
    raise ValueError('campaign claimed; inspect durable records before any further action')
write(S / 'source-stage.json', {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'base_source_root': '/oscar/scratch/afengler/sbi4finance-rough-main-legacy-20261111/repo',
    'base_submission': bind(Path('/oscar/scratch/afengler/sbi4finance-rough-main-legacy-20261111/submission.json')),
    'copy_method': 'read-only hardlinks; overlay may only add uniquely named files; bytecode writing disabled',
})
if digest(O) != overlay_sha:
    raise ValueError('overlay hash differs')
with tarfile.open(O) as archive:
    members = archive.getmembers()
    if any(m.name.startswith('/') or '..' in Path(m.name).parts or not m.isfile()
           for m in members):
        raise ValueError('unsafe overlay')
    if any(m.name.startswith('repo/') and (S / m.name).exists() for m in members):
        raise ValueError('overlay may only add new source names')
    archive.extractall(S, filter='data')

plan_path = P / 'plan.json'
review_path = P / 'pilot-readout.json'
if digest(plan_path) != plan_sha:
    raise ValueError('root-approved exact plan differs')
producer = R / 'lab/src/sbi4finance/calibration/rough_actual_layout_overnight_bank.py'
cli = R / 'lab/scripts/rough_actual_layout_overnight_bank.py'
if digest(producer) != producer_sha or digest(cli) != cli_sha:
    raise ValueError('reviewed producer or CLI differs')
sys.path[:0] = [str(R / 'lab'), str(R / 'lab/src'), str(R / 'tutorial/src')]
from sbi4finance.calibration import rough_actual_layout_overnight_bank as b

plan = b.pilot.read(plan_path)
b.validate(plan)
limits = plan['limits']
if (len(plan['jobs']) != 5980 or limits['concurrent_workers'] != 192 or
        limits['memory_gib_per_worker'] != 8 or
        limits['shard_wall_seconds'] != 43200 or
        len(plan['allocation_groups']) != 748 or
        limits['workers_per_allocation'] != 8 or
        limits['cpus_per_allocation'] != 8 or
        limits['concurrent_allocations'] != 24 or
        limits['memory_gib_per_allocation'] != 64):
    raise ValueError('reviewed whole-campaign scheduler bounds differ')

admission = {
    'format': b.ADMISSION,
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'bank_plan_sha256': digest(plan_path),
    'generation_authorized': True,
    'mode': 'candidate_only',
    'label_policy_authorized': True,
    'pilot_execution_reviewed': True,
    'pilot_numerical_review_passed': False,
    'scientific_acceptance': False,
    'numerical_policy': plan['numerical_policy'],
    'numerical_review': bind(review_path),
    'direct_user_request': 'Queue the full300k joint-c training target overnight using the restored original computational settings; preserve partial results and allow later inspection/training.',
    'heston_diagnostics_passed': False,
    'supersedes_previous_Heston_success_execution_gate': True,
    'root_reviewed_exact_plan_sha256': plan_sha,
    'scope': 'Remaining299000 joint training rows1000..299999, all5980 packed shards queued now; first1000 remain in8822042. Original engineering noise reference controls pricing budget only; observation law remains P+c*h*Z. No scientific acceptance or continuous-grid precision certification.',
}
admission_path = P / 'admission.json'
write(admission_path, admission)
b.validate_admission(plan_path, plan, admission_path)

env = os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE='1',
           PYTHONPATH=':'.join(str(R / name) for name in ['lab', 'lab/src', 'tutorial/src']),
           OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           NUMEXPR_NUM_THREADS='1', MPLCONFIGDIR=str(S / 'preflight/matplotlib'))
checked = subprocess.run([str(PY), str(cli), 'check', '--plan', str(plan_path)],
                         cwd='/tmp', env=env, text=True, capture_output=True)
(S / 'preflight/source-check.stdout.txt').write_text(checked.stdout)
(S / 'preflight/source-check.stderr.txt').write_text(checked.stderr)
if checked.returncode:
    raise ValueError('source/plan preflight failed; no jobs submitted')

qos = cmd(['sacctmgr', '-n', '-P', 'show', 'qos', 'where', 'name=carney-condo2',
           'format=Name,MaxTRESPU,MaxTRESPA,GrpTRES,MaxJobsPU,MaxJobsPA,MaxSubmitPU,MaxSubmitPA,MaxWall'])
config = cmd(['scontrol', 'show', 'config'])
partition = cmd(['scontrol', 'show', 'partition', 'batch'])
assoc = cmd(['sacctmgr', '-n', '-P', 'show', 'assoc', 'where',
             'account=carney-mjfrank-condo2',
             'format=Cluster,Account,User,Partition,ParentName,GrpJobs,GrpSubmitJobs,MaxJobs,MaxSubmitJobs,GrpTRES,MaxTRES'])
queue = cmd(['squeue', '-h', '-u', 'afengler', '-r', '-o', '%i|%q|%T|%C|%m|%D'])
shared = cmd(['squeue', '-h', '-A', 'carney-mjfrank-condo2', '-o', '%i|%q|%T|%C|%m|%D'])
resource_evidence = {
    'asof_utc': datetime.now(timezone.utc).isoformat(), 'qos': qos,
    'scheduler_config': '\n'.join(line for line in config.splitlines() if any(
        field in line for field in ['MaxArraySize', 'MaxJobCount', 'SchedulerParameters', 'AccountingStorageEnforce'])),
    'partition': partition, 'account_associations': assoc,
    'existing_user_queue': queue, 'shared_account_queue': shared,
    'requested': {'new_worker_concurrency': 192, 'memory_GiB': 1536,
                  'new_packed_jobs': 5980, 'new_physical_rows': 299000,
                  'array_count': 1, 'packed_allocations': 748,
                  'CPUs_per_allocation': 8, 'memory_GiB_per_allocation': 64,
                  'allocation_concurrency': 24, 'worker_wall_seconds': 43200},
    'existing_first1000': '8822042; remains active; QoS208CPU/2500G enforces aggregate overlap',
}
write(S / 'preflight/resources.json', resource_evidence)
if 'carney-condo2|cpu=208,mem=2500G' not in qos:
    raise ValueError('user entitlement changed; inspect before submitting')
if 'max_array_tasks=5000' not in config or 'MaxTime=UNLIMITED' not in partition:
    raise ValueError('array or wall limits differ from reviewed scheduling plan')

script = '''#!/bin/bash
#SBATCH --job-name=rough-main-jc-full
#SBATCH --account=carney-mjfrank-condo2
#SBATCH --partition=batch
#SBATCH --qos=carney-condo2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROUGH_REPO_ROOT/lab:$ROUGH_REPO_ROOT/lab/src:$ROUGH_REPO_ROOT/tutorial/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rough-bank-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
exec "$ROUGH_PYTHON" "$ROUGH_REPO_ROOT/lab/scripts/rough_actual_layout_overnight_bank.py" allocation --plan "$ROUGH_BANK_PLAN" --admission "$ROUGH_BANK_ADMISSION" --index "$SLURM_ARRAY_TASK_ID" --out "$ROUGH_PAYLOAD_ROOT"
'''
script_path = P / 'bank.sbatch'
with script_path.open('x') as stream:
    stream.write(script)
arrays = [{'offset': 0, 'count': 748, 'throttle': 24,
           'packed_shards_per_allocation': 8, 'last_allocation_shards': 4}]
record = {
    'format': 'rough_actual_layout_joint_300k_submission_v1',
    'created_utc': datetime.now(timezone.utc).isoformat(), 'host': socket.gethostname(),
    'stage_root': str(S), 'source_overlay': bind(O), 'plan': bind(plan_path),
    'admission': bind(admission_path), 'script': bind(script_path),
    'source_bindings_verified': len(plan['source_bindings']),
    'preflight_exit_code': checked.returncode, 'resource_evidence': bind(S / 'preflight/resources.json'),
    'training_target_rows': 300000, 'previously_queued_rows': 1000,
    'new_training_rows': 299000, 'packed_rows_per_shard': 50,
    'new_shard_count': 5980, 'new_task_count': 748, 'new_concurrency_cap': 192,
    'allocation_concurrency': 24, 'CPUs_per_allocation': 8,
    'memory_GiB_per_allocation': 64,
    'inference_parameter_names': plan['inference_blueprint']['parameter_names'],
    'SBI_training_readiness': False, 'scientific_acceptance': False,
    'automatic_retry': False, 'arrays': arrays,
}
write(S / 'preflight/preflight.json', record)
write(S / 'submission-claim.json', {'plan': bind(plan_path), 'admission': bind(admission_path),
                                  'array_offsets': arrays, 'automatic_retry': False})
ids = []
for array in arrays:
    offset, count = array['offset'], array['count']
    export = 'ALL,ROUGH_REPO_ROOT=' + str(R) + ',ROUGH_PYTHON=' + str(PY) + \
        ',ROUGH_BANK_PLAN=' + str(plan_path) + ',ROUGH_BANK_ADMISSION=' + str(admission_path) + \
        ',ROUGH_PAYLOAD_ROOT=' + str(S / 'payload')
    command = ['sbatch', '--parsable', '--array=0-' + str(count - 1) + '%24',
               '--export=' + export, '--output=' + str(S / 'logs/%A_%a.out'),
               '--error=' + str(S / 'logs/%A_%a.err'), '--chdir=' + str(S), str(script_path)]
    claim_path = S / ('array-offset-%06d-claim.json' % offset)
    write(claim_path, {'array': array, 'command': command, 'plan': bind(plan_path)})
    result = subprocess.run(command, text=True, capture_output=True)
    outcome = dict(array, command=command, sbatch_stdout=result.stdout,
                   sbatch_stderr=result.stderr, sbatch_returncode=result.returncode)
    if result.returncode:
        outcome['status'] = 'submission_failed'
        write(S / ('array-offset-%06d-submission.json' % offset), outcome)
        raise ValueError('submission failed; previous arrays remain queued; no automatic resubmit')
    job = result.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise ValueError('unexpected job ID; inspect claim before any retry')
    outcome.update(status='submitted', job_id=job)
    write(S / ('array-offset-%06d-submission.json' % offset), outcome)
    ids.append(job)
    array.update(job_id=job, submission_record=bind(S / ('array-offset-%06d-submission.json' % offset)))
    print(json.dumps({'status': 'submitted', 'job_id': job, 'offset': offset,
                      'count': count, 'throttle': 24, 'CPUs_per_task': 8}), flush=True)
record.update(status='submitted_all', job_ids=ids, completed_utc=datetime.now(timezone.utc).isoformat())
write(S / 'submission.json', record)
print(json.dumps({'status': 'all5980_packed_tasks_queued', 'job_ids': ids,
                  'training_rows_covered_with_first1000': 300000,
                  'record': bind(S / 'submission.json')}), flush=True)
time.sleep(3)
state = cmd(['squeue', '-h', '-j', ','.join(ids), '-r', '-o', '%i|%T|%R|%C|%m'])
(S / 'queue-after-submission.txt').write_text(state)
counts = {}
for line in state.splitlines():
    fields = line.split('|')
    counts[fields[1]] = counts.get(fields[1], 0) + 1
write(S / 'queue-summary.json', {'asof_utc': datetime.now(timezone.utc).isoformat(),
                               'job_ids': ids, 'counts': counts,
                               'queue_evidence': bind(S / 'queue-after-submission.txt')})
print(json.dumps({'job_ids': ids, 'counts': counts, 'record': bind(S / 'submission.json')}), flush=True)
