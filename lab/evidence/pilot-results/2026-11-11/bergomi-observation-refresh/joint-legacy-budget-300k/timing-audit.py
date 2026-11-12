"""Read-only Oscar row/allocation timing audit; never invokes a pricing kernel."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json
import random
import sqlite3
import statistics
import subprocess
import time

ROOT = Path('/oscar/scratch/afengler/sbi4finance-rough-main-full-20261111')
OLD = Path('/oscar/scratch/afengler/sbi4finance-rough-main-legacy-20261111')


def command(args):
    result = subprocess.run(args, text=True, capture_output=True, check=True)
    return result.stdout


def quantiles(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    result = {'count': len(values), 'mean': statistics.mean(values), 'sum': sum(values)}
    for name, probability in [('min', 0), ('p10', .1), ('median', .5),
                              ('p90', .9), ('p95', .95), ('p99', .99), ('max', 1)]:
        position = probability * (len(values) - 1)
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        result[name] = values[lower] + (values[upper] - values[lower]) * (position - lower)
    return result


def read_journal(path):
    connection = sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True, timeout=.1)
    try:
        draws = connection.execute('SELECT COUNT(*) FROM draws').fetchone()[0]
        rows = connection.execute('SELECT id,status,summary FROM rows ORDER BY id').fetchall()
        attempts = connection.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
    finally:
        connection.close()
    counts = Counter()
    durations = []
    errors = Counter()
    for _, status, summary in rows:
        counts[status] += 1
        record = json.loads(summary)
        if status in ('complete', 'complete_candidate'):
            durations.append(float(record['seconds']))
        elif status == 'failed':
            errors[record.get('error_type', '') + ': ' + record.get('error', '')] += 1
    return draws, counts, durations, errors, attempts


start = datetime.now(timezone.utc)
submission = json.loads((ROOT / 'submission.json').read_text())
launch = datetime.fromisoformat(submission['completed_utc'])
counts = Counter()
errors = Counter()
durations = []
draws = attempts = fully_finalized_shards = 0
per_shard = []
unreadable = []
for path in sorted((ROOT / 'payload').glob('shard-*/journal.sqlite')):
    try:
        d, c, t, e, a = read_journal(path)
    except sqlite3.Error as error:
        unreadable.append({'journal': str(path), 'error': str(error)})
        continue
    counts.update(c)
    errors.update(e)
    durations.extend(t)
    draws += d
    attempts += a
    finalized = sum(c.values())
    fully_finalized_shards += finalized == d
    per_shard.append({'shard': path.parent.name, 'prior_draws': d,
                      'row_states': dict(c), 'finalized': finalized,
                      'completed_row_seconds_sum': sum(t)})

queue_text = command(['squeue', '-h', '-j', '8823501', '-r', '-o', '%i|%T|%M|%C|%S|%R'])
queue_rows = [line.split('|') for line in queue_text.splitlines()]
queue_counts = Counter(row[1] for row in queue_rows)
asof = datetime.now(timezone.utc)
elapsed = (asof - launch).total_seconds()
complete = counts['complete'] + counts['complete_candidate']
failed = counts['failed']
finalized = complete + failed
remaining = 299000 - finalized
rate = finalized / elapsed if elapsed > 0 else None
bootstrap = json.loads((ROOT / 'bootstrap-check.json').read_text())
previous_time = datetime.fromisoformat(bootstrap['asof_utc'])
previous_finalized = sum(bootstrap['row_states'].values())
delta_seconds = (asof - previous_time).total_seconds()
fast = {
    'asof_utc': asof.isoformat(), 'launch_utc': launch.isoformat(),
    'elapsed_hours': elapsed / 3600, 'allocation_queue_states': dict(queue_counts),
    'master_row_states': dict(counts), 'master_finalized_rows': finalized,
    'master_remaining_unattempted_or_in_progress': remaining,
    'known_total_successes_including_initial996': complete + 996,
    'known_total_failures_including_initial4': failed + 4,
    'journals_read': len(per_shard), 'prior_draws_ledgered': draws,
    'fully_finalized_50row_shards': fully_finalized_shards,
    'incomplete_started_shards': len(per_shard) - fully_finalized_shards,
    'pricing_block_attempts_committed': attempts,
    'finalized_rows_per_wall_hour_since_launch': rate * 3600,
    'linear_remaining_hours_at_cumulative_throughput': remaining / rate / 3600,
    'since_previous_actual_snapshot': {
        'from_utc': previous_time.isoformat(), 'elapsed_hours': delta_seconds / 3600,
        'additional_finalized_rows': finalized - previous_finalized,
        'finalized_rows_per_wall_hour': (finalized - previous_finalized) / delta_seconds * 3600,
    },
    'scope': 'fast read-only whole-journal counts; unfinished workers are censored; failed rows retained; linear forecast is provisional',
}
print(json.dumps({'fast_readout': fast}), flush=True)

accounting = command(['sacct', '-n', '-X', '-j', '8823501', '-P',
                      '--format=JobID,State,ElapsedRaw,ReqCPUS,AllocCPUS,TotalCPU,Start,End'])
account_rows = [line.split('|') for line in accounting.splitlines()]
account_states = Counter(row[1] for row in account_rows)
allocation_cpu_seconds = sum(int(row[2]) * int(row[4]) for row in account_rows
                             if row[2].isdigit() and row[4].isdigit())
old_readout_path = OLD / 'terminal-readout.json'
old_readout = json.loads(old_readout_path.read_text())
old_durations = []
old_rows = Counter()
old_errors = Counter()
for path in sorted((OLD / 'payload').glob('shard-*/journal.sqlite')):
    try:
        _, c, t, e, _ = read_journal(path)
    except sqlite3.Error:
        continue
    old_rows.update(c)
    old_errors.update(e)
    old_durations.extend(t)
old_account = command(['sacct', '-n', '-X', '-j', '8822042', '-P', '--format=State'])

# This is a statistical resampling of already measured runtimes, not new prices.
# Failed row durations were not saved: use their stated860s budget as an estimate.
empirical = old_durations + [860.] * old_rows['failed']
rng = random.Random(20261111)
sampled_group_seconds = []
for _ in range(5000):
    sampled_group_seconds.append(max(sum(rng.choices(empirical, k=50)) for _ in range(8)))
group_stats = quantiles(sampled_group_seconds)
estimated_group_work_seconds = group_stats['mean'] * 748
barrier_total_hours = estimated_group_work_seconds / 24 / 3600
ideal_total_hours = statistics.mean(empirical) * 300000 / 192 / 3600

receipt = {
    'format': 'rough_actual_layout_readonly_timing_audit_v1',
    'read_started_utc': start.isoformat(), 'read_finished_utc': datetime.now(timezone.utc).isoformat(),
    **fast, 'master_successful_row_runtime_seconds': quantiles(durations),
    'master_failure_counts': dict(errors), 'unreadable_journals': unreadable,
    'requested_cpus_per_allocation': 8, 'maximum_running_allocations': 24,
    'requested_cpu_cap': 192, 'sacct_allocation_state_counts': dict(account_states),
    'sacct_allocation_elapsed_CPU_seconds_including_idle_and_running': allocation_cpu_seconds,
    'sacct_accounting_raw': accounting, 'squeue_raw': queue_text,
    'initial1000': {
        'row_states': dict(old_rows), 'sacct_terminal_states': dict(Counter(old_account.splitlines())),
        'successful_row_runtime_seconds': quantiles(old_durations),
        'failure_counts': dict(old_errors),
        'terminal_record_sha256': hashlib.sha256(old_readout_path.read_bytes()).hexdigest(),
        'failure_duration_assumption_seconds': 860,
    },
    'runtime_bootstrap': {
        'method': '5000 seeded iid group resamples from all initial1000 request runtimes; each group=max of8workers each summing50rows; four failures represented by860s budget; no financial simulations',
        'seed': 20261111, 'initial_request_runtime_samples': len(empirical),
        'request_mean_seconds_with_failed_budget_estimate': statistics.mean(empirical),
        'resampled_eight_worker_group_seconds': group_stats,
        'ideal_full300k_hours_at192busy_CPUs': ideal_total_hours,
        'barrier_model_full_master_hours_748groups_at24slots': barrier_total_hours,
        'barrier_model_remaining_hours_minus_elapsed': max(0., barrier_total_hours - elapsed / 3600),
        'estimated_barrier_multiplier': group_stats['mean'] / (50 * statistics.mean(empirical)),
    },
    'timing_limitations': [
        'Rows store successful duration seconds, but no UTC commit/start timestamps; failed rows store no duration. Filemtime was not used as completion time.',
        'Recent-window rate compares two actual snapshots; per-minute or per-row timestamp rates are unavailable.',
        'Live complete-row durations are censored toward fast rows; all failed rows and unfinished shard counts are reported.',
        'Packed groups wait for their slowest of8workers; iid bootstrap estimates that idle cost, but queueing, hardware contention, tail dependence, and final scheduling imbalance can extend ETA.',
        'Completion ETA means every requested row attempted; failures are retained and not automatically retried. Exactly300000 successful rows are not guaranteed.',
        'Allocated elapsed CPU seconds include idle sibling CPUs; sacct TotalCPU for running groups may lag until completion.',
    ],
    'scientific_acceptance': False, 'simulation_calls': 0,
    'producer_mutations': False, 'retries': False,
}
path = ROOT / ('timing-audit-' + str(time.time_ns()) + '.json')
with path.open('x') as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps({'detailed_record': str(path),
                  'record_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'master_success_runtime_seconds': receipt['master_successful_row_runtime_seconds'],
                  'initial1000': receipt['initial1000'],
                  'runtime_bootstrap': receipt['runtime_bootstrap'],
                  'accounting_states': dict(account_states),
                  'allocated_elapsed_CPU_hours': allocation_cpu_seconds / 3600,
                  'failure_counts': dict(errors)}), flush=True)
