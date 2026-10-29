"""Frozen bank preflight/final integrity; never trains or evaluates a posterior."""
import argparse
import hashlib
import json
import signal
import subprocess
import sys
import time
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, x):
    temporary = p.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(x, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(p)


def audit_appended(config, history, bank, bc, np):
    def digest(x):
        return hashlib.sha256(bank._json(x).encode()).hexdigest()

    def rng_digest(x):
        return bytes.fromhex(digest(np.random.PCG64(x).state))

    historical_states = {bytes.fromhex(x) for x in history['explicit_pcg64_states']}
    historical_seeds = set(history['explicit_seeds'])
    for seed in history['explicit_seeds']:
        historical_states.add(rng_digest(seed))
        for b in range(3):
            for r in range(7):
                seq = np.random.SeedSequence([seed, b, r])
                historical_states.add(rng_digest(seq))
                historical_seeds.add(int(seq.generate_state(1, dtype=np.uint64)[0]))
    examples = 0
    for c in history['configurations']:
        task, root = c['task'], c['root_seed']
        tid, task_sha = task['task_id'], digest(task)
        words = [int(task_sha[i:i + 8], 16) for i in range(0, 64, 8)]
        if root == 271300 and c['splits'] != {'train': 20000, 'validation': 5000, 'test': 7000}:
            raise ValueError('Historical known collection unexpectedly reaches appended IDs')
        for split, n in c['splits'].items():
            code = bank.SPLITS[split]
            for i in range(n):
                if tid in ('hkde_calibration_v1', 'rough_calibration_v1', 'rough_joint_calibration_v1') or tid.endswith('_option_panel_v1'):
                    identity = [0x4F5054, root, *words, code, i]
                    for purpose in ((0, 2, 3) if 'calibration' in tid else (0, 2)):
                        historical_states.add(rng_digest(np.random.SeedSequence([*identity, purpose])))
                    seed = int(np.random.SeedSequence([*identity, 1]).generate_state(1, dtype=np.uint64)[0])
                    historical_seeds.add(seed)
                    if tid.startswith('rough_') and 'calibration' in tid:
                        for b in range(len(task['numerics']['blocks'])):
                            for r in range(len(task['numerics']['paths_ladder'])):
                                seq = np.random.SeedSequence([seed, b, r])
                                historical_seeds.add(int(seq.generate_state(1, dtype=np.uint64)[0]))
                                historical_states.add(rng_digest(seq))
                    elif tid.endswith('_option_panel_v1'):
                        historical_states.add(rng_digest(seed))
                elif tid == 'gbm_drift_known_vol_bank_v1':
                    for purpose in (0, 1):
                        historical_states.add(rng_digest(np.random.SeedSequence([root, *words, code, i, purpose])))
                elif tid == 'svj_daily_parameter_state_v1':
                    for purpose in range(5):
                        historical_states.add(rng_digest(np.random.SeedSequence([0x53564A31, root & 0xffffffff, root >> 32, *words, code, i & 0xffffffff, i >> 32, purpose])))
                elif tid == 'heston_theta_state_v1':
                    identity = [0x48535431, root & 0xffffffff, root >> 32, *words, code]
                    historical_states.add(rng_digest(np.random.SeedSequence([*identity, 0, i & 0xffffffff, i >> 32])))
                    kernel = int(np.random.SeedSequence([*identity, 1]).generate_state(1, dtype=np.uint64)[0])
                    historical_seeds.add(kernel)
                    for purpose in range(3):
                        historical_states.add(rng_digest(np.random.SeedSequence([0x48535631, kernel & 0xffffffff, kernel >> 32, i & 0xffffffff, i >> 32, 0, 0, purpose])))
                else:
                    raise ValueError('Unsupported historical protocol ' + tid)
                examples += 1
        print(json.dumps({'audited_historical_root': root, 'task': tid}), flush=True)
    states, seeds = set(), set()
    prefix = bc._seed_prefix(config)
    for i in range(20000, 50000):
        identity = [*prefix, bank.SPLITS['train'], i]
        streams = [np.random.SeedSequence([*identity, p]) for p in (0, 2, 3)]
        base = bc._pricing_seed_from_prefix(prefix, bank.SPLITS['train'], i)
        values = [base]
        for b in range(3):
            for r in range(len(config['task']['numerics']['paths_ladder'])):
                seq = np.random.SeedSequence([base, b, r])
                values.append(int(seq.generate_state(1, dtype=np.uint64)[0]))
                streams.append(seq)
        for seq in streams:
            value = rng_digest(seq)
            if value in states or value in historical_states:
                raise ValueError('Initialized-state collision at train:' + str(i))
            states.add(value)
        for value in values:
            if value in seeds or value in historical_seeds:
                raise ValueError('Reduced pricing-seed collision at train:' + str(i))
            seeds.add(value)
    return {'passed': True, 'new_train_range': [20000, 50000], 'historical_declared_examples': examples,
            'historical_initialized_states': len(historical_states), 'historical_reduced_seeds': len(historical_seeds),
            'candidate_initialized_states': len(states), 'candidate_reduced_seeds': len(seeds),
            'state_intersections': 0, 'seed_intersections': 0,
            'scope': 'Finite recorded seed/state identities, including planned joint rows; not proof of mathematical stream independence.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('preflight', 'final'))
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--phase', choices=('known', 'joint'), required=True)
    a = p.parse_args()
    start = time.perf_counter()
    plan = json.loads(a.plan.read_text()); phase = plan['phases'][a.phase]
    root, repo, collection = (Path(phase[k]) for k in ('root', 'repo', 'collection'))
    output = root / (a.stage + '.json')
    if output.exists():
        raise FileExistsError(output)
    def stop(signum, frame):
        raise TimeoutError('Stage deadline; no automatic retries')
    signal.signal(signal.SIGALRM, stop); signal.signal(signal.SIGTERM, stop)
    signal.alarm(850 if a.stage == 'preflight' else 3500)
    sys.path.insert(0, str(repo / 'src'))
    import numpy as np
    from sbi4finance.calibration import bank, bank_collection as bc, tasks
    report = {'format': 'rough_bank_delivery_stage_v1', 'status': 'running', 'phase': a.phase,
              'stage': a.stage, 'plan_sha256': sha(a.plan), 'driver_sha256': sha(__file__),
              'no_financial_pricing_calls': True, 'no_posterior_evaluation': True}
    save(output, report)
    try:
        if subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != phase['revision']:
            raise ValueError('Wrong frozen source revision')
        if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True):
            raise ValueError('Frozen source modified')
        _, record = bc._read_collection(collection, current=True)
        if record['collection_id'] != phase['collection_id'] or sha(collection / 'collection.json') != phase['collection_sha256']:
            raise ValueError('Collection identity changed')
        config = bc._target(record, phase['train_target'])
        if config['root_seed'] != phase['root_seed'] or config['splits'] != phase['splits']:
            raise ValueError('Task/root/fixed holdouts changed')
        report.update(source_revision=phase['revision'], source_sha256=bank.source_hashes(config['task']),
                      runtime=bank.versions(), collection_id=record['collection_id'], task_fingerprint=bank.task_fingerprint(config))
        parts = bc._inventory(collection, record, full=True)
        inventory = [{'split': q['split'], 'start': q['start'], 'stop': q['stop'],
                      'path': str(q['path']), 'manifest_sha256': q['manifest_sha256'],
                      'array_sha256': q['array_sha256']} for q in parts]
        if a.stage == 'preflight':
            if len(parts) != phase['preserved_parts']:
                raise ValueError('Unexpected existing collection coverage')
            save(root / 'preserved-parts.json', inventory)
            if a.phase == 'known':
                history = json.loads(Path(plan['history']).read_text())
                if sha(plan['history']) != plan['history_sha256']:
                    raise ValueError('Retained history changed')
                report['appended_seed_audit'] = audit_appended(config, history, bank, bc, np)
                subprocess.run([sys.executable, str(repo / 'scripts/calibration_bank_collection.py'), 'plan',
                                '--collection', str(collection), '--train', '50000', '--chunk', '25',
                                '--out', str(root / 'jobs.json')], check=True, timeout=180)
                expected = [('train', x, x + 25) for x in range(20000, 50000, 25)]
                jobs = json.loads((root / 'jobs.json').read_text())
                if [(q['split'], q['start'], q['stop']) for q in jobs['jobs']] != expected:
                    raise ValueError('Append map changed IDs')
            else:
                original = Path(phase['original_map'])
                if sha(original) != phase['original_map_sha256']:
                    raise ValueError('Original joint map changed')
                if (sha(phase['original_seed_audit']) != phase['original_seed_audit_sha256']
                        or not json.loads(Path(phase['original_seed_audit']).read_text())['passed']):
                    raise ValueError('Original joint seed audit changed or failed')
                (root / 'jobs.json').write_bytes(original.read_bytes())
                jobs = json.loads((root / 'jobs.json').read_text())
                expected = {(q['split'], q['start'], q['stop']) for q in jobs['jobs'][:397]}
                if {(q['split'], q['start'], q['stop']) for q in parts} != expected:
                    raise ValueError('Preserved joint ranges are not original0:397')
                report['existing_full_bank_seed_audit'] = bc._audit_seeds(config)
                report['retained_original_seed_audit_sha256'] = phase['original_seed_audit_sha256']
            report.update(status='passed_preflight', preserved_parts=len(parts), preserved_inventory_sha256=sha(root / 'preserved-parts.json'),
                          job_map_sha256=sha(root / 'jobs.json'), map_tasks=len(jobs['jobs']),
                          append_worker_indices=phase['worker_indices'], test_use='Integrity only, no model/test outcomes.')
        else:
            pre = json.loads((root / 'preflight.json').read_text())
            if pre['status'] != 'passed_preflight' or sha(root / 'jobs.json') != pre['job_map_sha256']:
                raise ValueError('Preflight/map differs')
            previous = json.loads((root / 'preserved-parts.json').read_text())
            by_key = {(q['split'], q['start'], q['stop']): q for q in inventory}
            for q in previous:
                if by_key[(q['split'], q['start'], q['stop'])] != q:
                    raise ValueError('Existing immutable part changed')
            subprocess.run([sys.executable, str(repo / 'scripts/calibration_bank_collection.py'), 'freeze',
                            '--collection', str(collection), '--train', str(phase['train_target']),
                            '--out', phase['snapshot']], check=True, timeout=2400)
            prefix, bounds = bc._seed_prefix(config), np.asarray(config['task']['prior_bounds'])
            sigma = tasks.sigma(config['task']); exceptions = []; counts = {s: 0 for s in bank.SPLITS}
            for q in parts:
                with np.load(q['path'] / 'arrays.npz', allow_pickle=False) as z:
                    arrays = {k: z[k] for k in ('parameters', 'context_values', 'context_parameters', 'reference_prices',
                                               'observations', 'pricing_standard_errors', 'pricing_paths')}
                for row, i in enumerate(range(q['start'], q['stop'])):
                    identity = [*prefix, bank.SPLITS[q['split']], i]
                    truth = np.random.default_rng(np.random.SeedSequence([*identity, 0])).uniform(bounds[:, 0], bounds[:, 1])
                    levels, coefficients = (tasks.derived_context(config['task'], truth)
                                            if tasks.is_joint_curve_task(config['task']) else
                                            tasks.draw_context(config['task'], np.random.default_rng(np.random.SeedSequence([*identity, 3]))))
                    quotes = arrays['reference_prices'][row] + sigma * np.random.default_rng(np.random.SeedSequence([*identity, 2])).standard_normal(len(sigma))
                    if not (np.array_equal(truth, arrays['parameters'][row]) and np.array_equal(levels, arrays['context_values'][row])
                            and np.array_equal(coefficients, arrays['context_parameters'][row]) and np.array_equal(quotes, arrays['observations'][row, :, 0])):
                        raise ValueError('Prior/context/noise replay differs')
                ratios = arrays['pricing_standard_errors'] / sigma
                if np.any(ratios > .5):
                    raise ValueError('Hard numerical precision allowance exceeded')
                ladder = np.asarray(config['task']['numerics']['paths_ladder'])
                if not np.isin(arrays['pricing_paths'], ladder).all():
                    raise ValueError('Unrecorded pricing path count')
                maturity = np.asarray([c['maturity'] for c in config['task']['context']['contracts']])
                previous_horizon = 0.
                for block_index, block in enumerate(config['task']['numerics']['blocks']):
                    group = (maturity > previous_horizon + 1e-12) & (maturity <= block['horizon'] + 1e-12)
                    over = np.any(ratios[:, group] > .25, axis=1)
                    if not np.all(arrays['pricing_paths'][over, block_index] == ladder[-1]):
                        raise ValueError('Nominal precision exception before final rung')
                    previous_horizon = block['horizon']
                for row in np.flatnonzero(np.max(ratios, axis=1) > .25):
                    exceptions.append({'example_id': f"{q['split']}:{q['start'] + int(row)}", 'max_mcse_over_quote_sigma': float(ratios[row].max())})
                counts[q['split']] += q['stop'] - q['start']
            if counts != config['splits']:
                raise ValueError('Incomplete snapshot coverage')
            report.update(status='completed_checked_snapshot', snapshot=phase['snapshot'],
                          snapshot_manifest_sha256=sha(Path(phase['snapshot']) / 'manifest.json'), split_counts=counts,
                          preserved_parts_unchanged=len(previous), verified_parts=len(parts), exact_prior_context_noise_replay=True,
                          empirical_precision_exceptions=exceptions, numerical_scope=phase['numerical_scope'])
        report['elapsed_seconds'] = time.perf_counter() - start
        save(output, report)
        print(json.dumps({'phase': a.phase, 'stage': a.stage, 'status': report['status'], 'elapsed_seconds': report['elapsed_seconds']}), flush=True)
    except BaseException as e:
        report.update(status='failed', error_type=type(e).__name__, error=str(e), elapsed_seconds=time.perf_counter() - start)
        save(output, report)
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
