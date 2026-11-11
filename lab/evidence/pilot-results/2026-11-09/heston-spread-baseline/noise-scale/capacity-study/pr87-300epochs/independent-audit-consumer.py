"""Independent saved-data/state/result audit for the authorized300-epoch campaign.

No producer, financial model, neural runtime or reference sampler is imported.
"""
import argparse
from datetime import datetime, timezone
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
from statistics import NormalDist
import subprocess
import time

import numpy as np

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summary(array):
    return {'shape': list(array.shape), 'dtype': str(array.dtype),
            'sha256': hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}


def checked(spec):
    require(sha(spec['path']) == spec['sha256'], 'bound input changed: ' + spec['path'])
    return Path(spec['path'])


def rows_as_bytes(values):
    return set(np.ascontiguousarray(values).view(np.dtype((np.void, values.dtype.itemsize * values.shape[1]))).reshape(-1).tolist())


def verify_support(draws, bounds, saved):
    flat = draws.reshape(-1, 6)
    finite = np.isfinite(flat)
    below, above = flat < bounds[:, 0], flat > bounds[:, 1]
    physical = np.column_stack((flat[:, 0] <= 0, flat[:, 1] <= 0, flat[:, 2] <= 0,
        flat[:, 3] <= 0, (flat[:, 4] < -1) | (flat[:, 4] > 1), flat[:, 5] <= 0)) | ~finite
    for label, mask in [('outside_prior', below | above), ('below_prior', below), ('above_prior', above),
                        ('nonphysical', physical), ('nonfinite', ~finite)]:
        actual = {'sample_count': int(mask.any(axis=1).sum()), 'sample_fraction': float(mask.any(axis=1).mean()),
            'value_count': int(mask.sum()), 'per_dimension_count': mask.sum(axis=0).tolist(),
            'per_dimension_fraction': mask.mean(axis=0).tolist()}
        require(saved[label] == actual, 'raw support result differs: ' + label)
    require(saved['draws_discarded'] == saved['draws_clipped'] == saved['draws_resampled'] == 0,
            'raw draw support policy changed')


def audit_results(protocol, truth, historical_truth, native_quotes, historical_quotes, bounds, names):
    """Recompute allowed saved results independently, with no estimator import."""
    reports = []
    parent = read(checked(protocol['parent_10k_evaluation']))
    seed_contract = protocol['evaluation']
    require(seed_contract['native_sampling_seed'] == 709012 and seed_contract['historical_seed'] == 261112
            and seed_contract['historical_population_sbc_seed'] == 263112
            and seed_contract['historical_population_sampling_seed'] == '261112 + 1000 + zero_based_batch_start'
            and seed_contract['conditional_sampling_seed'] == '261112 + zero_based_case_index', 'sampling seed contract differs')
    for role, spec in protocol['evaluations'].items():
        path = Path(spec['output']) / 'evaluation.json'
        if role == 'secondary' and not path.exists():
            continue
        record = read(path)
        run_path = Path(protocol['output']) / 'run.json'
        run = read(run_path)
        epoch = 300 if role == 'primary' else run['best_validation_archive']['epoch']
        endpoint = 'primary_final' if role == 'primary' else 'secondary_best_validation'
        checkpoint_dir = Path(run['primary_checkpoint'] if role == 'primary' else run['best_validation_archive']['path'])
        require(record.get('format') == 'heston_pr87_followup_evaluation_v1'
                and record['task'] == protocol['task']['target']
                and record['protocol'] == identity(Path(protocol['task_path']).parent / 'protocol.json')
                and record['training_run'] == identity(run_path)
                and record['native_epoch'] == epoch and record['endpoint'] == endpoint
                and record['training_rows'] == protocol['contract']['train_rows']
                and record['checkpoint'] == identity(checkpoint_dir / 'checkpoint.json'),
                'saved endpoint task/protocol/checkpoint identity differs')
        require(record['status'] == 'complete' and record['scientific_acceptance'] is False
                and record['reserved_arrays_opened'] is False and record['new_reference_sampling'] is False,
                'evaluation incomplete or scope changed')
        require(record['seed'] == 261112 and record['native_seed'] == 709012, 'evaluation seeds differ')
        checked(record['checkpoint'])
        populations = {}
        for label, truths in [('native_validation', truth), ('population', historical_truth)]:
            saved = record[label]
            expected_count = 300 if label == 'native_validation' else 512
            require(saved['draws_per_case'] == expected_count
                    and saved['rows'] == (100 if label == 'native_validation' else 512), 'panel allocation differs')
            quotes = native_quotes if label == 'native_validation' else historical_quotes
            quote_hash = hashlib.sha256(np.asarray(quotes, dtype='<f8').tobytes()).hexdigest()
            truth_hash = hashlib.sha256(np.asarray(truths, dtype='<f8').tobytes()).hexdigest()
            require(quote_hash == saved['observations_sha256'] == parent[label]['observations_sha256']
                    and truth_hash == saved['truth_sha256'] == parent[label]['truth_sha256'], 'exact panel hashes differ')
            require(saved['sbc_seed'] == (711012 if label == 'native_validation' else 263112), 'SBC seed differs')
            payload = checked(saved['raw_draw_payload'])
            with np.load(payload, allow_pickle=False) as z:
                draws = z['draws'].copy()
            require(draws.shape == (len(truths), expected_count, 6), 'saved raw draw shape differs')
            verify_support(draws, bounds, saved['support'])
            finite = np.isfinite(draws).all()
            require((saved['calibration_status'] == 'defined') == finite, 'finite panel status differs')
            if finite:
                z = NormalDist().inv_cdf(.975)
                for level in [.5, .8, .9, .95]:
                    q = np.quantile(draws, [.5 - level / 2, .5 + level / 2], axis=1)
                    share = ((truths >= q[0]) & (truths <= q[1])).mean(axis=0)
                    n = len(truths)
                    centre = (share + z * z / (2 * n)) / (1 + z * z / n)
                    half = z * np.sqrt(share * (1 - share) / n + z * z / (4 * n**2)) / (1 + z * z / n)
                    require(np.array_equal(share, saved['coverage'][f'{level:g}']['coverage'])
                            and np.allclose(np.stack([centre - half, centre + half], 1),
                                            saved['coverage'][f'{level:g}']['wilson'], atol=1e-14, rtol=1e-14),
                            'coverage/Wilson result differs')
                rng = np.random.default_rng(saved['sbc_seed'])
                lower = np.sum(draws < truths[:, None, :], axis=1)
                ties = np.sum(draws == truths[:, None, :], axis=1)
                ranks = lower + rng.integers(0, ties + 1, size=ties.shape)
                unit = (ranks + rng.random(ranks.shape)) / (draws.shape[1] + 1)
                ordered = np.sort(unit, axis=0)
                grid = np.arange(1, len(truths) + 1)[:, None] / len(truths)
                ks = np.maximum(np.max(grid - ordered, axis=0), np.max(ordered - (grid - 1 / len(truths)), axis=0))
                band = float(np.sqrt(np.log(12 / .05) / (2 * len(truths))))
                require(np.array_equal(unit, saved['fractional_ranks'])
                        and np.array_equal(ks, saved['rank_dkw']['rank_ks'])
                        and band == saved['rank_dkw']['dkw_bound']
                        and (ks <= band).tolist() == saved['rank_dkw']['passed'], 'six-marginal SBC result differs')
                quantiles = np.quantile(draws, [.05, .5, .95], axis=1)
                recovery_path = checked(saved['recovery_payload'])
                with np.load(recovery_path, allow_pickle=False) as archive:
                    require(np.array_equal(archive['truth'], truths)
                            and np.array_equal(archive['quantiles'], quantiles.astype(np.float32)),
                            'saved recovery quantiles differ from every raw draw')
                rmse = np.sqrt(np.mean((quantiles[1] - truths)**2, axis=0))
                correlation = [float(np.corrcoef(truths[:, j], quantiles[1, :, j])[0, 1]) for j in range(6)]
                require(np.array_equal(rmse, saved['recovery']['median_rmse'])
                        and np.array_equal(correlation, saved['recovery']['truth_posterior_median_correlation'])
                        and np.array_equal(draws.std(axis=1, ddof=1).mean(axis=0), saved['mean_posterior_sd']),
                        'saved recovery/SD result differs')
            populations[label] = {'raw_draws': int(np.prod(draws.shape[:2])), 'independent_results_match': True,
                'observation_sha256': quote_hash, 'truth_sha256': truth_hash,
                'coverage90': saved['coverage']['0.9']['coverage'] if finite else None,
                'sbc_passed': saved['rank_dkw']['passed'] if finite else None,
                'outside_prior_sample_fraction': saved['support']['outside_prior']['sample_fraction'],
                'nonphysical_sample_fraction': saved['support']['nonphysical']['sample_fraction']}
        raw_path = checked(record['conditional_draws'])
        with np.load(raw_path, allow_pickle=False) as archive:
            conditional = {key: archive[key].copy() for key in archive.files}
        cases = {}
        for case_id, saved in record['cases'].items():
            neural = conditional[case_id]
            require(neural.shape == (4096, 6), 'conditional allocation differs')
            verify_support(neural, bounds, saved['support'])
            for prefix in ['reference_run', 'reference_diagnostics', 'reference_chain_payload']:
                require(sha(saved[prefix + '_path']) == saved[prefix + '_sha256'], 'existing reference changed')
            require(saved['reference_ready'] is True, 'reference not ready; no conditional verdict')
            with np.load(saved['reference_chain_payload_path'], allow_pickle=False) as archive:
                reference = archive['parameters'].reshape(-1, 6)
            result = saved['comparison']
            if not np.isfinite(neural).all():
                require(result['status'] == 'undefined_nonfinite_raw_draws', 'nonfinite conditional status differs')
                cases[case_id] = {'independent_results_match': True, 'raw_draws': len(neural),
                    'comparison_status': 'undefined_nonfinite_raw_draws', 'raw_draws_retained': True}
                continue
            require(result['status'] == 'defined' and np.isfinite(neural).all(), 'conditional summary undefined')
            nq, rq = np.quantile(neural, [.05, .5, .95], axis=0), np.quantile(reference, [.05, .5, .95], axis=0)
            # The historical agreement metric explicitly promotes physical draws
            # to float64; separately reported neural means retain raw precision.
            neural64 = np.asarray(neural, dtype=float)
            agreement_q = np.quantile(neural64, [.05, .5, .95], axis=0)
            err = np.abs(neural64.mean(axis=0) - reference.mean(axis=0)) / reference.std(axis=0)
            ratio = (agreement_q[2] - agreement_q[0]) / (rq[2] - rq[0])
            nc, rc = np.corrcoef(neural, rowvar=False), np.corrcoef(reference, rowvar=False)
            for key, actual in [('standardized_mean_error', err), ('width_ratio', ratio),
                    ('neural_mean', neural.mean(axis=0)), ('reference_mean', reference.mean(axis=0)),
                    ('reference_sd', reference.std(axis=0)), ('neural_quantiles_05_50_95', nq), ('reference_quantiles_05_50_95', rq)]:
                require(np.array_equal(actual, result[key]), 'conditional result differs: ' + key)
            for key, actual in [('neural_correlation', nc), ('reference_correlation', rc), ('signed_correlation_difference', nc - rc)]:
                require(np.array_equal(actual, result['dependence'][key]), 'conditional dependence result differs')
            unique = np.abs((nc - rc)[np.triu_indices(6, 1)])
            require(float(unique.max()) == result['dependence']['max_absolute_correlation_difference']
                    and float(np.median(unique)) == result['dependence']['median_absolute_correlation_difference'],
                    'conditional dependence aggregate differs')
            cases[case_id] = {'independent_results_match': True, 'raw_draws': len(neural),
                'standardized_mean_error': err.tolist(), 'width_ratio': ratio.tolist(),
                'max_absolute_correlation_difference': float(unique.max()),
                'outside_prior_sample_fraction': saved['support']['outside_prior']['sample_fraction']}
        reports.append({'role': role, 'epoch': record['native_epoch'], 'evaluation': identity(path),
            'populations': populations, 'conditional_cases': cases,
            'same_sampling_and_sbc_seed_contract': True,
            'raw_calibration_coverage_support_recovery_and_dependence_recomputed': True})
    require(len(reports) in (1, 2), 'missing primary evaluation')
    return reports


def data_audit(protocol, training, parent_record, parent_arrays, historical_truth):
    contract, task = protocol['contract'], protocol['task']
    run = read(training / 'run.json')
    require(run.get('format') == 'heston_pr87_followup_run_v1', 'new followup run identity required')
    rows = contract['train_rows']
    require(rows in (100000, 200000) and contract['epochs'] == 300
            and contract['parent_train_rows'] == 100000 and contract['validation_rows'] == 100,
            'data allocation or parent changed')
    require(run['protocol'] == contract, 'run scientific contract differs')
    require(run['protocol_sha256'] == sha(run['protocol_path']), 'run protocol hash differs')
    for spec in [*contract['source'].values(), *protocol['sources'].values(), protocol['task_file'],
                 protocol['parent_study'], protocol['parent_10k_evaluation'], protocol['prior_100k_evaluation']]:
        checked(spec)
    for key in ('run', 'training', 'data', 'whitening', 'protocol', 'final_checkpoint'):
        checked(contract['parent'][key])
    require(run['task'] == task == parent_record['task'], 'financial task differs')
    target, names = task['target'], task['target']['parameter_names']
    bounds = np.asarray(target['prior_bounds'])
    data_path = training / 'data.npz'
    require(sha(data_path) == run['data_payload_sha256'], 'current data payload hash differs')
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    require(set(arrays) == set(parent_arrays) and len(arrays) == 18
            and {k: summary(v) for k, v in arrays.items()} == run['data_arrays'], 'current array inventory/hashes differ')
    for key, array in arrays.items():
        n = rows if key.startswith('train_') else 100
        cols = 1 if key.split('_', 1)[1] in names else 54
        require(array.shape == (n, cols) and array.dtype == np.float64 and np.isfinite(array).all(),
                'current array shape/dtype/finiteness differs: ' + key)
        original = parent_arrays[key]
        prefix = array[:100000] if key.startswith('train_') else array
        require(prefix.dtype == original.dtype and np.array_equal(prefix, original),
                'fixed100k prefix or native100 validation changed: ' + key)
    parameters = {split: np.concatenate([arrays[split + '_' + name] for name in names], axis=1)
                  for split in ('train', 'validation')}
    require(np.all(parameters['train'] >= bounds[:, 0]) and np.all(parameters['train'] <= bounds[:, 1]),
            'training parameters outside physical Uniform prior')
    state = parent_record['numpy_generator_states']['after_append']
    require(contract['parent']['continuation_rng_state'] == state
            and run['numpy_generator_states']['before_append'] == state, 'continuation initial RNG differs')
    appended = rows - 100000
    rng = np.random.default_rng()
    rng.bit_generator.state = deepcopy(state)
    half_spread = np.asarray(task['records']['half_spread'])
    expected_prior = rng.uniform(bounds[:, 0], bounds[:, 1], (appended, 6))
    expected_noise = rng.standard_normal((appended, 54))
    require(np.array_equal(parameters['train'][100000:], expected_prior)
            and rng.bit_generator.state == run['numpy_generator_states']['after_append'],
            'appended prior block or final RNG state differs')
    expected_quotes = arrays['train_clean_prices'][100000:] + expected_prior[:, -1, None] * half_spread * expected_noise
    require(np.array_equal(expected_quotes, arrays['train_quotes'][100000:]), 'appended Gaussian quote/noise stream differs')
    for split in ('train', 'validation'):
        require(np.array_equal(arrays[split + '_observations'], arrays[split + '_quotes'] / half_spread),
                'native spread-scaled observations differ')
    if not appended:
        require(run['data_payload_sha256'] == parent_record['data_payload_sha256'], '100k reused data bytes differ')
    require(run['pricing_calls'] == appended and run['retained_parent_rows'] == 100000
            and run['retained_validation_rows'] == 100, 'new pricing/retained row accounting differs')
    physical = rows_as_bytes(parameters['train'][:, :5])
    validation = rows_as_bytes(parameters['validation'][:, :5])
    historical = rows_as_bytes(historical_truth[:, :5])
    require(len(physical) == rows and not (physical & validation) and not (physical & historical),
            'duplicate or holdout-overlapping physical targets')
    parent_w = Path(contract['parent']['whitening']['path'])
    require((training / 'whitening.json').read_bytes() == parent_w.read_bytes(), 'frozen parent whitening bytes differ')
    w = read(parent_w)
    x = parent_arrays['train_observations'][:10000]
    mean, covariance = x.mean(axis=0), np.cov(x, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = eigenvalues[-1] * 1e-8
    matrix = (eigenvectors / np.sqrt(np.maximum(eigenvalues, floor))) @ eigenvectors.T
    require(w['training_rows'] == 10000 and w['validation_used_for_fit'] is False and w['eigenvalue_floor'] == floor
            and np.array_equal(mean, w['mean']) and np.array_equal(covariance, w['covariance'])
            and np.array_equal(eigenvalues, w['eigenvalues']) and np.array_equal(eigenvectors, w['eigenvectors'])
            and np.array_equal(matrix, w['whitening_matrix']), 'whitening not exact original10k-only arithmetic')
    report = {'payload': identity(data_path), 'train_rows': rows, 'retained_parent_rows': 100000,
        'new_pricing_calls': appended, 'native_validation_rows': 100, 'arrays': 18,
        'all_arrays_native_float64_and_finite': True, 'all_parent_prefix_and_validation_rows_exact_including_dtype': True,
        'data_bytes_identical_to100k_parent': appended == 0, 'all_appended_prior_and_quote_noise_values_exact': True,
        'continuation_before_and_after_states_exact': True, 'appended_parameter_hashes': {
            name: summary(expected_prior[:, i:i + 1])['sha256'] for i, name in enumerate(names)},
        'appended_standard_normal_hash': summary(expected_noise)['sha256'],
        'unique_training_physical_parameter_rows': len(physical), 'overlap_native_validation': len(physical & validation),
        'overlap_historical512': len(physical & historical), 'whitening_parent_bytes_and_original10k_fit_exact': True}
    return report, run, arrays, parameters


def training_audit(protocol, training, run):
    contract = protocol['contract']
    rows, batches = contract['train_rows'], (contract['train_rows'] + 255) // 256
    updates = batches * 300
    require(run['status'] == 'complete' and run['completed_epochs'] == 300 and run['actual_optimizer_updates'] == updates
            and sha(training / 'run.json') == sha(training / 'training.json'), 'complete native training accounting differs')
    require(contract['example_presentations'] == rows * 300 and contract['expected_optimizer_updates'] == updates
            and contract['batches_per_epoch'] == batches and contract['tail_batch_rows'] == rows % 256,
            'predeclared native data exposure differs')
    require(contract['native_optimizer_budget'] == {'updates': updates, 'warmup_steps': updates // 20,
        'decay_steps': updates - updates // 20, 'initial_learning_rate': 5e-5, 'warmup_target': 5e-4, 'alpha': 0},
        'native optimizer schedule differs')
    history_path = training / 'native-epoch-history.json'
    require(sha(history_path) == run['native_epoch_history_sha256'], 'epoch history hash differs')
    history = read(history_path)
    require([row['epoch'] for row in history] == list(range(1, 301))
            and all(np.isfinite([row['loss'], row['val_loss']]).all() for row in history), 'native history differs')
    best = min(history, key=lambda row: row['val_loss'])
    require(best['epoch'] == run['best_validation_archive']['epoch'] and best['val_loss'] == run['best_validation_archive']['val_loss'],
            'strict first native validation minimum differs')
    require([entry['epoch'] for entry in run['checkpoints']] == [100, 200, 250, 300], 'archive schedule differs')
    states = []
    for entry in [*run['checkpoints'], run['best_validation_archive']]:
        directory = Path(entry['path'])
        require(sha(directory / 'checkpoint.json') == entry['checkpoint_sha256'], 'checkpoint declaration hash differs')
        cp = read(directory / 'checkpoint.json')
        require(cp.get('format') == 'heston_pr87_followup_checkpoint_v1', 'new followup checkpoint identity required')
        epoch, count, iteration, total = cp['native_epoch'], None, None, 0
        require(cp['protocol'] == contract and cp['actual_optimizer_updates'] == epoch * batches
                and cp['data_payload_sha256'] == run['data_payload_sha256'], 'checkpoint accounting/source differs')
        for filename, key in [('model.keras', 'model_sha256'), ('state.npz', 'state_sha256'), ('whitening.json', 'whitening_sha256')]:
            require(sha(directory / filename) == cp[key], 'checkpoint payload changed: ' + filename)
        require((directory / 'whitening.json').read_bytes() == (training / 'whitening.json').read_bytes(), 'checkpoint whitening changed')
        with np.load(directory / 'state.npz', allow_pickle=False) as archive:
            expected = {f'{group}_{i:04d}' for group, items in cp['state'].items() for i in range(len(items))}
            require(set(archive.files) == expected, 'full checkpoint state inventory differs')
            for group, items in cp['state'].items():
                for i, item in enumerate(items):
                    array = archive[f'{group}_{i:04d}']
                    require(summary(array) == {k: item[k] for k in ('shape', 'dtype', 'sha256')}, 'checkpoint state array differs')
                    total += 1
                    if group == 'optimizer' and item['path'].endswith('/iteration'):
                        iteration = int(array)
                    if group == 'non_trainable' and '/standardization' in item['path'] and array.shape == ():
                        count = int(array)
        require(iteration == epoch * batches and count == epoch * rows, 'actual optimizer/standardizer counters differ')
        states.append({'epoch': epoch, 'directory': directory.name, 'state_arrays_verified': total,
            'optimizer_updates': iteration, 'standardizer_presentations': count})
    return {'run': identity(training / 'run.json'), 'epochs': 300, 'batch_size': 256, 'batches_per_epoch': batches,
        'tail_batch_rows': rows % 256, 'presentations': rows * 300, 'optimizer_updates': updates,
        'best_validation_epoch': best['epoch'], 'native_optimizer_budget': contract['native_optimizer_budget'], 'checkpoints': states}


def independently_necessary(record):
    checks = []
    for panel in ('native_validation', 'population'):
        saved = record[panel]
        support = saved['support']
        finite = saved['calibration_status'] == 'defined' and support['nonfinite']['value_count'] == 0
        checks += [finite, finite and all(lo <= .9 <= hi for lo, hi in saved['coverage']['0.9']['wilson']),
            finite and all(saved['rank_dkw']['passed']), support['outside_prior']['sample_count'] == 0,
            support['nonphysical']['sample_count'] == 0]
    for case_id in ('synthetic-broad', 'synthetic-market-like', 'market'):
        entry = record['cases'][case_id]
        comparison, support = entry.get('comparison', {}), entry['support']
        defined = entry.get('reference_ready') is True and comparison.get('status') == 'defined'
        means, widths = comparison.get('standardized_mean_error', []), comparison.get('width_ratio', [])
        checks += [defined, defined and len(means) == 6 and all(x <= .3 for x in means),
            defined and len(widths) == 6 and all(.75 <= x <= 1.25 for x in widths),
            support['outside_prior']['sample_count'] == 0, support['nonphysical']['sample_count'] == 0,
            support['nonfinite']['value_count'] == 0]
    return all(checks)


def audit(evidence, *, selected_run=None, stage='final'):
    started = time.monotonic()
    evidence = Path(evidence).resolve()
    campaign_path = evidence / 'campaign-protocol.json'
    campaign = read(campaign_path)
    require(campaign['format'] == 'heston_pr87_followup_campaign_protocol_v1'
            and campaign['predeclared_before_first_fit'] is True and campaign['scientific_acceptance'] is False,
            'campaign identity changed')
    for spec in campaign['sources'].values():
        checked(spec)
    preserved_path = checked(campaign['preserved_artifacts'])
    preserved = read(preserved_path)
    for spec in preserved['files']:
        checked(spec)
        require(Path(spec['path']).stat().st_size == spec['bytes'], 'preserved file size changed')
    require(len(preserved['files']) == 565, 'preservation inventory changed')
    parent = Path(campaign['parent100k'])
    parent_record = read(parent / 'run.json')
    require(sha(parent / 'data.npz') == parent_record['data_payload_sha256'], 'parent100k data changed')
    with np.load(parent / 'data.npz', allow_pickle=False) as archive:
        parent_arrays = {key: archive[key] for key in archive.files}
    require({k: summary(v) for k, v in parent_arrays.items()} == parent_record['data_arrays'], 'parent100k arrays changed')
    first_protocol = read(checked(campaign['runs']['100k-300']['protocol']))
    input_path = checked(first_protocol['historical_population_inputs'])
    inputs = read(input_path)
    historical_path = input_path.parent / inputs['payload']
    require(sha(historical_path) == inputs['payload_sha256'], 'historical512 payload changed')
    with np.load(historical_path, allow_pickle=False) as archive:
        historical_truth, historical_quotes = archive['truth'].copy(), archive['quotes'].copy()
    report = {'format': 'heston_pr87_followup_independent_audit_v1', 'status': stage + '_verified',
        'audited_utc': datetime.now(timezone.utc).isoformat(), 'audit_source': identity(__file__),
        'source_commit': '843140c', 'current_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'campaign_protocol': identity(campaign_path), 'preserved_artifacts': {'manifest': identity(preserved_path),
            'files': len(preserved['files']), 'bytes': sum(row['bytes'] for row in preserved['files']), 'all_unchanged': True},
        'runs': {}, 'no_pricing_fit_model_load_or_posterior_query_performed': True,
        'reserved_arrays_opened': False, 'new_reference_sampling': False, 'scientific_acceptance': False,
        'scope': 'Independent saved-data/state/result audit; reused development diagnostics do not establish official quality.',
        'python': platform.python_version(), 'numpy': np.__version__}
    for name, spec in campaign['runs'].items():
        if selected_run and name != selected_run:
            continue
        protocol_path = checked(spec['protocol'])
        protocol = read(protocol_path)
        require(protocol['predeclared_before_first_campaign_fit'] is True
                and protocol['declared_utc'] == campaign['declared_utc'], 'scientific protocol was not frozen with campaign')
        training = Path(protocol['output'])
        if not training.exists():
            continue
        require(sha(protocol_path) == read(training / 'run.json')['protocol_sha256'], 'run no longer uses frozen scientific protocol')
        data_report, run, arrays, parameters = data_audit(protocol, training, parent_record, parent_arrays, historical_truth)
        item = {'protocol': identity(protocol_path), 'data': data_report}
        if stage in ('training', 'primary', 'results', 'final'):
            item['training'] = training_audit(protocol, training, run)
        if stage in ('primary', 'results', 'final'):
            if stage != 'primary' and run['best_validation_archive']['epoch'] != 300:
                require((Path(protocol['evaluations']['secondary']['output']) / 'evaluation.json').is_file(),
                        'distinct minimum-validation evaluation incomplete')
            item['saved_result_checks'] = audit_results(protocol, parameters['validation'], historical_truth,
                arrays['validation_quotes'], historical_quotes, np.asarray(protocol['task']['target']['prior_bounds']),
                protocol['task']['target']['parameter_names'])
        report['runs'][name] = item
    require(report['runs'], 'selected run data not available')
    if stage == 'final':
        execution_path = evidence / 'campaign.json'
        execution = read(execution_path)
        require(execution['status'] == 'complete' and execution['protocol'] == identity(campaign_path), 'campaign incomplete')
        proof_path = checked(execution['branch_proof'])
        proof = read(proof_path)
        first_primary = Path(first_protocol['evaluations']['primary']['output']) / 'evaluation.json'
        require(proof['primary_evaluation'] == identity(first_primary), 'branch used wrong endpoint')
        passed = independently_necessary(read(first_primary))
        require(proof['necessary_diagnostics_pass'] == passed and proof['run200k'] == (not passed)
                and proof['secondary_influenced_decision'] is False and execution['run200k'] == (not passed)
                and proof['branch_policy'] == campaign['branch_policy']
                and proof['readiness_source'] == campaign['sources']['renderer'],
                'branch proof differs from independently reproduced primary diagnostics')
        expected_runs = ['100k-300', '200k-300'] if not passed else ['100k-300']
        require(list(report['runs']) == expected_runs == list(execution['runs']), 'unplanned/missing fit or branch')
        expected_jobs = []
        for name in expected_runs:
            expected_jobs += [name + '_training', name + '_primary_evaluation']
            if report['runs'][name]['training']['best_validation_epoch'] != 300:
                expected_jobs.append(name + '_secondary_evaluation')
        require([job['id'] for job in execution['jobs']] == expected_jobs
                and all(job['status'] == 'complete' and job['returncode'] == 0
                    and job['wall_seconds_limit'] == (3600 if job['id'].endswith('_training') else 900)
                    for job in execution['jobs']), 'unexpected worker accounting, retry or budget')
        report.update(status='complete', campaign_execution=identity(execution_path),
            branch={'proof': identity(proof_path), 'primary_necessary_diagnostics_pass': passed,
                    'run200k': not passed, 'independently_reproduced': True, 'secondary_used': False},
            declared_workers_verified={'fits': len(expected_runs), 'posterior_evaluations': len(expected_jobs) - len(expected_runs),
                                       'worker_ids': expected_jobs, 'no_retry_or_unplanned_fit': True})
    report['seconds'] = time.monotonic() - started
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--run', choices=('100k-300', '200k-300'))
    parser.add_argument('--stage', choices=('data', 'training', 'primary', 'results', 'final'), default='final')
    args = parser.parse_args()
    result = audit(args.evidence, selected_run=args.run, stage=args.stage)
    if args.stage == 'final':
        require(args.run is None, 'final audit must cover the complete authorized campaign')
        archive = Path(args.evidence) / 'independent-audit-consumer.py'
        with archive.open('xb') as stream:
            stream.write(Path(__file__).read_bytes())
        require(sha(archive) == result['audit_source']['sha256'], 'archived consumer source differs from executed source')
        result['archived_audit_source'] = identity(archive)
        with (Path(args.evidence) / 'independent-audit.json').open('x') as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, allow_nan=False))

