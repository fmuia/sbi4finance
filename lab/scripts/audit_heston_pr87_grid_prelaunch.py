"""Independent saved-bank/state/raw-result audit for revised63 Heston.

No financial pricer, producer, neural runtime or MCMC sampler is imported.
Pure support and readiness checks retain the earlier independent-audit equations.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from statistics import NormalDist
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


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def truth(arrays, split, names):
    return np.column_stack([arrays[split + '_' + name].reshape(-1) for name in names])


def bank_audit(campaign):
    declaration = read(checked(campaign['bank_protocol']))
    directory = Path(declaration['output'])
    record = read(directory / 'bank.json')
    contract, task = declaration['contract'], declaration['task']
    names = task['target']['parameter_names']
    require(record['format'] == 'heston_pr87_grid_bank_v1' and record['status'] == 'complete'
            and record['protocol'] == contract and record['task'] == task
            and record['protocol_sha256'] == sha(campaign['bank_protocol']['path'])
            and sha(directory / 'bank.json') == sha(directory / 'generation.json'), 'completed source-bound bank differs')
    require(contract['train_rows'] == 300000 and contract['parent_train_rows'] == 200000
            and contract['appended_rows'] == 100000 and contract['validation_rows'] == 100
            and contract['historical_rows'] == 512 and contract['whitening_rows'] == 10000
            and record['pricing_calls'] == 300612, 'bank allocation or checked-label accounting differs')
    for spec in contract['source'].values(): checked(spec)
    parent = Path(contract['parent']['directory'])
    parent_record = read(checked(contract['parent']['run']))
    parent_arrays = load_arrays(checked(contract['parent']['data']))
    require(parent_record['protocol']['train_rows'] == 200000 and parent_record['completed_epochs'] == 300,
            'old54 parent allocation differs')
    require({key: summary(value) for key, value in parent_arrays.items()} == parent_record['data_arrays'], 'parent array hashes differ')
    historical = read(checked(contract['historical_inputs']['manifest']))
    historical_arrays = load_arrays(checked(contract['historical_inputs']['payload']))
    require(historical['payload_sha256'] == contract['historical_inputs']['payload']['sha256'], 'historical binding differs')
    data_path = directory / record['data_payload']
    require(sha(data_path) == record['data_payload_sha256'], 'new bank data hash differs')
    arrays = load_arrays(data_path)
    require({key: summary(value) for key, value in arrays.items()} == record['data_arrays'] and len(arrays) == 29,
            'new bank array inventory/hashes differ')
    truths = {split: truth(arrays, split, names) for split in ('train','validation','historical')}
    bounds, h = np.asarray(task['target']['prior_bounds']), np.asarray(task['records']['half_spread'])
    forward, strike, discount = [np.asarray(task['records'][key]) for key in ('forward','strike','discount')]
    put = np.asarray(task['records']['option_type']) == 'put'
    price_lower = discount*np.maximum(np.where(put,strike-forward,forward-strike),0)
    price_upper = discount*np.where(put,strike,forward)
    bound_tolerance = 1e-9*discount*np.maximum(forward,strike)
    old_target, new_target = deepcopy(parent_record['task']['target']), deepcopy(task['target'])
    require(new_target['pricing_controls'].pop('fourier_representation') == 'black_control_variate_v1', 'CV target tag differs')
    for value in (old_target,new_target):
        value.pop('quote_count'); value.pop('geometry_and_spreads_sha256')
    require(old_target == new_target, 'financial prior/noise settings changed')
    require(np.array_equal(truths['train'][:200000], truth(parent_arrays,'train',names))
            and np.array_equal(truths['validation'], truth(parent_arrays,'validation',names))
            and np.array_equal(truths['historical'], historical_arrays['truth'])
            and np.array_equal(arrays['historical_row_id'], historical_arrays['row_id'])
            and np.array_equal(arrays['validation_row_id'], np.array([f'native-validation:{i:06d}' for i in range(100)])),
            'paired physical/c truths or row identifiers changed')
    geometry_keys = ('forward','strike','discount','maturity','option_type','half_spread')
    matches = [(i,j) for i in range(54) for j in range(63)
               if all(parent_record['task']['records'][key][i] == task['records'][key][j] for key in geometry_keys)]
    require(matches == [(51,59)] and contract['shared_contracts'][0]['old_index'] == 51
            and contract['shared_contracts'][0]['new_index'] == 59, 'exact shared contract differs')
    rng = np.random.default_rng()
    initial = parent_record['numpy_generator_states']['after_append']
    require(contract['parent']['continuation_rng_state'] == initial
            and record['numpy_generator_states']['before_train'] == initial, 'initial PCG state differs')
    rng.bit_generator.state = deepcopy(initial)
    priors = rng.uniform(bounds[:,0], bounds[:,1], (100000,6))
    require(np.array_equal(truths['train'][200000:], priors), 'all100k appended iid prior rows differ')
    noise_hashes, shared_checks = {}, {}
    for split, rows in (('train',300000),('validation',100),('historical',512)):
        # The train entry state precedes the appended prior block; subsequent
        # entries precede only their fixed-truth Gaussian observation blocks.
        if split != 'train':
            require(record['numpy_generator_states']['before_' + split] == rng.bit_generator.state, 'split initial RNG differs')
        z = rng.standard_normal((rows,63))
        noise_hashes[split] = summary(z)['sha256']
        clean, quotes, observations = [arrays[split + '_' + key] for key in ('clean_prices','quotes','observations')]
        require(clean.shape == quotes.shape == observations.shape == (rows,63)
                and all(value.dtype == np.float64 and np.isfinite(value).all() for value in (clean,quotes,observations)),
                'numeric bank matrix shape/type/finiteness differs')
        require(np.all(clean >= price_lower-bound_tolerance) and np.all(clean <= price_upper+bound_tolerance),
                'saved clean price violates the existing financial bounds/tolerance')
        require(truths[split].shape == (rows,6) and truths[split].dtype == np.float64
                and np.isfinite(truths[split]).all() and np.all(truths[split] >= bounds[:,0])
                and np.all(truths[split] <= bounds[:,1]), 'generated prior support differs')
        expected = clean + truths[split][:,-1,None] * h * z
        if split in ('train','validation'):
            paired = 200000 if split == 'train' else 100
            scale = truths[split][:paired,-1] * h[59]
            old_clean, old_quote = parent_arrays[split+'_clean_prices'][:,51], parent_arrays[split+'_quotes'][:,51]
            old_z = (old_quote-old_clean)/scale
            expected[:paired,59] = clean[:paired,59] + scale*old_z
            recovered = (quotes[:paired,59]-clean[:paired,59])/scale
            bound = 8*np.finfo(np.float64).eps*(np.abs(clean[:paired,59])/scale+np.abs(old_z)+1)
            require(np.all(np.abs(recovered-old_z) <= bound), 'shared residual recomposition exceeds float64 bound')
            shared_checks[split] = {'paired_rows': paired, 'max_standard_normal_reconstruction_error': float(np.abs(recovered-old_z).max()),
                'max_rounding_bound': float(bound.max()), 'max_absolute_clean_shift': float(np.abs(clean[:paired,59]-old_clean).max())}
        require(np.array_equal(expected,quotes) and np.array_equal(quotes/h,observations), 'exact Gaussian quote/scaled observation stream differs')
        require(record['numpy_generator_states']['after_'+split] == rng.bit_generator.state, 'split final RNG differs')
        del z, expected
    physical = rows_as_bytes(truths['train'][:,:5])
    native, old512 = rows_as_bytes(truths['validation'][:,:5]), rows_as_bytes(truths['historical'][:,:5])
    require(len(physical) == 300000 and not (physical & native) and not (physical & old512), 'duplicate or evaluation-overlapping physical training row')
    w_path = directory/'whitening.json'; w = read(w_path)
    require(sha(w_path) == record['whitening_sha256'], 'new ZCA hash differs')
    x = arrays['train_observations'][:10000]
    mean, cov = x.mean(axis=0), np.cov(x,rowvar=False)
    eig, vec = np.linalg.eigh(cov); floor = eig[-1]*1e-8
    matrix = (vec/np.sqrt(np.maximum(eig,floor)))@vec.T
    require(w['training_rows'] == 10000 and w['validation_used_for_fit'] is False and w['eigenvalue_floor'] == floor
            and all(np.array_equal(actual,w[key]) for key,actual in (('mean',mean),('covariance',cov),('eigenvalues',eig),
                ('eigenvectors',vec),('whitening_matrix',matrix))), 'new63 ZCA differs from first10k-only native arithmetic')
    return {'record': identity(directory/'bank.json'), 'data': identity(data_path), 'arrays': len(arrays),
        'train_rows': 300000, 'paired_old54_truth_rows': 200000, 'appended_iid_rows': 100000,
        'native_validation_rows': 100, 'historical_rows': 512, 'checked_price_calls': 300612,
        'exact_prior_noise_quote_rng_reconstruction': True, 'standard_normal_hashes': noise_hashes,
        'all_saved_clean_prices_within_existing_financial_bound_tolerance': True,
        'appended_prior_hash': summary(priors)['sha256'], 'shared_contract_indices': matches, 'shared_residual_checks': shared_checks,
        'training_unique_physical_rows': len(physical), 'overlap_native100': len(physical&native), 'overlap_historical512': len(physical&old512),
        'new63_first10k_only_whitening_exact': True, 'whitening': identity(w_path)}, arrays, record


def training_audit(protocol, training, run):
    contract = protocol['contract']
    rows, batches = contract['train_rows'], (contract['train_rows'] + 255) // 256
    updates = batches * 300
    require(run['status'] == 'complete' and run['completed_epochs'] == 300 and run['actual_optimizer_updates'] == updates
            and sha(training / 'run.json') == sha(training / 'training.json'), 'complete native training accounting differs')
    require(contract['example_presentations'] == rows * 300 and contract['expected_optimizer_updates'] == updates
            and contract['batches_per_epoch'] == batches and contract['tail_batch_rows'] == rows % 256,
            'predeclared native data exposure differs')
    optimizer = contract['resolved_defaults']['optimizer']
    schedule = optimizer['config']['learning_rate']['config']
    require(optimizer['class_name'] == 'AdamW' and schedule == {'initial_learning_rate': 5e-5,
        'decay_steps': updates-updates//20, 'alpha': 0, 'warmup_target': 5e-4, 'warmup_steps': updates//20}
        and run['resolved_optimizer'] == optimizer, 'native optimizer schedule differs')
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
        require(cp.get('format') == 'heston_pr87_grid_checkpoint_v1', 'new grid checkpoint identity required')
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
        'best_validation_epoch': best['epoch'], 'native_optimizer_schedule': schedule, 'checkpoints': states}


def audit_results(protocol, protocol_path, truth, historical_truth, native_quotes, historical_quotes, bounds, names):
    """Recompute allowed saved results independently, with no estimator import."""
    reports = []
    parent = read(checked(protocol['old54_control']))
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
        require(record.get('format') == 'heston_pr87_grid_evaluation_v1'
                and record['task'] == protocol['task']['target']
                and record['protocol'] == identity(protocol_path)
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
            require(quote_hash == saved['observations_sha256']
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
                if prefix in saved: checked(saved[prefix])
            if saved['reference_ready'] is not True:
                require('comparison' not in saved and saved.get('status') == 'inconclusive', 'unready reference must remain inconclusive')
                cases[case_id] = {'independent_results_match': True, 'raw_draws': len(neural),
                    'reference_ready': False, 'comparison_status': 'inconclusive',
                    'outside_prior_sample_fraction': saved['support']['outside_prior']['sample_fraction']}
                continue
            with np.load(saved['reference_chain_payload']['path'], allow_pickle=False) as archive:
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
            'raw_calibration_coverage_support_recovery_and_ready_reference_dependence_recomputed': True,
            'necessary_diagnostics_pass': independently_necessary(record)})
    require(len(reports) in (1, 2), 'missing primary evaluation')
    return reports


def reference_audit(campaign, *, require_complete):
    protocol = read(checked(campaign['reference_protocol']))
    for spec in protocol['source_sha256'].values(): checked(spec)
    panel_path = checked(campaign['readiness']['conditional_panel'])
    panel = read(panel_path)
    require(protocol['panel_sha256'] == sha(panel_path) and protocol['design_sha256'] == panel['design']['sha256'],
            'new63 reference panel/design binding differs')
    report = {}
    for job in protocol['jobs']:
        case = next(row for row in panel['cases'] if row['id'] == job['id'])
        observation = np.asarray(case['observations'], dtype='<f8')
        obs_sha = hashlib.sha256(observation.tobytes()).hexdigest()
        require(job['task'] == panel['task'] and job['observation_sha256'] == obs_sha, 'reference job target differs')
        path = Path(job['evidence_out'])/'run.json'
        if not path.exists():
            require(not require_complete, 'reference job not completed')
            continue
        run = read(path)
        require(run['protocol'] == identity(campaign['reference_protocol']['path']) and run['task'] == panel['task']
                and run['source_sha256'] == protocol['source_sha256'] and run['smoke'] is False
                and run['truth_available_to_sampler'] is False and run['reserved_test_arrays_opened'] is False
                and run['scientific_acceptance'] is False and run['observation_sha256'] == obs_sha
                and np.array_equal(run['observations'], observation) and run['case_id'] == job['id']
                and run['panel_sha256'] == sha(panel_path) and run['design_sha256'] == panel['design']['sha256'],
                'production reference provenance differs')
        if require_complete: require(run['status'] == 'completed', 'reference process did not complete')
        item = {'run': identity(path), 'status': run['status'], 'reference_ready': run.get('reference_ready') is True,
            'observation_sha256': obs_sha, 'attempts': len(run['attempts'])}
        if run['status'] == 'completed':
            diagnostic_path = Path(job['evidence_out'])/'diagnostics.json'
            diagnostics = read(diagnostic_path)
            selected = run['attempts'][run['selected_attempt']]
            require(selected['diagnostics_sha256'] == sha(diagnostic_path)
                    and diagnostics['chain_payload_sha256'] == sha(run['chain_payload_path']), 'diagnosed reference payload changed')
            ready = diagnostics.get('reference_ready') is True and diagnostics.get('mcmc_diagnostics_passed') is True and diagnostics.get('pointwise_precision_passed') is True
            require(item['reference_ready'] == ready, 'production reference readiness differs')
            item.update(diagnostics=identity(diagnostic_path), chain_payload=identity(run['chain_payload_path']),
                mcmc_diagnostics_passed=diagnostics['mcmc_diagnostics_passed'],
                pointwise_precision_passed=diagnostics['pointwise_precision_passed'])
        report[job['id']] = item
    return report


def audit(evidence, *, stage='final', selected_run=None):
    started = time.monotonic()
    evidence = Path(evidence).resolve()
    path = evidence/'campaign-protocol.json'
    campaign = read(path)
    require(campaign['format'] == 'heston_pr87_grid_campaign_protocol_v1'
            and campaign['predeclared_before_bank'] is True and campaign['scientific_acceptance'] is False
            and list(campaign['runs']) == ['200k-300','300k-300'], 'campaign identity changed')
    for spec in campaign['sources'].values(): checked(spec)
    for spec in campaign['readiness'].values():
        if isinstance(spec,dict) and 'path' in spec: checked(spec)
    for key in ('historical_inputs','old54_control','native_smoke','bank_protocol','reference_protocol','preserved_artifacts'):
        checked(campaign[key])
    preserved = read(campaign['preserved_artifacts']['path'])
    require(len(preserved['files']) == preserved['file_count'] == 749, 'protected-file allocation differs')
    for row in preserved['files']:
        checked(row)
        require(Path(row['path']).stat().st_size == row['bytes'], 'protected file size differs')
    for spec in campaign['runs'].values(): checked(spec['protocol'])
    comparison_path = evidence/'old54-numerical-crosscheck.json'
    comparison = read(comparison_path)
    for key in ('producer_archive','declaration','old_design','old_model_source','new_representation'): checked(comparison[key])
    require(comparison['status'] == 'complete' and comparison['accepted_old_points'] == 98
            and len(comparison['points']) == 98 and [row['index'] for row in comparison['points']] == list(range(98))
            and all(row['old_status'] == 'accepted' for row in comparison['points'])
            and max(row['maximum_difference_over_minimum_sd'] for row in comparison['points']) == comparison['maximum_accepted_difference_over_minimum_sd'],
            'saved old54/CV numerical crosscheck differs')
    report = {'format': 'heston_pr87_grid_independent_audit_v1', 'status': stage+'_verified',
        'audited_utc': datetime.now(timezone.utc).isoformat(), 'audit_source': identity(__file__),
        'source_commit': 'a2c0c2a', 'campaign_protocol': identity(path),
        'protected_artifacts': {'manifest': campaign['preserved_artifacts'], 'files': len(preserved['files']),
            'bytes': sum(row['bytes'] for row in preserved['files']), 'all_unchanged': True},
        'saved_numerical_crosscheck': {'record':identity(comparison_path),'accepted_old_points':98,
            'maximum_accepted_difference_over_minimum_sd':comparison['maximum_accepted_difference_over_minimum_sd'],
            'aggregates_and_bound_sources_verified_without_pricing':True},
        'references': reference_audit(campaign,require_complete=stage in ('results','final')), 'runs': {},
        'no_pricing_fit_model_load_or_posterior_query_performed': True, 'reserved_arrays_opened': False,
        'new_reference_sampling': False, 'scientific_acceptance': False,
        'scope': 'Independent saved-data/state/result audit; reused development screens are necessary checks and do not establish official quality.',
        'python': platform.python_version(), 'numpy': np.__version__}
    if stage not in ('scope','abort'):
        bank_report, arrays, bank = bank_audit(campaign)
        report['bank'] = bank_report
        names = bank['task']['target']['parameter_names']
        for name,spec in campaign['runs'].items():
            if selected_run and name != selected_run: continue
            protocol_path = checked(spec['protocol']); protocol = read(protocol_path)
            require(protocol['declared_utc'] == campaign['declared_utc']
                    and protocol['predeclared_before_bank'] is True and protocol['predeclared_before_fit'] is True
                    and protocol['bank_protocol'] == campaign['bank_protocol']
                    and protocol['reference_protocol'] == campaign['reference_protocol']
                    and protocol['contract']['bank_contract'] == bank['protocol'], 'fit declaration differs from common frozen campaign')
            directory = Path(protocol['output'])
            if stage == 'bank' or not (directory/'run.json').exists(): continue
            run = read(directory/'run.json')
            require(run['protocol'] == protocol['contract'] and run['protocol_sha256'] == sha(protocol_path)
                    and run['data_payload_sha256'] == bank['data_payload_sha256']
                    and run['bank_record_sha256'] == bank_report['record']['sha256']
                    and run['data_arrays'] == bank['data_arrays'] and run['pricing_calls'] == 0
                    and run['task'] == bank['task']
                    and (directory/'whitening.json').read_bytes() == Path(bank_report['whitening']['path']).read_bytes(),
                    'fit bank/ZCA/source identity differs')
            if run['status'] != 'complete':
                require(stage not in ('results','final'), 'required fit incomplete')
                continue
            item = {'protocol': identity(protocol_path), 'training': training_audit(protocol,directory,run)}
            if stage in ('results','final'):
                if run['best_validation_archive']['epoch'] != 300:
                    require((Path(protocol['evaluations']['secondary']['output'])/'evaluation.json').exists(), 'distinct secondary evaluation incomplete')
                item['saved_result_checks'] = audit_results(protocol,protocol_path,truth(arrays,'validation',names),
                    truth(arrays,'historical',names),arrays['validation_quotes'],arrays['historical_quotes'],
                    np.asarray(bank['task']['target']['prior_bounds']),names)
            report['runs'][name] = item
    if stage == 'abort':
        execution_path = evidence/'campaign.json'; execution = read(execution_path)
        require(execution['status'] == 'failed' and execution['protocol'] == identity(path)
                and execution['error_type'] == 'RuntimeError'
                and execution['error'] == 'grid worker process/output failed: reference_synthetic-market-like',
                'expected immutable coupled-campaign failure differs')
        failed_path = evidence/'references/synthetic-market-like/run.json'; failed = read(failed_path)
        require(failed['status'] == 'failed' and failed['error_type'] == 'TimeoutError'
                and failed['error'] == 'reference CPU or wall budget exhausted'
                and failed['usage']['limit_seconds'] == 900
                and failed['usage']['wall_seconds'] >= 900 and failed['reference_ready'] is False,
                'reference cap failure differs')
        bank_directory = Path(read(campaign['bank_protocol']['path'])['output'])
        progress_path = bank_directory/'generation-progress.json'; progress = read(progress_path)
        require(progress['completed_pricing_calls'] == 290000 and progress['total_pricing_calls'] == 300612
                and not (bank_directory/'data.npz').exists()
                and not any(Path(read(spec['protocol']['path'])['output']).exists() for spec in campaign['runs'].values()),
                'abort bank/fit accounting differs')
        require([job['id'] for job in execution['jobs']] == ['bank','reference_synthetic-broad','reference_synthetic-market-like','reference_market']
                and [job['status'] for job in execution['jobs']] == ['stopped_after_peer_failure','complete','failed','stopped_after_peer_failure'],
                'abort process coupling or unexpected worker differs')
        report.update(status='abort_verified',campaign_execution=identity(execution_path),
            termination={'cause':'declared900s reference cap caused coupled campaign stop',
                'reference_failure':identity(failed_path),'usage':failed['usage'],
                'bank_record':identity(bank_directory/'bank.json'),'last_bank_progress':identity(progress_path),
                'last_completed_pricing_progress':290000,'planned_pricing_calls':300612,
                'bank_payload_completed':False,'new_fits':0,'posterior_evaluations':0,
                'scientific_quality_verdict':'not assessed; no completed bank/fit/evaluation',
                'automatic_retry_performed':False,'original_failed_records_unchanged':True})
    if stage == 'final':
        require(selected_run is None and list(report['runs']) == ['200k-300','300k-300'], 'complete two-fit campaign required')
        execution_path = evidence/'campaign.json'; execution = read(execution_path)
        require(execution['status'] == 'complete' and execution['protocol'] == identity(path), 'complete frozen campaign required')
        expected = {'bank':3600, **{'reference_'+case:950 for case in ('synthetic-broad','synthetic-market-like','market')}}
        for name,item in report['runs'].items():
            expected[name+'_training'] = 3600
            expected[name+'_primary_evaluation'] = 900
            if item['training']['best_validation_epoch'] != 300: expected[name+'_secondary_evaluation'] = 900
        require(len(execution['jobs']) == len(expected) and {job['id'] for job in execution['jobs']} == set(expected)
                and all(job['status'] == 'complete' and job['returncode'] == 0 and job['wall_seconds_limit'] == expected[job['id']]
                        for job in execution['jobs']), 'unplanned worker, retry or budget differs')
        report.update(status='complete',campaign_execution=identity(execution_path),
            declared_workers={'jobs':len(expected),'fits':2,'bank':1,'references':3,
                'posterior_endpoints':len(expected)-6,'no_retry_or_unplanned_fit':True})
    report['seconds'] = time.monotonic()-started
    return report


def recovery_prelaunch_audit(evidence):
    evidence = Path(evidence).resolve()
    path = evidence/'campaign-protocol.json'; campaign = read(path)
    failed = read(checked(campaign['recovery_of']))
    original_path = checked(failed['protocol']); original = read(original_path)
    require(campaign['format'] == 'heston_pr87_grid_recovery_campaign_protocol_v1' and failed['status'] == 'failed'
            and campaign['predeclared_before_bank'] is True and campaign['scientific_acceptance'] is False,
            'separate predeclared recovery required')
    for spec in campaign['sources'].values(): checked(spec)
    changed_sources = [key for key in original['sources'] if original['sources'][key] != campaign['sources'][key]]
    require(changed_sources == ['controller','evaluator','grid_reference'], 'unexpected recovery source change')
    old_reference_source = Path(original['sources']['grid_reference']['path']).read_text()
    reference_source = Path(campaign['sources']['grid_reference']['path']).read_text()
    old_body = old_reference_source[old_reference_source.index('"""',3)+3:]
    new_body = reference_source[reference_source.index('"""',3)+3:]
    require(new_body == old_body.replace('SEED, INNER_SECONDS, OUTER_SECONDS = 261112, 900, 950',
                'SEED, INNER_SECONDS, OUTER_SECONDS = 261112, 2400, 2450'), 'reference scientific source changed')
    require(Path(campaign['sources']['evaluator']['path']).read_text() == Path(original['sources']['evaluator']['path']).read_text().replace(
                'from scripts import run_heston_pr87_grid_reference as reference',
                'from scripts import run_heston_pr87_grid_reference_recovery as reference'), 'evaluator changed beyond reference import')
    require(campaign['readiness'] == original['readiness'] and campaign['native_smoke'] == original['native_smoke'], 'readiness proof changed')
    for key in ('parent_training','historical_inputs','old54_control'): require(campaign[key] == original[key], 'recovery input changed')
    bank = read(checked(campaign['bank_protocol'])); original_bank = read(checked(original['bank_protocol']))
    require(bank['contract'] == original_bank['contract'] and bank['task'] == original_bank['task'], 'recovery bank contract changed')
    for spec in bank['contract']['source'].values(): checked(spec)
    limits, old_limits = dict(campaign['limits']),dict(original['limits'])
    for key,old,new in [('reference_inner_seconds',900,2400),('reference_outer_seconds',950,2450)]:
        require(old_limits.pop(key) == old and limits.pop(key) == new, 'recovery reference limit differs')
    require(limits == old_limits, 'other resource budget changed')
    old_ref,ref = read(checked(original['reference_protocol'])),read(checked(campaign['reference_protocol']))
    settings,old_settings = dict(ref['settings']),dict(old_ref['settings'])
    for key,old,new in [('inner_cpu_and_wall_seconds',900,2400),('outer_total_wall_seconds',950,2450)]:
        require(old_settings.pop(key) == old and settings.pop(key) == new, 'reference setting cap differs')
    require(settings == old_settings and len(ref['jobs']) == 3, 'reference sampler/attempt/gate settings changed')
    for a,b in zip(old_ref['jobs'],ref['jobs'],strict=True):
        require({key:value for key,value in a.items() if key not in ('evidence_out','payload_out')}
                == {key:value for key,value in b.items() if key not in ('evidence_out','payload_out')}, 'reference target/case/seed changed')
    source_diff = [key for key in old_ref['source_sha256'] if old_ref['source_sha256'][key] != ref['source_sha256'][key]]
    require(source_diff == ['grid_reference'], 'reference numerical source changed')
    for spec in ref['source_sha256'].values(): checked(spec)
    preserved = read(checked(campaign['preserved_artifacts']))
    before = read(checked(original['preserved_artifacts']))
    require(preserved['file_count'] == len(preserved['files']) == 1017
            and {row['path'] for row in before['files']} <= {row['path'] for row in preserved['files']}, 'protected inventory differs')
    for row in preserved['files']:
        checked(row); require(Path(row['path']).stat().st_size == row['bytes'], 'preserved file size changed')
    runs = {}
    outputs = [bank['output'],campaign['bank_log']]
    for job in ref['jobs']: outputs += [job['evidence_out'],job['payload_out'],campaign['reference_logs'][job['id']]]
    for name,spec in campaign['runs'].items():
        study = read(checked(spec['protocol'])); old_study = read(checked(original['runs'][name]['protocol']))
        require(all(study[key] == old_study[key] for key in ('contract','task','task_file','evaluation','endpoint_policy','old54_control','conditional_panel'))
                and study['bank_protocol'] == campaign['bank_protocol'] and study['sources'] == campaign['sources']
                and study['reference_protocol'] == campaign['reference_protocol'] and study['declared_utc'] == campaign['declared_utc']
                and study['predeclared_before_bank'] is True and study['predeclared_before_fit'] is True, 'fit science/freeze changed')
        runs[name] = {'protocol':spec['protocol'],'contract_exactly_equal_original':True,
            'evaluation_and_endpoint_contract_exactly_equal_original':True}
        outputs += [study['output'],study['training_log']]
        outputs += [item[key] for item in study['evaluations'].values() for key in ('output','payload_output','log')]
    require(len(outputs) == len(set(outputs)) and not any(Path(item).exists() for item in outputs), 'outputs overlap or production already started')
    return {'format':'heston_pr87_grid_recovery_prelaunch_audit_v1','status':'prelaunch_verified',
        'audit_source':identity(__file__),'audited_utc':datetime.now(timezone.utc).isoformat(),
        'campaign_protocol':identity(path),'original_protocol':identity(original_path),'original_failed_campaign':campaign['recovery_of'],
        'bank_contract_exactly_equal_original':True,'bank_producer_unchanged':True,
        'reference_source_diff_only_documentation_and_resource_caps':True,'evaluator_diff_only_reference_import':True,
        'reference_sampler_cases_seeds_attempts_target_numerical_gates_identical':True,
        'source_binding_changes':changed_sources,'source_bindings':campaign['sources'],
        'protected_artifacts':{'manifest':campaign['preserved_artifacts'],'files':len(preserved['files']),
            'bytes':sum(row['bytes'] for row in preserved['files']),'includes_original749':True,'all_unchanged':True},
        'runs':runs,'no_pricing_fit_model_load_or_posterior_query_performed':True,'reserved_arrays_opened':False,
        'scientific_acceptance':False,'scope':'Independent source/contract/preservation readiness audit; no posterior quality verdict.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',required=True,type=Path)
    parser.add_argument('--stage',choices=('scope','abort','prelaunch','bank','training','results','final'),default='final')
    parser.add_argument('--run',choices=('200k-300','300k-300'))
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    result = recovery_prelaunch_audit(args.evidence) if args.stage == 'prelaunch' else audit(args.evidence,stage=args.stage,selected_run=args.run)
    if args.stage in ('final','abort','prelaunch'):
        name = {'final':'independent-audit-consumer.py','abort':'independent-abort-audit-consumer.py','prelaunch':'independent-prelaunch-audit-consumer.py'}[args.stage]
        archive = args.evidence/name
        with archive.open('xb') as stream: stream.write(Path(__file__).read_bytes())
        require(sha(archive) == sha(__file__),'archived consumer differs')
        result['archived_audit_source'] = identity(archive)
    destination = args.out or args.evidence/('independent-audit.json' if args.stage == 'final' else 'independent-audit-'+args.stage+'.json')
    with destination.open('x') as stream: stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'protected_files':result['protected_artifacts']['files'],
        'bank_rows':result.get('bank',{}).get('train_rows'),'runs':list(result['runs']),'seconds':result.get('seconds')}))
