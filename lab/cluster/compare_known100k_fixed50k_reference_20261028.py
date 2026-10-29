"""Compare saved 50k/100k neural panels against one unchanged 50k reference.

This finite metadata/statistics operation does not sample, price, train or submit
jobs. Existing reference chains and 50k comparison files remain read-only.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

BASE = Path('/oscar/scratch/afengler')
REPO = BASE / 'sbi4finance-training-recovery-v2-20261028/repo'
FIT = BASE / 'sbi4finance-independent-work-20261028/known100k/known'
BANK = BASE / 'sbi4finance-rough-known-backbone-20261027/snapshot-100k'
OLD_BANK = BASE / 'sbi4finance-rough-known-backbone-20261027/snapshot-50k'
REFERENCE = BASE / 'sbi4finance-scientific-validation-20261028/known50k-references'
OUT = BASE / 'sbi4finance-scientific-validation-20261028/known100k-fixed50k-reference'


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    sys.path.insert(0, str(REPO / 'src'))
    from sbi4finance.calibration import bank_collection

    plan = read(REFERENCE / 'plan.json')
    if subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() != plan['source_revision']:
        raise ValueError('Frozen c96 consumer changed')
    for path, digest in plan['frozen_sha256'].items():
        if sha(path) != digest: raise ValueError('Original frozen 50k input changed: ' + path)
    spec = importlib.util.spec_from_file_location('fixed_comparison', REPO / 'scripts/compare_full_surface_reference.py')
    comparison = importlib.util.module_from_spec(spec); spec.loader.exec_module(comparison)
    manifest, data = bank_collection.load_bank(BANK, splits=['validation'], roles=['observations', 'context_values', 'example_id'])
    old_manifest, old_data = bank_collection.load_bank(OLD_BANK, splits=['validation'], roles=['observations', 'context_values', 'example_id'])
    if manifest['configuration']['task'] != old_manifest['configuration']['task']:
        raise ValueError('50k/100k task contracts differ')
    for key in ('observations', 'context_values', 'example_id'):
        if not np.array_equal(data['validation'][key], old_data['validation'][key]):
            raise ValueError('Full shared validation holdout differs: ' + key)
    training = read(FIT / 'posterior/training.json'); evaluation = read(FIT / 'validation/evaluation.json')
    if (training['training_surfaces'] != 100000 or training['selected_epoch'] != 500
            or training['checkpoint_selection'] != 'final' or training['reserved_test_evaluated'] is not False
            or sha(FIT / 'posterior/model.keras') != training['checkpoint_sha256']
            or evaluation['checkpoint_sha256'] != training['checkpoint_sha256']
            or evaluation['bank_manifest_sha256'] != sha(BANK / 'manifest.json')
            or read(FIT / 'posterior/task.json') != manifest['configuration']['task']):
        raise ValueError('Saved 100k model/panel/task/bank binding differs')
    with np.load(FIT / 'validation/draws.npz', allow_pickle=False) as arrays:
        neural = arrays['draws'][:3].copy()
    if neural.shape != (3, 1000, 3) or not np.isfinite(neural).all():
        raise ValueError('Expected three saved finite 1000-draw panels')
    accounting = subprocess.check_output(['sacct', '-S', '2026-10-27', '-j', '8030892,8059665', '-n', '-P',
        '--format=JobID%64,State%40,ExitCode,ElapsedRaw,AllocCPUS'], text=True)
    rows = {r[0]: r for line in accounting.splitlines() if len(r := line.split('|')) >= 5 and '.' not in r[0]}
    cost = 0
    for job in ('8030892_0', '8030892_1', '8030892_2', '8059665'):
        row = rows[job]
        if row[1] != 'COMPLETED' or row[2] != '0:0': raise ValueError('Prior reference accounting unsettled')
        cost += int(row[3]) * int(row[4])
    if cost != 1544: raise ValueError('Settled reference cost differs')
    paths = [Path(__file__).resolve(), BANK / 'manifest.json', OLD_BANK / 'manifest.json',
        FIT / 'posterior/training.json', FIT / 'posterior/task.json', FIT / 'posterior/model.keras',
        FIT / 'validation/evaluation.json', FIT / 'validation/draws.npz', REFERENCE / 'plan.json',
        REFERENCE / 'run.json', REPO / 'scripts/compare_full_surface_reference.py',
        REPO / 'src/sbi4finance/calibration/validation.py']
    results = []
    for index, case in enumerate(plan['cases']):
        root = REFERENCE / f'case-{index}'
        record = read(root / 'reference.json'); diagnostics = read(root / 'diagnostics.json')
        old_comparison = read(root / 'saved-neural-comparison.json')
        if (evaluation['example_ids'][index] != case['surface_id'] or record['surface_id'] != case['surface_id']
                or record['settings']['seed'] != case['seed'] or record['task'] != manifest['configuration']['task']
                or not np.array_equal(data['validation']['observations'][index, :, 0], record['observed_prices'])
                or not np.array_equal(data['validation']['context_values'][index], record['known_curve_levels'])):
            raise ValueError('Matched case/quote/curve/seed binding differs')
        comparison._emulator_binding(record, record['task'], record['task_fingerprint'])
        with np.load(root / 'chains.npz', allow_pickle=False) as arrays: chains = arrays['parameters']
        if chains.shape != (4, 8000, 3) or sha(root / 'chains.npz') != old_comparison['chains_sha256']:
            raise ValueError('Original reference chain axes or bytes differ')
        names = record['task']['parameter_names']
        issues = comparison._reference_issues(diagnostics, names)
        issues += comparison._reference_issues(diagnostics['derived_quantities'], ['rho_eta'])
        if issues: raise ValueError('Fixed reference diagnostics failed: ' + str(issues))
        agreement = comparison.validation.agreement(neural[index], chains.reshape(-1, 3))
        parameters = {name: {'standardized_mean_error': agreement['standardized_mean_error'][j],
            'width_ratio_90': agreement['width_ratio'][j],
            'mean_error_within_0_3_reference_sd': agreement['standardized_mean_error'][j] <= .3,
            'relative_width_error_within_25_percent': abs(agreement['width_ratio'][j] - 1) <= .25}
            for j, name in enumerate(names)}
        passed = all(p['mean_error_within_0_3_reference_sd'] and p['relative_width_error_within_25_percent']
            for p in parameters.values())
        results.append({'surface_id': case['surface_id'], 'reference_seed': case['seed'],
            'reference_chain_shape': list(chains.shape), 'neural_draws_each_model': 1000,
            'reference_diagnostics_passed': True, 'scientific_acceptance': False,
            'known50k_neural': {'proposed_parameter_agreement': old_comparison['proposed_parameter_agreement'],
                'derived_parameter_comparison': old_comparison['derived_parameter_comparison']},
            'known100k_neural': {'proposed_parameter_agreement': {'parameters': parameters, 'all_checks_passed': passed},
                'derived_parameter_comparison': comparison._rough_derived_comparison(record['task'], neural[index], chains, diagnostics)}})
        paths.extend(root / name for name in ('reference.json', 'diagnostics.json', 'chains.npz', 'saved-neural-comparison.json'))
    report = {'format': 'known100k_neural_fixed50k_reference_comparison_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'source_revision': plan['source_revision'],
        'reference_root': str(REFERENCE), 'reference_plan_sha256': sha(REFERENCE / 'plan.json'),
        'reference_pricer': plan['pricer'], 'bank': str(BANK), 'bank_manifest_sha256': sha(BANK / 'manifest.json'),
        'posterior_checkpoint_sha256': training['checkpoint_sha256'],
        'shared_full_validation_holdout_cases': len(data['validation']['observations']),
        'new_mcmc_runs': 0, 'new_allocated_cpu_seconds': 0, 'cumulative_reference_cpu_seconds': cost,
        'new_neural_sampling': False, 'new_financial_pricing': False, 'reserved_test_evaluated': False,
        'scientific_acceptance': False, 'reference_target_scope': plan['target_scope'],
        'interpretation': 'Both saved neural models are compared with exactly the same 50k conditional-mean-emulator reference chains. The 100k pricer is deliberately not used. This holds the reference fixed, but does not establish agreement with the stochastic simulator or a continuous-time posterior, nor isolate a causal data-size effect.',
        'monte_carlo_qualification': 'Each neural panel reuses 1000 saved draws; these are matched observations, not paired neural random draws. Reference mean MCSE is retained in diagnostics. Quantile MCSE and neural Monte Carlo errors are unassessed; thresholds remain descriptive development screens.',
        'cases': results, 'frozen_sha256': {str(path): sha(path) for path in paths}}
    OUT.mkdir(exist_ok=False)
    save(OUT / 'comparison.json', report)
    (OUT / 'terminal-accounting.psv').write_text(accounting)
    np.savez_compressed(OUT / 'saved-known100k-neural-draws.npz', draws=neural)
    print(json.dumps({'root': str(OUT), 'comparison_sha256': sha(OUT / 'comparison.json'),
        'known50k_cases_passed': sum(c['known50k_neural']['proposed_parameter_agreement']['all_checks_passed'] for c in results),
        'known100k_cases_passed': sum(c['known100k_neural']['proposed_parameter_agreement']['all_checks_passed'] for c in results),
        'new_mcmc_runs': 0, 'cumulative_reference_cpu_seconds': cost}))


if __name__ == '__main__': main()
