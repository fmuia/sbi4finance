"""Replay the existing E1 slider values through the pinned marimo graph."""
import hashlib
import json
from pathlib import Path
import runpy
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import sbi4finance

root = Path('/tmp/sbi4finance-participant-final/isolated').resolve()
assert Path(sbi4finance.__file__).resolve().is_relative_to(root / 'src')
notebook = root / 'notebooks/01_gbm_foundations.py'
app = runpy.run_path(str(notebook))['app']
rows, histories = [], []
started = perf_counter()
for days in (252, 504):
    outputs, defs = app.run(defs={
        'history_length': SimpleNamespace(value=days),
        'history_seed': SimpleNamespace(value=2026),
    })
    context = defs['physical_context']
    elapsed = float(defs['observed_dt'].sum())
    variance = float(defs['posterior_std'][0] ** 2)
    expected = 1 / (context['prior_std'] ** -2 + elapsed / context['known_volatility'] ** 2)
    assert np.isclose(variance, expected, rtol=1e-13)
    histories.append(defs['observed_returns'])
    rows.append({'days': days, 'seed': 2026, 'elapsed_years': elapsed,
                 'posterior_mean': float(defs['posterior_mean'][0]),
                 'posterior_std': float(defs['posterior_std'][0]),
                 'posterior_variance': variance, 'analytic_variance': expected,
                 'output_count': len(outputs)})
assert np.array_equal(histories[0], histories[1][:, :252, :])
assert rows[1]['posterior_variance'] < rows[0]['posterior_variance']
fixed_year_variances = [float(1 / (context['prior_std'] ** -2 + np.full(n, 1/n).sum() / context['known_volatility'] ** 2)) for n in (252, 504)]
assert np.isclose(*fixed_year_variances, rtol=1e-13)
result = {'exercise': 'E1', 'source_revision': 'ca63424a140c2e512db97e40f0bb30010da3956a',
          'notebook_sha256': hashlib.sha256(notebook.read_bytes()).hexdigest(),
          'method': 'marimo app.run with both slider definitions overridden; Python-level network guard and BayesFlow import block active',
          'browser_slider_interaction_verified': False,
          'rows': rows, 'same_path_prefix': True,
          'fixed_one_year_variance_252_and_504_observations': fixed_year_variances,
          'elapsed_seconds': perf_counter() - started}
Path('/tmp/sbi4finance-participant-final/exercise-e1.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
