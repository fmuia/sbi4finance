"""Preserved CPU-only prior comparison from the data audit.

Run from the repository root with its pinned NumPy/SciPy environment. This
reconstructs declared prior draws only: it does not load the original bank,
price options, train an estimator, or open test data. The original execution
was an inline tool cell; this file preserves its calculation for review.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp

config = json.loads(Path('configs/tasks/calibration_heston_bank.json').read_text())
task = config['task']
bounds = np.asarray(task['prior_bounds'])
digest = '5e499526ff8f29a72f49785a1a36006b1d769d26bca8a6f83f7a8c8ac4d431fa'
words = [int(digest[i:i+8], 16) for i in range(0, 64, 8)]
cases = []
for i in range(3):
    report = json.loads(Path(f'docs/pilot-results/2026-10-27/backbone-runs/heston-100k/comparisons/case-{i}/comparison.json').read_text())
    cases.append([report['proposed_parameter_agreement']['parameters'][name]['reference_mean']
                  for name in task['parameter_names']])
cases = (np.asarray(cases) - bounds[:, 0]) / np.diff(bounds)[:, 0]
sets = {}
for root, n in [(2610, 50000), (271028, 100000)]:
    parameters = np.stack([
        np.random.default_rng(np.random.SeedSequence([0x4F5054, root, *words, 0, i, 0]))
        .uniform(bounds[:, 0], bounds[:, 1]) for i in range(n)])
    sets[root] = parameters
    unit = (parameters - bounds[:, 0]) / np.diff(bounds)[:, 0]
    print('root', root, 'n', n, 'means', parameters.mean(0).tolist(),
          'std', parameters.std(0).tolist(),
          'unitquantiles', np.quantile(unit, [.001, .01, .5, .99, .999], axis=0).tolist(),
          'corr', np.corrcoef(unit.T).round(5).tolist())
    for size in ([50000] if root == 2610 else [50000, 100000]):
        distance = np.linalg.norm(unit[None, :size, :] - cases[:, None, :], axis=2)
        print('root', root, 'prefix', size,
              'nearest_unitbox_distance', distance.min(1).tolist(),
              'within_radius0.1', np.count_nonzero(distance < .1, axis=1).tolist(),
              'radius0.2', np.count_nonzero(distance < .2, axis=1).tolist())
print('two_sample_marginal_KS_old50_vs_new50',
      [float(ks_2samp(sets[2610][:, j], sets[271028][:50000, j]).statistic) for j in range(4)])
