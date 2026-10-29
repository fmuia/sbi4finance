"""Cheap postprocessing of existing Heston references; no inference or pricing."""
import argparse
import hashlib
from importlib import metadata
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.special import logit
from scipy.stats import norm, skew, spearmanr


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normal_cdf_distance(values):
    ordered = np.sort(values)
    fitted = norm.cdf(ordered, values.mean(), values.std())
    empirical = np.arange(1, len(values) + 1) / len(values)
    return float(max(np.max(empirical - fitted), np.max(fitted - (empirical - 1 / len(values)))))


def empirical_cdf_distance(left, right):
    pooled = np.unique(np.concatenate([left, right]))
    a = np.searchsorted(np.sort(left), pooled, side='right') / len(left)
    b = np.searchsorted(np.sort(right), pooled, side='right') / len(right)
    return float(np.max(np.abs(a - b)))


def marginals(values, bounds):
    z = logit((values - bounds[:, 0]) / np.diff(bounds, axis=1).ravel())
    result = []
    for j in range(values.shape[1]):
        x, transformed = values[:, j], z[:, j]
        low, high = bounds[j]
        result.append({'mean': float(x.mean()), 'sd': float(x.std()),
                       'physical_skew': float(skew(x, bias=False)),
                       'logit_skew': float(skew(transformed, bias=False)),
                       'q01_q05_median_q95_q99': np.quantile(x, [.01, .05, .5, .95, .99]).tolist(),
                       'mean_distance_to_lower_bound_sd': float((x.mean() - low) / x.std()),
                       'mean_distance_to_upper_bound_sd': float((high - x.mean()) / x.std()),
                       'physical_moment_gaussian_cdf_distance': normal_cdf_distance(x),
                       'logit_moment_gaussian_cdf_distance': normal_cdf_distance(transformed),
                       'physical_gaussian_probability_outside_prior': float(
                           norm.cdf(low, x.mean(), x.std()) + norm.sf(high, x.mean(), x.std()))})
    return result, z


def bins(x, y, edges=None):
    if edges is None:
        edges = np.quantile(x, np.linspace(0, 1, 11))
    assigned = np.clip(np.searchsorted(edges, x, side='right') - 1, 0, 9)
    mean_x = np.array([x[assigned == k].mean() for k in range(10)])
    mean_y = np.array([y[assigned == k].mean() for k in range(10)])
    return mean_x, mean_y, edges


def pair_geometry(chains, i, j):
    values = chains.reshape(-1, chains.shape[-1])
    x, y = values[:, i], values[:, j]
    fit = np.polyfit(x, y, 1)
    bx, by, edges = bins(x, y)
    departures = (by - np.polyval(fit, bx)) / y.std()
    folds = []
    for held_out in range(4):
        train = chains[np.arange(4) != held_out].reshape(-1, chains.shape[-1])
        test = chains[held_out]
        mse = {}
        for degree in (1, 2):
            fitted = np.polyfit(train[:, i], train[:, j], degree)
            mse[str(degree)] = float(np.mean((test[:, j] - np.polyval(fitted, test[:, i]))**2) / y.var())
        folds.append({'held_out_chain': held_out, 'linear_mse_over_y_variance': mse['1'],
                      'quadratic_mse_over_y_variance': mse['2'],
                      'relative_mse_improvement': (mse['1'] - mse['2']) / mse['1']})
    return {'pearson': float(np.corrcoef(x, y)[0, 1]), 'spearman': float(spearmanr(x, y).statistic),
            'conditional_decile_mean_rms_departure_y_sd': float(np.sqrt(np.mean(departures**2))),
            'conditional_decile_mean_max_departure_y_sd': float(np.max(np.abs(departures))),
            'bin_mean_x': bx.tolist(), 'bin_mean_y': by.tolist(),
            'bin_departures_y_sd': departures.tolist(), 'linear_fit_slope_intercept': fit.tolist(),
            'chain_pearson': [float(np.corrcoef(chain[:, i], chain[:, j])[0, 1]) for chain in chains],
            'leave_one_chain_out_fits': folds,
            'interpretation': 'Descriptive physical-coordinate curvature; finite-bin noise remains. '
                              'No normality p-value, independent-data prediction claim or mode count.'}


def marginal_plot(ax, values, bound, title, label):
    low, high = bound
    sd, mean = values.std(), values.mean()
    left = min(values.min(), mean - 3.5 * sd)
    right = high + .7 * sd
    grid = np.linspace(left, right, 900)
    ax.hist(values, bins=70, density=True, color='#326b99', alpha=.42, label='PyMC reference')
    ax.plot(grid, norm.pdf(grid, mean, sd), color='#59636b', ls='--', lw=1.7,
            label='Gaussian in physical coordinates')
    inside = (grid > low) & (grid < high)
    u = (grid[inside] - low) / (high - low)
    z = logit((values - low) / (high - low))
    density = norm.pdf(logit(u), z.mean(), z.std()) / ((high - low) * u * (1 - u))
    ax.plot(grid[inside], density, color='#c9752e', lw=1.8, label='Gaussian in prior-box logit')
    ax.axvspan(high, right, color='#e9e9e9', zorder=-1)
    ax.axvline(high, color='#9a4141', ls=':', lw=1.7)
    ax.set(xlim=(left, right), xlabel=label, ylabel='Posterior density', title=title)
    ax.ticklabel_format(style='plain', useOffset=False, axis='x')
    ax.text(.03, .96, f'Physical skew: {skew(values):+.2f}\nLogit skew: {skew(z):+.2f}',
            transform=ax.transAxes, va='top', fontsize=9)


def plot_panel(chains, out):
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.2), layout='constrained')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    a = chains[0].reshape(-1, 4)
    b = chains[1].reshape(-1, 4)
    marginal_plot(axes[0, 0], a[:, 1], (.01, .16),
                  'A · Case 0: long-run variance near its upper bound', 'Long-run variance theta')
    marginal_plot(axes[0, 1], b[:, 2], (.1, 1.5),
                  'B · Case 1: strongly skewed vol-of-vol', 'Vol-of-vol')
    axes[0, 0].legend(loc='upper left', bbox_to_anchor=(.01, .75), fontsize=8, frameon=False)
    x, y = a[:, 2], a[:, 3]
    bx, by, edges = bins(x, y)
    fit = np.polyfit(x, y, 1)
    axes[1, 0].hist2d(x, y, bins=65, cmap='Blues', norm=LogNorm(vmin=1))
    axes[1, 0].plot(bx, np.polyval(fit, bx), '--', color='#c9752e', lw=2, label='Best straight line')
    axes[1, 0].plot(bx, by, 'o-', color='#172c3f', lw=1.6, ms=3.5, label='Decile conditional means')
    axes[1, 0].set(xlabel='Vol-of-vol', ylabel='Rho', title='C · Case 0: strong dependence, modest curvature')
    axes[1, 0].text(.04, .96, f'Pearson correlation: {np.corrcoef(x, y)[0, 1]:.3f}',
                    transform=axes[1, 0].transAxes, va='top', fontsize=9)
    axes[1, 0].legend(loc='lower right', fontsize=8, frameon=True, facecolor='white', framealpha=.9)
    colors = ['#87a6be', '#a1bdac', '#bfa8c7', '#ccba9c']
    for index, chain in enumerate(chains[0]):
        cx, cy, _ = bins(chain[:, 2], chain[:, 3], edges)
        axes[1, 1].plot(cx, (cy - np.polyval(fit, cx)) / y.std(), color=colors[index],
                        lw=1.3, label=f'Chain {index + 1}')
    residual = (by - np.polyval(fit, bx)) / y.std()
    axes[1, 1].plot(bx, residual, 'o-', color='#172c3f', lw=2, ms=4, label='Pooled')
    axes[1, 1].axhline(0, color='#59636b', ls='--', lw=1)
    axes[1, 1].set(xlabel='Vol-of-vol', ylabel='Departure from straight line\n(rho posterior SD)',
                  title='D · The small curve is visible across chains')
    axes[1, 1].text(.04, .96, f'Pooled RMS departure: {np.sqrt(np.mean(residual**2)):.3f} SD',
                    transform=axes[1, 1].transAxes, va='top', fontsize=9)
    axes[1, 1].legend(loc='lower center', ncol=3, fontsize=8, frameon=False)
    fig.suptitle('Heston posterior geometry: skew and a mildly curved ridge\n'
                 'Independent Fourier-likelihood / PyMC references · 4 chains × 8,000 draws per case', fontsize=15)
    fig.savefig(out / 'heston-posterior-geometry.png', dpi=170)
    fig.savefig(out / 'heston-posterior-geometry.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    declaration_path = args.repo / 'docs/pilot-results/2026-10-28/heston-candidate-acceptance/manifest.json'
    declaration = json.loads(declaration_path.read_text())
    bounds = np.asarray(declaration['task']['prior_bounds'])
    names = declaration['task']['parameter_names']
    result = {'format': 'heston_existing_posterior_geometry_v1', 'scientific_acceptance': False,
              'new_inference_or_pricing': False,
              'selection_scope': 'Post-hoc descriptive illustration from the three previously inspected development cases; '
                                 'not a prevalence claim over the prior, architecture benchmark or multimodality finding.',
              'candidate_sha256': declaration['candidate_sha256'], 'targets': names, 'prior_bounds': bounds.tolist(),
              'runtime': {name: metadata.version(name) for name in ('numpy', 'scipy', 'matplotlib')},
              'source_sha256': sha(__file__), 'declaration_sha256': sha(declaration_path), 'cases': []}
    chain_arrays = []
    for case in declaration['development_cases']:
        index = case['case']
        reference = args.data / f'case-{index}/reference/chains.npz'
        neural_path = args.data / f'results/new/100k-case-{index}.npz'
        original = args.repo / f'docs/pilot-results/2026-10-27/heston-reviewed/case-{index}/reference'
        diagnostic_path = original / 'diagnostics.json'
        assert sha(reference) == declaration['frozen_sha256'][case['reference'] + '/chains.npz']
        assert sha(neural_path) == declaration['frozen_sha256'][case['neural_draws']]
        assert sha(diagnostic_path) == declaration['frozen_sha256'][case['reference'] + '/diagnostics.json']
        diagnostics = json.loads(diagnostic_path.read_text())
        assert diagnostics['mcmc_diagnostics_passed']
        with np.load(reference, allow_pickle=False) as archive:
            chains = archive['parameters']
        with np.load(neural_path, allow_pickle=False) as archive:
            neural = archive['final']
        assert chains.shape == (4, 8000, 4) and neural.shape == (10000, 4)
        chain_arrays.append(chains)
        reference_marginals, z = marginals(chains.reshape(-1, 4), bounds)
        neural_marginals, _ = marginals(neural, bounds)
        pairs = {f'{names[i]} -> {names[j]}': pair_geometry(chains, i, j)
                 for i, j in itertools.combinations(range(4), 2)}
        result['cases'].append({'case': index, 'chain_shape': list(chains.shape), 'neural_shape': list(neural.shape),
             'input_sha256': {'reference_chains': sha(reference), 'neural_final': sha(neural_path),
                              'reference_diagnostics': sha(diagnostic_path)},
             'reference_parameter_diagnostics': diagnostics['parameters'],
             'reference_marginals': dict(zip(names, reference_marginals)),
             'reference_chain_skew': {name: [float(skew(chain[:, j], bias=False)) for chain in chains]
                                      for j, name in enumerate(names)},
             'reference_logit_chain_skew': {name: [float(skew(chain[:, j], bias=False))
                                                  for chain in z.reshape(4, 8000, 4)]
                                            for j, name in enumerate(names)},
             'neural_final_marginals': dict(zip(names, neural_marginals)), 'physical_pair_geometry': pairs,
             'neural_vs_reference_marginal_cdf_distance': {
                 name: empirical_cdf_distance(neural[:, j], chains[..., j].ravel())
                 for j, name in enumerate(names)},
             'reference_logit_correlation': np.corrcoef(z.T).tolist(),
             'neural_physical_correlation': np.corrcoef(neural.T).tolist()})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'geometry.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    plot_panel(chain_arrays, args.out)
    print(json.dumps({'cases': 3, 'plot': str(args.out / 'heston-posterior-geometry.png'),
                      'case0_vol_rho': result['cases'][0]['physical_pair_geometry']['vol_of_vol -> rho']}))


if __name__ == '__main__':
    main()
