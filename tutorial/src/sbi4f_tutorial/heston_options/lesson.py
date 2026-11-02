"""Heston option prices, illustrative paths and posterior-predictive quotes.

Edit the computations behind the Heston notebook here. Common tables and
plotting helpers are imported from shared.teaching.
"""

import numpy as np

from ..shared import surface
from ..shared.teaching import (
    LABELS, TENORS, PRODUCT_NAMES, SPX_SKEW_EXPONENT,
    load_json, load_npz, existing, design_arrays, smile_rows, flat_versus_smile,
    skew_term_structures, pairs, reserve_rows, price_histograms, agreement_rows,
    cost_rows, load_network, implied_vol_grid,
)


def heston_smile(task, row):
    """The 63 Fourier prices of one Heston parameter row (v0, theta, vol_of_vol, rho), kappa from the task.

    One quadrature rung of 1,024 nodes, the full ladder of the task where that rung fails, as in study S3.
    """
    from ..shared import pricing, tasks

    parameters, v0 = tasks._heston_split(tasks.model_parameters(task, np.asarray(row, dtype=float)))
    try:
        return pricing.heston_surface(parameters, v0, task["context"]["contracts"], nodes=1024, upper_limit=600.0)
    except ValueError:
        return tasks.heston_prices(task, parameters, v0)[0]


def heston_variance_swap_vol(v0, theta, kappa, days):
    """Heston variance-swap volatility (%) for maturities in business days: sqrt of the mean expected variance."""
    t = np.asarray(days, dtype=float) / surface.TRADING_DAYS
    v0, theta = np.asarray(v0, dtype=float)[..., None], np.asarray(theta, dtype=float)[..., None]
    w = theta + (v0 - theta) * (1 - np.exp(-kappa * t)) / (kappa * t)
    return 100 * np.sqrt(w)


def _heston_teaching_inputs(task):
    """Validate the four-target, zero-carry option task without a neural backend."""
    from ..shared.full_surface_teaching import heston_task

    heston_task(task)
    if task.get("measure") != "Q":
        raise ValueError("the option illustration requires the declared pricing measure Q")
    return np.asarray(task["prior_bounds"], dtype=float), np.asarray(task["context"]["quote_noise_std"], dtype=float)


def heston_illustrative_paths(task, parameters=None, *, paths=3, days=126, substeps=16, seed=20261101):
    """Small Q price/variance paths, separate from the Fourier option likelihood.

    Uses the vetted simulator's signed full-truncation Euler update and its
    left-variance Gaussian log-price step, with zero carry and no jumps. Store
    the signed auxiliary state; report max(state, 0) as effective variance.
    The default [v0, theta, sigma_v, rho] is [0.04, 0.04, 0.25, -0.7], with
    kappa from the task. These are illustrative parameters, not a calibration.
    Daily output includes time zero; each day uses `substeps` increments.
    No path here is used to calculate Fourier prices or banks.
    """
    bounds, _ = _heston_teaching_inputs(task)
    for value, maximum, name in ((paths, 8, "paths"), (days, 252, "days"), (substeps, 64, "substeps")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer from 1 to {maximum}")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer in [0, 2**64)")
    row = np.asarray([.04, .04, .25, -.7] if parameters is None else parameters, dtype=float)
    if row.shape != (4,) or not np.isfinite(row).all() or np.any(row < bounds[:, 0]) or np.any(row > bounds[:, 1]):
        raise ValueError("parameters must be one finite four-target row inside this task's prior")
    v0, theta, xi, rho = row
    kappa = float(task["fixed_parameters"]["kappa"])
    h = (1 / surface.TRADING_DAYS) / substeps
    normals = np.random.default_rng(seed).standard_normal((paths, days, substeps, 2))
    variance = np.full(paths, v0)
    states, returns = np.empty((paths, days + 1)), np.empty((paths, days))
    states[:, 0] = variance
    for day in range(days):
        total = np.zeros(paths)
        for step in range(substeps):
            positive = np.maximum(variance, 0.0)
            scale = np.sqrt(positive * h)
            zv, independent = normals[:, day, step, :].T
            total += -0.5 * positive * h + scale * (rho * zv + np.sqrt(1 - rho * rho) * independent)
            variance += kappa * (theta - positive) * h + xi * scale * zv
        returns[:, day], states[:, day + 1] = total, variance
    prices = np.exp(np.column_stack([np.zeros(paths), np.cumsum(returns, axis=1)]))
    if not np.isfinite(prices).all() or not np.isfinite(states).all():
        raise ValueError("nonfinite illustrative path; no redraw or clipping repair")
    return {"times_years": np.arange(days + 1) / surface.TRADING_DAYS, "prices": prices,
            "variances": np.maximum(states, 0), "variance_states": states, "log_returns": returns,
            "parameters": row.copy(), "parameter_names": list(task["parameter_names"]), "kappa": kappa,
            "seed": int(seed), "substeps": int(substeps), "scheme": "signed full-truncation Euler; left-variance log-price step",
            "scope": "Illustrative Q paths at stated parameters, with S0=1 and zero carry. This time discretization is not used for Fourier option prices, training banks or posterior inference."}


def heston_posterior_predictive(task, draw_sets, observations, *, truth=None, draws_per_set=32, seed=20261101):
    """Fourier means and replicated noisy quotes for a bounded saved-draw panel.

    Each value of draw_sets is [draw,4] or [chain,draw,4]. Evenly spaced flat
    indices select intact paired rows; this descriptive subset is not a new
    posterior/reference accuracy estimate. At most 64 rows/set and 128 total
    rows are priced, plus optional synthetic truth. `mean_prices` is P(theta);
    `replicated_quotes` is P(theta)+sigma*epsilon with independent standard
    normals per selected row/contract, drawn in the supplied set order. Noise
    is observation noise, not parameter or numerical-pricing error. Negative
    replicated quotes are retained. No implied-vol inversion occurs.
    """
    bounds, sigma = _heston_teaching_inputs(task)
    if isinstance(draws_per_set, (bool, np.bool_)) or not isinstance(draws_per_set, (int, np.integer)) or not 2 <= draws_per_set <= 64:
        raise ValueError("draws_per_set must be an integer from 2 to 64")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer in [0, 2**64)")
    observed = np.asarray(observations, dtype=float)
    if observed.shape != sigma.shape or not np.isfinite(observed).all():
        raise ValueError("observations must be the finite ordered 63-price vector")
    if not isinstance(draw_sets, dict) or not draw_sets:
        raise ValueError("draw_sets must be a nonempty dictionary of saved posterior arrays")
    selected = {}
    for label, values in draw_sets.items():
        draws = np.asarray(values, dtype=float)
        if (not isinstance(label, str) or not label or draws.ndim not in (2, 3) or draws.shape[-1] != 4
                or draws.size == 0 or not np.isfinite(draws).all()
                or np.any(draws < bounds[:, 0]) or np.any(draws > bounds[:, 1])):
            raise ValueError("each named posterior needs finite paired four-target rows inside this task's prior")
        flat = draws.reshape(-1, 4)
        indices = np.linspace(0, len(flat) - 1, min(draws_per_set, len(flat)), dtype=int)
        selected[label] = {"parameters": flat[indices].copy(), "source_indices": indices,
                           "source_shape": tuple(draws.shape)}
    if sum(len(item["parameters"]) for item in selected.values()) > 128:
        raise ValueError("at most 128 posterior rows may be priced in one illustration")
    truth_row = None if truth is None else np.asarray(truth, dtype=float)
    if truth_row is not None and (truth_row.shape != (4,) or not np.isfinite(truth_row).all()
                                   or np.any(truth_row < bounds[:, 0]) or np.any(truth_row > bounds[:, 1])):
        raise ValueError("synthetic truth must be one four-target row inside this task's prior")
    rng = np.random.default_rng(seed)
    for item in selected.values():
        means = np.asarray([heston_smile(task, row) for row in item["parameters"]])
        if means.shape != (len(item["parameters"]), 63) or not np.isfinite(means).all():
            raise ValueError("invalid Fourier mean-price panel")
        noise = rng.standard_normal(means.shape)
        item.update(mean_prices=means, standard_normal_noise=noise, replicated_quotes=means + sigma * noise)
    contracts = task["context"]["contracts"]
    return {"observations": observed.copy(), "quote_noise_std": sigma.copy(),
            "strikes": np.asarray([c["strike"] for c in contracts]),
            "maturities_years": np.asarray([c["maturity"] for c in contracts]),
            "option_types": np.asarray([c["option_type"] for c in contracts]),
            "truth_parameters": None if truth_row is None else truth_row.copy(),
            "truth_prices": None if truth_row is None else heston_smile(task, truth_row),
            "draw_sets": selected, "seed": int(seed),
            "scope": "Descriptive posterior-predictive illustration for this exact synthetic observation. Fourier mean-price variation and new quote noise are separate; a small panel does not establish calibration, model adequacy or reference accuracy."}
