"""From posterior draws to value-at-risk and expected shortfall.

The predictive distribution of the H-day return averages the model's forecast over the joint posterior
of the parameters and today's variance. It is simulated: from every posterior draw (theta, v_T) the daily
model runs H days forward with fresh shocks and jumps, and the pooled returns are draws from the
predictive law. The VaR is minus its (1 - level) quantile and the ES minus the average beyond the ES
quantile, both as positive losses in percent of the position.

The plug-in forecast fixes the parameters at a point (here the posterior median) and keeps only the
uncertainty about today's variance, filtered at that point. The conditional VaR, VaR(theta, v_T), is the
number an analyst would report if both were known; its posterior shows how uncertain the risk number is,
and the law of total variance splits that uncertainty into today's variance and the parameters.
"""

import numpy as np
from scipy.special import ndtr

from . import model
from .filtering import filtered_variance, variance_clouds

VAR_LEVEL = 0.99
ES_LEVEL = 0.975
HORIZONS = (1, 10)


def var_es(returns, var_level=VAR_LEVEL, es_level=ES_LEVEL):
    """VaR at var_level and ES at es_level of a return sample, as positive losses."""
    r = np.asarray(returns, dtype=float).ravel()
    q_es = np.quantile(r, 1 - es_level)
    return float(-np.quantile(r, 1 - var_level)), float(-r[r <= q_es].mean())


def horizon_returns(targets, horizons=HORIZONS, n_paths=20, seed=0):
    """Pooled H-day returns, {H: (m * n_paths,)}, simulated forward from each draw of (theta, v_T)."""
    paths = model.forward_returns(targets, max(horizons), n_paths, np.random.default_rng(seed))
    cumulative = paths.cumsum(axis=-1)
    return {h: cumulative[..., h - 1].ravel() for h in horizons}


def risk_numbers(targets, horizons=HORIZONS, n_paths=20, seed=0):
    """{H: {"var", "es"}} of the distribution simulated from the given draws."""
    pooled = horizon_returns(targets, horizons, n_paths, seed)
    return {h: dict(zip(("var", "es"), var_es(r))) for h, r in pooled.items()}


def plugin_draws(posterior, returns, seed=0, n_state=20_000, n_particles=2_000):
    """Draws for the plug-in forecast: parameters fixed at the posterior median, v_T from a filter at that point."""
    point = np.median(np.asarray(posterior)[:, : model.N_PARAMS], axis=0)
    _, cloud = filtered_variance(model.model_params(point), returns, seed, n_particles)
    v_t = np.random.default_rng(seed).choice(cloud, n_state)
    return np.column_stack([np.repeat(point[None], n_state, axis=0), model.log_spot_vol(v_t)])


def mixture_quantile(params, v, prob):
    """prob-quantile of the one-day return given parameters (m, 8) and v (m,), by bisection on the mixture CDF."""
    params = np.atleast_2d(params)
    mu, lam, mu_j, sigma_j = params[:, 0], params[:, 5], params[:, 6], params[:, 7]
    sd0, sd1 = np.sqrt(v), np.sqrt(v + sigma_j**2)
    lo = np.minimum(mu, mu + mu_j) - 12 * sd1
    hi = np.maximum(mu, mu + mu_j) + 12 * sd1
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cdf = (1 - lam) * ndtr((mid - mu) / sd0) + lam * ndtr((mid - mu - mu_j) / sd1)
        below = cdf < prob
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def conditional_var(targets, level=VAR_LEVEL):
    """One-day VaR(theta, v_T) for each draw (m, 9): minus the (1 - level) quantile of the return mixture."""
    targets = np.atleast_2d(targets)
    v_t = model.variance_from_log_spot_vol(targets[:, model.N_PARAMS])
    return -mixture_quantile(model.model_params(targets), v_t, 1 - level)


def variance_split(posterior, returns, seed=0, n_params=200, n_particles=1_000, level=VAR_LEVEL):
    """Shares of the posterior variance of the conditional VaR due to today's variance and to the parameters.

    The inner laws p(v_T | theta, y) come from a particle filter at each of n_params parameter draws.
    """
    rng = np.random.default_rng(seed)
    posterior = np.asarray(posterior)
    rows = rng.choice(len(posterior), size=min(n_params, len(posterior)), replace=False)
    params = model.model_params(posterior[rows, : model.N_PARAMS])
    _, clouds = variance_clouds(params, returns, seed, n_particles)
    n, k = clouds.shape
    cvar = -mixture_quantile(np.repeat(params, k, axis=0), clouds.ravel(), 1 - level).reshape(n, k)
    state = cvar.var(axis=1).mean()
    parameters = cvar.mean(axis=1).var()
    total = state + parameters
    return {"state_share": float(state / total), "parameter_share": float(parameters / total)}


def forecast(posterior, returns, seed=0, horizons=HORIZONS, n_paths=20, split=True):
    """Predictive and plug-in VaR / ES, and the posterior of the one-day conditional VaR, for one window."""
    predictive = risk_numbers(posterior, horizons, n_paths, seed)
    plugin = risk_numbers(plugin_draws(posterior, returns, seed), horizons, 1, seed + 1)
    cvar = conditional_var(posterior)
    out = {
        "predictive": predictive,
        "plugin_at_posterior_median": plugin,
        "conditional_var_1d": {
            "median": float(np.median(cvar)),
            "interval_90": [float(np.quantile(cvar, 0.05)), float(np.quantile(cvar, 0.95))],
            "prob_above_plugin_var": float(np.mean(cvar > plugin[1]["var"])) if 1 in plugin else None,
        },
    }
    if split:
        out["conditional_var_1d"].update(variance_split(posterior, returns, seed))
    return out
