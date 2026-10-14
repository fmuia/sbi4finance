"""Exact answers for the tutorial's first BayesFlow example: drift and volatility from daily returns.

Daily log returns in percent are independent normals,

    y_t = drift / 252 + (100 vol / sqrt(252)) eps_t,      eps_t ~ N(0, 1),

with drift in % per year and vol the annualized volatility as a fraction. The prior matches the first
two coordinates of the SVJ prior: drift ~ N(8, 10²) and log vol ~ N(log 0.25, 0.40²). The likelihood
depends on the data only through the sample mean and variance, so the exact posterior of (drift, log vol)
is a cheap two-dimensional grid, and the predictive distribution of an h-day return is a mixture of
normals. The lesson checks the amortized posterior against both.

Losses are percent log losses, minus the h-day log return in percent: VaR is minus the 1% quantile of
the return, and ES minus the mean return below its 2.5% quantile.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import norm

DAYS_PER_YEAR = 252
WINDOW = 1_000
DRIFT_PRIOR = (8.0, 10.0)  # mean and sd, % per year
LOG_VOL_PRIOR = (float(np.log(0.25)), 0.40)  # mean and sd of log annualized volatility
VAR_LEVEL, ES_LEVEL = 0.99, 0.975


def daily_moments(drift, log_vol):
    """Mean and standard deviation of one day's log return in %."""
    return np.asarray(drift) / DAYS_PER_YEAR, 100 * np.exp(np.asarray(log_vol)) / np.sqrt(DAYS_PER_YEAR)


def simulate_returns(drift, log_vol, n_days, rng):
    """Daily log returns in % for each (drift, log_vol) pair: shape (..., n_days)."""
    mean, sd = daily_moments(drift, log_vol)
    mean, sd = np.asarray(mean)[..., None], np.asarray(sd)[..., None]
    return mean + sd * rng.standard_normal(np.broadcast_shapes(mean.shape, sd.shape)[:-1] + (n_days,))


def exact_posterior(returns, n_grid=201, width=7.0):
    """Posterior of (drift, log vol) on a grid centred on the data, `width` likelihood sd either side.

    Returns a dict with the grid axes `drift` and `log_vol` (n_grid,), the probabilities `weights`
    (n_grid, n_grid) indexed [drift, log_vol], and the posterior `mean` and `sd` of the two coordinates.
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    mean, sd = returns.mean(), returns.std()
    drift_axis = DAYS_PER_YEAR * (mean + width * sd / np.sqrt(n) * np.linspace(-1, 1, n_grid))
    log_vol_hat = np.log(sd * np.sqrt(DAYS_PER_YEAR) / 100)
    log_vol_axis = log_vol_hat + width / np.sqrt(2 * n) * np.linspace(-1, 1, n_grid)
    drift, log_vol = np.meshgrid(drift_axis, log_vol_axis, indexing="ij")
    m, s = daily_moments(drift, log_vol)
    squares = ((returns - mean) ** 2).sum()
    log_post = (
        -n * np.log(s) - (squares + n * (mean - m) ** 2) / (2 * s**2)
        + norm.logpdf(drift, *DRIFT_PRIOR) + norm.logpdf(log_vol, *LOG_VOL_PRIOR)
    )
    weights = np.exp(log_post - logsumexp(log_post))
    post_mean = np.array([(weights * drift).sum(), (weights * log_vol).sum()])
    post_sd = np.sqrt(np.array([(weights * drift**2).sum(), (weights * log_vol**2).sum()]) - post_mean**2)
    return {"drift": drift_axis, "log_vol": log_vol_axis, "weights": weights, "mean": post_mean, "sd": post_sd}


def sample_posterior(posterior, n, rng):
    """Draws of (drift, log vol) from the grid posterior, spread uniformly within each grid cell."""
    weights = posterior["weights"].ravel()
    cells = rng.choice(weights.size, size=n, p=weights / weights.sum())
    i, j = np.unravel_index(cells, posterior["weights"].shape)
    step_d = posterior["drift"][1] - posterior["drift"][0]
    step_v = posterior["log_vol"][1] - posterior["log_vol"][0]
    return np.stack([
        posterior["drift"][i] + step_d * (rng.random(n) - 0.5),
        posterior["log_vol"][j] + step_v * (rng.random(n) - 0.5),
    ], axis=-1)


def horizon_moments(drift, log_vol, horizon):
    """Mean and sd of the `horizon`-day log return in %, given the parameters."""
    mean, sd = daily_moments(drift, log_vol)
    return horizon * mean, np.sqrt(horizon) * sd


def mixture_var_es(means, sds, weights=None, var_level=VAR_LEVEL, es_level=ES_LEVEL):
    """VaR and ES (percent log loss) of a return distributed as a mixture of normals."""
    means, sds = np.ravel(means).astype(float), np.ravel(sds).astype(float)
    weights = np.full(len(means), 1 / len(means)) if weights is None else np.ravel(weights) / np.sum(weights)

    def quantile(prob):
        cdf = lambda x: weights @ norm.cdf((x - means) / sds) - prob
        lo, hi = (means - 12 * sds).min(), (means + 12 * sds).max()
        return brentq(cdf, lo, hi, xtol=1e-10)

    alpha = 1 - es_level
    q_es = quantile(alpha)
    z = (q_es - means) / sds
    tail_mean = weights @ (means * norm.cdf(z) - sds * norm.pdf(z)) / alpha
    return -quantile(1 - var_level), -tail_mean


def predictive_var_es(drift, log_vol, horizon, weights=None):
    """Predictive VaR and ES of the h-day return: the mixture over posterior draws or grid points."""
    mean, sd = horizon_moments(drift, log_vol, horizon)
    return mixture_var_es(mean, sd, weights)


def exact_predictive_var_es(posterior, horizon):
    """Predictive VaR and ES under the exact grid posterior."""
    drift, log_vol = np.meshgrid(posterior["drift"], posterior["log_vol"], indexing="ij")
    keep = posterior["weights"] > 1e-12 * posterior["weights"].max()
    return predictive_var_es(drift[keep], log_vol[keep], horizon, posterior["weights"][keep])


def save_network(approximator, path):
    """Save a trained BayesFlow approximator without optimizer state: inference needs only the weights."""
    optimizer, compiled = getattr(approximator, "optimizer", None), approximator.compiled
    try:
        approximator.optimizer = None
        approximator.compiled = False
        approximator.save(path)
    finally:
        approximator.optimizer = optimizer
        approximator.compiled = compiled
