"""The audit: importance reweighting of network draws with an exact or Monte Carlo likelihood.

Generalized from hkde_correction (defensive mixture, self-normalized weights,
no clipping) and svj/audit (a Student t proposal fitted to the network's draws,
because a flow-matching network has no cheap density). Proposals live in the
logit coordinates of the prior box; densities carry the logit Jacobian, so the
weights are computed for the posterior in natural units.

The likelihood is the task's quote-noise model: independent Gaussians with
standard deviation sigma_c around the model prices. With a Monte Carlo pricer the
weights need an unbiased estimate of that likelihood, not of its logarithm
(importance sampling with an unbiased likelihood estimate targets the exact
posterior). If the estimated price is P_hat = P + s * zeta with zeta standard
normal, independent across quotes and of the quote noise, then
E[N(y; P_hat, sigma^2 - s^2)] = N(y; P, sigma^2) for s < sigma. The plug-in
N(y; P_hat, sigma^2) is instead unbiased for N(y; P, sigma^2 + s^2), and
N(y; P_hat, sigma^2 + s^2) for N(y; P, sigma^2 + 2 s^2): adding the Monte Carlo
variance counts it twice (design document, Appendix G.3).

Two conditions make the weights target the exact posterior. The Monte Carlo errors
must be independent across proposals: each proposal priced with its own seed. With
one seed for every proposal (common random numbers) the weights converge to the
posterior given that seed's error field, which is shifted, and narrower with
sigma^2 - s^2. And quotes priced on the same paths have correlated errors, so the
estimate must use their covariance S: E[N(y; P_hat, diag(sigma^2) - S)] =
N(y; P, diag(sigma^2)). gaussian_loglik_mc computes it; the caller supplies the
independent seeds.
"""

import time

import numpy as np
import scipy.linalg
from scipy.special import expit, logit, logsumexp
from scipy.stats import multivariate_t

from . import tasks

_TAG = 0x43414C43


def bounds_of(task):
    return np.asarray(task["prior_bounds"], dtype=np.float64)


def to_unbounded(parameters, bounds):
    unit = (np.asarray(parameters, dtype=float) - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    return logit(unit)


def to_bounded(z, bounds):
    return bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * expit(np.asarray(z, dtype=float))


def log_jacobian(z, bounds):
    """log |d theta / d z| for theta = lower + width * expit(z), summed over coordinates."""
    z = np.asarray(z, dtype=float)
    return np.sum(np.log(bounds[:, 1] - bounds[:, 0]) - np.logaddexp(0.0, z) - np.logaddexp(0.0, -z), axis=-1)


def log_prior(task):
    return -float(np.sum(np.log(np.diff(bounds_of(task), axis=1))))


def gaussian_loglik(task, observation, prices, mc_variance=None, max_ratio=0.9):
    """log of an unbiased estimate of p(y | theta) for each row of model prices [N, C].

    With exact prices (mc_variance None) this is the Gaussian quote-noise
    log-likelihood. With Monte Carlo prices whose errors have variance
    mc_variance [N, C], the Gaussian variance is sigma^2 - mc_variance, which
    makes exp(result) an unbiased estimate of the exact likelihood when the
    Monte Carlo errors are Gaussian and independent across quotes. Raises if a
    Monte Carlo standard error exceeds max_ratio * sigma: the estimate then has
    a variance too large to weight with.
    """
    sigma2 = tasks.sigma(task) ** 2
    prices = np.atleast_2d(np.asarray(prices, dtype=float))
    if mc_variance is None:
        variance = np.broadcast_to(sigma2, prices.shape)
    else:
        mc_variance = np.broadcast_to(np.asarray(mc_variance, dtype=float), prices.shape)
        if np.any(mc_variance < 0) or not np.isfinite(mc_variance).all():
            raise ValueError("Monte Carlo variances must be finite and non-negative")
        if np.any(mc_variance > max_ratio**2 * sigma2):
            worst = float(np.sqrt(np.max(mc_variance / sigma2)))
            raise ValueError(f"Monte Carlo SE / sigma = {worst:.3f} exceeds {max_ratio}: price with more paths")
        variance = sigma2 - mc_variance
    y = np.asarray(observation, dtype=float).reshape(-1)
    return -0.5 * np.sum((prices - y) ** 2 / variance + np.log(2 * np.pi * variance), axis=-1)


MAX_RELATIVE_EIGENVALUE = 0.5  # validity: diag(sigma^2) - S must stay a covariance (eigenvalue below 1)
MAX_RELATIVE_TRACE = 1.0       # efficiency: the log-likelihood estimate's variance is about sum (s_i / sigma_i)^2


def relative_mc_eigenvalue(sigma, covariance):
    """Largest eigenvalue of the Monte Carlo covariance in units of the quote noise: diag(1/sigma) S diag(1/sigma)."""
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    return float(np.linalg.eigvalsh(np.asarray(covariance, dtype=float) / np.outer(sigma, sigma))[-1])


def gaussian_loglik_mc(observation, prices, sigma, covariance, max_eigenvalue=MAX_RELATIVE_EIGENVALUE,
                       data_covariance=None):
    """log of an unbiased estimate of N(y; P, V) with V = diag(sigma^2) + data_covariance, from one Monte Carlo
    price vector.

    For P_hat = P + e with e ~ N(0, S), S the covariance of the price means (pricing.rough_surface returns it),
    E[N(y; P_hat, V - S)] = N(y; P, V), the convolution of two Gaussians. data_covariance is the error the data
    carry besides the quote noise: for a bank surface, the Monte Carlo error of its reference prices (the bank's
    pricing_covariance), since y = P_ref + noise; None for real quotes. Raises when the largest eigenvalue of S
    relative to diag(sigma^2) exceeds max_eigenvalue: the estimate is then too noisy to weight with.
    """
    y = np.asarray(observation, dtype=float).reshape(-1)
    p = np.asarray(prices, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    S = np.asarray(covariance, dtype=float)
    ratio = relative_mc_eigenvalue(sigma, S)
    if not np.isfinite(ratio) or ratio > max_eigenvalue:
        raise ValueError(f"Monte Carlo covariance / noise eigenvalue {ratio:.3f} exceeds {max_eigenvalue}: "
                         "price with more paths")
    V = np.diag(sigma**2) if data_covariance is None else np.diag(sigma**2) + np.asarray(data_covariance, float)
    chol = np.linalg.cholesky(V - S)
    r = scipy.linalg.solve_triangular(chol, y - p, lower=True)
    return float(-0.5 * r @ r - np.sum(np.log(np.diag(chol))) - 0.5 * len(y) * np.log(2 * np.pi))


def fit_proposal(draws, task, *, df=5, inflation=1.2):
    """Student t fitted to network draws [N, P] in logit coordinates."""
    z = to_unbounded(draws, bounds_of(task))
    if z.ndim != 2 or len(z) < 2 or not np.isfinite(z).all():
        raise ValueError("at least two finite network draws strictly inside the prior box required")
    shape = np.cov(z.T) * inflation**2 + 1e-9 * np.eye(z.shape[1])
    return multivariate_t(loc=z.mean(axis=0), shape=shape, df=df)


def propose(draws, task, num_samples, seed, *, prior_fraction=0.1, df=5, inflation=1.2):
    """IID proposals from (1 - f) t(draws) + f prior, and their log densities in natural units."""
    bounds = bounds_of(task)
    t = fit_proposal(draws, task, df=df, inflation=inflation)
    rng = np.random.default_rng(np.random.SeedSequence([_TAG, int(seed)]))
    from_prior = rng.random(num_samples) < prior_fraction
    z = np.atleast_2d(t.rvs(size=num_samples, random_state=rng))
    prior_rows = rng.uniform(bounds[:, 0], bounds[:, 1], (num_samples, len(bounds)))
    z = np.where(from_prior[:, None], to_unbounded(prior_rows, bounds), z)
    z = np.clip(z, -30, 30)  # keeps expit inside the open box in float64; |z| > 30 has density ~ 0
    theta = to_bounded(z, bounds)
    log_t = t.logpdf(z) - log_jacobian(z, bounds)  # density in natural units
    log_q = np.logaddexp(np.log1p(-prior_fraction) + log_t, np.log(prior_fraction) + log_prior(task))
    return theta, log_q, from_prior


def pareto_khat(log_weights):
    """Generalized Pareto shape of the largest weights (Zhang and Stephens 2009, as in PSIS).

    Below 0.5 the importance estimate is reliable, below 0.7 usable, above unreliable
    (Vehtari et al. 2024). Returns inf when the tail has fewer than five points.
    """
    x = np.asarray(log_weights, dtype=float)
    x = x - x.max()
    n = len(x)
    tail = int(np.ceil(min(0.2 * n, 3 * np.sqrt(n))))
    cutoff = max(np.sort(x)[-tail - 1], np.log(np.finfo(float).tiny))
    exceed = np.sort(np.exp(x[x > cutoff]) - np.exp(cutoff))
    m = len(exceed)
    if m <= 4:
        return np.inf
    prior_bs, prior_k = 3, 10
    m_est = 30 + int(m**0.5)
    b = 1 - np.sqrt(m_est / (np.arange(1, m_est + 1) - 0.5))
    b /= prior_bs * exceed[int(m / 4 + 0.5) - 1]
    b += 1 / exceed[-1]
    k = np.log1p(-b[:, None] * exceed).mean(axis=1)
    length = m * (np.log(-(b / k)) - k - 1)
    weights = 1 / np.exp(length - length[:, None]).sum(axis=1)
    keep = weights >= 10 * np.finfo(float).eps
    weights, b = weights[keep] / weights[keep].sum(), b[keep]
    b_post = np.sum(b * weights)
    k_post = np.log1p(-b_post * exceed).mean()
    return float((m * k_post + prior_k * 0.5) / (m + prior_k))


def correct(log_likelihood, log_q, task):
    """Self-normalized weights for prior x likelihood / proposal, with ESS and k-hat."""
    log_w = np.asarray(log_likelihood, dtype=float) + log_prior(task) - np.asarray(log_q, dtype=float)
    if np.isnan(log_w).any() or not np.isfinite(log_w).any():
        raise ValueError("importance weights failed: no finite weight")
    log_w = np.where(np.isfinite(log_w), log_w, -np.inf)
    log_norm = logsumexp(log_w)
    weights = np.exp(log_w - log_norm)
    ess = float(1.0 / np.sum(weights**2))
    finite = np.isfinite(log_w)
    return {"weights": weights, "ess": ess, "efficiency": ess / len(weights),
            "max_weight": float(weights.max()), "khat": pareto_khat(log_w[finite])}


def weighted_summary(parameters, weights, probabilities=(0.05, 0.5, 0.95)):
    """Weighted mean, sd and quantiles per column."""
    parameters = np.atleast_2d(np.asarray(parameters, dtype=float))
    weights = np.asarray(weights, dtype=float) / np.sum(weights)
    mean = weights @ parameters
    sd = np.sqrt(weights @ (parameters - mean) ** 2)
    quantiles = []
    for j in range(parameters.shape[1]):
        order = np.argsort(parameters[:, j])
        cdf = np.cumsum(weights[order])
        quantiles.append([float(parameters[order[min(np.searchsorted(cdf, p), len(cdf) - 1)], j])
                          for p in probabilities])
    return {"mean": mean, "sd": sd, "quantiles": np.array(quantiles), "probabilities": list(probabilities)}


def resample(parameters, weights, num_samples, seed):
    rng = np.random.default_rng(np.random.SeedSequence([_TAG, 1, int(seed)]))
    picks = rng.choice(len(weights), size=num_samples, replace=True, p=np.asarray(weights) / np.sum(weights))
    return np.asarray(parameters)[picks]


def audit(draws, observation, task, price_fn, *, num_proposals=2048, seed=0, prior_fraction=0.1,
          with_variance=False):
    """Reweight network draws for one surface. price_fn(rows) returns prices, or (prices, se) if with_variance."""
    start = time.process_time()
    theta, log_q, from_prior = propose(draws, task, num_proposals, seed, prior_fraction=prior_fraction)
    priced = price_fn(theta)
    prices, extra = (priced[0], priced[1] ** 2) if with_variance else (priced, None)
    log_lik = gaussian_loglik(task, observation, prices, extra)
    result = correct(log_lik, log_q, task)
    result.update(parameters=theta, log_likelihood=log_lik, from_prior=from_prior,
                  cpu_seconds=time.process_time() - start)
    return result


def random_walk_chain(log_target, start, scale, num_steps, seed):
    """Random-walk Metropolis in logit coordinates; log_target(z) includes the Jacobian."""
    rng = np.random.default_rng(np.random.SeedSequence([_TAG, 2, int(seed)]))
    z = np.array(start, dtype=float)
    current = log_target(z)
    chain = np.empty((num_steps, len(z)))
    accepted = 0
    chol = np.linalg.cholesky(np.atleast_2d(scale))
    for step in range(num_steps):
        proposal = z + chol @ rng.standard_normal(len(z))
        value = log_target(proposal)
        if np.log(rng.random()) < value - current:
            z, current = proposal, value
            accepted += 1
        chain[step] = z
    return chain, accepted / num_steps
