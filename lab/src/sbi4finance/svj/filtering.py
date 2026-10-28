"""Likelihood of the daily SVJ model: the fully adapted particle filter and the grid filter.

Both follow the filtering recursion over the latent variance (Appendix C of the SVJ design document, archived at 574b940).
Given yesterday's variance v, today's return is a two-component normal mixture,

    p(y_t | v) = (1 - lam) N(y_t; mu, v) + lam N(y_t; mu + mu_J, v + sigma_J²),

and given the return, today's variance is normal on each branch (no jump / jump) before the floor.
The particle filter weights particles by p(y_t | v), resamples systematically and draws each new
particle from its exact conditional law, so the particles stay equally weighted and the product of
the daily average weights is an unbiased estimate of the likelihood. The grid filter runs the same
recursion on a fixed grid in sqrt(v); it is deterministic and smooth in the parameters, which makes it
the reference for tests and the likelihood of the maximum-likelihood baseline.

Everything runs in JAX (float32) and is vectorized over parameter vectors with `jax.vmap`.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammainc, ndtr

from .model import DAYS_PER_YEAR, LAMBDA_MAX, N_PARAMS, V_MIN

LOG_2PI = float(np.log(2 * np.pi))


def model_params_jnp(coords):
    """JAX twin of `model.model_params`: coordinates (..., >=8) to the daily model's parameters."""
    drift, log_vol, log_h, log_c, z_rho, log_jpy, mean_jump, log_jsd = (coords[..., i] for i in range(N_PARAMS))
    theta = (100 * jnp.exp(log_vol)) ** 2 / DAYS_PER_YEAR
    kappa = jnp.log(2.0) / jnp.exp(log_h)
    sigma_v = jnp.exp(log_c) * jnp.sqrt(2 * kappa * theta)
    lam = jnp.minimum(jnp.exp(log_jpy) / DAYS_PER_YEAR, LAMBDA_MAX)
    return jnp.stack(
        [drift / DAYS_PER_YEAR, theta, kappa, sigma_v, jnp.tanh(z_rho), lam, mean_jump, jnp.exp(log_jsd)], axis=-1
    )


def _log_normal(y, mean, var):
    return -0.5 * (LOG_2PI + jnp.log(var) + (y - mean) ** 2 / var)


def _branches(params, v, y):
    """Mixture components of p(y | v) and the two normal laws of the next variance (before the floor)."""
    mu, theta, kappa, sigma_v, rho, lam, mu_j, sigma_j = (params[i] for i in range(N_PARAMS))
    sj2 = sigma_j**2
    log_no_jump = jnp.log1p(-lam) + _log_normal(y, mu, v)
    log_jump = jnp.log(lam) + _log_normal(y, mu + mu_j, v + sj2)
    drift = v + kappa * (theta - v)
    mean0 = drift + sigma_v * rho * (y - mu)
    var0 = sigma_v**2 * (1 - rho**2) * v
    jump_mean = mu_j + sj2 / (v + sj2) * (y - mu - mu_j)
    jump_var = sj2 * v / (v + sj2)
    mean1 = drift + sigma_v * rho * (y - mu - jump_mean)
    var1 = var0 + (sigma_v * rho) ** 2 * jump_var
    return log_no_jump, log_jump, mean0, var0, mean1, var1


def _stationary_draws(params, key, n):
    theta, kappa, sigma_v = params[1], params[2], params[3]
    c2 = sigma_v**2 / (2 * kappa * theta)
    return jnp.maximum(jax.random.gamma(key, 1 / c2, (n,)) * theta * c2, V_MIN)


def prng_key(seed):
    """Random keys for the filters: the XLA bit generator, markedly faster than threefry on CPU."""
    return jax.random.key(seed, impl="rbg")


def _particle_filter(params, y, key, n_particles):
    """One run of the fully adapted filter. Returns (log L-hat, final particles)."""
    params = params.astype(jnp.float32)
    key, init_key = jax.random.split(key)
    particles = _stationary_draws(params, init_key, n_particles)
    index = jnp.arange(n_particles)

    def day(carry, y_t):
        v, loglik, key = carry
        key, k_u, k_jump, k_v = jax.random.split(key, 4)
        log_no_jump, log_jump, _, _, _, _ = _branches(params, v, y_t)
        log_w = jnp.logaddexp(log_no_jump, log_jump)
        shift = jnp.max(log_w)
        w = jnp.exp(log_w - shift)
        total = jnp.sum(w)
        loglik = loglik + shift + jnp.log(total / n_particles)
        # systematic resampling with one uniform U: particle j gets floor(N C_j - U) - floor(N C_{j-1} - U) offspring
        cumulative = jnp.floor(n_particles * jnp.cumsum(w) / total - jax.random.uniform(k_u)) + 1
        offspring = jnp.diff(jnp.clip(cumulative, 0, n_particles).astype(jnp.int32), prepend=0)
        v = v[jnp.repeat(index, offspring, total_repeat_length=n_particles)]
        # each offspring moves with the exact law of v_t given v_{t-1} and y_t: jump or not, then a normal step
        log_no_jump, log_jump, mean0, var0, mean1, var1 = _branches(params, v, y_t)
        jump = jax.random.uniform(k_jump, (n_particles,)) < jnp.exp(log_jump - jnp.logaddexp(log_no_jump, log_jump))
        mean = jnp.where(jump, mean1, mean0)
        sd = jnp.sqrt(jnp.where(jump, var1, var0))
        v_new = jnp.maximum(mean + sd * jax.random.normal(k_v, (n_particles,)), V_MIN)
        return (v_new, loglik, key), None

    (particles, loglik, _), _ = jax.lax.scan(day, (particles, jnp.float32(0.0), key), y.astype(jnp.float32))
    return loglik, particles


@functools.partial(jax.jit, static_argnames="n_particles")
def particle_filter(params, y, key, n_particles=100):
    """Log likelihood estimate and the cloud of final particles (draws of v_T | y, params)."""
    return _particle_filter(params, y, key, n_particles)


@functools.lru_cache(maxsize=None)
def _batched_filter(n_particles):
    def one(params, y, key):
        loglik, particles = _particle_filter(params, y, key, n_particles)
        pick = jax.random.randint(jax.random.fold_in(key, 7), (), 0, n_particles)
        return loglik, particles[pick]

    return jax.jit(jax.vmap(one, in_axes=(0, None, 0)))


def loglik_batch(params, y, seed, n_particles=100, chunk=500):
    """Particle-filter log likelihoods for many parameter vectors (M, 8 model parameters).

    Returns (loglik, v_T) as NumPy arrays of shape (M,): one estimate per vector and one final particle
    chosen at random, which is a draw of today's variance from the joint posterior once weighted by L-hat.
    """
    params = np.asarray(params, dtype=np.float32)
    chunk = min(chunk, len(params))
    keys = jax.random.split(prng_key(seed), -(-len(params) // chunk) * chunk)
    run = _batched_filter(int(n_particles))
    y = jnp.asarray(y, dtype=jnp.float32)
    logliks, v_last = [], []
    for start in range(0, len(params), chunk):
        stop = min(start + chunk, len(params))
        block = np.zeros((chunk, N_PARAMS), dtype=np.float32)
        block[: stop - start] = params[start:stop]
        block[stop - start :] = params[start]  # pad so every chunk has one compiled shape
        loglik, v_t = run(jnp.asarray(block), y, keys[start : start + chunk])
        logliks.append(np.asarray(loglik)[: stop - start])
        v_last.append(np.asarray(v_t)[: stop - start])
    loglik = np.concatenate(logliks).astype(float)
    loglik[~np.isfinite(loglik)] = -np.inf
    return loglik, np.concatenate(v_last).astype(float)


def filtered_variance(params, y, seed, n_particles=2_000):
    """Final particle cloud at one parameter vector: equally weighted draws of v_T | y_{1:T}, params."""
    loglik, particles = particle_filter(jnp.asarray(params, jnp.float32), jnp.asarray(y), prng_key(seed), n_particles)
    return float(loglik), np.asarray(particles, dtype=float)


def loglik_sd(params, y, seed, n_particles, n_replicates=32):
    """Standard deviation of log L-hat over independent filter runs at one parameter vector."""
    loglik, _ = loglik_batch(np.repeat(np.asarray(params)[None], n_replicates, 0), y, seed, n_particles, n_replicates)
    return float(np.std(loglik[np.isfinite(loglik)]))


def choose_particle_count(params, y, seed, candidates=(50, 100, 200, 400), target_sd=1.3, n_replicates=32):
    """Smallest particle count whose log-likelihood noise is below `target_sd` (the study's rule)."""
    sd = np.inf
    for n_particles in candidates:
        sd = loglik_sd(params, y, seed, n_particles, n_replicates)
        if sd < target_sd:
            return n_particles, sd
    return candidates[-1], sd


def _grid_loglik(params, y, n_cells):
    """Runs in the dtype of `params` and `y`: float32 for tests, float64 under jax.enable_x64 for the optimizer."""
    params = jnp.asarray(params)
    y = jnp.asarray(y, params.dtype)
    theta, kappa, sigma_v = params[1], params[2], params[3]
    c = sigma_v / jnp.sqrt(2 * kappa * theta)
    edges_sqrt = jnp.linspace(0.0, 1.0, n_cells + 1, dtype=params.dtype) * jnp.sqrt(theta * (2 + 6 * c))
    centres = (0.5 * (edges_sqrt[:-1] + edges_sqrt[1:])) ** 2
    inner = edges_sqrt[1:-1] ** 2
    shape, scale = 1 / c**2, theta * c**2
    cdf0 = jnp.concatenate([jnp.zeros(1, params.dtype), gammainc(shape, inner / scale), jnp.ones(1, params.dtype)])
    alpha = jnp.diff(cdf0)
    zeros = jnp.zeros((n_cells, 1), params.dtype)

    def cell_masses(mean, var):
        # mass of each cell under N(mean_j, var_j); the outer cells extend to -inf and +inf, which mirrors the floor.
        # The CDF is evaluated at the finite inner edges only, so gradients stay finite.
        z = (inner[None, :] - mean[:, None]) / jnp.sqrt(var)[:, None]
        lower = jnp.concatenate([zeros, ndtr(z), zeros + 1], axis=1)
        upper = jnp.concatenate([zeros + 1, ndtr(-z), zeros], axis=1)
        z_left = jnp.concatenate([zeros - 1, z], axis=1)  # left edge of each cell; -inf for the first
        return jnp.where(z_left > 0, upper[:, :-1] - upper[:, 1:], lower[:, 1:] - lower[:, :-1])

    def day(alpha, y_t):
        log_no_jump, log_jump, mean0, var0, mean1, var1 = _branches(params, centres, y_t)
        log_w = jnp.logaddexp(log_no_jump, log_jump)
        shift = jnp.max(log_w)
        w = alpha * jnp.exp(log_w - shift)
        total = jnp.sum(w)
        p_jump = jnp.exp(log_jump - log_w)
        transition = (1 - p_jump)[:, None] * cell_masses(mean0, var0) + p_jump[:, None] * cell_masses(mean1, var1)
        return (w / total) @ transition, shift + jnp.log(total)

    _, daily = jax.lax.scan(day, alpha, y)
    return jnp.sum(daily)


@functools.partial(jax.jit, static_argnames="n_cells")
def grid_loglik(params, y, n_cells=100):
    """Deterministic approximate log likelihood on a grid of n_cells cells in sqrt(v) (Appendix C.9)."""
    return _grid_loglik(params, y, n_cells)


@functools.lru_cache(maxsize=None)
def _batched_clouds(n_particles):
    return jax.jit(jax.vmap(lambda p, y, k: _particle_filter(p, y, k, n_particles), in_axes=(0, None, 0)))


def variance_clouds(params, y, seed, n_particles=1_000):
    """Final particle clouds at many parameter vectors (M, 8): draws of v_T | y, params, shape (M, N)."""
    keys = jax.random.split(prng_key(seed), len(params))
    loglik, clouds = _batched_clouds(int(n_particles))(jnp.asarray(params, jnp.float32), jnp.asarray(y, jnp.float32), keys)
    return np.asarray(loglik, dtype=float), np.asarray(clouds, dtype=float)
