"""Helpers for the tutorial's advanced SVJ lesson: the filter day by day, and the agreement study's series.

Nothing here changes the pipeline. `filtered_path` runs the fully adapted particle filter of `filtering`
with the same model equations, but records what it knows on each day: quantiles of today's spot volatility
and the probability that today's return contained a jump. `agreement_series` regenerates the four held-out
series of the agreement study (scripts/svj_agreement.py), so a lesson can audit one of them live and compare
with the particle-MCMC results stored in docs/results/svj-agreement.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm

from . import model
from .filtering import _branches, _stationary_draws, prng_key

AGREEMENT_SEED = 31  # the seed of scripts/svj_agreement.py
AGREEMENT_CASES = (
    "rare jumps, low vol-of-variance",
    "rare jumps, high vol-of-variance",
    "frequent jumps, low vol-of-variance",
    "frequent jumps, high vol-of-variance",
)


@functools.partial(jax.jit, static_argnames="n_particles")
def _path(params, y, key, probs, n_particles):
    params = params.astype(jnp.float32)
    key, init_key = jax.random.split(key)
    particles = _stationary_draws(params, init_key, n_particles)
    index = jnp.arange(n_particles)

    def day(carry, y_t):
        v, key = carry
        key, k_u, k_jump, k_v = jax.random.split(key, 4)
        log_no_jump, log_jump, _, _, _, _ = _branches(params, v, y_t)
        log_w = jnp.logaddexp(log_no_jump, log_jump)
        w = jnp.exp(log_w - jnp.max(log_w))
        total = jnp.sum(w)
        jump_prob = jnp.sum(w * jnp.exp(log_jump - log_w)) / total  # P(jump on day t | y_1..y_t)
        cumulative = jnp.floor(n_particles * jnp.cumsum(w) / total - jax.random.uniform(k_u)) + 1
        offspring = jnp.diff(jnp.clip(cumulative, 0, n_particles).astype(jnp.int32), prepend=0)
        v = v[jnp.repeat(index, offspring, total_repeat_length=n_particles)]
        log_no_jump, log_jump, mean0, var0, mean1, var1 = _branches(params, v, y_t)
        jump = jax.random.uniform(k_jump, (n_particles,)) < jnp.exp(log_jump - jnp.logaddexp(log_no_jump, log_jump))
        mean = jnp.where(jump, mean1, mean0)
        sd = jnp.sqrt(jnp.where(jump, var1, var0))
        v_new = jnp.maximum(mean + sd * jax.random.normal(k_v, (n_particles,)), model.V_MIN)
        return (v_new, key), (jnp.quantile(v_new, probs), jump_prob)

    _, (quantiles, jump_prob) = jax.lax.scan(day, (particles, key), y.astype(jnp.float32))
    return quantiles, jump_prob


def filtered_path(params, returns, seed=0, n_particles=1_000, probs=(0.05, 0.5, 0.95)):
    """What the filter knows on each day, at one vector of model parameters (`model.model_params` units).

    Returns (spot_vol, jump_prob): spot_vol (T, len(probs)) holds quantiles of today's annualized spot
    volatility given the returns up to today, and jump_prob (T,) the probability that today's return
    contained a jump, given the returns up to today.
    """
    quantiles, jump_prob = _path(
        jnp.asarray(params, jnp.float32), jnp.asarray(returns, jnp.float32), prng_key(seed),
        jnp.asarray(probs, jnp.float32), int(n_particles),
    )
    return model.spot_vol_from_variance(np.asarray(quantiles, dtype=float)), np.asarray(jump_prob, dtype=float)


def agreement_series(k):
    """Series k (0 to 3) of the agreement study: (returns (T,), true targets (9,), name).

    Replays the draws of scripts/svj_agreement.py: four prior draws with the jump frequency and the
    CV of variance set to their 15% or 85% prior quantiles, then one simulation per case, in order.
    """
    if k not in range(len(AGREEMENT_CASES)):
        raise ValueError(f"k must be 0 to {len(AGREEMENT_CASES) - 1}")
    rng = np.random.default_rng(AGREEMENT_SEED)
    z = norm.ppf(0.85)
    prior = model.PRIOR
    cases = []
    for jumps in ("rare", "frequent"):
        for vol_of_var in ("low", "high"):
            coords = model.draw_coordinates(1, rng)[0]
            coords[5] = np.log(prior.jumps_per_year) + (z if jumps == "frequent" else -z) * prior.jumps_per_year_sd
            coords[3] = np.log(prior.cv) + (z if vol_of_var == "high" else -z) * prior.cv_sd
            cases.append(coords)
    for i, coords in enumerate(cases):
        sims = model.simulate_from_coordinates(coords[None], rng)
        if i == k:
            return sims["returns"][0], sims["targets"][0], AGREEMENT_CASES[k]
