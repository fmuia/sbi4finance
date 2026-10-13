"""Particle marginal Metropolis-Hastings: the exact posterior of one window (Appendix E).

Each iteration proposes new coordinates with a Gaussian random walk, runs a fresh particle filter at
the proposal and accepts with the usual Metropolis-Hastings ratio, keeping the stored likelihood
estimate of the current point. Because the filter's estimate is unbiased, the chain targets the exact
posterior; keeping one final particle of each accepted filter run adds today's variance to it.

This is the reference the amortized posterior is checked against, and the fallback for windows the
audit flags. Chains start at amortized draws and use the amortized covariance for the proposal,
which favours particle MCMC, so its measured cost is a lower bound.
"""

import functools
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import betaln, ndtri
from scipy.stats import rankdata

from . import model
from .filtering import _particle_filter, choose_particle_count, loglik_sd, model_params_jnp, prng_key


def log_prior_jnp(coords, prior=model.PRIOR):
    """JAX twin of `model.log_prior` for one coordinate vector."""
    out = 0.0
    for i, mean, sd in prior.normal_coordinates():
        out = out - 0.5 * ((coords[i] - mean) / sd) ** 2 - np.log(sd) - 0.5 * np.log(2 * np.pi)
    log_u, log_1mu = jax.nn.log_sigmoid(2 * coords[4]), jax.nn.log_sigmoid(-2 * coords[4])
    out = out + (prior.rho_a - 1) * log_u + (prior.rho_b - 1) * log_1mu - betaln(prior.rho_a, prior.rho_b)
    return out + np.log(2) + log_u + log_1mu


@functools.lru_cache(maxsize=None)
def _chain_runner(n_particles):
    def run_chain(y, start, chol, init_key, step_keys):
        k_init, k_pick = jax.random.split(init_key)
        loglik, cloud = _particle_filter(model_params_jnp(start), y, k_init, n_particles)
        state = (start, log_prior_jnp(start), loglik, cloud[jax.random.randint(k_pick, (), 0, n_particles)])

        def iteration(state, key):
            coords, log_p, loglik, v_t = state
            k_step, k_filter, k_pick, k_accept = jax.random.split(key, 4)
            proposal = coords + chol @ jax.random.normal(k_step, coords.shape)
            log_p_new = log_prior_jnp(proposal)
            loglik_new, cloud = _particle_filter(model_params_jnp(proposal), y, k_filter, n_particles)
            v_new = cloud[jax.random.randint(k_pick, (), 0, n_particles)]
            log_ratio = log_p_new + loglik_new - log_p - loglik
            accept = (jnp.log(jax.random.uniform(k_accept)) < log_ratio) & jnp.isfinite(loglik_new)
            new = (proposal, log_p_new, loglik_new, v_new)
            state = jax.tree_util.tree_map(lambda a, b: jnp.where(accept, a, b), new, state)
            return state, (state[0], state[3], accept)

        return jax.lax.scan(iteration, state, step_keys)[1]

    def run(y, starts, chol, init_keys, step_keys):
        return jax.vmap(run_chain, in_axes=(None, 0, None, 0, 0))(y, starts, chol, init_keys, step_keys)

    return jax.jit(run)


@dataclass
class PMMHResult:
    draws: np.ndarray  # (n_chains, n_kept, 9) in target coordinates
    accept_rate: float
    rhat: np.ndarray  # (9,) rank-normalized split R-hat
    ess_bulk: np.ndarray  # (9,)
    n_particles: int
    loglik_sd: float
    seconds: float

    def pooled(self):
        return self.draws.reshape(-1, self.draws.shape[-1])


def run_pmmh(y, start_draws, seed=0, n_chains=4, n_iter=12_500, burn_in=2_000, n_particles=None, target_sd=1.3):
    """Particle MCMC started from amortized draws (n, 9) with proposal covariance 2.38²/8 x their covariance."""
    start_time = time.process_time()
    rng = np.random.default_rng(seed)
    coords = np.asarray(start_draws)[:, : model.N_PARAMS]
    centre = coords.mean(axis=0)
    if n_particles is None:
        n_particles, sd = choose_particle_count(model.model_params(centre), y, seed, target_sd=target_sd)
    else:
        sd = loglik_sd(model.model_params(centre), y, seed, n_particles)
    cov = np.cov(coords.T) * 2.38**2 / model.N_PARAMS
    chol = np.linalg.cholesky(cov + 1e-10 * np.eye(model.N_PARAMS))
    starts = coords[rng.choice(len(coords), n_chains, replace=False)]
    init_keys, chain_keys = jnp.split(jax.random.split(prng_key(seed), 2 * n_chains), 2)
    step_keys = jax.vmap(lambda k: jax.random.split(k, n_iter))(chain_keys)
    chain_coords, chain_v, accepted = _chain_runner(int(n_particles))(
        jnp.asarray(y, jnp.float32),
        jnp.asarray(starts, jnp.float32),
        jnp.asarray(chol, jnp.float32),
        init_keys,
        step_keys,
    )
    chain_coords, chain_v = np.asarray(chain_coords, float), np.asarray(chain_v, float)
    draws = np.concatenate([chain_coords, model.log_spot_vol(chain_v)[..., None]], axis=-1)[:, burn_in:]
    rhat, ess = zip(*(rhat_ess(draws[..., j]) for j in range(draws.shape[-1])))
    return PMMHResult(
        draws=draws,
        accept_rate=float(np.asarray(accepted).mean()),
        rhat=np.array(rhat),
        ess_bulk=np.array(ess),
        n_particles=int(n_particles),
        loglik_sd=float(sd),
        seconds=time.process_time() - start_time,
    )


def _rank_normalize(chains):
    ranks = rankdata(chains, method="average").reshape(chains.shape)
    return ndtri((ranks - 3 / 8) / (chains.size + 1 / 4))


def _split(chains):
    half = chains.shape[1] // 2
    return np.concatenate([chains[:, :half], chains[:, half : 2 * half]], axis=0)


def _rhat(chains):
    n = chains.shape[1]
    within = chains.var(axis=1, ddof=1).mean()
    between_over_n = chains.mean(axis=1).var(ddof=1)
    return np.sqrt(((n - 1) / n * within + between_over_n) / within)


def _ess(chains):
    m, n = chains.shape
    centred = chains - chains.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centred, n=2 * n, axis=1)
    acov = np.fft.irfft(spectrum * np.conj(spectrum), axis=1)[:, :n] / n
    within = (acov[:, 0] * n / (n - 1)).mean()
    var_plus = within * (n - 1) / n + chains.mean(axis=1).var(ddof=1)
    rho = 1 - (within - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    pairs = rho[:-1:2] + rho[1::2]  # Geyer's initial positive, monotone sequence
    positive = np.argmax(pairs < 0) if np.any(pairs < 0) else len(pairs)
    pairs = np.minimum.accumulate(pairs[:positive])
    tau = max(-1 + 2 * pairs.sum(), 1 / np.log10(m * n))
    return m * n / tau


def rhat_ess(chains):
    """Rank-normalized split R-hat (max of bulk and folded) and bulk ESS for one quantity, chains (m, n)."""
    split = _split(np.asarray(chains, dtype=float))
    bulk = _rank_normalize(split)
    folded = _rank_normalize(np.abs(split - np.median(split)))
    return float(max(_rhat(bulk), _rhat(folded))), float(_ess(bulk))
