"""Audit by importance sampling: check and correct the amortized posterior of one window.

A Student t law with 5 degrees of freedom is fitted to the network's draws of the eight parameter
coordinates, with its scale inflated by 1.2. Proposals from it are weighted by prior times the
particle filter's likelihood estimate over the proposal density, all in the unbounded coordinates.
Because the estimate is unbiased, the weighted sample converges to the exact posterior as the number
of proposals grows; one final particle per filter run adds today's variance. The efficiency,
ESS / M, is the audit score: a low score means the network is off for this window, or the weighted
sample rests on few effective draws, and the window should go to particle MCMC.
"""

import time
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_t

from . import model
from .filtering import choose_particle_count, loglik_batch, loglik_sd

DEFAULT_THRESHOLD = 0.02  # an open decision in the design document; the study saw 0.2% to 10.5%


@dataclass
class AuditResult:
    efficiency: float
    ess: float
    n_proposals: int
    n_particles: int
    loglik_sd: float
    posterior: np.ndarray  # (n_out, 9) resampled in proportion to the weights
    proposals: np.ndarray  # (M, 9): coordinates and log spot vol from one final particle
    log_weights: np.ndarray  # (M,) normalized
    seconds: float

    def flagged(self, threshold=DEFAULT_THRESHOLD):
        return self.efficiency < threshold

    def weighted_mean(self):
        return np.exp(self.log_weights) @ self.proposals


def fit_proposal(draws, df=5, scale_inflation=1.2):
    coords = np.asarray(draws)[:, : model.N_PARAMS]
    shape = np.cov(coords.T) * scale_inflation**2 + 1e-9 * np.eye(model.N_PARAMS)
    return multivariate_t(loc=coords.mean(axis=0), shape=shape, df=df)


def audit(
    draws,
    returns,
    seed=0,
    n_proposals=3_000,
    df=5,
    scale_inflation=1.2,
    n_particles=None,
    target_sd=1.3,
    n_out=1_000,
):
    """Importance-sampling audit of amortized draws (n, 9) for one window of returns (T,)."""
    start = time.process_time()
    rng = np.random.default_rng(seed)
    proposal = fit_proposal(draws, df, scale_inflation)
    if n_particles is None:
        n_particles, sd = choose_particle_count(model.model_params(proposal.loc), returns, seed, target_sd=target_sd)
    else:
        sd = loglik_sd(model.model_params(proposal.loc), returns, seed, n_particles)
    coords = proposal.rvs(size=n_proposals, random_state=rng)
    loglik, v_t = loglik_batch(model.model_params(coords), returns, seed + 1, n_particles)
    log_w = model.log_prior(coords) + loglik - proposal.logpdf(coords)
    log_w = np.where(np.isfinite(log_w), log_w, -np.inf)
    log_w -= logsumexp(log_w)
    w = np.exp(log_w)
    ess = 1.0 / np.sum(w**2)
    proposals = np.concatenate([coords, model.log_spot_vol(v_t)[:, None]], axis=1)
    picks = rng.choice(n_proposals, size=n_out, replace=True, p=w)
    return AuditResult(
        efficiency=float(ess / n_proposals),
        ess=float(ess),
        n_proposals=n_proposals,
        n_particles=int(n_particles),
        loglik_sd=float(sd),
        posterior=proposals[picks],
        proposals=proposals,
        log_weights=log_w,
        seconds=time.process_time() - start,
    )
