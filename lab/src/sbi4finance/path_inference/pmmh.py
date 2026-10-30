"""Bounded four-parameter Heston particle marginal MH feasibility sampler.

The target is the bank's signed full-truncation *discrete* P model. The
one-step adapted filter or a substep bootstrap filter supplies a nonnegative
unbiased likelihood estimate, including zero-integrated-variance return atoms.
Its accepted random seed and likelihood estimate are part of the Markov state
and survive rejection.
This specialized method is not PyMC/NUTS or an accepted numerical reference.
"""

from dataclasses import dataclass
import math

import numpy as np
from numba import njit
from scipy.special import expit

from sbi4finance.path_inference.bank import validate_task
from sbi4finance.path_inference.reference import _filter


def validate_reference_task(task):
    validate_task(task)
    if task["model"] != "heston":
        raise ValueError("PMMH supports only the four-parameter Heston path task")


def prior_logit_to_physical(z, task):
    """Inverse prior CDF after sigmoid, in the bank's parameter order."""
    z = np.asarray(z, dtype=np.float64)
    bounds = np.asarray(task["prior_bounds"], dtype=np.float64)
    if z.shape != (len(bounds),) or not np.isfinite(z).all():
        raise ValueError("z must be a finite vector in task parameter order")
    u = expit(z)
    low, high = bounds.T
    values = low + (high - low) * u
    logarithmic = np.asarray(task["prior_types"]) == "log_uniform"
    values[logarithmic] = np.exp(np.log(low[logarithmic])
                               + u[logarithmic] * np.log(high[logarithmic] / low[logarithmic]))
    return values


def physical_to_prior_logit(values, task):
    values = np.asarray(values, dtype=np.float64)
    bounds = np.asarray(task["prior_bounds"], dtype=np.float64)
    if (values.shape != (len(bounds),) or not np.isfinite(values).all()
            or np.any(values <= bounds[:, 0]) or np.any(values >= bounds[:, 1])):
        raise ValueError("physical values must be strictly inside the prior support")
    low, high = bounds.T
    u = (values - low) / (high - low)
    logarithmic = np.asarray(task["prior_types"]) == "log_uniform"
    u[logarithmic] = (np.log(values[logarithmic] / low[logarithmic])
                      / np.log(high[logarithmic] / low[logarithmic]))
    return np.log(u) - np.log1p(-u)


def log_prior_in_logits(z):
    """Prior density times inverse-CDF/logit Jacobian: product u*(1-u)."""
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 1 or not np.isfinite(z).all():
        return -np.inf
    return float(np.sum(-np.logaddexp(0.0, -z) - np.logaddexp(0.0, z)))


@dataclass(frozen=True)
class LikelihoodEstimate:
    log_value: float
    seed: int
    minimum_ess: float = np.nan
    minimum_compatible: int = -1
    collapse: bool = False


@njit(cache=True, fastmath=False)
def _bootstrap_filter(returns, dt, substeps, theta, initial, drift, kappa, xi, rho,
                      particles, random_seed):
    """Bootstrap variance-path proposal, integrating W-perp out per return.

    Each particle proposes every variance Brownian increment under the prior.
    Conditional on those increments, the observed daily return is Normal with
    mean sum((mu-v/2)h + rho*sqrt(v*h)*Zv) and variance
    (1-rho²)*sum(v*h). Reweight/resample terminal SIGNED states. This proposal
    is different from the adapted one-step filter but has the same likelihood
    expectation when substeps=1. Numba's seeded RNG avoids O(T*L*N) buffers.
    """
    np.random.seed(random_seed)
    state = np.full(particles, initial)
    next_state = np.empty(particles)
    proposed = np.empty(particles)
    log_weights = np.empty(particles)
    cumulative = np.empty(particles)
    log_likelihood = 0.0
    minimum_ess = float(particles)
    minimum_compatible = particles
    h = dt / substeps
    atom_value = 0.0
    for step in range(substeps):
        atom_value += drift * h
    independent_fraction = 1 - rho * rho
    for t in range(len(returns)):
        observed = returns[t]
        observed_atom = observed == atom_value
        maximum = -np.inf
        compatible = 0
        for i in range(particles):
            value = state[i]
            mean, integrated_variance = 0.0, 0.0
            for step in range(substeps):
                positive = max(value, 0.0)
                scale = math.sqrt(positive * h)
                normal = np.random.standard_normal()
                mean += (drift - 0.5 * positive) * h + rho * scale * normal
                integrated_variance += positive * h
                value += kappa * (theta - positive) * h + xi * scale * normal
            if (not math.isfinite(value) or not math.isfinite(mean)
                    or not math.isfinite(integrated_variance)):
                return -np.inf, state, minimum_ess, minimum_compatible, t, 2
            proposed[i] = value
            log_weight = -np.inf
            if observed_atom:
                # The dominating measure includes the deterministic daily
                # zero-integral return atom; positive-density pieces have zero
                # probability at that atom and must receive zero weight.
                if integrated_variance == 0.0:
                    log_weight = 0.0
                    compatible += 1
            elif integrated_variance > 0.0:
                variance = independent_fraction * integrated_variance
                residual = observed - mean
                log_weight = -0.5 * (math.log(2 * math.pi * variance)
                                      + residual * residual / variance)
                compatible += 1
            log_weights[i] = log_weight
            maximum = max(maximum, log_weight)
        minimum_compatible = min(minimum_compatible, compatible)
        if not math.isfinite(maximum):
            return -np.inf, state, minimum_ess, minimum_compatible, t, 1
        total, square = 0.0, 0.0
        for i in range(particles):
            weight = math.exp(log_weights[i] - maximum)
            cumulative[i] = weight
            total += weight
            square += weight * weight
        minimum_ess = min(minimum_ess, total * total / square)
        log_likelihood += maximum + math.log(total) - math.log(particles)
        running = 0.0
        for i in range(particles):
            running += cumulative[i] / total
            cumulative[i] = running
        cumulative[-1] = 1.0
        offset = np.random.random()
        ancestor = 0
        for i in range(particles):
            position = (offset + i) / particles
            while ancestor < particles - 1 and position >= cumulative[ancestor]:
                ancestor += 1
            next_state[i] = proposed[ancestor]
        state, next_state = next_state, state
    return log_likelihood, state, minimum_ess, minimum_compatible, -1, 0


def particle_log_likelihood(returns, task, physical, *, particles, seed, method="auto"):
    """One fresh likelihood estimate, with independently seeded PF inputs.

    A particle collapse is a zero likelihood estimate and is a valid rejected
    PMMH proposal. A nonfinite transition is an error, never silently recoded.
    """
    values = np.asarray(returns, dtype=np.float64)
    p = np.asarray(physical, dtype=np.float64)
    if (values.ndim != 1 or not values.size or not np.isfinite(values).all()
            or p.shape != (4,) or not np.isfinite(p).all()
            or isinstance(particles, bool) or not isinstance(particles, (int, np.integer))
            or particles < 2 or not 0 <= int(seed) < 2**64):
        raise ValueError("finite one-history returns, four parameters, >=2 particles and uint64 seed required")
    kappa, theta, xi, rho = p
    if kappa <= 0 or theta <= 0 or xi < 0 or abs(rho) >= 1:
        raise ValueError("invalid Heston parameters")
    substeps = task["numerics"]["substeps"]
    if method not in ("auto", "adapted", "bootstrap"):
        raise ValueError("method must be auto, adapted or bootstrap")
    if method == "adapted" and substeps != 1:
        raise ValueError("adapted filter is only valid with one substep")
    use_adapted = substeps == 1 and method != "bootstrap"
    identity = [int(seed) & 0xffffffff, int(seed) >> 32]
    if use_adapted:
        resampling = np.random.Generator(np.random.PCG64(np.random.SeedSequence([*identity, 0])))
        transition = np.random.Generator(np.random.PCG64(np.random.SeedSequence([*identity, 1])))
        uniforms = resampling.random(len(values))
        normals = transition.standard_normal((len(values), particles))
        dt = np.full(len(values), task["dt_years"])
        result = _filter(values, dt, theta, task["fixed"]["initial_variance"],
                         task["fixed"]["drift"], kappa, xi, rho, uniforms, normals)
    else:
        # The uint64 accepted seed reproducibly determines this internal uint32
        # Numba/MT19937 seed. It is retained in the extended PMMH state.
        rng_seed = int(np.random.SeedSequence([*identity, 2]).generate_state(1)[0])
        result = _bootstrap_filter(values, task["dt_years"], substeps, theta,
                                   task["fixed"]["initial_variance"], task["fixed"]["drift"],
                                   kappa, xi, rho, particles, rng_seed)
    likelihood, _, ess, compatible, failed_time, failure = result
    if failure == 2:
        raise FloatingPointError(f"nonfinite PF transition at observation {failed_time}")
    if np.isnan(likelihood) or likelihood == np.inf:
        raise FloatingPointError("particle filter returned invalid likelihood estimate")
    return LikelihoodEstimate(float(likelihood), int(seed), float(ess), int(compatible), failure == 1)


def run_chain(estimator, initial_z, *, tune=500, draws=1000, seed=0,
              proposal_scale=0.35, adapt=True, callback=None, initial_estimate=None):
    """Symmetric random-walk PMMH in prior-CDF logits.

    estimator(z, fresh_seed) returns a LikelihoodEstimate. Initialization must
    have a positive finite estimate. Proposal likelihoods are evaluated once;
    the current estimate is never refreshed after rejection. Scalar proposal
    adaptation is confined to discarded tuning iterations and then frozen.
    """
    for value, name, minimum in ((tune, "tune", 0), (draws, "draws", 1), (seed, "seed", 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not np.isfinite(proposal_scale) or proposal_scale <= 0:
        raise ValueError("proposal_scale must be finite and positive")
    z = np.array(initial_z, dtype=np.float64, copy=True)
    if z.ndim != 1 or not z.size or not np.isfinite(z).all():
        raise ValueError("initial_z must be a finite vector")
    rng = np.random.Generator(np.random.PCG64(seed))
    next_seed = lambda: int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
    # A screened finite initial estimate may be supplied as part of the initial
    # extended state. Re-estimating it here could destroy an atom-compatible
    # initialization. It is never refreshed during a rejected update.
    current = estimator(z, next_seed()) if initial_estimate is None else initial_estimate
    if not np.isfinite(current.log_value):
        raise RuntimeError("initial likelihood estimate is zero/nonfinite; choose a different declared initialization")
    prior = log_prior_in_logits(z)
    count = tune + draws
    arrays = {"z": np.empty((count, len(z))), "log_likelihood_estimate": np.empty(count),
              "likelihood_seed": np.empty(count, dtype=np.uint64),
              "proposal_likelihood_seed": np.empty(count, dtype=np.uint64),
              "accepted": np.zeros(count, dtype=bool), "proposal_scale": np.empty(count),
              "minimum_ess": np.empty(count), "minimum_compatible": np.empty(count, dtype=np.int64),
              "proposal_collapsed": np.zeros(count, dtype=bool)}
    log_scale = float(np.log(proposal_scale))
    initial = {"z": z.tolist(), "log_likelihood_estimate": current.log_value,
               "likelihood_seed": current.seed, "proposal_scale": proposal_scale}
    for iteration in range(count):
        scale = float(np.exp(log_scale))
        candidate_z = z + scale * rng.standard_normal(len(z))
        candidate_seed = next_seed()
        candidate = estimator(candidate_z, candidate_seed)
        if np.isnan(candidate.log_value) or candidate.log_value == np.inf:
            raise FloatingPointError("invalid proposed likelihood estimate")
        candidate_prior = log_prior_in_logits(candidate_z)
        ratio = candidate.log_value + candidate_prior - current.log_value - prior
        accepted = bool(np.log(rng.random()) < min(0.0, ratio))
        if accepted:
            z, current, prior = candidate_z, candidate, candidate_prior
        arrays["z"][iteration] = z
        arrays["log_likelihood_estimate"][iteration] = current.log_value
        arrays["likelihood_seed"][iteration] = current.seed
        arrays["proposal_likelihood_seed"][iteration] = candidate_seed
        arrays["accepted"][iteration] = accepted
        arrays["proposal_scale"][iteration] = scale
        arrays["minimum_ess"][iteration] = current.minimum_ess
        arrays["minimum_compatible"][iteration] = current.minimum_compatible
        arrays["proposal_collapsed"][iteration] = candidate.collapse
        if adapt and iteration < tune:
            rate = min(0.05, (iteration + 10.0) ** -0.6)
            log_scale = float(np.clip(log_scale + rate * (float(accepted) - 0.234), np.log(0.01), np.log(3.0)))
        if callback is not None:
            callback(iteration + 1, arrays, {"rng_state": rng.bit_generator.state,
                     "current_z": z.tolist(), "current_likelihood_seed": current.seed,
                     "current_log_likelihood_estimate": current.log_value,
                     "proposal_scale_for_next_iteration": float(np.exp(log_scale))})
    return {"tune": {k: v[:tune] for k, v in arrays.items()},
            "draws": {k: v[tune:] for k, v in arrays.items()}, "initial": initial,
            "final_rng_state": rng.bit_generator.state,
            "final_proposal_scale": float(np.exp(log_scale)), "seed": int(seed)}
