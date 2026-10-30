"""Particle/quadrature reference for the one-step Heston theta/state task.

The conditional proposal is exact for the declared discrete full-truncation
scheme. Finite particle and theta-grid approximations still require convergence
checks. Inputs are observed returns and known task context, never latent truth.

The particle/quadrature kernel is recovered from heston_reference.py at project
commit dd001a3; its atom handling and signed-state transition are preserved.
"""

import math

import numpy as np
from numba import njit


def restricted_task(n_observations=252):
    """The restricted P control; theta denotes long-run variance, not kappa.

    This is deliberately separate from four/six-parameter path inference. Its
    custom particle/quadrature method is suitable for the return atom of the
    one-step discrete model and needs no differentiable likelihood. It is not
    PyMC, a continuous-time likelihood, or a rough-volatility reference.
    """
    return {
        "task_id": "heston_theta_state_v1", "measure": "P", "substeps": 1,
        "n_observations": _integer(n_observations, "n_observations", 1),
        "dt_years": 1.0 / 252, "initial_variance": 0.04,
        "theta_bounds": [0.02, 0.08], "prior": "log_uniform",
        "parameters": {"drift": 0.05, "kappa": 2.0,
                       "vol_of_vol": 0.2, "rho": -0.6},
    }


def posterior_summary(result):
    """Physical-coordinate theta summaries from the finite quadrature rule.

    Quantiles are interpolated at the midpoint CDF positions of the discrete
    masses, so their grid error must be included in the refinement comparison.
    """
    grid = np.asarray(result["theta_grid"], dtype=float)
    weights = np.asarray(result["grid_probabilities"], dtype=float)
    if (grid.ndim != 1 or weights.shape != grid.shape or len(grid) < 2
            or not np.isfinite(grid).all() or not np.isfinite(weights).all()
            or np.any(np.diff(grid) <= 0) or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1)):
        raise ValueError("require increasing finite grid and normalized nonnegative weights")
    mean = float(weights @ grid)
    sd = float(np.sqrt(weights @ (grid - mean) ** 2))
    positions = np.cumsum(weights) - 0.5 * weights
    quantiles = np.interp([0.05, 0.5, 0.95], positions, grid)
    return {"mean": mean, "sd": sd, "q05": float(quantiles[0]),
            "median": float(quantiles[1]), "q95": float(quantiles[2])}


def compare_posteriors(first, second):
    """Numerical differences; no acceptance claim follows from these alone."""
    a, b = posterior_summary(first), posterior_summary(second)
    grid_a, grid_b = (np.asarray(r["theta_grid"]) for r in (first, second))
    weights_a, weights_b = (np.asarray(r["grid_probabilities"]) for r in (first, second))
    locations = np.unique(np.r_[grid_a, grid_b])
    indices_a = np.searchsorted(grid_a, locations[:-1], side="right")
    indices_b = np.searchsorted(grid_b, locations[:-1], side="right")
    cdf_a = np.r_[0.0, np.cumsum(weights_a)][indices_a]
    cdf_b = np.r_[0.0, np.cumsum(weights_b)][indices_b]
    wasserstein = float(np.diff(locations) @ np.abs(cdf_a - cdf_b))
    scale = max(b["sd"], np.finfo(float).tiny)
    return {"absolute_mean_difference": abs(a["mean"] - b["mean"]),
            "mean_difference_in_second_sd": abs(a["mean"] - b["mean"]) / scale,
            "absolute_sd_difference": abs(a["sd"] - b["sd"]),
            "maximum_quantile_difference": max(abs(a[k] - b[k]) for k in ("q05", "median", "q95")),
            "wasserstein1_discrete": wasserstein,
            "wasserstein1_in_second_sd": wasserstein / scale}


@njit(cache=True, fastmath=False)
def _filter(returns, dt, theta, initial, drift, kappa, xi, rho, uniforms, normals):
    """Fully adapted auxiliary filter with systematic ancestor resampling."""
    times, particles = normals.shape
    state = np.full(particles, initial)
    next_state = np.empty(particles)
    log_weights = np.empty(particles)
    weights = np.empty(particles)
    cumulative = np.empty(particles)
    log_likelihood = 0.0
    minimum_ess = float(particles)
    minimum_compatible = particles
    independent_scale = math.sqrt(1.0 - rho * rho)
    for t in range(times):
        step, observed = dt[t], returns[t]
        atom = observed == drift * step
        maximum = -np.inf
        compatible = 0
        for i in range(particles):
            value = state[i]
            log_weight = -np.inf
            if atom:
                # Against Lebesgue + delta_(drift*dt), positive-variance
                # Gaussian components have zero mass at this observation.
                if value <= 0.0:
                    log_weight = 0.0
                    compatible += 1
            elif value > 0.0:
                residual = observed - (drift - 0.5 * value) * step
                standardized = residual / math.sqrt(value * step)
                log_weight = -0.5 * (math.log(2.0 * math.pi) + math.log(value)
                                      + math.log(step) + standardized * standardized)
                compatible += 1
            log_weights[i] = log_weight
            maximum = max(maximum, log_weight)
        minimum_compatible = min(minimum_compatible, compatible)
        if not math.isfinite(maximum):
            return -np.inf, state, minimum_ess, minimum_compatible, t, 1
        total, squared_total = 0.0, 0.0
        for i in range(particles):
            weight = math.exp(log_weights[i] - maximum)
            weights[i] = weight
            total += weight
            squared_total += weight * weight
        minimum_ess = min(minimum_ess, total * total / squared_total)
        # Previous particles are equally weighted after each adapted update.
        log_likelihood += maximum + math.log(total) - math.log(particles)
        running = 0.0
        for i in range(particles):
            running += weights[i] / total
            cumulative[i] = running
        cumulative[-1] = 1.0
        ancestor = 0
        for i in range(particles):
            position = (uniforms[t] + i) / particles
            # side='right' also skips zero-weight ancestors if the RNG gives 0.
            while ancestor < particles - 1 and position >= cumulative[ancestor]:
                ancestor += 1
            value = state[ancestor]
            if value > 0.0:
                residual = observed - (drift - 0.5 * value) * step
                mean = value + kappa * (theta - value) * step + xi * rho * residual
                scale = xi * math.sqrt(value * step) * independent_scale
                updated = mean + scale * normals[t, i]
            else:
                # Zero effective variance gives both a return atom and a
                # deterministic update of the signed auxiliary state.
                updated = value + kappa * theta * step
            if not math.isfinite(updated):
                return -np.inf, state, minimum_ess, minimum_compatible, t, 2
            next_state[i] = updated
        state, next_state = next_state, state
    return log_likelihood, state, minimum_ess, minimum_compatible, -1, 0


def _integer(value, name, minimum=0):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _scalar(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite scalar")
    result = float(value)
    if (not math.isfinite(result) or (positive and result <= 0)
            or (nonnegative and result < 0)):
        raise ValueError(f"invalid {name}")
    return result


def _inputs(log_returns, dt_years, task):
    if task.get("prior") != "log_uniform":
        raise ValueError("reference requires an explicitly declared log_uniform theta prior")
    if (task.get("task_id") != "heston_theta_state_v1" or task.get("measure") != "P"
            or _integer(task.get("substeps"), "substeps", 1) != 1):
        raise ValueError("reference requires the one-substep P heston_theta_state_v1 task")
    count = _integer(task.get("n_observations"), "n_observations", 1)
    observation_step = _scalar(task.get("dt_years"), "task dt_years", positive=True)
    values = np.asarray(log_returns, dtype=np.float64)
    if values.shape == (1, count, 1):
        values = values[0, :, 0]
    elif values.shape == (count, 1):
        values = values[:, 0]
    if values.shape != (count,) or not np.isfinite(values).all():
        raise ValueError("provide one finite history as [T], [T,1], or [1,T,1]")
    steps = np.asarray(dt_years, dtype=np.float64)
    if steps.ndim == 0:
        steps = np.full(count, steps.item())
    if (steps.shape != (count,) or not np.isfinite(steps).all()
            or not np.all(steps == observation_step)):
        raise ValueError("dt_years must match the task's fixed observation intervals")
    fixed = task.get("parameters", {})
    required = {"drift", "kappa", "vol_of_vol", "rho"}
    optional = {"jump_intensity", "jump_mean", "jump_std"}
    if not required.issubset(fixed) or set(fixed) - required - optional:
        raise ValueError("fixed parameters must specify drift, kappa, vol_of_vol and rho; theta is inferred")
    for name in optional:
        if _scalar(fixed.get(name, 0.0), name) != 0.0:
            raise ValueError("this reference supports no jumps")
    drift = _scalar(fixed["drift"], "drift")
    kappa = _scalar(fixed["kappa"], "kappa", positive=True)
    xi = _scalar(fixed["vol_of_vol"], "vol_of_vol", nonnegative=True)
    rho = _scalar(fixed["rho"], "rho")
    if abs(rho) > 1:
        raise ValueError("rho must lie in [-1,1]")
    initial = _scalar(task.get("initial_variance"), "initial_variance", nonnegative=True)
    bounds = np.asarray(task.get("theta_bounds"), dtype=np.float64)
    if (bounds.shape != (2,) or not np.isfinite(bounds).all()
            or bounds[0] <= 0 or bounds[1] <= bounds[0]):
        raise ValueError("theta_bounds must be finite with 0 < lower < upper")
    return np.ascontiguousarray(values), np.ascontiguousarray(steps), bounds, initial, drift, kappa, xi, rho


def reference_posterior(log_returns, dt_years, task, *, grid_size=48, particles=2048, seed=2026):
    """Approximate the joint posterior of theta and terminal signed auxiliary.

    The task declares a log-uniform theta prior on theta_bounds, a fixed known
    nonnegative initial variance and all other parameters. One full-truncation
    step per observation and no jumps are required. Exact observed atoms at
    drift*dt are handled without a variance floor or floating-point tolerance.

    Returns theta_grid[G], grid_probabilities[G], terminal_particles[G,N],
    log_likelihood[G], and diagnostics. Particles within a theta row have equal
    weight. For joint draws choose a theta index by grid_probabilities, then a
    terminal particle from that same row; do not sample the marginals separately.

    Each grid likelihood is a particle estimate; the quadrature posterior is
    not an exact posterior at finite G/N. Repeat seeds and refine both before
    using it to assess a neural approximation. Collapse raises RuntimeError;
    no grid point, history, or particle-filter run is silently replaced.
    """
    grid_size = _integer(grid_size, "grid_size", 2)
    particles = _integer(particles, "particles", 2)
    seed = _integer(seed, "seed")
    if seed >= 2**64:
        raise ValueError("seed must be below 2**64")
    values, steps, bounds, initial, drift, kappa, xi, rho = _inputs(log_returns, dt_years, task)
    nodes, quadrature_weights = np.polynomial.legendre.leggauss(grid_size)
    lower, upper = np.log(bounds)
    theta_grid = np.exp(lower + 0.5 * (nodes + 1.0) * (upper - lower))
    # Jacobian and log-uniform prior cancel, leaving the Legendre weights / 2.
    prior_weights = quadrature_weights / 2.0
    terminal = np.empty((grid_size, particles))
    log_likelihood = np.empty(grid_size)
    minimum_ess = np.empty(grid_size)
    minimum_compatible = np.empty(grid_size, dtype=np.int64)
    for index, theta in enumerate(theta_grid):
        identity = [seed & 0xffffffff, seed >> 32, index]
        resampling_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([*identity, 0])))
        transition_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([*identity, 1])))
        uniforms = resampling_rng.random(len(values))
        normals = transition_rng.standard_normal((len(values), particles))
        result = _filter(values, steps, theta, initial, drift, kappa, xi, rho, uniforms, normals)
        likelihood, states, ess, compatible, failed_time, failure = result
        if failure:
            reason = "no compatible finite particle weights" if failure == 1 else "nonfinite transition"
            raise RuntimeError(
                f"particle reference failed at theta grid index {index} (theta={theta:.17g}), "
                f"observation {failed_time}: {reason}; no posterior returned"
            )
        log_likelihood[index] = likelihood
        terminal[index] = states
        minimum_ess[index] = ess
        minimum_compatible[index] = compatible
    log_mass = log_likelihood + np.log(prior_weights)
    maximum = np.max(log_mass)
    mass = np.exp(log_mass - maximum)
    normalizer = mass.sum()
    probabilities = mass / normalizer
    return {
        "theta_grid": theta_grid,
        "grid_probabilities": probabilities,
        "terminal_particles": terminal,
        "log_likelihood": log_likelihood,
        "diagnostics": {
            "log_evidence": float(maximum + np.log(normalizer)),
            "atom_observations": int(np.count_nonzero(values == drift * steps)),
            "minimum_ancestor_ess_per_theta": minimum_ess.tolist(),
            "minimum_compatible_particles_per_theta": minimum_compatible.tolist(),
            "effective_grid_points": float(1.0 / np.sum(probabilities**2)),
            "grid_size": grid_size, "particles": particles, "seed": seed,
            "quadrature": "Gauss-Legendre in log(theta), log-uniform prior",
            "resampling": "systematic at every observation; terminal particles equally weighted",
            "scope": "finite-particle reference for the same one-step discrete model",
        },
    }
