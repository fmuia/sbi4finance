"""Bounded data-only starting points and proposal geometry for PyMC references.

This estimates transformed-space MAP locations and local proposal curvature.
It is neither a posterior approximation used as truth nor a sampler. The
physical uniform prior and PyMC's own Jacobian remain unchanged in sampling.
"""

import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit, logit


def transformed_objective(log_likelihood, bounds):
    """Negative log posterior in the bounded Uniform's logit value coordinates.

    Uniform log density -log(width) and the logit Jacobian log(width) cancel,
    leaving log_likelihood(theta(z)) + log(sigmoid(z) sigmoid(-z)).
    """
    bounds = np.asarray(bounds, dtype=float)

    def objective(z):
        z = np.asarray(z, dtype=float)
        if z.shape != (len(bounds),) or not np.isfinite(z).all():
            raise ValueError("one finite logit-coordinate vector is required")
        theta = bounds[:, 0] + np.diff(bounds, axis=1)[:, 0] * expit(z)
        value = float(log_likelihood(theta))
        if np.isnan(value) or value == np.inf:
            raise FloatingPointError("nonfinite deterministic log likelihood during initialization")
        return float(-value - np.sum(log_expit(z) + log_expit(-z)))

    return objective


def local_curvature(objective, center, step=0.01):
    """Central-difference Hessian and a regularized covariance, for proposals only."""
    center = np.asarray(center, dtype=float)
    dimension = len(center)
    axes = np.eye(dimension) * step
    value = objective(center)
    hessian = np.empty((dimension, dimension))
    for i in range(dimension):
        hessian[i, i] = (objective(center + axes[i]) - 2 * value + objective(center - axes[i])) / step**2
        for j in range(i):
            hessian[i, j] = hessian[j, i] = (
                objective(center + axes[i] + axes[j]) - objective(center + axes[i] - axes[j])
                - objective(center - axes[i] + axes[j]) + objective(center - axes[i] - axes[j])) / (4 * step**2)
    if not np.isfinite(hessian).all():
        raise ValueError("nonfinite local curvature; no proposal covariance is available")
    eigenvalues, vectors = np.linalg.eigh(hessian)
    precision_floor = max(1e-6, float(np.max(np.abs(eigenvalues))) * 1e-8)
    inverse = 1 / np.maximum(eigenvalues, precision_floor)
    covariance_eigenvalues = np.clip(inverse, 1e-6, 4.0)
    covariance = (vectors * covariance_eigenvalues) @ vectors.T
    return {"coordinates": "PyMC Uniform logit/value coordinates", "finite_difference_step": step,
            "hessian": hessian.tolist(), "raw_hessian_eigenvalues": eigenvalues.tolist(),
            "precision_eigenvalue_floor": precision_floor,
            "proposal_covariance_eigenvalue_bounds": [1e-6, 4.0],
            "proposal_covariance_eigenvalues": covariance_eigenvalues.tolist(),
            "regularization_applied": bool(np.any(eigenvalues <= precision_floor)
                                            or np.any(inverse != covariance_eigenvalues)),
            "covariance": covariance.tolist(),
            "scope": "local regularized proposal geometry; not posterior uncertainty or proof of unimodality"}


def data_fit_initialization(log_likelihood, bounds, chains, seed, *, fit_starts=None, fit_budget=2000):
    """Predeclared multistart Powell fits, then distinct dispersed chain starts.

    maxfev is split between fits, so optimizer requests never exceed fit_budget.
    Curvature and bounded start checks are counted separately. Every fit result
    remains visible, including failed termination and separated supported fits.
    """
    bounds = np.asarray(bounds, dtype=float)
    if (bounds.ndim != 2 or bounds.shape[1] != 2 or not len(bounds)
            or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1])
            or type(chains) is not int or chains < 2 or type(seed) is not int or seed < 0):
        raise ValueError("finite bounds, at least two chains and a nonnegative seed required")
    fit_starts = min(4, chains) if fit_starts is None else fit_starts
    if (type(fit_starts) is not int or not 2 <= fit_starts <= chains
            or type(fit_budget) is not int or fit_budget < 20 * fit_starts):
        raise ValueError("2..chains fit starts and at least 20 optimizer evaluations per fit required")
    started, cpu_started = time.perf_counter(), time.process_time()
    objective = transformed_objective(log_likelihood, bounds)
    fit_rng = np.random.default_rng(np.random.SeedSequence([0x464954, seed]))
    fractions = fit_rng.uniform(size=(fit_starts, len(bounds)))
    initial_z = np.clip(logit(fractions), -12, 12)
    evaluations = 0

    def counted(z):
        nonlocal evaluations
        evaluations += 1
        return objective(z)

    # This guard catches fresh random log-likelihood calls before fitting.
    first, repeated = objective(initial_z[0]), objective(initial_z[0])
    if not np.isfinite(first) or first != repeated:
        raise ValueError("data-only initialization requires a finite deterministic target")
    fits = []
    for index, z in enumerate(initial_z):
        quota = fit_budget // fit_starts + (index < fit_budget % fit_starts)
        before = evaluations
        result = minimize(counted, z, method="Powell", bounds=[(-12.0, 12.0)] * len(bounds),
                          options={"maxfev": quota, "xtol": 1e-4, "ftol": 1e-6})
        used = evaluations - before
        if used > quota:
            raise RuntimeError("optimizer exceeded its declared evaluation budget")
        fits.append({"fit": index, "initial_logit": z.tolist(),
                     "initial_parameters": (bounds[:, 0] + np.diff(bounds, axis=1)[:, 0] * expit(z)).tolist(),
                     "logit": result.x.tolist(), "negative_log_posterior_logit": float(result.fun),
                     "optimizer_success": bool(result.success), "message": str(result.message),
                     "function_evaluations": used, "max_function_evaluations": quota})
    finite = [fit for fit in fits if np.isfinite(fit["negative_log_posterior_logit"])]
    if not finite:
        raise ValueError("no finite data-only transformed MAP candidate was found")
    best = min(finite, key=lambda fit: fit["negative_log_posterior_logit"])
    center = np.asarray(best["logit"])
    curvature = local_curvature(objective, center)
    covariance = np.asarray(curvature["covariance"])
    factor = np.linalg.cholesky(covariance)
    supported = [fit for fit in finite
                 if fit["negative_log_posterior_logit"] <= best["negative_log_posterior_logit"] + 10]
    # Keep each separated supported fit represented instead of hiding it behind
    # the best location. A finite-budget fit location is not asserted to be a mode.
    representatives = [best]
    for fit in supported:
        if all(np.linalg.norm(np.linalg.solve(factor, np.asarray(fit["logit"]) - np.asarray(other["logit"]))) > 2
               for other in representatives):
            representatives.append(fit)
    jitter_rng = np.random.default_rng(np.random.SeedSequence([0x53544152, seed]))
    starts, details, start_checks = [], [], 0
    for chain in range(chains):
        fit = representatives[chain % len(representatives)]
        location = np.asarray(fit["logit"])
        for attempt in range(32):
            spread = 2.0 if attempt < 16 else 1.0
            candidate = location + spread * factor @ jitter_rng.standard_normal(len(bounds))
            target = objective(candidate)
            start_checks += 1
            theta = bounds[:, 0] + np.diff(bounds, axis=1)[:, 0] * expit(candidate)
            if (np.isfinite(target) and target <= best["negative_log_posterior_logit"] + 25
                    and np.all(theta > bounds[:, 0]) and np.all(theta < bounds[:, 1])
                    and all(not np.array_equal(theta, old) for old in starts)):
                starts.append(theta)
                details.append({"chain": chain, "fit": fit["fit"], "logit": candidate.tolist(),
                                "spread": spread, "attempts": attempt + 1,
                                "negative_log_posterior_logit": target})
                break
        else:
            raise ValueError(f"no finite, distinct data-supported start for chain {chain}")
    return {"kind": "data-fit", "bounds": bounds.tolist(), "seed": seed,
            "chain_starts": np.asarray(starts).tolist(), "chain_start_details": details,
            "start_protocol": "data-only transformed-space MAP fits plus seeded dispersed logit starts; no neural draws",
            "optimizer": "scipy.optimize.minimize Powell; transformed log posterior includes Uniform logit Jacobian",
            "optimization_logit_box": [-12.0, 12.0], "optimizer_box_limits_sampling_support": False,
            "fit_start_seed_protocol": "PCG64 SeedSequence([0x464954, seed])",
            "chain_start_seed_protocol": "PCG64 SeedSequence([0x53544152, seed])",
            "optimizer_evaluation_budget": fit_budget, "optimizer_evaluations": evaluations,
            "determinism_checks": 2, "curvature_evaluations": 1 + 2 * len(bounds)**2,
            "chain_start_checks": start_checks, "fits": fits, "best_fit": best["fit"],
            "supported_fit_max_objective_gap": 10.0,
            "separated_supported_fit_locations": [fit["fit"] for fit in representatives],
            "chain_start_max_objective_gap": 25.0, "curvature": curvature,
            "proposal": {"distribution": "PyMC MultivariateNormalProposal", "S_coordinates": "parameters_interval__",
                         "S": covariance.tolist(), "initial_scaling": 1.0, "tune": "scaling"},
            "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started,
            "scope": "initialization/proposal only; finite multistart search does not establish global mode coverage or posterior correctness"}
