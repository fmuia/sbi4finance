"""The classical route: weighted least squares, multi-start, Gauss–Newton intervals, Tikhonov (study S6).

The residuals are (P_c(theta) - y_c) / sigma_c, so least squares is maximum
likelihood under the quote-noise model and, with the uniform prior, the
posterior mode (design document, Appendix E). The search runs in the logit
coordinates of the prior box, so every iterate is a valid parameter; the
Gauss–Newton covariance is mapped back to natural units by the Jacobian of
the logit. A Monte Carlo pricer must be called with a fixed seed (common
random numbers), so that the objective is a deterministic function.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

from . import correction, tasks


def residual_function(task, observation, price_fn, *, z_reference=None, penalty=0.0):
    """r(z) = (P(theta(z)) - y) / sigma, with sqrt(penalty) (z - z_ref) appended for Tikhonov."""
    bounds = correction.bounds_of(task)
    y = np.asarray(observation, dtype=float).reshape(-1)
    sigma = tasks.sigma(task)

    def residuals(z):
        r = (price_fn(correction.to_bounded(z, bounds)) - y) / sigma
        if penalty > 0:
            r = np.concatenate([r, np.sqrt(penalty) * (z - z_reference)])
        return r

    return residuals


def gauss_newton_covariance(jacobian, z, bounds):
    """(J'J)^{-1} in logit coordinates, mapped to natural units: D (J'J)^{-1} D with D = d theta / d z."""
    information = jacobian.T @ jacobian
    try:
        covariance_z = np.linalg.inv(information)
    except np.linalg.LinAlgError:
        return None, np.inf
    unit = expit(z)
    d = (bounds[:, 1] - bounds[:, 0]) * unit * (1 - unit)
    return covariance_z * np.outer(d, d), float(np.linalg.cond(information))


def fit(task, observation, price_fn, starts, *, z_reference=None, penalty=0.0, diff_step=1e-4, max_nfev=200):
    """One least-squares fit per start; each returns its optimum, cost, Jacobian and Gauss–Newton covariance."""
    bounds = correction.bounds_of(task)
    residuals = residual_function(task, observation, price_fn, z_reference=z_reference, penalty=penalty)
    fits = []
    for start in np.atleast_2d(starts):
        result = least_squares(residuals, correction.to_unbounded(start, bounds), method="trf",
                               diff_step=diff_step, max_nfev=max_nfev, x_scale="jac")
        z = np.clip(result.x, -30, 30)
        covariance, condition = gauss_newton_covariance(result.jac[: len(tasks.sigma(task))], z, bounds)
        fits.append({"theta": correction.to_bounded(z, bounds), "z": z, "cost": float(result.cost),
                     "chi2": float(2 * result.cost), "covariance": covariance, "condition": condition,
                     "evaluations": int(result.nfev), "converged": bool(result.success),
                     "on_bound": bool(np.any(np.abs(z) > 8))})
    return fits


def distinct_modes(fits, *, chi2_tolerance=1.0, distance=0.05):
    """Optima within chi2_tolerance of the best, merged when closer than distance in prior-box units."""
    best = min(f["chi2"] for f in fits)
    near = sorted((f for f in fits if f["chi2"] <= best + chi2_tolerance), key=lambda f: f["chi2"])
    modes = []
    for f in near:
        unit = expit(f["z"])
        if all(np.max(np.abs(unit - expit(m["z"]))) > distance for m in modes):
            modes.append(f)
    return modes


def wald_interval(fit, level_z=1.6448536269514722):
    """Symmetric Gauss–Newton interval in natural units; None where the information matrix is singular."""
    if fit["covariance"] is None:
        return None
    sd = np.sqrt(np.maximum(np.diag(fit["covariance"]), 0))
    return np.stack([fit["theta"] - level_z * sd, fit["theta"] + level_z * sd], axis=1)


def latin_starts(task, count, seed):
    """Stratified starts over the prior box, one stratum per start in every coordinate."""
    bounds = correction.bounds_of(task)
    rng = np.random.default_rng(seed)
    unit = (np.stack([rng.permutation(count) for _ in bounds], axis=1) + rng.random((count, len(bounds)))) / count
    return bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
