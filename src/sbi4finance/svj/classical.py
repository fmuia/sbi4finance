"""The classical baseline: approximate maximum likelihood with the grid filter, and Wald intervals.

The grid filter's log likelihood is deterministic and smooth in the parameters, so automatic
differentiation through all T days gives its exact gradient. L-BFGS-B maximizes it inside artificial
bounds at the prior mean +/- 6 prior sd, from several starts (the study uses the prior mean and the
amortized posterior median). The Hessian comes from central differences of the exact gradient, and the
Wald intervals are built in the unbounded coordinates, which favours them (Appendix D). The computation
runs in float64 under `jax.enable_x64`, so the rest of the package keeps its float32 defaults.
"""

import functools
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from . import model
from .filtering import _grid_loglik, model_params_jnp


def prior_bounds(width=6.0, prior=model.PRIOR):
    """Box bounds on the eight coordinates: prior mean +/- width prior standard deviations."""
    lo, hi = np.empty(model.N_PARAMS), np.empty(model.N_PARAMS)
    for i, mean, sd in prior.normal_coordinates():
        lo[i], hi[i] = mean - width * sd, mean + width * sd
    z = model.draw_coordinates(200_000, np.random.default_rng(0), prior)[:, 4]  # atanh(rho) has no closed-form moments
    lo[4], hi[4] = z.mean() - width * z.std(), z.mean() + width * z.std()
    return lo, hi


@functools.lru_cache(maxsize=None)
def _objective(n_cells):
    def negloglik(coords, y):
        return -_grid_loglik(model_params_jnp(coords), y, n_cells)

    return jax.jit(jax.value_and_grad(negloglik)), jax.jit(jax.grad(negloglik))


@dataclass
class MLEResult:
    coords: np.ndarray  # (8,) estimate in the unbounded coordinates
    loglik: float
    se: np.ndarray  # (8,), NaN when the Hessian is not positive definite
    lower: np.ndarray  # 95% Wald interval
    upper: np.ndarray
    on_bound: np.ndarray  # (8,) bool
    hessian_positive_definite: bool
    n_evaluations: int
    cpu_s: float


def fit_mle(returns, starts, n_cells=100, step=1e-4):
    """Approximate MLE of the eight coordinates from the given starts (k, 8); keeps the best optimum."""
    start_time = time.process_time()
    lo, hi = prior_bounds()
    value_and_grad, grad = _objective(n_cells)
    with jax.enable_x64(True):
        y = jnp.asarray(returns, jnp.float64)

        def fun(c):
            value, gradient = value_and_grad(jnp.asarray(c, jnp.float64), y)
            value, gradient = float(value), np.asarray(gradient, dtype=float)
            if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                return 1e12, np.zeros_like(gradient)
            return value, gradient

        best, evaluations = None, 0
        for start in np.atleast_2d(starts):
            result = minimize(fun, np.clip(start, lo, hi), jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)))
            evaluations += result.nfev
            if best is None or result.fun < best.fun:
                best = result
        theta = best.x
        hessian = np.empty((model.N_PARAMS, model.N_PARAMS))
        for j in range(model.N_PARAMS):
            e = np.zeros(model.N_PARAMS)
            e[j] = step
            plus = np.asarray(grad(jnp.asarray(np.minimum(theta + e, hi)), y), dtype=float)
            minus = np.asarray(grad(jnp.asarray(np.maximum(theta - e, lo)), y), dtype=float)
            hessian[:, j] = (plus - minus) / (np.minimum(theta + e, hi)[j] - np.maximum(theta - e, lo)[j])
    positive, se = wald_standard_errors(hessian)
    span = hi - lo
    return MLEResult(
        coords=theta,
        loglik=-float(best.fun),
        se=se,
        lower=theta - 1.96 * se,
        upper=theta + 1.96 * se,
        on_bound=(theta - lo < 1e-4 * span) | (hi - theta < 1e-4 * span),
        hessian_positive_definite=positive,
        n_evaluations=int(evaluations),
        cpu_s=time.process_time() - start_time,
    )


def wald_standard_errors(hessian, rtol=1e-10):
    """Standard errors from the Hessian of the negative log likelihood, or NaN when no Wald interval exists.

    The Hessian must be positive definite with its smallest eigenvalue above rtol times its largest; a
    numerically singular Hessian gives no interval, like an indefinite one.
    """
    hessian = 0.5 * (np.asarray(hessian) + np.asarray(hessian).T)
    nan = np.full(len(hessian), np.nan)
    if not np.all(np.isfinite(hessian)):
        return False, nan
    eigenvalues = np.linalg.eigvalsh(hessian)
    if eigenvalues.min() <= rtol * max(eigenvalues.max(), 0.0) or eigenvalues.max() <= 0:
        return False, nan
    try:
        return True, np.sqrt(np.diag(np.linalg.inv(hessian)))
    except np.linalg.LinAlgError:
        return False, nan


def prior_mean():
    """Starting point: the prior mean of each coordinate."""
    coords = np.array([mean for _, mean, _ in model.PRIOR.normal_coordinates()])
    out = np.empty(model.N_PARAMS)
    out[[i for i, _, _ in model.PRIOR.normal_coordinates()]] = coords
    out[4] = model.draw_coordinates(200_000, np.random.default_rng(0))[:, 4].mean()
    return out
