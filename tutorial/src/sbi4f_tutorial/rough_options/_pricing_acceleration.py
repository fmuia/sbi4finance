"""Experimental terminal-moment acceleration for conditional rough pricing.

The public simulator remains the reference. This opt-in helper preserves its
three-channel innovations and left-point reductions, reuses the linear driver
for antithetic paths, and compiles only the terminal integration loops. It does
not change the discretization, path allocation, estimator, or fitted controls.
"""
from functools import lru_cache

import numpy as np

from .model import _model_parameters, _normals, _scalar, curve_values
from .simulator import rough_driver, simulate_rough


def _terminal_moments(driver, driver_variance, z, xi, dt, eta, indices):
    """Scalar left-to-right reductions; compiled without fastmath or threads."""
    m, n = driver.shape
    ivs = np.empty((2, m, len(indices)))
    iws = np.empty((2, m, len(indices)))
    maximum_variance = np.zeros(2)
    maximum_integrated = np.zeros(2)
    finite = True
    eta_squared = eta ** 2
    for side in range(2):
        sign = 1.0 if side == 0 else -1.0
        for path in range(m):
            iv, iw, selected = 0.0, 0.0, 0
            # The endpoint variance is not integrated, but the reference checks
            # it for overflow, so compute and check it here as well.
            for step in range(n + 1):
                d = 0.0 if step == 0 else sign * driver[path, step - 1]
                dv = 0.0 if step == 0 else driver_variance[step - 1]
                v = xi[step] * np.exp(eta * d - 0.5 * eta_squared * dv)
                if not np.isfinite(v):
                    finite = False
                if v > maximum_variance[side]:
                    maximum_variance[side] = v
                if step < n:
                    increment = v * dt
                    iv += increment
                    iw += np.sqrt(increment) * (sign * z[path, step, 0])
                    if not np.isfinite(iv) or not np.isfinite(iw):
                        finite = False
                    if iv > maximum_integrated[side]:
                        maximum_integrated[side] = iv
                    if selected < len(indices) and step + 1 == indices[selected]:
                        ivs[side, path, selected] = iv
                        iws[side, path, selected] = iw
                        selected += 1
    return ivs, iws, maximum_variance, maximum_integrated, finite


@lru_cache(maxsize=1)
def _compiled_kernel():
    try:
        from numba import njit
    except ImportError as error:
        raise RuntimeError("path_backend='numba' requires the optional acceleration dependency numba") from error
    return njit(cache=True, fastmath=False, parallel=False)(_terminal_moments)


def _certifies_omitted_paths(rho, z, dt, maximum_variance, maximum_integrated):
    """Use the reference conditional-only scalar overflow certificate."""
    with np.errstate(over="ignore", invalid="ignore"):
        normal_bound = max(np.max(z), -np.min(z))
        driver_bound = (abs(rho) + np.sqrt(1 - rho * rho)) * normal_bound
        drift_bound = 0.5 * maximum_variance
        return_bound = drift_bound * dt + np.sqrt(maximum_integrated) * driver_bound
        path_bound = z.shape[1] * return_bound
    bounds = np.asarray([driver_bound, drift_bound, return_bound, path_bound])
    return bool(np.isfinite(bounds).all() and np.max(bounds) < np.finfo(float).max / 8)


def antithetic_terminal_paths(parameters, curve, dt_years, innovations, indices):
    """Return compact IV/IW paths at sorted unique 1-based maturity indices.

    A single FFT produces both signs of the Gaussian driver. Uncertain overflow
    bounds fall back to the full reference, preserving its numerical support.
    There is no automatic fallback when the optional compiler is unavailable.
    """
    p = _model_parameters(parameters, "rough")
    dt = _scalar(dt_years, "dt_years")
    if dt <= 0:
        raise ValueError("dt_years must be positive")
    z = _normals(innovations)
    selected = np.asarray(indices)
    if (selected.ndim != 1 or not len(selected) or selected.dtype.kind not in "iu"
            or np.any(selected < 1) or np.any(selected > z.shape[1])):
        raise ValueError("sorted unique in-range integer maturity indices required")
    selected = selected.astype(np.int64)
    if np.any(np.diff(selected) <= 0):
        raise ValueError("sorted unique in-range integer maturity indices required")
    xi = curve_values(curve, dt * np.arange(z.shape[1] + 1))
    driver, driver_variance = rough_driver(p["hurst"], dt, z, method="hybrid")
    iv, iw, maximum_variance, maximum_integrated, finite = _compiled_kernel()(
        driver, driver_variance, z, xi, dt, p["eta"], selected)
    if finite and all(_certifies_omitted_paths(p["rho"], z, dt, v, i)
                      for v, i in zip(maximum_variance, maximum_integrated)):
        return tuple({"integrated_variance": iv[side], "volatility_driver_integral": iw[side]}
                     for side in range(2))
    # Preserve the reference's finite-trajectory checks and failure behavior for
    # extreme inputs, including the unused terminal variance and W-perp channel.
    result = []
    for sign in (1.0, -1.0):
        full = simulate_rough(p, curve, dt, sign * z, method="hybrid", conditional_only=True)
        result.append({key: value[:, selected - 1] for key, value in full.items()})
    return tuple(result)
