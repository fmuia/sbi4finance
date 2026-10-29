"""Riemann--Liouville Volterra rough Bergomi, not fractional Brownian motion.

Y_t=sqrt(2H) integral_0^t (t-s)^(H-1/2)dW_s, Var(Y_t)=t^(2H).
The hybrid kappa=1 scheme treats the singular last cell exactly and older cells
with their average kernel. Its own covariance normalizes lognormal variance.
A small-grid independent Gaussian-covariance reference includes the same asset W.
"""

from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.linalg import toeplitz

from sbi4finance.bergomi import (_asset_paths, _model_parameters, _normals,
                               _positive_integer, _q_inputs, _scalar, curve_values)


def hybrid_weights(hurst, dt_years, n_steps):
    """Brownian-increment coefficients and independent exact last-cell loading."""
    h = _scalar(hurst, "hurst")
    dt = _scalar(dt_years, "dt_years")
    n = _positive_integer(n_steps, "n_steps")
    if not 0 < h < 0.5 or dt <= 0:
        raise ValueError("require 0 < hurst < .5 and dt_years > 0")
    cells = np.arange(n + 1, dtype=float)
    weights = dt ** h * np.sqrt(2 * h) * np.diff(cells ** (h + 0.5)) / (h + 0.5)
    bridge = dt ** h * np.sqrt(max(1 - 2 * h / (h + 0.5) ** 2, 0))
    return weights, bridge


def hybrid_covariance(hurst, dt_years, n_steps):
    """Exact covariance of the *discrete approximation*, for numerical checks.

    Returns C_YY and C_Y,dW on positive grid times; this is not the continuum
    covariance used by rough_covariance. The independent cell residual is reused
    only at its own endpoint in this one-cell hybrid approximation.
    """
    weights, bridge = hybrid_weights(hurst, dt_years, n_steps)
    matrix = toeplitz(weights, np.r_[weights[0], np.zeros(n_steps - 1)])
    return matrix @ matrix.T + bridge ** 2 * np.eye(n_steps), matrix * np.sqrt(dt_years)


@lru_cache(maxsize=16)
def _reference_factors(hurst, dt_years, n_steps):
    covariance, cross = rough_covariance(hurst, dt_years, n_steps)
    residual = covariance - cross @ cross.T / dt_years
    residual = (residual + residual.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(residual)
    if eigenvalues.min() < -1e-10 * covariance.diagonal().max():
        raise ValueError("reference covariance is not positive semidefinite at numerical precision")
    factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
    return cross / np.sqrt(dt_years), factor, covariance.diagonal().copy()


def rough_covariance(hurst, dt_years, n_steps):
    """Continuum Gaussian grid covariance and joint dependence with W increments.

    Off-diagonal integrals use independent adaptive quadrature after removing
    the endpoint singularity; diagonals and C_Y,dW use analytic formulae.
    Capped at 128 steps: an O(n^3) reference, not a production simulator.
    """
    h = _scalar(hurst, "hurst")
    dt = _scalar(dt_years, "dt_years")
    n = _positive_integer(n_steps, "n_steps")
    if not 0 < h < 0.5 or dt <= 0 or n > 128:
        raise ValueError("reference needs 0 < hurst < .5, positive dt and <=128 steps")
    times = dt * np.arange(1, n + 1)
    alpha = h - 0.5
    covariance = np.diag(times ** (2 * h))
    for i, t in enumerate(times):
        for j in range(i + 1, n):
            distance = times[j] - t
            integral = quad(lambda z: (distance + t * z ** (1 / (alpha + 1))) ** alpha,
                            0, 1, epsabs=2e-12, epsrel=2e-12)[0]
            covariance[i, j] = covariance[j, i] = 2 * h * t ** (alpha + 1) * integral / (alpha + 1)
    endpoints = dt * np.arange(n + 1)
    upper = np.maximum(times[:, None] - endpoints[None, :-1], 0)
    lower = np.maximum(times[:, None] - endpoints[None, 1:], 0)
    cross = np.sqrt(2 * h) * (upper ** (h + 0.5) - lower ** (h + 0.5)) / (h + 0.5)
    return covariance, cross


def rough_driver(hurst, dt_years, innovations, *, method="hybrid"):
    """Generate the Gaussian field coupled to innovations[...,0]'s W increments."""
    z = _normals(innovations)
    n = z.shape[1]
    if method == "hybrid":
        weights, bridge = hybrid_weights(hurst, dt_years, n)
        fft_size = 1 << (2 * n - 1).bit_length()
        convolution = np.fft.irfft(np.fft.rfft(z[..., 0], n=fft_size, axis=1)
                                   * np.fft.rfft(weights, n=fft_size), n=fft_size, axis=1)[:, :n]
        driver = convolution + bridge * z[..., 1]
        variance = np.cumsum(weights ** 2) + bridge ** 2
    elif method == "covariance":
        loading, residual, variance = _reference_factors(float(hurst), float(dt_years), n)
        driver = z[..., 0] @ loading.T + z[..., 1] @ residual.T
    else:
        raise ValueError("method must be hybrid or covariance")
    return driver, variance


def simulate_rough(parameters, curve, dt_years, innovations, *, method="hybrid",
                   spot=1.0, rate=0.0, dividend=0.0, conditional_only=False):
    """Simulate from the original Q origin on a uniform grid.

    innovations [paths,steps,3] are independent standard normal channels for
    W, Gaussian residual, W-perp. Outputs include the full Volterra driver;
    current variance or current driver is NOT a sufficient continuation state.
    No restart API or P risk forecast is claimed. The hybrid normalizer is its
    discrete Var(Y_i), so E[v_i]=xi0(t_i) exactly for its Gaussian approximation.
    The covariance reference uses t_i^(2H); both retain log Euler asset bias.
    conditional_only=True returns only integrated_variance and
    volatility_driver_integral for conditional pricing. The three innovation
    channels and their order are unchanged, including the integrated-out asset
    channel; the default full-path API is unchanged.
    """
    if not isinstance(conditional_only, (bool, np.bool_)):
        raise ValueError("conditional_only must be boolean")
    p = _model_parameters(parameters, "rough")
    dt = _scalar(dt_years, "dt_years")
    if dt <= 0:
        raise ValueError("dt_years must be positive")
    z = _normals(innovations)
    spot, rate, dividend = _q_inputs(spot, rate, dividend)
    times = dt * np.arange(z.shape[1] + 1)
    xi = curve_values(curve, times)
    driver, driver_variance = rough_driver(p["hurst"], dt, z, method=method)
    driver = np.column_stack((np.zeros(len(z)), driver))
    driver_variance = np.r_[0., driver_variance]
    with np.errstate(over="raise", invalid="raise"):
        try:
            variance = xi * np.exp(p["eta"] * driver - 0.5 * p["eta"] ** 2 * driver_variance)
        except FloatingPointError as error:
            raise ValueError("variance overflow; numerical support exceeded") from error
    result = _asset_paths(variance, dt, z, p["rho"], spot, rate, dividend,
                          conditional_only=conditional_only)
    if conditional_only:
        return result
    result.update(volterra_driver=driver, driver_variance=driver_variance, times=times,
                  continuation_state_available=False, driver_method=method)
    return result
