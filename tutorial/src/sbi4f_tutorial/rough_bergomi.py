"""Rough Bergomi prices by Monte Carlo, in forward units: the live smile of slot 5.

The variance is v_t = xi0(t) exp(eta Y_t - eta^2 Var(Y_t) / 2), with Y_t = sqrt(2H) int_0^t (t-s)^(H-1/2) dW_s
a Riemann-Liouville Volterra process and xi0 the forward-variance curve. The price is correlated with W by rho.
Y is simulated by the hybrid scheme (the singular last cell exactly, older cells by their average kernel,
the convolution by FFT). A put is priced conditionally on the variance path: given int v dt and
int sqrt(v) dW, the price is lognormal, so the Black-Scholes formula integrates the independent noise out.
Paths come in antithetic pairs; two control variates per maturity with known mean zero remove most of the
Monte Carlo error. Calls follow by parity. This is the lab's rough_bergomi.py, bergomi.py and
calibration/pricing.py, restricted to what one smile needs.
"""

import numpy as np
from scipy.special import log_ndtr, ndtr

_PAIR_CHUNK = 2048  # antithetic pairs per simulation chunk; memory, not results, depends on it


def step_curve(knots, levels):
    """Right-continuous piecewise-constant forward variance: xi0(t) = levels[j] on [knots[j], knots[j+1])."""
    knots, levels = np.asarray(knots, dtype=float), np.asarray(levels, dtype=float)
    if knots[0] != 0 or len(knots) != len(levels) + 1 or np.any(np.diff(knots) <= 0) or np.any(levels <= 0):
        raise ValueError("knots from zero, one positive level per interval")
    ramp = 1e-10
    times, values = [0.0], [levels[0]]
    for j in range(1, len(levels)):
        times += [knots[j] - ramp, knots[j]]
        values += [levels[j - 1], levels[j]]
    times.append(knots[-1])
    values.append(levels[-1])
    return {"times": times, "values": values}


def curve_values(curve, times):
    """The curve, piecewise linear between its points, at the given times (no extrapolation)."""
    x, y = np.asarray(curve["times"], float), np.asarray(curve["values"], float)
    t = np.asarray(times, float)
    if np.any(t < 0) or np.any(t > x[-1] + 1e-12):
        raise ValueError("the curve must cover every time")
    return np.interp(t, x, y)


def hybrid_weights(hurst, dt, n_steps):
    """Weights of the Brownian increments in Y and the loading of the independent last-cell residual."""
    cells = np.arange(n_steps + 1, dtype=float)
    weights = dt ** hurst * np.sqrt(2 * hurst) * np.diff(cells ** (hurst + 0.5)) / (hurst + 0.5)
    bridge = dt ** hurst * np.sqrt(max(1 - 2 * hurst / (hurst + 0.5) ** 2, 0))
    return weights, bridge


def simulate(parameters, curve, dt, normals):
    """Integrated variance and int sqrt(v) dW on a uniform grid; normals [paths, steps, 3] (W, residual, W-perp)."""
    h, eta = parameters["hurst"], parameters["eta"]
    n = normals.shape[1]
    weights, bridge = hybrid_weights(h, dt, n)
    fft_size = 1 << (2 * n - 1).bit_length()
    convolution = np.fft.irfft(np.fft.rfft(normals[..., 0], n=fft_size, axis=1)
                               * np.fft.rfft(weights, n=fft_size), n=fft_size, axis=1)[:, :n]
    driver = np.column_stack((np.zeros(len(normals)), convolution + bridge * normals[..., 1]))
    driver_variance = np.r_[0., np.cumsum(weights ** 2) + bridge ** 2]
    xi = curve_values(curve, dt * np.arange(n + 1))
    variance = xi * np.exp(eta * driver - 0.5 * eta ** 2 * driver_variance)
    left = variance[:, :-1]
    paths = {"integrated_variance": np.cumsum(left * dt, axis=1),
             "volatility_driver_integral": np.cumsum(np.sqrt(left * dt) * normals[..., 0], axis=1)}
    if not all(np.isfinite(a).all() for a in paths.values()):
        raise ValueError("nonfinite trajectory; numerical support exceeded")
    return paths


def conditional_puts(paths, rho, contracts, indices):
    """Put prices conditional on each variance path, one column per contract; each lies in [0, K]."""
    columns = []
    for contract, index in zip(contracts, indices):
        k = contract["strike"]
        iv = paths["integrated_variance"][:, index - 1]
        iw = paths["volatility_driver_integral"][:, index - 1]
        log_forward = rho * iw - 0.5 * rho ** 2 * iv
        residual_variance = (1 - rho ** 2) * iv
        sd = np.sqrt(residual_variance)
        d1 = (log_forward - np.log(k) + 0.5 * residual_variance) / sd
        value = k * ndtr(-(d1 - sd)) - np.exp(log_forward + log_ndtr(-d1))
        columns.append(np.maximum(value, 0))
    return np.column_stack(columns)


def price_block(parameters, curve, contracts, *, horizon, n_steps, n_pairs, seed):
    """Prices, standard errors and covariance of the price means for one block of contracts."""
    dt = horizon / n_steps
    indices = np.rint(np.array([c["maturity"] for c in contracts]) / dt).astype(int)
    maturities = sorted(set(indices.tolist()))
    xi = curve_values(curve, dt * np.arange(n_steps))
    expected_iv = {i: float(np.sum(xi[:i]) * dt) for i in maturities}
    puts = [dict(c, option_type="put") for c in contracts]
    rho = parameters["rho"]
    rng = np.random.default_rng(seed)
    payoff_chunks, control_chunks = [], []
    remaining = n_pairs
    while remaining:
        m = min(_PAIR_CHUNK, remaining)
        z = rng.standard_normal((m, n_steps, 3))
        pair = []
        for sign in (1.0, -1.0):
            paths = simulate(parameters, curve, dt, sign * z)
            payoff = conditional_puts(paths, rho, puts, indices)
            controls = []
            for i in maturities:
                iv = paths["integrated_variance"][:, i - 1]
                iw = paths["volatility_driver_integral"][:, i - 1]
                controls.append(np.exp(rho * iw - 0.5 * rho**2 * iv) - 1.0)  # the effective forward, mean 0
                controls.append(iv - expected_iv[i])  # integrated variance, mean 0
            pair.append((payoff, np.column_stack(controls)))
        payoff_chunks.append(0.5 * (pair[0][0] + pair[1][0]))
        control_chunks.append(0.5 * (pair[0][1] + pair[1][1]))
        remaining -= m
    payoffs, controls = np.concatenate(payoff_chunks), np.concatenate(control_chunks)
    centered = controls - controls.mean(axis=0)
    coefficients = np.linalg.lstsq(centered, payoffs - payoffs.mean(axis=0), rcond=None)[0]
    adjusted = payoffs - controls @ coefficients
    prices = adjusted.mean(axis=0)
    residual = adjusted - prices
    dof = len(adjusted) - controls.shape[1] - 1
    covariance = residual.T @ residual / dof / len(adjusted)
    covariance = 0.5 * (covariance + covariance.T)
    prices = prices + np.array([1.0 - c["strike"] if c["option_type"] == "call" else 0.0 for c in contracts])
    return prices, np.sqrt(np.diag(covariance)), covariance


def surface_prices(parameters, curve, contracts, blocks, paths, seed):
    """Every contract priced in its maturity block with the given number of paths: prices and standard errors."""
    if not (0 < parameters["hurst"] < 0.5 and parameters["eta"] >= 0 and -1 < parameters["rho"] <= 0):
        raise ValueError("require 0 < H < 0.5, eta >= 0 and -1 < rho <= 0")
    maturities = np.array([c["maturity"] for c in contracts])
    assignment = np.full(len(contracts), -1)
    for b, block in reversed(list(enumerate(blocks))):
        assignment[maturities <= block["horizon"] + 1e-12] = b
    prices, errors = np.empty(len(contracts)), np.empty(len(contracts))
    for b, block in enumerate(blocks):
        chosen = np.flatnonzero(assignment == b)
        if len(chosen):
            stream = np.random.SeedSequence([seed, b, 0])
            prices[chosen], errors[chosen], _ = price_block(
                parameters, curve, [contracts[i] for i in chosen], horizon=block["horizon"],
                n_steps=block["n_steps"], n_pairs=paths // 2, seed=stream)
    return prices, errors
