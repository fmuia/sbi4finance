"""Bounded independent references for the daily inferred-drift teaching laws.

This is prior importance integration, not a learned likelihood or a PyMC run.
Asset noise and uniform physical drift are integrated out conditional on the
volatility innovations. The target is exactly the declared finite simulation
grid; Monte Carlo integration of that target remains approximate. Short-history
precision diagnostics do not certify a neural posterior or longer histories.

The signed full-truncation Heston law admits zero-integrated-variance days.
Their atom locations depend on inferred drift. Consequently only its first day
is supported as a posterior reference here (positive fixed initial variance).
Longer Heston histories produce prior atom-support diagnostics, never a
Gaussian-only likelihood passed off as the full mixed posterior.
"""

import hashlib
import time

import numpy as np
from scipy.special import log_ndtr, logsumexp
from scipy.stats import truncnorm

from . import bank


GATES = {"minimum_ess": 1000.0, "maximum_weight": .01,
         "maximum_mean_shift_in_posterior_sd": .10,
         "maximum_quantile_shift_in_posterior_sd": .15}
QUANTILES = np.array([.05, .5, .95])


def validate_task(task):
    bank.validate_task(task)
    if task["task_id"] != bank.TEACHING_TASK_IDS[task["model"]]:
        raise ValueError("reference requires the inferred-drift daily teaching task")
    if task["parameter_names"][-1] != "drift" or task["prior_types"][-1] != "uniform":
        raise ValueError("reference requires last-coordinate independent uniform drift")
    return task


def observations_sha256(observations):
    values = np.asarray(observations, dtype="<f8")
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def sample_nondrift_prior(task, count, rng):
    """Independent declared priors; no observed/simulated truth labels are used."""
    validate_task(task)
    count = _positive_integer(count, "count")
    bounds = np.asarray(task["prior_bounds"][:-1], dtype=float)
    values = rng.random((count, len(bounds)))
    for j, ((low, high), kind) in enumerate(zip(bounds, task["prior_types"][:-1], strict=True)):
        values[:, j] = (np.exp(np.log(low) + values[:, j] * np.log(high / low))
                        if kind == "log_uniform" else low + values[:, j] * (high - low))
    return values


def conditional_statistics(task, parameters, variance_normals, bridge_normals=None):
    """Return daily I=sum(v*h) and J=sum(sqrt(v*h)*Z_v), independently coded.

    ``parameters`` omits drift. Normals have shape [particle,day,internal_step].
    A rough bridge normal is independent of its variance Brownian channel. The
    driver starts at the original origin, with its discrete hybrid compensator.
    """
    validate_task(task)
    p = np.asarray(parameters, dtype=float)
    z = np.asarray(variance_normals, dtype=float)
    substeps = task["numerics"].get("substeps", task["numerics"].get("observation_substeps"))
    if (p.ndim != 2 or p.shape[1] != len(task["parameter_names"]) - 1
            or z.ndim != 3 or z.shape[0] != len(p) or z.shape[2] != substeps
            or not 1 <= z.shape[1] <= min(21, task["n_observations"])
            or not np.isfinite(p).all() or not np.isfinite(z).all()):
        raise ValueError("invalid short-history parameters/variance-normal shape")
    bounds = np.asarray(task["prior_bounds"][:-1])
    if np.any(p < bounds[:, 0]) or np.any(p > bounds[:, 1]):
        raise ValueError("parameters outside declared prior support")
    count, days, _ = z.shape
    h = task["dt_years"] / substeps
    if task["model"] == "heston":
        if bridge_normals is not None:
            raise ValueError("Heston has no independent Volterra bridge channel")
        kappa, level, sigma, _rho = p.T
        state = np.full(count, task["fixed"]["initial_variance"])
        integrated = np.zeros((count, days))
        leverage = np.zeros_like(integrated)
        for day in range(days):
            for step in range(substeps):
                positive = np.maximum(state, 0)
                increment = np.sqrt(positive * h) * z[:, day, step]
                integrated[:, day] += positive * h
                leverage[:, day] += increment
                state += kappa * (level - positive) * h + sigma * increment
    else:
        bridge_z = np.asarray(bridge_normals, dtype=float)
        if bridge_z.shape != z.shape or not np.isfinite(bridge_z).all():
            raise ValueError("rough reference requires independent bridge normals of matching shape")
        fine_count = days * substeps
        flat_z = z.reshape(count, fine_count)
        hurst, eta = p[:, 0:1], p[:, 1:2]
        grid = np.arange(fine_count, dtype=float)[None, :]
        weights = (h ** hurst * np.sqrt(2 * hurst)
                   * ((grid + 1) ** (hurst + .5) - grid ** (hurst + .5)) / (hurst + .5))
        bridge = h ** hurst * np.sqrt(np.maximum(1 - 2 * hurst / (hurst + .5) ** 2, 0))
        fft_size = 1 << (2 * fine_count - 1).bit_length()
        endpoints = (np.fft.irfft(np.fft.rfft(flat_z, n=fft_size, axis=1)
                     * np.fft.rfft(weights, n=fft_size, axis=1), n=fft_size, axis=1)[:, :fine_count]
                     + bridge * bridge_z.reshape(count, fine_count))
        endpoint_variance = np.cumsum(weights ** 2, axis=1) + bridge ** 2
        left_driver = np.column_stack((np.zeros(count), endpoints[:, :-1]))
        left_compensator = np.column_stack((np.zeros(count), endpoint_variance[:, :-1]))
        if task["model"] == "rough-known":
            curve = task["fixed"]["variance_level"]
        else:
            short, long, decay = p[:, 3:4], p[:, 4:5], p[:, 5:6]
            curve = long + (short - long) * np.exp(-decay * h * grid)
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            left = curve * np.exp(eta * left_driver - .5 * eta ** 2 * left_compensator)
        if not np.isfinite(left).all() or np.any(left <= 0):
            raise FloatingPointError("nonpositive/nonfinite rough variance; no floors or redraws")
        integrated = (left * h).reshape(count, days, substeps).sum(axis=2)
        leverage = (np.sqrt(left * h) * flat_z).reshape(count, days, substeps).sum(axis=2)
    if not np.isfinite(integrated).all() or not np.isfinite(leverage).all() or np.any(integrated < 0):
        raise FloatingPointError("invalid volatility sufficient statistics; no redraws")
    return integrated, leverage


def _log_interval_probability(lower, upper):
    """Stable log(Phi(upper)-Phi(lower)), including positive-tail intervals."""
    lower, upper = np.broadcast_arrays(np.asarray(lower, float), np.asarray(upper, float))
    if np.any(lower >= upper):
        raise ValueError("normal interval must have positive width")
    positive = lower > 0
    large = np.where(positive, log_ndtr(-lower), log_ndtr(upper))
    small = np.where(positive, log_ndtr(-upper), log_ndtr(lower))
    with np.errstate(divide="ignore", invalid="raise"):
        return large + np.log(-np.expm1(small - large))


def integrate_uniform_drift(observations, integrated, leverage, rho, dt, bounds):
    """Analytic Gaussian likelihood integral over independent uniform drift.

    Returns component log likelihood, truncated-Normal location/scale, mean
    and second moment. A zero conditional variance is rejected explicitly; it
    is a drift-dependent atom, not a Normal density with an arbitrary floor.
    """
    y = np.asarray(observations, dtype=float)
    iv, jv = np.asarray(integrated, float), np.asarray(leverage, float)
    rho = np.asarray(rho, float)
    if (y.ndim != 1 or iv.ndim != 2 or jv.shape != iv.shape or iv.shape[1] != len(y)
            or rho.shape != (len(iv),) or not np.isfinite(y).all()
            or not np.isfinite(iv).all() or not np.isfinite(jv).all()
            or not np.isfinite(rho).all() or np.any(np.abs(rho) >= 1)
            or np.any(iv <= 0) or not np.isfinite(dt) or dt <= 0):
        raise ValueError("positive finite Gaussian variances required; zero-variance atoms unsupported")
    low, high = map(float, bounds)
    if not np.isfinite([low, high]).all() or not low < high:
        raise ValueError("invalid uniform drift support")
    variance = (1 - rho[:, None] ** 2) * iv
    offset = -.5 * iv + rho[:, None] * jv
    precision = np.sum(dt ** 2 / variance, axis=1)
    location = np.sum(dt * (y[None, :] - offset) / variance, axis=1) / precision
    scale = 1 / np.sqrt(precision)
    lower, upper = (low - location) / scale, (high - location) / scale
    log_probability = _log_interval_probability(lower, upper)
    mode = np.clip(location, low, high)
    residual = y[None, :] - offset - dt * mode[:, None]
    log_at_mode = -.5 * np.sum(np.log(2 * np.pi * variance) + residual ** 2 / variance, axis=1)
    log_likelihood = (log_at_mode + .5 * precision * (mode - location) ** 2
                      + .5 * np.log(2 * np.pi / precision) + log_probability - np.log(high - low))
    # For almost-flat densities, avoid cancellation in truncated-Normal moment
    # formulas. Deterministic 32-node integration concerns moments only; the
    # importance weight above always uses the analytic Normal interval mass.
    smooth = ((upper - lower < .5)
              & (np.maximum(np.abs(lower), np.abs(upper)) * (upper - lower) < 1))
    mean, variance_mu = np.empty_like(location), np.empty_like(location)
    if np.any(~smooth):
        mean[~smooth], variance_mu[~smooth] = truncnorm.stats(
            lower[~smooth], upper[~smooth], loc=location[~smooth], scale=scale[~smooth], moments="mv")
    if np.any(smooth):
        nodes, weights = np.polynomial.legendre.leggauss(32)
        values = low + (nodes + 1) * (high - low) / 2
        exponent = -.5 * ((values[None, :] - location[smooth, None]) / scale[smooth, None]) ** 2
        normalized = np.exp(exponent - logsumexp(exponent, b=weights, axis=1)[:, None]) * weights
        mean[smooth] = normalized @ values
        variance_mu[smooth] = np.sum(normalized * (values[None, :] - mean[smooth, None]) ** 2, axis=1)
    second = variance_mu + mean ** 2
    if (not np.isfinite(log_likelihood).all() or not np.isfinite(mean).all()
            or not np.isfinite(second).all() or np.any(mean < low) or np.any(mean > high)
            or np.any(variance_mu < 0)):
        raise FloatingPointError("drift integration precision failed; no clipping or redraws")
    return log_likelihood, location, scale, mean, second


def _weighted_quantiles(values, weights):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return values[order[np.minimum(np.searchsorted(cumulative, QUANTILES), len(order) - 1)]]


def summarize_components(parameters, log_likelihood, location, scale, mean_mu, second_mu, bounds):
    normalizer = logsumexp(log_likelihood)
    weights = np.exp(log_likelihood - normalizer)
    means = np.r_[weights @ parameters, weights @ mean_mu]
    seconds = np.r_[weights @ (parameters ** 2), weights @ second_mu]
    sd = np.sqrt(np.maximum(seconds - means ** 2, 0))
    quantiles = np.empty((parameters.shape[1] + 1, len(QUANTILES)))
    for col in range(parameters.shape[1]):
        quantiles[col] = _weighted_quantiles(parameters[:, col], weights)
    lower, upper = (bounds[0] - location) / scale, (bounds[1] - location) / scale
    for col, probability in enumerate(QUANTILES):
        low, high = bounds
        for _ in range(44):
            midpoint = (low + high) / 2
            cdf = weights @ truncnorm.cdf(midpoint, lower, upper, loc=location, scale=scale)
            if cdf < probability:
                low = midpoint
            else:
                high = midpoint
        quantiles[-1, col] = (low + high) / 2
    ess = float(1 / np.sum(weights ** 2))
    maximum = float(weights.max())
    summary = {"samples": len(weights), "ess": ess, "relative_ess": ess / len(weights),
               "maximum_weight": maximum, "posterior_mean": means.tolist(),
               "posterior_sd": sd.tolist(), "quantile_probabilities": QUANTILES.tolist(),
               "posterior_quantiles": quantiles.tolist(),
               "log_evidence_estimate": float(normalizer - np.log(len(weights))),
               "evidence_relative_mcse_estimate": float(np.sqrt(max(1 / ess - 1 / len(weights), 0))),
               "single_run_weight_gates_pass": bool(ess >= GATES["minimum_ess"] and maximum <= GATES["maximum_weight"])}
    return summary, weights


def importance_reference(task, observations, *, samples, seed, chunk=256, draws=8192, deadline=None):
    """Independent prior/latent importance run with analytic drift integration."""
    validate_task(task)
    y = np.asarray(observations, dtype=float).reshape(-1)
    if not np.isfinite(y).all() or not 1 <= len(y) <= min(21, task["n_observations"]):
        raise ValueError("reference supports finite histories of one through 21 days")
    if task["model"] == "heston" and len(y) != 1:
        raise ValueError("multiday Heston requires atom-aware inference; use atom_support_diagnostic")
    samples, chunk, draws = (_positive_integer(samples, "samples"), _positive_integer(chunk, "chunk"),
                            _positive_integer(draws, "draws"))
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**64:
        raise ValueError("seed must be uint64")
    started = time.perf_counter()
    rng_p, rng_z, rng_bridge, rng_output = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(4)]
    dimension = len(task["parameter_names"]) - 1
    parameters = np.empty((samples, dimension))
    log_likelihood, location, scale, mean_mu, second_mu = np.empty((5, samples))
    substeps = task["numerics"].get("substeps", task["numerics"].get("observation_substeps"))
    for start in range(0, samples, chunk):
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("reference wall-clock cap reached; incomplete importance run discarded")
        stop = min(start + chunk, samples)
        p = sample_nondrift_prior(task, stop - start, rng_p)
        z = rng_z.standard_normal((len(p), len(y), substeps))
        bridge = (None if task["model"] == "heston" else rng_bridge.standard_normal(z.shape))
        iv, jv = conditional_statistics(task, p, z, bridge)
        rho_col = task["parameter_names"].index("rho")
        result = integrate_uniform_drift(y, iv, jv, p[:, rho_col], task["dt_years"], task["prior_bounds"][-1])
        parameters[start:stop] = p
        for array, values in zip((log_likelihood, location, scale, mean_mu, second_mu), result, strict=True):
            array[start:stop] = values
    bounds = task["prior_bounds"][-1]
    summary, weights = summarize_components(parameters, log_likelihood, location, scale, mean_mu, second_mu, bounds)
    selected = rng_output.choice(samples, size=draws, replace=True, p=weights)
    low, high = (bounds[0] - location[selected]) / scale[selected], (bounds[1] - location[selected]) / scale[selected]
    drift = truncnorm.ppf(rng_output.random(draws), low, high, loc=location[selected], scale=scale[selected])
    posterior = np.column_stack((parameters[selected], drift))
    if not np.isfinite(posterior).all():
        raise FloatingPointError("conditional drift inverse CDF failed")
    summary.update(method="prior_latent_importance_with_analytic_uniform_drift_integration",
                   framework="NumPy/SciPy; specialized bounded integration, not PyMC",
                   seed=int(seed), chunk=chunk, length=len(y), parameter_names=task["parameter_names"],
                   task_sha256=bank.task_fingerprint(task), observations_sha256=observations_sha256(y),
                   seconds=time.perf_counter() - started,
                   target="exact declared finite-grid law; Monte Carlo integration approximate",
                   limitations="Short-history Monte Carlo precision diagnostics only; two cases cannot establish conditional calibration or general joint accuracy")
    arrays = {"parameters_nondrift": parameters, "weights": weights,
              "drift_location": location, "drift_scale": scale,
              "posterior_draws": posterior, "observations": y}
    return summary, arrays


def atom_support_diagnostic(task, observations, *, samples, seed, chunk=256, deadline=None):
    """Prior-only Heston atom support, without claiming posterior atom weights."""
    validate_task(task)
    if task["model"] != "heston":
        raise ValueError("atom diagnostic is for signed full-truncation Heston")
    y = np.asarray(observations, dtype=float).reshape(-1)
    if not np.isfinite(y).all() or not 1 <= len(y) <= min(21, task["n_observations"]):
        raise ValueError("one through 21 finite observations required")
    samples, chunk = _positive_integer(samples, "samples"), _positive_integer(chunk, "chunk")
    rng_p, rng_z = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(2)]
    zero_by_day = np.zeros(len(y), dtype=np.int64)
    any_zero = 0
    started = time.perf_counter()
    for start in range(0, samples, chunk):
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("atom diagnostic cap reached; incomplete run discarded")
        count = min(chunk, samples - start)
        p = sample_nondrift_prior(task, count, rng_p)
        z = rng_z.standard_normal((count, len(y), task["numerics"]["substeps"]))
        iv, _ = conditional_statistics(task, p, z)
        zero = iv == 0
        zero_by_day += zero.sum(axis=0)
        any_zero += int(np.any(zero, axis=1).sum())
    candidates = y / task["dt_years"]
    bounds = task["prior_bounds"][-1]
    return {"method": "prior_atom_support_diagnostic_only", "status": "posterior_reference_inconclusive",
            "seed": int(seed), "samples": samples, "length": len(y), "chunk": chunk,
            "task_sha256": bank.task_fingerprint(task), "observations_sha256": observations_sha256(y),
            "prior_paths_with_any_zero_variance_day": any_zero,
            "prior_zero_variance_counts_by_day": zero_by_day.tolist(),
            "atom_compatible_drift_from_observed_return": candidates.tolist(),
            "atom_compatible_drift_inside_prior": ((candidates >= bounds[0]) & (candidates <= bounds[1])).tolist(),
            "seconds": time.perf_counter() - started,
            "limitations": "Prior support only, not posterior atom probability. Zero sampled atoms does not exclude support. Multiday posterior needs a mixed-measure, drift-atom-aware method; no Gaussian floors or omitted atoms are used."}


def compare_runs(summaries):
    """Prespecified necessary precision checks, never a general validation gate."""
    posterior = [r for r in summaries if "posterior_mean" in r]
    if len(posterior) < 2:
        return {"status": "inconclusive_insufficient_independent_runs", "gates": GATES,
                "scientific_acceptance": False}
    identity_keys = ("task_sha256", "observations_sha256", "parameter_names", "length")
    if any(any(r[k] != posterior[0][k] for k in identity_keys) for r in posterior[1:]):
        raise ValueError("reference precision comparison requires identical task and observations")
    if len({r["seed"] for r in posterior}) != len(posterior):
        raise ValueError("reference precision comparison requires independent random seeds")
    maximum_mean, maximum_quantile = 0.0, 0.0
    for i, first in enumerate(posterior):
        for second in posterior[i + 1:]:
            sd = np.sqrt((np.asarray(first["posterior_sd"]) ** 2 + np.asarray(second["posterior_sd"]) ** 2) / 2)
            if np.any(sd <= 0):
                raise FloatingPointError("zero posterior SD prevents precision comparison")
            maximum_mean = max(maximum_mean, float(np.max(np.abs(np.asarray(first["posterior_mean"]) - second["posterior_mean"]) / sd)))
            maximum_quantile = max(maximum_quantile, float(np.max(np.abs(np.asarray(first["posterior_quantiles"]) - second["posterior_quantiles"]) / sd[:, None])))
    passed = (all(r["single_run_weight_gates_pass"] for r in posterior)
              and maximum_mean <= GATES["maximum_mean_shift_in_posterior_sd"]
              and maximum_quantile <= GATES["maximum_quantile_shift_in_posterior_sd"])
    return {"status": "short_history_precision_checks_pass" if passed else "inconclusive_monte_carlo_precision",
            "gates": GATES, "independent_runs": len(posterior),
            "maximum_mean_shift_in_posterior_sd": maximum_mean,
            "maximum_quantile_shift_in_posterior_sd": maximum_quantile,
            "scientific_acceptance": False,
            "limitations": "Necessary finite-sample stability checks; agreement can miss rare high-weight regions. No certification of neural, long-history, conditional or joint posterior accuracy."}
