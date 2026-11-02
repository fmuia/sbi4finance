"""One-factor Bergomi and bounded European option-panel estimation under Q.

The initial forward variance curve is known context, expressed in annualized
variance units. Exact OU transitions do not make the asset integration exact.
Pricing integrates out the independent asset Brownian motion and averages
bounded conditional put prices; calls use parity only in the rho <= 0 domain.
"""

import numpy as np
from scipy.special import log_ndtr, ndtr

# Shared implementations are owned by the tutorial runtime.
from sbi4f_tutorial.rough_options.model import (
    _positive_integer,
    _scalar,
    curve_values,
    _model_parameters,
    _normals,
    _q_inputs,
    _asset_paths,
    conditional_put_payoffs,
)


def simulate_one_factor(parameters, curve, dt_years, innovations, *, factor_state=0.0,
                        elapsed_time=0.0, spot=1.0, rate=0.0, dividend=0.0):
    """Exact factor transitions plus Q log Euler, with a complete restart state.

    X satisfies dX=-kappa X dt+dW, X_0=0 at original origin; v_t=xi0(t)
    exp(eta X_t-eta**2 Var(X_t)/2). The three normal channels are W increments,
    OU bridge innovations, and independent asset increments. OU innovations
    preserve their exact covariance with W increments, not perfect correlation.

    Restart with terminal factor_state, accumulated elapsed_time, the SAME
    original curve, and current spot. Spot is a common scalar; log_returns and
    factor paths are independent of its value, so batched paths can restart at
    spot=1 then add their saved log spot. Do not replace the factor by variance.
    Latent outputs are simulator truth, not observed conditioning.
    """
    p = _model_parameters(parameters, "one_factor")
    z = _normals(innovations)
    dt = np.asarray(dt_years, float)
    if dt.shape != (z.shape[1],) or not np.isfinite(dt).all() or np.any(dt <= 0):
        raise ValueError("dt_years must be a positive finite vector matching steps")
    elapsed = _scalar(elapsed_time, "elapsed_time")
    if elapsed < 0:
        raise ValueError("elapsed_time must be nonnegative")
    spot, rate, dividend = _q_inputs(spot, rate, dividend)
    initial = np.asarray(factor_state, float)
    if initial.ndim > 1 or not np.isfinite(initial).all():
        raise ValueError("factor_state must be a finite scalar or path vector")
    try:
        initial = np.broadcast_to(initial, (len(z),))
    except ValueError as error:
        raise ValueError("factor_state must match number of paths") from error
    if elapsed == 0 and np.any(initial != 0):
        raise ValueError("original origin requires factor_state=0; continuation requires elapsed_time")
    times = elapsed + np.concatenate(([0.0], np.cumsum(dt)))
    xi = curve_values(curve, times)
    k, eta = p["kappa"], p["eta"]
    decay = np.exp(-k * dt)
    transition_variance = -np.expm1(-2 * k * dt) / (2 * k)
    covariance = -np.expm1(-k * dt) / k
    w_loading = covariance / np.sqrt(dt)
    bridge_loading = np.sqrt(np.maximum(transition_variance - w_loading ** 2, 0))
    factors = np.empty((len(z), len(dt) + 1))
    factors[:, 0] = initial
    for t in range(len(dt)):
        factors[:, t + 1] = decay[t] * factors[:, t] + w_loading[t] * z[:, t, 0] + bridge_loading[t] * z[:, t, 1]
    factor_variance = -np.expm1(-2 * k * times) / (2 * k)
    with np.errstate(over="raise", invalid="raise"):
        try:
            variance = xi * np.exp(eta * factors - 0.5 * eta ** 2 * factor_variance)
        except FloatingPointError as error:
            raise ValueError("variance overflow; numerical support exceeded") from error
    result = _asset_paths(variance, dt, z, p["rho"], spot, rate, dividend)
    result.update(factor_state=factors[..., None], elapsed_time=float(times[-1]), times=times,
                  driver_variance=factor_variance)
    return result


def option_panel(model, parameters, curve, contracts, *, spot=1.0, rate=0.0, dividend=0.0,
                 n_paths=8192, n_steps=128, seed=0, method="hybrid", chunk_size=1024):
    """Estimate a fixed Q European panel with shared-path MC covariance.

    model is 'one_factor' or 'rough'. Known curve uses original time in years.
    Maturities must be grid nodes on the uniform grid through max maturity.
    Contracts contain strike,maturity,option_type ('put' or 'call'; default put).
    Returned estimator_covariance is covariance of the price MEANS, not payoffs.
    SE excludes discretization/kernel bias and any separate quote error.
    Seed+configuration fully determines the paths, independent of chunk_size.
    """
    if model not in ("one_factor", "rough"):
        raise ValueError("model must be one_factor or rough")
    p = _model_parameters(parameters, model)
    spot, rate, dividend = _q_inputs(spot, rate, dividend)
    n_paths = _positive_integer(n_paths, "n_paths", 2)
    n_steps = _positive_integer(n_steps, "n_steps")
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    if model == "rough" and method not in ("hybrid", "covariance"):
        raise ValueError("rough method must be hybrid or covariance")
    if not isinstance(contracts, (list, tuple)) or not contracts:
        raise ValueError("contracts must be a nonempty ordered list")
    checked = []
    for contract in contracts:
        if not isinstance(contract, dict) or not {"strike", "maturity"}.issubset(contract) or set(contract) - {"strike", "maturity", "option_type"}:
            raise ValueError("each contract requires strike and maturity, optional option_type")
        k, t = _scalar(contract["strike"], "strike"), _scalar(contract["maturity"], "maturity")
        kind = contract.get("option_type", "put")
        if k <= 0 or t <= 0 or kind not in ("call", "put"):
            raise ValueError("positive strike/maturity and call or put required")
        checked.append({"strike": k, "maturity": t, "option_type": kind})
    horizon = max(c["maturity"] for c in checked)
    dt = horizon / n_steps
    raw_indices = np.array([c["maturity"] / dt for c in checked])
    indices = np.rint(raw_indices).astype(int)
    if np.any(indices < 1) or not np.allclose(indices, raw_indices, rtol=0, atol=1e-9):
        raise ValueError("all maturities must land on the declared uniform grid")
    curve_values(curve, [0, horizon])
    rng = np.random.default_rng(seed)
    prices, cross, count = np.zeros(len(checked)), np.zeros((len(checked), len(checked))), 0
    for first in range(0, n_paths, chunk_size):
        z = rng.standard_normal((min(chunk_size, n_paths - first), n_steps, 3))
        if model == "one_factor":
            paths = simulate_one_factor(p, curve, np.full(n_steps, dt), z,
                                        spot=spot, rate=rate, dividend=dividend)
        else:
            from sbi4finance.rough_bergomi import simulate_rough
            paths = simulate_rough(p, curve, dt, z, method=method,
                                   spot=spot, rate=rate, dividend=dividend)
        payoffs = conditional_put_payoffs(paths, p, checked, indices, spot=spot, rate=rate, dividend=dividend)
        # Centered batch combination avoids cancellation for almost-constant
        # conditional puts (e.g. deterministic variance with rho=0).
        batch_mean = payoffs.mean(axis=0)
        centered = payoffs - batch_mean
        difference = batch_mean - prices
        combined_count = count + len(payoffs)
        cross += centered.T @ centered + np.outer(difference, difference) * count * len(payoffs) / combined_count
        prices += difference * len(payoffs) / combined_count
        count = combined_count
    covariance = cross / ((n_paths - 1) * n_paths)
    covariance = (covariance + covariance.T) / 2
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    for i, contract in enumerate(checked):
        if contract["option_type"] == "call":
            t = contract["maturity"]
            prices[i] += spot * np.exp(-dividend * t) - contract["strike"] * np.exp(-rate * t)
    return {"prices": prices, "standard_errors": standard_errors,
            "estimator_covariance": covariance,
            "metadata": {"model": model, "measure": "Q", "n_paths": n_paths,
                         "n_steps": n_steps, "dt_years": dt, "seed": int(seed),
                         "driver_method": "exact_ou" if model == "one_factor" else method,
                         "estimator": "conditional_bounded_puts_and_call_parity",
                         "mc_error_includes_discretization_bias": False,
                         "quote_error_included": False}}
