"""Surface pricers for the calibration tasks, in forward units (spot 1, zero carry).

rough_surface prices under rough Bergomi by Monte Carlo. It reuses the hybrid
driver (rough_bergomi.simulate_rough) and the conditional Black–Scholes put
estimator (bergomi.conditional_put_payoffs), and adds three things:

* antithetic paths: every normal draw z is paired with -z, and pairs are
  averaged before any statistic is computed, so pairs are the iid units;
* two control variates per maturity with known mean zero: the effective
  forward exp(rho*int sqrt(V) dW - rho^2/2 int V dt) - 1, a discrete martingale,
  and int V dt minus its mean, which is exact because the hybrid scheme keeps
  E[v_i] = xi0(t_i); one regression per block removes their linear part;
* maturity blocks, each simulated on its own uniform grid with its own stream,
  so short maturities get fine steps; blocks are independent.

Calls follow from puts by parity, which is exact in forward units.

heston_surface prices with the Fourier integral of hkde.py with jumps off,
vectorized across contracts, at base and refined quadrature settings.
"""

import numpy as np

from ..bergomi import conditional_put_payoffs, curve_values
from ..rough_bergomi import simulate_rough

_PAIR_CHUNK = 2048  # antithetic pairs per simulation chunk; memory, not results, depends on it


def heston_variance_values(times, short_variance, long_variance, decay):
    """Positive Heston-shaped initial mean curve in variance units, not Heston dynamics."""
    t = np.asarray(times, dtype=float)
    coefficients = np.asarray([short_variance, long_variance, decay], dtype=float)
    if (coefficients.shape != (3,) or not np.isfinite(coefficients).all() or np.any(coefficients <= 0)
            or not np.isfinite(t).all() or np.any(t < 0)):
        raise ValueError("positive finite curve coefficients and nonnegative finite times required")
    # Convex weights avoid cancellation when the short level is far below the long level.
    weight = -np.expm1(-decay * t)
    return short_variance * (1.0 - weight) + long_variance * weight


def heston_integrated_variance(times, short_variance, long_variance, decay):
    """Analytic W(T)=integral_0^T xi(s)ds; expm1 resolves small decay*T."""
    t = np.asarray(times, dtype=float)
    heston_variance_values(t, short_variance, long_variance, decay)  # shared validation
    average_short_weight = np.ones_like(t)
    np.divide(-np.expm1(-decay * t), decay * t, out=average_short_weight, where=t != 0)
    return t * (short_variance * average_short_weight + long_variance * (1.0 - average_short_weight))


def heston_variance_curve(times, short_variance, long_variance, decay):
    """Sample the analytic curve at all simulator grid nodes, including time zero.

    The simulator consumes a piecewise-linear table. When every block grid is
    included, its node/left-cell evaluations equal the analytic curve. Its
    expected integrated-variance control remains the same discrete left sum,
    not the continuous W(T). Changing the grid changes that numerical model.
    """
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or len(t) < 2 or t[0] != 0 or np.any(np.diff(t) <= 0):
        raise ValueError("increasing curve grid from zero with at least two nodes required")
    return {"times": t.tolist(),
            "values": heston_variance_values(t, short_variance, long_variance, decay).tolist()}


def step_curve(knots, forward_variances):
    """Right-continuous piecewise-constant forward variance for bergomi.curve_values.

    knots are [0, T1, ..., Tn] and forward_variances the n interval levels:
    xi0(t) = f_j on [T_{j-1}, T_j). The left-point scheme uses xi0(t_{i-1}) on
    (t_{i-1}, t_i], so the level that starts at T_{j-1} applies from there on.
    """
    knots = np.asarray(knots, dtype=float)
    levels = np.asarray(forward_variances, dtype=float)
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


def _block_contracts(contracts, horizon, n_steps):
    dt = horizon / n_steps
    raw = np.array([c["maturity"] for c in contracts]) / dt
    indices = np.rint(raw).astype(int)
    if np.any(indices < 1) or np.any(indices > n_steps) or not np.allclose(indices, raw, rtol=0, atol=1e-9):
        raise ValueError("block contracts must land on the block's uniform grid")
    return dt, indices


def rough_block(parameters, curve, contracts, *, horizon, n_steps, n_pairs, seed, diagnostics=None):
    """Prices, standard errors and covariance of the price means for one block.

    contracts all have maturity <= horizon on the grid horizon / n_steps.
    n_pairs antithetic pairs are drawn from default_rng(seed). Given a dict,
    diagnostics receives, per contract, the variance of the antithetic pair
    means with and without the control variates: their ratio is 1 - R^2.
    """
    dt, indices = _block_contracts(contracts, horizon, n_steps)
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
            paths = simulate_rough(parameters, curve, dt, sign * z, method="hybrid", conditional_only=True)
            payoff = conditional_put_payoffs(paths, parameters, puts, indices, spot=1.0, rate=0.0, dividend=0.0)
            controls = []
            for i in maturities:
                iv = paths["integrated_variance"][:, i - 1]
                iw = paths["volatility_driver_integral"][:, i - 1]
                controls.append(np.exp(rho * iw - 0.5 * rho**2 * iv) - 1.0)
                controls.append(iv - expected_iv[i])
            pair.append((payoff, np.column_stack(controls)))
        payoff_chunks.append(0.5 * (pair[0][0] + pair[1][0]))
        control_chunks.append(0.5 * (pair[0][1] + pair[1][1]))
        remaining -= m
    payoffs, controls = np.concatenate(payoff_chunks), np.concatenate(control_chunks)
    centered = controls - controls.mean(axis=0)
    coefficients = np.linalg.lstsq(centered, payoffs - payoffs.mean(axis=0), rcond=None)[0]
    adjusted = payoffs - controls @ coefficients  # the controls' true means are zero
    prices = adjusted.mean(axis=0)
    residual = adjusted - prices
    dof = len(adjusted) - controls.shape[1] - 1
    covariance = residual.T @ residual / dof / len(adjusted)
    covariance = 0.5 * (covariance + covariance.T)
    if diagnostics is not None:
        diagnostics["plain_variance"] = payoffs.var(axis=0, ddof=1)
        diagnostics["controlled_variance"] = np.sum(residual**2, axis=0) / dof
    prices = prices + np.array([1.0 - c["strike"] if c["option_type"] == "call" else 0.0 for c in contracts])
    return prices, np.sqrt(np.diag(covariance)), covariance


def rough_surface(parameters, curve, contracts, blocks, paths_ladder, sigma, budget, seed, cap=None, diagnostics=None):
    """Price every contract in its block; escalate a block's paths until max SE / sigma <= budget.

    Each rung of the ladder uses a fresh stream SeedSequence([seed, block, rung]),
    so a surface's result is reproducible from its seed. A block that misses the
    budget on the last rung is kept only if its max SE / sigma is within cap
    (when a cap is given), and raises otherwise; nothing is redrawn silently.
    Returns prices, standard errors, the block-diagonal covariance of the price
    means, and the path count used per block. Given a list, diagnostics receives
    one entry per block for the rung used: its contracts and rough_block's
    variances with and without the control variates.
    """
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (len(contracts),))
    maturities = np.array([c["maturity"] for c in contracts])
    assignment = np.full(len(contracts), -1)
    for b, block in reversed(list(enumerate(blocks))):
        assignment[maturities <= block["horizon"] + 1e-12] = b
    if np.any(assignment < 0):
        raise ValueError("every contract needs a block whose horizon covers it")
    n = len(contracts)
    prices, errors, covariance = np.empty(n), np.empty(n), np.zeros((n, n))
    used = []
    for b, block in enumerate(blocks):
        chosen = np.flatnonzero(assignment == b)
        if not len(chosen):
            used.append(0)
            continue
        for rung, n_paths in enumerate(paths_ladder):
            stream = np.random.SeedSequence([seed, b, rung])
            found = {} if diagnostics is not None else None
            p, s, c = rough_block(parameters, curve, [contracts[i] for i in chosen],
                                  horizon=block["horizon"], n_steps=block["n_steps"],
                                  n_pairs=n_paths // 2, seed=stream, diagnostics=found)
            if np.all(s <= budget * sigma[chosen]):
                break
        else:
            if cap is None or np.max(s / sigma[chosen]) > cap:
                worst = int(chosen[np.argmax(s / sigma[chosen])])
                raise ValueError(f"Monte Carlo standard-error budget failed at contract {worst} "
                                 f"with {paths_ladder[-1]} paths (SE/sigma {np.max(s / sigma[chosen]):.3f})")
        prices[chosen], errors[chosen] = p, s
        covariance[np.ix_(chosen, chosen)] = c
        used.append(int(n_paths))
        if diagnostics is not None:
            diagnostics.append({"block": b, "contracts": chosen.tolist(), "paths": int(n_paths), **found})
    return prices, errors, covariance, np.array(used, dtype=float)


def heston_surface(parameters, initial_variance, contracts, *, nodes, upper_limit):
    """Fourier prices for every contract, one vectorized call per option type."""
    from ..hkde import european_price

    k = np.log(np.array([c["strike"] for c in contracts], dtype=float))
    maturity = np.array([c["maturity"] for c in contracts], dtype=float)
    kind = np.array([c["option_type"] for c in contracts])
    prices = np.empty(len(contracts))
    for value in np.unique(kind):
        chosen = kind == value
        prices[chosen] = european_price(1.0, np.exp(k[chosen]), 1.0, maturity[chosen], parameters,
                                        initial_variance, kind=str(value), nodes=nodes,
                                        upper_limit=upper_limit)
    return prices
