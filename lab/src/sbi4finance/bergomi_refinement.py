"""Paired time-grid price checks for the existing bounded Bergomi estimators.

One-factor coupling matches exact OU factor endpoints and Brownian increments.
Rough coupling preserves each hybrid grid's marginal law by aggregating iid
normal channels and rebuilding its field. It is common random numbers, not an
exact common-Wiener bridge of the singular integrals across grids. Neither
coupling removes discretization bias or certifies a continuum price.
"""

import numpy as np

from sbi4finance.bergomi import (
    _model_parameters, _normals, _positive_integer, _q_inputs, _scalar,
    conditional_put_payoffs, curve_values, simulate_one_factor,
)
from sbi4finance.rough_bergomi import simulate_rough


def coarsen_innovations(innovations, n_steps, *, model, dt_years, kappa=None):
    """Map finest [paths,steps,3] iid normals to one nested uniform grid.

    ``dt_years`` is the finest time step. Channels 0 and 2 are normalized block
    sums. For one_factor, channel 1 is the aggregate exact OU innovation after
    its projection on coarse channel 0, divided by its residual standard
    deviation. Hence each coarse channel is again an independent standard
    normal, and coarse OU endpoints equal the fine endpoints. For rough all
    three channels use block sums; the caller must rebuild its coarse field
    and normalizer instead of subsampling the fine field.
    """
    z = _normals(innovations)
    coarse_steps = _positive_integer(n_steps, "n_steps")
    dt = _scalar(dt_years, "dt_years")
    if model not in ("one_factor", "rough") or dt <= 0:
        raise ValueError("require model one_factor or rough and positive dt_years")
    if coarse_steps > z.shape[1] or z.shape[1] % coarse_steps:
        raise ValueError("coarse n_steps must divide the finest step count")
    if model == "one_factor":
        k = _scalar(kappa, "kappa") if kappa is not None else 0.0
        if k <= 0:
            raise ValueError("one_factor coupling requires kappa > 0")
    ratio = z.shape[1] // coarse_steps
    if ratio == 1:
        return z.copy()
    blocks = z.reshape(len(z), coarse_steps, ratio, 3)
    coarse = blocks.sum(axis=2) / np.sqrt(ratio)
    if model == "one_factor":
        def loadings(delta):
            variance = -np.expm1(-2 * k * delta) / (2 * k)
            loading = -np.expm1(-k * delta) / (k * np.sqrt(delta))
            return loading, np.sqrt(max(variance - loading * loading, 0.0))

        fine_w, fine_bridge = loadings(dt)
        coarse_w, coarse_bridge = loadings(dt * ratio)
        if fine_bridge == 0 or coarse_bridge == 0:
            raise ValueError("OU residual cannot be resolved at this kappa and time step")
        weights = np.exp(-k * dt * np.arange(ratio - 1, -1, -1))
        w_coefficients = (weights * fine_w - coarse_w / np.sqrt(ratio)) / coarse_bridge
        bridge_coefficients = weights * fine_bridge / coarse_bridge
        residual_variance = np.sum(w_coefficients**2 + bridge_coefficients**2)
        residual_cross = np.sum(w_coefficients) / np.sqrt(ratio)
        if abs(residual_variance - 1) > 1e-7 or abs(residual_cross) > 1e-7:
            raise ValueError("OU residual projection loses numerical precision at this grid")
        innovation = np.sum(weights * (fine_w * blocks[..., 0]
                                       + fine_bridge * blocks[..., 1]), axis=2)
        coarse[..., 1] = (innovation - coarse_w * coarse[..., 0]) / coarse_bridge
    return coarse


def _differences(difference_mean, difference_covariance, covariance, steps, pairs):
    """Summarize second-minus-first price differences using the full covariance."""
    independent_variances = []
    for first, second in pairs:
        independent = np.diag(covariance[first, :, first, :]
                              + covariance[second, :, second, :])
        independent_variances.append(np.maximum(independent, 0))
    paired = np.maximum(np.diag(difference_covariance), 0).reshape(difference_mean.shape)
    independent = np.asarray(independent_variances)
    # A ratio is undefined if both estimators have zero sampling variance.
    reduction = np.full_like(paired, np.nan)
    np.divide(independent, paired, out=reduction, where=paired > 0)
    reduction[(paired == 0) & (independent > 0)] = np.inf
    return {"step_pairs": np.asarray([[steps[a], steps[b]] for a, b in pairs]),
            "differences": difference_mean,
            "estimator_covariance": difference_covariance.reshape(len(pairs), difference_mean.shape[1],
                                                                  len(pairs), difference_mean.shape[1]),
            "paired_standard_errors": np.sqrt(paired),
            "independent_standard_errors": np.sqrt(independent),
            "variance_reduction": reduction}


def coupled_option_panels(model, parameters, curve, contracts, *, spot=1.0,
                          rate=0.0, dividend=0.0, n_paths=8192,
                          steps=(128, 256, 512, 1024), seed=0, chunk_size=256):
    """Estimate nested Q panels with paired grid differences and their MC errors.

    The finest iid stream is fixed by seed and is independent of chunk_size.
    Prices/SE have shape [grid,contract]. estimator_covariance has shape
    [grid,contract,grid,contract] and describes the means, not individual payoffs.
    Both adjacent and to_finest comparisons are second-minus-first. Their
    independent_standard_errors show the counterfactual error if grid means
    with the same marginal variances were estimated independently. The ratio
    of independent to paired variance can be NaN (both zero) or infinity
    (paired zero); this is deliberate, not evidence of discretization accuracy.
    All maturities must be nodes of every declared grid. Rough uses hybrid.
    """
    if model not in ("one_factor", "rough"):
        raise ValueError("model must be one_factor or rough")
    p = _model_parameters(parameters, model)
    spot, rate, dividend = _q_inputs(spot, rate, dividend)
    n_paths = _positive_integer(n_paths, "n_paths", 2)
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    if not isinstance(steps, (list, tuple, np.ndarray)) or len(steps) < 2:
        raise ValueError("steps must contain at least two nested increasing grids")
    grids = tuple(_positive_integer(n, "steps entry") for n in steps)
    if any(b <= a or b % a for a, b in zip(grids[:-1], grids[1:])):
        raise ValueError("steps must be strictly increasing with nested integer ratios")
    if not isinstance(contracts, (list, tuple)) or not contracts:
        raise ValueError("contracts must be a nonempty ordered list")
    checked = []
    for contract in contracts:
        if (not isinstance(contract, dict) or not {"strike", "maturity"}.issubset(contract)
                or set(contract) - {"strike", "maturity", "option_type"}):
            raise ValueError("each contract requires strike and maturity, optional option_type")
        strike = _scalar(contract["strike"], "strike")
        maturity = _scalar(contract["maturity"], "maturity")
        kind = contract.get("option_type", "put")
        if strike <= 0 or maturity <= 0 or kind not in ("put", "call"):
            raise ValueError("positive strike/maturity and call or put required")
        checked.append({"strike": strike, "maturity": maturity, "option_type": kind})
    horizon = max(c["maturity"] for c in checked)
    curve_values(curve, [0.0, horizon])
    indices = []
    for n in grids:
        raw = np.asarray([c["maturity"] * n / horizon for c in checked])
        nodes = np.rint(raw).astype(int)
        if np.any(nodes < 1) or not np.allclose(nodes, raw, rtol=0, atol=1e-9):
            raise ValueError("all maturities must land on every declared uniform grid")
        indices.append(nodes)
    n_contracts, base_width = len(checked), len(grids) * len(checked)
    adjacent = [(i, i + 1) for i in range(len(grids) - 1)]
    to_finest = [(i, len(grids) - 1) for i in range(len(grids) - 1)]
    difference_width = len(adjacent) * n_contracts
    width = base_width + 2 * difference_width
    mean, centered_cross, count = np.zeros(width), np.zeros((width, width)), 0
    rng = np.random.default_rng(seed)
    for first in range(0, n_paths, chunk_size):
        size = min(chunk_size, n_paths - first)
        finest = rng.standard_normal((size, grids[-1], 3))
        estimates = []
        for n, nodes in zip(grids, indices):
            z = coarsen_innovations(finest, n, model=model,
                                   dt_years=horizon / grids[-1], kappa=p.get("kappa"))
            if model == "one_factor":
                paths = simulate_one_factor(p, curve, np.full(n, horizon / n), z,
                                            spot=spot, rate=rate, dividend=dividend)
            else:
                paths = simulate_rough(p, curve, horizon / n, z, method="hybrid",
                                       spot=spot, rate=rate, dividend=dividend)
            estimates.append(conditional_put_payoffs(paths, p, checked, nodes,
                             spot=spot, rate=rate, dividend=dividend))
        # Accumulate price differences directly as well as joint grid prices.
        # Subtracting covariance blocks would lose precision near zero variance.
        differences = [estimates[b] - estimates[a] for a, b in adjacent + to_finest]
        estimates = np.concatenate(estimates + differences, axis=1)
        batch_mean = estimates.mean(axis=0)
        centered = estimates - batch_mean
        change = batch_mean - mean
        total = count + size
        centered_cross += centered.T @ centered + np.outer(change, change) * count * size / total
        mean += change * size / total
        count = total
    covariance = centered_cross / (n_paths * (n_paths - 1))
    covariance = (covariance + covariance.T) / 2
    errors = np.sqrt(np.maximum(np.diag(covariance)[:base_width], 0)).reshape(len(grids), n_contracts)
    full_covariance = covariance
    covariance = covariance[:base_width, :base_width].reshape(len(grids), n_contracts, len(grids), n_contracts)
    prices = mean[:base_width].reshape(len(grids), n_contracts)
    for index, contract in enumerate(checked):
        if contract["option_type"] == "call":
            maturity = contract["maturity"]
            prices[:, index] += (spot * np.exp(-dividend * maturity)
                                - contract["strike"] * np.exp(-rate * maturity))
    comparisons = []
    for index, pairs in enumerate((adjacent, to_finest)):
        span = slice(base_width + index * difference_width, base_width + (index + 1) * difference_width)
        comparisons.append(_differences(mean[span].reshape(len(pairs), n_contracts),
                           full_covariance[span, span], covariance, grids, pairs))
    return {"steps": np.asarray(grids), "prices": prices, "standard_errors": errors,
            "estimator_covariance": covariance,
            "adjacent": comparisons[0], "to_finest": comparisons[1],
            "metadata": {"model": model, "measure": "Q", "n_paths": n_paths,
                         "seed": int(seed), "steps": list(grids), "horizon_years": horizon,
                         "coupling": "exact_ou_endpoints" if model == "one_factor" else "hybrid_channel_block_crn",
                         "rough_common_wiener_bridge": False,
                         "estimator": "conditional_bounded_puts_and_call_parity",
                         "covariance_axes": ["grid", "contract", "grid", "contract"],
                         "mc_error_includes_discretization_bias": False,
                         "quote_error_included": False}}
