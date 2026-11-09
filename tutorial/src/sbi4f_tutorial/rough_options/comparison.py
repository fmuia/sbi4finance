"""Illustrative smile behavior: instant shape guides and task-matched prices."""

import json
import time

import numpy as np

from ..shared import surface, tasks
from ..shared.paths import CHECKPOINTS


def skew_shape_figure(hurst, spy_report):
    """Normalized short-maturity shape guides, not finite-maturity model prices.

    Heston's limiting skew is constant. The rough guide has exponent H-1/2.
    Both are normalized at the one-month empirical maturity; this deliberately
    compares shape rather than fitting a level or calibrating either model.
    """
    import matplotlib.pyplot as plt

    if isinstance(hurst, bool) or np.ndim(hurst) != 0 or not np.isfinite(hurst) or not 0 < hurst <= .5:
        raise ValueError("the shape guide requires 0 < H <= 0.5")
    days = np.asarray(spy_report["design"]["days"], dtype=float)
    empirical = np.abs(np.asarray(spy_report["atm_skew"], dtype=float))
    if (days.ndim != 1 or len(days) < 2 or empirical.shape != days.shape
            or not np.isfinite(days).all() or np.any(days <= 0) or np.any(np.diff(days) <= 0)
            or not np.isfinite(empirical).all() or np.any(empirical <= 0)):
        raise ValueError("positive finite ordered empirical maturities and nonzero ATM skews are required")
    anchor = np.flatnonzero(np.isclose(days, 21, rtol=0, atol=1e-12))
    if len(anchor) != 1:
        raise ValueError("the normalized comparison requires a 21-trading-day empirical skew")
    reference = 21.0
    grid = np.unique(np.r_[np.geomspace(min(.5, days[0] / 10), days[-1], 120), reference])
    figure, axis = plt.subplots(figsize=(8.5, 3.5), layout="constrained")
    axis.loglog(grid, (grid / reference) ** (hurst - .5), color="#7755aa", linewidth=2,
                label=f"rough short-time shape guide: exponent {hurst - .5:.2f}")
    axis.loglog(grid, np.ones_like(grid), color="#3a9874", linewidth=2,
                label="Heston short-time shape guide: exponent 0")
    axis.loglog(days, empirical / empirical[anchor[0]], "o", color="#111827", markersize=5,
                label="saved SPY skew / its one-month skew")
    axis.axvline(days[0], color="#94a3b8", linewidth=.8, linestyle=":")
    axis.set(xlim=(.5, 252), ylim=(.3, 30),
             xlabel="Maturity (trading days)", ylabel="Absolute ATM skew / one-month level",
             title="Short-maturity shape guides; no model fitting")
    axis.legend(fontsize=8)
    return figure


def _smile_statistics(implied, k, maturity, name):
    implied = np.asarray(implied, dtype=float)
    if implied.shape != k.shape:
        raise ValueError(f"{name} requires all 63 implied-volatility positions")
    for expiry in np.unique(maturity):
        indices = np.flatnonzero(maturity == expiry)
        nearest = indices[np.argsort(np.abs(k[indices]))[:5]]
        if len(nearest) != 5 or not np.isfinite(implied[nearest]).all():
            raise ValueError(f"{name} needs five finite near-ATM implied volatilities at every maturity")
    maturities, atm, skew, curvature = surface.smile_statistics(implied, k, maturity)
    if not all(np.isfinite(a).all() for a in (atm, skew, curvature)):
        raise ValueError(f"{name} ATM smile statistics could not be calculated")
    return {"implied": implied, "maturities": maturities, "atm": atm,
            "skew": skew, "curvature": curvature,
            "omitted_wing_quotes": int(np.sum(~np.isfinite(implied)))}


def compare_smiles(rough_result, *, volatility=.2, sigma_v=.4, rho=-.7):
    """Add checked Heston Fourier prices to an existing fixed-budget rough run.

    Both models use the same 63 contracts, zero carry, flat expected variance
    volatility^2 and correlation. Vol-of-vol parameters retain their distinct
    model meanings. This computes no new rough paths and performs no fit.
    """
    values = np.asarray([volatility, sigma_v, rho], dtype=float)
    if not np.isfinite(values).all() or volatility <= 0 or sigma_v <= 0 or not -1 < rho < 0:
        raise ValueError("positive finite volatility/vol-of-vol and -1 < rho < 0 required")
    if (not np.isclose(rough_result["vol"], volatility)
            or not np.isclose(rough_result["parameters"]["rho"], rho)):
        raise ValueError("the comparison requires the rough run's same flat variance level and correlation")
    precision = np.asarray([rough_result["cpu_seconds"], rough_result["max_se_over_sigma"]], dtype=float)
    paths = rough_result["paths"]
    if (not np.isfinite(precision).all() or np.any(precision < 0)
            or isinstance(paths, bool) or not isinstance(paths, (int, np.integer)) or paths < 8 or paths % 2):
        raise ValueError("finite nonnegative rough timing/MC SE and an even path count of at least eight required")
    task = json.loads((CHECKPOINTS / "heston/task.json").read_text())
    tasks.validate_task(task)
    contracts = task["context"]["contracts"]
    if contracts != tasks.rough_task()["context"]["contracts"]:
        raise ValueError("the Heston and rough illustrations require the same ordered contract design")
    k, maturity, kind = surface.contract_arrays(contracts)
    rough = {**_smile_statistics(rough_result["implied"], k, maturity, "rough Bergomi"),
             "parameters": dict(rough_result["parameters"])}
    row = [volatility**2, volatility**2, sigma_v, rho]
    parameters, v0 = tasks._heston_split(tasks.model_parameters(task, row))
    started = time.perf_counter()
    prices, refinement = tasks.heston_prices(task, parameters, v0)
    elapsed = time.perf_counter() - started
    implied = surface.implied_vol(prices, k, maturity, kind)
    heston = {**_smile_statistics(implied, k, maturity, "Heston"), "prices": prices,
              "parameters": {"initial_variance": v0, **parameters}}
    return {"contracts": contracts, "log_moneyness": k,
            "standardized_moneyness": surface.standardized_moneyness(k, maturity),
            "maturity_days": maturity * surface.TRADING_DAYS,
            "rough": rough, "heston": heston,
            "metadata": {"common_mean_variance": volatility**2, "common_correlation": rho,
                         "rough_paths_per_block": int(rough_result["paths"]),
                         "rough_cpu_seconds": float(rough_result["cpu_seconds"]),
                         "rough_max_mc_se_over_quote_sigma": float(rough_result["max_se_over_sigma"]),
                         "rough_precision_policy": "fixed path count; no Monte Carlo SE acceptance threshold",
                         "heston_seconds": elapsed,
                         "heston_max_refinement_over_quote_sigma": float(np.max(np.abs(refinement) / tasks.sigma(task))),
                         "scope": "Illustrative forward prices on the same design; no model fitting, posterior or market-fit claim."}}


def _outside_markers(axis, x, y, xlim, ylim, color):
    """Mark finite off-scale values at the boundary without changing data."""
    x, y = np.asarray(x), np.asarray(y)
    finite = np.isfinite(x) & np.isfinite(y)
    outside = finite & ((x < xlim[0]) | (x > xlim[1]) | (y < ylim[0]) | (y > ylim[1]))
    for marker, selected in (("v", outside & (y > ylim[1])),
                              ("^", outside & (y <= ylim[1]))):
        if selected.any():
            axis.plot(np.clip(x[selected], *xlim), np.clip(y[selected], *ylim),
                      marker, color=color, markersize=5, linestyle="none", clip_on=False)
    return int(outside.sum())


def comparison_figure(result, *, maturity_days=5, scale="detail"):
    """Actual smiles and skew on fixed axes; rough Monte Carlo is reused unchanged."""
    import matplotlib.pyplot as plt

    if scale not in ("detail", "wide"):
        raise ValueError("choose the detail or wide axis preset")
    iv_limits, skew_limits = ((0, 100), (.02, 10)) if scale == "detail" else ((0, 250), (.002, 100))
    x_limits, day_limits = (-2.05, 2.05), (5, 252)
    chosen = np.isclose(result["maturity_days"], maturity_days, rtol=0, atol=1e-9)
    if not chosen.any():
        raise ValueError("choose a maturity present in the shared contract design")
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.7), layout="constrained")
    smile_outside, skew_outside = 0, 0
    for key, label, color in (("rough", "rough Bergomi", "#7755aa"), ("heston", "Heston", "#3a9874")):
        model = result[key]
        omitted = model["omitted_wing_quotes"]
        x, iv = result["standardized_moneyness"][chosen], 100 * model["implied"][chosen]
        days, skew = model["maturities"] * surface.TRADING_DAYS, np.abs(model["skew"])
        left.plot(x, iv, "o-",
                  color=color, markersize=3, label=label + (f" ({omitted} wing IVs unavailable)" if omitted else ""))
        right.loglog(days, skew, "o-",
                     color=color, markersize=4, label=label)
        smile_outside += _outside_markers(left, x, iv, x_limits, iv_limits, color)
        skew_outside += _outside_markers(right, days, skew, day_limits, skew_limits, color)
    left.set(xlim=x_limits, ylim=iv_limits,
             xlabel="Standardized log-moneyness k / (0.20 √T)", ylabel="Implied volatility (%)",
             title=f"{maturity_days:g}-day smile; {smile_outside} points outside axes")
    right.set(xlim=day_limits, ylim=skew_limits,
              xlabel="Maturity (trading days)", ylabel="Absolute ATM IV skew, |dσ / dk|",
              title=f"Finite-grid skew; {skew_outside} points outside axes")
    left.legend(fontsize=8)
    right.legend(fontsize=8)
    m = result["metadata"]
    figure.suptitle(f"Rough: {m['rough_paths_per_block']:,} paths/block, max MC SE/quote SD {m['rough_max_mc_se_over_quote_sigma']:.2f}; "
                   "Heston: Fourier prices\nEqual mean variance and correlation; no fitting", fontsize=10)
    return figure
