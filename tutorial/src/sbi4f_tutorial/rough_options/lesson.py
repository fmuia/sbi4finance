"""Rough Bergomi forward-simulation and smile illustrations.

Edit the computations behind the rough-option notebook here. Shared tables
and plotting helpers are reused by both option lessons.
"""

import numpy as np

from ..shared import surface
from ..heston_options.lesson import heston_smile
from ..shared.teaching import (
    LABELS, TENORS, PRODUCT_NAMES, SPX_SKEW_EXPONENT,
    load_json, load_npz, existing, design_arrays, smile_rows, flat_versus_smile,
    skew_term_structures, pairs, reserve_rows, price_histograms, agreement_rows,
    cost_rows, load_network, implied_vol_grid,
)


def price_rough_smile(hurst, eta, rho, *, paths=8192, vol=0.20, seed=7):
    """One clean rough-Bergomi smile on the design grid at a flat curve: implied vols and skew statistics."""
    import time

    from ..shared import pricing, tasks

    task = tasks.rough_task()
    parameters = {"hurst": float(hurst), "eta": float(eta), "rho": float(rho)}
    curve = tasks.curve(task, np.full(tasks.context_size(task), vol**2))
    contracts = task["context"]["contracts"]
    started = time.process_time()
    prices, errors, _, _ = pricing.rough_surface(parameters, curve, contracts, task["numerics"]["blocks"], [paths],
                                                 tasks.sigma(task), np.inf, seed)
    cpu = time.process_time() - started
    k, maturity, kind = surface.contract_arrays(contracts)
    implied = surface.implied_vol(prices, k, maturity, kind)
    maturities, atm, skew, curvature = surface.smile_statistics(implied, k, maturity)
    exponent, scale = surface.skew_power_law(maturities, skew)
    return {"parameters": parameters, "vol": vol, "implied": implied, "maturities": maturities, "atm": atm,
            "skew": skew, "exponent": exponent, "scale": scale, "cpu_seconds": cpu, "paths": paths,
            "max_se_over_sigma": float(np.max(errors / tasks.sigma(task)))}


def live_smile_figure(result):
    from ..shared.plots import INK_2, SERIES, plt, style

    style()
    k, maturity, _, delta = design_arrays()
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.9), constrained_layout=True)
    days = sorted(TENORS)
    shades = plt.get_cmap("Blues")(np.linspace(0.35, 0.95, len(days)))
    for color, d in zip(shades, days):
        chosen = np.isclose(maturity * surface.TRADING_DAYS, d)
        left.plot(delta[chosen], 100 * result["implied"][chosen], color=color, linewidth=1.4, label=TENORS[d])
    left.axhline(100 * result["vol"], color=INK_2, linestyle=":", linewidth=1.0, label="√ξ₀, the flat curve")
    left.set_xlabel("standardized moneyness δ")
    left.set_ylabel("implied volatility (%)")
    left.legend(fontsize=7, ncol=2, loc="upper right")
    p = result["parameters"]
    left.set_title(f"rough Bergomi, H = {p['hurst']:.2f}, η = {p['eta']:.1f}, ρ = {p['rho']:.2f}", loc="left", fontsize=9)
    t = np.asarray(result["maturities"]) * surface.TRADING_DAYS
    right.loglog(t, np.abs(result["skew"]), "o", color=SERIES[0], ms=5, label="|ATM skew| of this smile")
    grid = np.geomspace(t.min(), t.max(), 40)
    right.loglog(grid, result["scale"] * (grid / surface.TRADING_DAYS) ** result["exponent"], "--", color=SERIES[0],
                 linewidth=1.0, label=f"fitted exponent {result['exponent']:.2f}")
    anchor = abs(result["skew"][0])
    right.loglog(grid, anchor * (grid / t[0]) ** (p["hurst"] - 0.5), ":", color=INK_2, linewidth=1.2,
                 label=f"slope H − ½ = {p['hurst'] - 0.5:.2f}")
    right.set_xticks(days, ["1w", "2w", "1m", "2m", "3m", "6m", "1y"])
    right.minorticks_off()
    low, high = right.get_ylim()
    values = np.unique(np.round(np.geomspace(low * 1.05, high / 1.05, 5), 2))
    right.set_yticks(values, [f"{v:.2f}" for v in values])
    right.set_xlabel("maturity")
    right.legend(fontsize=7.5, loc="lower left")
    right.set_title(f"{result['paths']:,} paths per block, {result['cpu_seconds']:.1f} CPU-s", loc="left", fontsize=9)
    return fig
