"""Loaders and figures for the slot-5 notebook. Everything reads accepted results; nothing trains.

The notebook lives in tutorial_final/notebooks and its results in
tutorial_final/results/calibration. A missing result returns None, so the
notebook can say "pending" instead of failing. Only the explicit live cells
load a network, through load_network, which initializes the neural backend.
"""

import json
from pathlib import Path

import numpy as np

from . import surface

LABELS = {"hurst": "H", "eta": "η", "rho": "ρ", "initial_variance": "v₀", "theta": "θ",
          "vol_of_vol": "σ_v", "rho_eta": "ρη"}
TENORS = {5: "1 week", 10: "2 weeks", 21: "1 month", 42: "2 months", 63: "3 months", 126: "6 months",
          252: "1 year"}
PRODUCT_NAMES = {"variance_swap": "Variance swap, 3m (variance strike)",
                 "volatility_swap": "Volatility swap, 3m (volatility strike)",
                 "forward_start_call": "Forward-start ATM call, 3m into 6m",
                 "put_out_of_grid": "90% put, 2y (outside the grid)"}
SPX_SKEW_EXPONENT = -0.4  # Bayer, Friz and Gatheral (2016)


def load_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def load_npz(path):
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def existing(path):
    path = Path(path)
    return path if path.exists() else None


def design_arrays():
    """Log-moneyness, maturity, option type and delta of the 63 design contracts, in task order."""
    contracts = surface.design_grid()
    k, maturity, kind = surface.contract_arrays(contracts)
    return k, maturity, kind, surface.standardized_moneyness(k, maturity)


def smile_rows(example):
    """Per maturity: ATM implied volatility, skew and curvature of the clean smile."""
    clean = example["clean"]
    rows = []
    formula = example.get("short_maturity_formula_skew")
    for j, t in enumerate(clean["maturities"]):
        days = int(round(t * surface.TRADING_DAYS))
        row = {"maturity": TENORS.get(days, f"{days} days"), "ATM implied vol (%)": 100 * clean["atm"][j],
               "ATM skew": clean["skew"][j], "ATM curvature": clean["curvature"][j]}
        if formula is not None:
            row["short-maturity formula skew"] = formula[j]
        rows.append(row)
    return rows


def flat_versus_smile(example, days, ax):
    """One maturity: the model smile, its noisy quotes and the one Black–Scholes volatility at the money."""
    from .plots import INK_2, SERIES

    k, maturity, _, delta = design_arrays()
    chosen = np.isclose(maturity * surface.TRADING_DAYS, days)
    clean = 100 * np.asarray(example["implied_vol_clean"])[chosen]
    noisy = 100 * np.asarray(example["implied_vol_noisy"])[chosen]
    bars = 100 * np.asarray(example["implied_vol_noise_sd"])[chosen]
    atm = clean[np.argmin(np.abs(delta[chosen]))]
    ax.plot(delta[chosen], clean, color=SERIES[0], linewidth=1.6, label="model smile")
    ax.errorbar(delta[chosen], noisy, yerr=bars, fmt="o", ms=4, color=SERIES[0], elinewidth=0.9, capsize=0,
                label="noisy quotes, ±1 sd")
    ax.axhline(atm, color=INK_2, linestyle="--", linewidth=1.2, label=f"Black–Scholes: one volatility, {atm:.1f}%")
    ax.set_xlabel("standardized moneyness δ = ln(K/F) / (0.2 √T)")
    ax.set_ylabel("implied volatility (%)")
    ax.legend(fontsize=8, loc="upper right")
    return ax


def skew_term_structures(examples, ax):
    """|ATM skew| against maturity, log–log, for each example, with a slope of −0.4 for reference."""
    from .plots import INK_2, SERIES

    for color, (name, example) in zip(SERIES, examples.items()):
        clean = example["clean"]
        t = np.asarray(clean["maturities"]) * surface.TRADING_DAYS
        ax.loglog(t, np.abs(clean["skew"]), "o-", color=color, ms=5, linewidth=1.2,
                  label=f"{name}: fitted exponent {clean['skew_exponent']:.2f}")
    first = next(iter(examples.values()))["clean"]
    t0 = 21.0
    s0 = abs(np.interp(t0, np.asarray(first["maturities"]) * surface.TRADING_DAYS, first["skew"]))
    grid = np.geomspace(5, 252, 20)
    ax.loglog(grid, s0 * (grid / t0) ** SPX_SKEW_EXPONENT, ":", color=INK_2, linewidth=1.2,
              label="slope −0.4, the SPX power law (BFG 2016)")
    ticks = sorted(TENORS)
    ax.set_xticks(ticks, ["1w", "2w", "1m", "2m", "3m", "6m", "1y"])
    ax.minorticks_off()
    low, high = ax.get_ylim()
    values = np.unique(np.round(np.geomspace(low * 1.05, high / 1.05, 5), 2))
    ax.set_yticks(values, [f"{v:.2f}" for v in values])
    ax.set_xlabel("maturity")
    ax.set_ylabel("|ATM skew|")
    ax.legend(fontsize=8, loc="lower left")
    return ax


def pairs(draw_sets, labels, truth=None, figsize=(7.2, 7.2), points=1500, seed=0):
    """Corner plot: one scatter per pair below the diagonal, one histogram per parameter on it."""
    from .plots import INK, SERIES, plt, style

    style()
    names = list(draw_sets)
    p = len(labels)
    fig, axes = plt.subplots(p, p, figsize=figsize, constrained_layout=True)
    rng = np.random.default_rng(seed)
    for i in range(p):
        for j in range(p):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            for color, name in zip(SERIES, names):
                draws = np.asarray(draw_sets[name])
                if i == j:
                    ax.hist(draws[:, i], bins=40, density=True, histtype="step", color=color, linewidth=1.3,
                            label=name)
                else:
                    shown = draws[rng.choice(len(draws), min(points, len(draws)), replace=False)]
                    ax.scatter(shown[:, j], shown[:, i], s=2, color=color, alpha=0.3)
            if truth is not None:
                if i == j:
                    ax.axvline(truth[i], color=INK, linewidth=1.0)
                else:
                    ax.scatter(truth[j], truth[i], marker="*", s=70, color=INK, edgecolor="white", linewidth=0.6,
                               zorder=4)
            if i == j:
                ax.set_yticks([])
            if i == p - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])
            elif j > 0:
                ax.set_yticklabels([])
    axes[0, 0].legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.05, 1.0))
    return fig


def reserve_rows(price_draws, row, level):
    """Point price, posterior mean, range and the reserve of a short position at the given level."""
    products = [str(p) for p in price_draws["products"]]
    prices = np.asarray(price_draws[f"prices_{row}"])
    point = np.asarray(price_draws[f"point_prices_{row}"])
    rows = []
    for j, product in enumerate(products):
        values = prices[:, j]
        q_low, q_high = np.quantile(values, [1 - level, level])
        reserve = max(q_high - point[j], 0.0)
        rows.append({"product": PRODUCT_NAMES.get(product, product), "least-squares price": point[j],
                     "posterior mean": values.mean(), f"{100 * (1 - level):.0f}% quantile": q_low,
                     f"{100 * level:.0f}% quantile": q_high, "reserve (short)": reserve,
                     "reserve / price (%)": 100 * reserve / point[j]})
    return rows


def price_histograms(price_draws, row, level, title):
    from .plots import INK, SERIES, plt, style

    style()
    products = [str(p) for p in price_draws["products"]]
    prices = np.asarray(price_draws[f"prices_{row}"])
    point = np.asarray(price_draws[f"point_prices_{row}"])
    fig, axes = plt.subplots(1, len(products), figsize=(11.5, 2.8), constrained_layout=True)
    for j, (ax, product) in enumerate(zip(axes, products)):
        values = prices[:, j]
        ax.hist(values, bins=40, color=SERIES[0], alpha=0.75)
        ax.axvline(point[j], color=SERIES[1], linewidth=1.8, label="least squares")
        ax.axvline(np.quantile(values, level), color=INK, linestyle="--", linewidth=1.1,
                   label=f"{100 * level:.0f}% quantile")
        ax.set_title(PRODUCT_NAMES.get(product, product), loc="left", fontsize=8.5)
        ax.set_yticks([])
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    return fig


def cost_rows(costs):
    """The S9 table: one likelihood evaluation, the exact route and the amortized route, per model."""
    rows = []
    rough = costs.get("rough_likelihood_evaluation_cpu_s")
    if rough:
        rows.append({"route": f"Rough Bergomi: one likelihood evaluation (63 prices, {costs['rough_paths']:,} paths per block)",
                     "CPU cost": f"{rough['median']:.1f} s"})
    exact = costs.get("exact_rough_posterior")
    if exact:
        rows.append({"route": f"Rough Bergomi: exact posterior by MCMC, 400 effective draws "
                              f"({exact['evaluations_for_400_effective_draws']:,.0f} evaluations, extrapolated)",
                     "CPU cost": f"{exact['cpu_hours']:.1f} h"})
    for name, label in (("rough", "Rough Bergomi"), ("heston", "Heston")):
        amortized = costs.get(f"{name}_amortized_posterior_1000_draws_cpu_s")
        if amortized:
            rows.append({"route": f"{label}: amortized posterior, 1,000 draws", "CPU cost": f"{amortized['median']:.2f} s"})
    heston = costs.get("heston_likelihood_evaluation_cpu_s")
    if heston:
        rows.append({"route": "Heston: one likelihood evaluation (63 Fourier prices)",
                     "CPU cost": f"{1000 * heston['median']:.0f} ms"})
    return rows


def price_rough_smile(hurst, eta, rho, *, paths=8192, vol=0.20, seed=7):
    """One clean rough-Bergomi smile on the design grid at a flat curve: implied vols and skew statistics."""
    import time

    from . import pricing, tasks

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
    from .plots import INK_2, SERIES, plt, style

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


def load_network(checkpoint):
    """(task, workflow) from a run directory with model.keras, task.json and training.json."""
    from . import inference

    checkpoint = Path(checkpoint)
    task = json.loads((checkpoint / "task.json").read_text())
    record = json.loads((checkpoint / "training.json").read_text())
    workflow = inference.build_workflow(task, subnet_widths=tuple(record["training"]["subnet_widths"]))
    workflow.load_approximator(checkpoint / "model.keras")
    return task, workflow
