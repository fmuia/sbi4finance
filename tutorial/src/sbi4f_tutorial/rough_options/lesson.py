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


def _session_settings(task, *, max_paths, max_seconds, seed):
    """Keep the published numerical task; truncate work, never its precision rule."""
    from ..shared import full_surface
    from ..shared.bank_helpers import _integer

    full_surface.validate_task(task)
    if task["model"] != "rough":
        raise ValueError("a complete-design rough-option task is required")
    max_paths = _integer(max_paths, "max_paths", minimum=2)
    seed = _integer(seed, "seed")
    if (isinstance(max_seconds, bool) or np.ndim(max_seconds) != 0
            or not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be positive and finite")
    numerics = task["numerics"]
    ladder = [n for n in numerics["paths_ladder"] if n <= max_paths]
    if not ladder:
        raise ValueError("max_paths is below the task's first Monte Carlo rung")
    # A looser last-rung cap belongs only to the task's actual last rung.
    cap = (numerics.get("max_mc_se_over_quote_sigma_last_rung")
           if ladder[-1] == numerics["paths_ladder"][-1] else None)
    return ladder, cap, float(max_seconds), seed


def _session_price(task, parameters, context_values, indices, ladder, cap, seed):
    from ..shared import pricing, tasks

    curve = (tasks.joint_curve(task, parameters) if tasks.is_joint_curve_task(task)
             else tasks.curve(task, context_values))
    contracts = [task["context"]["contracts"][i] for i in indices]
    prices, errors, _, paths = pricing.rough_surface(
        tasks.model_parameters(task, parameters), curve, contracts,
        task["numerics"]["blocks"], ladder, tasks.sigma(task)[indices],
        task["numerics"]["max_mc_se_over_quote_sigma"], seed, cap=cap)
    if (not np.isfinite(prices).all() or not np.isfinite(errors).all()
            or np.any(errors < 0)):
        raise ValueError("nonfinite prices or invalid Monte Carlo standard errors")
    return prices, errors, paths


def rough_posterior_predictive(task, posterior_draws, observations, *, context_values=None,
                               draws=8, maturity=0.25, seed=2026, max_paths=131072,
                               max_seconds=180, quote_replicates=8):
    """Reprice joint rows and add fresh quote errors, without writing lesson assets.

    The selected maturity retains its original block horizon, time grid and
    precision budget. Repricing uses independent streams per selected row.
    Reported model-price spread still contains Monte Carlo error; its standard
    errors exclude time-discretization bias. No draw is silently skipped on a
    budget failure. The wall-time guard is checked between complete surfaces.
    """
    import time

    from ..shared import full_surface, tasks
    from ..shared.bank_helpers import _integer

    ladder, cap, max_seconds, seed = _session_settings(
        task, max_paths=max_paths, max_seconds=max_seconds, seed=seed)
    draws = _integer(draws, "draws", minimum=2)
    quote_replicates = _integer(quote_replicates, "quote_replicates", minimum=1)
    if draws > 32 or quote_replicates > 32:
        raise ValueError("session illustration is capped at 32 parameter rows and 32 quote replicates per row")
    values = np.asarray(posterior_draws, dtype=float)
    bounds = np.asarray(task["prior_bounds"])
    if (values.ndim != 2 or values.shape[1] != len(bounds) or len(values) < draws
            or not np.isfinite(values).all() or np.any(values <= bounds[:, 0])
            or np.any(values >= bounds[:, 1])):
        raise ValueError("finite joint posterior rows strictly inside the task's prior bounds are required")
    observed = np.asarray(observations, dtype=float)
    if observed.shape != (63,) or not np.isfinite(observed).all():
        raise ValueError("one complete finite observation with 63 quotes is required")
    if full_surface.uses_known_curve(task):
        context_values = np.asarray(context_values, dtype=float)
        if (context_values.shape != (tasks.context_size(task),)
                or not np.isfinite(context_values).all() or np.any(context_values <= 0)):
            raise ValueError("the known-curve task requires its observed positive curve levels")
    elif context_values is not None:
        raise ValueError("joint-curve predictions derive the curve from each complete draw; supply no curve context")
    maturities = np.asarray([c["maturity"] for c in task["context"]["contracts"]])
    if isinstance(maturity, bool) or np.ndim(maturity) != 0 or not np.isfinite(maturity):
        raise ValueError("maturity must match one of the task's contract maturities")
    indices = np.flatnonzero(np.isclose(maturities, maturity, rtol=0, atol=1e-12))
    if not len(indices):
        raise ValueError("maturity must match one of the task's contract maturities")
    selected = np.linspace(0, len(values) - 1, draws, dtype=int)
    prices, errors, paths = [], [], []
    started = time.perf_counter()
    for index in selected:
        if time.perf_counter() - started > max_seconds:
            raise RuntimeError("posterior predictive exceeded its session time budget; no partial result returned")
        price_seed = int(np.random.SeedSequence([0x52505043, seed, int(index)]).generate_state(1)[0])
        try:
            p, e, used = _session_price(task, values[index], context_values, indices, ladder, cap, price_seed)
        except (ValueError, FloatingPointError, OverflowError) as error:
            raise ValueError(f"posterior row {index} could not meet the session pricing budget: {error}") from error
        prices.append(p)
        errors.append(e)
        paths.append(used)
    elapsed = time.perf_counter() - started
    if elapsed > max_seconds:
        raise RuntimeError("posterior predictive exceeded its session time budget; no partial result returned")
    prices, errors, paths = np.asarray(prices), np.asarray(errors), np.asarray(paths)
    sigma = tasks.sigma(task)[indices]
    noise_rng = np.random.default_rng(np.random.SeedSequence([0x52514E4F, seed]))
    replicated = prices[:, None, :] + sigma * noise_rng.standard_normal((draws, quote_replicates, len(indices)))
    return {
        "reference_prices": prices, "mc_se": errors,
        "replicated_quotes": replicated.reshape(-1, len(indices)),
        "observations": observed[indices], "quote_noise_std": sigma,
        "strikes": np.asarray([task["context"]["contracts"][i]["strike"] for i in indices]),
        "contract_indices": indices, "selected_parameters": values[selected].copy(),
        "metadata": {"draws": draws, "posterior_indices": selected.tolist(),
                     "quote_replicates_per_draw": quote_replicates, "seed": seed,
                     "maturity_years": float(maturity), "runtime_seconds": elapsed,
                     "session_time_budget_seconds": max_seconds,
                     "paths_per_draw_per_block": paths.tolist(), "paths_ladder": ladder,
                     "target_se_over_sigma": task["numerics"]["max_mc_se_over_quote_sigma"],
                     "allowed_last_rung_cap": cap,
                     "max_mc_se_over_sigma": float(np.max(errors / sigma)),
                     "mc_se_excludes_discretization_bias": True,
                     "scope": "session-only approximate posterior-predictive illustration; no posterior validation"},
    }


def generate_training_pairs(task, *, count=8, seed=2026, max_paths=131072, max_seconds=90):
    """A fresh bounded prior bank for demonstrating the native offline API.

    This samples the declared parameter and curve priors. It never reads the
    selected cached validation observation, rejects a difficult prior row, or
    writes assets. Numerical-budget failure stops the entire illustration.
    """
    import time

    from ..shared import tasks
    from ..shared.bank_helpers import _integer

    ladder, cap, max_seconds, seed = _session_settings(
        task, max_paths=max_paths, max_seconds=max_seconds, seed=seed)
    count = _integer(count, "count", minimum=2)
    if count > 64:
        raise ValueError("session training generation is capped at 64 examples")
    bounds = np.asarray(task["prior_bounds"])
    sigma = tasks.sigma(task)
    streams = np.random.SeedSequence([0x5254524E, seed]).spawn(4)
    prior_rng, context_rng, quote_rng, pricing_rng = [np.random.default_rng(s) for s in streams]
    parameters = prior_rng.uniform(bounds[:, 0], bounds[:, 1], size=(count, len(bounds)))
    prices, errors, contexts, paths = [], [], [], []
    started = time.perf_counter()
    for row in parameters:
        if time.perf_counter() - started > max_seconds:
            raise RuntimeError("training-pair generation exceeded its session time budget; no partial bank returned")
        levels = (tasks.derived_context(task, row)[0] if tasks.is_joint_curve_task(task)
                  else tasks.draw_context(task, context_rng)[0])
        price_seed = int(pricing_rng.integers(0, 2**32))
        try:
            p, e, used = _session_price(task, row, levels, np.arange(63), ladder, cap, price_seed)
        except (ValueError, FloatingPointError, OverflowError) as error:
            raise ValueError(f"fresh prior row {len(prices)} could not meet the session pricing budget: {error}") from error
        prices.append(p)
        errors.append(e)
        contexts.append(levels)
        paths.append(used)
    elapsed = time.perf_counter() - started
    if elapsed > max_seconds:
        raise RuntimeError("training-pair generation exceeded its session time budget; no partial bank returned")
    prices, errors = np.asarray(prices), np.asarray(errors)
    split = {"parameters": parameters, "reference_prices": prices,
             "observations": prices + sigma * quote_rng.standard_normal(prices.shape),
             "context_values": np.asarray(contexts), "pricing_standard_errors": errors}
    metadata = {"examples": count, "seed": seed, "runtime_seconds": elapsed,
                "session_time_budget_seconds": max_seconds,
                "paths_ladder": ladder, "paths_per_example_per_block": np.asarray(paths).tolist(),
                "target_se_over_sigma": task["numerics"]["max_mc_se_over_quote_sigma"],
                "allowed_last_rung_cap": cap,
                "max_mc_se_over_sigma": float(np.max(errors / sigma)),
                "scope": "fresh session-only training illustration; no accuracy or calibration claim"}
    return split, metadata



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


def conditioning_example(task, observed_prices, context_values=None):
    """Expose the exact pre-standardization vector used by the option adapter.

    One surface is one row. Curve levels are observed inputs only for the
    supplied-curve task; generating parameter labels never enter this vector.
    The workflow subsequently standardizes with statistics learned at fitting.
    """
    from ..shared import full_surface, tasks

    levels = None if context_values is None else np.asarray(context_values)[None, :]
    data = full_surface.conditioning_data(np.asarray(observed_prices)[None, :], task,
                                           context_values=levels)
    quotes = data["observations"][0]
    curve = data.get("curve_context", np.empty((1, 0)))[0]
    return {"scaled_quotes": quotes, "log_curve_levels": curve,
            "condition_vector": np.r_[quotes, curve],
            "prices": np.asarray(observed_prices), "quote_sd": tasks.sigma(task),
            "curve_levels": None if context_values is None else np.asarray(context_values)}


def conditioning_figure(task, observed_prices, context_values=None):
    """Show append-versus-transform operations on one actual market's inputs."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    data = conditioning_example(task, observed_prices, context_values)
    known = len(data["log_curve_levels"]) > 0
    figure, ax = plt.subplots(figsize=(10.5, 3.6), layout="constrained")
    ax.set(xlim=(0, 1), ylim=(-.12, 1.05)); ax.axis("off")

    def box(x, y, w, h, title, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.012",
                                    facecolor=color, edgecolor="#d8dde3"))
        ax.text(x+w/2, y+h*.76, title, ha="center", va="center", fontsize=12, weight="bold")
        ax.text(x+w/2, y+h*.32, text, ha="center", va="center", fontsize=10, linespacing=1.4)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13,
                                     color="#667581", linewidth=1.2))

    index = 4  # the at-the-money quote in the first expiry group
    box(.01, .56, .29, .36, "63 quotes → divide by quote SD",
        f"Example: {data['prices'][index]:.4g} / {data['quote_sd'][index]:.4g}\n"
        f"= {data['scaled_quotes'][index]:.3g}\n7 expiries × 9 strikes, in fixed order", "#f3f4f6")
    if known:
        box(.01, .07, .29, .36, "7 curve levels → natural log",
            f"Example: log({data['curve_levels'][0]:.4g})\n= {data['log_curve_levels'][0]:.3g}\nOne level per maturity interval", "#f1edf7")
        arrow((.31,.24),(.38,.47))
    else:
        box(.01, .07, .29, .36, "No supplied curve inputs",
            "Curve coefficients are targets\ninside the six-parameter vector", "#f1edf7")
    arrow((.31,.73),(.38,.58))
    count = len(data["condition_vector"])
    box(.39, .32, .27, .42, "Append the lists",
        ("63 quote/SD values\n+ 7 log curve levels\n= 70 entries in one row" if known else
         "63 quote/SD values\n= 63 entries in one row"), "#f7f5fa")
    arrow((.67,.53),(.73,.53))
    box(.74, .32, .25, .42, f"{count}-value market condition",
        "Standardize using training stats\nThen condition the flow\nNo target parameters appended", "#edf7f2")
    ax.text(.01,-.035,"Training labels: " + ", ".join(task["parameter_names"])
            + ". New-market inference supplies no labels.", fontsize=10, color="#667581")
    return figure
