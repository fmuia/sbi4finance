"""Small synthetic-data and supervised-pricer illustrations, without asset writes."""

import time

import numpy as np

from ..shared import full_surface, surface, tasks
from ..shared.bank_helpers import _integer
from . import model, simulator, surrogate


def simulate_dataset(task, parameters, *, context_values=None, maturity=.25, paths=4096,
                     seed=7, noise_seed=11, noise_scale=1.):
    """Paths -> terminal payoffs -> price estimates -> independent quote errors.

    Raw path payoffs illustrate the contract. Price estimates average bounded
    conditional put values, with calls obtained by parity. Integrating out the
    independent price shock makes those Monte Carlo SEs well-defined even when
    raw call payoffs lack a second moment. This small fixed-path illustration
    uses no antithetic/control-variate fit or bank precision acceptance gate.
    """
    full_surface.validate_task(task)
    if task["model"] != "rough":
        raise ValueError("a complete-design rough-option task is required")
    paths = _integer(paths, "paths", minimum=16)
    if paths > 4096:
        raise ValueError("the teaching example is capped at 4096 paths")
    seed, noise_seed = _integer(seed, "seed"), _integer(noise_seed, "noise_seed")
    if (isinstance(noise_scale, bool) or np.ndim(noise_scale) != 0
            or not np.isfinite(noise_scale) or not 0 <= noise_scale <= 10):
        raise ValueError("noise_scale must be finite and between zero and ten")
    if np.iscomplexobj(parameters) or np.ma.isMaskedArray(parameters):
        raise ValueError("one complete finite real parameter row is required")
    theta = np.asarray(parameters, dtype=float)
    bounds = np.asarray(task["prior_bounds"])
    if (theta.shape != (len(bounds),) or not np.isfinite(theta).all()
            or np.any(theta < bounds[:, 0]) or np.any(theta > bounds[:, 1])):
        raise ValueError("one parameter row inside the declared prior is required")
    if full_surface.uses_known_curve(task):
        if np.iscomplexobj(context_values) or np.ma.isMaskedArray(context_values):
            raise ValueError("positive unmasked real supplied curve levels are required")
        levels = np.asarray(context_values, dtype=float)
        if levels.shape != (tasks.context_size(task),) or not np.isfinite(levels).all() or np.any(levels <= 0):
            raise ValueError("positive supplied curve levels are required for the known-curve task")
        curve = tasks.curve(task, levels)
    else:
        if context_values is not None:
            raise ValueError("the joint curve comes from the parameter row, not supplied context")
        curve = tasks.joint_curve(task, theta)
    all_contracts = task["context"]["contracts"]
    all_maturities = np.asarray([c["maturity"] for c in all_contracts])
    if (isinstance(maturity, bool) or np.ndim(maturity) != 0 or not np.isfinite(maturity) or maturity <= 0):
        raise ValueError("choose a positive finite maturity present in the task")
    chosen = np.flatnonzero(np.isclose(all_maturities, maturity, rtol=0, atol=1e-12))
    if not len(chosen):
        raise ValueError("choose a maturity present in the task")
    contracts = [all_contracts[i] for i in chosen]
    block = next(b for b in task["numerics"]["blocks"] if maturity <= b["horizon"] + 1e-12)
    dt = block["horizon"] / block["n_steps"]
    index = int(np.rint(maturity / dt))
    if not np.isclose(index * dt, maturity, rtol=0, atol=1e-12):
        raise ValueError("the selected maturity must land on the original block grid")
    parameters = tasks.model_parameters(task, theta)
    started = time.perf_counter()
    innovations = np.random.default_rng(np.random.SeedSequence([0x44535054, seed])).standard_normal(
        (paths, block["n_steps"], 3))
    native = simulator.simulate_rough(parameters, curve, dt, innovations, method="hybrid",
                                       spot=1., rate=0., dividend=0.)
    terminal = np.exp(native["log_spot"][:, index])
    strikes = np.asarray([c["strike"] for c in contracts])
    is_put = np.asarray([c["option_type"] == "put" for c in contracts])
    raw_payoffs = np.maximum(np.where(is_put, strikes - terminal[:, None], terminal[:, None] - strikes), 0)
    puts = [dict(c, option_type="put") for c in contracts]
    conditional_puts = model.conditional_put_payoffs(native, parameters, puts, [index] * len(puts),
                                                     spot=1., rate=0., dividend=0.)
    prices = conditional_puts.mean(axis=0) + np.where(is_put, 0., 1. - strikes)
    mc_se = conditional_puts.std(axis=0, ddof=1) / np.sqrt(paths)
    if not all(np.isfinite(a).all() for a in (terminal, raw_payoffs, conditional_puts, prices, mc_se)):
        raise ValueError("nonfinite teaching simulation; no paths were redrawn or repaired")
    sigma = tasks.sigma(task)[chosen] * noise_scale
    errors = np.random.default_rng(np.random.SeedSequence([0x44535154, noise_seed])).standard_normal(len(chosen)) * sigma
    return {"terminal_prices": terminal, "payoffs": raw_payoffs, "conditional_put_values": conditional_puts,
            "mean_prices": prices, "mc_se": mc_se, "observations": prices + errors,
            "quote_errors": errors, "quote_noise_std": sigma, "strikes": strikes,
            "option_types": np.where(is_put, "put", "call"), "contract_indices": chosen,
            "parameters": theta.copy(), "metadata": {"paths": paths, "seed": seed, "noise_seed": noise_seed,
                "noise_scale": float(noise_scale), "maturity_years": float(maturity),
                "block": dict(block), "runtime_seconds": time.perf_counter() - started,
                "pricing_estimator": "iid bounded conditional puts; calls by put-call parity",
                "scope": "finite-grid teaching illustration; no bank MC precision gate or posterior validation"}}


def _outside(x, y, xlim, ylim):
    return int(np.sum((x < xlim[0]) | (x > xlim[1]) | (y < ylim[0]) | (y > ylim[1])))


def dataset_figure(result, *, contract=4, terminal_limits=(.5, 1.5), payoff_limits=(0, .5),
                   strike_limits=(.65, 1.55), price_limits=(-.01, .12)):
    """Fixed axes preserve comparisons; every omitted point is counted visibly."""
    import matplotlib.pyplot as plt

    contract = _integer(contract, "contract")
    if contract >= len(result["strikes"]) or result["option_types"][contract] != "put":
        raise ValueError("select a put for the bounded raw-payoff illustration")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), layout="constrained")
    terminal, payoffs = result["terminal_prices"], result["payoffs"][:, contract]
    axes[0].scatter(terminal, payoffs, s=5, alpha=.15, color="#7755aa")
    axes[0].axvline(result["strikes"][contract], color="#29303d", lw=.8, linestyle=":",
                    label=f"put strike / forward = {result['strikes'][contract]:.2f}")
    outside = _outside(terminal, payoffs, terminal_limits, payoff_limits)
    axes[0].set(xlim=terminal_limits, ylim=payoff_limits, xlabel="Terminal price / forward",
                ylabel="Put payoff / forward", title=f"1. Each path gives a payoff\n{outside} points outside fixed axes")
    axes[0].legend(fontsize=7)
    x = result["strikes"]
    axes[1].errorbar(x, result["mean_prices"], yerr=2 * result["mc_se"], fmt="o-", color="#7755aa", ms=4)
    axes[1].set(xlim=strike_limits, ylim=price_limits, xlabel="Strike / forward", ylabel="Estimated premium / forward",
                title=f"2. Average conditional put values\n{_outside(x, result['mean_prices'], strike_limits, price_limits)} means outside axes")
    axes[2].plot(x, result["mean_prices"], "--", color="#7755aa", label="estimated mean")
    axes[2].errorbar(x, result["observations"], yerr=result["quote_noise_std"], fmt="o", color="#2e9675", ms=4,
                     label="observation ± quote SD")
    axes[2].set(xlim=strike_limits, ylim=price_limits, xlabel="Strike / forward", ylabel="Observed premium / forward",
                title=f"3. Add independent quote errors\n{_outside(x, result['observations'], strike_limits, price_limits)} quotes outside axes")
    axes[2].legend(fontsize=7)
    fig.suptitle(f"{result['metadata']['paths']:,} paths · mean-price bars are ±2 MC SE · calls use put-call parity", fontsize=10)
    return fig


def pricer_validation_figure(pricer, task, validation_split, *, market=0, maturity_days=63,
                             strike_limits=(.65, 1.55), price_limits=(-.02, .12), residual_limits=(-25, 25)):
    """Compare physical predictions with clean numerical labels, then divide by quote SD."""
    import matplotlib.pyplot as plt

    full_surface.validate_task(task)
    market = _integer(market, "market")
    theta = np.asarray(validation_split["parameters"])
    if market >= len(theta):
        raise ValueError("choose one of the validation markets")
    contracts = task["context"]["contracts"]
    k, maturities, _ = surface.contract_arrays(contracts)
    chosen = np.isclose(maturities * surface.TRADING_DAYS, maturity_days, rtol=0, atol=1e-9)
    if not chosen.any():
        raise ValueError("choose a maturity present in the task")
    context = validation_split.get("context_values")
    context = None if context is None else np.asarray(context)[market:market + 1]
    predicted = pricer(surrogate.features(task, theta[market:market + 1], context))[0] * tasks.sigma(task)
    reference = np.asarray(validation_split["reference_prices"])[market]
    se = np.asarray(validation_split["pricing_standard_errors"])[market]
    if (predicted.shape != (63,) or reference.shape != (63,) or se.shape != (63,)
            or not all(np.isfinite(a).all() for a in (predicted, reference, se)) or np.any(se < 0)):
        raise ValueError("complete finite predictions, numerical labels and label MC SEs are required")
    residual = (predicted - reference) / tasks.sigma(task)
    x = np.exp(k[chosen])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    axes[0].errorbar(x, reference[chosen], yerr=2 * se[chosen], fmt="o", ms=5, color="#29303d",
                     label="MC mean-price label ±2 MC SE")
    axes[0].plot(x, predicted[chosen], "s--", ms=4, color="#7755aa", label="neural pricer prediction")
    label_out = _outside(x, reference[chosen], strike_limits, price_limits)
    pred_out = _outside(x, predicted[chosen], strike_limits, price_limits)
    axes[0].set(xlim=strike_limits, ylim=price_limits, xlabel="Strike / forward", ylabel="Option premium / forward",
                title=f"Physical prices · {label_out} labels / {pred_out} predictions outside axes")
    axes[0].legend(fontsize=7)
    axes[1].axhspan(-1, 1, color="#c5dfd2", alpha=.65, label="±1 quote SD reference")
    axes[1].axhline(0, color="#29303d", lw=.8)
    axes[1].plot(x, residual[chosen], "o-", ms=4, color="#7755aa")
    axes[1].set(xlim=strike_limits, ylim=residual_limits, xlabel="Strike / forward",
                ylabel="(prediction − MC mean label) / quote SD",
                title=f"Signed errors · {_outside(x, residual[chosen], strike_limits, residual_limits)} outside axes")
    axes[1].legend(fontsize=7)
    rmse = np.sqrt(np.mean(residual**2))
    fig.suptitle(f"Validation market {market + 1} · {maturity_days:g} days · full-surface RMSE {rmse:.2f} quote SDs\n"
                 "Large errors mean this small fit is unsuitable for posterior inference; the labels also have MC error.", fontsize=10)
    return fig
