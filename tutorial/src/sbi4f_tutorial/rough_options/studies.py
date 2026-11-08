"""Analytic teaching extensions for the current rough-option posterior.

These functions propagate complete approximate-posterior rows through variance
curve averages. They neither simulate realized payoffs nor fit a new estimator.
"""

import numpy as np

from ..shared import full_surface, pricing, tasks


MAX_MATURITY = 5.0


def _rows(task, draws):
    full_surface.validate_task(task)
    if not tasks.is_joint_curve_task(task):
        raise ValueError("the current six-target joint-curve task is required")
    if np.iscomplexobj(draws) or np.ma.isMaskedArray(draws):
        raise ValueError("complete real unmasked joint rows are required")
    values = np.asarray(draws, dtype=float)
    bounds = np.asarray(task["prior_bounds"], dtype=float)
    if (values.ndim != 2 or values.shape[1] != 6 or not len(values)
            or not np.isfinite(values).all() or np.any(values < bounds[:, 0])
            or np.any(values > bounds[:, 1]) or np.any(values[:, 3:] <= 0)):
        raise ValueError("finite joint rows within the declared support and positive curve coefficients are required")
    return values


def _times(times):
    if np.iscomplexobj(times) or np.ma.isMaskedArray(times):
        raise ValueError("real unmasked maturities are required")
    values = np.atleast_1d(np.asarray(times, dtype=float))
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > MAX_MATURITY):
        raise ValueError("maturities must be finite and between zero and five years")
    return values


def _integrals(values, times):
    return np.asarray([pricing.heston_integrated_variance(times, *row[3:]) for row in values])


def curve_functionals(task, draws):
    """Kvar(T)=integral_0^T xi(t)dt/T, one result per original paired row."""
    values = _rows(task, draws)
    maturities = np.array([.25, 1.0])
    rates = _integrals(values, maturities) / maturities
    return {"variance_swap_3m": rates[:, 0], "variance_swap_1y": rates[:, 1]}


def supplied_curve_functionals(task, curve_values):
    """Deterministic Kvar for the supplied piecewise-constant curve, not an ablation.

    Conditional on an exactly supplied curve these rates have zero parameter-
    induced width. The known-curve and joint examples have different observations.
    """
    full_surface.validate_task(task)
    if not full_surface.uses_known_curve(task):
        raise ValueError("a supplied-curve rough task is required")
    if np.iscomplexobj(curve_values) or np.ma.isMaskedArray(curve_values):
        raise ValueError("real positive curve levels are required")
    levels = np.asarray(curve_values, dtype=float)
    knots = tasks.knots(task)
    if levels.shape != (len(knots) - 1,) or not np.isfinite(levels).all() or np.any(levels <= 0):
        raise ValueError("one finite positive curve level per interval is required")
    rates = [float(np.sum(levels * np.maximum(0, np.minimum(knots[1:], t) - knots[:-1])) / t)
             for t in (.25, 1.0)]
    return dict(zip(("variance_swap_3m", "variance_swap_1y"), rates, strict=True))


def extended_integrated_variance(task, draws, times, alpha):
    """Keep xi unchanged through one year, then multiply it by exp(alpha*(t-1)).

    Alpha is one independent tail slope per complete joint row. Results have
    shape [rows, maturities]. The alpha=0 limit is evaluated without division by
    zero. Overflow is an error; rows are never clipped, dropped or redrawn.
    """
    values, times = _rows(task, draws), _times(times)
    if np.iscomplexobj(alpha) or np.ma.isMaskedArray(alpha):
        raise ValueError("one finite real tail slope per row is required")
    alpha = np.asarray(alpha, dtype=float)
    if alpha.shape != (len(values),) or not np.isfinite(alpha).all():
        raise ValueError("one finite tail slope per row is required")
    if max(c["maturity"] for c in task["context"]["contracts"]) > 1:
        raise ValueError("the tail extension requires all observed quotes to expire by one year")
    base = _integrals(values, np.minimum(times, 1.0))
    short, long, decay = values[:, 3], values[:, 4], values[:, 5]
    duration = np.maximum(times - 1.0, 0.0)
    with np.errstate(over="raise", invalid="raise"):
        try:
            def exponential_integral(rate):
                z = rate[:, None] * duration
                factor = np.ones_like(z)
                np.divide(np.expm1(z), z, out=factor, where=z != 0)
                return duration * factor

            tail = (long[:, None] * exponential_integral(alpha)
                    + ((short - long) * np.exp(-decay))[:, None] * exponential_integral(alpha - decay))
            result = base + tail
            if np.any(alpha == 0):
                result[alpha == 0] = _integrals(values[alpha == 0], times)
        except FloatingPointError as error:
            raise ValueError("tail integral overflow; no rows were discarded") from error
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("nonfinite or negative tail integral; no rows were discarded")
    return result


def tail_functionals(task, draws, width, *, seed=20261108, maturity=3.0):
    """An unlearned N(0,width^2) tail prior multiplied by the current q(theta|y).

    Reusing the seed retains the same independent standard normals as width
    changes. The zero-width baseline uses the same complete parameter rows and
    recovers their original exponential curve beyond one year.
    """
    values = _rows(task, draws)
    if isinstance(width, (bool, np.bool_)) or np.ndim(width) != 0 or not np.isfinite(width) or not 0 <= width <= .5:
        raise ValueError("tail-prior width must be between zero and 0.5 per year")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    t = _times(maturity)
    if len(t) != 1 or t[0] <= 1:
        raise ValueError("the tail exercise requires one maturity after one year and at most five years")
    maturity = float(t[0])
    alpha = float(width) * np.random.default_rng(int(seed)).standard_normal(len(values))
    totals = extended_integrated_variance(task, values, [1, maturity], alpha)
    base = _integrals(values, np.array([1.0, maturity]))
    return {"alpha": alpha, "variance_swap": totals[:, 1] / maturity,
            "forward_variance": (totals[:, 1] - totals[:, 0]) / (maturity - 1),
            "base_variance_swap": base[:, 1] / maturity,
            "base_forward_variance": (base[:, 1] - base[:, 0]) / (maturity - 1),
            "metadata": {"maturity_years": maturity, "tail_start_years": 1.0,
                         "tail_prior_sd_per_year": float(width), "seed": int(seed), "rows": len(values),
                         "scope": "analytic curve rates under q(theta|quotes) times an independent unchanged tail prior"}}


def curve_functional_figure(task, draws, *, comparison_draws=None):
    """Show paired curve averages, optionally contrasting independently shuffled marginals."""
    from ..shared.plots import SERIES, plt
    rates = curve_functionals(task, draws)
    comparison = None
    if comparison_draws is not None:
        original, shuffled = _rows(task, draws), _rows(task, comparison_draws)
        if original.shape != shuffled.shape or not np.array_equal(np.sort(original, axis=0), np.sort(shuffled, axis=0)):
            raise ValueError("the comparison must preserve the same parameter marginals")
        comparison = curve_functionals(task, shuffled)
    figure, axes = plt.subplots(1, 2, figsize=(9, 2.8), layout="constrained")
    for ax, (name, values), label in zip(axes, rates.items(), ("3 months", "1 year"), strict=True):
        bins = np.histogram_bin_edges(np.r_[values, comparison[name]] if comparison else values, bins=35)
        ax.hist(values, bins=bins, color=SERIES[0], alpha=.65, label="Paired posterior rows")
        if comparison:
            ax.hist(comparison[name], bins=bins, histtype="step", linewidth=1.5,
                    color=SERIES[1], label="Independent marginals (not posterior)")
        ax.set(xlabel="Annualized mean variance", ylabel="Draw count", title="Kvar · " + label)
        ax.legend(fontsize=7)
    figure.suptitle("Analytic curve rates from the current joint posterior", fontsize=11)
    return figure


def tail_figure(task, draws, width, seed=20261108, *, maturity=3.0):
    """Compare an independent tail prior with its original-curve baseline on the same rows."""
    from ..shared.plots import SERIES, plt
    result = tail_functionals(task, draws, width, seed=seed, maturity=maturity)
    figure, axes = plt.subplots(1, 2, figsize=(10, 2.9), layout="constrained")
    for ax, name, title in zip(axes, ("variance_swap", "forward_variance"),
                               (f"Kvar · 0–{maturity:g} years", f"Forward variance · 1–{maturity:g} years"), strict=True):
        values, base = result[name], result["base_" + name]
        bins = np.histogram_bin_edges(np.r_[values, base], bins=35)
        ax.hist(base, bins=bins, histtype="step", color="#64748b", linewidth=1.5, label="Original curve (α = 0)")
        ax.hist(values, bins=bins, alpha=.5, color=SERIES[0], label=f"Tail-prior SD = {width:g}/year")
        ax.set(xlabel="Annualized mean variance", ylabel="Draw count", title=title)
        ax.legend(fontsize=8)
    figure.suptitle("Same fitted curve through one year; additional tail uncertainty stays prior-driven", fontsize=10)
    return figure
