"""Calibration and analytic teaching extensions for current rough options.

These functions propagate complete approximate-posterior rows through variance
curve averages. An explicit local point fit uses a loaded deterministic neural
pricer. No function simulates future payoffs or trains a new neural estimator.
"""

import numpy as np

from ..shared import full_surface, pricing, tasks
from ..shared import parameter_transforms as transforms
from ..shared.full_surface_teaching import fingerprint


MAX_MATURITY = 5.0
CALIBRATION_LOGIT_LIMIT = 12.0


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


def point_calibration(pricer, task, observation, *, starts=3, seed=20261109, max_evaluations=500):
    """Local multistart least squares for one complete joint-curve observation.

    The native joint pricer accepts six parameter logits and returns 63 prices
    divided by their declared quote SDs. The residual is prediction - y/sigma;
    no logit Jacobian or prior-density penalty enters this point objective.
    The parameter box supplies constraints, rather than a claim of certainty.

    Optimize within z in [-12,12]^6 to avoid numerical endpoints. Start zero
    is the physical box midpoint; other starts are independent uniform-prior
    draws restricted to this explicitly retained numerical box. They never use
    the generating truth or posterior samples. max_evaluations is a HARD cap
    per start, including residual calls used for finite-difference derivatives.
    A failed/exhausted start is recorded; if none converges, raise without
    presenting an unfinished iterate as a fitted answer.

    Load and bind the native pricer to this exact task before calling. Identity
    is the loader's responsibility; a converged solver is not a global-minimum
    certificate or an accuracy check of the surrogate or posterior network.
    """
    from scipy.optimize import least_squares

    full_surface.validate_task(task)
    if not tasks.is_joint_curve_task(task):
        raise ValueError("point calibration requires the six-target joint-curve task")
    if not callable(pricer):
        raise ValueError("a callable native joint pricer is required")
    if np.iscomplexobj(observation) or np.ma.isMaskedArray(observation):
        raise ValueError("one complete real unmasked observation is required")
    observed = np.asarray(observation, dtype=float)
    if observed.shape != (63,) or not np.isfinite(observed).all():
        raise ValueError("one complete finite 63-price observation is required")
    for name, value, upper in (("starts", starts, 20), ("max_evaluations", max_evaluations, 10000)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not 1 <= value <= upper):
            raise ValueError(f"{name} must be an integer between one and {upper}")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    bounds = transforms.bounds_of(task)
    z_lower = np.full(6, -CALIBRATION_LOGIT_LIMIT)
    z_upper = -z_lower
    physical_lower = transforms.to_bounded(z_lower, bounds)
    physical_upper = transforms.to_bounded(z_upper, bounds)
    random = np.random.default_rng(int(seed))
    initial = [np.zeros(6)]
    if starts > 1:
        physical = random.uniform(physical_lower, physical_upper, size=(int(starts) - 1, 6))
        initial.extend(transforms.to_unbounded(physical, bounds))
    sigma = tasks.sigma(task)
    target = observed / sigma
    runs, candidates = [], []

    class EvaluationBudget(Exception):
        pass

    for index, start in enumerate(initial):
        calls = 0

        def residual(z):
            nonlocal calls
            if calls >= max_evaluations:
                raise EvaluationBudget
            calls += 1
            predicted = pricer(np.asarray(z)[None, :])
            if np.iscomplexobj(predicted) or np.ma.isMaskedArray(predicted):
                raise ValueError("the pricer returned complex or masked predictions")
            predicted = np.asarray(predicted, dtype=float)
            if predicted.shape != (1, 63) or not np.isfinite(predicted).all():
                raise ValueError("the pricer must return one finite complete scaled-price vector")
            result = predicted[0] - target
            if not np.isfinite(result).all():
                raise ValueError("the scaled price residual is nonfinite")
            return result

        record = {"start_index": index, "initial_logits": np.asarray(start).tolist(),
                  "initial_parameters": transforms.to_bounded(start, bounds).tolist()}
        try:
            fit = least_squares(residual, start, bounds=(z_lower, z_upper), method="trf",
                                jac="2-point", x_scale="jac", max_nfev=int(max_evaluations),
                                ftol=1e-8, xtol=1e-8, gtol=1e-8)
            chi2 = float(fit.fun @ fit.fun)
            usable = bool(fit.success and np.isfinite(chi2) and np.isfinite(fit.x).all())
            record.update(success=usable, status="converged" if usable else "not_converged",
                          message=str(fit.message), scipy_status=int(fit.status), chi2=chi2,
                          parameters=transforms.to_bounded(fit.x, bounds).tolist(),
                          active_numerical_bounds=np.asarray(fit.active_mask, dtype=int).tolist(),
                          evaluations=calls)
            if usable:
                candidates.append((chi2, index, fit.x.copy(), fit.fun.copy()))
        except EvaluationBudget:
            record.update(success=False, status="evaluation_budget", evaluations=calls,
                          message="Per-start residual-call budget exhausted, including numerical derivatives.")
        except (ValueError, FloatingPointError) as error:
            record.update(success=False, status="numerical_failure", evaluations=calls, message=str(error))
        runs.append(record)
    if not candidates:
        statuses = ", ".join(run["status"] for run in runs)
        raise RuntimeError(f"No local start converged; no point fit returned. Start outcomes: {statuses}.")
    chi2, selected, z, errors = min(candidates, key=lambda item: item[0])
    return {
        "parameters": transforms.to_bounded(z, bounds), "logits": z, "chi2": chi2,
        "residuals_noise_units": errors, "fitted_prices": (errors + target) * sigma,
        "observation": observed.copy(), "task_fingerprint": fingerprint(task),
        "selected_start": selected, "runs": runs,
        "settings": {"starts": int(starts), "seed": int(seed),
                     "max_residual_evaluations_per_start": int(max_evaluations),
                     "numerical_logit_bounds": [-CALIBRATION_LOGIT_LIMIT, CALIBRATION_LOGIT_LIMIT],
                     "retained_physical_bounds": np.column_stack([physical_lower, physical_upper]).tolist(),
                     "start_policy": "midpoint plus independent uniform-prior starts restricted to the numerical box",
                     "objective": "sum_c ((surrogate_price_c - observation_c) / quote_sd_c)^2; no Jacobian"},
        "scope": "Lowest-cost converged local multistart solution under the native neural pricer; no global-optimum or interval claim.",
    }


def calibration_effects(task, posterior_rows, point_fit=None):
    """Carry paired uncertainty and a point fit into the same curve and fair strike.

    With no fitted pricer point, return posterior propagation and None for
    every point field; no posterior median is substituted as a classical fit.
    This transforms a conditional parameter distribution into another expected
    product quantity. It does not forecast future realized variance, establish
    calibration, or imply that classical inference cannot supply intervals.
    """
    rows = _rows(task, posterior_rows)
    point = None
    if point_fit is not None:
        if not isinstance(point_fit, dict) or point_fit.get("task_fingerprint") != fingerprint(task):
            raise ValueError("the point fit must be bound to this exact task")
        point = _rows(task, np.asarray(point_fit["parameters"])[None, :])[0]
    times = np.linspace(0., 1., 161)
    curves = np.asarray([pricing.heston_variance_values(times, *row[3:]) for row in rows])
    point_curve = None if point is None else pricing.heston_variance_values(times, *point[3:])
    rates = curve_functionals(task, rows)["variance_swap_1y"]
    point_rate = None if point is None else float(curve_functionals(task, point[None, :])["variance_swap_1y"][0])
    return {"times_years": times, "posterior_curves": curves, "point_curve": point_curve,
            "posterior_variance_swap_1y": rates, "point_variance_swap_1y": point_rate,
            "point_parameters": None if point is None else point.copy(),
            "scope": "Analytic fair one-year variance-swap strikes, not realized payoffs or future risk forecasts."}


def calibration_effect_figure(task, posterior_rows, point_fit=None):
    """Show how uncertainty and a plug-in point reach another expected product.

    The curve band is pointwise central 90%, not simultaneous coverage. Neither
    the approximate posterior nor the locally optimized pricer target is assumed
    more accurate. A point estimate alone supplies no interval; bootstrap or
    other classical procedures could also propagate parameter uncertainty.
    """
    from ..shared.plots import SERIES, plt

    result = calibration_effects(task, posterior_rows, point_fit)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.2), layout="constrained")
    lo, middle, hi = np.quantile(result["posterior_curves"], [.05, .5, .95], axis=0)
    t = result["times_years"]
    axes[0].fill_between(t, lo, hi, color=SERIES[0], alpha=.25, label="Posterior curve · pointwise 90%")
    axes[0].plot(t, middle, color=SERIES[0], linewidth=1.3, label="Pointwise posterior median")
    if point_fit is not None:
        axes[0].plot(t, result["point_curve"], color=SERIES[2], ls="--", linewidth=1.5,
                     label="Local least-squares curve")
    axes[0].set(xlabel="Maturity (years)", ylabel="Annualized forward variance",
                title="Same option data → variance curves", xlim=(0, 1))
    axes[1].hist(result["posterior_variance_swap_1y"], bins=35, color=SERIES[0], alpha=.65,
                 label="Paired posterior rows")
    if point_fit is not None:
        axes[1].axvline(result["point_variance_swap_1y"], color=SERIES[2], ls="--", linewidth=1.5,
                        label="Plug-in least-squares strike")
    axes[1].set(xlabel="Fair one-year variance-swap strike", ylabel="Draw count",
                title="Carry the calibration to another product")
    for axis in axes:
        axis.legend(fontsize=7, loc="best")
    if point_fit is None:
        figure.suptitle("Posterior propagation; load a pricer to add the point-fit comparison", fontsize=10)
    return figure


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
