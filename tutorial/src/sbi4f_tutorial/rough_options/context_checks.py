"""Necessary curve-context support checks; no estimator or market-data loading."""

import numpy as np

from ..shared import tasks


def context_support(task, levels):
    """Test necessary interval-level ratios for the known linear-wobble prior.

    One common scale cancels from xi[j] / xi[0]. The remaining ratio has the
    extremal slope factor times exp(+/-2*wobble). A violation proves that the
    supplied curve lies outside training-context support. Passing does not
    establish full joint support: scale bounds and a common slope also matter.
    """
    tasks.validate_task(task)
    prior = task.get("context_prior", {}).get("forward_variance", {})
    if (task["model"] != "rough" or tasks.is_joint_curve_task(task)
            or tasks.curve_kind(prior) != "linear_wobble" or not prior):
        raise ValueError("this support check requires a known-curve linear_wobble context prior")
    if np.iscomplexobj(levels) or np.ma.isMaskedArray(levels):
        raise ValueError("positive finite unmasked real curve levels are required")
    values = np.asarray(levels, dtype=float)
    midpoints = tasks.interval_midpoints(task)
    if values.shape != midpoints.shape or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("one positive finite curve level per task interval is required")
    slopes = np.asarray(prior["slope_bounds"])
    slope_ratios = (1 + slopes[:, None] * midpoints) / (1 + slopes[:, None] * midpoints[0])
    wobble = prior["log_perturbation"]
    lower = slope_ratios.min(axis=0) * np.exp(-2 * wobble)
    upper = slope_ratios.max(axis=0) * np.exp(2 * wobble)
    lower[0] = upper[0] = 1.0  # the anchor uses the very same interval shock
    ratios = values / values[0]
    # Relative tolerance protects the mathematical boundary from rounding.
    outside = (ratios < lower * (1 - 1e-12)) | (ratios > upper * (1 + 1e-12))
    return {
        "observed_ratios": ratios, "lower_ratios": lower, "upper_ratios": upper,
        "midpoints_years": midpoints, "violating_intervals": (np.flatnonzero(outside) + 1).tolist(),
        "violates_necessary_bounds": bool(outside.any()),
        "status": "outside_support" if outside.any() else "not_ruled_out",
        "observed_end_ratio": float(ratios[-1]), "maximum_end_ratio": float(upper[-1]),
        "reference": {"anchor_interval": 1, "anchor_level": float(values[0]),
                      "slope_bounds": slopes.tolist(), "log_wobble_half_width": wobble,
                      "formula": "xi_j = level^2 * (1 + slope * midpoint_j) * exp(u_j), |u_j| <= wobble"},
        "scope": "Necessary ratio bounds only. A violation proves outside support; passing does not establish full joint context support.",
    }


def context_support_figure(task, levels):
    """Plot curve-level ratios against necessary bounds under the supplied task."""
    import matplotlib.pyplot as plt

    check = context_support(task, levels)
    time = check["midpoints_years"]
    figure, axis = plt.subplots(figsize=(8.5, 3.4), layout="constrained")
    axis.fill_between(time, check["lower_ratios"], check["upper_ratios"],
                      color="#94a3b8", alpha=.28, label="necessary context-prior bounds")
    axis.plot(time, check["observed_ratios"], "o-", color="#7755aa", markersize=5,
              label="supplied variance-curve levels / first level")
    outside = np.asarray(check["violating_intervals"], dtype=int) - 1
    if len(outside):
        axis.scatter(time[outside], check["observed_ratios"][outside], s=90,
                     facecolors="none", edgecolors="#b91c1c", linewidths=1.5, label="outside necessary bounds")
    axis.axhline(1, color="#64748b", linewidth=.8, linestyle=":")
    axis.set(xlabel="Interval midpoint (years)", ylabel="Variance level / first interval level",
             title="Can the supplied curve come from this context prior?")
    axis.legend(fontsize=8, loc="best")
    return figure
