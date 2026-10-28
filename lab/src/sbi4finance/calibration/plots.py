"""Figures for the calibration studies, in the SVJ reports' style (static PNG, light surface)."""

import numpy as np

from ..svj import plots as svj_plots
from ..svj.plots import BASELINE, INK, INK_2, SERIES, SURFACE, plt, style


def sbc(result, labels, title, points=201):
    """ECDF difference of fractional ranks with the simultaneous DKW band, via the SVJ panel helper."""
    from .validation import ecdf_difference
    grid, diff = ecdf_difference(result["fractional_ranks"], points)
    band = np.full(points, result["dkw_bound"])
    return svj_plots.sbc_panels({"z": grid, "ecdf_minus_uniform": diff, "lower": -band, "upper": band,
                                 "inside": result["passed"]}, labels, title)


recovery = svj_plots.recovery_panels
contraction = svj_plots.contraction_bars


def agreement(panels, labels, reference_name):
    """Medians and 90% intervals per parameter, standardized by the reference posterior of each panel.

    panels = [(title, reference_draws, network_draws, corrected_draws_or_None, truth)].
    """
    style()
    n = len(panels)
    fig, axes = plt.subplots(int(np.ceil(n / 2)), min(n, 2), figsize=(9.6, 1.0 + 0.55 * len(labels) * np.ceil(n / 2)),
                             sharex=True, constrained_layout=True, squeeze=False)
    methods = ((reference_name, INK_2, -0.22), ("network", SERIES[0], 0.0), ("network + importance sampling", SERIES[2], 0.22))
    y = np.arange(len(labels))
    extent = 3.0
    for _, reference, *others, truth in panels:
        centre, scale = reference.mean(0), reference.std(0)
        for draws in others:
            if draws is not None:
                extent = max(extent, np.abs(np.quantile((draws - centre) / scale, [0.05, 0.95], axis=0)).max())
        extent = max(extent, np.abs((truth - centre) / scale).max())
    for ax, (title, reference, network, corrected, truth) in zip(axes.ravel(), panels):
        centre, scale = reference.mean(0), reference.std(0)
        for (name, color, offset), draws in zip(methods, (reference, network, corrected)):
            if draws is None:
                continue
            lo, mid, hi = np.quantile((draws - centre) / scale, [0.05, 0.5, 0.95], axis=0)
            ax.hlines(y + offset, lo, hi, color=color, linewidth=1.5, label=name)
            ax.scatter(mid, y + offset, s=18, color=color, edgecolor=SURFACE, linewidth=0.8, zorder=3)
        ax.scatter((truth - centre) / scale, y, marker="*", s=55, color=INK, edgecolor=SURFACE, linewidth=0.5,
                   zorder=4, label="true value")
        ax.axvline(0, color=BASELINE, linewidth=0.8)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlim(-1.1 * extent, 1.1 * extent)
        ax.set_title(title, loc="left")
    for ax in axes[-1]:
        ax.set_xlabel(f"standardized by the {reference_name} posterior (mean 0, sd 1)")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside upper left", ncols=4)
    return fig
