"""Figures for the SVJ validation and real-data reports (static PNG, light surface)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, aqua: the first three validated slots
CRITICAL = "#d03b3b"


def style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 9.5,
            "axes.titleweight": "normal",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
            "legend.labelcolor": INK_2,
            "lines.solid_capstyle": "round",
        }
    )


def _grid(n, ncols=3, width=9.6, height_per_row=2.4):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height_per_row * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes


def sbc_panels(result, labels, title):
    """ECDF of fractional ranks minus the uniform CDF, one panel per target, with the simultaneous band."""
    style()
    z, diff = result["z"], result["ecdf_minus_uniform"]
    fig, axes = _grid(diff.shape[1])
    for j, ax in enumerate(axes[: diff.shape[1]]):
        ax.fill_between(z, result["lower"], result["upper"], color=GRID, alpha=0.8, linewidth=0)
        ax.axhline(0, color=BASELINE, linewidth=0.8)
        inside = bool(result["inside"][j])
        ax.plot(z, diff[:, j], color=SERIES[0] if inside else CRITICAL, linewidth=1.5)
        ax.set_title(labels[j] + ("" if inside else "  (outside band)"), loc="left")
        ax.set_xlim(0, 1)
        lim = max(0.08, 1.15 * np.abs(np.concatenate([result["lower"], result["upper"], diff[:, j]])).max())
        ax.set_ylim(-lim, lim)
    for ax in axes[-3:]:
        ax.set_xlabel("fractional rank of the true value")
    for ax in axes[::3]:
        ax.set_ylabel("ECDF − uniform")
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=10.5, fontweight="bold")
    return fig


def recovery_panels(truths, draws, labels, contraction, title, n_show=300):
    """Posterior median and 90% interval against the true value, in the units of the prior table."""
    style()
    truths, draws = truths[:n_show], draws[:n_show]
    median = np.median(draws, axis=1)
    lo, hi = np.quantile(draws, 0.05, axis=1), np.quantile(draws, 0.95, axis=1)
    fig, axes = _grid(truths.shape[1])
    for j, ax in enumerate(axes[: truths.shape[1]]):
        ax.vlines(truths[:, j], lo[:, j], hi[:, j], color=SERIES[0], alpha=0.18, linewidth=0.8)
        ax.scatter(truths[:, j], median[:, j], s=14, color=SERIES[0], edgecolor=SURFACE, linewidth=0.4, zorder=3)
        lims = np.quantile(np.concatenate([truths[:, j], median[:, j]]), [0.005, 0.995])
        pad = 0.05 * (lims[1] - lims[0])
        lims = (lims[0] - pad, lims[1] + pad)
        ax.plot(lims, lims, color=INK_2, linewidth=0.8)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(f"{labels[j]}  ·  contraction {contraction[j]:.2f}", loc="left")
    for ax in axes[-3:]:
        ax.set_xlabel("true value")
    for ax in axes[::3]:
        ax.set_ylabel("posterior median")
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=10.5, fontweight="bold")
    return fig


def contraction_bars(values, labels, title, reference=None, reference_label=None):
    """Horizontal bars of contraction per target; an optional second series (e.g. the design document's values)."""
    style()
    y = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(values) + 1.0), constrained_layout=True)
    height = 0.36 if reference is not None else 0.6
    offset = height / 2 + 0.03 if reference is not None else 0
    ax.barh(y - offset, values, height=height, color=SERIES[0], label="this run")
    for yi, v in zip(y - offset, values):
        ax.text(max(v, 0) + 0.01, yi, f"{v:.2f}", va="center", color=INK_2, fontsize=8)
    if reference is not None:
        ax.barh(y + offset, reference, height=height, color=SERIES[1], label=reference_label)
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=2, borderaxespad=0.2)
    ax.axvline(0, color=BASELINE, linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(min(0.0, float(np.min(values)) - 0.05), 1.08)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("contraction: 1 − posterior variance / prior variance")
    ax.set_title(title, loc="left", pad=22 if reference is not None else 6)
    return fig


def interval_dots(rows, nominal, xlabel, title):
    """Dot-and-whisker plot: rows = [(label, estimate, (lo, hi))], with a reference line at the nominal value."""
    style()
    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(rows) + 1.2), constrained_layout=True)
    y = np.arange(len(rows))
    ax.axvline(nominal, color=INK_2, linewidth=0.8)
    for yi, (label, est, (lo, hi)) in zip(y, rows):
        ax.hlines(yi, lo, hi, color=SERIES[0], linewidth=1.5)
        ax.scatter([est], [yi], s=36, color=SERIES[0], edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    return fig


def method_panels(panels, row_labels, methods, title, height_per_row=0.9, colors=None):
    """Side-by-side panels comparing methods per row, each value with a 95% interval.

    panels = [(xlabel, reference, {(row, method): (estimate, lo, hi)})]; one color per method (`colors` maps a
    method to its color, so the same method keeps its color across figures), with a legend under the plots.
    """
    colors = colors or {}
    style()
    fig, axes = plt.subplots(
        1, len(panels), figsize=(9.6, height_per_row * len(row_labels) + 1.6), sharey=True, constrained_layout=True
    )
    y = np.arange(len(row_labels))
    offsets = np.linspace(-0.15, 0.15, len(methods)) if len(methods) > 1 else [0.0]
    for ax, (xlabel, reference, values) in zip(np.atleast_1d(axes), panels):
        ax.axvline(reference, color=INK_2, linewidth=0.8)
        for k, (method, offset) in enumerate(zip(methods, offsets)):
            color = colors.get(method, SERIES[k % len(SERIES)])
            for r, row in enumerate(row_labels):
                if (row, method) not in values:
                    continue
                est, lo, hi = values[(row, method)]
                ax.hlines(y[r] + offset, lo, hi, color=color, linewidth=1.5)
                ax.scatter([est], [y[r] + offset], s=36, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3,
                           label=method if r == 0 else None)
        ax.set_yticks(y, row_labels)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(xlabel)
    np.atleast_1d(axes)[0].invert_yaxis()  # shared y: inverting one inverts all
    handles, names = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncols=len(methods))
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=10.5, fontweight="bold")
    return fig


def rolling_panels(dates, panels, title):
    """Stacked time-series panels sharing the date axis.

    Each panel is a dict: {"ylabel", "median", "lo", "hi", "extra": [(label, values, color)]} for a posterior
    band, or {"ylabel", "pvalues"} for misspecification p-values with the 5% line.
    """
    style()
    fig, axes = plt.subplots(len(panels), 1, figsize=(9.6, 2.1 * len(panels)), sharex=True, constrained_layout=True)
    for ax, panel in zip(np.atleast_1d(axes), panels):
        if "pvalues" in panel:
            p = np.asarray(panel["pvalues"])
            flagged = p < 0.05
            ax.scatter(dates[~flagged], p[~flagged], s=12, color=SERIES[0], edgecolor=SURFACE, linewidth=0.4, label="p ≥ 0.05")
            ax.scatter(dates[flagged], p[flagged], s=16, color=CRITICAL, edgecolor=SURFACE, linewidth=0.4, label="p < 0.05: flagged")
            ax.axhline(0.05, color=INK_2, linewidth=0.8)
            ax.set_ylim(0, 1.02)
            ax.legend(loc="upper left", ncols=2)
        else:
            ax.fill_between(dates, panel["lo"], panel["hi"], color=SERIES[0], alpha=0.12, linewidth=0)
            ax.plot(dates, panel["median"], color=SERIES[0], linewidth=1.5, label="posterior median, 90% band")
            for label, values, color in panel.get("extra", ()):
                ax.plot(dates, values, color=color, linewidth=1.2, label=label)
            if panel.get("extra"):
                ax.legend(loc="upper left", ncols=2)
        ax.set_ylabel(panel["ylabel"])
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=10.5, fontweight="bold")
    return fig


def agreement_panels(panels, labels):
    """Medians and 90% intervals per target, standardized by the particle-MCMC posterior of that target.

    panels = [(title, reference_draws, network_draws, corrected_draws, truth)], all (n, 9) in target coordinates.
    """
    style()
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6), sharex=True, constrained_layout=True)
    methods = (("particle MCMC", INK_2, -0.22), ("network", SERIES[0], 0.0), ("network + importance sampling", SERIES[2], 0.22))
    y = np.arange(len(labels))
    extent = 3.0
    for _, reference, network, corrected, truth in panels:
        centre, scale = reference.mean(0), reference.std(0)
        for draws in (network, corrected):
            extent = max(extent, np.abs(np.quantile((draws - centre) / scale, [0.05, 0.95], axis=0)).max())
        extent = max(extent, np.abs((truth - centre) / scale).max())
    for ax, (title, reference, network, corrected, truth) in zip(axes.ravel(), panels):
        centre, scale = reference.mean(0), reference.std(0)
        for (name, color, offset), draws in zip(methods, (reference, network, corrected)):
            std = (draws - centre) / scale
            lo, mid, hi = np.quantile(std, [0.05, 0.5, 0.95], axis=0)
            ax.hlines(y + offset, lo, hi, color=color, linewidth=1.5, label=name)
            ax.scatter(mid, y + offset, s=18, color=color, edgecolor=SURFACE, linewidth=0.8, zorder=3)
        ax.scatter((truth - centre) / scale, y, marker="*", s=55, color=INK, edgecolor=SURFACE, linewidth=0.5, zorder=4, label="true value")
        ax.axvline(0, color=BASELINE, linewidth=0.8)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlim(-1.1 * extent, 1.1 * extent)
        ax.set_title(title, loc="left")
    for ax in axes[1]:
        ax.set_xlabel("standardized by the particle-MCMC posterior (mean 0, sd 1)")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside upper left", ncols=4)
    return fig
