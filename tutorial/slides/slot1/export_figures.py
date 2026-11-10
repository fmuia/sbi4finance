"""Draw slot 1's figures: vector PDFs in the BayesOps theme's colours, at the size the slides show them.

Run from tutorial/:

    python slides/slot1/export_figures.py [--rubik PATH/TO/Rubik-Variable.ttf] [--recompute]

Seven of the eight PDFs come from the deck's running example (`illustration.py`: Heston on simulated quotes, the
model right by construction; the opening figure has two steps). The eighth, `coverage`, plots the stored validation
of the tutorial's Heston network (study S2). The example takes about four minutes to compute on four cores, mostly
five posteriors by importance sampling; its numbers are cached under tutorial/.cache/ (ignored by Git) and reused
until `illustration.py`, the settings below or `compute` change, or --recompute is given. The script ends by printing
every number the slides quote, and says which of them `slot1.tex` does not contain.

With --rubik (the variable font in the Overleaf project's fonts/ folder, or Google Fonts' Rubik[wght].ttf) the
figures use Rubik like the slides; without it, matplotlib's default sans-serif. The PDFs land in
slides/slot1/figures/slot1/.
"""

import argparse
import hashlib
import inspect
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sbi4f_tutorial.shared import plots  # noqa: E402

import illustration as il  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures" / "slot1"
CACHE = HERE.parents[1] / ".cache" / "slides-slot1"
VALIDATION = HERE.parents[1] / "assets" / "results" / "calibration" / "heston" / "s2" / "validation.json"

# The theme's colours (aviation #15153B, neonlake #6969FF) and the figures' second colour.
INK, INK_2, MUTED, GRID, BASELINE = "#15153b", "#3f3f63", "#7c7c99", "#e7e7f1", "#c4c4d6"
POSTERIOR, CLASSICAL = "#6969ff", "#c2187a"  # periwinkle: the posterior; magenta: least squares and its ranges

TEXT_WIDTH = 5.51  # inches: the 140 mm text width of a 16:9 Beamer frame
HOOK = "3 months"  # the reading of the opening example: 14 quotes, at 1 and 3 months
FIXED_KAPPA = (1.0, 2.0, 4.0)  # the habit of fixing the mean reversion, at three conventions
FAN = 120  # posterior curves drawn in a fan
SMILE_STRIKES = {"1 month": (0.875, 1.075), "3 months": (0.80, 1.125), "2 years": (0.60, 1.30)}
PRODUCTS = {"put": il.put_price, "forward_start": il.forward_start_call}
DASHED = (0, (4, 2))
RANGE_LEGEND = ("least squares, delta method", "posterior, 5% to 95%", "price under the generating parameters")


# ---------------------------------------------------------------------------------------------------------------
# Numbers


def key_of(reach):
    return reach.replace(" ", "_")


def _posterior(reach):
    return reach, il.posterior(il.designs()[reach])


def compute():
    """Every number the figures need, as a dictionary of arrays and scalars."""
    designs = il.designs()
    with ProcessPoolExecutor(max_workers=min(len(il.REACH), os.cpu_count() or 1)) as pool:
        posteriors = dict(pool.map(_posterior, il.REACH))
    data = {"truth/put": il.put_price(il.TRUTH), "truth/forward_start": il.forward_start_call(il.TRUTH)}

    def classical(prefix, result):
        data[f"{prefix}/fit"] = result.theta
        data[f"{prefix}/chi2"] = result.chi2
        data[f"{prefix}/on_bound"] = np.array(result.on_bound, dtype=str)
        for name, product in PRODUCTS.items():
            point, interval = il.delta_range(result, product)
            data[f"{prefix}/{name}/point"] = point
            data[f"{prefix}/{name}/delta"] = np.array(interval if interval is not None else (np.nan, np.nan))

    for reach in il.REACH:
        design, post, key = designs[reach], posteriors[reach], key_of(reach)
        result = il.fit(design)
        classical(key, result)
        for name, product in PRODUCTS.items():
            data[f"{key}/{name}/draws"] = np.array([product(theta) for theta in post.draws])
        data[f"{key}/draws"] = post.draws
        data[f"{key}/ess"] = post.effective_sample_size
        data[f"{key}/proposals"] = post.proposals
        data[f"{key}/failed"] = post.failed
        data[f"{key}/quotes"] = len(design.contracts)
        if result.covariance is not None:
            data[f"{key}/ellipse_kappa_theta"] = il.gauss_newton_ellipse(result, 2, 1)
            data[f"{key}/ellipse_xi_rho"] = il.gauss_newton_ellipse(result, 3, 4)

    # The opening example: smiles of the fit, the truth and a fan of posterior draws, quoted and not quoted.
    design, draws = designs[HOOK], posteriors[HOOK].draws
    mid, low, high = design.implied_quotes()
    data["hook/quotes"] = np.column_stack([np.exp(design.k), design.maturity, mid, low, high])
    data["hook/misfit"] = np.array([np.sqrt(np.mean(design.residuals(theta) ** 2)) for theta in draws])
    data["hook/misfit_fit"] = np.sqrt(data[f"{key_of(HOOK)}/chi2"] / len(design.contracts))
    for label, (lowest, highest) in SMILE_STRIKES.items():
        maturity = il.DAYS[il.REACH.index(label)] / 252
        strikes = np.linspace(lowest, highest, 41)
        key = f"hook/{key_of(label)}"
        data[f"{key}/strikes"] = strikes
        data[f"{key}/fit"] = il.smile(data[f"{key_of(HOOK)}/fit"], maturity, strikes)
        data[f"{key}/truth"] = il.smile(il.TRUTH, maturity, strikes)
        data[f"{key}/fan"] = np.array([il.smile(theta, maturity, strikes) for theta in draws[:FAN]])

    # The habit of fixing the mean reversion, on the opening example's quotes.
    for kappa in FIXED_KAPPA:
        classical(f"fixed/{kappa:g}", il.fit(design, fixed={2: kappa}))
    return data


def load(recompute=False):
    """The cached numbers, recomputed when `illustration.py`, the settings above or `compute` have changed."""
    recipe = json.dumps([HOOK, FIXED_KAPPA, FAN, SMILE_STRIKES, sorted(PRODUCTS), inspect.getsource(compute)])
    key = hashlib.sha256((HERE / "illustration.py").read_bytes() + recipe.encode()).hexdigest()[:16]
    path = CACHE / f"illustration-{key}.npz"
    if recompute or not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print("computing the running example (about four minutes on four cores) ...", flush=True)
        np.savez_compressed(path, **compute())
    with np.load(path) as stored:
        return {name: stored[name] for name in stored.files}


def quantiles(values, levels=(0.05, 0.5, 0.95)):
    return np.quantile(np.asarray(values, dtype=float), levels)


def percent(x):
    return 100.0 * np.asarray(x, dtype=float)


# ---------------------------------------------------------------------------------------------------------------
# Style


def add_rubik(variable_font, folder):
    """Static Regular, Medium and SemiBold instances of Rubik's variable font, registered with matplotlib.

    The instances are renamed to the family "Rubik": the variable font's default instance is Light, and some copies
    of it carry "Rubik Light" as their family name.
    """
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer

    for weight, face in ((400, "Regular"), (500, "Medium"), (600, "SemiBold")):
        source = TTFont(variable_font, recalcTimestamp=False)  # keep the font's own timestamp: stable PDF bytes
        font = instancer.instantiateVariableFont(source, {"wght": weight}, updateFontNames=True)
        names = {1: "Rubik", 2: "Regular", 4: f"Rubik {face}", 6: f"Rubik-{face}", 16: "Rubik", 17: face}
        for record in font["name"].names:
            if record.nameID in names:
                record.string = names[record.nameID]
        font.save(folder / f"Rubik-{face}.ttf")
        fm.fontManager.addfont(str(folder / f"Rubik-{face}.ttf"))
    return ["Rubik", "DejaVu Sans"]  # DejaVu supplies the Greek letters Rubik lacks


def style(families):
    """The notebooks' figure style in the deck's colours, with text sized for a figure shown at its drawn size."""
    plots.style()
    plt.rcParams.update({
        "font.family": families, "mathtext.fontset": "dejavusans", "pdf.fonttype": 42,
        "font.size": 7.0, "axes.titlesize": 7.5, "axes.labelsize": 7.0, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.titlepad": 3.5, "axes.labelpad": 2.0, "axes.titleweight": 500,
        "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff", "savefig.facecolor": "#ffffff",
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.6, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "grid.color": GRID, "grid.linewidth": 0.5, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2, "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
        "legend.labelcolor": INK_2, "lines.linewidth": 1.2, "text.color": INK,
    })


def canvas(width, height, **kwargs):
    """A figure `width` of the text width wide and `height` inches tall, so that the slide shows it unscaled."""
    return plt.subplots(figsize=(width * TEXT_WIDTH, height), constrained_layout=True, **kwargs)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", pad_inches=0.01, metadata={"CreationDate": None})
    plt.close(fig)


def key_line(colour, **kwargs):
    """A legend handle: a short line or a marker in a series colour, beside text in the ink colour."""
    return plt.Line2D([], [], color=colour, **kwargs)


def legend_above(fig, handles, labels):
    fig.legend(handles, labels, loc="outside upper left", ncols=len(handles), frameon=False, handlelength=1.6,
               columnspacing=1.4, handletextpad=0.5, borderaxespad=0.0)


def whisker(ax, y, low, high, centre, colour):
    """A range as a horizontal bar with a dot at its centre."""
    ax.plot([low, high], [y, y], color=colour, lw=1.7, solid_capstyle="round", zorder=3)
    ax.plot([centre], [y], "o", color=colour, ms=3.2, mec="white", mew=0.6, zorder=4)


def rows_axis(ax, positions, labels, xlabel):
    """An axis whose rows are named on the left, read from top to bottom."""
    ax.set_yticks(positions, labels)
    ax.set_ylim(max(positions) + 0.6, min(positions) - 0.6)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)


def range_handles():
    return [key_line(CLASSICAL, marker="o", ms=3.0, lw=1.7), key_line(POSTERIOR, marker="o", ms=3.0, lw=1.7),
            key_line(INK, lw=0.8, ls=DASHED)]


# ---------------------------------------------------------------------------------------------------------------
# Figures


def figure_many_fits(data, step):
    """The opening example: quotes at two maturities and a maturity nobody quotes.

    Step 1 shows the quotes and the least-squares fit. Step 2 adds a fan of posterior draws and, where there are no
    quotes, the smile of the parameters that generated them and the spread of the fan at the money.
    """
    panels = (("1 month", "quoted: 1 month"), ("3 months", "quoted: 3 months"), ("2 years", "not quoted: 2 years"))
    fig, axes = canvas(0.97, 1.60, ncols=3, sharey=True, gridspec_kw={"width_ratios": [1, 1, 1.3]})
    quotes = data["hook/quotes"]
    for ax, (label, title) in zip(axes, panels, strict=True):
        key = f"hook/{key_of(label)}"
        strikes = 100 * data[f"{key}/strikes"]
        quoted = np.isclose(quotes[:, 1], il.DAYS[il.REACH.index(label)] / 252)
        if step == 2:
            ax.plot(strikes, percent(data[f"{key}/fan"]).T, color=POSTERIOR, lw=0.45, alpha=0.22, zorder=2)
        ax.plot(strikes, percent(data[f"{key}/fit"]), color=CLASSICAL, lw=1.3, zorder=3)
        if quoted.any():
            mid, low, high = (percent(quotes[quoted, j]) for j in (2, 3, 4))
            ax.errorbar(100 * quotes[quoted, 0], mid, yerr=[mid - low, high - mid], fmt="o", color=INK, ms=2.6,
                        lw=0.8, capsize=1.5, capthick=0.8, zorder=5)
        else:
            ax.set_facecolor("#f5f5fd")
        if step == 2 and not quoted.any():
            ax.plot(strikes, percent(data[f"{key}/truth"]), color=INK, lw=1.0, ls=DASHED, zorder=4)
            spread = percent(data[f"{key}/fan"])[:, np.argmin(np.abs(strikes - 100.0))]
            ax.annotate("", xy=(100, spread.min()), xytext=(100, spread.max()), zorder=6,
                        arrowprops={"arrowstyle": "|-|,widthA=0.25,widthB=0.25", "color": INK, "lw": 0.8,
                                    "shrinkA": 0, "shrinkB": 0})
            ax.text(102, spread.max() + 0.6, f"{spread.min():.1f} to {spread.max():.1f}%", fontsize=6.3, color=INK,
                    va="bottom", ha="left")
        ax.set_title(title, loc="left")
        ax.set_xlabel("strike (% of forward)")
        ax.set_xlim(strikes[0], strikes[-1])
    axes[0].set_ylabel("implied volatility (%)")
    axes[0].set_ylim(8, 31)
    handles = [key_line(INK, marker="o", ms=2.3, lw=0.7), key_line(CLASSICAL, lw=1.3)]
    labels = ["quotes ± error band", "least-squares fit"]
    if step == 2:
        handles += [key_line(POSTERIOR, lw=1.0), key_line(INK, lw=1.0, ls=DASHED)]
        labels += [f"{FAN} posterior draws", "generating smile"]
    legend_above(fig, handles, labels)
    return fig


def figure_price_range(data):
    """The 2-year put under every posterior draw of the opening example: mark, 90% point and reserve."""
    key = key_of(HOOK)
    put = percent(data[f"{key}/put/draws"])
    mark, point90, least = put.mean(), np.quantile(put, 0.90), float(percent(data[f"{key}/put/point"]))
    fig, ax = canvas(0.88, 1.58)
    low, high = np.floor(put.min() * 2) / 2, np.ceil(np.quantile(put, 0.999) * 2) / 2
    counts, _, _ = ax.hist(put, bins=np.linspace(low, high, 49), color=POSTERIOR, alpha=0.5, lw=0, rwidth=0.88)
    top = counts.max()
    marks = ((least, CLASSICAL, "-", "right", -0.04, f"least squares\n{least:.2f}"),
             (mark, INK, "-", "left", 0.04, f"mark: the mean\n{mark:.2f}"),
             (point90, INK, DASHED, "left", 0.04, f"90% point\n{point90:.2f}"))
    for x, colour, dashes, side, shift, text in marks:
        ax.plot([x, x], [0, 1.12 * top], color=colour, lw=1.0, ls=dashes)
        ax.text(x + shift, 1.16 * top, text, ha=side, va="bottom", fontsize=6.5, color=INK_2, linespacing=1.15)
    ax.annotate("", xy=(point90, 0.80 * top), xytext=(mark, 0.80 * top),
                arrowprops={"arrowstyle": "<->", "color": INK, "lw": 0.7, "shrinkA": 0, "shrinkB": 0})
    ax.text(0.5 * (mark + point90), 0.83 * top, f"reserve {point90 - mark:.2f}", ha="center", va="bottom",
            fontsize=6.5, color=INK)
    ax.set(xlim=(low, high), ylim=(0, 1.34 * top), yticks=[])
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("price of the 2-year put, per 100 of forward: one value per posterior draw")
    return fig


def figure_posterior_cloud(data):
    """The posterior of the opening example in two planes, with least squares and its Gauss-Newton ellipse."""
    key = key_of(HOOK)
    draws, fit = data[f"{key}/draws"], data[f"{key}/fit"]
    fig, axes = canvas(0.92, 1.58, ncols=2, gridspec_kw={"width_ratios": [1.3, 1.0]})

    def volatility(variance):
        return 100 * np.sqrt(np.clip(variance, 0, None))

    planes = (  # x and y parameters, y scale, ellipse, title, labels, and the prior box as the axis limits
        (2, 1, volatility, "ellipse_kappa_theta", "left open by the quotes",
         "speed of mean reversion κ (per year)", "long-run volatility (%)", (0.1, 10.0), (10.0, 40.0)),
        (3, 4, np.asarray, "ellipse_xi_rho", "pinned by the quotes",
         "volatility of volatility", "correlation ρ", (0.1, 3.0), (-0.95, -0.10)),
    )
    for ax, (i, j, scale, ellipse, title, xlabel, ylabel, xlim, ylim) in zip(axes, planes, strict=True):
        ax.scatter(draws[:, i], scale(draws[:, j]), s=1.6, color=POSTERIOR, alpha=0.16, lw=0, zorder=2)
        outline = data[f"{key}/{ellipse}"]
        ax.plot(outline[:, 0], scale(outline[:, 1]), color=CLASSICAL, lw=0.9, zorder=3)
        ax.plot(fit[i], scale(fit[j]), "o", color=CLASSICAL, ms=3.4, mec="white", mew=0.6, zorder=4)
        ax.plot(il.TRUTH[i], scale(il.TRUTH[j]), "+", color=INK, ms=5.5, mew=1.0, zorder=5)
        ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
        ax.set_title(title, loc="left")
    handles = [key_line(POSTERIOR, marker="o", ms=2.6, lw=0), key_line(CLASSICAL, marker="o", ms=3.0, lw=0.9),
               key_line(INK, marker="+", ms=5, mew=1.0, lw=0)]
    legend_above(fig, handles, ["posterior draws", "least squares and its 90% ellipse", "generating parameters"])
    return fig


def figure_ranges_by_reach(data):
    """The 2-year put's 90% range, classical and posterior, as the quotes stop earlier and earlier."""
    rows = il.REACH[::-1]
    fig, ax = canvas(0.92, 1.40)
    for y, reach in enumerate(rows):
        key = key_of(reach)
        point = float(percent(data[f"{key}/put/point"]))
        low, high = percent(data[f"{key}/put/delta"])
        if np.isfinite(low):
            whisker(ax, y - 0.17, low, high, point, CLASSICAL)
        else:
            ax.plot([point], [y - 0.17], "o", mfc="white", mec=CLASSICAL, ms=3.2, mew=0.9, zorder=4)
            ax.text(point + 0.22, y - 0.17, "no range: the fit is on two bounds", va="center", fontsize=6.3,
                    color=INK_2)
        q = percent(quantiles(data[f"{key}/put/draws"]))
        whisker(ax, y + 0.17, q[0], q[2], q[1], POSTERIOR)
    ax.axvline(float(percent(data["truth/put"])), color=INK, lw=0.8, ls=DASHED, zorder=1)
    rows_axis(ax, range(len(rows)), [f"quotes up to {reach}" for reach in rows],
              "price of the 2-year put, per 100 of forward")
    legend_above(fig, range_handles(), RANGE_LEGEND)
    return fig


def figure_fixed_parameter(data):
    """The opening example's 2-year put when the mean reversion is fixed, fitted, or given a prior."""
    key = key_of(HOOK)
    fig, ax = canvas(0.92, 1.40)
    positions, labels = [0, 1, 2, 3.35, 4.35], [f"κ fixed at {kappa:g}" for kappa in FIXED_KAPPA]
    for y, kappa in enumerate(FIXED_KAPPA):
        low, high = percent(data[f"fixed/{kappa:g}/put/delta"])
        whisker(ax, positions[y], low, high, float(percent(data[f"fixed/{kappa:g}/put/point"])), CLASSICAL)
    low, high = percent(data[f"{key}/put/delta"])
    whisker(ax, positions[3], low, high, float(percent(data[f"{key}/put/point"])), CLASSICAL)
    q = percent(quantiles(data[f"{key}/put/draws"]))
    whisker(ax, positions[4], q[0], q[2], q[1], POSTERIOR)
    ax.axvline(float(percent(data["truth/put"])), color=INK, lw=0.8, ls=DASHED, zorder=1)
    ax.axhline(2.68, color=GRID, lw=0.8, zorder=1)
    rows_axis(ax, positions, labels + ["κ fitted: least squares", "κ uncertain: posterior"],
              "price of the 2-year put, per 100 of forward: 90% ranges")
    legend_above(fig, range_handles(), RANGE_LEGEND)
    return fig


def figure_joint_draws(data):
    """Two products priced under the same posterior draws, and a book that holds one against the other."""
    key = key_of(HOOK)
    put, start = percent(data[f"{key}/put/draws"]), percent(data[f"{key}/forward_start/draws"])
    fig, axes = canvas(0.92, 1.58, ncols=2, gridspec_kw={"width_ratios": [1.0, 1.3]})
    axes[0].scatter(put, start, s=1.6, color=POSTERIOR, alpha=0.16, lw=0)
    axes[0].set(xlabel="2-year put", ylabel="forward-start call")
    axes[0].set_title("one dot: one posterior draw, two prices", loc="left")
    rows = {"2-year put": put, "forward-start call": start, "book: long call, short put": start - put}
    for y, values in enumerate(rows.values()):
        q = quantiles(values)
        whisker(axes[1], y, q[0] - q[1], q[2] - q[1], 0.0, POSTERIOR)
        axes[1].text(q[2] - q[1] + 0.06, y, f"{q[2] - q[0]:.2f} wide", va="center", fontsize=6.3, color=INK_2)
    axes[1].axhline(1.5, color=GRID, lw=0.8, zorder=1)
    rows_axis(axes[1], range(len(rows)), list(rows), "90% range around the median, per 100 of forward")
    axes[1].set_title("the book's range is not the sum", loc="left")
    return fig


def figure_coverage():
    """Stated against measured coverage of the Heston network's intervals on 2,000 simulated smiles (study S2)."""
    stored = json.loads(VALIDATION.read_text())
    levels = sorted(stored["coverage"], key=float)
    names = {"initial_variance": "variance today", "theta": "long-run variance",
             "vol_of_vol": "volatility of volatility", "rho": "correlation"}
    markers = ("o", "s", "D", "^")
    fig, ax = canvas(0.92, 1.40)
    for y, level in enumerate(levels):
        stated, measured = 100 * float(level), percent(stored["coverage"][level]["coverage"])
        ax.plot([stated, stated], [y - 0.36, y + 0.36], color=INK, lw=1.1, zorder=2)
        for j, value in enumerate(measured):
            ax.plot([value], [y - 0.24 + 0.16 * j], markers[j], color=POSTERIOR, ms=3.6, mec="white", mew=0.4,
                    zorder=3)
        ax.text(stated + 1.6, y, f"measured {measured.min():.1f} to {measured.max():.1f}%", va="center",
                fontsize=6.3, color=INK_2)
    rows_axis(ax, range(len(levels)), [f"{100 * float(level):g}% intervals" for level in levels],
              f"share of {stored['surfaces']:,} simulated smiles whose interval contains the true parameter (%)")
    ax.set_xlim(40, 112)
    ax.set_xticks(range(40, 101, 10))
    handles = [key_line(INK, lw=1.1)] + [key_line(POSTERIOR, marker=marker, ms=3.6, lw=0) for marker in markers]
    legend_above(fig, handles, ["stated"] + [names[target] for target in stored["targets"]])
    return fig, stored


# ---------------------------------------------------------------------------------------------------------------
# The numbers the slides quote


def report(data, stored):
    """Every number the slides quote, printed so that slot1.tex can be checked against it."""
    key = key_of(HOOK)
    put, start = percent(data[f"{key}/put/draws"]), percent(data[f"{key}/forward_start/draws"])

    def span(values):
        low, _, high = quantiles(values)
        return high - low

    print(f"\nrunning example, seed {il.SEED}: (v0, theta, kappa, vol of vol, rho) = {il.TRUTH.tolist()}")
    print(f"under the generating parameters: 2-year put {percent(data['truth/put']):.2f}, forward-start call "
          f"{percent(data['truth/forward_start']):.2f}, per 100 of forward")
    for reach in il.REACH:
        k = key_of(reach)
        low, high = percent(data[f"{k}/put/delta"])
        q = percent(quantiles(data[f"{k}/put/draws"]))
        bounds = ", ".join(data[f"{k}/on_bound"])
        classical = f"{low:.2f} to {high:.2f}" if np.isfinite(low) else f"none (the fit is on a bound: {bounds})"
        print(f"  quotes up to {reach:8s} ({int(data[f'{k}/quotes']):2d} quotes): least squares "
              f"{percent(data[f'{k}/put/point']):.2f}, delta method {classical}; "
              f"posterior {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}; effective sample size {float(data[f'{k}/ess']):.0f} "
              f"of {int(data[f'{k}/proposals'])}, failed pricings {float(data[f'{k}/failed']):.2%}")
    for label in SMILE_STRIKES:
        k = f"hook/{key_of(label)}"
        at = np.argmin(np.abs(data[f"{k}/strikes"] - 1.0))
        fan = percent(data[f"{k}/fan"])[:, at]
        print(f"  opening example, {label}: at-the-money volatility of the {FAN} draws {fan.min():.1f} to "
              f"{fan.max():.1f}%, fit {percent(data[f'{k}/fit'][at]):.1f}%, "
              f"generating {percent(data[f'{k}/truth'][at]):.1f}%")
    low, high = np.quantile(data["hook/misfit"], [0.05, 0.95])
    print(f"  opening example: root-mean-square misfit in error bands, fit {float(data['hook/misfit_fit']):.2f}, "
          f"posterior draws {low:.2f} to {high:.2f} (5% to 95%)")
    mark, point90 = put.mean(), np.quantile(put, 0.9)
    print(f"  opening example, 2-year put: least squares {percent(data[f'{key}/put/point']):.2f}, mark {mark:.2f}, "
          f"90% point {point90:.2f}, reserve {point90 - mark:.2f} ({100 * (point90 / mark - 1):.0f}% of the mark)")
    for kappa in FIXED_KAPPA:
        k = f"fixed/{kappa:g}"
        low, high = percent(data[f"{k}/put/delta"])
        print(f"  kappa fixed at {kappa:g}: put {percent(data[f'{k}/put/point']):.2f}, range {low:.2f} to {high:.2f}, "
              f"chi-square {float(data[f'{k}/chi2']):.1f} (fitted: {float(data[f'{key}/chi2']):.1f})")
    print(f"  opening example, 90% widths: put {span(put):.2f}, forward-start call {span(start):.2f}, call minus put "
          f"{span(start - put):.2f}; correlation of the two prices {np.corrcoef(put, start)[0, 1]:.3f}")
    for level in sorted(stored["coverage"], key=float):
        measured = percent(stored["coverage"][level]["coverage"])
        print(f"  Heston network, {100 * float(level):g}% intervals: measured {measured.min():.1f} to "
              f"{measured.max():.1f}%")


def stale_numbers(data, stored):
    """The example's numbers that `slot1.tex` does not contain as they are printed above: sentences to update."""
    key = key_of(HOOK)
    put, start = percent(data[f"{key}/put/draws"]), percent(data[f"{key}/forward_start/draws"])
    at = np.argmin(np.abs(data["hook/2_years/strikes"] - 1.0))
    fan = percent(data["hook/2_years/fan"])[:, at]
    mark, point90 = put.mean(), np.quantile(put, 0.9)
    quoted = {"the fan's lowest at-the-money volatility": f"{fan.min():.1f}",
              "the fan's highest at-the-money volatility": f"{fan.max():.1f}",
              "the mark": f"{mark:.2f}", "the 90% point": f"{point90:.2f}", "the reserve": f"{point90 - mark:.2f}",
              "the least-squares price": f"{percent(data[f'{key}/put/point']):.2f}",
              "the generating price": f"{percent(data['truth/put']):.2f}"}
    for name, values in (("the put", put), ("the forward-start call", start), ("the book", start - put)):
        low, _, high = quantiles(values)
        quoted[f"the 90% width of {name}"] = f"{high - low:.2f}"
    for reach in il.REACH:
        k = key_of(reach)
        low, high = percent(data[f"{k}/put/delta"])
        q = percent(quantiles(data[f"{k}/put/draws"]))
        quoted[f"the posterior range, quotes up to {reach}"] = f"{q[0]:.2f} to {q[2]:.2f}"
        if np.isfinite(low):
            quoted[f"the delta-method range, quotes up to {reach}"] = f"{low:.2f} to {high:.2f}"
    for kappa in FIXED_KAPPA:
        low, high = percent(data[f"fixed/{kappa:g}/put/delta"])
        quoted[f"the range with kappa fixed at {kappa:g}"] = f"{low:.2f} to {high:.2f}"
    measured = percent(stored["coverage"]["0.9"]["coverage"])
    quoted["the measured coverage of 90% intervals"] = f"{measured.min():.1f}–{measured.max():.1f}"
    deck = (HERE / "slot1.tex").read_text()
    return {name: number for name, number in quoted.items() if number not in deck}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rubik", type=Path,
                        help="Rubik's variable font, e.g. the Overleaf project's fonts/Rubik-Variable.ttf")
    parser.add_argument("--recompute", action="store_true", help="ignore the cached numbers of the running example")
    args = parser.parse_args()
    data = load(args.recompute)
    with tempfile.TemporaryDirectory() as folder:
        style(add_rubik(args.rubik, Path(folder)) if args.rubik else ["DejaVu Sans"])
        save(figure_many_fits(data, 1), "many-fits-1")
        save(figure_many_fits(data, 2), "many-fits-2")
        save(figure_price_range(data), "price-range")
        save(figure_posterior_cloud(data), "posterior-cloud")
        save(figure_ranges_by_reach(data), "ranges-by-reach")
        save(figure_fixed_parameter(data), "fixed-parameter")
        save(figure_joint_draws(data), "joint-draws")
        fig, stored = figure_coverage()
        save(fig, "coverage")
    print(f"8 figures in {OUT}")
    report(data, stored)
    stale = stale_numbers(data, stored)
    print("\nslot1.tex quotes these numbers as printed above." if not stale else
          "\nslot1.tex does not quote these numbers as printed above; update the slides and the curriculum page:")
    for name, number in stale.items():
        print(f"  {name}: {number}")


if __name__ == "__main__":
    main()
