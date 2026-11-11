"""Draw slot 1's figures: vector PDFs in the BayesOps theme's colours, at the size the slides show them.

Run from tutorial/:

    python slides/slot1/export_figures.py [--rubik PATH/TO/Rubik-Variable.ttf] [--recompute]

The deck shows two kinds of evidence, and so does this script:

- six illustrations, from the deck's running example (`illustration.py`: a thin option chain simulated from Heston,
  the model right by construction; the opening figure has two steps);
- two results, from the Heston study on 63 SPX quotes as recorded on 11 November 2026 (`recorded-results.json`,
  which names the source of every number): what one posterior costs, and the coverage of the trained network.

The running example takes a few minutes to compute on four cores, mostly five posteriors. Its numbers are cached
under tutorial/.cache/ (ignored by Git) and reused until `illustration.py`, the settings below or `compute` change, or
--recompute is given. The script ends by printing every number the slides quote, and says which of them `slot1.tex`
does not contain.

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

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per worker process: the posteriors run side by side
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import illustration as il  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures" / "slot1"
CACHE = HERE.parents[1] / ".cache" / "slides-slot1"
RECORDED = json.loads((HERE / "recorded-results.json").read_text())

# The theme's colours (aviation #15153B, neonlake #6969FF), the figures' second colour, and the colours of a check.
INK, INK_2, MUTED, GRID, BASELINE = "#15153b", "#3f3f63", "#7c7c99", "#e7e7f1", "#c4c4d6"
POSTERIOR, CLASSICAL = "#6969ff", "#c2187a"  # periwinkle: the posterior; magenta: least squares and its ranges
HOLDS, CAVEAT = "#1a9c86", "#d9820a"  # a check that holds; one that holds with a caveat
UNQUOTED = "#f5f5fd"  # behind a panel where nothing is quoted

TEXT_WIDTH = 5.51  # inches: the 140 mm text width of a 16:9 Beamer frame
HOOK = "3 months"  # the reading of the opening example: 14 quotes, at 1 and 3 months
FIXED_KAPPA = (2.0, 4.0, 8.0)  # the habit of fixing the mean reversion, at three values
FAN = 120  # posterior curves drawn in a fan
SMILE_STRIKES = {"1 month": (0.835, 1.065), "3 months": (0.735, 1.105), "2 years": (0.55, 1.30)}
DASHED = (0, (4, 2))
RANGE_LEGEND = ("least squares, delta method", "posterior, 5% to 95%", "price under the generating parameters")
PARAMETERS = ("today's variance $v_0$", "mean reversion $\\kappa$", "long-run variance $v_\\infty$",
              "vol of vol $\\sigma_v$", "correlation $\\rho$", "error scale $c$")
NOTEBOOK_CASE = "synthetic-market-like"


# ---------------------------------------------------------------------------------------------------------------
# Numbers


def key_of(reach):
    return reach.replace(" ", "_")


def _posterior(reach):
    return reach, il.posterior(il.readings()[reach])


def compute():
    """Every number the illustrations need, as a dictionary of arrays and scalars."""
    readings = il.readings()
    with ProcessPoolExecutor(max_workers=min(len(il.REACH), os.cpu_count() or 1)) as pool:
        posteriors = dict(pool.map(_posterior, il.REACH))
    data = {f"truth/{name}": float(product(il.TRUTH)) for name, product in il.PRODUCTS.items()}

    def classical(prefix, result):
        data[f"{prefix}/fit"] = result.theta
        data[f"{prefix}/sum_of_squares"] = result.sum_of_squares
        data[f"{prefix}/scale"] = result.scale
        data[f"{prefix}/dof"] = result.dof
        data[f"{prefix}/on_bound"] = np.array(result.on_bound, dtype=str)
        for name, product in il.PRODUCTS.items():
            point, interval = il.delta_range(result, product)
            data[f"{prefix}/{name}/point"] = point
            data[f"{prefix}/{name}/delta"] = np.array(interval if interval is not None else (np.nan, np.nan))

    for reach in il.REACH:
        quotes, post, key = readings[reach], posteriors[reach], key_of(reach)
        result = il.fit(quotes)
        classical(key, result)
        for name, product in il.PRODUCTS.items():
            data[f"{key}/{name}/draws"] = product(post.draws[:, :5])
        data[f"{key}/draws"] = post.draws
        data[f"{key}/ess"] = post.effective_sample_size
        data[f"{key}/proposals"] = post.proposals
        data[f"{key}/failed"] = post.failed
        data[f"{key}/acceptance"] = post.acceptance
        data[f"{key}/quotes"] = quotes.n
        data[f"{key}/gate"] = max(quotes.gate(theta, il.CHECK_RUNG) for theta in post.draws[:200, :5])
        if result.covariance is not None:
            data[f"{key}/covariance"] = result.covariance

    # The opening example: smiles of the fit, the truth and a fan of posterior draws, quoted and not quoted.
    quotes, draws = readings[HOOK], posteriors[HOOK].draws[:, :5]
    mid, low, high = quotes.implied()
    data["hook/quotes"] = np.column_stack([quotes.strike, quotes.maturity, mid, low, high])
    data["hook/misfit"] = np.sqrt(quotes.sum_of_squares(draws) / quotes.n)
    data["hook/misfit_fit"] = np.sqrt(data[f"{key_of(HOOK)}/sum_of_squares"] / quotes.n)
    for label, (lowest, highest) in SMILE_STRIKES.items():
        maturity = il.MONTHS[il.REACH.index(label)] / 12.0
        strikes = np.linspace(lowest, highest, 51)
        key = f"hook/{key_of(label)}"
        data[f"{key}/strikes"] = strikes
        data[f"{key}/fit"] = il.smile(data[f"{key_of(HOOK)}/fit"], maturity, strikes)
        data[f"{key}/truth"] = il.smile(il.TRUTH, maturity, strikes)
        data[f"{key}/fan"] = np.array([il.smile(theta, maturity, strikes) for theta in draws[:FAN]])

    # The habit of fixing the mean reversion, on the opening example's quotes.
    for kappa in FIXED_KAPPA:
        classical(f"fixed/{kappa:g}", il.fit(quotes, fixed={1: kappa}))

    # The Heston notebook's own simulated case, on its 63 quotes: the classical intervals of the five parameters.
    notebook = il.notebook_case()
    result = il.fit(notebook)
    data["notebook/fit"] = result.theta
    data["notebook/intervals"] = il.parameter_intervals(result)
    data["notebook/scale"] = result.scale
    data["notebook/sum_of_squares"] = result.sum_of_squares
    data["notebook/quotes"] = notebook.n
    return data


def load(recompute=False):
    """The cached numbers, recomputed when `illustration.py`, the settings above or `compute` have changed."""
    recipe = json.dumps([HOOK, FIXED_KAPPA, FAN, SMILE_STRIKES, sorted(il.PRODUCTS), inspect.getsource(compute)])
    sources = b"".join((HERE / name).read_bytes() for name in ("illustration.py", "spx-design.json",
                                                               "market-like-case.json"))
    key = hashlib.sha256(sources + recipe.encode()).hexdigest()[:16]
    path = CACHE / f"illustration-{key}.npz"
    if recompute or not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print("computing the running example (a few minutes on four cores) ...", flush=True)
        np.savez_compressed(path, **compute())
    with np.load(path) as stored:
        return {name: stored[name] for name in stored.files}


def classical_fit(data, key):
    """The least-squares fit of a reading as `illustration.Fit`, from the cached numbers (all five parameters free)."""
    return il.Fit(np.asarray(data[f"{key}/fit"]), float(data[f"{key}/sum_of_squares"]), float(data[f"{key}/scale"]),
                  int(data[f"{key}/dof"]), tuple(range(len(il.NAMES))), np.asarray(data[f"{key}/covariance"]),
                  tuple(data[f"{key}/on_bound"]))


def quantiles(values, levels=(0.05, 0.5, 0.95)):
    return np.quantile(np.asarray(values, dtype=float), levels)


def percent(x):
    return 100.0 * np.asarray(x, dtype=float)


def notebook_agreement(data):
    """Least squares against the notebook's exact posterior on its 63 simulated quotes, for the five Heston parameters.

    Returns the classical 90% intervals, the reference's 5% and 95% points, the ratio of the two widths, and the
    distance of the fit from the reference's mean in reference standard deviations.
    """
    reference = RECORDED["reference_posteriors"][NOTEBOOK_CASE]
    classical = np.asarray(data["notebook/intervals"])
    exact = np.array([[q[0], q[2]] for q in reference["quantiles_05_50_95"][:5]])
    ratio = (classical[:, 1] - classical[:, 0]) / (exact[:, 1] - exact[:, 0])
    shift = (np.asarray(data["notebook/fit"]) - np.array(reference["mean"][:5])) / np.array(reference["sd"][:5])
    return classical, exact, ratio, shift


def costs():
    """What one posterior costs on the 63 SPX quotes: measured for Heston, derived for rough Bergomi."""
    references = RECORDED["references"].values()
    cpu = [run["cpu_seconds"] for run in references]
    evaluations = int(round(np.mean([run["likelihood_evaluations"] for run in references]), -3))
    low, high = RECORDED["rough_bergomi"]["pricing_cpu_seconds"]
    pricings = RECORDED["rough_bergomi"]["importance_sampling_pricings"]
    network = RECORDED["network"]
    tests = RECORDED["population"]["surfaces"]
    return {
        "network_seconds": network["posterior_seconds_total"] / network["posteriors_drawn"],
        "heston_minutes": (min(cpu) / 60, max(cpu) / 60), "evaluations": evaluations,
        "rough_sampling_hours": (pricings * low / 3600, pricings * high / 3600),
        "rough_mcmc_days": (evaluations * low / 86400, evaluations * high / 86400),
        "test_heston_days": (tests * min(cpu) / 86400, tests * max(cpu) / 86400),
        "test_rough_years": (tests * evaluations * low / (365 * 86400), tests * evaluations * high / (365 * 86400)),
        "bank_in_posteriors": network["training_rows"] / evaluations,
    }


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
    """The deck's figure style, with text sized for a figure shown at its drawn size."""
    plt.rcParams.update({
        "font.family": families, "mathtext.fontset": "dejavusans", "pdf.fonttype": 42,
        "font.size": 7.5, "axes.titlesize": 8.0, "axes.labelsize": 7.5, "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0, "axes.titlepad": 4.0, "axes.labelpad": 2.5, "axes.titleweight": 500,
        "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff", "savefig.facecolor": "#ffffff",
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.6, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-", "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2, "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
        "legend.frameon": False, "legend.labelcolor": INK_2, "lines.linewidth": 1.3, "lines.solid_capstyle": "round",
        "text.color": INK,
    })


HEIGHT = 1.60  # inches: every figure is this tall; \figmax in slot1.tex is just above it, so none is scaled


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


def whisker(ax, y, low, high, centre, colour, width=2.0):
    """A range as a horizontal bar with a dot at its centre, ringed in the surface colour."""
    ax.plot([low, high], [y, y], color=colour, lw=width, solid_capstyle="round", zorder=3)
    ax.plot([centre], [y], "o", color=colour, ms=4.6, mec="white", mew=0.8, zorder=4)


def rows_axis(ax, positions, labels, xlabel):
    """An axis whose rows are named on the left, read from top to bottom."""
    ax.set_yticks(positions, labels)
    ax.set_ylim(max(positions) + 0.6, min(positions) - 0.6)
    ax.tick_params(axis="y", length=0, labelsize=7.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)


def named_row(ax, y, name, detail):
    """A row name in two lines, left of the axis: what it is, and how it was obtained."""
    transform = ax.get_yaxis_transform()
    ax.text(-0.014, y - 0.17, name, transform=transform, ha="right", va="center", fontsize=8.0, color=INK,
            fontweight=500)
    ax.text(-0.014, y + 0.21, detail, transform=transform, ha="right", va="center", fontsize=6.8, color=MUTED)


def range_handles():
    return [key_line(CLASSICAL, marker="o", ms=3.6, lw=2.0), key_line(POSTERIOR, marker="o", ms=3.6, lw=2.0),
            key_line(INK, lw=0.9, ls=DASHED)]


# ---------------------------------------------------------------------------------------------------------------
# The illustrations


def figure_many_fits(data, step):
    """The opening example: quotes at two expiries and a maturity nobody quotes.

    Step 1 shows the quotes and the least-squares fit. Step 2 adds a fan of posterior draws and, where there are no
    quotes, the smile of the parameters that generated them and the spread of the fan at the money.
    """
    panels = (("1 month", "quoted: 1 month"), ("3 months", "quoted: 3 months"), ("2 years", "not quoted: 2 years"))
    fig, axes = canvas(0.97, HEIGHT, ncols=3, sharey=True, gridspec_kw={"width_ratios": [1, 1, 1.3]})
    quotes = data["hook/quotes"]
    for ax, (label, title) in zip(axes, panels, strict=True):
        key = f"hook/{key_of(label)}"
        strikes = 100 * data[f"{key}/strikes"]
        quoted = np.isclose(quotes[:, 1], il.MONTHS[il.REACH.index(label)] / 12.0)
        if step == 2:
            ax.plot(strikes, percent(data[f"{key}/fan"]).T, color=POSTERIOR, lw=0.5, alpha=0.22, zorder=2)
        ax.plot(strikes, percent(data[f"{key}/fit"]), color=CLASSICAL, lw=1.5, zorder=3)
        if quoted.any():
            mid, low, high = (percent(quotes[quoted, j]) for j in (2, 3, 4))
            ax.errorbar(100 * quotes[quoted, 0], mid, yerr=[mid - low, high - mid], fmt="o", color=INK, ms=3.0,
                        lw=0.9, capsize=1.8, capthick=0.9, zorder=5)
        else:
            ax.set_facecolor(UNQUOTED)
        if step == 2 and not quoted.any():
            ax.plot(strikes, percent(data[f"{key}/truth"]), color=INK, lw=1.1, ls=DASHED, zorder=4)
            spread = percent(data[f"{key}/fan"])[:, np.argmin(np.abs(strikes - 100.0))]
            ax.annotate("", xy=(100, spread.min()), xytext=(100, spread.max()), zorder=6,
                        arrowprops={"arrowstyle": "|-|,widthA=0.28,widthB=0.28", "color": INK, "lw": 0.9,
                                    "shrinkA": 0, "shrinkB": 0})
            ax.text(102.5, spread.max() + 0.5, f"{spread.min():.1f} to {spread.max():.1f}%", fontsize=7.2, color=INK,
                    va="bottom", ha="left")
        ax.set_title(title, loc="left")
        ax.set_xlabel("strike (% of forward)")
        ax.set_xlim(strikes[0], strikes[-1])
    axes[0].set_ylabel("implied volatility (%)")
    axes[0].set_ylim(9, 38)
    handles = [key_line(INK, marker="o", ms=2.6, lw=0.8), key_line(CLASSICAL, lw=1.5)]
    labels = ["quotes ± half-spread", "least-squares fit"]
    if step == 2:
        handles += [key_line(POSTERIOR, lw=1.1), key_line(INK, lw=1.1, ls=DASHED)]
        labels += [f"{FAN} other fits", "generating smile"]
    legend_above(fig, handles, labels)
    return fig


def figure_price_range(data):
    """The 2-year put under every posterior draw of the opening example: mark, 90% point and reserve."""
    key = key_of(HOOK)
    put = percent(data[f"{key}/put/draws"])
    mark, point90, least = put.mean(), np.quantile(put, 0.90), float(percent(data[f"{key}/put/point"]))
    truth = float(percent(data["truth/put"]))
    fig, ax = canvas(0.90, HEIGHT)
    low, high = np.floor(np.quantile(put, 0.001) * 2) / 2, np.ceil(np.quantile(put, 0.998) * 2) / 2
    counts, _, _ = ax.hist(put, bins=np.linspace(low, high, 49), color=POSTERIOR, alpha=0.5, lw=0, rwidth=0.88)
    top = counts.max()
    marks = ((least, CLASSICAL, "-", "right", -0.045, f"least squares\n{least:.2f}"),
             (mark, INK, "-", "left", 0.045, f"mark: the mean\n{mark:.2f}"),
             (point90, INK, DASHED, "left", 0.045, f"90% point\n{point90:.2f}"))
    for x, colour, dashes, side, shift, text in marks:
        ax.plot([x, x], [0, 1.12 * top], color=colour, lw=1.1, ls=dashes)
        ax.text(x + shift, 1.16 * top, text, ha=side, va="bottom", fontsize=7.2, color=INK_2, linespacing=1.15)
    ax.annotate("", xy=(point90, 0.80 * top), xytext=(mark, 0.80 * top),
                arrowprops={"arrowstyle": "<->", "color": INK, "lw": 0.8, "shrinkA": 0, "shrinkB": 0})
    ax.text(0.5 * (mark + point90), 0.83 * top, f"reserve {reserve(mark, point90):.2f}", ha="center", va="bottom",
            fontsize=7.2, color=INK)
    ax.plot([truth], [0], marker="^", color=INK, ms=4.5, clip_on=False, zorder=6)
    ax.legend([key_line(INK, marker="^", ms=4.5, lw=0)], [f"under the generating parameters: {truth:.2f}"],
              loc="upper right", bbox_to_anchor=(1.0, 0.62), handletextpad=0.2, borderaxespad=0.0)
    ax.set(xlim=(low, high), ylim=(0, 1.36 * top), yticks=[])
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("price of the 2-year put, per 100 of forward: one value per parameter set")
    return fig


def figure_posterior_cloud(data):
    """The posterior of the opening example in two planes, with least squares and its Gauss-Newton region."""
    key = key_of(HOOK)
    draws, result = data[f"{key}/draws"], classical_fit(data, key)
    fig, axes = canvas(0.94, HEIGHT, ncols=2, gridspec_kw={"width_ratios": [1.3, 1.0]})

    def volatility(variance):
        return 100 * np.sqrt(np.clip(variance, 0, None))

    planes = (  # x and y parameters with their scales, title, labels, and the prior box as the axis limits
        (1, 2, np.asarray, volatility, "left open by the quotes",
         "speed of mean reversion κ (per year)", "long-run volatility (%)", (0.1, 10.0), (10.0, 40.0)),
        (0, 4, volatility, np.asarray, "pinned by the quotes",
         "today's volatility (%)", "correlation ρ", (10.0, 40.0), (-0.95, -0.10)),
    )
    for ax, (i, j, x_scale, y_scale, title, xlabel, ylabel, xlim, ylim) in zip(axes, planes, strict=True):
        ax.scatter(x_scale(draws[:, i]), y_scale(draws[:, j]), s=2.0, color=POSTERIOR, alpha=0.16, lw=0, zorder=2)
        outline = il.gauss_newton_ellipse(result, i, j)
        ax.plot(x_scale(outline[:, 0]), y_scale(outline[:, 1]), color=CLASSICAL, lw=1.0, zorder=3)
        ax.plot(x_scale(result.theta[i]), y_scale(result.theta[j]), "o", color=CLASSICAL, ms=4.2, mec="white",
                mew=0.7, zorder=4)
        ax.plot(x_scale(il.TRUTH[i]), y_scale(il.TRUTH[j]), "+", color=INK, ms=6.5, mew=1.1, zorder=5)
        ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
        ax.set_title(title, loc="left")
    handles = [key_line(POSTERIOR, marker="o", ms=3.0, lw=0), key_line(CLASSICAL, marker="o", ms=3.6, lw=1.0),
               key_line(INK, marker="+", ms=6, mew=1.1, lw=0)]
    legend_above(fig, handles, ["posterior draws", "least squares and its 90% region", "generating parameters"])
    return fig


def figure_ranges_by_reach(data):
    """The 2-year put's 90% range, classical and posterior, as the quotes stop earlier and earlier."""
    rows = il.REACH[::-1]
    fig, ax = canvas(0.94, HEIGHT)
    for y, reach in enumerate(rows):
        key = key_of(reach)
        point = float(percent(data[f"{key}/put/point"]))
        low, high = percent(data[f"{key}/put/delta"])
        if np.isfinite(low):
            whisker(ax, y - 0.2, low, high, point, CLASSICAL)
        else:
            ax.plot([point], [y - 0.24], "o", mfc="white", mec=CLASSICAL, ms=4.2, mew=1.0, zorder=4)
            ax.text(point + 0.25, y - 0.25, "no range: the fit is on a bound", va="center", fontsize=7.0,
                    color=INK_2)
        q = percent(quantiles(data[f"{key}/put/draws"]))
        whisker(ax, y + 0.2, q[0], q[2], q[1], POSTERIOR)
    ax.axvline(float(percent(data["truth/put"])), color=INK, lw=0.9, ls=DASHED, zorder=1)
    labels = [f"quotes up to {reach} ({int(data[f'{key_of(reach)}/quotes'])})" for reach in rows]
    rows_axis(ax, range(len(rows)), labels, "price of the 2-year put, per 100 of forward")
    legend_above(fig, range_handles(), RANGE_LEGEND)
    return fig


def figure_fixed_parameter(data):
    """The opening example's 2-year put when the mean reversion is fixed, fitted, or given a prior."""
    key = key_of(HOOK)
    fig, ax = canvas(0.94, HEIGHT)
    positions, labels = [0, 1, 2, 3.4, 4.4], [f"κ fixed at {kappa:g}" for kappa in FIXED_KAPPA]
    for y, kappa in enumerate(FIXED_KAPPA):
        low, high = percent(data[f"fixed/{kappa:g}/put/delta"])
        whisker(ax, positions[y], low, high, float(percent(data[f"fixed/{kappa:g}/put/point"])), CLASSICAL)
    low, high = percent(data[f"{key}/put/delta"])
    whisker(ax, positions[3], low, high, float(percent(data[f"{key}/put/point"])), CLASSICAL)
    q = percent(quantiles(data[f"{key}/put/draws"]))
    whisker(ax, positions[4], q[0], q[2], q[1], POSTERIOR)
    ax.axvline(float(percent(data["truth/put"])), color=INK, lw=0.9, ls=DASHED, zorder=1)
    ax.axhline(2.7, color=BASELINE, lw=0.6, zorder=1)
    rows_axis(ax, positions, labels + ["κ fitted: least squares", "κ uncertain: posterior"],
              "price of the 2-year put, per 100 of forward: 90% ranges")
    legend_above(fig, range_handles(), RANGE_LEGEND)
    return fig


def figure_joint_draws(data):
    """Two products priced under the same posterior draws, and a book that holds one against the other."""
    key = key_of(HOOK)
    put, start = percent(data[f"{key}/put/draws"]), percent(data[f"{key}/forward_start/draws"])
    fig, axes = canvas(0.94, HEIGHT, ncols=2, gridspec_kw={"width_ratios": [1.0, 1.3]})
    axes[0].scatter(put, start, s=2.0, color=POSTERIOR, alpha=0.16, lw=0)
    axes[0].set(xlabel="2-year put", ylabel="forward-start call")
    axes[0].set_title("one dot: one posterior draw, two prices", loc="left")
    rows = {"2-year put": put, "forward-start call": start, "book: long call, short put": start - put}
    widest = 0.0
    for y, values in enumerate(rows.values()):
        q = quantiles(values)
        whisker(axes[1], y, q[0] - q[1], q[2] - q[1], 0.0, POSTERIOR)
        axes[1].text(q[2] - q[1] + 0.07, y, f"{q[2] - q[0]:.2f} wide", va="center", fontsize=7.2, color=INK_2)
        widest = max(widest, q[2] - q[1])
    axes[1].axhline(1.5, color=BASELINE, lw=0.6, zorder=1)
    rows_axis(axes[1], range(len(rows)), list(rows), "90% range around the median, per 100")
    axes[1].set_xlim(right=widest + 0.75)
    axes[1].set_title("the book's range is not the sum", loc="left")
    return fig


# ---------------------------------------------------------------------------------------------------------------
# The results


def figure_posterior_cost():
    """CPU time for one posterior on the 63 SPX quotes: exact methods, measured or derived, and the trained network."""
    cost = costs()
    rows = (  # name, method, range in seconds, value, status, colour, measured
        ("Heston", "exact posterior, by MCMC", [60 * x for x in cost["heston_minutes"]],
         "{:.0f}–{:.0f} CPU-minutes".format(*cost["heston_minutes"]), "measured", INK, True),
        ("Rough Bergomi", "exact, by importance sampling", [3600 * x for x in cost["rough_sampling_hours"]],
         "{:.0f}–{:.0f} CPU-hours".format(*cost["rough_sampling_hours"]), "estimated", INK, False),
        ("Rough Bergomi", "exact posterior, by MCMC", [86400 * x for x in cost["rough_mcmc_days"]],
         "100–180 CPU-days", "estimated, not run", INK, False),
        ("Heston", "trained network", [cost["network_seconds"]] * 2,
         f"{cost['network_seconds']:.1f} seconds", "measured, after training", POSTERIOR, True),
    )
    fig, ax = canvas(0.97, HEIGHT)
    for y, (name, method, (low, high), value, status, colour, measured) in zip((0, 1, 2, 3.25), rows, strict=True):
        if high > low:
            ax.plot([low, high], [y, y], color=colour, lw=6.0, solid_capstyle="round", alpha=1.0 if measured else 0.38,
                    zorder=3)
        else:
            ax.plot([low], [y], "o", color=colour, ms=7.5, mec="white", mew=1.0, zorder=3)
        ax.text(high * 2.2, y - 0.17, value, va="center", ha="left", fontsize=8.5, color=INK, fontweight=600)
        ax.text(high * 2.2, y + 0.21, status, va="center", ha="left", fontsize=6.8, color=MUTED)
        named_row(ax, y, name, method)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 2.2e9)
    ax.set_xticks([1, 60, 3600, 86400, 86400 * 30, 86400 * 365], ["1 s", "1 min", "1 hour", "1 day", "1 month", "1 year"])
    ax.minorticks_off()
    ax.set_yticks([])
    ax.set_ylim(3.85, -0.6)
    ax.grid(axis="y", visible=False)
    ax.axhline(2.62, color=BASELINE, lw=0.6)
    ax.set_xlabel("CPU time for one posterior on the 63 SPX quotes (logarithmic scale)")
    return fig


def figure_coverage():
    """How often the network's 90% intervals contain the truth on 512 simulated surfaces, parameter by parameter."""
    population = RECORDED["population"]
    coverage = percent(population["coverage"]["0.9"])
    wilson = percent(population["coverage_wilson95"]["0.9"])
    on_target = [low <= 90.0 <= high for low, high in wilson]
    order = [j for j in range(len(PARAMETERS)) if on_target[j]] + [j for j in range(len(PARAMETERS)) if not on_target[j]]
    positions = [y + (0.35 if not on_target[j] else 0.0) for y, j in enumerate(order)]
    fig, ax = canvas(0.97, HEIGHT)
    ax.axvline(90, color=INK, lw=1.0, zorder=2)
    for y, j in zip(positions, order, strict=True):
        colour = HOLDS if on_target[j] else CAVEAT
        ax.plot(wilson[j], [y, y], color=colour, lw=2.0, solid_capstyle="round", zorder=3)
        ax.plot([coverage[j]], [y], "o", color=colour, ms=6.0, mec="white", mew=1.0, zorder=4)
        ax.text(wilson[j][1] + 0.35, y, f"{coverage[j]:.1f}%", va="center", ha="left", fontsize=8.0, color=INK,
                fontweight=500)
        ax.text(-0.014, y, PARAMETERS[j], transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.8,
                color=INK_2)
        if not on_target[j]:
            ax.text(wilson[j][0] - 0.35, y, "intervals too wide", ha="right", va="center", fontsize=7.5, color=INK_2)
    split = sum(on_target) - 0.5 + 0.175
    ax.axhline(split, color=BASELINE, lw=0.6)
    ax.set_yticks([])
    ax.set_ylim(max(positions) + 0.7, -0.95)
    ax.set_xlim(85.5, 101.0)
    ticks = [86, 88, 90, 92, 94, 96, 98, 100]
    ax.set_xticks(ticks, [f"{tick}%" for tick in ticks])
    ax.grid(axis="y", visible=False)
    ax.text(90.25, -0.55, "stated: 90%", ha="left", va="center", fontsize=7.5, color=INK, fontweight=500)
    ax.text(100.8, 0.5 * (sum(on_target) - 1), f"{sum(on_target)} of {len(PARAMETERS)} parameters:\nthe 90% is met",
            ha="right", va="center", fontsize=7.5, color=INK_2, linespacing=1.25)
    ax.text(100.8, -0.55, "dot: measured · bar: its 95% confidence interval", ha="right", va="center", fontsize=6.5,
            color=MUTED)
    ax.set_xlabel(f"share of {population['surfaces']} simulated surfaces whose 90% interval contains the truth")
    return fig


# ---------------------------------------------------------------------------------------------------------------
# The numbers the slides quote


def reserve(mark, point90):
    """The 90% point less the mark, both as the slides print them, so that the three printed numbers add up."""
    return round(float(point90), 2) - round(float(mark), 2)


def quoted_numbers(data):
    """Every number the slides quote, as {description: text}: the report prints them and the deck is checked for them."""
    key = key_of(HOOK)
    put, start = percent(data[f"{key}/put/draws"]), percent(data[f"{key}/forward_start/draws"])
    at = np.argmin(np.abs(data["hook/2_years/strikes"] - 1.0))
    fan = percent(data["hook/2_years/fan"])[:, at]
    mark, point90 = put.mean(), np.quantile(put, 0.9)
    low, high = np.quantile(data["hook/misfit"], [0.05, 0.95])
    quoted = {
        "the best fit's misfit in half-spreads": f"{float(data['hook/misfit_fit']):.2f}",
        "the posterior draws' misfit in half-spreads": f"{low:.1f}–{high:.1f}",
        "the fan's at-the-money volatility at 2 years": f"{fan.min():.1f}–{fan.max():.1f}",
        "the least-squares price": f"{percent(data[f'{key}/put/point']):.2f}",
        "the mark": f"{mark:.2f}", "the 90% point": f"{point90:.2f}", "the reserve": f"{reserve(mark, point90):.2f}",
        "the reserve as a share of the mark": f"{100 * reserve(mark, point90) / mark:.0f}",
        "the generating price": f"{percent(data['truth/put']):.2f}",
    }
    for name, values in (("the put", put), ("the forward-start call", start), ("the book", start - put)):
        q = quantiles(values)
        quoted[f"the 90% width of {name}"] = f"{q[2] - q[0]:.2f}"
    for reach in il.REACH:
        k = key_of(reach)
        low, high = percent(data[f"{k}/put/delta"])
        q = percent(quantiles(data[f"{k}/put/draws"]))
        quoted[f"the posterior range, quotes up to {reach}"] = f"{q[0]:.2f} to {q[2]:.2f}"
        quoted[f"the least-squares price, quotes up to {reach}"] = f"{percent(data[f'{k}/put/point']):.2f}"
        if np.isfinite(low):
            quoted[f"the delta-method range, quotes up to {reach}"] = f"{low:.2f} to {high:.2f}"
    for kappa in FIXED_KAPPA:
        low, high = percent(data[f"fixed/{kappa:g}/put/delta"])
        quoted[f"the range with kappa fixed at {kappa:g}"] = f"{low:.2f} to {high:.2f}"
    _, _, ratio, _ = notebook_agreement(data)
    quoted["the notebook's 63 quotes: largest gap between the two widths, in percent"] = \
        f"{100 * np.max(np.abs(ratio - 1)):.0f}"
    cost = costs()
    coverage = percent(RECORDED["population"]["coverage"]["0.9"])
    quoted["the network's seconds per posterior"] = f"{cost['network_seconds']:.1f}"
    quoted["Heston's exact posterior in CPU-minutes"] = "{:.0f}–{:.0f}".format(*cost["heston_minutes"])
    quoted["rough Bergomi by importance sampling in CPU-hours"] = "{:.0f}–{:.0f}".format(*cost["rough_sampling_hours"])
    quoted["the lowest and highest coverage of the five parameters on target"] = \
        f"{np.sort(coverage)[0]:.0f}–{np.sort(coverage)[-2]:.0f}"
    quoted["the coverage for today's variance"] = f"{coverage[0]:.1f}"
    return quoted


def report(data):
    """Everything behind the slides, printed so that slot1.tex and the README can be checked against it."""
    key = key_of(HOOK)
    put, start = percent(data[f"{key}/put/draws"]), percent(data[f"{key}/forward_start/draws"])
    print(f"\nrunning example, seed {il.SEED}: (v0, kappa, long-run variance, vol of vol, rho) = {il.TRUTH.tolist()}")
    print(f"under the generating parameters: 2-year put {percent(data['truth/put']):.2f}, forward-start call "
          f"{percent(data['truth/forward_start']):.2f}, per 100 of forward")
    for reach in il.REACH:
        k = key_of(reach)
        low, high = percent(data[f"{k}/put/delta"])
        q = percent(quantiles(data[f"{k}/put/draws"]))
        bounds = ", ".join(data[f"{k}/on_bound"])
        classical = f"{low:.2f} to {high:.2f}" if np.isfinite(low) else f"none (the fit is on a bound: {bounds})"
        print(f"  quotes up to {reach:8s} ({int(data[f'{k}/quotes']):2d} quotes): least squares "
              f"{percent(data[f'{k}/put/point']):.2f}, error scale {float(data[f'{k}/scale']):.2f} on "
              f"{int(data[f'{k}/dof'])} degrees of freedom, delta method {classical}; "
              f"posterior {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}; effective sample size {float(data[f'{k}/ess']):.0f} "
              f"of {int(data[f'{k}/proposals'])}, failed pricings {float(data[f'{k}/failed']):.2%}, acceptance of "
              f"the moves {float(data[f'{k}/acceptance']):.2f}, pricing gate {float(data[f'{k}/gate']):.1e}")
        c = data[f"{k}/draws"][:, 5]
        print(f"      error scale c: posterior {np.quantile(c, 0.05):.2f} / {np.median(c):.2f} / "
              f"{np.quantile(c, 0.95):.2f}")
    for label in SMILE_STRIKES:
        k = f"hook/{key_of(label)}"
        at = np.argmin(np.abs(data[f"{k}/strikes"] - 1.0))
        fan = percent(data[f"{k}/fan"])[:, at]
        print(f"  opening example, {label}: at-the-money volatility of the {FAN} draws {fan.min():.1f} to "
              f"{fan.max():.1f}%, fit {percent(data[f'{k}/fit'][at]):.1f}%, "
              f"generating {percent(data[f'{k}/truth'][at]):.1f}%")
    low, high = np.quantile(data["hook/misfit"], [0.05, 0.95])
    print(f"  opening example: root-mean-square misfit in half-spreads, fit {float(data['hook/misfit_fit']):.2f}, "
          f"posterior draws {low:.2f} to {high:.2f} (5% to 95%)")
    design = percent(il.half_spread_volatility(np.array(il.DELTAS)))
    print(f"  the chain's half-spreads, as designed: {design.min():.2f} to {design.max():.2f} volatility points")
    mark, point90 = put.mean(), np.quantile(put, 0.9)
    print(f"  opening example, 2-year put: least squares {percent(data[f'{key}/put/point']):.2f}, mark {mark:.2f}, "
          f"90% point {point90:.2f}, reserve {reserve(mark, point90):.2f} "
          f"({100 * reserve(mark, point90) / mark:.0f}% of the mark)")
    for kappa in FIXED_KAPPA:
        k = f"fixed/{kappa:g}"
        low, high = percent(data[f"{k}/put/delta"])
        print(f"  kappa fixed at {kappa:g}: put {percent(data[f'{k}/put/point']):.2f}, range {low:.2f} to {high:.2f}, "
              f"sum of squared misses {float(data[f'{k}/sum_of_squares']):.1f} "
              f"(fitted: {float(data[f'{key}/sum_of_squares']):.1f})")
    q_put, q_start, q_book = quantiles(put), quantiles(start), quantiles(start - put)
    print(f"  opening example, 90% widths: put {q_put[2] - q_put[0]:.2f}, forward-start call "
          f"{q_start[2] - q_start[0]:.2f}, call minus put {q_book[2] - q_book[0]:.2f}; correlation of the two prices "
          f"{np.corrcoef(put, start)[0, 1]:.3f}")
    classical, exact, ratio, shift = notebook_agreement(data)
    print(f"\nthe notebook's simulated case, {int(data['notebook/quotes'])} quotes: least squares against its exact "
          f"posterior (error scale {float(data['notebook/scale']):.3f})")
    for name, c, e, r, s in zip(il.NAMES, classical, exact, ratio, shift, strict=True):
        print(f"  {name:10s} least squares {c[0]:.5f} to {c[1]:.5f}; exact posterior {e[0]:.5f} to {e[1]:.5f}; "
              f"width ratio {r:.3f}; fit minus posterior mean {s:+.2f} sd")
    cost = costs()
    print("\nrecorded results, Heston on 63 SPX quotes (recorded-results.json)")
    print(f"  network: {cost['network_seconds']:.2f} s per posterior; training bank "
          f"{RECORDED['network']['training_rows']:,} pricings = {cost['bank_in_posteriors']:.2f} exact posteriors")
    print("  one exact posterior: Heston {:.1f} to {:.1f} CPU-minutes ({:,} pricings)".format(
        *cost["heston_minutes"], cost["evaluations"]))
    print("  rough Bergomi, derived: importance sampling {:.1f} to {:.1f} CPU-hours; MCMC {:.0f} to {:.0f} "
          "CPU-days".format(*cost["rough_sampling_hours"], *cost["rough_mcmc_days"]))
    print("  the 512 posteriors of the coverage test by MCMC: Heston {:.1f} to {:.1f} CPU-days; rough Bergomi "
          "{:.0f} to {:.0f} CPU-years".format(*cost["test_heston_days"], *cost["test_rough_years"]))
    for level in RECORDED["population"]["levels"]:
        measured = percent(RECORDED["population"]["coverage"][str(level)])
        print(f"  coverage of {100 * level:g}% intervals: " + ", ".join(f"{x:.1f}" for x in measured))


def stale_numbers(data):
    """The numbers that `slot1.tex` does not contain as `quoted_numbers` gives them: sentences to update."""
    deck = (HERE / "slot1.tex").read_text()
    return {name: number for name, number in quoted_numbers(data).items() if number not in deck}


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
        save(figure_posterior_cost(), "posterior-cost")
        save(figure_coverage(), "coverage")
    print(f"9 figures in {OUT}")
    report(data)
    stale = stale_numbers(data)
    print("\nslot1.tex quotes these numbers as printed above." if not stale else
          "\nslot1.tex does not quote these numbers as printed above; update the slides and the curriculum page:")
    for name, number in stale.items():
        print(f"  {name}: {number}")


if __name__ == "__main__":
    main()
