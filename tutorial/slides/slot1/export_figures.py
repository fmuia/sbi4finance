"""Draw slot 1's figures: vector PDFs in the BayesOps theme's colours, at the size the slides show them.

Run from tutorial/:

    python slides/slot1/export_figures.py [--rubik PATH/TO/Rubik-Variable.ttf] [--recompute]

Seven of the eight PDFs come from the deck's running example (`illustration.py`: Heston on the simulated
"market-like" quotes of the Heston SPX notebook, the model right by construction; the opening figure has two steps). The
eighth, `coverage`, plots the checks of the option application's Heston network on 512 simulated surfaces; its
numbers are copied below from the application's record, since the slides' code reads nothing outside tutorial/.
The example takes about half an hour on four cores, mostly five posteriors by importance sampling; its numbers are
cached under tutorial/.cache/ (ignored by Git) and reused until `illustration.py`, `spx-design.json`,
`market-like-case.json`, the settings below or `compute` change, or --recompute is given. The script ends by
printing every number the slides quote, and says which of them `slot1.tex` does not contain.

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

# The checks of the option application's Heston network (300,000 training rows, flow matching, whitened inputs) on
# 512 surfaces simulated from the prior, 512 draws each: the share of surfaces whose central interval contains the
# truth. Copied from lab/plans/option-application/figure-sources/data.json, key heston_300k/coverage, on branch
# claude/tender-bardeen-l6tivw (commit ea6e25b), which records
# lab/evidence/pilot-results/2026-11-09/heston-spread-baseline/revised-grid/recovery-bank-independent/300k-300/
# evaluation/evaluation.json at commit 1c276bb72c (SHA-256 50312e17...0f90); documented in 3-full-detail.md, 8.2.
COVERAGE = {
    "surfaces": 512,
    "targets": ("v0", "kappa", "long_run", "vol_of_vol", "rho", "c"),
    "levels": {
        0.50: (0.675781, 0.533203, 0.556641, 0.515625, 0.535156, 0.500000),
        0.80: (0.925781, 0.816406, 0.843750, 0.830078, 0.847656, 0.789062),
        0.90: (0.974609, 0.900391, 0.923828, 0.892578, 0.919922, 0.902344),
        0.95: (0.992188, 0.941406, 0.958984, 0.931641, 0.966797, 0.949219),
    },
}

# The Heston SPX notebook's PyMC reference for the same 63 quotes (its case "market-like parameters"): four
# DEMetropolisZ chains of 12,000 draws, the exact Fourier likelihood, the same priors. Quantiles 5%, 50%, 95% of the
# parameters, from lab/evidence/pilot-results/2026-11-11/heston-actual63-notebook/references/synthetic-market-like/
# posterior.json (SHA-256 c83e9d38...cb32), and of the later-window variance as a volatility in %, from the 2,000
# draws of reference-display-draws.npz (SHA-256 a78a3510...4976), branch claude/trusting-dijkstra-hmgvcm, 2462b64.
# The slides' posterior of the 325-day reading (all 63 quotes) is compared with it in the report.
REFERENCE = {
    "v0": (0.027049986642360135, 0.027121354481854792, 0.02719199434877243),
    "kappa": (4.344922436891941, 4.410009460940039, 4.473384904009994),
    "long_run": (0.056044411660073436, 0.05632057884911886, 0.056608758574305855),
    "vol_of_vol": (1.8048672590115284, 1.817400469384277, 1.8300057457417425),
    "rho": (-0.6671271733545702, -0.6657683915740334, -0.6644440875540412),
    "c": (0.7727821652192872, 0.8959219539247977, 1.0540524714863415),
    "forward_variance": (22.94, 22.9776, 23.0153),
}

# The theme's colours (aviation #15153B, neonlake #6969FF) and the figures' second colour.
INK, INK_2, MUTED, GRID, BASELINE = "#15153b", "#3f3f63", "#7c7c99", "#e7e7f1", "#c4c4d6"
POSTERIOR, CLASSICAL = "#6969ff", "#c2187a"  # periwinkle: the posterior; magenta: least squares and its ranges

TEXT_WIDTH = 5.51  # inches: the 140 mm text width of a 16:9 Beamer frame
HOOK = "14 days"  # the reading of the opening example: 18 quotes, at 7 and 14 days
FIXED_KAPPA = (2.0, 4.0, 8.0)  # the habit of fixing the mean reversion, at three values
FAN = 120  # posterior curves drawn in a fan
# Standardized moneyness delta = ln(K/F) / (0.20 sqrt T), the Heston SPX notebook's coordinate (the design's
# z = 1.25 delta).
# The grids are the strikes of z from -5 to 2 and from -3 to 2.
Z_QUOTED = np.linspace(-4.0, 1.6, 57)  # along a quoted expiry
Z_FORWARD = np.linspace(-2.4, 1.6, 41)  # along the forward smile, 6 months to 1 year
PRODUCTS = il.PRODUCTS
# The 7-day reading's posterior is a thin curved ridge along which the box sets the ends: its proposal is adapted over
# more rounds and more slices, to a flatter posterior, with more draws from the prior, before the Metropolis moves.
THIN = {"7 days": {"adapt": (6000,) * 6, "pieces": 8, "temper": 0.3, "prior_share": 0.3, "most": 150000}}
DASHED = (0, (4, 2))
BOUND_NAMES = {"v0": "v₀", "kappa": "κ", "long_run": "the long-run variance", "vol_of_vol": "σ_v", "rho": "ρ"}
RANGE_LEGEND = ("least squares, delta method", "posterior, 5% to 95%", "price under the generating parameters")


# ---------------------------------------------------------------------------------------------------------------
# Numbers


def key_of(reach):
    return reach.replace(" ", "_")


def _posterior(reach):
    return reach, il.posterior(il.readings()[reach], **THIN.get(reach, {}))


def _strikes(z, maturity):
    return np.exp(np.asarray(z) * il.DISPLAY_REFERENCE * np.sqrt(maturity))


def compute():
    """Every number the figures need, as a dictionary of arrays and scalars."""
    readings = il.readings()
    with ProcessPoolExecutor(max_workers=min(len(il.REACH), os.cpu_count() or 1)) as pool:
        posteriors = dict(pool.map(_posterior, il.REACH))
    data = {f"truth/{name}": product(il.TRUTH) for name, product in PRODUCTS.items()}

    def classical(prefix, result):
        data[f"{prefix}/fit"] = result.theta
        data[f"{prefix}/sum_of_squares"] = result.sum_of_squares
        data[f"{prefix}/c"] = result.c
        data[f"{prefix}/on_bound"] = np.array(result.on_bound, dtype=str)
        for name, product in PRODUCTS.items():
            point, interval = il.delta_range(result, product)
            data[f"{prefix}/{name}/point"] = point
            data[f"{prefix}/{name}/delta"] = np.array(interval if interval is not None else (np.nan, np.nan))

    for reach in il.REACH:
        reading, post, key = readings[reach], posteriors[reach], key_of(reach)
        result = il.fit(reading)
        classical(key, result)
        for name, product in PRODUCTS.items():
            data[f"{key}/{name}/draws"] = np.array([product(theta) for theta in post.draws[:, :5]])
        data[f"{key}/draws"] = post.draws
        data[f"{key}/ess"] = post.effective_sample_size
        data[f"{key}/proposals"] = post.proposals
        data[f"{key}/failed"] = post.failed
        data[f"{key}/quotes"] = reading.n
        data[f"{key}/gate"] = max(il.gate(theta, reading.n) for theta in post.draws[:200, :5])
        if result.covariance is not None:
            data[f"{key}/ellipse_kappa_long_run"] = il.gauss_newton_ellipse(result, 1, 2)
            data[f"{key}/ellipse_xi_rho"] = il.gauss_newton_ellipse(result, 3, 4)

    # The opening example: smiles of the fit, the truth and a fan of posterior draws, quoted and not quoted.
    reading, draws = readings[HOOK], posteriors[HOOK].draws[:, :5]
    fit = data[f"{key_of(HOOK)}/fit"]
    mid, low, high = reading.implied_quotes()
    z = np.log(il.SPX["strike"][:reading.n]) / (il.DISPLAY_REFERENCE * np.sqrt(il.SPX["maturity"][:reading.n]))
    data["hook/quotes"] = np.column_stack([z, il.SPX["maturity"][:reading.n], mid, low, high])
    data["hook/misfit"] = np.array([np.sqrt(reading.sum_of_squares(theta) / reading.n) for theta in draws])
    data["hook/misfit_fit"] = np.sqrt(data[f"{key_of(HOOK)}/sum_of_squares"] / reading.n)
    for j in range(il.READINGS[HOOK]):
        maturity = il.SPX["maturity"][9 * j]
        key = f"hook/expiry_{j}"
        strikes = _strikes(Z_QUOTED, maturity)
        data[f"{key}/maturity"] = maturity
        data[f"{key}/fit"] = il.smile(fit, maturity, strikes)
        data[f"{key}/truth"] = il.smile(il.TRUTH, maturity, strikes)
        data[f"{key}/fan"] = np.array([il.smile(theta, maturity, strikes) for theta in draws[:FAN]])
    strikes = _strikes(Z_FORWARD, il.END - il.START)
    data["hook/forward/fit"] = il.forward_smile(fit, strikes)
    data["hook/forward/truth"] = il.forward_smile(il.TRUTH, strikes)
    data["hook/forward/fan"] = np.array([il.forward_smile(theta, strikes) for theta in draws[:FAN]])

    # The habit of fixing the mean reversion, on the opening example's quotes.
    for kappa in FIXED_KAPPA:
        classical(f"fixed/{kappa:g}", il.fit(reading, fixed={1: kappa}))
    return data


def load(recompute=False):
    """The cached numbers, recomputed when the example, its design, the settings above or `compute` have changed."""
    recipe = json.dumps([HOOK, FIXED_KAPPA, FAN, Z_QUOTED.tolist(), Z_FORWARD.tolist(), sorted(PRODUCTS), THIN,
                         inspect.getsource(compute)])
    inputs = ("illustration.py", "spx-design.json", "market-like-case.json")
    source = b"".join((HERE / name).read_bytes() for name in inputs)
    key = hashlib.sha256(source + recipe.encode()).hexdigest()[:16]
    path = CACHE / f"illustration-{key}.npz"
    if recompute or not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print("computing the running example (about half an hour on four cores) ...", flush=True)
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
    """The opening example: quotes at two expiries and the forward smile, which no option quotes.

    Step 1 shows the quotes and the least-squares fit. Step 2 adds a fan of posterior draws and, where there are no
    quotes, the forward smile of the parameters that generated them and the spread of the fan at the money.
    """
    expiries = il.READINGS[HOOK]
    fig, axes = canvas(0.97, 1.60, ncols=expiries + 1, sharey=True,
                       gridspec_kw={"width_ratios": [1] * expiries + [1.3]})
    quotes = data["hook/quotes"]
    for j, ax in enumerate(axes[:-1]):
        key = f"hook/expiry_{j}"
        maturity = float(data[f"{key}/maturity"])
        if step == 2:
            ax.plot(Z_QUOTED, percent(data[f"{key}/fan"]).T, color=POSTERIOR, lw=0.45, alpha=0.22, zorder=2)
        ax.plot(Z_QUOTED, percent(data[f"{key}/fit"]), color=CLASSICAL, lw=1.3, zorder=3)
        quoted = np.isclose(quotes[:, 1], maturity)
        mid, low, high = (percent(quotes[quoted, j]) for j in (2, 3, 4))
        ax.errorbar(quotes[quoted, 0], mid, yerr=[mid - low, high - mid], fmt="o", color=INK, ms=2.6, lw=0.8,
                    capsize=1.5, capthick=0.8, zorder=5)
        ax.set_title(f"quoted: {int(round(365 * maturity))} days", loc="left")
        ax.set_xlabel("moneyness δ")
        ax.set_xlim(Z_QUOTED[0] - 0.2, Z_QUOTED[-1] + 0.2)
    ax = axes[-1]
    ax.set_facecolor("#f5f5fd")
    if step == 2:
        fan = percent(data["hook/forward/fan"])
        ax.plot(Z_FORWARD, fan.T, color=POSTERIOR, lw=0.45, alpha=0.22, zorder=2)
        ax.plot(Z_FORWARD, percent(data["hook/forward/truth"]), color=INK, lw=1.0, ls=DASHED, zorder=4)
        spread = fan[:, np.argmin(np.abs(Z_FORWARD))]
        ax.annotate("", xy=(0, spread.min()), xytext=(0, spread.max()), zorder=6,
                    arrowprops={"arrowstyle": "|-|,widthA=0.25,widthB=0.25", "color": INK, "lw": 0.8,
                                "shrinkA": 0, "shrinkB": 0})
        ax.text(0.15, spread.max() + 0.6, f"{spread.min():.1f} to {spread.max():.1f}%", fontsize=6.3, color=INK,
                va="bottom", ha="left")
    ax.plot(Z_FORWARD, percent(data["hook/forward/fit"]), color=CLASSICAL, lw=1.3, zorder=3)
    ax.set_title("not quoted: the smile in 6 months", loc="left")
    ax.set_xlabel("moneyness δ, from 6 months to 1 year")
    ax.set_xlim(Z_FORWARD[0] - 0.2, Z_FORWARD[-1] + 0.2)
    axes[0].set_ylabel("implied volatility (%)")
    handles = [key_line(INK, marker="o", ms=2.3, lw=0.7), key_line(CLASSICAL, lw=1.3)]
    labels = ["quotes ± half-spread", "least-squares fit"]
    if step == 2:
        handles += [key_line(POSTERIOR, lw=1.0), key_line(INK, lw=1.0, ls=DASHED)]
        labels += [f"{FAN} posterior draws", "generating smile"]
    legend_above(fig, handles, labels)
    return fig


def figure_price_range(data):
    """The forward-start call under every posterior draw of the opening example: mark, 10% point and reserve."""
    key = key_of(HOOK)
    values = percent(data[f"{key}/forward_start/draws"])
    mark, point10, least = values.mean(), np.quantile(values, 0.10), float(percent(data[f"{key}/forward_start/point"]))
    fig, ax = canvas(0.88, 1.58)
    low, high = np.quantile(values, 0.001), np.quantile(values, 0.999)
    pad = 0.08 * (high - low)
    counts, _, _ = ax.hist(values, bins=np.linspace(low - pad, high + pad, 49), color=POSTERIOR, alpha=0.5, lw=0,
                           rwidth=0.88)
    top = counts.max()
    marks = ((point10, INK, DASHED, "right", -0.01 * (high - low), 1.16, f"10% point\n{point10:.2f}"),
             (mark, INK, "-", "left", 0.01 * (high - low), 1.16, f"mark: the mean\n{mark:.2f}"),
             (least, CLASSICAL, "-", "right", -0.01 * (high - low), 1.42, f"least squares\n{least:.2f}"))
    for x, colour, dashes, side, shift, height, text in marks:
        ax.plot([x, x], [0, (height - 0.04) * top], color=colour, lw=1.0, ls=dashes)
        ax.text(x + shift, height * top, text, ha=side, va="bottom", fontsize=6.5, color=INK_2, linespacing=1.15)
    # The reserve above the bars, its label on the arrow on a white ground so that no line runs through it.
    ax.annotate("", xy=(point10, 1.06 * top), xytext=(mark, 1.06 * top),
                arrowprops={"arrowstyle": "<->", "color": INK, "lw": 0.7, "shrinkA": 0, "shrinkB": 0})
    ax.text(0.5 * (mark + point10), 1.06 * top, f"reserve {mark - point10:.2f}", ha="center", va="center",
            fontsize=6.5, color=INK, zorder=5, bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none"})
    ax.set(xlim=(low - pad, high + pad), ylim=(0, 1.62 * top), yticks=[])
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("forward-start call, 6 months to 1 year, per 100 of forward: one value per posterior draw")
    return fig


def figure_posterior_cloud(data):
    """The posterior of the opening example in two planes, with least squares and its Gauss-Newton ellipse."""
    key = key_of(HOOK)
    draws, fit = data[f"{key}/draws"], data[f"{key}/fit"]
    fig, axes = canvas(0.92, 1.58, ncols=2, gridspec_kw={"width_ratios": [1.3, 1.0]})

    def volatility(variance):
        return 100 * np.sqrt(np.clip(variance, 0, None))

    planes = (  # x and y parameters, y scale, ellipse, title, labels, and the prior box as the axis limits
        (1, 2, volatility, "ellipse_kappa_long_run", "left open by the quotes",
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
    """The forward-start call's 90% range, classical and posterior, as the quotes stop earlier and earlier."""
    rows = il.REACH
    fig, ax = canvas(0.92, 1.58)
    left, right = 4.0, 5.3  # the thinnest reading's posterior runs beyond both ends: drawn clipped, with its numbers
    for y, reach in enumerate(rows):
        key = key_of(reach)
        point = float(percent(data[f"{key}/forward_start/point"]))
        low, high = percent(data[f"{key}/forward_start/delta"])
        if np.isfinite(low):
            whisker(ax, y - 0.17, low, high, point, CLASSICAL)
        else:  # the fit is on a bound of the box: its price, drawn at the edge when outside the axis, and no range
            shown = min(max(point, left + 0.01), right - 0.01)
            ax.plot([shown], [y - 0.17], "o", mfc="white", mec=CLASSICAL, ms=3.2, mew=0.9, zorder=4, clip_on=False)
            bounds = " and ".join(BOUND_NAMES.get(str(b), str(b)) for b in data[f"{key}/on_bound"])
            ax.text(shown + 0.03, y - 0.17, f"least squares {point:.2f}, {bounds} on its bound: no range",
                    va="center", ha="left", fontsize=6.3, color=INK_2, zorder=5,
                    bbox={"boxstyle": "square,pad=0.1", "fc": "white", "ec": "none"})
        q = percent(quantiles(data[f"{key}/forward_start/draws"]))
        if q[0] < left or q[2] > right:
            ax.annotate("", xy=(left, y + 0.17), xytext=(right, y + 0.17), zorder=3,
                        arrowprops={"arrowstyle": "<|-|>,head_length=0.3,head_width=0.15", "color": POSTERIOR,
                                    "lw": 1.7, "shrinkA": 0, "shrinkB": 0})
            ax.text(right - 0.02, y + 0.17 + 0.12, f"posterior {q[0]:.2f} to {q[2]:.2f}, beyond both ends",
                    ha="right", va="top", fontsize=6.3, color=INK_2, zorder=5,
                    bbox={"boxstyle": "square,pad=0.1", "fc": "white", "ec": "none"})
        else:
            whisker(ax, y + 0.17, q[0], q[2], q[1], POSTERIOR)
    ax.set_xlim(left, right)
    ax.axvline(float(percent(data["truth/forward_start"])), color=INK, lw=0.8, ls=DASHED, zorder=1)
    labels = [f"quotes up to {reach} ({int(data[f'{key_of(reach)}/quotes'])})" for reach in rows]
    rows_axis(ax, range(len(rows)), labels, "forward-start call, 6 months to 1 year, per 100 of forward")
    ax.set_ylim(len(rows) - 1 + 1.2, -0.6)
    legend_above(fig, range_handles(), RANGE_LEGEND)
    return fig


def figure_fixed_parameter(data):
    """The opening example's forward-start call when the mean reversion is fixed, fitted, or given a prior."""
    key = key_of(HOOK)
    fig, ax = canvas(0.92, 1.40)
    positions, labels = [0, 1, 2, 3.35, 4.35], [f"κ fixed at {kappa:g}" for kappa in FIXED_KAPPA]
    for y, kappa in enumerate(FIXED_KAPPA):
        low, high = percent(data[f"fixed/{kappa:g}/forward_start/delta"])
        whisker(ax, positions[y], low, high, float(percent(data[f"fixed/{kappa:g}/forward_start/point"])), CLASSICAL)
    low, high = percent(data[f"{key}/forward_start/delta"])
    whisker(ax, positions[3], low, high, float(percent(data[f"{key}/forward_start/point"])), CLASSICAL)
    q = percent(quantiles(data[f"{key}/forward_start/draws"]))
    whisker(ax, positions[4], q[0], q[2], q[1], POSTERIOR)
    ax.axvline(float(percent(data["truth/forward_start"])), color=INK, lw=0.8, ls=DASHED, zorder=1)
    ax.axhline(2.68, color=GRID, lw=0.8, zorder=1)
    rows_axis(ax, positions, labels + ["κ fitted: least squares", "κ uncertain: posterior"],
              "forward-start call, 6 months to 1 year, per 100 of forward: 90% ranges")
    legend_above(fig, range_handles(), RANGE_LEGEND)
    return fig


def figure_joint_draws(data):
    """Two products priced under the same posterior draws, and a book that holds one against the other."""
    key = key_of(HOOK)
    start, last = percent(data[f"{key}/forward_start/draws"]), percent(data[f"{key}/last_call/draws"])
    fig, axes = canvas(0.92, 1.58, ncols=2, gridspec_kw={"width_ratios": [1.0, 1.3]})
    axes[0].scatter(last, start, s=1.6, color=POSTERIOR, alpha=0.16, lw=0)
    axes[0].set(xlabel="325-day at-the-money call (listed)", ylabel="forward-start call")
    axes[0].set_title("one dot: one posterior draw, two prices", loc="left")
    rows = {"forward-start call": start, "325-day call": last, "book: long forward start,\nshort 325-day call":
            start - last}
    for y, values in enumerate(rows.values()):
        q = quantiles(values)
        whisker(axes[1], y, q[0] - q[1], q[2] - q[1], 0.0, POSTERIOR)
        axes[1].text(q[2] - q[1] + 0.02, y, f"{q[2] - q[0]:.2f} wide", va="center", fontsize=6.3, color=INK_2)
    axes[1].axhline(1.5, color=GRID, lw=0.8, zorder=1)
    rows_axis(axes[1], range(len(rows)), list(rows), "90% range around the median, per 100 of forward")
    axes[1].set_title("the book's range is not the sum", loc="left")
    return fig


def figure_coverage():
    """Stated against measured coverage of the Heston network's intervals on 512 simulated surfaces."""
    names = {"v0": "variance today", "kappa": "mean reversion", "long_run": "long-run variance",
             "vol_of_vol": "volatility of volatility", "rho": "correlation", "c": "error scale c"}
    markers = ("o", "s", "D", "^", "v", "P")
    levels = sorted(COVERAGE["levels"])
    fig, ax = canvas(0.92, 1.40)
    for y, level in enumerate(levels):
        stated, measured = 100 * level, percent(COVERAGE["levels"][level])
        ax.plot([stated, stated], [y - 0.38, y + 0.38], color=INK, lw=1.1, zorder=2)
        for j, value in enumerate(measured):
            ax.plot([value], [y - 0.30 + 0.12 * j], markers[j], color=POSTERIOR, ms=3.3, mec="white", mew=0.4,
                    zorder=3)
        ax.text(max(stated, measured.max()) + 1.8, y, f"measured {measured.min():.1f} to {measured.max():.1f}%",
                va="center", fontsize=6.3, color=INK_2)
    rows_axis(ax, range(len(levels)), [f"{100 * level:g}% intervals" for level in levels],
              f"share of {COVERAGE['surfaces']} simulated surfaces whose interval contains the true parameter (%)")
    ax.set_xlim(40, 118)
    ax.set_xticks(range(40, 101, 10))
    handles = [key_line(INK, lw=1.1)] + [key_line(POSTERIOR, marker=marker, ms=3.3, lw=0) for marker in markers]
    fig.legend(handles, ["stated"] + [names[target] for target in COVERAGE["targets"]], loc="outside upper left",
               ncols=4, frameon=False, handlelength=1.6, columnspacing=1.4, handletextpad=0.5, borderaxespad=0.0)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# The numbers the slides quote


def _range(values, digits=2):
    low, _, high = quantiles(values)
    return f"{low:.{digits}f} to {high:.{digits}f}"


def quoted_numbers(data):
    """Every number of the example and of the coverage figure that the slides quote, as the deck writes it."""
    key = key_of(HOOK)
    start = percent(data[f"{key}/forward_start/draws"])
    last = percent(data[f"{key}/last_call/draws"])
    fan = percent(data["hook/forward/fan"])[:, np.argmin(np.abs(Z_FORWARD))]
    mark, point10 = start.mean(), np.quantile(start, 0.10)
    misfit = np.quantile(data["hook/misfit"], [0.05, 0.95])
    quoted = {
        "the fan's lowest forward at-the-money volatility": f"{fan.min():.1f}",
        "the fan's highest forward at-the-money volatility": f"{fan.max():.1f}",
        "the fit's misfit in half-spreads": f"{float(data['hook/misfit_fit']):.2f}",
        "the posterior draws' misfit in half-spreads": f"{misfit[0]:.2f}–{misfit[1]:.2f}",
        "the mark": f"{mark:.2f}", "the 10% point": f"{point10:.2f}", "the reserve": f"{mark - point10:.2f}",
        "the reserve as a share of the mark": f"{100 * (1 - point10 / mark):.0f}\\%",
        "the least-squares price": f"{percent(data[f'{key}/forward_start/point']):.2f}",
        "the generating price": f"{percent(data['truth/forward_start']):.2f}",
    }
    quoted["the correlation of the forward start and the 325-day call"] = f"{np.corrcoef(start, last)[0, 1]:.3f}"
    for reach in il.REACH:
        quoted[f"the least-squares price, quotes up to {reach}"] = \
            f"{percent(data[f'{key_of(reach)}/forward_start/point']):.2f}"
    ess = {reach: float(data[f"{key_of(reach)}/ess"]) for reach in il.REACH}
    wide = [value for reach, value in ess.items() if reach not in THIN]
    quoted["the effective sample sizes of the wider readings"] = f"{min(wide):,.0f} to {max(wide):,.0f}"
    for reach in THIN:
        quoted[f"the effective sample size, quotes up to {reach}"] = f"{ess[reach]:,.0f}"
    for name, values in (("the forward-start call", start), ("the 325-day call", last), ("the book", start - last)):
        low, _, high = quantiles(values)
        quoted[f"the 90% width of {name}"] = f"{high - low:.2f}"
    for reach in il.REACH:
        k = key_of(reach)
        low, high = percent(data[f"{k}/forward_start/delta"])
        quoted[f"the posterior range, quotes up to {reach}"] = _range(percent(data[f"{k}/forward_start/draws"]))
        if np.isfinite(low):
            quoted[f"the delta-method range, quotes up to {reach}"] = f"{low:.2f} to {high:.2f}"
    for kappa in FIXED_KAPPA:
        low, high = percent(data[f"fixed/{kappa:g}/forward_start/delta"])
        quoted[f"the range with kappa fixed at {kappa:g}"] = f"{low:.2f} to {high:.2f}"
    measured = percent(COVERAGE["levels"][0.90])
    others = np.delete(measured, 0)
    quoted["the 90% coverage of five targets"] = f"{others.min():.0f}–{others.max():.0f}"
    quoted["the 90% coverage of v0"] = f"{measured[0]:.1f}"
    return quoted


def report(data):
    """Every number the example gives, printed so that slot1.tex and the curriculum page can be checked against it."""
    key = key_of(HOOK)
    start = percent(data[f"{key}/forward_start/draws"])
    print(f"\nrunning example, the notebook's market-like case (sampler seed {il.SEED}): "
          f"(v0, kappa, long-run variance, vol of vol, rho) = {il.TRUTH.tolist()}, c = {il.C_TRUE}")
    print(f"under the generating parameters: forward-start call {percent(data['truth/forward_start']):.2f}, "
          f"later-window variance {percent(data['truth/forward_variance']):.2f}% volatility, 325-day call "
          f"{percent(data['truth/last_call']):.2f}, per 100 of forward")
    for reach in il.REACH:
        k = key_of(reach)
        low, high = percent(data[f"{k}/forward_start/delta"])
        q = percent(quantiles(data[f"{k}/forward_start/draws"]))
        bounds = ", ".join(data[f"{k}/on_bound"])
        classical = f"{low:.2f} to {high:.2f}" if np.isfinite(low) else f"none (the fit is on a bound: {bounds})"
        c = quantiles(data[f"{k}/draws"][:, 5])
        fv = percent(quantiles(data[f"{k}/forward_variance/draws"]))
        lv, lh = percent(data[f"{k}/forward_variance/delta"])
        print(f"  quotes up to {reach:8s} ({int(data[f'{k}/quotes']):2d} quotes): forward start: least squares "
              f"{percent(data[f'{k}/forward_start/point']):.2f}, delta method {classical}; posterior {q[0]:.2f} / "
              f"{q[1]:.2f} / {q[2]:.2f}. Later-window variance (vol %): delta {lv:.2f} to {lh:.2f}, posterior "
              f"{fv[0]:.2f} to {fv[2]:.2f}. c: fit {float(data[f'{k}/c']):.2f}, posterior {c[0]:.2f} / {c[1]:.2f} / "
              f"{c[2]:.2f}. Effective sample size {float(data[f'{k}/ess']):.0f} of {int(data[f'{k}/proposals'])}, "
              f"failed pricings {float(data[f'{k}/failed']):.2%}, gate at 200 draws {float(data[f'{k}/gate']):.1e}")
    draws = data[f"{key}/draws"]
    for i, name in enumerate(il.NAMES + ("c",)):
        print(f"  opening example, posterior of {name}: {_range(draws[:, i], 4)}; fit "
              f"{data[f'{key}/fit'][i] if i < 5 else float(data[f'{key}/c']):.4f}")
    at = np.argmin(np.abs(Z_FORWARD))
    print(f"  opening example, forward smile at the money: fit {percent(data['hook/forward/fit'][at]):.1f}%, "
          f"generating {percent(data['hook/forward/truth'][at]):.1f}%")
    mark, point10 = start.mean(), np.quantile(start, 0.10)
    print(f"  opening example, forward-start call: mark {mark:.2f}, 10% point {point10:.2f}, reserve "
          f"{mark - point10:.2f}")
    for kappa in FIXED_KAPPA:
        k = f"fixed/{kappa:g}"
        low, high = percent(data[f"{k}/forward_start/delta"])
        print(f"  kappa fixed at {kappa:g}: forward start {percent(data[f'{k}/forward_start/point']):.2f}, range "
              f"{low:.2f} to {high:.2f}, sum of squares {float(data[f'{k}/sum_of_squares']):.1f} (fitted: "
              f"{float(data[f'{key}/sum_of_squares']):.1f}), c {float(data[f'{k}/c']):.2f} (fitted: "
              f"{float(data[f'{key}/c']):.2f})")
    full = data[f"{key_of(il.REACH[0])}/draws"]
    fv = percent(data[f"{key_of(il.REACH[0])}/forward_variance/draws"])
    print("  all 63 quotes against the notebook's PyMC reference, 5% / 50% / 95% (shift of each quantile in"
          " reference 90% widths):")
    for i, name in enumerate(il.NAMES + ("c", "forward_variance")):
        mine = quantiles(full[:, i] if i < 6 else fv)
        ref = np.array(REFERENCE[name])
        shift = (mine - ref) / (ref[2] - ref[0])
        print(f"    {name:16s} slides {mine[0]:.5g} / {mine[1]:.5g} / {mine[2]:.5g}; "
              f"PyMC {ref[0]:.5g} / {ref[1]:.5g} / {ref[2]:.5g}; shifts {shift[0]:+.2f} {shift[1]:+.2f} {shift[2]:+.2f}")
    last = percent(data[f"{key}/last_call/draws"])
    print(f"  correlation of the forward start and the 325-day call across draws: {np.corrcoef(start, last)[0, 1]:.3f}")
    for name, number in quoted_numbers(data).items():
        print(f"  {name}: {number}")


def stale_numbers(data):
    """The quoted numbers that `slot1.tex` does not contain as they are printed: sentences to update."""
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
        save(figure_coverage(), "coverage")
    print(f"8 figures in {OUT}")
    report(data)
    stale = stale_numbers(data)
    print("\nslot1.tex quotes these numbers as printed above." if not stale else
          "\nslot1.tex does not quote these numbers as printed above; update the slides and the curriculum page:")
    for name, number in stale.items():
        print(f"  {name}: {number}")


if __name__ == "__main__":
    main()
