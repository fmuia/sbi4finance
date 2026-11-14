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
HOOK = "6 months and 1 year"  # the reading of the opening example: 14 quotes, at the product's two dates
ONE = "1 year"  # the reading with one expiry, where an assumption sets the price
ASSUMED_VOLATILITY = (0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30, 0.325, 0.35, 0.375, 0.40)
FAN = 120  # posterior draws shown one by one
SMILE_STRIKES = {6: (0.65, 1.14), 12: (0.55, 1.20)}  # by expiry in months: the strikes a smile panel shows
DASHED = (0, (4, 2))
RANGE_LEGEND = ("least squares, delta method", "posterior, 5% to 95%", "price under the generating parameters")
READING_LABELS = {  # how a reading is named beside its row
    "1 month to 2 years": "5 expiries: 1 month to 2 years", "1 month to 1 year": "4 expiries: 1 month to 1 year",
    "3 months to 1 year": "3 expiries: 3 months to 1 year", "6 months and 1 year": "2 expiries: 6 months and 1 year",
    "1 year": "1 expiry: 1 year",
}
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
    with ProcessPoolExecutor(max_workers=min(len(readings), os.cpu_count() or 1)) as pool:
        posteriors = dict(pool.map(_posterior, readings))
    data = {f"truth/{name}": float(product(il.TRUTH)) for name, product in il.PRODUCTS.items()}

    for reach, quotes in readings.items():
        post, key = posteriors[reach], key_of(reach)
        result = il.fit(quotes)
        data[f"{key}/fit"] = result.theta
        data[f"{key}/sum_of_squares"] = result.sum_of_squares
        data[f"{key}/scale"] = result.scale
        data[f"{key}/dof"] = result.dof
        data[f"{key}/on_bound"] = np.array(result.on_bound, dtype=str)
        if result.covariance is not None:
            data[f"{key}/covariance"] = result.covariance
        for name, product in il.PRODUCTS.items():
            point, interval = il.delta_range(result, product)
            data[f"{key}/{name}/point"] = point
            data[f"{key}/{name}/delta"] = np.array(interval if interval is not None else (np.nan, np.nan))
            data[f"{key}/{name}/draws"] = product(post.draws[:, :5])
        data[f"{key}/draws"] = post.draws
        data[f"{key}/ess"] = post.effective_sample_size
        data[f"{key}/proposals"] = post.proposals
        data[f"{key}/failed"] = post.failed
        data[f"{key}/acceptance"] = post.acceptance
        data[f"{key}/quotes"] = quotes.n
        data[f"{key}/gate"] = max(quotes.gate(theta, il.CHECK_RUNG) for theta in post.draws[:200, :5])

    # The opening example: the quoted smiles under the fit and under a fan of posterior draws.
    quotes, draws = readings[HOOK], posteriors[HOOK].draws[:, :5]
    mid, low, high = quotes.implied()
    data["hook/quotes"] = np.column_stack([quotes.strike, quotes.maturity, mid, low, high])
    data["hook/misfit"] = np.sqrt(quotes.sum_of_squares(draws) / quotes.n)
    data["hook/misfit_fit"] = np.sqrt(data[f"{key_of(HOOK)}/sum_of_squares"] / quotes.n)
    for months, (lowest, highest) in SMILE_STRIKES.items():
        strikes = np.linspace(lowest, highest, 51)
        key = f"hook/{months}"
        data[f"{key}/strikes"] = strikes
        data[f"{key}/fit"] = il.smile(data[f"{key_of(HOOK)}/fit"], months / 12.0, strikes)
        data[f"{key}/fan"] = np.array([il.smile(theta, months / 12.0, strikes) for theta in draws[:FAN]])

    # One expiry: least squares with today's variance fixed at each assumed value.
    quotes = readings[ONE]
    assumed = [il.fit(quotes, fixed={0: volatility**2}) for volatility in ASSUMED_VOLATILITY]
    data["assumed/forward_start"] = np.array([float(il.forward_start_call(result.theta)) for result in assumed])
    data["assumed/misfit"] = np.array([np.sqrt(result.sum_of_squares / quotes.n) for result in assumed])
    data["assumed/bounds"] = np.array([len(result.on_bound) for result in assumed])

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
    recipe = json.dumps([HOOK, ONE, ASSUMED_VOLATILITY, FAN, sorted(SMILE_STRIKES.items()), sorted(il.PRODUCTS),
                         inspect.getsource(compute)])
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


def figure_prior_likelihood(data):
    """What a prior and a likelihood are, on the opening example's 6-month quotes.

    Left: the 6-month smiles of parameter sets drawn from the prior, before any quote. Right: the quotes, and two of
    those smiles scored against them: the best fit, and one that misses by several half-spreads.
    """
    months = 6
    quotes = il.readings()[HOOK]
    strikes = data[f"hook/{months}/strikes"]
    rng = np.random.default_rng(il.SEED)
    thetas = il.LOWER + (il.UPPER - il.LOWER) * rng.uniform(size=(FAN, len(il.NAMES)))
    smiles = np.array([il.smile(theta, months / 12.0, strikes) for theta in thetas])
    misfit = np.sqrt(quotes.sum_of_squares(thetas) / quotes.n)
    poor = int(np.argmin(np.abs(misfit - 10.0)))  # a prior draw that misses by about ten half-spreads
    fig, axes = canvas(0.94, HEIGHT, ncols=2)
    axes[0].plot(100 * strikes, percent(smiles).T, color=INK_2, lw=0.5, alpha=0.25)
    axes[0].set_title("the prior: smiles before any quote", loc="left")
    points = data["hook/quotes"]
    quoted = np.isclose(points[:, 1], months / 12.0)
    mid, low, high = (percent(points[quoted, j]) for j in (2, 3, 4))
    axes[1].plot(100 * strikes, percent(smiles[poor]), color=INK_2, lw=1.2, ls=DASHED, zorder=2)
    axes[1].plot(100 * strikes, percent(data[f"hook/{months}/fit"]), color=CLASSICAL, lw=1.5, zorder=3)
    axes[1].errorbar(100 * points[quoted, 0], mid, yerr=[mid - low, high - mid], fmt="o", color=INK, ms=3.0, lw=0.9,
                     capsize=1.8, capthick=0.9, zorder=5)
    axes[1].set_title("the likelihood: the quotes score each smile", loc="left")
    handles = [key_line(INK, marker="o", ms=2.6, lw=0.8), key_line(CLASSICAL, lw=1.5),
               key_line(INK_2, lw=1.2, ls=DASHED)]
    legend_above(fig, handles, ["quotes ± half-spread", f"misses by {float(data['hook/misfit_fit']):.1f} half-spreads: likely",
                                f"misses by {misfit[poor]:.0f} half-spreads: unlikely"])
    for ax, top in zip(axes, (60, 45), strict=True):
        ax.set(xlim=(100 * strikes[0], 100 * strikes[-1]), ylim=(5, top), xlabel="strike (% of forward), 6 months")
    axes[0].set_ylabel("implied volatility (%)")
    return fig


def figure_price_range(data):
    """The forward-start call under every posterior draw of the opening example: mark, 90% point and reserve."""
    key = key_of(HOOK)
    prices = percent(data[f"{key}/forward_start/draws"])
    mark, point90 = prices.mean(), np.quantile(prices, 0.90)
    least, truth = float(percent(data[f"{key}/forward_start/point"])), float(percent(data["truth/forward_start"]))
    fig, ax = canvas(0.90, HEIGHT)
    low, high = np.floor(np.quantile(prices, 0.001) * 5) / 5, np.ceil(np.quantile(prices, 0.999) * 5) / 5
    counts, _, _ = ax.hist(prices, bins=np.linspace(low, high, 49), color=POSTERIOR, alpha=0.5, lw=0, rwidth=0.88)
    top = counts.max()
    ax.plot([least, least], [0, 1.02 * top], color=CLASSICAL, lw=1.1)
    for x, dashes, side, shift, text in ((mark, "-", "right", -0.012, f"mark: the mean, {mark:.2f}"),
                                         (point90, DASHED, "left", 0.012, f"90% point, {point90:.2f}")):
        ax.plot([x, x], [0, 1.30 * top], color=INK, lw=1.1, ls=dashes)
        ax.text(x + shift, 1.20 * top, text, ha=side, va="bottom", fontsize=7.2, color=INK_2)
    ax.annotate("", xy=(point90, 1.10 * top), xytext=(mark, 1.10 * top),
                arrowprops={"arrowstyle": "<->", "color": INK, "lw": 0.8, "shrinkA": 0, "shrinkB": 0})
    ax.text(0.5 * (mark + point90), 1.13 * top, f"reserve {reserve(mark, point90):.2f}", ha="center", va="bottom",
            fontsize=7.2, color=INK)
    ax.plot([truth], [0], marker="^", color=INK, ms=4.5, clip_on=False, zorder=6)
    ax.legend([key_line(CLASSICAL, lw=1.1), key_line(INK, marker="^", ms=4.5, lw=0)],
              [f"least squares: {least:.2f}", f"under the generating parameters: {truth:.2f}"],
              loc="upper right", bbox_to_anchor=(1.0, 0.66), handletextpad=0.4, borderaxespad=0.0, handlelength=1.2)
    ax.set(xlim=(low, high), ylim=(0, 1.36 * top), yticks=[])
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("price of the forward-start call, per 100 of forward: one value per parameter set")
    return fig


def sampler_path(quotes, draws, steps=600, seed=7):
    """The first steps of a random-walk Metropolis chain on the posterior, as parameter sets, from a far start."""
    rng = np.random.default_rng(seed)
    box = lambda theta: np.log((theta - il.LOWER) / (il.UPPER - theta))  # noqa: E731
    factor = np.linalg.cholesky(0.5 * np.cov(box(draws).T))
    u = box(np.array([0.09, 2.0, 0.1024, 1.0, -0.3]))[None, :]  # a guess: volatility 30%, staying there; a mild skew
    value = il.log_posterior(quotes, u)[0]
    path = [u[0]]
    for _ in range(steps):
        proposal = u + rng.standard_normal(u.shape) @ factor.T
        new_value = il.log_posterior(quotes, proposal)[0]
        if np.log(rng.uniform()) < new_value[0] - value[0]:
            u, value = proposal, new_value
        path.append(u[0])
    return il.to_theta(np.array(path))


def figure_posterior(data, step):
    """The posterior of the opening example in two planes, built in three steps.

    Step 1: a sampler's first steps, from a far start to where the posterior is. Step 2: the posterior draws. Step 3:
    least squares and its quadratic region, and the generating parameters.
    """
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
    path = sampler_path(il.readings()[HOOK], draws[:, :5]) if step == 1 else None
    for ax, (i, j, x_scale, y_scale, title, xlabel, ylabel, xlim, ylim) in zip(axes, planes, strict=True):
        if step == 1:
            ax.plot(x_scale(path[:, i]), y_scale(path[:, j]), "-o", color=POSTERIOR, lw=0.6, ms=1.8, alpha=0.8,
                    zorder=2)
            ax.plot(x_scale(path[0, i]), y_scale(path[0, j]), "o", mfc="white", mec=POSTERIOR, ms=4.5, mew=1.0,
                    zorder=3)
        else:
            ax.scatter(x_scale(draws[:, i]), y_scale(draws[:, j]), s=2.0, color=POSTERIOR, alpha=0.16, lw=0, zorder=2)
        if step == 3:
            outline = il.gauss_newton_ellipse(result, i, j)
            ax.plot(x_scale(outline[:, 0]), y_scale(outline[:, 1]), color=CLASSICAL, lw=1.0, zorder=3)
            ax.plot(x_scale(result.theta[i]), y_scale(result.theta[j]), "o", color=CLASSICAL, ms=4.2, mec="white",
                    mew=0.7, zorder=4)
            ax.plot(x_scale(il.TRUTH[i]), y_scale(il.TRUTH[j]), "+", color=INK, ms=6.5, mew=1.1, zorder=5)
        ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
        if step > 1:
            ax.set_title(title, loc="left")
    if step == 1:
        handles = [key_line(POSTERIOR, marker="o", mfc="white", ms=4.5, mew=1.0, lw=0),
                   key_line(POSTERIOR, marker="o", ms=1.8, lw=0.6)]
        labels = ["a starting guess", f"a sampler's first {len(path) - 1} steps"]
    else:
        handles, labels = [key_line(POSTERIOR, marker="o", ms=3.0, lw=0)], [f"{len(draws):,} posterior draws"]
    if step == 3:
        handles += [key_line(CLASSICAL, marker="o", ms=3.6, lw=1.0), key_line(INK, marker="+", ms=6, mew=1.1, lw=0)]
        labels += ["least squares and its quadratic 90% region", "generating parameters"]
    legend_above(fig, handles, labels)
    return fig


def figure_summary(data):
    """The talk in three pictures: a posterior, the prices it gives a product, and what an exact posterior costs."""
    key = key_of(HOOK)
    draws, prices = data[f"{key}/draws"], percent(data[f"{key}/forward_start/draws"])
    cost = costs()
    fig, axes = canvas(0.97, HEIGHT, ncols=3)
    axes[0].scatter(draws[:, 1], 100 * np.sqrt(draws[:, 2]), s=2.0, color=POSTERIOR, alpha=0.16, lw=0)
    axes[0].set(xlim=(0.1, 10.0), ylim=(10.0, 40.0), xlabel="mean reversion", ylabel="long-run volatility (%)")
    axes[0].set_title("a distribution", loc="left")
    axes[1].hist(prices, bins=40, color=POSTERIOR, alpha=0.5, lw=0, rwidth=0.88)
    for x, dashes in ((prices.mean(), "-"), (np.quantile(prices, 0.9), DASHED)):
        axes[1].axvline(x, color=INK, lw=1.0, ls=dashes)
    axes[1].set(yticks=[], xlabel="price of an exotic")
    axes[1].spines["left"].set_visible(False)
    axes[1].grid(False)
    axes[1].set_title("a range for a price", loc="left")
    rows = (("exact,\nHeston", [60 * x for x in cost["heston_minutes"]], INK),
            ("exact,\nrough Bergomi", [86400 * x for x in cost["rough_mcmc_days"]], MUTED),
            ("amortized", [cost["network_seconds"]] * 2, POSTERIOR))
    for y, (name, span, colour) in enumerate(rows):
        axes[2].plot(span, [y, y], color=colour, lw=5.0, solid_capstyle="round")
        axes[2].plot(span[:1], [y], "o", color=colour, ms=5.0)
    axes[2].set_xscale("log")
    axes[2].set_xticks([1, 3600, 86400 * 365], ["1 s", "1 hour", "1 year"])
    axes[2].minorticks_off()
    axes[2].set_yticks(range(len(rows)), [name for name, _, _ in rows])
    axes[2].set(xlim=(0.05, 86400 * 365 * 8), ylim=(len(rows) - 0.4, -0.6), xlabel="CPU time for one posterior")
    axes[2].tick_params(axis="y", length=0)
    axes[2].grid(axis="y", visible=False)
    axes[2].set_title("a cost, paid once", loc="left")
    return fig


def figure_ranges_by_quotes(data):
    """The forward start's 90% range, classical and posterior, as fewer and fewer expiries are quoted."""
    rows = list(il.READINGS)[::-1]
    fig, ax = canvas(0.94, HEIGHT)
    for y, reach in enumerate(rows):
        key = key_of(reach)
        point = float(percent(data[f"{key}/forward_start/point"]))
        low, high = percent(data[f"{key}/forward_start/delta"])
        if np.isfinite(low):
            whisker(ax, y - 0.2, low, high, point, CLASSICAL)
        else:
            ax.plot([point], [y - 0.24], "o", mfc="white", mec=CLASSICAL, ms=4.2, mew=1.0, zorder=4)
            ax.text(point + 0.07, y - 0.25, "no range: the fit is on a bound", va="center", fontsize=7.0,
                    color=INK_2)
        q = percent(quantiles(data[f"{key}/forward_start/draws"]))
        whisker(ax, y + 0.2, q[0], q[2], q[1], POSTERIOR)
    ax.axvline(float(percent(data["truth/forward_start"])), color=INK, lw=0.9, ls=DASHED, zorder=1)
    labels = [f"{READING_LABELS[reach]} ({int(data[f'{key_of(reach)}/quotes'])})" for reach in rows]
    rows_axis(ax, range(len(rows)), labels, "price of the forward-start call, per 100 of forward")
    ax.set_xlim(right=6.95)
    legend_above(fig, range_handles(), RANGE_LEGEND)
    return fig


def figure_assumption(data):
    """Quotes at one expiry: the forward start under each assumed value of today's volatility, and under a prior."""
    key = key_of(ONE)
    draws, prices = data[f"{key}/draws"], percent(data[f"{key}/forward_start/draws"])
    fig, ax = canvas(0.94, HEIGHT)
    ax.scatter(100 * np.sqrt(draws[:, 0]), prices, s=2.0, color=POSTERIOR, alpha=0.16, lw=0, zorder=2)
    ax.plot(100 * np.array(ASSUMED_VOLATILITY), percent(data["assumed/forward_start"]), "-o", color=CLASSICAL, lw=1.3,
            ms=3.8, mec="white", mew=0.7, zorder=4)
    ax.axhline(float(percent(data["truth/forward_start"])), color=INK, lw=0.9, ls=DASHED, zorder=1)
    q = quantiles(prices)
    ax.plot([42.3, 42.3], [q[0], q[2]], color=POSTERIOR, lw=2.0, solid_capstyle="round", zorder=3, clip_on=False)
    ax.plot([42.3], [q[1]], "o", color=POSTERIOR, ms=4.6, mec="white", mew=0.8, zorder=4, clip_on=False)
    ax.text(10.0, 2.05, f"every point on the line fits the quotes equally: a misfit of "
            f"{float(np.max(data['assumed/misfit'])):.2f} half-spreads", fontsize=7.0, color=INK_2, va="bottom")
    ax.set(xlim=(9.0, 43.5), xlabel="today's volatility (%)", ylabel="forward start, per 100")
    ax.set_xticks([10, 15, 20, 25, 30, 35, 40])
    handles = [key_line(CLASSICAL, marker="o", ms=3.6, lw=1.3), key_line(POSTERIOR, marker="o", ms=3.0, lw=0),
               key_line(POSTERIOR, marker="o", ms=3.6, lw=2.0), key_line(INK, lw=0.9, ls=DASHED)]
    legend_above(fig, handles, ["best fit, today's volatility assumed", "posterior draws", "their 90% range",
                                "generating price"])
    return fig


def figure_joint_draws(data):
    """Two forward starts priced under the same posterior draws, and a book that holds one against the other."""
    key = key_of(HOOK)
    first, second = percent(data[f"{key}/forward_start/draws"]), percent(data[f"{key}/late_start/draws"])
    fig, axes = canvas(0.94, HEIGHT, ncols=2, gridspec_kw={"width_ratios": [1.0, 1.3]})
    axes[0].scatter(first, second, s=2.0, color=POSTERIOR, alpha=0.16, lw=0)
    axes[0].set(xlabel="forward start struck in 6 months", ylabel="struck in 9 months")
    axes[0].set_title("one dot: one posterior draw, two prices", loc="left")
    rows = {"struck in 6 months": first, "struck in 9 months": second, "book: long one, short the other": first - second}
    widest = 0.0
    for y, values in enumerate(rows.values()):
        q = quantiles(values)
        whisker(axes[1], y, q[0] - q[1], q[2] - q[1], 0.0, POSTERIOR)
        axes[1].text(q[2] - q[1] + 0.025, y, f"{q[2] - q[0]:.2f} wide", va="center", fontsize=7.2, color=INK_2)
        widest = max(widest, q[2] - q[1])
    axes[1].axhline(1.5, color=BASELINE, lw=0.6, zorder=1)
    rows_axis(axes[1], range(len(rows)), list(rows), "90% range around the median, per 100")
    axes[1].set_xlim(right=widest + 0.22)
    axes[1].set_title("the book's range is not the sum", loc="left")
    return fig


# ---------------------------------------------------------------------------------------------------------------
# The results


def figure_posterior_cost():
    """CPU time for one posterior on the 63 SPX quotes: exact methods, measured or derived, and the trained network."""
    cost = costs()
    rows = (  # name, method, range in seconds, value, status, colour, measured
        ("Heston", "Fourier pricing, MCMC", [60 * x for x in cost["heston_minutes"]],
         "{:.0f}–{:.0f} CPU-minutes".format(*cost["heston_minutes"]), "measured", INK, True),
        ("Rough Bergomi", "Monte Carlo pricing, importance sampling", [3600 * x for x in cost["rough_sampling_hours"]],
         "{:.0f}–{:.0f} CPU-hours".format(*cost["rough_sampling_hours"]), "estimated", INK, False),
        ("Rough Bergomi", "Monte Carlo pricing, MCMC", [86400 * x for x in cost["rough_mcmc_days"]],
         "100–180 CPU-days", "estimated, not run", INK, False),
        ("Heston", "network trained in advance", [cost["network_seconds"]] * 2,
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
    ax.set_xticks([1, 60, 3600, 86400, 86400 * 365], ["1 s", "1 min", "1 hour", "1 day", "1 year"])
    ax.minorticks_off()
    ax.set_yticks([])
    ax.set_ylim(3.85, -0.6)
    ax.grid(axis="y", visible=False)
    ax.axhline(2.62, color=BASELINE, lw=0.6)
    ax.set_xlabel("CPU time for one posterior on the 63 SPX quotes: our own runs, orders of magnitude (log scale)")
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
    first = percent(data[f"{key}/forward_start/draws"])
    mark, point90 = first.mean(), np.quantile(first, 0.9)
    assumed = percent(data["assumed/forward_start"])
    quoted = {
        "the least-squares price": f"{percent(data[f'{key}/forward_start/point']):.2f}",
        "the mark": f"{mark:.2f}", "the 90% point": f"{point90:.2f}", "the reserve": f"{reserve(mark, point90):.2f}",
        "the reserve as a share of the mark": f"{100 * reserve(mark, point90) / mark:.0f}",
        "the generating price": f"{percent(data['truth/forward_start']):.2f}",
        "one expiry: the best fit's price with today's volatility assumed at its lowest": f"{assumed[0]:.2f}",
        "one expiry: the best fit's price with today's volatility assumed at its highest": f"{assumed[-1]:.2f}",
    }
    for reach in il.READINGS:
        k = key_of(reach)
        low, high = percent(data[f"{k}/forward_start/delta"])
        q = percent(quantiles(data[f"{k}/forward_start/draws"]))
        quoted[f"the least-squares price, quotes at {reach}"] = f"{percent(data[f'{k}/forward_start/point']):.2f}"
        if np.isfinite(low):
            quoted[f"the posterior range, quotes at {reach}"] = f"{q[0]:.2f} to {q[2]:.2f}"
            quoted[f"the delta-method range, quotes at {reach}"] = f"{low:.2f} to {high:.2f}"
        else:  # a wide, prior-driven range, known less precisely: one decimal
            quoted[f"the posterior range, quotes at {reach}"] = f"{q[0]:.1f} to {q[2]:.1f}"
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
    first, second = percent(data[f"{key}/forward_start/draws"]), percent(data[f"{key}/late_start/draws"])
    print(f"\nrunning example, seed {il.SEED}: (v0, kappa, long-run variance, vol of vol, rho) = {il.TRUTH.tolist()}")
    print(f"under the generating parameters, per 100 of forward: forward-start call "
          f"{percent(data['truth/forward_start']):.2f} (6 months to 1 year), {percent(data['truth/late_start']):.2f} "
          f"(9 months to 1 year); the quoted 1-year at-the-money call {percent(data['truth/call']):.2f}")
    for reach in il.READINGS:
        k = key_of(reach)
        low, high = percent(data[f"{k}/forward_start/delta"])
        q = percent(quantiles(data[f"{k}/forward_start/draws"]))
        bounds = ", ".join(data[f"{k}/on_bound"])
        classical = f"{low:.2f} to {high:.2f}" if np.isfinite(low) else f"none (the fit is on a bound: {bounds})"
        print(f"  quotes at {reach:19s} ({int(data[f'{k}/quotes']):2d} quotes): forward start by least squares "
              f"{percent(data[f'{k}/forward_start/point']):.2f}, error scale {float(data[f'{k}/scale']):.2f} on "
              f"{int(data[f'{k}/dof'])} degrees of freedom, delta method {classical}; "
              f"posterior {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}; effective sample size {float(data[f'{k}/ess']):.0f} "
              f"of {int(data[f'{k}/proposals'])}, failed pricings {float(data[f'{k}/failed']):.2%}, acceptance of "
              f"the moves {float(data[f'{k}/acceptance']):.2f}, pricing gate {float(data[f'{k}/gate']):.1e}")
        draws = data[f"{k}/draws"]
        call = percent(quantiles(data[f"{k}/call/draws"]))
        print(f"      the quoted 1-year at-the-money call: posterior {call[0]:.2f} / {call[1]:.2f} / {call[2]:.2f}; "
              f"error scale c {np.quantile(draws[:, 5], 0.05):.2f} / {np.median(draws[:, 5]):.2f} / "
              f"{np.quantile(draws[:, 5], 0.95):.2f}; kappa {np.quantile(draws[:, 1], 0.05):.1f} to "
              f"{np.quantile(draws[:, 1], 0.95):.1f}; vol of vol {np.quantile(draws[:, 3], 0.05):.2f} to "
              f"{np.quantile(draws[:, 3], 0.95):.2f}; today's volatility "
              f"{100 * np.sqrt(np.quantile(draws[:, 0], 0.05)):.1f} to "
              f"{100 * np.sqrt(np.quantile(draws[:, 0], 0.95)):.1f}%")
    low, high = np.quantile(data["hook/misfit"], [0.05, 0.95])
    print(f"  opening example: root-mean-square misfit in half-spreads, fit {float(data['hook/misfit_fit']):.2f}, "
          f"posterior draws {low:.2f} to {high:.2f} (5% to 95%)")
    design = percent(il.half_spread_volatility(np.array(il.DELTAS)))
    print(f"  the chain's half-spreads, as designed: {design.min():.2f} to {design.max():.2f} volatility points")
    mark, point90 = first.mean(), np.quantile(first, 0.9)
    print(f"  opening example, forward-start call: least squares {percent(data[f'{key}/forward_start/point']):.2f}, "
          f"the {FAN} fits drawn {first[:FAN].min():.2f} to {first[:FAN].max():.2f}, mark {mark:.2f}, 90% point "
          f"{point90:.2f}, reserve {reserve(mark, point90):.2f} ({100 * reserve(mark, point90) / mark:.0f}% of the mark)")
    q_first, q_second, q_book = quantiles(first), quantiles(second), quantiles(first - second)
    print(f"  opening example, 90% widths: forward start {q_first[2] - q_first[0]:.2f}, later forward start "
          f"{q_second[2] - q_second[0]:.2f}, the first minus the second {q_book[2] - q_book[0]:.2f}; correlation of "
          f"the two prices {np.corrcoef(first, second)[0, 1]:.3f}")
    print("  one expiry, today's volatility assumed at " + ", ".join(f"{100 * v:g}%" for v in ASSUMED_VOLATILITY))
    print("      forward start of the best fit: " + ", ".join(f"{x:.2f}" for x in percent(data["assumed/forward_start"])))
    print("      misfit in half-spreads: " + ", ".join(f"{x:.2f}" for x in data["assumed/misfit"])
          + f"; fits on a bound: {int(np.sum(data['assumed/bounds'] > 0))}")
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
        save(figure_prior_likelihood(data), "prior-likelihood")
        for step in (1, 2, 3):
            save(figure_posterior(data, step), f"posterior-{step}")
        save(figure_price_range(data), "price-range")
        save(figure_ranges_by_quotes(data), "ranges-by-quotes")
        save(figure_assumption(data), "assumption")
        save(figure_joint_draws(data), "joint-draws")
        save(figure_posterior_cost(), "posterior-cost")
        save(figure_coverage(), "coverage")
        save(figure_summary(data), "summary")
    print(f"12 figures in {OUT}")
    report(data)
    stale = stale_numbers(data)
    print("\nslot1.tex quotes these numbers as printed above." if not stale else
          "\nslot1.tex does not quote these numbers as printed above; update the slides and the curriculum page:")
    for name, number in stale.items():
        print(f"  {name}: {number}")


if __name__ == "__main__":
    main()
