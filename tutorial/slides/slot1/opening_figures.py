"""Draw the figures of the opening slides: one option smile, and a model with two unknowns.

Run from tutorial/:

    python slides/slot1/opening_figures.py [--rubik PATH/TO/Rubik-Variable.ttf]

The opening slides make the case for Bayes on the smallest calibration that shows it: one simulated Heston smile at
one month, seven out-of-the-money quotes on the thin chain's strikes, with its half-spreads and its error law
(`illustration.py`). Two parameters are unknown, the vol of vol and the correlation; the other three are known. The
vol of vol is low (0.3), so the smile fixes little more than the product of the two unknowns, and the posterior is
long and bent. With the notebook's market-like vol of vol (1.8) the quotes pin both, and nothing is left to see.

With two unknowns the posterior is computed exactly, on a grid: uniform priors on a box, and normal quote errors of
one half-spread. The classical route is least squares with the Gauss-Newton covariance, for the same known error
size. The script ends by printing every number the opening slides quote. These are illustrations, not results.
"""

import argparse
import re
import tempfile
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
from scipy.optimize import least_squares

import export_figures as ef  # the deck's figure style and colours
import illustration as il
from sbi4f_tutorial.shared import surface

TRUTH = np.array([0.04, 2.0, 0.04, 0.30, -0.70])  # v0, kappa, long-run variance, vol of vol, correlation
MONTHS = 1
UNKNOWN = (3, 4)  # the vol of vol and the correlation
BOX = np.array([[0.10, 1.20], [-0.95, -0.10]])  # the uniform prior: vol of vol, then correlation
NAMES = ("vol of vol", "correlation")
GRID = 240
LEVEL = 0.90
Z = 1.6448536269514722  # the normal quantile of a central 90% interval
THRESHOLD = -0.5  # "how likely is a correlation below -0.5?"
PAIR_A, PAIR_B = ef.HOLDS, ef.CAVEAT  # two parameter pairs at the two ends of the posterior
HEIGHT_BAYES, HEIGHT_READ = 1.40, 1.18  # inches: sized to leave room for the row of cards under each figure


def smile_quotes():
    """Seven quotes at one month, observed once at TRUTH with errors of one half-spread (the thin chain's law)."""
    maturity = np.full(len(il.DELTAS), MONTHS / 12.0)
    delta = np.array(il.DELTAS)
    strike = np.exp(delta * il.REFERENCE_VOLATILITY * np.sqrt(maturity))
    put = strike <= 1.0
    clean = il.prices(TRUTH, strike, maturity, put, il.CHECK_RUNG)
    implied = surface.implied_vol(clean, np.log(strike), maturity, np.where(put, "put", "call"))
    half_spread = surface.bs_vega(np.log(strike), maturity, implied) * il.half_spread_volatility(delta)
    noise = np.random.default_rng(il.SEED).standard_normal(len(strike))
    return il.Quotes(strike, maturity, put, half_spread, clean + half_spread * noise, "one smile")


def theta(pairs):
    """Full Heston parameter sets from (vol of vol, correlation) pairs, the other parameters at TRUTH."""
    pairs = np.atleast_2d(pairs)
    thetas = np.tile(TRUTH, (len(pairs), 1))
    thetas[:, UNKNOWN] = pairs
    return thetas


def misses(quotes, pairs):
    return quotes.misses(theta(pairs))


def central(weights, values, levels=(0.05, 0.5, 0.95)):
    """Quantiles of `values` under `weights`."""
    order = np.argsort(values.ravel())
    cumulative = np.cumsum(weights.ravel()[order])
    return np.array([values.ravel()[order][np.searchsorted(cumulative, level)] for level in levels])


def compute():
    quotes = smile_quotes()
    xs, ys = (np.linspace(low, high, GRID) for low, high in BOX)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pairs = np.column_stack([X.ravel(), Y.ravel()])
    r = misses(quotes, pairs)
    log_posterior = (-0.5 * np.sum(r * r, axis=1)).reshape(X.shape)
    posterior = np.exp(log_posterior - log_posterior.max())
    posterior /= posterior.sum()

    # The classical route: least squares from the grid's best pair, and the Gauss-Newton covariance (J'J)^-1.
    start = pairs[np.argmax(posterior)]
    fit = least_squares(lambda p: misses(quotes, p)[0], start, bounds=(BOX[:, 0], BOX[:, 1]), x_scale=[0.1, 0.1])
    step = 1e-4
    jacobian = np.column_stack([(misses(quotes, fit.x + step * e)[0] - misses(quotes, fit.x - step * e)[0]) / (2 * step)
                                for e in np.eye(2)])
    covariance = np.linalg.inv(jacobian.T @ jacobian)
    on_bound = bool(np.any(np.isclose(fit.x, BOX[:, 0], atol=1e-6) | np.isclose(fit.x, BOX[:, 1], atol=1e-6)))

    # The 90% region of highest posterior density, and two pairs near its two ends, on the ridge.
    ordered = np.sort(posterior.ravel())[::-1]
    cut = ordered[np.searchsorted(np.cumsum(ordered), LEVEL)]
    region = posterior >= cut
    columns = np.flatnonzero(region.any(axis=1))
    ends = [columns[int(0.22 * len(columns))], columns[int(0.90 * len(columns))]]
    a, b = (np.array([xs[i], ys[np.argmax(posterior[i])]]) for i in ends)

    strikes = np.linspace(quotes.strike[0] * 0.996, quotes.strike[-1] * 1.004, 61)
    inside = np.einsum("...i,ij,...j->...", np.stack([X - fit.x[0], Y - fit.x[1]], -1), np.linalg.inv(covariance),
                       np.stack([X - fit.x[0], Y - fit.x[1]], -1)) <= -2.0 * np.log(1.0 - LEVEL)
    return {
        "quotes": quotes, "xs": xs, "ys": ys, "posterior": posterior, "region": region,
        "fit": fit.x, "covariance": covariance, "on_bound": on_bound,
        "classical": np.column_stack([fit.x - Z * np.sqrt(np.diag(covariance)), fit.x + Z * np.sqrt(np.diag(covariance))]),
        "quantiles": np.array([central(posterior, X), central(posterior, Y)]),
        "product": central(posterior, X * Y),
        "below": float(posterior[Y < THRESHOLD].sum()),
        "mass_in_ellipse": float(posterior[inside].sum()),
        "a": a, "b": b, "strikes": strikes,
        "smile_a": il.smile(theta(a)[0], MONTHS / 12.0, strikes), "smile_b": il.smile(theta(b)[0], MONTHS / 12.0, strikes),
        "rms": {name: float(np.sqrt(np.mean(misses(quotes, p)[0] ** 2))) for name, p in (("a", a), ("b", b), ("fit", fit.x))},
    }


# ---------------------------------------------------------------------------------------------------------------
# Figures


def parameter_axes(ax, ylabel=True):
    ax.set(xlim=(0.04, 1.26), ylim=(-1.0, -0.05), xlabel=NAMES[0])
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax.set_yticks([-0.8, -0.6, -0.4, -0.2])
    if ylabel:
        ax.set_ylabel(NAMES[1])


def figure_opening_bayes(data):
    """The prior, the quotes and the posterior, side by side."""
    quotes, xs, ys = data["quotes"], data["xs"], data["ys"]
    fig, axes = ef.canvas(0.97, HEIGHT_BAYES, ncols=3)

    ax = axes[0]
    ax.fill_between(BOX[0], BOX[1, 0], BOX[1, 1], color=ef.POSTERIOR, alpha=0.16, lw=0)
    ax.plot(BOX[0, [0, 1, 1, 0, 0]], BOX[1, [0, 0, 1, 1, 0]], color=ef.POSTERIOR, lw=0.8)
    ax.text(BOX[0].mean(), BOX[1].mean(), "every pair\nequally plausible", ha="center", va="center", fontsize=7.5,
            color=ef.INK_2, linespacing=1.25)
    ax.set_title("1 · the prior", loc="left")
    parameter_axes(ax)

    ax = axes[1]
    mid, low, high = (ef.percent(v) for v in quotes.implied())
    for smile, colour, label in ((data["smile_a"], PAIR_A, "pair A"), (data["smile_b"], PAIR_B, "pair B")):
        ax.plot(100 * data["strikes"], ef.percent(smile), color=colour, lw=1.4, zorder=3, label=label)
    ax.errorbar(100 * quotes.strike, mid, yerr=[mid - low, high - mid], fmt="o", color=ef.INK, ms=2.8, lw=0.9, capsize=1.8,
                capthick=0.9, zorder=5)
    ax.set(xlabel="strike (% of forward), 1 month", ylabel="implied volatility (%)", ylim=(16.5, 26.5))
    ax.set_title("2 · the quotes, with their bid–ask", loc="left")
    ax.legend(loc="upper right", handlelength=1.3, borderaxespad=0.1, labelspacing=0.25)

    ax = axes[2]
    shades = mcolors.LinearSegmentedColormap.from_list("posterior", ["#ffffff", ef.POSTERIOR])
    ax.contourf(xs, ys, data["posterior"].T, levels=np.linspace(0.04, 1.0, 9) * data["posterior"].max(), cmap=shades,
                extend="max")
    ax.contour(xs, ys, data["region"].T.astype(float), levels=[0.5], colors=[ef.INK], linewidths=0.9)
    for pair, colour, label in ((data["a"], PAIR_A, "A"), (data["b"], PAIR_B, "B")):
        ax.plot(*pair, "s", color=colour, ms=4.6, mec="white", mew=0.8, zorder=5)
        ax.annotate(label, pair, textcoords="offset points", xytext=(5, -2.5), color=colour, fontsize=8.0, fontweight=600)
    ax.set_title("3 · the posterior", loc="left")
    ax.text(1.22, -0.80, "the outline\nholds 90%", ha="right", va="center", fontsize=6.6, color=ef.MUTED, linespacing=1.2)
    parameter_axes(ax)
    return fig


def figure_opening_read(data):
    """The posterior of each unknown with its 90% range, against least squares and its 90% interval."""
    fig, axes = ef.canvas(0.97, HEIGHT_READ, ncols=2)
    grids = (data["xs"], data["ys"])
    for k, ax in zip((1, 0), axes, strict=True):  # the correlation first: the cards under the figure speak of it
        grid = grids[k]
        density = data["posterior"].sum(axis=1 - k)
        density = density / density.max()
        low, median, high = data["quantiles"][k]
        ax.plot(grid, density, color=ef.POSTERIOR, lw=1.4)
        keep = (grid >= low) & (grid <= high)
        ax.fill_between(grid[keep], 0.0, density[keep], color=ef.POSTERIOR, alpha=0.22, lw=0)
        rows = ((1.22, low, high, median, ef.POSTERIOR), (1.52, *data["classical"][k], data["fit"][k], ef.CLASSICAL))
        for y, a, b, centre, colour in rows:
            ef.whisker(ax, y, a, b, centre, colour)
            for value, side in ((a, "right"), (b, "left")):
                ax.text(value + (-0.012 if side == "right" else 0.012) * (grid[-1] - grid[0]), y, f"{value:.2f}".replace("-", "−"),
                        ha=side, va="center", fontsize=6.8, color=colour)
        ax.set(xlabel=NAMES[k], ylim=(0.0, 1.72), yticks=[])
        ax.grid(axis="y", visible=False)
    axes[0].set_xlim(-1.0, 0.0)
    axes[1].set_xlim(0.05, 1.0)
    axes[0].set_ylabel("plausibility")
    ef.legend_above(fig, [ef.key_line(ef.POSTERIOR, marker="o", ms=3.6, lw=2.0), ef.key_line(ef.CLASSICAL, marker="o", ms=3.6, lw=2.0)],
                    ["posterior: 90% range", "least squares: best fit and 90% interval"])
    return fig


def report(data):
    minus = lambda text: re.sub(r"-(?=\d)", "−", text)  # noqa: E731  (a minus sign before a number)
    print("\nThe opening example: one Heston smile at 1 month, 7 quotes; unknown vol of vol and correlation.")
    print(f"  generating pair: vol of vol {TRUTH[3]:.2f}, correlation {TRUTH[4]:.2f}")
    print(f"  least squares: vol of vol {data['fit'][0]:.2f}, correlation {data['fit'][1]:.2f}"
          f"{' (on a bound)' if data['on_bound'] else ''}; misses by {data['rms']['fit']:.2f} half-spreads (rms)")
    for k, name in enumerate(NAMES):
        low, median, high = data["quantiles"][k]
        print(minus(f"  {name}: posterior 90% {low:.2f} to {high:.2f} (median {median:.2f}); "
                    f"least squares 90% {data['classical'][k, 0]:.2f} to {data['classical'][k, 1]:.2f}"))
    print(minus(f"  product: posterior 90% {data['product'][0]:.3f} to {data['product'][2]:.3f}; generating {TRUTH[3] * TRUTH[4]:.3f}"))
    print(minus(f"  probability of a correlation below {THRESHOLD}: {100 * data['below']:.0f}%"))
    print(f"  posterior probability inside the least-squares 90% ellipse: {100 * data['mass_in_ellipse']:.0f}%")
    for name in ("a", "b"):
        print(minus(f"  pair {name.upper()}: vol of vol {data[name][0]:.2f}, correlation {data[name][1]:.2f}; "
                    f"misses by {data['rms'][name]:.2f} half-spreads (rms)"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rubik", type=Path, help="Rubik's variable font, so that the figures use the slides' typeface")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as folder:
        families = ef.add_rubik(args.rubik, Path(folder)) if args.rubik else ["DejaVu Sans"]
        ef.style(families)
        data = compute()
        ef.save(figure_opening_bayes(data), "opening-bayes")
        ef.save(figure_opening_read(data), "opening-read")
    report(data)


if __name__ == "__main__":
    main()
