"""Draw the figures of the opening argument (slides 1 to 4), in the deck's figure style.

Run from tutorial/:

    python slides/slot1/intro_figures.py [--rubik PATH/TO/Rubik-Variable.ttf]

The opening argument uses one small example on 20 years of US returns (`intro_example.py`, which says what is
computed and from which data):

- intro-weights: the weights of a portfolio of ten industries, from the sample averages, before and after one average
  is raised by 10% (slide 1);
- intro-horizon: the share in stocks of a buy-and-hold investor against the horizon, with the estimates taken as
  exact and with their uncertainty included (slide 2);
- intro-posterior: the prior, the data and the posterior of one expected return (slide 3);
- intro-prior: the same portfolio weight as the prior goes from no weight to certainty (slide 4).

The script ends by printing every number these slides quote, and says which of them `slot1.tex` does not contain.
These are illustrations on real returns, not results of the tutorial. With --rubik the figures use Rubik like the
slides (see `export_figures.py`). The PDFs land in slides/slot1/figures/slot1/.
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

import export_figures as ef  # the deck's figure style and colours
import intro_example as ex

HERE = Path(__file__).resolve().parent
plt = ef.plt
HORIZONS = np.arange(1, 21)  # years
RING = dict(marker="o", ms=3.9, mfc="white", mew=1.0, lw=0)  # a weight computed from the data
DOT = dict(marker="o", ms=4.6, mec="white", mew=0.8, lw=0)  # the same weight after the change
PALE = "#b4b4ff"  # the posterior's hue, lighter: the weight after the change, as a line


def percent_label(value, _=None):
    return f"{value:.0f}%".replace("-", "−")


def points(move):
    return f"{move:+.0f} points".replace("-", "−")


def annual(monthly):
    """A monthly return as percent a year."""
    return 1200.0 * monthly


def change_handles(what, changed):
    return ([ef.key_line(ef.INK_2, **RING), ef.key_line(ef.INK_2, **{**DOT, "mec": ef.INK_2})],
            [what, f"after raising {changed} average return by {100 * ex.CHANGE:.0f}%"])


def dumbbell(ax, before, after, at, colour, vertical=False):
    """A weight before (ring) and after (dot) the change, joined by a bar."""
    xs, ys = ([at, at], [before, after]) if vertical else ([before, after], [at, at])
    ax.plot(xs, ys, color=colour, lw=2.0, solid_capstyle="round", zorder=3)
    ax.plot(xs[:1], ys[:1], color=colour, zorder=4, **RING)
    ax.plot(xs[1:], ys[1:], color=colour, zorder=5, **DOT)


# ---------------------------------------------------------------------------------------------------------------
# Numbers


def compute(inputs):
    k = inputs.changed
    suggested = ex.suggested_weight(inputs.mean, inputs.covariance, inputs.months)
    grid = np.linspace(0.0, 1.0, 101)
    exact, uncertain = ex.shares(inputs, HORIZONS)
    return {
        "k": k, "suggested": suggested,
        "plugin": (ex.weights(inputs, 0.0), ex.weights(inputs, 0.0, raised=k)),
        "largest": np.array([np.abs(ex.weights(inputs, 0.0, raised=j) - ex.weights(inputs, 0.0)).max()
                             for j in range(len(inputs.codes))]),
        "grid": grid,
        "before": np.array([ex.weights(inputs, w)[k] for w in grid]),
        "after": np.array([ex.weights(inputs, w, raised=k)[k] for w in grid]),
        "at_suggested": (ex.weights(inputs, suggested), ex.weights(inputs, suggested, raised=k)),
        "certain": ex.weights(inputs, 1.0),
        "parameter": ex.one_parameter(inputs, suggested),
        "exact": exact, "uncertain": uncertain,
    }


# ---------------------------------------------------------------------------------------------------------------
# Figures


def figure_weights(inputs, data):
    """Slide 1: each industry's weight from the sample averages, before and after one average is raised."""
    before, after = (100.0 * w for w in data["plugin"])
    order = np.argsort(-before)  # rows from the largest weight down
    fig, ax = ef.canvas(0.97, ef.HEIGHT)
    ax.axvline(0.0, color=ef.BASELINE, lw=0.8, zorder=1)
    for y, i in enumerate(order):
        dumbbell(ax, before[i], after[i], y, ef.CLASSICAL)
    for i in np.argsort(-np.abs(after - before))[:2]:  # the two weights that move most
        y = int(np.where(order == i)[0][0])
        ax.text(max(before[i], after[i]) + 3.5, y, points(after[i] - before[i]), va="center", ha="left", fontsize=7.2,
                color=ef.INK, fontweight=500)
    ax.set_yticks(np.arange(len(order)), [inputs.names[i] for i in order])
    ax.set_ylim(len(order) - 0.45, -0.55)
    for label, i in zip(ax.get_yticklabels(), order):
        label.set_fontsize(6.8)
        if i == data["k"]:
            label.set_color(ef.INK)
            label.set_fontweight(600)
    ax.set_xlim(-76.0, 126.0)
    ax.set_xticks([-50, 0, 50, 100])
    ax.xaxis.set_major_formatter(percent_label)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    handles, labels = change_handles(f"portfolio weights from {ex.YEARS} years of monthly returns",
                                     "technology's")
    fig.legend(handles, labels, loc="outside upper left", ncols=2, handlelength=1.0, columnspacing=1.6,
               handletextpad=0.4, borderaxespad=0.0, borderpad=0.15)
    fig.get_layout_engine().set(h_pad=0.02)
    return fig


def figure_horizon(data):
    """Slide 2: the share in stocks against the horizon, the estimates taken as exact and with their uncertainty."""
    fig, ax = ef.canvas(0.80, ef.HEIGHT)
    for series, colour in ((data["exact"], ef.CLASSICAL), (data["uncertain"], ef.POSTERIOR)):
        ax.plot(HORIZONS, 100.0 * series, color=colour, lw=1.6, zorder=3)
        ax.plot(HORIZONS[-1:], 100.0 * series[-1:], color=colour, zorder=4, clip_on=False, **DOT)
        ax.text(HORIZONS[-1] + 0.45, 100.0 * series[-1], f"{100.0 * series[-1]:.0f}%", va="center", ha="left",
                fontsize=7.5, color=ef.INK, fontweight=500)
    ax.set_xlim(1.0, 20.0)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(percent_label)
    ax.set_xlabel("investment horizon, in years")
    ax.set_ylabel("share held in stocks")
    ax.grid(axis="x", visible=False)
    ax.legend([ef.key_line(ef.CLASSICAL, lw=1.6), ef.key_line(ef.POSTERIOR, lw=1.6)],
              ["estimated mean and volatility taken as exact", "their uncertainty included"],
              loc="lower left", handlelength=1.6, handletextpad=0.5, borderaxespad=0.1, labelspacing=0.2,
              borderpad=0.1)  # two lines that fit under the 25% gridline
    return fig


def figure_posterior(data):
    """Slide 3: what is believed about one expected return before the data, what the data say, and the posterior."""
    curves = {name: (annual(mean), annual(sd)) for name, (mean, sd) in data["parameter"].items()}
    x = np.linspace(-4.0, 30.0, 681)
    fig, ax = ef.canvas(0.97, ef.HEIGHT)
    mean, sd = curves["posterior"]
    low, high = stats.norm.ppf([0.05, 0.95], mean, sd)
    inside = (x >= low) & (x <= high)
    ax.fill_between(x, stats.norm.pdf(x, mean, sd), color=ef.POSTERIOR, alpha=0.10, lw=0)
    ax.fill_between(x[inside], stats.norm.pdf(x[inside], mean, sd), color=ef.POSTERIOR, alpha=0.22, lw=0)
    colours = {"prior": ef.MUTED, "data": ef.CLASSICAL, "posterior": ef.POSTERIOR}
    for name in ("prior", "data", "posterior"):
        shown = np.abs(x - curves[name][0]) <= 3.3 * curves[name][1]  # a curve stops where it reaches the baseline
        ax.plot(x[shown], stats.norm.pdf(x[shown], *curves[name]), color=colours[name],
                lw=1.6 if name == "posterior" else 1.3, zorder=3 if name == "posterior" else 2)
    top = stats.norm.pdf(mean, mean, sd)
    ax.text(21.3, 0.80 * top, f"with 90% probability, between\n{low:.1f}% and {high:.1f}% a year",
            ha="left", va="center", fontsize=7.2, color=ef.INK_2, linespacing=1.3)
    ax.plot([19.6, 20.8], [0.80 * top] * 2, color=ef.POSTERIOR, alpha=0.32, lw=5.0, solid_capstyle="butt")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0.0, 1.10 * top)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.xaxis.set_major_formatter(percent_label)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("expected return of technology in excess of bills, a year")
    words = {"prior": "prior", "data": f"{ex.YEARS} years of data", "posterior": "posterior"}
    ef.legend_above(fig, [ef.key_line(colours[name], lw=1.6) for name in words],
                    [f"{words[name]}: {curves[name][0]:.1f}% ± {curves[name][1]:.1f}" for name in words])
    return fig


def figure_prior(inputs, data):
    """Slide 4: technology's weight, and its move after the change, as the prior goes from no weight to certainty."""
    k, grid = data["k"], 100.0 * data["grid"]
    before, after = 100.0 * data["before"], 100.0 * data["after"]
    fig, ax = ef.canvas(0.97, ef.HEIGHT)
    ax.fill_between(grid, before, after, color=ef.POSTERIOR, alpha=0.13, lw=0, zorder=1)
    ax.plot(grid, after, color=PALE, lw=1.2, zorder=2)
    ax.plot(grid, before, color=ef.POSTERIOR, lw=1.6, zorder=2)
    suggested = 100.0 * data["suggested"]
    years = ex.prior_months(data["suggested"], inputs.months) / 12.0
    marks = ((0.0, data["plugin"], ef.CLASSICAL, "no prior"),
             (suggested, data["at_suggested"], ef.POSTERIOR, f"prior worth {years:.0f} years of data"),
             (100.0, (data["certain"], data["certain"]), ef.POSTERIOR, "prior taken as certain"))
    for at, (first, second), colour, name in marks:
        low, high = 100.0 * first[k], 100.0 * second[k]
        dumbbell(ax, low, high, at, colour, vertical=True)
        move = "no change" if round(high - low) == 0 else points(high - low)
        if at < 100.0:  # above the dot, and clear of the 100% gridline
            ax.text(at + 1.6, max(high + 3.0, 102.0) if high > 90.0 else high + 3.0, f"{name}: {move}", ha="left",
                    va="bottom", fontsize=7.2, color=ef.INK_2)
        else:  # under the two lines, where they meet
            ax.text(at - 1.6, 2.0, f"{name}: {move}", ha="right", va="bottom", fontsize=7.2, color=ef.INK_2)
    ax.set_xlim(-2.5, 102.5)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(percent_label)
    ax.set_ylim(0.0, 112.0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(percent_label)
    ax.set_xlabel("weight on the prior belief that the ten industries have the same expected return")
    ax.grid(axis="x", visible=False)
    handles, labels = change_handles("technology's weight in the portfolio", "its")
    fig.legend(handles, labels, loc="outside upper left", ncols=2, handlelength=1.0, columnspacing=1.6,
               handletextpad=0.4, borderaxespad=0.0, borderpad=0.15)
    fig.get_layout_engine().set(h_pad=0.02)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Report


def quoted_numbers(inputs, data):
    """The numbers the opening slides quote, as they should appear in slot1.tex."""
    k = data["k"]
    plug_before, plug_after = (100.0 * w for w in data["plugin"])
    held_before, held_after = (100.0 * w[k] for w in data["at_suggested"])
    retail = inputs.codes.index("Shops")
    curves = {name: (annual(mean), annual(sd)) for name, (mean, sd) in data["parameter"].items()}
    low, high = stats.norm.ppf([0.05, 0.95], *curves["posterior"])
    at = {years: int(np.where(HORIZONS == years)[0][0]) for years in (1, 10, 20)}
    market_error = annual(inputs.market_volatility) / np.sqrt(inputs.months)
    worth = ex.prior_months(data["suggested"], inputs.months) / 12.0
    return {
        "technology's average excess return, percent a year": f"{annual(inputs.mean[k]):.1f}\\%",
        "the same after the change": f"{annual(inputs.mean[k]) * (1 + ex.CHANGE):.1f}\\%",
        "its standard error": f"{annual(inputs.standard_error[k]):.1f}",
        "technology's plug-in weight before and after": f"{plug_before[k]:.0f}\\% to {plug_after[k]:.0f}\\%",
        "retail's plug-in weight before and after": f"{plug_before[retail]:.0f}\\% to {plug_after[retail]:.0f}\\%",
        "the plug-in move of technology's weight": f"{plug_after[k] - plug_before[k]:.0f} points",
        "the market's mean excess return and its standard error":
            f"{annual(inputs.market_mean):.1f}\\% a year $\\pm$ {market_error:.1f}",
        "the market's volatility": f"{100.0 * inputs.market_volatility * np.sqrt(12.0):.1f} percent",
        "the share in stocks, estimates taken as exact":
            f"{100.0 * data['exact'].min():.0f}\u2013{100.0 * data['exact'].max():.0f}\\%",
        "the share with their uncertainty, at 20 years": f"{100.0 * data['uncertain'][at[20]]:.0f}\\% at 20 years",
        "the same at 10 and 20 years, in the speaker notes":
            f"{100.0 * data['uncertain'][at[10]]:.0f} percent at 10 years and {100.0 * data['uncertain'][at[20]]:.0f} at 20",
        "the change in standard errors":
            f"{ex.CHANGE * inputs.mean[k] / inputs.standard_error[k]:.2f} standard errors",
        "the prior of technology's expected return": f"{curves['prior'][0]:.1f}\\% $\\pm$ {curves['prior'][1]:.1f}",
        "what the data say": f"{curves['data'][0]:.1f}\\% $\\pm$ {curves['data'][1]:.1f}",
        "the posterior": f"{curves['posterior'][0]:.1f}\\% $\\pm$ {curves['posterior'][1]:.1f}",
        "the weight on the prior that the data suggest": f"{100.0 * data['suggested']:.0f} percent",
        "the years of data that prior is worth": f"{worth:.0f} years",
        "technology's weight under that prior, before and after": f"{held_before:.0f}\\% to {held_after:.0f}\\%",
        "technology's weight with the prior taken as certain": f"{100.0 * data['certain'][k]:.0f}\\%",
        "the posterior's 90% interval, in the speaker notes": f"{low:.1f} and {high:.1f}",
    }


def report(inputs, data):
    """Everything behind the opening slides, printed so that slot1.tex and the README can be checked against it."""
    k = data["k"]
    before, after = (100.0 * w for w in data["plugin"])
    print(f"\nopening example: {inputs.months} months, {inputs.first_month} to {inputs.last_month}; maximum Sharpe "
          f"ratio portfolio of {len(inputs.codes)} industries, short positions allowed")
    print(f"  {inputs.names[k]}: average excess return {annual(inputs.mean[k]):.2f}% a year, standard error "
          f"{annual(inputs.standard_error[k]):.2f}; raised by {100 * ex.CHANGE:.0f}% to "
          f"{annual(inputs.mean[k]) * (1 + ex.CHANGE):.2f}%, a change of "
          f"{ex.CHANGE * inputs.mean[k] / inputs.standard_error[k]:.2f} standard errors")
    print("  plug-in weights, before and after, percent of the portfolio:")
    for i in np.argsort(-before):
        print(f"    {inputs.names[i]:14s} {before[i]:6.1f} {after[i]:6.1f}  ({after[i] - before[i]:+.1f})")
    print(f"  the largest plug-in move when each industry in turn is raised: {100 * data['largest'].min():.1f} to "
          f"{100 * data['largest'].max():.1f} points")
    print(f"  the weight on the prior that the data suggest: {data['suggested']:.3f}, a prior worth "
          f"{ex.prior_months(data['suggested'], inputs.months) / 12.0:.1f} years of data")
    for name, (first, second) in (("no prior", data["plugin"]), ("that prior", data["at_suggested"]),
                                  ("the prior taken as certain", (data["certain"], data["certain"]))):
        print(f"    {name}: {inputs.names[k]} {100 * first[k]:.1f}% to {100 * second[k]:.1f}%; all weights "
              f"{100 * first.min():.0f}% to {100 * first.max():.0f}%")
    raised = inputs.mean.copy()
    raised[k] *= 1.0 + ex.CHANGE
    again = ex.suggested_weight(raised, inputs.covariance, inputs.months)
    moved = ex.max_sharpe(*ex.predictive(raised, inputs.covariance, inputs.months, again)) - data["at_suggested"][0]
    print(f"    if the weight is estimated again after the change, it is {again:.3f} and {inputs.names[k]} moves by "
          f"{100 * moved[k]:.1f} points: the slides keep the prior's weight fixed")
    for name, weight in (("no prior", 0.0), ("that prior", data["suggested"])):
        first = ex.max_sharpe_long_only(*ex.predictive(inputs.mean, inputs.covariance, inputs.months, weight))
        second = ex.max_sharpe_long_only(*ex.predictive(raised, inputs.covariance, inputs.months, weight))
        print(f"    with no short positions, {name}: {int(np.sum(first > 0))} industries held, {inputs.names[k]} "
              f"{100 * first[k]:.0f}% to {100 * second[k]:.0f}%, the largest move "
              f"{100 * np.abs(second - first).max():.1f} points")
    for name, (mean, sd) in data["parameter"].items():
        print(f"  {inputs.names[k]}'s expected return, {name}: {annual(mean):.2f}% a year, standard deviation "
              f"{annual(sd):.2f}")
    print(f"  the market: mean excess log return {annual(inputs.market_mean):.2f}% a year, standard error "
          f"{annual(inputs.market_volatility) / np.sqrt(inputs.months):.2f}, volatility "
          f"{100 * inputs.market_volatility * np.sqrt(12):.2f}%; buy and hold, risk aversion {ex.RISK_AVERSION:g}")
    print("    horizon in years: " + ", ".join(f"{years}" for years in (1, 5, 10, 15, 20)))
    for name in ("exact", "uncertain"):
        print(f"    share in stocks, estimates {name}: "
              + ", ".join(f"{100 * data[name][years - 1]:.1f}%" for years in (1, 5, 10, 15, 20)))


def stale_numbers(inputs, data):
    """The numbers that `slot1.tex` does not contain as `quoted_numbers` gives them: sentences to update."""
    deck = (HERE / "slot1.tex").read_text()
    return {name: number for name, number in quoted_numbers(inputs, data).items() if number not in deck}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rubik", type=Path,
                        help="Rubik's variable font, e.g. the Overleaf project's fonts/Rubik-Variable.ttf")
    args = parser.parse_args()
    inputs = ex.load()
    data = compute(inputs)
    with tempfile.TemporaryDirectory() as folder:
        ef.style(ef.add_rubik(args.rubik, Path(folder)) if args.rubik else ["DejaVu Sans"])
        ef.save(figure_weights(inputs, data), "intro-weights")
        ef.save(figure_horizon(data), "intro-horizon")
        ef.save(figure_posterior(data), "intro-posterior")
        ef.save(figure_prior(inputs, data), "intro-prior")
    print(f"4 figures in {ef.OUT}")
    report(inputs, data)
    stale = stale_numbers(inputs, data)
    print("\nslot1.tex quotes these numbers as printed above." if not stale else
          "\nslot1.tex does not quote these numbers as printed above; update the slides:")
    for name, number in stale.items():
        print(f"  {name}: {number}")


if __name__ == "__main__":
    main()
