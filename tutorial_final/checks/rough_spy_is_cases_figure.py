"""Figure for rough_spy_is_cases.py: sparse quotes (A) and wide spreads (B), posterior against Gauss–Newton.

    python rough_spy_is_cases_figure.py

One row per case: the marginal posteriors of H and rho (weighted histograms) against the Gauss–Newton normal, with
the prior box shaded; and the one-sided 90% reserves of the products as ratios to the posterior's (the delta
method, and the Gauss–Newton normal truncated to the box and pushed through the pricer). The full 63-quote case
with the design noise (rough_spy_is.py, seed 0) is the first row, for reference.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sbi4finance.calibration.plots import BASELINE, INK, SERIES, plt, style  # noqa: E402

HERE = Path(__file__).resolve().parent / "rough_spy_is"
BOUNDS = [(0.02, 0.35), (0.5, 3.5), (-0.99, -0.2)]
PRODUCTS = {"volatility_swap": "63-day vol swap", "forward_start_call": "3-to-6-month forward start",
            "put_out_of_grid": "2-year 90% put"}
POST, GN, DELTA = SERIES
style()


def weighted_quantiles(values, weights, levels=(0.05, 0.95)):
    order = np.argsort(values)
    cdf = np.cumsum(weights[order]) / np.sum(weights)
    return np.array([values[order[min(np.searchsorted(cdf, p), len(cdf) - 1)]] for p in levels])


def rows():
    r0 = json.loads((HERE / "rough_spy_is_seed0.json").read_text())
    a0 = np.load(HERE / "rough_spy_is_seed0.npz")
    out = [{"label": "63 quotes, design noise", "theta": a0["theta"], "w": a0["w_post"],
            "mean": np.asarray(r0["fit"]["theta"]), "cov": np.asarray(r0["gauss_newton_covariance"]),
            "products": {k: {"posterior": v["reserve_long"]["posterior"], "posterior_s": v["reserve_short"]["posterior"],
                             "gauss": v["reserve_long"]["gaussian_pushforward"],
                             "gauss_s": v["reserve_short"]["gaussian_pushforward"],
                             "delta": v["reserve_long"]["delta"]} for k, v in r0["products"].items()},
            "importance": r0["importance"]}]
    rc = json.loads((HERE / "rough_spy_is_cases.json").read_text())
    ac = np.load(HERE / "rough_spy_is_cases.npz")
    for key in ("A", "B"):
        c = rc["cases"][key]
        out.append({"label": c["label"], "theta": ac["theta"], "w": ac[f"w_post_{key}"],
                    "mean": np.asarray(c["fit"]["theta"]), "cov": np.asarray(c["gauss_newton_covariance"]),
                    "products": {k: {"posterior": v["reserve_long"]["posterior"],
                                     "posterior_s": v["reserve_short"]["posterior"],
                                     "gauss": v["reserve_long"]["truncated_gaussian"],
                                     "gauss_s": v["reserve_short"]["truncated_gaussian"],
                                     "delta": v["reserve_long"]["delta"]} for k, v in c["products"].items()},
                    "importance": c["importance"]})
    return out


def main():
    data = rows()
    fig, axes = plt.subplots(len(data), 3, figsize=(13.0, 3.3 * len(data)), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for r, d in enumerate(data):
        for c, i in enumerate((0, 2)):
            ax = axes[r, c]
            mu, sd = d["mean"][i], np.sqrt(d["cov"][i, i])
            lo, hi = mu - 4 * sd, mu + 4 * sd
            bins = np.linspace(lo, hi, 33)
            ax.hist(d["theta"][:, i], bins=bins, weights=d["w"] / np.diff(bins)[0], color=POST, alpha=0.35)
            g = np.linspace(lo, hi, 300)
            ax.plot(g, norm.pdf(g, mu, sd), color=GN, linewidth=1.6)
            b = BOUNDS[i]
            for edge, side in ((b[0], "left"), (b[1], "right")):
                if lo < edge < hi:
                    ax.axvspan(lo if side == "left" else edge, edge if side == "left" else hi, color=BASELINE,
                               alpha=0.45, linewidth=0)
            top = ax.get_ylim()[1]
            q = weighted_quantiles(d["theta"][:, i], d["w"])
            ax.hlines(0.93 * top, *q, color=POST, linewidth=3)
            ax.hlines(0.86 * top, mu - 1.645 * sd, mu + 1.645 * sd, color=GN, linewidth=3)
            ax.set_xlim(lo, hi)
            ax.set_yticks([])
            name = "H" if i == 0 else "ρ"
            ax.set_title((f"{d['label']}\n" if c == 0 else "\n") + name, loc="left", fontsize=9)
        ax = axes[r, 2]
        labels, gauss, delta = [], [], []
        for key, name in PRODUCTS.items():
            p = d["products"][key]
            for side, word in (("", "long"), ("_s", "short")):
                labels.append(f"{name}, {word}")
                gauss.append(p["gauss" + side] / p["posterior" + side])
                delta.append(p["delta"] / p["posterior" + side])
        y = np.arange(len(labels))[::-1]
        ax.barh(y + 0.18, gauss, height=0.34, color=GN)
        ax.barh(y - 0.18, delta, height=0.34, color=DELTA)
        ax.axvline(1.0, color=INK, linewidth=0.8, linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlim(0.4, 2.0)
        imp = d["importance"]
        ax.set_title(f"one-sided 90% reserve / the posterior's (k̂ {imp['khat']:.2f}, ESS {imp['ess']:.0f})\n"
                     "orange: Gauss–Newton normal, priced; green: delta method", loc="left", fontsize=8.5)
    fig.suptitle("Rough Bergomi on SPY's 21 October smile: the exact posterior (blue) against Gauss–Newton (orange) as "
                 "the quotes say less; grey: outside the prior box; bars: 90% intervals\n32,768 paths, common random "
                 "numbers", fontsize=9.5, x=0.01, ha="left")
    fig.savefig(HERE.parent / "rough_spy_is_cases.png", dpi=150)
    print("saved", HERE.parent / "rough_spy_is_cases.png")


if __name__ == "__main__":
    main()
