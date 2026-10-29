"""Figure for rough_spy_is.py: the importance-sampled posterior on SPY against Gauss–Newton.

    python rough_spy_is_figure.py

Top row: the marginal posteriors of H, eta, rho (weighted histograms, seed 0; seed 1 as a line when present)
against the Gauss–Newton normal, with both 90% intervals.
Bottom row: the joint 90% regions of (eta, rho) and (H, rho), the posterior's (weighted kernel density) against
the Gauss–Newton ellipse of the same pair; and the one-sided 90% reserves of the products, as ratios to the
posterior's (Gaussian pushforward and delta method; bootstrap errors over proposals).
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2, gaussian_kde, norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbi4finance.calibration.plots import BASELINE, INK, SERIES, plt, style  # noqa: E402

HERE = Path(__file__).resolve().parent / "rough_spy_is"
NAMES = ["H", "η", "ρ"]
PRODUCTS = {"volatility_swap": "63-day vol swap", "forward_start_call": "3-to-6-month forward start",
            "put_out_of_grid": "2-year 90% put"}
POST, GN, DELTA = SERIES
style()


def load(seed):
    path = HERE / f"rough_spy_is_seed{seed}.npz"
    if not path.exists():
        return None, None
    return np.load(path), json.loads((HERE / f"rough_spy_is_seed{seed}.json").read_text())


def weighted_quantiles(values, weights, levels=(0.05, 0.95)):
    order = np.argsort(values)
    cdf = np.cumsum(weights[order]) / np.sum(weights)
    return np.array([values[order[min(np.searchsorted(cdf, p), len(cdf) - 1)]] for p in levels])


def ellipse(mean, cov, level=0.9, n=200):
    t = np.linspace(0, 2 * np.pi, n)
    vals, vecs = np.linalg.eigh(cov)
    r = np.sqrt(chi2.ppf(level, 2))
    return mean[:, None] + vecs @ (r * np.sqrt(vals)[:, None] * np.stack([np.cos(t), np.sin(t)]))


def joint_panel(ax, a, b, arrays, report):
    theta, w = arrays["theta"], arrays["w_post"]
    xy = theta[:, [a, b]].T
    kde = gaussian_kde(xy, weights=w)
    dens = kde(xy)
    order = np.argsort(dens)[::-1]
    level = dens[order][np.searchsorted(np.cumsum(w[order]) / w.sum(), 0.9)]
    cov = np.asarray(report["gauss_newton_covariance"])[np.ix_([a, b], [a, b])]
    mean = np.asarray(report["fit"]["theta"])[[a, b]]
    e = ellipse(mean, cov)
    pad = 0.25 * (e.max(axis=1) - e.min(axis=1))
    gx = np.linspace(e[0].min() - pad[0], e[0].max() + pad[0], 120)
    gy = np.linspace(e[1].min() - pad[1], e[1].max() + pad[1], 120)
    X, Y = np.meshgrid(gx, gy)
    Z = kde(np.stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    keep = w > 0.2 / len(w)
    ax.scatter(theta[keep, a], theta[keep, b], s=4, color=POST, alpha=0.25, linewidths=0)
    ax.contour(X, Y, Z, levels=[level], colors=[POST], linewidths=1.6)
    ax.plot(e[0], e[1], color=GN, linewidth=1.4, linestyle="--")
    ax.plot(*mean, marker="+", color=INK, markersize=9)
    ax.set_xlim(gx[0], gx[-1])
    ax.set_ylim(gy[0], gy[-1])
    ax.set_xlabel(NAMES[a])
    ax.set_ylabel(NAMES[b])
    ax.set_title(f"({NAMES[a]}, {NAMES[b]}): 90% regions, both marginal", loc="left", fontsize=9)


def main():
    a0, r0 = load(0)
    a1, r1 = load(1)
    fig = plt.figure(figsize=(13.0, 7.6), constrained_layout=True)
    top, bottom = fig.subfigures(2, 1)
    top_axes = top.subplots(1, 3)
    bottom_axes = bottom.subplots(1, 3)
    for i in range(3):
        ax = top_axes[i]
        x, w = a0["theta"][:, i], a0["w_post"]
        mu, sd = r0["fit"]["theta"][i], np.sqrt(np.asarray(r0["gauss_newton_covariance"])[i, i])
        lo, hi = mu - 4 * sd, mu + 4 * sd
        bins = np.linspace(lo, hi, 33)
        ax.hist(x, bins=bins, weights=w / np.diff(bins)[0], color=POST, alpha=0.35, label="posterior, seed 0")
        if a1 is not None:
            h, _ = np.histogram(a1["theta"][:, i], bins=bins, weights=a1["w_post"] / np.diff(bins)[0])
            ax.stairs(h, bins, color=POST, linewidth=1.0, label="posterior, seed 1")
        g = np.linspace(lo, hi, 300)
        ax.plot(g, norm.pdf(g, mu, sd), color=GN, linewidth=1.6, label="Gauss–Newton normal")
        top = ax.get_ylim()[1]
        q = weighted_quantiles(x, w)
        ax.hlines(0.93 * top, *q, color=POST, linewidth=3)
        ax.hlines(0.87 * top, mu - 1.645 * sd, mu + 1.645 * sd, color=GN, linewidth=3)
        e = r0["parameters"][["hurst", "eta", "rho"][i]]
        ax.set_title(f"{NAMES[i]}: Gauss–Newton 90% width / posterior's = {e['width_ratio_gn_over_posterior']:.2f}\n"
                     f"lower side {e['lower_ratio_gn_over_posterior']:.2f}, upper side "
                     f"{e['upper_ratio_gn_over_posterior']:.2f} (bars: 90% intervals)", loc="left", fontsize=8.5)
        ax.set_yticks([])
        ax.set_xlim(lo, hi)
        if i == 0:
            ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, 0.83))
    joint_panel(bottom_axes[0], 1, 2, a0, r0)
    joint_panel(bottom_axes[1], 0, 2, a0, r0)
    ax = bottom_axes[2]
    labels, rows = [], []
    for key, name in PRODUCTS.items():
        for side, word in (("reserve_long", "long"), ("reserve_short", "short")):
            s = r0["products"][key][side]
            p = s["posterior"]
            rows.append((s["gaussian_pushforward"] / p, s["delta"] / p,
                         np.hypot(s["gaussian_bootstrap_sd"] / p, s["gaussian_pushforward"] * s["posterior_bootstrap_sd"]
                                  / p**2)))
            labels.append(f"{name}, {word}")
    y = np.arange(len(rows))[::-1]
    ax.barh(y + 0.18, [r[0] for r in rows], height=0.34, xerr=[r[2] for r in rows], color=GN,
            label="Gauss–Newton normal through the pricer", error_kw={"elinewidth": 0.8})
    ax.barh(y - 0.18, [r[1] for r in rows], height=0.34, color=DELTA, label="delta method")
    ax.axvline(1.0, color=INK, linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlim(0.6, 1.4)
    ax.set_xlabel("one-sided 90% reserve / the posterior's")
    ax.set_title("Products: one-sided reserve / posterior's\norange: Gauss–Newton normal, priced\ngreen: delta method",
                 loc="left", fontsize=8.5)
    for x in (0.9, 1.1):
        ax.axvline(x, color=BASELINE, linewidth=0.6)
    imp = r0["importance"]
    fig.suptitle(f"Rough Bergomi on SPY (63 quotes, ξ0 known): importance-sampled posterior against Gauss–Newton, on "
                 f"the same pricer (32,768 paths, common random numbers)\nseed 0: {r0['proposals']} proposals, "
                 f"efficiency {imp['efficiency']:.2f}, k̂ {imp['khat']:.2f}"
                 + ("" if r1 is None else f"; seed 1: {r1['proposals']} proposals, efficiency "
                    f"{r1['importance']['efficiency']:.2f}, k̂ {r1['importance']['khat']:.2f}"),
                 fontsize=9, x=0.01, ha="left")
    fig.savefig(HERE.parent / "rough_spy_is.png", dpi=160)
    print("saved", HERE.parent / "rough_spy_is.png")


if __name__ == "__main__":
    main()
