"""Figures for the quick checks: rough_checks.json and heston_checks.json in this folder."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sbi4finance.calibration.plots import BASELINE, INK, SERIES, plt, style  # noqa: E402

HERE = Path(__file__).resolve().parent
style()


def rough_figure():
    r = json.loads((HERE / "rough_checks.json").read_text())
    names = ["H", "η", "ρ"]
    width = np.diff(np.asarray(r["prior_bounds"]), axis=1)[:, 0]
    sets = list(dict.fromkeys(s["quotes"] for s in r["scenarios"]))
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.2), constrained_layout=True)
    for k, q in enumerate(sets):
        rows = [s for s in r["scenarios"] if s["quotes"] == q]
        x = [100 * s["xi0_log_sd"] for s in rows]
        for i in range(3):
            y = [2 * 1.645 * s["seed1"]["sd"][i] / width[i] for s in rows]
            y2 = [2 * 1.645 * s["seed2"]["sd"][i] / width[i] for s in rows]
            axes[i].plot(x, y, marker="o", color=SERIES[k], label=q)
            axes[i].plot(x, y2, marker="x", linestyle=":", color=SERIES[k], linewidth=0.8)
        axes[3].plot(x, [s["seed1"]["corr_eta_rho"] for s in rows], marker="o", color=SERIES[k], label=q)
        axes[3].plot(x, [s["seed2"]["corr_eta_rho"] for s in rows], marker="x", linestyle=":", color=SERIES[k],
                     linewidth=0.8)
    for i in range(3):
        axes[i].axhline(1.0, color=INK, linewidth=0.8, linestyle="--")
        axes[i].set_title(f"{names[i]}: 90% local interval / prior range", loc="left", fontsize=9)
        axes[i].set_xlabel("uncertainty of each ξ0 level (% sd, log)")
        axes[i].set_ylim(bottom=0)
    axes[3].set_title("correlation of η and ρ", loc="left", fontsize=9)
    axes[3].set_xlabel("uncertainty of each ξ0 level (% sd, log)")
    axes[3].set_ylim(-1, 1)
    axes[3].axhline(0, color=BASELINE, linewidth=0.6)
    axes[0].legend(fontsize=7, loc="upper left")
    b = r["best"]["theta"]
    fig.suptitle(f"Rough Bergomi on SPY, local (Laplace) posterior at the least-squares fit "
                 f"H {b[0]:.3f}, η {b[1]:.2f}, ρ {b[2]:.2f}; dashed line: interval as wide as the prior box; "
                 f"dotted: second Monte Carlo seed", fontsize=9, x=0.01, ha="left")
    fig.savefig(HERE / "rough_checks.png", dpi=160)


def heston_figure():
    h = json.loads((HERE / "heston_checks.json").read_text())
    cases = list(h["cases"])
    labels = {"initial_variance": "v₀", "theta": "θ", "vol_of_vol": "σ_v", "rho": "ρ", "kappa": "κ"}
    bounds = {"initial_variance": (0.01, 0.16), "theta": (0.01, 0.16), "vol_of_vol": (0.1, 1.5), "rho": (-0.95, -0.1),
              "kappa": tuple(h["kappa_bounds"])}
    fig = plt.figure(figsize=(13.0, 7.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 5)
    for j, p in enumerate(labels):
        ax = fig.add_subplot(grid[0, j])
        for i, c in enumerate(cases):
            e = h["cases"][c]["parameters"].get(p)
            if e is None:
                continue
            q = e["posterior_q05_q50_q95"]
            ax.plot([q[0], q[2]], [i + 0.12, i + 0.12], color=SERIES[0], linewidth=2.2,
                    label="posterior 90%" if i == 1 else None)
            ax.plot(q[1], i + 0.12, "o", color=SERIES[0], markersize=3)
            if "gauss_newton_90" in e:
                g = e["gauss_newton_90"]
                ax.plot(g, [i - 0.12, i - 0.12], color=SERIES[1], linewidth=2.2,
                        label="Gauss–Newton 90%" if i == 1 else None)
            else:
                ax.text(e["best"], i - 0.12, "none", color=SERIES[1], fontsize=7, va="center")
        for edge in bounds[p]:
            ax.axvline(edge, color=INK, linewidth=0.7, linestyle="--")
        ax.set_yticks(range(len(cases)))
        ax.set_yticklabels(cases if j == 0 else [], fontsize=7.5)
        ax.set_ylim(len(cases) - 0.5, -0.5)
        ax.set_title(labels[p], loc="left")
        if p == "kappa":
            ax.set_xlim(-1.0, 8.6)
        if j == 0:
            ax.legend(fontsize=7, loc="lower left")
    # kappa profile
    ax = fig.add_subplot(grid[1, :2])
    prof = h["kappa_profile"]
    kap = np.array([p["kappa"] for p in prof])
    chi2 = np.array([p["chi2"] for p in prof])
    ax.plot(kap, chi2 - chi2.min(), marker="o", color=INK)
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
    ax.xaxis.set_major_locator(FixedLocator(kap))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("κ (fixed in each fit)")
    ax.set_ylabel("χ² above its minimum")
    ax.set_title("full grid: profile in κ, and product prices at each fit", loc="left", fontsize=9)
    twin = ax.twinx()
    ref = np.array(next(p["products"] for p in prof if p["kappa"] == 2.0))
    for k, (j, name) in enumerate([(2, "forward-start"), (3, "2-year put"), (0, "63-day variance swap")]):
        twin.plot(kap, [100 * (p["products"][j] / ref[j] - 1) for p in prof], marker="s", markersize=3,
                  color=SERIES[k % len(SERIES)], linewidth=1, label=name)
    twin.set_ylabel("price change vs κ = 2 (%)")
    twin.legend(fontsize=7, loc="upper center")
    # product ranges
    ax = fig.add_subplot(grid[1, 2:])
    prods = ["variance_swap", "volatility_swap", "forward_start_call", "put_out_of_grid"]
    pnames = ["63-day var swap", "63-day vol swap", "forward-start", "2-year put"]
    width = 0.8 / len(cases)
    for i, c in enumerate(cases):
        e = h["cases"][c]["products"]
        x = np.arange(len(prods)) + (i - (len(cases) - 1) / 2) * width
        ax.bar(x, [100 * e[p]["posterior_range_90_relative"] for p in prods], width * 0.9,
               color=plt.cm.viridis(0.1 + 0.8 * i / max(len(cases) - 1, 1)), label=c)
        ax.plot(x, [100 * e[p].get("delta_range_90_relative", np.nan) for p in prods], "_", color=INK, markersize=9,
                markeredgewidth=2, label="delta method" if i == 0 else None)
    ax.set_xticks(range(len(prods)))
    ax.set_xticklabels(pnames)
    ax.set_ylabel("90% range, % of price")
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
    ax.yaxis.set_major_locator(FixedLocator([2, 3, 5, 10, 20, 40]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title("product ranges: posterior (bars) against the delta method (black marks)", loc="left", fontsize=9)
    ax.legend(fontsize=6.5, ncol=2)
    fig.suptitle("Heston on SPY (21 Oct 2026): exact posterior against Gauss–Newton; dashed lines are the prior box",
                 fontsize=9.5, x=0.01, ha="left")
    fig.savefig(HERE / "heston_checks.png", dpi=160)


if __name__ == "__main__":
    for which in sys.argv[1:] or ["rough", "heston"]:
        {"rough": rough_figure, "heston": heston_figure}[which]()


def bs_figure():
    b = json.loads((HERE / "bs_hierarchy.json").read_text())
    sectors = list(b["sectors"])
    methods = ["per name", "proxy", "empirical Bayes", "hierarchical Bayes"]
    colors = {"per name": BASELINE, "proxy": SERIES[2], "empirical Bayes": SERIES[1], "hierarchical Bayes": SERIES[0]}
    fig = plt.figure(figsize=(13.0, 4.6), constrained_layout=True)
    grid = fig.add_gridspec(1, 4, width_ratios=[2.2, 1, 1, 1])
    ax = fig.add_subplot(grid[0])
    ex = b["sectors"][sectors[0]]["example"]
    truth = np.array(ex["truth"])
    order = np.argsort(truth)
    x = np.arange(len(truth))
    proxy = np.array(ex["proxy"])[0]
    ax.axhspan(100 * proxy[1], 100 * proxy[2], color=SERIES[2], alpha=0.15, label="proxy 90% band")
    ax.axhline(100 * proxy[0], color=SERIES[2], linewidth=1)
    for m, off in (("per name", -0.18), ("hierarchical Bayes", 0.18)):
        e = 100 * np.array(ex[m])[order]
        ax.vlines(x + off, e[:, 1], e[:, 2], color=colors[m], linewidth=2, label=f"{m} 90%")
    ax.scatter(x, 100 * truth[order], s=14, color=INK, zorder=5, label="true vol")
    ax.set_xlabel("illiquid names, sorted by true vol")
    ax.set_ylabel("implied vol (%)")
    ax.set_title(f"one sector ({sectors[0]}): each name's 90% interval", loc="left", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    for k, (key, title, unit) in enumerate((("rmse_vol_points", "error of the mark (RMSE)", "vol points"),
                                            ("coverage_90", "90% intervals: share covering the truth", "%"),
                                            ("mean_width_vol_points", "mean 90% width", "vol points"))):
        ax = fig.add_subplot(grid[k + 1])
        width = 0.2
        for j, m in enumerate(methods):
            vals = [b["sectors"][s]["scores"][m][key] * (100 if key == "coverage_90" else 1) for s in sectors]
            ax.bar(np.arange(len(sectors)) + (j - 1.5) * width, vals, width * 0.92, color=colors[m],
                   label=m if k == 0 else None)
        if key == "coverage_90":
            ax.axhline(90, color=INK, linewidth=0.8, linestyle="--")
            ax.set_ylim(50, 100)
        ax.set_xticks(range(len(sectors)))
        ax.set_xticklabels([s.replace(" + ", "\n+ ") for s in sectors], fontsize=7.5)
        ax.set_ylabel(unit)
        ax.set_title(title, loc="left", fontsize=9)
        if k == 0:
            ax.set_ylim(0, 5.2)
            ax.legend(fontsize=7, loc="upper left", ncol=1)
    s = b["settings"]
    fig.suptitle("Black–Scholes marks for one sector: liquid names quote the 63-point grid; illiquid names quote one "
                 f"21-day at-the-money call with {100 * s['illiquid_vol_noise']:.0f} vol points of noise; "
                 "true vols spread about 3 points across names", fontsize=9, x=0.01, ha="left")
    fig.savefig(HERE / "bs_hierarchy.png", dpi=160)
