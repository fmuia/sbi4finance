"""Figures of the course realignment plan (docs/course/v2-plan.md), drawn from the quick checks of 23-24 October.

    python scripts/course/realignment.py

Inputs (all committed): tutorial_final/checks/heston_checks.json, rough_spy_is/rough_spy_is_seed0.json,
rough_spy_is/rough_spy_is_cases.json, desk_case/desk_case_combined.json, desk_case/validate_thin_x1_batch{0,1}.json,
long_end/long_end.json, bs_hierarchy.json, thin_chains/ewi_2026-10-22.json.
Outputs: docs/course/images/v2-classical-vs-posterior.png, v2-coverage.png, v2-thin-chain.png, and
v2-evidence-numbers.json with every plotted number, so that the plan's text can be checked against it.
"""

import json

import numpy as np
from scipy.stats import binom

from common import COLORS, GREY, OUT, ROOT, plt, save, tidy

CHECKS = ROOT / "tutorial_final" / "checks"


def load(name):
    return json.loads((CHECKS / name).read_text())


def classical_vs_posterior(numbers):
    """Ratio of the classical 90% range or one-sided reserve to the posterior's, case by case."""
    heston = load("heston_checks.json")["cases"]
    seed0 = load("rough_spy_is/rough_spy_is_seed0.json")["products"]
    cases = load("rough_spy_is/rough_spy_is_cases.json")["cases"]
    desk = load("desk_case/desk_case_combined.json")["scales"]["1.0"]["products"]
    long_end = load("long_end/long_end.json")

    columns = []  # (label, group, [(product, side, ratio)])
    hlabels = {"full grid, kappa fixed": "Heston\n63 quotes\nκ = 2",
               "full grid, kappa free": "Heston\n63 quotes\nκ free",
               "up to 42 days, kappa fixed": "Heston\n36 quotes\n≤ 42 days",
               "3 x 5 grid, kappa free": "Heston\n15 quotes\nκ free"}
    for key, label in hlabels.items():
        pts = [(p, "both", v["delta_over_posterior_range"]) for p, v in heston[key]["products"].items()]
        columns.append((label, "Heston on SPY: delta-method range / posterior range", pts))

    def reserve_points(products, method):
        pts = []
        for p, v in products.items():
            if p == "variance_swap":
                continue
            for side in ("long", "short"):
                post = v[f"reserve_{side}"]["posterior"]
                pts.append((p, side, v[f"reserve_{side}"][method] / post))
        return pts

    columns.append(("rough\n63 quotes", "rough Bergomi: delta-method reserve / posterior reserve",
                    reserve_points(seed0, "delta")))
    columns.append(("rough\n63 quotes\n3× noise", "rough Bergomi: delta-method reserve / posterior reserve",
                    reserve_points(cases["B"]["products"], "delta")))
    columns.append(("rough\n15 quotes", "rough Bergomi: delta-method reserve / posterior reserve",
                    reserve_points(cases["A"]["products"], "delta")))
    pts, pts_fixed = [], []
    for p, v in desk.items():
        for side in ("long", "short"):
            pts.append((p, side, v["delta"][side] / v["posterior"][side]))
            pts_fixed.append((p, side, v["h_fixed_delta"][side] / v["posterior"][side]))
    columns.append(("rough\nfront months\n27 quotes", "rough Bergomi: delta-method reserve / posterior reserve", pts))
    columns.append(("rough\nfront months\nH fixed (desk)", "desk convention: reserve with H = 0.10 / posterior", pts_fixed))
    lpts = []
    for p, d, q in zip(long_end["products"], long_end["range90_desk_delta"], long_end["range90_posterior_mc"]):
        lpts.append((p, "both", d / q))
    columns.append(("rough\nlong end\nξ0 flat (desk)", "desk convention: range with ξ0 flat / posterior range", lpts))

    markers = {"variance_swap": "s", "volatility_swap": "D", "forward_start_call": "o", "forward_start": "o",
               "put_out_of_grid": "v", "put_1y_80": "v", "cliquet": "P", "autocallable": "X",
               "autocallable 3y": "X", "cliquet 3y": "P", "forward-start 2y-3y": "o", "put 5y 80%": "v",
               "variance swap 3y": "s"}
    floor = 0.04
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.axhspan(0.8, 1.25, color="0.9", zorder=0)
    ax.axhline(1.0, color="0.3", lw=0.8)
    for i, (label, group, pts) in enumerate(columns):
        colour = COLORS[0] if label.startswith("Heston") else (COLORS[3] if "desk" in label else COLORS[1])
        for j, (p, side, r) in enumerate(pts):
            x = i + (j - (len(pts) - 1) / 2) * 0.06
            y = max(r, floor)
            face = colour if side in ("both", "long") else "white"
            ax.scatter(x, y, marker=markers.get(p, "o"), s=38, facecolor=face, edgecolor=colour, lw=1.2, zorder=3)
            if r < floor:
                ax.annotate("0", (x, y), textcoords="offset points", xytext=(5, -3), fontsize=7, color=colour)
        numbers.setdefault("classical_vs_posterior", []).append(
            {"case": label.replace("\n", " "), "measure": group,
             "points": [{"product": p, "side": side, "ratio": r} for p, side, r in pts]})
    ax.set_yscale("log")
    ax.set_ylim(floor * 0.8, 2.2)
    ax.set_yticks([0.05, 0.1, 0.2, 0.5, 0.8, 1, 1.25, 2])
    ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "0.8", "1", "1.25", "2"])
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([c[0] for c in columns], fontsize=8)
    ax.set_ylabel("classical / posterior")
    ax.set_title("Where the classical answer and the posterior part: from a liquid smile (left) to what the "
                 "quotes do not see (right)", loc="left", fontsize=10)
    handles = [plt.Line2D([], [], marker=m, ls="", color="0.3", label=lab) for lab, m in
               [("forward start", "o"), ("put beyond the grid", "v"), ("volatility swap", "D"),
                ("variance swap", "s"), ("cliquet", "P"), ("autocallable", "X")]]
    handles += [plt.Line2D([], [], marker="o", ls="", markerfacecolor="0.3", markeredgecolor="0.3",
                           label="long side or two-sided range"),
                plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="0.3",
                           label="short side")]
    ax.legend(handles=handles, fontsize=7, ncol=4, loc="lower left", frameon=False)
    tidy(ax)
    save(fig, "v2-classical-vs-posterior.png")


def coverage(numbers):
    b0 = load("desk_case/validate_thin_x1_batch0.json")["report"]
    b1 = load("desk_case/validate_thin_x1_batch1.json")["report"]
    n = b0["replicas"] + b1["replicas"]
    methods = [("posterior", "posterior"), ("gauss_newton", "Gauss–Newton"), ("profile", "profile"),
               ("h_fixed", "desk: H = 0.10")]
    products = b0["products"]
    lo, hi = binom.ppf(0.025, n, 0.1) / n, binom.ppf(0.975, n, 0.1) / n
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={"width_ratios": [1.6, 1]})
    ax.axhspan(lo, hi, color="0.9", zorder=0)
    ax.axhline(0.1, color="0.3", lw=0.8)
    marks = ["X", "o", "v", "P"]
    rows = []
    for i, (key, label) in enumerate(methods):
        for s, side in enumerate(("long", "short")):
            for j, p in enumerate(products):
                rate = (b0["breach_rate"][key][side][p] * b0["replicas"]
                        + b1["breach_rate"][key][side][p] * b1["replicas"]) / n
                x = i + (s - 0.5) * 0.36 + (j - 1.5) * 0.06
                ax.scatter(x, rate, marker=marks[j], s=36, zorder=3, edgecolor=COLORS[i], lw=1.2,
                           facecolor=COLORS[i] if side == "long" else "white")
                rows.append({"method": label, "side": side, "product": p, "breach_rate": rate})
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m[1] for m in methods])
    ax.set_ylabel("share of chains whose truth breaches the 90% point")
    ax.set_title(f"Thin chains (SPY's front months), {n} simulated: one-sided 90% reserves", loc="left", fontsize=10)
    names = {"autocallable": "autocallable", "forward_start": "forward start", "put_1y_80": "1-year 80% put",
             "cliquet": "cliquet"}
    ax.legend(handles=[plt.Line2D([], [], marker=m, ls="", color="0.3", label=names.get(p, p.replace("_", " ")))
                       for m, p in zip(marks, products)]
              + [plt.Line2D([], [], marker="o", ls="", color="0.3", label="long side filled, short side open")],
              fontsize=7, frameon=False, loc="upper left")
    tidy(ax)
    numbers["desk_case_breach_rates"] = {"chains": n, "binomial_95_band": [lo, hi], "rows": rows}

    h = load("bs_hierarchy.json")["sectors"]
    labels = list(h)
    meth = ["per name", "proxy", "empirical Bayes", "hierarchical Bayes"]
    width = 0.19
    for k, m in enumerate(meth):
        vals = [h[s]["scores"][m]["coverage_90"] for s in labels]
        bx.bar(np.arange(len(labels)) + (k - 1.5) * width, vals, width, color=[GREY, COLORS[4], COLORS[1], COLORS[0]][k],
               label=m)
    bx.axhline(0.9, color="0.3", lw=0.8, ls="--")
    bx.set_ylim(0.5, 1.0)
    bx.set_xticks(range(len(labels)))
    bx.set_xticklabels(labels)
    bx.set_ylabel("coverage of 90% intervals, illiquid names")
    bx.set_title("Black–Scholes marks for thin names in one sector", loc="left", fontsize=10)
    bx.legend(fontsize=7, frameon=False, loc="upper center", ncol=2)
    tidy(bx)
    numbers["bs_hierarchy_coverage"] = {s: {m: h[s]["scores"][m]["coverage_90"] for m in meth} for s in labels}
    numbers["bs_hierarchy_width"] = {s: {m: h[s]["scores"][m]["mean_width_vol_points"] for m in meth} for s in labels}
    save(fig, "v2-coverage.png")


def thin_chain(numbers):
    ewi = load("thin_chains/ewi_2026-10-22.json")
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.8), sharey=True)
    summary = []
    for ax, e in zip(axes, ewi["expiries"]):
        for q in e["quotes"]:
            low = q["iv_bid"] if q["two_sided"] and q["iv_bid"] is not None else 0.0
            high = q["iv_ask"]
            ax.plot([q["k"], q["k"]], [100 * low, 100 * high], lw=3, solid_capstyle="butt",
                    color=COLORS[0] if q["two_sided"] else "0.75")
        mids = [(q["k"], q["iv_mid"]) for q in e["quotes"] if q["iv_mid"] is not None]
        ax.plot([m[0] for m in mids], [100 * m[1] for m in mids], "o", ms=3.5, color="0.15")
        ax.fill_between([-0.02, 0.02], 25 - 0.3, 25 + 0.3, color=COLORS[1], alpha=0.8)
        ax.set_title(f"{e['expiry']} ({e['days']} days)\nvariance-swap strip: {100 * e['strip_vol_on_bids']:.0f}% "
                     f"(bids) to {100 * e['strip_vol_on_asks']:.0f}% (asks)", fontsize=8.5, loc="left")
        ax.set_xlabel("log-moneyness")
        ax.set_ylim(0, 80)
        tidy(ax)
        summary.append({"expiry": e["expiry"], "out_of_the_money": e["out_of_the_money"],
                        "two_sided": e["two_sided"], "median_iv_spread_points": 100 * e["median_iv_spread"],
                        "strip_vol_on_bids": e["strip_vol_on_bids"], "strip_vol_on_asks": e["strip_vol_on_asks"]})
    axes[0].set_ylabel("implied volatility (%)")
    axes[-1].legend(handles=[plt.Line2D([], [], color=COLORS[0], lw=3, label="bid to ask, two-sided"),
                            plt.Line2D([], [], color="0.75", lw=3, label="ask only (no bid)"),
                            plt.Line2D([], [], color="0.15", marker="o", ls="", ms=3.5, label="mid"),
                            plt.Line2D([], [], color=COLORS[1], lw=6,
                                       label="SPY design noise at the money (±0.3 pt)")],
                   fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("A thin chain: EWI (iShares MSCI Italy), close of 22 October 2026; "
                 f"{int(ewi['volume_total'])} contracts traded across the whole chain that day",
                 fontsize=10, x=0.01, ha="left")
    numbers["ewi"] = {"volume_total": ewi["volume_total"], "expiries": summary}
    save(fig, "v2-thin-chain.png")


if __name__ == "__main__":
    numbers = {}
    classical_vs_posterior(numbers)
    coverage(numbers)
    thin_chain(numbers)
    (OUT / "v2-evidence-numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1)[:6000])
