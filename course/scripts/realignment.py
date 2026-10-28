"""Figures of the course's version-2 plan (course/v2-plan.md), drawn from the quick checks of 23-24 October.

    python course/scripts/realignment.py

Only evidence that survived the review of 24 October is drawn. The front-month case, its 80-chain validation and
the 15-quote and three-times-noise cases gave every method SPY's curve read from the full chain; they are
withdrawn (course/v2-plan.md, Section 4) and not drawn here.

Inputs (all committed): lab/checks/heston_checks.json, rough_spy_is/rough_spy_is_seed0.json,
long_end/long_end.json, bs_hierarchy.json, thin_chains/{EWI,EWG,EWQ}_2026-10-22.json, thin_chains/FEZ_2026-10-23.json.
Outputs: course/images/v2-classical-vs-posterior.png, v2-pooling.png, v2-thin-chain.png,
v2-thin-candidates.png, and v2-evidence-numbers.json with every plotted number, so that the plan's text can be
checked against it (course/scripts/check_v2_plan.py).
"""

import json

import numpy as np

from common import COLORS, GREY, OUT, ROOT, plt, save, tidy

CHECKS = ROOT / "lab" / "checks"
CANDIDATES = (("EWI", "EWI_2026-10-22.json"), ("EWG", "EWG_2026-10-22.json"), ("EWQ", "EWQ_2026-10-22.json"),
              ("FEZ", "FEZ_2026-10-23.json"))


def load(name):
    return json.loads((CHECKS / name).read_text())


def classical_vs_posterior(numbers):
    """Ratio of the classical 90% range or one-sided reserve to the posterior's, case by case."""
    heston = load("heston_checks.json")["cases"]
    seed0 = load("rough_spy_is/rough_spy_is_seed0.json")["products"]
    long_end = load("long_end/long_end.json")

    columns = []  # (label, group, [(product, side, ratio)])
    hlabels = {"full grid, kappa fixed": "Heston\n63 quotes\nκ = 2",
               "full grid, kappa free": "Heston\n63 quotes\nκ free",
               "up to 42 days, kappa fixed": "Heston\n36 quotes\n≤ 42 days",
               "3 x 5 grid, kappa free": "Heston\n15 quotes\nκ free"}
    for key, label in hlabels.items():
        pts = [(p, "both", v["delta_over_posterior_range"]) for p, v in heston[key]["products"].items()]
        columns.append((label, "Heston on SPY: delta-method range / posterior range", pts))
    pts = []
    for p, v in seed0.items():
        if p == "variance_swap":  # zero for both under rough Bergomi with the curve given
            continue
        for side in ("long", "short"):
            pts.append((p, side, v[f"reserve_{side}"]["delta"] / v[f"reserve_{side}"]["posterior"]))
    columns.append(("rough Bergomi\n63 quotes", "rough Bergomi on SPY: delta-method reserve / posterior reserve", pts))
    lpts = [(p, "both", d / q) for p, d, q in zip(long_end["products"], long_end["range90_desk_delta"],
                                                   long_end["range90_posterior_mc"])]
    columns.append(("rough Bergomi\nto five years\ncurve flat", "flat extrapolation's range / posterior range", lpts))

    markers = {"variance_swap": "s", "volatility_swap": "D", "forward_start_call": "o", "put_out_of_grid": "v",
               "autocallable 3y": "X", "cliquet 3y": "P", "forward-start 2y-3y": "o", "put 5y 80%": "v",
               "variance swap 3y": "s"}
    floor = 0.04
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.axhspan(0.8, 1.25, color="0.9", zorder=0)
    ax.axhline(1.0, color="0.3", lw=0.8)
    for i, (label, group, pts) in enumerate(columns):
        colour = COLORS[0] if label.startswith("Heston") else (COLORS[3] if "flat" in label else COLORS[1])
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
    ax.set_title("SPY, 21 October 2026: the classical answer against the posterior's", loc="left", fontsize=10)
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


def pooling(numbers):
    """Pooling across names in a simulated Black-Scholes sector: one sector's intervals, coverage against width, and
    the error of the mark of a product no illiquid name quotes."""
    h = load("bs_hierarchy.json")["sectors"]
    labels = list(h)
    meth = ["per name", "proxy", "empirical Bayes", "hierarchical Bayes"]
    colour = dict(zip(meth, [GREY, COLORS[4], COLORS[1], COLORS[0]]))
    small = "2 liquid + 6 illiquid"
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(13.2, 4.3), gridspec_kw={"width_ratios": [1.35, 1, 1]})

    # (a) the first simulated small sector: each illiquid name's 90% interval under each method, and its truth
    ex = h[small]["example"]
    truth = 100 * np.array(ex["truth"])
    for i, tv in enumerate(truth):
        for k, m in enumerate(meth):
            mark, lo, hi = (100 * v for v in ex[m][i])
            y = i + (k - 1.5) * 0.17
            covered = lo <= tv <= hi
            ax.plot([lo, hi], [y, y], color=colour[m], lw=2.2, solid_capstyle="butt", alpha=1.0 if covered else 0.45)
            ax.plot([mark], [y], "o", color=colour[m], ms=3.2)
        ax.plot([tv, tv], [i - 0.38, i + 0.38], color=COLORS[6], lw=1.4)
    for x in 100 * np.array(ex["liquid_marks"]):
        ax.axvline(x, color="0.75", lw=0.9, ls=":")
    ax.set_yticks(range(len(truth)), [f"name {i + 1}" for i in range(len(truth))])
    ax.invert_yaxis()
    ax.set_xlabel("implied volatility (%)")
    ax.set_title("One sector of 2 liquid and 6 illiquid names:\n90% intervals, truth (black), liquid marks (dotted)",
                 loc="left", fontsize=9.5)

    # (b) coverage against width, over all simulated sectors
    for s, marker in zip(labels, ("o", "s")):
        for m in meth:
            sc = h[s]["scores"][m]
            bx.plot(sc["mean_width_vol_points"], sc["coverage_90"], marker, color=colour[m], ms=8,
                    mfc=colour[m] if s == small else "white", mew=1.6)
    bx.axhline(0.9, color="0.3", lw=0.8, ls="--")
    bx.set_xlabel("mean width of the 90% interval (vol points)")
    bx.set_ylabel("coverage of 90% intervals, illiquid names")
    bx.set_ylim(0.5, 1.06)
    bx.set_title("Coverage against width\nfilled: 2 + 6 names (600 sectors); open: 8 + 32 (300)", loc="left",
                 fontsize=9.5)
    from matplotlib.lines import Line2D
    bx.legend(handles=[Line2D([], [], color=colour[m], marker="o", ls="", label=m) for m in meth], fontsize=7,
              frameon=False, loc="upper center", ncol=2)

    # (c) the error of the mark of a 6-month at-the-money call, which no illiquid name quotes
    width = 0.19
    for k, m in enumerate(meth):
        vals = [h[s]["scores"][m]["product_rmse_percent"] for s in labels]
        cx.bar(np.arange(len(labels)) + (k - 1.5) * width, vals, width, color=colour[m], label=m)
    cx.set_xticks(range(len(labels)), labels)
    cx.set_ylabel("root-mean-square error of the mark (% of price)")
    cx.set_title("Error of the mark: a 6-month at-the-money call", loc="left", fontsize=9.5)
    tidy(ax, bx, cx)
    fig.tight_layout()
    numbers["bs_hierarchy_coverage"] = {s: {m: h[s]["scores"][m]["coverage_90"] for m in meth} for s in labels}
    numbers["bs_hierarchy_width"] = {s: {m: h[s]["scores"][m]["mean_width_vol_points"] for m in meth} for s in labels}
    numbers["bs_hierarchy_product_rmse_percent"] = {s: {m: h[s]["scores"][m]["product_rmse_percent"] for m in meth}
                                                    for s in labels}
    numbers["bs_hierarchy_example_covered"] = {
        m: int(sum(100 * lo <= 100 * tv <= 100 * hi for (_, lo, hi), tv in zip(ex[m], ex["truth"]))) for m in meth}
    numbers["bs_hierarchy_tau"] = {s: {"truth": 0.10, "empirical_Bayes_median": h[s]["tau_eb_median"],
                                       "hierarchical_posterior_mean_median": h[s]["tau_hb_mean_median"]}
                                   for s in labels}
    save(fig, "v2-pooling.png")


def thin_candidates(numbers):
    """Median near-money bid-ask width per expiry to one year, for the four candidate chains."""
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    rows = {}
    for i, (ticker, name) in enumerate(CANDIDATES):
        chain = load(f"thin_chains/{name}")
        days, width, count = [], [], []
        for e in chain["expiries"]:
            if e["days"] > 366 or e["median_iv_spread_near_money"] is None:
                continue
            days.append(e["days"])
            width.append(100 * e["median_iv_spread_near_money"])
            count.append(e["two_sided_near_money"])
        ax.plot(days, width, "-", color=COLORS[i], lw=1)
        ax.scatter(days, width, s=[12 * c for c in count], color=COLORS[i], zorder=3, label=ticker)
        rows[ticker] = {"name": chain["name"], "asof": chain["source"], "volume_total": chain["volume_total"],
                        "open_interest_total": chain["open_interest_total"],
                        "expiries": [{"calendar_days": d, "median_near_money_width_vol_points": w,
                                      "two_sided_near_money": c} for d, w, c in zip(days, width, count)]}
    ax.axhline(2 * 0.3, color="0.3", lw=0.8, ls="--")
    ax.annotate("SPY's design noise at the money, as a bid-ask width (0.6 points)", (5, 0.62), fontsize=7,
                color="0.3", va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel("calendar days to expiry")
    ax.set_ylabel("median bid-ask width near the money (vol points)")
    ax.set_title("Four candidate thin chains, 22 and 23 October 2026 (marker area: two-sided quotes near the money)",
                 loc="left", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    tidy(ax)
    numbers["thin_candidates"] = rows
    save(fig, "v2-thin-candidates.png")


def thin_chain(numbers):
    ewi = load("thin_chains/EWI_2026-10-22.json")
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
    pooling(numbers)
    thin_chain(numbers)
    thin_candidates(numbers)
    (OUT / "v2-evidence-numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1)[:6000])
