"""Figures used by Lectures 6 and 15 (the file keeps an earlier lecture number): SPY's curve against the pilot's prior,
the product ranges, and Heston against rough Bergomi.

Usage: python course/scripts/lecture14.py   (writes course/images/l14-*.png and prints the numbers quoted in the text)

Every number comes from the cached study results in tutorial/assets/results/calibration, from the
SPY surface file, or from the studies' reported values (cited in the lecture); the only new
computation is the fit of the curve prior to the SPY forward-variance curve (Lecture 15, Figure 15.6).
"""

import json

import numpy as np

from common import COLORS, DAYS, GREY, ROOT, load_spy, plt, save, tidy
from sbi4finance.calibration import tasks

REPORT = {}
RESULTS = ROOT / "tutorial" / "assets" / "results" / "calibration"
PRODUCTS = ["variance_swap", "volatility_swap", "forward_start_call", "put_out_of_grid"]
PRODUCT_LABELS = ["variance swap\n63 d", "volatility swap\n63 d", "forward-start call\n63 → 126 d", "90% put\n2 years"]


def load(path):
    return json.loads((RESULTS / path).read_text())


# Figure 14.1: where each study stands (course/status.md) --
STATUS_COLORS = {"met": COLORS[4], "reported": COLORS[0], "partly met": COLORS[1], "not met": COLORS[3],
                 "pending": "0.88", "data done": "0.70", "": "white"}
STATUS = [  # (study, question, Heston, rough pilot, rough full bank, real smile)
    ("S1", "training bank", "met", "met", "pending", ""),
    ("S2", "calibration (SBC, coverage)", "met", "not met", "pending", ""),
    ("S3", "agreement with exact Heston", "partly met", "", "", ""),
    ("S4", "agreement with pricer route", "partly met", "", "pending", ""),
    ("S5", "audit", "", "reported", "pending", ""),
    ("S6", "classical route", "reported", "reported", "pending", ""),
    ("S7", "products and reserves", "reported", "reported", "pending", ""),
    ("S8", "the real smile", "", "", "", "data done"),
    ("S9", "costs", "reported", "reported", "pending", ""),
]


# Figure 14.3: the SPY forward-variance curve against the curve prior -----------------
def best_prior_fit(levels, mids, slopes):
    """Smallest max |u_j| over slopes s, with log sigma0^2 free: a minimax constant fit for each s."""
    best = (np.inf, None, None)
    for s in slopes:
        r = np.log(levels) - np.log1p(s * mids)
        c = 0.5 * (r.max() + r.min())
        value = 0.5 * (r.max() - r.min())
        if value < best[0]:
            best = (value, s, c)
    return best


def fig_spy_curve():
    spy = load_spy()
    levels = np.array(spy["forward_variance_levels"])
    task = tasks.rough_task()
    grid = np.asarray(task["model_context"]["forward_variance_knots"])
    mids = 0.5 * (grid[1:] + grid[:-1])
    prior = task["context_prior"]["forward_variance"]
    rng = np.random.default_rng(5)
    n = 20_000
    level = rng.uniform(*prior["level_bounds"], n)
    slope = rng.uniform(*prior["slope_bounds"], n)
    u = rng.uniform(-prior["log_perturbation"], prior["log_perturbation"], (n, len(mids)))
    curves = level[:, None] ** 2 * (1 + slope[:, None] * mids) * np.exp(u)
    rel = np.log(curves / curves[:, :1])
    spy_rel = np.log(levels / levels[0])
    bounded = best_prior_fit(levels, mids, np.linspace(*prior["slope_bounds"], 9001))
    free = best_prior_fit(levels, mids, np.linspace(-1 / mids.max() + 1e-3, 200, 400_001))
    sigma0 = np.sqrt(np.exp(bounded[2]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    edges = np.r_[0, DAYS]
    q = np.quantile(rel, [0.0, 0.05, 0.5, 0.95, 1.0], axis=0)
    ax.stairs(q[4], edges, baseline=q[0], fill=True, color=COLORS[0], alpha=0.15, label="curve prior: full range")
    ax.stairs(q[3], edges, baseline=q[1], fill=True, color=COLORS[0], alpha=0.35, label="curve prior: 5% to 95%")
    ax.stairs(spy_rel, edges, baseline=None, color=COLORS[3], lw=2.2, label="SPY, 21 Oct 2026")
    ax.set_xlabel("time, business days")
    ax.set_ylabel(r"$\log(\xi_0 / \xi_0^{\mathrm{first}})$, each design interval")
    ax.set_title("Shape of the curve: each level over the first one", fontsize=10)
    ax.set_ylim(top=1.6)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=2)
    ax = axes[1]
    for (value, s, c), color, name in ((bounded, COLORS[0], "best fit with the prior's slope bound"),
                                       (free, COLORS[1], "best fit with the slope unbounded")):
        resid = np.log(levels) - np.log1p(s * mids) - c
        ax.plot(np.arange(1, 8), resid, "o-", color=color, label=f"{name}: max $|u_j|$ = {value:.2f}")
    ax.axhspan(-prior["log_perturbation"], prior["log_perturbation"], color=GREY, alpha=0.2, lw=0)
    ax.text(7.2, 0.21, "prior: $|u_j| \\leq 0.2$", ha="right", va="bottom", fontsize=8.5, color="0.35")
    ax.set_xticks(range(1, 8))
    ax.set_xticklabels([f"{a}–{b}" for a, b in zip(np.r_[0, DAYS[:-1]], DAYS)], fontsize=8)
    ax.set_xlabel("design interval, business days")
    ax.set_ylabel(r"perturbation $u_j$ needed on each interval")
    ax.set_ylim(-0.5, 0.5)
    ax.set_title("How far the SPY curve is from the smooth prior curves", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    tidy(*axes)
    save(fig, "l14-spy-curve-and-prior.png")
    REPORT["spy_curve"] = dict(ratio_last_to_first=float(levels[-1] / levels[0]), forward_vols=np.sqrt(levels).tolist(),
                               prior_max_ratio=float(np.exp(rel[:, -1].max())),
                               prior_share_reaching_spy_ratio=float(np.mean(rel[:, -1] >= spy_rel[-1])),
                               bounded=dict(max_u=bounded[0], slope=bounded[1], sigma0=float(sigma0)),
                               free=dict(max_u=free[0], slope=free[1], sigma0=float(np.sqrt(np.exp(free[2])))),
                               increments=np.diff(np.log(levels)).tolist())


# Figure 14.4: product ranges and reserves (S7) --------------------------------------
def fig_products():
    heston = load("heston/s7/products.json")["surfaces"]
    rough = load("rough-pilot/s7/products.json")["surfaces"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    report = {}
    for ax, key, title in ((axes[0], "range_90_relative", "90% range of the price under the network posterior,\nshare of its mean"),
                           (axes[1], "reserve_short_relative", "short-side reserve under the network posterior,\nshare of the least-squares price")):
        for k, (surfaces, color, marker, name) in enumerate(((heston, COLORS[0], "o", "Heston, test rows 2,300 to 2,302"),
                                                             (rough, COLORS[1], "s", "rough-Bergomi pilot, test rows 290 to 292"))):
            for i, p in enumerate(PRODUCTS):
                vals = [100 * s["products"][p][key] for s in surfaces]
                ax.plot(np.full(3, i + (k - 0.5) * 0.25), vals, marker, color=color, ms=6,
                        label=name if i == 0 else None)
                report[f"{name.split(',')[0]} {p} {key}"] = [round(v, 2) for v in vals]
        ax.set_xticks(range(4))
        ax.set_xticklabels(PRODUCT_LABELS, fontsize=8.5)
        ax.set_ylabel("%")
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.set_ylim(-0.5, 20)
    tidy(*axes)
    save(fig, "l14-product-ranges.png")
    shares = {}
    for name, surfaces in (("heston", heston), ("rough", rough)):
        for p in PRODUCTS:
            shares[f"{name} {p}"] = [{k: round(v, 2) for k, v in s["products"][p]["variance_share_by_parameter"].items()}
                                     for s in surfaces]
    REPORT["products"] = dict(values=report, variance_shares=shares,
                              heston_rows=[s["test_row"] for s in heston], rough_rows=[s["test_row"] for s in rough],
                              mc_se_rough=[{p: s["products"][p]["mc_se_typical"] for p in PRODUCTS} for s in rough],
                              pricing_cpu_rough=[s["pricing_cpu_seconds"] for s in rough],
                              ls_cpu_rough=[s["least_squares_cpu_seconds"] for s in rough])


# Figure 14.5: the same smile under two models (S7 across models) ----------------------
def fig_across_models():
    across = load("rough-pilot/s7-across/across-models.json")["surfaces"]
    rough_draws = np.load(RESULTS / "rough-pilot/s7/price-draws.npz")
    heston_draws = np.load(RESULTS / "rough-pilot/s7-across/heston-price-draws.npz")
    row = 290
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    for i, (ax, p) in enumerate(zip(axes, PRODUCTS)):
        r = rough_draws[f"prices_{row}"][:, i]
        h = heston_draws[f"heston_prices_{row}"][:, i]
        lo, hi = min(r.min(), h.min()), max(r.max(), h.max())
        pad = 0.08 * (hi - lo)
        bins = np.linspace(lo - pad, hi + pad, 40)
        if np.ptp(r) > 0:
            ax.hist(r, bins, density=True, color=COLORS[0], alpha=0.5, label="rough Bergomi, network posterior")
        else:
            ax.axvline(r[0], color=COLORS[0], lw=3, alpha=0.6, label="rough Bergomi, network posterior (one value)")
        ax.hist(h, bins, density=True, color=COLORS[4], alpha=0.5, label="Heston, network reweighted by Fourier likelihood")
        ax.axvline(rough_draws[f"point_prices_{row}"][i], color=COLORS[0], ls="--", lw=1.2, label="rough least squares")
        ax.axvline(heston_draws[f"heston_point_prices_{row}"][i], color=COLORS[4], ls="--", lw=1.2, label="Heston least squares")
        gap = across[0]["products"][p]["gap_between_points_relative"]
        ax.set_title(f"{PRODUCT_LABELS[i].replace(chr(10), ', ')}\nHeston − rough: {100 * gap:+.1f}%", fontsize=9.5)
        ax.set_xlabel("price, forward units")
        ax.set_yticks([])
        ax.set_ylim(0, ax.get_ylim()[1] * 2.0)
        ax.locator_params(axis="x", nbins=4)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    tidy(*axes)
    save(fig, "l14-across-models.png")
    REPORT["across"] = [dict(row=s["test_row"], rough_chi2=s["rough_least_squares"]["chi2"],
                             heston_chi2=s["heston_least_squares"]["chi2"], heston_efficiency=s["heston_audit"]["efficiency"],
                             gaps={p: s["products"][p]["gap_between_points_relative"] for p in PRODUCTS},
                             overlap={p: s["products"][p]["ranges_overlap"] for p in PRODUCTS},
                             rough={p: [s["products"][p]["rough"][k] for k in ("point", "q05", "q95")] for p in PRODUCTS},
                             heston={p: [s["products"][p]["heston"][k] for k in ("point", "q05", "q95")] for p in PRODUCTS})
                        for s in across]


# Figure 14.6: the 40-minute segment and the evidence each act shows ----------------
ACTS = [  # (start, end, act, evidence, status)
    (0, 4, "The question", "the SPY smile; the rules' words", "available"),
    (4, 9, "GBM under Q", "flat line against the smile; skew term structure", "available"),
    (9, 16, "Heston", "exact against amortized posterior (S3)", "available"),
    (16, 21, "Rough Bergomi", "one priced surface and its cost (S9)", "available"),
    (21, 26, "The amortized posterior", "posterior pairs for the SPY smile (S8)", "pending"),
    (26, 31, "Is it right?", "SBC (S2), audit (S5; on SPY, S8), pricer route (S4)", "pilot only"),
    (31, 35, "The classical route", "objective landscape, Gauss–Newton region (S6)", "pilot only"),
    (35, 39, "From posterior to prices", "price distributions and reserves (S7)", "pilot only"),
    (39, 40, "Across models", "Heston against rough Bergomi (S7)", "pilot only"),
]
ACT_COLORS = {"available": COLORS[4], "pilot only": COLORS[1], "pending": "0.75"}


# Figure 14.7: which check guards which risk (Lecture 16, Section 16.7) --
RISKS = [
    ("posterior width is the noise model's", ["stated, exercise"]),
    ("model cannot produce the real smile", ["prior predictive", "posterior predictive (S8)", "across models (S7)"]),
    ("forward-variance curve carries an error", ["stated, exercise"]),
    ("Monte Carlo noise in the audit likelihood", ["audit (S5)"]),
    ("the ρ–η valley makes the network's job hard", ["SBC (S2)", "audit (S5)", "pricer route (S4)"]),
    ("design grid is not the real chain", ["stated, exercise"]),
    ("products depend on the curve beyond the data", ["stated, exercise"]),
    ("classical critique sounds overstated", ["classical route (S6)", "pricer route (S4)"]),
    ("cost claim sounds overstated", ["costs (S9)"]),
]
CHECKS = ["prior predictive", "SBC (S2)", "pricer route (S4)", "audit (S5)", "classical route (S6)",
          "across models (S7)", "posterior predictive (S8)", "costs (S9)", "stated, exercise"]


# Figure 14.8: what each lecture contributes to the tutorial ------------------------
LECTURE_ACTS = {
    1: ["The question", "From posterior to prices"],
    2: ["The question", "GBM under Q"],
    3: ["GBM under Q"],
    4: ["GBM under Q"],
    5: ["The question", "GBM under Q", "Rough Bergomi"],
    6: ["Heston", "Across models"],
    7: ["Rough Bergomi", "The amortized posterior"],
    8: ["The question", "From posterior to prices"],
    9: ["The classical route"],
    10: ["The amortized posterior", "From posterior to prices", "Across models"],
    11: ["Rough Bergomi"],
    12: ["The amortized posterior", "Is it right?"],
    13: ["Heston", "Is it right?"],
    14: [a[2] for a in ACTS],
}


def main():
    fig_spy_curve()
    fig_products()
    fig_across_models()
    print(json.dumps(REPORT, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))


if __name__ == "__main__":
    main()
