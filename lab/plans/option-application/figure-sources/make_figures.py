"""Draw the figures of the option-application documentation.

Every figure comes from `data.json` (see `extract_data.py`) or from a computation stated in
its docstring with the tutorial runtime's Heston pricer. Run from the repository root:

    PYTHONPATH=tutorial/src python lab/plans/option-application/figure-sources/make_figures.py

Needs numpy, scipy and matplotlib (all in the lab environment).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scipy.stats import norm  # noqa: E402

from sbi4f_tutorial.heston_options import model  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "images"
DATA = json.loads((HERE / "data.json").read_text())

LABELS = {"initial_variance": "$v_0$", "kappa": r"$\kappa$", "theta": r"$\bar v$",
          "vol_of_vol": r"$\xi$", "rho": r"$\rho$", "c": "$c$"}
NAMES = DATA["parameters"]
PARAM_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
                     "legend.fontsize": 8, "figure.dpi": 100, "savefig.dpi": 170,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- design and Black-Scholes

def design():
    d = DATA["design"]
    out = {key: np.asarray(d[key]) for key in ("forward", "strike", "discount", "maturity",
                                               "mid", "half_spread", "target_z")}
    out["put"] = np.asarray(d["option_type"]) == "put"
    out["expiration"] = np.asarray(d["expiration"])
    out["z"] = np.log(out["strike"] / out["forward"]) / (0.16 * np.sqrt(out["maturity"]))
    return out


def black(forward, strike, discount, maturity, sigma, put):
    s = sigma * np.sqrt(maturity)
    d1 = (np.log(forward / strike) + 0.5 * s * s) / s
    call = discount * (forward * norm.cdf(d1) - strike * norm.cdf(d1 - s))
    return np.where(put, call - discount * (forward - strike), call)


def implied_vol(price, forward, strike, discount, maturity, put):
    out = np.empty(np.size(price))
    for i, args in enumerate(np.broadcast(price, forward, strike, discount, maturity, put)):
        p, f, k, dsc, t, pt = args
        try:
            out[i] = brentq(lambda s: float(black(f, k, dsc, t, s, pt)) - p, 1e-4, 5.0)
        except ValueError:
            out[i] = np.nan
    return out


def vega(forward, strike, discount, maturity, sigma):
    s = sigma * np.sqrt(maturity)
    d1 = (np.log(forward / strike) + 0.5 * s * s) / s
    return discount * forward * norm.pdf(d1) * np.sqrt(maturity)


# ---------------------------------------------------------------- Heston helpers

MARKET = dict(zip(NAMES, DATA["heston_300k"]["cases"]["market"]["reference_mean"]))


def heston_parameters(theta):
    return {"kappa": theta["kappa"], "theta": theta["theta"], "vol_of_vol": theta["vol_of_vol"],
            "rho": theta["rho"], "jump_intensity": 0.0, "jump_up_probability": 0.5,
            "jump_up_rate": 10.0, "jump_down_rate": 10.0}


def heston_prices(theta, forward, strike, discount, maturity, put):
    call = model.european_price(forward, strike, discount, maturity, heston_parameters(theta),
                                theta["initial_variance"], kind="call",
                                upper_limit=9600.0, nodes=16384)
    return np.where(put, call - discount * (forward - strike), call)


def expected_variance(theta, start, end):
    v0, kappa, vbar = theta["initial_variance"], theta["kappa"], theta["theta"]
    weight = np.exp(-kappa * start) * (-np.expm1(-kappa * (end - start))) / (kappa * (end - start))
    return vbar + (v0 - vbar) * weight


def maturity_colors(maturities):
    unique = np.unique(maturities)
    cmap = plt.get_cmap("viridis")
    return {t: cmap(0.08 + 0.84 * i / (len(unique) - 1)) for i, t in enumerate(unique)}


def expiry_label(d, t):
    m = d["maturity"] == t
    return f"{d['expiration'][m][0]} ({t * 365:.0f} d)"


# ---------------------------------------------------------------- figures

def fig_smile_design():
    """The 63 quotes: implied volatility against z by expiry, and the layout in (z, T)."""
    d = design()
    iv = implied_vol(d["mid"], d["forward"], d["strike"], d["discount"], d["maturity"], d["put"])
    hs = d["half_spread"] / vega(d["forward"], d["strike"], d["discount"], d["maturity"], iv)
    colors = maturity_colors(d["maturity"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
    for t in np.unique(d["maturity"]):
        m = d["maturity"] == t
        o = np.argsort(d["z"][m])
        a.plot(d["z"][m][o], 100 * iv[m][o], "o-", ms=3.5, lw=1.2, color=colors[t],
               label=expiry_label(d, t))
    a.set_xlabel("standardized moneyness $z = \\ln(K/F)/(0.16\\sqrt{T})$")
    a.set_ylabel("implied volatility (%)")
    a.set_title("The smile: SPX implied volatility at the 63 quotes")
    a.axvline(0, color="0.75", lw=0.8, zorder=0)
    a.legend(title="expiry (days to expiry)", frameon=False, ncol=1, fontsize=7.5, title_fontsize=8)
    days = d["maturity"] * 365
    sc = b.scatter(d["z"], days, c=100 * hs, cmap="magma_r", s=46, vmin=0, vmax=0.3,
                   marker="o", edgecolors="0.3", linewidths=0.4)
    for z, t, put, spread in zip(d["z"], days, d["put"], 100 * hs):
        b.annotate("P" if put else "C", (z, t), fontsize=5.5, ha="center", va="center",
                   color="white" if spread > 0.12 else "black")
    b.set_yscale("log")
    b.set_yticks([7, 14, 30, 60, 90, 180, 330])
    b.set_yticklabels(["1 w", "2 w", "1 m", "2 m", "3 m", "6 m", "11 m"])
    b.set_xlabel("standardized moneyness $z$")
    b.set_ylabel("time to expiry")
    b.set_title("The design: 9 places per expiry, puts (P) and calls (C)")
    cb = fig.colorbar(sc, ax=b, pad=0.02)
    cb.set_label("half-spread (vol points)")
    save(fig, "smile-and-design.png")


def fig_heston_misfit():
    """Market against Heston at the reference posterior mean of the market case."""
    d = design()
    iv = implied_vol(d["mid"], d["forward"], d["strike"], d["discount"], d["maturity"], d["put"])
    hs = d["half_spread"] / vega(d["forward"], d["strike"], d["discount"], d["maturity"], iv)
    model_price = heston_prices(MARKET, d["forward"], d["strike"], d["discount"], d["maturity"], d["put"])
    model_iv = implied_vol(model_price, d["forward"], d["strike"], d["discount"], d["maturity"], d["put"])
    residual = 100 * (iv - model_iv)
    colors = maturity_colors(d["maturity"])
    unique = np.unique(d["maturity"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 3.8))
    for t in unique[[0, 4, 6]]:
        m = d["maturity"] == t
        f, dsc = d["forward"][m][0], d["discount"][m][0]
        zz = np.linspace(d["z"][m].min() - 0.2, d["z"][m].max() + 0.2, 60)
        kk = f * np.exp(0.16 * np.sqrt(t) * zz)
        put = zz <= 0
        p = heston_prices(MARKET, f, kk, dsc, t, put)
        a.plot(zz, 100 * implied_vol(p, f, kk, dsc, t, put), "-", color=colors[t], lw=1.4)
        a.plot(d["z"][m], 100 * iv[m], "o", ms=4.5, color=colors[t], label=expiry_label(d, t))
    a.set_xlabel("standardized moneyness $z$")
    a.set_ylabel("implied volatility (%)")
    a.set_title("Market quotes (dots) and Heston's best fit (lines)")
    a.legend(frameon=False, title="expiry", title_fontsize=8)
    for t in unique:
        m = d["maturity"] == t
        o = np.argsort(d["z"][m])
        b.plot(d["z"][m][o], residual[m][o], "o-", ms=3, lw=1.1, color=colors[t],
               label=expiry_label(d, t))
    med = 100 * np.median(hs)
    b.axhspan(-med, med, color="#e45756", alpha=0.35, lw=0, label=f"± median half-spread ({med:.2f} vp)")
    b.axhline(0, color="0.4", lw=0.7)
    rms = np.sqrt(np.nanmean(residual ** 2))
    b.set_xlabel("standardized moneyness $z$")
    b.set_ylabel("market minus model (vol points)")
    b.set_title(f"Heston's misses: {rms:.1f} vol points RMS, smooth across strikes")
    b.legend(frameon=False, fontsize=6.8, ncol=2, loc="upper right")
    save(fig, "heston-misfit.png")
    # Smoothness of the misses: correlation of neighbouring residuals along strikes (within an
    # expiry, ordered by z) and across adjacent expiries (same target z).
    along, across = [], []
    for i, t in enumerate(unique):
        m = np.flatnonzero(d["maturity"] == t)
        r = residual[m[np.argsort(d["z"][m])]]
        along += list(zip(r[:-1], r[1:]))
        if i + 1 < len(unique):
            n = np.flatnonzero(d["maturity"] == unique[i + 1])
            for target in np.unique(d["target_z"][m]):
                a_i, b_i = m[d["target_z"][m] == target], n[d["target_z"][n] == target]
                if len(a_i) and len(b_i):
                    across.append((residual[a_i[0]], residual[b_i[0]]))
    along, across = np.array(along), np.array(across)
    return {"rms_vp": float(rms), "median_half_spread_vp": float(med),
            "max_abs_vp": float(np.nanmax(np.abs(residual))),
            "price_rms_in_half_spreads": float(np.sqrt(np.mean(((d["mid"] - model_price) / d["half_spread"]) ** 2))),
            "neighbour_strike_correlation": float(np.corrcoef(along.T)[0, 1]),
            "adjacent_expiry_correlation": float(np.corrcoef(across.T)[0, 1]),
            "rms_vp_by_expiry_days": {str(round(365 * t)): float(np.sqrt(np.mean(residual[d["maturity"] == t] ** 2)))
                                      for t in unique}}


def simulate_heston(theta, years=1.0, steps_per_day=4, n_paths=400, seed=7):
    """Full-truncation Euler paths of X = S/F and V; daily closes; forward drift from the design."""
    rng = np.random.default_rng(seed)
    days = int(round(252 * years))
    dt = 1.0 / (252 * steps_per_day)
    v = np.full(n_paths, theta["initial_variance"])
    logx = np.zeros(n_paths)
    rho = theta["rho"]
    closes_x, closes_v = [np.ones(n_paths)], [v.copy()]
    for _ in range(days):
        for _ in range(steps_per_day):
            w1, w2 = rng.standard_normal((2, n_paths))
            vp = np.maximum(v, 0.0)
            db = rho * w1 + np.sqrt(1 - rho * rho) * w2
            logx += -0.5 * vp * dt + np.sqrt(vp * dt) * db
            v = v + theta["kappa"] * (theta["theta"] - vp) * dt + theta["vol_of_vol"] * np.sqrt(vp * dt) * w1
        closes_x.append(np.exp(logx))
        closes_v.append(np.maximum(v, 0.0))
    return np.arange(days + 1) / 252, np.array(closes_x), np.array(closes_v)


def fig_products():
    """Payoffs on simulated paths (Heston at the market reference mean) and the expected variance."""
    d = design()
    spot = 7571.54
    growth = np.log(d["forward"][-1] / spot) / d["maturity"][-1]
    t, x, v = simulate_heston(MARKET)
    s = spot * np.exp(growth * t)[:, None] * x
    fig, axs = plt.subplots(2, 2, figsize=(10.2, 6.6))
    a, b, c, e = axs.ravel()
    i1 = len(t) // 2
    picks = [3, 11, 25, 40]
    label_y = []
    for j, p in enumerate(picks[:3]):
        col = plt.get_cmap("tab10")(j)
        a.plot(t, s[:, p], lw=1, color=col)
        strike = s[i1, p] * np.exp(growth * (t[-1] - t[i1]))
        a.plot(t[i1], s[i1, p], "o", color=col, ms=4)
        a.hlines(strike, t[i1], t[-1], color=col, ls=":", lw=1.3)
        payoff = max(s[-1, p] - strike, 0.0)
        y = s[-1, p]
        while any(abs(y - other) < 170 for other in label_y):
            y -= 170
        label_y.append(y)
        a.annotate(f"pays {payoff:,.0f}", (t[-1], s[-1, p]), xytext=(t[-1] + 0.02, y), textcoords="data",
                   fontsize=7.5, color=col, va="center")
    a.axvline(t[i1], color="0.5", lw=0.8, ls="--")
    a.plot([], [], ls=":", color="0.3", label="strike, set at 6 months (dots)")
    a.legend(frameon=False, loc="upper left", fontsize=7.5)
    a.set_xlim(-0.02, 1.15)
    a.set_title("Forward-start call: strike fixed in 6 months, paid at 1 year")
    a.set_ylabel("index level")
    returns = np.diff(np.log(s), axis=0)
    running = np.sqrt(252 * np.cumsum(returns ** 2, axis=0) / np.arange(1, len(t))[:, None])
    k_vol = np.mean(np.sqrt(np.mean(v[1:], axis=0)))
    for j, p in enumerate(picks):
        b.plot(t[10:], 100 * running[9:, p], lw=1, color=plt.get_cmap("tab10")(j))
    b.axhline(100 * k_vol, color="black", lw=1.2, ls="--", label=f"fair strike, about {100 * k_vol:.0f}%")
    b.set_ylim(0, 60)
    b.set_title("Volatility swap: pays realized volatility minus a fixed strike")
    b.set_ylabel("running realized volatility (%)")
    b.legend(frameon=False, loc="upper right")
    hit = np.where(np.min(s / spot, axis=0) <= 0.7)[0]
    calm = [p for p in range(s.shape[1]) if p not in set(hit)][:6]
    for p in calm:
        c.plot(t, 100 * s[:, p] / spot, color="0.7", lw=0.9)
    if hit.size:
        p = hit[0]
        c.plot(t, 100 * s[:, p] / spot, color="#d62728", lw=1.4, label="barrier hit: the put comes alive")
        first = np.argmax(s[:, p] / spot <= 0.7)
        c.plot(t[first], 100 * s[first, p] / spot, "o", color="#d62728")
    c.axhline(70, color="black", ls="--", lw=1, label="barrier, 70%")
    c.axhline(100, color="0.4", ls=":", lw=1, label="strike, 100%")
    c.set_title(f"Down-and-in put: exists only if a close falls to 70% ({hit.size / s.shape[1]:.0%} of paths)")
    c.set_ylabel("index, % of today")
    c.set_xlabel("years")
    c.legend(frameon=False, loc="lower left", fontsize=7.5)
    tt = np.linspace(0, 1, 300)
    ev = MARKET["theta"] + (MARKET["initial_variance"] - MARKET["theta"]) * np.exp(-MARKET["kappa"] * tt)
    e.plot(tt, 100 * np.sqrt(ev), color="black", lw=1.5, label=r"$\sqrt{\mathbb{E}[V_t]}$")
    for name, (start, end), col in (("short window", DATA["design"]["products"]["control"], "#4c78a8"),
                                    ("later window", DATA["design"]["products"]["affected"], "#f58518")):
        avg = expected_variance(MARKET, start, end)
        e.axvspan(start, end, color=col, alpha=0.18)
        e.hlines(100 * np.sqrt(avg), start, end, color=col, lw=2.5,
                 label=f"{name}: {100 * np.sqrt(avg):.1f}%")
    for tm in np.unique(d["maturity"]):
        e.plot(tm, 15.2, "|", color="0.3", ms=9)
    e.text(0.02, 15.6, "quoted expiries", fontsize=7, color="0.3")
    e.set_title("Forward-starting variance: expected variance averaged over a window")
    e.set_ylabel("volatility (%)")
    e.set_xlabel("years")
    e.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "products.png")


def fig_whitening():
    """Two-quote illustration of whitening, and the controlled comparison for c."""
    rng = np.random.default_rng(3)
    n, c = 600, 1.3
    common = rng.normal(0, 150, n)
    x = np.column_stack([common + c * rng.standard_normal(n), common + 0.9 * c * rng.standard_normal(n)])
    mu, cov = x.mean(0), np.cov(x, rowvar=False)
    lam, q = np.linalg.eigh(cov)
    u = (x - mu) @ (q / np.sqrt(lam)) @ q.T
    w = DATA["whitening"]
    truth = np.asarray(w["truth_c"])
    fig, axs = plt.subplots(1, 4, figsize=(11.6, 3.0))
    axs[0].scatter(x[:, 0], x[:, 1], s=3, color="0.35")
    axs[0].set_title("Two quotes across the prior")
    axs[0].set_xlabel("quote 1 / half-spread")
    axs[0].set_ylabel("quote 2 / half-spread")
    axs[1].scatter(u[:, 0], u[:, 1], s=3, color="#2ca02c")
    axs[1].set_aspect("equal")
    axs[1].set_title("Same inputs, whitened")
    axs[1].set_xlabel("whitened coordinate 1")
    axs[1].set_ylabel("whitened coordinate 2")
    for ax, key, title in ((axs[2], "diagonal_c_quantiles", "Scaled one input at a time"),
                           (axs[3], "zca_c_quantiles", "Whitened inputs")):
        q50 = np.asarray(w[key])[1]
        r = np.corrcoef(truth, q50)[0, 1]
        ax.plot([0.5, 2], [0.5, 2], color="0.6", lw=0.8)
        ax.scatter(truth, q50, s=4, color="#d62728" if key.startswith("diag") else "#2ca02c")
        ax.set_xlim(0.45, 2.05)
        ax.set_ylim(0.45, 2.05)
        ax.set_title(f"{title}: correlation {abs(r) if abs(r) < 0.005 else r:.2f}")
        ax.set_xlabel("true $c$")
        ax.set_ylabel("posterior median of $c$")
    fig.tight_layout()
    save(fig, "whitening.png")


def fig_heston_calibration():
    """Coverage against nominal level, and SBC rank ECDFs with the DKW band (300k network)."""
    h = DATA["heston_300k"]
    levels = sorted(float(k) for k in h["coverage"])
    cov = np.array([h["coverage"][str(k) if str(k) in h["coverage"] else repr(k)] for k in levels])
    n = h["population_rows"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 3.8))
    lv = np.array(levels)
    band = 2 * np.sqrt(lv * (1 - lv) / n)
    a.fill_between(lv, lv - band, lv + band, color="0.88", label="±2 binomial s.e. (512 cases)")
    a.plot([0.45, 1], [0.45, 1], color="0.5", lw=0.8)
    for j, name in enumerate(NAMES):
        a.plot(lv, cov[:, j], "o-", ms=3.5, color=PARAM_COLORS[j], label=LABELS[name])
    a.set_xlabel("nominal level of the central interval")
    a.set_ylabel("how often it contains the truth")
    a.set_title("Coverage on 512 simulated surfaces")
    a.legend(frameon=False, ncol=2)
    ranks = np.asarray(h["fractional_ranks"])
    grid = np.linspace(0, 1, 201)
    bound = h["sbc_bound"]
    b.fill_between(grid, -bound, bound, color="0.88", label="95% band for a calibrated posterior")
    for j, name in enumerate(NAMES):
        ecdf = np.searchsorted(np.sort(ranks[:, j]), grid, side="right") / len(ranks)
        b.plot(grid, ecdf - grid, color=PARAM_COLORS[j], lw=1.3, label=LABELS[name])
    b.axhline(0, color="0.5", lw=0.7)
    b.set_xlabel("fractional rank of the truth among posterior draws")
    b.set_ylabel("ECDF minus uniform")
    b.set_title("Simulation-based calibration: rank of the truth")
    b.legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, "heston-calibration.png")


def fig_heston_recovery():
    """True parameter against posterior median on the 512 simulated surfaces, 90% bars on 60."""
    h = DATA["heston_300k"]
    truth = np.asarray(h["truth"])
    q = np.asarray(h["quantiles_05_50_95"])
    corr = h["truth_median_correlation"]
    fig, axs = plt.subplots(2, 3, figsize=(10.2, 6.1))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(truth), 60, replace=False)
    for j, (ax, name) in enumerate(zip(axs.ravel(), NAMES)):
        lo, hi = DATA["prior_bounds"][j]
        ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8)
        ax.scatter(truth[:, j], q[1, :, j], s=3, color=PARAM_COLORS[j], alpha=0.7)
        ax.errorbar(truth[pick, j], q[1, pick, j], yerr=[q[1, pick, j] - q[0, pick, j],
                    q[2, pick, j] - q[1, pick, j]], fmt="none", ecolor="0.2", elinewidth=0.7)
        ax.set_title(f"{LABELS[name]}: correlation {corr[j]:.3f}")
        ax.set_xlabel("true value")
        ax.set_ylabel("posterior median")
    fig.tight_layout()
    save(fig, "heston-recovery.png")


def fig_heston_reference():
    """Network against reference posterior on the three test cases, in units of the reference width."""
    cases = DATA["heston_300k"]["cases"]
    titles = {"synthetic-broad": "Simulated surface, broad", "synthetic-market-like":
              "Simulated surface, market-like", "market": "SPX market quotes"}
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.6), sharey=True)
    for ax, key in zip(axs, ("synthetic-broad", "synthetic-market-like", "market")):
        case = cases[key]
        ref = np.asarray(case["reference_quantiles_05_50_95"])
        net = np.asarray(case["neural_quantiles_05_50_95"])
        width = ref[2] - ref[0]
        for j, name in enumerate(NAMES):
            y = len(NAMES) - 1 - j
            r = (ref - ref[1]) / width
            nn = (net - ref[1]) / width
            ax.plot([r[0, j], r[2, j]], [y + 0.12, y + 0.12], color="0.25", lw=3)
            ax.plot([nn[0, j], nn[2, j]], [y - 0.12, y - 0.12], color=PARAM_COLORS[j], lw=3)
            ax.plot(nn[1, j], y - 0.12, "o", color=PARAM_COLORS[j], ms=3)
        ax.axvline(0, color="0.7", lw=0.7)
        ax.set_title(titles[key])
        ax.set_xlabel("position, in reference 90% widths")
        if key == "market":
            span = np.max(np.abs((net - ref[1]) / width)) * 1.08
            ax.set_xlim(-span, span)
            ax.set_xlabel("position, in reference 90% widths (wider scale)")
        else:
            ax.set_xlim(-2.2, 2.2)
    axs[0].set_yticks(range(len(NAMES)))
    axs[0].set_yticklabels([LABELS[n] for n in NAMES[::-1]])
    axs[0].plot([], [], color="0.25", lw=3, label="reference (MCMC) 90%")
    axs[0].plot([], [], color="#1f77b4", lw=3, label="network 90%")
    axs[0].legend(frameon=False, loc="lower left", fontsize=7.5)
    fig.tight_layout()
    save(fig, "heston-vs-reference.png")


def fig_information_by_region():
    """Local information: rise of posterior s.d. when a region of quotes is removed.

    Heston at the market reference mean, the 63 design quotes, errors c*h with c = 1, flat
    prior; s.d. from the inverse Fisher information (J^T Sigma^-1 J)^-1, central differences.
    """
    d = design()
    keys = ["initial_variance", "kappa", "theta", "vol_of_vol", "rho"]
    base = {k: MARKET[k] for k in keys}
    steps = {"initial_variance": 1e-5, "kappa": 1e-3, "theta": 1e-5, "vol_of_vol": 1e-4, "rho": 1e-4}
    jac = []
    for k in keys:
        up, dn = dict(base), dict(base)
        up[k] += steps[k]
        dn[k] -= steps[k]
        jac.append((heston_prices(up, d["forward"], d["strike"], d["discount"], d["maturity"], d["put"])
                    - heston_prices(dn, d["forward"], d["strike"], d["discount"], d["maturity"], d["put"]))
                   / (2 * steps[k]))
    jac = np.array(jac).T
    weights = 1 / d["half_spread"] ** 2
    prod_grad = []
    for start, end in (DATA["design"]["products"]["control"], DATA["design"]["products"]["affected"]):
        g = []
        for k in keys:
            up, dn = dict(base), dict(base)
            up[k] += steps[k]
            dn[k] -= steps[k]
            g.append((expected_variance(up, start, end) - expected_variance(dn, start, end)) / (2 * steps[k]))
        prod_grad.append(g)
    prod_grad = np.array(prod_grad)

    def sds(mask):
        info = jac[mask].T @ (weights[mask, None] * jac[mask])
        cov = np.linalg.inv(info)
        return np.concatenate([np.sqrt(np.diag(cov)), np.sqrt(np.einsum("ij,jk,ik->i", prod_grad, cov, prod_grad))])

    full = sds(np.ones(len(jac), bool))
    t = d["maturity"]
    regions = {
        "deep puts\n$z \\leq -2.25$": d["target_z"] <= -2.25,
        "near puts\n$z$ = −1.25, −0.5": (d["target_z"] > -2.25) & (d["target_z"] < 0),
        "at the money\n$z = 0$": d["target_z"] == 0,
        "calls\n$z > 0$": d["target_z"] > 0,
        "expiries up to\n1 month": t <= 31 / 365,
        "expiries from\n2 months": t >= 59 / 365,
    }
    rise = np.array([100 * (sds(~mask) / full - 1) for mask in regions.values()])
    cols = [LABELS[k] for k in keys] + ["short-window\nvariance", "later-window\nvariance"]
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    shown = np.log10(1 + np.maximum(rise, 0))
    im = ax.imshow(shown, cmap="Oranges", aspect="auto", vmin=0, vmax=np.log10(1 + 1000))
    for i in range(rise.shape[0]):
        for j in range(rise.shape[1]):
            ax.text(j, i, f"+{rise[i, j]:.0f}%" if rise[i, j] < 200 else f"×{1 + rise[i, j] / 100:.1f}",
                    ha="center", va="center", fontsize=8, color="black" if shown[i, j] < 2.2 else "white")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(list(regions))
    ax.set_title("Rise in posterior standard deviation when one region of quotes is removed")
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, "information-by-region.png")
    return {"regions": [r.replace("\n", " ") for r in regions], "columns": [c.replace("\n", " ") for c in cols],
            "rise_percent": np.round(rise, 1).tolist()}


def fig_variance_strip():
    """The model-free variance swap: integrand 2 OTM(k) e^{-k} for the last expiry, Heston prices."""
    d = design()
    t = d["maturity"].max()
    m = d["maturity"] == t
    f, dsc = d["forward"][m][0], d["discount"][m][0]
    k = np.linspace(-1.6, 0.6, 400)
    strikes = f * np.exp(k)
    put = k <= 0
    price = heston_prices(MARKET, f, strikes, dsc, t, put) / (dsc * f)
    integrand = 2 * price * np.exp(-k)
    w = np.trapezoid(integrand, k)
    kvar = MARKET["theta"] + (MARKET["initial_variance"] - MARKET["theta"]) * (-np.expm1(-MARKET["kappa"] * t)) / (MARKET["kappa"] * t)
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.fill_between(k, 0, integrand, where=put, color="#4c78a8", alpha=0.35, label="puts ($k \\leq 0$)")
    ax.fill_between(k, 0, integrand, where=~put, color="#f58518", alpha=0.35, label="calls ($k > 0$)")
    ax.plot(k, integrand, color="black", lw=1.2)
    kd = np.log(d["strike"][m] / f)
    ax.plot(kd, np.interp(kd, k, integrand), "o", color="black", ms=4, label="the design's quotes")
    ax.set_xlabel("log-moneyness $k = \\ln(K/F)$")
    ax.set_ylabel("$2\\,\\mathrm{OTM}(k)\\,e^{-k}$")
    ax.set_title(f"Variance swap from the strip, {t * 365:.0f} days: "
                 f"strip {100 * np.sqrt(w / t):.2f}% vs formula {100 * np.sqrt(kvar):.2f}%")
    ax.legend(frameon=False)
    save(fig, "variance-swap-strip.png")
    return {"strip_vol": float(np.sqrt(w / t)), "formula_vol": float(np.sqrt(kvar))}


def fig_push_forward():
    """Schematic: posterior draws pushed through a product, with mark and prudent value."""
    rng = np.random.default_rng(11)
    cov = [[1.0, -0.8], [-0.8, 1.0]]
    draws = rng.multivariate_normal([0, 0], cov, 1000)
    value = 100 + 1.4 * draws[:, 0] - 0.6 * draws[:, 1] + 0.25 * draws[:, 0] ** 2
    mark, q10 = value.mean(), np.quantile(value, 0.10)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.3))
    a.scatter(draws[:, 0], draws[:, 1], s=3, color="#4c78a8", alpha=0.6)
    a.set_xlabel("parameter 1")
    a.set_ylabel("parameter 2")
    a.set_title("Posterior draws: each is one complete model")
    b.hist(value, bins=45, color="#9ecae1", edgecolor="white")
    b.axvline(mark, color="black", lw=1.5, label=f"mark (mean) = {mark:.1f}")
    b.axvline(q10, color="#d62728", lw=1.5, ls="--", label=f"prudent value (10% quantile) = {q10:.1f}")
    b.annotate("", xy=(q10, b.get_ylim()[1] * 0.9), xytext=(mark, b.get_ylim()[1] * 0.9),
               arrowprops={"arrowstyle": "<->", "color": "0.3"})
    b.text((mark + q10) / 2, b.get_ylim()[1] * 0.93, "reserve", ha="center", fontsize=8)
    b.set_xlabel("product value under each draw")
    b.set_title("Pushed through the pricer: a distribution of values")
    b.legend(frameon=False, loc="upper right", fontsize=7.5)
    fig.suptitle("Schematic", x=0.02, ha="left", fontsize=8, color="0.4")
    fig.tight_layout()
    save(fig, "posterior-to-product.png")


def fig_costs():
    """CPU time for one posterior: measured (Heston MCMC, network) or derived from measured pricing (rough)."""
    rows = [("Network, after training", 164 / 615, "under 1 second"),
            ("Heston, MCMC reference", 898, "7–15 minutes (measured)"),
            ("Rough Bergomi, importance sampling", 108 * 1000, "16–30 hours (1,000 pricings\nof 1–2 minutes each)")]
    fig, ax = plt.subplots(figsize=(8.4, 2.4))
    colors = ["#2ca02c", "#4c78a8", "#f58518"]
    for i, (label, seconds, text) in enumerate(rows):
        ax.barh(i, seconds, left=0.05, color=colors[i], height=0.55)
        ax.text(seconds * 1.4, i, text, va="center", fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 3e7)
    ax.set_xticks([1, 60, 3600, 86400, 86400 * 30])
    ax.set_xticklabels(["1 second", "1 minute", "1 hour", "1 day", "1 month"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_title("CPU time for one posterior")
    save(fig, "posterior-costs.png")


def tau_information():
    """Precision of a smooth model-error size tau on the 63 quotes (tau-only Fisher information).

    Vol-point errors with covariance c^2 diag(h^2) + tau^2 K, K squared-exponential in (z, ln T);
    also the variance inflation of a level shift when smooth errors are treated as independent.
    """
    d = design()
    iv = implied_vol(d["mid"], d["forward"], d["strike"], d["discount"], d["maturity"], d["put"])
    hv = d["half_spread"] / vega(d["forward"], d["strike"], d["discount"], d["maturity"], iv)
    lt = np.log(d["maturity"])
    out = []
    for lz, lt_len in ((0.75, 1.15), (1.6, 2.0)):
        kern = np.exp(-(d["z"][:, None] - d["z"][None, :]) ** 2 / (2 * lz ** 2)
                      - (lt[:, None] - lt[None, :]) ** 2 / (2 * lt_len ** 2))
        inflation = float(np.ones(len(kern)) @ kern @ np.ones(len(kern)) / len(kern))
        for tau_vp in (1.0, 2.5):
            tau = tau_vp / 100
            inv = np.linalg.inv(np.diag(hv ** 2) + tau ** 2 * kern)
            grad = 2 * tau * kern
            info = 0.5 * np.trace(inv @ grad @ inv @ grad)
            rel = float(1 / np.sqrt(info) / tau)
            out.append({"lengths": [lz, lt_len], "tau_vp": tau_vp, "relative_sd": round(rel, 4),
                        "effective_directions": round(1 / (2 * rel ** 2), 1),
                        "level_shift_sd_factor": round(float(np.sqrt(inflation)), 2)})
    return out


def main():
    numbers = {}
    numbers["tau_information"] = tau_information()
    fig_smile_design()
    numbers["misfit"] = fig_heston_misfit()
    fig_products()
    fig_whitening()
    fig_heston_calibration()
    fig_heston_recovery()
    fig_heston_reference()
    numbers["information_by_region"] = fig_information_by_region()
    numbers["variance_strip"] = fig_variance_strip()
    fig_push_forward()
    fig_costs()
    (HERE / "numbers.json").write_text(json.dumps(numbers, indent=1) + "\n")
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
