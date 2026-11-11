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

LABELS = {"initial_variance": "$v_0$", "kappa": r"$\kappa$", "theta": r"$v_\infty$",
          "vol_of_vol": r"$\sigma_v$", "rho": r"$\rho$", "c": "$c$"}
SIGMA_REF = DATA["design"]["reference_volatility"]  # 0.16: the scale of the standardized moneyness delta
CASE_TITLES = {"synthetic-market-like": "Simulated quotes, market-like", "synthetic-broad":
               "Simulated quotes, stressed market", "market": "SPX quotes, 9 November 2026"}
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
    out["delta"] = np.log(out["strike"] / out["forward"]) / (SIGMA_REF * np.sqrt(out["maturity"]))
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


# ---------------------------------------------------------------- simulation from the prior

LADDER = ((256, 150.0), (512, 300.0), (1024, 600.0), (2048, 1200.0), (4096, 2400.0),
          (8192, 4800.0), (16384, 9600.0))  # Gauss-Legendre nodes and cutoff: the study's quadrature ladder
_RUNGS = {}


def _rung(index, d):
    if index not in _RUNGS:
        nodes, cutoff = LADDER[index]
        x, w = np.polynomial.legendre.leggauss(nodes)
        u = 0.5 * cutoff * (x + 1)
        _RUNGS[index] = (u, 0.5 * cutoff * w / (u * u + 0.25),
                         np.exp(1j * u[None, :] * np.log(d["forward"] / d["strike"])[:, None]))
    return _RUNGS[index]


def heston_surface(theta, d):
    """The 63 Heston prices for theta = (v0, kappa, v_inf, sigma_v, rho, ...).

    Lewis integral with the Black control variate (integrand: Black minus Heston characteristic
    function), refined on the ladder until two rungs agree within 0.01 half-spreads on every quote.
    """
    v0, kappa, v_inf, sigma_v, rho = theta[:5]
    pars = heston_parameters({"kappa": kappa, "theta": v_inf, "vol_of_vol": sigma_v, "rho": rho})
    maturities, index = np.unique(d["maturity"], return_inverse=True)
    total = v_inf * maturities + (v0 - v_inf) * (-np.expm1(-kappa * maturities)) / kappa
    base = black(d["forward"], d["strike"], d["discount"], d["maturity"],
                 np.sqrt(total[index] / d["maturity"]), d["put"])
    scale = d["discount"] * np.sqrt(d["forward"] * d["strike"]) / np.pi
    previous = None
    for r in range(1, len(LADDER)):
        u, weights, phase = _rung(r, d)
        phi_heston = model.characteristic_function(u[None, :] - 0.5j, maturities[:, None], pars, v0)
        phi_black = np.exp(-0.5 * total[:, None] * (u[None, :] ** 2 + 0.25))
        price = base + scale * (np.real(phase * (phi_black - phi_heston)[index]) @ weights)
        if previous is not None and np.max(np.abs(price - previous) / d["half_spread"]) <= 0.01:
            return previous
        previous = price
    raise ValueError("quadrature ladder exhausted")


def prior_simulation(n=3000, seed=20261111):
    """n draws of theta from the prior, their 63 clean prices, and x = y / h with y = P + c h eps."""
    d = design()
    rng = np.random.default_rng(seed)
    bounds = np.asarray(DATA["prior_bounds"])
    theta = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, 6))
    clean = np.array([heston_surface(row, d) for row in theta])
    y = clean + theta[:, 5:6] * d["half_spread"] * rng.standard_normal(clean.shape)
    return {"theta": theta, "clean": clean, "x": y / d["half_spread"]}


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
        o = np.argsort(d["delta"][m])
        a.plot(d["delta"][m][o], 100 * iv[m][o], "o-", ms=3.5, lw=1.2, color=colors[t],
               label=expiry_label(d, t))
    a.set_xlabel("standardized moneyness $\\delta = \\ln(K/F)/(\\sigma_{\\mathrm{ref}}\\sqrt{T})$, $\\sigma_{\\mathrm{ref}}$ = 16%")
    a.set_ylabel("implied volatility (%)")
    a.set_title("The smile: SPX implied volatility at the 63 quotes")
    a.axvline(0, color="0.75", lw=0.8, zorder=0)
    a.legend(title="expiry (days to expiry)", frameon=False, ncol=1, fontsize=7.5, title_fontsize=8)
    days = d["maturity"] * 365
    sc = b.scatter(d["delta"], days, c=100 * hs, cmap="magma_r", s=46, vmin=0, vmax=0.3,
                   marker="o", edgecolors="0.3", linewidths=0.4)
    for z, t, put, spread in zip(d["delta"], days, d["put"], 100 * hs):
        b.annotate("P" if put else "C", (z, t), fontsize=5.5, ha="center", va="center",
                   color="white" if spread > 0.12 else "black")
    b.set_yscale("log")
    b.set_yticks([7, 14, 30, 60, 90, 180, 330])
    b.set_yticklabels(["1 w", "2 w", "1 m", "2 m", "3 m", "6 m", "11 m"])
    b.set_xlabel("standardized moneyness $\\delta$")
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
        zz = np.linspace(d["delta"][m].min() - 0.2, d["delta"][m].max() + 0.2, 60)
        kk = f * np.exp(0.16 * np.sqrt(t) * zz)
        put = zz <= 0
        p = heston_prices(MARKET, f, kk, dsc, t, put)
        a.plot(zz, 100 * implied_vol(p, f, kk, dsc, t, put), "-", color=colors[t], lw=1.4)
        a.plot(d["delta"][m], 100 * iv[m], "o", ms=4.5, color=colors[t], label=expiry_label(d, t))
    a.set_xlabel("standardized moneyness $\\delta$")
    a.set_ylabel("implied volatility (%)")
    a.set_title("Market quotes (dots) and Heston's best fit (lines)")
    a.legend(frameon=False, title="expiry", title_fontsize=8)
    for t in unique:
        m = d["maturity"] == t
        o = np.argsort(d["delta"][m])
        b.plot(d["delta"][m][o], residual[m][o], "o-", ms=3, lw=1.1, color=colors[t],
               label=expiry_label(d, t))
    med = 100 * np.median(hs)
    b.axhspan(-med, med, color="#e45756", alpha=0.35, lw=0, label=f"± median half-spread ({med:.2f} vp)")
    b.axhline(0, color="0.4", lw=0.7)
    rms = np.sqrt(np.nanmean(residual ** 2))
    b.set_xlabel("standardized moneyness $\\delta$")
    b.set_ylabel("market minus model (vol points)")
    b.set_title(f"Heston's misses: {rms:.1f} vol points RMS, smooth across strikes")
    b.legend(frameon=False, fontsize=6.8, ncol=2, loc="upper right")
    save(fig, "heston-misfit.png")
    # Smoothness of the misses: correlation of neighbouring residuals along strikes (within an
    # expiry, ordered by z) and across adjacent expiries (same target z).
    along, across = [], []
    for i, t in enumerate(unique):
        m = np.flatnonzero(d["maturity"] == t)
        r = residual[m[np.argsort(d["delta"][m])]]
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


def fig_whitening_comparison():
    """Controlled comparison (54-quote design, same network and data): c recovered with and without whitening."""
    w = DATA["whitening"]
    truth = np.asarray(w["truth_c"])
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    for ax, key, title, color in ((axs[0], "diagonal_c_quantiles", "Each input scaled on its own", "#d62728"),
                                  (axs[1], "zca_c_quantiles", "Whitened inputs", "#2ca02c")):
        q05, q50, q95 = np.asarray(w[key])
        r = np.corrcoef(truth, q50)[0, 1]
        shown = np.arange(0, len(truth), 9)
        ax.plot([0.5, 2], [0.5, 2], color="0.6", lw=0.8)
        ax.errorbar(truth[shown], q50[shown], yerr=[q50[shown] - q05[shown], q95[shown] - q50[shown]],
                    fmt="none", ecolor=color, alpha=0.35, lw=0.8)
        ax.scatter(truth, q50, s=5, color=color, zorder=3)
        ax.set_xlim(0.45, 2.05)
        ax.set_ylim(0.3, 2.2)
        ax.set_title(f"{title}\ncorrelation of truth and median: {abs(r) if abs(r) < 0.005 else r:.2f}")
        ax.set_xlabel("true $c$")
    axs[0].set_ylabel("posterior median of $c$, with 90% intervals")
    fig.tight_layout()
    save(fig, "whitening-comparison.png")


def fig_whitening_spectrum(sim):
    """Why the inputs are whitened, on the prior simulation of the 63-quote design.

    mu and Sigma are fitted on the first 2,000 simulated inputs x = y/h; the other panels use the
    remaining 1,000. c_hat is the root mean square of the coordinates of x - mu along the principal
    directions whose variance is below twice E[c^2] = 1.75, the variance of the quote errors.
    """
    x, c = sim["x"], sim["theta"][:, 5]
    fit, held = slice(0, 2000), slice(2000, None)
    mu = x[fit].mean(axis=0)
    lam, vec = np.linalg.eigh(np.cov(x[fit], rowvar=False))
    lam, vec = lam[::-1], vec[:, ::-1]
    coords = (x[held] - mu) @ vec                      # coordinates along the principal directions
    white = coords / np.sqrt(lam) @ vec.T              # ZCA-whitened inputs
    noise = 1.75
    low = lam < 2 * noise
    c_hat = np.sqrt(np.mean(coords[:, low] ** 2, axis=1))
    k = np.arange(1, len(lam) + 1)
    fig, axs = plt.subplots(1, 4, figsize=(13.2, 3.3))
    axs[0].semilogy(k[~low], lam[~low], "o", ms=3.5, color="#4c78a8", label="set by the parameters")
    axs[0].semilogy(k[low], lam[low], "o", ms=3.5, color="#f58518", label=f"quote errors only ({low.sum()} directions)")
    axs[0].axhline(noise, color="0.4", ls="--", lw=0.9)
    axs[0].text(2, noise * 1.6, r"$\mathbb{E}[c^2] = 1.75$", fontsize=8, color="0.3")
    axs[0].set_xlabel("principal direction $k$")
    axs[0].set_ylabel(r"variance $\lambda_k$")
    axs[0].set_title(r"Variance of $x = y/h$ by direction")
    axs[0].legend(frameon=False, fontsize=7.5, loc="upper right")
    for ax, values, title in ((axs[1], x[held], "|correlation| of the 63 inputs"),
                              (axs[2], white, "after whitening")):
        image = ax.imshow(np.abs(np.corrcoef(values, rowvar=False)), vmin=0, vmax=1, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("quote $i$")
        ax.set_ylabel("quote $j$")
    fig.colorbar(image, ax=axs[2], shrink=0.8, pad=0.03)
    axs[3].plot([0.5, 2], [0.5, 2], color="0.6", lw=0.8)
    axs[3].scatter(c[held], c_hat, s=5, color="#f58518")
    axs[3].set_xlabel("true $c$")
    axs[3].set_ylabel(r"$\hat c$: size of $x$ along the orange directions")
    axs[3].set_title("The error scale is in the small directions")
    fig.tight_layout()
    save(fig, "whitening-spectrum.png")
    scaled = np.linalg.eigvalsh(np.corrcoef(x[fit], rowvar=False))
    return {"fit_rows": 2000, "held_out_rows": int(len(c[held])),
            "largest_variance": float(lam[0]), "smallest_variance": float(lam[-1]),
            "condition_number": float(lam[0] / lam[-1]),
            "condition_number_each_input_scaled": float(scaled[-1] / scaled[0]),
            "share_of_variance_in_first_three_directions": float(lam[:3].sum() / lam.sum()),
            "directions_below_twice_noise_variance": int(low.sum()),
            "mean_variance_of_those_directions": float(lam[low].mean()),
            "c_hat_correlation_with_c": float(np.corrcoef(c[held], c_hat)[0, 1]),
            "c_hat_relative_rms_error": float(np.sqrt(np.mean((c_hat / c[held] - 1) ** 2))),
            "largest_abs_correlation_after_whitening": float(np.max(np.abs(
                np.corrcoef(white, rowvar=False) - np.eye(len(lam)))))}


def fig_prior_predictive(sim):
    """150 smiles simulated from the prior (clean prices) against the market, at three expiries."""
    d = design()
    market_iv = implied_vol(d["mid"], d["forward"], d["strike"], d["discount"], d["maturity"], d["put"])
    unique = np.unique(d["maturity"])
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.2), sharey=True)
    for ax, t in zip(axs, unique[[0, 4, 6]]):
        m = np.flatnonzero(d["maturity"] == t)
        m = m[np.argsort(d["delta"][m])]
        for row in sim["clean"][:150]:
            iv = implied_vol(row[m], d["forward"][m], d["strike"][m], d["discount"][m], d["maturity"][m], d["put"][m])
            ax.plot(d["delta"][m], 100 * iv, color="#4c78a8", alpha=0.13, lw=0.8)
        ax.plot(d["delta"][m], 100 * market_iv[m], "o-", color="#f58518", ms=3.5, lw=1.6, label="market")
        ax.set_title(expiry_label(d, t))
        ax.set_xlabel(r"standardized moneyness $\delta$")
    axs[0].set_ylabel("implied volatility (%)")
    axs[0].set_ylim(0, 75)
    axs[0].plot([], [], color="#4c78a8", lw=0.8, label="150 draws from the prior")
    axs[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "prior-predictive.png")
    rank = (sim["x"] < (d["mid"] / d["half_spread"])[None, :]).mean(axis=0)
    return {"simulated_data_sets": int(len(sim["x"])),
            "market_quote_percentile_min": float(100 * rank.min()),
            "market_quote_percentile_max": float(100 * rank.max())}


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
    titles = CASE_TITLES
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.6), sharey=True)
    for ax, key in zip(axs, ("synthetic-market-like", "synthetic-broad", "market")):
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
        "deep puts\n$\\delta \\leq -2.25$": d["target_z"] <= -2.25,
        "near puts\n$\\delta$ = −1.25, −0.5": (d["target_z"] > -2.25) & (d["target_z"] < 0),
        "at the money\n$\\delta = 0$": d["target_z"] == 0,
        "calls\n$\\delta > 0$": d["target_z"] > 0,
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
    """The exact SPX posterior of Heston (1,000 reference draws) and what it implies for one product.

    Left: draws of (kappa, v_inf), coloured by the later-window variance strike. Right: the strike
    under every draw, as a volatility, with the mark, the 5-95% range and the prudent value.
    """
    case = DATA["references_regenerated"]["market"]
    draws = np.asarray(case["draws"])
    window = DATA["design"]["products"]["affected"]
    strike = 100 * np.sqrt(np.asarray(case["window_variance_draws"])[:, 1])
    mark, q05, q10, q95 = strike.mean(), *np.quantile(strike, [0.05, 0.10, 0.95])
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.0, 3.5), gridspec_kw={"width_ratios": [1, 1.15]})
    sc = a.scatter(draws[:, 1], 100 * np.sqrt(draws[:, 2]), c=strike, s=6, cmap="viridis")
    fig.colorbar(sc, ax=a, pad=0.02).set_label("variance strike of each draw (%)")
    a.set_xlabel(r"mean reversion $\kappa$")
    a.set_ylabel(r"long-run volatility $\sqrt{v_\infty}$ (%)")
    a.set_title(r"Posterior draws $\theta^{(s)}$: each is one complete model")
    b.hist(strike, bins=40, color="#9ecae1", edgecolor="white")
    top = b.get_ylim()[1]
    b.axvspan(q05, q95, color="0.85", alpha=0.6, lw=0, label=f"90% range: {q05:.2f}% to {q95:.2f}%")
    b.axvline(mark, color="black", lw=1.5, label=f"mark (mean): {mark:.2f}%")
    b.axvline(q10, color="#d62728", lw=1.5, ls="--", label=f"prudent value, long (10% quantile): {q10:.2f}%")
    b.annotate("", xy=(q10, top * 0.86), xytext=(mark, top * 0.86), arrowprops={"arrowstyle": "<->", "color": "0.3"})
    b.text((mark + q10) / 2, top * 0.89, "reserve", ha="center", fontsize=8)
    b.set_ylim(0, top * 1.45)
    b.set_xlabel(f"variance strike, {365 * window[0]:.0f} to {365 * window[1]:.0f} days, as volatility (%)")
    b.set_title(r"The same draws through the product: $g(\theta^{(s)})$")
    b.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.95, loc="upper left", fontsize=7.5)
    fig.tight_layout()
    save(fig, "posterior-to-product.png")
    short = 100 * np.sqrt(np.asarray(case["window_variance_draws"])[:, 0])
    summary = lambda v: {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "q05": float(np.quantile(v, 0.05)),
                         "q10": float(np.quantile(v, 0.10)), "q95": float(np.quantile(v, 0.95)),
                         "reserve_long": float(v.mean() - np.quantile(v, 0.10))}
    corr = np.corrcoef(draws[:, :5], rowvar=False)
    i, j = max(((a_, b_) for a_ in range(5) for b_ in range(a_)), key=lambda q: abs(corr[q]))
    return {"draws": int(len(strike)), "short_window_volatility_percent": summary(short),
            "later_window_volatility_percent": summary(strike),
            "strongest_parameter_correlation": {"pair": [NAMES[j], NAMES[i]], "value": float(corr[i, j])}}


def fig_costs():
    """CPU time for one posterior, from the number of pricings of the 63 quotes each method needs.

    Heston MCMC: measured (accepted runs of the regenerated references). Network: 615 posteriors in
    164 s. Rough Bergomi: pricings times the measured 1-2 CPU-minutes per pricing (derived, not run).
    """
    runs = [case["accepted_run"] for case in DATA["references_regenerated"].values()]
    evaluations = int(round(np.mean([r["likelihood_evaluations"] for r in runs]), -3))
    low, high = min(r["cpu_seconds"] for r in runs), max(r["cpu_seconds"] for r in runs)
    rows = [("Network, after training\n(no pricing)", 164 / 615, 164 / 615, "0.3 seconds", "#2ca02c"),
            (f"Heston, MCMC\n({evaluations:,} pricings)", low, high, f"{low / 60:.0f}–{high / 60:.0f} minutes (measured)", "#4c78a8"),
            ("Rough Bergomi, importance sampling\nfrom the network (1,000 pricings)", 1000 * 60, 1000 * 105,
             "17–29 hours", "#f58518"),
            (f"Rough Bergomi, MCMC\n({evaluations:,} pricings; not run)", evaluations * 60, evaluations * 105,
             "100–180 days", "#e45756")]
    fig, ax = plt.subplots(figsize=(9.0, 3.0))
    for i, (label, lo, hi, text, color) in enumerate(rows):
        ax.barh(i, hi, left=0.05, color=color, height=0.55, alpha=0.45)
        ax.barh(i, lo, left=0.05, color=color, height=0.55)
        ax.text(hi * 1.5, i, text, va="center", fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 2e9)
    ax.set_xticks([1, 60, 3600, 86400, 86400 * 30, 86400 * 365])
    ax.set_xticklabels(["1 second", "1 minute", "1 hour", "1 day", "1 month", "1 year"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_title("CPU time for one posterior")
    save(fig, "posterior-costs.png")
    return {"mcmc_likelihood_evaluations": [r["likelihood_evaluations"] for r in runs],
            "mcmc_cpu_minutes_accepted_run": [round(r["cpu_seconds"] / 60, 1) for r in runs],
            "mcmc_cpu_minutes_first_run": [round(case["first_run"]["cpu_seconds"] / 60, 1)
                                           for case in DATA["references_regenerated"].values()],
            "rough_mcmc_cpu_days": [round(evaluations * 60 / 86400), round(evaluations * 105 / 86400)]}


def fig_flow_matching():
    """Flow matching on a one-parameter toy, with the exact velocity field (no network).

    theta ~ U(-2, 2), y = theta^3 / 4 + 0.3 eps. As in BayesFlow: z_t = (1 - t) theta + t z_1 with
    noise z_1 ~ N(0, 1) at t = 1, and v(z, t, y) = E[z_1 - theta | z_t = z, y] = (z - E[theta | z_t = z, y]) / t,
    computed by quadrature on a grid of theta. Sampling integrates dz/dt = v from t = 1 to t = 0.01.
    """
    sigma, grid = 0.3, np.linspace(-2, 2, 1201)

    def log_posterior(y):
        lp = -(y - grid ** 3 / 4) ** 2 / (2 * sigma ** 2)
        return lp - lp.max() - np.log(np.trapezoid(np.exp(lp - lp.max()), grid))

    def velocity(z, t, lp):
        logw = lp[None, :] - (z[:, None] - (1 - t) * grid[None, :]) ** 2 / (2 * t * t)
        w = np.exp(logw - logw.max(axis=1, keepdims=True))
        return (z - (w * grid).sum(axis=1) / w.sum(axis=1)) / t

    times = np.linspace(1, 0.01, 160)
    rng = np.random.default_rng(5)
    cases = {1.0: "#d62728", -0.2: "#1f77b4"}
    paths, ends, report = {}, {}, {}
    for y in cases:
        lp = log_posterior(y)
        z = rng.standard_normal(1500)
        track = [z.copy()]
        for a, b in zip(times[:-1], times[1:]):          # Heun steps from t = a down to t = b
            k1 = velocity(z, a, lp)
            z = z + 0.5 * (b - a) * (k1 + velocity(z + (b - a) * k1, b, lp))
            track.append(z.copy())
        paths[y], ends[y] = np.array(track), z / (1 - times[-1])
        density = np.exp(lp)
        exact_mean = np.trapezoid(grid * density, grid)
        exact_sd = np.sqrt(np.trapezoid((grid - exact_mean) ** 2 * density, grid))
        report[str(y)] = {"exact_mean": float(exact_mean), "exact_sd": float(exact_sd),
                          "flow_mean": float(ends[y].mean()), "flow_sd": float(ends[y].std())}

    theta = rng.uniform(-2, 2, 400)                       # training pairs: theta, its y, independent noise
    y_sim = theta ** 3 / 4 + sigma * rng.standard_normal(400)
    noise = rng.standard_normal(400)
    shown = {y: np.flatnonzero(np.abs(y_sim - y) < 0.12)[:10] for y in cases}   # ten pairs near each observation
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12.0, 3.7), sharey=True, gridspec_kw={"width_ratios": [1, 1, 0.62]})
    for n in range(150):
        a.plot([1, 0], [noise[n], theta[n]], color="0.78", lw=0.4, zorder=1)
    for y, color in cases.items():
        for n in shown[y]:
            a.plot([1, 0], [noise[n], theta[n]], color=color, lw=1.1, zorder=3)
    a.set_title("Training pairs: straight lines from noise $z_1$\nto a simulated parameter $\\theta$")
    for y, color in cases.items():
        b.plot(times, paths[y][:, :22], color=color, lw=0.7, alpha=0.75)
        c.hist(ends[y], bins=np.linspace(-2.1, 2.1, 64), density=True, orientation="horizontal", color=color, alpha=0.35)
        c.plot(np.exp(log_posterior(y)), grid, color=color, lw=1.4, label=f"exact posterior, $y$ = {y:g}")
    b.set_title("Sampling: follow the averaged velocity\nfrom $t = 1$ to $t = 0$, for two observations $y$")
    c.set_title("End points $z_0$ (bars) against\nthe exact posteriors (lines)")
    c.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, fontsize=7.5, loc="lower right")
    c.set_xlabel("density")
    for ax in (a, b):
        ax.set_xlim(1.02, -0.02)
        ax.set_xlabel("time $t$  (noise at $t = 1$, posterior at $t = 0$)")
    a.set_ylabel(r"flow state $z_t$  (at $t = 0$: the parameter $\theta$)")
    a.set_ylim(-3, 3)
    for y, color in cases.items():
        a.plot([], [], color=color, lw=1.2, label=f"pairs with $y$ near {y:g}")
    a.plot([], [], color="0.7", lw=0.8, label="other pairs")
    a.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    save(fig, "flow-matching.png")
    return report


def fig_network():
    """What the estimator takes in and returns, at training and at use (diagram)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12.6, 5.1))
    ax.set_xlim(0, 126)
    ax.set_ylim(3, 54)
    ax.axis("off")

    def box(x, y, w, h, title, body, face="#eef4fb", edge="#4c78a8"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=face, ec=edge, lw=1.1))
        ax.text(x + w / 2, y + h - 2.3, title, ha="center", va="top", fontsize=9, fontweight="bold")
        ax.text(x + w / 2, y + h - 5.6, body, ha="center", va="top", fontsize=8.2, linespacing=1.45)

    def arrow(x0, y0, x1, y1, color="0.35", style="-|>", rad=0.0):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=11, color=color, lw=1.1,
                                     connectionstyle=f"arc3,rad={rad}"))

    ax.text(0.5, 52.5, "Training: once per model and design (the 63 contracts and their half-spreads)", fontsize=10.5,
            fontweight="bold", va="top")
    top, height, width = 31.5, 15.5, 21.6
    training = [("Prior", "draw $\\theta^{(n)} \\sim p(\\theta)$\n6 numbers:\n$v_0, \\kappa, v_\\infty, \\sigma_v, \\rho, c$"),
                ("Pricer", "$P(\\theta^{(n)})$\n63 model prices\n(Heston: Fourier integral)"),
                ("Error model", "$y^{(n)}_i = P_i(\\theta^{(n)}) + c^{(n)} h_i\\,\\varepsilon^{(n)}_i$\n63 simulated quotes"),
                ("Bank", "$N$ = 300,000 pairs\n$(\\theta^{(n)},\\ y^{(n)})$"),
                ("Fit", "whitening $(\\mu, W)$\nfrom 10,000 rows;\nvelocity network $v_\\phi$\nby flow matching")]
    for i, (title, body) in enumerate(training):
        x = 0.8 + i * 25.1
        box(x, top, width, height, title, body, face="#fdf1e3" if i == 4 else "#eef4fb", edge="#f58518" if i == 4 else "#4c78a8")
        if i:
            arrow(x - 3.5, top + height / 2, x - 0.4, top + height / 2)

    ax.text(0.5, 25.6, "Use: any quotes of these contracts, 0.3 s, no pricing", fontsize=10.5,
            fontweight="bold", va="top")
    bottom = 4.5
    use = [("Quotes", "$y \\in \\mathbb{R}^{63}$\nmid prices of the 63 options"),
           ("Scale and whiten", "$x = y / h$\n$\\tilde{x} = W\\,(x - \\mu) \\in \\mathbb{R}^{63}$"),
           ("Flow", "draw $z_1 \\sim \\mathcal{N}(0, I_6)$; solve\n$dz_t/dt = v_\\phi(z_t,\\ t,\\ \\tilde{x})$\nfrom $t = 1$ to $t = 0$"),
           ("Back to parameter units", "$\\theta = m_\\theta + s_\\theta \\odot z_0$\n($m_\\theta$, $s_\\theta$: mean and s.d. of\neach parameter over the bank)"),
           ("Posterior draws", "$\\theta^{(1)}, \\dots, \\theta^{(S)} \\sim q_\\phi(\\theta \\mid y)$\none per noise draw $z_1$")]
    for i, (title, body) in enumerate(use):
        x = 0.8 + i * 25.1
        box(x, bottom, width, height, title, body, face="#e8f5e9" if i == 4 else "#eef4fb", edge="#2ca02c" if i == 4 else "#4c78a8")
        if i:
            arrow(x - 3.5, bottom + height / 2, x - 0.4, bottom + height / 2)
    fit_x = 0.8 + 4 * 25.1 + width / 2
    for offset, column, label in ((-5, 1, "$\\mu,\\ W$"), (0, 2, "$\\phi$"), (5, 3, "$m_\\theta,\\ s_\\theta$")):
        x0, y0 = fit_x + offset, top - 0.7
        x1, y1 = 0.8 + column * 25.1 + width / 2 + 4, bottom + height + 0.7
        arrow(x0, y0, x1, y1, color="#f58518")
        ax.text(x0 + 0.24 * (x1 - x0), y0 + 0.24 * (y1 - y0), label, color="#b35c00", fontsize=9, ha="center",
                va="center", bbox={"fc": "white", "ec": "none", "pad": 1.2})
    save(fig, "network-inputs-outputs.png")


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
        kern = np.exp(-(d["delta"][:, None] - d["delta"][None, :]) ** 2 / (2 * lz ** 2)
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
    simulation = prior_simulation()
    numbers["whitening_spectrum"] = fig_whitening_spectrum(simulation)
    numbers["prior_predictive"] = fig_prior_predictive(simulation)
    fig_whitening_comparison()
    fig_network()
    numbers["flow_matching_toy"] = fig_flow_matching()
    fig_heston_calibration()
    fig_heston_recovery()
    fig_heston_reference()
    numbers["information_by_region"] = fig_information_by_region()
    numbers["variance_strip"] = fig_variance_strip()
    numbers["posterior_to_product"] = fig_push_forward()
    numbers["posterior_costs"] = fig_costs()
    (HERE / "numbers.json").write_text(json.dumps(numbers, indent=1) + "\n")
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
