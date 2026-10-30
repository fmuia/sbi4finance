"""Figures of Lecture 5: implied volatility and the smile.

Usage: python course/scripts/lecture05.py   (writes course/images/l05-*.png and prints the numbers quoted in the text)
"""

import json

import numpy as np
from scipy.stats import norm

from common import COLORS, DAYS, DELTAS, GREY, ROOT, call, d12, load_spy, otm, plt, save, sf, tidy
from sbi4finance import hkde

REPORT = {}
HESTON = dict(kappa=2.0, theta=0.085, vol_of_vol=0.8, rho=-0.525, jump_intensity=0.0,
              jump_up_probability=0.5, jump_up_rate=10.0, jump_down_rate=10.0)
V0 = 0.085


def grid_k(j):
    return DELTAS * sf.SIGMA_REF * np.sqrt(DAYS[j] / 252)


def spy_iv(spy):
    prices = np.array(spy["prices_forward_units"]).reshape(len(DAYS), len(DELTAS))
    iv = np.empty_like(prices)
    for j in range(len(DAYS)):
        iv[j] = sf.implied_vol(prices[j], grid_k(j), DAYS[j] / 252, np.where(DELTAS <= 0, "put", "call"))
    return prices, iv


def fig_inversion():
    T = 21 / 252
    s = np.linspace(0.02, 0.6, 600)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    for color, (delta, name) in zip(COLORS, ((-1.0, r"put, $\delta = -1$"), (0.0, r"ATM, $\delta = 0$"),
                                             (1.0, r"call, $\delta = +1$"))):
        k = delta * 0.2 * np.sqrt(T)
        ax.plot(s, otm(k, s**2 * T), color=color, label=name)
    kq = -0.2 * np.sqrt(T)
    y = 0.0080
    simp = float(sf.implied_vol(y, kq, T, "put"))
    ax.axhline(y, color="0.3", ls="--", lw=1)
    ax.plot([simp, simp], [0, y], color="0.3", ls=":", lw=1)
    ax.plot(simp, y, "o", color=COLORS[0])
    ax.annotate(rf"quote $y$ = {y:.4f}", (0.36, y - 0.0035), fontsize=9, color="0.3")
    ax.annotate(rf"$\sigma_{{imp}}$ = {simp:.1%}", (simp + 0.015, 0.0012), fontsize=9, color="0.3")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"volatility $\sigma$")
    ax.set_ylabel("price, fraction of the forward")
    ax.set_title(r"Each price is increasing in $\sigma$: one crossing per quote")
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    REPORT["inversion_example"] = dict(y=y, k=float(kq), sigma_imp=simp)

    ax = axes[1]
    for color, (delta, y, name) in zip((COLORS[0], COLORS[3]),
                                       ((0.0, float(otm(0.0, 0.25**2 * T)), "ATM quote (true 25%)"),
                                        (2.0, float(otm(2 * 0.2 * np.sqrt(T), 0.25**2 * T)), r"$\delta = 2$ call (true 25%)"))):
        k = delta * 0.2 * np.sqrt(T)
        lo, hi, errs_b = 1e-4, 5.0, []
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if call(k, mid**2 * T) > y:
                hi = mid
            else:
                lo = mid
            errs_b.append(abs(0.5 * (lo + hi) - 0.25))
        sig, errs_n = 0.6, []
        for _ in range(12):
            d1, _ = d12(k, sig**2 * T)
            v = np.sqrt(T) * norm.pdf(d1)
            sig = sig - (call(k, sig**2 * T) - y) / v
            errs_n.append(abs(sig - 0.25) if np.isfinite(sig) else np.nan)
        ax.semilogy(range(1, 41), np.maximum(errs_b, 1e-16), color=color, ls=":", label=f"bisection, {name}")
        ax.semilogy(range(1, 13), np.maximum(np.abs(errs_n), 1e-16), "o-", color=color, ms=3.5, label=f"Newton from 60%, {name}")
        REPORT[f"newton_{delta}"] = [float(e) for e in errs_n]
    k = 2 * 0.2 * np.sqrt(T)
    y = float(otm(k, 0.25**2 * T))
    sig, errs = 0.10, []
    with np.errstate(all="ignore"):
        for _ in range(3):
            d1, _ = d12(k, sig**2 * T)
            sig = sig - (call(k, sig**2 * T) - y) / (np.sqrt(T) * norm.pdf(d1))
            errs.append(abs(sig - 0.25))
    ax.semilogy(range(1, 4), np.minimum(errs, 1e3), "x-", color=GREY, ms=6,
                label=r"Newton from 10%, $\delta = 2$ call: diverges")
    REPORT["newton_diverges"] = [float(e) for e in errs]
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"error $|\sigma_n - \sigma_{imp}|$")
    ax.set_title("Newton is fast but can diverge; bisection is safe")
    ax.set_ylim(1e-16, 1e3)
    ax.legend(frameon=False, fontsize=8)
    tidy(*axes)
    save(fig, "l05-inversion.png")


def fig_conditioning():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    vegas = []
    for color, j in zip(COLORS, range(len(DAYS))):
        T = DAYS[j] / 252
        vega = sf.bs_vega(grid_k(j), T, sf.SIGMA_REF)
        vegas.append(vega)
        axes[0].plot(DELTAS, vega, "o-", color=color, ms=3.5, label=f"{DAYS[j]} d")
        axes[1].semilogy(DELTAS, 1e-4 / vega * 100, "o-", color=color, ms=3.5, label=f"{DAYS[j]} d")
    axes[0].set_ylabel(r"vega $\sqrt{T}\,\varphi(d_1)$ at 20%")
    axes[0].set_title("Vega on the design grid")
    axes[1].set_ylabel("implied-volatility error, volatility points")
    axes[1].set_title(r"Error in $\sigma_{imp}$ from a price error of $10^{-4}$ of the forward")
    for ax in axes:
        ax.set_xlabel(r"standardized moneyness $\delta$")
    axes[0].legend(frameon=False, fontsize=8, ncol=2, title="maturity", title_fontsize=8)
    tidy(*axes)
    save(fig, "l05-conditioning.png")
    v = np.array(vegas)
    REPORT["vega"] = dict(atm_1w=float(v[0, 4]), d2_1w=float(v[0, 8]), atm_1y=float(v[-1, 4]), d2_1y=float(v[-1, 8]),
                          err_1w_d2=float(1e-4 / v[0, 8] * 100), err_1y_atm=float(1e-4 / v[-1, 4] * 100))


def fig_spy_smiles(spy, iv):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for color, j in zip(COLORS, range(len(DAYS))):
        axes[0].plot(DELTAS, iv[j] * 100, "o-", color=color, ms=3.5, label=f"{DAYS[j]} d")
        axes[1].plot(grid_k(j), iv[j] * 100, "o-", color=color, ms=3.5, label=f"{DAYS[j]} d")
    axes[0].set_xlabel(r"standardized moneyness $\delta$")
    axes[1].set_xlabel(r"log-moneyness $k = \ln(K/F)$")
    for ax in axes[:2]:
        ax.set_ylabel("implied volatility, %")
    axes[0].set_title("SPY smiles, 21 October 2026, against $\\delta$")
    axes[1].set_title("The same smiles against $k$")
    axes[0].legend(frameon=False, fontsize=8, ncol=2, title="maturity", title_fontsize=8)
    ax = axes[2]
    T = DAYS / 252
    atm = np.array(spy["atm_implied_vol"])
    wvar = np.array(spy["diagnostics"]["design_total_variance"])
    ax.plot(DAYS, atm * 100, "o-", color=COLORS[0], ms=4, label="at-the-money implied volatility")
    ax.plot(DAYS, np.sqrt(wvar / T) * 100, "s-", color=COLORS[3], ms=4, label=r"variance-swap volatility $\sqrt{w_{var}(T)/T}$")
    ax.set_xscale("log")
    ax.set_xticks(DAYS)
    ax.set_xticklabels([str(d) for d in DAYS])
    ax.set_xlabel("maturity, trading days")
    ax.set_ylabel("% per year")
    ax.set_title("Term structure: level rises with maturity")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l05-spy-smiles.png")
    REPORT["spy"] = dict(atm=[float(a) for a in atm], varswap_vol=[float(v) for v in np.sqrt(wvar / T)],
                         wing_put_1y=float(iv[-1, 0]), wing_call_1y=float(iv[-1, -1]),
                         wing_put_1w=float(iv[0, 0]), wing_call_1w=float(iv[0, -1]), min_1w=float(iv[0].min()),
                         argmin_1w_delta=float(DELTAS[np.argmin(iv[0])]))


def fig_skew_curvature(spy):
    T = DAYS / 252
    skew = np.abs(np.array(spy["atm_skew"]))
    curv = np.array(spy["atm_curvature"])
    b_s, a_s = np.polyfit(np.log(T), np.log(skew), 1)
    b_c, a_c = np.polyfit(np.log(T), np.log(curv), 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.loglog(DAYS, skew, "o", color=COLORS[0], ms=5, label=r"SPY $|\psi(T)| = |\partial\sigma_{imp}/\partial k|$ at $k=0$")
    ax.loglog(DAYS, np.exp(a_s) * T**b_s, color=COLORS[0], ls="--", label=rf"fit $\propto T^{{{b_s:.2f}}}$")
    ax.set_title("ATM skew steepens as maturity shortens")
    ax.set_ylabel("absolute skew, per unit of $k$")
    ax = axes[1]
    ax.loglog(DAYS, curv, "o", color=COLORS[3], ms=5, label=r"SPY $\partial^2\sigma_{imp}/\partial k^2$ at $k=0$")
    ax.loglog(DAYS, np.exp(a_c) * T**b_c, color=COLORS[3], ls="--", label=rf"fit $\propto T^{{{b_c:.2f}}}$")
    ax.set_title("ATM curvature explodes as maturity shortens")
    ax.set_ylabel("curvature, per unit of $k^2$")
    for ax in axes:
        ax.set_xticks(DAYS)
        ax.set_xticklabels([str(d) for d in DAYS])
        ax.set_xlabel("maturity, trading days")
        ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l05-skew-and-curvature.png")
    REPORT["skew_fit"] = dict(exponent=float(b_s), scale=float(np.exp(a_s)), skew=[float(x) for x in skew])
    REPORT["curv_fit"] = dict(exponent=float(b_c), scale=float(np.exp(a_c)), curv=[float(x) for x in curv])


def heston_calls(k, T):
    return np.array([float(hkde.european_price(1.0, np.exp(x), 1.0, T, HESTON, V0, kind="call",
                                               nodes=4096, upper_limit=3000.0)) for x in k])


def fig_smile_mechanisms():
    T = 21 / 252
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    k = np.linspace(-0.25, 0.25, 201)
    mix = 0.5 * (call(k, 0.15**2 * T) + call(k, 0.25**2 * T))
    iv = sf.implied_vol(mix, k, T, "call")
    ax.plot(k, iv * 100, color=COLORS[0], lw=2, label="50/50 mixture of 15% and 25%")
    for s in (0.15, 0.25):
        ax.axhline(s * 100, color="0.6", ls=":", lw=1)
    ax.axhline(np.sqrt(0.5 * (0.15**2 + 0.25**2)) * 100, color=COLORS[1], ls="--", lw=1,
               label=r"$\sqrt{\mathbb{E}[\sigma^2]}$ = 20.6%")
    ax.set_xlabel(r"log-moneyness $k$")
    ax.set_ylabel("implied volatility, %")
    ax.set_title("Random volatility, independent of the price: a smile")
    ax.legend(frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 0.08))
    REPORT["mixture_atm"] = float(iv[100])

    ax = axes[1]
    k = np.linspace(-0.25, 0.2, 91)
    iv_h = sf.implied_vol(heston_calls(k, T), k, T, "call")
    atm = float(np.interp(0.0, k, iv_h))
    ax.plot(k, iv_h * 100, color=COLORS[3], lw=2, label=r"random variance, correlation $-0.525$")
    ax.axhline(atm * 100, color="0.5", ls="--", lw=1, label=f"Black–Scholes at the ATM level, {atm:.1%}")
    ax.set_xlabel(r"log-moneyness $k$")
    ax.set_ylabel("implied volatility, %")
    ax.set_title("Random volatility, negatively correlated: a skew")
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[2]
    h = 0.004
    kk = np.linspace(-0.45, 0.35, 401)
    cm, c0, cp = heston_calls(kk - h, T), heston_calls(kk, T), heston_calls(kk + h, T)
    ck = (cp - cm) / (2 * h)
    ckk = (cp - 2 * c0 + cm) / h**2
    dens = (ckk - ck) / np.exp(kk)  # density of ln X_T by Breeden-Litzenberger
    w = atm**2 * T
    logn = norm.pdf(kk, loc=-0.5 * w, scale=np.sqrt(w))
    ax.semilogy(kk, np.maximum(dens, 1e-6), color=COLORS[3], lw=2, label="skewed model (from its prices)")
    ax.semilogy(kk, logn, color="0.5", ls="--", label="Black–Scholes at the ATM level")
    ax.set_ylim(1e-3, 20)
    ax.set_xlabel(r"$\ln X_T$")
    ax.set_ylabel("density (log scale)")
    ax.set_title("The skew is a heavier left tail")
    ax.legend(frameon=False, fontsize=8.5, loc="lower center")
    tidy(*axes)
    save(fig, "l05-smile-mechanisms.png")
    REPORT["heston_atm"] = atm
    REPORT["tail_ratio_-0.25"] = float(np.interp(-0.25, kk, dens) / np.interp(-0.25, kk, logn))
    REPORT["tail_ratio_+0.2"] = float(np.interp(0.2, kk, dens) / np.interp(0.2, kk, logn))


def fig_spy_calendar(iv):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ws = []
    for color, j in zip(COLORS, range(len(DAYS))):
        w = iv[j] ** 2 * DAYS[j] / 252
        ws.append(w)
        axes[0].plot(grid_k(j), w, "o-", color=color, ms=3.5, label=f"{DAYS[j]} d")
    axes[0].set_xlabel(r"log-moneyness $k$")
    axes[0].set_ylabel(r"total implied variance $w = \sigma_{imp}^2 T$")
    axes[0].set_title("SPY: total implied variance by maturity")
    axes[0].legend(frameon=False, fontsize=8, ncol=2, title="maturity", title_fontsize=8)
    worst = np.inf
    for color, kk in zip(COLORS, (-0.05, 0.0, 0.05)):
        vals, ds = [], []
        for j in range(len(DAYS)):
            k = grid_k(j)
            if k[0] <= kk <= k[-1]:
                vals.append(float(np.interp(kk, k, ws[j])))
                ds.append(DAYS[j])
        axes[1].plot(ds, vals, "o-", color=color, ms=3.5, label=f"$k$ = {kk:+.2f}")
        worst = min(worst, float(np.min(np.diff(vals))))
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xticks(DAYS)
    axes[1].set_xticklabels([str(d) for d in DAYS])
    axes[1].set_xlabel("maturity, trading days")
    axes[1].set_ylabel("total implied variance")
    axes[1].set_title("At fixed $k$ it rises with $T$: no calendar arbitrage")
    axes[1].legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l05-spy-calendar.png")
    REPORT["calendar_min_increment"] = worst


def fig_variance_swap(spy, prices):
    exps = spy["diagnostics"]["expirations"]
    days = np.array([e["days"] for e in exps], dtype=float)
    wrep = np.array([e["w_replicated"] for e in exps])
    covered = np.array([e["replication_covered"] for e in exps])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    trunc = []
    wvar = np.array(spy["diagnostics"]["design_total_variance"])
    for color, j in zip(COLORS, (0, 2, 4, 6)):
        k = grid_k(j)
        integrand = 2 * prices[j] * np.exp(-k)
        ax.plot(k, integrand, "o-", color=color, ms=3.5, label=f"{DAYS[j]} d")
    for j in range(len(DAYS)):
        k = grid_k(j)
        trunc.append(float(np.trapezoid(2 * prices[j] * np.exp(-k), k)))
    ax.set_xlabel(r"log-moneyness $k$")
    ax.set_ylabel(r"$2\,\mathrm{OTM}(k)\,e^{-k}$")
    ax.set_title("Replication integrand on the design grid")
    ax.legend(frameon=False, fontsize=8.5, title="maturity", title_fontsize=8.5)
    ax = axes[1]
    T = days / 252
    ax.plot(days[covered], np.sqrt(wrep[covered] / T[covered]) * 100, "o", color=COLORS[3], ms=4,
            label=r"full strip, $\delta \leq -8$ reached")
    ax.plot(days[~covered], np.sqrt(wrep[~covered] / T[~covered]) * 100, "x", color=GREY, ms=6,
            label="strip cut short (excluded)")
    ax.plot(DAYS, np.sqrt(np.array(trunc) / (DAYS / 252)) * 100, "s--", color=COLORS[2], ms=4,
            label=r"design grid only, $|\delta| \leq 2$")
    ax.plot(DAYS, np.array(spy["atm_implied_vol"]) * 100, "^-", color=COLORS[0], ms=4, label="ATM implied volatility")
    ax.set_xscale("log")
    ax.set_xlabel("maturity, trading days")
    ax.set_ylabel(r"$\sqrt{w_{var}(T)/T}$, % per year")
    ax.set_title("Variance-swap volatility from SPY options")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    xi = np.array(spy["forward_variance_levels"])
    edges = np.r_[0, DAYS]
    ax.stairs(np.sqrt(xi) * 100, edges, color=COLORS[3], lw=2, baseline=None, label=r"$\sqrt{\xi_0(t)}$ on each interval")
    ax.plot(DAYS, np.sqrt(wvar / (DAYS / 252)) * 100, "s", color=COLORS[2], ms=4, label=r"$\sqrt{w_{var}(T)/T}$ (average up to $T$)")
    ax.set_xlabel("time $t$, trading days")
    ax.set_ylabel("% per year")
    ax.set_title("SPY forward-variance curve, in volatility units")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l05-variance-swap.png")
    REPORT["varswap"] = dict(covered=int(covered.sum()), n=len(covered),
                             design_vs_trunc=[[float(np.sqrt(a / (d / 252))), float(np.sqrt(b / (d / 252)))]
                                              for a, b, d in zip(wvar, trunc, DAYS)],
                             xi_vol=[float(v) for v in np.sqrt(xi)])


SPY_QUOTES = ROOT / "lab" / "checks" / "spy_quotes" / "SPY_2026-10-21.json"


def fig_spy_quotes(spy, path=SPY_QUOTES, name="l05-spy-quotes.png"):
    """At four design maturities, the nearest expiration whose strip covers the replication range: its quotes, the
    spline read from them, the strip's range and the design grid."""
    if not path.exists():
        REPORT["spy_quotes"] = f"skipped: {path.name} is missing (lab/checks/spy_quotes/spy_quotes.py)"
        return
    data = json.loads(path.read_text())
    if data["chain_sha256"] != spy["chain_sha256"]:
        raise ValueError(f"{path.name} was read from another chain than the tutorial's SPY surface")
    grid = np.array(data["implied_vol_grid"]).reshape(len(DAYS), len(DELTAS))
    lo, hi = data["replication_deltas"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    report = []
    covered = [e for e in data["expirations"] if e["replication_covered"]] or data["expirations"]
    for ax, design_days in zip(axes.ravel(), (5, 21, 63, 252)):
        e = min(covered, key=lambda e: abs(e["days"] - design_days))
        scale = sf.SIGMA_REF * np.sqrt(e["days"] / 252)
        q = e["quotes"]
        d = np.array([x["k"] for x in q]) / scale
        bid, mid, ask = (100 * np.array([x[key] for x in q]) for key in ("iv_bid", "iv_mid", "iv_ask"))
        kept = np.array([x["kept"] for x in q])
        spline_d = np.array(e["spline"]["k"]) / scale
        spline_iv = 100 * np.array(e["spline"]["iv"])
        ax.axvspan(*e["strip_deltas"], color=COLORS[0], alpha=0.07, lw=0, label="the strip's range")
        for x in (lo, hi):
            ax.axvline(x, color=GREY, ls=":", lw=1, label=rf"replication range, $\delta$ = {lo:g} to {hi:g}"
                       if x == lo else None)
        ax.vlines(d[~kept], bid[~kept], ask[~kept], color=GREY, lw=1, alpha=0.6, label="removed by cleaning")
        ax.vlines(d[kept], bid[kept], ask[kept], color=COLORS[0], lw=1.6, label="quote, bid to ask")
        ax.plot(spline_d, spline_iv, color=COLORS[1], lw=1, label="smoothing spline")
        # the design points at their log-moneyness, in this expiration's delta
        j = list(DAYS).index(design_days)
        ax.plot(DELTAS * np.sqrt(design_days / e["days"]), 100 * grid[j], "o", color=COLORS[6], ms=4.5, zorder=5,
                label="design grid")
        ax.set_ylim(0, min(1.05 * ask[kept].max(), 150))
        ax.set_xlabel(r"standardized moneyness $\delta$ at the expiration's maturity")
        ax.set_ylabel("implied volatility (%)")
        ax.set_title(f"{e['expiration']}: {e['days']} days (design maturity {design_days})", fontsize=10)
        half = (ask[kept] - bid[kept]) / 2
        miss = (np.interp(d[kept], spline_d, spline_iv) - mid[kept]) / half
        report.append(dict(design_days=design_days, days=e["days"], kept=int(kept.sum()), removed=int((~kept).sum()),
                           no_bid=e["quotes_no_bid"], strip_deltas=e["strip_deltas"],
                           replication_covered=e["replication_covered"],
                           spline_rms_miss_half_spreads=float(np.sqrt(np.mean(miss ** 2)))))
    axes[0, 0].legend(frameon=False, fontsize=8)
    tidy(*axes.ravel())
    save(fig, name)
    REPORT["spy_quotes"] = report


def main():
    spy = load_spy()
    prices, iv = spy_iv(spy)
    fig_inversion()
    fig_conditioning()
    fig_spy_smiles(spy, iv)
    fig_skew_curvature(spy)
    fig_smile_mechanisms()
    fig_spy_calendar(iv)
    fig_variance_swap(spy, prices)
    fig_spy_quotes(spy)
    print(json.dumps(REPORT, indent=1))


if __name__ == "__main__":
    main()
