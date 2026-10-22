# Draft kept for Lecture 10 (Bayesian calibration of Black-Scholes on SPY, figure 29). Not run by the course; paths are those of the original draft.
"""Draw the Appendix B figures of the calibration design document (Black-Scholes in detail).

Figure 23: the law of X_T = S_T / F_T, and the call price as an area under it.
Figure 24: N(d2), N(d1) and the call price c = N(d1) - e^k N(d2) against k.
Figure 25: no-arbitrage bounds, limits in sigma*sqrt(T), and the ATM approximation.
Figure 26: vega and the design quote noise on the 63-contract grid, with SPY's noise.
Figure 27: where a smile comes from: a mixture of volatilities, Heston's smile and density.
Figure 28: no-arbitrage checks on the SPY smile of 21 October 2026.
Figure 29: Bayesian calibration of Black-Scholes to 63 quotes: synthetic and SPY.

Closed-form, Fourier and cached SPY data only. Usage:
    python scripts/plot_design_black_scholes_appendix.py [--out docs/design/images]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sbi4finance import hkde  # noqa: E402
from sbi4finance.calibration import surface as sf  # noqa: E402

COLORS = ["#087F8C", "#D97732", "#738095", "#B33E43", "#5B8C3A", "#7A4E9C", "#2F2F2F"]
DAYS = np.array(sf.DESIGN_DAYS)
DELTAS = np.array(sf.DESIGN_DELTAS)
HESTON = dict(kappa=2.0, theta=0.085, vol_of_vol=0.8, rho=-0.525, jump_intensity=0.0,
              jump_up_probability=0.5, jump_up_rate=10.0, jump_down_rate=10.0)
V0 = 0.085
SPY = ROOT / "tutorial_final/results/calibration/spy/surface.json"
REPORT = {}


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)


def d12(k, w):
    s = np.sqrt(w)
    d1 = (-k + 0.5 * w) / s
    return d1, d1 - s


def call_w(k, w):
    d1, d2 = d12(k, w)
    return norm.cdf(d1) - np.exp(k) * norm.cdf(d2)


def otm_iv(price, k, T):
    kind = np.where(np.asarray(k) <= 0, "put", "call")
    return sf.implied_vol(price, k, T, kind)


# Figure 23 -------------------------------------------------------------------
def figure_terminal_law(path):
    x = np.linspace(1e-3, 2.6, 2000)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for color, s in zip(COLORS, (0.05, 0.1, 0.2, 0.4)):
        w = s * s
        q = np.exp(-(np.log(x) + 0.5 * w) ** 2 / (2 * w)) / (x * np.sqrt(2 * np.pi * w))
        ax.plot(x, q, color=color, label=rf"$\sigma\sqrt{{T}}$ = {s}")
    ax.axvline(1, color="0.6", ls="--", lw=1)
    ax.text(1.12, 6.6, r"$\mathbb{E}[X_T]=1$", color="0.4", fontsize=9)
    ax.set_xlim(0, 2.2)
    ax.set_xlabel(r"$X_T = S_T/F_T$")
    ax.set_ylabel("density")
    ax.set_title("Law of the terminal price in forward units")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    s, k = 0.2, 0.1
    w = s * s
    q = np.exp(-(np.log(x) + 0.5 * w) ** 2 / (2 * w)) / (x * np.sqrt(2 * np.pi * w))
    ax.plot(x, q, color=COLORS[0], label=r"density of $X_T$, $\sigma\sqrt{T}=0.2$")
    itm = x > np.exp(k)
    ax.fill_between(x[itm], 0, q[itm], color=COLORS[0], alpha=0.2)
    d1, d2 = d12(k, w)
    ax.text(1.13, 1.7, "shaded area\n" + rf"$=\mathbb{{Q}}(X_T>e^k)=N(d_2)={norm.cdf(d2):.3f}$", fontsize=9, color=COLORS[0])
    twin = ax.twinx()
    twin.plot(x, np.maximum(x - np.exp(k), 0), color=COLORS[1], label=r"payoff $(X_T-e^k)^+$")
    twin.set_ylabel("payoff", color=COLORS[1])
    twin.set_ylim(0, 1.2)
    twin.spines[["top"]].set_visible(False)
    ax.axvline(np.exp(k), color="0.6", ls=":", lw=1)
    ax.text(np.exp(k) + 0.01, 2.0, r"$e^k$", color="0.4")
    ax.set_xlim(0.4, 1.8)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"$X_T$")
    ax.set_ylabel("density")
    c = call_w(k, w)
    ax.set_title(rf"Call at $k$ = {k}: price $c$ = {c:.4f}")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="upper left")
    tidy(axes[0]); ax.spines[["top"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)
    REPORT["fig23"] = dict(k=k, sigma_sqrtT=s, N_d2=float(norm.cdf(d2)), N_d1=float(norm.cdf(d1)), c=float(c))


# Figure 24 -------------------------------------------------------------------
def figure_probabilities(path):
    w = 0.2 ** 2
    k = np.linspace(-0.6, 0.6, 600)
    d1, d2 = d12(k, w)
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.plot(k, norm.cdf(d2), color=COLORS[1], label=r"$N(d_2)=\mathbb{Q}(X_T>e^k)$")
    ax.plot(k, norm.cdf(d1), color=COLORS[2], label=r"$N(d_1)=\mathbb{Q}^S(X_T>e^k)$, the delta")
    ax.plot(k, np.exp(k) * norm.cdf(d2), color=COLORS[1], ls="--", label=r"$e^k N(d_2)$")
    ax.plot(k, call_w(k, w), color=COLORS[0], lw=2, label=r"$c = N(d_1) - e^k N(d_2)$")
    ax.plot(k, np.maximum(1 - np.exp(k), 0), color="0.6", ls=":", label=r"intrinsic value $(1-e^k)^+$")
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel(r"log-moneyness $k$")
    ax.set_ylabel("probability or price (fraction of the forward)")
    ax.set_title(r"The two probabilities in the call price, $\sigma\sqrt{T}$ = 0.2")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    tidy(ax)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


# Figure 25 -------------------------------------------------------------------
def figure_bounds(path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    k = np.linspace(-1.5, 1.5, 800)
    ax.fill_between(k, np.maximum(1 - np.exp(k), 0), 1, color="0.93", label="no-arbitrage region")
    ax.plot(k, np.maximum(1 - np.exp(k), 0), color="0.3", ls=":", label=r"lower bound $(1-e^k)^+$")
    ax.axhline(1, color="0.3", ls="--", lw=1, label="upper bound 1")
    for color, s in zip(COLORS, (0.02, 0.2, 1.0, 3.0)):
        ax.plot(k, call_w(k, s * s), color=color, label=rf"$\sigma\sqrt{{T}}$ = {s}")
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel(r"log-moneyness $k$")
    ax.set_ylabel("call price, fraction of the forward")
    ax.set_title("Bounds and limits of the call price")
    ax.legend(frameon=False, fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.56))

    ax = axes[1]
    s = np.linspace(0.001, 1.0, 400)
    exact = 2 * norm.cdf(s / 2) - 1
    approx = s / np.sqrt(2 * np.pi)
    ax.plot(s, exact, color=COLORS[0], label=r"exact $2N(\sigma\sqrt{T}/2)-1$")
    ax.plot(s, approx, color=COLORS[1], ls="--", label=r"approximation $\sigma\sqrt{T}/\sqrt{2\pi}$")
    ax.set_xlabel(r"$\sigma\sqrt{T}$")
    ax.set_ylabel("ATM call price, fraction of the forward")
    ax.set_title("At-the-money price and its linear approximation")
    twin = ax.twinx()
    twin.plot(s, (approx - exact) / exact * 100, color=COLORS[3], lw=1, label="relative error, %")
    twin.plot(s, s ** 2 / 24 * 100, color=COLORS[3], ls=":", lw=1, label=r"$w/24$, %")
    twin.set_ylabel("relative error, %", color=COLORS[3])
    twin.spines[["top"]].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="upper left")
    tidy(axes[0]); ax.spines[["top"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)
    REPORT["fig25"] = dict(rel_error_at_0p2=float((0.2 / np.sqrt(2 * np.pi)) / (2 * norm.cdf(0.1) - 1) - 1))


# Figure 26 -------------------------------------------------------------------
def figure_conditioning(path, spy):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    noise_spy = np.array(spy["noise_sd"]).reshape(len(DAYS), len(DELTAS))
    vegas, sigmas = [], []
    for color, day in zip(COLORS, DAYS):
        T = day / sf.TRADING_DAYS
        k = DELTAS * sf.SIGMA_REF * np.sqrt(T)
        vega = sf.bs_vega(k, T, sf.SIGMA_REF)
        sig = vega * sf.implied_vol_noise(DELTAS)
        vegas.append(vega); sigmas.append(sig)
        axes[0].plot(DELTAS, vega, "o-", color=color, ms=3.5, label=f"{day} d")
        axes[1].plot(DELTAS, sig, "o-", color=color, ms=3.5, label=f"{day} d")
    axes[0].set_ylabel(r"vega $\sqrt{T}\,\varphi(d_1)$ at $\sigma_{ref}$ = 20%")
    axes[0].set_title("Vega on the design grid, by maturity")
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"quote-noise sd $\sigma_c$, fraction of the forward")
    axes[1].set_title(r"Quote-noise sd $\sigma_c$ = vega $\times\, s_{iv}(\delta)$")
    for ax in axes:
        ax.set_xlabel(r"standardized moneyness $\delta$")
        tidy(ax)
    axes[0].legend(frameon=False, fontsize=8.5, ncol=2, title="maturity", title_fontsize=8.5)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)
    vegas = np.array(vegas)
    REPORT["fig26"] = dict(vega_1w_atm=float(vegas[0, 4]), vega_1w_d2=float(vegas[0, 8]),
                           vega_1y_atm=float(vegas[-1, 4]), vega_1y_d2=float(vegas[-1, 8]),
                           spy_over_design_median=float(np.median(noise_spy / np.array(sigmas))))


# Figure 27 -------------------------------------------------------------------
def heston_calls(k, T):
    return np.array([float(hkde.european_price(1.0, np.exp(x), 1.0, T, HESTON, V0, kind="call",
                                               nodes=4096, upper_limit=3000.0)) for x in k])


def figure_smile_density(path):
    T = 21 / sf.TRADING_DAYS
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    ax = axes[0]
    k = np.linspace(-0.25, 0.25, 201)
    mix = 0.5 * (call_w(k, 0.15 ** 2 * T) + call_w(k, 0.25 ** 2 * T))
    iv = sf.implied_vol(mix, k, T, "call")
    ax.plot(k, iv * 100, color=COLORS[0], lw=2, label="50/50 mixture of 15% and 25%")
    for s, ls in ((0.15, ":"), (0.25, ":")):
        ax.axhline(s * 100, color="0.6", ls=ls, lw=1)
    ax.axhline(np.sqrt(0.5 * (0.15 ** 2 + 0.25 ** 2)) * 100, color=COLORS[1], ls="--", lw=1,
               label=r"$\sqrt{\mathbb{E}[\sigma^2]}$ = 20.6%")
    ax.set_xlabel(r"log-moneyness $k$")
    ax.set_ylabel("implied volatility, %")
    ax.set_title("A random volatility makes a smile, $T$ = 1 month")
    ax.legend(frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 0.08))
    REPORT["fig27_mix_atm"] = float(iv[100])

    ax = axes[1]
    k = np.linspace(-0.25, 0.2, 91)
    c = heston_calls(k, T)
    iv_h = sf.implied_vol(c, k, T, "call")
    atm = float(np.interp(0.0, k, iv_h))
    ax.plot(k, iv_h * 100, color=COLORS[3], lw=2, label=r"Heston, $\rho_H$ = −0.525")
    ax.axhline(atm * 100, color="0.5", ls="--", lw=1, label=f"Black–Scholes at the ATM level, {atm:.1%}")
    ax.set_xlabel(r"log-moneyness $k$")
    ax.set_ylabel("implied volatility, %")
    ax.set_title("A correlated random volatility makes a skew")
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[2]
    h = 0.004
    kk = np.linspace(-0.45, 0.35, 401)
    cm, c0, cp = heston_calls(kk - h, T), heston_calls(kk, T), heston_calls(kk + h, T)
    ck = (cp - cm) / (2 * h)
    ckk = (cp - 2 * c0 + cm) / h ** 2
    dens = (ckk - ck) / np.exp(kk)      # density of ln X_T, by Breeden-Litzenberger
    w = atm ** 2 * T
    logn = norm.pdf(kk, loc=-0.5 * w, scale=np.sqrt(w))
    ax.semilogy(kk, np.maximum(dens, 1e-6), color=COLORS[3], lw=2, label="Heston (from its prices)")
    ax.semilogy(kk, logn, color="0.5", ls="--", label="Black–Scholes at the ATM level")
    ax.set_ylim(1e-3, 20)
    ax.set_xlabel(r"$\ln X_T$")
    ax.set_ylabel("density (log scale)")
    ax.set_title("The skew is a heavier left tail")
    ax.legend(frameon=False, fontsize=8.5, loc="lower center")
    for a in axes:
        tidy(a)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)
    left = kk < -0.2
    REPORT["fig27_tail_ratio_at_-0.25"] = float(np.interp(-0.25, kk, dens) / np.interp(-0.25, kk, logn))
    REPORT["fig27_heston_atm"] = atm


# Figure 28 -------------------------------------------------------------------
def figure_spy_arbitrage(path, spy):
    prices = np.array(spy["prices_forward_units"]).reshape(len(DAYS), len(DELTAS))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ws, ks = [], []
    for color, day, row in zip(COLORS, DAYS, prices):
        T = day / sf.TRADING_DAYS
        k = DELTAS * sf.SIGMA_REF * np.sqrt(T)
        iv = otm_iv(row, k, T)
        w = iv ** 2 * T
        ws.append(w); ks.append(k)
        axes[0].plot(k, w, "o-", color=color, ms=3.5, label=f"{day} d")
    axes[0].set_xlabel(r"log-moneyness $k$")
    axes[0].set_ylabel(r"total implied variance $w = \sigma_{imp}^2 T$")
    axes[0].set_title("SPY, 21 Oct 2026: total variance must rise with $T$")
    axes[0].legend(frameon=False, fontsize=8.5, ncol=2, title="maturity", title_fontsize=8.5)
    # calendar check: at each k of a shorter maturity inside the next one's range, w must not fall
    worst = np.inf
    for j in range(len(DAYS) - 1):
        nxt = np.interp(ks[j], ks[j + 1], ws[j + 1])
        worst = min(worst, float(np.min(nxt - ws[j])))
    REPORT["fig28_min_calendar_gap"] = worst

    ax = axes[1]
    j = 2  # one month
    T = DAYS[j] / sf.TRADING_DAYS
    k = DELTAS * sf.SIGMA_REF * np.sqrt(T)
    calls = np.where(k <= 0, prices[j] + 1 - np.exp(k), prices[j])   # put-call parity in forward units
    x = np.exp(k)
    ax.plot(x, calls, "o-", color=COLORS[2], ms=4, label="call price (puts converted by parity)")
    ax.plot(x, np.maximum(1 - x, 0), color="0.6", ls=":", label=r"intrinsic $(1-x)^+$")
    slope = np.diff(calls) / np.diff(x)
    convex = np.diff(slope)
    ax.set_xlabel(r"strike in forward units $x = K/F = e^k$")
    ax.set_ylabel("call price, fraction of the forward")
    ax.set_title("SPY, 1 month: decreasing and convex in the strike")
    ax.legend(frameon=False, fontsize=8.5)
    REPORT["fig28_slopes_range"] = [float(slope.min()), float(slope.max())]
    REPORT["fig28_min_second_difference"] = float(convex.min())
    for a in axes:
        tidy(a)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


# Figure 29 -------------------------------------------------------------------
def bs_prices_grid(sig):
    """63 OTM prices, maturity-major; sig is a scalar or one value per maturity."""
    sig = np.broadcast_to(np.asarray(sig, dtype=float), (len(DAYS),))
    out = []
    for s, day in zip(sig, DAYS):
        T = day / sf.TRADING_DAYS
        k = DELTAS * sf.SIGMA_REF * np.sqrt(T)
        out.append(sf.bs_price(k, T, s, np.where(DELTAS <= 0, "put", "call")))
    return np.concatenate(out)


def posterior_sigma(y, sd, block=None):
    """Grid posterior of one sigma under a uniform prior on (0.05, 0.6)."""
    def nll(s):
        p = bs_prices_grid(s) if block is None else block(s)
        return 0.5 * np.sum(((y - p) / sd) ** 2)
    best = minimize_scalar(nll, bounds=(0.05, 0.6), method="bounded", options=dict(xatol=1e-10)).x
    # local curvature for the grid width
    e = 1e-5
    curv = (nll(best + e) - 2 * nll(best) + nll(best - e)) / e ** 2
    sd_post = 1 / np.sqrt(curv)
    grid = np.linspace(best - 8 * sd_post, best + 8 * sd_post, 801)
    logp = -np.array([nll(s) for s in grid])
    p = np.exp(logp - logp.max())
    p /= np.trapezoid(p, grid)
    mean = np.trapezoid(grid * p, grid)
    sd_ = np.sqrt(np.trapezoid((grid - mean) ** 2 * p, grid))
    return grid, p, mean, sd_, best


def figure_posterior(path, spy):
    contracts = sf.design_grid()
    sd_design = np.array(sf.quote_noise(contracts))
    rng = np.random.default_rng(11)
    truth = 0.20
    y_syn = bs_prices_grid(truth) + sd_design * rng.standard_normal(63)
    y_spy = np.array(spy["prices_forward_units"])
    sd_spy = np.array(spy["noise_sd"])

    g1, p1, m1, s1, _ = posterior_sigma(y_syn, sd_design)
    g2, p2, m2, s2, b2 = posterior_sigma(y_spy, sd_spy)
    fisher = 1 / np.sqrt(np.sum(1 / sf.implied_vol_noise(np.tile(DELTAS, len(DAYS))) ** 2))

    z_syn = (y_syn - bs_prices_grid(m1)) / sd_design
    z_one = (y_spy - bs_prices_grid(m2)) / sd_spy
    per = []
    for j, day in enumerate(DAYS):
        sl = slice(9 * j, 9 * j + 9)
        T = day / sf.TRADING_DAYS
        k = DELTAS * sf.SIGMA_REF * np.sqrt(T)
        kind = np.where(DELTAS <= 0, "put", "call")
        blk = lambda s, k=k, T=T, kind=kind: sf.bs_price(k, T, s, kind)
        _, _, mj, _, _ = posterior_sigma(y_spy[sl], sd_spy[sl], block=blk)
        per.append(mj)
    z_per = (y_spy - bs_prices_grid(np.array(per))) / sd_spy

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    ax.plot((g1 - truth) * 100, p1 / p1.max(), color=COLORS[0], label=f"synthetic BS, truth 20%: mean {m1:.2%}")
    ax.plot((g2 - m2) * 100, p2 / p2.max(), color=COLORS[3], ls="--",
            label=f"SPY: mean {m2:.2%} (centred at 0)")
    ax.axvline(0, color="0.7", lw=0.8)
    ax.set_xlabel(r"$\sigma$ minus reference, volatility points")
    ax.set_ylabel("posterior density (peak = 1)")
    ax.set_title(f"Posterior of σ: sd {s1 * 100:.3f} (synthetic), {s2 * 100:.3f} (SPY) vol pts")
    ax.set_xlim(-0.5, 0.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    for ax, z, title in ((axes[1], z_one, "SPY residuals, one σ for all maturities"),
                         (axes[2], z_per, "SPY residuals, one σ per maturity")):
        for color, j in zip(COLORS, range(len(DAYS))):
            ax.plot(DELTAS, z[9 * j:9 * j + 9], "o-", color=color, ms=3.5, label=f"{DAYS[j]} d")
        ax.axhspan(-2, 2, color="0.9", zorder=0, label=r"$\pm 2$")
        ax.set_yscale("symlog", linthresh=2)
        ticks = [-10, -5, -2, 0, 2, 5, 10, 20]
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel(r"standardized moneyness $\delta$")
        ax.set_ylabel(r"$(y_c - c_{BS}) / \sigma_c$")
        ax.set_title(title)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    for a in axes:
        tidy(a)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)
    REPORT["fig29"] = dict(syn_mean=m1, syn_sd=s1, fisher_sd=float(fisher), syn_absz_max=float(np.abs(z_syn).max()),
                           syn_frac_within2=float(np.mean(np.abs(z_syn) < 2)),
                           spy_mean=m2, spy_sd=s2, spy_absz_max=float(np.abs(z_one).max()),
                           spy_frac_within2=float(np.mean(np.abs(z_one) < 2)),
                           per_maturity=[float(x) for x in per], per_absz_max=float(np.abs(z_per).max()),
                           per_frac_within2=float(np.mean(np.abs(z_per) < 2)),
                           spy_atm=[float(x) for x in spy["atm_implied_vol"]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/design/images")
    out = Path(parser.parse_args().out)
    spy = json.loads(SPY.read_text())
    figure_terminal_law(out / "calibration-23-bs-terminal-law.png")
    figure_probabilities(out / "calibration-24-bs-probabilities.png")
    figure_bounds(out / "calibration-25-bs-bounds-atm.png")
    figure_conditioning(out / "calibration-26-vega-and-quote-noise.png", spy)
    figure_smile_density(out / "calibration-27-smile-and-density.png")
    figure_spy_arbitrage(out / "calibration-28-spy-no-arbitrage.png", spy)
    figure_posterior(out / "calibration-29-bs-posterior.png", spy)
    print(json.dumps(REPORT, indent=1, default=float))


if __name__ == "__main__":
    main()
