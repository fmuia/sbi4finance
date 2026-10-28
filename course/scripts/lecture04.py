"""Figures of Lecture 4: the Black-Scholes model.

Usage: python scripts/course/lecture04.py   (writes docs/course/images/l04-*.png and prints the numbers quoted in the text)
"""

import json

import numpy as np
from scipy.stats import norm

from common import COLORS, GREY, call, d12, plt, save, tidy

REPORT = {}


def hedge_pnl(mu, sigma_true, sigma_model, n_steps, T=0.25, paths=20_000, seed=0):
    """Sell one at-the-money call at the model price, delta-hedge n_steps times, return the final P&L.

    Forward units with zero rates and dividends: the forward is the spot, S_0 = 1, strike 1.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    S = np.ones(paths)
    w0 = sigma_model**2 * T
    cash = call(0.0, w0)  # premium received
    for i in range(n_steps):
        tau = T - i * dt
        d1, _ = d12(np.log(1.0 / S), sigma_model**2 * tau)
        delta = norm.cdf(d1)
        S_new = S * np.exp((mu - 0.5 * sigma_true**2) * dt + sigma_true * np.sqrt(dt) * rng.standard_normal(paths))
        cash = cash + delta * (S_new - S)
        S = S_new
    return cash - np.maximum(S - 1.0, 0.0), call(0.0, w0)


def fig_hedging():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    bins = np.linspace(-0.012, 0.012, 100)
    out = {}
    for color, mu in zip((COLORS[0], COLORS[3]), (-0.10, 0.25)):
        pnl, prem = hedge_pnl(mu, 0.2, 0.2, 63, seed=1)
        ax.hist(pnl * 100, bins=bins * 100, density=True, alpha=0.5, color=color,
                label=rf"$\mu$ = {mu:+.0%}: mean {pnl.mean() * 100:+.3f}, sd {pnl.std() * 100:.3f}")
        out[mu] = (float(pnl.mean()), float(pnl.std()))
    ax.set_xlabel("hedged P&L, % of the forward")
    ax.set_ylabel("density")
    ax.set_title("Daily hedging: the drift does not matter")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    REPORT["drift"] = dict(premium=float(prem), results=out)

    ax = axes[1]
    ns = np.array([4, 13, 63, 252, 1260])
    sds = []
    for n in ns:
        pnl, _ = hedge_pnl(0.08, 0.2, 0.2, int(n), seed=2, paths=10_000)
        sds.append(pnl.std())
    sds = np.array(sds)
    ax.loglog(ns, sds * 100, "o-", color=COLORS[0], ms=4, label="sd of the hedged P&L")
    ax.loglog(ns, sds[2] * 100 * np.sqrt(ns[2] / ns), color=GREY, ls=":", label=r"$\propto 1/\sqrt{n}$")
    ax.set_xticks(ns)
    ax.set_xticklabels(["monthly", "weekly", "daily", "4 a day", "20 a day"], fontsize=8)
    ax.set_xlabel("rebalancing frequency over three months")
    ax.set_ylabel("% of the forward")
    ax.set_title("The hedge becomes exact as rebalancing becomes continuous")
    ax.legend(frameon=False, fontsize=8.5)
    REPORT["rebalancing_sd"] = {int(n): float(s) for n, s in zip(ns, sds)}

    ax = axes[2]
    bins = np.linspace(-0.03, 0.03, 120)
    out = {}
    for color, st in zip((COLORS[0], COLORS[2], COLORS[3]), (0.15, 0.20, 0.25)):
        pnl, prem = hedge_pnl(0.0, st, 0.2, 252, seed=3)
        ax.hist(pnl * 100, bins=bins * 100, density=True, alpha=0.5, color=color,
                label=rf"realized $\sigma$ = {st:.0%}: mean {pnl.mean() * 100:+.2f}")
        out[st] = float(pnl.mean())
    ax.set_xlabel("P&L of the seller, % of the forward")
    ax.set_title("Hedged at 20%: the seller gains if realized vol is lower")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    tidy(*axes)
    save(fig, "l04-hedging.png")
    # approximation: E[P&L] ~ (sigma_model^2 - sigma_true^2)/2 * E int Gamma S^2 dt ~ vega * (sigma_model - sigma_true)
    vega = np.sqrt(0.25) * norm.pdf(0.5 * 0.2 * np.sqrt(0.25))
    REPORT["vol_mismatch"] = dict(means=out, vega_approx={st: float(vega * (0.2 - st)) for st in (0.15, 0.25)})


def fig_terminal_law():
    x = np.linspace(1e-3, 2.6, 2000)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    for color, s in zip(COLORS, (0.05, 0.1, 0.2, 0.4)):
        w = s * s
        q = np.exp(-(np.log(x) + 0.5 * w) ** 2 / (2 * w)) / (x * np.sqrt(2 * np.pi * w))
        ax.plot(x, q, color=color, label=rf"$\sigma\sqrt{{T}}$ = {s}")
    ax.axvline(1, color="0.6", ls="--", lw=1)
    ax.text(1.12, 6.6, r"$\mathbb{E}[X_T]=1$", color="0.4", fontsize=9)
    ax.set_xlim(0, 2.2)
    ax.set_xlabel(r"$X_T = S_T/F$")
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
    ax.text(1.13, 1.7, "shaded area\n" + rf"$=\mathbb{{Q}}(X_T>e^k)=N(d_2)={norm.cdf(d2):.3f}$", fontsize=9,
            color=COLORS[0])
    twin = ax.twinx()
    twin.plot(x, np.maximum(x - np.exp(k), 0), color=COLORS[1], label=r"payoff $(X_T-e^k)^+$")
    twin.set_ylabel("payoff", color=COLORS[1])
    twin.set_ylim(0, 1.2)
    twin.spines[["top"]].set_visible(False)
    ax.axvline(np.exp(k), color="0.6", ls=":", lw=1)
    ax.text(np.exp(k) + 0.01, 2.45, r"$e^k$", color="0.4")
    ax.set_xlim(0.4, 1.8)
    ax.set_ylim(0, 2.8)
    ax.set_xlabel(r"$X_T$")
    ax.set_ylabel("density")
    c = call(k, w)
    ax.set_title(rf"Call at $k$ = {k}: price $c$ = {c:.4f}")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="upper left")
    tidy(axes[0])
    ax.spines[["top"]].set_visible(False)
    save(fig, "l04-terminal-law.png")
    REPORT["call_example"] = dict(k=k, w=w, N_d2=float(norm.cdf(d2)), N_d1=float(norm.cdf(d1)), c=float(c))


def fig_probabilities():
    w = 0.2**2
    k = np.linspace(-0.6, 0.6, 600)
    d1, d2 = d12(k, w)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(k, norm.cdf(d2), color=COLORS[1], label=r"$N(d_2)=\mathbb{Q}(X_T>e^k)$")
    ax.plot(k, norm.cdf(d1), color=COLORS[2], label=r"$N(d_1)=\mathbb{Q}^S(X_T>e^k)$, the delta")
    ax.plot(k, np.exp(k) * norm.cdf(d2), color=COLORS[1], ls="--", label=r"$e^k N(d_2)$")
    ax.plot(k, call(k, w), color=COLORS[0], lw=2, label=r"$c = N(d_1) - e^k N(d_2)$")
    ax.plot(k, np.maximum(1 - np.exp(k), 0), color="0.6", ls=":", label=r"intrinsic value $(1-e^k)^+$")
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel(r"log-moneyness $k$")
    ax.set_ylabel("probability, or price as a fraction of the forward")
    ax.set_title(r"The two probabilities in the call price, $\sigma\sqrt{T}$ = 0.2")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    tidy(ax)
    save(fig, "l04-probabilities.png")


def fig_prices():
    k = np.linspace(-0.5, 0.5, 801)
    otm = lambda kk, w: np.where(kk < 0, call(kk, w) - 1 + np.exp(kk), call(kk, w))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    T = 21 / 252
    for color, sigma in zip(COLORS, (0.10, 0.20, 0.30, 0.40)):
        axes[0].plot(k, otm(k, sigma**2 * T), color=color, label=rf"$\sigma$ = {sigma:.0%}")
    axes[0].set_title(r"Volatility varies, $T$ = 1 month")
    sigma = 0.20
    for color, days, name in zip(COLORS, (5, 21, 63, 252), ("1 week", "1 month", "3 months", "1 year")):
        T = days / 252
        axes[1].plot(k, otm(k, sigma**2 * T), color=color, label=rf"$T$ = {name}")
        grid = np.arange(-2, 2.01, 0.5) * 0.2 * np.sqrt(T)
        axes[1].plot(grid, otm(grid, sigma**2 * T), "o", color=color, ms=3.5)
    axes[1].set_title(r"Maturity varies, $\sigma$ = 20% (dots: the 9 design strikes)")
    for ax in axes:
        ax.axvline(0, color="0.8", lw=0.8, zorder=0)
        ax.set_xlabel(r"log-moneyness $k = \ln(K/F)$")
        ax.text(-0.50, 0.004, "puts", color="0.45", fontsize=9)
        ax.text(0.44, 0.004, "calls", color="0.45", fontsize=9)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("out-of-the-money price, fraction of the forward")
    tidy(*axes)
    save(fig, "l04-prices.png")


def fig_bounds():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
    ax = axes[0]
    k = np.linspace(-1.5, 1.5, 800)
    ax.fill_between(k, np.maximum(1 - np.exp(k), 0), 1, color="0.93", label="no-arbitrage region")
    ax.plot(k, np.maximum(1 - np.exp(k), 0), color="0.3", ls=":", label=r"lower bound $(1-e^k)^+$")
    ax.axhline(1, color="0.3", ls="--", lw=1, label="upper bound 1")
    for color, s in zip(COLORS, (0.02, 0.2, 1.0, 3.0)):
        ax.plot(k, call(k, s * s), color=color, label=rf"$\sigma\sqrt{{T}}$ = {s}")
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel(r"log-moneyness $k$")
    ax.set_ylabel("call price, fraction of the forward")
    ax.set_title("Bounds and limits of the call price")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17))
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
    twin.plot(s, s**2 / 24 * 100, color=COLORS[3], ls=":", lw=1, label=r"$w/24$, %")
    twin.set_ylabel("relative error, %", color=COLORS[3])
    twin.spines[["top"]].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    tidy(axes[0])
    ax.spines[["top"]].set_visible(False)
    save(fig, "l04-bounds-and-atm.png")
    REPORT["atm_rel_error_0.2"] = float((0.2 / np.sqrt(2 * np.pi)) / (2 * norm.cdf(0.1) - 1) - 1)


def fig_greeks():
    k = np.linspace(-0.4, 0.4, 600)
    sigma = 0.2
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True)
    for color, days, name in zip(COLORS, (21, 63, 252), ("1 month", "3 months", "1 year")):
        T = days / 252
        w = sigma**2 * T
        d1, d2 = d12(k, w)
        vega = np.sqrt(T) * norm.pdf(d1)
        axes[0, 0].plot(k, norm.cdf(d1), color=color, label=name)
        axes[0, 1].plot(k, norm.pdf(d1) / (sigma * np.sqrt(T)), color=color, label=name)
        axes[1, 0].plot(k, vega, color=color, label=name)
        axes[1, 1].plot(k, vega * d1 * d2 / sigma, color=color, label=name)
        axes[1, 1].axvline(-w / 2, color=color, ls=":", lw=0.8)
        axes[1, 1].axvline(w / 2, color=color, ls=":", lw=0.8)
    axes[0, 0].set_title(r"Delta $N(d_1)$")
    axes[0, 1].set_title(r"Gamma $\varphi(d_1)/(\sigma\sqrt{T})$ (forward units)")
    axes[1, 0].set_title(r"Vega $\sqrt{T}\,\varphi(d_1)$")
    axes[1, 1].set_title(r"Vomma $\mathrm{vega}\cdot d_1 d_2/\sigma$; zero at $k=\pm w/2$ (dotted)")
    axes[1, 1].axhline(0, color="0.8", lw=0.8)
    for ax in axes[1]:
        ax.set_xlabel(r"log-moneyness $k$")
    axes[0, 0].legend(frameon=False, fontsize=9, title=r"maturity, $\sigma$ = 20%", title_fontsize=9)
    tidy(*axes.ravel())
    save(fig, "l04-greeks.png")


def main():
    fig_hedging()
    fig_terminal_law()
    fig_probabilities()
    fig_prices()
    fig_bounds()
    fig_greeks()
    print(json.dumps(REPORT, indent=1))


if __name__ == "__main__":
    main()
