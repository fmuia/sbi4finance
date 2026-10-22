"""Figures of Lecture 1: options, quotes and volatility.

Usage: python scripts/course/lecture01.py   (writes docs/course/images/l01-*.png and prints the numbers quoted in the text)
"""

import json

import numpy as np
from scipy.stats import norm

from common import COLORS, DAYS, DELTAS, GREY, call, load_spy, load_voo, plt, save, sf, tidy

REPORT = {}


def bs_dollar_call(S, K, T, sigma):
    """Call price in currency units with zero rates and dividends (the forward equals the spot)."""
    return S * call(np.log(K / S), sigma * sigma * T)


def fig_voo():
    simple, logr = load_voo()
    level = 100 * np.cumprod(1 + simple)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1]))
    axes[0].plot(level.index, level.values, color=COLORS[0])
    axes[0].set_ylabel("VOO total-return index (3 Jan 2020 = 100)")
    axes[0].set_title("The underlying: an S&P 500 index fund, daily, 2020–2026")
    axes[1].plot(logr.index, logr.values * 100, color=COLORS[2], lw=0.6)
    axes[1].set_ylabel("daily log return, %")
    worst = logr.idxmin()
    axes[1].annotate(f"{worst.date()}: {logr.min():.1%}", (worst, logr.min() * 100),
                     xytext=(worst + np.timedelta64(200, "D"), logr.min() * 100 + 1), fontsize=9,
                     arrowprops=dict(arrowstyle="-", color=GREY))
    tidy(*axes)
    save(fig, "l01-voo-price-and-returns.png")
    years = len(logr) / 252
    REPORT["voo"] = dict(n=len(logr), first=str(logr.index[0].date()), last=str(logr.index[-1].date()),
                         years=years, total_growth=float(level.iloc[-1] / 100),
                         mean_annual_log=float(logr.mean() * 252), vol_annual=float(logr.std() * np.sqrt(252)),
                         worst_day=str(worst.date()), worst=float(logr.min()), best=float(logr.max()),
                         best_day=str(logr.idxmax().date()))


def fig_payoffs():
    S = np.linspace(50, 150, 400)
    K, prem_c, prem_p = 100.0, bs_dollar_call(100, 100, 1, 0.2), None
    prem_p = prem_c  # at the money with zero carry, call and put cost the same (put-call parity, Lecture 2)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax = axes[0]
    ax.plot(S, S - K, color=COLORS[6])
    ax.set_title(r"Long forward: $S_T - K$")
    ax = axes[1]
    ax.plot(S, np.maximum(S - K, 0), color=COLORS[0], label=r"payoff $(S_T-K)^+$")
    ax.plot(S, np.maximum(S - K, 0) - prem_c, color=COLORS[0], ls="--", label=f"profit after premium {prem_c:.2f}")
    ax.set_title("Long call, strike 100")
    ax = axes[2]
    ax.plot(S, np.maximum(K - S, 0), color=COLORS[1], label=r"payoff $(K-S_T)^+$")
    ax.plot(S, np.maximum(K - S, 0) - prem_p, color=COLORS[1], ls="--", label=f"profit after premium {prem_p:.2f}")
    ax.set_title("Long put, strike 100")
    for ax in axes:
        ax.axhline(0, color="0.8", lw=0.8)
        ax.axvline(K, color="0.85", lw=0.8)
        ax.set_xlabel(r"price at maturity $S_T$")
        ax.set_ylabel("value at maturity")
        if ax is not axes[0]:
            ax.legend(frameon=False, fontsize=8.5, loc="upper center")
    tidy(*axes)
    save(fig, "l01-payoffs.png")
    REPORT["premium_atm_1y_20pct"] = float(prem_c)


def fig_protective_put():
    K, sigma, T, mu = 90.0, 0.2, 1.0, 0.07
    prem = bs_dollar_call(100, K, T, sigma) - 100 + K  # put by parity with zero carry
    S = np.linspace(50, 150, 400)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(S, S, color=COLORS[2], label="stock alone")
    ax.plot(S, S + np.maximum(K - S, 0) - prem, color=COLORS[0], label=f"stock + put struck at 90 (premium {prem:.2f})")
    ax.axhline(K - prem, color=COLORS[0], ls=":", lw=1)
    ax.set_xlabel(r"price at maturity $S_T$")
    ax.set_ylabel("position value at maturity")
    ax.set_title("A put insures a holding of the stock")
    ax.legend(frameon=False, fontsize=8.5)
    rng = np.random.default_rng(1)
    ST = 100 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * rng.standard_normal(200_000))
    r_stock = ST / 100 - 1
    r_hedged = (ST + np.maximum(K - ST, 0) - prem) / 100 - 1
    bins = np.linspace(-0.6, 0.9, 120)
    ax = axes[1]
    ax.hist(r_stock * 100, bins=bins * 100, density=True, color=COLORS[2], alpha=0.5, label="stock alone")
    ax.hist(r_hedged * 100, bins=bins * 100, density=True, color=COLORS[0], alpha=0.5, label="stock + put")
    ax.set_xlabel("one-year return, %")
    ax.set_ylabel("density")
    ax.set_title("The put removes the left tail, at the cost of its premium")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l01-protective-put.png")
    REPORT["protective_put"] = dict(premium=float(prem), floor=float(K - prem),
                                    p5_stock=float(np.quantile(r_stock, 0.05)),
                                    p5_hedged=float(np.quantile(r_hedged, 0.05)),
                                    mean_stock=float(r_stock.mean()), mean_hedged=float(r_hedged.mean()))


def fig_intrinsic():
    S = np.linspace(70, 130, 400)
    K, sigma = 100.0, 0.2
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.plot(S, np.maximum(S - K, 0), color="0.4", ls=":", label="intrinsic value $(S-K)^+$")
    for color, (T, name) in zip(COLORS, ((1 / 12, "1 month"), (0.25, "3 months"), (1.0, "1 year"))):
        ax.plot(S, bs_dollar_call(S, K, T, sigma), color=color, label=f"value, {name} to maturity")
    T = 0.25
    v = bs_dollar_call(S, K, T, sigma)
    ax.fill_between(S, np.maximum(S - K, 0), v, color=COLORS[1], alpha=0.12, label="time value, 3 months")
    for x, lab in ((88, "out of\nthe money"), (100, "at the\nmoney"), (112, "in the\nmoney")):
        ax.text(x, 31, lab, ha="center", fontsize=8.5, color="0.35")
    ax.axvline(K, color="0.85", lw=0.8)
    ax.set_xlabel("price of the underlying today, $S$")
    ax.set_ylabel("call value, strike 100")
    ax.set_title("Intrinsic value and time value of a call (σ = 20%)")
    ax.set_ylim(-1, 34)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    tidy(ax)
    save(fig, "l01-intrinsic-and-time-value.png")
    REPORT["atm_time_value_3m"] = float(bs_dollar_call(100, 100, 0.25, 0.2))


def fig_spy_grid(spy):
    exps = spy["diagnostics"]["expirations"]
    exp_days = np.array([e["days"] for e in exps], dtype=float)
    exp_fwd = np.array([e["forward"] for e in exps])
    prices = np.array(spy["prices_forward_units"]).reshape(len(DAYS), len(DELTAS))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    strikes = []
    for color, day in zip(COLORS, DAYS):
        T = day / sf.TRADING_DAYS
        F = float(np.interp(day, exp_days, exp_fwd))
        K = F * np.exp(DELTAS * sf.SIGMA_REF * np.sqrt(T))
        strikes.append(K)
        ax.scatter(np.full(9, day), K, color=[COLORS[0] if d <= 0 else COLORS[1] for d in DELTAS], s=16, zorder=3)
    ax.plot(exp_days, exp_fwd, color="0.3", lw=1, label="forward $F_T$")
    ax.scatter([], [], color=COLORS[0], s=16, label="put (strike below the forward)")
    ax.scatter([], [], color=COLORS[1], s=16, label="call (strike above the forward)")
    ax.set_xscale("log")
    ax.set_xticks(DAYS)
    ax.set_xticklabels([str(d) for d in DAYS])
    ax.set_xlabel("maturity, trading days")
    ax.set_ylabel("strike, US dollars")
    ax.set_title("The 63 contracts read from the SPY chain")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax = axes[1]
    for color, day, row in zip(COLORS, DAYS, prices):
        ax.semilogy(DELTAS, row, "o-", color=color, ms=3.5, label=f"{day} d")
    ax.set_xlabel(r"standardized moneyness $\delta$")
    ax.set_ylabel("price, fraction of the forward")
    ax.set_title("Their prices, SPY close of 21 October 2026")
    ax.legend(frameon=False, fontsize=8, ncol=2, title="maturity", title_fontsize=8)
    tidy(*axes)
    save(fig, "l01-spy-design-grid.png")
    strikes = np.array(strikes)
    REPORT["spy_grid"] = dict(strike_range_1w=[float(strikes[0, 0]), float(strikes[0, -1])],
                              strike_range_1y=[float(strikes[-1, 0]), float(strikes[-1, -1])],
                              atm_price_1w=float(prices[0, 4]), atm_price_1y=float(prices[-1, 4]),
                              wing_put_1y=float(prices[-1, 0]), forward_1w=float(np.interp(5, exp_days, exp_fwd)),
                              forward_1y=float(np.interp(252, exp_days, exp_fwd)))


def fig_exotics():
    rng = np.random.default_rng(4)
    n, T, sigma = 126, 0.5, 0.25
    dt = T / n
    t = np.linspace(0, T, n + 1)
    paths = 100 * np.exp(np.cumsum(np.c_[np.zeros(3), (-0.5 * sigma**2 * dt + sigma * np.sqrt(dt)
                                                        * rng.standard_normal((3, n)))], axis=1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    i1 = n // 2
    for color, p in zip(COLORS, paths):
        ax.plot(t * 252, p, color=color, lw=1)
        pay = max(p[-1] / p[i1] - 1, 0)
        ax.plot([t[i1] * 252, T * 252], [p[i1], p[i1]], color=color, ls=":", lw=1)
        ax.text(T * 252 + 2, p[-1], f"{pay:.1%}", color=color, fontsize=8.5, va="center")
    ax.axvline(t[i1] * 252, color="0.85")
    ax.set_title(r"Forward-start call: strike set to $S_{T_1}$ at $T_1$")
    ax.set_xlabel("trading days")
    ax.set_ylabel("price of the underlying")
    ax = axes[1]
    B = 115
    for color, p in zip(COLORS, paths):
        hit = np.any(p >= B)
        ax.plot(t * 252, p, color=color, lw=1, alpha=0.4 if hit else 1)
        pay = 0.0 if hit else max(p[-1] - 100, 0)
        ax.text(T * 252 + 2, p[-1], "knocked out" if hit else f"{pay:.1f}", color=color, fontsize=8.5, va="center")
    ax.axhline(B, color=COLORS[3], ls="--", lw=1)
    ax.text(3, B + 1, "barrier 115", color=COLORS[3], fontsize=8.5)
    ax.axhline(100, color="0.85")
    ax.set_title("Up-and-out call, strike 100: dies if the barrier is touched")
    ax.set_xlabel("trading days")
    ax = axes[2]
    finals = []
    for color, p in zip(COLORS, paths):
        r = np.diff(np.log(p))
        running = 252 * np.cumsum(r**2) / np.arange(1, n + 1)
        ax.plot(t[1:] * 252, np.sqrt(running) * 100, color=color, lw=1)
        finals.append((np.sqrt(running[-1]), color))
    for rank, (v, color) in enumerate(sorted(finals)):
        ax.text(T * 252 + 2, 22 + 3 * rank, f"{v:.1%}", color=color, fontsize=8.5, va="center")
    ax.axhline(sigma * 100, color="0.5", ls=":", lw=1)
    ax.set_ylim(10, 40)
    ax.set_title("Variance swap: pays realized variance minus a strike")
    ax.set_xlabel("trading days")
    ax.set_ylabel(r"running realized volatility $\sqrt{Q_t}$, %")
    tidy(*axes)
    save(fig, "l01-exotic-payoffs.png")


def fig_realized_vol():
    _, logr = load_voo()
    roll = logr.rolling(21).std() * np.sqrt(252)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.plot(roll.index, roll.values * 100, color=COLORS[3], lw=1)
    yearly = logr.groupby(logr.index.year).std() * np.sqrt(252)
    for year, v in yearly.items():
        ax.hlines(v * 100, np.datetime64(f"{year}-01-01"), np.datetime64(f"{year}-12-31"), color="0.3", lw=2)
    ax.set_ylabel("annualized volatility, %")
    ax.set_title("Realized volatility of VOO: 21-day window (red), calendar year (black)")
    ax = axes[1]
    z = (logr - logr.mean()) / logr.std()
    bins = np.linspace(-10, 10, 101)
    ax.hist(z, bins=bins, density=True, color=COLORS[2], alpha=0.7, label="VOO daily log returns, standardized")
    x = np.linspace(-10, 10, 400)
    ax.plot(x, norm.pdf(x), color="0.2", ls="--", label="standard normal")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1)
    ax.set_xlabel("return in standard deviations")
    ax.set_ylabel("density (log scale)")
    ax.set_title("Daily returns have far heavier tails than a normal law")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l01-voo-realized-vol.png")
    beyond4 = int((np.abs(z) > 4).sum())
    REPORT["realized"] = dict(roll_min=float(roll.min()), roll_max=float(roll.max()),
                              roll_max_date=str(roll.idxmax().date()),
                              yearly={int(k): float(v) for k, v in yearly.items()},
                              beyond4=beyond4, expected_beyond4_normal=float(len(z) * 2 * norm.sf(4)),
                              kurtosis=float(((z**4).mean())))


def main():
    spy = load_spy()
    fig_voo()
    fig_payoffs()
    fig_protective_put()
    fig_intrinsic()
    fig_spy_grid(spy)
    fig_exotics()
    fig_realized_vol()
    print(json.dumps(REPORT, indent=1))


if __name__ == "__main__":
    main()
