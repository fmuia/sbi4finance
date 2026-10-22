"""Figures of Lecture 2: no-arbitrage without a model.

Usage: python scripts/course/lecture02.py   (writes docs/course/images/l02-*.png and prints the numbers quoted in the text)
"""

import json

import numpy as np
from scipy.stats import norm

from common import COLORS, DAYS, DELTAS, GREY, load_spy, plt, save, sf, tidy

REPORT = {}
SPOT = 773.09  # SPY close of 21 October 2026 (design document, study S8)


def expirations(spy):
    e = spy["diagnostics"]["expirations"]
    days = np.array([x["days"] for x in e], dtype=float)
    return (days, np.array([x["forward"] for x in e]), np.array([x["discount"] for x in e]),
            np.array([x["parity_regression_discount"] for x in e]))


def fig_rates(spy):
    days, fwd, disc, preg = expirations(spy)
    T = days / 252
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(days, disc, "o-", color=COLORS[0], ms=3.5, label="discount factor $D(T)$")
    ax.set_xlabel("maturity, trading days")
    ax.set_ylabel("discount factor")
    ax.set_title("Discount factors from US Treasury yields")
    twin = ax.twinx()
    twin.plot(days, -np.log(disc) / T * 100, color=COLORS[1], lw=1, label=r"rate $r(T) = -\ln D(T)/T$")
    twin.set_ylabel("continuously compounded rate, %", color=COLORS[1])
    twin.set_ylim(3.5, 5.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="lower left")
    ax = axes[1]
    ax.plot(days, preg, "o", color=COLORS[3], ms=4, label="from the put–call parity regression")
    ax.plot(days, disc, "-", color=COLORS[0], label="from Treasury yields")
    ax.axhline(1, color="0.8", lw=0.8)
    ax.set_xlabel("maturity, trading days")
    ax.set_ylabel("discount factor")
    ax.set_title("Parity on American puts biases the discount upward")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    tidy(axes[0], axes[1])
    twin.spines[["top"]].set_visible(False)
    save(fig, "l02-rates-and-discounts.png")
    REPORT["rates"] = dict(r_1w=float(-np.log(disc[days == 5][0]) / (5 / 252)),
                           r_1y=float(-np.log(np.interp(252, days, disc)) / 1.0),
                           parity_above_one=int((preg > 1).sum()), n=len(preg),
                           parity_max=float(preg.max()), parity_max_days=int(days[np.argmax(preg)]))


def fig_forward(spy):
    days, fwd, disc, _ = expirations(spy)
    T = days / 252
    keep = days >= 5
    carry = np.log(fwd / SPOT) / T
    r = -np.log(disc) / T
    q = r - carry
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(days, fwd, "o-", color=COLORS[0], ms=3.5, label="forward $F_T$ from put–call parity")
    ax.axhline(SPOT, color="0.4", ls="--", lw=1, label=f"spot close {SPOT}")
    ax.set_xlabel("maturity, trading days")
    ax.set_ylabel("US dollars")
    ax.set_title("SPY forward curve, 21 October 2026")
    ax.legend(frameon=False, fontsize=8.5)
    ax = axes[1]
    ax.plot(days[keep], r[keep] * 100, color=COLORS[1], label="interest rate $r$ (Treasury)")
    ax.plot(days[keep], carry[keep] * 100, "o-", color=COLORS[0], ms=3.5, label=r"carry $\ln(F_T/S_0)/T = r - q$")
    ax.plot(days[keep], q[keep] * 100, "o-", color=COLORS[3], ms=3.5, label="implied dividend yield $q$")
    ax.axhline(0, color="0.85", lw=0.8)
    ax.set_xlabel("maturity, trading days")
    ax.set_ylabel("% per year")
    ax.set_title("The forward's premium over spot is the carry")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l02-spy-forward-and-carry.png")
    long = days >= 100
    REPORT["forward"] = dict(F_5=float(fwd[days == 5][0]), F_1y=float(np.interp(252, days, fwd)),
                             F_last=float(fwd[-1]), last_days=int(days[-1]),
                             q_long_median=float(np.median(q[long])), carry_long_median=float(np.median(carry[long])),
                             r_long_median=float(np.median(r[long])),
                             q_short_range=[float(q[keep & (days < 30)].min()), float(q[keep & (days < 30)].max())])


def spy_calls(spy, j):
    prices = np.array(spy["prices_forward_units"]).reshape(len(DAYS), len(DELTAS))
    T = DAYS[j] / 252
    k = DELTAS * sf.SIGMA_REF * np.sqrt(T)
    calls = np.where(DELTAS <= 0, prices[j] + 1 - np.exp(k), prices[j])
    return k, np.exp(k), calls, T


def fig_shape(spy):
    j = 4  # 63 days: wide enough a strike range to show the shape
    k, x, c, T = spy_calls(spy, j)
    atm = spy["atm_implied_vol"][j]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    xx = np.linspace(0.8, 1.2, 400)
    ax.fill_between(xx, np.maximum(1 - xx, 0), 1, color="0.94", label="no-arbitrage region")
    ax.plot(xx, np.maximum(1 - xx, 0), color="0.4", ls=":", label=r"lower bound $(1-x)^+$")
    ax.plot(x, c, "o-", color=COLORS[0], ms=4, label="SPY call prices (puts converted by parity)")
    ax.set_ylim(-0.005, 0.25)
    ax.set_xlabel(r"strike in forward units $x = K/F$")
    ax.set_ylabel("call price, fraction of the forward")
    ax.set_title("SPY, 63 days: inside the bounds, decreasing, convex")
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1]
    xm = 0.5 * (x[1:] + x[:-1])
    slope = np.diff(c) / np.diff(x)
    ax.plot(xm, -slope, "o-", color=COLORS[0], ms=4, label=r"SPY: $-\Delta c/\Delta x$")
    w = atm**2 * T
    xs = np.linspace(x[0], x[-1], 300)
    d1 = (-np.log(xs) + 0.5 * w) / np.sqrt(w)
    ax.plot(xs, norm.cdf(d1 - np.sqrt(w)), color=GREY, ls="--", label=f"log-normal at the ATM level {atm:.1%}")
    ax.set_xlabel(r"strike in forward units $x$")
    ax.set_ylabel(r"$\mathbb{Q}(X_T > x)$")
    ax.set_title("The slope is minus a probability")
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[2]
    x2 = x[1:-1]
    curv = 2 * ((c[2:] - c[1:-1]) / (x[2:] - x[1:-1]) - (c[1:-1] - c[:-2]) / (x[1:-1] - x[:-2])) / (x[2:] - x[:-2])
    ax.plot(x2, curv, "o-", color=COLORS[0], ms=4, label="SPY: second difference of $c$")
    dens = np.exp(-(np.log(xs) + 0.5 * w) ** 2 / (2 * w)) / (xs * np.sqrt(2 * np.pi * w))
    ax.plot(xs, dens, color=GREY, ls="--", label="log-normal at the ATM level")
    ax.set_xlabel(r"strike in forward units $x$")
    ax.set_ylabel(r"density $q(x)$")
    ax.set_title("The curvature is the density of $X_T$")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l02-bounds-and-shape.png")
    left = x2 < 0.95
    REPORT["shape"] = dict(T_days=int(DAYS[j]), atm=float(atm), x_range=[float(x[0]), float(x[-1])],
                           prob_above_lowest=float(-slope[0]), lognormal_prob_above_lowest=float(norm.cdf(
                               (-np.log(xm[0]) + 0.5 * w) / np.sqrt(w) - np.sqrt(w))),
                           min_curv=float(curv.min()),
                           dens_left=[float(v) for v in curv[left]],
                           logn_left=[float(np.interp(v, xs, dens)) for v in x2[left]])


def fig_calendar(spy):
    prices = np.array(spy["prices_forward_units"]).reshape(len(DAYS), len(DELTAS))
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    out = {}
    for color, kk in zip(COLORS, (-0.05, -0.02, 0.0, 0.02, 0.05)):
        vals, ds = [], []
        for j, day in enumerate(DAYS):
            T = day / 252
            k = DELTAS * sf.SIGMA_REF * np.sqrt(T)
            calls = np.where(DELTAS <= 0, prices[j] + 1 - np.exp(k), prices[j])
            if k[0] <= kk <= k[-1]:
                # interpolate the total implied variance (smooth in k), then convert back to a price
                iv = sf.implied_vol(calls, k, T, "call")
                v = float(np.interp(kk, k, iv))
                vals.append(float(sf.bs_call(kk, T, v)))
                ds.append(day)
        ax.plot(ds, vals, "o-", color=color, ms=3.5, label=f"$k$ = {kk:+.2f}")
        out[kk] = vals
    ax.set_xscale("log")
    ax.set_xticks(DAYS)
    ax.set_xticklabels([str(d) for d in DAYS])
    ax.set_xlabel("maturity, trading days")
    ax.set_ylabel("call price, fraction of the forward")
    ax.set_title("SPY: at fixed $k$, a call is worth more with more time")
    ax.legend(frameon=False, fontsize=8.5, title="log-moneyness", title_fontsize=8.5)
    tidy(ax)
    save(fig, "l02-calendar.png")
    REPORT["calendar_monotone"] = {str(k): bool(np.all(np.diff(v) > 0)) for k, v in out.items()}


def fig_replication():
    """Replicate the log contract -2 ln X_T with a finite strip of puts and calls."""
    X = np.linspace(0.5, 1.6, 600)
    g = -2 * np.log(X)

    def strip(dx):
        strikes = np.arange(dx, 3.0 + 1e-9, dx)
        weights = 2 / strikes**2 * dx  # g''(K) dK
        rep = -2 * (X - 1)  # g(1) + g'(1)(X - 1), with g(1) = 0 and g'(1) = -2
        for K, wgt in zip(strikes, weights):
            if K < 1 - 1e-12:
                rep = rep + wgt * np.maximum(K - X, 0)
            elif K > 1 + 1e-12:
                rep = rep + wgt * np.maximum(X - K, 0)
            else:
                rep = rep + 0.5 * wgt * (np.maximum(K - X, 0) + np.maximum(X - K, 0))
        return rep

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.plot(X, g, color="0.2", lw=2, label=r"target payoff $g(X_T) = -2\ln X_T$")
    ax.plot(X, -2 * (X - 1), color=GREY, ls=":", label=r"forward part $g(1) + g'(1)(X_T - 1)$")
    ax.plot(X, strip(0.2), color=COLORS[1], ls="--", label="forward + strip, strike spacing 0.2")
    ax.set_xlabel(r"$X_T = S_T/F$")
    ax.set_ylabel("payoff")
    ax.set_title("A curved payoff built from a forward and options")
    ax.legend(frameon=False, fontsize=8.5)
    ax = axes[1]
    errors = {}
    for color, dx in zip(COLORS, (0.2, 0.1, 0.05)):
        err = strip(dx) - g
        ax.plot(X, err, color=color, label=f"strike spacing {dx}")
        errors[dx] = float(np.max(np.abs(err)[(X > 0.7) & (X < 1.4)]))
    ax.axhline(0, color="0.85")
    ax.set_xlabel(r"$X_T$")
    ax.set_ylabel("replication error")
    ax.set_title("The error falls with the square of the spacing")
    ax.legend(frameon=False, fontsize=8.5)
    ax = axes[2]
    K = np.linspace(0.3, 2.5, 400)
    ax.plot(K, 2 / K**2, color=COLORS[0])
    ax.axvline(1, color="0.85")
    ax.text(0.62, 15, "puts", color="0.4")
    ax.text(1.6, 5, "calls", color="0.4")
    ax.set_xlabel("strike in forward units $K$")
    ax.set_ylabel(r"weight $g''(K) = 2/K^2$ per unit of strike")
    ax.set_title("The log contract weights low strikes most")
    tidy(*axes)
    save(fig, "l02-static-replication.png")
    REPORT["replication_max_error_0.7_1.4"] = errors


def main():
    spy = load_spy()
    fig_rates(spy)
    fig_forward(spy)
    fig_shape(spy)
    fig_calendar(spy)
    fig_replication()
    print(json.dumps(REPORT, indent=1))


if __name__ == "__main__":
    main()
