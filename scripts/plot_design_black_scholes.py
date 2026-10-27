"""Draw the two Black-Scholes figures of the calibration design document.

Figure 18: out-of-the-money prices in forward units against log-moneyness,
for several volatilities and several maturities.
Figure 19: the price of one contract as a function of sigma, with a quote and
its implied volatility, and the vega against log-moneyness.

Closed-form only; no bank, no network. Usage:
    python scripts/plot_design_black_scholes.py [--out docs/design/images]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

DAYS = 252.0
SIGMA_REF = 0.20
COLORS = ["#087F8C", "#D97732", "#738095", "#B33E43"]


def d1_d2(k, T, sigma):
    sd = sigma * np.sqrt(T)
    d1 = (-k + 0.5 * sd**2) / sd
    return d1, d1 - sd


def call(k, T, sigma):
    d1, d2 = d1_d2(k, T, sigma)
    return norm.cdf(d1) - np.exp(k) * norm.cdf(d2)


def put(k, T, sigma):
    return call(k, T, sigma) - 1.0 + np.exp(k)


def otm(k, T, sigma):
    """Put below the forward, call above it, as on the design grid."""
    return np.where(k < 0, put(k, T, sigma), call(k, T, sigma))


def vega(k, T, sigma):
    d1, _ = d1_d2(k, T, sigma)
    return np.sqrt(T) * norm.pdf(d1)


def implied_vol(y, k, T):
    return brentq(lambda s: float(otm(k, T, s)) - y, 1e-4, 5.0)


def figure_prices(path):
    k = np.linspace(-0.5, 0.5, 801)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    T = 21 / DAYS
    for color, sigma in zip(COLORS, (0.10, 0.20, 0.30, 0.40)):
        axes[0].plot(k, otm(k, T, sigma), color=color, label=rf"$\sigma$ = {sigma:.0%}")
    axes[0].set_title(r"Volatility varies, $T$ = 1 month")

    sigma = 0.20
    for color, days, name in zip(COLORS, (5, 21, 63, 252), ("1 week", "1 month", "3 months", "1 year")):
        T = days / DAYS
        axes[1].plot(k, otm(k, T, sigma), color=color, label=rf"$T$ = {name}")
        grid = np.arange(-2, 2.01, 0.5) * SIGMA_REF * np.sqrt(T)
        axes[1].plot(grid, otm(grid, T, sigma), "o", color=color, ms=3.5)
    axes[1].set_title(r"Maturity varies, $\sigma$ = 20% (dots: the 9 design strikes)")

    for ax in axes:
        ax.axvline(0, color="0.8", lw=0.8, zorder=0)
        ax.set_xlabel(r"log-moneyness $k = \ln(K/F)$")
        ax.text(-0.50, 0.004, "puts", color="0.45", fontsize=9)
        ax.text(0.44, 0.004, "calls", color="0.45", fontsize=9)
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("out-of-the-money price, fraction of the forward")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def figure_inversion(path):
    T = 21 / DAYS
    s = np.linspace(0.02, 0.6, 600)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    contracts = ((-1.0, "put, $\\delta = -1$"), (0.0, "ATM, $\\delta = 0$"), (1.0, "call, $\\delta = +1$"))
    for color, (delta, name) in zip(COLORS, contracts):
        k = delta * SIGMA_REF * np.sqrt(T)
        ax.plot(s, otm(k, T, s), color=color, label=name)
    k_q = -1.0 * SIGMA_REF * np.sqrt(T)
    y = 0.0080  # an illustrative quote for the delta = -1 put
    s_imp = implied_vol(y, k_q, T)
    ax.axhline(y, color="0.3", ls="--", lw=1)
    ax.plot([s_imp, s_imp], [0, y], color="0.3", ls=":", lw=1)
    ax.plot(s_imp, y, "o", color=COLORS[0])
    ax.annotate(rf"quote $y$ = {y:.4f}", (0.03, y), xytext=(0.36, y - 0.0035), fontsize=9, color="0.3")
    ax.annotate(rf"$\sigma_{{imp}}$ = {s_imp:.1%}", (s_imp, 0), xytext=(s_imp + 0.015, 0.0012),
                fontsize=9, color="0.3")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"volatility $\sigma$")
    ax.set_ylabel("price, fraction of the forward")
    ax.set_title(r"Price is increasing in $\sigma$, $T$ = 1 month")
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.0, 0.88))

    ax = axes[1]
    k = np.linspace(-0.5, 0.5, 801)
    for color, days, name in zip(COLORS, (5, 21, 63, 252), ("1 week", "1 month", "3 months", "1 year")):
        ax.plot(k, vega(k, days / DAYS, 0.20), color=color, label=rf"$T$ = {name}")
    ax.set_xlabel(r"log-moneyness $k$")
    ax.set_ylabel(r"vega $\partial c/\partial\sigma = \sqrt{T}\,\varphi(d_1)$")
    ax.set_title(r"Slope of the map $\sigma \mapsto$ price, $\sigma$ = 20%")
    ax.legend(frameon=False, fontsize=9)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return y, s_imp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/design/images")
    out = Path(parser.parse_args().out)
    figure_prices(out / "calibration-18-black-scholes-prices.png")
    y, s_imp = figure_inversion(out / "calibration-19-implied-volatility.png")
    print(f"quote {y} -> implied volatility {s_imp:.4f}")


if __name__ == "__main__":
    main()
