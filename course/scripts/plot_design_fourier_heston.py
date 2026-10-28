"""Draw the Appendix J figures of the calibration design document.

Figure 20: two Fourier inversions of the same Heston characteristic function
(Heston's two-probability form and Lewis's half-shift form): integrand
envelopes and the prices they return.
Figure 21: at-the-money skew against maturity, Heston (Fourier prices) against
the leading-order rough-Bergomi power law.
Figure 22: a forward-variance curve of the training prior against the best
Heston curve E[V_t] = theta + (v0 - theta) exp(-kappa t).

Closed-form and Fourier only; no bank, no network. Usage:
    python course/scripts/plot_design_fourier_heston.py [--out course/design-document/images]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lab" / "src"))
from sbi4finance import hkde  # noqa: E402

DAYS = 252.0
COLORS = ["#087F8C", "#D97732", "#738095", "#B33E43"]
DESIGN_DAYS = np.array([5, 10, 21, 42, 63, 126, 252])
# Centre of the Heston task's prior (the main text's Heston example surface).
HESTON = dict(kappa=2.0, theta=0.085, vol_of_vol=0.8, rho=-0.525, jump_intensity=0.0,
              jump_up_probability=0.5, jump_up_rate=10.0, jump_down_rate=10.0)
V0 = 0.085
# Centre of the rough-Bergomi prior.
H, ETA, RHO = 0.185, 2.0, -0.595


def phi(u, T, params=HESTON, v0=V0):
    return hkde.characteristic_function(u, T, params, v0)


def bs_call(k, T, s):
    sd = s * np.sqrt(T)
    d1 = (-k + 0.5 * sd**2) / sd
    return norm.cdf(d1) - np.exp(k) * norm.cdf(d1 - sd)


def implied_vol(c, k, T):
    return brentq(lambda s: bs_call(k, T, s) - c, 1e-4, 5.0, xtol=1e-14)


def lewis_call(k, T, params=HESTON, v0=V0, nodes=8192, upper=3000.0):
    return float(hkde.european_price(1.0, np.exp(k), 1.0, T, params, v0, kind="call",
                                     nodes=nodes, upper_limit=upper))


def heston_two_probability_call(k, T):
    """Heston (1993) form: c = P1 - e^k P2, each by Gil-Pelaez inversion."""
    def p(shift):
        f = lambda u: (np.exp(-1j * u * k) * phi(u + shift, T) / (1j * u)).real
        return 0.5 + quad(f, 1e-12, 4000.0, limit=4000)[0] / np.pi
    return p(-1j) - np.exp(k) * p(0.0)


def figure_inversions(path):
    T, k = 5 / DAYS, 0.02
    u = np.logspace(-1, 3.4, 600)
    gil = np.abs(phi(u, T)) / u                      # envelope of P2's integrand
    gil1 = np.abs(phi(u - 1j, T)) / u                # envelope of P1's integrand
    lew = np.abs(phi(u - 0.5j, T)) / (u**2 + 0.25)   # envelope of Lewis's integrand

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.loglog(u, gil, color=COLORS[1], label=r"Heston $P_2$: $|\varphi(u)|/u$")
    ax.loglog(u, gil1, color=COLORS[2], ls="--", label=r"Heston $P_1$: $|\varphi(u-i)|/u$")
    ax.loglog(u, lew, color=COLORS[0], label=r"Lewis: $|\varphi(u-\frac{i}{2})|/(u^2+\frac{1}{4})$")
    ax.set_xlabel(r"frequency $u$")
    ax.set_ylabel("integrand envelope")
    ax.set_ylim(1e-8, 20)
    ax.set_title("Integrands, Heston at the prior centre, $T$ = 1 week")
    ax.legend(frameon=False, fontsize=9, loc="lower left")

    ax = axes[1]
    ks = np.linspace(-0.06, 0.06, 13)
    lew_p = np.array([lewis_call(x, T) for x in ks])
    gil_p = np.array([heston_two_probability_call(x, T) for x in ks])
    ivs = [implied_vol(c, x, T) for c, x in zip(lew_p, ks)]
    ax.plot(ks, np.array(ivs) * 100, "o-", color=COLORS[0], ms=3.5, label="Lewis (one integral)")
    ivs2 = [implied_vol(c, x, T) for c, x in zip(gil_p, ks)]
    ax.plot(ks, np.array(ivs2) * 100, "x", color=COLORS[1], ms=7, label="Heston 1993 (two integrals)")
    ax.set_xlabel(r"log-moneyness $k$")
    ax.set_ylabel("implied volatility, %")
    gap = np.max(np.abs(lew_p - gil_p))
    ax.set_title(f"Same smile, $T$ = 1 week (largest price gap {gap:.0e})")
    ax.legend(frameon=False, fontsize=9)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return gap


def heston_atm_skew(T, params):
    h = 0.02 * np.sqrt(T)
    up = implied_vol(lewis_call(h, T, params), h, T)
    dn = implied_vol(lewis_call(-h, T, params), -h, T)
    return (up - dn) / (2 * h)


def figure_skew(path):
    T = np.geomspace(2 / DAYS, 1.0, 30)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for color, xi in zip(COLORS[1:], (0.4, 0.8, 1.6)):
        params = dict(HESTON, vol_of_vol=xi)
        skew = np.array([heston_atm_skew(t, params) for t in T])
        ax.loglog(T * DAYS, np.abs(skew), color=color, label=rf"Heston, $\sigma_v$ = {xi}")
        limit = abs(HESTON["rho"] * xi / (4 * np.sqrt(V0)))
        ax.axhline(limit, color=color, ls=":", lw=1)
    c = RHO * ETA * np.sqrt(2 * H) / (2 * (H + 0.5) * (H + 1.5))
    ax.loglog(T * DAYS, np.abs(c) * T ** (H - 0.5), color=COLORS[0], ls="--",
              label=rf"rough Bergomi, leading order $\propto T^{{H-1/2}}$, $H$ = {H}")
    ax.set_ylim(0.03, 2.0)
    ax.set_xlabel("maturity, trading days")
    ax.set_ylabel(r"$|\psi(T)|$, ATM skew per unit of $k$")
    ax.set_title("At-the-money skew against maturity")
    ax.set_xticks([2, 5, 10, 21, 63, 126, 252])
    ax.set_xticklabels(["2", "5", "10", "21", "63", "126", "252"])
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def prior_curve(seed=7, sigma0=0.25, slope=0.15):
    """One forward-variance curve of the training prior: piecewise constant."""
    rng = np.random.default_rng(seed)
    edges = np.concatenate([[0.0], DESIGN_DAYS / DAYS])
    mids = 0.5 * (edges[:-1] + edges[1:])
    levels = sigma0**2 * (1 + slope * mids) * np.exp(rng.uniform(-0.2, 0.2, len(mids)))
    return edges, levels


def heston_curve(t, v0, theta, kappa):
    return theta + (v0 - theta) * np.exp(-kappa * t)


def heston_total(T, v0, theta, kappa):
    return theta * T + (v0 - theta) * (-np.expm1(-kappa * T)) / kappa


def figure_curve(path):
    edges, levels = prior_curve()
    Tm = edges[1:]
    w_mkt = np.cumsum(levels * np.diff(edges))      # total variance at the design maturities
    kvar_mkt = np.sqrt(w_mkt / Tm)                   # variance-swap strikes, in volatility

    def fit(kappa=None):
        def resid(x):
            v0, theta = np.exp(x[:2])
            kap = kappa if kappa is not None else np.exp(x[2])
            return np.sqrt(heston_total(Tm, v0, theta, kap) / Tm) - kvar_mkt
        x0 = [np.log(levels[0]), np.log(levels[-1])] + ([] if kappa is not None else [np.log(2.0)])
        x = least_squares(resid, x0).x
        v0, theta = np.exp(x[:2])
        return v0, theta, (kappa if kappa is not None else np.exp(x[2]))

    fits = {r"Heston, $\kappa$ = 2 fixed": fit(2.0), r"Heston, $\kappa$ free": fit()}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    t = np.linspace(0, 1, 500)
    ax = axes[0]
    ax.stairs(levels, edges * DAYS, color="0.2", lw=1.6, baseline=None, label=r"$\xi_0(t)$, one prior draw")
    for color, (name, p) in zip(COLORS[:2], fits.items()):
        ax.plot(t * DAYS, heston_curve(t, *p), color=color, label=name)
    ax.set_xlabel("t, trading days")
    ax.set_ylabel("forward variance, per year")
    ax.set_title("Forward variance: input curve against Heston's family")
    ax.set_ylim(0.04, 0.09)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    ax = axes[1]
    for color, (name, p) in zip(COLORS[:2], fits.items()):
        err = (np.sqrt(heston_total(Tm, *p) / Tm) - kvar_mkt) * 100
        ax.plot(DESIGN_DAYS, err, "o-", color=color, label=name)
    ax.axhline(0, color="0.2", lw=1, label="rough Bergomi (exact by construction)")
    ax.set_xscale("log")
    ax.set_xticks(DESIGN_DAYS)
    ax.set_xticklabels([str(d) for d in DESIGN_DAYS])
    ax.set_xlabel("maturity, trading days")
    ax.set_ylabel(r"error in $\sqrt{K_{var}}$, volatility points")
    ax.set_title("Variance-swap strikes: best Heston fit minus input")
    ax.set_ylim(-1.0, 1.5)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    worst = {n: float(np.max(np.abs(np.sqrt(heston_total(Tm, *p) / Tm) - kvar_mkt)) * 100)
             for n, p in fits.items()}
    return worst, fits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="course/design-document/images")
    out = Path(parser.parse_args().out)
    gap = figure_inversions(out / "calibration-20-fourier-inversions.png")
    figure_skew(out / "calibration-21-skew-term-structure.png")
    worst, fits = figure_curve(out / "calibration-22-heston-variance-curve.png")
    print(f"largest price gap between the two inversions: {gap:.2e}")
    print("worst variance-swap error (vol points):", worst)
    print("fits (v0, theta, kappa):", fits)


if __name__ == "__main__":
    main()
