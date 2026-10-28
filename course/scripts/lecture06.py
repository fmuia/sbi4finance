"""Figures of Lecture 6: stochastic volatility, the Heston model.

Usage: python course/scripts/lecture06.py   (writes course/images/l06-*.png and prints the numbers quoted in the text)
"""

import json
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares
from scipy.special import roots_legendre
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm

from common import COLORS, DAYS, DELTAS, GREY, call, load_spy, plt, save, sf, tidy
from sbi4finance import hkde
from sbi4finance.calibration import pricing

REPORT = {}
CENTRE = dict(v0=0.085, theta=0.085, kappa=2.0, xi=0.8, rho=-0.525)  # centre of the tutorial's Heston prior


def hk(p):
    """Repository HKDE parameter dictionary with jumps off."""
    return dict(kappa=p["kappa"], theta=p["theta"], vol_of_vol=p["xi"], rho=p["rho"], jump_intensity=0.0,
                jump_up_probability=0.5, jump_up_rate=10.0, jump_down_rate=10.0)


# --------------------------------------------------------------------------- characteristic functions
def cf(u, T, p):
    """E[exp(iu ln X_T)] in the decaying-exponential (trap-free) form."""
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    b = p["kappa"] - p["rho"] * p["xi"] * iu
    d = np.sqrt(b * b + p["xi"] ** 2 * (u * u + iu))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    D = (b - d) / p["xi"] ** 2 * (1 - e) / (1 - g * e)
    C = p["kappa"] * p["theta"] / p["xi"] ** 2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    return np.exp(C + D * p["v0"])


def cf_original(u, T, p):
    """Heston's (1993) growing-exponential form, evaluated with the principal branch of the logarithm."""
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    b = p["kappa"] - p["rho"] * p["xi"] * iu
    d = np.sqrt(b * b + p["xi"] ** 2 * (u * u + iu))
    G = (b + d) / (b - d)
    e = np.exp(d * T)
    D = (b + d) / p["xi"] ** 2 * (1 - e) / (1 - G * e)
    C = p["kappa"] * p["theta"] / p["xi"] ** 2 * ((b + d) * T - 2 * np.log((1 - G * e) / (1 - G)))
    return np.exp(C + D * p["v0"])


# --------------------------------------------------------------------------- four inversions
def lewis(k, T, p, phi=cf, nodes=4096, U=3000.0):
    roots, weights = roots_legendre(nodes)
    u = 0.5 * U * (roots + 1)
    w = 0.5 * U * weights
    x = -np.atleast_1d(np.asarray(k, dtype=float))
    f = phi(u - 0.5j, T, p)
    integrand = np.real(np.exp(1j * np.outer(x, u)) * f) / (u**2 + 0.25)
    return 1 - np.exp(-x / 2) / np.pi * (integrand @ w)


def decay_cutoff(p, T, digits=30.0):
    """Cutoff where the exponential tail exp(-c u) of the Lewis integrand has fallen by e^-digits."""
    rate = (p["v0"] + p["kappa"] * p["theta"] * T) * np.sqrt(1 - p["rho"] ** 2) / p["xi"]
    return digits / rate


def lewis_composite(k, T, p, U=None, width=1.5, order=64):
    """Reference Lewis price: composite Gauss-Legendre, panels of width <= `width`, 64 nodes each.

    Single Gauss-Legendre rules with many nodes are off by 1e-11 to 1e-10 of the forward here (the report
    keys inversions_reference_check); panels of moderate order converge to about 1e-15.
    """
    U = decay_cutoff(p, T) if U is None else U
    panels = int(np.ceil(U / width))
    r, w = roots_legendre(order)
    edges = np.linspace(0.0, U, panels + 1)
    a, b = edges[:-1, None], edges[1:, None]
    u = (0.5 * (b - a) * (r[None, :] + 1) + a).ravel()
    wt = (0.5 * (b - a) * w[None, :]).ravel()
    x = -np.atleast_1d(np.asarray(k, dtype=float))
    f = cf(u - 0.5j, T, p)
    integrand = np.real(np.exp(1j * np.outer(x, u)) * f) / (u**2 + 0.25)
    return 1 - np.exp(-x / 2) / np.pi * (integrand @ wt)


def gil_pelaez(k, T, p):
    def prob(shift):
        f = lambda u: (np.exp(-1j * u * k) * cf(u + shift, T, p) / (1j * u)).real
        return 0.5 + quad(f, 1e-10, 3000.0, limit=4000)[0] / np.pi
    return prob(-1j) - np.exp(k) * prob(0.0)


def carr_madan(k, T, p, alpha=0.75):
    def integrand(u):
        z = cf(u - (alpha + 1) * 1j, T, p) / (alpha**2 + alpha - u**2 + 1j * (2 * alpha + 1) * u)
        return (np.exp(-1j * u * k) * z).real
    return np.exp(-alpha * k) / np.pi * quad(integrand, 0, 3000.0, limit=4000)[0]


def cos_put(k, T, p, N=256, L=12.0):
    wbar = max(p["v0"], p["theta"]) * T
    a, b = -0.5 * wbar - L * np.sqrt(wbar), -0.5 * wbar + L * np.sqrt(wbar)
    kk = np.arange(N)
    uk = kk * np.pi / (b - a)
    c, dd = a, k                       # the put pays (e^k - e^y)^+ for y in (a, k)
    psi = np.where(kk == 0, dd - c, (np.sin(uk * (dd - a)) - np.sin(uk * (c - a))) / np.where(kk == 0, 1, uk))
    chi = (np.cos(uk * (dd - a)) * np.exp(dd) - np.cos(uk * (c - a)) * np.exp(c)
           + uk * np.sin(uk * (dd - a)) * np.exp(dd) - uk * np.sin(uk * (c - a)) * np.exp(c)) / (1 + uk**2)
    V = 2 / (b - a) * (np.exp(k) * psi - chi)
    terms = np.real(cf(uk, T, p) * np.exp(-1j * uk * a)) * V
    terms[0] *= 0.5
    return terms.sum()


# --------------------------------------------------------------------------- smiles from the repository pricer
def contracts_for(k, T):
    k = np.atleast_1d(k)
    T = np.broadcast_to(np.atleast_1d(T), k.shape)
    return [{"strike": float(np.exp(a)), "maturity": float(b), "option_type": "put" if a <= 0 else "call"}
            for a, b in zip(k, T)]


def surface_prices(p, k, T, nodes=4096, U=3000.0):
    return pricing.heston_surface(hk(p), p["v0"], contracts_for(k, T), nodes=nodes, upper_limit=U)


def implied(p, k, T, **kw):
    k = np.atleast_1d(k)
    T = np.broadcast_to(np.atleast_1d(T), k.shape)
    prices = surface_prices(p, k, T, **kw)
    return sf.implied_vol(prices, k, T, np.where(k <= 0, "put", "call"))


def atm_skew(p, T, h=0.01):
    """Local ATM skew by a central difference in k."""
    T = np.atleast_1d(T)
    hh = h * np.sqrt(T)
    up = implied(p, hh, T)
    dn = implied(p, -hh, T)
    return (up - dn) / (2 * hh)


def quad_skew(p, T):
    """ATM skew with the estimator used for SPY: a quadratic in k through delta in {-1, -0.5, 0, 0.5, 1}."""
    out = []
    for t in np.atleast_1d(T):
        k = np.array([-1, -0.5, 0, 0.5, 1]) * sf.SIGMA_REF * np.sqrt(t)
        iv = implied(p, k, np.full(5, t))
        out.append(np.polyfit(k, iv, 2)[1])
    return np.array(out)


def grid_k(day):
    return DELTAS * sf.SIGMA_REF * np.sqrt(day / 252)


# --------------------------------------------------------------------------- simulation
def simulate(p, years, steps_per_day, n_paths, seed, record_every=None):
    """Full-truncation Euler for (ln X, V)."""
    rng = np.random.default_rng(seed)
    n = int(round(years * 252 * steps_per_day))
    dt = years / n
    v = np.full(n_paths, p["v0"])
    x = np.zeros(n_paths)
    record_every = record_every or steps_per_day
    vs, xs = [v.copy()], [x.copy()]
    for i in range(n):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        vp = np.maximum(v, 0.0)
        x = x - 0.5 * vp * dt + np.sqrt(vp * dt) * (p["rho"] * z1 + np.sqrt(1 - p["rho"] ** 2) * z2)
        v = v + p["kappa"] * (p["theta"] - vp) * dt + p["xi"] * np.sqrt(vp * dt) * z1
        if (i + 1) % record_every == 0:
            vs.append(np.maximum(v, 0.0))
            xs.append(x.copy())
    return np.array(vs), np.array(xs)


# =========================================================================== figures
def fig_cir():
    kappa, theta = 2.0, 0.085
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    v = np.linspace(1e-4, 0.4, 800)
    out = {}
    for color, xi in zip(COLORS, (0.3, np.sqrt(2 * kappa * theta), 0.8)):
        shape, scale = 2 * kappa * theta / xi**2, xi**2 / (2 * kappa)
        ax.plot(np.sqrt(v) * 100, gamma_dist.pdf(v, shape, scale=scale) * 2 * np.sqrt(v),
                color=color, label=rf"$\sigma_v$ = {xi:.2f}: shape $2\kappa\theta/\sigma_v^2$ = {shape:.2f}")
        out[f"{xi:.3f}"] = dict(shape=float(shape), p_vol_below_10=float(gamma_dist.cdf(0.01, shape, scale=scale)))
    ax.set_xlabel(r"volatility $\sqrt{V}$, %")
    ax.set_ylabel(r"stationary density of $\sqrt{V}$")
    ax.set_ylim(0, 8)
    ax.set_title(r"Stationary law of the variance, $\kappa$ = 2, $\theta$ = 0.085")
    ax.legend(frameon=False, fontsize=8.5)
    ax = axes[1]
    for color, xi in zip((COLORS[0], COLORS[2]), (0.3, 0.8)):
        p = dict(CENTRE, xi=xi)
        V, _ = simulate(p, 2.0, 8, 1, seed=3)
        ax.plot(np.arange(len(V)) / 252, np.sqrt(V[:, 0]) * 100, color=color, lw=0.9, label=rf"$\sigma_v$ = {xi}")
        out[f"path_vol_range_xi_{xi}"] = [float(np.sqrt(max(V[:, 0].min(), 0))), float(np.sqrt(V[:, 0].max()))]
    ax.set_xlabel("years")
    ax.set_ylabel(r"$\sqrt{V_t}$, %")
    ax.set_title("One variance path each: Feller satisfied and violated")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l06-cir-feller.png")
    rng = np.random.default_rng(0)
    th = rng.uniform(0.01, 0.16, 200_000)
    xi = rng.uniform(0.1, 1.5, 200_000)
    REPORT["feller"] = dict(by_xi=out, centre_2kt=2 * kappa * theta, centre_xi2=0.8**2,
                            prior_share_violating=float(np.mean(xi**2 > 2 * 2.0 * th)))


def fig_paths():
    p = dict(CENTRE)
    V, X = simulate(p, 2.0, 8, 1, seed=11)
    v = V[:, 0]
    x = X[:, 0]
    r = np.diff(x)
    dv = np.diff(v)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.2), sharex=True)
    t = np.arange(len(v)) / 252
    axes[0].plot(t, 100 * np.exp(x), color=COLORS[0], lw=1)
    axes[0].set_ylabel("price, start 100")
    axes[0].set_title("A Heston path at the centre of the tutorial's prior (two years, daily)")
    axes[1].plot(t, np.sqrt(v) * 100, color=COLORS[3], lw=1)
    axes[1].axhline(np.sqrt(p["theta"]) * 100, color=GREY, ls=":", lw=1)
    axes[1].set_ylabel(r"$\sqrt{V_t}$, %")
    axes[2].plot(t[1:], r * 100, color=COLORS[2], lw=0.6)
    axes[2].set_ylabel("daily log return, %")
    axes[2].set_xlabel("years")
    tidy(*axes)
    save(fig, "l06-paths.png")
    seg = {}
    for a, b in ((0, 0.5), (0.5, 1), (1, 1.25), (1.25, 1.5), (1.5, 2)):
        m = (t >= a) & (t < b)
        seg[f"{a}-{b}"] = dict(mean_vol=float(np.sqrt(np.maximum(v[m], 0)).mean()),
                               price_start=float(100 * np.exp(x[m][0])), price_end=float(100 * np.exp(x[m][-1])))
    REPORT["paths_segments"] = seg
    # many paths for the statistics of clustering and leverage
    Vm, Xm = simulate(p, 1.0, 8, 4000, seed=12)
    rm = np.diff(Xm, axis=0)
    dvm = np.diff(Vm, axis=0)
    corr = float(np.mean([np.corrcoef(rm[:, j], dvm[:, j])[0, 1] for j in range(400)]))
    ab = np.abs(rm)
    ac1 = float(np.mean([np.corrcoef(ab[:-1, j], ab[1:, j])[0, 1] for j in range(400)]))
    z = (rm / rm.std()).ravel()
    REPORT["paths"] = dict(corr_return_dvar=corr, autocorr_abs_returns_lag1=ac1,
                           kurtosis=float(np.mean(z**4)), vol_range=[float(np.sqrt(v.min())), float(np.sqrt(v.max()))])


def fig_expected_variance(spy):
    t = np.linspace(0, 1.0, 400)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    ax = axes[0]
    theta = 0.085
    for color, v0 in zip((COLORS[0], COLORS[3]), (0.04, 0.16)):
        for ls, kappa in zip((":", "-", "--"), (0.5, 2.0, 8.0)):
            ev = theta + (v0 - theta) * np.exp(-kappa * t)
            ax.plot(t * 252, np.sqrt(ev) * 100, color=color, ls=ls,
                    label=rf"$v_0$ = {v0}, $\kappa$ = {kappa}" if True else None)
    p = dict(CENTRE, v0=0.04)
    V, _ = simulate(p, 1.0, 8, 20000, seed=5, record_every=8 * 21)
    tm = np.arange(V.shape[0]) * 21
    ax.plot(tm, np.sqrt(V.mean(axis=1)) * 100, "o", color=COLORS[0], ms=4, label=r"simulated mean, $v_0$ = 0.04, $\kappa$ = 2")
    ax.axhline(np.sqrt(theta) * 100, color=GREY, lw=0.8)
    ax.set_xlabel("t, days (252 a year)")
    ax.set_ylabel(r"$\sqrt{\mathbb{E}[V_t]}$, % per year")
    ax.set_title(r"Expected variance relaxes to $\theta$ at rate $\kappa$")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    REPORT["expected_variance_mc_check"] = dict(simulated=float(V.mean(axis=1)[-1]),
                                                exact=float(theta + (0.04 - theta) * np.exp(-2.0)))

    ax = axes[1]
    T = DAYS / 252
    wvar = np.array(spy["diagnostics"]["design_total_variance"])
    xi_levels = np.array(spy["forward_variance_levels"])
    edges = np.r_[0, DAYS]
    ax.stairs(np.sqrt(xi_levels) * 100, edges, color="0.2", lw=1.8, baseline=None, label=r"SPY $\sqrt{\xi_0(t)}$")

    def w_heston(Tm, v0, th, ka):
        return th * Tm + (v0 - th) * (-np.expm1(-ka * Tm)) / ka

    def fit_curve(kappa_fixed=None):
        def resid(z):
            v0, th = np.exp(z[:2])
            ka = kappa_fixed if kappa_fixed is not None else np.exp(z[2])
            return (np.sqrt(w_heston(T, v0, th, ka) / T) - np.sqrt(wvar / T)) * 100
        z0 = [np.log(xi_levels[0]), np.log(xi_levels[-1])] + ([] if kappa_fixed is not None else [np.log(2.0)])
        sol = least_squares(resid, z0)
        v0, th = np.exp(sol.x[:2])
        ka = kappa_fixed if kappa_fixed is not None else float(np.exp(sol.x[2]))
        return float(v0), float(th), float(ka), float(np.max(np.abs(sol.fun)))

    fits = {"kappa = 2 fixed": fit_curve(2.0), "kappa free": fit_curve()}
    tt = np.linspace(0, 1, 400)
    for color, (name, (v0, th, ka, err)) in zip((COLORS[0], COLORS[1]), fits.items()):
        ax.plot(tt * 252, np.sqrt(th + (v0 - th) * np.exp(-ka * tt)) * 100, color=color,
                label=rf"Heston fit, $\kappa$ {'= 2' if 'fixed' in name else 'free'}: max error {err:.2f} vol pts")
    ax.set_xlabel("t, days (252 a year)")
    ax.set_ylabel("% per year")
    ax.set_title("SPY's forward-variance curve and Heston's best match")
    ax.legend(frameon=False, fontsize=8)
    tidy(*axes)
    save(fig, "l06-expected-variance.png")
    REPORT["xi_fit"] = {k: dict(v0=v[0], theta=v[1], kappa=v[2], max_err_volpts=v[3]) for k, v in fits.items()}
    REPORT["spy_xi_vol"] = [float(x) for x in np.sqrt(xi_levels)]


def fig_trap():
    p = dict(CENTRE)
    u = np.linspace(0.001, 25, 2500)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    T = 5.0
    ax.plot(u, cf(u, T, p).real, color=COLORS[0], lw=2, label="decaying-exponential form")
    ax.plot(u, cf_original(u, T, p).real, color=COLORS[3], ls="--", lw=1.2, label="original form, principal log")
    ax.set_xlabel("frequency $u$")
    ax.set_ylabel(r"Re $\varphi_T(u)$")
    ax.set_title(r"The same function, two formulas: $T$ = 5 years")
    ax.set_xlim(0, 10)
    ax.legend(frameon=False, fontsize=8.5)
    ax = axes[1]
    k = np.linspace(-0.6, 0.6, 25)
    out = {}
    for color, T in zip(COLORS, (1.0, 2.0, 5.0)):
        good = lewis(k, T, p, nodes=2048, U=200.0)
        bad = lewis(k, T, p, phi=cf_original, nodes=2048, U=200.0)
        iv_good = sf.implied_vol(good, k, T, "call")
        iv_bad = sf.implied_vol(bad, k, T, "call")
        ax.plot(k, (iv_bad - iv_good) * 100, "o-", color=color, ms=3, label=f"$T$ = {T:.0f} years")
        out[T] = dict(max_price_error=float(np.max(np.abs(bad - good))), max_iv_error_pts=float(np.nanmax(np.abs(iv_bad - iv_good)) * 100))
    ax.axhline(0, color="0.8")
    ax.set_xlabel(r"log-moneyness $k$")
    ax.set_ylabel("implied-volatility error, volatility points")
    ax.set_title("Error of prices computed with the original form")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l06-heston-trap.png")
    REPORT["trap"] = out
    uu = np.linspace(0.001, 60, 6000)
    REPORT["trap_max_cf_gap_T5"] = float(np.max(np.abs(cf(uu, 5.0, p) - cf_original(uu, 5.0, p))))
    REPORT["trap_max_cf_gap_T1"] = float(np.max(np.abs(cf(uu, 1.0, p) - cf_original(uu, 1.0, p))))
    uf = np.linspace(0.001, 10, 20001)
    good, bad = cf(uf, 5.0, p).real, cf_original(uf, 5.0, p).real
    i = int(np.argmax(np.abs(good - bad) > 1e-6))
    REPORT["trap_first_departure_T5"] = dict(u=float(uf[i]), correct=float(good[i]), original=float(bad[i]))


def fig_inversions():
    p = dict(CENTRE)
    T = 21 / 252
    ks = np.linspace(-0.25, 0.2, 10)
    t0 = time.process_time()
    # Reference: composite Gauss-Legendre Lewis integral to the exponential cutoff (converged to about 1e-15)
    ref = lewis_composite(ks, T, p)
    REPORT["inversions_reference_check"] = dict(
        reference_cutoff=float(decay_cutoff(p, T)),
        cos_1024_vs_reference=float(np.max(np.abs(np.array([cos_put(k, T, p, N=1024) for k in ks]) + 1 - np.exp(ks) - ref))),
        old_reference_16384_8000_vs_reference=float(np.max(np.abs(lewis(ks, T, p, nodes=16384, U=8000.0) - ref))),
        lewis_32768_400_vs_reference=float(np.max(np.abs(lewis(ks, T, p, nodes=32768, U=400.0) - ref))),
        composite_halved_width_change=float(np.max(np.abs(lewis_composite(ks, T, p, width=0.75) - ref))))
    methods = {}
    t0 = time.process_time(); lw = lewis(ks, T, p); methods["Lewis (Gauss–Legendre, 4096 nodes)"] = (lw, time.process_time() - t0)
    t0 = time.process_time(); gp = np.array([gil_pelaez(k, T, p) for k in ks]); methods["Heston 1993 (two integrals)"] = (gp, time.process_time() - t0)
    t0 = time.process_time(); cm = np.array([carr_madan(k, T, p) for k in ks]); methods[r"Carr–Madan, $\alpha$ = 0.75"] = (cm, time.process_time() - t0)
    t0 = time.process_time(); co = np.array([cos_put(k, T, p) for k in ks]) + 1 - np.exp(ks); methods["COS, 256 terms"] = (co, time.process_time() - t0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3))
    ivref = sf.implied_vol(ref, ks, T, "call")
    axes[0].plot(ks, ivref * 100, color="0.2", lw=2, label="reference (Lewis, composite quadrature)")
    marks = ["o", "x", "s", "^"]
    out = {}
    for color, m, (name, (pr, sec)) in zip(COLORS, marks, methods.items()):
        axes[0].plot(ks, sf.implied_vol(pr, ks, T, "call") * 100, m, color=color, ms=5, mfc="none", label=name)
        axes[1].semilogy(ks, np.maximum(np.abs(pr - ref), 1e-17), m + "-", color=color, ms=4, label=name)
        out[name] = dict(max_abs_error=float(np.max(np.abs(pr - ref))), min_abs_error=float(np.min(np.abs(pr - ref))),
                         cpu_s_10_prices=float(sec))
    axes[0].set_xlabel(r"log-moneyness $k$")
    axes[0].set_ylabel("implied volatility, %")
    axes[0].set_title("Four inversions of one characteristic function, $T$ = 1 month")
    axes[0].legend(frameon=False, fontsize=8)
    sig = np.array(sf.quote_noise(contracts_for(ks, T)))
    axes[1].semilogy(ks, sig, color=GREY, ls=":", label=r"quote noise $\sigma_c$ at that strike")
    axes[1].set_xlabel(r"log-moneyness $k$")
    axes[1].set_ylabel("|difference from the reference|, fraction of forward")
    axes[1].set_title("All agree far below the quote noise")
    axes[1].legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    tidy(*axes)
    save(fig, "l06-four-inversions.png")
    REPORT["inversions"] = out
    REPORT["inversions_min_quote_noise"] = float(sig.min())
    REPORT["inversions_min_quote_noise_at_k"] = float(ks[np.argmin(sig)])
    REPORT["design_grid_k_range_1m"] = float(grid_k(21).max())


DECAY = {}
REF_CHECK = {}


def decay_rate(p, T):
    """Predicted and fitted exponential decay rate of |phi_T(u - i/2)| beyond the crossover frequency."""
    rate = (p["v0"] + p["kappa"] * p["theta"] * T) * np.sqrt(1 - p["rho"] ** 2) / p["xi"]
    cross = 1.0 / (p["xi"] * np.sqrt(1 - p["rho"] ** 2) * T)
    u = np.linspace(5 * cross, 5 * cross + 200.0 / rate, 400)     # well past the crossover, before underflow
    fitted = -np.polyfit(u, np.log(np.abs(cf(u - 0.5j, T, p))), 1)[0]
    return dict(predicted_rate=float(rate), fitted_rate=float(fitted), crossover_u=float(cross),
                fit_range=[float(u[0]), float(u[-1])], reference_cutoff=float(45.0 / rate))


def fig_ladder():
    ladder = ((512, 400.0), (1024, 600.0), (2048, 1500.0), (4096, 3000.0), (8192, 6000.0))
    cases = {
        "centre of the prior, 1 year": (dict(CENTRE), 252),
        r"$v_0 = \theta$ = 0.01, $\sigma_v$ = 1.5, $\rho$ = −0.95, 1 week": (
            dict(v0=0.01, theta=0.01, kappa=2.0, xi=1.5, rho=-0.95), 5),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    out = {}
    for color, (name, (p, day)) in zip((COLORS[0], COLORS[3]), cases.items()):
        T = day / 252
        ks = grid_k(day)
        sig = np.array(sf.quote_noise(contracts_for(ks, np.full(9, T))))
        rate = (p["v0"] + p["kappa"] * p["theta"] * T) * np.sqrt(1 - p["rho"] ** 2) / p["xi"]
        ref = lewis_composite(ks, T, p)
        REF_CHECK[name] = float(np.max(np.abs(lewis_composite(ks, T, p, width=0.75) - ref) / sig))
        otm_ref = np.where(ks <= 0, ref - 1 + np.exp(ks), ref)
        errs, viol = [], []
        for n, U in ladder:
            c = lewis(ks, T, p, nodes=n, U=U)
            otm_c = np.where(ks <= 0, c - 1 + np.exp(ks), c)
            errs.append(float(np.max(np.abs(otm_c - otm_ref) / sig)))
            viol.append(int(np.sum(otm_c < 0)))
        axes[0].semilogy(range(1, 6), np.maximum(errs, 1e-16), "o-", color=color, ms=4, label=name)
        u = np.geomspace(0.1, 6000, 3000)
        env = np.abs(cf(u - 0.5j, T, p)) / (u**2 + 0.25)
        axes[1].loglog(u, env, color=color, label=name)
        axes[1].loglog(u, np.exp(-rate * u) / (u**2 + 0.25), ":", color=color, lw=1.2)
        DECAY[name] = decay_rate(p, T)
        out[name] = dict(max_error_over_sigma=errs, negative_otm_prices=viol,
                         rung_agreement=[float(abs(a - b)) for a, b in zip(errs[:-1], errs[1:])])
    axes[0].axhline(0.1, color=GREY, ls=":", lw=1)
    axes[0].text(3.2, 0.13, r"$0.1\,\sigma_c$: the acceptance tolerance", fontsize=8.5, color="0.4")
    axes[0].set_xticks(range(1, 6))
    axes[0].set_xticklabels([f"{n}\n{U:.0f}" for n, U in ladder], fontsize=8)
    axes[0].set_xlabel("rung: Gauss–Legendre nodes / cutoff $U$")
    axes[0].set_ylabel(r"largest error over the 9 strikes / $\sigma_c$")
    axes[0].set_title("Price error along the quadrature ladder")
    axes[0].legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(0.0, 0.45))
    axes[1].set_xlabel("frequency $u$")
    axes[1].set_ylabel(r"$|\varphi_T(u - i/2)|/(u^2 + 1/4)$")
    axes[1].set_ylim(1e-14, 10)
    axes[1].set_title(r"Exponential tail $e^{-cu}$ (dotted), $c = (v_0 + \kappa\theta T)\sqrt{1-\rho^2}/\sigma_v$", fontsize=10)
    tidy(*axes)
    save(fig, "l06-quadrature-ladder.png")
    REPORT["ladder"] = out
    REPORT["ladder_decay"] = DECAY
    REPORT["ladder_reference_halved_width_change_over_sigma"] = REF_CHECK
    hard = dict(v0=0.01, theta=0.01, kappa=2.0, xi=1.5, rho=-0.95)
    T = 5 / 252
    con = contracts_for(grid_k(5), np.full(9, T))
    sig = np.array(sf.quote_noise(con))
    rungs = []
    for n, U in ladder:
        try:
            rungs.append(pricing.heston_surface(hk(hard), hard["v0"], con, nodes=n, upper_limit=U))
        except ValueError:
            rungs.append(None)
    REPORT["ladder_repo_pricer_hard_case"] = dict(
        rung_raises_bound_violation=[r is None for r in rungs],
        accepted_pair_agreement_over_sigma=[None if a is None or b is None else float(np.max(np.abs(a - b) / sig))
                                            for a, b in zip(rungs[:-1], rungs[1:])],
        own_lewis_first_pair_disagreement_over_sigma=float(np.max(np.abs(lewis(grid_k(5), T, hard, nodes=512, U=400.0)
                                                                         - lewis(grid_k(5), T, hard, nodes=1024, U=600.0)) / sig)))
    uu = np.geomspace(0.1, 20000, 40000)
    decay = {}
    for name, (p, day) in zip(("centre_1y", "hard_1w"), ((dict(CENTRE), 252), (hard, 5))):
        env = np.abs(cf(uu - 0.5j, day / 252, p)) / (uu**2 + 0.25)
        decay[name] = float(uu[np.argmax(env < 1e-10)])
    REPORT["ladder_integrand_below_1e-10_at_u"] = decay


SWEEPS = {
    "v0": ("v0", (0.04, 0.085, 0.16), r"$v_0$", "l06-smile-v0.png", "atm"),
    "theta": ("theta", (0.04, 0.085, 0.16), r"$\theta$", "l06-smile-theta.png", "atm"),
    "kappa": ("kappa", (0.5, 2.0, 8.0), r"$\kappa$", "l06-smile-kappa.png", "atm"),
    "xi": ("xi", (0.4, 0.8, 1.5), r"$\sigma_v$", "l06-smile-volofvol.png", "skew"),
    "rho": ("rho", (-0.9, -0.525, 0.0), r"$\rho$", "l06-smile-rho.png", "skew"),
}


def fig_sweep(key):
    name, values, label, fname, right = SWEEPS[key]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    out = {}
    for color, val in zip(COLORS, values):
        p = dict(CENTRE, **{name: val})
        iv1 = implied(p, grid_k(21), 21 / 252)
        iv12 = implied(p, grid_k(252), 1.0)
        axes[0].plot(DELTAS, iv1 * 100, "o-", color=color, ms=3.5, label=f"{label} = {val}")
        axes[1].plot(DELTAS, iv12 * 100, "o-", color=color, ms=3.5, label=f"{label} = {val}")
        T = DAYS / 252
        if right == "atm":
            atm = implied(p, np.zeros(len(T)), T)
            axes[2].plot(DAYS, atm * 100, "o-", color=color, ms=3.5, label=f"{label} = {val}")
            rv = atm
        elif key == "rho":
            sk = quad_skew(p, T)
            axes[2].semilogx(DAYS, sk, "o-", color=color, ms=3.5, label=f"{label} = {val}")
            rv = sk
        else:
            sk = quad_skew(p, T)
            axes[2].loglog(DAYS, np.abs(sk), "o-", color=color, ms=3.5, label=f"{label} = {val}")
            rv = sk
        out[str(val)] = dict(iv_1m=[float(x) for x in iv1], iv_1y=[float(x) for x in iv12], right=[float(x) for x in rv])
    for ax, title in zip(axes[:2], ("1 month", "1 year")):
        ax.set_xlabel(r"standardized moneyness $\delta$")
        ax.set_ylabel("implied volatility, %")
        ax.set_title(f"Smile at {title}")
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[2].set_xscale("log")
    axes[2].set_xticks(DAYS)
    axes[2].set_xticklabels([str(d) for d in DAYS])
    axes[2].set_xlabel("maturity, business days")
    if right == "atm":
        axes[2].set_ylabel("ATM implied volatility, %")
        axes[2].set_title("At-the-money term structure")
    else:
        axes[2].set_ylabel(r"$\psi(T)$, ATM skew" if key == "rho" else r"$|\psi(T)|$, ATM skew")
        axes[2].set_title("At-the-money skew term structure")
    fig.suptitle(f"Heston: varying {label}, others at the centre of the prior", y=1.0, fontsize=11)
    tidy(*axes)
    save(fig, fname)
    REPORT[f"sweep_{key}"] = out


def fig_skew(spy):
    T = np.geomspace(2 / 252, 1.0, 24)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    out = {}
    for color, xi in zip(COLORS[1:], (0.4, 0.8, 1.5)):
        p = dict(CENTRE, xi=xi)
        sk = np.abs(quad_skew(p, T))
        ax.loglog(T * 252, sk, color=color, label=rf"Heston, $\sigma_v$ = {xi}")
        lim = abs(p["rho"] * xi / (4 * np.sqrt(p["v0"])))
        ax.axhline(lim, color=color, ls=":", lw=1)
        out[xi] = dict(limit=float(lim), at_2d=float(sk[0]), at_1m=float(np.interp(21, T * 252, sk)), at_1y=float(sk[-1]))
    spy_sk = np.abs(np.array(spy["atm_skew"]))
    ax.loglog(DAYS, spy_sk, "o", color=COLORS[0], ms=6, label=r"SPY, 21 Oct 2026 ($\propto T^{-0.32}$)")
    ax.set_xticks([2, 5, 10, 21, 63, 126, 252])
    ax.set_xticklabels(["2", "5", "10", "21", "63", "126", "252"])
    ax.set_xlabel("maturity, business days")
    ax.set_ylabel(r"$|\psi(T)|$, ATM skew per unit of $k$")
    ax.set_title("Heston's skew flattens at short maturities; the market's steepens")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    tidy(ax)
    save(fig, "l06-skew-term-structure.png")
    REPORT["skew"] = out


def fig_spy_fit(spy):
    contracts = sf.design_grid()
    y = np.array(spy["prices_forward_units"])
    sig = np.array(spy["noise_sd"])
    names = ["v0", "theta", "kappa", "xi", "rho"]
    lo = np.array([1e-4, 1e-4, 0.05, 0.02, -0.999])
    hi = np.array([1.0, 1.0, 30.0, 5.0, 0.999])
    counter = {"n": 0}

    def model(z, kappa_fixed=None):
        p = dict(zip(names, z))
        if kappa_fixed is not None:
            p["kappa"] = kappa_fixed
        return pricing.heston_surface(hk(p), p["v0"], contracts, nodes=4096, upper_limit=3000.0)

    def resid(z, kappa_fixed=None):
        counter["n"] += 1
        try:
            return (model(z, kappa_fixed) - y) / sig
        except ValueError:
            return np.full(len(y), 1e3)

    best = {}
    t0 = time.process_time()
    for label, kfix in (("kappa free", None), ("kappa = 2 fixed", 2.0)):
        sols = []
        for start in ([0.02, 0.05, 2.0, 0.8, -0.7], [0.015, 0.03, 5.0, 1.2, -0.9], [0.03, 0.06, 1.0, 0.5, -0.5]):
            z0 = np.array(start)
            if kfix is not None:
                lo2, hi2 = lo.copy(), hi.copy()
                lo2[2], hi2[2] = 1.999, 2.001
                z0[2] = 2.0
                sol = least_squares(lambda z: resid(z, kfix), z0, bounds=(lo2, hi2), x_scale="jac")
            else:
                sol = least_squares(resid, z0, bounds=(lo, hi), x_scale="jac")
            sols.append(sol)
        sol = min(sols, key=lambda s: s.cost)
        best[label] = sol
    REPORT["spy_fit_cpu_s"] = float(time.process_time() - t0)
    REPORT["spy_fit_evaluations"] = counter["n"]
    sol = best["kappa free"]
    z = sol.x
    pfit = dict(zip(names, z))
    fine = pricing.heston_surface(hk(pfit), pfit["v0"], contracts, nodes=8192, upper_limit=6000.0)
    res = ((fine - y) / sig).reshape(len(DAYS), len(DELTAS))
    k_all, T_all, kind = sf.contract_arrays(contracts)
    iv_mkt = sf.implied_vol(y, k_all, T_all, kind).reshape(len(DAYS), len(DELTAS))
    iv_fit = sf.implied_vol(fine, k_all, T_all, kind).reshape(len(DAYS), len(DELTAS))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
    ax = axes[0]
    for color, j in zip((COLORS[0], COLORS[1], COLORS[3]), (0, 2, 6)):
        ax.plot(DELTAS, iv_mkt[j] * 100, "o", color=color, ms=5, label=f"SPY, {DAYS[j]} d")
        ax.plot(DELTAS, iv_fit[j] * 100, "-", color=color, lw=1.5, label=f"Heston fit, {DAYS[j]} d")
    ax.set_xlabel(r"standardized moneyness $\delta$")
    ax.set_ylabel("implied volatility, %")
    ax.set_title("Best weighted least-squares Heston fit to SPY")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax = axes[1]
    im = ax.imshow(res, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    for j in range(len(DAYS)):
        for i in range(len(DELTAS)):
            ax.text(i, j, f"{res[j, i]:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(res[j, i]) > 2 else "0.1")
    ax.set_xticks(range(len(DELTAS)))
    ax.set_xticklabels([f"{d:g}" for d in DELTAS])
    ax.set_yticks(range(len(DAYS)))
    ax.set_yticklabels([f"{d} d" for d in DAYS])
    ax.set_xlabel(r"standardized moneyness $\delta$")
    ax.set_title(r"Residuals (model − quote) / $\sigma_c$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax = axes[2]
    T = DAYS / 252
    ax.loglog(DAYS, np.abs(np.array(spy["atm_skew"])), "o", color=COLORS[0], ms=6, label="SPY")
    fit_skew = np.abs(sf.smile_statistics(iv_fit.ravel(), k_all, T_all)[2])
    ax.loglog(DAYS, fit_skew, "s-", color=COLORS[3], ms=4, label="Heston fit (same estimator)")
    TT = np.geomspace(1 / 252, 1.0, 20)
    ax.loglog(TT * 252, np.abs(quad_skew(pfit, TT)), ":", color=COLORS[3], lw=1, label="Heston fit, all maturities")
    ax.axhline(abs(pfit["rho"] * pfit["xi"] / (4 * np.sqrt(pfit["v0"]))), color=GREY, ls="--", lw=0.8,
               label=r"Heston limit $|\rho\sigma_v|/(4\sqrt{v_0})$")
    ax.set_xticks(DAYS)
    ax.set_xticklabels([str(d) for d in DAYS])
    ax.set_xlabel("maturity, business days")
    ax.set_ylabel(r"$|\psi(T)|$")
    ax.set_title("Skew term structure: SPY against the Heston fit")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(axes[0], axes[2])
    save(fig, "l06-spy-fit.png")
    REPORT["spy_fit"] = {
        label: dict(params={n: float(v) for n, v in zip(names, s.x)}, rms=float(np.sqrt(np.mean(s.fun**2))),
                    max_abs=float(np.max(np.abs(s.fun))), share_within_2=float(np.mean(np.abs(s.fun) < 2)),
                    feller_2kt_minus_xi2=float(2 * s.x[2] * s.x[1] - s.x[3] ** 2) if label == "kappa free"
                    else float(2 * 2.0 * s.x[1] - s.x[3] ** 2))
        for label, s in best.items()}
    REPORT["spy_fit_fine_residuals"] = dict(rms=float(np.sqrt(np.mean(res**2))), max_abs=float(np.max(np.abs(res))),
                                            by_maturity_rms=[float(np.sqrt(np.mean(r**2))) for r in res],
                                            atm_skew_fit=[float(v) for v in fit_skew],
                                            atm_curv_fit=[float(v) for v in sf.smile_statistics(iv_fit.ravel(), k_all, T_all)[3]],
                                            limit_skew=float(abs(pfit["rho"] * pfit["xi"] / (4 * np.sqrt(pfit["v0"])))),
                                            fit_skew_exponent=float(sf.skew_power_law(DAYS / 252, fit_skew)[0]),
                                            residuals=[[float(v) for v in r] for r in res])
    vega_spy = sf.bs_vega(k_all, T_all, iv_mkt.ravel())
    noise_pts = (sig / vega_spy).reshape(len(DAYS), len(DELTAS)) * 100
    REPORT["spy_noise_in_vol_points_at_spy_iv"] = dict(min=float(noise_pts.min()), max=float(noise_pts.max()),
                                                       atm_1w=float(noise_pts[0, 4]), d2_1w=float(noise_pts[0, 8]),
                                                       dm2_1w=float(noise_pts[0, 0]), atm_1y=float(noise_pts[6, 4]),
                                                       d2_1y=float(noise_pts[6, 8]))
    REPORT["spy_fit_iv_gap_pts"] = dict(d2_1w=float((iv_fit[0, 8] - iv_mkt[0, 8]) * 100),
                                        dm1_1w=float((iv_fit[0, 2] - iv_mkt[0, 2]) * 100),
                                        max_abs=float(np.nanmax(np.abs(iv_fit - iv_mkt)) * 100))
    # the variance-swap curve the smile fit implies, against SPY's replicated one
    wvar = np.array(spy["diagnostics"]["design_total_variance"])
    w_fit = pfit["theta"] * T + (pfit["v0"] - pfit["theta"]) * (-np.expm1(-pfit["kappa"] * T)) / pfit["kappa"]
    REPORT["spy_fit_varswap_vol"] = dict(model=[float(v) for v in np.sqrt(w_fit / T)], spy=[float(v) for v in np.sqrt(wvar / T)])


def main():
    spy = load_spy()
    fig_cir()
    fig_paths()
    fig_expected_variance(spy)
    fig_trap()
    fig_inversions()
    fig_ladder()
    for key in SWEEPS:
        fig_sweep(key)
    fig_skew(spy)
    fig_spy_fit(spy)
    print(json.dumps(REPORT, indent=1, default=float))


if __name__ == "__main__":
    main()
