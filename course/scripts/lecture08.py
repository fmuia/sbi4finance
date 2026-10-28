"""Figures of Lecture 8: calibration on a derivatives desk.

Usage: python course/scripts/lecture08.py   (writes course/images/l08-*.png and prints the numbers quoted in the text)
"""

import json
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import ncx2, norm

from common import COLORS, GREY, ROOT, call, plt, save, sf, tidy
from sbi4finance import hkde
from sbi4finance.calibration import pricing, products, tasks

REPORT = {}
RESULTS = ROOT / "tutorial" / "assets" / "results" / "calibration"
KAPPA = 2.0
NOJUMP = dict(jump_intensity=0.0, jump_up_probability=0.5, jump_up_rate=10.0, jump_down_rate=10.0)


def heston_params(theta, xi, rho, kappa=KAPPA):
    return dict(NOJUMP, kappa=kappa, theta=theta, vol_of_vol=xi, rho=rho)


def heston_call(k, T, v0, theta, xi, rho, kappa=KAPPA, nodes=4096, upper=3000.0):
    return float(hkde.european_price(1.0, np.exp(k), 1.0, T, heston_params(theta, xi, rho, kappa), v0, kind="call",
                                     nodes=nodes, upper_limit=upper))


def cir_density(v, v0, theta, xi, T, kappa=KAPPA):
    """Density of the Heston variance at T given v0 (a scaled non-central chi-square)."""
    c = 2 * kappa / (xi**2 * (1 - np.exp(-kappa * T)))
    df = 4 * kappa * theta / xi**2
    nc = 2 * c * v0 * np.exp(-kappa * T)
    return 2 * c * ncx2.pdf(2 * c * v, df, nc)


def heston_forward_start(v0, theta, xi, rho, T1, T2, kappa=KAPPA, n=80):
    """E[(S_T2/S_T1 - 1)^+] = E[c_H(0, T2 - T1; V_T1)]: Heston is Markov in (S, V) and the payoff is scale-free."""
    c = 2 * kappa / (xi**2 * (1 - np.exp(-kappa * T1)))
    df = 4 * kappa * theta / xi**2
    nc = 2 * c * v0 * np.exp(-kappa * T1)
    # Gauss-Legendre on the quantile scale of V_T1
    u, wts = np.polynomial.legendre.leggauss(n)
    p = 0.5 * (u + 1) * (1 - 2e-6) + 1e-6
    v = ncx2.ppf(p, df, nc) / (2 * c)
    values = np.array([heston_call(0.0, T2 - T1, max(x, 1e-10), theta, xi, rho, kappa) for x in v])
    return float(np.sum(0.5 * wts * (1 - 2e-6) * values))


# -------------------------------------------------------------------------------------------------
def fig_swaps():
    v0 = theta = 0.04
    T = 63 / 252
    n_steps = 4 * 63
    xis = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    kvol_mc, kvol_se, var_q, kvol_bl = [], [], [], []
    samples = {}
    for xi in xis:
        paths = products.heston_paths(dict(kappa=KAPPA, theta=theta, vol_of_vol=xi), v0, T, n_steps, 32768, seed=4)
        Q = paths["I"][:, -1] / T
        m, se = products.volatility_swap(paths, T)
        kvol_mc.append(m)
        kvol_se.append(se)
        var_q.append(Q.var())
        kvol_bl.append(np.sqrt(theta) - Q.var() / (8 * theta**1.5))
        if xi in (0.3, 0.9):
            samples[xi] = Q
    kvar = theta  # flat forward variance: E[Q_T] = theta exactly
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    bins = np.linspace(0, 0.12, 120)
    for color, xi in zip((COLORS[0], COLORS[3]), (0.3, 0.9)):
        Q = samples[xi]
        ax.hist(Q, bins=bins, density=True, alpha=0.45, color=color,
                label=rf"$\sigma_v$ = {xi}: sd of $Q_T$ = {Q.std():.4f}")
    ax.axvline(kvar, color="0.2", ls="--", lw=1, label=r"$K_{var} = \mathbb{E}[Q_T]$ = 0.04")
    ax.set_xlabel(r"realized variance over 63 days, $Q_T$")
    ax.set_ylabel("density")
    ax.set_title("Same variance-swap strike, different spread")
    ax.legend(frameon=False, fontsize=8.5)
    ax = axes[1]
    ax.axhline(np.sqrt(kvar) * 100, color="0.2", ls="--", lw=1, label=r"$\sqrt{K_{var}}$ = 20%")
    ax.errorbar(xis, np.array(kvol_mc) * 100, yerr=1.96 * np.array(kvol_se) * 100, fmt="o-", color=COLORS[0], ms=4,
                label=r"$K_{vol} = \mathbb{E}[\sqrt{Q_T}]$, Monte Carlo")
    ax.plot(xis, np.array(kvol_bl) * 100, "s--", color=COLORS[1], ms=4,
            label=r"$\sqrt{K_{var}} - \mathrm{Var}(Q_T)/(8K_{var}^{3/2})$")
    ax.set_xlabel(r"Heston volatility of variance $\sigma_v$")
    ax.set_ylabel("% per year")
    ax.set_title("The volatility swap pays for the spread")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    tidy(*axes)
    save(fig, "l08-variance-and-volatility-swaps.png")
    REPORT["swaps"] = dict(xi=xis.tolist(), kvol=[float(v) for v in kvol_mc], kvol_bl=[float(v) for v in kvol_bl],
                           var_q=[float(v) for v in var_q], kvol_se_max=float(np.max(kvol_se)),
                           sd_q={str(k): float(v.std()) for k, v in samples.items()})


def fig_forward_start_deterministic():
    T1, T2 = 0.25, 0.5
    tau = T2 - T1
    s_flat = 0.20
    s_early = 0.25
    s_late = np.sqrt((s_flat**2 * T2 - s_early**2 * T1) / tau)
    cases = {"A: flat 20%": (s_flat, s_flat), f"B: 25% then {s_late:.1%}": (s_early, s_late)}
    out = {}
    for name, (a, b) in cases.items():
        w1, w2 = a**2 * T1, a**2 * T1 + b**2 * tau
        out[name] = dict(vanilla_T1=float(call(0.0, w1)), vanilla_T2=float(call(0.0, w2)),
                         forward_start=float(call(0.0, b**2 * tau)), varswap_T2=float(w2 / T2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for color, (name, (a, b)) in zip((COLORS[0], COLORS[3]), cases.items()):
        ax.step([0, T1, T2], np.array([a, b, b]) * 100, where="post", color=color, lw=2, label=name)
    ax.set_xlabel("time $t$, years")
    ax.set_ylabel(r"deterministic volatility $\sigma(t)$, %")
    ax.set_ylim(10, 28)
    ax.set_title("Two volatility term structures")
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    labels = ["ATM call,\n6 months", "variance swap,\n6 months", "ATM call,\n3 months", "forward-start\n3 → 6 months"]
    keys = ["vanilla_T2", "varswap_T2", "vanilla_T1", "forward_start"]
    xs = np.arange(len(keys))
    for i, (color, name) in enumerate(zip((COLORS[0], COLORS[3]), cases)):
        vals = [out[name][k] * (1 if k != "varswap_T2" else 1) for k in keys]
        ax.bar(xs + (i - 0.5) * 0.36, vals, 0.36, color=color, label=name)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("price in forward units (variance swap: strike)")
    ax.set_title("Equal on the 6-month quotes, different on the rest")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l08-forward-start-deterministic.png")
    REPORT["fs_deterministic"] = dict(s_late=float(s_late), cases=out)


def fig_forward_start_stochastic():
    v0 = theta = 0.04
    rho = -0.7
    T1, T2 = 0.25, 0.5
    tau = T2 - T1
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    v = np.linspace(1e-4, 0.14, 400)
    twin = ax.twinx()
    for color, xi in zip((COLORS[0], COLORS[3]), (0.3, 0.9)):
        ax.plot(v, cir_density(v, v0, theta, xi, T1), color=color, label=rf"density of $V_{{T_1}}$, $\sigma_v$ = {xi}")
        grid = np.linspace(0.002, 0.14, 25)
        twin.plot(grid, [heston_call(0.0, tau, g, theta, xi, rho) * 100 for g in grid], color=color, ls="--", lw=1)
    twin.set_ylabel(r"conditional ATM price over $[T_1, T_2]$, % (dashed)")
    twin.spines[["top"]].set_visible(False)
    ax.axvline(theta, color="0.6", ls=":", lw=1)
    ax.set_xlabel(r"variance at the strike-setting date, $V_{T_1}$")
    ax.set_ylabel("density (solid)")
    ax.set_title("The forward-start averages a price over the law of $V_{T_1}$")
    ax.set_ylim(0, 60)
    ax.text(0.004, 56, "spike at 0 cut", fontsize=8, color=COLORS[3])
    ax.legend(frameon=False, fontsize=8.5, loc="center right")
    xis = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    exact, naive, atm1, atm2 = [], [], [], []
    for xi in xis:
        fs = heston_forward_start(v0, theta, xi, rho, T1, T2)
        c1 = heston_call(0.0, T1, v0, theta, xi, rho)
        c2 = heston_call(0.0, T2, v0, theta, xi, rho)
        s1 = float(sf.implied_vol(c1, 0.0, T1, "call"))
        s2 = float(sf.implied_vol(c2, 0.0, T2, "call"))
        s_fwd = np.sqrt((s2**2 * T2 - s1**2 * T1) / tau)
        exact.append(fs)
        naive.append(float(call(0.0, s_fwd**2 * tau)))
        atm1.append(s1)
        atm2.append(s2)
    exact, naive = np.array(exact), np.array(naive)
    ax = axes[1]
    ax.plot(xis, (exact / naive - 1) * 100, "o-", color=COLORS[3], ms=4)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel(r"Heston volatility of variance $\sigma_v$ ($\rho = -0.7$)")
    ax.set_ylabel("Heston forward-start price over the\nBlack–Scholes price at the forward ATM vol, −1, %")
    ax.set_title("Today's ATM vols do not price the forward-start")
    tidy(axes[0], axes[1])
    save(fig, "l08-forward-start-stochastic.png")
    # Monte Carlo check of the quadrature at one point
    paths = products.heston_paths(dict(kappa=KAPPA, theta=theta, vol_of_vol=0.9), v0, T2, 504, 16384, seed=9)
    mc, mc_se = products.forward_start_call(paths, rho, T1, T2)
    REPORT["fs_stochastic"] = dict(xi=xis.tolist(), exact=[float(v) for v in exact], naive=[float(v) for v in naive],
                                   atm_T1=atm1, atm_T2=atm2, gap_pct=[float(v) for v in (exact / naive - 1) * 100],
                                   mc_check_xi_0_9=[float(mc), float(mc_se), float(exact[8])],
                                   bs_flat_20=float(call(0.0, 0.04 * tau)),
                                   conditional_at_v_0_04={str(xi): float(heston_call(0.0, tau, 0.04, theta, xi, rho))
                                                          for xi in (0.3, 0.9)},
                                   rule_0_4_sqrt_v_tau=float(0.4 * np.sqrt(0.04 * tau)))


def fig_cliquet():
    tau = 1 / 12
    sig = np.linspace(0.02, 0.8, 200)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    out = {}
    for color, (F, C) in zip(COLORS, ((0.0, 0.02), (0.0, 0.05), (-0.01, 0.03))):
        w = sig**2 * tau
        p = F + call(np.log(1 + F), w) - call(np.log(1 + C), w)
        ax.plot(sig * 100, p * 100, color=color, label=f"floor {F:+.0%}, cap {C:+.0%}")
        out[f"{F},{C}"] = dict(peak_sigma=float(sig[np.argmax(p)]), peak=float(p.max()),
                               at_10=float(np.interp(0.1, sig, p)), at_20=float(np.interp(0.2, sig, p)),
                               at_30=float(np.interp(0.3, sig, p)))
    ax.set_xlabel(r"Black–Scholes volatility $\sigma$, %")
    ax.set_ylabel("price of one monthly period, % of notional")
    ax.set_title("A capped period: the price levels off, then falls")
    ax.legend(frameon=False, fontsize=8.5)
    ax = axes[1]
    R = np.linspace(-0.08, 0.08, 400)
    ax.plot(R * 100, np.clip(R, 0.0, 0.02) * 100, color=COLORS[0], label="floor 0%, cap 2%")
    ax.plot(R * 100, np.maximum(R, 0) * 100, color="0.6", ls=":", label="uncapped forward-start")
    ax.set_xlabel(r"period return $S_{t_i}/S_{t_{i-1}} - 1$, %")
    ax.set_ylabel("period payoff, % of notional")
    ax.set_title("Payoff of one cliquet period")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l08-cliquet.png")
    REPORT["cliquet"] = out


def fig_profile():
    """Least squares of Heston (kappa = 2) to one noisy synthetic smile, profiled over the vol of vol."""
    task = tasks.heston_task()
    C = task["context"]["contracts"]
    s = tasks.sigma(task)
    F = task["fixed_parameters"]

    def surf(v0, th, xi, rho):
        params = dict(F, theta=th, vol_of_vol=xi, rho=rho)
        try:
            return pricing.heston_surface(params, v0, C, nodes=1024, upper_limit=600)
        except ValueError:
            return tasks.heston_prices(task, params, v0)[0]

    truth = (0.04, 0.05, 0.4, -0.7)
    y = surf(*truth) + s * np.random.default_rng(3).standard_normal(63)
    full = least_squares(lambda x: (surf(*x) - y) / s, list(truth),
                         bounds=([0.01, 0.01, 0.1, -0.95], [0.16, 0.16, 1.5, -0.1]))
    chi_best = 2 * full.cost
    xis = np.round(np.arange(0.30, 0.551, 0.025), 3)
    rows = []
    T = 63 / 252
    for xi in xis:
        r = least_squares(lambda x: (surf(x[0], x[1], xi, x[2]) - y) / s, [full.x[0], full.x[1], full.x[3]],
                          bounds=([0.01, 0.01, -0.95], [0.16, 0.16, -0.1]))
        v0, th, rho = r.x
        kvar = th + (v0 - th) * (1 - np.exp(-KAPPA * T)) / (KAPPA * T)
        fs = heston_forward_start(v0, th, xi, rho, 63 / 252, 126 / 252)
        put2y = heston_call(np.log(0.9), 2.0, v0, th, xi, rho) - 1 + 0.9
        paths = products.heston_paths(dict(kappa=KAPPA, theta=th, vol_of_vol=xi), v0, T, 252, 32768, seed=12)
        kvol = products.volatility_swap(paths, T)[0]
        rows.append(dict(xi=float(xi), v0=float(v0), theta=float(th), rho=float(rho), chi2=float(2 * r.cost),
                         varswap=float(kvar), volswap=float(kvol), fstart=float(fs), put2y=float(put2y)))
    best = min(rows, key=lambda r_: r_["chi2"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    d = np.array([r_["chi2"] for r_ in rows]) - chi_best
    ax.plot(xis, d, "o-", color=COLORS[0], ms=4)
    ax.axhline(2.71, color="0.4", ls="--", lw=1)
    ax.text(0.30, 3.2, r"$\Delta\chi^2 = 2.71$", fontsize=8.5, color="0.35")
    ax.axvline(truth[2], color="0.6", ls=":", lw=1)
    ax.set_xlabel(r"$\sigma_v$, held fixed while $v_0$, $\theta_H$, $\rho$ are refitted")
    ax.set_ylabel(r"$\chi^2 - \chi^2_{\min}$ over 63 quotes")
    ax.set_title("Heston, one noisy synthetic smile: the profile of the fit")
    ax.set_ylim(0, 25)
    ax = axes[1]
    names = {"varswap": "variance swap 63 d", "volswap": "volatility swap 63 d", "fstart": "forward-start 63 → 126 d",
             "put2y": "2-year 90% put"}
    for color, (k, name) in zip(COLORS, names.items()):
        vals = np.array([r_[k] for r_ in rows])
        ax.plot(xis, (vals / best[k] - 1) * 100, "o-", color=color, ms=3.5, label=name)
    from scipy.interpolate import CubicSpline
    fine = np.linspace(xis[0], xis[-1], 4001)
    inside_f = fine[CubicSpline(xis, d)(fine) <= 2.71]
    lo_x, hi_x = float(inside_f.min()), float(inside_f.max())
    ax.axvspan(lo_x, hi_x, color="0.9", zorder=0, label=r"$\Delta\chi^2 \leq 2.71$ (interpolated)")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel(r"$\sigma_v$")
    ax.set_ylabel("price relative to the best fit, %")
    ax.set_title("Prices along the profile")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l08-heston-profile.png")
    band_range = {}
    for k in names:
        rel = CubicSpline(xis, np.array([r_[k] for r_ in rows]) / best[k] - 1)(inside_f) * 100
        band_range[k] = [float(rel.min()), float(rel.max())]
    rho_f = CubicSpline(xis, np.array([r_["rho"] for r_ in rows]))([lo_x, hi_x])
    REPORT["profile"] = dict(truth=truth, best_full=[float(v) for v in full.x], chi2_best=float(chi_best), rows=rows,
                             band_xi_interpolated=[lo_x, hi_x], band_rho_at_edges=[float(v) for v in rho_f],
                             band_range_pct_interpolated=band_range)


def fig_cached_products():
    d = json.loads((RESULTS / "heston" / "s7" / "products.json").read_text())
    names = {"variance_swap": "variance swap 63 d", "volatility_swap": "volatility swap 63 d",
             "forward_start_call": "forward-start 63 → 126 d", "put_out_of_grid": "2-year 90% put"}
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    y = 0
    ticks, labels = [], []
    out = {}
    for k, name in names.items():
        for j, srf in enumerate(d["surfaces"]):
            p = srf["products"][k]
            ref = p["point"]
            ax.plot([(p["q05"] / ref - 1) * 100, (p["q95"] / ref - 1) * 100], [y, y], color=COLORS[j], lw=5, alpha=0.5)
            ax.plot(0, y, "|", color="k", ms=12)
            ax.plot((p["point"] + p["reserve_short"]) / ref * 100 - 100, y, "v", color=COLORS[3], ms=7)
            ticks.append(y)
            labels.append(f"{name}, smile {srf['test_row']}")
            out[f"{k}/{srf['test_row']}"] = dict(range_rel=p["range_90_relative"], reserve_rel=p["reserve_short_relative"])
            y -= 1
        y -= 0.6
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlabel("price relative to the least-squares price, %")
    ax.plot([], [], color=COLORS[0], lw=5, alpha=0.5, label="5% to 95% of the price under the network posterior")
    ax.plot([], [], "|", color="k", ms=12, label="least-squares (point) price")
    ax.plot([], [], "v", color=COLORS[3], label="90% quantile: point + short-side reserve")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=3)
    ax.set_title("Heston, three test smiles (cached S7, network posterior): range, point and reserve")
    tidy(ax)
    save(fig, "l08-reserve.png")
    REPORT["cached_products"] = out


def fig_across_models():
    d = json.loads((RESULTS / "rough-pilot" / "s7-across" / "across-models.json").read_text())
    names = {"variance_swap": "variance swap 63 d", "volatility_swap": "volatility swap 63 d",
             "forward_start_call": "forward-start 63 → 126 d", "put_out_of_grid": "2-year 90% put"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    out = {}
    for ax, row in zip(axes, (0, 2)):
        srf = d["surfaces"][row]
        for i, (k, name) in enumerate(names.items()):
            ref = srf["products"][k]["rough"]["point"]
            for dy, model, color in ((0.15, "rough", COLORS[0]), (-0.15, "heston", COLORS[4])):
                p = srf["products"][k][model]
                lo, hi = (p["q05"] / ref - 1) * 100, (p["q95"] / ref - 1) * 100
                ax.plot([lo, hi], [-i + dy] * 2, color=color, lw=6, alpha=0.6, solid_capstyle="butt")
                ax.plot((p["point"] / ref - 1) * 100, -i + dy, "o", color=color, ms=5)
            P = srf["products"][k]
            out[f"{srf['test_row']}/{k}"] = dict(
                gap=P["gap_between_points_relative"],
                range_rough=(P["rough"]["q95"] - P["rough"]["q05"]) / P["rough"]["mean"],
                range_heston=(P["heston"]["q95"] - P["heston"]["q05"]) / P["heston"]["mean"],
                reserve_rough=P["rough"]["reserve_short_relative"], reserve_heston=P["heston"]["reserve_short_relative"],
                overlap=bool(not (P["rough"]["q95"] < P["heston"]["q05"] or P["heston"]["q95"] < P["rough"]["q05"])))
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set_yticks([0, -1, -2, -3])
        ax.set_yticklabels(list(names.values()), fontsize=8.5)
        ax.set_xlabel("price relative to the rough-Bergomi least-squares price, %")
        ax.set_title(f"Rough-pilot smile {srf['test_row']}: χ² {srf['rough_least_squares']['chi2']:.1f} (rough), "
                     f"{srf['heston_least_squares']['chi2']:.1f} (Heston)", fontsize=10)
        tidy(ax)
    axes[0].plot([], [], color=COLORS[0], lw=6, alpha=0.6, label="rough Bergomi pilot-network posterior, 5% to 95%")
    axes[0].plot([], [], color=COLORS[4], lw=6, alpha=0.6, label="Heston network posterior (reweighted), 5% to 95%")
    axes[0].plot([], [], "o", color="0.4", label="least-squares point")
    fig.legend(frameon=False, fontsize=8.5, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(ROOT / "course" / "images" / "l08-across-models.png", dpi=160)
    plt.close(fig)
    REPORT["across"] = out


def leverage_bias():
    """Daily monitoring under rough Bergomi: E[sum r_i^2] exceeds the integral of xi0 by a leverage term.

    First order in eta, per day of length D, flat curve xi:
    -rho eta sqrt(2H) sqrt(xi) D^(H+1/2) / ((H+1/2)(H+3/2)) + xi D / 4, relative to xi D.
    The Monte Carlo check simulates 63 days with 16 steps per day and averages the exact conditional
    expectation of the daily squared log return given the variance path.
    """
    from sbi4finance.rough_bergomi import simulate_rough
    D, xi, days, per_day, n = 1 / 252, 0.04, 63, 16, 8192
    out = {}
    for label, (H, eta, rho) in {"prior centre": (0.185, 2.0, -0.595),
                                 "pilot smile 293, best fit": (0.0684, 1.3235, -0.6632),
                                 "steep skew": (0.07, 1.9, -0.9)}.items():
        first = (-rho * eta * np.sqrt(2 * H) * np.sqrt(xi) * D ** (H + 0.5) / ((H + 0.5) * (H + 1.5))
                 + 0.25 * xi * D)
        z = np.random.default_rng(0).standard_normal((n, days * per_day, 3))
        sim = simulate_rough({"hurst": H, "eta": eta, "rho": rho}, {"times": [0.0, 1.0], "values": [xi, xi]},
                             D / per_day, z)
        left = sim["variance"][:, :-1]
        dw = np.sqrt(left * D / per_day) * z[..., 0]
        A = (left * D / per_day).reshape(n, days, per_day).sum(-1)
        X = dw.reshape(n, days, per_day).sum(-1)
        est = (0.25 * A**2 - rho * X * A).sum(1) / (xi * D * days)
        out[label] = dict(parameters=[H, eta, rho], first_order_pct=float(first * 100),
                          monte_carlo_pct=float(est.mean() * 100), monte_carlo_se_pct=float(est.std() / np.sqrt(n) * 100))
    REPORT["variance_swap_daily_monitoring"] = out


def main():
    t0 = time.time()
    fig_swaps()
    fig_forward_start_deterministic()
    fig_forward_start_stochastic()
    fig_cliquet()
    fig_profile()
    fig_cached_products()
    fig_across_models()
    leverage_bias()
    REPORT["seconds"] = time.time() - t0
    print(json.dumps(REPORT, indent=1, default=float))


if __name__ == "__main__":
    main()
