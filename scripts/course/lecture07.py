"""Figures of Lecture 7: rough volatility, the rough Bergomi model.

Usage: python scripts/course/lecture07.py   (writes docs/course/images/l07-*.png and prints the numbers quoted in the text)

Monte Carlo runs are kept small (2^13 to 2^15 paths); every price is reported with its standard error.
"""

import json
import time

import numpy as np
from scipy.special import polygamma
from scipy.stats import norm

from common import COLORS, DAYS, DELTAS, GREY, load_spy, load_voo, plt, save, sf, tidy
from sbi4finance.bergomi import conditional_put_payoffs
from sbi4finance.calibration import pricing, tasks
from sbi4finance.rough_bergomi import hybrid_weights, rough_driver, simulate_rough

REPORT = {}
MEDIAN = dict(hurst=0.185, eta=2.0, rho=-0.595)  # centre of the tutorial's prior
XI = 0.25**2                                       # flat forward variance, 25% a year
TASK = tasks.rough_task()
LEVELS = np.full(7, XI)
SIGMA = np.array(tasks.sigma(TASK))
K_ALL, T_ALL, KIND = sf.contract_arrays(TASK["context"]["contracts"])


def flat_curve(horizon):
    return pricing.step_curve([0.0, horizon], [XI])


def surface(params, seed=7, levels=LEVELS):
    """63 prices, standard errors and CPU seconds for one parameter set (common random numbers via the seed)."""
    t0 = time.process_time()
    prices, errors, _, _, used = tasks.price(TASK, [params["hurst"], params["eta"], params["rho"]], levels, seed)
    return prices, errors, time.process_time() - t0, used


def implied_grid(prices):
    return sf.implied_vol(prices, K_ALL, T_ALL, KIND).reshape(len(DAYS), len(DELTAS))


def iv_se(prices, errors):
    """Implied-volatility standard error, price error divided by vega at the implied volatility."""
    iv = sf.implied_vol(prices, K_ALL, T_ALL, KIND)
    return (errors / sf.bs_vega(K_ALL, T_ALL, iv)).reshape(len(DAYS), len(DELTAS))


# =========================================================================== 1. evidence from data
def fig_roughness_data():
    _, r = load_voo()
    r = r.values
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    lags = np.arange(1, 13)
    n = 5
    m = len(r) // n
    lrv = np.log((r[: m * n] ** 2).reshape(m, n).sum(axis=1) * 252 / n)
    zetas = []
    for color, q in zip(COLORS, (0.5, 1.0, 1.5, 2.0)):
        mq = np.array([np.mean(np.abs(lrv[l:] - lrv[:-l]) ** q) for l in lags])
        b, a = np.polyfit(np.log(lags), np.log(mq), 1)
        zetas.append(b)
        axes[0].loglog(lags, mq, "o", color=color, ms=4, label=rf"$q$ = {q}: slope {b:.3f}")
        axes[0].loglog(lags, np.exp(a) * lags**b, "-", color=color, lw=1)
    axes[0].set_xticks([1, 2, 4, 8, 12])
    axes[0].set_xticklabels(["1", "2", "4", "8", "12"])
    axes[0].set_xlabel(r"lag $\Delta$, weeks")
    axes[0].set_ylabel(r"$m(q, \Delta) = \mathrm{mean}\,|\log RV_{t+\Delta} - \log RV_t|^q$")
    axes[0].set_title("VOO weekly realized variance, 2020–2026")
    axes[0].legend(frameon=False, fontsize=8.5)
    qs = np.array([0.5, 1.0, 1.5, 2.0])
    H = float(np.polyfit(qs, zetas, 1)[0])
    axes[1].plot(qs, zetas, "o", color=COLORS[0], ms=6, label="slopes from the left panel")
    axes[1].plot(qs, H * qs, "-", color=COLORS[0], lw=1, label=rf"$\zeta_q = qH$, $H$ = {H:.3f}")
    axes[1].plot(qs, 0.5 * qs, "--", color=GREY, lw=1, label=r"Brownian log-variance, $H = 0.5$")
    # measurement-noise correction for q = 2: weekly RV from 5 normal returns has Var(log RV) = trigamma(5/2)
    m2 = np.array([np.mean((lrv[l:] - lrv[:-l]) ** 2) for l in lags])
    noise = 2 * polygamma(1, n / 2)
    corrected = m2 - noise
    ok = corrected > 0
    Hc = float(np.polyfit(np.log(lags[ok]), np.log(corrected[ok]), 1)[0] / 2)
    axes[1].plot([2.0], [2 * Hc], "s", color=COLORS[3], ms=7, label=rf"$q$ = 2 after removing sampling noise: $H$ = {Hc:.2f}")
    axes[1].set_xlabel("$q$")
    axes[1].set_ylabel(r"scaling exponent $\zeta_q$")
    axes[1].set_title("Monofractal scaling with a small exponent")
    axes[1].legend(frameon=False, fontsize=8)
    tidy(*axes)
    save(fig, "l07-roughness-from-data.png")
    out["weekly"] = dict(weeks=int(m), zetas=[float(z) for z in zetas], H=H, noise_m2=float(noise),
                         m2_lag1=float(m2[0]), H_noise_corrected=Hc)
    # sensitivity to the block length
    n2 = 10
    m10 = len(r) // n2
    l10 = np.log((r[: m10 * n2] ** 2).reshape(m10, n2).sum(axis=1))
    z10 = [np.polyfit(np.log(lags), np.log([np.mean(np.abs(l10[l:] - l10[:-l]) ** q) for l in lags]), 1)[0] for q in qs]
    out["biweekly_H"] = float(np.polyfit(qs, z10, 1)[0])
    m2b = np.array([np.mean((l10[l:] - l10[:-l]) ** 2) for l in lags]) - 2 * polygamma(1, n2 / 2)
    okb = m2b > 0
    out["biweekly_H_noise_corrected"] = float(np.polyfit(np.log(lags[okb]), np.log(m2b[okb]), 1)[0] / 2)
    REPORT["roughness_data"] = out


# =========================================================================== 2. the Volterra driver
def fig_driver():
    dt = 1 / (252 * 8)
    n = 252 * 8
    rng = np.random.default_rng(1)
    z = rng.standard_normal((1, n, 3))
    t = dt * np.arange(1, n + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for color, (H, off) in zip(COLORS, ((0.1, 3.0), (0.185, 1.0), (0.3, -1.0), (0.45, -3.0))):
        y, _ = rough_driver(H, dt, z)
        axes[0].plot(t, y[0] + off, color=color, lw=0.7, label=f"$H$ = {H}")
    axes[0].set_yticks([])
    axes[0].set_xlabel("$t$, years")
    axes[0].set_title(r"$Y_t$ for four $H$, same Brownian increments")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right", ncol=2)
    zs = rng.standard_normal((4000, n, 3))
    var_check, incr = {}, {}
    for color, H in zip(COLORS, (0.1, 0.185, 0.3)):
        y, dvar = rough_driver(H, dt, zs)
        sel = [8, 40, 200, 1000, n - 1]
        axes[1].loglog(t[sel], y[:, sel].var(axis=0), "o", color=color, ms=5)
        axes[1].loglog(t, t ** (2 * H), color=color, lw=1, label=rf"$t^{{2H}}$, $H$ = {H}")
        var_check[H] = [float(v) for v in y[:, sel].var(axis=0) / t[sel] ** (2 * H)]
        hs = np.array([1, 2, 4, 8, 16, 32, 64, 128])
        i0 = n // 2
        inc = np.array([np.mean((y[:, i0 + h] - y[:, i0]) ** 2) for h in hs])
        b = np.polyfit(np.log(hs * dt), np.log(inc), 1)[0]
        axes[2].loglog(hs * dt * 252, inc, "o-", color=color, ms=4, label=rf"$H$ = {H}: slope {b:.2f}")
        incr[H] = float(b)
    axes[1].set_xlabel("$t$, years")
    axes[1].set_ylabel(r"Var$(Y_t)$")
    axes[1].set_title(r"Variance $t^{2H}$ (dots: 4,000 simulated paths)")
    axes[1].legend(frameon=False, fontsize=8.5)
    axes[2].set_xlabel(r"increment length $h$, trading days")
    axes[2].set_ylabel(r"$\mathbb{E}[(Y_{t+h} - Y_t)^2]$ at $t$ = 0.5")
    axes[2].set_title(r"Increments scale as $h^{2H}$: roughness")
    axes[2].legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l07-volterra-driver.png")
    REPORT["driver"] = dict(variance_ratio_to_t2H=var_check, increment_slopes=incr)


# =========================================================================== 3. variance and its law
def fig_variance_law():
    dt = 1 / (252 * 4)
    n = 252 * 4
    rng = np.random.default_rng(2)
    z = rng.standard_normal((20000, n, 3))
    paths = simulate_rough(MEDIAN, flat_curve(1.0), dt, z)
    V = paths["variance"]
    t = dt * np.arange(n + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for color, j in zip(COLORS, range(3)):
        axes[0].plot(t * 252, np.sqrt(V[j]) * 100, color=color, lw=0.7)
    axes[0].axhline(25, color="0.3", ls="--", lw=1, label=r"$\sqrt{\xi_0}$ = 25%")
    axes[0].set_xlabel("t, trading days")
    axes[0].set_ylabel(r"$\sqrt{V_t}$, %")
    axes[0].set_title(r"Three variance paths at the prior centre ($H$ = 0.185, $\eta$ = 2)")
    axes[0].legend(frameon=False, fontsize=8.5)
    q = np.quantile(V / XI, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    ax = axes[1]
    ax.fill_between(t[1:] * 252, q[0, 1:], q[4, 1:], color=COLORS[0], alpha=0.15, label="5%–95%")
    ax.fill_between(t[1:] * 252, q[1, 1:], q[3, 1:], color=COLORS[0], alpha=0.3, label="25%–75%")
    ax.plot(t[1:] * 252, q[2, 1:], color=COLORS[0], label="median (simulated)")
    ax.plot(t[1:] * 252, np.exp(-0.5 * MEDIAN["eta"] ** 2 * t[1:] ** (2 * MEDIAN["hurst"])), "--", color=COLORS[3],
            lw=1, label=r"median $e^{-\eta^2 t^{2H}/2}$")
    ax.plot(t[1:] * 252, (V / XI).mean(axis=0)[1:], color="0.2", lw=1.5, label="mean (simulated) = 1")
    ax.set_yscale("log")
    ax.set_xlabel("t, trading days")
    ax.set_ylabel(r"$V_t / \xi_0(t)$")
    ax.set_title("The law of $V_t$: mean fixed, median falling")
    ax.legend(frameon=False, fontsize=8)
    tidy(*axes)
    save(fig, "l07-variance-law.png")
    REPORT["variance_law"] = dict(mean_ratio_end=float((V / XI).mean(axis=0)[-1]),
                                  mean_ratio_se=float((V[:, -1] / XI).std() / np.sqrt(len(V))),
                                  median_1y=float(q[2, -1]), median_theory_1y=float(np.exp(-2.0)),
                                  q95_1y=float(q[4, -1]), q05_1y=float(q[0, -1]),
                                  median_1m=float(q[2, 21 * 4]), theory_1m=float(np.exp(-0.5 * 4 * (21 / 252) ** 0.37)))


# =========================================================================== 4. non-Markovian: the past matters
def fig_memory():
    H, eta = MEDIAN["hurst"], MEDIAN["eta"]
    dt = 1 / 252
    n_past, n_future = 63, 189
    n = n_past + n_future
    w, bridge = hybrid_weights(H, dt, n)
    rng = np.random.default_rng(4)
    z = rng.standard_normal((20000, n_past))
    resid = rng.standard_normal((20000, n_past))    # the hybrid scheme's independent last-cell residuals
    # driver at step n_past: convolution of the past increments plus the last cell's residual
    y_now = z @ w[:n_past][::-1] + bridge * resid[:, -1]
    target = 0.0
    trend = z[:, -21:].sum(axis=1) - z[:, :21].sum(axis=1)  # recent minus early shocks
    close = np.abs(y_now - target) < 0.02
    a = np.flatnonzero(close)[np.argmax(trend[close])]
    b = np.flatnonzero(close)[np.argmin(trend[close])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    cond = {}
    for color, idx, name in ((COLORS[3], a, "recent shocks up"), (COLORS[0], b, "recent shocks down")):
        ypast = np.array([np.sum(w[:i][::-1] * z[idx, :i]) + bridge * resid[idx, i - 1] for i in range(1, n_past + 1)])
        # conditional mean of Y and conditional variance of V at future steps given the past increments
        m_fut, v_fut = [], []
        for h in range(1, n_future + 1):
            coeff = w[h:h + n_past][::-1]
            m_fut.append(np.sum(coeff * z[idx]))
            v_fut.append(np.sum(w[:h] ** 2) + bridge**2)
        m_fut, v_fut = np.array(m_fut), np.array(v_fut)
        tot = np.cumsum(w**2)[n_past:n] + bridge**2
        xi_t = XI * np.exp(eta * m_fut - 0.5 * eta**2 * tot + 0.5 * eta**2 * v_fut)
        days_past = np.arange(1, n_past + 1)
        days_fut = n_past + np.arange(1, n_future + 1)
        axes[0].plot(days_past, ypast, color=color, lw=1, label=f"past with {name}")
        axes[0].plot(days_fut, m_fut, color=color, ls="--", lw=1)
        axes[1].plot(days_fut, np.sqrt(xi_t) * 100, color=color, label=f"{name}")
        cond[name] = dict(y_now=float(y_now[idx]), xi_next_day=float(np.sqrt(xi_t[0])), xi_6m=float(np.sqrt(xi_t[-1])))
    axes[0].axvline(n_past, color="0.8")
    axes[0].set_xlabel("trading days")
    axes[0].set_ylabel("$Y_t$ (solid) and $\\mathbb{E}[Y_u \\mid$ past$]$ (dashed)")
    axes[0].set_title("Two pasts that end at the same $Y_t$")
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[1].axhline(25, color=GREY, ls=":", lw=1)
    axes[1].set_xlabel("future date $u$, trading days")
    axes[1].set_ylabel(r"$\sqrt{\mathbb{E}[V_u \mid \mathrm{past}]}$, %")
    axes[1].set_title("Same $V_t$, different forward-variance curves")
    axes[1].legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l07-memory.png")
    REPORT["memory"] = cond


# =========================================================================== 5. the hybrid scheme
def fig_hybrid():
    dt = 1 / (252 * 8)
    n = 504
    i = np.arange(1, n + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    out = {}
    for color, H in zip(COLORS, (0.1, 0.185, 0.3)):
        alpha = H - 0.5
        exact = (i * dt) ** (2 * H)
        left = np.cumsum(2 * H * (np.arange(1, n + 1) * dt) ** (2 * alpha) * dt)  # kernel at the far end of each cell
        w, bridge = hybrid_weights(H, dt, n)
        hybrid = np.cumsum(w**2) + bridge**2
        axes[0].plot(i, left / exact, ":", color=color, lw=1.5, label=rf"Riemann sum, $H$ = {H}")
        axes[0].plot(i, hybrid / exact, "-", color=color, lw=1.5, label=rf"hybrid scheme, $H$ = {H}")
        out[H] = dict(left_ratio_first=float(left[0] / exact[0]), left_ratio_end=float(left[-1] / exact[-1]),
                      hybrid_ratio_first=float(hybrid[0] / exact[0]), hybrid_ratio_end=float(hybrid[-1] / exact[-1]))
    axes[0].axhline(1, color="0.8")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("step $i$ (8 steps per trading day)")
    axes[0].set_ylabel(r"discrete Var$(Y_i)$ / $t_i^{2H}$")
    axes[0].set_title("Variance captured by each scheme")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    H = 0.185
    alpha = H - 0.5
    k = np.arange(1, 9)
    bstar = ((k ** (alpha + 1) - (k - 1) ** (alpha + 1)) / (alpha + 1)) ** (1 / alpha)
    s = np.linspace(0.02, 8, 800)
    axes[1].plot(s, s**alpha, color="0.3", label=r"kernel $(t_i - s)^{H - 1/2}$, in units of $\Delta$")
    axes[1].plot(k, k**alpha, "o", color=COLORS[3], ms=5, label="Riemann sum: kernel at $k\\Delta$")
    axes[1].plot(bstar, bstar**alpha, "s", color=COLORS[0], ms=5, label=r"hybrid: kernel at $b_k^*\Delta$")
    axes[1].set_ylim(0, 3)
    axes[1].set_xlabel(r"distance $(t_i - s)/\Delta$")
    axes[1].set_ylabel("kernel value")
    axes[1].set_title(r"Where the kernel is evaluated, $H$ = 0.185")
    axes[1].legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l07-hybrid-scheme.png")
    REPORT["hybrid"] = dict(ratios=out, bstar=[float(b) for b in bstar])


# =========================================================================== 6. variance reduction
def fig_estimators():
    dt = 1 / (252 * 8)
    n = 63 * 8
    T = n * dt
    rng = np.random.default_rng(5)
    m = 8192
    z = rng.standard_normal((m, n, 3))
    deltas = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    k = deltas * sf.SIGMA_REF * np.sqrt(T)
    contracts = [{"strike": float(np.exp(x)), "maturity": T, "option_type": "put"} for x in k]
    idx = np.full(len(k), n)
    res = {}
    estimates = {}
    for sign in (1.0, -1.0):
        paths = simulate_rough(MEDIAN, flat_curve(T), dt, sign * z)
        ST = np.exp(paths["log_spot"][:, -1])
        raw = np.maximum(np.exp(k)[None, :] - ST[:, None], 0.0)
        condl = conditional_put_payoffs(paths, MEDIAN, contracts, idx, spot=1.0, rate=0.0, dividend=0.0)
        estimates[sign] = (raw, condl)
    raw = estimates[1.0][0]
    condl = estimates[1.0][1]
    anti = 0.5 * (estimates[1.0][1] + estimates[-1.0][1])
    # controls with one block, as in the repository pricer
    diag = {}
    contracts_all = [{"strike": float(np.exp(x)), "maturity": T, "option_type": "put"} for x in k]
    p_cv, s_cv, _ = pricing.rough_block(MEDIAN, flat_curve(T), contracts_all, horizon=T, n_steps=n, n_pairs=m, seed=5,
                                         diagnostics=diag)
    var_raw = raw.var(axis=0, ddof=1)
    var_cond = condl.var(axis=0, ddof=1)
    var_anti = anti.var(axis=0, ddof=1) * 2      # per path, so that the comparison is at equal path counts
    var_cv = diag["controlled_variance"] * 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    j = 0
    bins = np.linspace(0, 0.03, 80)
    ax.hist(condl[:, j], bins=bins, color=COLORS[0], alpha=0.5, density=True, label="conditional Black–Scholes put")
    ax.hist(raw[:, j], bins=bins, color=COLORS[3], histtype="step", lw=1.5, density=True, label=r"raw payoff $(K - X_T)^+$")
    ax.set_yscale("log")
    ax.set_xlabel(r"per-path estimate, $\delta = -2$ put, 63 days")
    ax.set_ylabel("density (log scale)")
    ax.set_title("One number per path: payoff against conditional price")
    ax.legend(frameon=False, fontsize=8.5)
    ax = axes[1]
    labels = ["raw payoff", "conditional BS", "+ antithetic", "+ control variates"]
    for color, dd in zip(COLORS, range(len(deltas))):
        ratios = np.array([1.0, var_cond[dd] / var_raw[dd], var_anti[dd] / var_raw[dd], var_cv[dd] / var_raw[dd]])
        ax.semilogy(range(4), ratios, "o-", color=color, ms=4, label=rf"$\delta$ = {deltas[dd]:+.0f}")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("variance per path, relative to the raw payoff")
    ax.set_title("Each step of variance reduction, 63-day puts")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l07-variance-reduction.png")
    sig = np.array(sf.quote_noise(contracts_all))
    REPORT["variance_reduction"] = dict(
        deltas=deltas.tolist(), raw=[float(v) for v in var_raw],
        cond_over_raw=[float(a / b) for a, b in zip(var_cond, var_raw)],
        anti_over_raw=[float(a / b) for a, b in zip(var_anti, var_raw)],
        cv_over_raw=[float(a / b) for a, b in zip(var_cv, var_raw)],
        se_over_sigma_cv=[float(a / b) for a, b in zip(s_cv, sig)],
        se_over_sigma_raw=[float(np.sqrt(a / (2 * m)) / b) for a, b in zip(var_raw, sig)],
        prices_cv=[float(v) for v in p_cv], pairs=m)


# =========================================================================== 7. cost and error budget
def fig_budget():
    T = 10 / 252
    n = 80
    contracts = [{"strike": float(np.exp(x)), "maturity": T, "option_type": "put"} for x in DELTAS * 0.2 * np.sqrt(T)]
    sig = np.array(sf.quote_noise(contracts))
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    pairs = np.array([2**11, 2**12, 2**13, 2**14])
    ses, cpu = [], []
    for npairs in pairs:
        t0 = time.process_time()
        _, s, _ = pricing.rough_block(MEDIAN, flat_curve(T), contracts, horizon=T, n_steps=n, n_pairs=int(npairs), seed=9)
        cpu.append(time.process_time() - t0)
        ses.append(s / sig)
    ses = np.array(ses)
    for color, j in zip(COLORS, (0, 2, 4, 8)):
        axes[0].loglog(2 * pairs, ses[:, j], "o-", color=color, ms=4, label=rf"$\delta$ = {DELTAS[j]:+.0f}")
    axes[0].loglog(2 * pairs, ses[0, 0] * np.sqrt(pairs[0] / pairs), ":", color=GREY, label=r"$\propto N^{-1/2}$")
    axes[0].axhline(0.25, color=COLORS[3], ls="--", lw=1)
    axes[0].text(2 * pairs[0], 0.27, r"budget $s_c \leq 0.25\,\sigma_c$", color=COLORS[3], fontsize=8.5)
    axes[0].set_xlabel("paths $N$")
    axes[0].set_ylabel(r"Monte Carlo standard error / $\sigma_c$")
    axes[0].set_title("10-day block at the prior centre")
    axes[0].legend(frameon=False, fontsize=8.5)
    out["block10"] = dict(paths=(2 * pairs).tolist(), se_over_sigma=[[float(v) for v in r] for r in ses], cpu_s=[float(c) for c in cpu])
    ax = axes[1]
    prices, errors, sec, used = surface(MEDIAN, seed=0)
    r = (errors / SIGMA).reshape(len(DAYS), len(DELTAS))
    im = ax.imshow(r, cmap="viridis", aspect="auto", vmin=0, vmax=0.25)
    for jj in range(len(DAYS)):
        for ii in range(len(DELTAS)):
            ax.text(ii, jj, f"{r[jj, ii]:.2f}", ha="center", va="center", fontsize=7, color="white" if r[jj, ii] < 0.15 else "0.1")
    ax.set_xticks(range(len(DELTAS)))
    ax.set_xticklabels([f"{d:g}" for d in DELTAS])
    ax.set_yticks(range(len(DAYS)))
    ax.set_yticklabels([f"{d} d" for d in DAYS])
    ax.set_xlabel(r"standardized moneyness $\delta$")
    ax.set_title(rf"One surface, $2^{{15}}$ paths per block: $s_c/\sigma_c$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tidy(axes[0])
    save(fig, "l07-error-budget.png")
    out["surface"] = dict(cpu_s=float(sec), paths=[float(u) for u in used], max=float(r.max()), median=float(np.median(r)),
                          argmax=[int(v) for v in np.unravel_index(np.argmax(r), r.shape)])
    REPORT["budget"] = out
    return prices, errors


# =========================================================================== 8. parameters and the smile
def fig_sweeps(base):
    sets = {
        "hurst": (0.08, 0.185, 0.30),
        "eta": (1.2, 2.0, 2.8),
        "rho": (-0.9, -0.595, -0.3),
    }
    labels = {"hurst": "$H$", "eta": r"$\eta$", "rho": r"$\rho$"}
    cache = {}
    for name, values in sets.items():
        for v in values:
            p = dict(MEDIAN, **{name: v})
            key = (p["hurst"], p["eta"], p["rho"])
            if key not in cache:
                cache[key] = base if key == (0.185, 2.0, -0.595) else surface(p, seed=0)[:2]
    out = {}
    for name, values in sets.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        res = {}
        for color, v in zip(COLORS, values):
            p = dict(MEDIAN, **{name: v})
            prices, errors = cache[(p["hurst"], p["eta"], p["rho"])]
            iv = implied_grid(prices)
            se = iv_se(prices, errors)
            for ax, j in zip(axes[:2], (0, 6)):
                ax.errorbar(DELTAS, iv[j] * 100, yerr=2 * se[j] * 100, fmt="o-", color=color, ms=3.5, capsize=2,
                            label=f"{labels[name]} = {v}")
            _, lev, skew, curv = sf.smile_statistics(iv.ravel(), K_ALL, T_ALL)
            axes[2].loglog(DAYS, np.abs(skew), "o-", color=color, ms=4, label=f"{labels[name]} = {v}")
            expo = sf.skew_power_law(DAYS / 252, skew)[0]
            res[str(v)] = dict(atm=[float(x) for x in lev], skew=[float(x) for x in skew], curv=[float(x) for x in curv],
                               skew_exponent=float(expo), iv_1w=[float(x) for x in iv[0]], iv_1y=[float(x) for x in iv[6]],
                               max_iv_se_pts=float(np.nanmax(se) * 100))
        for ax, title in zip(axes[:2], ("1 week", "1 year")):
            ax.set_xlabel(r"standardized moneyness $\delta$")
            ax.set_ylabel("implied volatility, %")
            ax.set_title(f"Smile at {title} (bars: 2 Monte Carlo s.e.)")
        axes[0].legend(frameon=False, fontsize=8.5)
        axes[2].set_xticks(DAYS)
        axes[2].set_xticklabels([str(d) for d in DAYS])
        axes[2].set_xlabel("maturity, business days")
        axes[2].set_ylabel(r"$|\psi(T)|$, ATM skew")
        axes[2].set_title("At-the-money skew term structure")
        fig.suptitle(f"Rough Bergomi: varying {labels[name]}, others at the prior centre, flat 25% curve", y=1.0, fontsize=11)
        tidy(*axes)
        save(fig, f"l07-smile-{name}.png")
        out[name] = res
    REPORT["sweeps"] = out
    return cache


# =========================================================================== 9. the short-maturity skew formula
def skew_formula(p, T):
    H = p["hurst"]
    return p["rho"] * p["eta"] * np.sqrt(2 * H) / (2 * (H + 0.5) * (H + 1.5)) * T ** (H - 0.5)


def fig_skew_formula(cache):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    out = {}
    TT = np.geomspace(5 / 252, 1.0, 50)
    for color, H in zip(COLORS, (0.08, 0.185, 0.30)):
        p = dict(MEDIAN, hurst=H)
        prices, _ = cache[(H, 2.0, -0.595)]
        skew = sf.smile_statistics(implied_grid(prices).ravel(), K_ALL, T_ALL)[2]
        ax.loglog(DAYS, np.abs(skew), "o", color=color, ms=5, label=f"simulated, $H$ = {H}")
        ax.loglog(TT * 252, np.abs(skew_formula(p, TT)), "--", color=color, lw=1, label="short-maturity formula")
        f = skew_formula(p, DAYS / 252)
        out[H] = dict(ratio_formula_over_sim=[float(a / b) for a, b in zip(f, skew)],
                      sim_exponent=float(sf.skew_power_law(DAYS / 252, skew)[0]), H_minus_half=H - 0.5)
    ax.set_xticks(DAYS)
    ax.set_xticklabels([str(d) for d in DAYS])
    ax.set_xlabel("maturity, business days")
    ax.set_ylabel(r"$|\psi(T)|$")
    ax.set_title(r"Simulated ATM skew against $\rho\eta\sqrt{2H}\,T^{H-1/2}/(2(H+\frac{1}{2})(H+\frac{3}{2}))$")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    tidy(ax)
    save(fig, "l07-skew-formula.png")
    REPORT["skew_formula"] = out


# =========================================================================== 10. what the smile identifies
def fig_identification(cache):
    prod = MEDIAN["rho"] * MEDIAN["eta"]
    etas = (1.4, 2.0, 2.8)
    sets = [dict(MEDIAN, eta=e, rho=prod / e) for e in etas]
    surfaces = []
    for p in sets:
        key = (p["hurst"], p["eta"], round(p["rho"], 6))
        if abs(p["eta"] - 2.0) < 1e-12:
            surfaces.append(cache[(0.185, 2.0, -0.595)])
        else:
            surfaces.append(surface(p, seed=0)[:2])
    p_alt = dict(MEDIAN, rho=-0.7)            # change the product instead
    alt = surface(p_alt, seed=0)[:2]
    base_iv = implied_grid(surfaces[1][0])
    s_iv = sf.implied_vol_noise(DELTAS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    out = {}
    for ax, j in zip(axes[:2], (0, 6)):
        for color, p, (prices, errors) in zip((COLORS[0], COLORS[2], COLORS[3]), sets, surfaces):
            iv = implied_grid(prices)
            ax.plot(DELTAS, iv[j] * 100, "o-", color=color, ms=3.5,
                    label=rf"$\eta$ = {p['eta']}, $\rho$ = {p['rho']:.3f}")
        ax.set_xlabel(r"standardized moneyness $\delta$")
        ax.set_ylabel("implied volatility, %")
        ax.set_title(f"Same $\\rho\\eta$ = {prod:.2f}, {DAYS[j]} days")
    axes[0].legend(frameon=False, fontsize=8.5)
    ax = axes[2]
    for color, p, (prices, errors) in zip((COLORS[0], COLORS[3]), (sets[0], sets[2]), (surfaces[0], surfaces[2])):
        d = (implied_grid(prices) - base_iv) / s_iv[None, :]
        ax.plot(np.arange(63), d.ravel(), "o", color=color, ms=3.5, label=rf"$\eta$ = {p['eta']} at fixed $\rho\eta$")
        out[p["eta"]] = dict(max_abs=float(np.nanmax(np.abs(d))), rms=float(np.sqrt(np.nanmean(d**2))),
                             max_at=[int(v) for v in np.unravel_index(np.nanargmax(np.abs(d)), d.shape)])
    d = (implied_grid(alt[0]) - base_iv) / s_iv[None, :]
    ax.plot(np.arange(63), d.ravel(), "x", color=COLORS[1], ms=4, label=r"$\rho$ = −0.70 at $\eta$ = 2 ($\rho\eta$ = −1.40)")
    out["rho_change"] = dict(max_abs=float(np.nanmax(np.abs(d))), rms=float(np.sqrt(np.nanmean(d**2))),
                             max_at=[int(v) for v in np.unravel_index(np.nanargmax(np.abs(d)), d.shape)])
    levels = {}
    for p, (prices, errors) in zip(sets + [p_alt], surfaces + [alt]):
        iv = implied_grid(prices)
        _, lev, skew, curv = sf.smile_statistics(iv.ravel(), K_ALL, T_ALL)
        levels[f"eta={p['eta']},rho={p['rho']:.3f}"] = dict(atm=[float(x) for x in lev], skew=[float(x) for x in skew],
                                                           curv=[float(x) for x in curv], iv_1w=[float(x) for x in iv[0]],
                                                           iv_1y=[float(x) for x in iv[6]])
    out["smile_statistics"] = levels
    for jj in range(1, 7):
        ax.axvline(9 * jj - 0.5, color="0.9", lw=0.8)
    ax.axhspan(-1, 1, color="0.92", zorder=0)
    ax.set_xticks(9 * np.arange(7) + 4)
    ax.set_xticklabels([f"{d} d" for d in DAYS], fontsize=8)
    ax.set_xlabel(r"contract (9 values of $\delta$ per maturity)")
    ax.set_ylabel(r"difference from the prior centre / $s_{iv}(\delta)$")
    ax.set_title("Differences in units of the quote noise")
    ax.legend(frameon=False, fontsize=8)
    tidy(*axes)
    save(fig, "l07-identification.png")
    REPORT["identification"] = dict(product=float(prod), sets=[{k: float(v) for k, v in s.items()} for s in sets], diffs=out)
    return sets


# =========================================================================== 11. products: forward start, swaps
def products(p, n_pairs=2**14, seed=21):
    dt = 1 / (252 * 4)
    T1, T2 = 63, 126
    n = T2 * 4
    rng = np.random.default_rng(seed)
    vals = []
    remaining = n_pairs
    while remaining:
        m = min(2048, remaining)
        z = rng.standard_normal((m, n, 3))
        pair = []
        for sign in (1.0, -1.0):
            paths = simulate_rough(p, flat_curve(T2 * 1 / 252), dt, sign * z)
            iv = paths["integrated_variance"]
            iw = paths["volatility_driver_integral"]
            i1, i2 = T1 * 4 - 1, T2 * 4 - 1
            dv = iv[:, i2] - iv[:, i1]
            dw = iw[:, i2] - iw[:, i1]
            fwd = np.exp(p["rho"] * dw - 0.5 * p["rho"] ** 2 * dv)
            tot = (1 - p["rho"] ** 2) * dv
            sd = np.sqrt(tot)
            d1 = (np.log(fwd) + 0.5 * tot) / sd
            fs = fwd * norm.cdf(d1) - norm.cdf(d1 - sd)          # conditional Black–Scholes forward-start call
            q = iv[:, i1] / (T1 / 252)                           # realized variance over the first 63 days
            pair.append(np.column_stack([fs, q, np.sqrt(q)]))
        vals.append(0.5 * (pair[0] + pair[1]))
        remaining -= m
    vals = np.concatenate(vals)
    return vals.mean(axis=0), vals.std(axis=0, ddof=1) / np.sqrt(len(vals))


def fig_products(sets):
    out = {}
    means, ses = [], []
    for p in sets:
        m, s = products(p)
        means.append(m)
        ses.append(s)
    means, ses = np.array(means), np.array(ses)
    tau = 63 / 252
    bs_fs = 2 * norm.cdf(np.sqrt(XI) * np.sqrt(tau) / 2) - 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    etas = [p["eta"] for p in sets]
    names = ["forward-start ATM call, 63 → 126 days, % of notional", "variance swap, 63 days: fair strike as volatility, %",
             "volatility swap, 63 days: fair strike, %"]
    vals = [means[:, 0] * 100, np.sqrt(means[:, 1]) * 100, means[:, 2] * 100]
    errs = [ses[:, 0] * 100, ses[:, 1] / (2 * np.sqrt(means[:, 1])) * 100, ses[:, 2] * 100]
    refs = [bs_fs * 100, np.sqrt(XI) * 100, np.sqrt(XI) * 100]
    for ax, title, v, e, r in zip(axes, names, vals, errs, refs):
        ax.errorbar(etas, v, yerr=2 * e, fmt="o-", color=COLORS[0], ms=5, capsize=3)
        ax.axhline(r, color=GREY, ls=":", lw=1, label="Black–Scholes at 25% / the curve's level")
        ax.set_xlabel(r"$\eta$ (with $\rho = -1.19/\eta$)")
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    tidy(*axes)
    save(fig, "l07-products.png")
    REPORT["products"] = dict(etas=etas, forward_start=[float(v) for v in means[:, 0]], forward_start_se=[float(v) for v in ses[:, 0]],
                              varswap_vol=[float(v) for v in np.sqrt(means[:, 1])], varswap_se=[float(v) for v in ses[:, 1]],
                              volswap=[float(v) for v in means[:, 2]], volswap_se=[float(v) for v in ses[:, 2]],
                              bs_forward_start=float(bs_fs))


# =========================================================================== 12. the time convention
def fig_time_convention():
    _, r = load_voo()
    gap = r.index.to_series().diff().dt.days.values[1:]
    sq = r.values[1:] ** 2
    base = sq[gap == 1].mean()
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    gaps = (1, 3, 4)
    for g in gaps:
        sel = sq[gap == g]
        out[str(g)] = dict(days=int(sel.size), ratio=float(sel.mean() / base), se=float(sel.std(ddof=1) / np.sqrt(sel.size) / base))
    x = np.arange(len(gaps))
    ax.bar(x, [out[str(g)]["ratio"] for g in gaps], yerr=[2 * out[str(g)]["se"] for g in gaps], color=COLORS[0],
           alpha=0.8, capsize=4, label="VOO, mean squared log return (bars: 2 s.e.)")
    ax.plot(x, gaps, "s--", color=COLORS[3], ms=6, label="if variance accrued per calendar day")
    ax.axhline(1, color=GREY, ls=":", lw=1, label="if variance accrued per trading day")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g} day{'s' if g > 1 else ''}\n({out[str(g)]['days']} returns)" for g in gaps])
    ax.set_xlabel("calendar days since the previous close")
    ax.set_ylabel("variance relative to one-day returns")
    ax.set_title("Variance accrues on trading days, 2020–2026")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax = axes[1]
    Hs = np.linspace(0.02, 0.35, 100)
    ax.plot(Hs, 252.0 ** -Hs, color=COLORS[0], label=r"time in trading days instead of years: $252^{-H}$")
    ax.plot(Hs, (365 / 252) ** Hs, color=COLORS[2], label=r"business days divided by 365: $(365/252)^{H}$")
    ax.axvline(MEDIAN["hurst"], color=GREY, ls=":", lw=1)
    ax.set_xlabel("$H$")
    ax.set_ylabel(r"factor on $\eta$")
    ax.set_title(r"$\eta$ is measured in year$^{-H}$")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l07-time-convention.png")
    REPORT["time_convention"] = dict(by_gap=out, eta_day_factor_at_median=float(252.0 ** -MEDIAN["hurst"]),
                                     eta_365_factor_at_median=float((365 / 252) ** MEDIAN["hurst"]))


def main():
    t0 = time.time()
    fig_roughness_data()
    fig_driver()
    fig_variance_law()
    fig_memory()
    fig_hybrid()
    fig_estimators()
    base = fig_budget()
    cache = fig_sweeps(base)
    fig_skew_formula(cache)
    sets = fig_identification(cache)
    fig_products(sets)
    fig_time_convention()
    REPORT["wall_seconds"] = time.time() - t0
    print(json.dumps(REPORT, indent=1, default=float))


if __name__ == "__main__":
    main()
