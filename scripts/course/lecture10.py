"""Figures of Lecture 10: Bayesian calibration.

Usage: python scripts/course/lecture10.py   (writes docs/course/images/l10-*.png and prints the numbers quoted in the text)

Uses the toy smile of lecture09.py and, for Heston, the repository's Fourier pricer and product pricer.
"""

import json
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist
from scipy.stats import norm

import lecture09 as L9
from common import COLORS, DAYS, DELTAS, GREY, ROOT, load_spy, plt, save, sf, tidy
from sbi4finance.calibration import pricing, products, tasks

REPORT = {}
RESULTS = ROOT / "tutorial_final" / "results" / "calibration"
SIG_BOX = (0.05, 0.60)


# ------------------------------------------------------------------------------------------------
# Black-Scholes: exact posterior of one sigma on a grid
# ------------------------------------------------------------------------------------------------
def bs_prices(sig, days=DAYS):
    out = []
    for day in days:
        T = day / 252
        k = DELTAS * sf.SIGMA_REF * np.sqrt(T)
        out.append(sf.bs_price(k, T, sig, np.where(DELTAS <= 0, "put", "call")))
    return np.concatenate(out)


def grid_posterior_1d(loglik, lo, hi, centre, width, n=2001):
    """Posterior on a fine grid around the likelihood peak, under a uniform prior on (lo, hi)."""
    g = np.linspace(max(lo, centre - width), min(hi, centre + width), n)
    lp = np.array([loglik(x) for x in g])
    p = np.exp(lp - lp.max())
    p /= np.trapezoid(p, g)
    cdf = np.cumsum(p) * (g[1] - g[0])
    cdf /= cdf[-1]
    mean = np.trapezoid(g * p, g)
    sd = np.sqrt(np.trapezoid((g - mean) ** 2 * p, g))
    return dict(grid=g, density=p, mean=float(mean), sd=float(sd),
                q05=float(np.interp(0.05, cdf, g)), q95=float(np.interp(0.95, cdf, g)), mode=float(g[np.argmax(p)]))


def fig_bs_synthetic():
    sd = np.array(sf.quote_noise(sf.design_grid()))
    rng = np.random.default_rng(21)
    y = bs_prices(0.20) + sd * rng.standard_normal(63)
    loglik = lambda s: -0.5 * np.sum(((y - bs_prices(s)) / sd) ** 2)
    post = grid_posterior_1d(loglik, *SIG_BOX, 0.2, 0.01)
    se = 1 / np.sqrt(np.sum(np.concatenate([sf.bs_vega(DELTAS * 0.2 * np.sqrt(d / 252), d / 252, post["mode"])
                                            for d in DAYS]) ** 2 / sd**2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.fill_between([SIG_BOX[0] * 100, SIG_BOX[1] * 100], 0, 1 / (SIG_BOX[1] - SIG_BOX[0]) / 100, color=COLORS[2],
                    alpha=0.3, label=r"prior: uniform on 5% to 60%")
    ax.plot(post["grid"] * 100, post["density"] / 100, color=COLORS[0], label="posterior")
    ax.set_xlim(0, 65)
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1e3)
    ax.set_xlabel(r"$\sigma$, %")
    ax.set_ylabel("density per volatility point (log scale)")
    ax.set_title("63 quotes turn a wide prior into a narrow posterior")
    ax.legend(frameon=False, fontsize=8.5)
    ax = axes[1]
    g = post["grid"]
    ax.plot((g - 0.2) * 100, post["density"] / 100, color=COLORS[0], lw=2, label="exact posterior (grid)")
    ax.plot((g - 0.2) * 100, norm.pdf(g, post["mode"], se) / 100, color=COLORS[1], ls="--",
            label=r"normal at $\hat\sigma$ with the Gauss–Newton sd")
    ax.axvspan((post["q05"] - 0.2) * 100, (post["q95"] - 0.2) * 100, color=COLORS[0], alpha=0.1,
               label="90% credible interval")
    ax.set_xlim(-0.35, 0.3)
    ax.set_ylim(0, post["density"].max() / 100 * 1.45)
    ax.set_xlabel(r"$\sigma$ minus the true 20%, volatility points")
    ax.set_ylabel("density per volatility point")
    ax.set_title("A well-identified model: posterior = Gauss–Newton")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    tidy(*axes)
    save(fig, "l10-black-scholes-synthetic.png")
    REPORT["bs_synthetic"] = dict(mean=post["mean"], sd=post["sd"], q05=post["q05"], q95=post["q95"], gn_se=float(se),
                                  prior_sd=float((SIG_BOX[1] - SIG_BOX[0]) / np.sqrt(12)))


def spy_iv(prices):
    out = []
    for j, day in enumerate(DAYS):
        T = day / 252
        out.append(sf.implied_vol(prices[9 * j:9 * j + 9], DELTAS * 0.2 * np.sqrt(T), T,
                                  np.where(DELTAS <= 0, "put", "call")))
    return np.concatenate(out)


def fig_bs_spy(spy):
    y = np.array(spy["prices_forward_units"])
    sd = np.array(spy["noise_sd"])
    loglik = lambda s: -0.5 * np.sum(((y - bs_prices(s)) / sd) ** 2)
    post = grid_posterior_1d(loglik, *SIG_BOX, 0.145, 0.01)
    z_one = (y - bs_prices(post["mean"])) / sd
    per = []
    for j, day in enumerate(DAYS):
        sl = slice(9 * j, 9 * j + 9)
        ll = lambda s, sl=sl, day=day: -0.5 * np.sum(((y[sl] - bs_prices(s, [day])) / sd[sl]) ** 2)
        per.append(grid_posterior_1d(ll, *SIG_BOX, 0.15, 0.08))
    sig_per = np.concatenate([[p["mean"]] * 9 for p in per])
    z_per = (y - np.concatenate([bs_prices(p["mean"], [d]) for p, d in zip(per, DAYS)])) / sd
    chi_one = float(np.sum(z_one**2))
    chi_per = float(np.sum(z_per**2))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    ax.plot(DAYS, [p["mean"] * 100 for p in per], "o-", color=COLORS[3], ms=4, label=r"posterior mean of $\sigma_j$, one per maturity")
    ax.fill_between(DAYS, [p["q05"] * 100 for p in per], [p["q95"] * 100 for p in per], color=COLORS[3], alpha=0.2)
    ax.axhline(post["mean"] * 100, color=COLORS[0], lw=2, label=f"one σ for all: {post['mean']:.2%} ± {post['sd'] * 100:.3f} pts")
    ax.plot(DAYS, np.array(spy["atm_implied_vol"]) * 100, "s--", color="0.4", ms=4, label="SPY at-the-money implied vol")
    ax.set_xscale("log")
    ax.set_xticks(DAYS)
    ax.set_xticklabels([str(d) for d in DAYS])
    ax.set_xlabel("maturity, business days")
    ax.set_ylabel("% per year")
    ax.set_title("Black–Scholes posteriors on SPY, 21 October 2026")
    ax.legend(frameon=False, fontsize=8)
    for ax, z, title in ((axes[1], z_one, f"Residuals, one σ: χ² = {chi_one:,.0f} on 62 dof"),
                         (axes[2], z_per, f"Residuals, one σ per maturity: χ² = {chi_per:,.0f} on 56 dof")):
        for color, j in zip(COLORS, range(len(DAYS))):
            ax.plot(DELTAS, z[9 * j:9 * j + 9], "o-", color=color, ms=3.5, label=f"{DAYS[j]} d")
        ax.axhspan(-2, 2, color="0.9", zorder=0)
        ax.set_yscale("symlog", linthresh=2)
        ticks = [-10, -5, -2, 0, 2, 5, 10, 20]
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel(r"standardized moneyness $\delta$")
        ax.set_ylabel(r"$(y_c - P_c)/\sigma_c$ at the posterior mean")
        ax.set_title(title, fontsize=10)
    axes[1].legend(frameon=False, fontsize=7.5, ncol=2, loc="lower left")
    tidy(*axes)
    save(fig, "l10-black-scholes-spy.png")
    REPORT["bs_spy"] = dict(mean=post["mean"], sd=post["sd"], q05=post["q05"], q95=post["q95"],
                            chi2_one=chi_one, chi2_per=chi_per,
                            p_value_one=float(chi2_dist.sf(chi_one, 62)),
                            absz_max_one=float(np.abs(z_one).max()), within2_one=float(np.mean(np.abs(z_one) < 2)),
                            absz_max_per=float(np.abs(z_per).max()), within2_per=float(np.mean(np.abs(z_per) < 2)),
                            per_mean=[p["mean"] for p in per], per_sd=[p["sd"] for p in per],
                            atm=spy["atm_implied_vol"])

    # posterior predictive check in implied-volatility units
    rng = np.random.default_rng(3)
    draws = rng.choice(post["grid"], size=400, p=post["density"] / post["density"].sum())
    reps = np.array([bs_prices(s) + sd * rng.standard_normal(63) for s in draws])
    iv_obs = spy_iv(y)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    out = {}
    for ax, j in zip(axes, (0, 4, 6)):
        T = DAYS[j] / 252
        k = DELTAS * 0.2 * np.sqrt(T)
        iv_rep = np.array([sf.implied_vol(r[9 * j:9 * j + 9], k, T, np.where(DELTAS <= 0, "put", "call")) for r in reps])
        lo, hi = np.nanquantile(iv_rep, 0.05, axis=0), np.nanquantile(iv_rep, 0.95, axis=0)
        ax.fill_between(DELTAS, lo * 100, hi * 100, color=COLORS[0], alpha=0.25,
                        label="replicated quotes, 5%–95% (posterior predictive)")
        ax.plot(DELTAS, iv_obs[9 * j:9 * j + 9] * 100, "o-", color=COLORS[3], ms=4, label="observed SPY quotes")
        ax.set_title(f"{DAYS[j]} business days")
        ax.set_xlabel(r"standardized moneyness $\delta$")
        out[int(DAYS[j])] = dict(outside=int(np.sum((iv_obs[9 * j:9 * j + 9] < lo) | (iv_obs[9 * j:9 * j + 9] > hi))),
                                 nan_share=float(np.mean(np.isnan(iv_rep))))
    axes[0].set_ylabel("implied volatility, %")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    tidy(*axes)
    save(fig, "l10-posterior-predictive-spy.png")
    REPORT["ppc_spy"] = out


# ------------------------------------------------------------------------------------------------
# Toy smile: the posterior of a weakly identified pair
# ------------------------------------------------------------------------------------------------
ETA_G = np.linspace(*L9.ETA_BOX, 321)
RHO_G = np.linspace(*L9.RHO_BOX, 241)
EG, RG = np.meshgrid(ETA_G, RHO_G)


def toy_posterior(y):
    lp = -0.5 * L9.toy_chi2(y, EG, RG)
    p = np.exp(lp - lp.max())
    return p / p.sum()


def quantiles_1d(values, weights, qs):
    idx = np.argsort(values)
    v, w = values[idx], weights[idx]
    c = np.cumsum(w)
    c /= c[-1]
    return [float(np.interp(q, c, v)) for q in qs]


def fig_toy_posterior():
    rng = np.random.default_rng(7)
    y = L9.toy_data(L9.TRUTH_EDGE, rng)
    p = toy_posterior(y)
    best, _ = L9.toy_fit(y)
    cov, _ = L9.toy_gn_cov(*best)
    flat = p.ravel()
    order = np.argsort(flat)[::-1]
    level90 = flat[order][np.searchsorted(np.cumsum(flat[order]), 0.9)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    ax = axes[0]
    ax.contourf(EG, RG, p, levels=[level90, p.max() * 2], colors=[COLORS[0]], alpha=0.35)
    ax.contour(EG, RG, p, levels=np.quantile(flat[flat > 1e-12], [0.9, 0.99, 0.999]), colors=COLORS[0], linewidths=0.7)
    vals, vecs = np.linalg.eigh(cov)
    t = np.linspace(0, 2 * np.pi, 200)
    ell = best[:, None] + vecs @ (np.sqrt(vals)[:, None] * np.vstack([np.cos(t), np.sin(t)])) * np.sqrt(4.605)
    ax.plot(ell[0], ell[1], color=COLORS[1], lw=1.5, ls="--", label="Gauss–Newton 90% ellipse")
    ax.plot([], [], color=COLORS[0], lw=8, alpha=0.4, label="posterior 90% region")
    ax.plot(*L9.TRUTH_EDGE, "*", color="k", ms=11, label="truth")
    ax.set_xlim(*L9.ETA_BOX)
    ax.set_ylim(-1.3, -0.2)
    ax.axhline(L9.RHO_BOX[0], color=COLORS[3], ls=":", lw=1)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\rho$")
    ax.set_title("Toy smile: the posterior follows the valley")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax = axes[1]
    pe = p.sum(axis=0)
    ax.plot(ETA_G, pe / pe.max(), color=COLORS[0], label="posterior of $\\eta$")
    ax.plot(ETA_G, norm.pdf(ETA_G, best[0], np.sqrt(cov[0, 0])) / norm.pdf(0, 0, np.sqrt(cov[0, 0])), color=COLORS[1],
            ls="--", label="Gauss–Newton normal")
    ax.axvline(L9.TRUTH_EDGE[0], color="k", ls=":", lw=1)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("density (peak = 1)")
    ax.set_title(r"$\eta$ alone: wide and skewed")
    ax.legend(frameon=False, fontsize=8.5)
    ax = axes[2]
    prod = (EG * RG).ravel()
    jit = np.random.default_rng(0)
    pick = jit.choice(flat.size, size=400_000, p=flat)
    de, dr = ETA_G[1] - ETA_G[0], RHO_G[1] - RHO_G[0]
    sample = (EG.ravel()[pick] + jit.uniform(-de / 2, de / 2, pick.size)) * \
        np.clip(RG.ravel()[pick] + jit.uniform(-dr / 2, dr / 2, pick.size), L9.RHO_BOX[0], L9.RHO_BOX[1])
    h, edges = np.histogram(sample, bins=np.linspace(-1.1, -0.6, 100), density=True)
    ax.plot(0.5 * (edges[1:] + edges[:-1]), h / h.max(), color=COLORS[0], label=r"posterior of $\rho\eta$")
    ax.axvline(L9.TRUTH_EDGE[0] * L9.TRUTH_EDGE[1], color="k", ls=":", lw=1)
    ax.set_xlabel(r"$\rho\eta$")
    ax.set_title(r"The product: narrow")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l10-toy-posterior.png")
    q = lambda v: quantiles_1d(v, flat, [0.05, 0.5, 0.95])
    sd_gn = np.sqrt(np.diag(cov))
    REPORT["toy_posterior"] = dict(eta=q(EG.ravel()), rho=q(RG.ravel()), product=q(prod),
                                   gn_eta=[float(best[0] - 1.645 * sd_gn[0]), float(best[0] + 1.645 * sd_gn[0])],
                                   gn_rho=[float(best[1] - 1.645 * sd_gn[1]), float(best[1] + 1.645 * sd_gn[1])],
                                   best=[float(v) for v in best])


def fig_coverage():
    rng = np.random.default_rng(17)
    n = 2000
    out = {}
    eta_flat, rho_flat = EG.ravel(), RG.ravel()

    def credible(y):
        p = toy_posterior(y).ravel()
        return quantiles_1d(eta_flat, p, [0.05, 0.95]), quantiles_1d(rho_flat, p, [0.05, 0.95])

    for label, truth in (("near the bound", L9.TRUTH_EDGE), ("inside the box", L9.TRUTH_INSIDE)):
        be = br = ge = gr = 0
        for i in range(n):
            y = L9.toy_data(truth, rng)
            ie, ir = credible(y)
            be += ie[0] <= truth[0] <= ie[1]
            br += ir[0] <= truth[1] <= ir[1]
            best, sd, _ = L9.gn_interval(y)
            ge += abs(best[0] - truth[0]) <= 1.645 * sd[0]
            gr += abs(best[1] - truth[1]) <= 1.645 * sd[1]
        out[label] = dict(bayes_eta=be / n, bayes_rho=br / n, gn_eta=ge / n, gn_rho=gr / n)
    truths, hits = [], []
    for i in range(n):
        tr = (rng.uniform(*L9.ETA_BOX), rng.uniform(*L9.RHO_BOX))
        y = L9.toy_data(tr, rng)
        ie, ir = credible(y)
        best, sd, _ = L9.gn_interval(y)
        truths.append(tr)
        hits.append((ie[0] <= tr[0] <= ie[1], ir[0] <= tr[1] <= ir[1],
                     abs(best[0] - tr[0]) <= 1.645 * sd[0], abs(best[1] - tr[1]) <= 1.645 * sd[1]))
    truths, hits = np.array(truths), np.array(hits, dtype=float)
    out["truths from the prior"] = dict(bayes_eta=hits[:, 0].mean(), bayes_rho=hits[:, 1].mean(),
                                        gn_eta=hits[:, 2].mean(), gn_rho=hits[:, 3].mean())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    labels = list(out)
    xs = np.arange(len(labels))
    for i, (key, name, color) in enumerate((("bayes_eta", r"posterior, $\eta$", COLORS[0]),
                                            ("gn_eta", r"Gauss–Newton, $\eta$", COLORS[2]),
                                            ("bayes_rho", r"posterior, $\rho$", COLORS[1]),
                                            ("gn_rho", r"Gauss–Newton, $\rho$", COLORS[5]))):
        ax.bar(xs + (i - 1.5) * 0.2, [out[l][key] * 100 for l in labels], 0.2, color=color, label=name)
    ax.axhline(90, color="0.3", ls="--", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(60, 100)
    ax.set_ylabel("share of 90% intervals that contain the truth, %")
    ax.set_title("Toy smile, 2,000 data sets per case")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax = axes[1]
    bins = np.linspace(*L9.RHO_BOX, 9)
    idx = np.clip(np.digitize(truths[:, 1], bins) - 1, 0, len(bins) - 2)
    cen = 0.5 * (bins[1:] + bins[:-1])
    for col, name, color, ls in ((0, r"posterior, $\eta$", COLORS[0], "-"), (2, r"Gauss–Newton, $\eta$", COLORS[2], "--"),
                                 (1, r"posterior, $\rho$", COLORS[1], "-"), (3, r"Gauss–Newton, $\rho$", COLORS[5], "--")):
        ax.plot(cen, [hits[idx == i, col].mean() * 100 for i in range(len(cen))], "o", ls=ls, color=color, ms=4, label=name)
    ax.axhline(90, color="0.3", ls="--", lw=1)
    ax.set_ylim(60, 100)
    ax.set_xlabel(r"true $\rho$ (truths drawn from the prior)")
    ax.set_ylabel("coverage within the bin, %")
    ax.set_title("Averaged over the prior, the posterior is exact; bin by bin, not")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    tidy(*axes)
    save(fig, "l10-credible-vs-confidence.png")
    REPORT["coverage"] = out
    REPORT["coverage_bins"] = dict(centres=[float(c) for c in cen],
                                   bayes_eta=[float(hits[idx == i, 0].mean()) for i in range(len(cen))],
                                   bayes_rho=[float(hits[idx == i, 1].mean()) for i in range(len(cen))],
                                   gn_eta=[float(hits[idx == i, 2].mean()) for i in range(len(cen))],
                                   gn_rho=[float(hits[idx == i, 3].mean()) for i in range(len(cen))])


# ------------------------------------------------------------------------------------------------
# MCMC
# ------------------------------------------------------------------------------------------------
def split_rhat(chains):
    """chains [m, n]: split-R-hat (Gelman et al. 2013)."""
    m, n = chains.shape
    half = n // 2
    c = np.concatenate([chains[:, :half], chains[:, half:2 * half]])
    means = c.mean(axis=1)
    W = c.var(axis=1, ddof=1).mean()
    B = half * means.var(ddof=1)
    var = (half - 1) / half * W + B / half
    return float(np.sqrt(var / W))


def ess(chains):
    """Effective sample size pooled over chains, with Geyer's initial positive sequence.

    Returns the ESS and the first 60 autocorrelations.
    """
    m, n = chains.shape
    x = chains - chains.mean(axis=1, keepdims=True)
    f = np.fft.rfft(x, n=2 * n, axis=1)
    acov = np.fft.irfft(f * np.conj(f), axis=1)[:, :n] / n
    acf = acov.mean(axis=0) / acov.mean(axis=0)[0]
    pairs = acf[: (n // 2) * 2].reshape(-1, 2).sum(axis=1)   # Gamma_k = rho_2k + rho_2k+1
    stop = np.argmax(pairs < 0) if np.any(pairs < 0) else len(pairs)
    tau = -1 + 2 * pairs[:stop].sum()
    return float(m * n / max(tau, 1e-9)), acf[:60]


def metropolis(logp, start, step_cov, n, rng):
    d = len(start)
    L = np.linalg.cholesky(step_cov)
    x = np.array(start, dtype=float)
    lx = logp(x)
    out = np.empty((n, d))
    acc = 0
    for i in range(n):
        prop = x + L @ rng.standard_normal(d)
        lp = logp(prop)
        if np.log(rng.uniform()) < lp - lx:
            x, lx = prop, lp
            acc += 1
        out[i] = x
    return out, acc / n


def fig_toy_mcmc():
    rng = np.random.default_rng(7)
    y = L9.toy_data(L9.TRUTH_EDGE, rng)

    def logp(t):
        if not (L9.ETA_BOX[0] < t[0] < L9.ETA_BOX[1] and L9.RHO_BOX[0] < t[1] < L9.RHO_BOX[1]):
            return -np.inf
        return -0.5 * float(L9.toy_chi2(y, np.array(t[0]), np.array(t[1])))

    p_grid = toy_posterior(y).ravel()
    m_e, m_r = np.sum(p_grid * EG.ravel()), np.sum(p_grid * RG.ravel())
    c_post = np.cov(np.vstack([EG.ravel(), RG.ravel()]), aweights=p_grid)
    rng = np.random.default_rng(1)
    starts = [(3.3, -0.25), (0.6, -0.95), (2.0, -0.4), (1.2, -0.7)]
    out = {}
    for label, cov in (("steps too small", np.diag([0.02, 0.01]) ** 2),
                       ("covariance-scaled steps", c_post * 2.38**2 / 2)):
        chains, accs = [], []
        for st in starts:
            c, a = metropolis(logp, st, cov, 4000, rng)
            chains.append(c)
            accs.append(a)
        out[label] = (np.array(chains), accs)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, label in zip(axes[:2], out):
        chains, accs = out[label]
        for color, c in zip(COLORS, chains):
            ax.plot(c[:, 0], color=color, lw=0.6)
        r = split_rhat(chains[:, 1000:, 0])
        e, _ = ess(chains[:, 1000:, 0])
        ax.set_title(f"{label}: acceptance {np.mean(accs):.0%}, " + r"$\hat R$" + f" = {r:.2f}, ESS = {e:.0f}",
                     fontsize=10)
        ax.set_xlabel("iteration")
        ax.set_ylabel(r"$\eta$")
        ax.axvline(1000, color="0.6", ls=":", lw=1)
        REPORT.setdefault("toy_mcmc", {})[label] = dict(acceptance=float(np.mean(accs)), rhat=r, ess=e)
    ax = axes[2]
    chains, _ = out["covariance-scaled steps"]
    pooled = chains[:, 1000:, 0].ravel()
    ax.hist(pooled, bins=60, density=True, color=COLORS[0], alpha=0.4, label="4 chains after burn-in, 12,000 draws")
    p = toy_posterior(y)
    pe = p.sum(axis=0)
    ax.plot(ETA_G, pe / np.trapezoid(pe, ETA_G), color="0.2", label="exact posterior (grid)")
    _, acf = ess(chains[:, 1000:, 0])
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("density")
    ax.set_title("The chains reproduce the grid posterior")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l10-toy-mcmc.png")


# ------------------------------------------------------------------------------------------------
# Heston: exact posterior by MCMC on a synthetic smile, then products
# ------------------------------------------------------------------------------------------------
TASK = tasks.heston_task()
CONTRACTS = TASK["context"]["contracts"]
SIGMA_C = tasks.sigma(TASK)
FIXED = TASK["fixed_parameters"]
LO = np.array([0.01, 0.01, 0.1, -0.95])
HI = np.array([0.16, 0.16, 1.5, -0.1])
TRUTH_H = (0.04, 0.05, 0.4, -0.7)


def heston_surface(t):
    params = dict(FIXED, theta=t[1], vol_of_vol=t[2], rho=t[3])
    try:
        return pricing.heston_surface(params, t[0], CONTRACTS, nodes=1024, upper_limit=600)
    except ValueError:
        return tasks.heston_prices(TASK, params, t[0])[0]


def to_z(t):
    u = (np.asarray(t) - LO) / (HI - LO)
    return np.log(u / (1 - u))


def from_z(z):
    return LO + (HI - LO) / (1 + np.exp(-np.asarray(z)))


def heston_logpost(z, y):
    t = from_z(z)
    try:
        r = (y - heston_surface(t)) / SIGMA_C
    except ValueError:
        return -np.inf
    # uniform prior on the box, written in logit coordinates: log Jacobian of the logistic map
    u = (t - LO) / (HI - LO)
    return -0.5 * float(r @ r) + float(np.sum(np.log(u * (1 - u))))


def fig_heston_mcmc():
    rng = np.random.default_rng(3)
    y = heston_surface(TRUTH_H) + SIGMA_C * rng.standard_normal(63)
    fit = least_squares(lambda t: (heston_surface(t) - y) / SIGMA_C, list(TRUTH_H), bounds=(LO, HI))
    t_hat = fit.x
    J = np.empty((63, 4))
    z_hat = to_z(t_hat)
    for j in range(4):
        e = np.zeros(4)
        e[j] = 1e-4
        J[:, j] = (heston_surface(from_z(z_hat + e)) - heston_surface(from_z(z_hat - e))) / (2e-4) / SIGMA_C
    cov_z = np.linalg.inv(J.T @ J)
    Jt = np.empty((63, 4))
    for j in range(4):
        h = 1e-6 * (HI[j] - LO[j])
        e = np.zeros(4)
        e[j] = h
        Jt[:, j] = (heston_surface(t_hat + e) - heston_surface(t_hat - e)) / (2 * h) / SIGMA_C
    cov_t = np.linalg.inv(Jt.T @ Jt)
    # Stage 1 (adaptation, discarded): proposal from the Gauss-Newton covariance in logit coordinates.
    # Stage 2 (kept): proposal from the covariance of the pooled stage-1 draws, scaled by 2.38^2 / d.
    n_chain, n_adapt, n_keep = 4, 1500, 4000
    burn = n_adapt
    t0 = time.time()
    stage1 = []
    for c in range(n_chain):
        start = z_hat + rng.multivariate_normal(np.zeros(4), cov_z * 4)
        ch, _ = metropolis(lambda z: heston_logpost(z, y), start, cov_z * 2.38**2 / 4, n_adapt, rng)
        stage1.append(ch)
    stage1 = np.array(stage1)
    cov_adapt = np.cov(stage1[:, n_adapt // 3:, :].reshape(-1, 4).T)
    chains, accs = [], []
    for c in range(n_chain):
        ch, a = metropolis(lambda z: heston_logpost(z, y), stage1[c, -1], cov_adapt * 2.38**2 / 4, n_keep, rng)
        chains.append(np.vstack([stage1[c], ch]))
        accs.append(a)
    seconds = time.time() - t0
    chains = np.array(chains)
    draws_z = chains[:, burn:, :]
    draws = from_z(draws_z.reshape(-1, 4))
    names = [r"$v_0$", r"$\theta_H$", r"$\sigma_v$", r"$\rho$"]
    rh = [split_rhat(from_z(draws_z)[..., j]) for j in range(4)]
    es = [ess(from_z(draws_z)[..., j])[0] for j in range(4)]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6))
    for j in range(4):
        ax = axes[0, j]
        for color, ch in zip(COLORS, from_z(chains)[..., j]):
            ax.plot(ch, color=color, lw=0.5)
        ax.axvline(burn, color="0.6", ls=":", lw=1)
        ax.set_title(f"{names[j]}: " + r"$\hat R$" + f" = {rh[j]:.3f}, ESS = {es[j]:.0f}", fontsize=10)
        ax.set_xlabel("iteration")
        ax = axes[1, j]
        ax.hist(draws[:, j], bins=50, density=True, color=COLORS[0], alpha=0.5, label="posterior (MCMC)")
        xx = np.linspace(draws[:, j].min(), draws[:, j].max(), 200)
        ax.plot(xx, norm.pdf(xx, t_hat[j], np.sqrt(cov_t[j, j])), color=COLORS[1], ls="--", label="Gauss–Newton normal")
        ax.axvline(TRUTH_H[j], color="k", ls=":", lw=1.2, label="truth")
        ax.set_xlabel(names[j])
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        if j == 0:
            ax.legend(frameon=False, fontsize=8)
    tidy(*axes.ravel())
    save(fig, "l10-heston-mcmc.png")
    q = np.quantile(draws, [0.05, 0.5, 0.95], axis=0)
    corr = np.corrcoef(draws.T)
    REPORT["heston_mcmc"] = dict(truth=TRUTH_H, t_hat=[float(v) for v in t_hat], chi2_hat=float(2 * fit.cost),
                                 acceptance=[float(a) for a in accs], rhat=rh, ess=es, seconds=seconds,
                                 evaluations=n_chain * (n_adapt + n_keep), kept=n_chain * n_keep,
                                 q05=[float(v) for v in q[0]], q50=[float(v) for v in q[1]], q95=[float(v) for v in q[2]],
                                 gn_sd=[float(v) for v in np.sqrt(np.diag(cov_t))],
                                 post_sd=[float(v) for v in draws.std(axis=0)],
                                 corr_xi_rho=float(corr[2, 3]), corr_v0_theta=float(corr[0, 1]))
    return draws, t_hat


def fig_products(draws, t_hat):
    rng = np.random.default_rng(5)
    pick = draws[rng.choice(len(draws), 300, replace=False)]
    spec = products.default_specification()

    def price(t, seed):
        params = dict(FIXED, initial_variance=t[0], theta=t[1], vol_of_vol=t[2], rho=t[3])
        return products.price_draw("heston", params, None, spec, 8192, seed)[0]

    t0 = time.time()
    common = np.array([price(t, 2026) for t in pick])
    independent = np.array([price(t, 3000 + i) for i, t in enumerate(pick[:150])])
    point = price(t_hat, 2026)
    truth_price = price(np.array(TRUTH_H), 2026)
    seconds = time.time() - t0
    names = ["variance swap 63 d", "volatility swap 63 d", "forward-start 63 → 126 d", "2-year 90% put"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0))
    out = {}
    for j, ax in enumerate(axes):
        rel = (common[:, j] / point[j] - 1) * 100
        ax.hist(rel, bins=40, color=COLORS[0], alpha=0.5, density=True)
        q90 = np.quantile(common[:, j], 0.9)
        q10 = np.quantile(common[:, j], 0.1)
        ax.axvline(0, color=COLORS[1], lw=2, label="least-squares price")
        ax.axvline((q90 / point[j] - 1) * 100, color=COLORS[3], ls="--", label="90% quantile")
        ax.axvline((truth_price[j] / point[j] - 1) * 100, color="k", ls=":", label="price at the truth")
        ax.set_title(names[j], fontsize=10)
        ax.set_xlabel("price relative to the least-squares price, %")
        out[names[j]] = dict(point=float(point[j]), mean=float(common[:, j].mean()),
                             q05=float(np.quantile(common[:, j], 0.05)), q95=float(np.quantile(common[:, j], 0.95)),
                             range90_rel=float((np.quantile(common[:, j], 0.95) - np.quantile(common[:, j], 0.05))
                                               / common[:, j].mean()),
                             reserve_short_rel=float((q90 - point[j]) / point[j]),
                             reserve_long_rel=float((point[j] - q10) / point[j]),
                             truth=float(truth_price[j]),
                             sd_common=float(common[:150, j].std()), sd_independent=float(independent[:, j].std()))
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("density")
    tidy(*axes)
    save(fig, "l10-heston-products.png")
    # law of total variance: share of each product's variance explained by each parameter alone
    fig, axes2 = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes2[0]
    labels = [r"$v_0$", r"$\theta_H$", r"$\sigma_v$", r"$\rho$"]
    shares = np.zeros((4, 4))
    for j in range(4):
        for i in range(4):
            shares[j, i] = np.corrcoef(pick[:, i], common[:, j])[0, 1] ** 2
    for i in range(4):
        ax.bar(np.arange(4) + (i - 1.5) * 0.2, shares[:, i] * 100, 0.2, color=COLORS[i], label=labels[i])
    ax.set_xticks(range(4))
    ax.set_xticklabels([n.replace(" 6", "\n6").replace(" → ", "\n→ ") for n in names], fontsize=8.5)
    ax.set_ylabel(r"$R^2$ of the price on one parameter, %")
    ax.set_title("Which parameter each price's range comes from")
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper center")
    ax.set_ylim(0, 110)
    ax = axes2[1]
    j = 2
    ax.plot(pick[:150, 2], independent[:, j], "o", color=COLORS[2], ms=3, alpha=0.6,
            label=f"independent random numbers per draw: sd {independent[:, j].std() / point[j] * 100:.2f}%")
    ax.plot(pick[:150, 2], common[:150, j], "o", color=COLORS[0], ms=3, alpha=0.8,
            label=f"common random numbers: sd {common[:150, j].std() / point[j] * 100:.2f}%")
    ax.set_xlabel(r"$\sigma_v$ of the posterior draw")
    ax.set_ylabel("forward-start price")
    ax.set_title("Common random numbers isolate the parameter effect")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    tidy(*axes2)
    save(fig, "l10-products-variance-and-crn.png")
    REPORT["products"] = dict(seconds=seconds, results=out,
                              shares={names[j]: {labels[i]: float(shares[j, i]) for i in range(4)} for j in range(4)})


def fig_cached_ranges():
    h = json.loads((RESULTS / "heston" / "s7" / "products.json").read_text())
    r = json.loads((RESULTS / "rough-pilot" / "s7" / "products.json").read_text())
    keys = ["variance_swap", "volatility_swap", "forward_start_call", "put_out_of_grid"]
    names = ["variance swap\n63 d", "volatility swap\n63 d", "forward-start\n63 → 126 d", "2-year\n90% put"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    out = {}
    for m, (model, d, color) in enumerate((("Heston", h, COLORS[4]), ("rough Bergomi (pilot)", r, COLORS[0]))):
        for s, srf in enumerate(d["surfaces"]):
            vals = [srf["products"][k]["range_90_relative"] * 100 for k in keys]
            ax.bar(np.arange(4) + (m * 3 + s - 2.5) * 0.13, vals, 0.13, color=color, alpha=0.5 + 0.17 * s,
                   label=model if s == 0 else None)
            out[f"{model}/{srf['test_row']}"] = [float(v) for v in vals]
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("90% range of the price, % of the posterior mean")
    ax.set_title("Three test smiles per model (cached S7): the variance swap's range is zero only under rough Bergomi")
    ax.legend(frameon=False, fontsize=9)
    tidy(ax)
    save(fig, "l10-cached-ranges.png")
    REPORT["cached_ranges"] = out


def main():
    import sys
    only = set(sys.argv[1:])
    run = lambda name: not only or name in only
    t0 = time.time()
    spy = load_spy()
    if run("bs"):
        fig_bs_synthetic()
        fig_bs_spy(spy)
    if run("toy"):
        fig_toy_posterior()
        fig_coverage()
        fig_toy_mcmc()
    if run("heston"):
        draws, t_hat = fig_heston_mcmc()
        fig_products(draws, t_hat)
    if run("cached"):
        fig_cached_ranges()
    REPORT["seconds"] = time.time() - t0
    print(json.dumps(REPORT, indent=1, default=float))


if __name__ == "__main__":
    main()
