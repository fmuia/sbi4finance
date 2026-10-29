"""Figures of Lecture 13: validating an amortized posterior.

Usage: python course/scripts/lecture13.py   (writes course/images/l13-*.png and prints the numbers quoted in the text)

Most figures read the cached study results in tutorial/assets/results/calibration. Two compute
new numbers: the toy simulations of Figures 13.2 and 13.7, and the posterior predictive check of
Figure 13.10, which runs random-walk chains with the Heston Fourier likelihood (about five
CPU-minutes on two cores).
"""

import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2, multivariate_t, norm

from common import COLORS, DAYS, DELTAS, GREY, ROOT, load_spy, plt, save, tidy
from sbi4finance.calibration import correction, pricing, tasks, validation
from scipy.special import ndtri
from scipy.stats import rankdata


# Rank-normalized split R-hat and bulk ESS, the repository's sbi4finance.svj.pmmh.rhat_ess (Vehtari et al.
# 2021), copied here because that module imports JAX, which the course scripts do not need.
def _rank_normalize(chains):
    ranks = rankdata(chains, method="average").reshape(chains.shape)
    return ndtri((ranks - 3 / 8) / (chains.size + 1 / 4))


def _split(chains):
    half = chains.shape[1] // 2
    return np.concatenate([chains[:, :half], chains[:, half: 2 * half]], axis=0)


def _rhat(chains):
    n = chains.shape[1]
    within = chains.var(axis=1, ddof=1).mean()
    between_over_n = chains.mean(axis=1).var(ddof=1)
    return np.sqrt(((n - 1) / n * within + between_over_n) / within)


def _ess(chains):
    m, n = chains.shape
    centred = chains - chains.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centred, n=2 * n, axis=1)
    acov = np.fft.irfft(spectrum * np.conj(spectrum), axis=1)[:, :n] / n
    within = (acov[:, 0] * n / (n - 1)).mean()
    var_plus = within * (n - 1) / n + chains.mean(axis=1).var(ddof=1)
    rho = 1 - (within - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    pairs = rho[:-1:2] + rho[1::2]
    positive = np.argmax(pairs < 0) if np.any(pairs < 0) else len(pairs)
    pairs = np.minimum.accumulate(pairs[:positive])
    tau = max(-1 + 2 * pairs.sum(), 1 / np.log10(m * n))
    return m * n / tau


def rhat_ess(chains):
    split = _split(np.asarray(chains, dtype=float))
    bulk = _rank_normalize(split)
    folded = _rank_normalize(np.abs(split - np.median(split)))
    return float(max(_rhat(bulk), _rhat(folded))), float(_ess(bulk))


REPORT = {}
RESULTS = ROOT / "tutorial" / "assets" / "results" / "calibration"
HESTON_NAMES = [r"$v_0$", r"$\theta_H$", r"$\sigma_v$", r"$\rho$"]
ROUGH_NAMES = [r"$H$", r"$\eta$", r"$\rho$", r"$\rho\eta$"]


def load(path):
    return json.loads((RESULTS / path).read_text())


# Figure 13.1: prior predictive check, with SPY marked ----------------------------
STAT_LABELS = ["1-month ATM implied volatility", r"1-month ATM skew $\partial\sigma/\partial k$",
               r"skew exponent $b$ in $|\psi(T)| = aT^{b}$", r"1-month ATM curvature $\partial^2\sigma/\partial k^2$"]


def spy_statistics():
    spy = load_spy()
    month = list(DAYS).index(21)
    return [spy["atm_implied_vol"][month], spy["atm_skew"][month], spy["skew_power_law"]["exponent"],
            spy["atm_curvature"][month]]


def fig_prior_predictive():
    real = spy_statistics()
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.8))
    report = {"spy": real}
    for row, (title, path) in enumerate((("rough Bergomi, pilot: 400 held-out surfaces",
                                          "rough-pilot/prior-predictive/prior-predictive.json"),
                                         ("Heston: 2,000 held-out surfaces",
                                          "heston/prior-predictive/prior-predictive.json"))):
        data = load(path)
        clean = np.array(data["clean"]["values"], dtype=float)
        noisy = np.array(data["noisy"]["values"], dtype=float)
        shares = []
        for j in range(4):
            ax = axes[row, j]
            c = clean[:, j][np.isfinite(clean[:, j])]
            n = noisy[:, j][np.isfinite(noisy[:, j])]
            lo, hi = np.quantile(np.r_[c, n], [0.01, 0.99])
            lo, hi = min(lo, real[j]), max(hi, real[j])
            pad = 0.05 * (hi - lo)
            bins = np.linspace(lo - pad, hi + pad, 41)
            ax.hist(c, bins, density=True, color=COLORS[0], alpha=0.45, label="clean model smiles")
            ax.hist(n, bins, density=True, histtype="step", color=COLORS[2], lw=1.2, label="noisy quotes")
            ax.axvline(real[j], color=COLORS[3], lw=2, label="SPY, 21 Oct 2026")
            if j == 2:
                ax.axvline(-0.4, color="k", lw=1, ls=":", label="SPX, about $-0.4$")
            ax.set_xlabel(STAT_LABELS[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{title.split(':')[0]}\ndensity", fontsize=9)
            shares.append(float(np.mean(c <= real[j])))
        axes[row, 0].set_title(title, fontsize=10, loc="left")
        report["rough" if row == 0 else "heston"] = {"share_of_clean_below_spy": shares,
                                                     "clean_q05_q50_q95": [np.quantile(clean[:, j][np.isfinite(clean[:, j])], [0.05, 0.5, 0.95]).tolist() for j in range(4)],
                                                     "noisy_q05_q95": [np.quantile(noisy[:, j][np.isfinite(noisy[:, j])], [0.05, 0.95]).tolist() for j in range(4)]}
    handles, labels = axes[0, 2].get_legend_handles_labels()
    axes[0, 3].legend(handles, labels, frameon=False, fontsize=8, loc="upper right")
    tidy(*axes.ravel())
    save(fig, "l13-prior-predictive.png")
    REPORT["prior_predictive"] = report


# Figure 13.2: what SBC sees, on a model with an exact posterior --------------------
def fig_sbc_shapes():
    rng = np.random.default_rng(1)
    count, samples = 2000, 400
    theta = rng.standard_normal(count)
    y = theta + 0.5 * rng.standard_normal(count)
    mean, sd = 0.8 * y, np.sqrt(0.2)  # exact posterior of theta ~ N(0,1), y ~ N(theta, 0.5^2)
    cases = [("exact posterior", mean, sd), ("overconfident: sd × 0.6", mean, 0.6 * sd),
             ("underconfident: sd × 1.6", mean, 1.6 * sd), ("biased: mean + 0.5 sd", mean + 0.5 * sd, sd),
             ("the prior: ignores the data", 0 * mean, 1.0)]
    fig, axes = plt.subplots(2, 5, figsize=(16, 5.8), sharey="row")
    report = {}
    for j, (name, mu, s) in enumerate(cases):
        draws = (mu[:, None] + s * rng.standard_normal((count, samples)))[:, :, None]
        result = validation.sbc(draws, theta[:, None], seed=j, alpha=0.05)
        unit = np.asarray(result["fractional_ranks"])
        grid, diff = validation.ecdf_difference(unit)
        ax = axes[0, j]
        ax.hist(unit[:, 0], bins=20, range=(0, 1), color=COLORS[0], alpha=0.6)
        ax.axhline(count / 20, color="k", lw=0.8, ls="--")
        ax.set_title(name, fontsize=9.5)
        ax.set_xlabel("fractional rank of the truth", fontsize=8.5)
        ax = axes[1, j]
        band = result["dkw_bound"]
        ax.fill_between([0, 1], -band, band, color=GREY, alpha=0.25, lw=0)
        failed = not result["passed"][0]
        ax.plot(grid, diff[:, 0], color=COLORS[3] if failed else COLORS[0], lw=1.8)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("fractional rank of the truth", fontsize=8.5)
        cover = float(np.mean(np.abs(theta - mu) <= norm.ppf(0.95) * s))
        contraction = 1 - s**2 / 1.0 if np.ndim(s) == 0 else float("nan")
        ax.text(0.03, 0.95, f"largest distance {result['rank_ks'][0]:.3f}\n"
                f"{'outside' if failed else 'inside'} the band\n90% coverage {cover:.0%}\ncontraction {contraction:.2f}",
                transform=ax.transAxes, va="top", fontsize=8)
        report[name] = dict(ks=result["rank_ks"][0], band=band, passed=result["passed"][0], coverage_90=cover,
                            contraction=contraction)
    axes[0, 0].set_ylabel("surfaces per bin")
    axes[1, 0].set_ylabel("ECDF of ranks minus uniform")
    for ax in axes[1]:
        ax.set_ylim(-0.3, 0.46)
    tidy(*axes.ravel())
    save(fig, "l13-sbc-shapes.png")
    REPORT["sbc_toy"] = report


# Figure 13.3: SBC results of the two networks --------------------------------------
def fig_sbc_results():
    heston = load("heston/s2/validation.json")["sbc"]
    rough = load("rough-pilot/s2/validation.json")["sbc"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, data, names, title in ((axes[0], heston, HESTON_NAMES, "Heston network, 2,000 held-out smiles"),
                                   (axes[1], rough, ROUGH_NAMES, "rough-Bergomi pilot network, 250 held-out smiles")):
        ks = np.array(data["rank_ks"])
        colors = [COLORS[0] if p else COLORS[3] for p in data["passed"]]
        ax.bar(range(4), ks, color=colors, width=0.55)
        ax.axhline(data["dkw_bound"], color="k", ls="--", lw=1)
        ax.text(-0.45, data["dkw_bound"] + 0.002, f"DKW band {data['dkw_bound']:.3f}", ha="left", va="bottom", fontsize=8.5)
        for i, v in enumerate(ks):
            ax.text(i, v - 0.003, f"{v:.3f}", ha="center", va="top", fontsize=8.5, color="white")
        ax.set_xticks(range(4))
        ax.set_xticklabels(names)
        ax.set_ylabel("largest distance of the rank ECDF from uniform")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 0.16)
    tidy(*axes)
    save(fig, "l13-sbc-results.png")
    REPORT["sbc_results"] = dict(heston=heston["rank_ks"], heston_band=heston["dkw_bound"], rough=rough["rank_ks"],
                                 rough_band=rough["dkw_bound"], rough_passed=rough["passed"])


# Figure 13.4: coverage and contraction ------------------------------------------
def fig_coverage():
    heston = load("heston/s2/validation.json")
    rough = load("rough-pilot/s2/validation.json")
    levels = ["0.5", "0.8", "0.9", "0.95"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, data, names, title in ((axes[0], heston, HESTON_NAMES, "Heston, 2,000 smiles"),
                                   (axes[1], rough, ROUGH_NAMES, "rough-Bergomi pilot, 250 smiles")):
        for j in range(4):
            cov = np.array([data["coverage"][lv]["coverage"][j] for lv in levels])
            wil = np.array([data["coverage"][lv]["wilson"][j] for lv in levels])
            x = np.arange(4) + (j - 1.5) * 0.14
            nominal = np.array([float(lv) for lv in levels])
            ax.errorbar(x, 100 * (cov - nominal), yerr=100 * np.abs(wil.T - cov), fmt="o", color=COLORS[j], ms=4,
                        capsize=2, lw=1, label=names[j])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"{float(lv):.0%}" for lv in levels])
        ax.set_xlabel("nominal level of the central interval")
        ax.set_ylabel("coverage minus nominal, percentage points")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-16, 10)
        ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower left")
    ax = axes[2]
    hc = [heston["contraction"][k] for k in heston["contraction"]]
    rc = [rough["contraction"][k] for k in ("hurst", "eta", "rho", "rho_eta")]
    ax.bar(np.arange(4) - 0.18, hc, width=0.34, color=COLORS[0], label="Heston: " + ", ".join(HESTON_NAMES))
    ax.bar(np.arange(4) + 0.18, rc, width=0.34, color=COLORS[1], label="rough pilot: " + ", ".join(ROUGH_NAMES))
    for i in range(4):
        ax.text(i - 0.18, hc[i] + 0.005, f"{hc[i]:.2f}", ha="center", fontsize=7.5)
        ax.text(i + 0.18, rc[i] + 0.005, f"{rc[i]:.2f}", ha="center", fontsize=7.5)
    ax.set_ylim(0.85, 1.075)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["1st target", "2nd", "3rd", "4th"])
    ax.set_ylabel("contraction  $1 - \\mathrm{posterior\\ var} / \\mathrm{prior\\ var}$")
    ax.set_title("How much the smile narrows the prior", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    tidy(*axes)
    save(fig, "l13-coverage-contraction.png")
    REPORT["coverage"] = dict(heston={lv: heston["coverage"][lv]["coverage"] for lv in levels},
                              rough={lv: rough["coverage"][lv]["coverage"] for lv in levels},
                              heston_contraction=hc, rough_contraction=rc,
                              heston_recovery=heston.get("recovery_correlation"),
                              rough_recovery=rough["recovery_correlation"])


# Figure 13.5: agreement with the exact Heston posterior ---------------------------
def fig_heston_agreement():
    data = load("heston/s3/heston-check.json")
    surfaces = data["surfaces"]
    err = np.array([s["network_vs_reweighted"]["standardized_mean_error"] for s in surfaces])
    width = np.array([s["network_vs_reweighted"]["width_ratio"] for s in surfaces])
    ok = (err < 0.3) & (np.abs(width - 1) < 0.25)
    draws = np.load(RESULTS / "heston/s3/posterior-draws.npz")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    ax.add_patch(plt.Rectangle((0.75, 0), 0.5, 0.3, color=COLORS[4], alpha=0.12, lw=0))
    for j in range(4):
        ax.scatter(width[:, j], err[:, j], s=16, color=COLORS[j], label=f"{HESTON_NAMES[j]}: {ok[:, j].mean():.0%} within")
    ax.set_xlabel("90% width of the network over the exact 90% width")
    ax.set_ylabel("|mean of network − exact mean| / exact sd")
    ax.set_title("50 test smiles: the network against the exact posterior\n(shaded: the criterion)", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax = axes[1]
    rows = [2000, 2001, 2002, 2003]
    y = 0
    ticks, labels = [], []
    for r in rows:
        chain = draws[f"chain_{r}"]
        m, s = chain.mean(0), chain.std(0)
        for j in range(4):
            for k, (key, color, off) in enumerate((("chain", GREY, 0.25), ("network", COLORS[0], 0.0),
                                                   ("reweighted", COLORS[4], -0.25))):
                z = (draws[f"{key}_{r}"][:, j] - m[j]) / s[j]
                q05, q50, q95 = np.quantile(z, [0.05, 0.5, 0.95])
                ax.plot([q05, q95], [y + off, y + off], color=color, lw=2.2 if key == "chain" else 1.6)
                ax.plot(q50, y + off, "o", color=color, ms=3.5)
            ax.plot((draws[f"truth_{r}"][j] - m[j]) / s[j], y, "*", color=COLORS[3], ms=8)
            ticks.append(y)
            labels.append(f"smile {r}, {HESTON_NAMES[j]}")
            y -= 1
        y -= 0.5
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(-8.2, 3.5)
    ax.set_xticks(range(-4, 4))
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("value in units of the exact posterior: (θ − exact mean) / exact sd")
    ax.plot([], [], color=GREY, lw=2.2, label="exact: 4 random-walk chains")
    ax.plot([], [], color=COLORS[0], lw=1.6, label="network")
    ax.plot([], [], color=COLORS[4], lw=1.6, label="network reweighted by the audit")
    ax.plot([], [], "*", color=COLORS[3], ms=8, label="true value")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("90% intervals and medians on the four smiles with chains", fontsize=10)
    tidy(*axes)
    save(fig, "l13-heston-agreement.png")
    chain_checks = [{k: s[k]["standardized_mean_error"] for k in ("network_vs_chain", "reweighted_vs_chain")}
                    | {"width_" + k: s[k]["width_ratio"] for k in ("network_vs_chain", "reweighted_vs_chain")}
                    for s in surfaces[:4]]
    eff = np.array([s["audit"]["efficiency"] for s in surfaces])
    khat = np.array([s["audit"]["khat"] for s in surfaces])
    misses = [i for i in range(50) if not ok[i].all()]
    REPORT["heston_agreement"] = dict(share_per_target=ok.mean(0).tolist(), share_all=float(ok.all(1).mean()),
                                      misses=len(misses), misses_by_v0_only=int(sum((~ok[i, 0]) for i in misses)),
                                      v0_errors_of_misses=sorted(err[misses, 0].round(3).tolist()),
                                      chain_checks=chain_checks,
                                      reweighted_vs_chain_max_error=float(max(max(c["reweighted_vs_chain"]) for c in chain_checks)),
                                      reweighted_vs_chain_width=[float(min(min(c["width_reweighted_vs_chain"]) for c in chain_checks)),
                                                                 float(max(max(c["width_reweighted_vs_chain"]) for c in chain_checks))],
                                      audit_efficiency_q=[float(eff.min()), float(np.median(eff)), float(eff.max())],
                                      khat_above_0_7=int((khat > 0.7).sum()),
                                      khat_above_0_7_among_misses=int(sum(khat[i] > 0.7 for i in misses)),
                                      truth_z_2001=((draws["truth_2001"] - draws["chain_2001"].mean(0)) / draws["chain_2001"].std(0)).tolist())


# Figure 13.6: agreement with the neural-pricer route -----------------------------
def share(surfaces, key):
    e = np.array([s[key]["standardized_mean_error"] for s in surfaces])
    w = np.array([s[key]["width_ratio"] for s in surfaces])
    ok = (e < 0.3) & (np.abs(w - 1) < 0.25)
    return np.r_[ok.mean(0), ok.all(1).mean()]


def fig_pricer_route():
    s3 = load("heston/s3/heston-check.json")["surfaces"]
    s4 = load("heston/s4/pricer-route.json")["surfaces"]
    s4e = load("heston/s4-exact/pricer-route.json")["surfaces"]
    groups = [("network against the exact posterior\n(S3, rows 2,000 to 2,049)", share(s3, "network_vs_reweighted"), COLORS[0]),
              ("network against the pricer route\n(S4, rows 2,050 to 2,099)", share(s4, "agreement"), COLORS[1]),
              ("pricer route against the exact posterior\n(S4 on the smiles of S3)", share(s4e, "pricer_vs_exact"), COLORS[4])]
    fig, ax = plt.subplots(figsize=(10, 4.3))
    for k, (label, values, color) in enumerate(groups):
        x = np.arange(5) + (k - 1) * 0.27
        ax.bar(x, 100 * values, width=0.25, color=color, label=label.replace("\n", " "))
        for xi, v in zip(x, values):
            ax.text(xi, 100 * v + 1, f"{100 * v:.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(5))
    ax.set_xticklabels(HESTON_NAMES + ["all four"])
    ax.set_ylabel("share of 50 smiles within the criterion, %")
    ax.set_ylim(0, 142)
    ax.set_yticks(range(0, 101, 20))
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=1)
    tidy(ax)
    save(fig, "l13-pricer-route-agreement.png")
    REPORT["pricer_route"] = {g[0].split("\n")[0]: g[1].tolist() for g in groups}


# Figure 13.7: importance sampling on a toy target --------------------------------
def efficiency_theory(delta, lam, dims=1):
    """Limit of ESS / M for target N(0, I) and proposal N(delta e_1, lam^2 I)."""
    lam = np.asarray(lam, dtype=float)
    ok = 2 * lam**2 > 1
    per_dim = np.where(ok, np.sqrt(np.clip(2 * lam**2 - 1, 1e-12, None)) / lam**2, 0.0)
    return np.where(ok, per_dim**dims * np.exp(-np.asarray(delta)**2 / np.clip(2 * lam**2 - 1, 1e-12, None)), 0.0)


def is_run(delta, lam, rng, *, defensive, dims=3, M=256, broad=4.0):
    """One audit on a Gaussian target N(0, I): returns (efficiency, khat)."""
    centre = np.zeros(dims)
    centre[0] = delta
    if defensive:
        t = multivariate_t(loc=centre, shape=(1.2 * lam) ** 2 * np.eye(dims), df=5)
        wide = rng.random(M) < 0.1
        x = np.where(wide[:, None], broad * rng.standard_normal((M, dims)), t.rvs(size=M, random_state=rng))
        log_q = np.logaddexp(np.log(0.9) + t.logpdf(x),
                             np.log(0.1) + norm.logpdf(x, scale=broad).sum(1))
    else:
        x = centre + lam * rng.standard_normal((M, dims))
        log_q = norm.logpdf(x, loc=centre, scale=lam).sum(1)
    log_w = norm.logpdf(x).sum(1) - log_q
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    return 1 / np.sum(w**2) / M, correction.pareto_khat(log_w)


def fig_importance_toy():
    rng = np.random.default_rng(7)
    reps = 60
    deltas = np.linspace(0, 3, 13)
    lams = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0])
    out = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    dd = np.linspace(0, 3, 200)
    ax.semilogy(dd, efficiency_theory(dd, 1.0), color="k", lw=1.2, label=r"theory, same width: $e^{-\delta^2}$")
    for defensive, color, name in ((False, COLORS[0], "Gaussian proposal at the network"),
                                   (True, COLORS[1], "the audit's defensive proposal")):
        res = np.array([[is_run(d, 1.0, rng, defensive=defensive) for _ in range(reps)] for d in deltas])
        ax.semilogy(deltas, np.median(res[:, :, 0], 1), "o", color=color, ms=4, label=name + ", median of 60")
        out[f"offset_{'defensive' if defensive else 'plain'}"] = dict(delta=deltas.tolist(),
                                                                      efficiency=np.median(res[:, :, 0], 1).tolist(),
                                                                      khat=np.median(res[:, :, 1], 1).tolist())
    ax.axhline(0.02, color=COLORS[3], ls=":", lw=1)
    ax.text(0.02, 0.023, "2% flag", ha="left", fontsize=8, color=COLORS[3])
    ax.set_xlabel(r"network's mean error $\delta$, in posterior sd (one of 3 coordinates)")
    ax.set_ylabel("efficiency ESS / M")
    ax.set_ylim(1e-3, 1.5)
    ax.set_title("A shifted network: efficiency falls like $e^{-\\delta^2}$", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    for k, (defensive, color, name) in enumerate(((False, COLORS[0], "Gaussian proposal at the network"),
                                                  (True, COLORS[1], "the audit's defensive proposal"))):
        res = np.array([[is_run(0.0, lam, rng, defensive=defensive) for _ in range(reps)] for lam in lams])
        axes[1].semilogx(lams, np.median(res[:, :, 0], 1), "o-", color=color, ms=4, lw=1, label=name)
        q = np.quantile(res[:, :, 1], [0.1, 0.5, 0.9], axis=1)
        axes[2].semilogx(lams, q[1], "o-", color=color, ms=4, lw=1, label=name + ": median")
        axes[2].fill_between(lams, q[0], q[2], color=color, alpha=0.15, lw=0)
        out[f"width_{'defensive' if defensive else 'plain'}"] = dict(lam=lams.tolist(),
                                                                     efficiency=np.median(res[:, :, 0], 1).tolist(),
                                                                     khat=q[1].tolist())
    ll = np.linspace(0.72, 3, 200)
    axes[1].semilogx(ll, efficiency_theory(0.0, ll, dims=3), color="k", lw=1.2,
                     label=r"theory, Gaussian: $\left(\sqrt{2\lambda^2-1}/\lambda^2\right)^3$")
    axes[1].axvline(1 / np.sqrt(2), color=GREY, ls=":", lw=1)
    axes[1].text(0.505, 1.0, r"left of $\lambda = 1/\sqrt{2}$:" + "\nthe weights have\ninfinite variance", fontsize=7.5,
                 color="0.35", va="top")
    axes[1].set_xlabel(r"network's width over the true width, $\lambda$ (all 3 coordinates)")
    axes[1].set_ylabel("efficiency ESS / M")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("A network of the wrong width", fontsize=10)
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper right")
    ax = axes[2]
    for level in (0.5, khat_threshold(256), 0.7):
        ax.axhline(level, color=COLORS[3], ls=":", lw=1)
    ax.text(0.95, 0.76, "dotted: 0.5, 0.58 (threshold for M = 256), 0.7", ha="left", va="bottom", fontsize=7.5,
            color=COLORS[3])
    ax.set_xlabel(r"network's width over the true width, $\lambda$")
    ax.set_ylabel(r"Pareto $\hat k$ of the weights (10% to 90% shaded)")
    ax.set_title("Too narrow a network gives heavy-tailed weights", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    for a in axes[1:]:
        a.set_xticks([0.5, 0.7, 1, 1.5, 2, 3])
        a.set_xticklabels(["0.5", "0.7", "1", "1.5", "2", "3"])
        a.xaxis.set_minor_formatter(plt.NullFormatter())
    tidy(*axes)
    save(fig, "l13-importance-sampling-toy.png")
    out["theory"] = dict(e_minus_1=float(np.exp(-1)), e_minus_4=float(np.exp(-4)),
                         lam_1_2_d3=float(efficiency_theory(0.0, 1.2, 3)), lam_0_8_d3=float(efficiency_theory(0.0, 0.8, 3)),
                         lam_1_45_d3=float(efficiency_theory(0.0, 1.2 * 1.45, 3)))
    REPORT["importance_toy"] = out


def khat_threshold(m):
    """Sample-size dependent k-hat threshold of Vehtari et al. (2024): min(1 - 1 / log10 M, 0.7)."""
    return float(min(1 - 1 / np.log10(m), 0.7))


# Figure 13.8: the audit on the rough-Bergomi pilot ---------------------------------
def fig_rough_audit():
    data = load("rough-pilot/s5/audit.json")
    surfaces = data["surfaces"]
    eff = np.array([s["efficiency"] for s in surfaces])
    khat = np.array([s["khat"] for s in surfaces])
    err = np.array([s["network_vs_audited"]["standardized_mean_error"] for s in surfaces])
    width = np.array([s["network_vs_audited"]["width_ratio"] for s in surfaces])
    worst = err.max(1)
    heston_khat = np.array([s["audit"]["khat"] for s in load("heston/s3/heston-check.json")["surfaces"]])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    threshold = khat_threshold(256)
    flagged = khat > 0.7
    middle = (khat > threshold) & ~flagged
    ok = khat <= threshold
    ax.semilogy(worst[ok], eff[ok], "o", color=COLORS[0], ms=5, label=rf"$\hat k \leq {threshold:.2f}$")
    ax.semilogy(worst[middle], eff[middle], "o", color=COLORS[1], ms=5, label=rf"${threshold:.2f} < \hat k \leq 0.7$")
    ax.semilogy(worst[flagged], eff[flagged], "o", color=COLORS[3], ms=5, label=r"$\hat k > 0.7$")
    dd = np.linspace(0, 4, 200)
    ax.semilogy(dd, np.exp(-dd**2), "k--", lw=1, label=r"$e^{-\delta^2}$")
    ax.axhline(0.02, color=COLORS[3], ls=":", lw=1)
    ax.set_xlim(0, 6.5)
    ax.set_ylim(5e-4, 1.2)
    n_off = int((worst > 6.5).sum())
    ax.text(6.4, 0.0012, f"{n_off} smile{'s' if n_off != 1 else ''} with δ > 6.5\nnot shown", ha="right", fontsize=7.5, color="0.35")
    ax.set_xlabel(r"largest error of the network's mean, $\delta$ (audited sd)")
    ax.set_ylabel("audit efficiency ESS / M")
    ax.set_title("Rough-Bergomi pilot: 40 held-out smiles, M = 256", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax = axes[1]
    bins = np.linspace(-0.6, 1.6, 23)
    ax.hist(np.clip(heston_khat, -0.6, 1.6), bins, color=COLORS[0], alpha=0.5, label="Heston, 50 smiles, M = 2,048")
    ax.hist(np.clip(khat, -0.6, 1.6), bins, histtype="step", color=COLORS[1], lw=2, label="rough pilot, 40 smiles, M = 256")
    for level in (0.5, threshold, 0.7):
        ax.axvline(level, color=COLORS[3], ls=":", lw=1)
    ax.set_xlabel(r"Pareto $\hat k$ of the audit's weights")
    ax.set_ylabel("smiles")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title(r"How heavy the weight tails are", fontsize=10)
    tidy(*axes)
    save(fig, "l13-rough-audit.png")
    reliable = (eff > 0.05) & (khat < 0.7)
    REPORT["rough_audit"] = dict(surfaces=len(surfaces), efficiency_q05_q50_q95=np.quantile(eff, [0.05, 0.5, 0.95]).tolist(),
                                 flagged_khat=int(flagged.sum()), threshold_256=threshold,
                                 flagged_khat_threshold=int((khat > threshold).sum()),
                                 heston_threshold_2048=khat_threshold(2048),
                                 heston_flagged_threshold=int((heston_khat > khat_threshold(2048)).sum()),
                                 below_2pct=int((eff < 0.02).sum()),
                                 reliable=int(reliable.sum()),
                                 reliable_median_error=np.median(err[reliable], 0).tolist(),
                                 reliable_median_width=np.median(width[reliable], 0).tolist(),
                                 off_scale=n_off, heston_khat_above_0_7=int((heston_khat > 0.7).sum()),
                                 heston_khat_below_0_5=int((heston_khat < 0.5).sum()),
                                 rough_khat_below_0_5=int((khat < 0.5).sum()))


# Figures 13.9 and 13.10: goodness of fit and the posterior predictive check -------
HESTON = tasks.heston_task()
CONTRACTS = HESTON["context"]["contracts"]
SIGMA = tasks.sigma(HESTON)
BOUNDS = correction.bounds_of(HESTON)
EXAMPLE = json.loads((RESULTS / "smile/example-rough.json").read_text())


def heston_prices(theta):
    parameters, v0 = tasks._heston_split(tasks.model_parameters(HESTON, theta))
    try:
        return pricing.heston_surface(parameters, v0, CONTRACTS, nodes=1024, upper_limit=600.0)
    except ValueError:
        return tasks.heston_prices(HESTON, parameters, v0)[0]


def log_target(z, y):
    theta = correction.to_bounded(z[None], BOUNDS)[0]
    try:
        prices = heston_prices(theta)
    except ValueError:
        return -np.inf
    loglik = correction.gaussian_loglik(HESTON, y, prices[None])[0]
    return loglik + correction.log_prior(HESTON) + correction.log_jacobian(z[None], BOUNDS)[0]


def chain_job(args):
    y, start, scale, steps, seed = args
    started = time.process_time()
    chain, acc = correction.random_walk_chain(lambda z: log_target(z, y), start, scale, steps, seed)
    return chain, acc, time.process_time() - started


def heston_posterior_for(y, *, chains=4, steps=3000, burn=500):
    """Exact Heston posterior for quotes y: multi-start least squares, then random-walk chains."""
    def residual(z):
        theta = correction.to_bounded(z[None], BOUNDS)[0]
        try:
            return (y - heston_prices(theta)) / SIGMA
        except ValueError:
            return np.full(len(y), 1e3)
    rng = np.random.default_rng(3)
    started = time.process_time()
    fits = [least_squares(residual, rng.uniform(-2, 2, 4), method="lm", max_nfev=400) for _ in range(6)]
    best = min(fits, key=lambda r: np.sum(r.fun**2))
    ls_cpu = time.process_time() - started
    cov = np.linalg.inv(best.jac.T @ best.jac + 1e-6 * np.eye(4))
    scale = cov * 2.38**2 / 4
    starts = best.x + np.linalg.cholesky(cov) @ rng.standard_normal((4, chains)) * 0.5
    with ProcessPoolExecutor(2) as pool:
        results = list(pool.map(chain_job, [(y, starts[:, i], scale, steps, 100 + i) for i in range(chains)]))
    z = np.stack([r[0][burn:] for r in results])
    theta = correction.to_bounded(z.reshape(-1, 4), BOUNDS).reshape(z.shape)
    diagnostics = [rhat_ess(theta[:, :, d]) for d in range(4)]
    return dict(least_squares=correction.to_bounded(best.x[None], BOUNDS)[0], chi2=float(np.sum(best.fun**2)),
                least_squares_cpu=ls_cpu, draws=theta.reshape(-1, 4), rhat=[d[0] for d in diagnostics],
                ess=[d[1] for d in diagnostics], acceptance=[r[1] for r in results],
                chain_cpu=float(sum(r[2] for r in results)))


def posterior_predictive(y, draws, rng, n=200):
    chosen = draws[rng.choice(len(draws), n, replace=False)]
    prices = np.stack([heston_prices(t) for t in chosen])
    residual = (y - prices) / SIGMA
    replicated = prices + SIGMA * rng.standard_normal(prices.shape)
    t_obs = np.sum(residual**2, axis=1)
    t_rep = np.sum(((replicated - prices) / SIGMA) ** 2, axis=1)
    return dict(median_residual=np.median(residual, axis=0), t_obs=t_obs, t_rep=t_rep,
                p_value=float(np.mean(t_rep >= t_obs)), beyond_two=float(np.mean(np.abs(np.median(residual, 0)) > 2)))


def fig_goodness_of_fit(example_chi2, example_true_chi2):
    classical = load("heston/s6/classical.json")["surfaces"]
    best = np.array([s["best_chi2"] for s in classical])
    on_bound = np.array([s["on_bound"] for s in classical])
    across = load("rough-pilot/s7-across/across-models.json")["surfaces"]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    bins = np.linspace(20, 120, 41)
    ax.hist(best, bins, density=True, color=COLORS[0], alpha=0.45,
            label="Heston smiles fitted by Heston: smallest χ², 200 smiles (S6)")
    xx = np.linspace(20, 120, 400)
    ax.plot(xx, chi2.pdf(xx, 59), color="k", lw=1.2, label=r"$\chi^2$ with $63 - 4 = 59$ degrees of freedom")
    ax.axvline(chi2.ppf(0.99, 59), color=GREY, ls=":", lw=1)
    ax.text(chi2.ppf(0.99, 59) + 0.8, 0.044, "99% point", fontsize=7.5, color="0.35")
    for k, s in enumerate(across):
        h, r = s["heston_least_squares"]["chi2"], s["rough_least_squares"]["chi2"]
        ax.plot([r], [0.003 + 0.004 * k], "v", color=COLORS[1], ms=7, label="rough-Bergomi smiles fitted by rough Bergomi (rows 290–292)" if k == 0 else None)
        if h < 120:
            ax.plot([h], [0.003 + 0.004 * k], "s", color=COLORS[3], ms=6, label="the same smiles fitted by Heston" if k == 0 else None)
        else:
            ax.annotate(f"row {s['test_row']}: Heston χ² = {h:,.0f}", xy=(119, 0.003 + 0.004 * k), xytext=(97, 0.024),
                        fontsize=8, color=COLORS[3], arrowprops=dict(arrowstyle="->", color=COLORS[3]))
    ax.annotate(f"example rough smile fitted by Heston: {example_chi2:.0f}", xy=(119, 0.03), xytext=(92, 0.034),
                fontsize=8, color=COLORS[5], arrowprops=dict(arrowstyle="->", color=COLORS[5]))
    ax.set_xlabel(r"$\chi^2 = \sum_c ((y_c - P_c)/\sigma_c)^2$ at the least-squares point, 63 quotes")
    ax.set_ylabel("density")
    ax.set_xlim(20, 120)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    tidy(ax)
    save(fig, "l13-goodness-of-fit.png")
    REPORT["goodness_of_fit"] = dict(heston_mean=float(best.mean()), heston_median=float(np.median(best)),
                                     heston_q05_q95=np.quantile(best, [0.05, 0.95]).tolist(),
                                     chi2_59_q05_q50_q95=chi2.ppf([0.05, 0.5, 0.95], 59).tolist(),
                                     chi2_59_q99=float(chi2.ppf(0.99, 59)),
                                     share_above_q99=float(np.mean(best > chi2.ppf(0.99, 59))),
                                     on_bound=int(on_bound.sum()),
                                     across=[(s["test_row"], s["rough_least_squares"]["chi2"], s["heston_least_squares"]["chi2"]) for s in across],
                                     example_heston_chi2=example_chi2, example_rough_true_chi2=example_true_chi2)


def residual_map(ax, values, title):
    grid = values.reshape(len(DAYS), len(DELTAS))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-6, vmax=6, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(grid[i, j]) > 3.5 else "k")
    ax.set_xticks(range(len(DELTAS)))
    ax.set_xticklabels([f"{d:g}" for d in DELTAS], fontsize=7.5)
    ax.set_yticks(range(len(DAYS)))
    ax.set_yticklabels([f"{d} d" for d in DAYS], fontsize=7.5)
    ax.set_xlabel(r"standardized moneyness $\delta$", fontsize=8.5)
    ax.set_title(title, fontsize=9.5)
    return im


def fig_posterior_predictive():
    rng = np.random.default_rng(11)
    draws = np.load(RESULTS / "heston/s3/posterior-draws.npz")
    y_heston = draws["observation_2000"]
    correct = posterior_predictive(y_heston, draws["chain_2000"], rng)
    y_rough = np.array(EXAMPLE["noisy_quotes"])
    true_chi2 = float(np.sum(((y_rough - np.array(EXAMPLE["prices"])) / SIGMA) ** 2))
    fitted = heston_posterior_for(y_rough)
    wrong = posterior_predictive(y_rough, fitted["draws"], rng)
    fig = plt.figure(figsize=(16, 4.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.06, 1.05])
    a0, a1, cax, a2 = (fig.add_subplot(gs[i]) for i in range(4))
    residual_map(a0, correct["median_residual"], "Heston smile (test row 2,000) under\nits exact Heston posterior")
    im = residual_map(a1, wrong["median_residual"], "rough-Bergomi example smile under\nits Heston posterior")
    a0.set_ylabel("maturity, business days", fontsize=8.5)
    fig.colorbar(im, cax=cax, label="median standardized residual $(y_c - P_c)/\\sigma_c$")
    for res, color, name in ((correct, COLORS[0], "Heston smile, Heston model"),
                             (wrong, COLORS[3], "rough smile, Heston model")):
        a2.scatter(res["t_rep"], res["t_obs"], s=8, color=color, alpha=0.6,
                   label=f"{name}: p = {res['p_value']:.2f}")
    lim = [20, 240]
    a2.plot(lim, lim, color="k", lw=0.8)
    a2.set_xlim(*lim)
    a2.set_ylim(*lim)
    a2.set_xlabel(r"$T(y^{\mathrm{rep}}, \theta)$: replicated quotes")
    a2.set_ylabel(r"$T(y^{\mathrm{obs}}, \theta)$: observed quotes")
    a2.set_title("Discrepancy $T = \\sum_c ((y_c - P_c(\\theta))/\\sigma_c)^2$\nat 200 posterior draws", fontsize=9.5)
    a2.legend(frameon=False, fontsize=8, loc="upper left")
    tidy(a2)
    save(fig, "l13-posterior-predictive.png")
    REPORT["posterior_predictive"] = dict(
        correct=dict(p_value=correct["p_value"], beyond_two=correct["beyond_two"],
                     max_abs=float(np.abs(correct["median_residual"]).max()),
                     t_obs_median=float(np.median(correct["t_obs"]))),
        wrong=dict(p_value=wrong["p_value"], beyond_two=wrong["beyond_two"],
                   max_abs=float(np.abs(wrong["median_residual"]).max()),
                   t_obs_median=float(np.median(wrong["t_obs"])),
                   residual_grid=np.round(wrong["median_residual"].reshape(7, 9), 2).tolist()),
        heston_fit=dict(least_squares=fitted["least_squares"].tolist(), chi2=fitted["chi2"],
                        least_squares_cpu=fitted["least_squares_cpu"], rhat=fitted["rhat"], ess=fitted["ess"],
                        acceptance=fitted["acceptance"], chain_cpu=fitted["chain_cpu"],
                        posterior_mean=fitted["draws"].mean(0).tolist(), posterior_sd=fitted["draws"].std(0).tolist()),
        example_true_parameters=EXAMPLE["parameters"], example_true_chi2=true_chi2)
    return fitted["chi2"], true_chi2


def main():
    fig_prior_predictive()
    fig_sbc_shapes()
    fig_sbc_results()
    fig_coverage()
    fig_heston_agreement()
    fig_pricer_route()
    fig_importance_toy()
    fig_rough_audit()
    example_chi2, true_chi2 = fig_posterior_predictive()
    fig_goodness_of_fit(example_chi2, true_chi2)
    print(json.dumps(REPORT, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))


if __name__ == "__main__":
    main()
