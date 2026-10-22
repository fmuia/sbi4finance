"""Figures of Lecture 11: why the exact posterior is expensive.

Usage: python scripts/course/lecture11.py   (writes docs/course/images/l11-*.png and prints the numbers quoted in the text)
"""

import json

import numpy as np
from scipy.stats import norm

from common import COLORS, DAYS, DELTAS, GREY, ROOT, plt, save, sf, tidy

REPORT = {}
RESULTS = ROOT / "tutorial_final" / "results" / "calibration"


def load(path):
    return json.loads((RESULTS / path).read_text())


# Figure 11.1 -----------------------------------------------------------------
def fig_costs(costs):
    heston_runs = np.array(costs["heston_likelihood_evaluation_cpu_s"]["runs"])
    rough_runs = np.array(costs["rough_likelihood_evaluation_cpu_s"]["runs"])
    h_eval = costs["heston_likelihood_evaluation_cpu_s"]["median"]
    r_eval = costs["rough_likelihood_evaluation_cpu_s"]["median"]
    n_eval = costs["exact_rough_posterior"]["evaluations_for_400_effective_draws"]
    h_post = costs["exact_heston_posterior"]["cpu_seconds"]
    r_post = costs["exact_rough_posterior"]["cpu_hours"] * 3600
    bank_unit = load("rough-pilot/s1/bank-quality.json")["generation"]["cpu_seconds_per_surface"]
    r_post_bank = n_eval * bank_unit
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax = axes[0]
    ax.bar([0, 1], [h_eval, r_eval], color=[COLORS[0], COLORS[3]], width=0.55, alpha=0.85)
    ax.scatter(np.zeros(len(heston_runs)) + np.linspace(-0.12, 0.12, len(heston_runs)), heston_runs,
               color="0.2", s=8, zorder=3, label="individual runs")
    ax.scatter(np.ones(len(rough_runs)) + np.linspace(-0.08, 0.08, len(rough_runs)), rough_runs,
               color="0.2", s=8, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Heston\n(Fourier integral)", "rough Bergomi\n(Monte Carlo, $2^{15}$ paths)"])
    ax.set_ylabel("CPU-seconds per likelihood evaluation")
    ax.set_title("One evaluation of the likelihood of 63 quotes")
    for x, v in ((0, h_eval), (1, r_eval)):
        ax.text(x, v * 1.6, f"{v:.3g} s", ha="center", fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax = axes[1]
    ax.bar([0, 1, 2], [h_post, r_post, r_post_bank], color=[COLORS[0], COLORS[3], COLORS[3]], width=0.55,
           alpha=0.85)
    ax.bar([2], [r_post_bank], color="none", edgecolor="k", hatch="//", width=0.55, lw=0)
    ax.set_yscale("log")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Heston", "rough Bergomi:\nprior centre, first rung\n(lower bound)",
                        "rough Bergomi:\npilot bank's average\ncost per surface"], fontsize=8.5)
    ax.set_ylabel("CPU-seconds per surface")
    ax.set_title(f"Exact posterior: {n_eval:,.0f} evaluations, 400 effective draws")
    ax.text(0, h_post * 1.6, f"{h_post:.0f} s = {h_post / 60:.1f} min", ha="center", fontsize=9)
    ax.text(1, r_post * 1.6, f"{r_post / 3600:.1f} h", ha="center", fontsize=9)
    ax.text(2, r_post_bank * 1.6, f"{r_post_bank / 3600:.1f} h", ha="center", fontsize=9)
    ax.set_ylim(top=r_post_bank * 12)
    axes[0].set_ylim(top=r_eval * 12)
    tidy(*axes)
    save(fig, "l11-likelihood-and-posterior-cost.png")
    REPORT["costs"] = dict(heston_eval=h_eval, rough_eval=r_eval, ratio=r_eval / h_eval, evaluations=n_eval,
                           heston_post_s=h_post, rough_post_h=r_post / 3600, bank_unit=bank_unit,
                           rough_post_bank_h=r_post_bank / 3600, per_effective_draw=n_eval / 400,
                           heston_run_range=[float(heston_runs.min()), float(heston_runs.max())],
                           rough_run_range=[float(rough_runs.min()), float(rough_runs.max())])


# Figure 11.2 -----------------------------------------------------------------
def fig_mc_noise():
    rng = np.random.default_rng(0)
    C, reps = 63, 200_000
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    out = {}
    bins = np.linspace(-20, 6, 200)
    for color, a in zip((COLORS[0], COLORS[1], COLORS[3]), (0.05, 0.25, 0.5)):
        eps = rng.standard_normal((reps, C))
        zeta = rng.standard_normal((reps, C))
        diff = np.sum(eps * a * zeta - 0.5 * a**2 * zeta**2, axis=1)
        ax.hist(diff, bins=bins, density=True, histtype="step", lw=1.6, color=color,
                label=rf"$s_c/\sigma_c$ = {a}: mean {diff.mean():.2f}, sd {diff.std():.2f}")
        out[a] = dict(mean=float(diff.mean()), sd=float(diff.std()), mean_theory=float(-0.5 * C * a**2),
                      sd_theory=float(np.sqrt(C * (a**2 + 0.5 * a**4))),
                      width_inflation=float(np.sqrt(1 + a**2) - 1))
    ax.set_xlabel(r"error of the log-likelihood, $\hat\ell - \ell$")
    ax.set_ylabel("density")
    ax.set_title("63 quotes priced with independent Monte Carlo noise")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    REPORT["mc_noise"] = out

    # Common random numbers: Black-Scholes priced by Monte Carlo on the design grid, one parameter sigma
    ax = axes[1]
    contracts = sf.design_grid()
    k, T, kind = sf.contract_arrays(contracts)
    sigma_c = np.array(sf.quote_noise(contracts))
    truth = 0.20
    clean = sf.bs_price(k, T, truth, kind)
    y = clean + sigma_c * np.random.default_rng(1).standard_normal(len(clean))
    N = 2**15
    is_put = kind == "put"

    def mc_prices(s, z):
        # terminal law is exact for Black-Scholes; antithetic pairs; one stream per maturity
        out = np.empty(len(k))
        se = np.empty(len(k))
        for t in np.unique(T):
            idx = np.flatnonzero(T == t)
            zz = z[np.searchsorted(np.unique(T), t)]
            X = np.exp(-0.5 * s * s * t + s * np.sqrt(t) * np.r_[zz, -zz])
            for i in idx:
                pay = np.maximum(np.exp(k[i]) - X, 0) if is_put[i] else np.maximum(X - np.exp(k[i]), 0)
                pairs = 0.5 * (pay[:N // 2] + pay[N // 2:])
                out[i] = pairs.mean()
                se[i] = pairs.std(ddof=1) / np.sqrt(len(pairs))
        return out, se

    def loglik(p):
        return -0.5 * np.sum(((y - p) / sigma_c) ** 2)

    grid = np.linspace(0.1965, 0.2035, 71)
    exact = np.array([loglik(sf.bs_price(k, T, s, kind)) for s in grid])
    zfix = np.random.default_rng(2).standard_normal((len(DAYS), N // 2))
    crn, ses = [], []
    for s in grid:
        p, se = mc_prices(s, zfix)
        crn.append(loglik(p))
        ses.append(se)
    crn = np.array(crn)
    rng = np.random.default_rng(3)
    indep = np.array([loglik(mc_prices(s, rng.standard_normal((len(DAYS), N // 2)))[0]) for s in grid])
    ratio = np.array(ses[35]) / sigma_c
    ref = exact.max()
    ax.plot(grid * 100, exact - ref, color="0.2", lw=2, label="exact prices")
    ax.plot(grid * 100, crn - ref, color=COLORS[0], lw=1.5, label="Monte Carlo, common random numbers")
    ax.plot(grid * 100, indep - ref, "o", color=COLORS[3], ms=3, label="Monte Carlo, a new seed at each point")
    ax.set_ylim(-30, 8)
    ax.set_xlabel(r"Black–Scholes volatility $\sigma$, %")
    ax.set_ylabel(r"log-likelihood minus the exact maximum")
    ax.set_title(r"One parameter, 63 design quotes, $2^{15}$ paths per maturity")
    ax.legend(frameon=False, fontsize=8.5, loc="lower center")
    tidy(*axes)
    save(fig, "l11-monte-carlo-likelihood.png")
    imax_e, imax_c = int(np.argmax(exact)), int(np.argmax(crn))
    REPORT["crn"] = dict(paths=N, se_over_sigma_median=float(np.median(ratio)), se_over_sigma_max=float(ratio.max()),
                         argmax_exact=float(grid[imax_e]), argmax_crn=float(grid[imax_c]),
                         crn_shift_at_max=float(crn[imax_c] - exact[imax_e]),
                         indep_sd_around_exact=float(np.std(indep - exact)),
                         crn_minus_exact_sd=float(np.std(crn - exact)))


# Figure 11.3 -----------------------------------------------------------------
def rwm(d, steps, scale, cov_chol, rng):
    x = np.zeros(d)
    prec = np.linalg.inv(cov_chol @ cov_chol.T)
    lp = -0.5 * x @ prec @ x
    out = np.empty(steps)
    acc = 0
    for i in range(steps):
        prop = x + scale * rng.standard_normal(d)
        lq = -0.5 * prop @ prec @ prop
        if np.log(rng.random()) < lq - lp:
            x, lp = prop, lq
            acc += 1
        out[i] = x[0]
    return out, acc / steps


def ess(x):
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf = acf / acf[0]
    tau, t = 1.0, 1
    while t + 1 < n:
        pair = acf[t] + acf[t + 1]
        if pair < 0:
            break
        tau += 2 * pair
        t += 2
    return n / tau


def fig_rwm(costs):
    rng = np.random.default_rng(4)
    dims = [1, 2, 3, 4, 6, 8, 12, 16]
    steps = 60_000
    iso, corr = [], []
    accs = []
    for d in dims:
        chain, acc = rwm(d, steps, 2.38 / np.sqrt(d), np.eye(d), rng)
        iso.append(steps / ess(chain[5000:]) )
        accs.append(acc)
        # correlated target: all pairwise correlations 0.9, isotropic proposal tuned to the smallest scale
        cov = 0.9 * np.ones((d, d)) + 0.1 * np.eye(d)
        L = np.linalg.cholesky(cov)
        small = np.sqrt(np.linalg.eigvalsh(cov).min())
        chain, _ = rwm(d, steps, 2.38 / np.sqrt(d) * small, L, rng)
        corr.append(steps / ess(chain[5000:]) if d > 1 else np.nan)
    iso, corr = np.array(iso), np.array(corr)
    heston_per = costs["exact_heston_posterior"]["evaluations_for_400_effective_draws"] / 400
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    ax.plot(dims, iso, "o-", color=COLORS[0], ms=4, label="independent Gaussian target, optimal step")
    ax.plot(dims, corr, "s-", color=COLORS[3], ms=4, label="correlation 0.9 between all pairs")
    ax.plot(dims, np.array(dims) / 0.3, color=GREY, ls=":", label=r"$d/0.3$ (Roberts et al. 1997)")
    ax.plot([4], [heston_per], "*", color=COLORS[1], ms=13, label=f"Heston chains of study S3: {heston_per:.1f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("number of parameters $d$")
    ax.set_ylabel("likelihood evaluations per effective draw")
    ax.set_title("Random-walk Metropolis slows down with dimension")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[1]
    per_draw_cost = costs["rough_likelihood_evaluation_cpu_s"]["median"]
    ess_target = np.array([100, 200, 400, 1000])
    for color, per, name in ((COLORS[0], iso[dims.index(3)], "3 independent parameters"),
                             (COLORS[1], heston_per, "as the Heston chains"),
                             (COLORS[3], corr[dims.index(3)], "3 parameters, correlation 0.9")):
        ax.plot(ess_target, ess_target * per * per_draw_cost / 3600, "o-", color=color, ms=4, label=name)
    ax.axhline(costs["exact_rough_posterior"]["cpu_hours"], color="0.4", ls="--", lw=1)
    ax.text(110, costs["exact_rough_posterior"]["cpu_hours"] * 1.12, "design document: 17 CPU-hours", fontsize=8.5,
            color="0.3")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("effective draws required")
    ax.set_ylabel("CPU-hours per surface at 5.0 CPU-s per evaluation")
    ax.set_title("What a rough-Bergomi chain would cost")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    tidy(*axes)
    save(fig, "l11-random-walk-cost.png")
    REPORT["rwm"] = dict(dims=dims, iso=[float(v) for v in iso], corr=[float(v) for v in corr],
                         acceptance=[float(a) for a in accs], heston_per_effective_draw=float(heston_per),
                         hours_400_iso3=float(400 * iso[2] * per_draw_cost / 3600),
                         hours_400_corr3=float(400 * corr[2] * per_draw_cost / 3600))


# Figure 11.4 -----------------------------------------------------------------
def fig_pseudo_marginal():
    rng = np.random.default_rng(5)
    steps = 150_000
    sds = np.linspace(0, 3.5, 15)
    rel, accs = [], []

    def chain(sd, steps, seed):
        r = np.random.default_rng(seed)
        x, lhat = 0.0, -0.5 * 0.0 + sd * r.standard_normal() - 0.5 * sd**2
        out = np.empty(steps)
        acc = 0
        for i in range(steps):
            prop = x + 2.38 * r.standard_normal()
            lprop = -0.5 * prop**2 + sd * r.standard_normal() - 0.5 * sd**2
            if np.log(r.random()) < lprop - lhat:
                x, lhat = prop, lprop
                acc += 1
            out[i] = x
        return out, acc / steps

    base, _ = chain(0.0, steps, 10)
    base_ess = ess(base)
    for i, sd in enumerate(sds):
        c, a = chain(sd, steps, 20 + i)
        rel.append(ess(c) / base_ess)
        accs.append(a)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    for color, sd, seed in ((COLORS[0], 0.5, 31), (COLORS[3], 3.0, 32)):
        c, a = chain(sd, 3000, seed)
        ax.plot(c, color=color, lw=0.8, label=rf"log-likelihood noise sd {sd}: acceptance {a:.0%}")
    ax.set_ylim(-4, 5.5)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\theta$")
    ax.set_title("Pseudo-marginal chains on a standard normal target")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=1)
    ax = axes[1]
    ax.plot(sds, rel, "o-", color=COLORS[0], ms=4, label="effective draws per step, relative to exact")
    ax.plot(sds, np.array(accs) / accs[0], "s--", color=COLORS[2], ms=3.5, label="acceptance rate, relative to exact")
    ax.axvline(1.2, color=COLORS[1], ls=":", lw=1.2)
    ax.text(1.25, 1.6, "recommended\nnoise sd 1.2", color=COLORS[1], fontsize=8.5)
    ax.axvline(REPORT["mc_noise"][0.25]["sd"], color=COLORS[3], ls=":", lw=1.2)
    ax.text(REPORT["mc_noise"][0.25]["sd"] + 0.05, 1.6, "63 quotes at\n$s_c = 0.25\\,\\sigma_c$", color=COLORS[3], fontsize=8.5)
    ax.set_yscale("log")
    ax.set_ylim(2e-4, 6)
    ax.set_xlabel("standard deviation of the log-likelihood estimate")
    ax.set_ylabel("relative to the exact likelihood")
    ax.set_title("Noisier likelihoods make chains stick")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    tidy(*axes)
    save(fig, "l11-pseudo-marginal.png")
    REPORT["pseudo_marginal"] = dict(sds=[float(s) for s in sds], rel_ess=[float(r) for r in rel],
                                     acceptance=[float(a) for a in accs])


# Figure 11.5 -----------------------------------------------------------------
def fig_universe(costs):
    exact_h = costs["exact_rough_posterior"]["cpu_hours"]
    bank_unit = load("rough-pilot/s1/bank-quality.json")["generation"]["cpu_seconds_per_surface"]
    exact_bank_h = costs["exact_rough_posterior"]["evaluations_for_400_effective_draws"] * bank_unit / 3600
    net_s = costs["rough_amortized_posterior_1000_draws_cpu_s"]["median"]
    audit_s = 425.35490193349995  # rough-pilot/s5/audit.json, median CPU-seconds per surface
    heston_s = costs["exact_heston_posterior"]["cpu_seconds"]
    train_h = 370.0  # design document, "The simulator and the training bank": 112,000 surfaces at 3.3 CPU-h per 1,000
    n = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    ax.loglog(n, n * exact_h, "o-", color=COLORS[3], ms=4, label="exact rough-Bergomi MCMC, lower bound")
    ax.loglog(n, n * exact_bank_h, "o--", color=COLORS[3], ms=3.5, alpha=0.7,
              label="exact rough-Bergomi MCMC, bank's average surface cost")
    ax.loglog(n, n * (net_s + audit_s) / 3600, "o-", color=COLORS[0], ms=4, label="amortized posterior + audit")
    ax.loglog(n, n * net_s / 3600, "o--", color=COLORS[0], ms=3.5, alpha=0.7, label="amortized posterior alone")
    ax.loglog(n, n * heston_s / 3600, "s-", color=COLORS[2], ms=3.5, label="exact Heston MCMC (Fourier)")
    ax.set_xlabel("smiles calibrated in one morning")
    ax.set_ylabel("CPU-hours each morning")
    ax.set_title("Daily cost across a desk's underlyings")
    ax.set_ylim(5e-4, 1e10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax = axes[1]
    m = np.linspace(0, 60, 601)
    ax.plot(m, m * exact_h, color=COLORS[3], label="exact MCMC, lower bound")
    ax.plot(m, m * exact_bank_h, color=COLORS[3], ls="--", alpha=0.7, label="exact MCMC, bank's average surface cost")
    ax.plot(m, train_h + m * (net_s + audit_s) / 3600, color=COLORS[0],
            label="bank once (training ⟨S2⟩ not included),\nthen network + audit")
    be = train_h / (exact_h - (net_s + audit_s) / 3600)
    be_bank = train_h / (exact_bank_h - (net_s + audit_s) / 3600)
    for x in (be, be_bank):
        ax.axvline(x, color="0.5", ls=":", lw=1)
    ax.text(31, 120, f"break-even: {be_bank:.0f} smiles against the dashed line,\n{be:.0f} against the solid one",
            fontsize=9, color="0.3")
    ax.set_ylim(0, 1500)
    ax.set_xlabel("smiles calibrated in total")
    ax.set_ylabel("cumulative CPU-hours")
    ax.set_title("The training bank pays for itself after 9 to 22 smiles")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    tidy(*axes)
    save(fig, "l11-desk-scaling.png")
    REPORT["universe"] = dict(break_even=float(be), break_even_bank=float(be_bank), exact_bank_h=float(exact_bank_h),
                              hundred_exact_bank_h=float(100 * exact_bank_h), per_smile_amortized_audit_s=float(net_s + audit_s),
                              hundred_exact_h=float(100 * exact_h), hundred_amortized_audit_h=float(100 * (net_s + audit_s) / 3600))


def typical_noise():
    """Log-likelihood noise sd sqrt(C (a^2 + a^4 / 2)) if every quote had the bank's median or 90% error."""
    q = load("rough-pilot/s1/bank-quality.json")["error_quantiles_all_quotes"]
    out = {}
    for key in ("q50", "q90", "max"):
        a = q[key]
        out[key] = dict(a=a, bias=-0.5 * 63 * a**2, sd=float(np.sqrt(63 * (a**2 + 0.5 * a**4))))
    REPORT["typical_noise"] = out


def main():
    costs = load("costs/costs.json")
    typical_noise()
    fig_costs(costs)
    fig_mc_noise()
    fig_rwm(costs)
    fig_pseudo_marginal()
    fig_universe(costs)
    print(json.dumps(REPORT, indent=1, default=float))


if __name__ == "__main__":
    main()
