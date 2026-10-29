"""Synthetic Black–Scholes case: implied-vol marks for one sector of single names, liquid and illiquid.

    python bs_hierarchy.py OUT.json

Truth: log sigma_i ~ N(log 0.30, 0.10^2) across names (sector dispersion of about 3 vol points).
Liquid names: the 63-quote design grid (sigma_ref 30%), design noise (0.3 to 1 vol point per quote).
Illiquid names: one 21-day at-the-money call, noise of 3 vol points (a bid-ask about 6 points wide).
Methods, scored on the illiquid names over repeated sectors:
  per name    least squares on each name's quotes, Gauss–Newton (vega) 90% interval;
  proxy       the mean of the liquid names' marks, +-1.645 times their cross-sectional sd;
  empirical Bayes  log sigma_i ~ N(mu, tau^2) with (mu, tau) at the maximum of the marginal likelihood;
  hierarchical Bayes  the same model with priors on (mu, tau) and their uncertainty integrated out.
Every likelihood is the exact Gaussian quote-noise likelihood; integrals are on grids (no sampling).
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbi4finance.calibration import surface  # noqa: E402

MU_TRUE, TAU_TRUE = np.log(0.30), 0.10
SIGMA_REF = 0.30
LIQUID_CONTRACTS = surface.design_grid(sigma_ref=SIGMA_REF)
LIQ_K, LIQ_T, LIQ_KIND = surface.contract_arrays(LIQUID_CONTRACTS)
LIQ_NOISE = surface.quote_noise(LIQUID_CONTRACTS, sigma_ref=SIGMA_REF)
ILL_T, ILL_K, ILL_VOL_NOISE = 21 / 252, 0.0, 0.03
PRODUCT_T = 0.5  # six-month at-the-money call, quoted for no illiquid name

GRID = np.linspace(np.log(0.05), np.log(1.2), 1200)  # log sigma grid for illiquid names
DG = GRID[1] - GRID[0]
MU_GRID = np.linspace(np.log(0.12), np.log(0.75), 61)
TAU_GRID = np.linspace(0.005, 0.6, 60)
H_MU, H_TAU = [a.ravel() for a in np.meshgrid(MU_GRID, TAU_GRID, indexing="ij")]
LOG_HYPERPRIOR = norm.logpdf(H_TAU, 0, 0.3)  # flat in mu over its grid; half-normal(0.3) on tau
NODE = norm.pdf(GRID[None, :], H_MU[:, None], H_TAU[:, None])  # [hyper nodes, grid]


def bs_call(k, t, s):
    sd = s * np.sqrt(t)
    d1 = (-k + 0.5 * sd**2) / sd
    return norm.cdf(d1) - np.exp(k) * norm.cdf(d1 - sd)


def bs(k, t, s, kind):
    call = bs_call(k, t, s)
    return np.where(kind == "put", call - (1 - np.exp(k)), call)


def vega(k, t, s):
    sd = s * np.sqrt(t)
    return norm.pdf((-k + 0.5 * sd**2) / sd) * np.sqrt(t)


ILL_NOISE = vega(ILL_K, ILL_T, SIGMA_REF) * ILL_VOL_NOISE  # price noise, fixed (not parameter-dependent)
ILL_PRICE_GRID = bs_call(ILL_K, ILL_T, np.exp(GRID))


def liquid_fit(y):
    """Least squares in log sigma, and its Gauss–Newton sd in log sigma (the likelihood is Gaussian there)."""
    f = lambda u: np.sum(((bs(LIQ_K, LIQ_T, np.exp(u), LIQ_KIND) - y) / LIQ_NOISE) ** 2)
    u = minimize_scalar(f, bounds=(np.log(0.03), np.log(2.0)), method="bounded", options={"xatol": 1e-7}).x
    j = vega(LIQ_K, LIQ_T, np.exp(u)) * np.exp(u) / LIQ_NOISE
    return u, 1 / np.sqrt(np.sum(j**2))


def implied_vol(price):
    if not (0 < price < 1):
        return np.nan
    return minimize_scalar(lambda s: (bs_call(ILL_K, ILL_T, s) - price) ** 2, bounds=(1e-3, 3.0), method="bounded",
                           options={"xatol": 1e-9}).x


def grid_summary(density):
    """Mean and 90% interval of sigma from a density on the log-sigma grid."""
    p = density / density.sum()
    cdf = np.cumsum(p)
    s = np.exp(GRID)
    return float(np.sum(p * s)), float(s[np.searchsorted(cdf, 0.05)]), float(s[np.searchsorted(cdf, 0.95)])


def one_sector(n_liquid, n_illiquid, rng, tau=TAU_TRUE, ill_vol_noise=ILL_VOL_NOISE):
    """One sector: true vols, the four methods' marks and 90% intervals for the illiquid names."""
    ill_noise = vega(ILL_K, ILL_T, SIGMA_REF) * ill_vol_noise
    truth = np.exp(rng.normal(MU_TRUE, tau, n_liquid + n_illiquid))
    liq = [liquid_fit(bs(LIQ_K, LIQ_T, s, LIQ_KIND) + LIQ_NOISE * rng.standard_normal(len(LIQ_K)))
           for s in truth[:n_liquid]]
    y_ill = bs_call(ILL_K, ILL_T, truth[n_liquid:]) + ill_noise * rng.standard_normal(n_illiquid)
    lik = np.exp(-0.5 * ((ILL_PRICE_GRID[None, :] - y_ill[:, None]) / ill_noise) ** 2)  # [illiquid, grid]
    marg = NODE @ lik.T * DG  # [hyper, illiquid]: integral of likelihood x N(log sigma; mu, tau)
    log_marg = np.log(np.maximum(marg, 1e-300)).sum(axis=1)
    for u, sd in liq:
        log_marg += norm.logpdf(u, H_MU, np.sqrt(H_TAU**2 + sd**2))
    out = {m: [] for m in ("per name", "proxy", "empirical Bayes", "hierarchical Bayes")}
    # empirical Bayes: (mu, tau) at the maximum of the marginal likelihood
    h_star = int(np.argmax(log_marg))
    # hierarchical Bayes: posterior weights over (mu, tau)
    log_w = log_marg + LOG_HYPERPRIOR
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    liq_marks = np.exp([u for u, _ in liq])
    proxy_mean = liq_marks.mean()
    proxy_sd = liq_marks.std(ddof=1) if n_liquid > 1 else np.nan
    for i in range(n_illiquid):
        s_hat = implied_vol(y_ill[i])
        half = 1.645 * ill_noise / vega(ILL_K, ILL_T, s_hat) if np.isfinite(s_hat) else np.nan
        out["per name"].append((s_hat, s_hat - half, s_hat + half))
        out["proxy"].append((proxy_mean, proxy_mean - 1.645 * proxy_sd, proxy_mean + 1.645 * proxy_sd))
        out["empirical Bayes"].append(grid_summary(lik[i] * NODE[h_star]))
        mix = NODE.T @ (w / np.maximum(marg[:, i], 1e-300))
        out["hierarchical Bayes"].append(grid_summary(lik[i] * mix))
    tau_post = float(np.sum(w * H_TAU))
    return truth[n_liquid:], {m: np.array(v) for m, v in out.items()}, \
        {"tau_eb": float(H_TAU[h_star]), "tau_hb_mean": tau_post, "liquid_marks": liq_marks.tolist(),
         "y_illiquid": y_ill.tolist()}


def score(truths, estimates):
    t = np.concatenate(truths)
    e = np.concatenate(estimates)
    ok = np.isfinite(e[:, 0])
    point, lo, hi = e[ok, 0], e[ok, 1], e[ok, 2]
    price_t, price_e = bs_call(0.0, PRODUCT_T, t[ok]), bs_call(0.0, PRODUCT_T, point)
    return {"rmse_vol_points": float(100 * np.sqrt(np.mean((point - t[ok]) ** 2))),
            "coverage_90": float(np.mean((lo <= t[ok]) & (t[ok] <= hi))),
            "mean_width_vol_points": float(100 * np.mean(hi - lo)),
            "product_rmse_percent": float(100 * np.sqrt(np.mean((price_e / price_t - 1) ** 2))),
            "share_without_estimate": float(1 - ok.mean())}


METHODS = ("per name", "proxy", "empirical Bayes", "hierarchical Bayes")


def run(n_liquid, n_illiquid, sectors, seed, tau=TAU_TRUE, ill_vol_noise=ILL_VOL_NOISE):
    """Scores of the four methods over repeated sectors, the first sector as an example, and tau diagnostics."""
    rng = np.random.default_rng(seed)
    truths, est, extra = [], {m: [] for m in METHODS}, []
    for r in range(sectors):
        t, e, x = one_sector(n_liquid, n_illiquid, rng, tau, ill_vol_noise)
        truths.append(t)
        extra.append(x)
        for m in est:
            est[m].append(e[m])
        if r == 0:
            example = {"truth": t.tolist(), **{m: e[m].tolist() for m in e}, **x}
    return {"replications": sectors, "scores": {m: score(truths, est[m]) for m in est}, "example": example,
            "tau_eb_share_at_lowest_node": float(np.mean([x["tau_eb"] <= TAU_GRID[0] + 1e-12 for x in extra])),
            "tau_eb_median": float(np.median([x["tau_eb"] for x in extra])),
            "tau_hb_mean_median": float(np.median([x["tau_hb_mean"] for x in extra]))}


def main(out):
    report = {"settings": {"mu_true_vol": 0.30, "tau_true_log": TAU_TRUE, "illiquid_vol_noise": ILL_VOL_NOISE,
                           "illiquid_quote": "21-day at-the-money call", "liquid_quotes": len(LIQ_K),
                           "hyperpriors": "mu flat on [log 0.12, log 0.75]; tau half-normal(0.3) on [0.005, 0.6]",
                           "product": "6-month at-the-money call"}, "sectors": {}}
    for label, n_liq, n_ill, reps in (("8 liquid + 32 illiquid", 8, 32, 300), ("2 liquid + 6 illiquid", 2, 6, 600)):
        report["sectors"][label] = run(n_liq, n_ill, reps, 20261023 + n_liq)
        print(f"\n== {label} ({reps} sectors)")
        for m, s in report["sectors"][label]["scores"].items():
            print(f"  {m:20s} RMSE {s['rmse_vol_points']:.2f} vol pts  coverage {100 * s['coverage_90']:.1f}%  "
                  f"width {s['mean_width_vol_points']:.2f}  6m-call RMSE {s['product_rmse_percent']:.1f}%  "
                  f"no estimate {100 * s['share_without_estimate']:.1f}%", flush=True)
        r = report["sectors"][label]
        print(f"  tau: EB median {r['tau_eb_median']:.3f} (at the lowest node in {100 * r['tau_eb_share_at_lowest_node']:.0f}%),"
              f" HB posterior mean median {r['tau_hb_mean_median']:.3f}; truth {TAU_TRUE}")
    Path(out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
