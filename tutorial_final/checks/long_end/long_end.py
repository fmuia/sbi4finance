"""The long end of SPY's surface: how much of a long-dated exotic's uncertainty comes from inputs the quotes do not pin.

    python long_end.py OUT.json

Model: rough Bergomi at SPY's least-squares fit (H 0.077, eta 2.68, rho -0.77; rough_checks.json), with a
forward-variance curve xi0 on 11 intervals: the 7 design intervals to one year and 4 long intervals
[1, 1.5], [1.5, 2.25], [2.25, 3], [3, 5] years.
Data (all SPY, 21 Oct 2026):
  the 63 design quotes (Jacobian in (H, eta, rho, log xi_1..7) from rough_checks.json, design noise);
  the 7 short xi0 levels from replication, each with a 5% sd in log;
  replicated variance-swap vols of the four covered LEAP expirations (1.20, 1.30, 1.71, 2.23 years),
  each with a stated sd in vol points (VS_SD).
Prior on the long xi0 levels: a random walk in log xi from the one-year level, sd S_PRIOR per sqrt(year)
between interval midpoints (flat extrapolation is its mean; the desk's practice is its zero-width limit).
Posterior: Gaussian (Laplace) in (H, eta, rho, log xi_1..11). Products by Monte Carlo (16,384 antithetic paths,
weekly steps to five years, common random numbers): gradients by central differences give the delta-method
ranges; 400 posterior draws give the Monte Carlo ranges.
Ranges compared: desk (H, eta, rho uncertain; xi0 fixed: short replicated, long flat) against the posterior
(everything uncertain), with the posterior's variance split by input group.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from sbi4finance.calibration import pricing, products, tasks  # noqa: E402

HERE = Path(__file__).resolve().parent
ROUGH = json.loads((HERE.parent / "rough_checks.json").read_text())
SPY = json.loads((Path(__file__).resolve().parents[3] / "tutorial_final/results/calibration/spy/surface.json").read_text())
TASK = json.loads((Path(__file__).resolve().parents[3] / "configs/tasks/calibration_rough_pilot.json").read_text())["task"]
SIGMA = tasks.sigma(TASK)
THETA = np.asarray(ROUGH["best"]["theta"])  # H, eta, rho
JAC = np.asarray(ROUGH["jacobian_seed1"])  # [63, 3 + 7]
XI_SHORT = np.asarray(SPY["forward_variance_levels"])
SHORT_KNOTS = list(TASK["model_context"]["forward_variance_knots"])  # 0 ... 1 year
LONG_KNOTS = [1.5, 2.25, 3.0, 5.0]
KNOTS = np.array(SHORT_KNOTS + LONG_KNOTS)
LEAPS = [(e["days"] / 252, e["w_replicated"]) for e in SPY["diagnostics"]["expirations"]
         if e["days"] > 252 and e["replication_covered"]]
VS_SD = 0.5 / 100  # sd of a replicated LEAP variance-swap vol
S_PRIOR = 0.20  # random-walk sd of log xi per sqrt(year) beyond the data
XI_SHORT_SD = 0.05
N_PATHS, STEPS, HORIZON = 16384, 260, 5.0
STEP = np.array([0.005, 0.03, 0.01] + [0.02] * (len(KNOTS) - 1))
NAMES = ["autocallable 3y", "cliquet 3y", "forward-start 2y-3y", "put 5y 80%", "variance swap 3y"]
GROUPS = {"H, η, ρ (from the quotes)": slice(0, 3), "ξ0 to 1 year (replicated)": slice(3, 10),
          "ξ0 1 to 2.25 years (LEAPs)": slice(10, 12), "ξ0 2.25 to 5 years (prior only)": slice(12, 14)}


def overlap(t, lo, hi):
    return np.clip(t - lo, 0.0, hi - lo)


def vs_rows(log_xi):
    """Replicated total variance w(T) = sum_j xi_j overlap_j(T) and its gradient in log xi."""
    xi = np.exp(log_xi)
    rows, w = [], []
    for t, _ in LEAPS:
        o = np.array([overlap(t, KNOTS[j], KNOTS[j + 1]) for j in range(len(xi))])
        w.append(float(o @ xi))
        rows.append(o * xi)
    return np.array(w), np.array(rows)


def prior_precision_long(s_prior=None):
    """Random walk in log xi from the one-year level: increments with variance S^2 * (midpoint spacing)."""
    s_prior = S_PRIOR if s_prior is None else s_prior
    mids = 0.5 * (KNOTS[1:] + KNOTS[:-1])
    n_long = len(LONG_KNOTS)
    first = len(KNOTS) - 1 - n_long  # index of the first long level among the 11
    d = np.zeros((n_long, len(KNOTS) - 1))
    var = np.zeros(n_long)
    for i in range(n_long):
        j = first + i
        d[i, j], d[i, j - 1] = 1.0, -1.0
        var[i] = s_prior**2 * (mids[j] - mids[j - 1])
    return d.T @ np.diag(1 / var) @ d  # precision on all 11 log xi (only long levels and the 1-year link)


def posterior(log_xi0, s_prior=None, vs_sd=None):
    """Laplace posterior in x = (H, eta, rho, log xi_1..11): mean (one Gauss-Newton step from x0) and covariance."""
    vs_sd = VS_SD if vs_sd is None else vs_sd
    n = 3 + len(KNOTS) - 1
    x0 = np.concatenate([THETA, log_xi0])
    info = np.zeros((n, n))
    grad = np.zeros(n)
    jn = JAC / SIGMA[:, None]  # option quotes: residuals are zero at the least-squares fit to first order
    info[:10, :10] += jn.T @ jn
    short = np.zeros((7, n))
    short[:, 3:10] = np.eye(7)
    info += short.T @ short / XI_SHORT_SD**2  # short xi0 at the replicated levels
    w_model, rows = vs_rows(log_xi0)
    t = np.array([t for t, _ in LEAPS])
    w_obs = np.array([w for _, w in LEAPS])
    sd_w = 2 * np.sqrt(w_obs / t) * vs_sd * t  # d w = 2 sigma T d sigma
    jv = np.zeros((len(t), n))
    jv[:, 3:] = rows
    info += jv.T @ np.diag(1 / sd_w**2) @ jv
    grad += jv.T @ ((w_obs - w_model) / sd_w**2)
    prior = np.zeros((n, n))
    prior[3:, 3:] = prior_precision_long(s_prior)
    info += prior
    grad += -prior @ (x0 - np.concatenate([THETA, log_xi0]))  # zero: the prior mean is the start (flat)
    cov = np.linalg.inv(info)
    return x0 + cov @ grad, cov


def ranges(report, s_prior, vs_sd):
    """Delta-method 90% ranges (fractions of price) of the five products, desk and posterior, from a long_end.json."""
    grad = np.asarray(report["gradient"])
    flat = np.log(np.asarray(report["xi_flat"]))
    _, cov = posterior(flat, s_prior, vs_sd)
    jn = JAC[:, :3] / SIGMA[:, None]
    cov_desk = np.zeros_like(cov)
    cov_desk[:3, :3] = np.linalg.inv(jn.T @ jn)
    price = np.asarray(report["price_posterior_mean_inputs"])
    desk = 2 * 1.645 * np.sqrt(np.einsum("pi,ij,pj->p", grad, cov_desk, grad)) / np.asarray(report["price_desk_point"])
    post = 2 * 1.645 * np.sqrt(np.einsum("pi,ij,pj->p", grad, cov, grad)) / price
    groups = {"short end (option quotes)": slice(0, 10), "ξ0 1 to 2.25 years (LEAPs)": slice(10, 12),
              "ξ0 2.25 to 5 years (prior only)": slice(12, 14)}
    parts = np.array([np.einsum("pi,ij,pj->p", grad[:, g], cov[g, g], grad[:, g]) for g in groups.values()])
    return desk, post, dict(zip(groups, parts / parts.sum(axis=0))), cov


def curve_of(log_xi):
    return pricing.step_curve(list(KNOTS), list(np.exp(log_xi)))


def price_products(x, seed=0):
    params = {"hurst": float(x[0]), "eta": float(x[1]), "rho": float(x[2])}
    paths = products.rough_paths(params, curve_of(x[3:]), HORIZON, STEPS, N_PATHS, seed)
    rho = params["rho"]
    idx = [products._index(paths, t) for t in (1.0, 2.0, 3.0)]
    z = np.random.default_rng(seed + 777).standard_normal((N_PATHS // 2, 3))
    z = np.concatenate([z, -z])
    log_s, prev, s = np.zeros(N_PATHS), 0, []
    for k, i in enumerate(idx):
        d_i = paths["I"][:, i] - paths["I"][:, prev]
        d_j = paths["J"][:, i] - paths["J"][:, prev]
        log_s = log_s + rho * d_j - 0.5 * d_i + np.sqrt(np.maximum((1 - rho**2) * d_i, 0)) * z[:, k]
        s.append(np.exp(log_s))
        prev = i
    s1, s2, s3 = s
    coupon = 0.08
    called1, called2, called3 = s1 >= 1, (s1 < 1) & (s2 >= 1), (s1 < 1) & (s2 < 1) & (s3 >= 1)
    alive = (s1 < 1) & (s2 < 1) & (s3 < 1)
    autocall = (called1 * (1 + coupon) + called2 * (1 + 2 * coupon) + called3 * (1 + 3 * coupon)
                + alive * np.where(s3 >= 0.6, 1.0, s3))
    returns = [s1 - 1, s2 / s1 - 1, s3 / s2 - 1]
    cliquet = sum(np.clip(r, 0.0, 0.08) for r in returns)
    pair = lambda v: 0.5 * (v[: N_PATHS // 2] + v[N_PATHS // 2:])
    fwd = products.forward_start_call(paths, rho, 2.0, 3.0, 1.0)[0]
    put = products.vanilla(paths, rho, 0.8, 5.0, "put")[0]
    vs = float(paths["mean_I"][products._index(paths, 3.0)] / 3.0)
    values = np.array([pair(autocall).mean(), pair(cliquet).mean(), fwd, put, vs])
    se = np.array([pair(autocall).std() / np.sqrt(N_PATHS / 2), pair(cliquet).std() / np.sqrt(N_PATHS / 2),
                   np.nan, np.nan, 0.0])
    return values, se


def _price(args):
    return price_products(*args)[0]


def main(out):
    log_xi_flat = np.log(np.concatenate([XI_SHORT, np.full(len(LONG_KNOTS), XI_SHORT[-1])]))
    mean, cov = posterior(log_xi_flat)
    x_desk = np.concatenate([THETA, log_xi_flat])
    # desk: H, eta, rho from the 63 quotes alone (Gauss-Newton), xi0 fixed
    jn = JAC[:, :3] / SIGMA[:, None]
    cov_desk = np.zeros_like(cov)
    cov_desk[:3, :3] = np.linalg.inv(jn.T @ jn)
    with ProcessPoolExecutor(2) as pool:
        jobs = [(x_desk, 0), (mean, 0)]
        for i in range(len(mean)):
            for sign in (1, -1):
                x = mean.copy()
                x[i] += sign * STEP[i]
                jobs.append((x, 0))
        values = list(pool.map(_price, jobs))
        p_desk, p_post = values[0], values[1]
        grad = np.stack([(values[2 + 2 * i] - values[3 + 2 * i]) / (2 * STEP[i]) for i in range(len(mean))], axis=1)
        rng = np.random.default_rng(5)
        draws = rng.multivariate_normal(mean, cov, 400)
        draws[:, 2] = np.clip(draws[:, 2], -0.999, -0.001)
        mc = np.stack(list(pool.map(_price, [(d, 0) for d in draws], chunksize=10)))
    se = price_products(mean, 0)[1]
    var_desk = np.einsum("pi,ij,pj->p", grad, cov_desk, grad)
    var_post = np.einsum("pi,ij,pj->p", grad, cov, grad)
    shares = {g: (np.einsum("pi,ij,pj->p", grad[:, s], cov[s, s], grad[:, s]) / var_post).tolist()
              for g, s in GROUPS.items()}
    q = np.quantile(mc, [0.05, 0.5, 0.95], axis=0)
    report = {"settings": {"vs_sd_vol_points": 100 * VS_SD, "prior_sd_per_sqrt_year": S_PRIOR,
                           "xi_short_sd": XI_SHORT_SD, "paths": N_PATHS, "steps": STEPS, "theta": THETA.tolist(),
                           "knots": KNOTS.tolist(), "leaps": LEAPS},
              "products": NAMES, "price_desk_point": p_desk.tolist(), "price_posterior_mean_inputs": p_post.tolist(),
              "mc_se_at_mean": se.tolist(),
              "range90_desk_delta": (2 * 1.645 * np.sqrt(var_desk)).tolist(),
              "range90_posterior_delta": (2 * 1.645 * np.sqrt(var_post)).tolist(),
              "range90_posterior_mc": (q[2] - q[0]).tolist(), "posterior_mc_quantiles": q.tolist(),
              "variance_shares": shares, "posterior_mean": mean.tolist(),
              "posterior_sd": np.sqrt(np.diag(cov)).tolist(), "xi_flat": np.exp(log_xi_flat).tolist(),
              "xi_posterior_mean": np.exp(mean[3:]).tolist(), "gradient": grad.tolist()}
    Path(out).write_text(json.dumps(report, indent=1))
    print("long xi: flat", np.round(np.exp(log_xi_flat[-4:]), 4), "posterior mean", np.round(np.exp(mean[-4:]), 4),
          "posterior sd (log)", np.round(np.sqrt(np.diag(cov))[-4:], 3))
    for j, n in enumerate(NAMES):
        print(f"{n:22s} desk point {p_desk[j]:.4f}  posterior-input point {p_post[j]:.4f}  90% range: desk "
              f"{100 * 2 * 1.645 * np.sqrt(var_desk[j]) / p_desk[j]:.2f}%  posterior delta "
              f"{100 * 2 * 1.645 * np.sqrt(var_post[j]) / p_post[j]:.2f}%  posterior MC "
              f"{100 * (q[2, j] - q[0, j]) / q[1, j]:.2f}%  | shares "
              + "  ".join(f"{g.split(' (')[0]} {100 * shares[g][j]:.0f}%" for g in GROUPS))


if __name__ == "__main__":
    for arg in sys.argv[2:]:
        k, v = arg.split("=")
        globals()[k] = float(v)
    main(sys.argv[1])
