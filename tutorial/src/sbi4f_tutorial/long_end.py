"""The long end of SPY's surface: how much of a long-dated exotic's uncertainty comes from inputs the quotes do not pin.

The notebooks use the Gaussian posterior and its delta-method ranges below, from the cached long_end.json
(its product gradients), rough_checks.json and SPY's surface in tutorial/assets/. The products' Monte Carlo
prices and the full study are lab/checks/long_end/long_end.py.

Model: rough Bergomi at SPY's least-squares fit (H 0.077, eta 2.68, rho -0.77; rough_checks.json), with a
forward-variance curve xi0 on 11 intervals: the 7 design intervals to one year and 4 long intervals
[1, 1.5], [1.5, 2.25], [2.25, 3], [3, 5] years.
Data (all SPY, 21 Oct 2026):
  the 63 design quotes (Jacobian in (H, eta, rho, log xi_1..7) from rough_checks.json, design noise);
  the 7 short xi0 levels from replication, each with a 5% sd in log;
  replicated variance-swap vols of the four covered LEAP expirations (1.20, 1.30, 1.71, 2.23 years),
  each with a stated sd in vol points (VS_SD).
Prior on the long xi0 levels: a random walk in log xi from the one-year level, sd S_PRIOR per sqrt(year)
between interval midpoints (flat extrapolation is its mean and its zero-width limit).
Posterior: Gaussian (Laplace) in (H, eta, rho, log xi_1..11). Products by Monte Carlo (16,384 antithetic paths,
weekly steps to five years, common random numbers): gradients by central differences give the delta-method
ranges; 400 posterior draws give the Monte Carlo ranges.
Ranges compared: flat extrapolation, keyed "desk" in the JSON (H, eta, rho uncertain; xi0 fixed: short replicated,
long flat) against the posterior
(everything uncertain), with the posterior's variance split by input group.
"""

import json

import numpy as np

from . import tasks
from .paths import QUICK_CHECKS, RESULTS

ROUGH = json.loads((QUICK_CHECKS / "rough_checks.json").read_text())
SPY = json.loads((RESULTS / "spy" / "surface.json").read_text())
DESIGN = tasks.rough_design()
SIGMA = DESIGN["sigma"]
THETA = np.asarray(ROUGH["best"]["theta"])  # H, eta, rho
JAC = np.asarray(ROUGH["jacobian_seed1"])  # [63, 3 + 7]
XI_SHORT = np.asarray(SPY["forward_variance_levels"])
SHORT_KNOTS = list(DESIGN["knots"])  # 0 ... 1 year
LONG_KNOTS = [1.5, 2.25, 3.0, 5.0]
KNOTS = np.array(SHORT_KNOTS + LONG_KNOTS)
LEAPS = [(e["days"] / 252, e["w_replicated"]) for e in SPY["diagnostics"]["expirations"]
         if e["days"] > 252 and e["replication_covered"]]
VS_SD = 0.5 / 100  # sd of a replicated LEAP variance-swap vol
S_PRIOR = 0.20  # random-walk sd of log xi per sqrt(year) beyond the data
XI_SHORT_SD = 0.05


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
