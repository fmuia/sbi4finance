"""Posterior of the joint-curve rough-Bergomi task at one synthetic smile, by importance sampling.

The likelihood uses the task's Monte Carlo pricer on one fixed seed (common random numbers), so the target is the
posterior of that fixed-seed pricer, the construction of lab/checks/rough_spy_is.py; the quotes are that pricer at
the truth plus the design noise. The prior is the task's uniform box.

The proposal is built in coordinates the quotes pin down: (H, eta, rho, short, xi(0.5), xi(1)), with xi(t) the
curve at t years, in place of (long, decay). In those coordinates the Gauss-Newton normal is a good description; the
map back to (long, decay) is closed-form,

    long = (xi1 * short - xi05**2) / (xi1 + short - 2 * xi05),    decay = -2 log((xi05 - long) / (short - long)),

and its Jacobian enters the proposal density. The proposal is a multivariate t (5 degrees of freedom) at one
Gauss-Newton step from the truth, with twice the Gauss-Newton covariance. Draws with no exponential curve through
their three values, or outside the prior box, get zero weight. Reported: effective sample size, the largest weight,
weighted quantiles of the six parameters and of two curve products beyond the last quote, against the Gauss-Newton
(Laplace) interval in the model's own parameters at the truth.

Needs the joint task of PR #48. Run from lab/:
python checks/why_bayes/joint_is_posterior.py OUT.npz H ETA RHO SHORT LONG DECAY [N_PROPOSALS] [NOISE_SEED]
The recorded case: checks/why_bayes/joint_is_A.npz 0.09 2.6 -0.73 0.02 0.06 0.5 1024 1
"""

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

from joint_gn_screen import HI, LO, NAMES, SIGMA, Z90, curve_products, jacobian, surface

T_PIN = (0.5, 1.0)  # the curve's values here replace (long, decay) in the proposal's coordinates


def to_pinned(theta):
    """(H, eta, rho, short, long, decay) -> (H, eta, rho, short, xi(0.5), xi(1))."""
    s, l, k = theta[3:]
    return np.r_[theta[:4], [l + (s - l) * np.exp(-k * t) for t in T_PIN]]


def from_pinned(u):
    """The inverse map; None where no exponential curve with positive decay passes through the three values."""
    s, a, b = u[3:]
    denominator = b + s - 2 * a
    if denominator == 0:
        return None
    l = (b * s - a * a) / denominator
    ratio = (a - l) / (s - l) if s != l else -1.0
    if not 0 < ratio < 1:
        return None
    return np.r_[u[:4], l, -np.log(ratio) / T_PIN[0]]


def log_abs_det(theta):
    """log |d(pinned)/d(theta)|: the 2 x 2 block of (xi(0.5), xi(1)) against (long, decay)."""
    s, l, k = theta[3:]
    e1, e2 = np.exp(-k * T_PIN[0]), np.exp(-k * T_PIN[1])
    d = (1 - e1) * (-(s - l) * T_PIN[1] * e2) - (-(s - l) * T_PIN[0] * e1) * (1 - e2)
    return np.log(abs(d))


def pinned_jacobian(theta):
    """d(pinned)/d(theta), analytic."""
    s, l, k = theta[3:]
    a = np.eye(6)
    for row, t in zip((4, 5), T_PIN):
        e = np.exp(-k * t)
        a[row, 3:] = [e, 1 - e, -(s - l) * t * e]
    return a


def chi2(theta, y):
    return float(np.sum(((surface(theta) - y) / SIGMA) ** 2))


def weighted_quantiles(values, weights, qs):
    order = np.argsort(values)
    c = np.cumsum(weights[order])
    return np.interp(qs, (c - 0.5 * weights[order]) / c[-1], values[order])


def main():
    out = sys.argv[1]
    truth = np.array([float(v) for v in sys.argv[2:8]])
    n = int(sys.argv[8]) if len(sys.argv) > 8 else 1024
    noise_seed = int(sys.argv[9]) if len(sys.argv) > 9 else 1
    rng = np.random.default_rng(noise_seed)
    y = surface(truth) + SIGMA * rng.standard_normal(len(SIGMA))
    J = jacobian(surface, truth) / SIGMA[:, None]
    cov = np.linalg.inv(J.T @ J)  # Gauss-Newton in the model's parameters, at the truth
    Ju = J @ np.linalg.inv(pinned_jacobian(truth))
    cov_u = np.linalg.inv(Ju.T @ Ju)
    r = (y - surface(truth)) / SIGMA
    centre = to_pinned(truth) + cov_u @ Ju.T @ r  # one Gauss-Newton step towards the least-squares point
    proposal = stats.multivariate_t(loc=centre, shape=2.0 * cov_u, df=5, seed=noise_seed + 1000)
    u = proposal.rvs(size=n)
    thetas = [from_pinned(row) for row in u]
    ok = np.array([t is not None and bool(np.all((t >= LO) & (t <= HI))) for t in thetas])
    draws = np.array([t if t is not None else np.full(6, np.nan) for t in thetas])
    with ProcessPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(chi2, draws[ok], [y] * int(ok.sum())))
    c2 = np.full(n, np.inf)
    c2[ok] = values
    logq = np.full(n, -np.inf)
    logq[ok] = proposal.logpdf(u[ok]) + np.array([log_abs_det(t) for t in draws[ok]])
    logw = np.full(n, -np.inf)
    logw[ok] = -0.5 * c2[ok] - logq[ok]
    w = np.exp(logw - logw[ok].max())
    w /= w.sum()
    ess = 1.0 / np.sum(w**2)
    sd = np.sqrt(np.diag(cov))
    print(f"truth {truth.tolist()}; proposals {n}, valid and inside the prior box {ok.sum()}; ESS {ess:.0f}; "
          f"largest weight {w.max():.3f}; min chi2 {np.min(c2):.1f} on {len(SIGMA)} quotes")
    for j, name in enumerate(NAMES):
        q = weighted_quantiles(draws[ok, j], w[ok], [0.05, 0.5, 0.95])
        g = (truth[j] - Z90 * sd[j], truth[j] + Z90 * sd[j])
        print(f"  {name:15s} truth {truth[j]:.4f} | posterior 5/50/95% {q[0]:.4f} {q[1]:.4f} {q[2]:.4f}"
              f" | Gauss-Newton 90% {g[0]:.4f} {g[1]:.4f} | prior [{LO[j]}, {HI[j]}]")
    cp = np.full((n, 2), np.nan)
    cp[ok] = [curve_products(t) for t in draws[ok]]
    g_curve = jacobian(curve_products, truth, rel=1e-4)
    cp_sd = np.sqrt(np.einsum("ij,jk,ik->i", g_curve, cov, g_curve))
    cp0 = curve_products(truth)
    for i, label in enumerate(["2y variance swap strike", "1y->2y forward variance"]):
        q = weighted_quantiles(cp[ok, i], w[ok], [0.05, 0.10, 0.5, 0.90, 0.95])
        print(f"  {label:24s} truth {cp0[i]:.4f} | posterior 5/10/50/90/95% " + " ".join(f"{v:.4f}" for v in q)
              + f" | Gauss-Newton 90% {cp0[i] - Z90 * cp_sd[i]:.4f} {cp0[i] + Z90 * cp_sd[i]:.4f}")
    m = draws[ok]
    pc = np.cov(m.T, aweights=w[ok])
    pc = pc / np.sqrt(np.outer(np.diag(pc), np.diag(pc)))
    gc = cov / np.outer(sd, sd)
    print("  correlation, posterior | Gauss-Newton: (long, decay) %.2f | %.2f; (eta, rho) %.2f | %.2f" % (
        pc[4, 5], gc[4, 5], pc[1, 2], gc[1, 2]))
    np.savez(out, truth=truth, quotes=y, pinned=u, draws=draws, weights=w, chi2=c2, valid=ok, gn_cov=cov,
             gn_cov_pinned=cov_u, curve_products=cp, names=np.array(NAMES))


if __name__ == "__main__":
    main()
