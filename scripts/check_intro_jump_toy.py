"""Feasibility check for the proposed intro example: hidden jumps and Wald intervals.

    pixi run python scripts/check_intro_jump_toy.py                # full table, about 20 minutes
    pixi run python scripts/check_intro_jump_toy.py --series 20    # quick look

An independent re-implementation of the constant-volatility jump-model study behind Figure 08
of the design document, whose code is not in the repository. It prints the coverage table in
docs/tutorial-intro-and-linkedin-plan.md, section 2.7. It is a check of the proposal, not the
intro lesson.

Daily log returns in percent follow y_t = mu + sd * eps_t + B_t * xi_t, with
B_t ~ Bernoulli(jumps per year / 252) and xi_t ~ N(mean jump, jump sd^2). Every jump day is
hidden among ordinary days. For each simulated 1,000-day series:

- approximate ML maximizes the exact likelihood with L-BFGS-B from three starts, inside bounds
  at the prior mean +/- 6 prior sd, and builds 95% Wald intervals in the unbounded coordinates;
  a Hessian that is not positive definite counts as a miss;
- where requested, the exact posterior of the three jump parameters is evaluated on a grid,
  with the diffusion volatility and the drift at their true values, and gives 95% equal-tailed
  intervals.

The jump priors are slot 5's; the diffusion volatility and drift priors match its long-run
volatility and drift.
"""

import argparse

import numpy as np
from scipy.optimize import minimize

DAYS, YEAR = 1_000, 252
# Unbounded coordinates, in order: log jumps per year, log jump sd (%), mean jump (%),
# log diffusion sd (% per day), drift (% per day). Prior (mean, sd) for each.
NAMES = ("log_jumps_per_year", "log_jump_sd", "mean_jump", "log_sd", "drift")
PRIOR = np.array([
    (np.log(2.5), 0.75),
    (np.log(3.0), 0.45),
    (-1.0, 2.5),
    (np.log(100 * 0.25 / np.sqrt(YEAR)), 0.40),
    (8.0 / YEAR, 10.0 / YEAR),
])
BOUNDS = [(m - 6 * s, m + 6 * s) for m, s in PRIOR]
MAX_DAILY_JUMP_PROBABILITY = 0.5  # as in slot 5's prior


def true_values(jumps_per_year=2.5, jump_sd=3.0, mean_jump=-1.0, vol=0.16, drift=8.0):
    return np.array([np.log(jumps_per_year), np.log(jump_sd), mean_jump,
                     np.log(100 * vol / np.sqrt(YEAR)), drift / YEAR])


def simulate(theta, rng):
    jumps = rng.random(DAYS) < np.exp(theta[0]) / YEAR
    diffusion = np.exp(theta[3]) * rng.standard_normal(DAYS)
    return theta[4] + diffusion + jumps * rng.normal(theta[2], np.exp(theta[1]), DAYS)


def loglik(y, log_lam, log_jump_sd, mean_jump, log_sd, drift):
    """Exact log-likelihood, vectorized over broadcastable parameter arrays; y has shape (days,)."""
    lam, mean_jump, drift = (np.asarray(a)[..., None] for a in (np.exp(log_lam) / YEAR, mean_jump, drift))
    v1 = np.exp(2 * np.asarray(log_sd))[..., None]
    v2 = v1 + np.exp(2 * np.asarray(log_jump_sd))[..., None]
    no_jump = np.log1p(-lam) - 0.5 * np.log(2 * np.pi * v1) - 0.5 * (y - drift) ** 2 / v1
    jump = np.log(lam) - 0.5 * np.log(2 * np.pi * v2) - 0.5 * (y - drift - mean_jump) ** 2 / v2
    return np.logaddexp(no_jump, jump).sum(-1)


def hessian(f, x, h=1e-3):
    k = len(x)
    steps = np.eye(k) * h
    H = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            H[i, j] = (f(x + steps[i] + steps[j]) - f(x + steps[i] - steps[j])
                       - f(x - steps[i] + steps[j]) + f(x - steps[i] - steps[j])) / (4 * h * h)
    return H


def wald(y, theta, free):
    """Bounded MLE over the coordinates in `free` (others at `theta`); 95% Wald intervals or None."""
    def full(x):
        z = theta.copy()
        z[list(free)] = x
        return z

    nll = lambda x: -float(loglik(y, *full(x)))
    sample = np.array([0.0, 0.0, 0.0, np.log(y.std()), y.mean()])
    starts = [np.r_[PRIOR[:3, 0], sample[3:]], np.r_[np.log(0.5), np.log(6.0), -3.0, sample[3:]],
              np.r_[np.log(20.0), np.log(1.0), 0.0, sample[3:]]]
    bounds = [BOUNDS[i] for i in free]
    best = min((minimize(nll, np.clip(s[list(free)], *np.array(bounds).T), method="L-BFGS-B", bounds=bounds)
                for s in starts), key=lambda r: r.fun)
    on_bound = any(min(abs(v - lo), abs(v - hi)) < 1e-3 for v, (lo, hi) in zip(best.x, bounds))
    H = hessian(nll, best.x)
    try:
        if not (np.all(np.isfinite(H)) and np.all(np.linalg.eigvalsh(H) > 0)):
            return on_bound, None
        se = np.sqrt(np.diag(np.linalg.inv(H)))
    except np.linalg.LinAlgError:
        return on_bound, None
    return on_bound, np.stack([best.x - 1.96 * se, best.x + 1.96 * se], axis=1)


def posterior_intervals(y, theta, jump_free):
    """Exact grid posterior of the free jump coordinates (0, 1 and optionally 2); 95% intervals of 0 and 1."""
    axes = [np.linspace(PRIOR[0, 0] - 4.5 * PRIOR[0, 1], PRIOR[0, 0] + 4.5 * PRIOR[0, 1], 61),
            np.linspace(PRIOR[1, 0] - 4.5 * PRIOR[1, 1], PRIOR[1, 0] + 4.5 * PRIOR[1, 1], 41),
            np.linspace(PRIOR[2, 0] - 4.5 * PRIOR[2, 1], PRIOR[2, 0] + 4.5 * PRIOR[2, 1], 31)
            if 2 in jump_free else np.array([theta[2]])]
    L, S, M = np.meshgrid(*axes, indexing="ij")
    log_post = np.stack([loglik(y, L[i], S[i], M[i], theta[3], theta[4]) for i in range(len(axes[0]))])
    for j, Z in enumerate((L, S, M)):
        if j in jump_free:
            log_post = log_post - 0.5 * ((Z - PRIOR[j, 0]) / PRIOR[j, 1]) ** 2
    w = np.exp(log_post - log_post.max())
    w /= w.sum()
    marginals = (w.sum(axis=(1, 2)), w.sum(axis=(0, 2)))
    # each grid point carries its mass at its centre, so the CDF there counts half of it
    return [np.interp([0.025, 0.975], np.cumsum(m) - 0.5 * m, a) for m, a in zip(marginals, axes)]


def draw_jump_truth(rng, base):
    theta = base.copy()
    while True:
        theta[0] = rng.normal(*PRIOR[0])
        if np.exp(theta[0]) / YEAR < MAX_DAILY_JUMP_PROBABILITY:
            break
    theta[1] = rng.normal(*PRIOR[1])
    theta[2] = rng.normal(*PRIOR[2])
    return theta


def study(label, free, n_series, *, vol=0.16, from_prior=False, bayes=False, seed=0):
    rng = np.random.default_rng(seed)
    base = true_values(vol=vol)
    tallies = {k: [] for k in ("bound", "pd", "wald_lam", "wald_sd", "bayes_lam", "bayes_sd")}
    for _ in range(n_series):
        theta = draw_jump_truth(rng, base) if from_prior else base
        y = simulate(theta, rng)
        on_bound, intervals = wald(y, theta, free)
        tallies["bound"].append(on_bound)
        tallies["pd"].append(intervals is not None)
        for key, j in (("wald_lam", 0), ("wald_sd", 1)):
            tallies[key].append(intervals is not None and intervals[j, 0] <= theta[j] <= intervals[j, 1])
        if bayes:
            for key, j, (lo, hi) in zip(("bayes_lam", "bayes_sd"), (0, 1),
                                         posterior_intervals(y, theta, [i for i in free if i < 3])):
                tallies[key].append(lo <= theta[j] <= hi)
    rate = {k: np.mean(v) if v else np.nan for k, v in tallies.items()}
    exact = f"{rate['bayes_lam']:.0%} / {rate['bayes_sd']:.0%}" if bayes else "not run"
    print(f"| {label} | {rate['bound']:.0%} | {1 - rate['pd']:.0%} | "
          f"{rate['wald_lam']:.0%} / {rate['wald_sd']:.0%} | {exact} |", flush=True)


SETTINGS = {
    "2": ("2 free (jump frequency, jump sd), truth at the prior median", (0, 1), {"bayes": True}),
    "3": ("3 free (+ mean jump), truth at the prior median", (0, 1, 2), {"bayes": True}),
    "3p": ("3 free, truths drawn from the prior", (0, 1, 2), {"bayes": True, "from_prior": True}),
    "4": ("4 free (+ diffusion vol), truth at the prior median", (0, 1, 2, 3), {}),
    "5": ("5 free (+ drift), truth at the prior median", (0, 1, 2, 3, 4), {}),
    "5v": ("5 free, 25% diffusion volatility", (0, 1, 2, 3, 4), {"vol": 0.25}),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--series", type=int, default=200, help="simulated series per setting")
    parser.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=list(SETTINGS))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(f"{args.series} series per setting; truth 2.5 jumps a year, jump sd 3%, mean jump -1%, "
          "diffusion vol 16% unless stated, drift 8% a year\n")
    print("| Setting | MLE on a bound | Hessian not PD | Wald 95% coverage, jump frequency / jump sd "
          "| Exact posterior, same |")
    print("|---|---|---|---|---|")
    for key in args.settings:  # each setting keeps its own seed, so a subset reproduces the same rows
        label, free, options = SETTINGS[key]
        study(label, free, args.series, seed=args.seed + list(SETTINGS).index(key), **options)


if __name__ == "__main__":
    main()
