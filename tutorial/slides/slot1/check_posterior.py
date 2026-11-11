"""An independent check of the running example's posteriors: adaptive Metropolis chains on the same target.

Run from tutorial/, after `export_figures.py` has computed the example:

    python slides/slot1/check_posterior.py [READING ...]        e.g.  "3 months" "1 month"

`illustration.posterior` draws by importance sampling followed by population moves. This script samples the same
posterior, p(theta, c | quotes) with c integrated out, by a different algorithm that shares only the likelihood:
several random-walk Metropolis chains in logit coordinates, started at draws from the prior, whose proposal covariance
is adapted to the chain's own history during the first half and then frozen. It reports the chains' convergence
(split R-hat and effective sample size of the 2-year put) and compares, for the put and for each parameter, the 5%,
50% and 95% points of the two samples in units of the 90% width of the chains' sample.

Nothing here feeds the slides. It needs only the tutorial's own dependencies.
"""

import argparse

import numpy as np

import export_figures as ef
import illustration as il

CHAINS, STEPS = 8, 60000  # per chain; the first half adapts and is dropped


def chains(quotes, seed=7):
    """Adaptive Metropolis draws in logit coordinates, shape (CHAINS, STEPS // 2, 5), and their acceptance rate."""
    rng = np.random.default_rng(seed)
    dim = len(il.NAMES)
    u = np.empty((0, dim))
    while len(u) < CHAINS:  # starts: draws from the prior with a finite likelihood
        trial = rng.logistic(size=(4 * CHAINS, dim))
        value, _ = il.log_posterior(quotes, trial)
        u = np.vstack([u, trial[np.isfinite(value)]])
    u = u[:CHAINS]
    value, _ = il.log_posterior(quotes, u)
    kept = np.empty((CHAINS, STEPS, dim))
    factor = np.stack([0.3 * np.eye(dim)] * CHAINS)
    accepted = 0
    for step in range(STEPS):
        if step < STEPS // 2 and step >= 1000 and step % 500 == 0:  # adapt to each chain's own second half so far
            for chain in range(CHAINS):
                history = kept[chain, step // 2:step]
                covariance = np.cov(history.T) * 2.38**2 / dim + 1e-8 * np.eye(dim)
                factor[chain] = np.linalg.cholesky(covariance)
        proposal = u + np.einsum("cij,cj->ci", factor, rng.standard_normal((CHAINS, dim)))
        new_value, _ = il.log_posterior(quotes, proposal)
        accept = np.log(rng.uniform(size=CHAINS)) < new_value - value
        u, value = np.where(accept[:, None], proposal, u), np.where(accept, new_value, value)
        kept[:, step] = u
        if step >= STEPS // 2:
            accepted += int(accept.sum())
    return kept[:, STEPS // 2:], accepted / (CHAINS * (STEPS - STEPS // 2))


def split_rhat(x):
    """Split R-hat of chains x with shape (chains, draws)."""
    half = x.shape[1] // 2
    parts = np.vstack([x[:, :half], x[:, half:2 * half]])
    within = parts.var(axis=1, ddof=1).mean()
    between = half * parts.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((half - 1) / half * within + between / half) / within))


def effective_sample_size(x):
    """A batch-means effective sample size of chains x with shape (chains, draws)."""
    batches = 40
    size = x.shape[1] // batches
    means = x[:, :batches * size].reshape(x.shape[0], batches, size).mean(axis=2)
    variance_of_mean = means.var(axis=1, ddof=1).mean() / batches
    return float(x.shape[0] * x.var(axis=1, ddof=1).mean() / variance_of_mean)


def check(reach, data):
    quotes = il.readings()[reach]
    u, acceptance = chains(quotes)
    theta = il.to_theta(u[:, ::20])  # thinned: every 20th draw
    put = 100 * il.put_price(theta.reshape(-1, 5)).reshape(theta.shape[:2])
    reference = np.quantile(put, [0.05, 0.5, 0.95])
    width = reference[2] - reference[0]
    sampled = np.quantile(100 * np.asarray(data[f"{ef.key_of(reach)}/put/draws"]), [0.05, 0.5, 0.95])
    print(f"\n{reach} ({quotes.n} quotes): {CHAINS} chains of {STEPS // 2} draws after {STEPS // 2}; acceptance "
          f"{acceptance:.2f}; split R-hat of the put {split_rhat(put):.3f}; effective sample size "
          f"{effective_sample_size(put):.0f}")
    print(f"  2-year put, 5% / 50% / 95%: chains {reference[0]:.2f} / {reference[1]:.2f} / {reference[2]:.2f}; "
          f"slides {sampled[0]:.2f} / {sampled[1]:.2f} / {sampled[2]:.2f}; largest difference "
          f"{np.max(np.abs(sampled - reference)) / width:.1%} of the 90% width")
    draws = np.asarray(data[f"{ef.key_of(reach)}/draws"])[:, :5]
    flat = theta.reshape(-1, 5)
    for j, name in enumerate(il.NAMES):
        a, b = np.quantile(flat[:, j], [0.05, 0.5, 0.95]), np.quantile(draws[:, j], [0.05, 0.5, 0.95])
        print(f"  {name:10s} chains {a[0]:.4f} / {a[1]:.4f} / {a[2]:.4f}; slides {b[0]:.4f} / {b[1]:.4f} / "
              f"{b[2]:.4f}; largest difference {np.max(np.abs(a - b)) / (a[2] - a[0]):.1%} of the 90% width; "
              f"split R-hat {split_rhat(theta[:, :, j]):.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("readings", nargs="*", default=["3 months", "1 month"], help="readings to check")
    args = parser.parse_args()
    data = ef.load()
    for reach in args.readings:
        check(reach, data)


if __name__ == "__main__":
    main()
