"""Desk case, second round of importance sampling: proposals fitted to the first round's posterior.

    python desk_case_round2.py OUT_DIR [--proposals 1536]

The first round (desk_case.py) drew its proposals around the Gauss–Newton fit, and the posterior turned out longer
than that: with the front-month quotes only it follows the H–rho ridge far beyond the Gauss–Newton ellipse, and the
weights were unreliable (Pareto k-hat 0.64 at the design noise, 1.14 at 1.5 times it). This round fits Student t
components (5 degrees of freedom, in logit space) to the first round's weighted posterior at each noise level,
scale 1.5x and, for the wider level, also 2.5x; 10% stays on the prior. The two rounds are combined with the
deterministic-mixture weights (each point's density is the sample-size-weighted mixture of both rounds' proposal
densities), and the analysis of desk_analysis.py is rerun on all the points.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import desk_analysis as da  # noqa: E402
import desk_case as dc  # noqa: E402
import rough_spy_is as base  # noqa: E402
from sbi4finance.calibration import correction  # noqa: E402


def sample(ts, fractions, n, seed):
    rng = np.random.default_rng(np.random.SeedSequence([31, seed]))
    comp = rng.choice(len(fractions), size=n, p=fractions)
    z = np.empty((n, 3))
    for j, t in enumerate(ts):
        m = comp == j
        if m.any():
            z[m] = np.atleast_2d(t.rvs(size=int(m.sum()), random_state=rng))
    m = comp == len(ts)
    z[m] = correction.to_unbounded(rng.uniform(base.BOUNDS[:, 0], base.BOUNDS[:, 1], (int(m.sum()), 3)), base.BOUNDS)
    return correction.to_bounded(np.clip(z, -30, 30), base.BOUNDS), comp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--proposals", type=int, default=1536)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    out = Path(args.out_dir)
    started = time.time()
    fit = da.fits(out)
    r1 = np.load(out / "desk_case.npz")
    full = np.load(HERE.parent / "rough_spy_is" / "rough_spy_is_seed0.npz")
    ts1, fr1 = da.round1_components(fit)
    w = {k: da.posterior_weights(r1["prices"], r1["log_q"], k)[0]["weights"] for k in da.SCALES}
    ts2 = [da.weighted_t(r1["theta"], w[1.0], 1.5), da.weighted_t(r1["theta"], w[1.5], 1.5),
           da.weighted_t(r1["theta"], w[1.5], 2.5)]
    fr2 = [0.3, 0.3, 0.3, 0.1]
    theta2, comp2 = sample(ts2, fr2, args.proposals, 2)
    with ProcessPoolExecutor(args.workers) as pool:
        priced = list(pool.map(base.surface_job, [(row, dc.SEED) for row in theta2], chunksize=8))
        print(f"[round 2] priced {len(theta2)} proposals on 63 quotes ({time.time() - started:.0f} s)", flush=True)
        prod2 = np.stack(list(pool.map(dc.product_job, [(row, dc.SEED) for row in theta2], chunksize=8)))
        print(f"[round 2] priced products ({time.time() - started:.0f} s)", flush=True)
    prices2 = np.stack([p for p, _ in priced])
    se2 = np.stack([s for _, s in priced])
    theta = np.concatenate([r1["theta"], theta2])
    n1, n2 = len(r1["theta"]), len(theta2)
    log_q1, log_q2 = da.mixture_logpdf(ts1, fr1, theta), da.mixture_logpdf(ts2, fr2, theta)
    log_q = np.logaddexp(np.log(n1) + log_q1, np.log(n2) + log_q2) - np.log(n1 + n2)
    prices = np.concatenate([r1["prices"], prices2])
    prod = np.concatenate([r1["products"], prod2])
    report = da.analyse(theta, log_q, prices, prod, r1["full_products"], full["w_post"], fit)
    report.update(rounds={"round_1": n1, "round_2": n2}, round_2_components=np.bincount(comp2, minlength=4).tolist(),
                  max_se_over_sigma_front=float(np.max(se2[:, dc.FRONT] / base.SIGMA[dc.FRONT])),
                  wall_seconds=time.time() - started)
    np.savez_compressed(out / "desk_case_combined.npz", theta=theta, log_q=log_q, prices=prices, products=prod,
                        round=np.r_[np.zeros(n1, int), np.ones(n2, int)], se_round2=se2)
    (out / "desk_case_combined.json").write_text(json.dumps(report, indent=1))
    print(da.summary(report), flush=True)


if __name__ == "__main__":
    main()
