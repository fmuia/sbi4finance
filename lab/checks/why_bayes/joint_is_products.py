"""The tutorial's forward start and 2-year 90% put under an importance-sampling posterior from joint_is_posterior.py.

Prices a systematic resample of 300 posterior draws (common random numbers, joint_gn_screen.SEED), and
compares weighted quantiles and the short- and long-side reserves at the 90% level with the delta method at the
truth: the price at the truth stands in for the least-squares mark, and the delta-method reserve is 1.2816
standard deviations.

Needs the joint task of PR #48. Run from lab/: python checks/why_bayes/joint_is_products.py IN.npz
"""

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from joint_gn_screen import PRODUCT_PATHS, SEED, TASK, jacobian, mc_products
from joint_is_posterior import weighted_quantiles
from sbi4finance.calibration import pricing, products, tasks

RESAMPLE = 300


def mc_products_flat(theta):
    """The same products with the curve held flat beyond one year at its 6-12 month average (extend_curve's rule)."""
    s, l, k = theta[3:]
    times = np.arange(0, 2 * 252 + 1) / 252
    w = pricing.heston_integrated_variance(np.array([0.5, 1.0]), s, l, k)
    flat = (w[1] - w[0]) / 0.5
    values = np.where(times <= 1.0, pricing.heston_variance_values(times, s, l, k), flat)
    curve = {"times": times.tolist(), "values": values.tolist()}
    prices, _ = products.price_draw("rough", tasks.model_parameters(TASK, theta), curve,
                                    products.default_specification(), PRODUCT_PATHS, SEED)
    return prices[[2, 3]]


def main():
    d = np.load(sys.argv[1])
    draws, w, truth, cov = d["draws"], d["weights"], d["truth"], d["gn_cov"]
    # A systematic resample of RESAMPLE draws by weight; each distinct draw is priced once and weighted by its count.
    positions = (np.arange(RESAMPLE) + np.random.default_rng(0).uniform()) / RESAMPLE
    picked = np.minimum(np.searchsorted(np.cumsum(w), positions), len(w) - 1)
    keep, counts = np.unique(picked, return_counts=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        prices = np.array(list(pool.map(mc_products, draws[keep])))
    wk = counts / counts.sum()
    mark = mc_products(truth)
    g = jacobian(mc_products, truth)
    sd = np.sqrt(np.einsum("ij,jk,ik->i", g, cov, g))
    flat = mc_products_flat(truth)
    g_known = jacobian(mc_products_flat, truth)[:, :3]
    sd_flat = np.sqrt(np.einsum("ij,jk,ik->i", g_known, np.linalg.inv(np.linalg.inv(cov)[:3, :3]), g_known))
    print(f"priced {len(keep)} proposals carrying {w[keep].sum():.4f} of the weight")
    for i, name in enumerate(["forward start 63->126d", "2-year 90% put"]):
        print(f"  {name}, curve held flat beyond 1y at the truth: {flat[i]:.5f} ({100 * (flat[i] / mark[i] - 1):+.2f}%"
              f" against the true curve); its 90% range from (H, eta, rho) alone: +-{100 * 1.6449 * sd_flat[i] / flat[i]:.2f}%")
    for i, name in enumerate(["forward start 63->126d", "2-year 90% put"]):
        q = weighted_quantiles(prices[:, i], wk, [0.05, 0.10, 0.5, 0.90, 0.95])
        short_post, long_post = q[3] - mark[i], mark[i] - q[1]
        short_dm = long_dm = 1.2816 * sd[i]
        print(f"  {name}: mark {mark[i]:.5f}; posterior 5/10/50/90/95% " + " ".join(f"{v:.5f}" for v in q)
              + f"; delta-method 90% {mark[i] - 1.6449 * sd[i]:.5f} {mark[i] + 1.6449 * sd[i]:.5f}")
        print(f"    reserve short: posterior {short_post:.5f} ({100 * short_post / mark[i]:.2f}%), delta {short_dm:.5f}"
              f" ({100 * short_dm / mark[i]:.2f}%) | long: posterior {long_post:.5f} ({100 * long_post / mark[i]:.2f}%),"
              f" delta {long_dm:.5f} ({100 * long_dm / mark[i]:.2f}%)")
    np.savez(sys.argv[1].replace(".npz", "-products.npz"), keep=keep, prices=prices, weights=wk, mark=mark, delta_sd=sd,
             flat=flat, flat_sd=sd_flat, names=np.array(["forward_start_call", "put_2y_90"]))


if __name__ == "__main__":
    main()
