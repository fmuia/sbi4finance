"""Audit the six-target thin network on single chains by importance sampling with the Monte Carlo likelihood.

For each chain of a thin_joint_breach.py sample file: a proposal from the network's draws, 90% a Student t (5
degrees of freedom) fitted in the logit coordinates of the prior box and widened by 1.2, 10% the uniform prior;
each proposal priced on its own paths, the Gaussian likelihood with the layout's noise less the Monte Carlo
variance (Lecture 8, Section 8.9); self-normalized weights, their effective sample size and Pareto k-hat
(correction.pareto_khat). Reported per chain beside the network's: the 5%, 50% and 95% quantiles of the six
targets and of the forward variance from six months to one year, the curve beyond the last quote; and the 10%, 50%
and 90% quantiles of the six thin products under --product-draws draws resampled systematically from the weighted
proposals, priced as thin_joint_breach.py prices them (8,192 paths, seed --seed + 1), beside the network's from that
study's posterior arm (DIR/rows-posterior-*.jsonl) when it has run. Rows already in OUT are skipped.

Needs the joint task of PR #48. Run from lab/:
python checks/why_bayes/thin_joint_audit.py --bank BANK --posterior DIR/posterior-0-100.npz --rows 0 1 2 --out OUT.jsonl
    [--proposals 512 --paths 8192 --product-draws 200 --seed 0]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import expit, logit, logsumexp
from scipy.stats import multivariate_t

from sbi4finance.calibration import bank, correction, pricing, tasks, thin_products

import thin_joint

_TASK = {}
PRODUCT_PATHS = 8192


def _price(args):
    theta, seed, paths = args
    task = _TASK["task"]
    prices, se, _, _ = pricing.rough_surface(tasks.model_parameters(task, theta), tasks.joint_curve(task, theta),
                                             task["context"]["contracts"], task["numerics"]["blocks"], [paths],
                                             thin_joint.noise(task), np.inf, seed, cap=np.inf)
    return prices, se


def _products(args):
    theta, seed = args
    return thin_joint.products(_TASK["task"], theta, PRODUCT_PATHS, seed)


def _init(task):
    _TASK["task"] = task


def audit_chain(task, quotes, draws, proposals, paths, seed, pool):
    box = thin_joint.bounds(task)
    width = box[:, 1] - box[:, 0]
    z = logit(np.clip((draws - box[:, 0]) / width, 1e-9, 1 - 1e-9))
    t = multivariate_t(loc=z.mean(axis=0), shape=np.cov(z.T) * 1.2**2, df=5)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 77]))
    from_prior = rng.random(proposals) < 0.1
    zp = np.clip(np.atleast_2d(t.rvs(size=proposals, random_state=rng)), -30, 30)
    theta = box[:, 0] + width * expit(zp)
    theta[from_prior] = box[:, 0] + width * rng.random((int(from_prior.sum()), 6))
    z_all = logit(np.clip((theta - box[:, 0]) / width, 1e-12, 1 - 1e-12))
    log_jac = np.sum(np.log(width) - np.logaddexp(0.0, z_all) - np.logaddexp(0.0, -z_all), axis=-1)
    log_prior = -np.sum(np.log(width))
    log_q = np.logaddexp(np.log(0.9) + t.logpdf(z_all) - log_jac, np.log(0.1) + log_prior)
    jobs = [(row, seed * 100_003 + i, paths) for i, row in enumerate(theta)]
    priced = list(pool.map(_price, jobs))
    prices = np.array([p for p, _ in priced])
    mc_var = np.array([s for _, s in priced]) ** 2
    variance = np.maximum(thin_joint.noise(task) ** 2 - mc_var, 1e-30)
    log_lik = -0.5 * np.sum((prices - quotes) ** 2 / variance + np.log(2 * np.pi * variance), axis=-1)
    log_w = log_lik + log_prior - log_q
    w = np.exp(log_w - logsumexp(log_w))
    return theta, w, {"ess": float(1 / np.sum(w**2)), "max_weight": float(w.max()),
                      "khat": float(correction.pareto_khat(log_w)),
                      "max_mc_se_over_noise": float(np.sqrt(mc_var.max(axis=0) / thin_joint.noise(task) ** 2).max())}


def weighted_quantiles(values, weights, qs):
    order = np.argsort(values)
    c = np.cumsum(weights[order])
    return np.interp(qs, (c - 0.5 * weights[order]) / c[-1], values[order]).tolist()


def systematic_resample(weights, n, rng):
    positions = (rng.random() + np.arange(n)) / n
    return np.minimum(np.searchsorted(np.cumsum(weights), positions), len(weights) - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proposals", type=int, default=512)
    parser.add_argument("--paths", type=int, default=8192)
    parser.add_argument("--product-draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest, _ = bank.load_bank(args.bank, splits=())
    task = manifest["configuration"]["task"]
    post = np.load(args.posterior)
    index = {int(r): i for i, r in enumerate(post["rows"])}
    network_products = {r["row"]: r["posterior"] for f in args.posterior.parent.glob("rows-posterior-*.jsonl")
                        for r in thin_joint.read_rows(f)}
    done = {r["row"] for r in thin_joint.read_rows(args.out)} if args.out.exists() else set()
    qs = [0.05, 0.5, 0.95]
    names = thin_joint.NAMES + ["forward_variance_6m_1y"]
    with ProcessPoolExecutor(max_workers=4, initializer=_init, initargs=(task,)) as pool:
        for row in args.rows:
            if row in done:
                continue
            i = index[row]
            t0 = time.time()
            theta, w, diagnostics = audit_chain(task, post["quotes"][i], post["draws"][i], args.proposals, args.paths,
                                                args.seed * 1_000_003 + row, pool)
            net = np.column_stack([post["draws"][i], thin_joint.forward_variance(post["draws"][i], 0.5, 1.0)])
            aud = np.column_stack([theta, thin_joint.forward_variance(theta, 0.5, 1.0)])
            truth = np.r_[post["truth"][i], thin_joint.forward_variance(post["truth"][i], 0.5, 1.0)]
            pick = systematic_resample(w, args.product_draws, np.random.default_rng(np.random.SeedSequence([args.seed, 9, row])))
            seed = args.seed + 1
            priced = np.array(list(pool.map(_products, [(theta[k], seed) for k in pick] + [(post["truth"][i], seed)])))
            products = {"names": list(thin_products.NAMES), "distinct_draws": int(len(np.unique(pick))),
                        "truth": priced[-1].tolist(),
                        "audited": {"lower": np.quantile(priced[:-1], 0.1, axis=0).tolist(),
                                    "median": np.median(priced[:-1], axis=0).tolist(),
                                    "upper": np.quantile(priced[:-1], 0.9, axis=0).tolist()}}
            if row in network_products:
                products["network"] = network_products[row]
            record = {"row": row, **diagnostics, "seconds": time.time() - t0, "truth": dict(zip(names, truth.tolist())),
                      "network": {n: np.quantile(net[:, j], qs).tolist() for j, n in enumerate(names)},
                      "audited": {n: weighted_quantiles(aud[:, j], w, qs) for j, n in enumerate(names)},
                      "products": products}
            thin_joint.append_row(args.out, record)
            print(f"row {row}: ESS {diagnostics['ess']:.0f} of {args.proposals}, k-hat {diagnostics['khat']:.2f}, "
                  f"{record['seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
