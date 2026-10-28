"""SPY under rough Bergomi with the liquid network (study S8): posterior, audit, predictive check, products.

    python lab/studies/real_smile.py --checkpoint S2_RUN \
        --surface tutorial/assets/results/calibration/spy/surface.json --out NEW_DIR \
        [--draws 1000 --proposals 1024 --paths 32768 --product-draws 400 --product-paths 16384 \
         --predictive-draws 200 --workers 16 --seed 0]

The surface is SPY's design grid of 21 October 2026 (scripts/build_surface.py): 63 prices in forward units, their
noise sds and the seven forward-variance levels replicated from the strip. The liquid network is given the levels
as known, as in training, so every result is conditional on the replicated curve. Steps:

1. The network's draws of (H, eta, rho).
2. The audit: a Student t fitted to the draws (10% prior), each proposal priced on its own seed at --paths, the
   unbiased likelihood N(y; P_hat, diag(noise^2) - S) (correction.gaussian_loglik_mc; real quotes, so no bank
   error term), self-normalized weights with ESS and k-hat (audit.price_proposal does the pricing).
3. The posterior predictive check: --predictive-draws resampled draws priced on their own seeds, replicated
   quotes with the quote noise, the share of the 63 quotes inside their 90% predictive band, standardized
   residuals by maturity, and the chi-square of the quotes at the posterior's best draw.
4. The design document's products (sbi4finance.calibration.products) under the audited posterior and under the
   network alone, common random numbers, with the curve held flat beyond one year (products.extend_curve).

The grid values are smoothing-spline readings, not independent quotes (lab/studies/README.md); the
likelihood treats them as independent with the design noise, so the predictive check is the test of that model.
Writes real-smile.json, draws.npz and predictive.png.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sbi4finance.calibration import correction, inference, products, tasks  # noqa: E402
from audit import price_proposal  # noqa: E402
from evaluate_calibration import load_network  # noqa: E402
from heston_check import jsonable  # noqa: E402

_TAG = 0x53385350  # "S8SP"


def seed_of(seed, purpose, j):
    return int(np.random.SeedSequence([_TAG, int(seed), int(purpose), int(j)]).generate_state(1, dtype=np.uint32)[0])


def price_job(job):
    task, row, curve, paths, seed = job
    return price_proposal(row, seed, task, curve, task["context"]["contracts"], paths)


def product_job(job):
    task, row, curve, spec, paths, seed = job
    return products.price_draw("rough", tasks.model_parameters(task, row), curve, spec, paths, seed)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--proposals", type=int, default=1024)
    parser.add_argument("--paths", type=int, default=32768)
    parser.add_argument("--product-draws", type=int, default=400)
    parser.add_argument("--product-paths", type=int, default=16384)
    parser.add_argument("--predictive-draws", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    task, workflow, record = load_network(args.checkpoint)
    if task["model"] != "rough":
        raise ValueError("S8 is the rough-Bergomi network on SPY")
    surface = json.loads(args.surface.read_text())
    y = np.asarray(surface["prices_forward_units"], dtype=float)
    noise = np.asarray(surface["noise_sd"], dtype=float)
    levels = np.asarray(surface["forward_variance_levels"], dtype=float)
    sigma = tasks.sigma(task)
    if y.shape != sigma.shape or levels.shape != (tasks.context_size(task),):
        raise ValueError("the surface does not match the task's design grid")
    if np.any(noise < sigma * (1 - 1e-9)):
        raise ValueError("the surface's noise is below the design noise the network was trained with")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    curve = tasks.curve(task, levels)
    network = inference.sample(workflow, task, y[None, :, None], levels[None, :], args.draws, seed=args.seed)[0]

    pool = ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"))
    theta, log_q, from_prior = correction.propose(network, task, args.proposals, seed_of(args.seed, 1, 0))
    priced = list(pool.map(price_job, [(task, row, curve, args.paths, seed_of(args.seed, 2, j))
                                       for j, row in enumerate(theta)], chunksize=4))
    log_lik = np.full(len(theta), -np.inf)
    unpriced = []
    for j, (p, cov, info) in enumerate(priced):
        if p is None:
            unpriced.append(info)
        else:
            log_lik[j] = correction.gaussian_loglik_mc(y, p, noise, cov)
    result = correction.correct(log_lik, log_q, task)
    summary = correction.weighted_summary(theta, result["weights"])
    names = task["parameter_names"]
    posterior = {n: {"mean": float(summary["mean"][i]), "sd": float(summary["sd"][i]),
                     "q05_q50_q95": summary["quantiles"][i].tolist()} for i, n in enumerate(names)}
    network_summary = {n: {"mean": float(network[:, i].mean()), "sd": float(network[:, i].std()),
                           "q05_q50_q95": np.quantile(network[:, i], [0.05, 0.5, 0.95]).tolist()}
                       for i, n in enumerate(names)}

    # the posterior predictive check
    rows = correction.resample(theta, result["weights"], args.predictive_draws, seed_of(args.seed, 3, 0))
    pp = [p for p, _, _ in pool.map(price_job, [(task, row, curve, args.paths, seed_of(args.seed, 4, j))
                                                 for j, row in enumerate(rows)], chunksize=4) if p is not None]
    pp = np.array(pp)
    rng = np.random.default_rng(seed_of(args.seed, 5, 0))
    replicated = pp + noise * rng.standard_normal(pp.shape)
    low, high = np.quantile(replicated, [0.05, 0.95], axis=0)
    standardized = (y - replicated.mean(axis=0)) / replicated.std(axis=0)
    best = int(np.argmax(log_lik))
    chi2_best = float(np.sum(((y - priced[best][0]) / noise) ** 2))
    days = np.rint(np.array([c["maturity"] for c in task["context"]["contracts"]]) * 252).astype(int)
    inside = (y >= low) & (y <= high)
    predictive = {"quotes": int(len(y)), "share_inside_90": float(inside.mean()), "chi2_at_best_proposal": chi2_best,
                  "standardized_residuals": standardized.tolist(),
                  "by_maturity": {str(d): {"share_inside_90": float(inside[days == d].mean()),
                                           "max_abs_standardized": float(np.abs(standardized[days == d]).max())}
                                  for d in sorted(set(days))}}

    # products under the posterior and under the network alone, common random numbers
    spec = products.default_specification()
    long_curve = products.extend_curve(curve, spec["put_out_of_grid"]["maturity"])
    product_seed = seed_of(args.seed, 6, 0)
    picks = correction.resample(theta, result["weights"], args.product_draws, seed_of(args.seed, 7, 0))
    net_pick = network[np.random.default_rng(seed_of(args.seed, 8, 0)).choice(len(network), args.product_draws,
                                                                               replace=False)]
    post_prices = np.array(list(pool.map(product_job, [(task, r, long_curve, spec, args.product_paths, product_seed)
                                                       for r in picks], chunksize=4)))
    net_prices = np.array(list(pool.map(product_job, [(task, r, long_curve, spec, args.product_paths, product_seed)
                                                      for r in net_pick], chunksize=4)))
    pool.shutdown()
    product_report = {name: {"posterior_q10_q50_q90": np.quantile(post_prices[:, j], [0.1, 0.5, 0.9]).tolist(),
                             "network_q10_q50_q90": np.quantile(net_prices[:, j], [0.1, 0.5, 0.9]).tolist()}
                      for j, name in enumerate(products.PRODUCTS)}
    report = {"study": "S8", "surface": str(args.surface), "source": surface.get("source"),
              "checkpoint_sha256": hashlib.sha256((args.checkpoint / "model.keras").read_bytes()).hexdigest(),
              "task_fingerprint": record["task_fingerprint"],
              "settings": {k: getattr(args, k) for k in ("draws", "proposals", "paths", "product_draws",
                                                         "product_paths", "predictive_draws", "seed")},
              "audit": {"ess": result["ess"], "efficiency": result["efficiency"], "khat": result["khat"],
                        "max_weight": result["max_weight"], "unpriced_proposals": len(unpriced),
                        "unpriced_reasons": unpriced[:3], "reliable": bool(result["khat"] <= 0.7)},
              "posterior": posterior, "network": network_summary, "posterior_predictive": predictive,
              "products": product_report, "product_specification": spec,
              "wall_seconds": time.perf_counter() - started}
    (args.out / "real-smile.json").write_text(json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.out / "draws.npz", proposals=theta, weights=result["weights"], network=network,
                        posterior_products=post_prices, network_products=net_prices, predictive_prices=pp)
    figure(y, low, high, days, args.out / "predictive.png")
    print(json.dumps(jsonable({"audit": report["audit"], "share_inside_90": predictive["share_inside_90"]}), indent=1))


def figure(y, low, high, days, path):
    from sbi4finance.calibration.plots import SERIES, plt, style

    style()
    maturities = sorted(set(days))
    fig, axes = plt.subplots(1, len(maturities), figsize=(14, 3.2), constrained_layout=True, squeeze=False)
    for ax, d in zip(axes[0], maturities):
        sel = days == d
        x = np.arange(sel.sum())
        ax.fill_between(x, low[sel] / y[sel], high[sel] / y[sel], color=SERIES[1], alpha=0.3, label="90% predictive")
        ax.plot(x, np.ones(sel.sum()), "o", color="0.2", ms=3, label="SPY")
        ax.set_title(f"{d} days", fontsize=9, loc="left")
        ax.set_xlabel("design delta, left to right")
    axes[0][0].set_ylabel("price / SPY's")
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.savefig(path, dpi=160)


if __name__ == "__main__":
    main()
