"""The classical route on the same surfaces (study S6).

    python lab/studies/classical_route.py --bank BANK --checkpoint RUN --out NEW_DIR \
        [--pricer S4_DIR/pricer] [--first 2100 --surfaces 200 --starts 10]

For each test surface: weighted least squares from --starts stratified starts
(the desk's multi-start), the Gauss–Newton 90% interval at the best optimum, the
amortized posterior's 90% interval, and the number of distinct optima within
one unit of chi-square of the best. Coverage of the truth is reported for both
interval types. Heston uses the exact Fourier pricer; rough Bergomi uses the
neural pricer of S4 (deep calibration) when --pricer is given, since a Monte
Carlo objective per iterate is the cost the lesson is about, and otherwise the
Monte Carlo pricer itself at --least-squares-paths paths with a fixed seed
(common random numbers), affordable for a few surfaces. A Tikhonov fit toward the prior centre
is run at four penalties on the first surfaces to show how the answer moves.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sbi4finance.calibration import bank_collection as bank  # noqa: E402
from sbi4finance.calibration import classical, correction, pricer_net, tasks  # noqa: E402
from evaluate_calibration import LABELS, load_network, posterior_draws  # noqa: E402
from heston_check import exact_prices, jsonable  # noqa: E402

PENALTIES = (0.01, 0.1, 1.0, 10.0)
_STATE = {}


def _price_fn(task, pricer_dir, context, mc_paths=None, seed=0):
    if task["model"] == "hkde":
        return lambda theta: exact_prices(task, theta)
    if pricer_dir is None:
        from sbi4finance.calibration import pricing

        curve = tasks.curve(task, context)
        contracts, blocks, sigma = task["context"]["contracts"], task["numerics"]["blocks"], tasks.sigma(task)
        return lambda theta: pricing.rough_surface(tasks.model_parameters(task, theta), curve, contracts, blocks,
                                                   [mc_paths], sigma, np.inf, seed)[0]
    pricer = _STATE.setdefault(str(pricer_dir), pricer_net.Pricer.load(pricer_dir))
    sigma = tasks.sigma(task)
    return lambda theta: pricer(pricer_net.features(task, theta, context))[0] * sigma


def fit_job(job):
    task, pricer_dir, observation, context, starts, tikhonov, mc_paths = job
    price_fn = _price_fn(task, pricer_dir, context, mc_paths)
    started = time.process_time()
    fits = classical.fit(task, observation, price_fn, starts)
    best = min(fits, key=lambda f: f["chi2"])
    result = {"fits": fits, "best": best, "modes": classical.distinct_modes(fits),
              "cpu_seconds": time.process_time() - started}
    if tikhonov:
        bounds = correction.bounds_of(task)
        centre = np.zeros(len(bounds))
        result["tikhonov"] = [{"penalty": lam, "theta": classical.fit(task, observation, price_fn, [best["theta"]],
                                                                      z_reference=centre, penalty=lam)[0]["theta"]}
                              for lam in PENALTIES]
    return result


def landscape(task, observation, price_fn, result, draws, pair, labels, path, points=45):
    """chi-square over a parameter pair at the best values of the others, with optima, ellipse and posterior."""
    from sbi4finance.calibration.plots import BASELINE, INK, SERIES, plt, style

    style()
    best = result["best"]["theta"]
    bounds = correction.bounds_of(task)
    a, b = pair
    span_a = np.quantile(draws[:, a], [0.001, 0.999])
    span_b = np.quantile(draws[:, b], [0.001, 0.999])
    pad_a, pad_b = 0.6 * np.diff(span_a)[0], 0.6 * np.diff(span_b)[0]
    grid_a = np.linspace(max(bounds[a, 0], span_a[0] - pad_a), min(bounds[a, 1], span_a[1] + pad_a), points)
    grid_b = np.linspace(max(bounds[b, 0], span_b[0] - pad_b), min(bounds[b, 1], span_b[1] + pad_b), points)
    sigma = tasks.sigma(task)
    chi2 = np.empty((points, points))
    for i, x in enumerate(grid_a):
        for j, y in enumerate(grid_b):
            theta = best.copy()
            theta[a], theta[b] = x, y
            chi2[j, i] = np.sum(((price_fn(theta) - observation) / sigma) ** 2)
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    levels = result["best"]["chi2"] + np.array([1, 2.3, 4.6, 9.2, 20, 50, 100])
    contours = ax.contour(grid_a, grid_b, chi2, levels=levels, colors=BASELINE, linewidths=0.8)
    ax.clabel(contours, fmt=lambda v: f"+{v - result['best']['chi2']:.0f}", fontsize=6.5)
    ax.scatter(draws[:, a], draws[:, b], s=3, color=SERIES[0], alpha=0.25, label="posterior draws")
    for k, mode in enumerate(result["fits"]):
        ax.scatter(mode["theta"][a], mode["theta"][b], s=28, color=SERIES[1], edgecolor="white", linewidth=0.6,
                   zorder=4, label="multi-start optima" if k == 0 else None)
    covariance = result["best"]["covariance"]
    if covariance is not None:
        sub = covariance[np.ix_([a, b], [a, b])]
        values, vectors = np.linalg.eigh(sub)
        angle = np.linspace(0, 2 * np.pi, 200)
        circle = np.stack([np.cos(angle), np.sin(angle)])
        ellipse = vectors @ (np.sqrt(np.maximum(values, 0) * 4.605)[:, None] * circle)
        ax.plot(best[a] + ellipse[0], best[b] + ellipse[1], color=INK, linewidth=1.2, linestyle="--",
                label="Gauss–Newton 90% region")
    ax.set_xlim(grid_a[0], grid_a[-1])
    ax.set_ylim(grid_b[0], grid_b[-1])
    ax.set_xlabel(labels[a])
    ax.set_ylabel(labels[b])
    ax.legend(loc="lower left", fontsize=7.5)
    ax.set_title("least-squares objective (chi-square above its minimum), other parameters at their best values",
                 loc="left", fontsize=8.5)
    fig.savefig(path, dpi=160)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pricer", type=Path)
    parser.add_argument("--first", type=int, default=2100)
    parser.add_argument("--surfaces", type=int, default=200)
    parser.add_argument("--starts", type=int, default=10)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--least-squares-paths", type=int, default=8192)
    parser.add_argument("--landscape-points", type=int, default=45)
    args = parser.parse_args()
    manifest, splits = bank.load_bank(args.bank, splits=("test",), roles=bank.TRAINING_ROLES)
    task, workflow, _ = load_network(args.checkpoint)
    mc_paths = args.least_squares_paths if task["model"] == "rough" and args.pricer is None else None
    test = splits["test"]
    rows = np.arange(args.first, args.first + args.surfaces)
    args.out.mkdir(parents=True, exist_ok=False)
    network = posterior_draws(workflow, task, test, rows, args.draws, args.seed)
    jobs = [(task, args.pricer, test["observations"][r, :, 0],
             test["context_values"][r] if test["context_values"].shape[1] else None,
             classical.latin_starts(task, args.starts, args.seed + int(r)), i < 5, mc_paths) for i, r in enumerate(rows)]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        # chunks of 4 cut the overhead of 200 cheap Fourier fits; a few costly Monte Carlo fits need one per worker
        results = list(pool.map(fit_job, jobs, chunksize=4 if len(jobs) >= 8 * args.workers else 1))
    truth = test["parameters"][rows]
    names = task["parameter_names"]
    gauss_newton, posterior, records = [], [], []
    for i, result in enumerate(results):
        interval = classical.wald_interval(result["best"])
        q05, q95 = np.quantile(network[i], [0.05, 0.95], axis=0)
        gauss_newton.append(None if interval is None else ((interval[:, 0] <= truth[i]) & (truth[i] <= interval[:, 1])))
        posterior.append((q05 <= truth[i]) & (truth[i] <= q95))
        records.append({"test_row": int(rows[i]), "truth": truth[i].tolist(), "best_theta": result["best"]["theta"].tolist(),
                        "best_chi2": result["best"]["chi2"], "modes": len(result["modes"]),
                        "mode_thetas": [m["theta"].tolist() for m in result["modes"]],
                        "on_bound": result["best"]["on_bound"], "condition": result["best"]["condition"],
                        "gauss_newton_90": None if interval is None else interval.tolist(),
                        "posterior_90": np.stack([q05, q95], 1).tolist(), "cpu_seconds": result["cpu_seconds"],
                        "tikhonov": [{"penalty": t["penalty"], "theta": t["theta"].tolist()} for t in result.get("tikhonov", [])]})
    valid = [g for g in gauss_newton if g is not None]
    summary = {
        "surfaces": len(rows), "starts": args.starts,
        "gauss_newton_coverage_90": dict(zip(names, np.mean(valid, axis=0).tolist())) if valid else None,
        "gauss_newton_singular": int(sum(g is None for g in gauss_newton)),
        "posterior_coverage_90": dict(zip(names, np.mean(posterior, axis=0).tolist())),
        "modes": {"mean": float(np.mean([r["modes"] for r in records])),
                  "share_with_two_or_more": float(np.mean([r["modes"] >= 2 for r in records]))},
        "best_on_bound_share": float(np.mean([r["on_bound"] for r in records])),
        "cpu_seconds_per_surface_median": float(np.median([r["cpu_seconds"] for r in records])),
        "pricer": ("exact Fourier" if task["model"] == "hkde" else "neural pricer (S4)" if args.pricer
                   else f"Monte Carlo, {mc_paths} paths, fixed seed"),
    }
    report = {"study": "S6", "task_id": task["task_id"], "task_fingerprint": manifest["task_fingerprint"],
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "parameters": names, "labels": [LABELS[n] for n in names], "summary": summary, "surfaces": records}
    (args.out / "classical.json").write_text(json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.out / "network-draws.npz", draws=network[:20], rows=rows[:20])
    pair = [names.index("eta"), names.index("rho")] if task["model"] == "rough" else [names.index("vol_of_vol"), names.index("rho")]
    context = test["context_values"][rows[0]] if test["context_values"].shape[1] else None
    landscape(task, test["observations"][rows[0], :, 0], _price_fn(task, args.pricer, context, mc_paths), results[0],
              network[0], pair, [LABELS[n] for n in names], args.out / "landscape.png", points=args.landscape_points)
    print(json.dumps(jsonable(summary), indent=1))


if __name__ == "__main__":
    main()
