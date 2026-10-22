"""Products under the posterior: price ranges, reserves and what drives them (study S7).

    python tutorial_final/studies/products_study.py --bank BANK --checkpoint RUN --out NEW_DIR \
        [--pricer S4_DIR/pricer] [--first 2300 --surfaces 3 --draws 400 --paths 8192 --workers 4]

For each surface: posterior draws from the network; each draw prices the
variance swap, the volatility swap, the forward-start call and the 2-year 90%
put by Monte Carlo with common random numbers (the same normals for every draw);
the least-squares point (best of a multi-start) prices them too. Reported per
product: the point price, the posterior mean and 90% range, the reserve on the
short side (90% quantile minus the point price), and the share of the price
variance explained linearly by each parameter.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sbi4finance.calibration import bank, classical, products, tasks  # noqa: E402
from sbi4finance.calibration.plots import INK, SERIES, plt, style  # noqa: E402
from classical_route import _price_fn  # noqa: E402
from evaluate_calibration import LABELS, load_network, posterior_draws  # noqa: E402
from heston_check import jsonable  # noqa: E402

PRODUCT_LABELS = {"variance_swap": "variance swap, 3m (strike, variance)",
                  "volatility_swap": "volatility swap, 3m (strike, volatility)",
                  "forward_start_call": "forward-start ATM call, 3m into 6m",
                  "put_out_of_grid": "90% put, 2y (outside the grid)"}


def model_parameters(task, row):
    parameters = tasks.model_parameters(task, row)
    return parameters if task["model"] == "rough" else parameters  # Heston keeps initial_variance inside


def price_job(job):
    task, row, context, spec, paths, seed = job
    model = "rough" if task["model"] == "rough" else "heston"
    curve = None
    if model == "rough":
        curve = products.extend_curve(tasks.curve(task, context), spec["put_out_of_grid"]["maturity"])
    return products.price_draw(model, model_parameters(task, row), curve, spec, paths, seed)


def variance_shares(parameters, prices):
    """R^2 of a single-parameter linear fit of the price, per parameter (shares need not sum to one)."""
    shares = []
    for j in range(parameters.shape[1]):
        x = parameters[:, j]
        if np.std(x) == 0 or np.std(prices) == 0:
            shares.append(0.0)
            continue
        shares.append(float(np.corrcoef(x, prices)[0, 1] ** 2))
    return shares


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pricer", type=Path)
    parser.add_argument("--first", type=int, default=2300)
    parser.add_argument("--surfaces", type=int, default=3)
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--paths", type=int, default=8192)
    parser.add_argument("--starts", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest, splits = bank.load_bank(args.bank)
    task, workflow, _ = load_network(args.checkpoint)
    test = splits["test"]
    rows = np.arange(args.first, args.first + args.surfaces)
    spec = products.default_specification()
    args.out.mkdir(parents=True, exist_ok=False)
    network = posterior_draws(workflow, task, test, rows, args.draws, args.seed)
    names = task["parameter_names"]
    has_context = test["context_values"].shape[1] > 0
    records, saved = [], {}
    style()
    fig, axes = plt.subplots(len(rows), len(products.PRODUCTS), figsize=(11, 2.3 * len(rows)), constrained_layout=True,
                             squeeze=False)
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for i, row in enumerate(rows):
            context = test["context_values"][row] if has_context else None
            fits = classical.fit(task, test["observations"][row, :, 0], _price_fn(task, args.pricer, context),
                                 classical.latin_starts(task, args.starts, args.seed + int(row)))
            point = min(fits, key=lambda f: f["chi2"])["theta"]
            started = time.process_time()
            jobs = [(task, theta, context, spec, args.paths, args.seed) for theta in [point, *network[i]]]
            priced = list(pool.map(price_job, jobs, chunksize=8))
            cpu = time.process_time() - started
            point_prices = priced[0][0]
            draws_prices = np.stack([p[0] for p in priced[1:]])
            entry = {"test_row": int(row), "truth": test["parameters"][row].tolist(), "least_squares": point.tolist(),
                     "products": {}}
            for j, product in enumerate(products.PRODUCTS):
                values = draws_prices[:, j]
                entry["products"][product] = {
                    "point": float(point_prices[j]), "mean": float(values.mean()),
                    "q05": float(np.quantile(values, 0.05)), "q95": float(np.quantile(values, 0.95)),
                    "range_90_relative": float((np.quantile(values, 0.95) - np.quantile(values, 0.05)) / values.mean()),
                    "reserve_short": products.reserve(values, point_prices[j]),
                    "reserve_short_relative": products.reserve(values, point_prices[j]) / point_prices[j],
                    "variance_share_by_parameter": dict(zip(names, variance_shares(network[i], values))),
                    "mc_se_typical": float(np.median([p[1][j] for p in priced])),
                }
                ax = axes[i, j]
                ax.hist(values, bins=40, color=SERIES[0], alpha=0.75)
                ax.axvline(point_prices[j], color=SERIES[1], linewidth=1.6, label="least squares")
                ax.axvline(np.quantile(values, 0.9), color=INK, linewidth=1.0, linestyle="--", label="90% quantile")
                if i == 0:
                    ax.set_title(PRODUCT_LABELS[product], loc="left", fontsize=8.5)
                ax.grid(axis="y", visible=False)
            entry["pricing_cpu_seconds"] = cpu
            records.append(entry)
            saved.update({f"prices_{int(row)}": draws_prices, f"point_prices_{int(row)}": point_prices,
                          f"draws_{int(row)}": network[i], f"least_squares_{int(row)}": point,
                          f"truth_{int(row)}": test["parameters"][row]})
    axes[0, 0].legend(loc="upper left", fontsize=7.5)
    model = "rough Bergomi" if task["model"] == "rough" else "Heston"
    fig.suptitle(f"{model}: product prices under the posterior, {args.draws} draws per surface", x=0.01, ha="left",
                 color=INK, fontsize=10.5, fontweight="bold")
    fig.savefig(args.out / "price-distributions.png", dpi=160)
    report = {"study": "S7", "task_id": task["task_id"], "task_fingerprint": manifest["task_fingerprint"],
              "specification": spec, "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "parameters": names, "labels": [LABELS[n] for n in names], "surfaces": records}
    (args.out / "products.json").write_text(json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.out / "price-draws.npz", products=np.array(products.PRODUCTS), **saved)
    print(json.dumps(jsonable([{p: {k: round(v, 5) if isinstance(v, float) else v
                                    for k, v in r["products"][p].items() if k != "variance_share_by_parameter"}
                                for p in products.PRODUCTS} for r in records]), indent=1))


if __name__ == "__main__":
    main()
