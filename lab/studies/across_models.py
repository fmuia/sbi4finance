"""The same smile under two models, rough Bergomi and Heston (study S7, across models).

    python lab/studies/across_models.py --rough-bank BANK --rough-products S7_DIR \
        --heston-checkpoint RUN --out NEW_DIR [--draws 400 --proposals 2048 --starts 10 --workers 4]

For each smile of the rough-Bergomi products study (S7_DIR, which holds
products.json and price-draws.npz), all simulated from rough Bergomi, so that
Heston is the misspecified model here:

1. the Heston posterior: the Heston network's draws, corrected by importance
   reweighting with the exact Fourier likelihood (the audit of study S3); its
   efficiency and k-hat say how far the smile lies from the smiles the Heston
   network was trained on;
2. the Heston least-squares fit (Fourier pricer, multi-start) and its
   chi-square, against the rough fit's chi-square at the rough least-squares
   point (Monte Carlo at 2^15 paths per block);
3. the four products priced under the Heston posterior and at the Heston point,
   with the specification, paths and seed of the rough products study.

The two tasks share the 63 contracts, the forward and the quote-noise model, so
the same quotes condition both. The gap between the two models' prices is the
across-model range the within-model posterior range sits under.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys

from matplotlib.ticker import MaxNLocator
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sbi4finance.calibration import bank, classical, correction, pricing, products, tasks  # noqa: E402
from sbi4finance.calibration.plots import INK, SERIES, plt, style  # noqa: E402
from evaluate_calibration import load_network, posterior_draws  # noqa: E402
from heston_check import audit_job, exact_prices, jsonable  # noqa: E402
from products_study import PRODUCT_LABELS, price_job  # noqa: E402

ROUGH_CHI2_PATHS = 2**15


def summary(values, point):
    return {"point": float(point), "mean": float(values.mean()), "q05": float(np.quantile(values, 0.05)),
            "q95": float(np.quantile(values, 0.95)),
            "range_90_relative": float((np.quantile(values, 0.95) - np.quantile(values, 0.05)) / values.mean()),
            "reserve_short_relative": products.reserve(values, point) / point}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rough-bank", type=Path, required=True)
    parser.add_argument("--rough-products", type=Path, required=True, help="the rough-Bergomi S7 run directory")
    parser.add_argument("--heston-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--network-draws", type=int, default=1000)
    parser.add_argument("--proposals", type=int, default=2048)
    parser.add_argument("--starts", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest, splits = bank.load_bank(args.rough_bank)
    rough_task = manifest["configuration"]["task"]
    rough = json.loads((args.rough_products / "products.json").read_text())
    rough_prices = np.load(args.rough_products / "price-draws.npz")
    if rough["task_fingerprint"] != manifest["task_fingerprint"]:
        raise ValueError("the products study was not run on this bank")
    heston_task, workflow, _ = load_network(args.heston_checkpoint)
    for key in ("contracts", "quote_noise_std", "spot", "rate", "dividend"):
        if heston_task["context"][key] != rough_task["context"][key]:
            raise ValueError(f"the two tasks differ in {key}; the same quotes cannot condition both")
    spec, paths, seed = rough["specification"], rough["settings"]["paths"], rough["settings"]["seed"]
    test = splits["test"]
    rows = np.array([s["test_row"] for s in rough["surfaces"]])
    observations = test["observations"][rows, :, 0]
    args.out.mkdir(parents=True, exist_ok=False)
    network = posterior_draws(workflow, heston_task, {"observations": test["observations"][rows],
                                                      "context_values": np.zeros((len(rows), 0))},
                              np.arange(len(rows)), args.network_draws, args.seed)
    sigma = tasks.sigma(heston_task)
    records, saved = [], {}
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        audits = list(pool.map(audit_job, [(heston_task, observations[i], network[i], args.proposals, args.seed + i)
                                           for i in range(len(rows))]))
        for i, row in enumerate(rows):
            fits = classical.fit(heston_task, observations[i], lambda theta: exact_prices(heston_task, theta),
                                 classical.latin_starts(heston_task, args.starts, args.seed + int(row)))
            best = min(fits, key=lambda f: f["chi2"])
            draws = audits[i]["draws"][: args.draws]
            priced = list(pool.map(price_job, [(heston_task, theta, None, spec, paths, seed)
                                               for theta in [best["theta"], *draws]], chunksize=8))
            heston_point = priced[0][0]
            heston_values = np.stack([p[0] for p in priced[1:]])
            context = test["context_values"][row]
            rough_entry = rough["surfaces"][i]
            rough_fit = pricing.rough_surface(tasks.model_parameters(rough_task, np.array(rough_entry["least_squares"])),
                                              tasks.curve(rough_task, context), rough_task["context"]["contracts"],
                                              rough_task["numerics"]["blocks"], [ROUGH_CHI2_PATHS],
                                              tasks.sigma(rough_task), np.inf, seed)[0]
            entry = {"test_row": int(row), "truth_rough": test["parameters"][row].tolist(),
                     "heston_network_mean": network[i].mean(axis=0).tolist(),
                     "heston_posterior": {k: np.asarray(v).tolist() for k, v in audits[i]["summary"].items()},
                     "heston_audit": {k: audits[i][k] for k in ("efficiency", "khat", "max_weight", "cpu_seconds")},
                     "heston_least_squares": {"theta": best["theta"].tolist(), "chi2": best["chi2"]},
                     "rough_least_squares": {"theta": rough_entry["least_squares"],
                                             "chi2": float(np.sum(((rough_fit - observations[i]) / sigma) ** 2)),
                                             "chi2_paths": ROUGH_CHI2_PATHS},
                     "quotes": len(sigma), "products": {}}
            for j, product in enumerate(products.PRODUCTS):
                h = summary(heston_values[:, j], heston_point[j])
                r = rough_entry["products"][product]
                entry["products"][product] = {
                    "heston": h, "rough": {k: r[k] for k in ("point", "mean", "q05", "q95", "range_90_relative",
                                                             "reserve_short_relative")},
                    "gap_between_points_relative": (h["point"] - r["point"]) / r["point"],
                    "gap_between_means_relative": (h["mean"] - r["mean"]) / r["mean"],
                    "ranges_overlap": bool(h["q05"] <= r["q95"] and r["q05"] <= h["q95"])}
            records.append(entry)
            saved.update({f"heston_prices_{int(row)}": heston_values, f"heston_point_prices_{int(row)}": heston_point,
                          f"heston_draws_{int(row)}": draws})
    report = {"study": "S7 across models", "rough_task_fingerprint": manifest["task_fingerprint"],
              "heston_checkpoint_sha256": hashlib.sha256((args.heston_checkpoint / "model.keras").read_bytes()).hexdigest(),
              "rough_products_sha256": hashlib.sha256((args.rough_products / "products.json").read_bytes()).hexdigest(),
              "specification": spec, "paths": paths, "seed": seed,
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "heston_parameters": heston_task["parameter_names"], "surfaces": records}
    (args.out / "across-models.json").write_text(json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.out / "heston-price-draws.npz", products=np.array(products.PRODUCTS), **saved)

    style()
    fig, axes = plt.subplots(len(rows), len(products.PRODUCTS), figsize=(11, 2.3 * len(rows)), constrained_layout=True,
                             squeeze=False)
    for i, row in enumerate(rows):
        for j, product in enumerate(products.PRODUCTS):
            ax = axes[i, j]
            r_values, h_values = rough_prices[f"prices_{int(row)}"][:, j], saved[f"heston_prices_{int(row)}"][:, j]
            bins = np.linspace(min(r_values.min(), h_values.min()), max(r_values.max(), h_values.max()), 45)
            if bins[0] == bins[-1]:
                bins = np.linspace(bins[0] * 0.999, bins[0] * 1.001, 45)
            ax.hist(r_values, bins=bins, color=SERIES[0], alpha=0.7, label="rough Bergomi posterior")
            ax.hist(h_values, bins=bins, color=SERIES[2], alpha=0.6, label="Heston posterior")
            ax.axvline(rough_prices[f"point_prices_{int(row)}"][j], color=SERIES[0], linewidth=1.6)
            ax.axvline(saved[f"heston_point_prices_{int(row)}"][j], color=SERIES[2], linewidth=1.6)
            if i == 0:
                ax.set_title(PRODUCT_LABELS[product], loc="left", fontsize=8.5)
            ax.grid(axis="y", visible=False)
            ax.xaxis.set_major_locator(MaxNLocator(4))
    axes[0, 0].legend(loc="upper left", fontsize=7)
    fig.suptitle("The same smiles under two models: product prices under each posterior (lines: least squares)",
                 x=0.01, ha="left", color=INK, fontsize=10.5, fontweight="bold")
    fig.savefig(args.out / "across-models.png", dpi=160)
    print(json.dumps(jsonable([{p: {k: round(v, 4) if isinstance(v, float) else v for k, v in e["products"][p].items()
                                    if k.startswith("gap") or k == "ranges_overlap"} for p in products.PRODUCTS}
                               | {"chi2_heston": round(e["heston_least_squares"]["chi2"], 1),
                                  "chi2_rough": round(e["rough_least_squares"]["chi2"], 1),
                                  "audit_efficiency": round(e["heston_audit"]["efficiency"], 3)} for e in records]),
                     indent=1))


if __name__ == "__main__":
    main()
