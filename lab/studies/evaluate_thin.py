"""Calibration of the thin-chain posterior (study S2-T) on simulated thin chains.

    python lab/studies/evaluate_thin.py --bank BANK --checkpoint RUN --out NEW_DIR \
        [--first 4000 --surfaces 2000 --draws 1000 --scale 1.0 --seed 0]

Each of --surfaces test surfaces from row --first becomes a thin chain: its clean prices on the checkpoint's layout plus
a fresh quote error with the layout's noise times --scale (independent of the bank's stored noise). The network
returns draws of (H, eta, rho) and the seven log forward-variance levels; rho * eta is added. Writes
validation.json (SBC with a simultaneous band, coverage with Wilson intervals, contraction against each
target's prior variance) and sbc.png. Test rows 4000 to 5999 are S2-T's and 3000 to 3999 are S10's
(lab/studies/README.md).
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sbi4finance.calibration import bank, plots, thin, validation  # noqa: E402

_TAG = 0x45564C54


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_thin_network(checkpoint):
    checkpoint = Path(checkpoint)
    task = json.loads((checkpoint / "task.json").read_text())
    record = json.loads((checkpoint / "training.json").read_text())
    if not record["training"].get("thin_layout"):
        raise ValueError("not a thin-chain checkpoint")
    layout = thin.load_layout(record["training"]["thin_layout"]["path"], task)
    if sha256(record["training"]["thin_layout"]["path"]) != record["training"]["thin_layout"]["sha256"]:
        raise ValueError("the layout file changed since training")
    workflow = thin.build_thin_workflow(task, subnet_widths=tuple(record["training"]["subnet_widths"]))
    workflow.load_approximator(checkpoint / "model.keras")
    return task, layout, workflow, record


def simulated_chains(split, rows, layout, scale, seed):
    """Clean layout prices of the rows plus a fresh error with sd scale * noise, one stream per row."""
    idx, noise = thin.layout_arrays(layout)
    quotes = np.empty((len(rows), len(idx)))
    for i, row in enumerate(rows):
        rng = np.random.default_rng(np.random.SeedSequence([_TAG, int(seed), int(row)]))
        quotes[i] = split["reference_prices"][row, idx] + scale * noise * rng.standard_normal(len(idx))
    return quotes


def prior_logit_variance(task, n=200_000, seed=0):
    """Variance of each target's prior in the logit coordinates of its support (pi^2/3 for a uniform)."""
    draws = thin.draw_prior(task, np.random.default_rng(seed), n)
    bounds = thin.target_bounds(task)
    unit = np.clip((draws - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]), 1e-12, 1 - 1e-12)
    return logit(unit).var(axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--first", type=int, default=4000)
    parser.add_argument("--surfaces", type=int, default=2000)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest, splits = bank.load_bank(args.bank, splits=("test",))
    task, layout, workflow, record = load_thin_network(args.checkpoint)
    if task != manifest["configuration"]["task"] or record["task_fingerprint"] != manifest["task_fingerprint"]:
        raise ValueError("checkpoint and bank tasks differ")
    test = splits["test"]
    rows = np.arange(args.first, min(args.first + args.surfaces, len(test["parameters"])))
    if not len(rows):
        raise ValueError("no test rows in range")
    args.out.mkdir(parents=True, exist_ok=False)
    quotes = simulated_chains(test, rows, layout, args.scale, args.seed)
    started = time.perf_counter()
    draws = np.concatenate([thin.sample_thin(workflow, task, layout, quotes[s:s + 250], args.draws, args.scale,
                                             seed=args.seed + s) for s in range(0, len(rows), 250)])
    seconds = (time.perf_counter() - started) / len(rows)
    truth = np.hstack([test["parameters"][rows], np.log(test["context_values"][rows])])
    names = thin.target_names(task)
    bounds = thin.target_bounds(task)
    prior_variance = list(prior_logit_variance(task))
    h, e, r = (names.index(n) for n in ("hurst", "eta", "rho"))
    draws = np.concatenate([draws, (draws[..., r] * draws[..., e])[..., None]], axis=-1)
    truth = np.concatenate([truth, (truth[:, r] * truth[:, e])[:, None]], axis=1)
    product = [bounds[r, 0] * bounds[e, 1], bounds[r, 1] * bounds[e, 0]]
    bounds = np.vstack([bounds, product])
    names = names + ["rho_eta"]
    prior_variance.append(validation.product_prior_variance(bounds[r], bounds[e], product))
    result = validation.sbc(draws, truth, seed=args.seed)
    cover = validation.coverage(draws, truth)
    shrink = validation.contraction(draws, bounds, prior_variance)
    report = {
        "study": "S2-T", "task_fingerprint": manifest["task_fingerprint"], "layout": layout["ticker"],
        "layout_contracts": len(layout["contract_indices"]), "noise_scale": args.scale,
        "bank_manifest_sha256": sha256(args.bank / "manifest.json"),
        "checkpoint_sha256": sha256(args.checkpoint / "model.keras"),
        "surfaces": int(len(rows)), "test_rows": [int(rows[0]), int(rows[-1]) + 1], "draws_per_surface": args.draws,
        "targets": names, "sbc": {k: v for k, v in result.items() if k != "fractional_ranks"},
        "coverage": cover, "contraction": dict(zip(names, shrink)),
        "prior_variance_logit": dict(zip(names, [float(v) for v in prior_variance])),
        "seconds_per_surface": seconds,
        "note": "contraction against each target's prior variance in the logit coordinates of its support",
    }
    (args.out / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.out / "draws.npz", draws=draws[:300], truth=truth[:300])
    labels = ["H", "η", "ρ"] + [f"log ξ{j + 1}" for j in range(len(names) - 4)] + ["ρη"]
    plots.sbc(result, labels, f"Thin chain ({layout['ticker']} layout, noise x{args.scale:g}): SBC on "
                              f"{len(rows):,} simulated chains").savefig(args.out / "sbc.png", dpi=160)
    print(json.dumps({"sbc_passed": dict(zip(names, result["passed"])),
                      "coverage_90": dict(zip(names, np.round(cover["0.9"]["coverage"], 3).tolist())),
                      "contraction": dict(zip(names, np.round(shrink, 3).tolist()))}, indent=1))


if __name__ == "__main__":
    main()
