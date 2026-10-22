"""Statistical calibration and identification of the amortized posterior (study S2).

    python tutorial_final/studies/evaluate_calibration.py --bank BANK --checkpoint RUN --out NEW_DIR \
        [--surfaces 2000 --draws 1000 --seed 0]

Uses the first --surfaces test surfaces; later test surfaces stay free for the
exact checks and the products. Writes validation.json, sbc.png, recovery.png
and contraction.png. For rough Bergomi the product rho * eta is reported too.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sbi4finance.calibration import bank, inference, plots, validation  # noqa: E402

LABELS = {"hurst": "H", "eta": "η", "rho": "ρ", "initial_variance": "v₀", "theta": "θ",
          "vol_of_vol": "σ_v", "rho_eta": "ρη"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_network(checkpoint):
    task = json.loads((Path(checkpoint) / "task.json").read_text())
    record = json.loads((Path(checkpoint) / "training.json").read_text())
    workflow = inference.build_workflow(task, subnet_widths=tuple(record["training"]["subnet_widths"]))
    workflow.load_approximator(Path(checkpoint) / "model.keras")
    return task, workflow, record


def posterior_draws(workflow, task, split, rows, draws, seed, batch=250):
    out = []
    for start in range(0, len(rows), batch):
        chosen = rows[start:start + batch]
        out.append(inference.sample(workflow, task, split["observations"][chosen], split["context_values"][chosen],
                                    draws, seed=seed + start))
    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--surfaces", type=int, default=2000)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest, splits = bank.load_bank(args.bank)
    task, workflow, record = load_network(args.checkpoint)
    if task != manifest["configuration"]["task"] or record["task_fingerprint"] != manifest["task_fingerprint"]:
        raise ValueError("checkpoint and bank tasks differ")
    test = splits["test"]
    rows = np.arange(min(args.surfaces, len(test["parameters"])))
    names = list(task["parameter_names"])
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    draws = posterior_draws(workflow, task, test, rows, args.draws, args.seed)
    batch_seconds = (time.perf_counter() - started) / len(rows)
    single = time.perf_counter()
    inference.sample(workflow, task, test["observations"][:1], test["context_values"][:1], args.draws, seed=args.seed)
    single_seconds = time.perf_counter() - single
    truth = test["parameters"][rows]
    labels, bounds = [LABELS[n] for n in names], np.asarray(task["prior_bounds"])
    prior_variance = None
    if task["model"] == "rough":
        h, e, r = (names.index(n) for n in ("hurst", "eta", "rho"))
        draws = np.concatenate([draws, (draws[..., r] * draws[..., e])[..., None]], axis=-1)
        truth = np.concatenate([truth, (truth[:, r] * truth[:, e])[:, None]], axis=1)
        names, labels = names + ["rho_eta"], labels + [LABELS["rho_eta"]]
        product = [bounds[r, 0] * bounds[e, 1], bounds[r, 1] * bounds[e, 0]]
        bounds = np.vstack([bounds, product])
        prior_variance = [None] * (len(names) - 1) + [validation.product_prior_variance(bounds[r], bounds[e], product)]
    result = validation.sbc(draws, truth, seed=args.seed)
    cover = validation.coverage(draws, truth)
    shrink = validation.contraction(draws, bounds, prior_variance)
    recovered = validation.recovery(draws, truth)
    model_targets = list(task["parameter_names"])
    correlation = validation.posterior_correlations(draws[..., : len(model_targets)], bounds[: len(model_targets)])
    report = {
        "study": "S2", "task_id": task["task_id"], "task_fingerprint": manifest["task_fingerprint"],
        "bank_manifest_sha256": sha256(args.bank / "manifest.json"),
        "checkpoint_sha256": sha256(args.checkpoint / "model.keras"),
        "surfaces": int(len(rows)), "test_rows": [0, int(len(rows))], "draws_per_surface": args.draws,
        "targets": names, "sbc": {k: v for k, v in result.items() if k != "fractional_ranks"},
        "coverage": cover, "contraction": dict(zip(names, shrink)),
        "recovery_correlation": dict(zip(names, recovered["correlation"])),
        "posterior_correlation_logit": {f"{model_targets[a]}|{model_targets[b]}": v for (a, b), v in correlation.items()},
        "seconds_per_surface": {"batched": batch_seconds, "single": single_seconds},
        "contraction_prior_variance_logit": dict(zip(names, [np.pi**2 / 3 if v is None else v
                                                             for v in (prior_variance or [None] * len(names))])),
        "note": ("contraction in the logit coordinates of each target's range; rho_eta's prior is not uniform on "
                 "its range, so its prior variance there is computed from the prior box by Monte Carlo"
                 if prior_variance else "contraction in the logit coordinates of the prior box"),
    }
    (args.out / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.out / "draws.npz", draws=draws[:300], truth=truth[:300])
    model = "rough Bergomi" if task["model"] == "rough" else "Heston"
    plots.sbc(result, labels, f"{model}: simulation-based calibration on {len(rows):,} held-out surfaces").savefig(args.out / "sbc.png", dpi=160)
    plots.recovery(truth, draws, labels, shrink, f"{model}: recovery on held-out surfaces").savefig(args.out / "recovery.png", dpi=160)
    plots.contraction(shrink, labels, f"{model}: what a smile identifies").savefig(args.out / "contraction.png", dpi=160)
    print(json.dumps({"sbc_passed": result["passed"], "contraction": dict(zip(names, np.round(shrink, 3).tolist())),
                      "coverage_90": cover["0.9"]["coverage"], "seconds_per_surface": report["seconds_per_surface"]}, indent=1))


if __name__ == "__main__":
    main()
