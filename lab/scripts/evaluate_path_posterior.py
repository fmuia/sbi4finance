"""Length-stratified development diagnostics; the reserved test split stays closed."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sbi4finance.path_inference import bank, inference  # noqa: E402


def evaluate(args):
    started = time.monotonic()
    if args.split != "validation":
        raise ValueError("this development command only opens validation; reserved tests remain closed")
    for key in ("cases", "draws", "batch_size"):
        inference.integer(getattr(args, key), key)
    task, workflow, training = inference.load_checkpoint(args.checkpoint)
    manifest = bank.load_manifest(args.bank)
    if inference.fingerprint(manifest["task"]) != inference.fingerprint(task):
        raise ValueError("evaluation bank task differs from checkpoint task")
    split = bank.load_split(args.bank, "validation")
    count = min(args.cases, len(split["observations"]))
    truth = split["parameters"][:count]
    prior = inference.prior_samples(task, 100_000, args.seed + 11)
    report = {"format": "path_development_evaluation_v1", "task_sha256": inference.fingerprint(task),
              "checkpoint_sha256": training["model_sha256"], "targets": task["parameter_names"],
              "cases": count, "draws": args.draws, "reserved_test_evaluated": False,
              "supported_lengths": [1, task["n_observations"]], "lengths": {},
              "references": "not compared in this command", "scientific_acceptance": False,
              "split": "validation", "seed": args.seed}
    args.out.mkdir(parents=True, exist_ok=False)
    saved = {"truth": truth[:32], "observations": split["observations"][:32],
             "trajectory_id": split["trajectory_id"][:32]}
    # Include arbitrary lengths, not just the display landmarks, in development.
    lengths = sorted(set(task["prefix_lengths"] + [1, min(17, task["n_observations"]),
                                                 max(1, task["n_observations"] - 7)]))
    physical = np.asarray(task["prior_bounds"])
    for length in lengths:
        batches = []
        for start in range(0, count, args.batch_size):
            observations = split["observations"][start:min(count, start + args.batch_size), :length]
            batches.append(inference.sample(workflow, task, observations, args.draws,
                                             seed=args.seed + 10007 * length + start))
        draws = np.concatenate(batches)
        metrics = inference.diagnostics(draws, truth, prior, seed=args.seed + length)
        metrics["elapsed_years"] = length * task["dt_years"]
        metrics["out_of_prior_fraction"] = np.mean(
            (draws <= physical[:, 0]) | (draws >= physical[:, 1]), axis=(0, 1)).tolist()
        metrics["nonfinite_count"] = int(np.sum(~np.isfinite(draws)))
        if "eta" in task["parameter_names"] and "rho" in task["parameter_names"]:
            eta, rho = (task["parameter_names"].index(k) for k in ("eta", "rho"))
            metrics["rho_eta"] = inference.diagnostics(
                (draws[..., eta] * draws[..., rho])[..., None],
                (truth[:, eta] * truth[:, rho])[:, None],
                (prior[:, eta] * prior[:, rho])[:, None], seed=args.seed + length)
        report["lengths"][str(length)] = metrics
        saved[f"draws_{length}"] = draws[:32]
        np.savez_compressed(args.out / "teaching-cases.npz", **saved)
        report["elapsed_seconds"] = time.monotonic() - started
        (args.out / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"evaluated {count} histories at length {length}", flush=True)
    report["teaching_cases_sha256"] = inference.sha256(args.out / "teaching-cases.npz")
    (args.out / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=256)
    parser.add_argument("--draws", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--seed", type=int, default=0)
    report = evaluate(parser.parse_args())
    print(json.dumps({"lengths": list(report["lengths"]), "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
