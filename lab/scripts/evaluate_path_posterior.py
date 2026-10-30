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


def requested_lengths(task, values=None):
    if values is None:
        return sorted(set(task["prefix_lengths"] + [1, min(17, task["n_observations"]),
                                                     max(1, task["n_observations"] - 7)]))
    if isinstance(values, str):
        values = [int(value) for value in values.replace(",", " ").split()]
    lengths = sorted(set(inference.integer(value, "evaluation length") for value in values))
    if not lengths or lengths[-1] > task["n_observations"]:
        raise ValueError("requested lengths must lie between one and task maximum")
    return lengths


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


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
    offset = inference.integer(getattr(args, "offset", 0), "offset", minimum=0)
    stop = min(offset + args.cases, len(split["observations"]))
    if stop <= offset:
        raise ValueError("requested validation cohort is empty")
    split = {key: value[offset:stop] if isinstance(value, np.ndarray) else value
             for key, value in split.items()}
    count = stop - offset
    truth = split["parameters"][:count]
    prior = inference.prior_samples(task, 100_000, args.seed + 11)
    report = {"format": "path_development_evaluation_v1", "task_sha256": inference.fingerprint(task),
              "checkpoint_sha256": training["model_sha256"], "targets": task["parameter_names"],
              "cases": count, "draws": args.draws, "reserved_test_evaluated": False,
              "supported_lengths": [1, task["n_observations"]], "lengths": {},
              "references": "not compared in this command", "scientific_acceptance": False,
              "split": "validation", "offset": offset, "seed": args.seed,
              "case_identities_sha256": inference.fingerprint(split["trajectory_id"].tolist()),
              "prior_checkpoint_development_pool_size": training.get("validation_cases_per_length"),
              "cohort_note": "offsets identify validation rows; later rows remain validation, not reserved tests"}
    args.out.mkdir(parents=True, exist_ok=False)
    saved = {"truth": truth[:32], "observations": split["observations"][:32],
             "trajectory_id": split["trajectory_id"][:32]}
    case_summary = {"truth": truth, "trajectory_id": split["trajectory_id"]}
    # Include arbitrary lengths, not just the display landmarks, in development.
    lengths = requested_lengths(task, getattr(args, "lengths", None))
    physical = np.asarray(task["prior_bounds"])
    for length in lengths:
        batches = []
        for start in range(0, count, args.batch_size):
            observations = split["observations"][start:min(count, start + args.batch_size), :length]
            batches.append(inference.sample(workflow, task, observations, args.draws,
                                             seed=args.seed + 10007 * length + offset + start))
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
        lower, upper = np.quantile(draws, [.05, .95], axis=1)
        case_summary[f"posterior_mean_{length}"] = draws.mean(axis=1)
        case_summary[f"lower90_{length}"] = lower
        case_summary[f"upper90_{length}"] = upper
        case_summary[f"posterior_variance_{length}"] = draws.var(axis=1)
        np.savez_compressed(args.out / "teaching-cases.npz", **saved)
        np.savez_compressed(args.out / "case-summary.npz", **case_summary)
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.out / "evaluation.json", report)
        print(f"evaluated {count} histories at length {length}", flush=True)
    report["teaching_cases_sha256"] = inference.sha256(args.out / "teaching-cases.npz")
    report["case_summary_file"] = "case-summary.npz"
    report["case_summary_sha256"] = inference.sha256(args.out / "case-summary.npz")
    atomic_json(args.out / "evaluation.json", report)
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
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--lengths", type=int, nargs="+", help="Explicit arbitrary history lengths to evaluate")
    report = evaluate(parser.parse_args())
    print(json.dumps({"lengths": list(report["lengths"]), "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
