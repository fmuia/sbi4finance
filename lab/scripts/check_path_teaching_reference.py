"""Bounded independent short-history controls on inspected validation cases.

Inputs are a hash-verified native bank or curated development replay. The
reference sees only observed return prefixes, never their generating labels.
No reserved test or unseen validation cohort is consumed by this command.
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy

from sbi4finance.path_inference import bank, teaching_reference as reference


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def load_inputs(*, bank_path=None, replay=None, cases=(0, 1)):
    """Read requested inspected validation rows and record their bound source."""
    if (bank_path is None) == (replay is None):
        raise ValueError("provide exactly one native bank or curated replay")
    if (not cases or len(set(cases)) != len(cases)
            or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < 1024 for i in cases)):
        raise ValueError("reference is restricted to unique inspected validation IDs 0:1024")
    selected, candidate = {}, {}
    if bank_path is not None:
        directory = Path(bank_path).resolve()
        manifest = bank.load_manifest(directory)
        task = reference.validate_task(manifest["task"])
        shards = []
        for record in manifest["splits"]["validation"]:
            wanted = [i for i in cases if record["start"] <= i < record["start"] + record["count"]]
            if not wanted:
                continue
            arrays = bank._load_part(directory, manifest, "validation", record)
            for index in wanted:
                row = index - record["start"]
                selected[index] = {"observations": arrays["observations"][row, :, 0].copy(),
                                   "trajectory_id": str(arrays["trajectory_id"][row])}
            shards.append({"path": record["path"], "sha256": record["sha256"]})
        source = {"kind": "native_bank", "directory": str(directory),
                  "bank_manifest_sha256": _hash(directory / "manifest.json"),
                  "verified_validation_shards": shards}
    else:
        from sbi4finance.path_inference.teaching import load_cache
        directory = Path(replay).resolve()
        task, bundle, arrays = load_cache(directory)
        reference.validate_task(task)
        for row, identity in enumerate(arrays["trajectory_id"]):
            index = int(str(identity).rsplit(":", 1)[-1])
            if index in cases:
                selected[index] = {"observations": arrays["observations"][row, :, 0].copy(),
                                   "trajectory_id": str(identity)}
                candidate[index] = {int(k.removeprefix("draws_")): values[row].copy()
                                    for k, values in arrays.items() if k.startswith("draws_")}
        evaluation = json.loads((directory / "evaluation.json").read_text())
        source = {"kind": "development_replay", "directory": str(directory),
                  "bundle_sha256": _hash(directory / "bundle.json"),
                  "bank_manifest_sha256": evaluation["bank_manifest_sha256"],
                  "evaluation_sha256": bundle["files"]["evaluation.json"],
                  "source_case_panel_sha256": bundle["source_case_panel_sha256"],
                  "checkpoint_sha256": bundle["checkpoint_sha256"]}
    if set(selected) != set(cases):
        raise ValueError("not every requested validation case is present in verified input")
    source.update(split="validation", inspected_case_indices=list(cases),
                  reserved_test_evaluated=False, unseen_validation_evaluated=False)
    return task, source, selected, candidate


def _candidate_comparison(draws, reference_runs, precision_status):
    best = max(reference_runs, key=lambda r: (r["samples"], r["ess"]))
    mean, sd = np.asarray(best["posterior_mean"]), np.asarray(best["posterior_sd"])
    quantiles = np.quantile(draws, reference.QUANTILES, axis=0).T
    return {"role": "exploratory_marginal_comparison_to_short_history_reference",
            "reference_precision_status": precision_status,
            "reference_seed": best["seed"], "candidate_draws": len(draws),
            "candidate_mean": np.mean(draws, axis=0).tolist(),
            "candidate_quantiles": quantiles.tolist(),
            "mean_difference_in_reference_sd": ((np.mean(draws, axis=0) - mean) / sd).tolist(),
            "quantile_difference_in_reference_sd": ((quantiles - best["posterior_quantiles"]) / sd[:, None]).tolist(),
            "scientific_acceptance": False,
            "limitations": "Marginal development-case comparison only; Monte Carlo uncertainty and finite neural draws remain. No joint-posterior or general conditional validation."}


def run(args):
    if (not args.samples or any(n < 1 for n in args.samples) or args.samples != sorted(set(args.samples))
            or args.replicates < 2 or args.chunk < 1 or args.draws < 1
            or not np.isfinite(args.max_seconds) or args.max_seconds <= 0
            or not args.lengths or args.lengths != sorted(set(args.lengths))
            or any(n < 1 or n > 21 for n in args.lengths)):
        raise ValueError("positive distinct increasing counts/lengths<=21 and >=2 replicates required")
    task, source, cases, candidates = load_inputs(bank_path=args.bank, replay=args.replay, cases=args.cases)
    if max(args.lengths) > task["n_observations"]:
        raise ValueError("requested prefix exceeds task length")
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    deadline = started + args.max_seconds
    report = {"format": "daily_drift_short_reference_v1", "task": task,
              "task_sha256": bank.task_fingerprint(task), "source": source,
              "method": "independent prior/latent importance; exact Gaussian asset-noise and uniform drift integration",
              "source_code_sha256": {"teaching_reference.py": _hash(reference.__file__),
                                     "check_path_teaching_reference.py": _hash(__file__)},
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
              "configuration": {"base_seed": args.seed, "samples": args.samples, "replicates": args.replicates,
                                "chunk": args.chunk, "draws": args.draws, "lengths": args.lengths,
                                "max_seconds": args.max_seconds},
              "gates": reference.GATES, "cases": {}, "scientific_acceptance": False,
              "reserved_test_evaluated": False,
              "limitations": "Two inspected development cases and short histories only. Numerical stability is necessary, not a proof of importance-tail exploration or conditional/joint accuracy."}
    cap_reached = False
    for index, case in cases.items():
        case_report = {"trajectory_id": case["trajectory_id"],
                       "full_observations_sha256": reference.observations_sha256(case["observations"]), "lengths": {}}
        report["cases"][str(index)] = case_report
        for length in args.lengths:
            runs, failures = [], []
            current = {"runs": runs, "failures": failures}
            case_report["lengths"][str(length)] = current
            diagnostic = task["model"] == "heston" and length > 1
            for count in args.samples:
                for replicate in range(args.replicates):
                    seed = int(np.random.SeedSequence([args.seed, bank.MODELS.index(task["model"]),
                                                      index, length, count, replicate]).generate_state(1, dtype=np.uint64)[0])
                    filename = f"case-{index}-length-{length}-n-{count}-rep-{replicate}"
                    try:
                        options = dict(samples=count, seed=seed, chunk=args.chunk, deadline=deadline)
                        if diagnostic:
                            summary = reference.atom_support_diagnostic(task, case["observations"][:length], **options)
                        else:
                            summary, arrays = reference.importance_reference(task, case["observations"][:length], draws=args.draws, **options)
                            path = destination / f"{filename}.npz"
                            np.savez_compressed(path, **arrays)
                            summary["payload"] = {"file": path.name, "sha256": _hash(path)}
                        summary.update(replicate=replicate, trajectory_id=case["trajectory_id"])
                        runs.append(summary)
                        _write(destination / f"{filename}.json", summary)
                        print(json.dumps({"case": index, "length": length, "samples": count, "replicate": replicate,
                                          "ess": summary.get("ess"), "seconds": summary["seconds"],
                                          "diagnostic_only": diagnostic}), flush=True)
                    except (TimeoutError, FloatingPointError) as error:
                        failures.append({"samples": count, "replicate": replicate, "error": str(error)})
                        if isinstance(error, TimeoutError):
                            cap_reached = True
                            break
                    _write(destination / "reference-summary.json", report)
                if cap_reached:
                    break
            current["comparison"] = ({"status": "posterior_reference_inconclusive_atom_handling_required",
                                       "scientific_acceptance": False} if diagnostic else reference.compare_runs(runs))
            expected = len(args.samples) * args.replicates
            if failures or len(runs) != expected:
                current["comparison"]["status"] = "inconclusive_incomplete_or_failed_runs"
            if not diagnostic and runs and index in candidates and length in candidates[index]:
                current["candidate_comparison"] = _candidate_comparison(candidates[index][length], runs, current["comparison"]["status"])
            _write(destination / "reference-summary.json", report)
            if cap_reached:
                break
        if cap_reached:
            break
    report.update(seconds=time.perf_counter() - started, wall_clock_cap_reached=cap_reached,
                  status="bounded_diagnostic_complete" if not cap_reached else "bounded_diagnostic_incomplete_cap_reached")
    _write(destination / "reference-summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bank")
    source.add_argument("--replay")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 21])
    parser.add_argument("--samples", type=int, nargs="+", default=[32768, 65536])
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--draws", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20261031181)
    parser.add_argument("--max-seconds", type=float, default=600)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "seconds": result["seconds"], "scientific_acceptance": False}), flush=True)


if __name__ == "__main__":
    main()
