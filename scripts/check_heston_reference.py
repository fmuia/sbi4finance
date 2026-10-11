"""Check the small Heston reference by particle/grid refinement and independent repeats.

Uses a fixed prefix of a declared bank split. This establishes a numerical
reference pilot, not neural acceptance or real-market model adequacy.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from sbi4finance import heston_reference
from sbi4finance.heston_bank import load_bank


SETTINGS = ((24, 1024), (48, 4096), (48, 4096))
THRESHOLDS = {"maximum_mean_difference_in_sd": 0.10,
              "maximum_relative_sd_difference": 0.10,
              "correlation_difference": 0.10}


def summary(reference):
    theta = reference["theta_grid"]
    state = reference["terminal_particles"]
    probabilities = reference["grid_probabilities"]
    mean = np.array([probabilities @ theta, probabilities @ state.mean(axis=1)])
    theta_centered, state_centered = theta - mean[0], state - mean[1]
    variance = np.array([probabilities @ theta_centered**2,
                         probabilities @ (state_centered**2).mean(axis=1)])
    std = np.sqrt(variance)
    covariance = probabilities @ (theta_centered * state_centered.mean(axis=1))
    return {"mean": mean.tolist(), "std": std.tolist(),
            "correlation": float(covariance / np.prod(std))}


def compare(left, right):
    scale = np.asarray(right["std"])
    result = {
        "maximum_mean_difference_in_sd": float(np.max(np.abs(np.asarray(left["mean"]) - right["mean"]) / scale)),
        "maximum_relative_sd_difference": float(np.max(np.abs(np.asarray(left["std"]) / scale - 1))),
        "correlation_difference": float(abs(left["correlation"] - right["correlation"])),
    }
    result["passed"] = all(result[key] <= value for key, value in THRESHOLDS.items())
    return result


def sample_reference(reference, size, *, seed):
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(reference["theta_grid"]), size=size, p=reference["grid_probabilities"])
    columns = rng.integers(reference["terminal_particles"].shape[1], size=size)
    return np.column_stack([reference["theta_grid"][rows], reference["terminal_particles"][rows, columns]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new reference-report directory")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--seed", type=int, default=6831)
    args = parser.parse_args()
    manifest, splits = load_bank(args.bank)
    data = splits[args.split]
    task = manifest["configuration"]["task"]
    if not 1 <= args.cases <= len(data["observations"]) or args.seed < 0:
        parser.error("cases must select a nonempty prefix of the split; seed must be nonnegative")
    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    for case in range(args.cases):
        runs = []
        for run, (grid, particles) in enumerate(SETTINGS):
            seed = args.seed + 100 * case + run
            started = perf_counter()
            reference = heston_reference.reference_posterior(
                data["observations"][case], data["dt_years"], task,
                grid_size=grid, particles=particles, seed=seed)
            elapsed = perf_counter() - started
            moments = summary(reference)
            if run == len(SETTINGS) - 1:
                np.savez(args.out / f"case-{case}.npz", observations=data["observations"][case],
                         theta_grid=reference["theta_grid"], grid_probabilities=reference["grid_probabilities"],
                         terminal_particles=reference["terminal_particles"])
            runs.append({"grid_size": grid, "particles": particles, "seed": seed,
                         "seconds_including_compile_if_needed": elapsed,
                         "summary": moments, "diagnostics": reference["diagnostics"]})
        refinement = compare(runs[0]["summary"], runs[1]["summary"])
        repeat = compare(runs[1]["summary"], runs[2]["summary"])
        result = {"example_id": str(data["example_id"][case]), "runs": runs,
                  "coarse_to_fine": refinement, "independent_fine_repeat": repeat,
                  "passed": refinement["passed"] and repeat["passed"]}
        results.append(result)
        print(json.dumps({"example_id": result["example_id"], "passed": result["passed"],
                          "refinement": refinement, "repeat": repeat}), flush=True)
    report = {
        "data_origin": "synthetic", "measure": "P", "evidence_scope": "S reference-inference pilot",
        "bank_manifest_sha256": hashlib.sha256((args.bank / "manifest.json").read_bytes()).hexdigest(),
        "task": task, "split": args.split, "case_selection": f"first {args.cases} independent base histories",
        "source_sha256": {"heston_reference.py": hashlib.sha256(Path(heston_reference.__file__).read_bytes()).hexdigest(),
                          "check_heston_reference.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "thresholds": THRESHOLDS, "cases": results,
        "passed": all(row["passed"] for row in results),
        "limits": "Approximate quadrature/particle reference for the declared daily discrete model. Moment/correlation stability is diagnostic, not exact posterior certification; no continuous-time or market acceptance.",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["passed"]:
        raise SystemExit("Reference stability gate failed; inspect report before judging neural inference.")


if __name__ == "__main__":
    main()
