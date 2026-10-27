"""Sample-based core diagnostics on declared validation or frozen test data.

This reports simulator calibration/recovery, not conditional MCMC agreement or
market validity. Use validation during development; reserve test for acceptance.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbi4finance.calibration import bank_collection as bank, full_surface, validation  # noqa: E402


def evaluate(args):
    task, workflow, record = full_surface.load_checkpoint(args.checkpoint)
    manifest, splits = bank.load_bank(args.bank, splits=(args.split,), roles=bank.TRAINING_ROLES)
    if task != manifest["configuration"]["task"] or record["task_fingerprint"] != manifest["task_fingerprint"]:
        raise ValueError("bank and checkpoint tasks differ")
    split = splits[args.split]
    if args.first < 0 or args.surfaces < 1 or args.draws < 2 or args.batch_size < 1 or args.seed < 0:
        raise ValueError("nonnegative first/seed, positive sizes and at least two draws required")
    stop = args.first + args.surfaces
    if stop > len(split["parameters"]):
        raise ValueError("requested panel exceeds the split; no silent truncation")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    curves = split.get("context_values")
    draws = np.concatenate([full_surface.sample(workflow, task, split["observations"][start:min(start + args.batch_size, stop)],
                                  args.draws, encoder=record["network"]["encoder"], seed=args.seed + start,
                                  context_values=None if curves is None else curves[start:min(start + args.batch_size, stop)])
                             for start in range(args.first, stop, args.batch_size)])
    truth = split["parameters"][args.first:stop]
    bounds = np.asarray(task["prior_bounds"])
    sbc = validation.sbc(draws, truth, seed=args.seed)
    correlations = validation.recovery(draws, truth)["correlation"] if len(truth) > 1 else [None] * len(bounds)
    correlations = [float(value) if value is not None and np.isfinite(value) else None for value in correlations]
    report = {"format": "full_surface_evaluation_v1", "task_fingerprint": record["task_fingerprint"],
              "checkpoint_sha256": record["checkpoint_sha256"], "network": record["network"],
              "bank_manifest_sha256": hashlib.sha256((args.bank / "manifest.json").read_bytes()).hexdigest(),
              "evaluated_split": args.split, "reserved_test_evaluated": args.split == "test",
              "example_ids": split["example_id"][args.first:stop].tolist(), "seed": args.seed,
              "draws_per_surface": args.draws, "targets": task["parameter_names"],
              "sbc": {k: v for k, v in sbc.items() if k != "fractional_ranks"},
              "coverage": validation.coverage(draws, truth),
              "contraction_logit": dict(zip(task["parameter_names"], validation.contraction(draws, bounds))),
              "recovery_correlation": dict(zip(task["parameter_names"], correlations)),
              "scope": "synthetic complete-design diagnostics; not conditional reference agreement or scientific acceptance"}
    if task["model"] == "rough":
        eta, rho = (task["parameter_names"].index(name) for name in ("eta", "rho"))
        diagnostic_draws = np.concatenate([draws, (draws[..., eta] * draws[..., rho])[..., None]], axis=-1)
        diagnostic_truth = np.column_stack([truth, truth[:, eta] * truth[:, rho]])
        # Keep the existing parameter-only fields and saved arrays unchanged.
        # This separate simultaneous family includes the derived rho*eta target,
        # matching the original known-curve course calibration screen.
        diagnostic_sbc = validation.sbc(diagnostic_draws, diagnostic_truth, seed=args.seed)
        report["diagnostics_with_derived_targets"] = {
            "targets": [*task["parameter_names"], "rho_eta"],
            "sbc": {k: v for k, v in diagnostic_sbc.items() if k != "fractional_ranks"},
            "coverage": validation.coverage(diagnostic_draws, diagnostic_truth),
            "scope": "one simultaneous rank-screen family containing primary parameters and rho*eta; "
                     "no uniform-prior assumption or logit contraction is applied to the derived target"}
    report["seconds"] = time.perf_counter() - started
    (args.out / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.out / "draws.npz", draws=draws, truth=truth)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--surfaces", type=int, default=100)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    result = evaluate(parser.parse_args())
    print(json.dumps({"surfaces": len(result["example_ids"]), "evaluated_split": result["evaluated_split"],
                      "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
