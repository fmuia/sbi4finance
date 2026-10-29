"""One reviewed Heston follow-up inside a four-CPU, 30-minute allocation.

This only sequences existing tools. It never resumes a sampler, launches another
case, accepts a posterior, or replaces the required manual trace/product review.
"""

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def commands(bank, checkpoint, index, out, *, tune=4000, draws=8000):
    """Exact production recipe; small overrides are for software tests only."""
    if index not in (0, 1, 2):
        raise ValueError("only the three declared development cases are supported")
    seed = 271026 + index
    return [
        ("reference", [sys.executable, str(ROOT / "scripts/calibration_reference.py"),
            "--bank", str(bank), "--split", "validation", "--index", str(index),
            "--out", str(out / "reference"), "--tune", str(tune), "--draws", str(draws),
            "--chains", "4", "--cores", "4", "--mp-context", "spawn", "--seed", str(seed),
            "--initialization", "data-fit", "--fit-starts", "4", "--fit-budget", "2000",
            "--checkpoint-every", "100", "--no-progress"]),
        ("precision", [sys.executable, str(ROOT / "scripts/check_heston_reference_precision.py"),
            "--reference", str(out / "reference"), "--points", "16", "--out", str(out / "precision")]),
        ("comparison", [sys.executable, str(ROOT / "scripts/compare_full_surface_reference.py"),
            "--checkpoint", str(checkpoint), "--reference", str(out / "reference"),
            "--precision", str(out / "precision/precision.json"), "--out", str(out / "comparison"),
            "--draws", "10000", "--seed", str(seed)]),
    ]


def validate_inputs(bank, checkpoint, index):
    """Match the old observations before spending the reference allocation."""
    import numpy as np
    from sbi4finance.calibration import bank_collection

    anchor = ROOT / f"evidence/pilot-results/2026-10-26/heston-reference-parallel/case-{index}/reference.json"
    original = json.loads(anchor.read_text())
    manifest, data = bank_collection.load_bank(bank, splits=["validation"], roles=["observations", "example_id"])
    split = data["validation"]
    task = manifest["configuration"]["task"]
    if (digest(bank / "manifest.json") != original["bank_manifest_sha256"]
            or task != original["task"] or original["settings"]["seed"] != 271026 + index
            or str(split["example_id"][index]) != original["surface_id"]
            or not np.array_equal(split["observations"][index, :, 0], original["observed_prices"])):
        raise ValueError("bank must match the original manifest, case, 63 quotes, task/noise and seed exactly")
    training = json.loads((checkpoint / "training.json").read_text())
    settings = training["run_contract"]["settings"]
    if (json.loads((checkpoint / "task.json").read_text()) != task
            or training["network"]["family"] != "spline" or settings["epochs"] != 300 or settings["seed"] != 0
            or digest(checkpoint / "model.keras") != training["checkpoint_sha256"]):
        raise ValueError("the matching 300-epoch seed-0 spline development checkpoint is required")
    return {"anchor_sha256": digest(anchor), "bank_manifest_sha256": digest(bank / "manifest.json"),
            "surface_id": original["surface_id"], "task": task, "observed_prices": original["observed_prices"],
            "checkpoint_sha256": training["checkpoint_sha256"],
            "checkpoint_metadata_sha256": {n: digest(checkpoint/n) for n in ("task.json", "training.json")}}


def guards(out):
    """Expose separate guards; none substitutes for reviewing the retained trace."""
    diagnostics = json.loads((out / "reference/diagnostics.json").read_text())
    precision = json.loads((out / "precision/precision.json").read_text())
    comparison = json.loads((out / "comparison/comparison.json").read_text())
    products = comparison.get("variance_product_comparison") or {}
    return {
        "parameter_mcmc_passed": diagnostics.get("mcmc_diagnostics_passed") is True,
        "analytic_product_mcmc_passed": diagnostics.get("variance_functionals", {}).get("mcmc_diagnostics_passed") is True,
        "all_precision_points_priced": precision.get("all_points_priced") is True,
        "precision_pricing_sources_match": precision.get("pricing_sources_match_reference") is True,
        "guarded_comparison_completed": comparison.get("status") == "completed_development_comparison",
        "comparison_product_diagnostics_passed": products.get("status") == "descriptive_development_comparison",
        "proposed_neural_parameter_checks_passed": (comparison.get("proposed_parameter_agreement") or {}).get("all_checks_passed"),
    }


def _write(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) < 4:
        raise ValueError("run through core_cpu.sbatch inside an allocation with at least four CPUs")
    if metadata.version("bayesflow") != "2.1.0":
        raise ValueError("the locked BayesFlow 2.1.0 core-reference runtime is required")
    stages = commands(args.bank, args.checkpoint, args.index, args.out)
    args.out.mkdir(parents=True, exist_ok=False)
    source_files = ["scripts/run_heston_reference_case.py", "scripts/calibration_reference.py",
                    "scripts/check_heston_reference_precision.py", "scripts/compare_full_surface_reference.py",
                    "src/sbi4finance/hkde.py", "src/sbi4finance/calibration/pricing.py", "pyproject.toml", "pixi.lock"]
    state = {"format": "heston_reference_case_run_v1", "status": "preflight", "case_index": args.index,
             "seed": 271026 + args.index, "scientific_acceptance": False, "sampler_resumable": False,
             "trace_and_product_review": "pending_manual_review", "launch_other_cases": False,
             "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "source_sha256": {name: digest(ROOT/name) for name in source_files},
             "versions": {name: metadata.version(name) for name in ("bayesflow", "pymc", "pytensor", "numpy", "scipy")},
             "slurm": {name: os.environ.get(name) for name in ("SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE")},
             "stages": [{"name": name, "command": command, "status": "pending"} for name, command in stages]}
    state_path = args.out / "run.json"
    _write(state_path, state)
    try:
        state["inputs"] = validate_inputs(args.bank, args.checkpoint, args.index)
        for row in state["stages"]:
            state["status"], row["status"] = row["name"], "running"
            _write(state_path, state)
            started = time.perf_counter()
            with (args.out / f"{row['name']}.log").open("w") as log:
                result = subprocess.run(row["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            row.update(returncode=result.returncode, seconds=time.perf_counter()-started,
                       status="completed" if result.returncode == 0 else "failed")
            _write(state_path, state)
            if result.returncode:
                raise RuntimeError(f"{row['name']} failed; retained log and partial outputs must be inspected")
        state["guards"] = guards(args.out)
        # Neural agreement is reported separately. A valid reference can expose
        # a weak estimator; that result must not be mislabeled as sampler failure.
        reference_checks = [value for key, value in state["guards"].items()
                            if key != "proposed_neural_parameter_checks_passed"]
        state["status"] = "ready_for_manual_review" if all(reference_checks) else "inconclusive_reference"
        state["next_cases_authorized"] = False
        _write(state_path, state)
        return 0 if all(reference_checks) else 2
    except Exception as error:
        state.update(status="failed", failure={"type": type(error).__name__, "error": str(error)})
        _write(state_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.bank, args.checkpoint, args.out = (p.resolve() for p in (args.bank, args.checkpoint, args.out))
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
