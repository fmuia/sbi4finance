"""Two fixed Heston fits and development checks inside their GPU allocations.

Sequences existing commands; never prices data, reruns MCMC, opens test arrays,
resumes an old run, selects between families, or claims scientific acceptance.
"""

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RECIPES = {"spline-1000": "configs/heston_20k_spline_1000epochs.json",
           "flow-matching-500": "configs/heston_20k_flow_matching_500epochs.json"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_inputs(bank, baseline, references, recipe):
    """Inspect metadata/checksums only; the trainer opens train/validation arrays."""
    manifest = json.loads((bank/"manifest.json").read_text())
    training = json.loads((baseline/"training.json").read_text())
    task = json.loads((baseline/"task.json").read_text())
    contract = training["run_contract"]
    if (digest(bank/"manifest.json") != recipe["bank_manifest_sha256"]
            or manifest["configuration"]["splits"] != recipe["splits"]
            or manifest["configuration"]["task"] != task
            or manifest["task_fingerprint"] != recipe["task_fingerprint"]
            or contract["dataset"]["manifest_sha256"] != recipe["bank_manifest_sha256"]
            or digest(baseline/"model.keras") != recipe["baseline_checkpoint_sha256"]
            or training["checkpoint_sha256"] != recipe["baseline_checkpoint_sha256"]
            or training["task_fingerprint"] != recipe["task_fingerprint"]
            or contract["settings"]["epochs"] != 300 or contract["settings"]["seed"] != 0):
        raise ValueError("the original20k/5k/5k Heston snapshot, exact task/noise and300-epoch baseline are required")
    environment = {name: digest(ROOT/name) for name in ("pyproject.toml", "pixi.lock")}
    if environment != contract["environment_sha256"]:
        raise ValueError("locked environment bytes differ from the existing BF2.1 baseline")
    runtime = {"python": platform.python_version(), "keras_backend_env": os.environ.get("KERAS_BACKEND"),
               **{name: metadata.version(name) for name in ("bayesflow", "keras", "jax", "numpy", "scipy")}}
    if runtime != contract["runtime"] or runtime["bayesflow"] != "2.1.0" or runtime["keras_backend_env"] != "jax":
        raise ValueError("the byte-locked BF2.1 baseline runtime and JAX backend are required")
    anchors = ROOT/"docs/pilot-results/2026-10-27/heston-reviewed"
    hashes = {}
    for index in recipe["reference_comparison"]["case_indices"]:
        case = references/f"case-{index}"
        reference = json.loads((case/"reference/reference.json").read_text())
        diagnostics = json.loads((case/"reference/diagnostics.json").read_text())
        precision = json.loads((case/"precision/precision.json").read_text())
        files = ("reference/reference.json", "reference/diagnostics.json", "precision/precision.json")
        if (any(digest(case/name) != digest(anchors/f"case-{index}"/name) for name in files)
                or reference["task"] != task or reference["surface_id"] != f"validation:{index}"
                or reference["settings"]["seed"] != 271026+index
                or diagnostics.get("mcmc_diagnostics_passed") is not True
                or diagnostics.get("variance_functionals", {}).get("mcmc_diagnostics_passed") is not True
                or precision.get("all_points_priced") is not True
                or precision.get("pricing_sources_match_reference") is not True
                or digest(case/"reference/chains.npz") != precision["input_sha256"]["chains.npz"]
                or any(digest(case/"reference"/name) != precision["input_sha256"][name]
                       for name in ("reference.json", "diagnostics.json"))):
            raise ValueError(f"case{index} must match the exact reviewed reference, diagnostics and precision inputs")
        hashes[f"case-{index}"] = {name: digest(case/name) for name in (*files, "reference/chains.npz")}
    return {"bank_manifest_sha256": digest(bank/"manifest.json"), "task_fingerprint": manifest["task_fingerprint"],
            "baseline_checkpoint_sha256": digest(baseline/"model.keras"), "runtime": runtime,
            "environment_sha256": environment, "reference_sha256": hashes,
            "test_access": "split counts in metadata only; no test arrays opened"}


def commands(args, recipe):
    checkpoint, budget = args.out/"checkpoint", recipe["execution_budget"]
    network, settings, evaluation = recipe["network"], recipe["training"], recipe["evaluation"]
    training = [sys.executable, str(ROOT/"scripts/train_full_surface.py"), "--bank", str(args.bank),
                "--out", str(checkpoint), "--family", network["family"], "--encoder", network["encoder"],
                "--widths", ",".join(map(str, network["widths"]))]
    for key in ("summary_dim", "flow_steps", "mean_steps"):
        training += ["--"+key.replace("_", "-"), str(network[key])]
    for key in ("epochs", "batch_size", "chunk_epochs", "seed"):
        training += ["--"+key.replace("_", "-"), str(settings[key])]
    validation = [sys.executable, str(ROOT/"scripts/evaluate_full_surface.py"), "--bank", str(args.bank),
                  "--checkpoint", str(checkpoint), "--out", str(args.out/"validation")]
    for key in ("split", "first", "surfaces", "draws", "batch_size", "seed"):
        validation += ["--"+key.replace("_", "-"), str(evaluation[key])]
    stages = [("environment", [sys.executable, str(ROOT/"scripts/check_full_surface_environment.py"), "--require-gpu"],
               budget["environment_seconds"]), ("training", training, budget["training_seconds"]),
              ("validation", validation, budget["validation_seconds"])]
    for index in recipe["reference_comparison"]["case_indices"]:
        case = args.references/f"case-{index}"
        stages.append((f"comparison-{index}", [sys.executable, str(ROOT/"scripts/compare_full_surface_reference.py"),
            "--checkpoint", str(checkpoint), "--reference", str(case/"reference"),
            "--precision", str(case/"precision/precision.json"), "--out", str(args.out/f"comparisons/case-{index}"),
            "--draws", str(recipe["reference_comparison"]["neural_draws"]),
            "--seed", str(recipe["reference_comparison"]["seed_base"]+index)], budget["comparison_seconds_per_case"]))
    return stages


def _write(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 4:
        raise ValueError("a four-CPU Slurm GPU allocation is required; no login-node training")
    if any(os.environ.get(name) != "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise ValueError("set each numerical-library thread variable to1 as in the core launchers")
    recipe_path = ROOT/RECIPES[args.fit]
    recipe = json.loads(recipe_path.read_text())
    started = time.perf_counter()
    deadline = started+recipe["execution_budget"]["total_seconds"]
    stages = commands(args, recipe)
    args.out.mkdir(parents=True, exist_ok=False)
    source_names = [RECIPES[args.fit], "scripts/run_heston_training_followup.py", "scripts/train_full_surface.py",
                    "scripts/evaluate_full_surface.py", "scripts/compare_full_surface_reference.py",
                    "src/sbi4finance/calibration/full_surface.py", "src/sbi4finance/calibration/inference.py"]
    state = {"format": "heston_training_followup_run_v1", "status": "preflight", "recipe": recipe,
             "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "source_sha256": {name: digest(ROOT/name) for name in source_names},
             "scientific_acceptance": False, "reserved_test_evaluated": False, "automatic_retry": False,
             "stages": [{"name": name, "command": command, "timeout_cap_seconds": cap, "status": "pending"}
                        for name, command, cap in stages]}
    state_path = args.out/"run.json"
    _write(state_path, state)
    try:
        state["inputs"] = validate_inputs(args.bank, args.baseline, args.references, recipe)
        for row in state["stages"]:
            remaining = deadline-time.perf_counter()
            if remaining <= 0:
                raise TimeoutError("whole-run wall budget exhausted; no automatic continuation")
            row.update(status="running", effective_timeout_seconds=min(remaining, row["timeout_cap_seconds"]))
            state["status"] = row["name"]
            _write(state_path, state)
            before = time.perf_counter()
            try:
                with (args.out/f"{row['name']}.log").open("w") as log:
                    result = subprocess.run(row["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=row["effective_timeout_seconds"], check=False)
                row.update(returncode=result.returncode, status="completed" if result.returncode == 0 else "failed")
                if result.returncode:
                    raise RuntimeError(f"{row['name']} failed; preserve its outputs and inspect the log")
            except subprocess.TimeoutExpired:
                row["status"] = "timeout"
                raise
            finally:
                row["seconds"] = time.perf_counter()-before
                _write(state_path, state)
        evaluation = json.loads((args.out/"validation/evaluation.json").read_text())
        if (evaluation.get("reserved_test_evaluated") is not False or evaluation.get("evaluated_split") != "validation"
                or evaluation.get("example_ids") != [f"validation:{i}" for i in range(500)]):
            raise ValueError("the evaluation must retain exactly the declared500 development validation IDs")
        state["development_comparisons"] = {}
        for index in recipe["reference_comparison"]["case_indices"]:
            comparison = json.loads((args.out/f"comparisons/case-{index}/comparison.json").read_text())
            state["development_comparisons"][f"validation:{index}"] = {
                "status": comparison["status"],
                "proposed_parameter_checks_passed": (comparison.get("proposed_parameter_agreement") or {}).get("all_checks_passed"),
                "analytic_products": comparison.get("variance_product_comparison")}
        state.update(status="completed_development_checks", seconds=time.perf_counter()-started,
                     interpretation="Within-family validation selection only; failed neural checks are retained, not rerun or accepted.")
        _write(state_path, state)
        return state
    except Exception as error:
        state.update(status="incomplete" if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)) else "failed",
                     seconds=time.perf_counter()-started, failure={"type": type(error).__name__, "error": str(error)})
        _write(state_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", choices=RECIPES)
    for name in ("bank", "baseline", "references", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    for name in ("bank", "baseline", "references", "out"):
        setattr(args, name, getattr(args, name).resolve())
    result = run(args)
    print(json.dumps({"status": result["status"], "scientific_acceptance": False}))


if __name__ == "__main__":
    main()
