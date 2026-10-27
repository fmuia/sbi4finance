"""One matched 50k Heston fit, validation panel and three cached references.

Sequences existing commands in one bounded GPU allocation. No pricing, MCMC,
reserved test evaluation, automatic continuation or scientific acceptance.
"""
import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_heston_training_followup import digest, _write  # noqa: E402

RECIPE = ROOT / "configs/heston_backbone_50k_fit.json"
ANCHORS = ROOT / "docs/pilot-results/2026-10-27/heston-reviewed"
BASELINE = ROOT / "docs/pilot-results/2026-10-27/followup/heston/flow-matching-500/checkpoint"


def validate_inputs(bank, references, recipe):
    """Freeze snapshot/task/runtime/reference bytes without opening test arrays."""
    manifest = json.loads((bank / "manifest.json").read_text())
    task = json.loads((BASELINE / "task.json").read_text())
    contract = json.loads((BASELINE / "training.json").read_text())["run_contract"]
    if (digest(bank / "manifest.json") != recipe["bank_manifest_sha256"]
            or manifest["configuration"]["splits"] != recipe["splits"]
            or manifest["configuration"]["task"] != task
            or manifest["task_fingerprint"] != recipe["task_fingerprint"]):
        raise ValueError("the checked50k/5k/5k snapshot and original Heston task are required")
    environment = {name: digest(ROOT / name) for name in ("pyproject.toml", "pixi.lock")}
    runtime = {"python": platform.python_version(), "keras_backend_env": os.environ.get("KERAS_BACKEND"),
               **{name: metadata.version(name) for name in ("bayesflow", "keras", "jax", "numpy", "scipy")}}
    if (environment != contract["environment_sha256"] or runtime != contract["runtime"]
            or runtime["bayesflow"] != "2.1.0" or runtime["keras_backend_env"] != "jax"):
        raise ValueError("the locked BF2.1/JAX environment used by the development baseline is required")
    hashes = {str(bank / "manifest.json"): digest(bank / "manifest.json")}
    cases = []
    for index in recipe["reference_comparison"]["case_indices"]:
        case = references / f"case-{index}"
        reference = json.loads((case / "reference/reference.json").read_text())
        diagnostics = json.loads((case / "reference/diagnostics.json").read_text())
        precision = json.loads((case / "precision/precision.json").read_text())
        files = ("reference/reference.json", "reference/diagnostics.json", "reference/chains.npz", "precision/precision.json")
        if (any(digest(case / name) != digest(ANCHORS / f"case-{index}" / name) for name in files)
                or reference["task"] != task or reference["surface_id"] != f"validation:{index}"
                or reference["settings"]["seed"] != 271026 + index
                or diagnostics.get("mcmc_diagnostics_passed") is not True
                or diagnostics.get("variance_functionals", {}).get("mcmc_diagnostics_passed") is not True
                or precision.get("all_points_priced") is not True
                or precision.get("pricing_sources_match_reference") is not True):
            raise ValueError(f"case{index} must be the exact completed reviewed reference")
        hashes.update({str(case / name): digest(case / name) for name in files})
        cases.append({"index": index, "surface_id": reference["surface_id"],
                      "observation_bank_manifest_sha256": reference["bank_manifest_sha256"],
                      "observed_prices": reference["observed_prices"]})
    return {"bank_manifest_sha256": recipe["bank_manifest_sha256"], "task_fingerprint": recipe["task_fingerprint"],
            "environment_sha256": environment, "runtime": runtime, "sha256": hashes, "cases": cases,
            "scope": "reference observations are the original readiness cases; train/validation snapshot is a distinct collection; test counts metadata only"}


def commands(args, recipe):
    """Construct only the frozen fit, development evaluation and cached checks."""
    checkpoint, budget = args.out / "checkpoint", recipe["execution_budget"]
    network, training, evaluation = recipe["network"], recipe["training"], recipe["evaluation"]
    fit = [sys.executable, str(ROOT / "scripts/train_full_surface.py"), "--bank", str(args.bank),
           "--out", str(checkpoint), "--family", network["family"], "--encoder", network["encoder"],
           "--widths", ",".join(map(str, network["widths"]))]
    # None means OMIT this flag, not the invalid string '--flow-steps None'.
    for key in ("summary_dim", "flow_steps", "mean_steps"):
        if network[key] is not None:
            fit += ["--" + key.replace("_", "-"), str(network[key])]
    for key in ("epochs", "batch_size", "chunk_epochs", "seed"):
        fit += ["--" + key.replace("_", "-"), str(training[key])]
    evaluate = [sys.executable, str(ROOT / "scripts/evaluate_full_surface.py"), "--bank", str(args.bank),
                "--checkpoint", str(checkpoint), "--out", str(args.out / "validation")]
    for key in ("split", "first", "surfaces", "draws", "batch_size", "seed"):
        evaluate += ["--" + key.replace("_", "-"), str(evaluation[key])]
    stages = [("environment", [sys.executable, str(ROOT / "scripts/check_full_surface_environment.py"), "--require-gpu"], budget["environment_seconds"]),
              ("training", fit, budget["training_seconds"])]
    for index in recipe["reference_comparison"]["case_indices"]:
        case = args.references / f"case-{index}"
        stages.append((f"comparison-{index}", [sys.executable, str(ROOT / "scripts/compare_full_surface_reference.py"),
            "--checkpoint", str(checkpoint), "--reference", str(case / "reference"),
            "--precision", str(case / "precision/precision.json"), "--out", str(args.out / f"comparisons/case-{index}"),
            "--draws", str(recipe["reference_comparison"]["neural_draws"]),
            "--seed", str(recipe["reference_comparison"]["seed_base"] + index)], budget["comparison_seconds_per_case"]))
    # Preserve matched-reference results even if the larger validation panel
    # uses its remaining allowance. Each maximum is clipped by the whole-run deadline.
    stages.append(("validation", evaluate, budget["validation_seconds"]))
    return stages


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 4:
        raise ValueError("one four-CPU GPU Slurm allocation is required")
    if any(os.environ.get(name) != "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise ValueError("numerical-library thread variables must equal1")
    recipe = json.loads(RECIPE.read_text()); started = time.perf_counter()
    deadline = started + recipe["execution_budget"]["total_seconds"]
    args.out.mkdir(parents=True, exist_ok=False)
    sources = [RECIPE, Path(__file__), ROOT / "scripts/train_full_surface.py", ROOT / "scripts/evaluate_full_surface.py",
               ROOT / "scripts/compare_full_surface_reference.py", ROOT / "src/sbi4finance/calibration/full_surface.py"]
    state = {"format": "heston_backbone_fit_run_v1", "status": "preflight", "recipe": recipe,
             "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
             "scientific_acceptance": False, "reserved_test_evaluated": False, "automatic_retry": False,
             "stages": [{"name": name, "command": command, "timeout_cap_seconds": cap, "status": "pending"}
                        for name, command, cap in commands(args, recipe)]}
    path = args.out / "run.json"; _write(path, state)
    def interrupted(signum, frame):
        raise SystemExit(f"received signal{signum}; preserve completed outputs, no retry")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        state["inputs"] = validate_inputs(args.bank, args.references, recipe)
        _write(path, state)
        for row in state["stages"]:
            remaining = deadline - time.perf_counter()
            if remaining <= 0: raise TimeoutError("whole-run allowance exhausted")
            row.update(status="running", effective_timeout_seconds=min(remaining, row["timeout_cap_seconds"]))
            state["status"] = row["name"]; _write(path, state); before = time.perf_counter()
            try:
                with (args.out / f"{row['name']}.log").open("w") as log:
                    result = subprocess.run(row["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=row["effective_timeout_seconds"], check=False)
                row.update(returncode=result.returncode, status="completed" if result.returncode == 0 else "failed")
                if result.returncode: raise RuntimeError(f"{row['name']} failed; inspect retained log")
            except subprocess.TimeoutExpired:
                row["status"] = "timeout"; raise
            finally:
                row["seconds"] = time.perf_counter() - before; _write(path, state)
        evaluation = json.loads((args.out / "validation/evaluation.json").read_text())
        ids = [f"validation:{i}" for i in range(recipe["evaluation"]["first"], recipe["evaluation"]["first"] + recipe["evaluation"]["surfaces"])]
        training = json.loads((args.out / "checkpoint/training.json").read_text())
        if (evaluation.get("reserved_test_evaluated") is not False or evaluation.get("example_ids") != ids
                or training.get("flow_matching_integrate_kwargs") != {"method": "tsit5", "steps": "adaptive"}
                or training.get("training_surfaces") != recipe["splits"]["train"]):
            raise ValueError("completed outputs differ from the declared data/native policy")
        if any(digest(Path(p)) != h for p, h in state["inputs"]["sha256"].items()):
            raise ValueError("frozen inputs changed during the unit")
        if any(digest(ROOT / name) != value for name, value in state["source_sha256"].items()):
            raise ValueError("frozen source/config bytes changed during the unit")
        state["development_comparisons"] = {str(i): json.loads((args.out / f"comparisons/case-{i}/comparison.json").read_text())["proposed_parameter_agreement"] for i in recipe["reference_comparison"]["case_indices"]}
        state.update(status="completed_development_checks", seconds=time.perf_counter() - started,
                     interpretation="Selected50k fit; failures retained.2000 development validation cases plus3conditional cases do not constitute full tutorial acceptance.")
        _write(path, state); return state
    except BaseException as error:
        state.update(status="incomplete" if isinstance(error, (TimeoutError, subprocess.TimeoutExpired, SystemExit)) else "failed", seconds=time.perf_counter() - started,
                     failure={"type": type(error).__name__, "error": str(error)})
        _write(path, state); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "references", "out"): parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    for name in ("bank", "references", "out"): setattr(args, name, getattr(args, name).resolve())
    state = run(args); print(json.dumps({"status": state["status"], "scientific_acceptance": False}))
if __name__ == "__main__": main()
