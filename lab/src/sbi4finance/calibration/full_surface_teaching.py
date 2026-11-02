"""Small offline tutorial replay bundles; importing initializes no neural backend.

The exported final-epoch checkpoint is a derived publication artifact. Original
training metadata and historical checkpoint selection are preserved separately.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

# Shared implementations are owned by the tutorial runtime.
from sbi4f_tutorial.shared.full_surface_teaching import (
    NAMES,
    CACHE_FILES,
    ROUGH_NAMES,
    NEURAL_FILES,
    digest,
    fingerprint,
    require,
    heston_task,
    variance_products,
    load_cache,
    rough_task,
    load_neural_replay,
    load_tutorial_artifact,
    load_tutorial_checkpoint,
    rough_variance_products,
)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def export_checkpoint(source, out, *, gate, product_checks):
    """Publish a new final-500 directory without mutating the selected-402 run."""
    source, out, gate, product_checks = map(Path, (source, out, gate, product_checks))
    require(not out.exists(), "output must be a new directory")
    task = heston_task(json.loads((source / "task.json").read_text()))
    original = json.loads((source / "training.json").read_text())
    resume = json.loads((source / "resume.json").read_text())
    candidate = source / resume["checkpoint_files"]["last"]
    model_sha = digest(candidate)
    result = json.loads(gate.read_text())
    products = json.loads(product_checks.read_text())
    require(original["format"] == "full_surface_training_v1" and original["task_fingerprint"] == fingerprint(task),
            "source task/training mismatch")
    require(digest(source / "model.keras") == original["checkpoint_sha256"], "source selected model changed")
    require(resume["epochs_done"] == 500 and original["selected_epoch"] == 402,
            "this publication expects the audited selected402/final500 run")
    require(result.get("checkpoint_sha256") == model_sha and result.get("rank_screen_passed") is True
            and result.get("reserved_test_evaluated") is True and result.get("scientific_acceptance") is False,
            "a passed scoped frozen-candidate gate for these exact model bytes is required")
    require(products.get("candidate_sha256") == model_sha and products.get("scientific_acceptance") is False,
            "analytic product evidence must identify the same scoped candidate")
    out.mkdir(parents=True)
    for name, path in (("model.keras", candidate), ("task.json", source / "task.json"),
                       ("origin-training.json", source / "training.json"), ("gate.json", gate),
                       ("analytic-product-checks.json", product_checks)):
        shutil.copyfile(path, out / name)
    derived = {**original, "checkpoint_sha256": model_sha, "selected_epoch": 500,
               "exported_candidate_epoch": 500, "original_validation_selected_epoch": 402,
               "selection": "post-hoc development-selected final endpoint, frozen before scoped held-out gate",
               "same_seed_reload_exact": False, "scientific_validation": False,
               "export_provenance": {"origin_training_sha256": digest(source / "training.json"),
                   "origin_selected_checkpoint_sha256": original["checkpoint_sha256"],
                   "source_final_checkpoint": str(candidate), "gate_sha256": digest(gate),
                   "analytic_product_checks_sha256": digest(product_checks),
                   "seeded_export_reload_verified": False,
                   "scope": "one frozen candidate; not full tutorial, joint or path-product acceptance"}}
    write(out / "training.json", derived)
    publication = {"format": "full_surface_checkpoint_export_v1", "scientific_acceptance": False,
                   "task_fingerprint": fingerprint(task), "candidate_sha256": model_sha,
                   "exported_candidate_epoch": 500, "original_validation_selected_epoch": 402,
                   "files": {p.name: digest(p) for p in sorted(out.iterdir())}}
    write(out / "export.json", publication)
    return publication


def prepare_cache(checkpoint, declaration, case_index, out):
    """Replay a previously inspected development case; no bank/model load or inference."""
    checkpoint, declaration, out = map(Path, (checkpoint, declaration, out))
    require(not out.exists(), "output must be a new directory")
    export = json.loads((checkpoint / "export.json").read_text())
    for name, sha in export["files"].items():
        require(digest(checkpoint / name) == sha, f"changed exported file: {name}")
    task = heston_task(json.loads((checkpoint / "task.json").read_text()))
    binding = json.loads(declaration.read_text())
    require(binding["candidate_sha256"] == export["candidate_sha256"], "candidate/declaration mismatch")
    cases = [c for c in binding["development_cases"] if c["case"] == case_index]
    require(len(cases) == 1, "expected one bound development case")
    case = cases[0]
    reference, neural = Path(case["reference"]), Path(case["neural_draws"])
    inputs = [reference / n for n in ("reference.json", "diagnostics.json", "chains.npz")] + [neural]
    for path in inputs:
        require(binding["frozen_sha256"].get(str(path)) == digest(path), f"unbound or changed cache source: {path}")
    record = json.loads((reference / "reference.json").read_text())
    require(record["task"] == task and record.get("split") != "test"
            and not str(record.get("surface_id", "")).startswith("test:"), "cache must be matched development data")
    with np.load(reference / "chains.npz", allow_pickle=False) as data:
        chains = data["parameters"].copy()
    with np.load(neural, allow_pickle=False) as data:
        draws = data["final"].copy()
    arrays = {"observations": np.asarray(record["observed_prices"], dtype=float),
              "neural_draws": draws, "reference_chains": chains}
    for name, value in variance_products(task, draws).items(): arrays["neural_" + name] = value
    for name, value in variance_products(task, chains).items(): arrays["reference_" + name] = value
    out.mkdir(parents=True)
    shutil.copyfile(checkpoint / "task.json", out / "task.json")
    for name in ("reference.json", "diagnostics.json"): shutil.copyfile(reference / name, out / name)
    all_products = json.loads((checkpoint / "analytic-product-checks.json").read_text())
    selected = [row for row in all_products["cases"] if row["case"] == case_index]
    require(len(selected) == 1, "missing analytic product evidence for development case")
    write(out / "products.json", {"case": case_index, "scientific_acceptance": False, **selected[0]})
    np.savez_compressed(out / "posterior.npz", **arrays)
    report = {"format": "full_surface_teaching_v1", "role": "cached_development_example",
              "case_id": record["surface_id"], "parameter_names": NAMES, "task_fingerprint": fingerprint(task),
              "checkpoint_sha256": export["candidate_sha256"], "exported_candidate_epoch": 500,
              "original_validation_selected_epoch": 402, "scientific_acceptance": False,
              "checkpoint_status": "frozen_candidate_gate_passed_not_full_acceptance",
              "neural_draw_count": len(draws), "reference_chain_shape": list(chains.shape),
              "source_sha256": {str(p): digest(p) for p in [declaration, *inputs]},
              "files": {name: digest(out / name) for name in sorted(CACHE_FILES)},
              "limitations": "Previously inspected development case. Analytic variance rates only; no path-product, "
                             "market, training-seed or universal conditional acceptance. This initial cache is not "
                             "the prospective interior teaching example."}
    write(out / "bundle.json", report)
    load_cache(out)
    return report


def prepare_development_cache(checkpoint, case_root, out):
    """Package the completed interior case with its own descriptive products."""
    checkpoint, case_root, out = map(Path, (checkpoint, case_root, out))
    require(not out.exists(), "output must be a new directory")
    export = json.loads((checkpoint / "export.json").read_text())
    for name, sha in export["files"].items():
        require(digest(checkpoint / name) == sha, f"changed exported file: {name}")
    task = heston_task(json.loads((checkpoint / "task.json").read_text()))
    run = json.loads((case_root / "run.json").read_text())
    declaration = json.loads((case_root / "manifest.json").read_text())
    require(run.get("format") == "heston_interior_development_run_v1"
            and run.get("status") == "completed_development_case"
            and run.get("frozen_inputs_verified") is True
            and run.get("scientific_acceptance") is False, "completed checked interior development case required")
    require(run["manifest_sha256"] == digest(case_root / "manifest.json")
            and declaration["candidate_sha256"] == export["candidate_sha256"], "interior declaration/candidate mismatch")
    for name, sha in run["outputs"].items():
        require(digest(case_root / name) == sha, f"changed interior output: {name}")
    interior = json.loads((case_root / "interior-check.json").read_text())
    diagnostics = json.loads((case_root / "reference/diagnostics.json").read_text())
    require(interior.get("interior_passed") is True and diagnostics.get("mcmc_diagnostics_passed") is True
            and run["stages"]["precision"].get("all_points_priced") is True,
            "interior, reference convergence and precision checks must pass")
    reference = json.loads((case_root / "reference/reference.json").read_text())
    neural_record = json.loads((case_root / "neural/run.json").read_text())
    require(reference["task"] == task and reference.get("split") == "development"
            and neural_record["candidate_sha256"] == export["candidate_sha256"], "interior task/readout mismatch")
    with np.load(case_root / "neural/draws.npz", allow_pickle=False) as data: neural = data["parameters"].copy()
    with np.load(case_root / "reference/chains.npz", allow_pickle=False) as data: chains = data["parameters"].copy()
    with np.load(case_root / "observation.npz", allow_pickle=False) as data: observations = data["observed_prices"].copy()
    require(np.array_equal(observations, reference["observed_prices"]), "interior observation/reference mismatch")
    arrays = {"observations": observations, "neural_draws": neural, "reference_chains": chains}
    products = {}
    for prefix, values in (("neural_", neural), ("reference_", chains)):
        for name, value in variance_products(task, values).items():
            arrays[prefix + name] = value
            products[prefix + name] = {"mean": float(value.mean()), "q10_q90": np.quantile(value, [.1, .9]).tolist()}
    out.mkdir(parents=True)
    shutil.copyfile(checkpoint / "task.json", out / "task.json")
    shutil.copyfile(case_root / "reference/reference.json", out / "reference.json")
    shutil.copyfile(case_root / "reference/diagnostics.json", out / "diagnostics.json")
    write(out / "products.json", {"scientific_acceptance": False, "scope": "descriptive analytic products for this "
          "interior development case; no inherited boundary-case product gate or new quantile acceptance rule",
          "units": "annualized variance", "products": products})
    np.savez_compressed(out / "posterior.npz", **arrays)
    inputs = [case_root / n for n in ("run.json", "manifest.json", *run["outputs"])]
    report = {"format": "full_surface_teaching_v1", "role": "cached_development_example",
              "case_id": reference["surface_id"], "parameter_names": NAMES, "task_fingerprint": fingerprint(task),
              "checkpoint_sha256": export["candidate_sha256"], "exported_candidate_epoch": 500,
              "original_validation_selected_epoch": 402, "scientific_acceptance": False,
              "checkpoint_status": "frozen_candidate_gate_passed_not_full_acceptance",
              "neural_draw_count": len(neural), "reference_chain_shape": list(chains.shape),
              "source_sha256": {str(p): digest(p) for p in inputs},
              "files": {name: digest(out / name) for name in sorted(CACHE_FILES)},
              "limitations": "Ex-ante midpoint development example; convergence/refinement/interior checks passed. "
                             "Analytic products descriptive only, not inherited case0 gates or reserved acceptance."}
    write(out / "bundle.json", report)
    load_cache(out)
    return report


def prepare_rough_replay(run_root, snapshot, out, case_index=0):
    """Copy an explicit saved validation case; no model load or new inference.

    The default retains the original validation:0 fixture. Selecting a different
    case does not certify its geometry or conditional accuracy. Only its native
    part and saved posterior batch are decoded; a combined truth panel is also
    checked when available.
    """
    run_root, snapshot, out = map(Path, (run_root, snapshot, out))
    require(not out.exists(), "output must be a new directory")
    require(not isinstance(case_index, (bool, np.bool_)) and isinstance(case_index, (int, np.integer))
            and case_index >= 0, "case index must be a nonnegative integer")
    case_index = int(case_index); case_id = f"validation:{case_index}"
    task = rough_task(json.loads((run_root / "posterior/task.json").read_text()))
    training = json.loads((run_root / "posterior/training.json").read_text())
    evaluation = json.loads((run_root / "validation/evaluation.json").read_text())
    run = json.loads((run_root / "run.json").read_text())
    manifest = json.loads((snapshot / "manifest.json").read_text())
    require(run.get("status") == "completed" and run.get("frozen_inputs_verified") is True
            and run.get("reserved_test_evaluated") is False and run.get("scientific_acceptance") is False,
            "completed development run with frozen inputs required")
    require(manifest["configuration"]["task"] == task
            and digest(snapshot / "manifest.json") == run["bank"]["manifest_sha256"]
            and training["task_fingerprint"] == fingerprint(task) == evaluation["task_fingerprint"]
            and training["checkpoint_sha256"] == evaluation["checkpoint_sha256"] == run["posterior_sha256"]
            and evaluation.get("evaluated_split") == "validation" and evaluation.get("reserved_test_evaluated") is False,
            "bank/model/validation identity differs")
    ids = evaluation["example_ids"]
    require(len(ids) == len(set(ids)) and case_id in ids, "requested validation case is absent or ambiguous")
    evaluation_row = ids.index(case_id)
    selected = [p for p in manifest["parts"] if p["split"] == "validation"
                and p["start"] <= case_index < p["stop"]]
    require(len(selected) == 1, "requested validation part missing or ambiguous")
    part = selected[0]; part_root = (snapshot / part["path"]).resolve()
    part_row = case_index - part["start"]
    collection = (snapshot / manifest["collection"]["path"]).resolve()
    require(part_root == collection / "parts" / f"validation-{part['start']:010d}-{part['stop']:010d}"
            and digest(part_root / "manifest.json") == part["manifest_sha256"]
            and digest(part_root / "arrays.npz") == part["array_sha256"], "validation part identity differs")
    with np.load(part_root / "arrays.npz", allow_pickle=False) as saved:
        require(str(saved["example_id"][part_row]) == case_id, "validation row identity differs")
        arrays = {"observations": saved["observations"][part_row, :, 0].copy(),
                  "truth": saved["parameters"][part_row].copy()}
        if task["task_id"] == "rough_calibration_v1":
            arrays["context_values"] = saved["context_values"][part_row].copy()
    seed = evaluation.get("seed", 0)
    require(type(seed) is int and seed >= 0, "invalid saved validation sampling seed")
    batches = []
    for path in (run_root / "validation-partial").glob("batch-*.npz"):
        suffix = path.stem.removeprefix("batch-")
        require(suffix.isdigit() and path.name == f"batch-{int(suffix):04d}.npz", "invalid saved batch name")
        if int(suffix) <= seed + case_index: batches.append((int(suffix), path))
    require(batches, "saved posterior batch missing for requested case")
    batch_seed, partial = max(batches, key=lambda item: item[0])
    with np.load(partial, allow_pickle=False) as saved:
        first = np.asarray(saved["first"])
        require(first.shape == () and first.dtype.kind in "iu" and int(first) == batch_seed,
                "saved draw batch starts at a different validation row")
        draws = saved["draws"]; batch_row = case_index - (batch_seed - seed)
        require(draws.ndim == 3 and draws.shape[2] == len(task["parameter_names"])
                and 0 <= batch_row < len(draws), "saved posterior batch does not contain requested case")
        arrays["neural_draws"] = draws[batch_row].copy()
    combined = run_root / "validation/draws.npz"
    if combined.exists():
        # Read truth only: NumPy does not decode the large combined draw member.
        with np.load(combined, allow_pickle=False) as saved:
            truth = saved["truth"]
            require(truth.shape == (len(ids), len(task["parameter_names"]))
                    and np.array_equal(truth[evaluation_row], arrays["truth"]),
                    "saved evaluation truth differs from bank row")
    out.mkdir(parents=True)
    for name, source in (("task.json", "posterior/task.json"), ("training.json", "posterior/training.json"),
                         ("evaluation.json", "validation/evaluation.json")):
        shutil.copyfile(run_root / source, out / name)
    np.savez_compressed(out / "posterior.npz", **arrays)
    sources = [run_root / n for n in ("run.json", "posterior/task.json", "posterior/training.json",
               "validation/evaluation.json")] + [snapshot / "manifest.json", part_root / "manifest.json",
               part_root / "arrays.npz", partial] + ([combined] if combined.exists() else [])
    report = {"format": "full_surface_neural_replay_v1", "role": "cached_development_neural_replay",
        "case_id": case_id, "validation_case_index": case_index, "evaluation_row": evaluation_row,
        "saved_batch_seed": batch_seed, "saved_batch_row": batch_row,
        "combined_truth_crosscheck": combined.exists(),
        "selection": ("fixed first development validation row; no case search" if case_index == 0 else
                      "explicitly requested development row; external selection criteria are not certified by this export"),
        "task_fingerprint": fingerprint(task), "parameter_names": task["parameter_names"],
        "checkpoint_sha256": training["checkpoint_sha256"], "selected_epoch": training["selected_epoch"],
        "bank_manifest_sha256": digest(snapshot / "manifest.json"), "neural_draw_count": len(arrays["neural_draws"]),
        "known_curve_is_observed": task["task_id"] == "rough_calibration_v1",
        "scientific_acceptance": False, "reserved_test_evaluated": False, "conditional_reference_available": False,
        "source_sha256": {str(p): digest(p) for p in sources},
        "builder_source_sha256": digest(__file__),
        "files": {name: digest(out / name) for name in sorted(NEURAL_FILES)},
        "limitations": "Development replay only, possibly prior-truncated. This export does not certify interior-case selection, "
            "PyMC agreement, independent pricing/refinement, or downstream-product acceptance. "
            "Joint generating curves are intentionally absent from inference inputs."}
    write(out / "bundle.json", report); load_neural_replay(out)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export-checkpoint")
    for name in ("source", "out", "gate", "product-checks"): p.add_argument("--" + name, type=Path, required=True)
    p = sub.add_parser("cache")
    for name in ("checkpoint", "declaration", "out"): p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--case-index", type=int, default=0)
    p = sub.add_parser("cache-development")
    for name in ("checkpoint", "case-root", "out"): p.add_argument("--" + name, type=Path, required=True)
    p = sub.add_parser("rough-replay")
    for name in ("run-root", "snapshot", "out"): p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--case-index", type=int, default=0, help="explicit validation ID; defaults to the existing case0 fixture")
    args = vars(parser.parse_args()); command = args.pop("command")
    if command == "export-checkpoint": result = export_checkpoint(**args)
    elif command == "cache-development": result = prepare_development_cache(**args)
    elif command == "rough-replay": result = prepare_rough_replay(**args)
    else: result = prepare_cache(**args)
    print(json.dumps({"format": result["format"], "scientific_acceptance": False}))


if __name__ == "__main__":
    main()
