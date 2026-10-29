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

NAMES = ["initial_variance", "theta", "vol_of_vol", "rho"]
CACHE_FILES = {"task.json", "posterior.npz", "reference.json", "diagnostics.json", "products.json"}
ROUGH_NAMES = {"rough_calibration_v1": ["hurst", "eta", "rho"],
               "rough_joint_calibration_v1": ["hurst", "eta", "rho", "short_variance", "long_variance", "decay"]}
NEURAL_FILES = {"task.json", "posterior.npz", "evaluation.json", "training.json"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(task):
    return hashlib.sha256((json.dumps(task, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def heston_task(task):
    require(task.get("model") == "hkde" and task.get("parameter_names") == NAMES,
            "this cache supports the four-target Heston task only")
    context = task.get("context", {})
    contracts = context.get("contracts", [])
    noise = np.asarray(context.get("quote_noise_std"), dtype=float)
    bounds = np.asarray(task.get("prior_bounds"), dtype=float)
    require(len(contracts) == 63 and noise.shape == (63,) and np.isfinite(noise).all() and np.all(noise > 0),
            "complete ordered 63-contract design and positive quote noise required")
    require(bounds.shape == (4, 2) and np.isfinite(bounds).all() and np.all(bounds[:, 0] < bounds[:, 1]),
            "invalid parameter bounds")
    require(context.get("spot") == 1 and context.get("rate") == 0 and context.get("dividend") == 0,
            "expected spot-one zero-carry task")
    require(task.get("fixed_parameters", {}).get("kappa", 0) > 0
            and task["fixed_parameters"].get("jump_intensity") == 0, "zero-jump fixed-kappa Heston required")
    for contract in contracts:
        require(contract.get("option_type") in ("call", "put")
                and np.isfinite([contract.get("strike", np.nan), contract.get("maturity", np.nan)]).all()
                and min(contract["strike"], contract["maturity"]) > 0, "invalid contract")
    return task


def variance_products(task, draws):
    """Analytic annualized variance rates, preserving chain/draw axes."""
    kappa = task["fixed_parameters"]["kappa"]
    v0, theta = draws[..., 0], draws[..., 1]
    def rate(start, end):
        weight = np.exp(-kappa * start) * -np.expm1(-kappa * (end - start)) / (kappa * (end - start))
        return weight * v0 + (1 - weight) * theta
    return {"variance_swap_1y": rate(0, 1), "forward_variance_0_5_to_1y": rate(.5, 1)}


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


def load_cache(path, *, expected_task=None, expected_checkpoint_sha256=None):
    """Return task, report and cached arrays offline; requires NumPy, never BayesFlow."""
    path = Path(path)
    report = json.loads((path / "bundle.json").read_text())
    require(report.get("format") == "full_surface_teaching_v1" and set(report.get("files", {})) == CACHE_FILES,
            "expected complete full_surface_teaching_v1 cache")
    for name, sha in report["files"].items():
        require(digest(path / name) == sha, f"teaching cache checksum mismatch: {name}")
    task = heston_task(json.loads((path / "task.json").read_text()))
    require(report["task_fingerprint"] == fingerprint(task) and report["parameter_names"] == NAMES,
            "cache task identity differs")
    require(expected_task is None or task == expected_task, "cache differs from expected complete task")
    require(expected_checkpoint_sha256 is None or report["checkpoint_sha256"] == expected_checkpoint_sha256,
            "cache differs from expected checkpoint")
    require(report.get("scientific_acceptance") is False and report.get("role") == "cached_development_example"
            and report.get("checkpoint_status") == "frozen_candidate_gate_passed_not_full_acceptance"
            and not report["case_id"].startswith("test:"), "cached development/acceptance scope changed")
    with np.load(path / "posterior.npz", allow_pickle=False) as data: arrays = dict(data)
    count, shape = report["neural_draw_count"], report["reference_chain_shape"]
    require(type(count) is int and count > 1 and len(shape) == 3 and shape[-1] == 4
            and shape[0] > 1 and shape[1] > 1, "invalid draw counts or chain axes")
    shapes = {"observations": (63,), "neural_draws": (count, 4), "reference_chains": tuple(shape)}
    for name in ("variance_swap_1y", "forward_variance_0_5_to_1y"):
        shapes["neural_" + name] = (count,); shapes["reference_" + name] = tuple(shape[:2])
    require(set(arrays) == set(shapes), "cache arrays differ from complete replay contract")
    for name, expected in shapes.items():
        require(arrays[name].shape == expected and np.isfinite(arrays[name]).all(), f"invalid cached {name}")
    bounds = np.asarray(task["prior_bounds"])
    for name in ("neural_draws", "reference_chains"):
        require(np.all(arrays[name] >= bounds[:, 0]) and np.all(arrays[name] <= bounds[:, 1]),
                "cached parameter draws outside prior")
    reference = json.loads((path / "reference.json").read_text())
    require(reference["task"] == task and reference["surface_id"] == report["case_id"]
            and np.array_equal(reference["observed_prices"], arrays["observations"]),
            "cache observation/reference identity differs")
    for prefix, draws in (("neural_", arrays["neural_draws"]), ("reference_", arrays["reference_chains"])):
        for name, expected in variance_products(task, draws).items():
            require(np.allclose(arrays[prefix + name], expected, rtol=1e-12, atol=1e-14),
                    "parameter/product pairing changed")
    return task, report, arrays


def rough_task(task):
    names = ROUGH_NAMES.get(task.get("task_id"))
    require(task.get("model") == "rough" and names is not None and task.get("parameter_names") == names,
            "expected complete known-curve or joint-curve rough task")
    context = task.get("context", {}); contracts = context.get("contracts", [])
    noise = np.asarray(context.get("quote_noise_std"), dtype=float)
    bounds = np.asarray(task.get("prior_bounds"), dtype=float)
    require(len(contracts) == 63 and noise.shape == (63,) and np.isfinite(noise).all() and np.all(noise > 0)
            and bounds.shape == (len(names), 2) and np.isfinite(bounds).all()
            and np.all(bounds[:, 0] < bounds[:, 1]), "invalid full-surface rough task")
    return task


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


def load_neural_replay(path, *, expected_task=None, expected_checkpoint_sha256=None):
    """Offline rough replay; never fabricate missing reference chains or products."""
    path = Path(path); report = json.loads((path / "bundle.json").read_text())
    require(report.get("format") == "full_surface_neural_replay_v1" and set(report.get("files", {})) == NEURAL_FILES,
            "expected complete rough neural replay")
    for name, sha in report["files"].items(): require(digest(path / name) == sha, "replay checksum mismatch: " + name)
    task = rough_task(json.loads((path / "task.json").read_text()))
    training = json.loads((path / "training.json").read_text()); evaluation = json.loads((path / "evaluation.json").read_text())
    require(report["task_fingerprint"] == fingerprint(task) == training["task_fingerprint"] == evaluation["task_fingerprint"]
            and report["parameter_names"] == task["parameter_names"]
            and training.get("format") == "full_surface_training_v1"
            and report["selected_epoch"] == training["selected_epoch"]
            and report["checkpoint_sha256"] == training["checkpoint_sha256"] == evaluation["checkpoint_sha256"]
            and report["bank_manifest_sha256"] == evaluation["bank_manifest_sha256"], "replay task/model/bank identity differs")
    require(expected_task is None or task == expected_task, "replay differs from expected complete task")
    require(expected_checkpoint_sha256 is None or report["checkpoint_sha256"] == expected_checkpoint_sha256,
            "replay differs from expected checkpoint")
    ids = evaluation.get("example_ids", [])
    case_id = report.get("case_id")
    require(isinstance(case_id, str) and case_id.startswith("validation:")
            and case_id.removeprefix("validation:").isdigit()
            and case_id == f"validation:{int(case_id.removeprefix('validation:'))}"
            and len(ids) == len(set(ids)) and case_id in ids, "development replay case identity differs")
    require(report.get("validation_case_index", int(case_id.split(":")[1])) == int(case_id.split(":")[1])
            and report.get("evaluation_row", ids.index(case_id)) == ids.index(case_id), "development replay case identity differs")
    require(report.get("role") == "cached_development_neural_replay"
            and report.get("scientific_acceptance") is False and report.get("reserved_test_evaluated") is False
            and report.get("conditional_reference_available") is False and evaluation.get("evaluated_split") == "validation"
            and evaluation.get("reserved_test_evaluated") is False, "development replay scope changed")
    with np.load(path / "posterior.npz", allow_pickle=False) as saved: arrays = dict(saved)
    count, dimension = report["neural_draw_count"], len(task["parameter_names"])
    known = task["task_id"] == "rough_calibration_v1"
    require(report.get("known_curve_is_observed") == known, "known/joint curve boundary changed")
    shapes = {"observations": (63,), "truth": (dimension,), "neural_draws": (count, dimension)}
    if known: shapes["context_values"] = (7,)
    require(type(count) is int and count > 1 and set(arrays) == set(shapes), "invalid replay arrays or draw count")
    for name, shape in shapes.items():
        require(arrays[name].shape == shape and np.isfinite(arrays[name]).all(), "invalid replay array: " + name)
    if known: require(np.all(arrays["context_values"] > 0), "observed curve levels must be positive")
    bounds = np.asarray(task["prior_bounds"])
    for name in ("truth", "neural_draws"):
        require(np.all(arrays[name] >= bounds[:, 0]) and np.all(arrays[name] <= bounds[:, 1]), "replay parameters outside prior")
    return task, report, arrays


def load_tutorial_artifact(manifest, name):
    """Pinned cached role for the review notebook; does not download or load models."""
    manifest = Path(manifest); catalogue = json.loads(manifest.read_text())
    require(catalogue.get("format") == "full_surface_tutorial_artifacts_v1"
            and catalogue.get("scientific_acceptance") is False, "invalid tutorial artifact catalogue")
    entry = catalogue["artifacts"][name]; path = (manifest.parent / entry["cache"]).resolve()
    require(digest(path / "bundle.json") == entry["bundle_sha256"], "catalogue/cache identity differs")
    require(entry["cache_format"] in ("full_surface_teaching_v1", "full_surface_neural_replay_v1"),
            "unknown tutorial cache format")
    loader = load_cache if entry["cache_format"] == "full_surface_teaching_v1" else load_neural_replay
    task, report, arrays = loader(path, expected_checkpoint_sha256=entry["checkpoint_sha256"])
    require(fingerprint(task) == entry["task_fingerprint"], "catalogue/task identity differs")
    if loader is load_cache:
        arrays["truth"] = np.asarray(json.loads((path / "reference.json").read_text())["generating_parameters"])
    return task, {**report, "artifact_role": "cached", "model_source": entry["model_source"]}, arrays


def load_tutorial_checkpoint(manifest, name, directory):
    """Explicit live role; BF2.1 initializes only when this function is called."""
    task, report, _ = load_tutorial_artifact(manifest, name)
    from . import full_surface
    live_task, workflow, record = full_surface.load_checkpoint(directory)
    require(live_task == task and record["checkpoint_sha256"] == report["checkpoint_sha256"],
            "live checkpoint differs from the selected tutorial artifact")
    return live_task, workflow, record


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
