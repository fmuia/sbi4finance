"""Small offline Heston replay bundles; importing this module initializes no backend.

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export-checkpoint")
    for name in ("source", "out", "gate", "product-checks"): p.add_argument("--" + name, type=Path, required=True)
    p = sub.add_parser("cache")
    for name in ("checkpoint", "declaration", "out"): p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--case-index", type=int, default=0)
    args = vars(parser.parse_args()); command = args.pop("command")
    result = export_checkpoint(**args) if command == "export-checkpoint" else prepare_cache(**args)
    print(json.dumps({"format": result["format"], "scientific_acceptance": False}))


if __name__ == "__main__":
    main()
