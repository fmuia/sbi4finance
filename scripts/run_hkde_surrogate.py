"""Fit and audit one fixed-contract HKDE price surrogate; never rerun inference."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import scipy

from sbi4finance import hkde_reference as reference, hkde_surrogate as surrogate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")


def parameters_for_split(bounds, seed, split, count):
    """Distinct named streams; all labels for one joint row remain together."""
    if split not in ("train", "validation", "test"):
        raise ValueError("unknown regression split")
    stream = ("train", "validation", "test").index(split)
    rng = np.random.default_rng(np.random.SeedSequence([0x484B5352, seed, stream]))
    bounds = np.asarray(bounds)
    return rng.uniform(bounds[:, 0], bounds[:, 1], size=(count, 2))


def error_summary(predicted, prices, parameters, floor):
    error = predicted - prices
    flat = np.unravel_index(np.argmax(np.abs(error)), error.shape)
    return {"rows": len(prices), "max_abs_error_currency": np.max(np.abs(error)),
            "rmse_currency": np.sqrt(np.mean(error**2)),
            "max_abs_error_by_contract": np.max(np.abs(error), axis=0),
            "rmse_by_contract": np.sqrt(np.mean(error**2, axis=0)),
            "max_floor_relative_error": np.max(np.abs(error) / np.maximum(np.abs(prices), floor)),
            "worst_parameters": parameters[flat[0]], "worst_contract_index": flat[1]}


def put_bound_violation(task, contracts, prices):
    context = task["context"]
    maturity = np.array([c["maturity"] for c in contracts])
    strike = np.array([c["strike"] for c in contracts])
    discount = np.exp(-context["rate"] * maturity)
    forward = context["spot"] * np.exp((context["rate"] - context["dividend"]) * maturity)
    low, high = discount * np.maximum(strike - forward, 0), discount * strike
    return float(max(0.0, np.max(low - prices), np.max(prices - high)))


def load_baseline(protocol, task):
    """Bind immutable historical arrays without requiring a passed neural fit."""
    directory = Path(protocol["baseline_results"])
    paths = {name: directory / name for name in
             ("reference.json", "reference-grid.npz", "neural-validation.json", "validation-checks.npz")}
    report = json.loads(paths["reference.json"].read_text())
    neural = json.loads(paths["neural-validation.json"].read_text())
    original_protocol = Path(protocol["baseline_protocol"])
    config = Path(protocol["task_config"])
    if (not report["passed"] or report["task"] != task
            or report["target_contracts"] != protocol["targets"]
            or report["protocol_sha256"] != sha(original_protocol)
            or report["bank_config_sha256"] != sha(config)
            or report["reference_grid_sha256"] != sha(paths["reference-grid.npz"])
            or neural["reference_report_sha256"] != sha(paths["reference.json"])
            or neural["checks_sha256"] != sha(paths["validation-checks.npz"])
            or neural["split"] != "validation" or neural["reserved_test_evaluated"]):
        raise ValueError("matching accepted numerical reference and original validation artifacts required")
    # Historical script guards may have changed after that run. Bind the actual
    # numerical price/weight code, while retaining exact historical report hashes.
    for name in ("hkde_reference.py", "hkde.py", "option_bank.py", "option_panel.py", "heston.py", "gbm.py"):
        if sha(Path(reference.__file__).with_name(name)) != report["source_sha256"][name]:
            raise ValueError(f"baseline numerical source changed: {name}")
    with np.load(paths["reference-grid.npz"], allow_pickle=False) as saved:
        grid = dict(saved)
    with np.load(paths["validation-checks.npz"], allow_pickle=False) as saved:
        checks = {key: saved[key] for key in
                  ("case_ids", "observations", "conditional_draws", "target_draws", "reference_target_quantiles")}
    if (not np.array_equal(checks["case_ids"], grid["case_ids"])
            or not np.array_equal(checks["observations"], grid["observations"])):
        raise ValueError("conditional cases or observed panels differ")
    return grid, checks, neural, {name: sha(path) for name, path in paths.items()}


def paired_propagation(predicted, direct, probabilities, weights=None):
    """Same draws/weights isolate price substitution, not posterior error."""
    if weights is None:
        pred_q = np.quantile(predicted, probabilities, axis=1).transpose(1, 2, 0)
        direct_q = np.quantile(direct, probabilities, axis=1).transpose(1, 2, 0)
        mean_shift = np.mean(predicted - direct, axis=1)
    else:
        pred_q = np.array([reference.weighted_quantiles(predicted, w, probabilities) for w in weights])
        direct_q = np.array([reference.weighted_quantiles(direct, w, probabilities) for w in weights])
        mean_shift = weights @ (predicted - direct)
    return pred_q, direct_q, mean_shift


def benchmark(direct, fast, parameters, repeats):
    # Both routes get one warmup, and the timed direct route recomputes prices.
    direct(parameters)
    fast(parameters)
    times = {"direct": [], "surrogate": []}
    for _ in range(repeats):
        for name, function in (("direct", direct), ("surrogate", fast)):
            start = time.perf_counter()
            function(parameters)
            times[name].append(time.perf_counter() - start)
    d, s = [float(np.median(times[name])) for name in ("direct", "surrogate")]
    return {"row_count": len(parameters), "direct_seconds": times["direct"],
            "surrogate_seconds": times["surrogate"], "direct_median_seconds": d,
            "surrogate_median_seconds": s, "speedup": d / s,
            "scope": "warm in-process CPU pricing; input validation included; no neural sampling or cached-price lookup"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("configs/hkde_surrogate_protocol.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    protocol = json.loads(args.protocol.read_text())
    task = json.loads(Path(protocol["task_config"]).read_text())["task"]
    contracts, settings = protocol["targets"], protocol["checks"]
    gates = settings["gates"]
    labels = protocol["labels"]
    if (labels["nodes"] != task["numerics"]["refined_nodes"]
            or labels["upper_limit"] != task["numerics"]["refined_upper_limit"]):
        raise ValueError("regression label map must match the accepted numerical reference")
    grid, old, neural_report, input_hashes = load_baseline(protocol, task)
    args.out.mkdir(parents=True, exist_ok=False)
    timing, arrays, banks = {}, {}, {}
    direct = lambda rows: reference.price_parameters(task, rows, contracts)

    def generate(split):
        start = time.perf_counter()
        rows = parameters_for_split(task["prior_bounds"], protocol["root_seed"], split, protocol["splits"][split])
        prices = direct(rows)
        np.savez_compressed(args.out / f"bank-{split}.npz", parameters=rows, prices=prices,
                            example_id=np.array([f"{split}:{i}" for i in range(len(rows))]))
        timing[f"generate_write_{split}"] = time.perf_counter() - start
        banks[split] = {"rows": len(rows), "labels_per_row": len(contracts),
                        "sha256": sha(args.out / f"bank-{split}.npz")}
        return rows, prices

    train_rows, train_prices = generate("train")
    validation_rows, validation_prices = generate("validation")
    start = time.perf_counter()
    model = surrogate.fit(task, contracts, train_rows, train_prices, degree=protocol["model"]["degree"])
    timing["fit"] = time.perf_counter() - start
    start = time.perf_counter()
    surrogate.save_model(args.out / "model.npz", model)
    timing["save_model"] = time.perf_counter() - start
    initial_predictions = surrogate.predict(model, validation_rows)
    start = time.perf_counter()
    loaded = surrogate.load_model(args.out / "model.npz")
    validation_predictions = surrogate.predict(loaded, validation_rows)
    timing["fresh_file_load_and_256_predictions"] = time.perf_counter() - start
    reload_exact = np.array_equal(initial_predictions, validation_predictions)
    if not reload_exact:
        raise ValueError("reloaded model predictions differ")
    fast = lambda rows: surrogate.predict(loaded, rows)
    grid_predictions = fast(grid["parameters"])
    neural_shape = old["conditional_draws"].shape
    neural_rows = old["conditional_draws"].reshape(-1, 2)
    neural_predictions = fast(neural_rows).reshape(neural_shape)
    errors = {name: error_summary(predicted, prices, rows, settings["relative_error_denominator_floor_currency"])
              for name, predicted, prices, rows in
              (("validation", validation_predictions, validation_prices, validation_rows),
               ("stress_grid", grid_predictions, grid["target_prices"], grid["parameters"]),
               ("neural_draws", neural_predictions.reshape(-1, 2), old["target_draws"].reshape(-1, 2), neural_rows))}
    probabilities = settings["quantiles"]
    neural_q, direct_q, neural_mean_shift = paired_propagation(neural_predictions, old["target_draws"], probabilities)
    grid_q, reference_q, grid_mean_shift = paired_propagation(grid_predictions, grid["target_prices"], probabilities, grid["weights"])
    if not np.allclose(reference_q, old["reference_target_quantiles"], rtol=0, atol=1e-12):
        raise ValueError("cached reference quantiles differ from weighted reconstruction")
    propagation = {"case_ids": old["case_ids"], "neural_direct_minus_grid_quantiles": direct_q - reference_q,
                   "neural_surrogate_minus_grid_quantiles": neural_q - reference_q,
                   "neural_paired_quantile_shift": neural_q - direct_q,
                   "grid_paired_quantile_shift": grid_q - reference_q,
                   "neural_paired_mean_shift": neural_mean_shift, "grid_paired_mean_shift": grid_mean_shift}
    bounds = np.array(task["prior_bounds"])
    axes = [np.linspace(lo, hi, labels["refinement_axis_points"]) for lo, hi in bounds]
    refinement_rows = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
    start = time.perf_counter()
    ordinary = direct(refinement_rows)
    refined = reference.price_parameters(task, refinement_rows, contracts,
                                        nodes=labels["refinement_nodes"], upper_limit=labels["refinement_upper_limit"])
    timing["label_refinement"] = time.perf_counter() - start
    refinement_error = float(np.max(np.abs(ordinary - refined)))
    price_sets = [validation_predictions, grid_predictions, neural_predictions.reshape(-1, 2)]
    metrics = {
        "max_abs_price_error_currency": max(value["max_abs_error_currency"] for value in errors.values()),
        "price_rmse_currency": max(np.max(value["rmse_by_contract"]) for value in errors.values()),
        "max_abs_paired_mean_error_currency": max(np.max(np.abs(neural_mean_shift)), np.max(np.abs(grid_mean_shift))),
        "max_abs_paired_quantile_error_currency": max(np.max(np.abs(neural_q-direct_q)), np.max(np.abs(grid_q-reference_q))),
        "max_label_refinement_difference_currency": refinement_error,
        "max_design_condition_number": model["condition_number"],
        "max_put_bound_violation_currency": max(put_bound_violation(task, contracts, prices) for prices in price_sets),
    }
    gate_results = {key: bool(value <= gates[key]) for key, value in metrics.items()}
    validation_passed = all(gate_results.values()) and reload_exact
    test_evaluated, test_passed, test_gates = False, None, None
    if validation_passed:
        test_rows, test_prices = generate("test")
        test_predictions = fast(test_rows)
        errors["test"] = error_summary(test_predictions, test_prices, test_rows, settings["relative_error_denominator_floor_currency"])
        test_gates = {
            "max_abs_price_error_currency": bool(errors["test"]["max_abs_error_currency"] <= gates["max_abs_price_error_currency"]),
            "price_rmse_currency": bool(max(errors["test"]["rmse_by_contract"]) <= gates["price_rmse_currency"]),
            "put_bounds": put_bound_violation(task, contracts, test_predictions) <= gates["max_put_bound_violation_currency"],
        }
        test_evaluated, test_passed = True, all(test_gates.values())
        arrays.update(test_parameters=test_rows, test_error=test_predictions-test_prices)
    benchmark_rows = old["conditional_draws"][0, :settings["benchmark_draws"]]
    benchmarks = {name: benchmark(direct, fast, rows, settings["benchmark_repeats"])
                  for name, rows in (("single", benchmark_rows[:1]), ("batch", benchmark_rows))}
    setup = timing["generate_write_train"] + timing["fit"] + timing["save_model"]
    saving = benchmarks["batch"]["direct_median_seconds"] - benchmarks["batch"]["surrogate_median_seconds"]
    arrays.update(grid_parameters=grid["parameters"], grid_shape=np.array([len(grid["axis0"]),len(grid["axis1"])]),
                  grid_error=grid_predictions-grid["target_prices"], validation_parameters=validation_rows,
                  validation_error=validation_predictions-validation_prices,
                  refinement_parameters=refinement_rows, refinement_error=ordinary-refined)
    np.savez_compressed(args.out / "checks.npz", **arrays)
    paths = [args.protocol, Path(protocol["task_config"]), Path(protocol["baseline_protocol"])]
    sources = [Path(__file__), Path(surrogate.__file__), Path(reference.__file__), Path(reference.__file__).with_name("hkde.py")]
    timing["total"] = time.perf_counter() - started
    report = {
        "experiment_id": protocol["experiment_id"], "evidence_scope": "S", "measure": "Q",
        "passed": bool(validation_passed and test_passed), "validation_passed": validation_passed,
        "test_evaluated": test_evaluated, "test_passed": test_passed, "test_gates": test_gates,
        "thresholds": gates, "metrics": metrics, "gates": gate_results, "task": task, "targets": contracts,
        "fit": {"degree": model["degree"], "rank": model["rank"], "condition_number": model["condition_number"],
                "coefficients_per_output": (model["degree"]+1)**2, "training_rows": len(train_rows)},
        "same_predictions_after_reload": reload_exact, "banks": banks, "errors": errors,
        "data_contract": {"origin": "synthetic", "purpose": "supervised fixed-contract pricing regression",
                          "inputs": "parameters[N,2] in task.parameter_names order",
                          "targets": "prices[N,2], deterministic Fourier expectations in target-contract order",
                          "grouping": "all contract labels for a joint parameter row stay in its split",
                          "context": "known Q task and fixed contracts; no observed market panel or latent truth conditioning",
                          "conditioning_cutoff": "not applicable to static synthetic model-price regression"},
        "propagation": propagation, "timings_seconds": timing, "benchmark": benchmarks,
        "amortization": {"training_setup_seconds": setup,
                         "training_setup_break_even_4096_row_queries": setup / saving if saving > 0 else None,
                         "complete_run_break_even_4096_row_queries": timing["total"] / saving if saving > 0 else None,
                         "scope": "fractional pricing-batch break-even; complete run includes this audit/benchmark, but excludes previously cached reference/inference studies"},
        "baseline_inference": {"passed": neural_report["passed"],
                               "max_abs_price_quantile_error_currency": neural_report["metrics"]["max_abs_target_quantile_error_currency"],
                               "recomputed_max_abs_price_quantile_error_currency": np.max(np.abs(direct_q-reference_q)),
                               "with_surrogate_max_abs_price_quantile_error_currency": np.max(np.abs(neural_q-reference_q)),
                               "gate_currency": neural_report["thresholds"]["max_abs_target_quantile_error_currency"],
                               "posterior_final_test_evaluated": False, "new_sbc_run": False},
        "input_sha256": {**{str(p):sha(p) for p in paths}, **input_hashes},
        "artifact_sha256": {p.name:sha(p) for p in args.out.glob("*.npz")},
        "source_sha256": {p.name:sha(p) for p in sources},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "platform": platform.system(), "machine": platform.machine()},
        "limits": ["Fixed two contracts/context and bounded parameters; no extrapolation, Greeks or generic surface claim.",
                   "Regression targets numerical expectations, never noisy observations or single payoffs.",
                   "Finite validation and refinement sets are not a uniform continuum error certificate.",
                   "Paired propagation isolates substitution error; the original posterior remains provisional.",
                   "The new regression test is distinct from the untouched posterior final test.",
                   "Load/predict timing is within the current process, not full cold process startup."]}
    write_json(args.out / "report.json", report)
    print(json.dumps(jsonable({k:report[k] for k in ("passed", "validation_passed", "test_passed", "metrics", "baseline_inference", "benchmark")}), indent=2))
    if not report["passed"]:
        raise SystemExit("Frozen surrogate checks failed; report retained, no tuning sweep.")


if __name__ == "__main__":
    main()
