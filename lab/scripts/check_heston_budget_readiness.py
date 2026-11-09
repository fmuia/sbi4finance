"""Evaluate the saved full Heston model before admitting shorter training runs.

This opens validation data and the three recorded development references only.
It never trains, generates a bank, changes a model or opens the reserved test
split. Reports and draws go to a new caller-owned directory.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checked_support(draws, bounds):
    import numpy as np

    draws = np.asarray(draws)
    return bool(draws.size and draws.ndim > 0 and not np.iscomplexobj(draws) and draws.shape[-1] == len(bounds)
                and np.isfinite(draws).all() and np.all(draws > bounds[:, 0]) and np.all(draws < bounds[:, 1]))


def validate_panels(protocol, available):
    """Reject ranges that could silently select another case or repeat a draw."""
    panel = protocol["data_roles"]["development"]
    predictive = protocol["data_roles"]["posterior_predictive"]
    for settings, keys in ((panel, ("start", "stop", "draws", "seed", "batch_size")),
                           (predictive, ("start", "stop", "paired_draws", "seed"))):
        if any(type(settings.get(key)) is not int for key in keys):
            raise ValueError("integer panel indices, counts and seeds are required")
    if (panel.get("split") != "validation" or predictive.get("split") != "validation"
            or not 0 <= panel["start"] < panel["stop"] <= available
            or panel["draws"] < 2 or panel["batch_size"] < 1 or panel["seed"] < 0
            or not panel["start"] <= predictive["start"] < predictive["stop"] <= panel["stop"]
            or not 2 <= predictive["paired_draws"] <= panel["draws"] or predictive["seed"] < 0):
        raise ValueError("validation panels must have valid nested ranges and distinct selected draws")


def run(args):
    import numpy as np
    from sbi4finance.calibration import bank_collection as bank, full_surface, tasks, validation
    from sbi4finance.calibration.heston_budget_diagnostics import conditional_agreement
    from sbi4finance.calibration.performance import PerformanceRecorder
    from scripts.compare_full_surface_reference import _reference_issues

    performance = PerformanceRecorder()
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("format") != "heston_training_budget_protocol_v1":
        raise ValueError("a frozen Heston budget protocol is required")
    if digest(args.bank / "manifest.json") != protocol["bank"]["manifest_sha256"]:
        raise ValueError("bank differs from protocol")
    if digest(args.checkpoint / "model.keras") != protocol["control"]["checkpoint_sha256"]:
        raise ValueError("readiness must evaluate the saved full-model control")
    with performance.phase("checkpoint_construction_and_reload"):
        task, workflow, record = full_surface.load_checkpoint(args.checkpoint)
    if record["task_fingerprint"] != protocol["bank"]["task_fingerprint"]:
        raise ValueError("control task differs from protocol")
    with performance.phase("development_data_loading"):
        manifest, splits = bank.load_bank(args.bank, splits=("validation",), roles=bank.TRAINING_ROLES)
    if manifest["configuration"]["task"] != task:
        raise ValueError("control and bank tasks differ")
    split = splits["validation"]
    validate_panels(protocol, len(split["parameters"]))
    bounds, names = np.asarray(task["prior_bounds"]), task["parameter_names"]
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"format": "heston_training_budget_readiness_v1", "status": "running",
              "protocol_sha256": digest(args.protocol), "control_checkpoint_sha256": record["checkpoint_sha256"],
              "bank_manifest_sha256": digest(args.bank / "manifest.json"), "reserved_test_evaluated": False,
              "training_performed": False, "new_mcmc_performed": False, "gates": {}, "conditional": {},
              "source_sha256": {"scripts/check_heston_budget_readiness.py": digest(__file__),
                  "src/sbi4finance/calibration/heston_budget_diagnostics.py": digest(
                      ROOT / "src/sbi4finance/calibration/heston_budget_diagnostics.py")}}
    write(args.out / "readiness.json", report)
    try:
        # Complete conditional cases first; the same observations and seeds are
        # declared before any candidate exists. Preserve chains and neural rows.
        conditional = protocol["conditional"]
        for index in conditional["case_indices"]:
            reference_root = ROOT / conditional["source_template"].format(case=index)
            reference = json.loads((reference_root / "reference.json").read_text())
            diagnostics = json.loads((reference_root / "diagnostics.json").read_text())
            if reference["task"] != task or _reference_issues(diagnostics, names):
                raise ValueError(f"case {index}: task mismatch or inconclusive source MCMC diagnostics")
            with np.load(reference_root / "chains.npz", allow_pickle=False) as saved:
                chains = saved["parameters"]
            if chains.shape != (reference["settings"]["chains"], reference["settings"]["draws"], len(names)):
                raise ValueError("reference chain/draw axes differ from metadata")
            if not checked_support(chains, bounds):
                raise ValueError("reference draws outside finite declared prior support")
            with performance.phase("conditional_posterior_sampling"):
                draws = full_surface.sample(workflow, task, np.asarray(reference["observed_prices"])[None, :],
                    conditional["draws"], encoder=record["network"]["encoder"], seed=conditional["seed_base"] + index)[0]
            np.savez_compressed(args.out / f"conditional-{index}.npz", parameters=draws)
            with performance.phase("conditional_uncertainty"):
                uncertainty = conditional.get("uncertainty_calculation", {})
                result = conditional_agreement(draws, chains, diagnostics, names,
                    seed=conditional["seed_base"] + 100 + index,
                    replicates=uncertainty.get("replicates_per_block_length", 400),
                    family_checks=uncertainty.get("family_checks", 24),
                    mean_limit=conditional["mean_error_reference_sd_max"],
                    width_limits=tuple(conditional["central90_width_ratio"]))
            result["finite_in_support_without_clipping"] = checked_support(draws, bounds)
            result["input_sha256"] = {name: digest(reference_root / name)
                                      for name in ("reference.json", "diagnostics.json", "chains.npz")}
            precision = reference_root.parent / "precision/precision.json"
            if precision.exists():
                result["historical_pointwise_precision_sha256"] = digest(precision)
                result["precision_scope"] = "Historical pointwise numerical sensitivity; not posterior stability."
            report["conditional"][str(index)] = result
            write(args.out / "readiness.json", report)

        panel = protocol["data_roles"]["development"]
        if panel["split"] != "validation":
            raise ValueError("readiness development must use validation only")
        batches = []
        with performance.phase("development_posterior_sampling"):
            for start in range(panel["start"], panel["stop"], panel["batch_size"]):
                stop = min(start + panel["batch_size"], panel["stop"])
                draws = full_surface.sample(workflow, task, split["observations"][start:stop], panel["draws"],
                    encoder=record["network"]["encoder"], seed=panel["seed"] + start)
                np.savez_compressed(args.out / f"development-{start:04d}.npz", parameters=draws)
                batches.append(draws)
        draws = np.concatenate(batches)
        truth = split["parameters"][panel["start"]:panel["stop"]]
        with performance.phase("development_diagnostics"):
            ranks = validation.sbc(draws, truth, seed=panel["seed"])
            coverage = validation.coverage(draws, truth)
            recovered = validation.recovery(draws, truth)
            prior_variance = (bounds[:, 1] - bounds[:, 0]) ** 2 / 12
            report["development"] = {"example_ids": split["example_id"][panel["start"]:panel["stop"]].tolist(),
                "rank_screen": {key: value for key, value in ranks.items() if key != "fractional_ranks"},
                "coverage": coverage, "recovery_correlation": [float(value) if np.isfinite(value) else None
                                                               for value in recovered["correlation"]],
                "normalized_posterior_mean_mse": (((draws.mean(1) - truth) ** 2).mean(0) / prior_variance).tolist(),
                "mean_physical_central90_width": (recovered["q95"] - recovered["q05"]).mean(0).tolist()}
        support = checked_support(draws, bounds) and all(item["finite_in_support_without_clipping"]
                                                        for item in report["conditional"].values())
        report["gates"]["finite_in_support_without_clipping"] = support
        report["gates"]["development_rank_screen"] = all(ranks["passed"])
        report["gates"]["conditional_all_cases"] = all(item["status"] == "pass"
                                                         for item in report["conditional"].values())
        if support:
            # Intact parameter rows preserve posterior dependence. Each quote
            # replicate adds the declared noise separately from parameter spread.
            predictive = protocol["data_roles"]["posterior_predictive"]
            rng = np.random.default_rng(predictive["seed"])
            selected = np.linspace(0, panel["draws"] - 1, predictive["paired_draws"], dtype=int)
            means = []
            failures = []
            with performance.phase("posterior_predictive_pricing"):
                for index in range(predictive["start"], predictive["stop"]):
                    rows = []
                    for draw_index in selected:
                        try:
                            parameters, v0 = tasks._heston_split(tasks.model_parameters(task, draws[index - panel["start"], draw_index]))
                            rows.append(tasks.heston_prices(task, parameters, v0)[0])
                        except ValueError as error:
                            failures.append({"observation": index, "draw": int(draw_index), "error": str(error)})
                            rows.append(np.full(63, np.nan))
                    means.append(rows)
            means = np.asarray(means)
            if not failures:
                replicated = means + tasks.sigma(task) * rng.standard_normal(means.shape)
                observed = split["observations"][predictive["start"]:predictive["stop"], :, 0]
                q05, q95 = np.quantile(replicated, [.05, .95], axis=1)
                report["posterior_predictive"] = {"status": "completed_descriptive_check",
                    "observations": len(means), "paired_draws": len(selected),
                    "central90_quote_coverage": float(np.mean((observed >= q05) & (observed <= q95))),
                    "mean_absolute_residual_over_quote_sigma": float(np.mean(np.abs(observed - means.mean(1)) / tasks.sigma(task))),
                    "scope": "Fixed development panel; replicated quotes include new Gaussian observation noise. No new adequacy/acceptance rule."}
                np.savez_compressed(args.out / "predictive.npz", mean_prices=means, replicated_quotes=replicated, observations=observed)
            else:
                report["posterior_predictive"] = {"status": "failed", "failures": failures}
            report["gates"]["predictive_pricing_completed"] = not failures
        else:
            report["gates"]["predictive_pricing_completed"] = False
        failed = (not support or not all(ranks["passed"]) or
                  any(item["status"] == "fail" for item in report["conditional"].values()) or
                  not report["gates"]["predictive_pricing_completed"])
        report["status"] = "fail" if failed else ("pass" if all(report["gates"].values()) else "inconclusive")
        report["performance"] = performance.report(work={"optimizer_updates": 0, "test_observations": 0,
            "development_observations": len(truth), "development_posterior_draws": int(draws.shape[0] * draws.shape[1]),
            "conditional_posterior_draws": len(conditional["case_indices"]) * conditional["draws"]})
        report["matrix_launch_allowed"] = False
        report["next_gate"] = "Control pass alone is insufficient: exact-cap smoke and a cost/resource admission receipt are also required."
        write(args.out / "readiness.json", report)
        return report
    except Exception as error:
        report.update(status="failed_execution", failure={"type": type(error).__name__, "error": str(error)},
                      performance=performance.report(), matrix_launch_allowed=False)
        write(args.out / "readiness.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/heston_training_budget_study.json")
    for name in ("bank", "checkpoint", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("positive thread count required")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ.setdefault("KERAS_BACKEND", "jax")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    result = run(args)
    print(json.dumps({"status": result["status"], "gates": result["gates"], "matrix_launch_allowed": False}))


if __name__ == "__main__":
    main()
