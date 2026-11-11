"""Describe prior context from immutable ready old54 market chains only.

No model loading, pricing, neural sampling or MCMC occurs. This is one-snapshot
risk-neutral working-model evidence, not a universal bound recommendation or a
counterfactual posterior under the original fixed-kappa tutorial task.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_heston_spread_baseline as canonical

PHYSICAL_NAMES = ["initial_variance", "kappa", "theta", "vol_of_vol", "rho"]


def read(path): return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": canonical.digest(path)}


def protect(path, protected, expected=None):
    spec = binding(path)
    if expected is not None and spec["sha256"] != expected: raise ValueError("saved prior-evidence input identity differs")
    protected[spec["path"]] = spec["sha256"]
    return spec


def indicator_summary(values, threshold, batch_length):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] != 4 or values.shape[1] % batch_length:
        raise ValueError("four chain-preserving arrays and complete within-chain batches required")
    indicator = values > threshold
    batches = indicator.reshape(values.shape[0], values.shape[1]//batch_length, batch_length).mean(axis=2)
    pooled = batches.reshape(-1)
    return {"threshold": threshold, "strictly_greater": True, "sample_count": int(indicator.size),
        "count": int(indicator.sum()), "fraction": float(indicator.mean()),
        "per_chain_fraction": indicator.mean(axis=1).tolist(), "batch_length": batch_length,
        "batches_per_chain": batches.shape[1], "total_batches": int(batches.size), "draws_discarded": 0,
        "mcse_batch_means": float(np.sqrt(pooled.var(ddof=1)/len(pooled))),
        "method": "non-overlapping batches formed separately within each retained chain; sqrt(sample_variance(all batch means)/number_of_batches)",
        "scope": "descriptive Monte Carlo error of this saved-chain indicator; depends on batch length, not data uncertainty or a new posterior fit"}


def reference_summary(directory, *, infer_c, batch_length, protected):
    directory = Path(directory).resolve(); run_path = directory/"run.json"; diagnostic_path = directory/"diagnostics.json"
    run, diagnostic = read(run_path), read(diagnostic_path)
    run_binding = protect(run_path, protected); diagnostic_binding = protect(diagnostic_path, protected)
    if (run.get("status") != "completed" or run.get("reference_ready") is not True
            or diagnostic.get("reference_ready") is not True or diagnostic.get("mcmc_diagnostics_passed") is not True
            or diagnostic.get("pointwise_precision_passed") is not True or run.get("smoke") is not False
            or run.get("reserved_test_arrays_opened") is not False
            or run["task"].get("quote_count") != 54 or run["task"].get("infer_c") is not infer_c
            or run["names"] != PHYSICAL_NAMES+(["c"] if infer_c else [])
            or (not infer_c and run["task"].get("fixed_c") != 1.)):
        raise ValueError("ready immutable old54 fixed-c1 or joint-c market reference required")
    panel_path = Path(run["panel_path"]); panel = read(panel_path)
    panel_binding = protect(panel_path, protected, run["panel_sha256"])
    design_binding = protect(panel["design_path"], protected, panel["design_sha256"])
    case = next(case for case in panel["cases"] if case["id"] == "market")
    y = np.asarray(case["observations"], dtype=float)
    chains_path = Path(run["chain_payload_path"])
    chain_binding = protect(chains_path, protected, run["chain_payload_sha256"])
    canonical.check_reference_binding(run, diagnostic, chains_path, target=run["task"], design_sha256=panel["design_sha256"],
        case_id="market", observations=y, panel_sha256=panel_binding["sha256"], products=panel["products"], diagnostics_path=diagnostic_path)
    with np.load(chains_path, allow_pickle=False) as archive: draws = archive["parameters"].copy()
    retained = run["attempts"][run["selected_attempt"]]["draws"]
    bounds = np.asarray(run["bounds"])
    if (draws.shape != (4, retained, len(run["names"])) or not np.isfinite(draws).all()
            or np.any(draws < bounds[:, 0]) or np.any(draws > bounds[:, 1])):
        raise ValueError("saved chain/draw/parameter axes or prior support differ")
    summaries = {}
    for index, name in enumerate(run["names"]):
        saved = diagnostic["parameters"]["parameters["+name+"]"]
        if saved["r_hat"] > 1.01 or min(saved["ess_bulk"], saved["ess_tail"]) < 400:
            raise ValueError("saved reference does not pass its convergence requirements")
        summaries[name] = {"quantiles_05_50_95": np.quantile(draws[:, :, index], [.05, .5, .95]).tolist(),
            "saved_diagnostics": {key: saved[key] for key in ("mean", "sd", "mcse_mean", "r_hat", "ess_bulk", "ess_tail")}}
    result = {"run": run_binding, "diagnostics": diagnostic_binding, "chains": chain_binding,
        "panel": panel_binding, "design": design_binding, "task": run["task"], "reference_ready": True,
        "chain_shape": list(draws.shape), "axes": ["chain", "draw", "parameter"], "parameter_names": run["names"],
        "parameters": summaries, "vol_of_vol_greater_than_original_cap": indicator_summary(draws[:, :, 3], 1.5, batch_length),
        "sampler": run["sampler"], "settings": run["settings"], "recorded_source_bindings": run["source_sha256"],
        "reserved_arrays_opened": False, "new_sampling": False}
    if infer_c:
        c = draws[:, :, -1]
        result["c_boundary_context"] = {"prior_bounds": run["task"]["prior_bounds"][-1],
            "fraction_c_ge_1p95": float((c >= 1.95).mean()),
            "interpretation": "upper-bound pressure under the specified likelihood/prior; residual-scale/model/data limitation, not pure quote-noise identification"}
    return result


def build(original_task_path, current_task_path, fixed_reference, joint_reference, out):
    out = Path(out)
    if out.exists(): raise FileExistsError("prior-realism evidence identity must be new")
    protected = {}
    original_binding = protect(original_task_path, protected); current_binding = protect(current_task_path, protected)
    original, current_wrapper = read(original_task_path), read(current_task_path)
    current = current_wrapper["target"]
    old_bounds = dict(zip(original["parameter_names"], original["prior_bounds"], strict=True))
    current_bounds = dict(zip(current["parameter_names"], current["prior_bounds"], strict=True))
    if (original["fixed_parameters"].get("kappa") != 2. or old_bounds["vol_of_vol"] != [.1, 1.5]
            or current_bounds["kappa"] != [.1, 10.] or current_bounds["vol_of_vol"] != [.1, 3.]
            or current_bounds["c"] != [.5, 2.] or current["quote_count"] != 63):
        raise ValueError("original/current prior and task context differs")
    fixed = reference_summary(fixed_reference, infer_c=False, batch_length=100, protected=protected)
    joint = reference_summary(joint_reference, infer_c=True, batch_length=200, protected=protected)
    if fixed["panel"] != joint["panel"] or fixed["design"] != joint["design"]:
        raise ValueError("the two saved references must use the same old54 market observations")
    for name in PHYSICAL_NAMES:
        fixed_bounds = fixed["task"]["prior_bounds"][fixed["parameter_names"].index(name)]
        joint_bounds = joint["task"]["prior_bounds"][joint["parameter_names"].index(name)]
        if fixed_bounds != current_bounds[name] or joint_bounds != current_bounds[name]:
            raise ValueError("physical-prior context differs between references/current task")
    if any(canonical.digest(path) != sha for path, sha in protected.items()): raise ValueError("protected saved evidence changed")
    result = {"format": "heston_pr87_grid_prior_realism_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "user_decision": "retain current broad priors; no prior or parameter change authorized by this evidence",
        "original_task": {"binding": original_binding, "parameter_names": original["parameter_names"],
            "prior_bounds": original["prior_bounds"], "fixed_kappa": original["fixed_parameters"]["kappa"], "measure": original["measure"]},
        "current_task": {"binding": current_binding, "target": current},
        "old54_market_references": {"fixed_c1": fixed, "joint_c": joint},
        "interpretation": [
            "Both saved old54 market posteriors center vol_of_vol just below1.5, but the1.5 cap excludes a material upper tail under their current variable-kappa working models.",
            "The original task also fixes kappa=2; these variable-kappa chains cannot be reused as a counterfactual posterior for that original task.",
            "A single SPX snapshot under specified risk-neutral Heston, carry and Gaussian half-spread noise assumptions does not establish universal realistic parameter bounds.",
            "The joint-c reference presses against its upper c=2 prior bound; this limits model/noise interpretation and does not justify a retrospective prior change.",
            "These references concern the old54 observation target. They are prior-context evidence only and cannot validate the revised63 posterior."],
        "producer": binding(__file__), "canonical_binding_validator": binding(canonical.__file__),
        "protected_inputs": [{"path": p, "sha256": sha} for p, sha in sorted(protected.items())],
        "protected_inputs_unchanged": True, "chain_axes_preserved": True, "new_neural_sampling": False,
        "new_pricing": False, "new_reference_sampling": False, "reserved_arrays_opened": False,
        "scientific_acceptance": False, "prior_changed": False}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream: json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original-task", "current-task", "fixed-reference", "joint-reference", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args(); result = build(args.original_task,args.current_task,args.fixed_reference,args.joint_reference,args.out)
    print(json.dumps({"out": binding(args.out), "xi_tail_fractions": {name: value["vol_of_vol_greater_than_original_cap"]["fraction"]
        for name, value in result["old54_market_references"].items()}}))
