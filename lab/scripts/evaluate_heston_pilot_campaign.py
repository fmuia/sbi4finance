"""Development evaluation for the controlled full-54 Heston campaign.

Only validation members are read. The final 2,000-case acceptance panel is
reserved and deliberately unsupported here until a winner is frozen.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

# Support the documented ``python scripts/...`` invocation as well as -m.
LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "src"))

import numpy as np
from scipy.stats import kstest

from sbi4finance.calibration import heston_budget_diagnostics as uncertainty
from sbi4finance.calibration import validation
from scripts.run_heston_observation_pilot import expected_variance, write_json


NAMES = ["initial_variance", "kappa", "theta", "vol_of_vol", "rho", "tau"]
INITIAL_REFERENCE_INDICES = [0, 2, 4, 6, 8, 10]
FAMILY_CHECKS = 2 * 8 * 13  # two metrics, six coordinates + two products, 13 cases


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def values_digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def neural_support_report(values, bounds):
    """Audit native float32 rounding at bounds without modifying any draw.

    Only float32 arrays receive two ULPs at the respective endpoint. Float64
    arrays and all reference chains retain strict physical support checks.
    """
    values, bounds = np.asarray(values), np.asarray(bounds, dtype=float)
    if values.shape[-1] != len(bounds) or not np.isfinite(values).all():
        raise ValueError("finite neural draws with matching parameter dimension required")
    flat = values.reshape(-1, len(bounds)).astype(float)
    tolerance = (2 * np.abs(np.spacing(bounds.astype(np.float32))).astype(float)
                 if values.dtype == np.dtype("float32") else np.zeros_like(bounds))
    below, above = np.maximum(bounds[:, 0] - flat, 0.), np.maximum(flat - bounds[:, 1], 0.)
    material = (below > tolerance[:, 0]) | (above > tolerance[:, 1])
    return {"dtype": str(values.dtype), "draw_count": len(flat), "modified_draw_count": 0,
            "endpoint_tolerance_lower_upper": tolerance.tolist(),
            "strict_below_count_by_parameter": np.sum(below > 0, axis=0).tolist(),
            "strict_above_count_by_parameter": np.sum(above > 0, axis=0).tolist(),
            "maximum_below_deviation_by_parameter": np.max(below, axis=0).tolist(),
            "maximum_above_deviation_by_parameter": np.max(above, axis=0).tolist(),
            "representational_excursion_draw_count": int(np.sum(np.any((below > 0) | (above > 0), axis=1) & ~np.any(material, axis=1))),
            "material_outside_draw_count": int(np.sum(np.any(material, axis=1))),
            "scope": "Native float32 inverse-adapter output, at most two endpoint-specific ULPs; no clipping, dropped cases or tolerance for reference chains."}


def validate_reference_observations(record, expected, kind):
    expected = np.asarray(expected, dtype=float)
    if expected.shape != (54,) or not np.isfinite(expected).all():
        raise ValueError("exact finite full54 expected observations required")
    if kind == "market":
        forbidden = ("index", "case_json_sha256", "panel_sha256", "panel_sha")
        if record.get("format") != "heston_spxw_reference_v1" or any(key in record for key in forbidden):
            raise ValueError("market comparison requires an original frozen-market reference, not a synthetic case")
    elif kind == "synthetic":
        if record.get("format") != "heston_training_conditional_reference_v1" or "observations" not in record:
            raise ValueError("synthetic comparison requires an explicitly identified synthetic observation")
    else:
        raise ValueError("reference kind must be market or synthetic")
    if "observations" in record:
        actual = np.asarray(record["observations"], dtype=float)
        if not np.array_equal(actual, expected) or record.get("observation_sha256") != values_digest(expected):
            raise ValueError("reference observation vector or observation hash mismatch")
    elif "observation_sha256" in record and record["observation_sha256"] != values_digest(expected):
        raise ValueError("implicit market observation hash differs from frozen design mids")
    return {"kind": kind, "expected_observation_sha256": values_digest(expected),
            "binding": "explicit vector and float64 SHA" if "observations" in record else "original market-reference format and exact frozen-design SHA; mids implicit in that design"}


def validate_synthetic_case(run, case_record, declared, panel_hash, design_hash, records):
    if (run.get("index") != declared["index"] or case_record.get("index") != declared["index"]
            or case_record.get("id") != declared["id"]
            or ("id" in run and run["id"] != declared["id"])):
        raise ValueError("synthetic reference index or ID differs from the declared case")
    if any(case_record.get(key) != value for key, value in declared.items()):
        raise ValueError("synthetic truth or errors differ from the declared panel case")
    if (run.get("panel_sha256", run.get("panel_sha")) != panel_hash
            or case_record.get("panel_sha256") != panel_hash
            or run.get("design_sha256") != design_hash or case_record.get("design_sha256") != design_hash):
        raise ValueError("synthetic case design or panel identity mismatch")
    observed = np.asarray(case_record.get("observations"), dtype=float)
    validate_reference_observations(run, observed, "synthetic")
    if case_record.get("observation_sha256") != values_digest(observed):
        raise ValueError("synthetic case observation hash mismatch")
    clean = np.asarray(case_record.get("clean_prices"), dtype=float)
    errors = np.asarray(declared["standard_normal_errors"], dtype=float)
    tau = declared["parameters"][-1]
    sigma = np.hypot(np.asarray(records["half_spread"]), tau * np.asarray(records["market_vega"]))
    if clean.shape != (54,) or not np.isfinite(clean).all() or not np.array_equal(clean + errors * sigma, observed):
        raise ValueError("synthetic observation does not equal saved clean prices plus declared Gaussian errors")
    return observed


def declare_reference_panel(design, design_hash, bounds, seed=261111):
    """Freeze all 12 truths and error vectors before new checkpoint outcomes."""
    bounds = np.asarray(bounds, dtype=float)
    rng = np.random.default_rng(np.random.SeedSequence([0x50414E45, seed]))
    broad = rng.uniform(bounds[:5, 0], bounds[:5, 1], size=(6, 5))
    local = np.array([[.027, 4.3, .0565, 1.8, -.667], [.0258, 3.5, .060, 1.65, -.695],
                      [.0285, 5.3, .0535, 2., -.64], [.0265, 6.5, .054, 2.3, -.62]])
    boundary = np.array([[.0105, .15, .0105, min(2.9, bounds[3, 1] - .01), -.94],
                         [.159, bounds[1, 1] - .2, .159, min(2.9, bounds[3, 1] - .01), -.105]])
    physical = np.concatenate([broad, local, boundary])
    if np.any(physical <= bounds[:5, 0]) or np.any(physical >= bounds[:5, 1]):
        raise ValueError("fixed local/boundary panel lies outside this task support")
    cases = []
    n = len(design["records"]["mid"])
    for index, point in enumerate(physical):
        tau_bounds = [0., .02] if index < 6 else ([.004, .01] if index < 10 else [0., .0003])
        tau_rng = np.random.default_rng(np.random.SeedSequence([0x50414E45, seed, index, 1]))
        noise_rng = np.random.default_rng(np.random.SeedSequence([0x50414E45, seed, index, 2]))
        tau = float(tau_rng.uniform(*tau_bounds))
        cases.append({"index": index, "id": f"conditional:{index:03d}",
                      "category": "broad_prior" if index < 6 else ("market_like" if index < 10 else "low_tau_boundary"),
                      "parameters": [*point.tolist(), tau], "tau_sampling_bounds": tau_bounds,
                      "spread_multiplier": 1., "standard_normal_errors": noise_rng.standard_normal(n).tolist()})
    return {"format": "heston_campaign_declared_conditional_panel_v1", "names": NAMES,
            "task_bounds": bounds.tolist(), "design_sha256": design_hash, "seed": seed,
            "initial_reference_indices": INITIAL_REFERENCE_INDICES, "cases": cases,
            "noise_protocol": "Independent PCG64 SeedSequence([0x50414E45,seed,index,1]) for tau; suffix2 for quote errors. Never regenerate failures or condition on quote positivity.",
            "scope": "12 fixed development reference observations, selected before new campaign outcomes. Four local physical truths use the previous market region; this is not untouched acceptance or a prior-predictive frequency study."}


def holm(pvalues):
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order])
    result = np.empty(len(values))
    result[order] = np.minimum(adjusted, 1.)
    return result


def calibration_metrics(draws, truth, names, seed):
    """Finite-IID-posterior randomized ranks, coordinate KS and Holm family."""
    ranks = validation.sbc(draws, truth, seed=seed)
    fractional = np.asarray(ranks.pop("fractional_ranks"))
    tests = [kstest(fractional[:, i], "uniform", method="exact") for i in range(len(names))]
    adjusted = holm([test.pvalue for test in tests])
    grid, difference = validation.ecdf_difference(fractional)
    return {"names": list(names), "cases": len(truth), "draws_per_case": draws.shape[1],
            "coverage": validation.coverage(draws, truth, levels=(.5, .8, .9, .95)),
            "rank_randomization": "(number_below + UniformInteger(0,ties) + Uniform(0,1))/(posterior_draws+1); continuous Uniform under the calibrated IID-posterior null",
            "coordinate_ks": {name: {"statistic": float(test.statistic), "p_value": float(test.pvalue),
                                      "holm_adjusted_p_value": float(adjusted[i]), "holm_reject_at_05": bool(adjusted[i] <= .05)}
                              for i, (name, test) in enumerate(zip(names, tests))},
            "holm_family_size": len(names), "rank_dkw_screen": ranks,
            "ecdf_grid": grid.tolist(), "ecdf_minus_uniform": difference.tolist(),
            "posterior_mean_rmse": np.sqrt(np.mean((draws.mean(axis=1) - truth)**2, axis=0)).tolist(),
            "scope": "Prior-predictive development marginal calibration; no conditional-accuracy or acceptance verdict."}


def fresh_development(physical, clean, records, bounds, seed):
    """Fresh independent tau and full error vectors for each validation row."""
    physical, clean = np.asarray(physical), np.asarray(clean)
    bounds = np.asarray(bounds)
    if physical.shape != (len(clean), 5) or clean.ndim != 2 or clean.shape[1] != len(records["mid"]):
        raise ValueError("five physical parameters and complete clean quote rows required")
    if not np.isfinite(physical).all() or not np.isfinite(clean).all() or np.any(physical < bounds[:5, 0]) or np.any(physical > bounds[:5, 1]):
        raise ValueError("nonfinite bank data or physical latents outside task prior")
    rng = np.random.default_rng(np.random.SeedSequence([0x4556414C, seed]))
    tau = rng.uniform(bounds[-1, 0], bounds[-1, 1], len(clean))
    truth = np.column_stack([physical, tau])
    sd = np.hypot(np.asarray(records["half_spread"])[None, :], tau[:, None] * np.asarray(records["market_vega"])[None, :])
    return truth, clean + sd * rng.standard_normal(clean.shape)


def validate_identity(metadata, design_hash, bank_record, bank_record_hash, bank_payload_hash):
    task = metadata["task"]
    if (metadata["design_sha256"] != design_hash or task["design_sha256"] != design_hash
            or metadata["names"] != NAMES or task["parameter_names"] != NAMES
            or metadata["bounds"] != task["prior_bounds"]
            or metadata["task_fingerprint"] != fingerprint(task)
            or metadata["bank_record_sha256"] != bank_record_hash
            or metadata["bank_payload_sha256"] != bank_payload_hash
            or bank_record["payload_sha256"] != bank_payload_hash
            or bank_record["identity"]["task_fingerprint"] != metadata["task_fingerprint"]
            or bank_record["identity"]["task"] != task):
        raise ValueError("checkpoint, task, design or bank identities differ")
    if metadata.get("layout", "full") != "full" or task["tau_training_bounds"] != [0., .02]:
        raise ValueError("only the declared full-layout tau task is supported")
    if not 0 < metadata.get("actual_optimizer_updates", 1) <= metadata.get("planned_updates", 1):
        raise ValueError("actual updates must be positive and no greater than planned updates")


def augmented_targets(draws, design):
    return np.stack([expected_variance(draws, interval) for interval in design["products"].values()], axis=-1)


def load_validation_members(path, count):
    """Projected NPZ reads leave training and reserved-test members unopened."""
    with np.load(path, allow_pickle=False) as archive:
        return (archive["validation_parameters"][:count], archive["validation_clean"][:count],
                archive["validation_row_id"][:count])


def reference_comparison(neural, directory, metadata, design, design_hash, seed, replicates, *, expected_observations, reference_kind):
    directory = Path(directory)
    record = json.loads((directory / "run.json").read_text())
    diagnostic = json.loads((directory / "diagnostics.json").read_text())
    if (record["design_sha256"] != design_hash or record["names"] != metadata["names"]
            or record["bounds"] != metadata["bounds"] or record["arm"] != "full"
            or record["spread_multiplier"] != 1. or not record["infer_discrepancy"]):
        raise ValueError("reference and campaign target identities differ")
    observation_binding = validate_reference_observations(record, expected_observations, reference_kind)
    if digest(directory / "chains.npz") != diagnostic["chain_payload_sha256"]:
        raise ValueError("reference draws differ from the diagnosed payload")
    with np.load(directory / "chains.npz", allow_pickle=False) as values:
        chains = values["parameters"]
    if chains.shape != (record["settings"]["chains"], record["settings"]["draws"], len(NAMES)):
        raise ValueError("reference chain/draw structure differs from run record")
    bounds = np.asarray(metadata["bounds"])
    if not np.isfinite(chains).all() or np.any(chains < bounds[:, 0]) or np.any(chains > bounds[:, 1]):
        raise ValueError("reference draws violate strict finite declared prior support")
    neural_support = neural_support_report(neural, bounds)
    if neural_support["material_outside_draw_count"]:
        raise ValueError("neural draws materially outside prior: " + json.dumps(neural_support))
    combined_neural = np.concatenate([neural, augmented_targets(neural, design)], axis=-1)
    combined_chains = np.concatenate([chains, augmented_targets(chains, design)], axis=-1)
    combined_diagnostics = copy.deepcopy(diagnostic)
    combined_diagnostics["parameters"].update(diagnostic["products"]["parameters"])
    result = uncertainty.conditional_agreement(combined_neural, combined_chains, combined_diagnostics,
                [*NAMES, *design["products"]], seed=seed, replicates=replicates, family_checks=FAMILY_CHECKS)
    result["identity"] = {"run_sha256": digest(directory / "run.json"), "chains_sha256": digest(directory / "chains.npz"),
                          "diagnostics_sha256": digest(directory / "diagnostics.json"), "observation": observation_binding}
    result["neural_support"] = neural_support
    result["scope"] += "; family predeclared across market plus all12 synthetic reference cases, six coordinates plus two products"
    return result


def evaluate(args):
    from scripts.train_heston_pilot_campaign import load_checkpoint
    from scripts.train_heston_pilot import sample
    workflow, metadata = load_checkpoint(args.checkpoint)
    design = json.loads(args.design.read_text())
    design_hash = digest(args.design)
    records = design["records"]
    bank_record = json.loads((args.bank / "bank.json").read_text())
    payload_path = args.bank / bank_record["payload"]
    if payload_path.resolve().parent != args.bank.resolve():
        raise ValueError("bank payload must remain inside bank directory")
    validate_identity(metadata, design_hash, bank_record, digest(args.bank / "bank.json"), digest(payload_path))
    if len(records["mid"]) != 54 or args.cases > 512 or args.cases < 1:
        raise ValueError("initial evaluation supports full54 and at most512 development cases")
    args.out.mkdir(parents=True, exist_ok=False)
    physical, clean, row_ids = load_validation_members(payload_path, args.cases)
    if len(physical) != args.cases or len(np.unique(row_ids)) != args.cases:
        raise ValueError("insufficient or duplicate validation row IDs")
    bounds = np.asarray(metadata["bounds"])
    truth, observations = fresh_development(physical, clean, records, bounds, args.seed)
    mask, multipliers = np.ones(54, dtype=bool), np.ones(args.cases)
    started = time.perf_counter()
    batches = [sample(workflow, NAMES, observations[start:start+16], multipliers[start:start+16], records,
                      mask, args.draws, args.seed + 1000 + start) for start in range(0, args.cases, 16)]
    posterior = np.concatenate(batches)
    market = sample(workflow, NAMES, np.asarray(records["mid"])[None, :], np.ones(1), records,
                    mask, args.market_draws, args.seed + 2000)[0]
    support = {"development": neural_support_report(posterior, bounds), "market": neural_support_report(market, bounds)}
    if any(item["material_outside_draw_count"] for item in support.values()):
        write_json(args.out / "support-failure.json", support)
        raise ValueError("material posterior support violation; exact deviations saved without modifying draws")
    np.savez_compressed(args.out / "draws.npz", validation_truth=truth, validation_observations=observations,
                        validation_draws=posterior, validation_row_id=row_ids, market_draws=market)
    report = {"format": "heston_full_campaign_development_evaluation_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "scientific_acceptance": False, "scope": __doc__, "checkpoint": metadata,
              "settings": {"cases": args.cases, "draws_per_case": args.draws, "batch_size": 16,
                           "market_draws": args.market_draws, "seed": args.seed, "bootstrap_replicates_per_blocklength": args.bootstrap},
              "cohort": "first512 validation base rows at most; reused validation latents/clean prices with independent fresh tau and Gaussian errors; c=1. Reserved final2000 untouched.",
              "cohort_sha256": fingerprint({"row_ids": row_ids.tolist(), "physical": physical.tolist(), "clean": clean.tolist()}),
              "draws_sha256": digest(args.out / "draws.npz"), "script_sha256": digest(__file__),
              "parameter_calibration": calibration_metrics(posterior, truth, NAMES, args.seed + 3000),
              "product_calibration": calibration_metrics(augmented_targets(posterior, design), augmented_targets(truth, design), list(design["products"]), args.seed + 3001),
              "numeric_support_boundary_draw_count": int(np.any((posterior <= bounds[:, 0]) | (posterior >= bounds[:, 1]), axis=-1).sum()),
              "neural_support": support,
              "market": reference_comparison(market, args.reference, metadata, design, design_hash, args.seed + 4000, args.bootstrap,
                         expected_observations=records["mid"], reference_kind="market"),
              "wall_seconds": time.perf_counter() - started}
    if args.reference_panel is not None:
        panel = json.loads(args.reference_panel.read_text())
        if panel["design_sha256"] != design_hash or panel["task_bounds"] != metadata["bounds"]:
            raise ValueError("synthetic reference panel has a different task identity")
        report["conditional_references"] = {}
        retained = {}
        for case in panel["cases"]:
            directory = args.conditional_root / f"case-{case['index']:03d}"
            if not all((directory / name).exists() for name in ("chains.npz", "diagnostics.json", "run.json", "case.json")):
                report["conditional_references"][case["id"]] = {"status": "not_run", "index": case["index"]}
                continue
            run = json.loads((directory / "run.json").read_text())
            case_path = directory / "case.json"
            case_record = json.loads(case_path.read_text())
            if run["case_json_sha256"] != digest(case_path):
                raise ValueError("synthetic case or panel identity differs from reference")
            observed = validate_synthetic_case(run, case_record, case, digest(args.reference_panel), design_hash, records)
            neural = sample(workflow, NAMES, observed[None, :], np.ones(1), records, mask,
                            args.market_draws, args.seed + 5000 + case["index"])[0]
            retained[f"case_{case['index']:03d}"] = neural
            report["conditional_references"][case["id"]] = reference_comparison(
                neural, directory, metadata, design, design_hash, args.seed + 6000 + case["index"], args.bootstrap,
                expected_observations=observed, reference_kind="synthetic")
        if retained:
            np.savez_compressed(args.out / "conditional-draws.npz", **retained)
            report["conditional_draws_sha256"] = digest(args.out / "conditional-draws.npz")
        report["reference_panel_sha256"] = digest(args.reference_panel)
    report["wall_seconds"] = time.perf_counter() - started
    write_json(args.out / "evaluation.json", report)
    print(json.dumps({"out": str(args.out), "market_status": report["market"]["status"], "scientific_acceptance": False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    panel = sub.add_parser("panel")
    panel.add_argument("--design", type=Path, required=True)
    panel.add_argument("--out", type=Path, required=True)
    panel.add_argument("--kappa-max", type=float, default=10.)
    panel.add_argument("--xi-max", type=float, default=3.)
    panel.add_argument("--seed", type=int, default=261111)
    run = sub.add_parser("evaluate")
    for name in ("checkpoint", "design", "bank", "reference", "out"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--cases", type=int, default=512)
    run.add_argument("--draws", type=int, default=512)
    run.add_argument("--market-draws", type=int, default=4096)
    run.add_argument("--bootstrap", type=int, default=200)
    run.add_argument("--seed", type=int, default=261112)
    run.add_argument("--reference-panel", type=Path)
    run.add_argument("--conditional-root", type=Path)
    args = parser.parse_args()
    if args.command == "evaluate":
        if (args.reference_panel is None) != (args.conditional_root is None):
            parser.error("reference-panel and conditional-root must be supplied together")
        evaluate(args)
    else:
        design = json.loads(args.design.read_text())
        bounds = [[.01, .16], [.1, args.kappa_max], [.01, .16], [.1, args.xi_max], [-.95, -.1], [0., .02]]
        declaration = declare_reference_panel(design, digest(args.design), bounds, args.seed)
        if args.out.exists():
            raise FileExistsError(args.out)
        write_json(args.out, declaration)


if __name__ == "__main__":
    main()
