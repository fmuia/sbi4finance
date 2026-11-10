"""Development checks for the fixed-c Heston checkpoint, never asset acceptance.

Use the declared four-case panel and chain-preserving matched references.
Missing or unconverged references remain inconclusive. Only validation members
of the historical clean bank may be read for population diagnostics.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbi4finance.calibration import heston_budget_diagnostics, validation


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def observations_digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def support_report(draws, bounds):
    """Report float32 endpoint rounding separately from material violations."""
    values, bounds = np.asarray(draws), np.asarray(bounds)
    if (values.ndim < 2 or bounds.shape != (values.shape[-1], 2)
            or not np.isfinite(values).all()):
        raise ValueError("finite posterior draws and matching physical bounds required")
    tolerance = 8 * np.maximum(np.abs(np.spacing(bounds[:, 0].astype(np.float32))),
                               np.abs(np.spacing(bounds[:, 1].astype(np.float32))))
    outside = (values < bounds[:, 0]) | (values > bounds[:, 1])
    material = (values < bounds[:, 0]-tolerance) | (values > bounds[:, 1]+tolerance)
    return {"outside_value_count": int(outside.sum()),
            "material_outside_value_count": int(material.sum()),
            "rounding_tolerance": tolerance.tolist(), "draws_clipped": False}


def compare_draws(neural, chains, diagnostic, names, products, *, seed, replicates=400,
                  family_checks=112):
    from scripts.run_heston_spread_reference import product_values
    combined_neural = np.concatenate([neural, product_values(neural, products)], axis=-1)
    combined_chains = np.concatenate([chains, product_values(chains, products)], axis=-1)
    combined_diagnostic = copy.deepcopy(diagnostic)
    combined_diagnostic["parameters"].update(diagnostic["products"]["parameters"])
    return heston_budget_diagnostics.conditional_agreement(
        combined_neural, combined_chains, combined_diagnostic,
        [*names, *products], seed=seed, replicates=replicates,
        family_checks=family_checks, mean_limit=.3, width_limits=(.75, 1.25))


def check_reference_binding(record, diagnostics, chains_path, *, target, design_sha256,
                            case_id, observations, panel_sha256, products, diagnostics_path):
    """Reject target/data substitutions before using a converged-looking trace."""
    if (record["task"] != target or record["design_sha256"] != design_sha256
            or record["case_id"] != case_id
            or record["observation_sha256"] != observations_digest(observations)
            or not np.array_equal(record["observations"], observations)
            or record["panel_sha256"] != panel_sha256 or record["products"] != products
            or record["attempts"][record["selected_attempt"]]["diagnostics_sha256"] != digest(diagnostics_path)
            or diagnostics["chain_payload_sha256"] != digest(chains_path)):
        raise ValueError("reference target, observation, design or diagnosed payload differs")


def validate_development_arrays(truth, clean, bounds, count, quote_count):
    if (truth.shape != (count, len(bounds)) or clean.shape != (count, quote_count)
            or not np.isfinite(truth).all() or not np.isfinite(clean).all()
            or np.any(truth < bounds[:, 0]) or np.any(truth > bounds[:, 1])):
        raise ValueError("validation data shape, finiteness or physical support differs")


def evaluate(checkpoint, panel_path, references, bank_path, *, seed=261112,
             conditional_draws=4096, validation_rows=512, validation_draws=512,
             replicates=400):
    from scripts.train_heston_spread_baseline import load_checkpoint, sample
    from scripts.run_heston_spread_reference import load_case
    workflow, metadata = load_checkpoint(checkpoint)
    task = metadata["task"]
    panel_path, bank_path, references = Path(panel_path), Path(bank_path), Path(references)
    panel = json.loads(panel_path.read_text())
    target = task["target"]
    if (target != panel["fixed_task"] or target["infer_c"] or target["fixed_c"] != 1.
            or task["design_sha256"] != panel["design_sha256"]):
        raise ValueError("only the declared fixed-c=1 task and matched panel are supported")
    bank_record = json.loads((bank_path / "bank.json").read_text())
    payload = bank_path / bank_record["payload"]
    if (metadata["bank_record_sha256"] != digest(bank_path / "bank.json")
            or metadata["bank_identity"] != bank_record["identity"]
            or metadata["bank_payload_sha256"] != bank_record["payload_sha256"]
            or digest(payload) != bank_record["payload_sha256"]):
        raise ValueError("checkpoint and validation bank payload differ")
    bounds = np.asarray(target["prior_bounds"])
    names = target["parameter_names"]
    result = {"format": "heston_spread_development_evaluation_v1", "seed": seed,
        "checkpoint": str(Path(checkpoint)), "panel_sha256": digest(panel_path),
        "task": target, "source_sha256": digest(__file__), "cases": {},
        "scientific_acceptance": False, "reserved_test_arrays_opened": False,
        "scope": "Development conditional/population checks; no final-pool or lesson acceptance."}
    for i, case in enumerate(panel["cases"]):
        case_id = case["id"]
        _, _, _, y = load_case(panel_path, case_id)
        neural = sample(workflow, task, y[None], conditional_draws, seed=seed+i)[0]
        support = support_report(neural, bounds)
        if support["material_outside_value_count"]:
            raise ValueError(f"posterior support violation at {case_id}: {support}")
        entry = {"neural_support": support,
            "neural_quantiles_05_50_95": np.quantile(neural, [.05,.5,.95], axis=0).tolist()}
        directory = references / case_id
        required = [directory / name for name in ("run.json", "diagnostics.json")]
        if not all(path.exists() for path in required):
            entry.update(status="inconclusive", reason="matched completed reference unavailable")
        else:
            record, diagnostic = [json.loads(path.read_text()) for path in required]
            if record.get("status") != "completed" or "chain_payload_path" not in record:
                entry.update(status="inconclusive", reason="matched reference has not completed")
                result["cases"][case_id] = entry
                continue
            chains_path = Path(record["chain_payload_path"])
            check_reference_binding(record, diagnostic, chains_path, target=target,
                design_sha256=task["design_sha256"], case_id=case_id, observations=y,
                panel_sha256=digest(panel_path), products=panel["products"], diagnostics_path=required[1])
            with np.load(chains_path, allow_pickle=False) as values:
                chains = values["parameters"]
            retained = record["attempts"][record["selected_attempt"]]["draws"]
            if chains.shape != (record["settings"]["chains"], retained, len(names)):
                raise ValueError("reference chain/draw dimensions differ from its run record")
            if (not np.isfinite(chains).all() or np.any(chains < bounds[:, 0])
                    or np.any(chains > bounds[:, 1])):
                raise ValueError("reference posterior violates declared prior support")
            entry["comparison"] = compare_draws(neural, chains, diagnostic, names,
                panel["products"], seed=seed+100+i, replicates=replicates)
            entry["status"] = entry["comparison"]["status"]
            entry["reference_diagnostics_passed"] = diagnostic["mcmc_diagnostics_passed"]
            if diagnostic.get("reference_ready") is not True:
                entry.update(status="inconclusive", reason="matched reference is smoke, unconverged or numerically unresolved")
        result["cases"][case_id] = entry
    with np.load(payload, allow_pickle=False) as values:
        truth = values["validation_parameters"][:validation_rows].copy()
        clean = values["validation_clean"][:validation_rows].copy()
    h = np.asarray(task["records"]["half_spread"])
    validate_development_arrays(truth, clean, bounds, validation_rows, len(h))
    rng = np.random.default_rng(np.random.SeedSequence([0x53504556, seed]))
    noisy = clean + h * rng.standard_normal(clean.shape)
    draws = np.concatenate([sample(workflow, task, noisy[start:start+16], validation_draws,
                                  seed=seed+1000+start)
                            for start in range(0, len(noisy), 16)])
    support = support_report(draws, bounds)
    if support["material_outside_value_count"]:
        raise ValueError("development posterior support violation")
    ranks = validation.sbc(draws, truth, seed=seed+2000)
    ranks.pop("fractional_ranks")
    result["population"] = {"rows": validation_rows, "draws_per_case": validation_draws,
        "support": support, "coverage": validation.coverage(draws, truth), "rank_dkw": ranks,
        "scope": "Prior-predictive development calibration under fixed c=1; not conditional or market acceptance."}
    statuses = [entry["status"] for entry in result["cases"].values()]
    result["conditional_status"] = ("fail" if "fail" in statuses else
        "pass" if statuses and all(value == "pass" for value in statuses) else "inconclusive")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=261112)
    parser.add_argument("--conditional-draws", type=int, default=4096)
    parser.add_argument("--validation-rows", type=int, default=512)
    parser.add_argument("--validation-draws", type=int, default=512)
    parser.add_argument("--replicates", type=int, default=400)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = evaluate(args.checkpoint,args.panel,args.references,args.bank,seed=args.seed,
        conditional_draws=args.conditional_draws,validation_rows=args.validation_rows,
        validation_draws=args.validation_draws,replicates=args.replicates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix+".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    temporary.replace(args.out)
    print(json.dumps({"out":str(args.out),"conditional_status":result["conditional_status"],
                      "scientific_acceptance":False}))


if __name__ == "__main__":
    main()
