"""Bundle one published synthetic HKDE validation case for offline teaching.

Replays cached paired posterior draws and model prices; no training, fresh
inference, market access, or reserved-test evaluation. Requires the simulator
workstream's original bank and published provisional-checkpoint artifacts.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New bundle directory")
    parser.add_argument("--simulator-root", type=Path, default=root)
    args = parser.parse_args()
    require(not args.out.exists(), "output must be a new directory")
    simulator = args.simulator_root.resolve()
    sys.path.insert(0, str(root / "src"))
    import sbi4finance
    # Review-checkout bridge only; scientific code stays in its owning module.
    sbi4finance.__path__.append(str(simulator / "src/sbi4finance"))
    import numpy as np
    from sbi4finance import hkde_reference as reference
    from sbi4finance.datasets import hkde_panel_prices, load_hkde_bank_panel, write_artifact

    module_root = simulator / "src/sbi4finance"
    require(Path(reference.__file__).resolve() == module_root / "hkde_reference.py",
            "pricing module did not resolve to the requested simulator checkout")
    checkpoint = simulator / "checkpoints/hkde-theta-intensity-provisional"
    results = simulator / "docs/results/hkde-inference"
    task_bytes = (checkpoint / "task.json").read_bytes()
    task = json.loads(task_bytes)
    training = json.loads((checkpoint / "training.json").read_text())
    protocol_path = checkpoint / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    validation_path = results / "neural-validation.json"
    validation = json.loads(validation_path.read_text())
    reference_path = results / "reference.json"
    accepted_reference = json.loads(reference_path.read_text())
    checks_path = results / "validation-checks.npz"
    bank_manifest_hash = sha(args.bank / "manifest.json")
    model_hash, protocol_hash = sha(checkpoint / "model.keras"), sha(protocol_path)
    require(task == training["bank_configuration"]["task"] == accepted_reference["task"],
            "scientific tasks differ")
    require(model_hash == training["checkpoint_sha256"] == validation["checkpoint_sha256"],
            "checkpoint differs from training/validation records")
    require(protocol_hash == training["protocol_sha256"] == validation["protocol_sha256"]
            == accepted_reference["protocol_sha256"], "protocol linkage differs")
    require(bank_manifest_hash == training["bank_manifest_sha256"] == validation["bank_manifest_sha256"],
            "bank differs from the published training/validation bank")
    require(sha(reference_path) == training["reference_report_sha256"]
            == validation["reference_report_sha256"] and accepted_reference["passed"],
            "numerical reference report is changed or unaccepted")
    require(sha(checks_path) == validation["checks_sha256"], "cached validation arrays changed")
    require(validation["split"] == "validation" and validation["passed"] is False
            and not validation["reserved_test_evaluated"] and not training["reserved_test_evaluated"],
            "this bundle expects the published provisional validation state")
    driver_hashes = {}
    for record in (training, validation, accepted_reference):
        for name, digest in record["source_sha256"].items():
            path = simulator / "scripts" / name if name.startswith(("train_", "evaluate_", "check_")) else module_root / name
            if name in ("train_hkde.py", "evaluate_hkde.py"):
                # These historical producers are not executed by cache replay.
                driver_hashes[name] = {"recorded": digest, "checkout": sha(path)}
            else:
                require(sha(path) == digest, f"published scientific source changed: {name}")

    case_id, target_index = "validation:257", 0
    target = protocol["targets"][target_index]
    require(target == {"strike": 95.0, "maturity": 0.75, "option_type": "put"}
            and target not in task["context"]["contracts"], "expected a held-out 95-strike nine-month put")
    panel = load_hkde_bank_panel(args.bank, example_id=case_id, checkpoint_task=task)
    conditioning = np.asarray(hkde_panel_prices(panel, checkpoint_task=task), dtype=np.float64)
    with np.load(checks_path, allow_pickle=False) as cached:
        ids = cached["case_ids"].tolist()
        require(ids.count(case_id) == 1 and ids == validation["case_ids"], "cached case identity changed")
        index = ids.index(case_id)
        require(np.array_equal(conditioning, cached["observations"][index, :, 0]),
                "bank observation differs from the cached conditioning panel")
        require(np.array_equal([panel.truth[name] for name in task["parameter_names"]], cached["truth"][index]),
                "bank truth differs from the cached evaluation labels")
        draws = cached["conditional_draws"][index].copy()
        prices = cached["target_draws"][index, :, target_index].copy()
        reference_quantiles = cached["reference_target_quantiles"][index, target_index].copy()
    count = protocol["evaluation"]["conditional_draws"]
    require(count == validation["conditional_draws_per_case"] and draws.shape == (count, 2)
            and prices.shape == (count,) and np.isfinite(draws).all() and np.isfinite(prices).all(),
            "invalid cached paired draws")
    repriced = reference.price_parameters(task, draws[:3], [target])[:, 0]
    require(np.allclose(repriced, prices[:3], rtol=0, atol=1e-10), "cached parameter/price pairing changed")
    probabilities = protocol["evaluation"]["quantiles"]
    case_report = validation["cases"][index]
    require(case_report["case_id"] == case_id
            and np.allclose(np.quantile(prices, probabilities), case_report["target_quantiles"][target_index],
                            rtol=0, atol=1e-12), "cached price quantiles differ from the report")
    gate_name = "max_abs_target_quantile_error_currency"
    gate = protocol["evaluation"]["gates"][gate_name]
    require(validation["thresholds"][gate_name] == gate and not validation["gates"][gate_name],
            "expected the recorded failed price-error gate")

    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "task.json").write_bytes(task_bytes)
    write_artifact(args.out / "panel", panel.metadata, panel.rows)
    np.savez_compressed(args.out / "posterior.npz", parameter_draws=draws,
                        model_price_draws=prices, conditioning_prices=conditioning)
    bundle = {
        "format": "hkde_teaching_v1", "case_id": case_id,
        "parameter_names": task["parameter_names"], "target_contract": target,
        "quantiles": probabilities, "reference_price_quantiles": reference_quantiles.tolist(),
        "checkpoint_status": "provisional", "scientific_validation_passed": validation["passed"],
        "price_error_gate_currency": gate,
        "original_max_price_quantile_error_currency": validation["metrics"][gate_name],
        "reserved_test_evaluated": False, "cached_draw_count": count,
        "sources": {
            "simulator_commit": subprocess.check_output(
                ["git", "-C", str(simulator), "rev-parse", "HEAD"], text=True).strip(),
            "checkpoint_sha256": model_hash, "bank_manifest_sha256": bank_manifest_hash,
            "checks_sha256": sha(checks_path), "validation_report_sha256": sha(validation_path),
            "reference_report_sha256": sha(reference_path), "protocol_sha256": protocol_hash,
            "adapter_sha256": sha(module_root / "hkde_inference.py"),
            "pricing_source_sha256": sha(module_root / "hkde_reference.py"),
            "builder_sha256": sha(__file__),
            "bank_panel_loader_sha256": sha(root / "src/sbi4finance/datasets/banks.py"),
            "producer_driver_sha256": driver_hashes,
        },
        "files": {name: sha(args.out / name) for name in (
            "task.json", "panel/manifest.json", "panel/observations.csv", "posterior.npz")},
        "notes": [
            "Cached validation replay, not fresh inference or new statistical validation.",
            "Paired parameter rows propagate into Q model-price uncertainty, not realized payoffs or P risk.",
            "Synthetic truth is evaluation-only; the held-out target never conditions the posterior.",
            "The original failed price gate remains visible; no reserved-test evaluation or model copy.",
            "Historical training/audit driver hashes are preserved separately from checkout hashes; "
            "those drivers were subsequently strengthened with provenance guards. This builder does not rerun them.",
        ],
    }
    (args.out / "bundle.json").write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n")
    print(f"Prepared {args.out}: {case_id}, {count} cached paired draws; checkpoint remains provisional.")


if __name__ == "__main__":
    main()
