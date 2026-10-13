"""Opt-in local-data/checkpoint round trip using cached HKDE validation panels.

No bank generation, training, market data, or reserved-test evaluation. Before
the simulator PR merges, pass its checkout with --simulator-root.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ.setdefault("KERAS_BACKEND", "jax")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator-root", type=Path, default=root)
    parser.add_argument("--out", type=Path, help="Optional new JSON report file")
    args = parser.parse_args()
    simulator = args.simulator_root.resolve()
    # Temporary bridge between review checkouts; never part of the library API.
    sys.path.insert(0, str(root / "src"))
    import sbi4finance
    sbi4finance.__path__.append(str(simulator / "src/sbi4finance"))
    import numpy as np
    from sbi4finance import hkde_inference as inference
    from sbi4finance.datasets import load_artifact, write_artifact
    from sbi4finance.datasets.options import prepare_hkde_panel, hkde_panel_prices

    source = simulator / "src/sbi4finance/hkde_inference.py"
    require(Path(inference.__file__).resolve() == source,
            "HKDE adapter did not resolve to the requested simulator checkout")
    checkpoint = simulator / "checkpoints/hkde-theta-intensity-provisional"
    results = simulator / "docs/results/hkde-inference"
    task = json.loads((checkpoint / "task.json").read_text())
    training = json.loads((checkpoint / "training.json").read_text())
    validation = json.loads((results / "neural-validation.json").read_text())
    checks = results / "validation-checks.npz"
    model_hash = sha(checkpoint / "model.keras")
    require(task == training["bank_configuration"]["task"], "saved tasks differ")
    require(training["adapter_id"] == inference.ADAPTER_ID, "adapter ID differs")
    require(training["checkpoint_sha256"] == model_hash == validation["checkpoint_sha256"],
            "checkpoint hash differs from its training/validation records")
    require(training["source_sha256"]["hkde_inference.py"] == sha(source),
            "checkpoint adapter source changed")
    require(validation["split"] == "validation" and not validation["reserved_test_evaluated"]
            and validation["checks_sha256"] == sha(checks), "cached validation checks differ")
    with np.load(checks, allow_pickle=False) as saved:
        panels, labels, ids = [saved[key][:2].copy() for key in ("observations", "truth", "case_ids")]
    require(panels.shape == (2, 9, 1), "expected two complete cached validation panels")

    def equal(left, right):
        return left.keys() == right.keys() and all(np.array_equal(left[k], right[k]) for k in left)

    def conditions_for(artifact):
        prices = np.asarray(hkde_panel_prices(artifact, checkpoint_task=task))[None, :, None]
        return inference.conditioning_data(prices, task)

    loaded_conditions = []
    with tempfile.TemporaryDirectory(prefix="hkde-data-check-") as directory:
        for index, panel in enumerate(panels):
            records = [{"contract_id": f"c{j}", **contract, "price": float(price[0]), "quote_time": None}
                       for j, (contract, price) in enumerate(zip(task["context"]["contracts"], panel, strict=True))]
            metadata = {
                "schema_version": 1, "artifact_id": str(ids[index]), "task_id": task["task_id"],
                "artifact_type": "option_panel", "representation": "raw_quote", "measure": "Q",
                "data_origin": "synthetic", "purpose": "validation",
                "conventions": {"quote_form": "single_premium", "price_unit": "currency",
                                "exercise": "European", "settlement": "cash",
                                "premium_basis": "per_underlying_unit"},
                "provenance": {"source": "committed HKDE synthetic validation panels",
                               "checks_sha256": sha(checks), "case_id": str(ids[index])},
                "truth": dict(zip(task["parameter_names"], map(float, labels[index]), strict=True)),
            }
            prepared = prepare_hkde_panel(records, metadata=metadata, task=task)
            destination = Path(directory) / f"panel-{index}"
            write_artifact(destination, prepared.metadata, prepared.rows)
            loaded = load_artifact(destination)
            conditions = conditions_for(loaded)
            require(equal(conditions, inference.conditioning_data(panel[None], task)),
                    "CSV/JSON round trip changed conditioning")
            changed = deepcopy(loaded)
            changed.metadata["truth"] = {"unused_label": -999.0}
            for row in changed.rows:
                row["quote_time"] = "auxiliary metadata does not condition the model"
            require(equal(conditions, conditions_for(changed)),
                    "truth or auxiliary metadata leaked into conditioning")
            changed.metadata.pop("truth")
            changed.metadata["columns"] = [c for c in changed.metadata["columns"] if c["role"] != "auxiliary"]
            for row in changed.rows:
                row.pop("quote_time")
            unlabelled = Path(directory) / f"unlabelled-{index}"
            write_artifact(unlabelled, changed.metadata, changed.rows)
            unlabelled_conditions = conditions_for(load_artifact(unlabelled))
            require(equal(conditions, unlabelled_conditions),
                    "unlabelled observations changed conditioning")
            loaded_conditions.append(unlabelled_conditions["observations"])

        rejected = []
        for change in ("order", "contract", "context", "noise", "prior", "representation", "unit"):
            bad = deepcopy(loaded)
            if change == "order":
                bad.rows = tuple(reversed(bad.rows))
            elif change == "contract":
                bad.rows[0]["strike"] += 1.0
            elif change == "context":
                bad.metadata["scientific_task"]["context"]["spot"] += 1.0
            elif change == "noise":
                bad.metadata["scientific_task"]["context"]["quote_noise_std"] *= 2.0
            elif change == "prior":
                bad.metadata["scientific_task"]["prior_bounds"][0][0] *= 0.9
            elif change == "representation":
                bad.metadata["representation"] = "processed_iv"
            else:
                bad.metadata["conventions"]["price_unit"] = "decimal_iv"
            try:
                conditions_for(bad)
            except ValueError:
                rejected.append(change)
            else:
                raise RuntimeError(f"incompatible {change} was accepted")

    workflow = inference.build_workflow(task)
    workflow.load_approximator(checkpoint / "model.keras")
    direct = workflow.sample(num_samples=32, conditions=inference.conditioning_data(panels, task), seed=71026)
    restored = workflow.sample(num_samples=32, conditions={"observations": np.concatenate(loaded_conditions)}, seed=71026)
    require(equal(direct, restored), "same-seed posterior draws changed after local IO")
    for name, (lower, upper) in zip(task["parameter_names"], task["prior_bounds"], strict=True):
        require(np.isfinite(restored[name]).all() and np.all(restored[name] > lower)
                and np.all(restored[name] < upper), f"invalid posterior draws for {name}")
    report = {
        "passed": True, "evidence_scope": "synthetic data/adapter interoperability only",
        "source_commit": subprocess.check_output(["git", "-C", str(simulator), "rev-parse", "HEAD"], text=True).strip(),
        "data_source_sha256": {name: sha(root / name) for name in (
            "src/sbi4finance/datasets/local.py", "src/sbi4finance/datasets/options.py",
            "scripts/check_hkde_data.py")},
        "checkpoint_sha256": model_hash, "task_sha256": sha(checkpoint / "task.json"),
        "checks_sha256": sha(checks), "adapter_sha256": sha(source), "adapter_id": inference.ADAPTER_ID,
        "case_ids": [str(value) for value in ids], "draws_per_case": 32, "seed": 71026,
        "conditions_exact": True, "seeded_draws_exact": True, "truth_and_auxiliary_excluded": True,
        "rejected_incompatibilities": rejected, "reserved_test_evaluated": False,
        "checkpoint_status": "provisional", "original_scientific_validation_passed": validation["passed"],
        "limits": "This checks transport and conditioning, not posterior accuracy or real-market fit.",
    }
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out:
        with args.out.open("x", encoding="utf-8") as handle:
            handle.write(output)
    print(output, end="")


if __name__ == "__main__":
    main()
