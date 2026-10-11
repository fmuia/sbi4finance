"""Opt-in HKDE bank -> local panel -> checkpoint interoperability check.

Use --bank for an existing simulator bank, or generate an independent five-row
smoke bank with the simulator's writer. Only a selected validation observation
conditions the checkpoint. This is not a new posterior-accuracy evaluation.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
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
    parser.add_argument("--bank", type=Path, help="Existing option_bank_v1 directory")
    parser.add_argument("--example-id", default="validation:0")
    parser.add_argument("--out", type=Path, help="Optional new JSON report file")
    args = parser.parse_args()
    simulator = args.simulator_root.resolve()
    # Temporary bridge between review checkouts; never part of the library API.
    sys.path.insert(0, str(root / "src"))
    import sbi4finance
    sbi4finance.__path__.append(str(simulator / "src/sbi4finance"))
    import numpy as np
    from sbi4finance import hkde_inference as inference, option_bank
    from sbi4finance.datasets import (
        hkde_panel_prices, load_artifact, load_hkde_bank_panel, write_artifact,
    )

    source = simulator / "src/sbi4finance/hkde_inference.py"
    bank_source = simulator / "src/sbi4finance/option_bank.py"
    require(Path(inference.__file__).resolve() == source
            and Path(option_bank.__file__).resolve() == bank_source,
            "HKDE adapter/bank reader did not resolve to the requested checkout")
    checkpoint = simulator / "checkpoints/hkde-theta-intensity-provisional"
    validation_path = simulator / "docs/results/hkde-inference/neural-validation.json"
    task = json.loads((checkpoint / "task.json").read_text())
    training = json.loads((checkpoint / "training.json").read_text())
    validation = json.loads(validation_path.read_text())
    model_hash = sha(checkpoint / "model.keras")
    require(task == training["bank_configuration"]["task"], "saved tasks differ")
    require(training["adapter_id"] == inference.ADAPTER_ID, "adapter ID differs")
    require(training["checkpoint_sha256"] == model_hash == validation["checkpoint_sha256"],
            "checkpoint hash differs from its training/validation records")
    require(training["source_sha256"]["hkde_inference.py"] == sha(source),
            "checkpoint adapter source changed")
    require(validation["split"] == "validation" and not validation["reserved_test_evaluated"],
            "expected the recorded validation report without reserved-test evaluation")

    def equal(left, right):
        return left.keys() == right.keys() and all(np.array_equal(left[k], right[k]) for k in left)

    def conditions_for(artifact):
        prices = np.asarray(hkde_panel_prices(artifact, checkpoint_task=task))[None, :, None]
        return inference.conditioning_data(prices, task)

    with tempfile.TemporaryDirectory(prefix="hkde-bank-data-check-") as directory:
        temporary = Path(directory)
        if args.bank is None:
            bank_path = temporary / "smoke-bank"
            smoke_config = {
                "task": task, "root_seed": 72026,
                "splits": {"train": 2, "validation": 2, "test": 1},
            }
            require(smoke_config["root_seed"] != training["bank_configuration"]["root_seed"],
                    "smoke bank must use an independent seed")
            option_bank.write_bank(bank_path, smoke_config)
        else:
            bank_path = args.bank.resolve()
        # The owner reader checks all split files; no test observations enter
        # inference or metrics here, including the independent smoke test row.
        manifest, arrays = option_bank.load_bank(bank_path)
        prepared = load_hkde_bank_panel(
            bank_path, example_id=args.example_id, checkpoint_task=task,
        )
        indices = np.flatnonzero(arrays["validation"]["example_id"] == args.example_id)
        require(len(indices) == 1, "selected validation example is missing or ambiguous")
        index = int(indices[0])
        observation = arrays["validation"]["observations"][index:index + 1]
        direct_conditions = inference.conditioning_data(observation, task)
        expected_truth = dict(zip(task["parameter_names"],
                                  map(float, arrays["validation"]["parameters"][index]), strict=True))
        require(prepared.truth == expected_truth, "synthetic truth was not preserved separately")
        require(set(prepared.observations) == {"price"}, "unexpected conditioning columns")
        write_artifact(temporary / "panel", prepared.metadata, prepared.rows)
        loaded = load_artifact(temporary / "panel")
        restored_conditions = conditions_for(loaded)
        require(equal(direct_conditions, restored_conditions), "local IO changed conditioning")

        unlabelled = deepcopy(loaded)
        unlabelled.metadata.pop("truth")
        auxiliary = {column["name"] for column in unlabelled.metadata["columns"]
                     if column["role"] == "auxiliary"}
        unlabelled.metadata["columns"] = [column for column in unlabelled.metadata["columns"]
                                           if column["name"] not in auxiliary]
        for row in unlabelled.rows:
            for key in auxiliary:
                row.pop(key)
        write_artifact(temporary / "unlabelled", unlabelled.metadata, unlabelled.rows)
        require(equal(restored_conditions, conditions_for(load_artifact(temporary / "unlabelled"))),
                "truth or auxiliary labels affect conditioning")
        forbidden = {"reference_prices", "parameters", "pricing_standard_errors",
                     "pricing_covariance", "refinement_difference"}
        require(not forbidden.intersection(restored_conditions), "bank labels leaked into conditions")

        rejected = []
        changed_task = deepcopy(task)
        changed_task["context"]["quote_noise_std"] *= 2
        corrupt_bank = temporary / "corrupt-bank"
        corrupt_bank.mkdir()
        for filename in ("manifest.json", "train.npz", "validation.npz", "test.npz"):
            shutil.copyfile(bank_path / filename, corrupt_bank / filename)
        (corrupt_bank / "validation.npz").write_bytes(b"corrupted validation split")
        for reason, path, identity, expected_task in (
            ("train_id", bank_path, "train:0", task),
            ("test_id", bank_path, "test:0", task),
            ("absent_validation_id", bank_path,
             f"validation:{manifest['configuration']['splits']['validation']}", task),
            ("checkpoint_noise", bank_path, args.example_id, changed_task),
            ("corrupted_validation_split", corrupt_bank, args.example_id, task),
        ):
            try:
                load_hkde_bank_panel(path, example_id=identity, checkpoint_task=expected_task)
            except ValueError:
                rejected.append(reason)
            else:
                raise RuntimeError(f"incompatible {reason} was accepted")

        workflow = inference.build_workflow(task)
        workflow.load_approximator(checkpoint / "model.keras")
        direct = workflow.sample(num_samples=32, conditions=direct_conditions, seed=72027)
        restored = workflow.sample(num_samples=32, conditions=restored_conditions, seed=72027)
        require(equal(direct, restored), "same-seed posterior draws changed after local IO")
        for name, (lower, upper) in zip(task["parameter_names"], task["prior_bounds"], strict=True):
            require(np.isfinite(restored[name]).all() and np.all(restored[name] > lower)
                    and np.all(restored[name] < upper), f"invalid posterior draws for {name}")
        report = {
            "passed": True, "evidence_scope": "synthetic bank/data/adapter interoperability only",
            "source_commit": subprocess.check_output(
                ["git", "-C", str(simulator), "rev-parse", "HEAD"], text=True).strip(),
            "data_source_sha256": {name: sha(root / name) for name in (
                "src/sbi4finance/datasets/local.py", "src/sbi4finance/datasets/options.py",
                "src/sbi4finance/datasets/banks.py", "scripts/check_hkde_bank_data.py")},
            "bank_source": "generated independent smoke bank" if args.bank is None else "existing bank",
            "bank_manifest_sha256": sha(bank_path / "manifest.json"),
            "bank_configuration_sha256": manifest["configuration_sha256"],
            "bank_task_fingerprint": manifest["task_fingerprint"],
            "bank_root_seed": manifest["configuration"]["root_seed"],
            "bank_split_counts": manifest["configuration"]["splits"],
            "validation_split_sha256": manifest["splits"]["validation"]["sha256"],
            "checkpoint_sha256": model_hash, "task_sha256": sha(checkpoint / "task.json"),
            "adapter_sha256": sha(source), "adapter_id": inference.ADAPTER_ID,
            "bank_reader_sha256": sha(bank_source), "scientific_report_sha256": sha(validation_path),
            "example_id": args.example_id, "draws_per_case": 32, "seed": 72027,
            "conditions_exact": True, "seeded_draws_exact": True,
            "truth_preserved_separately": True, "truth_and_auxiliary_excluded": True,
            "rejected_incompatibilities": rejected,
            "reserved_test_evaluated": False,
            "checkpoint_status": "provisional",
            "original_scientific_validation_passed": validation["passed"],
            "limits": "Transport and conditioning only; no new posterior-accuracy or market evidence. "
                      "The smoke writer creates an independent test row, but no test row is evaluated.",
        }
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out:
        with args.out.open("x", encoding="utf-8") as handle:
            handle.write(output)
    print(output, end="")


if __name__ == "__main__":
    main()
