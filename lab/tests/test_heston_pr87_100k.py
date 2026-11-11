"""Fixed parent prefix/W, native continuation order and guarded new loaders."""
from copy import deepcopy
from importlib import metadata
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import reproduce_heston_pr87_100k as trainer
from scripts import train_heston_spread_scale as scale
from test_heston_spread_training import fixture


def parent_fixture(root):
    design, _, _, _ = fixture(root)
    task, model = scale.make_task(design)
    rng = np.random.default_rng(92)
    arrays = {}
    spread = np.asarray(task["records"]["half_spread"])
    for split, count in (("train", 128), ("validation", 32)):
        parameters = rng.uniform(model.bounds[:, 0], model.bounds[:, 1], (count, 6))
        quotes = rng.normal(size=(count, 54))
        arrays.update({f"{split}_{name}": parameters[:, index:index + 1]
                       for index, name in enumerate(model.names)})
        arrays.update({f"{split}_observations": quotes / spread,
                       f"{split}_quotes": quotes, f"{split}_clean_prices": quotes + .1})
    whitening = trainer.NotebookWhitening(arrays["train_observations"])
    parentdir = root / "parent"
    parentdir.mkdir()
    trainer.write_json(parentdir / "whitening.json", whitening.to_dict())
    return {"directory": parentdir, "task": task, "arrays": arrays, "whitening": whitening,
            "identity": {"directory": str(parentdir), "train_rows": 128, "validation_rows": 32,
                "after_validation_rng_state": deepcopy(rng.bit_generator.state),
                "whitening": trainer._identity(parentdir / "whitening.json")}}, model


class ParentDataGuards(unittest.TestCase):
    def test_original_prefix_and_all_validation_arrays_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent, _ = parent_fixture(Path(temporary))
            arrays = {key: value.copy() for key, value in parent["arrays"].items()}
            trainer._validate_parent_prefix(arrays, parent)
            arrays["train_kappa"][0, 0] += .01
            with self.assertRaisesRegex(ValueError, "prefix"):
                trainer._validate_parent_prefix(arrays, parent)
            arrays = {key: value.copy() for key, value in parent["arrays"].items()}
            arrays["validation_quotes"][0, 0] += .01
            with self.assertRaisesRegex(ValueError, "validation"):
                trainer._validate_parent_prefix(arrays, parent)

    def test_loader_checks_identity_and_every_prefix_before_model_deserialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent, _ = parent_fixture(root)
            training = root / "training"
            checkpoint = training / "checkpoints" / "epoch-000002"
            checkpoint.mkdir(parents=True)
            arrays = {key: np.concatenate((value, value)) if key.startswith("train_") else value.copy()
                      for key, value in parent["arrays"].items()}
            contract = {"format": "heston_pr87_100k_protocol_v1", "parent": parent["identity"],
                "smoke_test_only": True, "epochs": 2, "batches_per_epoch": 1,
                "expected_optimizer_updates": 2, "train_rows": 256, "validation_rows": 32}
            protocol = root / "protocol.json"
            trainer.write_json(protocol, contract)
            for destination in (training, checkpoint):
                (destination / "whitening.json").write_bytes((parent["directory"] / "whitening.json").read_bytes())
            (checkpoint / "model.keras").write_bytes(b"never deserialize these fixture bytes")
            (checkpoint / "state.npz").write_bytes(b"never deserialize these fixture bytes")
            record = {"format": "heston_pr87_100k_checkpoint_v1", "task": parent["task"],
                "task_fingerprint": trainer.fingerprint(parent["task"]),
                "parameter_names": parent["task"]["target"]["parameter_names"],
                "bounds": parent["task"]["target"]["prior_bounds"], "protocol": contract,
                "parent": parent["identity"], "protocol_path": str(protocol),
                "protocol_sha256": trainer.digest(protocol), "data_payload": "data.npz",
                "smoke_test_only": True, "primary_epoch": 2, "native_epoch": 2,
                "actual_optimizer_updates": 2, "model_sha256": trainer.digest(checkpoint / "model.keras"),
                "state_sha256": trainer.digest(checkpoint / "state.npz"),
                "whitening_sha256": trainer.digest(checkpoint / "whitening.json")}

            def persist():
                trainer.write_npz(training / "data.npz", arrays)
                record.update(data_payload_sha256=trainer.digest(training / "data.npz"),
                    data_arrays={key: trainer.array_summary(value) for key, value in arrays.items()})
                trainer.write_json(checkpoint / "checkpoint.json", record)
                run = {**record, "format": "heston_pr87_100k_run_v1", "status": "complete",
                    "completed_epochs": 2, "checkpoints": [{"epoch": 2, "path": str(checkpoint),
                        "checkpoint_sha256": trainer.digest(checkpoint / "checkpoint.json")}]}
                for filename in ("run.json", "training.json"):
                    trainer.write_json(training / filename, run)

            persist()
            with patch.object(trainer, "_read_parent", return_value=parent), patch.object(
                    trainer, "protocol_contract", return_value=contract), patch.object(
                    trainer, "build_workflow", side_effect=RuntimeError("model boundary")) as build:
                with self.assertRaisesRegex(RuntimeError, "model boundary"):
                    trainer.load_checkpoint(checkpoint)
                build.reset_mock()
                original = protocol.read_bytes()
                protocol.write_bytes(original + b" ")
                with self.assertRaises(ValueError):
                    trainer.load_checkpoint(checkpoint)
                protocol.write_bytes(original)
                with patch.object(trainer, "protocol_contract", return_value={**contract, "changed_source": True}):
                    with self.assertRaises(ValueError):
                        trainer.load_checkpoint(checkpoint)
                with (training / "data.npz").open("ab") as stream:
                    stream.write(b"corrupted generated payload")
                with self.assertRaises(ValueError):
                    trainer.load_checkpoint(checkpoint)
                persist()
                original = (training / "whitening.json").read_bytes()
                (training / "whitening.json").write_bytes(original + b" ")
                with self.assertRaises(ValueError):
                    trainer.load_checkpoint(checkpoint)
                (training / "whitening.json").write_bytes(original)
                # Rehashing a modified prefix must still fail the parent comparison.
                arrays["train_kappa"][0, 0] += .01
                persist()
                with self.assertRaisesRegex(ValueError, "prefix"):
                    trainer.load_checkpoint(checkpoint)
                build.assert_not_called()


try:
    CORE = metadata.version("bayesflow") == "2.1.0"
except metadata.PackageNotFoundError:
    CORE = False


@unittest.skipUnless(CORE, "requires locked BF2.1 environment")
class NativeContinuationTests(unittest.TestCase):
    def test_restored_parent_rng_all_appended_priors_then_noises_and_exact_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent, model = parent_fixture(Path(temporary))
            before = deepcopy(parent["identity"]["after_validation_rng_state"])
            old_whitening = parent["whitening"].to_dict()
            clean = np.arange(54, dtype=float) + 100.
            progress = []
            with patch.object(trainer.HestonSpreadPilot, "prices", return_value=clean) as pricing:
                arrays, states = trainer.append_data(parent, 7, progress=progress.append)
            rng = np.random.default_rng()
            rng.bit_generator.state = before
            parameters = np.asarray([rng.uniform(model.bounds[:, 0], model.bounds[:, 1]) for _ in range(7)])
            spread = np.asarray(parent["task"]["records"]["half_spread"])
            quotes = np.asarray([clean + c * spread * rng.standard_normal(54) for c in parameters[:, -1]])
            np.testing.assert_array_equal(arrays["train_quotes"][128:], quotes)
            np.testing.assert_array_equal(arrays["train_observations"][128:], quotes / spread)
            for index, name in enumerate(model.names):
                np.testing.assert_array_equal(arrays["train_" + name][128:, 0], parameters[:, index])
            trainer._validate_parent_prefix(arrays, parent)
            self.assertEqual(states, {"before_append": before, "after_append": rng.bit_generator.state})
            self.assertEqual(parent["identity"]["after_validation_rng_state"], before)
            self.assertEqual(parent["whitening"].to_dict(), old_whitening)
            self.assertEqual(pricing.call_count, 7)
            self.assertEqual(progress, [7])
            for call, row in zip(pricing.call_args_list, parameters, strict=True):
                np.testing.assert_array_equal(call.args[0], row)

    def test_full_native_optimizer_budget_is_resolved_without_pricing(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent, _ = parent_fixture(Path(temporary))
            parent["identity"]["train_rows"] = 10000
            parent["identity"]["validation_rows"] = 100
            with patch.object(trainer, "_read_parent", return_value=parent), patch.object(
                    trainer, "source_bindings", return_value={"producer": "fixture"}), patch.object(
                    trainer.HestonSpreadPilot, "prices", side_effect=AssertionError("contract may not price")):
                contract = trainer.protocol_contract(parent["directory"])
            self.assertEqual((contract["train_rows"], contract["appended_rows"], contract["epochs"],
                              contract["batch_size"], contract["batches_per_epoch"], contract["tail_batch_rows"],
                              contract["example_presentations"], contract["expected_optimizer_updates"]),
                             (100000, 90000, 200, 256, 391, 160, 20000000, 78200))
            self.assertEqual(contract["native_optimizer_budget"], {"updates": 78200, "warmup_steps": 3910,
                "decay_steps": 74290, "initial_learning_rate": 5e-5, "warmup_target": 5e-4, "alpha": 0})
            optimizer = contract["resolved_defaults"]["optimizer"]
            self.assertEqual(optimizer["class_name"], "AdamW")
            self.assertEqual(optimizer["config"]["weight_decay"], .005)
            self.assertEqual(optimizer["config"]["clipnorm"], 1.5)
            self.assertEqual(contract["checkpoint_epochs"], [20, 50, 75, 83, 100, 150, 200])
            self.assertEqual(contract["max_wall_seconds"], 3600)
            self.assertFalse(contract["automatic_extension"])
            self.assertEqual(contract["retries"], 0)
            self.assertFalse(contract["fresh_noise_per_pass"])


if __name__ == "__main__":
    unittest.main()
