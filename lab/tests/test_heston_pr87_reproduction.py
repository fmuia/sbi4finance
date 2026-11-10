"""Notebook-order generation, literal whitening, and raw-target boundaries."""

from importlib import metadata
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import reproduce_heston_pr87 as trainer
from scripts import train_heston_spread_scale as scale
from test_heston_spread_training import fixture


class NotebookWhiteningTests(unittest.TestCase):
    def test_loader_refuses_protocol_source_and_generated_payload_tampering_before_deserialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            design, _, _, _ = fixture(root)
            task, _ = scale.make_task(design)
            training = root / "training"
            checkpoint = training / "checkpoints" / "epoch-000002"
            checkpoint.mkdir(parents=True)
            notebook = root / "notebook.ipynb"
            notebook.write_text("fixture notebook")
            protocol = root / "protocol.json"
            contract = {"source": {"producer": "frozen"},
                        "notebook": {"path": str(notebook), "sha256": trainer.digest(notebook)}}
            trainer.write_json(protocol, contract)
            observations = np.random.default_rng(12).normal(size=(128, 54))
            trainer.write_npz(training / "data.npz", {"train_observations": observations})
            whitening = trainer.NotebookWhitening(observations).to_dict()
            for path in (training / "whitening.json", checkpoint / "whitening.json"):
                trainer.write_json(path, whitening)
            (checkpoint / "model.keras").write_bytes(b"must never deserialize")
            (checkpoint / "state.npz").write_bytes(b"must never deserialize")
            record = {"format": "heston_pr87_reproduction_checkpoint_v1", "task": task,
                "task_fingerprint": trainer.fingerprint(task), "parameter_names": task["target"]["parameter_names"],
                "bounds": task["target"]["prior_bounds"], "protocol": contract,
                "protocol_path": str(protocol), "protocol_sha256": trainer.digest(protocol),
                "data_payload": "data.npz", "data_payload_sha256": trainer.digest(training / "data.npz"),
                "model_sha256": trainer.digest(checkpoint / "model.keras"),
                "state_sha256": trainer.digest(checkpoint / "state.npz"),
                "whitening_sha256": trainer.digest(checkpoint / "whitening.json")}
            trainer.write_json(training / "run.json", record)
            trainer.write_json(checkpoint / "checkpoint.json", record)
            original_protocol = protocol.read_bytes()
            with patch.object(trainer, "build_workflow") as build, patch.object(
                    trainer, "source_bindings", return_value=contract["source"]):
                protocol.write_bytes(original_protocol + b" ")
                with self.assertRaises(ValueError):
                    trainer.load_checkpoint(checkpoint)
                protocol.write_bytes(original_protocol)
                with patch.object(trainer, "source_bindings", return_value={"producer": "changed"}):
                    with self.assertRaises(ValueError):
                        trainer.load_checkpoint(checkpoint)
                with (training / "data.npz").open("ab") as stream:
                    stream.write(b"changed generated payload")
                with self.assertRaises(ValueError):
                    trainer.load_checkpoint(checkpoint)
                build.assert_not_called()

    def test_native_equation_round_trip_and_corruption(self):
        observations = np.random.default_rng(39).normal(size=(128, 54))
        observations[:, 1] = observations[:, 0] + 1e-8 * observations[:, 1]
        mean = observations.mean(axis=0)
        covariance = np.cov(observations, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        matrix = (eigenvectors / np.sqrt(np.maximum(eigenvalues, eigenvalues[-1] * 1e-8))) @ eigenvectors.T
        fitted = trainer.NotebookWhitening(observations)
        np.testing.assert_array_equal(fitted.transform(observations), (observations - mean) @ matrix)
        saved = fitted.to_dict()
        restored = trainer.NotebookWhitening.from_dict(saved)
        heldout = np.random.default_rng(40).normal(size=(3, 54)) * 1e8
        np.testing.assert_array_equal(restored.transform(heldout), fitted.transform(heldout))
        self.assertFalse(saved["validation_used_for_fit"])
        saved["whitening_matrix"][0][0] += 1.
        with self.assertRaises(ValueError):
            trainer.NotebookWhitening.from_dict(saved)

    def test_raw_quote_wrapper_preserves_outside_support_draws(self):
        class Native:
            estimator = None
            def sample(self, **kwargs):
                self.kwargs = kwargs
                return {name: np.full((2, 3, 1), np.nan if getattr(self, "nonfinite", False) else -100. + i)
                        for i, name in enumerate(names)}

        names = ["initial_variance", "kappa", "theta", "vol_of_vol", "rho", "c"]
        task = {"records": {"half_spread": np.full(54, 2.).tolist()},
                "target": {"parameter_names": names}}
        observations = np.random.default_rng(44).normal(size=(128, 54))
        whitening = trainer.NotebookWhitening(observations)
        native = Native()
        wrapper = trainer.RawQuoteWorkflow(native, task, whitening)
        quotes = np.full((2, 54), -4.)
        draws = wrapper.sample(quotes, 3, seed=73)
        np.testing.assert_array_equal(draws[0, 0], np.arange(6) - 100.)
        np.testing.assert_array_equal(native.kwargs["conditions"]["observations"], whitening.transform(quotes / 2.))
        self.assertEqual(native.kwargs["seed"], 73)
        self.assertEqual(set(native.kwargs["conditions"]), {"observations"})
        native.nonfinite = True
        self.assertTrue(np.isnan(wrapper.sample(quotes, 3, seed=73)).all())


try:
    CORE = metadata.version("bayesflow") == "2.1.0"
except metadata.PackageNotFoundError:
    CORE = False


@unittest.skipUnless(CORE, "requires locked BF2.1 environment")
class NotebookSimulatorTests(unittest.TestCase):
    def test_exact_shared_rng_native_all_priors_then_all_noises(self):
        with tempfile.TemporaryDirectory() as temporary:
            design, _, _, _ = fixture(Path(temporary))
            task, model = scale.make_task(design)
            clean = np.arange(54, dtype=float) + 100.
            with patch.object(trainer.HestonSpreadPilot, "prices", return_value=clean) as pricing:
                data, extra, states = trainer.generate_data(task, 7, 3)
            rng = np.random.default_rng(trainer.SEED)
            halfspread = np.asarray(task["records"]["half_spread"])
            for split, rows in (("train", 7), ("validation", 3)):
                parameters = np.asarray([rng.uniform(model.bounds[:, 0], model.bounds[:, 1]) for _ in range(rows)])
                expected_quotes = np.asarray([clean + c * halfspread * rng.standard_normal(54) for c in parameters[:, -1]])
                np.testing.assert_array_equal(extra[split + "_quotes"], expected_quotes)
                np.testing.assert_array_equal(data[split]["observations"], expected_quotes / halfspread)
                for index, name in enumerate(model.names):
                    np.testing.assert_array_equal(data[split][name][:, 0], parameters[:, index])
                self.assertEqual(states["after_" + split], rng.bit_generator.state)
            self.assertEqual(pricing.call_count, 10)
            # Fresh physical labels were passed into the existing checked pricer.
            for call, row in zip(pricing.call_args_list[:7], np.column_stack(
                    [data["train"][name][:, 0] for name in model.names]), strict=True):
                np.testing.assert_array_equal(call.args[0], row)

    def test_adapter_has_exactly_raw_concatenation_and_rename(self):
        with tempfile.TemporaryDirectory() as temporary:
            design, _, _, _ = fixture(Path(temporary))
            task, _ = scale.make_task(design)
            workflow = trainer.build_workflow(task)
            raw = {name: np.array([[i + .25]], dtype=np.float64)
                   for i, name in enumerate(task["target"]["parameter_names"])}
            raw["observations"] = np.ones((1, 54), dtype=np.float64)
            adapted = workflow.adapter(raw)
            np.testing.assert_array_equal(adapted["inference_variables"], np.arange(6)[None, :] + .25)
            # BF.Adapter() supplies its own native cast; no cast is added here.
            self.assertEqual(adapted["inference_variables"].dtype, np.float32)
            self.assertEqual(set(adapted), {"inference_variables", "inference_conditions"})
            optimizer = workflow.build_optimizer(200, 40, strategy="offline")
            config = optimizer.get_config()
            self.assertEqual(type(optimizer).__name__, "AdamW")
            self.assertEqual(config["weight_decay"], .005)
            self.assertEqual(config["clipnorm"], 1.5)
            lr = config["learning_rate"]["config"]
            self.assertEqual((lr["initial_learning_rate"], lr["warmup_target"], lr["warmup_steps"], lr["decay_steps"]),
                             (5e-5, 5e-4, 400, 7600))


if __name__ == "__main__":
    unittest.main()
