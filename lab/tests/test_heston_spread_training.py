"""Fresh noise, immutable-bank isolation, native updates and first failure capture."""

from importlib import metadata
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import train_heston_spread_baseline as trainer


def fixture(root, train_rows=128, validation_rows=32):
    design = root / "design.json"
    records = {"forward": [100.] * 54, "strike": np.linspace(80., 120., 54).tolist(),
               "discount": [.99] * 54, "maturity": [.5] * 54,
               "option_type": ["put"] * 27 + ["call"] * 27,
               "mid": [2.] * 54, "half_spread": np.linspace(.02, .1, 54).tolist()}
    design.write_text(json.dumps({"records": records}))
    task, pilot = trainer.make_task(design)
    original = {"parameter_names": [*pilot.names, "tau"], "prior_bounds": [*pilot.bounds.tolist(), [0., .02]],
                "design_sha256": trainer.digest(design), "pricing_controls": {
                    "measurement_spread_multiplier": .5, "numerical_fraction": pilot.numerical_fraction,
                    "quadrature_ladder": [list(rung) for rung in pilot.quadrature_ladder]}}
    identity = {"task": original, "task_fingerprint": trainer.fingerprint(original),
                "train_rows": train_rows, "validation_rows": validation_rows, "test_rows": 2}
    bank = root / "bank"
    bank.mkdir()
    rng = np.random.default_rng(42)
    payload = {}
    for split, count in (("train", train_rows), ("validation", validation_rows)):
        parameters = rng.uniform(pilot.bounds[:, 0], pilot.bounds[:, 1], (count, 5))
        # Software fixture, not a numerical Heston bank: no pricing is performed.
        clean = np.linspace(.01, 3., 54)[None, :] + parameters[:, :1]
        payload.update({split + "_parameters": parameters, split + "_clean": clean,
                        split + "_row_id": np.array([f"{split}:{i:08d}" for i in range(count)])})
    payload.update(test_parameters=np.full((2, 5), np.nan), test_clean=np.full((2, 54), np.nan))
    np.savez_compressed(bank / "bank.npz", **payload)
    record = {"format": "heston_pilot_campaign_clean_bank_v1", "identity": identity,
              "payload": "bank.npz", "payload_sha256": trainer.digest(bank / "bank.npz")}
    (bank / "bank.json").write_text(json.dumps(record))
    return design, bank, task, payload


class SpreadTrainingDataTests(unittest.TestCase):
    def test_loader_reads_only_named_training_development_members_and_selects_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, bank, task, payload = fixture(Path(temporary))
            real_load, accesses = np.load, []

            class Archive:
                def __init__(self, path, **kwargs):
                    self.archive = real_load(path, **kwargs)
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    self.archive.close()
                @property
                def files(self):
                    raise AssertionError("do not enumerate reserved bank members")
                def __getitem__(self, key):
                    accesses.append(key)
                    if key.startswith("test_"):
                        raise AssertionError("reserved pool was read")
                    return self.archive[key]

            with patch.object(trainer.np, "load", side_effect=Archive):
                data, record = trainer.load_clean_bank(bank, task, 16, smoke_test_only=True)
            self.assertEqual(set(accesses), {f"{split}_{key}" for split in ("train", "validation")
                                             for key in ("parameters", "clean", "row_id")})
            np.testing.assert_array_equal(data["train"]["parameters"], payload["train_parameters"][:16])
            self.assertFalse(record["reserved_arrays_opened"])
            self.assertEqual(record["simulation_calls"], 0)
            with self.assertRaisesRegex(ValueError, "production requires"):
                trainer.load_clean_bank(bank, task, 16)
            changed = json.loads((bank / "bank.json").read_text())
            changed["identity"]["task"]["prior_bounds"][0][1] = .2
            (bank / "bank.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "differs"):
                trainer.load_clean_bank(bank, task, 16, smoke_test_only=True)

    def test_noise_is_gaussian_fresh_probe_invariant_and_not_a_target(self):
        clean = np.zeros((20000, 3))
        h = np.array([.02, .05, .1])
        first = trainer.noisy_quotes(clean, h, 5, 9011)
        np.testing.assert_array_equal(first, trainer.noisy_quotes(clean, h, 5, 9011))
        self.assertFalse(np.array_equal(first, trainer.noisy_quotes(clean, h, 6, 9011)))
        self.assertFalse(np.array_equal(first, trainer.noisy_quotes(clean, h, 5, 9011, namespace=1)))
        np.testing.assert_allclose((first / h).var(axis=0), np.ones(3), atol=.03)
        np.testing.assert_allclose((first / h).mean(axis=0), np.zeros(3), atol=.02)
        self.assertTrue(np.any(first < 0))
        with tempfile.TemporaryDirectory() as temporary:
            _, _, task, payload = fixture(Path(temporary))
            quotes = -np.ones((2, 54))
            data = trainer.training_data(task, quotes, payload["train_parameters"][:2])
            self.assertEqual(set(data), {"observations", *task["target"]["parameter_names"]})
            self.assertNotIn("c", data)
            self.assertNotIn("tau", data)
            np.testing.assert_array_equal(data["observations"], quotes / np.asarray(task["records"]["half_spread"]))
            plan = trainer.BatchPlan(128, 16, 5, 9011)
            np.testing.assert_array_equal(plan.indices(0), trainer.BatchPlan(128, 16, 5, 9011).indices(0))


try:
    CORE = metadata.version("bayesflow") == "2.1.0"
except metadata.PackageNotFoundError:
    CORE = False


@unittest.skipUnless(CORE, "requires locked BF2.1 JAX environment")
class SpreadTrainingWorkflowTests(unittest.TestCase):
    def test_diagnostic_step_matches_native_bayesflow_update_without_advancing_probe_state(self):
        import jax
        import keras
        with tempfile.TemporaryDirectory() as temporary:
            _, bank, task, _ = fixture(Path(temporary))
            split, _ = trainer.load_clean_bank(bank, task, 16, smoke_test_only=True)
            keras.utils.set_random_seed(9011)
            workflow = trainer.build_workflow(task)
            plan = trainer.BatchPlan(16, 16, 1, 9011)
            batch = trainer.adapted_batch(workflow, task, split["train"], plan, 0, 9011)[-1]
            workflow.estimator.build_from_data(batch)
            workflow.estimator.optimizer.build(workflow.estimator.trainable_variables)
            state = trainer.current_state(workflow.estimator)
            before = trainer.state_summary(workflow.estimator, state)
            diagnostic = trainer.compiled_step(workflow.estimator)
            metrics, actual, gradients, flags = diagnostic(state, batch)
            native_metrics, expected = jax.jit(workflow.estimator.stateless_train_step)(state, batch)
            for current, native in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
                np.testing.assert_array_equal(np.asarray(current), np.asarray(native))
            self.assertTrue(all(bool(value) for value in flags.values()))
            self.assertTrue(all(np.isfinite(value).all() for value in gradients))
            self.assertEqual(float(metrics["loss"]), float(native_metrics["loss"]))
            self.assertEqual(before, trainer.state_summary(workflow.estimator, trainer.current_state(workflow.estimator)))

    def test_actual_tiny_training_checkpoint_reload_and_first_gradient_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            design, bank, task, _ = fixture(root)
            result = trainer.run_training(design, bank, root / "good", train_rows=128, updates=5,
                probe_rows=16, log_every=5, smoke_test_only=True, checkpoint_updates=(2, 5))
            self.assertEqual(result["completed_updates"], 5)
            self.assertEqual(result["row_presentations"], 5*128)
            self.assertTrue(result["same_seed_reload_exact"])
            loaded, record = trainer.load_checkpoint(root / "good/checkpoints/update-000005")
            self.assertEqual(int(loaded.estimator.optimizer.iterations.numpy()), 5)
            self.assertFalse(record["resume_supported"])
            self.assertEqual(record["names"], task["target"]["parameter_names"])
            real_step = trainer.compiled_step

            def injected_gradient_failure(model):
                native = real_step(model)
                calls = 0
                def step(state, batch):
                    nonlocal calls
                    metrics, candidate, gradients, flags = native(state, batch)
                    calls += 1
                    if calls == 2:
                        gradients = list(gradients)
                        gradients[0] = np.full(np.shape(gradients[0]), np.nan, dtype=np.asarray(gradients[0]).dtype)
                        flags = {**flags, "gradients": False}
                    return metrics, candidate, gradients, flags
                return step

            with patch.object(trainer, "compiled_step", side_effect=injected_gradient_failure):
                with self.assertRaisesRegex(FloatingPointError, "gradients"):
                    trainer.run_training(design, bank, root / "bad", train_rows=128, updates=5,
                        probe_rows=16, log_every=5, smoke_test_only=True, checkpoint_updates=(1, 5))
            failure = json.loads((root / "bad/failure/failure.json").read_text())
            self.assertEqual(failure["absolute_batch_step"], 1)
            self.assertEqual(failure["attempted_optimizer_update"], 2)
            self.assertEqual(failure["completed_finite_updates"], 1)
            self.assertEqual(failure["failed_fields"], ["gradients"])
            self.assertTrue(failure["gradients_available"])
            self.assertTrue((root / "bad/checkpoints/update-000001/model.keras").is_file())
            self.assertFalse((root / "bad/checkpoints/update-000005").exists())
            with np.load(root / "bad/failure/offending-batch-and-gradients.npz") as values:
                self.assertTrue(np.isnan(values["gradient_0000"]).all())
                self.assertTrue(np.isfinite(values["adapted_inference_conditions"]).all())
            progress = json.loads((root / "bad/progress.json").read_text())
            self.assertEqual(progress["status"], "failed")
            self.assertEqual(progress["completed_updates"], 1)
            real_batch = trainer.adapted_batch

            def injected_input_failure(*args, **kwargs):
                indices, quotes, raw, adapted = real_batch(*args, **kwargs)
                if args[4] == 1:
                    adapted["inference_conditions"] = np.array(adapted["inference_conditions"], copy=True)
                    adapted["inference_conditions"][0, 3] = np.nan
                return indices, quotes, raw, adapted

            with patch.object(trainer, "adapted_batch", side_effect=injected_input_failure):
                with self.assertRaisesRegex(FloatingPointError, "adapted input"):
                    trainer.run_training(design, bank, root / "bad-input", train_rows=128, updates=5,
                        probe_rows=16, log_every=5, smoke_test_only=True, checkpoint_updates=(1, 5))
            failure = json.loads((root / "bad-input/failure/failure.json").read_text())
            self.assertEqual(failure["absolute_batch_step"], 1)
            self.assertEqual(failure["phase"], "before_update")
            self.assertEqual(failure["completed_finite_updates"], 1)
            self.assertFalse(failure["gradients_available"])
            self.assertEqual(failure["failed_fields"], ["adapted_input:inference_conditions"])
            with np.load(root / "bad-input/failure/offending-batch-and-gradients.npz") as values:
                self.assertTrue(np.isnan(values["adapted_inference_conditions"][0, 3]))
                self.assertTrue(np.isfinite(values["quotes"]).all())
            def unexpected_adapter_exception(*args, **kwargs):
                result = real_batch(*args, **kwargs)
                if args[4] == 1:
                    raise RuntimeError("injected unexpected adapter exception before returning")
                return result

            with patch.object(trainer, "adapted_batch", side_effect=unexpected_adapter_exception):
                with self.assertRaisesRegex(RuntimeError, "unexpected adapter exception"):
                    trainer.run_training(design, bank, root / "adapter-error", train_rows=128, updates=5,
                        probe_rows=16, log_every=5, smoke_test_only=True, checkpoint_updates=(1, 5))
            failure = json.loads((root / "adapter-error/failure/failure.json").read_text())
            self.assertEqual(failure["absolute_batch_step"], 1)
            self.assertEqual(failure["completed_finite_updates"], 1)
            self.assertEqual(failure["phase"], "batch_adaptation")
            self.assertFalse(failure["batch_tensors_available"])
            self.assertEqual(failure["adapted_batch"], {})
            self.assertFalse(failure["gradients_available"])
            self.assertIn("unavailable", failure["batch_capture_scope"])
            with np.load(root / "adapter-error/failure/offending-batch-and-gradients.npz") as values:
                self.assertEqual(set(values.files), {"indices", "quotes"})
                self.assertEqual(values["indices"].size, 0)
                self.assertEqual(values["quotes"].size, 0)
            self.assertTrue((root / "adapter-error/checkpoints/update-000001/model.keras").is_file())


if __name__ == "__main__":
    unittest.main()
