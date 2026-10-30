"""Path conditions, arbitrary-prefix masks, support and real BF2.1 serialization."""

from importlib import metadata
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from sbi4finance.path_inference import bank, inference

try:
    CORE = metadata.version("bayesflow") == "2.1.0"
except metadata.PackageNotFoundError:
    CORE = False


def small_task():
    return dict(bank.default_task("heston"), n_observations=29,
                prefix_lengths=[1, 7, 13, 21, 29], history_length_range=[1, 29])


class InputContractTests(unittest.TestCase):
    def setUp(self):
        self.task = small_task()
        self.rng = np.random.default_rng(83)
        self.values = self.rng.normal(size=(4, 29, 1)) * .01
        self.split = {"observations": self.values,
                      "parameters": inference.prior_samples(self.task, 4, 83)}

    def test_all_integer_lengths_are_accepted(self):
        for length in range(1, 30):
            data = inference.conditioning_data(self.values[:, :length], self.task)
            self.assertEqual(data["observations"].shape, (4, 29, 2))
            self.assertEqual(data["observations"].dtype, np.float32)
            self.assertEqual(data["time_context"].dtype, np.float32)
            self.assertEqual(data["summary_mask"].dtype, np.bool_)
            self.assertTrue(data["summary_mask"][:, :length].all())
            self.assertFalse(data["summary_mask"][:, length:].any())
            np.testing.assert_array_equal(data["observations"][:, length:], 0)

    def test_future_and_labels_do_not_enter_conditions(self):
        lengths = np.array([1, 7, 17, 29])
        original = inference.conditioning_data(self.values, self.task, lengths=lengths)
        changed = self.values.copy()
        for row, length in enumerate(lengths):
            changed[row, length:] = np.nan
        other = inference.conditioning_data(changed, self.task, lengths=lengths)
        for key in original:
            np.testing.assert_array_equal(original[key], other[key])
        self.assertEqual(set(original), {"observations", "time_context", "summary_mask"})

    def test_invalid_observed_values_and_lengths_fail(self):
        for bad in (np.zeros((4, 0, 1)), np.ones((4, 30, 1)), np.full((4, 29, 1), np.nan)):
            with self.assertRaises(ValueError):
                inference.conditioning_data(bad, self.task)
        for lengths in ([0, 1, 2, 3], [30, 1, 2, 3], [1., 2., 3., 4.], [1, 2]):
            with self.assertRaises(ValueError):
                inference.conditioning_data(self.values, self.task, lengths=lengths)

    def test_fixed_scaling_preserves_return_amplitude(self):
        a = inference.conditioning_data(self.values, self.task)
        b = inference.conditioning_data(self.values * 2, self.task)
        np.testing.assert_array_equal(b["observations"][..., 0], 2 * a["observations"][..., 0])
        np.testing.assert_array_equal(b["observations"][..., 1], a["observations"][..., 1])

    def test_prior_transform_round_trip_all_models(self):
        for model in ("heston", "rough-known", "rough-joint"):
            task = bank.default_task(model)
            physical = inference.prior_samples(task, 100, 901)
            restored = inference.inverse_parameters(inference.transform_parameters(physical, task), task)
            np.testing.assert_allclose(restored, physical, rtol=2e-14, atol=1e-15)
        invalid = self.split["parameters"].copy()
        invalid[0, 0] = self.task["prior_bounds"][0][0]
        with self.assertRaises(ValueError):
            inference.transform_parameters(invalid, self.task)

    def test_prefix_augmentation_is_seeded_and_covers_unseen_lengths(self):
        data = inference.training_data(self.split, self.task)
        a, b = (inference.random_prefixes(self.task, 900) for _ in range(2))
        for _ in range(40):
            first, second = a(data), b(data)
            for key in first:
                np.testing.assert_array_equal(first[key], second[key])
            self.assertTrue(np.all(np.sum(first["summary_mask"], axis=1) >= 1))
            np.testing.assert_array_equal(first["parameters"], data["parameters"])
        self.assertGreater(np.sum(a.exposure > 0), len(self.task["prefix_lengths"]))

    def test_metrics_preserve_variance_expansion(self):
        prior = self.rng.normal(size=(1000, 2))
        truth = self.rng.normal(size=(20, 2))
        samples = self.rng.normal(size=(20, 50, 2)) * 3
        result = inference.diagnostics(samples, truth, prior)
        self.assertGreater(min(result["mean_posterior_to_prior_variance"]), 1)
        self.assertEqual(result["cases"], 20)

    def test_mixture_policy_adds_short_history_exposure_without_losing_support(self):
        data = inference.training_data(self.split, self.task)
        data = {k: np.tile(v, (250,) + (1,) * (v.ndim - 1)) for k, v in data.items()}
        uniform = inference.random_prefixes(self.task, 991, policy="uniform")
        mixture = inference.random_prefixes(self.task, 991, policy="uniform_log_mixture")
        for _ in range(10):
            first, second = uniform(data), mixture(data)
            np.testing.assert_array_equal(first["parameters"], second["parameters"])
            self.assertTrue(np.all(second["summary_mask"].sum(axis=1) >= 1))
        self.assertTrue(np.all(mixture.exposure[1:] > 0))
        self.assertGreater(mixture.exposure[1:5].sum(), 1.5 * uniform.exposure[1:5].sum())
        with self.assertRaises(ValueError):
            inference.random_prefixes(self.task, 83, policy="unknown")


@unittest.skipUnless(CORE, "requires locked BayesFlow2.1 core runtime")
class ActualWorkflowTests(unittest.TestCase):
    def test_weights_only_warm_start_resets_optimizer_and_checks_bank_identity(self):
        os.environ.setdefault("KERAS_BACKEND", "jax")
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("path_train_warm_test", root / "scripts/train_path_posterior.py")
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)
        task = dict(bank.default_task("rough-joint"), n_observations=17,
                    prefix_lengths=[1, 4, 8, 13, 17], history_length_range=[1, 17])
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            bank.write_bank(task, work / "bank", train=8, validation=4, test=2, chunk=4, seed=83)
            settings = dict(bank=work / "bank", epochs=1, batch_size=4, seed=83, max_seconds=120,
                            validation_cases=4, hidden_dim=8, summary_dim=8, depth=2, bins=4, width=8,
                            prefix_policy="uniform", learning_rate=5e-4, init_checkpoint=None)
            parent = trainer.run_training(SimpleNamespace(**settings, out=work / "parent"))
            self.assertEqual(parent["optimizer_steps"], 2)
            # An original uniform checkpoint without the new policy field still
            # loads, so historical native bytes need not be rewritten.
            legacy = json.loads((work / "parent/training.json").read_text())
            legacy.pop("prefix_policy")
            (work / "parent/training.json").write_text(json.dumps(legacy))
            inference.load_checkpoint(work / "parent")
            settings.update(init_checkpoint=work / "parent", prefix_policy="uniform_log_mixture", learning_rate=1e-4,
                            epochs=2, checkpoint_selection="final")
            child = trainer.run_training(SimpleNamespace(**settings, out=work / "child"))
            self.assertEqual(child["optimizer_steps"], 4)
            self.assertEqual(child["initialization"]["initial_optimizer_steps"], 0)
            self.assertEqual(child["initialization"]["parent_model_sha256"], parent["model_sha256"])
            self.assertEqual(child["prefix_policy"], "uniform_log_mixture")
            self.assertEqual(child["checkpoint_selection"], "final")
            self.assertEqual(child["selected_epoch"], child["epochs_completed"])
            self.assertEqual(child["model_sha256"], child["final_sha256"])
            self.assertTrue(child["same_seed_reload_exact"])
            self.assertTrue({1, 2, 4, 8, 17}.issubset(child["validation_landmarks"]))
            inference.load_checkpoint(work / "child")
            for script in ("evaluate_path_posterior", "check_path_one_return"):
                spec = importlib.util.spec_from_file_location(script + "_warm_test", root / "scripts" / (script + ".py"))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                arguments = SimpleNamespace(bank=work / "bank", checkpoint=work / "child", out=work / script,
                                            split="validation", cases=3, draws=8, batch_size=2, seed=83,
                                            offset=1, lengths=[1, 2, 17], quadrature_nodes=64)
                checked = module.evaluate(arguments) if script == "evaluate_path_posterior" else module.run(arguments)
                self.assertEqual(checked["offset"], 1)
                if script == "evaluate_path_posterior":
                    self.assertEqual(set(checked["lengths"]), {"1", "2", "17"})
                    with np.load(arguments.out / "case-summary.npz") as summary:
                        self.assertEqual(summary["truth"].shape, (3, 6))
                        self.assertEqual(summary["posterior_mean_17"].shape, (3, 6))
            bank.write_bank(task, work / "other-bank", train=8, validation=4, test=2, chunk=4, seed=84)
            settings["bank"] = work / "other-bank"
            with self.assertRaisesRegex(ValueError, "same task and bank"):
                trainer.run_training(SimpleNamespace(**settings, out=work / "bad-child"))

    def test_masked_arbitrary_prefix_training_reload_and_future_invariance(self):
        os.environ.setdefault("KERAS_BACKEND", "jax")
        import bayesflow  # noqa: F401
        import keras
        task = small_task()
        rng = np.random.default_rng(83)
        split = {"observations": rng.normal(size=(8, 29, 1)) * .01,
                 "parameters": inference.prior_samples(task, 8, 83)}
        network = {"hidden_dim": 8, "summary_dim": 8, "depth": 2, "bins": 4, "widths": [8, 8]}
        keras.utils.set_random_seed(83)
        workflow = inference.build_workflow(task, network=network, optimizer=keras.optimizers.Adam(5e-4))
        data = inference.training_data(split, task)
        history = workflow.fit_offline(data=data, validation_data=data, epochs=1, batch_size=4,
                                        augmentations=inference.random_prefixes(task, 83),
                                        keep_optimizer=True, verbose=0).history
        self.assertTrue(np.isfinite(history["loss"]).all())
        lengths = np.array([1, 17])
        conditions = inference.conditioning_data(split["observations"][:2], task, lengths=lengths)
        before = np.asarray(workflow.summary_network(conditions["observations"].astype("float32"),
                                                      mask=conditions["summary_mask"], training=False))
        changed = conditions["observations"].copy()
        for row, length in enumerate(lengths):
            changed[row, length:] = 1000
        after = np.asarray(workflow.summary_network(changed.astype("float32"),
                                                     mask=conditions["summary_mask"], training=False))
        np.testing.assert_array_equal(before, after)
        # Verify BayesFlow itself forwards the semantic mask through sampling,
        # not merely that direct calls to the recurrent network accept a mask.
        original_draws = workflow.sample(num_samples=4, conditions=conditions, sample_shape=(), seed=113)
        changed_conditions = {**conditions, "observations": changed}
        changed_draws = workflow.sample(num_samples=4, conditions=changed_conditions, sample_shape=(), seed=113)
        np.testing.assert_array_equal(original_draws["parameters"], changed_draws["parameters"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.keras"
            workflow.estimator.save(path)
            restored = inference.build_workflow(task, network=network)
            restored.load_estimator(path)
            a = inference.sample(workflow, task, split["observations"][:2], 8, lengths=lengths, seed=19)
            b = inference.sample(restored, task, split["observations"][:2], 8, lengths=lengths, seed=19)
            self.assertEqual(a.shape, (2, 8, 4))
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
