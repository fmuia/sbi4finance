"""Known curves are conditions for the original rough task, never joint-task labels.

The actual BF2.1 check uses tiny dummy labels and no financial pricing calls.
It verifies training/serialization/software behavior, not inference quality.
"""

import argparse
from importlib import metadata
import importlib.util
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from sbi4finance.calibration import full_surface, inference, tasks, validation
from sbi4finance.gbm_bank import _json

ROOT = Path(__file__).resolve().parents[1]
try:
    CORE_ENVIRONMENT = metadata.version("bayesflow") == "2.1.0"
except metadata.PackageNotFoundError:
    CORE_ENVIRONMENT = False


def script_module(name):
    spec = importlib.util.spec_from_file_location("known_curve_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dummy_split(task, count, seed=83):
    rng = np.random.default_rng(seed)
    bounds = np.asarray(task["prior_bounds"])
    quotes = rng.normal(size=(count, 63, 1)) * .001
    return {"observations": quotes, "reference_prices": quotes[..., 0],
            "parameters": rng.uniform(bounds[:, 0], bounds[:, 1], (count, len(bounds))),
            "context_values": rng.uniform(.01, .16, (count, 7)),
            "example_id": np.array([f"validation:{i}" for i in range(count)])}


class KnownCurveContractTests(unittest.TestCase):
    def setUp(self):
        self.task = tasks.rough_task()
        self.split = dummy_split(self.task, 3)

    def test_conditions_match_original_known_curve_coordinates(self):
        self.assertIs(full_surface.validate_task(self.task), self.task)
        old = inference.conditioning_data(self.split["observations"], self.split["context_values"], self.task)
        new = full_surface.training_data(self.split, self.task)
        np.testing.assert_array_equal(new["observations"], old["observations"])
        np.testing.assert_array_equal(new["curve_context"], old["context"])
        self.assertEqual(set(new), {"observations", "curve_context", *self.task["parameter_names"]})
        changed = {**self.split, "parameters": self.split["parameters"] * 1.001,
                   "context_parameters": np.full((3, 2), np.nan)}
        other = full_surface.training_data(changed, self.task)
        for name in ("observations", "curve_context"):
            np.testing.assert_array_equal(new[name], other[name])

    def test_known_curve_is_required_positive_finite_real_and_batch_aligned(self):
        masked = np.ma.array(self.split["context_values"], mask=False)
        masked.mask[0, 0] = True
        invalid = (None, np.ones((2, 7)), np.ones((3, 8)), np.ones((3, 7, 1)),
                   np.zeros((3, 7)), -np.ones((3, 7)), np.full((3, 7), np.nan),
                   np.full((3, 7), np.inf), masked, np.ones((3, 7), dtype=complex))
        for levels in invalid:
            with self.subTest(shape=np.shape(levels)), self.assertRaisesRegex(ValueError, "known curve"):
                full_surface.conditioning_data(self.split["observations"], self.task, context_values=levels)
        with self.assertRaisesRegex(ValueError, "known curve"):
            full_surface.training_data({key: value for key, value in self.split.items() if key != "context_values"},
                                       self.task)

    def test_known_curve_set_inputs_keep_global_curve_separate_and_noise_off_curve(self):
        for encoder in ("deepset", "set_transformer"):
            with self.subTest(encoder=encoder):
                data = full_surface.training_data(self.split, self.task, encoder)
                self.assertEqual(data["observations"].shape, (3, 63, 5))
                self.assertEqual(data["curve_context"].shape, (3, 7))
                np.testing.assert_array_equal(data["curve_context"], np.log(self.split["context_values"]))
                noisy = full_surface.fresh_noise(93, encoder)(data)
                np.testing.assert_array_equal(noisy["curve_context"], data["curve_context"])
                np.testing.assert_array_equal(noisy["observations"][..., 1:], data["observations"][..., 1:])
                self.assertFalse(np.array_equal(noisy["observations"][..., 0], data["observations"][..., 0]))
                self.assertEqual(set(data), {"observations", "curve_context", *self.task["parameter_names"]})

    def test_fresh_noise_changes_quotes_only_and_keeps_curves(self):
        data = full_surface.training_data(self.split, self.task, clean=True)
        augment = full_surface.fresh_noise(93)
        first, second = augment(data), augment(data)
        self.assertFalse(np.array_equal(first["observations"], second["observations"]))
        self.assertFalse(np.array_equal(data["observations"], first["observations"]))
        for name in ("curve_context", *self.task["parameter_names"]):
            np.testing.assert_array_equal(first[name], data[name])
            np.testing.assert_array_equal(second[name], data[name])

    def test_sample_passes_known_curve_and_joint_and_heston_ignore_auxiliaries(self):
        for task in (self.task, tasks.joint_rough_task(), tasks.heston_task()):
            with self.subTest(task=task["task_id"]):
                bounds = np.asarray(task["prior_bounds"])
                sampled = {name: np.full((3, 4, 1), bounds[i].mean())
                           for i, name in enumerate(task["parameter_names"])}
                workflow = SimpleNamespace(sample=mock.Mock(return_value=sampled))
                curves = self.split["context_values"] if full_surface.uses_known_curve(task) else np.full((3, 7), np.nan)
                drawn = full_surface.sample(workflow, task, self.split["observations"], 4,
                                            context_values=curves, seed=93)
                self.assertEqual(drawn.shape, (3, 4, len(bounds)))
                conditions = workflow.sample.call_args.kwargs["conditions"]
                if full_surface.uses_known_curve(task):
                    np.testing.assert_array_equal(conditions["curve_context"], np.log(curves))
                else:
                    self.assertEqual(set(conditions), {"observations"})

    def test_evaluation_threads_curve_batches_and_reports_rho_eta_family(self):
        module = script_module("evaluate_full_surface")
        for task in (self.task, tasks.joint_rough_task()):
            with self.subTest(task=task["task_id"]), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                split = dummy_split(task, 5)
                bounds = np.asarray(task["prior_bounds"])
                rng = np.random.default_rng(92)
                samples = rng.uniform(bounds[:, 0], bounds[:, 1], (5, 8, len(bounds)))
                fingerprint = hashlib.sha256(_json(task).encode()).hexdigest()
                manifest = {"configuration": {"task": task}, "task_fingerprint": fingerprint}
                bank = root / "bank"
                bank.mkdir()
                (bank / "manifest.json").write_text(json.dumps(manifest))
                record = {"network": {"encoder": "vector"}, "task_fingerprint": fingerprint,
                          "checkpoint_sha256": "dummy-checkpoint"}
                args = argparse.Namespace(bank=bank, checkpoint=root / "checkpoint", split="validation",
                                          first=1, surfaces=3, draws=8, batch_size=2, seed=91, out=root / "out")

                def sample(workflow, actual_task, quotes, count, **kwargs):
                    start = kwargs["seed"] - args.seed
                    stop = start + len(quotes)
                    self.assertEqual(actual_task, task)
                    self.assertEqual(count, 8)
                    np.testing.assert_array_equal(kwargs["context_values"], split["context_values"][start:stop])
                    return samples[start:stop]

                with (mock.patch.object(module.full_surface, "load_checkpoint", return_value=(task, object(), record)),
                      mock.patch.object(module.bank, "load_bank", return_value=(manifest, {"validation": split})) as load,
                      mock.patch.object(module.full_surface, "sample", side_effect=sample)):
                    report = module.evaluate(args)
                self.assertEqual(load.call_args.kwargs["splits"], ("validation",))
                self.assertEqual(report["targets"], task["parameter_names"])
                self.assertEqual(report["performance"]["work"]["posterior_draws"], 24)
                self.assertEqual(report["performance"]["work"]["sample_calls"], 2)
                self.assertEqual(len(report["sbc"]["rank_ks"]), len(bounds))
                diagnostics = report["diagnostics_with_derived_targets"]
                self.assertEqual(diagnostics["targets"], [*task["parameter_names"], "rho_eta"])
                eta, rho = (task["parameter_names"].index(name) for name in ("eta", "rho"))
                draws, truth = samples[1:4], split["parameters"][1:4]
                augmented_draws = np.concatenate([draws, (draws[..., eta] * draws[..., rho])[..., None]], axis=-1)
                augmented_truth = np.column_stack([truth, truth[:, eta] * truth[:, rho]])
                expected = validation.sbc(augmented_draws, augmented_truth, seed=args.seed)
                self.assertEqual(diagnostics["sbc"], {k: v for k, v in expected.items() if k != "fractional_ranks"})
                self.assertEqual(len(diagnostics["coverage"]["0.95"]["coverage"]), len(bounds) + 1)
                self.assertFalse(report["reserved_test_evaluated"])
                with np.load(args.out / "draws.npz") as saved:
                    np.testing.assert_array_equal(saved["draws"], draws)
                    np.testing.assert_array_equal(saved["truth"], truth)


@unittest.skipUnless(CORE_ENVIRONMENT, "requires the locked BayesFlow2.1 core environment")
class ActualKnownCurveTests(unittest.TestCase):
    def test_set_encoders_train_reload_and_preserve_global_curve_under_quote_permutation(self):
        os.environ.setdefault("KERAS_BACKEND", "jax")
        import keras
        task = tasks.rough_task(); split = dummy_split(task, 8, seed=83)
        for encoder in ("deepset", "set_transformer"):
            with self.subTest(encoder=encoder), tempfile.TemporaryDirectory() as temporary:
                keras.utils.set_random_seed(83)
                settings = dict(family="affine", encoder=encoder, widths=(8, 8), summary_dim=8)
                workflow = full_surface.build_workflow(task, **settings)
                data = full_surface.training_data(split, task, encoder)
                history = workflow.fit_offline(data=data, validation_data=data, epochs=1, batch_size=8, verbose=0).history
                self.assertTrue(np.isfinite(history["loss"]).all())
                before = full_surface.sample(workflow, task, split["observations"][:2], 4, encoder=encoder,
                                              context_values=split["context_values"][:2], seed=4)
                self.assertEqual(before.shape, (2, 4, 3))
                path = Path(temporary) / "model.keras"; workflow.estimator.save(path)
                restored = full_surface.build_workflow(task, **settings); restored.load_estimator(path)
                after = full_surface.sample(restored, task, split["observations"][:2], 4, encoder=encoder,
                                             context_values=split["context_values"][:2], seed=4)
                np.testing.assert_array_equal(before, after)
                conditions = full_surface.conditioning_data(split["observations"][:2], task, encoder,
                                                             context_values=split["context_values"][:2])
                conditions["observations"] = conditions["observations"][:, ::-1]
                sampled = restored.sample(num_samples=4, conditions=conditions, sample_shape=(), seed=4)
                permuted = np.concatenate([sampled[name] for name in task["parameter_names"]], axis=-1)
                np.testing.assert_allclose(before, permuted, rtol=1e-5, atol=1e-6)

    def test_tiny_training_run_reload_and_native_sampling_with_three_targets(self):
        os.environ.setdefault("KERAS_BACKEND", "jax")
        module = script_module("train_full_surface")
        task = tasks.rough_task()
        fingerprint = hashlib.sha256(_json(task).encode()).hexdigest()
        manifest = {"configuration": {"task": task}, "task_fingerprint": fingerprint}
        train, val = dummy_split(task, 16, seed=10), dummy_split(task, 8, seed=11)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank = root / "bank"
            bank.mkdir()
            (bank / "manifest.json").write_text(json.dumps(manifest))
            args = argparse.Namespace(bank=bank, out=root / "run", family="flow_matching", encoder="vector",
                                      widths="8,8", summary_dim=32, flow_steps=None, mean_steps=4,
                                      epochs=1, batch_size=8, chunk_epochs=1, seed=7)
            with mock.patch.object(module.bank, "load_bank", return_value=(manifest, {"train": train, "validation": val})) as load:
                report = module.run_training(args)
            self.assertEqual(load.call_args.kwargs["splits"], ("train", "validation"))
            self.assertTrue(report["same_seed_reload_exact"])
            self.assertFalse(report["reserved_test_evaluated"])
            self.assertIn("supplied known curve levels", report["conditioning"])
            _, workflow, saved = full_surface.load_checkpoint(args.out)
            self.assertEqual(saved["flow_matching_integrate_kwargs"], {"method": "tsit5", "steps": "adaptive"})
            draws = full_surface.sample(workflow, task, val["observations"][:2], 4,
                                        context_values=val["context_values"][:2], seed=8)
            self.assertEqual(draws.shape, (2, 4, 3))
            self.assertTrue(np.isfinite(draws).all())


if __name__ == "__main__":
    unittest.main()
