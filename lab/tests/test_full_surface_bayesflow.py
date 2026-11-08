"""Actual BF2.1 API checks; run in the isolated core environment.

Tiny generated labels test software behavior, not posterior accuracy.
"""

from importlib import metadata
import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from sbi4finance.calibration import full_surface, tasks

try:
    CORE_ENVIRONMENT = metadata.version("bayesflow") == "2.1.0"
except metadata.PackageNotFoundError:
    CORE_ENVIRONMENT = False


@unittest.skipUnless(CORE_ENVIRONMENT, "requires the locked BayesFlow2.1 core environment")
class ActualBayesFlowTests(unittest.TestCase):
    def test_real_interrupted_chunk_recovery_and_no_test_read(self):
        os.environ.setdefault("KERAS_BACKEND", "jax")
        script = Path(__file__).resolve().parents[1] / "scripts/train_full_surface.py"
        spec = importlib.util.spec_from_file_location("core_train_for_test", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from sbi4finance.calibration import bank

        class InterruptedAfterCommit(Exception):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {"task": tasks.heston_task(), "root_seed": 1283,
                      "splits": {"train": 17, "validation": 8, "test": 2}}
            bank.write_bank(root / "bank", config)
            args = argparse.Namespace(bank=root / "bank", out=root / "run", family="spline", encoder="vector",
                         widths="16,16", summary_dim=32, flow_steps=4, mean_steps=4, epochs=2,
                         batch_size=8, chunk_epochs=1, seed=3)
            original_commit = module.recovery.commit_resume_state
            def interrupt(*arguments, **kwargs):
                original_commit(*arguments, **kwargs)
                raise InterruptedAfterCommit
            with mock.patch.object(module.recovery, "commit_resume_state", side_effect=interrupt):
                with self.assertRaises(InterruptedAfterCommit):
                    module.run_training(args)
            with mock.patch.object(module.bank, "load_bank", wraps=module.bank.load_bank) as load:
                report = module.run_training(args)
                self.assertEqual(load.call_args.kwargs["splits"], ("train", "validation"))
            self.assertEqual(report["resumed_at_epochs"], [1])
            self.assertEqual(len(report["history"]["loss"]), 2)
            self.assertFalse(report["reserved_test_evaluated"])
            self.assertTrue(report["same_seed_reload_exact"])
            work = report["performance"]["work"]
            self.assertEqual(work["optimizer_updates"], 6)
            self.assertEqual(work["optimizer_updates_this_invocation"], 3)
            self.assertEqual(work["training_presentations"], 34)
            self.assertEqual(work["training_presentations_this_invocation"], 17)
            self.assertEqual(work["unique_training_rows"], 17)
            self.assertEqual(work["last_batch_size"], 1)
            self.assertEqual(work["full_batches"], 4)
            self.assertEqual(work["partial_batches"], 2)
            self.assertEqual(work["online_simulated_rows"], 0)
            self.assertEqual(len(work["training_example_ids_sha256"]), 64)
            phases = report["performance"]["phases"]
            self.assertEqual(phases["validation"]["calls"], 1)
            self.assertEqual(phases["first_training_batch_including_compile"]["calls"], 1)
            self.assertIn("resume_reload", phases)
            full_surface.load_checkpoint(args.out)
            training_path = args.out / "training.json"
            original = training_path.read_text()
            for change_bound_contract in (False, True):
                altered = json.loads(original)
                altered["network"]["family"] = "affine"
                if change_bound_contract:
                    altered["run_contract"]["network"]["family"] = "affine"
                training_path.write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError, "architecture"):
                    full_surface.load_checkpoint(args.out)
            training_path.write_text(original)
            changed = argparse.Namespace(**{**vars(args), "family": "affine"})
            with self.assertRaises(ValueError):
                module.run_training(changed)

    def test_selected_families_and_encoders_train_reload_and_sample(self):
        os.environ.setdefault("KERAS_BACKEND", "jax")
        import keras
        task = tasks.heston_task()
        bounds = np.asarray(task["prior_bounds"])
        rng = np.random.default_rng(313)
        split = {"observations": rng.normal(size=(16, 63, 1)) * .001,
                 "parameters": rng.uniform(bounds[:, 0], bounds[:, 1], size=(16, len(bounds)))}
        configurations = [(family, "vector") for family in full_surface.FAMILIES]
        configurations += [("spline", encoder) for encoder in ("deepset", "set_transformer")]
        for family, encoder in configurations:
            with self.subTest(family=family, encoder=encoder), tempfile.TemporaryDirectory() as temp:
                keras.utils.set_random_seed(313)
                data = full_surface.training_data(split, task, encoder)
                settings = dict(family=family, encoder=encoder, widths=(16, 16), flow_steps=4)
                workflow = full_surface.build_workflow(task, **settings)
                history = workflow.fit_offline(data=data, validation_data=data, epochs=1, batch_size=8,
                                                verbose=0).history
                self.assertTrue(np.isfinite(history["loss"]).all())
                self.assertTrue(np.isfinite(history["val_loss"]).all())
                before = full_surface.sample(workflow, task, split["observations"][:2], 4, encoder=encoder, seed=3)
                self.assertEqual(before.shape, (2, 4, 4))
                self.assertTrue(np.all(before >= bounds[:, 0]) and np.all(before <= bounds[:, 1]))
                path = Path(temp) / "model.keras"
                workflow.estimator.save(path)
                restored = full_surface.build_workflow(task, **settings)
                restored.load_estimator(path)
                after = full_surface.sample(restored, task, split["observations"][:2], 4, encoder=encoder, seed=3)
                np.testing.assert_array_equal(before, after)
                if encoder != "vector":
                    # Includes the trained adapter and its standardization, not just the summary layer.
                    conditions = full_surface.conditioning_data(split["observations"][:2], task, encoder)
                    conditions["observations"] = conditions["observations"][:, ::-1, :]
                    permuted = workflow.sample(num_samples=4, conditions=conditions, sample_shape=(), seed=3)
                    permuted = np.concatenate([permuted[name] for name in task["parameter_names"]], axis=-1)
                    np.testing.assert_allclose(before, permuted, rtol=1e-5, atol=1e-6)
                if family == "mean_flow":
                    for steps in (1, 2, 8):
                        self.assertEqual(full_surface.sample(restored, task, split["observations"][:2], 4,
                                           seed=3, mean_steps=steps).shape, (2, 4, 4))


if __name__ == "__main__":
    unittest.main()
