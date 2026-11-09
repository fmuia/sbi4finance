"""Finite update schedules must stop exactly and preserve the endpoint on reload."""
from argparse import Namespace
from importlib import metadata
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, PropertyMock

import numpy as np

from scripts import train_full_surface as trainer
from scripts.train_full_surface_budget import BatchPlan, make_dataset, update_guard
from sbi4finance.calibration import bank, full_surface, tasks

try:
    CORE = metadata.version("bayesflow") == "2.1.0"
except metadata.PackageNotFoundError:
    CORE = False


class BatchPlanTests(unittest.TestCase):
    def test_partial_pass_has_unique_rows_and_probe_invariant_noise(self):
        plan = BatchPlan(10, 4, 5, 7)
        np.testing.assert_array_equal(np.sort(np.concatenate([plan.indices(i) for i in range(3)])), np.arange(10))
        self.assertEqual([len(plan.indices(i)) for i in range(5)], [4, 4, 2, 4, 4])
        np.testing.assert_array_equal(plan.indices(0), BatchPlan(10, 4, 5, 7).indices(0))
        np.testing.assert_array_equal(plan.noise(1, (4, 63)), plan.noise(1, (4, 63)))
        self.assertFalse(np.array_equal(plan.indices(0), plan.indices(3)))
        with self.assertRaises(IndexError): plan.indices(5)
        for bad in (True, 0, -1, 1.5):
            with self.assertRaises(ValueError): BatchPlan(10, 4, bad, 7)


@unittest.skipUnless(CORE, "requires the isolated BF2.1 core environment")
class UpdateBudgetTests(unittest.TestCase):
    def fixture(self, root, updates=3):
        task = tasks.heston_task()
        config = {"task": task, "root_seed": 12, "splits": {"train": 16, "validation": 8, "test": 4}}
        manifest = {"configuration": config, "task_fingerprint": bank.task_fingerprint(config)}
        bank_path = root / "bank"; bank_path.mkdir()
        (bank_path / "manifest.json").write_text(json.dumps(manifest))
        rng = np.random.default_rng(51); bounds = np.asarray(task["prior_bounds"])
        def split(n, name):
            values = rng.normal(.02, .001, (n, 63))
            return {"parameters": rng.uniform(bounds[:, 0], bounds[:, 1], (n, 4)),
                    "reference_prices": values, "observations": values[..., None],
                    "context_values": np.empty((n, 0)), "example_id": np.array([f"{name}:{i}" for i in range(n)])}
        splits = {"train": split(16, "train"), "validation": split(8, "validation")}
        args = Namespace(bank=bank_path, out=root / "run", family="flow_matching", encoder="vector",
                         widths="8,8", summary_dim=8, flow_steps=None, mean_steps=4,
                         epochs=2, batch_size=8, chunk_epochs=1, seed=3, updates=updates,
                         checkpoint_selection="final")
        return args, manifest, splits

    def test_exact_partial_pass_reload_and_reject_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, manifest, splits = self.fixture(Path(temporary))
            with patch.object(trainer.bank, "load_bank", return_value=(manifest, splits)) as load:
                record = trainer.run_training(args)
            self.assertEqual(load.call_args.kwargs["splits"], ("train", "validation"))
            self.assertEqual(load.call_args.kwargs["roles"], trainer.bank.TRAINING_ROLES)
            self.assertEqual(record["selected_update"], 3)
            self.assertEqual(record["checkpoint_selection"], "endpoint")
            self.assertTrue(record["same_seed_reload_exact"])
            self.assertFalse(record["reserved_test_evaluated"])
            _, restored, loaded = full_surface.load_checkpoint(args.out)
            self.assertEqual(int(restored.estimator.optimizer.iterations.numpy()), 3)
            self.assertEqual(loaded["checkpoint_sha256"], record["checkpoint_sha256"])
            with np.load(args.out / "training-exposure.npz") as exposure:
                self.assertEqual(exposure["presentations"].sum(), 24)
                self.assertEqual(np.count_nonzero(exposure["presentations"]), 16)
            self.assertFalse((args.out / "resume.json").exists())
            with self.assertRaisesRegex(ValueError, "nonempty training output"):
                trainer.run_training(args)
            args.updates = 4
            with self.assertRaisesRegex(ValueError, "nonempty training output"):
                trainer.run_training(args)

    def test_batch_accounting_never_reads_callback_model_and_verifies_actual_endpoint(self):
        plan = BatchPlan(16, 8, 3, 3)
        callback = update_guard(plan)
        actual = [0]
        model = SimpleNamespace(steps_per_execution=1,
            optimizer=SimpleNamespace(iterations=SimpleNamespace(numpy=lambda: actual[0])))
        callback.set_model(model)
        callback.on_train_begin()
        # Keras' real JAX model getter synchronizes all model/optimizer state.
        # Forbid access while checking every scheduled callback and row visit.
        with patch.object(type(callback), "model", new_callable=PropertyMock,
                          side_effect=AssertionError("per-batch model access forces JAX state sync")) as getter:
            for batch in range(3):
                callback.on_train_batch_end(batch)
            self.assertEqual(getter.call_count, 0)
        self.assertEqual(callback.updates, 3)
        self.assertEqual(callback.visits.sum(), 24)
        self.assertEqual(np.count_nonzero(callback.visits), 16)
        # Callback counts alone cannot certify that the optimizer advanced.
        with self.assertRaisesRegex(RuntimeError, "exact requested update"):
            callback.on_train_end()
        actual[0] = 3
        callback.on_train_end()
        with self.assertRaisesRegex(RuntimeError, "exceeded"):
            callback.on_train_batch_end(3)

    def test_multi_batch_execution_rejected_before_any_update(self):
        import keras
        from sbi4finance.calibration import inference
        with tempfile.TemporaryDirectory() as temporary:
            args, manifest, splits = self.fixture(Path(temporary))
            task = manifest["configuration"]["task"]
            keras.utils.set_random_seed(5)
            optimizer = inference.offline_optimizer(3)
            workflow = full_surface.build_workflow(task, optimizer=optimizer, family="flow_matching", widths=(8, 8))
            workflow.estimator.compile(optimizer=optimizer, steps_per_execution=4)
            train = full_surface.training_data(splits["train"], task, clean=True)
            plan = BatchPlan(16, 8, 3, 3)
            dataset = make_dataset(train, workflow.adapter, plan)
            workflow.estimator.build_from_data(dataset[0])
            before = [np.asarray(w).copy() for w in workflow.estimator.trainable_variables]
            with self.assertRaisesRegex(ValueError, "steps_per_execution=1"):
                workflow.estimator.fit(dataset=dataset, epochs=1, shuffle=False, verbose=0,
                                         callbacks=[update_guard(plan)])
            self.assertEqual(int(optimizer.iterations.numpy()), 0)
            for initial, current in zip(before, workflow.estimator.trainable_variables, strict=True):
                np.testing.assert_array_equal(initial, np.asarray(current))

    def test_endpoint_weights_deterministic_and_schedule_budget_specific(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, manifest, splits = self.fixture(root)
            with patch.object(trainer.bank, "load_bank", return_value=(manifest, splits)):
                first = trainer.run_training(args)
                args.out = root / "second"
                second = trainer.run_training(args)
            _, a, _ = full_surface.load_checkpoint(root / "run")
            _, b, _ = full_surface.load_checkpoint(root / "second")
            for x, y in zip(a.estimator.trainable_variables, b.estimator.trainable_variables, strict=True):
                np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
            for record in (first, second):
                work = record["performance"]["work"]
                self.assertEqual(work["optimizer_updates"], 3)
                schedule = work["optimizer_schedule"]["serialized"]["config"]
                self.assertEqual(schedule["warmup_steps"], 0)
                self.assertEqual(schedule["decay_steps"], 3)
