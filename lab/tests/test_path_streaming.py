"""Bounded I/O, deterministic offline epochs, and real production-length fits."""

from importlib import metadata
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np

from sbi4finance.path_inference import bank, inference

try:
    CORE = metadata.version("bayesflow") == "2.1.0"
except metadata.PackageNotFoundError:
    CORE = False

if CORE:
    from sbi4finance.path_inference import dataset


@unittest.skipUnless(CORE, "requires locked BayesFlow2.1 core runtime")
class StreamingDatasetTests(unittest.TestCase):
    def test_deterministic_first_n_epochs_cache_and_adapter_boundary(self):
        task = bank.teaching_task("rough-known", observations=17)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "bank"
            bank.write_bank(task, path, 23, 7, 3, chunk=5, seed=83)
            expected = bank.generate(task, "train", 0, 13, seed=83)
            with mock.patch.object(bank, "load_split", side_effect=AssertionError("whole split loaded")):
                with mock.patch.object(bank, "_load_part", wraps=bank._load_part) as load:
                    a = dataset.ShardedPathDataset(path, adapter=None, training_limit=13, batch_size=4,
                                                  seed=84, prefix_policy="uniform_log_mixture", cache_shards=1)
                    b = dataset.ShardedPathDataset(path, adapter=None, training_limit=13, batch_size=4,
                                                  seed=84, prefix_policy="uniform_log_mixture", cache_shards=1)
                    np.testing.assert_array_equal(np.sort(a.order), np.arange(13))
                    first = a.order.copy()
                    for index in range(len(a)):
                        np.testing.assert_array_equal(a.batch_indices(index), b.batch_indices(index))
                        batch, repeated, same = a[index], a[index], b[index]
                        for key in batch:
                            np.testing.assert_array_equal(batch[key], repeated[key])
                            np.testing.assert_array_equal(batch[key], same[key])
                        self.assertEqual(set(batch), {"observations", "parameters", "time_context", "summary_mask"})
                        self.assertEqual(batch["observations"].shape[1:], (17, 2))
                        self.assertTrue(np.all(batch["observations"][~batch["summary_mask"]] == 0))
                    for value in (a, b):
                        value.on_epoch_end()
                    self.assertFalse(np.array_equal(first, a.order))
                    np.testing.assert_array_equal(a.order, b.order)
                    for index in range(len(a)):
                        a[index]
                    # Each instance checks each accessed shard once, not once
                    # per epoch or framework signature probe.
                    self.assertEqual(a.reader.hash_verifications, 3)
                    self.assertEqual(b.reader.hash_verifications, 3)
                    self.assertEqual(load.call_count, 6)
                    self.assertGreater(a.reader.loads, a.reader.hash_verifications)
                    self.assertLessEqual(a.reader.peak_cache_shards, 1)
                    rows = np.array([12, 0, 7])
                    raw = a.raw_batch(rows)
                    for key in raw:
                        np.testing.assert_array_equal(raw[key], expected[key][rows])
                    self.assertEqual(len(a[-1]["parameters"]), 1)

    def test_validation_subset_only_reads_required_shards_and_integrity_is_guarded(self):
        task = bank.teaching_task("heston", observations=17)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "bank"
            manifest = bank.write_bank(task, path, 23, 7, 3, chunk=5, seed=85)
            with mock.patch.object(bank, "load_split", side_effect=AssertionError("whole split loaded")):
                with mock.patch.object(bank, "_load_part", wraps=bank._load_part) as load:
                    part = dataset.load_subset(path, "validation", 3, manifest=manifest)
                    self.assertEqual(part["observations"].shape, (3, 17, 1))
                    self.assertEqual(load.call_count, 1)
                    self.assertEqual(load.call_args.args[2], "validation")
            with self.assertRaisesRegex(ValueError, "reserved test"):
                dataset.load_subset(path, "test", 1)
            reader = dataset.ShardReader(path, manifest, cache_shards=1)
            record = manifest["splits"]["train"][0]
            reader.load("train", record)
            (path / record["path"]).write_bytes((path / record["path"]).read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "changed after verification"):
                reader.load("train", record)
            fresh = dataset.ShardReader(path, manifest, cache_shards=1)
            with self.assertRaisesRegex(ValueError, "checksum"):
                fresh.load("train", record)

    def test_identity_subset_limits_memory_projection_and_comparison_budgets(self):
        task = bank.teaching_task("rough-joint")
        fake = {"task": task, "task_sha256": bank.task_fingerprint(task), "root_seed": 86,
                "split_counts": {"train": 1_000_000, "validation": 2000, "test": 2000},
                "splits": {"train": [{"start": i, "count": 1000} for i in range(0, 1_000_000, 1000)]}}
        for count in (100_000, 1_000_000):
            memory = dataset.streaming_memory_projection(fake, count, 128, 256,
                sorted(set(task["prefix_lengths"] + [1, 2, 4, 8, 17, 32, 1253])), cache_shards=2)
            self.assertLess(memory["estimated_peak_host_bytes"], memory["host_guard_bytes"])
            self.assertLess(memory["estimated_peak_host_bytes"], 2 * 1024 ** 3)
        self.assertEqual(dataset.planned_optimizer_steps(100_000, 128, 50), 39100)
        self.assertEqual(dataset.planned_optimizer_steps(1_000_000, 128, 5), 39065)
        labels = inference.prior_samples(task, 100, 86)
        drift_index = task["parameter_names"].index("drift")
        self.assertTrue(np.any(labels[:, drift_index] < 0))
        np.testing.assert_allclose(inference.inverse_parameters(inference.transform_parameters(labels, task), task),
                                   labels, rtol=1e-13, atol=1e-13)
        small = bank.teaching_task("heston", observations=17)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "bank"
            manifest = bank.write_bank(small, path, 23, 7, 3, chunk=5, seed=86)
            first = dataset.subset_identity(path, manifest, 13)
            self.assertEqual(first, dataset.subset_identity(path, manifest, 13))
            self.assertNotEqual(first["subset_sha256"], dataset.subset_identity(path, manifest, 23)["subset_sha256"])
            self.assertEqual(sum(r["selected_count"] for r in first["source_shards"]), 13)
            with self.assertRaisesRegex(ValueError, "no larger"):
                dataset.training_count(manifest, 24)


@unittest.skipUnless(CORE, "requires locked BayesFlow2.1 core runtime")
class ActualStreamingWorkflowTests(unittest.TestCase):
    def test_real_production_length_all_models_fresh_training_reload_and_masks(self):
        import keras
        lab = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("path_streaming_trainer_test", lab / "scripts/train_path_posterior.py")
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for model, dimension in (("heston", 5), ("rough-known", 4), ("rough-joint", 7)):
                with self.subTest(model=model):
                    task = bank.teaching_task(model)
                    self.assertEqual(task["n_observations"], 1260)
                    self.assertEqual(len(task["parameter_names"]), dimension)
                    path = root / (model + "-bank")
                    bank.write_bank(task, path, train=7, validation=2, test=2, chunk=2, seed=87)
                    args = SimpleNamespace(bank=path, out=root / model, epochs=2, batch_size=2,
                        seed=20261031087, max_seconds=3300, validation_cases=2,
                        hidden_dim=None, summary_dim=None, depth=None, bins=None, width=None,
                        prefix_policy="uniform", learning_rate=5e-4, init_checkpoint=None,
                        checkpoint_selection="final", streaming=True, training_limit=3, cache_shards=1)
                    with mock.patch.object(bank, "load_split", side_effect=AssertionError("whole split loaded")):
                        report = trainer.run_training(args)
                    self.assertEqual(report["optimizer_steps"], 4)
                    self.assertEqual(report["planned_optimizer_steps"], 4)
                    self.assertEqual(report["planned_training_views"], 6)
                    self.assertEqual(report["completed_training_views"], 6)
                    self.assertEqual(report["bank_training_trajectories"], 7)
                    self.assertEqual(report["training_trajectories"], 3)
                    self.assertEqual(report["network"], inference.DEFAULT_NETWORK)
                    self.assertEqual(report["initialization"]["initial_optimizer_steps"], 0)
                    self.assertEqual(report["initialization"]["mode"], "fresh")
                    self.assertEqual(report["effective_backend_seed"], 20261031087 % 2 ** 32)
                    self.assertTrue(report["same_seed_reload_exact"])
                    self.assertEqual(report["validation_landmarks"],
                                     sorted(set(task["prefix_lengths"] + [1, 2, 4, 8, 17, 32, 1253])))
                    self.assertEqual(report["model_sha256"], inference.sha256(args.out / "model.keras"))
                    self.assertEqual(report["bank_manifest_sha256"], inference.sha256(path / "manifest.json"))
                    self.assertLessEqual(report["streaming"]["observed"]["peak_cache_shards"], 1)
                    loaded_task, workflow, restored = inference.load_checkpoint(args.out)
                    self.assertEqual(loaded_task, task)
                    observed = dataset.load_subset(path, "validation", 2)["observations"]
                    lengths = np.array([1, 1253])
                    baseline = inference.sample(workflow, task, observed, 4, lengths=lengths, seed=88)
                    changed = observed.copy()
                    for row, length in enumerate(lengths):
                        changed[row, length:] = np.nan
                    other = inference.sample(workflow, task, changed, 4, lengths=lengths, seed=88)
                    np.testing.assert_array_equal(baseline, other)
                    self.assertEqual(baseline.shape, (2, 4, dimension))
                    # Adapter labels cannot enter summary variables or context.
                    stream = dataset.ShardedPathDataset(path, adapter=workflow.adapter, training_limit=3,
                                                        batch_size=2, seed=89, cache_shards=1)
                    batch = stream[0]
                    self.assertEqual(set(batch), {"inference_variables", "summary_variables", "inference_conditions", "summary_mask"})
                    self.assertEqual(batch["inference_variables"].shape, (2, dimension))
                    self.assertEqual(batch["summary_variables"].shape, (2, 1260, 2))
                    self.assertEqual(batch["inference_conditions"].shape, (2, 2))
                    keras.backend.clear_session()


if __name__ == "__main__":
    unittest.main()
