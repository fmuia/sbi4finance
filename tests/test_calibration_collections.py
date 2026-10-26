"""Collection identity and recovery checks; prices are mocked, no numerical study runs."""

import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import runpy
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

from sbi4finance.calibration import bank, bank_collection as collection, tasks


def configuration(model="heston", train=3, root_seed=2611):
    task = (tasks.heston_task(days=(10, 63), deltas=(-1.0, 0.0, 1.0)) if model == "heston"
            else tasks.rough_task(curve_prior=tasks.FULL_BANK_CURVE_PRIOR))
    return {"task": task, "root_seed": root_seed,
            "splits": {"train": train, "validation": 2, "test": 2}}


def cheap_price(task, parameters, context_values, seed, diagnostics=None):
    """Keep the real per-ID seed dispatch and prior/curve/noise draws, replace only pricing."""
    c = len(task["context"]["contracts"])
    values = np.random.default_rng(seed).uniform(0.01, 0.02, c) + 0.001 * np.sum(parameters)
    return values, np.zeros(c), np.zeros((c, c)), np.zeros(c), np.zeros(tasks.block_count(task))


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pricer = patch.object(tasks, "price", side_effect=cheap_price)
        self.price = self.pricer.start()
        self.addCleanup(self.pricer.stop)

    def build(self, path, config, train=None, chunk=2):
        collection.create_collection(path, config)
        plan = collection.plan_extension(path, train or config["splits"]["train"], chunk)
        # Reverse scheduling exercises job-order independence.
        for job in reversed(plan["jobs"]):
            collection.run_job(path, job)
        return plan

    def compare(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            assert_array_equal(left[key], right[key], err_msg=key)

    def test_growing_bank_preserves_prefix_and_never_reprices_finished_rows(self):
        for model in ("heston", "rough"):
            with self.subTest(model=model):
                config = configuration(model)
                root = self.root / model
                old_plan = self.build(root, config)
                snapshot = self.root / (model + "-three")
                collection.freeze_snapshot(root, 3, snapshot)
                _, before = collection.load_bank(snapshot)
                calls = self.price.call_count
                for job in old_plan["jobs"]:
                    collection.run_job(root, job)
                self.assertEqual(self.price.call_count, calls)
                extra = collection.plan_extension(root, 7, 3)
                self.assertTrue(extra["jobs"])
                self.assertTrue(all(j["split"] == "train" and j["start"] >= 3 for j in extra["jobs"]))
                self.assertEqual(sum(j["stop"] - j["start"] for j in extra["jobs"]), 4)
                for job in extra["jobs"]:
                    collection.run_job(root, job)
                self.assertEqual(self.price.call_count - calls, 4)
                larger = self.root / (model + "-seven")
                collection.freeze_snapshot(root, 7, larger)
                _, after = collection.load_bank(larger)
                self.compare(before["train"], {k: v[:3] for k, v in after["train"].items()})
                for split in ("validation", "test"):
                    self.compare(before[split], after[split])
                self.assertEqual(len(set(after["train"]["example_id"])), 7)
                # Counts, order and chunking do not enter the financial experiment's RNG identity.
                direct = copy.deepcopy(config)
                direct["splits"]["train"] = 7
                self.compare(bank.generate_examples(direct, "train", range(7)), after["train"])
                frozen_bytes = (snapshot / "manifest.json").read_bytes()
                with self.assertRaises(FileExistsError):
                    collection.freeze_snapshot(root, 3, snapshot)
                self.assertEqual((snapshot / "manifest.json").read_bytes(), frozen_bytes)

    def test_adopt_existing_shards_reuses_data_despite_larger_split_count(self):
        config = configuration()
        paths = []
        for i, job in enumerate(bank.job_map(config, 2)):
            path = self.root / f"legacy-{i}"
            bank.write_shard(path, config, job["split"], job["start"], job["stop"])
            paths.append(path)
        calls = self.price.call_count
        grown = copy.deepcopy(config)
        grown["splits"]["train"] = 7
        root = self.root / "collection"
        collection.create_collection(root, grown)
        collection.adopt_shards(root, reversed(paths))
        self.assertEqual(calls, self.price.call_count)
        plan = collection.plan_extension(root, 7, 2)
        self.assertTrue(all(j["split"] == "train" and j["start"] >= 3 for j in plan["jobs"]))
        self.assertEqual(sum(j["stop"] - j["start"] for j in plan["jobs"]), 4)
        for job in plan["jobs"]:
            collection.run_job(root, job)
        snapshot = self.root / "snapshot"
        collection.freeze_snapshot(root, 7, snapshot)
        _, arrays = collection.load_bank(snapshot)
        self.compare(bank.generate_examples(grown, "train", range(7)), arrays["train"])

    def test_new_root_has_different_experiments_and_cannot_be_adopted(self):
        config = configuration()
        other = configuration(root_seed=2612)
        self.assertFalse(np.array_equal(bank.generate_examples(config, "train", [0])["parameters"],
                                       bank.generate_examples(other, "train", [0])["parameters"]))
        shard = self.root / "other"
        bank.write_shard(shard, other, "train", 0, 2)
        root = self.root / "collection"
        collection.create_collection(root, config)
        with self.assertRaises(ValueError):
            collection.adopt_shards(root, [shard])

    def test_changed_runtime_or_sources_cannot_extend_a_collection(self):
        root = self.root / "collection"
        plan = self.build(root, configuration())
        for name in ("source_hashes", "versions"):
            with self.subTest(identity=name), patch.object(bank, name, return_value={"changed": "identity"}):
                with self.assertRaises(ValueError):
                    collection.plan_extension(root, 7, 2)
                with self.assertRaises(ValueError):
                    collection.run_job(root, plan["jobs"][0])

    def test_valid_payload_under_wrong_range_name_is_not_reused(self):
        root = self.root / "collection"
        collection.create_collection(root, configuration())
        plan = collection.plan_extension(root, 3, 2)
        training = next(job for job in plan["jobs"] if job["split"] == "train" and job["start"] == 0)
        later = next(job for job in plan["jobs"] if job["split"] == "train" and job["start"] == 2)
        collection.run_job(root, training)
        part = next((root / "parts").iterdir())
        wrong_name = part.with_name(f"train-{later['start']:010d}-{later['stop']:010d}")
        shutil.copytree(part, wrong_name)
        with self.assertRaises(ValueError):
            collection.run_job(root, later)

    def test_projected_load_does_not_decode_covariance_or_final_test(self):
        root = self.root / "collection"
        self.build(root, configuration())
        snapshot = self.root / "snapshot"
        collection.freeze_snapshot(root, 3, snapshot)
        from numpy.lib.npyio import NpzFile
        original = NpzFile.__getitem__
        opened = []

        def selected(archive, key):
            opened.append(key)
            self.assertNotEqual(key, "pricing_covariance")
            return original(archive, key)

        with patch.object(NpzFile, "__getitem__", selected):
            manifest, loaded = collection.load_bank(snapshot, splits=("train", "validation"),
                                                     roles=collection.TRAINING_ROLES)
        self.assertEqual(set(loaded), {"train", "validation"})
        self.assertNotIn("pricing_covariance", opened)
        self.assertEqual(set(loaded["train"]), set(collection.TRAINING_ROLES))
        self.assertEqual(manifest["configuration"]["splits"]["train"], 3)

    def test_corruption_blocks_reuse_instead_of_silently_regenerating(self):
        root = self.root / "collection"
        plan = self.build(root, configuration())
        snapshot = self.root / "snapshot"
        collection.freeze_snapshot(root, 3, snapshot)
        payload = next(root.rglob("arrays.npz"))
        payload.write_bytes(payload.read_bytes() + b"corrupt")
        calls = self.price.call_count
        with self.assertRaises(ValueError):
            collection.plan_extension(root, 7, 2)
        with self.assertRaises(ValueError):
            collection.load_bank(snapshot)
        # Finding corruption must not trigger numerical recomputation.
        self.assertEqual(calls, self.price.call_count)

    def test_missing_ranges_cannot_be_published(self):
        root = self.root / "collection"
        collection.create_collection(root, configuration())
        plan = collection.plan_extension(root, 3, 2)
        collection.run_job(root, plan["jobs"][0])
        with self.assertRaises(ValueError):
            collection.freeze_snapshot(root, 3, self.root / "incomplete")

    def test_exact_pricing_seed_audit_includes_holdouts(self):
        config = configuration()
        actual = []

        def record_seed(task, parameters, context_values, seed, diagnostics=None):
            actual.append(seed)
            return cheap_price(task, parameters, context_values, seed, diagnostics)

        with patch.object(tasks, "price", side_effect=record_seed):
            for split, count in config["splits"].items():
                bank.generate_examples(config, split, range(count))
        expected = [collection.pricing_seed(config, split, i)
                    for split, count in config["splits"].items() for i in range(count)]
        self.assertEqual(actual, expected)
        self.assertEqual(len(set(expected)), sum(config["splits"].values()))
        root = self.root / "collection"
        self.build(root, config)
        snapshot = self.root / "snapshot"
        manifest = collection.freeze_snapshot(root, 3, snapshot)
        self.assertEqual(manifest["seed_audit"]["examples_checked"], 7)
        self.assertEqual(manifest["seed_audit"]["duplicate_pricing_seeds"], 0)

    def test_repeated_pricing_seed_blocks_planning_and_publication(self):
        root = self.root / "collection"
        self.build(root, configuration())

        class RepeatedSequence:
            def __init__(self, *args, **kwargs):
                pass

            def generate_state(self, count, dtype=None):
                return np.zeros(count, dtype=dtype)

        with patch.object(collection.np.random, "SeedSequence", RepeatedSequence):
            with self.assertRaisesRegex(ValueError, "seed collision"):
                collection.plan_extension(root, 7, 2)
            with self.assertRaisesRegex(ValueError, "seed collision"):
                collection.freeze_snapshot(root, 3, self.root / "collision")
        self.assertFalse((self.root / "collision").exists())

    def test_concurrent_retry_prices_one_range_once(self):
        root = self.root / "collection"
        collection.create_collection(root, configuration())
        plan = collection.plan_extension(root, 3, 2)
        job = plan["jobs"][0]
        calls = self.price.call_count
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda unused: collection.run_job(root, job), range(2)))
        self.assertEqual(self.price.call_count - calls, job["stop"] - job["start"])
        self.assertEqual(results[0]["sha256"], results[1]["sha256"])

    def test_an_older_overlapping_plan_cannot_add_duplicate_rows(self):
        root = self.root / "collection"
        collection.create_collection(root, configuration())
        small = collection.plan_extension(root, 3, 2)
        wide = collection.plan_extension(root, 3, 3)
        first = next(job for job in small["jobs"] if job["split"] == "train" and job["start"] == 0)
        overlapping = next(job for job in wide["jobs"] if job["split"] == "train")
        collection.run_job(root, first)
        calls = self.price.call_count
        with self.assertRaisesRegex(ValueError, "overlap"):
            collection.run_job(root, overlapping)
        self.assertEqual(calls, self.price.call_count)

    def test_aligned_earlier_snapshot_can_be_frozen_after_growth(self):
        root = self.root / "collection"
        self.build(root, configuration(), train=6, chunk=2)
        snapshot = self.root / "earlier"
        collection.freeze_snapshot(root, 2, snapshot)
        _, arrays = collection.load_bank(snapshot)
        self.assertEqual(len(arrays["train"]["parameters"]), 2)
        with self.assertRaisesRegex(ValueError, "boundary|through"):
            collection.freeze_snapshot(root, 3, self.root / "middle")

    def test_legacy_projected_reader_matches_the_original_loader(self):
        config = configuration()
        legacy = self.root / "legacy"
        bank.write_bank(legacy, config)
        _, original = bank.load_bank(legacy, splits=("train", "validation"))
        _, projected = collection.load_bank(legacy, splits=("train", "validation"),
                                              roles=collection.TRAINING_ROLES)
        for split in projected:
            self.compare({k: original[split][k] for k in collection.TRAINING_ROLES}, projected[split])

    def test_changed_job_cannot_reuse_another_range_or_collection(self):
        root = self.root / "collection"
        plan = self.build(root, configuration())
        changed = copy.deepcopy(plan["jobs"][0])
        changed["stop"] = changed["start"] + 1
        with self.assertRaises(ValueError):
            collection.run_job(root, changed)
        other = self.root / "other"
        collection.create_collection(other, configuration(root_seed=123))
        with self.assertRaises(ValueError):
            collection.run_job(other, plan["jobs"][0])

    def test_entropy_integer_domain_is_explicit(self):
        for value in (-1, 2**32, True):
            with self.subTest(root_seed=value):
                config = configuration(root_seed=value)
                with self.assertRaises(ValueError):
                    collection.create_collection(self.root / str(value), config)

    def test_cli_help_and_empty_plan_map(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "calibration_bank_collection.py"
        help_result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        root = self.root / "collection"
        self.build(root, configuration())
        destination = self.root / "jobs.json"
        result = subprocess.run([sys.executable, str(script), "plan", "--collection", str(root),
                                 "--train", "3", "--out", str(destination)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(destination.read_text())["jobs"], [])

    def test_cli_compact_map_runs_the_same_planned_job(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "calibration_bank_collection.py"
        root = self.root / "collection"
        collection.create_collection(root, configuration())
        destination = self.root / "jobs.json"
        argv = [str(script), "plan", "--collection", str(root), "--train", "3", "--chunk", "2",
                "--out", str(destination)]
        with patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
            runpy.run_path(str(script), run_name="__main__")
        record = json.loads(destination.read_text())
        self.assertIn("configuration", record)
        self.assertTrue(all("configuration" not in job for job in record["jobs"]))
        calls = self.price.call_count
        argv = [str(script), "shard", "--job-map", str(destination), "--task-index", "0"]
        with patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
            runpy.run_path(str(script), run_name="__main__")
        job = record["jobs"][0]
        self.assertEqual(self.price.call_count - calls, job["stop"] - job["start"])


if __name__ == "__main__":
    unittest.main()
