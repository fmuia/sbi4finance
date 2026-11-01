"""Checks of numerical-law identity, deterministic path splitting and integrity."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from sbi4finance.path_inference import bank
from sbi4finance.rough_bergomi import rough_driver


def small_task(model):
    task = bank.default_task(model)
    task.update(n_observations=12, history_length_range=[1, 12], prefix_lengths=[1, 3, 12])
    return task


class PathBankTests(unittest.TestCase):
    def test_integrity_check_releases_each_shard_without_materializing_splits(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "bank"
            manifest = bank.write_bank(small_task("heston"), path, 7, 3, 0, chunk=2, seed=91)
            original = bank._load_part
            references = []

            def checked_load(*args):
                self.assertTrue(all(reference() is None for reference in references),
                                "the preceding decoded shard is still retained")
                arrays = original(*args)
                references[:] = [weakref.ref(value) for value in arrays.values()]
                return arrays

            with patch.object(bank, "load_split", side_effect=AssertionError("whole split loaded")):
                with patch.object(bank, "_load_part", side_effect=checked_load):
                    report = bank.verify_bank(path)
            self.assertTrue(report["passed"])
            self.assertEqual(report["counts"], {"train": 7, "validation": 3, "test": 0})
            self.assertEqual(report["checked_shards"], 6)
            self.assertEqual(report["task_sha256"], manifest["task_sha256"])
            self.assertLess(report["largest_decoded_shard_bytes"], 20_000)
            self.assertTrue(all(reference() is None for reference in references))

    def test_streamed_integrity_rejects_corruption_and_unlisted_shards(self):
        for mutation, message in (("bytes", "checksum"), ("size", "payload size"),
                                  ("extra", "inventory")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as root:
                path = Path(root) / "bank"
                manifest = bank.write_bank(small_task("heston"), path, 2, 1, 1, chunk=2, seed=92)
                payload = path / manifest["splits"]["train"][0]["path"]
                if mutation == "bytes":
                    data = bytearray(payload.read_bytes())
                    data[len(data) // 2] ^= 1
                    payload.write_bytes(data)
                elif mutation == "size":
                    with payload.open("ab") as stream:
                        stream.write(b"changed")
                else:
                    (path / "train" / "unlisted.npz").write_bytes(payload.read_bytes())
                with self.assertRaisesRegex(ValueError, message):
                    bank.verify_bank(path)

    def test_tasks_declare_physical_measure_and_full_history_range(self):
        for model, dimension, maximum in (("heston", 4, 1260), ("rough-known", 3, 1008),
                                           ("rough-joint", 6, 1008)):
            task = bank.default_task(model)
            self.assertEqual(task["measure"], "P")
            self.assertEqual(task["history_length_range"], [1, maximum])
            self.assertEqual(task["prefix_lengths_role"], "validation_landmarks")
            self.assertEqual(len(task["parameter_names"]), dimension)
        self.assertEqual(bank.default_task("rough-joint")["numerics"]["variance_compensator"],
                         "discrete_hybrid_variance")
        self.assertEqual(bank.default_task("heston")["numerics"]["substeps"], 8)

    def test_chunks_reproduce_exact_trajectories_for_every_model(self):
        for model in bank.MODELS:
            task = small_task(model)
            whole = bank.generate(task, "train", 0, 5, 77)
            chunks = [bank.generate(task, "train", start, count, 77) for start, count in ((0, 2), (2, 3))]
            for key in whole:
                np.testing.assert_array_equal(whole[key], np.concatenate([x[key] for x in chunks]))
            self.assertEqual(set(whole), {"observations", "parameters", "trajectory_id"})
            self.assertEqual(whole["observations"].shape, (5, 12, 1))

    def test_split_and_root_random_domains_are_independent(self):
        task = small_task("rough-joint")
        train = bank.generate(task, "train", 0, 3, 10)
        validation = bank.generate(task, "validation", 0, 3, 10)
        another_root = bank.generate(task, "train", 0, 3, 11)
        for other in (validation, another_root):
            self.assertFalse(np.array_equal(train["parameters"], other["parameters"]))
            self.assertFalse(np.array_equal(train["observations"], other["observations"]))
            self.assertFalse(set(train["trajectory_id"]) & set(other["trajectory_id"]))

    def test_future_innovations_do_not_change_observed_prefix(self):
        factory = bank._rng
        prefix = 4
        for model in bank.MODELS:
            task = small_task(model)
            baseline = bank.generate(task, "train", 0, 2, 44)

            def changed_rng(digest, split, identifier, seed, domain):
                generator = factory(digest, split, identifier, seed, domain)
                if domain == 0:
                    return generator

                class FutureChanged:
                    def standard_normal(self, shape):
                        values = generator.standard_normal(shape)
                        if model == "heston":
                            values[prefix:] += 2.0
                        else:
                            values[:, prefix:] += 2.0
                        return values

                return FutureChanged()

            with patch.object(bank, "_rng", side_effect=changed_rng):
                changed = bank.generate(task, "train", 0, 2, 44)
            np.testing.assert_array_equal(baseline["parameters"], changed["parameters"])
            # FFT convolution can change rounding at machine precision when
            # later innovations change; its causal convolution is unchanged.
            np.testing.assert_allclose(baseline["observations"][:, :prefix],
                                       changed["observations"][:, :prefix], rtol=1e-12, atol=1e-14)
            self.assertFalse(np.array_equal(baseline["observations"][:, prefix:],
                                            changed["observations"][:, prefix:]))

    def test_heston_first_return_has_physical_drift_and_leverage(self):
        task = small_task("heston")
        # An exact single-step Gaussian calculation checks the P law and
        # leverage routing; the primary bank uses eight internal steps.
        task["numerics"]["substeps"] = 1
        task["fixed"]["drift"] = 0.17
        arrays = bank.generate(task, "validation", 9, 1, 123)
        p = dict(zip(task["parameter_names"], arrays["parameters"][0]))
        normals = bank._rng(bank.task_fingerprint(task), "validation", 9, 123, 1).standard_normal((12, 1, 2))
        variance, dt = task["fixed"]["initial_variance"], task["dt_years"]
        expected = ((0.17 - 0.5 * variance) * dt + np.sqrt(variance * dt)
                    * (p["rho"] * normals[0, 0, 0] + np.sqrt(1 - p["rho"] ** 2) * normals[0, 0, 1]))
        self.assertAlmostEqual(arrays["observations"][0, 0, 0], expected, places=15)

    def test_rough_returns_use_declared_curve_and_discrete_compensator(self):
        for model in ("rough-known", "rough-joint"):
            task = small_task(model)
            arrays = bank.generate(task, "validation", 7, 1, 23)
            p = dict(zip(task["parameter_names"], arrays["parameters"][0]))
            normals = bank._rng(bank.task_fingerprint(task), "validation", 7, 23, 1).standard_normal((1, 12, 3))
            driver, compensation = rough_driver(p["hurst"], task["dt_years"], normals)
            times = task["dt_years"] * np.arange(12)
            curve = (np.full(12, 0.04) if model == "rough-known" else
                     p["long_variance"] + (p["short_variance"] - p["long_variance"]) * np.exp(-p["decay"] * times))
            left = curve * np.exp(p["eta"] * np.r_[0, driver[0, :-1]]
                                  - 0.5 * p["eta"] ** 2 * np.r_[0, compensation[:-1]])
            expected = ((task["fixed"]["drift"] - 0.5 * left) * task["dt_years"]
                        + np.sqrt(left * task["dt_years"]) * (p["rho"] * normals[0, :, 0]
                                     + np.sqrt(1 - p["rho"] ** 2) * normals[0, :, 2]))
            np.testing.assert_allclose(arrays["observations"][0, :, 0], expected, rtol=1e-14, atol=1e-16)
            self.assertGreater(np.max(np.abs(compensation - (task["dt_years"] * np.arange(1, 13)) ** (2 * p["hurst"]))), 0)

    def test_prior_labels_respect_original_physical_support(self):
        for model in bank.MODELS:
            task = small_task(model)
            draws = bank.generate(task, "train", 10, 31, 57)["parameters"]
            bounds = np.asarray(task["prior_bounds"])
            self.assertTrue(np.all(draws >= bounds[:, 0]))
            self.assertTrue(np.all(draws <= bounds[:, 1]))

    def test_banks_verify_and_chunk_partition_does_not_change_samples(self):
        with tempfile.TemporaryDirectory() as root:
            roots = [Path(root) / "a", Path(root) / "b"]
            task = small_task("heston")
            for path, chunk in zip(roots, (2, 3)):
                bank.write_bank(task, path, 5, 3, 0, chunk=chunk, seed=9)
            for split in bank.SPLITS:
                a, b = [bank.load_split(path, split) for path in roots]
                self.assertEqual(a.pop("task"), task)
                b.pop("task")
                for key in a:
                    np.testing.assert_array_equal(a[key], b[key])
            self.assertEqual(bank.load_manifest(roots[0])["split_counts"], {"train": 5, "validation": 3, "test": 0})

    def test_writer_keeps_generation_chunk_bounded(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(bank, "generate", wraps=bank.generate) as generate:
                bank.write_bank(small_task("rough-known"), Path(root) / "bank", 5, 1, 0, chunk=2)
            self.assertTrue(all(call.args[3] <= 2 for call in generate.call_args_list))

    def test_checksum_and_split_identity_corruption_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "bank"
            bank.write_bank(small_task("rough-known"), path, 2, 0, 0)
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            record = manifest["splits"]["train"][0]
            part = path / record["path"]
            original = part.read_bytes()
            part.write_bytes(original + b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum"):
                bank.load_split(path, "train")
            part.write_bytes(original)
            with np.load(part, allow_pickle=False) as saved:
                arrays = dict(saved)
            arrays["trajectory_id"][0] = arrays["trajectory_id"][1]
            np.savez(part, **arrays)
            record["sha256"] = hashlib.sha256(part.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "trajectory IDs"):
                bank.load_split(path, "train")

    def test_task_or_compensator_changes_cannot_pass_old_identity(self):
        task = small_task("rough-known")
        changed = copy.deepcopy(task)
        changed["numerics"]["variance_compensator"] = "continuum_t2H"
        with self.assertRaisesRegex(ValueError, "compensator"):
            bank.task_fingerprint(changed)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "bank"
            bank.write_bank(task, path, 1, 0, 0)
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["task"]["fixed"]["drift"] = 0.15
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                bank.load_manifest(path)

    def test_teaching_tasks_share_daily_design_and_infer_drift_without_changing_legacy_defaults(self):
        before = {model: bank.default_task(model) for model in bank.MODELS}
        for model, dimension in (("heston", 5), ("rough-known", 4), ("rough-joint", 7)):
            task = bank.teaching_task(model)
            bank.validate_task(task)
            self.assertEqual(task["task_version"], 2)
            self.assertEqual(task["task_id"], bank.TEACHING_TASK_IDS[model])
            self.assertEqual(task["dt_years"], 1 / 252)
            self.assertEqual(task["n_observations"], 1260)
            self.assertEqual(task["prefix_lengths"], [21, 63, 126, 252, 504, 756, 1260])
            self.assertEqual(task["history_length_range"], [1, 1260])
            self.assertEqual(len(task["parameter_names"]), dimension)
            self.assertEqual(task["parameter_names"][-1], "drift")
            self.assertEqual(task["prior_bounds"][-1], [-.10, .20])
            self.assertEqual(task["prior_types"][-1], "uniform")
            self.assertNotIn("drift", task["fixed"])
            if model == "heston":
                self.assertEqual(task["numerics"]["substeps"], 8)
            else:
                self.assertEqual(task["numerics"]["observation_substeps"], 4)
            self.assertEqual(bank.default_task(model), before[model])
            self.assertNotEqual(bank.task_fingerprint(task), bank.task_fingerprint(before[model]))

    def test_teaching_smoke_lengths_chunk_identity_and_negative_drift_labels(self):
        for model in bank.MODELS:
            task = bank.teaching_task(model, observations=9)
            self.assertEqual(task["prefix_lengths"], [9])
            self.assertEqual(task["history_length_range"], [1, 9])
            whole = bank.generate(task, "train", 0, 32, 2026103101)
            chunks = [bank.generate(task, "train", 0, 13, 2026103101),
                      bank.generate(task, "train", 13, 19, 2026103101)]
            for key in whole:
                np.testing.assert_array_equal(whole[key], np.concatenate([part[key] for part in chunks]))
            self.assertEqual(whole["observations"].shape, (32, 9, 1))
            self.assertTrue(np.any(whole["parameters"][:, -1] < 0))
            self.assertTrue(np.any(whole["parameters"][:, -1] > 0))
            self.assertEqual(set(whole), {"observations", "parameters", "trajectory_id"})

    def test_paired_drift_changes_shift_each_daily_return_by_delta_times_dt(self):
        factory = bank._rng
        for model in bank.MODELS:
            task = bank.teaching_task(model, observations=11)

            def generate_with_drift_quantile(quantile):
                def controlled(digest, split, identifier, seed, domain):
                    generator = factory(digest, split, identifier, seed, domain)
                    if domain != 0:
                        return generator

                    class ControlledParameterPrior:
                        def random(self, size):
                            values = generator.random(size)
                            values[-1] = quantile
                            return values

                    return ControlledParameterPrior()

                with patch.object(bank, "_rng", side_effect=controlled):
                    return bank.generate(task, "validation", 3, 4, 2026103103)

            low, high = generate_with_drift_quantile(.2), generate_with_drift_quantile(.8)
            np.testing.assert_array_equal(low["parameters"][:, :-1], high["parameters"][:, :-1])
            np.testing.assert_array_equal(low["trajectory_id"], high["trajectory_id"])
            shift = (high["parameters"][:, -1] - low["parameters"][:, -1])[:, None, None] * task["dt_years"]
            np.testing.assert_allclose(high["observations"] - low["observations"],
                                       np.broadcast_to(shift, high["observations"].shape), rtol=1e-11, atol=1e-15)

    def test_teaching_rough_daily_returns_sum_full_origin_intraday_path(self):
        for model in ("rough-known", "rough-joint"):
            task = bank.teaching_task(model, observations=7)
            generated = bank.generate(task, "validation", 17, 1, 2026103104)
            p = dict(zip(task["parameter_names"], generated["parameters"][0]))
            internal_count = 7 * 4
            dt = task["dt_years"] / 4
            normals = bank._rng(bank.task_fingerprint(task), "validation", 17, 2026103104, 1).standard_normal(
                (1, internal_count, 3))
            driver, compensation = rough_driver(p["hurst"], dt, normals)
            time = dt * np.arange(internal_count)
            curve = (np.full(internal_count, .04) if model == "rough-known" else
                     p["long_variance"] + (p["short_variance"] - p["long_variance"]) * np.exp(-p["decay"] * time))
            variance = curve * np.exp(p["eta"] * np.r_[0., driver[0, :-1]]
                                      - .5 * p["eta"] ** 2 * np.r_[0., compensation[:-1]])
            fine = ((p["drift"] - .5 * variance) * dt + np.sqrt(variance * dt)
                    * (p["rho"] * normals[0, :, 0] + np.sqrt(1 - p["rho"] ** 2) * normals[0, :, 2]))
            expected = fine.reshape(7, 4).sum(axis=1)
            np.testing.assert_allclose(generated["observations"][0, :, 0], expected, rtol=1e-14, atol=1e-16)

    def test_teaching_law_and_inferred_drift_contract_cannot_be_changed_silently(self):
        task = bank.teaching_task("rough-joint", observations=7)
        for mutate in (
            lambda t: t.pop("task_version"),
            lambda t: t.update(dt_years=1 / 1008),
            lambda t: t["fixed"].update(drift=.05),
            lambda t: t["numerics"].update(observation_substeps=1),
            lambda t: t["prior_types"].__setitem__(-1, "log_uniform"),
        ):
            changed = copy.deepcopy(task)
            mutate(changed)
            with self.assertRaises(ValueError):
                bank.validate_task(changed)
        heston = bank.teaching_task("heston", observations=7)
        heston["numerics"]["substeps"] = 1
        with self.assertRaisesRegex(ValueError, "eight"):
            bank.validate_task(heston)

    def test_teaching_banks_record_measured_payload_and_verify(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "teaching-bank"
            task = bank.teaching_task("rough-joint", observations=7)
            written = bank.write_bank(task, out, 5, 3, 1, chunk=2, seed=2026103105)
            actual = sum(path.stat().st_size for path in out.glob("*/*.npz"))
            self.assertEqual(written["payload_bytes"], actual)
            self.assertEqual(sum(written["payload_bytes_by_split"].values()), actual)
            for split, records in written["splits"].items():
                self.assertEqual(sum(record["bytes"] for record in records), written["payload_bytes_by_split"][split])
                loaded = bank.load_split(out, split)
                self.assertEqual(loaded["task"], task)
                self.assertEqual(loaded["parameters"].shape[1], 7)
            self.assertGreaterEqual(written["timing_seconds"]["total_including_verification"],
                                    written["timing_seconds"]["total"])


if __name__ == "__main__":
    unittest.main()
