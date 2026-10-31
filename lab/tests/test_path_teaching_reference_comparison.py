import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.stats import norm, truncnorm

from sbi4finance.path_inference import bank, teaching_reference


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_path_teaching_reference.py"
_SPEC = importlib.util.spec_from_file_location("compare_path_teaching_reference", _SCRIPT)
comparison = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(comparison)


class TeachingReferenceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.task = {"parameter_names": ["theta", "drift"], "prior_bounds": [[0, 1], [-.1, .2]],
                     "prior_types": ["uniform", "uniform"]}
        self.payload = {"parameters_nondrift": np.array([[.1], [.6]]), "weights": np.array([.3, .7]),
                        "drift_location": np.array([0, .2]), "drift_scale": np.array([.1, .05])}

    def test_drift_cdf_is_weighted_conditional_mixture_not_location_or_resampling(self):
        values = comparison.component_values(self.payload, 1, .05, [-.1, .2])
        expected = [truncnorm.cdf(.05, (-.1 - m) / s, (.2 - m) / s, loc=m, scale=s)
                    for m, s in zip(self.payload["drift_location"], self.payload["drift_scale"], strict=True)]
        assert_allclose(values, expected, atol=1e-14)
        mixture = float(self.payload["weights"] @ values)
        self.assertGreater(abs(mixture - .3), .01)
        assert_allclose(comparison.component_values(self.payload, 1, -.1, [-.1, .2]), 0)
        assert_allclose(comparison.component_values(self.payload, 1, .2, [-.1, .2]), 1)
        assert_allclose(comparison.component_values(self.payload, 0, .1, [-.1, .2]), [1, 0])

    def test_conditional_drift_moments_against_numerical_integrals(self):
        locations = np.array([0, .2, 10, -3])
        scales = np.array([.1, .05, 10, .8])
        means, variances = comparison.drift_moments(locations, scales, [-.1, .2])
        for j, (location, scale) in enumerate(zip(locations, scales, strict=True)):
            density = lambda x: norm.pdf(x, loc=location, scale=scale)
            mass = quad(density, -.1, .2, epsabs=1e-14)[0]
            mean = quad(lambda x: x * density(x), -.1, .2, epsabs=1e-14)[0] / mass
            variance = quad(lambda x: (x - mean) ** 2 * density(x), -.1, .2, epsabs=1e-14)[0] / mass
            assert_allclose(means[j], mean, atol=2e-12)
            assert_allclose(variances[j], variance, atol=2e-12)

    def test_log_uniform_prior_targets_and_evidence_weighted_pooling(self):
        task = {"prior_bounds": [[1, np.e], [-.1, .2]], "prior_types": ["log_uniform", "uniform"]}
        prior = comparison.prior_summary(task)
        assert_allclose(prior["mean"], [np.e - 1, .05], atol=1e-14)
        assert_allclose(prior["quantiles"][0], np.exp(comparison.PROBABILITIES))
        self.assertAlmostEqual(comparison.prior_cdf(np.sqrt(np.e), [1, np.e], "log_uniform"), .5)
        payloads = [self.payload, {**self.payload, "parameters_nondrift": self.payload["parameters_nondrift"] + .1}]
        pooled, shares = comparison.pool_runs(payloads, [{"samples": 2, "log_evidence_estimate": 0},
                                                       {"samples": 2, "log_evidence_estimate": np.log(3)}])
        assert_allclose(shares, [.25, .75])
        assert_allclose(pooled["weights"], [.075, .175, .225, .525])
        self.assertAlmostEqual(pooled["weights"].sum(), 1)

    def test_neural_uncertainty_and_interval_mass_use_all_importance_components(self):
        rng = np.random.default_rng(54)
        draws = np.column_stack((rng.uniform(0, 1, 512), rng.uniform(-.1, .2, 512)))
        summary = comparison.component_summary(self.payload, self.task)
        result = comparison.compare_draws(self.task, draws, self.payload, summary, [self.payload])
        for col, name in enumerate(self.task["parameter_names"]):
            current = result[name]
            self.assertAlmostEqual(current["neural_mean_iid_mcse"], draws[:, col].std(ddof=1) / np.sqrt(512))
            self.assertAlmostEqual(current["iid_asymptotic_nominal90_probability_mcse"], np.sqrt(.09 / 512))
            interval = np.quantile(draws[:, col], [.05, .95])
            expected = self.payload["weights"] @ (comparison.component_values(self.payload, col, interval[1], [-.1, .2])
                                                 - comparison.component_values(self.payload, col, interval[0], [-.1, .2]))
            self.assertAlmostEqual(current["reference_mass_inside_neural90"], expected)
            self.assertEqual(np.asarray(current["neural_quantile_order_statistic_95pct_intervals"]).shape, (3, 2))
        self.assertEqual(comparison.POSTHOC_FLAGS["absolute_mean_difference_in_reference_sd"], .2)

    def test_exact_one_day_density_bound_all_models(self):
        for model in bank.MODELS:
            task = bank.teaching_task(model, observations=21)
            bound = comparison.likelihood_upper_bound(task)
            self.assertTrue(np.isfinite(bound) and bound > 0)
            summary, payload = teaching_reference.importance_reference(task, [.003], samples=128, seed=71, draws=16)
            maximum = payload["weights"].max() * 128 * np.exp(summary["log_evidence_estimate"])
            self.assertLess(maximum, bound)

    def test_end_to_end_hash_bound_cache_reference_and_identity_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = bank.teaching_task("heston", observations=21)
            native = bank.generate(task, "validation", 0, 4, 851)
            replay, directory = root / "replay", root / "reference"
            replay.mkdir()
            directory.mkdir()
            manifest_hash, source_case_hash, model_hash = "a" * 64, "b" * 64, "c" * 64
            arrays = {"observations": native["observations"], "truth": native["parameters"],
                      "trajectory_id": native["trajectory_id"],
                      "draws_1": np.repeat(native["parameters"][:, None, :], 512, axis=1)}
            record = {"task": task, "task_sha256": bank.task_fingerprint(task),
                      "source": {"kind": "development_replay", "split": "validation",
                                 "bank_manifest_sha256": manifest_hash, "source_case_panel_sha256": source_case_hash,
                                 "reserved_test_evaluated": False, "unseen_validation_evaluated": False}, "cases": {}}
            for index in [0, 1]:
                runs = []
                for replicate in [0, 1]:
                    summary, payload = teaching_reference.importance_reference(task, native["observations"][index, :1, 0],
                                                                              samples=8192, seed=190 + 2 * index + replicate,
                                                                              draws=512)
                    path = directory / f"case-{index}-rep-{replicate}.npz"
                    np.savez_compressed(path, **payload)
                    summary.update(payload={"file": path.name, "sha256": comparison.sha256(path)},
                                   trajectory_id=str(native["trajectory_id"][index]))
                    runs.append(summary)
                    if replicate == 0:
                        arrays["draws_1"][index] = payload["posterior_draws"]
                precision = teaching_reference.compare_runs(runs)
                self.assertEqual(precision["status"], "short_history_precision_checks_pass")
                record["cases"][str(index)] = {
                    "trajectory_id": str(native["trajectory_id"][index]),
                    "full_observations_sha256": teaching_reference.observations_sha256(native["observations"][index, :, 0]),
                    "lengths": {"1": {"runs": runs, "comparison": precision}}}
            training = {"task_sha256": record["task_sha256"], "model_sha256": model_hash,
                        "bank_manifest_sha256": manifest_hash}
            evaluation = {"task_sha256": record["task_sha256"], "checkpoint_sha256": model_hash,
                          "bank_manifest_sha256": manifest_hash, "draws": 512,
                          "split": "validation", "offset": 0, "reserved_test_evaluated": False}
            for name, value in [("task.json", task), ("training.json", training), ("evaluation.json", evaluation)]:
                (replay / name).write_text(json.dumps(value))
            np.savez_compressed(replay / "posterior.npz", **arrays)
            bundle = {"format": "trajectory_teaching_replay_v1", "scientific_acceptance": False,
                      "source_split": "validation", "reserved_test_evaluated": False, "model": "heston",
                      "task_sha256": record["task_sha256"], "checkpoint_sha256": model_hash,
                      "source_case_panel_sha256": source_case_hash, "cached_lengths": [1],
                      "case_ids": native["trajectory_id"].tolist(),
                      "files": {name: comparison.sha256(replay / name)
                                for name in ["task.json", "training.json", "evaluation.json", "posterior.npz"]}}
            (replay / "bundle.json").write_text(json.dumps(bundle))
            report_path = directory / "reference-summary.json"
            report_path.write_text(json.dumps(record))
            result = comparison.run(replay, directory, root / "comparison.json")
            self.assertEqual(result["status"], "descriptive_comparison_complete_no_scientific_acceptance")
            self.assertFalse(result["scientific_acceptance"])
            self.assertEqual(set(result["cases"]), {"0", "1"})
            self.assertEqual(result["cases"]["0"]["neural_draws"], 512)
            # Metadata and numerical identity are checked independently of a
            # valid replay's own file hashes.
            record["source"]["source_case_panel_sha256"] = "d" * 64
            report_path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "case-panel"):
                comparison.run(replay, directory, root / "invalid-source.json")
            record["source"].pop("source_case_panel_sha256")
            report_path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "case-panel"):
                comparison.run(replay, directory, root / "missing-source.json")
            record["source"]["source_case_panel_sha256"] = source_case_hash
            record["cases"]["0"]["full_observations_sha256"] = "e" * 64
            report_path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "observations"):
                comparison.run(replay, directory, root / "invalid-observations.json")
            record["cases"]["0"]["full_observations_sha256"] = teaching_reference.observations_sha256(native["observations"][0, :, 0])
            report_path.write_text(json.dumps(record))
            payload = directory / record["cases"]["0"]["lengths"]["1"]["runs"][0]["payload"]["file"]
            payload.write_bytes(payload.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "hash/path"):
                comparison.run(replay, directory, root / "invalid-payload.json")


if __name__ == "__main__":
    unittest.main()
