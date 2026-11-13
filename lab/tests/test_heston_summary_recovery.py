"""Recovery exports must reproduce recorded raw-draw diagnostics without filtering."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import export_heston_summary_recovery as recovery


class SummaryRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(9104)
        bounds = recovery.BOUNDS
        cls.truth = rng.uniform(bounds[:, 0], bounds[:, 1], (512, 6))
        cls.draws = (cls.truth[:, None] + rng.normal(size=(512, 512, 6)) *
                     (.04 * np.ptp(bounds, axis=1))).astype(np.float32)
        cls.q = np.quantile(cls.draws, [.05, .5, .95], axis=1)
        lower, median, upper = cls.q
        outside = ((cls.draws < bounds[:, 0]) | (cls.draws > bounds[:, 1])).sum(axis=(0, 1))
        cls.compact = {"population": {"parameter_names": list(recovery.NAMES), "rows": 512,
                        "draws_per_case": 512, "raw_draws_used": 262144,
                        "truth_sha256": sha256(np.asarray(cls.truth, dtype="<f8").tobytes()).hexdigest(),
                        "support": {"draws_discarded": 0, "draws_clipped": 0, "draws_resampled": 0,
                                    "outside_prior_draws_retained_for_calibration": True,
                                    "outside_prior": {"per_dimension_count": outside.tolist()}}},
                       "parameters": {name: {
                           "rmse": float(np.sqrt(np.mean((median[:, j] - cls.truth[:, j]) ** 2))),
                           "coverage90": float(np.mean((cls.truth[:, j] >= lower[:, j]) & (cls.truth[:, j] <= upper[:, j]))),
                           "median_width90": float(np.median(upper[:, j] - lower[:, j]))}
                           for j, name in enumerate(recovery.NAMES)}}

    def test_raw_quantiles_and_metrics_include_outside_prior_draws(self):
        original = self.draws.copy()
        quantiles, metrics = recovery.summarize_draws(self.draws, self.truth, self.compact)
        self.assertTrue(np.array_equal(quantiles, self.q))
        self.assertTrue(np.array_equal(self.draws, original))
        self.assertGreater(sum(self.compact["population"]["support"]["outside_prior"]["per_dimension_count"]), 0)
        self.assertTrue(np.any(quantiles[0] < recovery.BOUNDS[:, 0]))
        self.assertEqual(metrics["coverage90"], [self.compact["parameters"][name]["coverage90"] for name in recovery.NAMES])

    def test_changed_truth_metrics_and_support_cannot_be_plotted(self):
        wrong_truth = self.truth.copy()
        wrong_truth[0, 0] += .01
        with self.assertRaisesRegex(ValueError, "truth identity"):
            recovery.summarize_draws(self.draws, wrong_truth, self.compact)
        for change in ("metric", "support", "filter"):
            compact = deepcopy(self.compact)
            if change == "metric":
                compact["parameters"]["c"]["rmse"] += .01
            elif change == "support":
                compact["population"]["support"]["outside_prior"]["per_dimension_count"][0] += 1
            else:
                compact["population"]["support"]["draws_clipped"] = 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                recovery.summarize_draws(self.draws, self.truth, compact)

    def test_source_payload_hash_is_checked_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "draws.npz"
            path.write_bytes(b"original private payload")
            expected = recovery.digest(path)
            self.assertEqual(recovery.checked_file(path, expected), path)
            path.write_bytes(b"changed private payload")
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                recovery.checked_file(path, expected)


if __name__ == "__main__":
    unittest.main()
