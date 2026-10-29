"""Target and input-contract checks without requiring MCMC convergence in CI."""

import unittest
from unittest.mock import patch
from types import SimpleNamespace
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from sbi4finance.calibration import reference, tasks
from scripts.calibration_reference import PartialDrawWriter


class HestonReferenceTargetTests(unittest.TestCase):
    def test_gaussian_likelihood_and_prior_support(self):
        task = tasks.heston_task()
        quotes = np.zeros(63)
        theta = np.mean(task["prior_bounds"], axis=1)
        with patch.object(tasks, "price", return_value=(quotes, None, None, None, None)):
            likelihood = reference.heston_log_likelihood(task, quotes)
            expected = -0.5 * np.log(2 * np.pi * tasks.sigma(task)**2).sum()
            self.assertAlmostEqual(likelihood(theta), expected)
            outside = theta.copy()
            outside[0] = task["prior_bounds"][0][0] - 1
            self.assertEqual(likelihood(outside), -np.inf)

    def test_incomplete_quotes_and_conditional_rough_are_rejected(self):
        masked = np.ma.array(np.zeros(63), mask=np.arange(63) == 0)
        for quotes in (np.zeros(62), np.full(63, np.nan), np.zeros((1, 63, 2)),
                       np.full(63, 1j), masked):
            with self.assertRaises(ValueError):
                reference.heston_log_likelihood(tasks.heston_task(), quotes)
        with self.assertRaises(ValueError):
            reference.heston_log_likelihood(tasks.rough_task(), np.zeros(63))

    def test_pricing_failure_is_not_zero_likelihood(self):
        task = tasks.heston_task()
        likelihood = reference.heston_log_likelihood(task, np.zeros(63))
        with patch.object(tasks, "price", side_effect=ValueError("precision failed")):
            with self.assertRaisesRegex(ValueError, "precision failed"):
                likelihood(np.mean(task["prior_bounds"], axis=1))

    def test_random_log_likelihood_is_rejected_before_sampling(self):
        counter = iter(range(10))
        with self.assertRaisesRegex(ValueError, "deterministic"):
            reference.sample_deterministic_reference(lambda _: float(next(counter)),
                                                       [[0, 1]], ["x"], draws=2, tune=2, chains=2)

    def test_prior_initialization_and_native_chain_seeds_are_recorded(self):
        first = reference.reference_initialization([[-4, 4]], 4, 271026)
        self.assertEqual(first, reference.reference_initialization([[-4, 4]], 4, 271026))
        self.assertEqual([c["spawn_key"] for c in first["chains"]], [[0], [1], [2], [3]])
        starts = np.asarray(first["prior_starts"])
        self.assertEqual(starts.shape, (4, 1))
        self.assertTrue(np.all((starts > -4) & (starts < 4)))
        self.assertEqual(len(set(c["model_initialization_seed"] for c in first["chains"])), 4)

    def test_parallelism_respects_allocation_and_rejects_fork(self):
        with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "2"}):
            reference.validate_parallelism(2, 4, "spawn")
            for cores, context in ((3, "spawn"), (0, "spawn"), (1, "fork")):
                with self.assertRaises(ValueError):
                    reference.validate_parallelism(cores, 4, context)


class PartialDrawTests(unittest.TestCase):
    def test_preallocated_tail_is_excluded_and_warmup_is_marked(self):
        class Trace:
            def __len__(self):
                return 3

            def get_values(self, name):
                self.name = name
                return np.array([[0.1], [0.2], [0.3], [np.nan], [np.nan]])

        with tempfile.TemporaryDirectory() as temporary:
            writer = PartialDrawWriter(Path(temporary) / "partial", chains=2, tune=2, draws=3, every=3)
            writer(trace=Trace(), draw=SimpleNamespace(chain=1, draw_idx=2))
            with np.load(writer.path / "chain-001.npz", allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays["parameters"], [[0.1], [0.2], [0.3]])
                np.testing.assert_array_equal(arrays["tuning"], [True, True, False])
                np.testing.assert_array_equal(arrays["draw_idx"], [0, 1, 2])
            progress = json.loads((writer.path / "progress.json").read_text())
            self.assertEqual(progress["posterior_draw_counts"], [0, 1])
            self.assertFalse(progress["sampler_resumable"])
            self.assertFalse(progress["scientific_acceptance"])

    def test_callback_rejects_inconsistent_draw_indexing(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = PartialDrawWriter(Path(temporary) / "partial", chains=2, tune=2, draws=3)
            with self.assertRaisesRegex(ValueError, "indexing"):
                writer(trace=[], draw=SimpleNamespace(chain=0, draw_idx=0))


if __name__ == "__main__":
    unittest.main()
