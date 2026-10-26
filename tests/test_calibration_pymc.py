"""Target and input-contract checks without requiring MCMC convergence in CI."""

import unittest
from unittest.mock import patch

import numpy as np

from sbi4finance.calibration import reference, tasks


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


if __name__ == "__main__":
    unittest.main()
