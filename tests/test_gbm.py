"""Independent scientific references for the first synthetic GBM workflow."""

import unittest
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from sbi4finance.gbm import (
    drift_log_likelihood,
    drift_posterior,
    european_price,
    predictive_log_return,
    prices_from_log_returns,
    simulate_log_returns,
)


class GBMReferenceTests(unittest.TestCase):
    def test_simulation_moments_independence_and_continuation(self):
        dt = np.array([0.125, 0.375, 0.5])
        count = 50_000
        returns = simulate_log_returns(
            np.full(count, 0.08), 0.2, dt, rng=np.random.default_rng(182)
        )
        self.assertEqual(returns.shape, (count, 3, 1))
        self.assertEqual(returns.dtype, np.dtype("float64"))
        samples = returns[:, :, 0]
        variances = 0.04 * dt
        # These sampling-error tolerances assess the law, not exact seeded values.
        self.assertTrue(
            np.all(np.abs(samples.mean(axis=0) - 0.06 * dt)
                   < 5.5 * np.sqrt(variances / count))
        )
        covariance = np.cov(samples, rowvar=False)
        tolerance = 5.5 * np.sqrt(
            (np.outer(variances, variances) + np.diag(variances**2))
            / (count - 1)
        )
        self.assertTrue(np.all(np.abs(covariance - np.diag(variances)) < tolerance))

        whole = simulate_log_returns(
            [0.03, 0.08], 0.2, dt, rng=np.random.default_rng(71)
        )
        stream = np.random.default_rng(71)
        first = simulate_log_returns([0.03, 0.08], 0.2, dt[:1], rng=stream)
        rest = simulate_log_returns([0.03, 0.08], 0.2, dt[1:], rng=stream)
        assert_array_equal(whole, np.concatenate([first, rest], axis=1))

        deterministic = simulate_log_returns(
            0.1, 0.0, [0.25, 0.75], rng=np.random.default_rng(9)
        )
        assert_allclose(deterministic, [[[0.025], [0.075]]])
        prices = prices_from_log_returns(deterministic, 100.0)
        self.assertEqual(prices.shape, (1, 3, 1))
        assert_allclose(prices[0, :, 0], [100.0, 102.5315120524429, 110.51709180756477])

    def test_likelihood_has_correct_density_and_batch_broadcast(self):
        history = np.array([[[0.115], [-0.055]]])
        result = drift_log_likelihood(history, 0.08, 0.2, [0.25, 0.75])
        self.assertEqual(result.shape, (1,))
        assert_allclose(result, [1.5513203085780245], rtol=0, atol=1e-13)
        candidates = np.array([-0.1, 0.0, 0.08])
        batch = drift_log_likelihood(history, candidates, 0.2, [0.25, 0.75])
        self.assertEqual(batch.shape, (3,))
        for index, candidate in enumerate(candidates):
            assert_allclose(
                batch[index],
                drift_log_likelihood(history, candidate, 0.2, [0.25, 0.75])[0],
            )

    def test_posterior_matches_worked_example_and_grid_integration(self):
        history = np.array([[[0.05], [-0.02]]])
        mean, std = drift_posterior(
            history, 0.2, [0.25, 0.75], prior_mean=0.0, prior_std=0.1
        )
        self.assertEqual(mean.shape, (1,))
        self.assertEqual(std.shape, (1,))
        assert_allclose(mean, [0.01], rtol=0, atol=1e-14)
        assert_allclose(std**2, [0.008], rtol=0, atol=1e-14)

        # Numerically integrate prior times likelihood without using the
        # conjugate posterior expression as the reference calculation.
        grid = np.linspace(-0.8, 0.8, 16_001)
        log_weight = (
            drift_log_likelihood(history, grid, 0.2, [0.25, 0.75])
            - 0.5 * (grid / 0.1) ** 2
        )
        weights = np.exp(log_weight - log_weight.max())
        weights /= weights.sum()
        grid_mean = np.dot(weights, grid)
        grid_variance = np.dot(weights, (grid - grid_mean) ** 2)
        assert_allclose(mean, [grid_mean], rtol=0, atol=1e-11)
        assert_allclose(std**2, [grid_variance], rtol=0, atol=1e-11)

        # An observed prefix may be a view into a stored path. Future values
        # outside that prefix must not enter conditioning or change the input.
        path = np.array([[[0.05], [-0.02], [3.0]]])
        prefix = path[:, :2, :]
        path[:, 2:, :] = -7.0
        after = drift_posterior(
            prefix, 0.2, [0.25, 0.75], prior_mean=0.0, prior_std=0.1
        )
        assert_array_equal(prefix, history)
        assert_allclose(after[0], mean)
        assert_allclose(after[1], std)

    def test_predictive_integrates_drift_uncertainty(self):
        mean, std = predictive_log_return(0.01, np.sqrt(0.008), 0.2, 0.25)
        self.assertEqual(mean.shape, (1,))
        self.assertEqual(std.shape, (1,))
        assert_allclose(mean, [-0.0025], rtol=0, atol=1e-14)
        assert_allclose(std**2, [0.0105], rtol=0, atol=1e-14)
        zero_mean, zero_std = predictive_log_return([0.01, 0.02], 0.1, 0.2, 0.0)
        assert_array_equal(zero_mean, [0.0, 0.0])
        assert_array_equal(zero_std, [0.0, 0.0])

    def test_european_prices_match_reference_parity_and_zero_variance(self):
        forward = 100.0 * np.exp(0.03)
        discount = np.exp(-0.05)
        call = european_price(forward, 100.0, discount, 0.04, option="call")
        put = european_price(forward, 100.0, discount, 0.04, option="put")
        self.assertIsInstance(call, np.ndarray)
        self.assertEqual(call.shape, ())
        # S=K=100, r=.05, q=.02, T=1, sigma=.2 Black--Scholes references.
        assert_allclose(call, 9.227005508154036, rtol=0, atol=2e-12)
        assert_allclose(put, 6.330080627549918, rtol=0, atol=2e-12)

        forwards = np.array([90.0, 100.0, 110.0])
        calls = european_price(forwards, 100.0, 0.95, 0.0, option="call")
        puts = european_price(forwards, 100.0, 0.95, 0.0, option="put")
        assert_allclose(calls, [0.0, 0.0, 9.5], rtol=0, atol=1e-13)
        assert_allclose(puts, [9.5, 0.0, 0.0], rtol=0, atol=1e-13)
        assert_allclose(calls - puts, 0.95 * (forwards - 100.0))

    def test_invalid_inputs_fail_explicitly(self):
        history = np.zeros((1, 2, 1))
        invalid_calls = {
            "scalar time grid": lambda: simulate_log_returns(0.0, 0.2, 1.0, rng=np.random.default_rng(1)),
            "nonpositive time step": lambda: simulate_log_returns(0.0, 0.2, [0.0], rng=np.random.default_rng(1)),
            "negative volatility": lambda: simulate_log_returns(0.0, -0.2, [1.0], rng=np.random.default_rng(1)),
            "non-generator RNG": lambda: simulate_log_returns(0.0, 0.2, [1.0], rng=1),
            "squeezed returns": lambda: drift_log_likelihood([0.0, 0.0], 0.0, 0.2, [0.5, 0.5]),
            "mismatched time grid": lambda: drift_log_likelihood(history, 0.0, 0.2, [1.0]),
            "degenerate likelihood": lambda: drift_log_likelihood(history, 0.0, 0.0, [0.5, 0.5]),
            "degenerate prior": lambda: drift_posterior(history, 0.2, [0.5, 0.5], prior_mean=0.0, prior_std=0.0),
            "negative horizon": lambda: predictive_log_return(0.0, 0.1, 0.2, -1.0),
            "nonpositive initial price": lambda: prices_from_log_returns(history, 0.0),
            "nonfinite forward": lambda: european_price(np.nan, 100.0, 1.0, 0.04),
            "negative integrated variance": lambda: european_price(100.0, 100.0, 1.0, -0.01),
            "unknown option type": lambda: european_price(100.0, 100.0, 1.0, 0.04, option="straddle"),
        }
        for name, call in invalid_calls.items():
            with self.subTest(name=name):
                with self.assertRaises((TypeError, ValueError)):
                    call()

    def test_demo_serializes_deterministic_and_unsampled_tail_prices(self):
        from scripts.gbm_reference import run

        config_path = Path(__file__).resolve().parents[1] / "configs/tasks/gbm_drift.json"
        config = json.loads(config_path.read_text())
        config["returns"]["forecast_draws"] = 20
        config["pricing"]["mc_draws"] = 20
        config["pricing"]["volatility"] = 0.0
        report, _ = run(config)
        prices = report["pricing_result"]
        json.dumps(report, allow_nan=False)
        assert_allclose(prices["mc_calls"], prices["reference_calls"], atol=1e-13)
        self.assertEqual(prices["mc_standard_errors"], [0.0, 0.0, 0.0])
        self.assertEqual(prices["error_in_standard_errors"], [None, None, None])

        config["pricing"]["volatility"] = 0.2
        config["pricing"]["strikes"] = [1e6]
        report, _ = run(config)
        json.dumps(report, allow_nan=False)
        self.assertEqual(report["pricing_result"]["error_in_standard_errors"], [None])


if __name__ == "__main__":
    unittest.main()
