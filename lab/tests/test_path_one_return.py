"""One-return rough reference agrees with independent scalar integration."""

import unittest

import numpy as np
from scipy.integrate import quad

from sbi4finance.path_inference import bank, one_return


class OneReturnTests(unittest.TestCase):
    def setUp(self):
        self.task = bank.default_task("rough-joint")
        self.returns = np.array([0.0, .01, -.02, .04])

    def test_scalar_moments_agree_with_independent_adaptive_quadrature(self):
        result = one_return.ShortVariancePosterior(self.returns, self.task, nodes=128)
        mean, sd = result.moments()
        dt, mu = self.task["dt_years"], self.task["fixed"]["drift"]
        for index, observed in enumerate(self.returns):
            def integrand(log_variance, power):
                variance = np.exp(log_variance)
                loglike = -.5 * (np.log(2 * np.pi * variance * dt)
                                 + (observed - (mu - .5 * variance) * dt) ** 2 / (variance * dt))
                return np.exp(loglike) * variance ** power
            values = [quad(integrand, result.a, result.b, args=(power,), epsabs=1e-13, epsrel=1e-12)[0]
                      for power in (0, 1, 2)]
            exact_mean = values[1] / values[0]
            exact_sd = np.sqrt(values[2] / values[0] - exact_mean ** 2)
            self.assertAlmostEqual(mean[index], exact_mean, places=13)
            self.assertAlmostEqual(sd[index], exact_sd, places=12)

    def test_quantiles_and_cdf_are_consistent(self):
        result = one_return.ShortVariancePosterior(self.returns, self.task, nodes=128)
        for probability in (.05, .5, .95):
            np.testing.assert_allclose(result.cdf(result.quantile(probability)), probability, atol=2e-11)
        np.testing.assert_array_equal(result.cdf(np.full(4, result.low)), 0)
        np.testing.assert_allclose(result.cdf(np.full(4, result.high)), 1, atol=1e-13)

    def test_only_short_variance_updates_after_one_return(self):
        reference, scalar = one_return.reference(self.returns, self.task, nodes=128)
        for index, name in enumerate(self.task["parameter_names"]):
            if name == "short_variance":
                self.assertGreater(np.ptp(reference["mean"][:, index]), .01)
            else:
                np.testing.assert_array_equal(reference["mean"][:, index], reference["mean"][0, index])
                np.testing.assert_array_equal(reference["sd"][:, index], reference["sd"][0, index])
        self.assertEqual(scalar.index, 3)

    def test_wrong_task_is_rejected(self):
        with self.assertRaises(ValueError):
            one_return.reference(self.returns, bank.default_task("rough-known"))


if __name__ == "__main__":
    unittest.main()
