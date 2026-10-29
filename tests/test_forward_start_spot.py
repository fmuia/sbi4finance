"""Independent payoff checks for the spot-notional forward-start contract."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from sbi4finance.calibration import products


class SpotForwardStartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A two-step stochastic-variance law: future variance depends on the
        # first variance shock. This makes start-notional weighting matter.
        cls.pairs = 2**16
        z = np.random.default_rng(280029).normal(size=(cls.pairs, 4))
        z = np.concatenate([z, -z])
        cls.rho = -.6
        first_i = np.full(len(z), .04 * .25)
        first_j = np.sqrt(first_i) * z[:, 0]
        next_variance = np.maximum(.04 + .1 * z[:, 0], .0001)
        next_i = .25 * next_variance
        next_j = np.sqrt(next_i) * z[:, 1]
        cls.paths = {
            "times": np.array([0., .25, .5]),
            "I": np.column_stack([np.zeros(len(z)), first_i, first_i + next_i]),
            "J": np.column_stack([np.zeros(len(z)), first_j, first_j + next_j]),
            "mean_I": np.array([0., .01, .02]), "mean_I_is_exact": False,
        }
        # Direct asset simulation uses independent orthogonal Brownian
        # increments, rather than the conditional Black-Scholes helper.
        start_asset = np.exp(cls.rho * first_j - .5 * first_i
                             + np.sqrt((1-cls.rho**2) * first_i) * z[:, 2])
        asset_ratio = np.exp(cls.rho * next_j - .5 * next_i
                             + np.sqrt((1-cls.rho**2) * next_i) * z[:, 3])
        payoff = start_asset * np.maximum(asset_ratio - 1., 0.)
        paired = .5 * (payoff[:cls.pairs] + payoff[cls.pairs:])
        cls.direct = paired.mean()
        cls.direct_se = paired.std(ddof=1) / np.sqrt(cls.pairs)

    def test_matches_direct_asset_difference_payoff(self):
        price, se = products.forward_start_call_spot(self.paths, self.rho, .25, .5)
        self.assertLess(abs(price-self.direct), 4*np.hypot(se, self.direct_se))
        ratio, _ = products.forward_start_call(self.paths, self.rho, .25, .5)
        self.assertGreater(abs(ratio-self.direct), 4*np.hypot(se, self.direct_se))

    def test_start_zero_is_the_same_contract_as_a_vanilla_call(self):
        actual = products.forward_start_call_spot(self.paths, self.rho, 0., .5, 1.05)
        expected = products.vanilla(self.paths, self.rho, 1.05, .5, "call")
        assert_allclose(actual, expected, rtol=0, atol=1e-14)

    def test_invalid_contract_is_rejected(self):
        for start, end, strike in [(-.25, .5, 1.), (.5, .25, 1.), (.25, .5, 0.)]:
            with self.subTest(start=start, end=end, strike=strike), self.assertRaises(ValueError):
                products.forward_start_call_spot(self.paths, self.rho, start, end, strike)


if __name__ == "__main__":
    unittest.main()
