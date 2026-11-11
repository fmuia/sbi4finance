"""Independent mathematical oracles for the rough-pricing optimization.

Expected drivers use explicit causal sums and analytic cell coefficients, never
the production FFT or weight helper. The deterministic pricing limit uses an
independent scalar Black formula with the discrete left-point variance integral.
"""
import importlib.util
import math
import unittest

import numpy as np

from sbi4f_tutorial.rough_options import _pricing_acceleration as acceleration
from sbi4f_tutorial.rough_options.simulator import rough_driver
from sbi4f_tutorial.shared.pricing import rough_block


def direct_hybrid(hurst, dt, z):
    """Small direct Gaussian-grid oracle from the hybrid cell integrals."""
    paths, steps, _ = z.shape
    coefficients = [math.sqrt(2*hurst)*dt**hurst
                    * ((lag+1)**(hurst+.5)-lag**(hurst+.5))/(hurst+.5)
                    for lag in range(steps)]
    bridge = dt**hurst*math.sqrt(max(1-2*hurst/(hurst+.5)**2, 0))
    variance = np.array([math.fsum(w*w for w in coefficients[:step+1])+bridge**2
                         for step in range(steps)])
    driver = np.empty((paths, steps))
    for path in range(paths):
        for step in range(steps):
            driver[path, step] = math.fsum(coefficients[step-cell]*float(z[path, cell, 0])
                                          for cell in range(step+1))+bridge*float(z[path, step, 1])
    return driver, variance


class RoughDriverOracleTests(unittest.TestCase):
    def test_fft_matches_explicit_causal_sum_and_analytic_variance(self):
        for hurst in (.025, .12, .49):
            for steps in (1, 7, 17, 63):
                with self.subTest(hurst=hurst, steps=steps):
                    dt = .75/steps
                    z = np.random.default_rng(101+steps).standard_normal((3, steps, 3))
                    expected = direct_hybrid(hurst, dt, z)
                    found = rough_driver(hurst, dt, z)
                    for actual, oracle in zip(found, expected):
                        np.testing.assert_allclose(actual, oracle, rtol=2e-13, atol=2e-14)

    def test_antithetic_sign_identity_at_actual_production_grid_lengths(self):
        for steps in (622, 1241, 664, 1327, 1791, 950, 1796):
            z = np.random.default_rng(20261111+steps).standard_normal((3, steps, 3))
            for hurst in (.025, .12, .35, .499999):
                with self.subTest(steps=steps, hurst=hurst):
                    positive, variance = rough_driver(hurst, .75/steps, z)
                    negative, other_variance = rough_driver(hurst, .75/steps, -z)
                    np.testing.assert_allclose(negative, -positive, rtol=0, atol=2e-14)
                    np.testing.assert_array_equal(other_variance, variance)


@unittest.skipUnless(importlib.util.find_spec("numba"), "optional numba acceleration is not installed")
class RoughPricingOracleTests(unittest.TestCase):
    def test_accelerated_integrals_match_direct_scalar_model(self):
        steps, dt = 31, .75/31
        indices = [1, 2, 7, 16, 31]
        curve = {"times": [0., .75], "values": [.02, .05]}
        z = np.random.default_rng(812).standard_normal((4, steps, 3))
        for hurst, eta, rho in ((.025, 3.4, -.98), (.12, 1.5, -.7), (.34, .51, -.21)):
            with self.subTest(hurst=hurst, eta=eta, rho=rho):
                driver, driver_variance = direct_hybrid(hurst, dt, z)
                found = acceleration.antithetic_terminal_paths(
                    {"hurst": hurst, "eta": eta, "rho": rho}, curve, dt, z, indices)
                for side, sign in enumerate((1., -1.)):
                    expected_iv, expected_iw = [], []
                    for path in range(len(z)):
                        iv, iw, selected_iv, selected_iw = 0., 0., [], []
                        for step in range(steps):
                            y = 0. if step == 0 else sign*driver[path, step-1]
                            q = 0. if step == 0 else driver_variance[step-1]
                            xi = .02+.03*step/steps
                            v = xi*math.exp(eta*y-.5*eta**2*q)
                            iv += v*dt
                            iw += math.sqrt(v*dt)*sign*float(z[path, step, 0])
                            if step+1 in indices:
                                selected_iv.append(iv); selected_iw.append(iw)
                        expected_iv.append(selected_iv); expected_iw.append(selected_iw)
                    np.testing.assert_allclose(found[side]["integrated_variance"], expected_iv, rtol=3e-12, atol=2e-14)
                    np.testing.assert_allclose(found[side]["volatility_driver_integral"], expected_iw, rtol=3e-12, atol=2e-14)

    def test_zero_eta_prices_match_independent_black_formula(self):
        steps, dt = 32, .5/32
        curve = {"times": [0., .5], "values": [.03, .09]}
        contracts, expected = [], []
        normal_cdf = lambda x: .5*math.erfc(-x/math.sqrt(2))
        for index in (16, 32):
            # Exact left-grid sum of xi_j=.03+.06*j/steps, not its
            # continuous integral: the benchmark must preserve this grid law.
            integrated = dt*(index*.03+.06/steps*index*(index-1)/2)
            sd = math.sqrt(integrated)
            for strike in (.7, .9, 1., 1.1, 1.3):
                d1 = (-math.log(strike)+.5*integrated)/sd
                d2 = d1-sd
                for kind in ("put", "call"):
                    contracts.append({"maturity": index*dt, "strike": strike, "option_type": kind})
                    expected.append(strike*normal_cdf(-d2)-normal_cdf(-d1) if kind == "put"
                                    else normal_cdf(d1)-strike*normal_cdf(d2))
        for backend in ("reference", "numba"):
            with self.subTest(backend=backend):
                prices, errors, covariance = rough_block(
                    {"hurst": .12, "eta": 0., "rho": 0.}, curve, contracts,
                    horizon=.5, n_steps=steps, n_pairs=64, seed=95, path_backend=backend)
                np.testing.assert_allclose(prices, expected, rtol=0, atol=5e-15)
                self.assertLess(float(np.max(errors)), 5e-15)
                self.assertLess(float(np.max(np.abs(covariance))), 3e-29)


if __name__ == "__main__":
    unittest.main()
