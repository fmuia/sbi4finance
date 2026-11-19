"""Independent checks of the exact Heston forward-start call and volatility swap."""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.special import ndtr, ndtri
from scipy.stats import ncx2

from sbi4f_tutorial.heston_options import model
from sbi4finance.calibration import heston_exact_products as exact

JUMPS = dict(jump_intensity=0., jump_up_probability=.5, jump_up_rate=10., jump_down_rate=10.)
# The market-like simulation's generating values (v0, kappa, theta, vol_of_vol, rho), rounded.
MARKET_LIKE = (0.02708, 4.3356, 0.05652, 1.7993, -0.6675)


def points(price, tenor=.25):
    """Black-equivalent volatility points of an undiscounted at-the-money call in forward units."""
    return 100 * 2 / np.sqrt(tenor) * ndtri((1 + price) / 2)


def forward_start(v0, kappa, theta, vol_of_vol, rho, reset=.25, expiry=.5):
    price, _ = exact.forward_start_call(dict(kappa=kappa, theta=theta, vol_of_vol=vol_of_vol, rho=rho),
                                        v0, reset, expiry)
    return float(price)


def volatility_swap(v0, kappa, theta, vol_of_vol, maturity=.25):
    value, _ = exact.volatility_swap(dict(kappa=kappa, theta=theta, vol_of_vol=vol_of_vol), v0, maturity)
    return float(value)


def quantile_average(v0, kappa, theta, vol_of_vol, rho, count, reset=.25, tenor=.25):
    """Average the tutorial runtime's Heston call over midpoint quantiles of the exact law of V(reset)."""
    scale = vol_of_vol**2 * -np.expm1(-kappa * reset) / (4 * kappa)
    law = ncx2(4 * kappa * theta / vol_of_vol**2, v0 * np.exp(-kappa * reset) / scale, scale=scale)
    variances = law.ppf((np.arange(count) + .5) / count)
    parameters = dict(kappa=kappa, theta=theta, vol_of_vol=vol_of_vol, rho=rho, **JUMPS)
    return float(np.mean([model.european_price(1., 1., 1., tenor, parameters, float(v),
                                               upper_limit=1500., nodes=2048) for v in variances]))


class HestonExponentTests(unittest.TestCase):
    def test_exponent_reproduces_the_tutorial_characteristic_function(self):
        rng = np.random.default_rng(20261119)
        sets = [MARKET_LIKE[1:], (0.1, 0.01, 3., -0.95), (10., 0.16, 0.1, -0.1)]
        sets += [tuple(rng.uniform([0.1, 0.01, 0.1, -0.95], [10., 0.16, 3., -0.1])) for _ in range(5)]
        frequencies = np.r_[0.01, 0.7, 3., 25., 140., 900.]
        for kappa, theta, vol_of_vol, rho in sets:
            parameters = dict(kappa=kappa, theta=theta, vol_of_vol=vol_of_vol, rho=rho, **JUMPS)
            for u in (frequencies - 0.5j, frequencies + 0j):
                for tenor in (.02, .25, 1.):
                    a, b = exact.heston_exponent(u, tenor, kappa, theta, vol_of_vol, rho)
                    for v0 in (0., 0.02, 0.15):
                        with self.subTest(parameters=(kappa, theta, vol_of_vol, rho), tenor=tenor, v0=v0):
                            expected = model.characteristic_function(u, tenor, parameters, v0)
                            assert_allclose(np.exp(a + b * v0), expected, rtol=1e-11, atol=1e-300)

    def test_cir_mgf_has_the_transition_mean_and_variance(self):
        v0, kappa, theta, vol_of_vol, horizon = .03, 2.5, .05, 1.2, .25
        mean = theta + (v0 - theta) * np.exp(-kappa * horizon)
        variance = (v0 * vol_of_vol**2 / kappa * (np.exp(-kappa * horizon) - np.exp(-2 * kappa * horizon))
                    + theta * vol_of_vol**2 / (2 * kappa) * (1 - np.exp(-kappa * horizon))**2)
        step = 1e-20
        slope = np.imag(exact.cir_log_mgf(1j * step, horizon, v0, kappa, theta, vol_of_vol)) / step
        self.assertAlmostEqual(slope, mean, delta=1e-14)
        h = 1e-4  # curvature from central differences of the real log-MGF
        values = [exact.cir_log_mgf(z, horizon, v0, kappa, theta, vol_of_vol) for z in (-h, 0., h)]
        self.assertAlmostEqual((values[0] - 2 * values[1] + values[2]) / h**2, variance, delta=1e-8)

    def test_cir_mgf_matches_direct_integration_over_the_transition_density(self):
        v0, kappa, theta, vol_of_vol, horizon = .04, 3., .06, .3, .25  # Feller holds: bounded density
        scale = vol_of_vol**2 * -np.expm1(-kappa * horizon) / (4 * kappa)
        law = ncx2(4 * kappa * theta / vol_of_vol**2, v0 * np.exp(-kappa * horizon) / scale, scale=scale)
        for z in (-3. + 0j, -20. + 15j):
            real = quad(lambda v: np.real(np.exp(z * v)) * law.pdf(v), 0, np.inf, epsabs=1e-14, epsrel=1e-12)[0]
            imag = quad(lambda v: np.imag(np.exp(z * v)) * law.pdf(v), 0, np.inf, epsabs=1e-14, epsrel=1e-12)[0]
            with self.subTest(z=z):
                assert_allclose(np.exp(exact.cir_log_mgf(z, horizon, v0, kappa, theta, vol_of_vol)),
                                real + 1j * imag, rtol=1e-9)


class ForwardStartTests(unittest.TestCase):
    def test_matches_the_conditional_call_averaged_over_the_exact_variance_law(self):
        # The reference never uses heston_exponent or the MGF: it prices each conditional call with
        # the tutorial runtime and averages over quantiles of scipy's noncentral chi-square. Its
        # midpoint error falls like 1/count; Richardson extrapolation removes most of it.
        for case in (MARKET_LIKE, (0.04, 2., 0.05, 0.3, -0.7)):
            coarse, fine = (points(quantile_average(*case, count)) for count in (500, 2000))
            extrapolated = (4 * fine - coarse) / 3
            with self.subTest(case=case):
                self.assertLess(abs(points(forward_start(*case)) - extrapolated), 1e-3)
                self.assertLess(abs(points(forward_start(*case)) - fine), 3e-3)

    def test_a_reset_today_is_the_tutorial_runtime_vanilla_call(self):
        v0, kappa, theta, vol_of_vol, rho = MARKET_LIKE
        parameters = dict(kappa=kappa, theta=theta, vol_of_vol=vol_of_vol, rho=rho, **JUMPS)
        vanilla = model.european_price(1., 1., 1., .25, parameters, v0, upper_limit=1500., nodes=4096)
        # The runtime's single Gauss-Legendre interval is accurate to about 1e-10 here (a piecewise
        # integration of its own integrand agrees with this module to 1e-12); 1e-9 is 5e-7 points.
        self.assertAlmostEqual(forward_start(*MARKET_LIKE, reset=0., expiry=.25), float(vanilla), delta=1e-9)

    def test_deterministic_variance_gives_the_black_price(self):
        v0, kappa, theta, rho, reset, expiry = .04, 2., .06, -.7, .25, .5
        total = theta * (expiry - reset) + (v0 - theta) * (np.exp(-kappa * reset) - np.exp(-kappa * expiry)) / kappa
        black = 2 * ndtr(np.sqrt(total) / 2) - 1
        # The leading correction is first order in rho * vol_of_vol.
        self.assertLess(abs(forward_start(v0, kappa, theta, 1e-5, rho, reset, expiry) / black - 1), 1e-6)
        self.assertLess(abs(forward_start(v0, kappa, theta, 1e-3, 0., reset, expiry) / black - 1), 1e-6)

    def test_batches_match_single_parameter_sets_across_chunks(self):
        rng = np.random.default_rng(7)
        draws = rng.uniform([0.01, 0.1, 0.01, 0.1, -0.95], [0.16, 10., 0.16, 3., -0.1], size=(600, 5))
        parameters = dict(kappa=draws[:, 1], theta=draws[:, 2], vol_of_vol=draws[:, 3], rho=draws[:, 4])
        prices, changes = exact.forward_start_call(parameters, draws[:, 0], .25, .5)
        self.assertEqual(prices.shape, (600,))
        self.assertLess(changes.max(), 1e-7)
        for i in (0, 511, 512, 599):
            assert_allclose(prices[i], forward_start(*draws[i]), rtol=1e-13, atol=0)

    def test_invalid_contracts_and_parameters_are_rejected(self):
        good = dict(kappa=2., theta=.05, vol_of_vol=1., rho=-.5)
        for reset, expiry in ((-.1, .5), (.5, .5), (.5, .25)):
            with self.subTest(reset=reset, expiry=expiry), self.assertRaises(ValueError):
                exact.forward_start_call(good, .04, reset, expiry)
        for key, value in (("kappa", 0.), ("theta", -.01), ("vol_of_vol", 0.), ("rho", -1.), ("rho", np.nan)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                exact.forward_start_call({**good, key: value}, .04, .25, .5)
        with self.assertRaises(ValueError):
            exact.forward_start_call(good, -.01, .25, .5)
        with self.assertRaises(TypeError):
            exact.forward_start_call([2., .05, 1., -.5], .04, .25, .5)


class VolatilitySwapTests(unittest.TestCase):
    def test_matches_adaptive_integration_of_the_laplace_transform(self):
        for v0, kappa, theta, vol_of_vol in (MARKET_LIKE[:4], (0.01, 0.1, 0.01, 3.), (0.16, 10., 0.16, 0.1)):
            def integrand(x):
                return -np.expm1(exact.integrated_variance_log_laplace(x * x / .25, .25, v0, kappa, theta,
                                                                       vol_of_vol)) / (x * x)
            edges = [0, 1, 10, 100, 1e3, 1e4, 1e5, 1e6]
            total = sum(quad(integrand, lo, hi, limit=500, epsabs=1e-15, epsrel=1e-13)[0]
                        for lo, hi in zip(edges[:-1], edges[1:]))
            total += 1 / edges[-1]  # the transform has vanished beyond the last edge
            with self.subTest(parameters=(v0, kappa, theta, vol_of_vol)):
                self.assertAlmostEqual(volatility_swap(v0, kappa, theta, vol_of_vol), total / np.sqrt(np.pi),
                                       delta=1e-12)

    def test_low_vol_of_vol_matches_the_second_order_expansion(self):
        # E sqrt(X) = sqrt(m) - Var(X) / (8 m^1.5) + O(vol_of_vol^4), with the exact variance of
        # integrated CIR variance. This checks the vol_of_vol dependence of the transform.
        v0, kappa, theta, maturity = .04, 2., .06, .25
        mean = theta + (v0 - theta) * -np.expm1(-kappa * maturity) / (kappa * maturity)
        self.assertLess(abs(volatility_swap(v0, kappa, theta, 1e-3) / np.sqrt(mean) - 1), 1e-6)
        vol_of_vol = .05

        def variance_at(s):
            return (v0 * vol_of_vol**2 / kappa * (np.exp(-kappa * s) - np.exp(-2 * kappa * s))
                    + theta * vol_of_vol**2 / (2 * kappa) * (1 - np.exp(-kappa * s))**2)
        variance = 2 * quad(lambda s: variance_at(s) * -np.expm1(-kappa * (maturity - s)) / kappa, 0, maturity,
                            epsabs=1e-16, epsrel=1e-13)[0] / maturity**2
        correction = variance / (8 * mean**1.5)
        residual = volatility_swap(v0, kappa, theta, vol_of_vol) - (np.sqrt(mean) - correction)
        self.assertLess(abs(residual), 1e-3 * correction)

    def test_batches_shapes_and_validation(self):
        values, changes = exact.volatility_swap(dict(kappa=[2., 5.], theta=.05, vol_of_vol=[1., 2.]),
                                                [.03, .02], .25)
        self.assertEqual(values.shape, (2,))
        self.assertLess(changes.max(), 1e-10)
        assert_allclose(values[1], volatility_swap(.02, 5., .05, 2.), rtol=1e-13, atol=0)
        self.assertEqual(np.shape(exact.volatility_swap(dict(kappa=2., theta=.05, vol_of_vol=1.), .03, .25)[0]), ())
        for v0, maturity in ((0., .25), (.03, 0.), (.03, -1.)):
            with self.subTest(v0=v0, maturity=maturity), self.assertRaises(ValueError):
                exact.volatility_swap(dict(kappa=2., theta=.05, vol_of_vol=1.), v0, maturity)


if __name__ == "__main__":
    unittest.main()
