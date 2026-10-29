"""Marginal laws, exact OU nesting and paired price-error accounting."""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.special import ndtr

from sbi4finance.bergomi import conditional_put_payoffs, simulate_one_factor
from sbi4finance.bergomi_refinement import coarsen_innovations, coupled_option_panels
from sbi4finance.rough_bergomi import hybrid_covariance, rough_driver, simulate_rough


CURVE = {"times": [0., 1.], "values": [.04, .04]}
ONE = {"kappa": 2., "eta": 1., "rho": -.65}
ROUGH = {"hurst": .1, "eta": 1., "rho": -.65}
CONTRACTS = [{"strike": .95, "maturity": .5},
             {"strike": 1.05, "maturity": 1.},
             {"strike": 1.05, "maturity": 1., "option_type": "call"}]


class BergomiRefinementTests(unittest.TestCase):
    def test_linear_coupling_preserves_independent_standard_normal_channels(self):
        # The linear map applied to an orthonormal basis gives its exact Gram
        # matrix. No sampling tolerance can hide an incorrect coarse marginal.
        fine_steps, coarse_steps = 24, 6
        basis = np.eye(3 * fine_steps).reshape(3 * fine_steps, fine_steps, 3)
        for model in ("one_factor", "rough"):
            z = coarsen_innovations(basis, coarse_steps, model=model,
                                    dt_years=1 / fine_steps, kappa=2.)
            matrix = z.reshape(len(z), -1)
            assert_allclose(matrix.T @ matrix, np.eye(3 * coarse_steps), atol=2e-12)
            assert_allclose(z[..., 0], basis[..., 0].reshape(len(z), coarse_steps, 4).sum(axis=2)/2)
            assert_allclose(z[..., 2], basis[..., 2].reshape(len(z), coarse_steps, 4).sum(axis=2)/2)

    def test_exact_ou_endpoints_and_brownian_cross_covariance(self):
        fine_steps, coarse_steps, kappa = 32, 8, ONE["kappa"]
        basis = np.eye(3 * fine_steps).reshape(3 * fine_steps, fine_steps, 3)
        z = coarsen_innovations(basis, coarse_steps, model="one_factor",
                                dt_years=1 / fine_steps, kappa=kappa)
        fine = simulate_one_factor(ONE, CURVE, np.full(fine_steps, 1 / fine_steps), basis)
        coarse = simulate_one_factor(ONE, CURVE, np.full(coarse_steps, 1 / coarse_steps), z)
        assert_allclose(coarse["factor_state"], fine["factor_state"][:, ::4], atol=2e-15)
        dt = 1 / coarse_steps
        first_factor = coarse["factor_state"][:, 1, 0]
        first_dw = np.sqrt(dt) * z[:, 0, 0]
        assert_allclose(first_factor @ first_factor, -np.expm1(-2*kappa*dt)/(2*kappa), atol=1e-15)
        assert_allclose(first_factor @ first_dw, -np.expm1(-kappa*dt)/kappa, atol=1e-15)

    def test_rough_rebuild_has_coarse_discrete_covariance_and_own_normalizer(self):
        fine_steps, coarse_steps = 32, 8
        basis = np.eye(3 * fine_steps).reshape(3 * fine_steps, fine_steps, 3)
        z = coarsen_innovations(basis, coarse_steps, model="rough", dt_years=1 / fine_steps)
        driver, variance = rough_driver(ROUGH["hurst"], 1 / coarse_steps, z)
        expected, cross = hybrid_covariance(ROUGH["hurst"], 1 / coarse_steps, coarse_steps)
        assert_allclose(driver.T @ driver, expected, atol=2e-15)
        assert_allclose(driver.T @ (z[..., 0] / np.sqrt(coarse_steps)), cross, atol=2e-15)
        assert_allclose(variance, expected.diagonal(), atol=2e-15)
        fine_driver, fine_variance = rough_driver(ROUGH["hurst"], 1 / fine_steps, basis)
        self.assertGreater(np.max(np.abs(fine_driver[:, 3::4] - driver)), .01)
        self.assertGreater(np.max(np.abs(fine_variance[3::4] - variance)), 1e-5)

    def test_rough_near_brownian_limit_has_nested_driver_endpoints(self):
        fine = np.random.default_rng(733).standard_normal((9, 32, 3))
        coarse = coarsen_innovations(fine, 8, model="rough", dt_years=1/32)
        fine_y, _ = rough_driver(.499999, 1/32, fine)
        coarse_y, _ = rough_driver(.499999, 1/8, coarse)
        assert_allclose(coarse_y, fine_y[:, 3::4], atol=8e-6)

    def test_full_covariance_and_paired_errors_match_direct_payoff_differences(self):
        grids, n_paths, seed = (4, 8, 16), 321, 715
        finest = np.random.default_rng(seed).standard_normal((n_paths, grids[-1], 3))
        for model, p in (("one_factor", ONE), ("rough", ROUGH)):
            direct = []
            for steps in grids:
                z = coarsen_innovations(finest, steps, model=model,
                                        dt_years=1/grids[-1], kappa=p.get("kappa"))
                if model == "one_factor":
                    paths = simulate_one_factor(p, CURVE, np.full(steps, 1/steps), z)
                else:
                    paths = simulate_rough(p, CURVE, 1/steps, z)
                direct.append(conditional_put_payoffs(paths, p, CONTRACTS,
                              [steps//2, steps, steps], spot=1., rate=0., dividend=0.))
            result = coupled_option_panels(model, p, CURVE, CONTRACTS,
                     n_paths=n_paths, steps=grids, seed=seed, chunk_size=71)
            expected_cov = np.cov(np.concatenate(direct, axis=1), rowvar=False, ddof=1)/n_paths
            assert_allclose(result["estimator_covariance"].reshape(9, 9), expected_cov, atol=1e-18)
            for name, pairs in (("adjacent", [(0, 1), (1, 2)]), ("to_finest", [(0, 2), (1, 2)])):
                differences = np.concatenate([direct[b]-direct[a] for a, b in pairs], axis=1)
                expected_diff_cov = np.cov(differences, rowvar=False, ddof=1)/n_paths
                observed = result[name]
                assert_allclose(observed["differences"], differences.mean(axis=0).reshape(2, 3), atol=1e-17)
                assert_allclose(observed["estimator_covariance"].reshape(6, 6), expected_diff_cov, atol=1e-20)
                assert_allclose(observed["paired_standard_errors"]**2,
                                np.diag(expected_diff_cov).reshape(2, 3), atol=1e-20)
                for row, (a, b) in enumerate(pairs):
                    joint = result["estimator_covariance"]
                    independent = np.diag(joint[a, :, a, :] + joint[b, :, b, :])
                    paired = independent - 2*np.diag(joint[a, :, b, :])
                    assert_allclose(observed["independent_standard_errors"][row]**2, independent, atol=1e-20)
                    assert_allclose(observed["paired_standard_errors"][row]**2, paired, atol=1e-19)
                    self.assertTrue(np.all(np.diag(joint[a, :, b, :]) > 0))
                    assert_allclose(observed["variance_reduction"][row], independent/paired, rtol=1e-12)
                    self.assertTrue(np.all(observed["variance_reduction"][row] > 1))
            self.assertGreaterEqual(np.linalg.eigvalsh(expected_cov).min(), -1e-18)

    def test_chunk_invariance_and_call_parity_on_each_grid(self):
        for model, p in (("one_factor", ONE), ("rough", ROUGH)):
            a = coupled_option_panels(model, p, CURVE, CONTRACTS, steps=(4, 8, 16),
                                      n_paths=513, seed=729, chunk_size=31, rate=.02, dividend=.01)
            b = coupled_option_panels(model, p, CURVE, CONTRACTS, steps=(4, 8, 16),
                                      n_paths=513, seed=729, chunk_size=256, rate=.02, dividend=.01)
            assert_allclose(a["prices"], b["prices"], atol=1e-15)
            assert_allclose(a["estimator_covariance"], b["estimator_covariance"], atol=1e-19)
            for key in ("adjacent", "to_finest"):
                assert_allclose(a[key]["differences"], b[key]["differences"], atol=1e-17)
                assert_allclose(a[key]["paired_standard_errors"], b[key]["paired_standard_errors"], atol=1e-18)
            assert_allclose(a["prices"][:, 2]-a["prices"][:, 1], np.exp(-.01)-1.05*np.exp(-.02), atol=1e-15)
            assert_allclose(a["estimator_covariance"][:, 1], a["estimator_covariance"][:, 2], atol=1e-20)

    def test_flat_deterministic_variance_is_black_scholes_with_zero_refinement(self):
        maturity, strike, rate, dividend = 1., 1.05, .02, .01
        d1 = (np.log(1/strike) + (rate-dividend+.02)*maturity)/np.sqrt(.04*maturity)
        expected = strike*np.exp(-rate*maturity)*ndtr(-d1+.2) - np.exp(-dividend*maturity)*ndtr(-d1)
        for model, p in (("one_factor", ONE), ("rough", ROUGH)):
            result = coupled_option_panels(model, dict(p, eta=0., rho=0.), CURVE,
                     [{"strike": strike, "maturity": maturity}], steps=(4, 8, 16),
                     n_paths=35, seed=20, chunk_size=9, rate=rate, dividend=dividend)
            assert_allclose(result["prices"], expected, atol=1e-15)
            self.assertLess(result["standard_errors"].max(), 1e-15)
            self.assertLess(np.max(abs(result["adjacent"]["differences"])), 1e-15)
            self.assertLess(result["adjacent"]["paired_standard_errors"].max(), 1e-15)
            self.assertFalse(result["metadata"]["mc_error_includes_discretization_bias"])
            self.assertFalse(result["metadata"]["quote_error_included"])

    def test_rejects_invalid_grids_context_domains_and_unresolved_ou_bridge(self):
        for grids in ((8,), (8, 8), (16, 8), (6, 16), (4., 8), (False, 8)):
            with self.subTest(grids=grids), self.assertRaises(ValueError):
                coupled_option_panels("one_factor", ONE, CURVE, CONTRACTS,
                                      steps=grids, n_paths=8)
        for changes in ({"n_paths": 1}, {"chunk_size": 0}, {"spot": 0}):
            with self.assertRaises(ValueError):
                coupled_option_panels("rough", ROUGH, CURVE, CONTRACTS, steps=(4, 8), **changes)
        with self.assertRaises(ValueError):
            coupled_option_panels("rough", ROUGH, CURVE,
                [{"strike": 1., "maturity": .3}, {"strike": 1., "maturity": 1.}], steps=(4, 8), n_paths=8)
        z = np.zeros((2, 16, 3))
        for changes in ({"n_steps": 7}, {"model": "wrong"}, {"dt_years": 0}, {"kappa": 0}):
            args = {"n_steps": 4, "model": "one_factor", "dt_years": 1/16, "kappa": 2., **changes}
            with self.assertRaises(ValueError):
                coarsen_innovations(z, **args)
        with self.assertRaises(ValueError):
            coarsen_innovations(z, 4, model="one_factor", dt_years=1/16, kappa=1e-20)


if __name__ == "__main__":
    unittest.main()
