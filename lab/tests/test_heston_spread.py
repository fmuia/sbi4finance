"""Likelihood and numerical contracts of the one-component Heston task."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from sbi4finance.calibration.heston_pilot import HestonPilot, black_price
from sbi4finance.calibration.heston_spread import (
    HestonSpreadPilot, conditional_c_log_density, conditional_c_mode,
)


THETA = np.array([.04, 2., .05, .4, -.7])


def records():
    f = np.array([100., 102., 105., 105.])
    k = f * np.array([.9, 1., 1.05, .95])
    d, t = np.array([.99, .98, .96, .96]), np.array([.1, .25, .75, .75])
    kind = np.array(["put", "call", "call", "put"])
    return dict(forward=f, strike=k, discount=d, maturity=t, option_type=kind,
                mid=black_price(f, k, d, t, [.25, .22, .18, .23], kind),
                half_spread=np.array([.03, .02, .04, .03]))


class HestonSpreadTests(unittest.TestCase):
    def test_arbitrary_finite_observations_need_no_iv_or_vega(self):
        data = records()
        data["mid"] = np.array([-2., 0., 1e4, 3.])
        data["market_iv"] = [np.nan] * 4
        data["market_vega"] = [-1.] * 4
        with patch("sbi4finance.calibration.heston_pilot.observed_black_vega",
                   side_effect=AssertionError("spread-only law must not invert IV")):
            pilot = HestonSpreadPilot(data)
            self.assertTrue(np.isfinite(pilot.log_likelihood(THETA)))
        self.assertNotIn("market_vega", pilot.records)
        data["mid"][0] = 99.
        self.assertEqual(pilot.records["mid"][0], -2.)
        self.assertFalse(pilot.records["mid"].flags.writeable)
        self.assertEqual(pilot.names, ["initial_variance", "kappa", "theta", "vol_of_vol", "rho"])

    def test_price_engine_is_reused_and_masks_preserve_order(self):
        pilot, old = HestonSpreadPilot(records()), HestonPilot(records(), spread_multiplier=.5)
        assert_allclose(pilot.prices(THETA), old.prices(THETA), rtol=0, atol=1e-12)
        selected = [3, 0]
        assert_allclose(pilot.prices(THETA, selected), pilot.prices(THETA)[selected], atol=1e-10)
        independent, error = pilot.independent_prices(THETA, selected)
        assert_allclose(independent, pilot.prices(THETA, selected), atol=1e-5)
        self.assertTrue(np.all(error < .02 * pilot.measurement_sigma[selected]))

    def test_fixed_and_inferred_c_normalized_likelihood_and_boundaries(self):
        fixed, joint = HestonSpreadPilot(records(), c=1.3), HestonSpreadPilot(records(), infer_c=True)
        clean = fixed.records["mid"] + [.01, -.02, .03, -.04]
        observed = np.array([-1., 2., 3., 4.])
        for c in (.5, 1.3, 2.):
            sigma = c * records()["half_spread"]
            expected = -.5 * np.sum(((observed-clean)/sigma)**2) - np.sum(np.log(sigma)) - 2*np.log(2*np.pi)
            with patch.object(joint, "prices", return_value=clean):
                self.assertAlmostEqual(joint.log_likelihood(np.r_[THETA, c], observations=observed), expected, places=10)
        with patch.object(fixed, "prices", return_value=clean), patch.object(joint, "prices", return_value=clean):
            self.assertAlmostEqual(fixed.log_likelihood(THETA, observations=observed),
                                   joint.log_likelihood(np.r_[THETA, 1.3], observations=observed))
        for c in (.49, 2.01):
            self.assertEqual(joint.log_likelihood(np.r_[THETA, c]), -np.inf)
        self.assertEqual(joint.names[-1], "c")
        assert_allclose(joint.bounds[-1], [.5, 2.])

    def test_scale_changes_concentration_not_fixed_scale_mle(self):
        data = records()
        slopes = np.array([2., 3., 5., 7.])
        data["mid"] = slopes * .07 + np.array([.01, -.02, .03, -.01])
        pilots = [HestonSpreadPilot(data, c=c) for c in (.5, 1., 2.)]
        fits = []
        for pilot in pilots:
            with patch.object(pilot, "prices", side_effect=lambda theta, mask: slopes * theta[0]):
                fit = minimize_scalar(lambda x: -pilot.log_likelihood(np.r_[x, THETA[1:]]),
                                      bounds=(.01, .16), method="bounded", options={"xatol":1e-12})
                fits.append(fit.x)
        self.assertTrue(.02 < fits[0] < .15)
        weighted_expected = np.sum(slopes * data["mid"] / data["half_spread"]**2) / np.sum(slopes**2 / data["half_spread"]**2)
        self.assertAlmostEqual(fits[0], weighted_expected, places=7)
        assert_allclose(fits, fits[0], atol=1e-8)

    def test_conditional_c_oracle_integrates_to_one_and_has_correct_mode(self):
        for residual in (np.zeros(1), np.zeros(4), np.array([.6, -.8, 1.2, -1.4]), np.full(4, 4.)):
            expected = float(np.clip(np.sqrt(np.mean(residual**2)), .5, 2.))
            self.assertAlmostEqual(conditional_c_mode(residual), expected)
            density = lambda c: np.exp(conditional_c_log_density(c, residual))
            self.assertAlmostEqual(quad(density, .5, 2., epsabs=1e-10)[0], 1., places=9)
            mode_logp = conditional_c_log_density(expected, residual)
            self.assertTrue(np.all(conditional_c_log_density(np.linspace(.5, 2., 101), residual) <= mode_logp + 1e-12))
            self.assertTrue(np.isfinite(conditional_c_log_density([.5, 2.], residual)).all())
            self.assertTrue(np.isneginf(conditional_c_log_density([.49, 2.01], residual)).all())
        joint = HestonSpreadPilot(records(), infer_c=True)
        residual = np.array([.6, -.8, 1.2, -1.4])
        clean = joint.records["mid"] - residual * joint.records["half_spread"]
        with patch.object(joint, "prices", return_value=clean):
            self.assertAlmostEqual(joint.conditional_c_mode(THETA), conditional_c_mode(residual))
            grid = np.array([.5, 1., 1.6, 2.])
            log_likelihood = np.array([joint.log_likelihood(np.r_[THETA, c]) for c in grid])
            log_oracle = joint.conditional_c_log_density(grid, THETA)
            assert_allclose(log_likelihood-log_likelihood[0], log_oracle-log_oracle[0], atol=1e-11)

    def test_selected_likelihood_and_refinement_use_supplied_observations(self):
        pilot = HestonSpreadPilot(records(), infer_c=True)
        theta, selected = np.r_[THETA, 1.3], [3, 0]
        observations = np.array([-1., 2., -3., 4.])
        price = pilot.prices(theta, selected)
        sigma = pilot.observation_sigma(theta, selected)
        expected = -.5*np.sum(((observations[selected]-price)/sigma)**2 + np.log(2*np.pi*sigma**2))
        self.assertAlmostEqual(pilot.log_likelihood(theta, selected, observations=observations), expected, places=10)
        report = pilot.refinement_report(theta, selected, observations=observations)
        y = observations[selected]
        shift = -.5*np.sum(((y-report["refined_prices"])/sigma)**2 - ((y-report["base_prices"])/sigma)**2)
        self.assertAlmostEqual(report["log_likelihood_shift"], shift, places=13)

    def test_numerical_gate_never_loosens_with_c(self):
        pilot = HestonSpreadPilot(records(), infer_c=True,
                                  quadrature_ladder=((16, 20.), (32, 40.)))
        assert_allclose(pilot.measurement_sigma, .5 * records()["half_spread"])
        for c in (.5, 2.):
            with patch.object(pilot, "_rung_prices", side_effect=[np.ones(4), np.ones(4)+.001]):
                with self.assertRaisesRegex(FloatingPointError, "refinement gate"):
                    pilot.prices(np.r_[THETA, c])

    def test_simulation_is_seeded_and_uses_only_spread_noise(self):
        pilot = HestonSpreadPilot(records(), infer_c=True)
        clean = np.array([1., 2., 3., 4.])
        theta = np.r_[THETA, 1.2]
        first = pilot.simulate(theta, seed=19, size=5, clean_prices=clean)
        expected = clean + np.random.default_rng(19).standard_normal((5, 4)) * 1.2 * records()["half_spread"]
        assert_allclose(first, expected, rtol=0, atol=0)
        assert_allclose(first, pilot.simulate(theta, seed=19, size=5, clean_prices=clean), rtol=0, atol=0)
        self.assertFalse(np.array_equal(first, pilot.simulate(theta, seed=20, size=5, clean_prices=clean)))
        assert_allclose(first[:, [3, 0]], pilot.simulate(theta, seed=19, size=5, mask=[3, 0], clean_prices=clean), rtol=0, atol=0)

    def test_task_identity_binds_target_geometry_spreads_and_numerics(self):
        pilot = HestonSpreadPilot(records())
        task = pilot.task_definition()
        self.assertEqual(pilot.validate_task(task), task)
        changed_observations = records()
        changed_observations["mid"][:] = -1.
        self.assertEqual(HestonSpreadPilot(changed_observations).task_definition(), task)
        for altered in (HestonSpreadPilot(records(), c=2.), HestonSpreadPilot(records(), infer_c=True),
                        HestonSpreadPilot(records(), numerical_fraction=.01)):
            with self.assertRaisesRegex(ValueError, "identity"):
                altered.validate_task(task)
        mutated = copy.deepcopy(task)
        mutated["prior_bounds"][0][0] *= .9
        with self.assertRaisesRegex(ValueError, "identity"):
            pilot.validate_task(mutated)
        for key in ("strike", "half_spread"):
            data = records()
            data[key][0] *= 1.01
            with self.assertRaisesRegex(ValueError, "identity"):
                HestonSpreadPilot(data).validate_task(task)

    def test_dimensions_finiteness_positive_scales_and_prior_support(self):
        for bounds in ((0., 2.), (-1., 2.), (2., .5), (.5, np.inf), (.5, 1., 2.)):
            with self.assertRaises(ValueError):
                HestonSpreadPilot(records(), c_bounds=bounds)
        for c in (0., .49, 2.01, np.nan, True):
            with self.assertRaises(ValueError):
                HestonSpreadPilot(records(), c=c)
        for key, bad in (("mid", np.nan), ("half_spread", 0.), ("maturity", -1.)):
            data = records()
            data[key][0] = bad
            with self.assertRaises(ValueError):
                HestonSpreadPilot(data)
        with self.assertRaises(ValueError):
            HestonSpreadPilot(records(), bounds=((.01,.16),)*4 + ((-1.1,-.1),))
        pilot = HestonSpreadPilot(records())
        for bad in (THETA[:4], np.r_[THETA, 1.], THETA * np.nan):
            with self.assertRaises(ValueError):
                pilot.log_likelihood(bad)
        with self.assertRaisesRegex(ValueError, "real"):
            pilot.log_likelihood(THETA.astype(complex) + 1j)
        with self.assertRaisesRegex(ValueError, "real"):
            pilot.log_likelihood(THETA, observations=np.ones(4, dtype=complex))
        for bad in ([1., 2.], [1., 2., 3., np.nan]):
            with self.assertRaises(ValueError):
                pilot.log_likelihood(THETA, observations=bad)
        outside = THETA.copy(); outside[0] = .009
        self.assertEqual(pilot.log_likelihood(outside), -np.inf)
        for mask in ([], [0,0], [4], [True,False]):
            with self.assertRaises(ValueError):
                pilot.log_likelihood(THETA, mask=mask)
        with self.assertRaisesRegex(ValueError, "infer_c"):
            pilot.conditional_c_mode(THETA)


if __name__ == "__main__":
    unittest.main()
