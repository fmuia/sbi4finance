"""Lean conditional outputs preserve full-path integrals and exact pricing streams."""

from contextlib import contextmanager
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

from sbi4finance.bergomi import _asset_paths
from sbi4finance.rough_bergomi import simulate_rough
from sbi4finance.calibration import pricing, tasks


@contextmanager
def full_output_pricer():
    original = simulate_rough

    def full(*args, **kwargs):
        kwargs["conditional_only"] = False
        return original(*args, **kwargs)

    with patch.object(pricing, "simulate_rough", side_effect=full):
        yield


class ConditionalOutputTests(unittest.TestCase):
    def test_integrals_are_bit_exact_and_default_api_is_full(self):
        curve = pricing.heston_variance_curve(np.linspace(0, .25, 33), .09, .01, 2)
        for method in ("hybrid", "covariance"):
            for eta, rho in ((0.0, 0.0), (1.5, -.7), (3.4, -.95)):
                z = np.random.default_rng(31).standard_normal((128, 32, 3))
                parameters = {"hurst": .08, "eta": eta, "rho": rho}
                full = simulate_rough(parameters, curve, .25 / 32, z, method=method)
                lean = simulate_rough(parameters, curve, .25 / 32, z, method=method, conditional_only=True)
                self.assertTrue({"log_returns", "log_spot", "variance", "volterra_driver"} <= set(full))
                self.assertEqual(set(lean), {"integrated_variance", "volatility_driver_integral"})
                for key in lean:
                    assert_array_equal(lean[key], full[key])

    def test_actual_rough_block_prices_errors_and_covariance_are_bit_exact(self):
        contracts = tasks.joint_rough_task(days=(21, 63), deltas=(-1, 0, 1))["context"]["contracts"]
        for steps in (24, 96):
            curve = pricing.heston_variance_curve(np.linspace(0, .25, steps + 1), .09, .02, 3)
            for seed in (3, 712):
                parameters = {"hurst": .12, "eta": 1.6, "rho": -.65}
                kwargs = dict(horizon=.25, n_steps=steps, n_pairs=256, seed=seed)
                with full_output_pricer():
                    before = pricing.rough_block(parameters, curve, contracts, **kwargs)
                after = pricing.rough_block(parameters, curve, contracts, **kwargs)
                for left, right in zip(before, after):
                    assert_array_equal(left, right)

    def test_adaptive_rungs_seed_shapes_and_diagnostics_are_unchanged(self):
        task = tasks.joint_rough_task(days=(21, 63), deltas=(-1, 0, 1),
                   blocks=[{"horizon": .25, "n_steps": 24}], paths_ladder=(128, 256, 512),
                   budget=1e-9, last_rung_cap=1e6)
        row = np.array([.12, 1.6, -.65, .09, .02, 3])
        context, _ = tasks.derived_context(task, row)
        before_diagnostics, after_diagnostics = [], []
        with full_output_pricer():
            before = tasks.price(task, row, context, 2611, before_diagnostics)
        original = pricing.np.random.default_rng
        shapes = []

        class RecordedGenerator:
            def __init__(self, seed):
                self.rng = original(seed)

            def standard_normal(self, shape):
                shapes.append(shape)
                return self.rng.standard_normal(shape)

        with patch.object(pricing.np.random, "default_rng", side_effect=RecordedGenerator):
            after = tasks.price(task, row, context, 2611, after_diagnostics)
        self.assertTrue(shapes)
        self.assertTrue(all(shape[-1] == 3 for shape in shapes))
        self.assertEqual(after[-1].tolist(), [512])
        for left, right in zip(before, after):
            assert_array_equal(left, right)
        for left, right in zip(before_diagnostics, after_diagnostics):
            self.assertEqual(set(left), set(right))
            for key in left:
                assert_array_equal(left[key], right[key])

    def test_extreme_inputs_fall_back_and_preserve_failure_semantics(self):
        # Individually finite z can overflow the unused asset combination before
        # multiplication by a tiny volatility. The conservative bound must force
        # the original calculation, preserving its rejection.
        variance = np.full((1, 3), 1e-300)
        z = np.zeros((1, 2, 3))
        z[..., 0], z[..., 2] = -1.7e308, 1.7e308
        for lean in (False, True):
            with np.errstate(all="ignore"), self.assertRaisesRegex(ValueError, "nonfinite trajectory"):
                _asset_paths(variance, 1.0, z, -.7, 1.0, 0.0, 0.0, conditional_only=lean)
        # Another extreme case is finite; fallback must not reject it merely
        # because the sufficient scalar certificate could not establish safety.
        z[..., 0] = 0
        with np.errstate(all="ignore"):
            full = _asset_paths(variance, 1.0, z, -.7, 1.0, 0.0, 0.0)
            lean = _asset_paths(variance, 1.0, z, -.7, 1.0, 0.0, 0.0, conditional_only=True)
        for key in lean:
            assert_array_equal(full[key], lean[key])
        for lean in (False, True):
            with np.errstate(all="ignore"), self.assertRaises(ValueError):
                _asset_paths(np.full((1, 3), 1e308), 2.0, np.zeros((1, 2, 3)),
                             -.7, 1., 0., 0., conditional_only=lean)

    def test_invalid_inputs_are_not_hidden_by_lean_output(self):
        parameters = {"hurst": .1, "eta": 1.5, "rho": -.7}
        curve = {"times": [0., 1.], "values": [.04, .04]}
        z = np.zeros((4, 8, 3))
        z[0, 0, 2] = np.inf  # Even the integrated-out channel remains validated.
        for lean in (False, True):
            with self.assertRaisesRegex(ValueError, "innovations"):
                simulate_rough(parameters, curve, .1, z, conditional_only=lean)
            with self.assertRaisesRegex(ValueError, "positive"):
                simulate_rough(parameters, curve, 0., np.zeros((4, 8, 3)), conditional_only=lean)


if __name__ == "__main__":
    unittest.main()
