"""Independent analytic checks for the explanatory posterior-CDF calculation."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.special import ndtr


spec = importlib.util.spec_from_file_location(
    "gate_audit", Path(__file__).resolve().parents[1] / "scripts/audit_hkde_calibration_gate.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ReferenceCDFTests(unittest.TestCase):
    def test_constant_likelihood_recovers_uniform_prior_on_unequal_axes(self):
        axes = (np.array([2., 2.1, 2.8, 4.]), np.array([-3., -1.5, -.2, 0.]))
        truth = np.array([[2., -3.], [3., -1.5], [4., 0.]])
        prices = np.zeros((len(axes[0])*len(axes[1]), 1))
        result = audit.pit_from_prices(axes, prices, np.zeros((3, 1, 1)), truth, .25)
        np.testing.assert_allclose(result, [[0., 0.], [.5, .5], [1., 1.]], atol=1e-14)

    def test_linear_prices_match_independent_truncated_normal_posteriors(self):
        # Refine both integration and interpolation below the analytic tolerance.
        axes = (np.linspace(-1., 2., 1201), np.linspace(-2., 1., 1001))
        prices = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
        observations = np.array([[[.3], [-.7]], [[1.8], [-1.7]]])
        truth = np.array([[.4, -.9], [1.9, -1.5]])
        sigma = .4
        result = audit.pit_from_prices(axes, prices, observations, truth, sigma)
        means = observations[..., 0]
        lo, hi = np.array([a[0] for a in axes]), np.array([a[-1] for a in axes])
        expected = ((ndtr((truth-means)/sigma)-ndtr((lo-means)/sigma)) /
                    (ndtr((hi-means)/sigma)-ndtr((lo-means)/sigma)))
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
