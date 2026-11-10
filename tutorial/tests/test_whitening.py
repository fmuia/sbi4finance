"""Full-covariance preprocessing retains small-variance information and state."""

from dataclasses import FrozenInstanceError
import copy
import json
import unittest

import numpy as np

from sbi4f_tutorial.shared.whitening import FullCovarianceWhitening, fit_whitening


class WhiteningTests(unittest.TestCase):
    def test_correlated_low_variance_directions_are_whitened_without_reduction(self):
        rng = np.random.default_rng(731)
        rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        values = (rng.normal(size=(6000, 4)) * [3., .7, .1, .001]) @ rotation.T
        fitted = fit_whitening(values)
        self.assertTrue(np.all(fitted.eigenvalues > fitted.eigenvalue_floor))
        transformed = fitted.transform(values)
        self.assertEqual(transformed.shape, values.shape)
        self.assertEqual(transformed.dtype, np.float64)
        np.testing.assert_allclose(transformed.mean(axis=0), 0., atol=1e-11)
        np.testing.assert_allclose(np.cov(transformed, rowvar=False), np.eye(4), atol=2e-9)
        np.testing.assert_allclose(fitted.inverse(transformed), values, rtol=2e-10, atol=2e-10)
        # A vector and batches with extra leading dimensions use the same map.
        np.testing.assert_allclose(fitted.transform(values[0]), transformed[0])
        np.testing.assert_array_equal(fitted.transform(values[:6].reshape(2, 3, 4)), transformed[:6].reshape(2, 3, 4))

    def test_active_floor_keeps_all_directions_and_an_invertible_transform(self):
        rng = np.random.default_rng(129)
        latent = rng.normal(size=(1000, 3))
        values = np.column_stack((latent[:, 0], latent[:, 0] + 1e-10 * latent[:, 1],
                                  1e-6 * latent[:, 2], np.zeros(1000)))
        fitted = fit_whitening(values)
        self.assertTrue(np.any(fitted.eigenvalues < fitted.eigenvalue_floor))
        self.assertEqual(fitted.dimension, 4)
        np.testing.assert_array_equal(fitted.safe_eigenvalues,
                                      np.maximum(fitted.eigenvalues, fitted.eigenvalue_floor))
        self.assertTrue(np.all(np.linalg.eigvalsh(fitted.whitening_matrix) > 0))
        arbitrary = np.array([[2., -1., 5., 7.]])
        self.assertEqual(fitted.transform(arbitrary).shape, (1, 4))
        np.testing.assert_allclose(fitted.inverse(fitted.transform(arbitrary)), arbitrary, rtol=2e-10, atol=2e-10)

    def test_float64_centering_preserves_variation_before_any_network_cast(self):
        rng = np.random.default_rng(57)
        values = 1e8 + rng.normal(scale=.01, size=(2000, 2))
        self.assertTrue(np.all(values.astype(np.float32) == np.float32(1e8)))
        fitted = fit_whitening(values)
        transformed = fitted.transform(values)
        self.assertEqual(transformed.dtype, np.float64)
        np.testing.assert_allclose(np.cov(transformed, rowvar=False), np.eye(2), atol=1e-9)
        self.assertGreater(float(np.max(np.abs(transformed))), 1.)
        np.testing.assert_allclose(fitted.inverse(transformed), values, atol=0., rtol=0.)

    def test_json_roundtrip_reconstructs_exact_state_without_mutable_aliases(self):
        rng = np.random.default_rng(18)
        values = rng.normal(size=(300, 5))
        fitted = fit_whitening(values)
        restored = FullCovarianceWhitening.from_dict(json.loads(json.dumps(fitted.to_dict(), allow_nan=False)))
        self.assertEqual(restored.to_dict(), fitted.to_dict())
        np.testing.assert_array_equal(restored.transform(values), fitted.transform(values))
        self.assertFalse(restored.mean.flags.writeable)
        self.assertFalse(restored.whitening_matrix.flags.writeable)
        with self.assertRaises(ValueError):
            restored.mean[0] = 3.
        with self.assertRaises(FrozenInstanceError):
            restored.training_rows = 4
        record = fitted.to_dict()
        restored = FullCovarianceWhitening.from_dict(record)
        record["mean"][0] += 100.
        np.testing.assert_array_equal(restored.mean, fitted.mean)

    def test_invalid_or_inconsistent_serialized_state_is_rejected(self):
        good = fit_whitening(np.random.default_rng(6).normal(size=(100, 3))).to_dict()
        changes = [
            lambda r: r.update(format="unknown"),
            lambda r: r.update(extra=1),
            lambda r: r.pop("mean"),
            lambda r: r.update(dimension=2),
            lambda r: r.update(dimension=True),
            lambda r: r.update(training_rows=1),
            lambda r: r["mean"].__setitem__(0, float("nan")),
            lambda r: r.update(whitening_matrix=[[1.]]),
            lambda r: r["whitening_matrix"][0].__setitem__(1, 20.),
            lambda r: r.update(whitening_matrix=(-np.eye(3)).tolist()),
            lambda r: r["inverse_matrix"][0].__setitem__(0, r["inverse_matrix"][0][0] + .1),
            lambda r: r["safe_eigenvalues"].__setitem__(0, -1.),
            lambda r: r.update(relative_eigenvalue_floor=.9),
        ]
        for change in changes:
            record = copy.deepcopy(good)
            change(record)
            with self.subTest(record=record):
                with self.assertRaises(ValueError):
                    FullCovarianceWhitening.from_dict(record)

    def test_invalid_fit_and_transform_inputs_are_rejected(self):
        for values in ([], [[1., 2.]], np.empty((3, 0)), np.ones((3, 2)), np.full((3, 2), .1),
                       [[1., np.nan], [2., 3.]], [[1j], [2j]], [["1"], ["2"]]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    fit_whitening(values)
        values = np.random.default_rng(9).normal(size=(100, 2))
        for floor in (0., -1., 1.1, float("nan"), float("inf"), True, np.bool_(True), 1j, "1e-8"):
            with self.subTest(floor=floor):
                with self.assertRaises(ValueError):
                    fit_whitening(values, floor)
        fitted = fit_whitening(values)
        for invalid in ([1.], [[1., 2., 3.]], [[np.inf, 2.]], [1j, 2j], np.empty((0, 2))):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    fitted.transform(invalid)
                with self.assertRaises(ValueError):
                    fitted.inverse(invalid)


if __name__ == "__main__":
    unittest.main()
