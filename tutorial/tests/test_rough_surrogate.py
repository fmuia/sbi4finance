"""The pricer uses clean labels, physical parameter logits and explicit output units."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.special import logit

from sbi4f_tutorial.rough_options import surrogate
from sbi4f_tutorial.shared import backbones, parameter_transforms, tasks


class RoughSurrogateTests(unittest.TestCase):
    def test_known_curve_features_and_joint_curve_roles_match_the_trained_task(self):
        for name in ("rough-known100k-validation0", "rough-joint100k-validation0"):
            task, _, case = backbones.load_example(name)
            bounds = np.asarray(task["prior_bounds"])
            theta = case["neural_draws"][:2]
            context = (np.repeat(case["context_values"][None, :], 2, axis=0)
                       if "context_values" in case else np.full((2, 7), np.nan))
            features = surrogate.features(task, theta, context)
            np.testing.assert_array_equal(features[:, :len(bounds)],
                                          logit((theta - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])))
            if tasks.is_joint_curve_task(task):
                self.assertEqual(features.shape, (2, 6))
                np.testing.assert_array_equal(features, surrogate.features(task, theta))
            else:
                self.assertEqual(features.shape, (2, 10))
                np.testing.assert_array_equal(features[:, 3:], np.log(context))

    def test_box_transform_roundtrip_and_jacobian_match_the_physical_uniform_prior(self):
        task, _, case = backbones.load_example("rough-joint100k-validation0")
        bounds = parameter_transforms.bounds_of(task)
        theta = case["neural_draws"][:3]
        z = parameter_transforms.to_unbounded(theta, bounds)
        np.testing.assert_allclose(parameter_transforms.to_bounded(z, bounds), theta, rtol=5e-16)
        width = bounds[:, 1] - bounds[:, 0]
        unit = (theta - bounds[:, 0]) / width
        expected = np.log(width * unit * (1 - unit)).sum(axis=-1)
        np.testing.assert_allclose(parameter_transforms.log_jacobian(z, bounds), expected)

    def test_errors_compare_clean_price_labels_in_quote_noise_units(self):
        task, _, case = backbones.load_example("rough-known100k-validation0")
        theta = case["neural_draws"][:2]
        sigma = tasks.sigma(task)
        values = np.broadcast_to(2 * sigma, (2, 63)).copy()
        split = {"parameters": theta, "reference_prices": values,
                 "observations": np.full((2, 63), 999.0),
                 "context_values": np.repeat(case["context_values"][None, :], 2, axis=0)}
        fake_pricer = lambda x: np.full((len(x), 63), 2.25)
        np.testing.assert_array_equal(surrogate.targets(task, values), np.full((2, 63), 2.0))
        np.testing.assert_array_equal(surrogate.errors(fake_pricer, task, split), np.full((2, 63), .25))
        split["observations"][:] = -111.0
        np.testing.assert_array_equal(surrogate.errors(fake_pricer, task, split), np.full((2, 63), .25))

    def test_numpy_payloads_keep_the_activation_marker_and_exact_predictions(self):
        rng = np.random.default_rng(37)
        weights = [(rng.normal(size=(3, 4)), rng.normal(size=4)),
                   (rng.normal(size=(4, 2)), rng.normal(size=2))]
        x = rng.normal(size=(5, 3))
        with tempfile.TemporaryDirectory() as temporary:
            for activation in ("gelu_tanh", "gelu_erf"):
                original = surrogate.Pricer(weights, np.zeros(3), np.ones(3), [1, 2], [3, 4], activation=activation)
                path = Path(temporary) / activation
                original.save(path)
                restored = surrogate.Pricer.load(path)
                self.assertEqual(restored.activation, activation)
                np.testing.assert_array_equal(original(x), restored(x))


if __name__ == "__main__":
    unittest.main()
