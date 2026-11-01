"""Independent integration checks for the tutorial's mean-curve functionals."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from sbi4finance.calibration.full_surface_teaching import rough_variance_products


LAB = Path(__file__).resolve().parents[1]


def task(known=False):
    name = "calibration_rough_known_backbone_20k.json" if known else "calibration_rough_joint_development_20k.json"
    return json.loads((LAB / "configs" / name).read_text())["task"]


def midpoint(specification, shape=()):
    return np.broadcast_to(np.mean(specification["prior_bounds"], axis=1),
                           (*shape, len(specification["parameter_names"]))).copy()


class RoughVarianceProductTests(unittest.TestCase):
    def test_joint_matches_independent_gauss_legendre_integration_and_keeps_paired_rows(self):
        specification = task()
        bounds = np.asarray(specification["prior_bounds"])
        draws = bounds[:, 0] + np.random.default_rng(37).random((2, 9, 6)) * np.diff(bounds).ravel()
        actual = rough_variance_products(specification, draws)
        nodes, weights = np.polynomial.legendre.leggauss(96)
        for key, start, end in (("variance_swap_1y", 0, 1), ("forward_variance_0_5_to_1y", .5, 1)):
            t = start + (nodes + 1) * (end - start) / 2
            curve = (draws[..., 4, None]
                     + (draws[..., 3, None] - draws[..., 4, None]) * np.exp(-draws[..., 5, None] * t))
            expected = np.sum(curve * weights, axis=-1) / 2
            self.assertEqual(actual[key].shape, (2, 9))
            assert_allclose(actual[key], expected, rtol=3e-14, atol=2e-16)
        permuted = draws[:, ::-1]
        for key, value in rough_variance_products(specification, permuted).items():
            assert_array_equal(value, actual[key][:, ::-1])
        unpaired = draws.copy(); unpaired[..., 5] = unpaired[:, ::-1, 5]
        self.assertGreater(np.max(np.abs(rough_variance_products(specification, unpaired)["variance_swap_1y"]
                                           - actual["variance_swap_1y"])), .001)

    def test_joint_flat_curve_and_noncurve_parameters_do_not_change_products(self):
        specification = task()
        draws = midpoint(specification, (3,))
        draws[:, 3:5] = .07
        draws[:, 5] = [.25, 2, 8]
        for value in rough_variance_products(specification, draws).values():
            assert_allclose(value, .07, rtol=0, atol=2e-17)
        draws[:, 3:5] = [.02, .13]
        draws[:, 5] = 2
        draws[:, :3] = np.asarray(specification["prior_bounds"])[:3, 0]
        before = rough_variance_products(specification, draws)
        draws[:, :3] = np.asarray(specification["prior_bounds"])[:3, 1]
        for key, value in rough_variance_products(specification, draws).items():
            assert_array_equal(value, before[key])

    def test_joint_small_and_large_decay_are_stable_convex_means(self):
        specification = task()
        specification["prior_bounds"][5] = [1e-14, 1e6]
        draws = midpoint(specification, (2,))
        draws[:, 3:5] = [.015, .15]
        draws[:, 5] = [1e-14, 1e6]
        products = rough_variance_products(specification, draws)
        for value in products.values():
            self.assertTrue(np.all(value >= .015))
            self.assertTrue(np.all(value <= .15))
            assert_allclose(value[0], .015, rtol=0, atol=2e-15)
        assert_allclose(products["variance_swap_1y"][1], .15 - (.15 - .015) / 1e6, rtol=2e-15)
        self.assertEqual(products["forward_variance_0_5_to_1y"][1], .15)

    def test_known_actual_knots_integrate_intervals_and_are_constant_per_draw(self):
        specification = task(known=True)
        levels = np.array([.02, .03, .04, .05, .06, .07, .08])
        draws = midpoint(specification, (2, 11))
        products = rough_variance_products(specification, draws, context_values=levels)
        expected = np.dot(np.diff(specification["model_context"]["forward_variance_knots"]), levels)
        assert_allclose(products["variance_swap_1y"], expected, rtol=2e-15)
        assert_array_equal(products["forward_variance_0_5_to_1y"], np.full((2, 11), .08))
        draws[..., :3] = np.asarray(specification["prior_bounds"])[:, 0]
        for key, value in rough_variance_products(specification, draws, context_values=levels).items():
            assert_array_equal(value, products[key])

    def test_known_overlap_clips_at_half_and_one_year_and_ignores_later_intervals(self):
        specification = task(known=True)
        specification["model_context"]["forward_variance_knots"] = [0, .1, .2, .4, .6, .8, 1.2, 2]
        levels = np.array([.01, .02, .03, .04, .05, .06, 10.])
        draws = midpoint(specification)
        products = rough_variance_products(specification, draws, context_values=levels)
        assert_allclose(products["variance_swap_1y"], .1*.01 + .1*.02 + .2*.03 + .2*.04 + .2*.05 + .2*.06)
        assert_allclose(products["forward_variance_0_5_to_1y"], (.1*.04 + .2*.05 + .2*.06) / .5)
        self.assertEqual(products["variance_swap_1y"].shape, ())

    def test_rejects_invalid_task_draw_shape_support_and_horizon(self):
        specification = task()
        draws = midpoint(specification, (2,))
        for mutation in (lambda t: t.update(parameter_names=t["parameter_names"][::-1]),
                         lambda t: t.update(task_id="other"),
                         lambda t: t.update(measure="P"),
                         lambda t: t["model_context"].update(curve_family="other"),
                         lambda t: t["model_context"].update(forward_variance_knots=[0, .1, .2, .3, .4, .5, .6, .9]),
                         lambda t: t["model_context"].update(forward_variance_knots=[0, .1, .2, .2, .4, .5, .6, 1]),
                         lambda t: t["model_context"].update(forward_variance_knots=[0, .1, .2, .3, .4, .5, .6, np.nan])):
            changed = copy.deepcopy(specification); mutation(changed)
            with self.subTest(task=changed), self.assertRaises(ValueError):
                rough_variance_products(changed, draws)
        for changed in (np.zeros((2, 5)), np.empty((0, 6)), draws * np.nan, draws + 100, np.array(1.)):
            with self.subTest(shape=changed.shape), self.assertRaises(ValueError):
                rough_variance_products(specification, changed)
        with self.assertRaisesRegex(ValueError, "inferred curve coefficients"):
            rough_variance_products(specification, draws, context_values=np.full(7, .04))

    def test_known_rejects_missing_nonfinite_nonpositive_and_batched_curves(self):
        specification = task(known=True)
        draws = midpoint(specification, (2,))
        for levels in (None, [1]*6, np.ones((2, 7)), [1]*6 + [0], [1]*6 + [np.nan]):
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                rough_variance_products(specification, draws, context_values=levels)

    def test_tutorial_copy_matches_lab_and_imports_no_backend_or_scipy(self):
        probe = '''
import importlib.abc, json, sys
import numpy as np
class Offline(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'bayesflow', 'keras', 'jax', 'scipy', 'sbi4finance'}:
            raise AssertionError('Unexpected offline import: ' + name)
sys.meta_path.insert(0, Offline())
from sbi4f_tutorial.full_surface_teaching import rough_variance_products
task = json.loads(sys.argv[1])
draws = np.asarray(sys.argv[2:8], dtype=float)
print(json.dumps({k: float(v) for k, v in rough_variance_products(task, draws).items()}))
'''
        specification = task(); draws = midpoint(specification)
        result = subprocess.run([sys.executable, "-c", probe, json.dumps(specification), *map(str, draws)],
                                capture_output=True, text=True, timeout=20,
                                env={**os.environ, "PYTHONPATH": str(LAB.parent / "tutorial/src")})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for key, value in json.loads(result.stdout).items():
            self.assertEqual(value, rough_variance_products(specification, draws)[key])


if __name__ == "__main__":
    unittest.main()
