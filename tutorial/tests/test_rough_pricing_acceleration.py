"""Parity and support checks for the explicitly selected terminal path kernel."""
import importlib.util
import sys
import unittest
from unittest.mock import patch

import numpy as np

from sbi4f_tutorial.shared import pricing
from sbi4f_tutorial.shared.pricing import rough_block
from sbi4f_tutorial.rough_options import _pricing_acceleration as acceleration
from sbi4f_tutorial.rough_options.simulator import simulate_rough


@unittest.skipUnless(importlib.util.find_spec("numba"), "optional numba acceleration is not installed")
class RoughPricingAccelerationTests(unittest.TestCase):
    curve = {"times": [0., .25, 1.], "values": [.04, .08, .03]}

    def assert_nested_close(self, first, second):
        if isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_nested_close(first[key], second[key])
        elif isinstance(first, str):
            self.assertEqual(first, second)
        else:
            np.testing.assert_allclose(first, second, rtol=2e-11, atol=1e-13)

    def test_full_estimator_moments_across_chunks_and_parameter_edges(self):
        contracts = [{"maturity": t, "strike": k, "option_type": kind}
                     for t in (.25, .5) for k, kind in ((.85, "put"), (1., "call"), (1.15, "put"))]
        for p in ({"hurst": .1, "eta": 1.5, "rho": -.7},
                  {"hurst": .025, "eta": 3., "rho": -.95},
                  {"hurst": .49, "eta": 0., "rho": 0.},
                  {"hurst": .15, "eta": 2.2, "rho": -1.}):
            with self.subTest(parameters=p):
                diagnostics = [{"capture_raw_control": True} for _ in range(2)]
                outputs = [rough_block(p, self.curve, contracts, horizon=.5, n_steps=32,
                                      n_pairs=2053, seed=42, diagnostics=d, path_backend=backend)
                           for backend, d in zip(("reference", "numba"), diagnostics)]
                for reference, candidate in zip(*outputs):
                    np.testing.assert_allclose(candidate, reference, rtol=2e-11, atol=1e-13)
                self.assert_nested_close(diagnostics[0], diagnostics[1])

    def test_compact_paths_match_reference_left_point_integrals(self):
        p = {"hurst": .08, "eta": 2., "rho": -.8}
        z = np.random.default_rng(71).standard_normal((13, 17, 3))
        indices = np.array([1, 3, 17])
        found = acceleration.antithetic_terminal_paths(p, self.curve, .01, z, indices)
        for side, sign in enumerate((1., -1.)):
            reference = simulate_rough(p, self.curve, .01, sign*z, conditional_only=True)
            for key in reference:
                np.testing.assert_allclose(found[side][key], reference[key][:, indices-1], rtol=3e-14, atol=1e-16)

    def test_uncertain_certificate_uses_reference_fallback(self):
        p = {"hurst": .1, "eta": 1., "rho": -.7}
        z = np.random.default_rng(92).standard_normal((3, 4, 3))
        with patch.object(acceleration, "_certifies_omitted_paths", return_value=False), \
                patch.object(acceleration, "simulate_rough", wraps=simulate_rough) as reference:
            found = acceleration.antithetic_terminal_paths(p, self.curve, .01, z, [1, 4])
        self.assertEqual(reference.call_count, 2)
        self.assertEqual(found[0]["integrated_variance"].shape, (3, 2))
        for side, sign in enumerate((1., -1.)):
            full = simulate_rough(p, self.curve, .01, sign*z, conditional_only=True)
            for key in full:
                np.testing.assert_array_equal(found[side][key], full[key][:, [0, 3]])

    def test_changed_chunk_size_and_partial_final_chunk(self):
        p = {"hurst": .13, "eta": 1.8, "rho": -.65}
        contracts = [{"maturity": .25, "strike": 1., "option_type": "call"}]
        with patch.object(pricing, "_PAIR_CHUNK", 7):
            reference = rough_block(p, self.curve, contracts, horizon=.25, n_steps=8,
                                    n_pairs=19, seed=64)
            candidate = rough_block(p, self.curve, contracts, horizon=.25, n_steps=8,
                                    n_pairs=19, seed=64, path_backend="numba")
        for first, second in zip(reference, candidate):
            np.testing.assert_allclose(first, second, rtol=2e-11, atol=1e-13)

    def test_unused_endpoint_variance_overflow_is_rejected(self):
        # The final variance is unused by left-point pricing. Its overflow must
        # still fail, matching the full simulator's support check.
        p = {"hurst": .1, "eta": 100., "rho": -.7}
        z = np.zeros((2, 1, 3)); z[:, 0, 1] = 1e6
        for run in (lambda: simulate_rough(p, self.curve, .01, z, conditional_only=True),
                    lambda: acceleration.antithetic_terminal_paths(p, self.curve, .01, z, [1])):
            with self.assertRaisesRegex(ValueError, "variance overflow"):
                run()

    def test_indices_are_validated(self):
        p = {"hurst": .1, "eta": 1., "rho": -.7}
        z = np.zeros((2, 4, 3))
        for indices in ([], [0], [5], [2, 1], np.array([3, 2], dtype=np.uint64), [1, 1], [1.5]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                acceleration.antithetic_terminal_paths(p, self.curve, .01, z, indices)


class RoughPricingBackendSelectionTests(unittest.TestCase):
    def test_missing_optional_compiler_has_an_explicit_error(self):
        acceleration._compiled_kernel.cache_clear()
        try:
            with patch.dict(sys.modules, {"numba": None}), self.assertRaisesRegex(RuntimeError, "optional acceleration dependency"):
                acceleration._compiled_kernel()
        finally:
            acceleration._compiled_kernel.cache_clear()

    def test_unknown_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "path_backend"):
            rough_block({}, {}, [], horizon=1., n_steps=1, n_pairs=2, seed=1, path_backend="unknown")


if __name__ == "__main__":
    unittest.main()
