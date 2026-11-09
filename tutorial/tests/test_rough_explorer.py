"""The small explorer uses the shared pricing driver and controlled innovations."""

import unittest

import numpy as np

from sbi4f_tutorial.rough_options import explorer, simulator


class RoughExplorerTests(unittest.TestCase):
    def test_explorer_matches_the_canonical_hybrid_paths_and_compensator(self):
        result = explorer.explore_paths(.12, 1.4, -.65, .25, seed=13, paths=6, steps=64)
        innovations = np.random.default_rng(13).standard_normal((6, 64, 3))
        native = simulator.simulate_rough(
            {"hurst": .12, "eta": 1.4, "rho": -.65},
            {"times": [0., 1.], "values": [.25**2, .25**2]}, 1 / 64, innovations,
            method="hybrid", spot=1., rate=0., dividend=0.)
        self.assertEqual(result["prices"].shape, (6, 65))
        np.testing.assert_array_equal(result["times_years"], native["times"])
        np.testing.assert_array_equal(result["prices"], np.exp(native["log_spot"]))
        np.testing.assert_array_equal(result["variances"], native["variance"])
        np.testing.assert_array_equal(result["driver_variance"], native["driver_variance"])
        np.testing.assert_array_equal(result["volterra_driver"], native["volterra_driver"])
        np.testing.assert_allclose(result["variances"], .25**2 * np.exp(
            1.4 * result["volterra_driver"] - .5 * 1.4**2 * result["driver_variance"]), rtol=1e-15)
        self.assertFalse(np.array_equal(result["driver_variance"], result["times_years"] ** (.24)),
                         "the hybrid scheme uses its own Gaussian-grid variance")

    def test_seed_and_path_prefix_are_reproducible_and_new_seed_changes_the_paths(self):
        a = explorer.explore_paths(seed=7, paths=6, steps=64)
        b = explorer.explore_paths(seed=7, paths=8, steps=64)
        c = explorer.explore_paths(seed=8, paths=6, steps=64)
        np.testing.assert_array_equal(a["prices"], b["prices"][:6])
        np.testing.assert_array_equal(a["variances"], b["variances"][:6])
        self.assertFalse(np.array_equal(a["prices"], c["prices"]))
        np.testing.assert_array_equal(a["prices"][:, 0], np.ones(6))
        np.testing.assert_array_equal(a["variances"][:, 0], np.full(6, .2**2))

    def test_leverage_changes_prices_but_not_the_variance_driver(self):
        a = explorer.explore_paths(rho=-.3, paths=6, steps=64)
        b = explorer.explore_paths(rho=-.9, paths=6, steps=64)
        np.testing.assert_array_equal(a["variances"], b["variances"])
        self.assertFalse(np.array_equal(a["prices"], b["prices"]))

    def test_zero_variance_amplitude_and_supplied_curve_scale_have_expected_effects(self):
        a = explorer.explore_paths(eta=0., hurst=.05, paths=6, steps=64)
        b = explorer.explore_paths(eta=0., hurst=.3, paths=6, steps=64)
        np.testing.assert_array_equal(a["variances"], np.full((6, 65), .2**2))
        np.testing.assert_array_equal(a["prices"], b["prices"])
        c = explorer.explore_paths(volatility=.2, paths=6, steps=64)
        d = explorer.explore_paths(volatility=.3, paths=6, steps=64)
        np.testing.assert_allclose(d["variances"], c["variances"] * (.3 / .2)**2, rtol=1e-15)

    def test_invalid_controls_fail_without_silent_clipping(self):
        for kwargs in [{"volatility": 0}, {"volatility": float("nan")}, {"paths": 65},
                       {"steps": 0}, {"seed": True}, {"seed": -1}, {"hurst": .5}, {"rho": .2}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                explorer.explore_paths(**kwargs)


if __name__ == "__main__":
    unittest.main()
