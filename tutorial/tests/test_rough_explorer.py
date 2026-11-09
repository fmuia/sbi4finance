"""The small explorer uses the shared pricing driver and controlled innovations."""

import unittest
from unittest.mock import patch

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
        np.testing.assert_array_equal(a["standard_normals"], b["standard_normals"])
        np.testing.assert_array_equal(a["brownian_increments"], b["brownian_increments"])
        self.assertFalse(np.array_equal(a["prices"], b["prices"]))

    def test_exposed_components_decompose_the_canonical_log_euler_update(self):
        with patch.object(simulator, "simulate_rough", wraps=simulator.simulate_rough) as shared:
            result = explorer.explore_paths(.08, 2.2, -.8, .3, seed=17, paths=6, steps=64)
        self.assertEqual(shared.call_count, 1, "no path can be rejected or silently redrawn")
        z = np.random.default_rng(17).standard_normal((6, 64, 3))
        np.testing.assert_array_equal(result["standard_normals"], z)
        np.testing.assert_array_equal(result["brownian_increments"], z[..., [0, 2]] / 8)
        d_b = -.8 * z[..., 0] / 8 + np.sqrt(1 - .8**2) * z[..., 2] / 8
        np.testing.assert_allclose(result["price_brownian_increments"], d_b, rtol=1e-15, atol=1e-17)
        np.testing.assert_allclose(result["asset_log_drift"], -.5 * result["variances"][:, :-1] / 64,
                                   rtol=1e-15)
        np.testing.assert_allclose(result["asset_log_diffusion"], np.sqrt(result["variances"][:, :-1]) * d_b,
                                   rtol=1e-13, atol=1e-17)
        np.testing.assert_array_equal(result["relative_increments"], np.expm1(result["log_returns"]))
        np.testing.assert_allclose(result["relative_increments"],
                                   result["prices"][:, 1:] / result["prices"][:, :-1] - 1,
                                   rtol=1e-10, atol=1e-15)
        np.testing.assert_array_equal(result["variance_compensator"], - .5 * 2.2**2 * result["driver_variance"])
        np.testing.assert_allclose(result["variance_multiplier"], np.exp(result["log_variance_multiplier"]),
                                   rtol=1e-15)
        self.assertEqual(result["settings"]["brownian_channels"], ["W", "W-perp"])

    def test_view_windows_stay_fixed_when_parameters_and_seed_change(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for kwargs in [{"hurst": .02, "eta": 3., "rho": -.95, "volatility": .4, "seed": 7},
                       {"hurst": .4, "eta": .2, "rho": -.05, "volatility": .1, "seed": 17},
                       {"hurst": .2, "eta": 1.5, "rho": -.7, "volatility": .2, "seed": 27}]:
            result = explorer.explore_paths(**kwargs, paths=6, steps=64)
            for preset in explorer.AXIS_PRESETS:
                with self.subTest(kwargs=kwargs, preset=preset):
                    fig = explorer.component_figure(result, preset=preset)
                    names = ("driver", "variance", "relative_increments", "prices")
                    for axis, name in zip(fig.axes, names, strict=True):
                        self.assertEqual(axis.get_xlim(), (0., 1.))
                        self.assertEqual(axis.get_ylim(), explorer.AXIS_PRESETS[preset][name])
                        self.assertEqual(axis.get_yscale(), "log" if name == "variance" else "linear")
                    plt.close(fig)
                    fig = explorer.path_figure(result, preset=preset)
                    for axis, name in zip(fig.axes, ("prices", "volatility_percent"), strict=True):
                        self.assertEqual(axis.get_xlim(), (0., 1.))
                        self.assertEqual(axis.get_ylim(), explorer.AXIS_PRESETS[preset][name])
                    plt.close(fig)

    def test_outside_values_are_counted_marked_and_remain_in_the_plot_data(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = explorer.explore_paths(eta=0., paths=2, steps=8)
        for name, fill in [("prices", 1.), ("variances", .04), ("volterra_driver", 0.), ("relative_increments", 0.)]:
            result[name] = np.full_like(result[name], fill)
        for name, low, high in [("prices", .1, 3.), ("variances", 1e-8, 100.),
                                 ("volterra_driver", -8., 8.), ("relative_increments", -.8, .8)]:
            result[name][0, -2] = low
            result[name][1, -1] = high
        saved = {name: value.copy() for name, value in result.items() if isinstance(value, np.ndarray)}
        stats = explorer.axis_statistics(result)
        self.assertEqual(stats["points_outside"], 8)
        self.assertTrue(all(row["outside"] == 2 and row["highlighted_outside"] == 1
                            for row in stats["components"]))
        fig = explorer.component_figure(result)
        for axis, name in zip(fig.axes, ("volterra_driver", "variances", "relative_increments", "prices"), strict=True):
            self.assertTrue(any(np.array_equal(line.get_ydata(), result[name][0]) for line in axis.lines))
            self.assertGreater(len(axis.collections), 0, "outside coordinates must receive boundary markers")
        for name, value in saved.items():
            np.testing.assert_array_equal(result[name], value)
        plt.close(fig)

    def test_unknown_view_presets_are_not_silently_autoscaled(self):
        result = explorer.explore_paths(paths=2, steps=8)
        for kwargs in [{"preset": "auto"}, {"view": "unknown"}, {"highlighted_path": 2}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                explorer.axis_statistics(result, **kwargs)

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
