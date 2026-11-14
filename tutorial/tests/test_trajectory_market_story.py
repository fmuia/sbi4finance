"""The long market snapshot and illustrative model keep their actual identities."""

import unittest

import numpy as np

from sbi4f_tutorial.trajectories import lesson, market_story, simulation


class MarketStoryTests(unittest.TestCase):
    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_long_snapshot_preserves_adjusted_price_returns_and_old_window(self):
        full = lesson.load_market_example("voo-2020-20261113")
        self.assertEqual((len(full["dates"]), full["dates"][0], full["dates"][-1]),
                         (1726, "2020-01-03", "2026-11-13"))
        np.testing.assert_array_equal(full["adjusted_close"][1:] / full["adjusted_close"][:-1] - 1,
                                      full["simple_returns"])
        np.testing.assert_array_equal(np.log1p(full["simple_returns"]), full["log_returns"])
        self.assertEqual(len(lesson.load_market_example()["dates"]), 256)
        with self.assertRaises(ValueError):
            lesson.load_market_example("unpackaged-market-window")

    def test_empirical_tail_retains_ties_and_extremes(self):
        thresholds, probability = market_story.absolute_survival([0., .01, -.01, .02])
        np.testing.assert_array_equal(thresholds, [0., .01, .02])
        np.testing.assert_array_equal(probability, [1., .75, .25])
        with self.assertRaises(ValueError):
            market_story.absolute_survival([0., np.nan])

    def test_illustration_is_canonical_fixed_law_not_a_market_fit(self):
        model = market_story.model_example()
        parameters = np.array([[model["parameters"][name] for name in model["task"]["parameter_names"]]] * 16)
        canonical = simulation.simulate_parameters(model["task"], parameters,
                                                   seed=model["seed"], backend="numpy")
        np.testing.assert_array_equal(canonical[:, :, 0], model["returns"])
        self.assertEqual(model["returns"].shape, (16, 1260))
        self.assertFalse(np.array_equal(model["returns"][0], model["returns"][1]))
        figure = market_story.comparison_figure(lesson.load_market_example("voo-2020-20261113"), model)
        self.assertEqual(len(figure.axes), 4)
        self.assertEqual(figure.axes[0].get_ylim(), figure.axes[2].get_ylim())
        self.assertEqual(figure.axes[1].get_xlim(), figure.axes[3].get_xlim())
        self.assertEqual(figure.axes[1].get_yscale(), "log")


if __name__ == "__main__":
    unittest.main()
