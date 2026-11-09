"""One simulated market keeps paths, numerical prices and quote errors separate."""

import unittest

import numpy as np

from sbi4f_tutorial.rough_options import data_story
from sbi4f_tutorial.shared import backbones, tasks


class RoughDataStoryTests(unittest.TestCase):
    def setUp(self):
        self.task, _, case = backbones.load_example("rough-known100k-validation0")
        self.theta = np.asarray([.1, 1.5, -.7])
        self.context = case["context_values"]

    def simulate(self, **settings):
        return data_story.simulate_dataset(self.task, self.theta, context_values=self.context, paths=64, **settings)

    def test_payoff_means_parity_and_quote_noise_are_distinct_operations(self):
        result = self.simulate(seed=37, noise_seed=19)
        terminal = result["terminal_prices"][:, None]
        strikes = result["strikes"]
        put = result["option_types"] == "put"
        expected = np.maximum(np.where(put, strikes - terminal, terminal - strikes), 0)
        np.testing.assert_array_equal(result["payoffs"], expected)
        conditional = result["conditional_put_values"]
        self.assertTrue(np.all(conditional >= 0))
        self.assertTrue(np.all(conditional <= strikes + 1e-14))
        np.testing.assert_array_equal(result["mean_prices"], conditional.mean(0) + np.where(put, 0, 1 - strikes))
        np.testing.assert_array_equal(result["mc_se"], conditional.std(0, ddof=1) / np.sqrt(64))
        np.testing.assert_array_equal(result["observations"], result["mean_prices"] + result["quote_errors"])
        np.testing.assert_array_equal(result["quote_noise_std"], tasks.sigma(self.task)[result["contract_indices"]])
        self.assertEqual(result["metadata"]["block"], self.task["numerics"]["blocks"][1])

    def test_changing_quote_noise_never_changes_the_paths_or_mean_prices(self):
        a = self.simulate(seed=37, noise_seed=19)
        b = self.simulate(seed=37, noise_seed=20)
        c = self.simulate(seed=37, noise_seed=19, noise_scale=2)
        d = self.simulate(seed=37, noise_seed=19, noise_scale=0)
        for key in ("terminal_prices", "payoffs", "conditional_put_values", "mean_prices", "mc_se"):
            np.testing.assert_array_equal(a[key], b[key])
            np.testing.assert_array_equal(a[key], c[key])
        self.assertFalse(np.array_equal(a["quote_errors"], b["quote_errors"]))
        np.testing.assert_array_equal(c["quote_errors"], 2 * a["quote_errors"])
        np.testing.assert_array_equal(d["observations"], d["mean_prices"])

    def test_joint_curve_is_derived_and_never_accepted_as_observed_context(self):
        task, _, case = backbones.load_example("rough-joint100k-validation0")
        with self.assertRaisesRegex(ValueError, "not supplied context"):
            data_story.simulate_dataset(task, case["truth"], context_values=np.full(7, .04), paths=16)

    def test_validation_plot_shows_physical_prices_and_signed_quote_noise_errors_on_fixed_axes(self):
        import matplotlib.pyplot as plt
        sigma = tasks.sigma(self.task)
        validation = {"parameters": self.theta[None, :], "context_values": self.context[None, :],
                      "reference_prices": 2 * sigma[None, :], "pricing_standard_errors": .1 * sigma[None, :]}
        predictor = lambda values: np.full((len(values), 63), 3.5)
        figure = data_story.pricer_validation_figure(predictor, self.task, validation)
        np.testing.assert_allclose(figure.axes[1].lines[-1].get_ydata(), np.full(9, 1.5))
        self.assertEqual(figure.axes[1].get_ylim(), (-25, 25))
        self.assertEqual(figure.axes[0].get_ylim(), (-.02, .12))
        self.assertIn("1.50 quote SDs", figure._suptitle.get_text())
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
