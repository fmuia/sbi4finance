"""Shape guides, fixed-grid pricing and fair inputs stay distinct."""

import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from sbi4f_tutorial.rough_options import comparison
from sbi4f_tutorial.shared import tasks
from sbi4f_tutorial.shared.paths import CHECKPOINTS, RESULTS


class RoughComparisonTests(unittest.TestCase):
    def setUp(self):
        self.example = json.loads((RESULTS / "smile/example-rough.json").read_text())
        self.rough = {"parameters": self.example["parameters"], "vol": .25,
                      "implied": self.example["implied_vol_clean"], "paths": 32768,
                      "cpu_seconds": self.example["pricing"]["cpu_seconds"],
                      "max_se_over_sigma": self.example["pricing"]["max_mc_se_over_sigma"]}

    def test_comparison_uses_canonical_fourier_prices_same_mean_variance_and_correlation(self):
        with patch("sbi4f_tutorial.shared.pricing.rough_surface", side_effect=AssertionError("do not rerun MC")):
            actual = comparison.compare_smiles(self.rough, volatility=.25, rho=self.rough["parameters"]["rho"])
        task = json.loads((CHECKPOINTS / "heston/task.json").read_text())
        p, v0 = tasks._heston_split(tasks.model_parameters(task, [.25**2, .25**2, .4, self.rough["parameters"]["rho"]]))
        expected, _ = tasks.heston_prices(task, p, v0)
        np.testing.assert_array_equal(actual["heston"]["prices"], expected)
        self.assertEqual(actual["heston"]["parameters"]["theta"], actual["metadata"]["common_mean_variance"])
        self.assertEqual(actual["heston"]["parameters"]["rho"], actual["rough"]["parameters"]["rho"])
        self.assertEqual(actual["metadata"]["rough_paths_per_block"], 32768)
        self.assertTrue(np.isfinite(actual["rough"]["skew"]).all())
        np.testing.assert_allclose(actual["standardized_moneyness"].reshape(7, 9)[0], np.arange(-2, 2.1, .5), atol=1e-14)

    def test_missing_atm_inversions_fail_but_unavailable_wings_are_counted(self):
        missing = copy.deepcopy(self.rough)
        missing["implied"][4] = np.nan
        with self.assertRaisesRegex(ValueError, "five finite near-ATM"):
            comparison.compare_smiles(missing, volatility=.25, rho=missing["parameters"]["rho"])
        missing = copy.deepcopy(self.rough)
        missing["implied"][0] = np.nan
        actual = comparison.compare_smiles(missing, volatility=.25, rho=missing["parameters"]["rho"])
        self.assertEqual(actual["rough"]["omitted_wing_quotes"], 1)
        np.testing.assert_allclose(actual["rough"]["skew"], self.example["clean"]["skew"])

    def test_different_mean_variance_or_correlation_cannot_be_labelled_as_matched(self):
        with self.assertRaisesRegex(ValueError, "same flat variance"):
            comparison.compare_smiles(self.rough)
        with self.assertRaisesRegex(ValueError, "same flat variance"):
            comparison.compare_smiles(self.rough, volatility=.25, rho=-.4)

    def test_shape_guides_normalize_at_the_same_maturity_without_using_a_pricer(self):
        import matplotlib.pyplot as plt
        spy = json.loads((RESULTS / "spy/surface.json").read_text())
        with patch("sbi4f_tutorial.shared.pricing.rough_surface", side_effect=AssertionError("shape only")):
            figure = comparison.skew_shape_figure(.1, spy)
        rough, heston, observed = figure.axes[0].lines[:3]
        self.assertEqual(rough.get_ydata()[np.flatnonzero(rough.get_xdata() == 21)[0]], 1)
        np.testing.assert_array_equal(heston.get_ydata(), np.ones_like(heston.get_ydata()))
        self.assertEqual(observed.get_ydata()[2], 1)
        np.testing.assert_allclose(rough.get_ydata(), (rough.get_xdata() / 21) ** -.4)
        self.assertEqual(figure.axes[0].get_ylim(), (.3, 30))
        self.assertEqual(figure.axes[0].get_xlim(), (.5, 252))
        alternative = comparison.skew_shape_figure(.4, spy)
        self.assertEqual(alternative.axes[0].get_ylim(), figure.axes[0].get_ylim())
        plt.close(alternative)
        plt.close(figure)
        invalid = copy.deepcopy(spy)
        invalid["design"]["days"][2] = 20
        with self.assertRaisesRegex(ValueError, "21-trading-day"):
            comparison.skew_shape_figure(.1, invalid)

    def test_unreported_monte_carlo_precision_cannot_be_displayed_as_measured(self):
        missing = copy.deepcopy(self.rough)
        missing["max_se_over_sigma"] = np.nan
        with self.assertRaisesRegex(ValueError, "rough timing/MC SE"):
            comparison.compare_smiles(missing, volatility=.25, rho=missing["parameters"]["rho"])

    def test_axis_presets_stay_fixed_when_model_values_change_and_report_off_scale_points(self):
        import matplotlib.pyplot as plt

        result = comparison.compare_smiles(self.rough, volatility=.25, rho=self.rough["parameters"]["rho"])
        original = comparison.comparison_figure(result)
        changed = copy.deepcopy(result)
        changed["rough"]["implied"] *= 10
        changed["rough"]["skew"] *= 100
        raw = changed["rough"]["implied"].copy()
        figure = comparison.comparison_figure(changed)
        for before, after in zip(original.axes, figure.axes):
            self.assertEqual(before.get_xlim(), after.get_xlim())
            self.assertEqual(before.get_ylim(), after.get_ylim())
        self.assertEqual(figure.axes[0].get_xlim(), (-2.05, 2.05))
        self.assertEqual(figure.axes[0].get_ylim(), (0, 100))
        self.assertEqual(figure.axes[1].get_xlim(), (5, 252))
        self.assertEqual(figure.axes[1].get_ylim(), (.02, 10))
        self.assertNotIn("0 points outside axes", figure.axes[0].get_title())
        np.testing.assert_array_equal(changed["rough"]["implied"], raw)
        wide = comparison.comparison_figure(changed, scale="wide")
        self.assertEqual(wide.axes[0].get_ylim(), (0, 250))
        self.assertEqual(wide.axes[1].get_ylim(), (.002, 100))
        for item in (original, figure, wide):
            plt.close(item)


if __name__ == "__main__":
    unittest.main()
