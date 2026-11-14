"""The closing Bergomi study renders without payloads and preserves plot units."""

import importlib.util
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


LAB = Path(__file__).resolve().parents[1]
NOTEBOOK = LAB / "notebooks/heston_spx_calibration.py"


@unittest.skipUnless(importlib.util.find_spec("marimo") and importlib.util.find_spec("bayesflow"),
                     "requires the core notebook environment")
class HestonSPXBergomiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts import heston_grid_collaborator as delivery

        spec = importlib.util.spec_from_file_location("spx_bergomi_integration_under_test", NOTEBOOK)
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.module
        with (patch.dict(os.environ, {"HESTON_BUNDLE": ""}),
              patch.object(delivery, "load_checkpoint", side_effect=AssertionError("private checkpoint accessed")),
              patch.object(delivery, "load_bank", side_effect=AssertionError("private bank accessed"))):
            spec.loader.exec_module(cls.module)
            cls.outputs, cls.defs = cls.module.app.run()

    @classmethod
    def tearDownClass(cls):
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_saved_section_is_independent_of_the_main_checkpoint(self):
        self.assertTrue(self.outputs)
        self.assertIsNone(self.defs["BUNDLE"])
        self.assertIsNone(self.defs["estimator"])
        self.assertEqual(self.defs["bergomi_expiry"].value, 0)
        record = self.defs["bergomi_results"]
        self.assertEqual(record["training"]["rows"], 200_000)
        self.assertEqual(record["parameter_names"][-1], "c")
        self.assertEqual(record["parameter_bounds"][-1], [.5, 50.])
        self.assertFalse(record["scientific_acceptance"])
        # The separate saved experiment must not widen the live Heston task.
        np.testing.assert_allclose(self.defs["BOUNDS"][-1], [.5, 2.])

    def test_fit_panels_use_the_same_observations_residual_sign_and_units(self):
        record = self.defs["bergomi_results"]
        geometry = record["geometry"]
        observed = np.asarray(geometry["observed_raw"])
        vega = np.asarray(self.defs["MARKET_VEGA"])
        expiry_index = np.asarray(geometry["expiry_index"])
        z = np.asarray(self.defs["DELTA"])
        figure = self.defs["bergomi_fit_figure"]
        self.assertEqual(len(figure.axes), 3)
        for axis, expiry in zip(figure.axes, [0, 3, 6]):
            indices = np.flatnonzero(expiry_index == expiry)
            indices = indices[np.argsort(z[indices])]
            for line, model in zip(axis.lines[:2], ["heston", "bergomi"]):
                expected = 100 * (np.asarray(record["fit_comparison"][model]["price_mean_raw"])
                                  - observed) / vega
                np.testing.assert_allclose(line.get_xdata(), z[indices])
                np.testing.assert_allclose(line.get_ydata(), expected[indices])

    def test_bergomi_display_symbols_match_the_trajectory_lesson(self):
        labels = self.defs["BERGOMI_PARAMETER_LABELS"]
        symbols = {name: re.search(r"\$([^$]+)\$", label).group(1)
                   for name, label in labels.items()}
        self.assertEqual(symbols, {"hurst": "H", "eta": r"\eta", "rho": r"\rho",
                                   "short_variance": "S", "long_variance": "L",
                                   "decay": "k", "c": "c"})
        self.assertEqual(list(labels), self.defs["bergomi_results"]["parameter_names"])
        self.assertEqual([axis.get_title() for axis in self.defs["bergomi_recovery_figure"].axes[:7]],
                         list(labels.values()))

    def test_heston_displayed_rms_and_heatmap_use_the_same_volatility_point_residuals(self):
        observed = np.asarray(self.defs["MID"])
        expected = 100 * (np.asarray(self.defs["market_fit"]) - observed) / self.defs["MARKET_VEGA"]
        expected_rms = float(np.sqrt(np.mean(expected ** 2)))
        np.testing.assert_allclose(self.defs["market_residual_vol_points"], expected)
        self.assertAlmostEqual(self.defs["market_rms"], expected_rms, places=12)
        summary = self.defs["heston_misfit_vol_summary"]
        self.assertAlmostEqual(summary["rms"], expected_rms, places=12)
        expiry, z = np.asarray(self.defs["EXPIRY"]), np.asarray(self.defs["DELTA"])
        rows = [np.flatnonzero(expiry == index) for index in range(7)]
        rows = [indices[np.argsort(z[indices])] for indices in rows]
        grid = np.asarray(self.defs["heston_misfit_figure"].axes[0].images[0].get_array())
        np.testing.assert_allclose(grid, np.asarray([expected[indices] for indices in rows]))
        np.testing.assert_allclose(summary["expiry_rms"],
                                   [np.sqrt(np.mean(expected[indices] ** 2)) for indices in rows])

    def test_noisy_band_inclusion_is_preserved_for_every_contract_and_method(self):
        record = self.defs["bergomi_results"]
        observed = np.asarray(record["geometry"]["observed_raw"])
        for method in ("mcmc", "amortizer"):
            with self.subTest(method=method):
                predictive = record["predictives"][method]
                raw = np.asarray(predictive["noisy_q05_q50_q95_raw"])
                converted = self.defs["price_residual_vol_points"](raw, observed)
                self.assertEqual(converted.shape, (3, 63))
                np.testing.assert_allclose(converted, 100 * (raw - observed) / self.defs["MARKET_VEGA"])
                raw_inside = (raw[0] <= observed) & (observed <= raw[2])
                converted_inside = (converted[0] <= 0) & (0 <= converted[2])
                np.testing.assert_array_equal(raw_inside, converted_inside)
                self.assertEqual(int(converted_inside.sum()), predictive["noisy_in_band_count"])

    def test_saved_heston_market_views_include_both_methods_without_an_estimator(self):
        self.assertIsNone(self.defs["estimator"])
        record = self.defs["market_predictives"]
        self.assertEqual(set(record["methods"]), {"pymc", "amortizer"})
        expiry, z = np.asarray(self.defs["EXPIRY"]), np.asarray(self.defs["DELTA"])
        fit = self.defs["heston_market_fit_figure"]
        predictive = self.defs["heston_market_predictive_figure"]
        self.assertEqual(len(fit.axes), 8)
        self.assertEqual(len(predictive.axes), 8)
        for index, axis in enumerate(fit.axes[:7]):
            indices = np.flatnonzero(expiry == index)
            indices = indices[np.argsort(z[indices])]
            for line, method in zip(axis.lines[:2], ("pymc", "amortizer")):
                quantiles = np.asarray(record["methods"][method]["clean_q05_q50_q95_raw"])
                expected, _ = self.defs["display_units"].implied_volatility_points(
                    quantiles, self.defs["FORWARD"], self.defs["STRIKE"], self.defs["MATURITY"],
                    self.defs["DISCOUNT"], self.defs["KIND"])
                np.testing.assert_allclose(line.get_xdata(), z[indices])
                np.testing.assert_allclose(line.get_ydata(), expected[1, indices])
        for figure in (fit, predictive):
            axis = figure.axes[7]
            self.assertEqual(axis.get_yscale(), "log")
            self.assertEqual(len(axis.patches), 2)
            for patch, method in zip(axis.patches, ("pymc", "amortizer")):
                histogram = record["methods"][method]["c_histogram"]
                np.testing.assert_allclose(patch.get_data().edges, histogram["edges"])
                np.testing.assert_allclose(patch.get_data().values, histogram["density"])

    def test_heston_market_predictive_medians_and_bands_use_the_paired_method_quantiles(self):
        record = self.defs["market_predictives"]
        observed = np.asarray(self.defs["CASES"]["market"])
        vega = np.asarray(self.defs["MARKET_VEGA"])
        expiry, z = np.asarray(self.defs["EXPIRY"]), np.asarray(self.defs["DELTA"])
        figure = self.defs["heston_market_predictive_figure"]
        for index, axis in enumerate(figure.axes[:7]):
            indices = np.flatnonzero(expiry == index)
            indices = indices[np.argsort(z[indices])]
            for method_index, method in enumerate(("pymc", "amortizer")):
                clean = 100 * (np.asarray(record["methods"][method]["clean_q05_q50_q95_raw"])
                               - observed) / vega
                noisy = 100 * (np.asarray(record["methods"][method]["noisy_q05_q50_q95_raw"])
                               - observed) / vega
                np.testing.assert_allclose(axis.lines[method_index].get_xdata(), z[indices])
                np.testing.assert_allclose(axis.lines[method_index].get_ydata(), clean[1, indices])
                for collection, expected in ((axis.collections[2 * method_index], noisy),
                                              (axis.collections[2 * method_index + 1], clean)):
                    paths = collection.get_paths()
                    self.assertEqual(len(paths), 1)
                    vertices = paths[0].vertices
                    for contract in indices:
                        boundary = vertices[np.isclose(vertices[:, 0], z[contract], rtol=0, atol=1e-12), 1]
                        self.assertGreaterEqual(len(boundary), 2)
                        np.testing.assert_allclose([boundary.min(), boundary.max()],
                                                   expected[[0, 2], contract])

    def test_parameter_and_noisy_panels_keep_paired_quantiles_and_common_scale(self):
        self.check_predictive_figure(self.defs, 0)

    def test_bergomi_recovery_shows_all_raw_cases_and_seven_parameters(self):
        record = self.defs["bergomi_results"]
        recovery = record["recovery"]
        truth = np.asarray(recovery["truth"])
        quantiles = np.asarray(recovery["q05_q50_q95"])
        figure = self.defs["bergomi_recovery_figure"]
        self.assertEqual(truth.shape, (800, 7))
        self.assertEqual(len(figure.axes), 8)
        for parameter, axis in enumerate(figure.axes[:7]):
            offsets = axis.collections[1].get_offsets()
            np.testing.assert_allclose(offsets[:, 0], truth[:, parameter])
            np.testing.assert_allclose(offsets[:, 1], quantiles[1, :, parameter])
            self.assertEqual(len(axis.collections[0].get_segments()), 40)
            np.testing.assert_allclose(axis.lines[0].get_xdata(), record["parameter_bounds"][parameter])
            np.testing.assert_allclose(axis.lines[0].get_ydata(), record["parameter_bounds"][parameter])

    def test_optional_product_pricing_starts_collapsed(self):
        self.assertIn("data-expanded='[]'", self.defs["use_section"].text)
        self.assertIn("Optional: forward-start and volatility-swap pricing", self.defs["use_section"].text)
        self.assertTrue(self.defs["use_summary"])

    def test_last_expiry_can_be_selected_without_a_model(self):
        # App.run overrides only the selector's cell; the saved posterior is fixed.
        definitions = dict(self.defs)
        definitions.pop("bergomi_uncertainty_figure")
        definitions["bergomi_expiry"] = SimpleNamespace(value=6)
        with patch.dict(os.environ, {"HESTON_BUNDLE": ""}):
            _, result = self.module.app.run(defs=definitions)
        self.check_predictive_figure(result, 6)

    def check_predictive_figure(self, definitions, expiry):
        record = definitions["bergomi_results"]
        geometry = record["geometry"]
        observed = np.asarray(geometry["observed_raw"])
        vega = np.asarray(definitions["MARKET_VEGA"])
        z = np.asarray(definitions["DELTA"])
        indices = np.flatnonzero(np.asarray(geometry["expiry_index"]) == expiry)
        indices = indices[np.argsort(z[indices])]
        figure = definitions["bergomi_uncertainty_figure"]
        self.assertEqual(len(figure.axes), 3)
        clean, noisy = figure.axes[1:]
        self.assertEqual(clean.get_title(), "Parameter uncertainty only")
        self.assertEqual(noisy.get_title(), "Parameters + observation noise")
        self.assertEqual(clean.get_ylim(), noisy.get_ylim())
        for axis, kind in [(clean, "clean"), (noisy, "noisy")]:
            for line, method in zip(axis.lines[:2], ["mcmc", "amortizer"]):
                quantiles = np.asarray(record["predictives"][method][f"{kind}_q05_q50_q95_raw"])
                expected = 100 * (quantiles[1, indices] - observed[indices]) / vega[indices]
                np.testing.assert_allclose(line.get_xdata(), z[indices])
                np.testing.assert_allclose(line.get_ydata(), expected)


if __name__ == "__main__":
    unittest.main()
