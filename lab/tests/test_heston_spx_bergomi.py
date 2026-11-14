"""The closing Bergomi study renders without payloads and preserves plot units."""

import importlib.util
import os
from pathlib import Path
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
        spreads = np.asarray(geometry["half_spread_raw"])
        expiry_index = np.asarray(geometry["expiry_index"])
        z = np.asarray(self.defs["DELTA"])
        figure = self.defs["bergomi_fit_figure"]
        self.assertEqual(len(figure.axes), 3)
        for axis, expiry in zip(figure.axes, [0, 3, 6]):
            indices = np.flatnonzero(expiry_index == expiry)
            indices = indices[np.argsort(z[indices])]
            for line, model in zip(axis.lines[:2], ["heston", "bergomi"]):
                expected = (np.asarray(record["fit_comparison"][model]["price_mean_raw"])
                            - observed) / spreads
                np.testing.assert_allclose(line.get_xdata(), z[indices])
                np.testing.assert_allclose(line.get_ydata(), expected[indices])

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
        spreads = np.asarray(geometry["half_spread_raw"])
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
                expected = (quantiles[1, indices] - observed[indices]) / spreads[indices]
                np.testing.assert_allclose(line.get_xdata(), z[indices])
                np.testing.assert_allclose(line.get_ydata(), expected)


if __name__ == "__main__":
    unittest.main()
