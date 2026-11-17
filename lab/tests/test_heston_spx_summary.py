"""The SPX summary section runs from saved data and stays bound to its baseline."""

from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


LAB = Path(__file__).resolve().parents[1]
NOTEBOOK = LAB / "notebooks/heston_spx_calibration.py"


@unittest.skipUnless(importlib.util.find_spec("marimo") and importlib.util.find_spec("bayesflow"),
                     "requires the core notebook environment")
class HestonSPXSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts import heston_grid_collaborator as delivery

        spec = importlib.util.spec_from_file_location("spx_summary_integration_under_test", NOTEBOOK)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        with (patch.dict(os.environ, {"HESTON_BUNDLE": ""}),
              patch.object(delivery, "load_checkpoint", side_effect=AssertionError("private checkpoint accessed")) as checkpoint,
              patch.object(delivery, "load_bank", side_effect=AssertionError("private bank accessed")) as bank):
            spec.loader.exec_module(module)
            cls.outputs, cls.defs = module.app.run()
            checkpoint.assert_not_called()
            bank.assert_not_called()
        cls.bind = staticmethod(cls.defs["bind_summary_study"])

    @classmethod
    def tearDownClass(cls):
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_saved_investigation_renders_without_private_model_or_bank(self):
        self.assertTrue(self.outputs)
        self.assertIsNone(self.defs["BUNDLE"])
        self.assertIsNone(self.defs["estimator"])
        self.assertEqual(len(self.defs["summary_runs"]), 19)
        self.assertEqual(len(self.defs["summary_primary_runs"]), 10)
        self.assertEqual(self.defs["summary_parameter"].value, 5)
        self.assertEqual(set(self.defs["summary_dimensions"].value), {8, 7, 6, 5, 4})
        self.assertEqual(self.defs["summary_model"].value, 6)
        self.assertEqual(set(self.defs["summary_recovery_figures"]), {None, 64, 32, 16, 8, 7, 6, 5, 4, 2})
        self.assertTrue(all(path.is_file() for path in self.defs["summary_recovery_figures"].values()))

    def test_binding_rejects_a_different_spx_task_budget_population_or_reference(self):
        arguments = [self.defs["summary_record"], self.defs["summary_runs"], self.defs["EVALUATED"],
                     self.defs["TRAINING"], self.defs["NAMES"], self.defs["BOUNDS"]]
        self.bind(*arguments)
        for field in ("names", "bounds", "budget", "truth", "observations", "training_run", "bank", "reference"):
            values = deepcopy(arguments)
            record, runs, baseline, training, names, bounds = values
            if field == "names":
                values[4] = ["other_parameter", *names[1:]]
            elif field == "bounds":
                values[5] = np.array(bounds, copy=True)
                values[5][0, 0] += .001
            elif field == "budget":
                training["rows"] -= 1
            elif field in ("truth", "observations"):
                baseline["population"][field + "_sha256"] = "0" * 64
            elif field == "training_run":
                baseline["training_run"]["sha256"] = "0" * 64
            elif field == "bank":
                baseline["common_bank"]["sha256"] = "0" * 64
            else:
                baseline["cases"]["synthetic-broad"]["reference_chain_payload"]["sha256"] = "0" * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.bind(*values)

    def test_main_comparison_uses_declared_final_checkpoints(self):
        primary = self.defs["summary_primary_runs"]
        self.assertTrue(all(row["endpoint"] == "primary_final" and row["evaluation"]["epoch"] == 300
                            for row in primary))
        by_dimension = self.defs["summary_by_dimension"]
        record = self.defs["summary_record"]
        for dimension in (7, 6, 5):
            fit = next(fit for fit in record["fits"] if fit["dimension"] == dimension)
            expected = fit["endpoints"]["primary_final"]["population"]["rmse"][-1]
            self.assertEqual(by_dimension[dimension]["evaluation"]["parameters"]["c"]["rmse"], expected)
        five = by_dimension[5]["evaluation"]["parameters"]
        six = by_dimension[6]["evaluation"]["parameters"]
        self.assertGreater(five["c"]["rmse"], 3 * six["c"]["rmse"])
        self.assertAlmostEqual(five["c"]["median_width90"] / record["protocol"]["population"]["prior_width90"][-1], 1, delta=.02)
        self.assertEqual(len([row for row in self.defs["summary_runs"]
                              if row["endpoint"] == "secondary_validation_minimum"]), 9)


if __name__ == "__main__":
    unittest.main()
