"""Raw unbounded posterior diagnostics preserve the native six-target recipe."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import evaluate_heston_pr87_reproduction as evaluation
from scripts import render_heston_pr87_reproduction as report

NAMES = ["initial_variance", "kappa", "theta", "vol_of_vol", "rho", "c"]
BOUNDS = [[.01, .16], [.1, 10.], [.01, .16], [.1, 3.], [-.95, -.1], [.5, 2.]]
TARGET = {"infer_c": True, "parameter_names": NAMES, "prior_bounds": BOUNDS, "quote_count": 54}


def fixture(rows=100, count=30):
    rng = np.random.default_rng(45)
    bounds = np.asarray(BOUNDS)
    truth = rng.uniform(bounds[:, 0], bounds[:, 1], (rows, 6))
    draws = truth[:, None, :] + rng.normal(size=(rows, count, 6)) * (bounds[:, 1] - bounds[:, 0]) * .2
    return draws, truth, np.zeros((rows, 54))


class RawDiagnosticsTests(unittest.TestCase):
    def test_protocol_matches_native_recovery_and_historical_seeds_only_final(self):
        contract = evaluation.evaluation_contract(TARGET)
        self.assertEqual((contract["native_validation_rows"], contract["native_validation_draws"]), (100, 300))
        self.assertEqual((contract["historical_population_rows"], contract["historical_draws_per_case"]), (512, 512))
        self.assertEqual(contract["historical_seed"], 261112)
        self.assertEqual(contract["conditional_draws"], 4096)
        self.assertIn("primary final epoch200 only", contract["checkpoint_selection"])
        self.assertFalse(contract["new_reference_sampling"])
        self.assertEqual(contract["max_total_posterior_evaluation_seconds"], 900)

    def test_outside_prior_draws_remain_in_calibration_and_recovery(self):
        draws, truth, quotes = fixture()
        draws[0, 0] = [-1, 30, -1, 8, -2, -1]
        before = draws.copy()
        summary, quantiles = evaluation.population_summary(draws, truth, quotes, TARGET, seed=17)
        self.assertEqual(summary["calibration_status"], "defined")
        self.assertEqual(summary["raw_draws_used"], draws.shape[0] * draws.shape[1])
        self.assertEqual(summary["support"]["draws_discarded"], 0)
        self.assertEqual(summary["support"]["draws_clipped"], 0)
        self.assertGreater(summary["support"]["outside_prior"]["sample_count"], 0)
        np.testing.assert_array_equal(draws, before)
        np.testing.assert_array_equal(quantiles, np.quantile(before, [.05, .5, .95], axis=1))
        self.assertEqual(summary["coverage"], evaluation.validation.coverage(before, truth))
        self.assertEqual(summary["fractional_ranks"], evaluation.validation.sbc(before, truth, seed=17)["fractional_ranks"])

    def test_support_distinguishes_prior_from_physical_validity(self):
        midpoint = np.asarray(BOUNDS).mean(axis=1)
        draws = np.tile(midpoint, (4, 1))
        draws[0, 5] = .4
        draws[1, 4] = -1.2
        draws[2, 1] = -1
        draws[3, 0] = np.nan
        support = evaluation.raw_support(draws, BOUNDS, NAMES)
        self.assertEqual(support["outside_prior"]["sample_count"], 3)
        self.assertEqual(support["nonphysical"]["sample_count"], 3)
        self.assertEqual(support["outside_prior"]["per_dimension_count"], [0, 1, 0, 0, 1, 1])
        self.assertEqual(support["nonfinite"]["per_dimension_count"], [1, 0, 0, 0, 0, 0])

    def test_nonfinite_draw_makes_whole_panel_undefined_without_filtering(self):
        draws, truth, quotes = fixture()
        draws[0, 0, 0] = np.nan
        summary, quantiles = evaluation.population_summary(draws, truth, quotes, TARGET, seed=17)
        self.assertIsNone(quantiles)
        self.assertEqual(summary["calibration_status"], "undefined_nonfinite_raw_draws")
        self.assertEqual(summary["raw_draws_used"], 3000)
        self.assertEqual(summary["support"]["draws_discarded"], 0)
        self.assertIsNone(summary["coverage"])

    def test_nonphysical_products_skipped_but_all_six_targets_compared(self):
        draws, _, _ = fixture(rows=2)
        neural = draws[0].copy(); neural[0, 0] = -1
        comparison = evaluation.descriptive_comparison(neural, draws, NAMES, {"any": [0, 1]}, BOUNDS)
        self.assertEqual(comparison["status"], "defined")
        self.assertEqual(comparison["quantities"], NAMES)
        self.assertEqual(len(comparison["neural_mean"]), 6)
        self.assertEqual(comparison["derived_products"]["status"], "skipped")
        self.assertIn("no filtering", comparison["derived_products"]["reason"])
        self.assertEqual(comparison["dependence"]["unique_parameter_pairs"], 15)
        np.testing.assert_allclose(comparison["neural_mean"], neural.mean(axis=0))


class CompactReportTests(unittest.TestCase):
    def test_report_complete_offline_and_under_one_mib(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training = root / "training"; checkpoint = training / "checkpoints" / "epoch-000200"
            checkpoint.mkdir(parents=True)
            evaluation.fixed.write_json(checkpoint / "checkpoint.json", {})
            evaluation.fixed.write_json(training / "native-epoch-history.json", [{"epoch": i, "loss": 1 / i, "val_loss": 2 / i} for i in range(1, 201)])
            protocol = {"task": {"target": TARGET}, "evaluation": evaluation.evaluation_contract(TARGET),
                "sources": {"renderer": evaluation.binding(report.__file__)}}
            protocol_path = root / "protocol.json"; evaluation.fixed.write_json(protocol_path, protocol)
            record = {"format": "heston_pr87_reproduction_evaluation_v1", "status": "complete", "task": TARGET,
                "protocol": evaluation.binding(protocol_path), "native_epoch": 200, "checkpoint": evaluation.binding(checkpoint / "checkpoint.json"),
                "protected_inputs_unchanged": True, "new_reference_sampling": False, "reserved_arrays_opened": False,
                "cases": {}, "matched_mcmc_report": {"not_opened_in_unit_fixture": True}}
            for label, rows, count in (("native_validation", 100, 300), ("population", 512, 512)):
                draws, truth, quotes = fixture(rows=rows, count=count)
                summary, quantiles = evaluation.population_summary(draws, truth, quotes, TARGET, seed=17)
                path = root / (label + ".npz")
                evaluation.fixed.write_npz(path, {"truth": truth, "quantiles": quantiles.astype(np.float32)})
                summary["recovery_payload"] = evaluation.binding(path)
                record[label] = summary
            evaluation_path = root / "evaluation.json"; evaluation.fixed.write_json(evaluation_path, record)
            out = root / "report.html"
            with patch.object(report, "selected_references", return_value=None):
                summary = report.build(protocol_path, evaluation_path, out)
            document = out.read_text()
            self.assertLess(summary["report_bytes"], 1024 * 1024)
            self.assertEqual(summary["inline_svg_count"], 4)
            self.assertIn("FlowMatching objective", document)
            self.assertIn("cannot attribute changes to architecture alone", document)
            self.assertIn("nonfinite", document)
            self.assertNotIn("<script", document)
            self.assertNotIn('src="http', document)
            self.assertTrue(summary["protected_inputs_unchanged"])

    def test_renderer_rejects_drifted_recovery_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "recovery.npz"
            evaluation.fixed.write_npz(path, {"truth": np.zeros((100, 6)), "quantiles": np.zeros((3, 100, 6))})
            spec = evaluation.binding(path)
            evaluation.fixed.write_npz(path, {"truth": np.ones((100, 6))})
            with self.assertRaisesRegex(ValueError, "identity differs"):
                report.load_recovery({"calibration_status": "defined", "recovery_payload": spec}, {})


if __name__ == "__main__":
    unittest.main()
