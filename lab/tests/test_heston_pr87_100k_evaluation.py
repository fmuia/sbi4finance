"""A versioned100k consumer preserves panel identities and endpoint restrictions."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import evaluate_heston_pr87_100k as evaluation
from scripts import evaluate_heston_pr87_reproduction as parent_evaluation
from scripts import render_heston_pr87_100k as report
from scripts import export_heston_pr87_100k_figures as figures

NAMES = ["initial_variance", "kappa", "theta", "vol_of_vol", "rho", "c"]
BOUNDS = [[.01, .16], [.1, 10.], [.01, .16], [.1, 3.], [-.95, -.1], [.5, 2.]]
TARGET = {"infer_c": True, "parameter_names": NAMES, "prior_bounds": BOUNDS, "quote_count": 54}


def run_fixture(root, best_epoch=200):
    training = root / "training"
    primary = training / "checkpoints" / "epoch-000200"
    best = training / "checkpoints" / "best-validation"
    for directory in (primary, best):
        directory.mkdir(parents=True)
        evaluation.fixed.write_json(directory / "checkpoint.json", {})
    curve = [{"epoch": i, "loss": 1 / i, "val_loss": 2 / i} for i in range(1, 201)]
    evaluation.fixed.write_json(training / "native-epoch-history.json", curve)
    run = {"format": "heston_pr87_100k_run_v1", "status": "complete", "completed_epochs": 200,
        "actual_optimizer_updates": 78200, "protocol": {"train_rows": 100000, "validation_rows": 100,
            "epochs": 200, "batch_size": 256}, "primary_checkpoint": str(primary),
        "best_validation_archive": {"epoch": best_epoch, "path": str(best)},
        "native_epoch_history_sha256": evaluation.fixed.digest(training / "native-epoch-history.json")}
    evaluation.fixed.write_json(training / "run.json", run)
    return run, primary, best


def population_fixture(root, label, rows, count):
    rng = np.random.default_rng(45)
    bounds = np.asarray(BOUNDS)
    truth = rng.uniform(bounds[:, 0], bounds[:, 1], (rows, 6))
    draws = truth[:, None, :] + rng.normal(size=(rows, count, 6)) * (bounds[:, 1] - bounds[:, 0]) * .2
    summary, quantiles = evaluation.population_summary(draws, truth, np.zeros((rows, 54)), TARGET, seed=17)
    path = root / (label + ".npz")
    evaluation.fixed.write_npz(path, {"truth": truth, "quantiles": quantiles.astype(np.float32)})
    summary["recovery_payload"] = evaluation.binding(path)
    return summary


def report_fixture(root, best_epoch=200):
    run, primary_checkpoint, secondary_checkpoint = run_fixture(root, best_epoch)
    panels = {label: population_fixture(root, label, rows, count)
        for label, rows, count in (("native_validation", 100, 300), ("population", 512, 512))}
    parent = {"format": "heston_pr87_reproduction_evaluation_v1", "status": "complete", "task": TARGET,
        "native_epoch": 200, **panels}
    parent_path = root / "parent.json"; evaluation.fixed.write_json(parent_path, parent)
    protocol = {"task": {"target": TARGET}, "evaluation": evaluation.evaluation_contract(TARGET),
        "parent_10k_evaluation": evaluation.binding(parent_path), "sources": {
            "renderer": evaluation.binding(report.__file__), "parent_renderer": evaluation.binding(report.parent_report.__file__)}}
    protocol_path = root / "protocol.json"; evaluation.fixed.write_json(protocol_path, protocol)
    primary = {"format": "heston_pr87_100k_evaluation_v1", "status": "complete", "task": TARGET,
        "protocol": evaluation.binding(protocol_path), "native_epoch": 200,
        "checkpoint": evaluation.binding(primary_checkpoint / "checkpoint.json"), "endpoint": "primary_final",
        "actual_optimizer_updates": 78200, "training_rows": 100000, "training_presentations": 20000000,
        "training_run": evaluation.binding(primary_checkpoint.parent.parent / "run.json"),
        "parent_10k_evaluation": protocol["parent_10k_evaluation"], "protected_inputs_unchanged": True,
        "new_reference_sampling": False, "reserved_arrays_opened": False, "cases": {},
        "matched_mcmc_report": {"not_opened_in_unit_fixture": True}, **panels}
    primary_path = root / "primary.json"; evaluation.fixed.write_json(primary_path, primary)
    secondary = deepcopy(primary)
    secondary.update(native_epoch=best_epoch, actual_optimizer_updates=best_epoch * 391,
        training_presentations=best_epoch * 100000, endpoint="secondary_best_validation",
        checkpoint=evaluation.binding(secondary_checkpoint / "checkpoint.json"))
    secondary_path = root / "secondary.json"; evaluation.fixed.write_json(secondary_path, secondary)
    return protocol_path, primary_path, secondary_path, primary


class EndpointTests(unittest.TestCase):
    def test_unchanged_sampling_contract_and_no_equal_compute_claim(self):
        new, old = evaluation.evaluation_contract(TARGET), parent_evaluation.evaluation_contract(TARGET)
        for key in ("native_validation_rows", "native_validation_draws", "native_sampling_seed",
                "historical_population_rows", "historical_draws_per_case", "historical_seed",
                "historical_population_sampling_seed", "historical_population_sbc_seed",
                "conditional_cases", "conditional_draws", "conditional_sampling_seed", "raw_draw_policy", "calibration", "order"):
            self.assertEqual(new[key], old[key], key)
        self.assertEqual(new["max_posterior_evaluation_seconds_per_endpoint"], 900)
        self.assertEqual(new["primary_presentations"], 20000000)
        self.assertFalse(new["equal_compute_comparison"])
        self.assertFalse(new["scientific_acceptance"])

    def test_final_and_distinct_minimum_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, primary, best = run_fixture(Path(temporary), best_epoch=93)
            self.assertEqual(evaluation.checkpoint_endpoint(primary, {"native_epoch": 200, "actual_optimizer_updates": 78200}, run), "primary_final")
            self.assertEqual(evaluation.checkpoint_endpoint(best, {"native_epoch": 93, "actual_optimizer_updates": 93 * 391}, run), "secondary_best_validation")
            for checkpoint, epoch in ((best, 94), (primary, 93), (primary.parent / "epoch-000100", 100)):
                with self.assertRaisesRegex(ValueError, "endpoint"):
                    evaluation.checkpoint_endpoint(checkpoint, {"native_epoch": epoch, "actual_optimizer_updates": epoch * 391}, run)
            with self.assertRaisesRegex(ValueError, "accounting"):
                evaluation.checkpoint_endpoint(primary, {"native_epoch": 200, "actual_optimizer_updates": 8000}, run)

    def test_duplicate_best_epoch200_is_never_secondary(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, _, best = run_fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "endpoint"):
                evaluation.checkpoint_endpoint(best, {"native_epoch": 200, "actual_optimizer_updates": 78200}, run)

    def test_raw_diagnostics_are_frozen_parent_helpers(self):
        self.assertIs(evaluation.raw_support, parent_evaluation.raw_support)
        self.assertIs(evaluation.population_summary, parent_evaluation.population_summary)
        self.assertIs(evaluation.descriptive_comparison, parent_evaluation.descriptive_comparison)


class CompactReportTests(unittest.TestCase):
    def test_two_full_endpoints_stay_compact_and_offline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, primary, secondary, _ = report_fixture(root, best_epoch=93)
            out = root / "report.html"
            with patch.object(report, "selected_references", return_value=None):
                summary = report.build(protocol, primary, out, secondary_path=secondary)
            document = out.read_text()
            self.assertLess(summary["report_bytes"], 1024 * 1024)
            self.assertEqual(summary["inline_svg_count"], 9)
            self.assertEqual(document.count('Three unchanged conditional cases:4096'), 2)
            self.assertIn("20 million versus2 million", document)
            self.assertIn("does not isolate a data-size effect at equal compute", document)
            self.assertIn("frozen parent full-covariance ZCA", document)
            self.assertIn("minimum-validation epoch93", document)
            self.assertIn("Official-quality readiness is not established", document)
            self.assertNotIn("<script", document)
            self.assertNotIn('src="http', document)
            self.assertFalse(summary["scientific_acceptance"])
            self.assertFalse(summary["promotion"])

    def test_distinct_minimum_requires_full_secondary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, primary, _, _ = report_fixture(root, best_epoch=93)
            with self.assertRaisesRegex(ValueError, "secondary diagnostics required"):
                report.build(protocol, primary, root / "report.html")

    def test_same_final_minimum_omits_secondary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, primary, secondary, _ = report_fixture(root)
            with patch.object(report, "selected_references", return_value=None):
                result = report.build(protocol, primary, root / "report.html")
            self.assertIsNone(result["secondary_evaluation"])
            with self.assertRaisesRegex(ValueError, "secondary diagnostics required"):
                report.build(protocol, primary, root / "duplicate.html", secondary_path=secondary)

    def test_readiness_checks_every_target_and_no_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, record = report_fixture(root)
            for panel in ("native_validation", "population"):
                record[panel]["coverage"]["0.9"]["wilson"] = [[.88, .92]] * 6
                record[panel]["rank_dkw"]["passed"] = [True] * 6
                record[panel]["support"] = {key: {"sample_count": 0, "value_count": 0} for key in ("outside_prior", "nonphysical", "nonfinite")}
            for case in evaluation.previous.CASES:
                record["cases"][case] = {"reference_ready": True, "comparison": {"status": "defined",
                    "standardized_mean_error": [.1] * 6, "width_ratio": [1.] * 6},
                    "support": {key: {"sample_count": 0, "value_count": 0} for key in ("outside_prior", "nonphysical", "nonfinite")}}
            self.assertTrue(report.readiness(record)["necessary_diagnostics_pass"])
            record["population"]["rank_dkw"]["passed"][0] = False
            result = report.readiness(record)
            self.assertFalse(result["necessary_diagnostics_pass"])
            self.assertFalse(result["official_quality_established"])
            self.assertFalse(result["scientific_acceptance"])
            self.assertFalse(result["promotion"])

    def test_saved_panels_must_match_parent_truths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, primary_path, _, record = report_fixture(root)
            record["population"]["truth_sha256"] = "changed"
            evaluation.fixed.write_json(primary_path, record)
            with self.assertRaisesRegex(ValueError, "preserved10k"):
                report.build(protocol, primary_path, root / "report.html")

    def test_matplotlib_export_has_new_identity_and100k_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, primary, _, _ = report_fixture(root)
            result = figures.build(protocol, primary, root / "figures")
            self.assertEqual(result["format"], "heston_pr87_100k_scientific_figures_v1")
            self.assertEqual(len(result["exports"]), 5)
            svg = (root / "figures" / "native-learning-curves.svg").read_text()
            self.assertIn("100,000 noisy datasets", svg)
            self.assertNotIn("10,000 noisy datasets", svg)
            self.assertTrue(result["protected_inputs_unchanged"])
            self.assertFalse(result["model_loaded"])


if __name__ == "__main__": unittest.main()
