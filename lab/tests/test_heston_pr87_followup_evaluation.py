"""300-epoch diagnostics preserve frozen panels and predeclared endpoints."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import evaluate_heston_pr87_followup as evaluation
from scripts import evaluate_heston_pr87_100k as parent_evaluation
from scripts import render_heston_pr87_followup as report
from scripts import render_heston_pr87_100k as parent_report
from scripts import export_heston_pr87_followup_figures as figures
from tests.test_heston_pr87_100k_evaluation import TARGET, report_fixture as old_fixture


def fixture(root, rows=100000, best_epoch=300):
    protocol_path, primary_path, secondary_path, primary = old_fixture(root)
    protocol = evaluation.read(protocol_path)
    prior_path = root / "prior100k.json"
    evaluation.fixed.write_json(prior_path, primary)
    protocol["prior_100k_evaluation"] = evaluation.binding(prior_path)
    protocol["evaluation"] = evaluation.evaluation_contract(TARGET)
    protocol["sources"] = {"renderer": evaluation.binding(report.__file__), "parent_renderer": evaluation.binding(parent_report.__file__)}
    evaluation.fixed.write_json(protocol_path, protocol)
    training = Path(primary["training_run"]["path"]).parent
    final = training / "checkpoints" / "epoch-000300"
    final.mkdir(); evaluation.fixed.write_json(final / "checkpoint.json", {})
    curve_path = training / "native-epoch-history.json"
    evaluation.fixed.write_json(curve_path, [{"epoch": i, "loss": 1 / i, "val_loss": 2 / i} for i in range(1, 301)])
    run = evaluation.read(training / "run.json")
    run.update(format="heston_pr87_followup_run_v1", completed_epochs=300,
        actual_optimizer_updates=300 * ((rows + 255) // 256), primary_checkpoint=str(final),
        native_epoch_history_sha256=evaluation.fixed.digest(curve_path))
    run["protocol"].update(train_rows=rows, epochs=300)
    run["best_validation_archive"]["epoch"] = best_epoch
    evaluation.fixed.write_json(training / "run.json", run)
    primary.update(format="heston_pr87_followup_evaluation_v1", native_epoch=300,
        actual_optimizer_updates=run["actual_optimizer_updates"], training_rows=rows, training_presentations=300 * rows,
        training_accounting=evaluation.training_accounting(run), protocol=evaluation.binding(protocol_path),
        training_run=evaluation.binding(training / "run.json"), prior_100k_evaluation=protocol["prior_100k_evaluation"],
        checkpoint=evaluation.binding(final / "checkpoint.json"))
    evaluation.fixed.write_json(primary_path, primary)
    secondary = deepcopy(primary)
    secondary.update(native_epoch=best_epoch, actual_optimizer_updates=best_epoch * ((rows + 255) // 256),
        training_presentations=best_epoch * rows, endpoint="secondary_best_validation",
        checkpoint=evaluation.binding(Path(run["best_validation_archive"]["path"]) / "checkpoint.json"))
    evaluation.fixed.write_json(secondary_path, secondary)
    return protocol_path, primary_path, secondary_path, primary, run


class EndpointTests(unittest.TestCase):
    def test_sampling_contract_unchanged_except_declared_training(self):
        new, old = evaluation.evaluation_contract(TARGET), parent_evaluation.evaluation_contract(TARGET)
        for key in ("native_validation_rows", "native_validation_draws", "native_sampling_seed",
                "historical_population_rows", "historical_draws_per_case", "historical_seed",
                "historical_population_sampling_seed", "historical_population_sbc_seed", "conditional_cases",
                "conditional_draws", "conditional_sampling_seed", "raw_draw_policy", "calibration", "order"):
            self.assertEqual(new[key], old[key], key)
        self.assertEqual(new["supported_train_rows"], [100000, 200000])
        self.assertEqual(new["epochs"], 300)
        self.assertEqual(new["max_posterior_evaluation_seconds_per_endpoint"], 900)
        self.assertFalse(new["equal_compute_comparison"])
        self.assertFalse(new["scientific_acceptance"])

    def test_both_sizes_count_every_tail_and_only_final_or_minimum(self):
        with tempfile.TemporaryDirectory() as temporary:
            for rows, steps, tail in ((100000, 391, 160), (200000, 782, 64)):
                _, _, _, primary, run = fixture(Path(temporary) / str(rows), rows, 287)
                accounting = evaluation.training_accounting(run)
                self.assertEqual(accounting["primary_optimizer_updates"], steps * 300)
                self.assertEqual(accounting["primary_presentations"], rows * 300)
                self.assertEqual(accounting["tail_batch_rows"], tail)
                self.assertEqual(evaluation.checkpoint_endpoint(Path(primary["checkpoint"]["path"]).parent, primary, run), "primary_final")
                best = run["best_validation_archive"]["path"]
                self.assertEqual(evaluation.checkpoint_endpoint(best, {"native_epoch": 287, "actual_optimizer_updates": steps * 287}, run), "secondary_best_validation")
                for epoch in (100, 200, 250, 286):
                    with self.assertRaisesRegex(ValueError, "endpoint"):
                        evaluation.checkpoint_endpoint(best, {"native_epoch": epoch, "actual_optimizer_updates": steps * epoch}, run)

    def test_duplicate_final_and_old_update_accounting_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, _, primary, run = fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "endpoint"):
                evaluation.checkpoint_endpoint(run["best_validation_archive"]["path"], primary, run)
            primary["actual_optimizer_updates"] = 78200
            with self.assertRaisesRegex(ValueError, "accounting"):
                evaluation.checkpoint_endpoint(run["primary_checkpoint"], primary, run)
            run["protocol"]["train_rows"] = 50000
            with self.assertRaisesRegex(ValueError, "100k or200k"):
                evaluation.training_accounting(run)

    def test_frozen_raw_policies_and_necessary_screens_reused(self):
        self.assertIs(evaluation.raw_support, parent_evaluation.raw_support)
        self.assertIs(evaluation.population_summary, parent_evaluation.population_summary)
        self.assertIs(report.readiness, parent_report.readiness)


class SavedConsumerTests(unittest.TestCase):
    def test200k_two_complete_endpoints_compact_offline_and_baselines_visible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, primary, secondary, _, _ = fixture(root, 200000, 287)
            out = root / "report.html"
            with patch.object(report, "selected_references", return_value=None):
                result = report.build(protocol, primary, out, secondary_path=secondary)
            text = out.read_text()
            self.assertLess(result["report_bytes"], 1024 * 1024)
            self.assertEqual(result["inline_svg_count"], 9)
            for expected in ("100k parent — final epoch200", "200k primary — final epoch300", "60 million primary training presentations",
                    "same 200,000 noisy datasets", "frozen ZCA matrix fitted on 10k rows", "minimum-validation epoch287"):
                self.assertIn(expected, text)
            self.assertNotIn("<script", text)
            self.assertFalse(result["scientific_acceptance"])
            self.assertFalse(result["promotion"])

    def test_secondary_present_exactly_when_minimum_differs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, primary, _, _, _ = fixture(root / "different", best_epoch=287)
            with self.assertRaisesRegex(ValueError, "secondary diagnostics required"):
                report.build(protocol, primary, root / "missing.html")
            protocol, primary, secondary, _, _ = fixture(root / "same")
            with patch.object(report, "selected_references", return_value=None):
                result = report.build(protocol, primary, root / "same.html")
            self.assertIsNone(result["secondary_evaluation"])
            with self.assertRaisesRegex(ValueError, "secondary diagnostics required"):
                report.build(protocol, primary, root / "extra.html", secondary_path=secondary)

    def test_prior100k_panel_truth_identity_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path, primary_path, _, primary, _ = fixture(root)
            protocol = evaluation.read(protocol_path)
            prior_path = Path(protocol["prior_100k_evaluation"]["path"])
            prior = evaluation.read(prior_path); prior["population"]["truth_sha256"] = "changed"
            evaluation.fixed.write_json(prior_path, prior)
            protocol["prior_100k_evaluation"] = evaluation.binding(prior_path)
            evaluation.fixed.write_json(protocol_path, protocol)
            primary["protocol"] = evaluation.binding(protocol_path); primary["prior_100k_evaluation"] = protocol["prior_100k_evaluation"]
            evaluation.fixed.write_json(primary_path, primary)
            with self.assertRaisesRegex(ValueError, "preserved10k/100k"):
                report.build(protocol_path, primary_path, root / "drift.html")

    def test_exact_rare_conditional_support_and_full300_epoch_history(self):
        text = report.conditional_support_table({"cases": {"market": {"support": {"samples": 4096,
            "outside_prior": {"sample_count": 2, "sample_fraction": 2 / 4096},
            "nonphysical": {"sample_count": 0}, "nonfinite": {"sample_count": 0}}}}})
        self.assertIn("0.0488%", text)
        with self.assertRaisesRegex(ValueError, "300-epoch"):
            report.checked_history([{"epoch": i, "loss": 1., "val_loss": 1.} for i in range(1, 201)])

    def test_matplotlib_full300_epoch_correct_size_and_new_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, primary, _, _, _ = fixture(root, 200000)
            result = figures.build(protocol, primary, root / "figures")
            self.assertEqual(result["format"], "heston_pr87_followup_scientific_figures_v1")
            self.assertEqual(len(result["exports"]), 5)
            svg = (root / "figures" / "native-learning-curves.svg").read_text()
            self.assertIn("200,000 noisy datasets", svg)
            self.assertIn("epoch300 is primary", svg)
            self.assertFalse(result["model_loaded"])
            self.assertFalse(result["new_neural_sampling"])


if __name__ == "__main__": unittest.main()
