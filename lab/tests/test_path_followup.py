"""Follow-up budget, dependency and paired-comparison integrity checks."""

import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


LAB = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


launcher = module("path_followup_launcher", LAB / "cluster/submit_path_followup.py")
comparison = module("path_followup_comparison", LAB / "scripts/compare_path_refinement.py")
conditional = module("path_conditional_comparison", LAB / "scripts/compare_path_reference.py")


class FollowupTests(unittest.TestCase):
    def test_graph_has_only_two_bounded_gpu_candidates(self):
        specs = launcher.job_specs(Path("/old"), Path("/new"), "/cpu", "/gpu", "/reference")
        seen = set()
        for spec in specs:
            self.assertTrue(set(spec["dependencies"]) <= seen)
            seen.add(spec["name"])
        graphics = [spec for spec in specs if spec["gpus"]]
        self.assertEqual({spec["name"] for spec in graphics}, {"fit-heston", "fit-rough-joint"})
        for spec in graphics:
            self.assertEqual(spec["minutes"], 10)
            self.assertEqual(spec["command"][spec["command"].index("--checkpoint-selection") + 1], "final")
            self.assertEqual(spec["command"][spec["command"].index("--prefix-policy") + 1], "uniform_log_mixture")
        self.assertEqual(len(specs), 20)
        self.assertAlmostEqual(sum(s["minutes"] * s["cpus"] / 60 for s in specs), 315 / 60)
        self.assertAlmostEqual(sum((s["minutes"] * 60 + 240) * s["gpus"] for s in specs), 1680)
        self.assertEqual(specs[-1]["dependency_condition"], "afterany")

    def test_budget_rejects_old_spending_plus_new_exposure(self):
        with tempfile.TemporaryDirectory() as temporary:
            old = Path(temporary)
            (old / "pmmh").mkdir()
            (old / "submission-receipt.json").write_text(json.dumps({"runtime_lock_sha256": "sha",
                "jobs": [{"job_id": "1", "cpus": 4, "gpus": 1, "minutes": 50}]}))
            (old / "pmmh/submission-receipt.json").write_text('{"jobs": []}')
            specs = launcher.job_specs(old, old / "new", "/cpu", "/gpu", "/reference")
            lines = "1|COMPLETED|10000|4|cpu=4,gres/gpu=1|\n"
            lines += "8109953|FAILED|8|2|cpu=2|\n8109954|CANCELLED|0|0||\n8109969|FAILED|5|1|cpu=1|\n"
            with patch.object(launcher.subprocess, "run", return_value=types.SimpleNamespace(stdout=lines)):
                with self.assertRaisesRegex(RuntimeError, "ceiling"):
                    launcher.reconcile(old, specs)
            lines = lines.replace("|10000|", "|2434|")
            with patch.object(launcher.subprocess, "run", return_value=types.SimpleNamespace(stdout=lines)):
                result, runtime = launcher.reconcile(old, specs)
                self.assertLess(result["total_gpu_hours_including_cleanup"], 3)
                self.assertEqual(runtime, "sha")

    def test_paired_risk_detects_improvement_and_regression(self):
        truth = np.random.default_rng(8).normal(size=(256, 2))
        old, better, worse = truth + .5, truth + .1, truth + 1
        yes = comparison.paired_risk(old, better, truth, np.zeros(2), bootstraps=200)
        no = comparison.paired_risk(old, worse, truth, np.zeros(2), bootstraps=200)
        self.assertTrue(all(yes["nondegradation_pass"]))
        self.assertFalse(any(no["nondegradation_pass"]))
        with self.assertRaises(ValueError):
            comparison.paired_risk(old[:-1], better, truth, np.zeros(2))

    def test_exact_check_is_bound_to_the_evaluated_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "one-return.json").write_text(json.dumps({"task_sha256": "task", "checkpoint_sha256": "baseline"}))
            with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
                comparison.exact_screen(root, np.array(["case"]), slice(None),
                                        {"task_sha256": "task", "checkpoint_sha256": "candidate"})

    def test_interval_regression_cannot_hide_behind_unchanged_mean_risk(self):
        truth = np.zeros((100, 1))
        low, high = np.full_like(truth, -1), np.full_like(truth, 1)
        low[90:] = 1
        worse = low.copy()
        worse[70:] = 1
        self.assertFalse(comparison.interval_screen(low, high, worse, high, truth)["pass"][0])
        self.assertTrue(comparison.interval_screen(low, high, low, high, truth)["pass"][0])

    def test_failed_reference_gate_never_loads_missing_neural_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / "reference.json"
            summary.write_text('{"current":{"convergence_gates":{"passed":false}}}')
            result = conditional.run(types.SimpleNamespace(reference_summary=summary,
                chains=root / "missing-chains", out=root / "comparison"))
            self.assertEqual(result["status"], "inconclusive_reference")
            self.assertFalse(result["neural_models_sampled"])
            self.assertFalse(result["reference_accepted"])


if __name__ == "__main__":
    unittest.main()
