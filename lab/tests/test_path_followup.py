"""Follow-up budget, dependency and paired-comparison integrity checks."""

import contextlib
import importlib.util
import io
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
reporter = module("path_followup_reporter", LAB / "cluster/report_path_followup.py")


class FollowupTests(unittest.TestCase):
    def original_followup(self, root):
        old, original = root / "old", root / "original-followup"
        old.mkdir()
        (old / "pmmh").mkdir()
        original.mkdir()
        main = {"runtime_lock_sha256": "runtime-sha", "jobs": [
            {"name": name, "job_id": str(index + 1), "cpus": 4, "gpus": 1,
             "minutes": 15 if index == 0 else 50}
            for index, name in enumerate(("gpu-smoke", "fit-rough-known", "fit-rough-joint", "fit-heston"))]}
        (old / "submission-receipt.json").write_text(json.dumps(main))
        # Duplicate scheduler identity must not be charged a second time.
        (old / "pmmh/submission-receipt.json").write_text(json.dumps({"jobs": [main["jobs"][0]]}))
        specs = launcher.job_specs(old, original, "/cpu", "/gpu", "/reference")
        source = {"old_run": str(old), "run_root": str(original), "source_commit": "original-source",
                  "jobs": [{**spec, "job_id": str(100 + i)} for i, spec in enumerate(specs)]}
        (original / "submission-receipt.json").write_text(json.dumps(source))
        return old, original, main, source

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
        reducer = next(spec for spec in specs if spec["name"] == "pmmh-summary")
        self.assertEqual(reducer["dependency_condition"], "afterany")

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

    def test_repair_submits_thirteen_jobs_with_five_minute_gpu_caps_and_external_references(self):
        old, run, original = Path("/old"), Path("/repair"), Path("/original")
        specs = launcher.job_specs(old, run, "/cpu", "/gpu", "/reference", repair_of=original)
        self.assertEqual(len(specs), 13)
        self.assertFalse(set(s["name"] for s in specs) & set(launcher.REUSED_REFERENCE_JOBS))
        seen = {"pmmh-summary", "noise"}
        for spec in specs:
            self.assertTrue(set(spec["dependencies"]) <= seen)
            seen.add(spec["name"])
        graphics = [spec for spec in specs if spec["gpus"]]
        self.assertEqual(len(graphics), 2)
        for spec in graphics:
            self.assertEqual(spec["minutes"], 5)
            self.assertEqual(spec["command"][spec["command"].index("--max-seconds") + 1], "180")
        self.assertEqual(sum((s["minutes"] * 60 + 240) * s["gpus"] for s in specs), 1080)
        conditional_spec = next(s for s in specs if s["name"] == "conditional-heston")
        self.assertEqual(conditional_spec["command"][conditional_spec["command"].index("--chains") + 1],
                         str(original / "pmmh"))
        self.assertEqual(conditional_spec["command"][conditional_spec["command"].index("--reference-summary") + 1],
                         str(original / "pmmh-summary/summary.json"))
        dependency = launcher.dependency_argument(conditional_spec, {"pmmh-summary": "111", "fit-heston": "222"})
        self.assertEqual(dependency, "--dependency=afterok:111:222")
        self.assertEqual(specs[-1]["dependency_condition"], "afterany")
        self.assertTrue({"pmmh-summary", "noise"} <= set(specs[-1]["dependencies"]))

    def test_repair_accounting_includes_failed_original_gpu_jobs_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, original, main, source = self.original_followup(root)
            specs = launcher.job_specs(old, root / "repair", "/cpu", "/gpu", "/reference", repair_of=original)
            lines = "".join(f"{job['job_id']}|COMPLETED|{seconds}|4|cpu=4,gres/gpu=1|\n"
                            for job, seconds in zip(main["jobs"], (177, 2760, 2756, 2434)))
            for job in source["jobs"]:
                if job["gpus"]:
                    seconds = 48 if job["name"] == "fit-heston" else 34
                    lines += f"{job['job_id']}|FAILED|{seconds}|4|cpu=4,gres/gpu=1|\n"
                elif job["name"] in launcher.REUSED_REFERENCE_JOBS:
                    lines += f"{job['job_id']}|COMPLETED|10|{job['cpus']}|cpu={job['cpus']}|\n"
                else:
                    lines += f"{job['job_id']}|CANCELLED|0|0||\n"
            lines += "8109953|FAILED|8|2|cpu=2|\n8109954|CANCELLED|0|0||\n8109969|FAILED|5|1|cpu=1|\n"
            with patch.object(launcher.subprocess, "run", return_value=types.SimpleNamespace(stdout=lines)) as query:
                budget, _ = launcher.reconcile(old, specs, repair_of=original)
            self.assertAlmostEqual(budget["existing_gpu_hours_including_cleanup"] * 3600, 9649)
            self.assertAlmostEqual(budget["total_gpu_hours_including_cleanup"] * 3600, 10729)
            identifiers = query.call_args.args[0][4].split(",")
            self.assertEqual(len(identifiers), len(set(identifiers)))
            self.assertEqual(len(budget["repair_receipt_ancestry"]), 1)
            too_much = lines.replace("|FAILED|34|4|", "|FAILED|134|4|")
            with patch.object(launcher.subprocess, "run", return_value=types.SimpleNamespace(stdout=too_much)):
                with self.assertRaisesRegex(RuntimeError, "ceiling"):
                    launcher.reconcile(old, specs, repair_of=original)

    def test_repair_submission_preserves_reused_ids_and_source_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, original, _, source = self.original_followup(root)
            destination = root / "repair"
            argv = ["submit_path_followup.py", "--old-run", str(old), "--run-root", str(destination),
                    "--cpu-python", "/cpu", "--gpu-python", "/gpu", "--reference-python", "/reference",
                    "--source-commit", "repaired-source", "--repair-of", str(original)]
            with patch("sys.argv", argv), patch.object(launcher, "reconcile", return_value=({}, "runtime-sha")), \
                    patch.object(launcher.subprocess, "run") as submit:
                submit.side_effect = [types.SimpleNamespace(stdout=f"{200 + i}\n") for i in range(13)]
                with contextlib.redirect_stdout(io.StringIO()):
                    launcher.main()
            receipt = json.loads((destination / "submission-receipt.json").read_text())
            self.assertEqual(len(receipt["jobs"]), 13)
            self.assertEqual(len(receipt["reused_jobs"]), 7)
            self.assertEqual(receipt["repair_of"], str(original.resolve()))
            self.assertEqual(receipt["reference_run"], str(original.resolve()))
            self.assertTrue(receipt["submission_complete"])
            external = {job["name"]: job["job_id"] for job in source["jobs"]}
            conditional_job = next(job for job in receipt["jobs"] if job["name"] == "conditional-heston")
            fit_job = next(job for job in receipt["jobs"] if job["name"] == "fit-heston")
            self.assertIn(f"--dependency=afterok:{external['pmmh-summary']}:{fit_job['job_id']}", conditional_job["sbatch"])
            final = receipt["jobs"][-1]
            dependency = next(x for x in final["sbatch"] if x.startswith("--dependency="))
            self.assertTrue(dependency.startswith("--dependency=afterany:"))
            self.assertIn(external["pmmh-summary"], dependency.split(":"))
            self.assertIn(external["noise"], dependency.split(":"))
            for reused in receipt["reused_jobs"]:
                self.assertEqual(reused["source_commit"], "original-source")
                self.assertEqual(len(reused["source_receipt"]["sha256"]), 64)

    def test_repair_report_reads_original_reference_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, original = root / "repair", root / "original"
            run.mkdir()
            (original / "pmmh-summary").mkdir(parents=True)
            (original / "noise").mkdir()
            (original / "pmmh-summary/summary.json").write_text('{"current":{"status":"reference-status"}}')
            (original / "noise/panel.json").write_text('{"points":[0,1,2,3]}')
            receipt = {"source_commit": "repair", "budget": {}, "repair_of": str(original),
                       "reference_run": str(original), "jobs": [{"job_id": "9"}],
                       "reused_jobs": [{"job_id": "2"}, {"job_id": "3"}]}
            (run / "submission-receipt.json").write_text(json.dumps(receipt))
            with patch("sys.argv", ["report", "--run-root", str(run)]), \
                    patch.object(reporter.subprocess, "run", return_value=types.SimpleNamespace(stdout="", stderr="", returncode=0)):
                reporter.main()
            result = json.loads((run / "followup-summary.json").read_text())
            self.assertEqual(result["artifacts"]["pmmh-summary/summary.json"]["current"]["status"], "reference-status")
            self.assertEqual(result["artifacts"]["noise/panel.json"]["points"], [0, 1, 2, 3])
            self.assertEqual(result["artifact_sources"]["pmmh-summary/summary.json"], str(original / "pmmh-summary/summary.json"))

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
