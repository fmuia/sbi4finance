"""Independent scale-up lanes, hard resource bounds and durable launch records."""

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


LAB = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


launcher = module("path_scaleup_launcher", LAB / "cluster/submit_path_scaleup.py")
reporter = module("path_scaleup_reporter", LAB / "cluster/report_path_scaleup.py")


def arguments(run, *, dry=False):
    args = ["submit_path_scaleup.py", "--run-root", str(run), "--cpu-python", sys.executable,
            "--gpu-python", sys.executable, "--source-commit", "frozen-source",
            "--runtime-lock-sha256", "a" * 64]
    return args + (["--dry-run"] if dry else [])


class ScaleupLauncherTests(unittest.TestCase):
    def specs(self):
        return launcher.job_specs(Path("/new-campaign"), "/cpu/python", "/gpu/python")

    def test_lanes_are_independent_with_only_necessary_artifact_dependencies(self):
        specs = self.specs()
        self.assertEqual(len(specs), 29)
        by_name = {spec["name"]: spec for spec in specs}
        self.assertEqual(by_name["software"]["dependencies"], [])
        for model in launcher.MODELS:
            prefix = model + "-"
            self.assertEqual(by_name[prefix + "pilot"]["dependencies"], [])
            self.assertEqual(by_name[prefix + "gpu-readiness"]["dependencies"], [])
            self.assertEqual(by_name[prefix + "bank"]["dependencies"], [prefix + "pilot"])
            self.assertEqual(by_name[prefix + "bank-check"]["dependencies"], [prefix + "bank"])
            for label in ("100k", "1m"):
                self.assertEqual(by_name[prefix + "fit-" + label]["dependencies"],
                                 [prefix + "bank-check", prefix + "gpu-readiness"])
                self.assertEqual(by_name[prefix + "evaluation-" + label]["dependencies"], [prefix + "fit-" + label])
            self.assertEqual(by_name[prefix + "comparison"]["dependency_condition"], "afterany")
            for spec in specs:
                if spec["model"] == model:
                    self.assertTrue(all(by_name[name]["model"] == model for name in spec["dependencies"]))
                    self.assertNotIn("software", spec["dependencies"])
        self.assertEqual(specs[-1]["dependency_condition"], "afterany")
        self.assertEqual(set(specs[-1]["dependencies"]), set(by_name) - {"report"})

    def test_reservations_include_every_cleanup_and_reject_overbudget_or_cross_lane_graph(self):
        specs = self.specs()
        budget = launcher.resource_budget(specs)
        self.assertAlmostEqual(budget["reserved_cpu_hours_including_cleanup"], 42.25)
        self.assertAlmostEqual(budget["reserved_gpu_hours_including_cleanup"], 7.85)
        self.assertEqual(budget["cleanup_seconds_per_job"], 240)
        excessive = copy.deepcopy(specs)
        next(spec for spec in excessive if "fit-100k" in spec["name"])["minutes"] = 90
        with self.assertRaisesRegex(RuntimeError, "ceiling"):
            launcher.resource_budget(excessive)
        crossed = copy.deepcopy(specs)
        next(spec for spec in crossed if spec["name"] == "rough-known-bank")["dependencies"] = ["heston-pilot"]
        with self.assertRaisesRegex(ValueError, "unrelated lanes"):
            launcher.resource_budget(crossed)

    def test_nested_training_pair_uses_streaming_same_seed_and_declared_exposure(self):
        specs = self.specs()
        for model in launcher.MODELS:
            fits = [spec for spec in specs if spec["model"] == model and "-fit-" in spec["name"]]
            self.assertEqual(len(fits), 2)
            values = []
            for spec in fits:
                command = spec["command"]
                value = lambda flag: command[command.index(flag) + 1]
                self.assertIn("--streaming", command)
                self.assertEqual(value("--cache-shards"), "2")
                self.assertEqual(value("--max-seconds"), "3300")
                self.assertEqual(value("--checkpoint-selection"), "final")
                self.assertEqual(value("--prefix-policy"), "uniform_log_mixture")
                self.assertEqual(spec["minutes"], 70)
                self.assertNotIn("--init-checkpoint", command)
                values.append((int(value("--training-limit")), int(value("--epochs")), value("--seed"), value("--bank")))
            self.assertEqual(values[0][:2], (100000, 50))
            self.assertEqual(values[1][:2], (1000000, 5))
            self.assertEqual(values[0][2:], values[1][2:])
            self.assertEqual(values[0][0] * values[0][1], values[1][0] * values[1][1])
            bank = next(spec for spec in specs if spec["name"] == model + "-bank")["command"]
            self.assertIn("--teaching", bank)
            self.assertEqual(bank[bank.index("--train") + 1], "1000000")
            self.assertEqual(bank[bank.index("--validation") + 1], "4000")
            self.assertEqual(bank[bank.index("--test") + 1], "4000")
            for spec in specs:
                if spec["model"] == model and "-evaluation-" in spec["name"]:
                    self.assertEqual(spec["command"][spec["command"].index("--cases") + 1], "1024")
        self.assertFalse(any("pmmh" in str(spec["command"]) for spec in specs))

    def test_sbatch_uses_absolute_source_and_correct_long_time_requests(self):
        args = types.SimpleNamespace(cpu_account="cpu-account", gpu_account="gpu-account",
            cpu_partition="batch", gpu_partition="gpu", cpu_qos=None, gpu_qos="norm-gpu",
            readiness_gpu_partition="gpu-debug", readiness_gpu_qos="gpu-debug", runtime_lock_sha256="runtime")
        specs = self.specs()
        identifiers = {spec["name"]: str(700 + i) for i, spec in enumerate(specs)}
        for spec in specs:
            command = launcher.sbatch_command(spec, args, Path("/run"), LAB, identifiers)
            self.assertIn(f"--chdir={LAB}", command)
            self.assertIn("--no-requeue", command)
            self.assertTrue(any(f"SBI4FINANCE_SOURCE_LAB={LAB}" in item for item in command))
            if "-fit-" in spec["name"]:
                self.assertIn("--time=01:10:00", command)
            elif spec["name"].endswith("-bank"):
                self.assertIn("--time=01:15:00", command)
            elif spec["name"].endswith("gpu-readiness"):
                self.assertIn("--partition=gpu-debug", command)
                self.assertIn("--time=00:05:00", command)
                self.assertIn("--qos=gpu-debug", command)
            if spec["dependencies"]:
                expected = "--dependency=" + spec["dependency_condition"] + ":" + ":".join(identifiers[name] for name in spec["dependencies"])
                self.assertIn(expected, command)

    def test_dry_run_is_read_only_and_prints_every_exact_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "new"
            output = io.StringIO()
            with patch("sys.argv", arguments(run, dry=True)), \
                    patch.object(launcher, "source_inventory", return_value={"pixi.lock": "source-sha"}), \
                    patch.object(launcher.subprocess, "run") as submit, contextlib.redirect_stdout(output):
                launcher.main()
            self.assertFalse(run.exists())
            submit.assert_not_called()
            self.assertEqual(sum(line.startswith("sbatch ") for line in output.getvalue().splitlines()), 29)

    def test_typed_gpu_requests_cover_readiness_and_fits_without_changing_budget(self):
        specs = self.specs()
        args = types.SimpleNamespace(cpu_account="carney-mjfrank-condo2", gpu_account="carney-mjfrank-condo2",
            cpu_partition="batch", gpu_partition="gpu-he", cpu_qos=None, gpu_qos="carney-gcondo2",
            readiness_gpu_partition=None, readiness_gpu_qos=None, runtime_lock_sha256="runtime", gpu_type="l40s")
        identifiers = {spec["name"]: str(700 + i) for i, spec in enumerate(specs)}
        gpu_jobs = 0
        for spec in specs:
            command = launcher.sbatch_command(spec, args, Path("/run"), LAB, identifiers)
            if spec["gpus"]:
                gpu_jobs += 1
                self.assertIn("--gres=gpu:l40s:1", command)
                self.assertIn("--partition=gpu-he", command)
                self.assertIn("--qos=carney-gcondo2", command)
            else:
                self.assertFalse(any(flag.startswith("--gres=") for flag in command))
        self.assertEqual(gpu_jobs, 9)
        self.assertEqual(launcher.resource_budget(specs)["reserved_gpu_hours_including_cleanup"], 7.85)

    def test_successful_submission_receipt_binds_all_jobs_runtime_and_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "new"
            with patch("sys.argv", arguments(run)), \
                    patch.object(launcher, "source_inventory", return_value={"pixi.lock": "source-sha"}), \
                    patch.object(launcher.subprocess, "run") as submit, contextlib.redirect_stdout(io.StringIO()):
                def response(*args, **kwargs):
                    receipt = json.loads((run / "submission-receipt.json").read_text())
                    self.assertEqual(receipt["jobs"][-1]["state"], "submission_attempt_pending")
                    self.assertEqual(kwargs["cwd"], LAB)
                    return types.SimpleNamespace(stdout=str(9000 + len(receipt["jobs"])) + "\n", stderr="", returncode=0)
                submit.side_effect = response
                launcher.main()
            receipt = json.loads((run / "submission-receipt.json").read_text())
            self.assertEqual(len(receipt["jobs"]), 29)
            self.assertTrue(receipt["submission_complete"])
            self.assertEqual(receipt["source_commit"], "frozen-source")
            self.assertEqual(receipt["runtime_lock_sha256"], "a" * 64)
            self.assertEqual(receipt["source_sha256"], {"pixi.lock": "source-sha"})
            self.assertFalse(receipt["scientific_acceptance"])
            self.assertFalse(receipt["reserved_test_evaluated"])
            self.assertFalse(receipt["automatic_retries"])
            self.assertTrue(all(job["state"] == "submitted" and job["stderr"] == "" for job in receipt["jobs"]))

    def test_submission_failure_retains_stderr_and_stops_without_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "new"
            with patch("sys.argv", arguments(run)), \
                    patch.object(launcher, "source_inventory", return_value={"pixi.lock": "source-sha"}), \
                    patch.object(launcher.subprocess, "run") as submit, contextlib.redirect_stdout(io.StringIO()):
                submit.side_effect = [types.SimpleNamespace(stdout="9001\n", stderr="", returncode=0),
                                      types.SimpleNamespace(stdout="", stderr="Invalid qos", returncode=1)]
                with self.assertRaisesRegex(RuntimeError, "Invalid qos"):
                    launcher.main()
            receipt = json.loads((run / "submission-receipt.json").read_text())
            self.assertEqual(submit.call_count, 2)
            self.assertFalse(receipt["submission_complete"])
            self.assertEqual(receipt["jobs"][0]["job_id"], "9001")
            self.assertEqual(receipt["jobs"][1]["stderr"], "Invalid qos")
            self.assertEqual(receipt["jobs"][1]["state"], "submission_uncertain_no_retry")
            with patch("sys.argv", arguments(run)), patch.object(launcher.subprocess, "run") as duplicate:
                with self.assertRaisesRegex(RuntimeError, "no jobs resubmitted"):
                    launcher.main()
            duplicate.assert_not_called()

    def test_report_retains_failed_and_missing_lanes_and_excludes_batch_accounting_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            receipt = {"source_commit": "source", "source_lab": str(LAB), "runtime_lock_sha256": "runtime",
                       "submission_complete": True, "budget": {"cleanup_seconds_per_job": 240},
                       "jobs": [{"name": "software", "job_id": "1", "state": "submitted", "cpus": 2, "gpus": 0, "minutes": 8},
                                {"name": "heston-fit-100k", "job_id": "2", "state": "submitted", "cpus": 4, "gpus": 1, "minutes": 70}]}
            (run / "submission-receipt.json").write_text(json.dumps(receipt))
            directory = run / "fits/heston/100k"
            directory.mkdir(parents=True)
            (directory / "training.json").write_text('{"epochs_completed":2,"optimizer_steps":1564,"status":"soft_wall_stop"}')
            (run / "comparison/heston").mkdir(parents=True)
            (run / "comparison/heston/comparison.json").write_text('{"status":"inconclusive_missing_large_fit"}')
            raw = "1|FAILED|1:0|10|2|cpu=2|1024K|\n1.batch|FAILED|1:0|10|2|cpu=2|1024K|\n2|TIMEOUT|0:15|4200|4|cpu=4,gres/gpu=1|2048K|\n"
            with patch.object(reporter.subprocess, "run", return_value=types.SimpleNamespace(stdout=raw, stderr="", returncode=0)):
                result = reporter.collect(run)
            self.assertEqual(result["accounting"]["actual_allocated_cpu_seconds_snapshot"], 20 + 4 * 4200)
            self.assertEqual(result["software"]["execution"]["status"], "FAILED")
            self.assertTrue(result["models"]["rough-known"]["bank"]["missing"])
            self.assertFalse(result["models"]["heston"]["exposure_comparison"]["both_planned_epoch_counts_completed"])
            self.assertEqual(result["models"]["heston"]["comparison"]["records"]["comparison.json"]["record"]["status"], "inconclusive_missing_large_fit")
            self.assertFalse(result["scientific_acceptance"])
            self.assertTrue((run / "scaleup-summary.json").exists())

    def test_accounting_metadata_failure_does_not_prevent_artifact_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            receipt = {"source_commit": "source", "source_lab": str(LAB), "runtime_lock_sha256": "runtime",
                       "budget": {"cleanup_seconds_per_job": 240}, "jobs": [
                           {"name": "software", "state": "submitted", "job_id": "1", "cpus": 2, "gpus": 0, "minutes": 8}]}
            (run / "submission-receipt.json").write_text(json.dumps(receipt))
            with patch.object(reporter.subprocess, "run", side_effect=subprocess.TimeoutExpired("sacct", 45)):
                result = reporter.collect(run)
            self.assertIsNone(result["accounting_query_returncode"])
            self.assertIn("TimeoutExpired", result["accounting_query_stderr"])
            self.assertEqual(result["accounting"]["jobs"]["software"]["status"], "accounting_unavailable")
            self.assertTrue((run / "scaleup-summary.json").exists())


if __name__ == "__main__":
    unittest.main()
