"""Independent CPU phases, bounded allocations and durable no-retry submission."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_path_workshop_acceptance as fixtures

LAB = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("path_closeout_launcher_tests", LAB / "cluster/submit_path_closeout.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class CloseoutLauncherTests(unittest.TestCase):
    def setUp(self):
        self.campaign = fixtures.WorkshopAcceptanceTests("test_selection_is_immutable_and_retains_the_known_development_limitations")
        self.campaign.setUp()
        self.addCleanup(self.campaign.doCleanups)
        self.run = self.campaign.run.parent / "closeout"
        self.out = self.campaign.run.parent / "submission.json"

    def arguments(self, phase="confirmation", *, dry=False, review=None, out=None):
        values = ["submit_path_closeout.py", "--source-lab", str(LAB), "--source-commit", "closeout-source",
            "--source-archive-sha256", "archive-hash", "--runtime-lock-sha256", "runtime-lock",
            "--python", sys.executable, "--run-root", str(self.run), "--phase", phase,
            "--selection", str(self.campaign.selection_path), "--out", str(out or self.out)]
        if dry:
            values += ["--dry-run"]
        if review:
            values += ["--review", str(review)]
        return values

    def execute(self, argv, responses=None):
        with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()), patch.object(
                launcher.subprocess, "run", side_effect=responses) as submit:
            value = launcher.main()
        return value, submit

    def test_each_phase_has_three_independent_cpu_jobs_and_all_six_fit_the_cap(self):
        for phase, minutes in (("confirmation", 30), ("final", 40)):
            specs = launcher.job_specs(self.campaign.selection, self.run, sys.executable, phase,
                self.campaign.selection_path, self.campaign.run.parent / "review.json")
            self.assertEqual(len(specs), 3)
            for job in specs:
                self.assertEqual(job["dependencies"], [])
                self.assertEqual((job["cpus"], job["gpus"], job["minutes"], job["memory"]), (2, 0, minutes, "12G"))
                self.assertIn("2976" if phase == "confirmation" else "4000", job["command"])
                self.assertIn("validation" if phase == "confirmation" else "test", job["command"])
        self.assertEqual(launcher.budget()["all_six_jobs_cpu_hours_including_cleanup"], 7.8)
        self.assertEqual(launcher.budget()["gpu_hours"], 0)

    def test_dry_run_checks_all_inputs_and_writes_no_guard_or_output(self):
        receipt, submit = self.execute(self.arguments(dry=True))
        submit.assert_not_called()
        self.assertFalse(self.run.exists())
        self.assertFalse(self.out.exists())
        self.assertFalse((Path(self.campaign.selection["state_directory"]) / "submission-confirmation.json").exists())
        self.assertFalse(receipt["submission_complete"])

    def test_success_records_each_attempt_and_refuses_duplicates_even_at_another_outpath(self):
        responses = [SimpleNamespace(stdout=f"{901 + i}\n", stderr="", returncode=0) for i in range(3)]
        receipt, submit = self.execute(self.arguments(), responses)
        self.assertEqual(submit.call_count, 3)
        self.assertTrue(receipt["submission_complete"])
        self.assertEqual([row["job_id"] for row in receipt["jobs"]], ["901", "902", "903"])
        attempts = self.out.with_name(self.out.name + ".attempts")
        self.assertEqual(len(list(attempts.glob("*.json"))), 6)
        self.assertEqual(json.loads((attempts / "00-heston.json").read_text())["state"], "submission_attempt_pending")
        self.assertIn("--chdir=" + str(LAB), receipt["jobs"][0]["sbatch"])
        self.assertFalse(any("--dependency" in arg or "--gres" in arg for arg in receipt["jobs"][0]["sbatch"]))
        with self.assertRaisesRegex(FileExistsError, "already claimed"):
            self.execute(self.arguments(out=self.out.with_name("alternative.json")))

    def test_partial_error_preserves_identifiers_and_never_resubmits(self):
        responses = [SimpleNamespace(stdout="901\n", stderr="", returncode=0),
                     SimpleNamespace(stdout="", stderr="Invalid account", returncode=1)]
        with self.assertRaisesRegex(RuntimeError, "failed sbatch"):
            self.execute(self.arguments(), responses)
        receipt = json.loads(self.out.read_text())
        self.assertFalse(receipt["submission_complete"])
        self.assertEqual(len(receipt["jobs"]), 2)
        self.assertEqual(receipt["jobs"][0]["job_id"], "901")
        self.assertEqual(receipt["jobs"][1]["stderr"], "Invalid account")
        with patch.object(launcher.subprocess, "run") as submit:
            with patch("sys.argv", self.arguments()), self.assertRaisesRegex(FileExistsError, "no resubmissions"):
                launcher.main()
        submit.assert_not_called()

    def test_timeout_is_durable_and_has_no_automatic_retry(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            self.execute(self.arguments(), [subprocess.TimeoutExpired("sbatch", 60, stderr=b"unknown submission outcome")])
        receipt = json.loads(self.out.read_text())
        self.assertEqual(receipt["jobs"][0]["state"], "submission_uncertain_no_retry")
        self.assertEqual(receipt["jobs"][0]["stderr"], "unknown submission outcome")

    def test_final_preflight_requires_review_but_does_not_consume_test_access(self):
        assessment, _ = self.campaign.assess(coverage=.8)
        review, _ = self.campaign.review(assessment, "evaluate_frozen_development_candidate")
        _, submit = self.execute(self.arguments("final", dry=True, review=review))
        submit.assert_not_called()
        self.assertFalse(list(Path(self.campaign.selection["state_directory"]).glob("test-opening-*.json")))
        self.assertFalse(self.run.exists())

    def test_changed_input_blocks_every_submission(self):
        model = Path(self.campaign.selection["models"]["rough-joint"]["checkpoint"]) / "model.keras"
        model.write_bytes(b"changed")
        with patch.object(launcher.subprocess, "run") as submit:
            with patch("sys.argv", self.arguments()), self.assertRaisesRegex(ValueError, "frozen artifact"):
                launcher.main()
        submit.assert_not_called()
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
