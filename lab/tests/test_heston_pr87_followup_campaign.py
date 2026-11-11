"""Guard frozen two-fit scope, primary-only branching and hard worker limits."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import scripts
from scripts import run_heston_pr87_followup as controller


class FollowupCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        git = patch.object(controller.subprocess, "check_output", return_value="fixture-revision\n")
        git.start()
        self.addCleanup(git.stop)
        self.root = Path(self.temp.name)
        self.input = self.root / "preserved.json"
        self.input.write_text('{}\n')
        self.bound = controller.identity(self.input)
        self.sources = {"controller": controller.identity(controller.__file__)}
        for key in ("producer", "evaluator", "renderer"):
            path = self.root / (key + ".py")
            path.write_text("# never executed: supervised worker fixture\n")
            self.sources[key] = controller.identity(path)
        preserved = self.root / "preserved-artifacts-before.json"
        preserved.write_text(json.dumps({"files": [{**self.bound, "bytes": self.input.stat().st_size}]}))
        self.preserved = controller.identity(preserved)
        self.campaign = {"format": controller.CAMPAIGN_FORMAT, "limits": controller.LIMITS,
            "branch_policy": controller.BRANCH_POLICY, "predeclared_before_first_fit": True,
            "scientific_acceptance": False, "sources": self.sources, "preserved_artifacts": self.preserved,
            "parent100k": str(self.root / "fixed-parent100k"), "runs": {},
            "branch_proof_path": str(self.root / "branch-proof.json")}
        self.studies = {}
        for name, rows in controller.RUNS:
            evidence = self.root / name
            payload = self.root / (name + "-payload")
            evidence.mkdir(); payload.mkdir()
            task = evidence / "task.json"
            task.write_text('{}\n')
            study = {"format": controller.FORMAT, "campaign": controller.LIMITS,
                "branch_policy": controller.BRANCH_POLICY, "endpoint_policy": controller.ENDPOINT_POLICY,
                "predeclared_before_fit": True, "predeclared_before_first_campaign_fit": True,
                "scientific_acceptance": False,
                "contract": {"format": "heston_pr87_followup_protocol_v1", "train_rows": rows,
                    "epochs": 300, "validation_rows": 100, "batch_size": 256},
                "sources": self.sources, "task": {}, "task_path": str(task), "task_file": controller.identity(task),
                "parent_training": self.campaign["parent100k"], "parent_study": self.bound,
                "parent_10k_evaluation": self.bound, "prior_100k_evaluation": self.bound,
                "original_whitening_study": self.bound, "historical_population_inputs": self.bound,
                "preserved_artifacts": self.preserved, "output": str(payload / "training"),
                "training_log": str(payload / "training.log"), "evaluations": {}}
            for role in ("primary", "secondary"):
                study["evaluations"][role] = {"checkpoint": "epoch-000300" if role == "primary" else "best-validation",
                    "output": str(evidence / role), "payload_output": str(payload / role), "log": str(payload / (role + ".log"))}
            self.studies[name] = study
            self.persist_study(name, rows)
        self.protocol = self.root / "campaign-protocol.json"
        self.persist_campaign()

    def persist_study(self, name, rows):
        path = self.root / name / "protocol.json"
        path.write_text(json.dumps(self.studies[name]))
        self.campaign["runs"][name] = {"train_rows": rows, "protocol": controller.identity(path)}

    def persist_campaign(self):
        self.protocol.write_text(json.dumps(self.campaign))

    def fake_worker(self, *, primary_pass, best_epoch=250, fail_first=False, timeout=False):
        calls = []
        class Worker:
            def __init__(self, command, **kwargs):
                calls.append(command)
                self.pid, self.returncode = 8000 + len(calls), int(fail_first and len(calls) == 1)
                self.command = command
                if self.returncode:
                    return
                study = json.loads(Path(command[command.index("--protocol") + 1]).read_text())
                target = Path(command[command.index("--out") + 1])
                target.mkdir()
                if "--payload-out" not in command:
                    record = {"status": "complete", "best_validation_archive": {"epoch": best_epoch}}
                    filename = "training.json"
                else:
                    role = "primary" if "epoch-000300" in command[command.index("--checkpoint") + 1] else "secondary"
                    record = {"format": "heston_pr87_followup_evaluation_v1", "status": "complete",
                        "endpoint": "primary_final" if role == "primary" else "secondary_best_validation",
                        "native_epoch": 300 if role == "primary" else best_epoch,
                        "training_rows": study["contract"]["train_rows"],
                        # A passing secondary must never rescue a failing primary.
                        "fixture_readiness": primary_pass if role == "primary" else True}
                    filename = "evaluation.json"
                (target / filename).write_text(json.dumps(record))

            def wait(self, timeout=None):
                expected = 900 if "--payload-out" in self.command else 3600
                if timeout != expected:
                    raise AssertionError("worker wall cap differs")
                if timeout_error:
                    raise subprocess.TimeoutExpired(self.command, timeout)
                return self.returncode

        timeout_error = timeout
        return Worker, calls

    def launch_fixture(self, *, primary_pass, best_epoch=250, fail_first=False, timeout=False):
        worker, calls = self.fake_worker(primary_pass=primary_pass, best_epoch=best_epoch,
                                        fail_first=fail_first, timeout=timeout)
        renderer = SimpleNamespace(__file__=self.sources["renderer"]["path"],
            readiness=lambda record: {"necessary_diagnostics_pass": record["fixture_readiness"],
                                      "scientific_acceptance": False})
        with patch.dict(sys.modules, {"scripts.render_heston_pr87_followup": renderer}), patch.object(
                scripts, "render_heston_pr87_followup", renderer, create=True), patch.object(
                controller.subprocess, "Popen", worker), patch.object(controller.shared, "stop"):
            result = controller.run(self.protocol, self.root / "campaign.json")
        return result, calls

    def test_pass_stops_after_first_fit_even_with_distinct_secondary(self):
        before = {name: spec["protocol"]["sha256"] for name, spec in self.campaign["runs"].items()}
        result, calls = self.launch_fixture(primary_pass=True)
        self.assertEqual(len(calls), 3)
        self.assertEqual(list(result["runs"]), ["100k-300"])
        self.assertFalse(result["run200k"])
        self.assertFalse(result["scientific_acceptance"])
        self.assertFalse((Path(self.studies["200k-300"]["output"])).exists())
        for name, spec in self.campaign["runs"].items():
            self.assertEqual(controller.identity(spec["protocol"]["path"])["sha256"], before[name])

    def test_failing_primary_runs_exactly_second_fit_and_keeps_secondary(self):
        result, calls = self.launch_fixture(primary_pass=False)
        self.assertEqual(len(calls), 6)
        self.assertEqual(list(result["runs"]), ["100k-300", "200k-300"])
        self.assertTrue(result["run200k"])
        proof = json.loads((self.root / "branch-proof.json").read_text())
        self.assertFalse(proof["necessary_diagnostics_pass"])
        self.assertFalse(proof["secondary_influenced_decision"])
        self.assertIn("100k-300/primary/evaluation.json", proof["primary_evaluation"]["path"])
        self.assertFalse(result["scientific_acceptance"])

    def test_same_final_minimum_never_gets_duplicate_secondary_queries(self):
        result, calls = self.launch_fixture(primary_pass=False, best_epoch=300)
        self.assertEqual(len(calls), 4)
        self.assertTrue(all("secondary_skipped" in value for value in result["runs"].values()))

    def test_failures_stop_without_fallback_fit_or_retry(self):
        worker, calls = self.fake_worker(primary_pass=False, fail_first=True)
        with patch.object(controller.subprocess, "Popen", worker), patch.object(controller.shared, "stop"):
            with self.assertRaisesRegex(RuntimeError, "job failed"):
                controller.run(self.protocol, self.root / "campaign.json")
            with self.assertRaisesRegex(FileExistsError, "no retries"):
                controller.run(self.protocol, self.root / "campaign.json")
        self.assertEqual(len(calls), 1)

    def test_training_timeout_stops_worker_and_refuses_fallback(self):
        worker, calls = self.fake_worker(primary_pass=False, timeout=True)
        with patch.object(controller.subprocess, "Popen", worker), patch.object(controller.shared, "stop") as stop:
            with self.assertRaisesRegex(TimeoutError, "wall cap"):
                controller.run(self.protocol, self.root / "campaign.json")
        self.assertEqual(len(calls), 1)
        self.assertTrue(stop.called)

    def test_changed_frozen_or_preserved_source_refuses_before_claim(self):
        controller.validate(self.campaign)
        self.input.write_text('{"changed":true}\n')
        with patch.object(controller.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "changed"):
                controller.run(self.protocol, self.root / "campaign.json")
            launch.assert_not_called()
        self.assertFalse((self.root / "campaign.claim.json").exists())

    def test_existing_conditional_output_refuses_even_if_first_could_pass(self):
        Path(self.studies["200k-300"]["evaluations"]["secondary"]["payload_output"]).mkdir()
        with patch.object(controller.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(FileExistsError, "no retries"):
                controller.run(self.protocol, self.root / "campaign.json")
            launch.assert_not_called()

    def test_changed_scope_and_shared_outputs_refuse(self):
        modified = copy.deepcopy(self.campaign)
        modified["branch_policy"] = {**modified["branch_policy"], "secondary_influences_branch": True}
        with self.assertRaisesRegex(ValueError, "predeclared"):
            controller.validate(modified)
        self.studies["200k-300"]["evaluations"]["primary"]["output"] = self.studies["100k-300"]["evaluations"]["primary"]["output"]
        self.persist_study("200k-300", 200000)
        with self.assertRaisesRegex(ValueError, "distinct"):
            controller.validate(self.campaign)

    def test_branch_refuses_incomplete_or_secondary_evaluation(self):
        path = self.root / "bad-evaluation.json"
        path.write_text(json.dumps({"format": "heston_pr87_followup_evaluation_v1", "status": "complete",
            "endpoint": "secondary_best_validation", "native_epoch": 250, "training_rows": 100000}))
        fake = SimpleNamespace(readiness=lambda _: {"necessary_diagnostics_pass": False})
        with patch.dict(sys.modules, {"scripts.render_heston_pr87_followup": fake}), patch.object(
                scripts, "render_heston_pr87_followup", fake, create=True):
            with self.assertRaisesRegex(ValueError, "primary evaluation"):
                controller.branch_decision(path)


if __name__ == "__main__":
    unittest.main()
