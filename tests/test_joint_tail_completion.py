"""Metadata bridge regression: payload receipts alone never release training."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / "docs/pilot-results/2026-10-28/bank-delivery/tail-recovery/complete_joint_tail.py"
SPEC = importlib.util.spec_from_file_location("joint_tail_completion", SOURCE)
bridge = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(bridge)


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)
        self.native = self.root / "native"; self.native.mkdir()
        self.original = self.root / "original"; self.original.mkdir()
        self.stage = self.root / "stage"; self.stage.mkdir()
        config = {"snapshot": str(self.native), "cpu_hours": 1500, "prior_cpu_seconds": 3141589}
        self.config = self.root / "campaign.json"; bridge.save(self.config, config)
        jobs = [{"split": "validation", "start": 4925, "stop": 4950, "job_sha256": "a"},
                {"split": "test", "start": 0, "stop": 25, "job_sha256": "b"}]
        configuration = {"root_seed": 271200, "splits": {"train": 20000, "validation": 5000, "test": 7000}}
        self.jobs = self.root / "jobs.json"; bridge.save(self.jobs, {"configuration": configuration, "jobs": jobs})
        common = {"collection_id": "collection", "configuration": configuration,
                  "source_sha256": {"numerics": "fixed"}, "versions": {"numpy": "pinned"}, "task_fingerprint": "task"}
        for directory in (self.native, self.original):
            bridge.save(directory / "manifest.json", {**common, "parts": [{"split": "train", "start": 0, "stop": 25,
                "path": "../parts/train-0", "array_sha256": "unchanged", "manifest_sha256": "unchanged"}]})
        self.phase = {"collection_id": "collection", "root_seed": 271200, "splits": configuration["splits"],
                      "root": str(self.stage), "snapshot": str(self.original), "repo": str(self.root), "revision": "producer"}
        stage_plan = self.root / "stage-plan.json"; bridge.save(stage_plan, {"phases": {"joint": self.phase}})
        self.attempts = [{"role": "workers", "state": "submitted", "job_id": "20", "indices": [0, 1], "array_offset": 0},
                         {"role": "final", "state": "submitted", "job_id": "21"}]
        self.execution = self.root / "native-execution.json"
        bridge.save(self.execution, {"state": "completed_checked_snapshot", "planned_workers": 2,
            "config": {**config, "expected_operator_sha256": "frozen detached addition"}, "attempts": self.attempts,
            "snapshot": str(self.native), "snapshot_manifest_sha256": bridge.sha(self.native / "manifest.json")})
        self.plan = {"root": str(self.root), "recovery_execution": str(self.execution), "recovery_config": str(self.config),
            "recovery_jobs": str(self.jobs), "stage_plan": str(stage_plan), "prior_cpu_seconds": 3141589,
            "native_jobs": jobs, "tested_finalizer_sha256": "tested", "tested_finalizer": "tested.py", "python": "python",
            "readiness": str(self.root / "readiness.json"), "original_execution_sha256": "original-stopped",
            "prior_recovery_execution_sha256": "morning-stopped", "preserved_inventory_sha256": "1278-parts",
            "original_job_map_sha256": "original-map", "plan_sha256": "bridge-plan"}
        self.rows = {i: {"job_id": i, "state": "COMPLETED", "exit_code": "0:0", "cpus": 1, "elapsed": 10}
                     for i in ("20_0", "20_1", "21", "22")}
        self.final_attempt = {"role": "final", "state": "submitted", "job_id": "22"}
        bridge.save(self.stage / "final.json", {"status": "completed_checked_snapshot", "plan_sha256": bridge.sha(stage_plan),
            "driver_sha256": "tested", "preserved_parts_unchanged": 1278, "verified_parts": 1280,
            "exact_prior_context_noise_replay": True, "snapshot": str(self.original),
            "snapshot_manifest_sha256": bridge.sha(self.original / "manifest.json"), "split_counts": self.phase["splits"],
            "collection_id": "collection", "source_revision": "producer", "source_sha256": common["source_sha256"],
            "runtime": common["versions"], "task_fingerprint": "task"})

    def test_completed_receipts_wait_for_allocation_then_publish_without_overwrite(self):
        # Even an existing native snapshot/checked ledger cannot pass pending accounting.
        self.rows["20_1"]["state"] = "RUNNING"
        self.assertIsNone(bridge.generic_ready(self.plan, self.rows))
        self.rows["20_1"]["state"] = "COMPLETED"; generic = bridge.generic_ready(self.plan, self.rows)
        # Even a successful full-integrity report cannot release anything before terminal accounting.
        self.rows["22"]["state"] = "RUNNING"
        self.assertIsNone(bridge.completion_report(self.plan, generic, self.final_attempt, self.rows, "start"))
        self.assertFalse(Path(self.plan["readiness"]).exists())
        self.rows["22"].update(state="TIMEOUT", exit_code="0:0")
        with self.assertRaisesRegex(ValueError, "did not complete"):
            bridge.completion_report(self.plan, generic, self.final_attempt, self.rows, "start")
        self.rows["22"]["state"] = "COMPLETED"
        report = bridge.completion_report(self.plan, generic, self.final_attempt, {"22": self.rows["22"]}, "start")
        self.assertEqual(report["cumulative_allocated_cpu_seconds"], 3141589 + 40)
        bridge.publish(self.plan, report)
        self.assertEqual(bridge.sha(self.root / "completion.json"), bridge.sha(self.plan["readiness"]))
        self.assertFalse(report["scientific_acceptance"]); self.assertFalse(report["reserved_test_evaluated"])
        before = Path(self.plan["readiness"]).read_bytes()
        with self.assertRaises(FileExistsError): bridge.publish(self.plan, report)
        self.assertEqual(Path(self.plan["readiness"]).read_bytes(), before)

    def test_preserved_part_or_finalizer_identity_mismatch_blocks_completion(self):
        generic = bridge.generic_ready(self.plan, self.rows)
        report = bridge.read(self.stage / "final.json"); report["preserved_parts_unchanged"] = 1277
        bridge.save(self.stage / "final.json", report)
        with self.assertRaisesRegex(ValueError, "integrity result"):
            bridge.completion_report(self.plan, generic, self.final_attempt, {"22": self.rows["22"]}, "start")
        self.assertFalse(Path(self.plan["readiness"]).exists())

    def test_unknown_finalizer_submission_is_recorded_before_scheduler_and_not_retried(self):
        # Source contract requires absent destinations at submission.
        (self.stage / "final.json").unlink(); (self.original / "manifest.json").unlink(); self.original.rmdir()
        ledger = {}; calls = []
        def ambiguous(*args, **kwargs):
            calls.append(args)
            saved = bridge.read(self.root / "execution.json")
            self.assertEqual(saved["integrity_attempt"]["state"], "submission_attempted")
            raise TimeoutError("unknown scheduler response")
        with self.assertRaises(TimeoutError): bridge.submit_integrity(self.plan, ledger, self.root, run=ambiguous)
        self.assertEqual(ledger["integrity_attempt"]["state"], "unknown_submission")
        with self.assertRaisesRegex(ValueError, "already attempted"):
            bridge.submit_integrity(self.plan, ledger, self.root, run=ambiguous)
        self.assertEqual(len(calls), 1)

    def test_delayed_accounting_then_completion_keeps_one_finalizer_attempt(self):
        self.plan["frozen_sha256"] = {}
        plan_path = self.root / "bridge-plan.json"; bridge.save(plan_path, self.plan)
        calls = []
        def submit(plan, ledger, root):
            calls.append("one finalizer")
            ledger["integrity_attempt"] = self.final_attempt.copy()
        # First native query and first finalizer query are delayed; all receipts
        # remain untouched. The following read-only polls complete successfully.
        answers = [subprocess.TimeoutExpired("sacct", 45), self.rows,
                   subprocess.CalledProcessError(1, "sacct"), self.rows, {"22": self.rows["22"]}]
        with mock.patch.object(sys, "argv", ["bridge", "--plan", str(plan_path)]), \
             mock.patch.object(bridge, "accounting", side_effect=answers), \
             mock.patch.object(bridge, "submit_integrity", side_effect=submit), \
             mock.patch.object(bridge.time, "sleep"):
            bridge.main()
        self.assertEqual(calls, ["one finalizer"])
        ledger = bridge.read(self.root / "execution.json")
        self.assertEqual(ledger["state"], "completed_checked_snapshot")
        self.assertEqual(ledger["last_accounting_error"]["type"], "CalledProcessError")
        self.assertTrue(Path(self.plan["readiness"]).exists())


if __name__ == "__main__": unittest.main()
