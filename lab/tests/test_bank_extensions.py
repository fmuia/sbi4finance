"""Dependency/once-only tests; no numerical work or scheduler submission."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location("bank_extensions", Path(__file__).resolve().parents[1] / "scripts/arm_bank_extensions.py")
ext = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(ext)


class ExtensionsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        collection = self.root / "collection"; collection.mkdir()
        record = {"configuration": {"task": {"model": "rough"}}, "source_sha256": {"numerics": "frozen"},
                  "versions": {"numpy": "pinned"}, "task_fingerprint": "task"}
        ext.save(collection / "collection.json", record)
        snapshot = self.root / "snapshot"; snapshot.mkdir()
        splits = {"train": 50000, "validation": 5000, "test": 7000}
        ext.save(snapshot / "manifest.json", {"collection_id": "id", "configuration": {
            "task": record["configuration"]["task"], "root_seed": 271300, "splits": splits}})
        self.final = self.root / "final.json"
        ext.save(self.final, {"status": "completed_checked_snapshot", "snapshot_manifest_sha256": ext.sha(snapshot / "manifest.json"),
            "collection_id": "id", "split_counts": splits, "source_revision": "revision",
            "source_sha256": record["source_sha256"], "runtime": record["versions"], "task_fingerprint": "task"})
        self.jobs = [{"role": "preflight", "state": "submitted", "job_id": "1"},
                     {"role": "workers", "state": "submitted", "job_id": "2", "indices": [0, 1]},
                     {"role": "final", "state": "submitted", "job_id": "3"}]
        self.original = self.root / "original.json"
        ext.save(self.original, {"started_utc": "2026-10-28", "phases": {"known": {
            "status": "completed_checked_snapshot", "snapshot": str(snapshot),
            "snapshot_manifest_sha256": ext.sha(snapshot / "manifest.json"), "jobs": self.jobs}}})
        self.route = {"collection": str(collection), "collection_sha256": ext.sha(collection / "collection.json"),
            "collection_id": "id", "root_seed": 271300, "producer_revision": "revision",
            "dependency_snapshot": str(snapshot), "dependency_final": str(self.final),
            "dependency_splits": splits, "expected_original_workers": 2}
        self.plan = {"original_execution": str(self.original), "routes": {"known": self.route}}
        self.states = {"known": {"state": "waiting"}}
        self.rows = {i: {"state": "COMPLETED", "exit_code": "0:0"} for i in ["1", "2_0", "2_1", "3"]}
        self.scheduler = SimpleNamespace(accounting=lambda *_: self.rows)

    def test_requires_checked_snapshot_identity_and_settled_original_jobs(self):
        self.assertTrue(ext.readiness(self.plan, "known", self.states, self.scheduler)[0])
        self.rows["2_1"]["state"] = "RUNNING"
        self.assertFalse(ext.readiness(self.plan, "known", self.states, self.scheduler)[0])
        self.rows["2_1"]["state"] = "COMPLETED"
        report = json.loads(self.final.read_text()); report["runtime"] = {"numpy": "changed"}; ext.save(self.final, report)
        with self.assertRaisesRegex(ValueError, "identity"):
            ext.readiness(self.plan, "known", self.states, self.scheduler)

    def test_transient_readonly_accounting_timeout_remains_waiting(self):
        def timeout(*_): raise TimeoutError("temporary sacct timeout")
        scheduler = SimpleNamespace(accounting=timeout)
        ready, reason = ext.readiness(self.plan, "known", self.states, scheduler)
        self.assertFalse(ready); self.assertEqual(reason, "waiting_for_accounting:TimeoutError")

    def test_joint_resource_dependency_does_not_launch_or_ignore_known_failure(self):
        self.plan["routes"]["joint"] = {**self.route, "after_route": "known"}
        ready, reason = ext.readiness(self.plan, "joint", self.states, self.scheduler)
        self.assertFalse(ready); self.assertIn("resource_dependency", reason)
        self.states["known"]["state"] = "blocked_campaign"
        with self.assertRaisesRegex(ValueError, "resource_dependency_failed"):
            ext.readiness(self.plan, "joint", self.states, self.scheduler)

    def test_unknown_launch_is_persisted_once_without_retry(self):
        launcher = self.root / "launcher.py"; launcher.write_text("# frozen\n")
        config = self.root / "campaign.json"; ext.save(config, {"campaign_root": str(self.root / "campaign")})
        self.route["campaign_config"] = str(config)
        self.plan.update(python=sys.executable, launcher=str(launcher))
        ledger = {"routes": {"known": {"state": "waiting"}}}; calls = []
        def ambiguous(*args, **kwargs):
            calls.append(args)
            saved = json.loads((self.root / "execution.json").read_text())
            self.assertEqual(saved["routes"]["known"]["state"], "launch_attempted")
            raise TimeoutError("response unknown")
        with self.assertRaises(TimeoutError): ext.launch(self.plan, "known", ledger, self.root, run=ambiguous)
        self.assertEqual(ledger["routes"]["known"]["state"], "unknown_launch")
        with self.assertRaises(RuntimeError): ext.launch(self.plan, "known", ledger, self.root, run=ambiguous)
        self.assertEqual(len(calls), 1)

    def test_detached_waiter_receipt_precedes_child_and_is_once_only(self):
        p = self.root / "plan.json"; plan = {"root": str(self.root / "waiter"), "python": sys.executable}
        ext.save(p, plan); calls = []
        def fake_popen(*args, **kwargs):
            calls.append(args)
            self.assertTrue((Path(plan["root"]) / "launch-attempt.json").exists())
            self.assertTrue(kwargs["start_new_session"])
            return SimpleNamespace(pid=555)
        receipt = ext.detach(p, popen=fake_popen)
        self.assertEqual(receipt["pid"], 555)
        with self.assertRaises(FileExistsError): ext.detach(p, popen=fake_popen)
        self.assertEqual(len(calls), 1)

    def test_reviewed_recovery_uses_new_successful_jobs_not_historical_timeouts(self):
        old = json.loads(self.original.read_text())
        old["phases"]["joint"] = {"status": "generation_active", "jobs": [{"job_id": "old_failed"}]}
        ext.save(self.original, old)
        recovery = self.root / "recovery.json"
        jobs = [{"role": "workers", "state": "submitted", "job_id": "99", "indices": [404, 992]},
                {"role": "final", "state": "submitted", "job_id": "100"}]
        original = json.loads(self.original.read_text())
        phase = {**original["phases"]["known"], "jobs": jobs}
        ext.save(recovery, {"status": "completed_checked_recovery", "started_utc": "2026-10-28", "phases": {"joint": phase}})
        plan = self.root / "recovery-plan.json"; ext.save(plan, {"preserved_parts": 971})
        binding = {"original_execution_sha256": ext.sha(self.original), "preserved_inventory_sha256": "inventory",
                   "preserved_parts": 971, "original_job_map_sha256": "map", "reviewed_failed_indices": [404],
                   "new_worker_indices": [404, 992]}
        report = json.loads(self.final.read_text())
        report.update(recovery_execution=str(recovery), recovery_execution_sha256=ext.sha(recovery),
            original_execution_sha256=binding["original_execution_sha256"], preserved_inventory_sha256="inventory",
            preserved_parts_unchanged=971, original_job_map_sha256="map", reviewed_failed_indices=[404],
            new_worker_indices=[404,992], plan_sha256=ext.sha(plan))
        ext.save(self.final, report)
        route = {**self.route, "dependency_execution": str(recovery), "recovery_plan": str(plan), "reviewed_recovery": binding}
        self.plan["routes"] = {"joint": route}
        self.rows = {i: {"state": "COMPLETED", "exit_code": "0:0"} for i in ["99_404", "99_992", "100"]}
        self.assertTrue(ext.readiness(self.plan, "joint", {}, self.scheduler)[0])
        report["reviewed_failed_indices"] = []; ext.save(self.final, report)
        with self.assertRaisesRegex(ValueError, "recovery completion"):
            ext.readiness(self.plan, "joint", {}, self.scheduler)

    def test_external_known_campaign_is_a_checked_and_settled_resource_gate(self):
        p = self.root / "resource-campaign"; p.mkdir()
        config = self.root / "resource-config.json"
        ext.save(config, {"campaign_root": str(p), "snapshot": self.route["dependency_snapshot"]})
        dep = {"config": str(config), "collection_id": "id", "splits": self.route["dependency_splits"]}
        run = {"state": "generation_active", "started_utc": "2026-10-28"}
        ext.save(p / "execution.json", run)
        self.assertFalse(ext.external_campaign_ready(dep, self.scheduler)[0])
        run.update(state="completed_checked_snapshot", collection_id="id", attempts=[
            {"role": "workers", "state": "submitted", "job_id": "2", "indices": [0,1], "array_offset": 0},
            {"role": "final", "state": "submitted", "job_id": "3"}],
            snapshot_manifest_sha256=ext.sha(Path(self.route["dependency_snapshot"]) / "manifest.json"))
        ext.save(p / "execution.json", run)
        self.assertTrue(ext.external_campaign_ready(dep, self.scheduler)[0])
        run["state"] = "stopped_for_review"; ext.save(p / "execution.json", run)
        with self.assertRaisesRegex(ValueError, "resource_campaign_failed"):
            ext.external_campaign_ready(dep, self.scheduler)

    def test_native_tail_recovery_requires_exact_ranges_full_cost_and_new_successes(self):
        # Failed journals stay frozen; only two native missing ranges are rerun.
        old = self.root / "stopped-recovery.json"
        ext.save(old, {"status": "stopped_for_review", "failed": ["8046699_997", "8046699_1000"]})
        config_path = self.root / "tail-config.json"
        config = {"cpu_hours": 1500, "prior_cpu_seconds": 4000000}
        ext.save(config_path, config)
        jobs_path = self.root / "tail-jobs.json"
        jobs = [{"split": "test", "start": 50, "stop": 75, "job_sha256": "range997"},
                {"split": "test", "start": 125, "stop": 150, "job_sha256": "range1000"}]
        ext.save(jobs_path, {"jobs": jobs})
        attempts = [{"role": "workers", "state": "submitted", "job_id": "90", "indices": [0,1], "array_offset": 0},
                    {"role": "final", "state": "submitted", "job_id": "91"}]
        execution = self.root / "tail-execution.json"
        manifest_sha = ext.sha(Path(self.route["dependency_snapshot"]) / "manifest.json")
        native_snapshot = self.root / "native-snapshot"; native_snapshot.mkdir()
        native_manifest = json.loads((Path(self.route["dependency_snapshot"]) / "manifest.json").read_text())
        native_manifest["publication"] = "native tail snapshot precedes independent integrity snapshot"
        ext.save(native_snapshot / "manifest.json", native_manifest)
        native_sha = ext.sha(native_snapshot / "manifest.json")
        self.assertNotEqual(native_sha, manifest_sha)
        ext.save(execution, {"state": "completed_checked_snapshot", "started_utc": "2026-10-28",
            "snapshot": str(native_snapshot), "snapshot_manifest_sha256": native_sha,
            "source_sha256": {"numerics": "frozen"}, "runtime": {"numpy": "pinned"},
            "producer_git_revision": "revision",
            "collection_id": "id", "config": config, "planned_workers": 2, "attempts": attempts})
        fields = {"prior_recovery_execution_sha256": ext.sha(old), "preserved_parts_unchanged": 1278,
                  "reviewed_failed_allocations": ["8046699_997", "8046699_1000"]}
        binding = {"jobs": str(jobs_path), "campaign_config": str(config_path), "expected_jobs": jobs,
                   "cpu_hours": 1500, "prior_cpu_seconds": 4000000,
                   "snapshot": str(native_snapshot),
                   "frozen_provenance": {str(old): ext.sha(old)}, "completion_fields": fields}
        report = json.loads(self.final.read_text())
        report.update(fields, recovery_execution=str(execution), recovery_execution_sha256=ext.sha(execution),
            recovery_jobs_sha256=ext.sha(jobs_path), recovery_config_sha256=ext.sha(config_path),
            recovery_snapshot_manifest_sha256=native_sha,
            integrity_attempts=[{"role": "final", "state": "submitted", "job_id": "92"}],
            integrity_started_utc="2026-10-28")
        ext.save(self.final, report)
        self.route.update(native_recovery=binding, dependency_execution=str(execution))
        self.rows = {i: {"state": "COMPLETED", "exit_code": "0:0"} for i in ["90_0", "90_1", "91", "92"]}
        self.assertTrue(ext.readiness(self.plan, "known", self.states, self.scheduler)[0])
        self.rows["90_1"]["state"] = "TIMEOUT"
        self.assertFalse(ext.readiness(self.plan, "known", self.states, self.scheduler)[0])
        self.rows["90_1"]["state"] = "COMPLETED"
        del self.rows["92"]
        self.assertFalse(ext.readiness(self.plan, "known", self.states, self.scheduler)[0])
        self.rows["92"] = {"state": "COMPLETED", "exit_code": "0:0"}
        config["prior_cpu_seconds"] = 0; ext.save(config_path, config)
        report["recovery_config_sha256"] = ext.sha(config_path); ext.save(self.final, report)
        with self.assertRaisesRegex(ValueError, "budget differs"):
            ext.readiness(self.plan, "known", self.states, self.scheduler)
        config["prior_cpu_seconds"] = 4000000; ext.save(config_path, config)
        report["recovery_config_sha256"] = ext.sha(config_path); ext.save(self.final, report)
        ext.save(old, {"status": "rewritten_as_success"})
        with self.assertRaisesRegex(ValueError, "frozen provenance"):
            ext.readiness(self.plan, "known", self.states, self.scheduler)


if __name__ == "__main__": unittest.main()
