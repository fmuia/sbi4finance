"""Tiny native banks with mocked prices and a fake scheduler; no Slurm jobs."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sbi4finance.calibration import bank_collection as bc, tasks
from test_calibration_collections import cheap_price, configuration

SPEC = importlib.util.spec_from_file_location("bank_campaign", Path(__file__).resolve().parents[1] / "scripts/run_bank_campaign.py")
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


class FakeScheduler:
    def __init__(self, config, fail=False):
        self.config, self.fail, self.rows, self.calls = config, fail, {}, []

    def submit(self, argv):
        self.calls.append(argv)
        root = Path(self.config["campaign_root"])
        ledger = json.loads((root / "execution.json").read_text())
        attempt = ledger["attempts"][-1]
        assert attempt["state"] == "submission_attempted"  # Must exist BEFORE sbatch.
        if self.fail:
            raise TimeoutError("ambiguous scheduler response")
        identifier = str(100 + len(self.calls))
        if attempt["role"] == "workers":
            jobs = json.loads((root / "jobs.json").read_text())
            for i in attempt["indices"]:
                job = {k: jobs[k] for k in ("collection_id", "collection_sha256", "configuration")}
                job.update(jobs["jobs"][i])
                bc.run_job(jobs["collection"], job)
                self.rows[identifier + "_" + str(i-attempt["array_offset"])] = {
                    "state": "COMPLETED", "elapsed": 1, "cpus": 1, "exit_code": "0:0"}
        else:
            bc.freeze_snapshot(self.config["collection"], self.config["train"], self.config["snapshot"])
            self.rows[identifier] = {"state": "COMPLETED", "elapsed": 1, "cpus": 1, "exit_code": "0:0"}
        return identifier

    def accounting(self, attempts, started):
        return self.rows.copy()


class BankCampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.price = patch.object(tasks, "price", side_effect=cheap_price)
        self.price.start(); self.addCleanup(self.price.stop)
        self.config_file = self.root / "task.json"
        self.config_file.write_text(json.dumps(configuration(train=2)))

    def config(self, name="one", train=2):
        return {"task_config": str(self.config_file), "collection": str(self.root / "collection"),
                "snapshot": str(self.root / ("snapshot-" + name)), "campaign_root": str(self.root / name),
                "train": train, "chunk": 1, "max_workers": 2, "max_array_size": 2,
                "worker_wall_seconds": 60, "final_wall_seconds": 60, "cleanup_seconds": 10,
                "prior_cpu_seconds": 0, "cpu_hours": 1., "poll_seconds": 1,
                "controller_wall_hours": 1., "memory_gib": 1, "python": "locked-python",
                "account": "test", "partition": "test", "seed_policy": "native_collection"}

    def test_native_append_preserves_existing_data_and_fixed_holdouts(self):
        c = self.config(); first = campaign.run(c, scheduler=FakeScheduler(c), sleep=lambda _: None)
        self.assertEqual(first["state"], "completed_checked_snapshot")
        before = {str(p): p.read_bytes() for p in (self.root / "collection/parts").rglob("*") if p.is_file()}
        c = self.config("two", train=5)
        c["exclude_nodes"] = ["node2339", "node2344"]
        scheduler = FakeScheduler(c)
        result = campaign.run(c, scheduler=scheduler, sleep=lambda _: None)
        self.assertEqual(result["planned_workers"], 3)
        self.assertEqual(result["state"], "completed_checked_snapshot")
        for path, value in before.items():
            self.assertEqual(Path(path).read_bytes(), value)
        jobs = json.loads((self.root / "two/jobs.json").read_text())
        self.assertTrue(all(j["split"] == "train" and j["start"] >= 2 for j in jobs["jobs"]))
        self.assertEqual(result["budget"]["nonterminal_workers"], 0)
        arrays = [a for a in result["attempts"] if a["role"] == "workers"]
        self.assertEqual([a["array_offset"] for a in arrays], [0, 2])
        self.assertTrue(any("--array=0-0%1" in argv for argv in scheduler.calls))
        self.assertTrue(all("--exclude=node2339,node2344" in argv for argv in scheduler.calls))

    def test_budget_counts_missing_records_and_reserves_final_publication(self):
        c = self.config()
        attempts = [{"role": "workers", "state": "submitted", "job_id": "1", "indices": [0, 1],
                     "array_offset": 0, "wall_seconds": 60}]
        b = campaign.exposure(c, attempts, {})
        self.assertEqual(b["exposure_cpu_seconds"], 210)
        self.assertEqual(b["nonterminal_workers"], 2)
        rows = {"1_0": {"state": "COMPLETED", "elapsed": 5, "cpus": 1, "exit_code": "0:0"}}
        self.assertEqual(campaign.exposure(c, attempts, rows)["exposure_cpu_seconds"], 145)
        root = self.root / "budget"; root.mkdir()
        ledger = {"attempts": attempts}
        c["cpu_hours"] = 210/3600
        with self.assertRaisesRegex(RuntimeError, "ceiling"):
            campaign.submit(c, ledger, root, FakeScheduler(c), rows, "workers", [2])
        self.assertEqual(len(ledger["attempts"]), 1)

    def test_unknown_submission_persists_once_and_never_retries(self):
        c = self.config(); scheduler = FakeScheduler(c, fail=True)
        with self.assertRaises(TimeoutError):
            campaign.run(c, scheduler=scheduler, sleep=lambda _: None)
        report = json.loads((Path(c["campaign_root"]) / "execution.json").read_text())
        self.assertEqual(report["state"], "stopped_for_review")
        self.assertEqual(report["attempts"][0]["state"], "unknown_submission")
        with self.assertRaises(FileExistsError):
            campaign.run(c, scheduler=scheduler, sleep=lambda _: None)
        self.assertEqual(len(scheduler.calls), 1)

    def test_changed_task_must_use_a_distinct_collection(self):
        c = self.config(); campaign.run(c, scheduler=FakeScheduler(c), sleep=lambda _: None)
        task = json.loads(self.config_file.read_text()); task["task"]["fixed_parameters"]["kappa"] = 3.
        self.config_file.write_text(json.dumps(task))
        c = self.config("changed", train=3); scheduler = FakeScheduler(c)
        with self.assertRaises(ValueError):
            campaign.run(c, scheduler=scheduler, sleep=lambda _: None)
        self.assertFalse(scheduler.calls)

    def test_certificate_cannot_be_reused_for_an_expanded_id_domain(self):
        c = self.config()
        record = bc.create_collection(c["collection"], json.loads(self.config_file.read_text()))
        p = self.root / "certificate.json"
        value = {"format": "calibration_cross_collection_seed_certificate_v1", "passed": True,
                 "collection_id": record["collection_id"], "rng_protocol": record["rng_protocol"],
                 "task_fingerprint": record["task_fingerprint"], "root_seed": record["configuration"]["root_seed"],
                 "covered_splits": record["configuration"]["splits"]}
        p.write_text(json.dumps(value))
        c.update(seed_policy="certificate", seed_certificate=str(p), seed_certificate_sha256=campaign.sha(p))
        campaign.check_certificate(c, record, record["configuration"])
        target = copy.deepcopy(record["configuration"]); target["splits"]["train"] += 1
        with self.assertRaisesRegex(ValueError, "requested IDs"):
            campaign.check_certificate(c, record, target)

    def test_detached_receipt_survives_terminal_and_is_once_only(self):
        c = self.config(); calls = []
        class Child:
            pid = 12345
        def popen(*args, **kwargs):
            calls.append((args, kwargs)); return Child()
        receipt = campaign.detach(c, popen=popen)
        self.assertEqual(receipt["pid"], 12345)
        self.assertTrue(calls[0][1]["start_new_session"])
        self.assertEqual(receipt["state"], "launched_not_yet_checked")
        self.assertTrue(Path(receipt["log"]).exists())
        frozen = json.loads((Path(c["campaign_root"])/"campaign.json").read_text())
        self.assertEqual(frozen.pop("expected_operator_sha256"), campaign.sha(campaign.__file__))
        self.assertEqual(frozen, c)
        with self.assertRaises(FileExistsError):
            campaign.detach(c, popen=popen)
        self.assertEqual(len(calls), 1)

    def test_accounting_includes_previous_local_evening_from_real_capture(self):
        fixture = json.loads((Path(__file__).resolve().parents[1] /
            "docs/pilot-results/2026-10-28/bank-followons-accounting-recovery/accounting-date-failure.json").read_text())
        self.assertEqual(fixture["raw_sacct"]["2026-10-28"], "")
        called = []
        def captured(argv, **kwargs):
            called.append(argv)
            return type("Result", (), {"stdout": fixture["raw_sacct"][argv[argv.index("-S")+1]]})()
        with patch.object(campaign.subprocess, "run", side_effect=captured):
            rows = campaign.Slurm().accounting([{"job_id": fixture["job_id"]}], fixture["original_started_utc"])
        self.assertEqual(called[0][called[0].index("-S")+1], "2026-10-27")
        self.assertEqual(set(rows), {"8032794"})  # Exclude .batch/.extern step records.
        self.assertEqual(rows["8032794"], {"state": "COMPLETED", "elapsed": 445, "cpus": 1, "exit_code": "0:0"})


if __name__ == "__main__":
    unittest.main()
