"""Readiness and once-only scheduling guards; no Slurm or neural backend calls."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("post_bank_operator", ROOT / "cluster/submit_full_surface_backbones_20261028.py")
operator = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(operator)


class PostBankOperatorTests(unittest.TestCase):
    def fixture(self, root):
        bank = root / "bank"; bank.mkdir()
        route = {"bank": str(bank), "ready": str(root / "final.json"), "collection_id": "same-collection",
                 "producer_revision": "frozen-producer", "root_seed": 11, "task_id": "rough_calibration_v1",
                 "splits": {"train": 50000, "validation": 5000, "test": 7000}}
        manifest = {"configuration": {"root_seed": 11, "task": {"task_id": route["task_id"]},
                       "splits": route["splits"]}, "collection_id": route["collection_id"],
                    "task_fingerprint": "task", "source_sha256": {"pricing.py": "producer-code"}}
        operator.save(bank / "manifest.json", manifest)
        final = {"status": "completed_checked_snapshot", "snapshot": str(bank),
                 "snapshot_manifest_sha256": operator.sha(bank / "manifest.json"), "split_counts": route["splits"],
                 "collection_id": route["collection_id"], "source_revision": route["producer_revision"],
                 "source_sha256": manifest["source_sha256"], "task_fingerprint": "task"}
        return route, final

    def test_manifest_existence_does_not_release_incomplete_or_wrong_bank(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); route, final = self.fixture(root)
            self.assertIsNone(operator.readiness(route))
            operator.save(root / "final.json", {**final, "status": "failed"})
            with self.assertRaisesRegex(ValueError, "successful checked snapshot"): operator.readiness(route)
            operator.save(root / "final.json", {**final, "source_revision": "different-producer"})
            with self.assertRaisesRegex(ValueError, "identity/counts"): operator.readiness(route)
            operator.save(root / "final.json", final)
            self.assertEqual(operator.readiness(route)["manifest_sha256"], final["snapshot_manifest_sha256"])

    def test_ambiguous_scheduler_timeout_is_recorded_before_contact_and_never_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); route, final = self.fixture(root); (root / "logs").mkdir()
            operator.save(root / "final.json", final)
            plan = {"root": str(root), "repo": str(root), "python": "/python", "routes": {"known": route}}
            ledger = {"routes": {"known": {"state": "waiting"}}}
            def timeout(*args, **kwargs):
                persisted = operator.read(root / "execution.json")["routes"]["known"]
                self.assertEqual(persisted["state"], "submission_attempted")
                raise subprocess.TimeoutExpired("sbatch", 45)
            with patch.object(operator, "source_check"), patch.object(operator.subprocess, "run", side_effect=timeout) as call:
                operator.submit(plan, ledger, "known")
                self.assertEqual(ledger["routes"]["known"]["state"], "unknown_or_failed_submission")
                with self.assertRaisesRegex(ValueError, "already has a recorded attempt"):
                    operator.submit(plan, ledger, "known")
                self.assertEqual(call.call_count, 1)

    def test_complete_two_job_exposure_fits_declared_separate_caps(self):
        self.assertLessEqual(2 * operator.CPUS * (operator.WALL + operator.CLEANUP), 9 * 3600)
        self.assertLessEqual(2 * (operator.WALL + operator.CLEANUP), 2.25 * 3600)


if __name__ == "__main__": unittest.main()
