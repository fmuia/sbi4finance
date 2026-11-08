"""Admission is explicit; independent lanes consume reservations on failure."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts import run_heston_budget_study as runner


class BudgetRunnerTests(unittest.TestCase):
    def fixture(self, root):
        protocol_path = runner.DEFAULT_PROTOCOL
        protocol = runner.read_json(protocol_path)
        readiness = root / "readiness.json"
        readiness.write_text(json.dumps({"protocol_sha256": runner.digest(protocol_path),
                                         "control_checkpoint_sha256": protocol["control"]["checkpoint_sha256"],
                                         "status": "pass", "reserved_test_evaluated": False}))
        smoke = root / "smoke.json"
        smoke.write_text(json.dumps({"same_seed_reload_exact": True, "scientific_validation": False,
                                    "run_contract": {"format": "full_surface_update_budget_run_v1", "settings": {"updates": 3}},
                                    "selected_update": 3, "performance": {"work": {"optimizer_updates": 3}}}))
        admission = {"format": "heston_training_budget_admission_v1",
                     "protocol_sha256": runner.digest(protocol_path), "readiness_sha256": runner.digest(readiness),
                     "smoke_training_sha256": runner.digest(smoke), "source_revision": "frozen-revision",
                     "host_cpus": 4, "gpu_count": 1, "cleanup_seconds_per_lane": 5, "lane_wall_seconds": {"small-u00030-s0": 20, "full-u00030-s0": 30},
                     "already_used": {"allocated_cpu_seconds": 40, "gpu_seconds": 10},
                     "reserved_confirmation": {"allocated_cpu_seconds": 2000, "gpu_seconds": 500},
                     "approved_for_execution": True}
        path = root / "admission.json"; path.write_text(json.dumps(admission))
        return protocol_path, readiness, smoke, path, admission

    def test_admission_binds_evidence_and_caps_total_including_final(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); p, r, s, a, admission = self.fixture(root)
            runner.validate_admission(p, r, s, a, "frozen-revision")
            for changes, error in (({"source_revision": "changed"}, "identity changed"),
                                   ({"approved_for_execution": False}, "explicitly admitted"),
                                   ({"reserved_confirmation": {"allocated_cpu_seconds": 21600, "gpu_seconds": 0}}, "resource ceiling")):
                a.write_text(json.dumps({**admission, **changes}))
                with self.assertRaisesRegex(ValueError, error):
                    runner.validate_admission(p, r, s, a, "frozen-revision")
            a.write_text(json.dumps(admission))
            r.write_text(r.read_text().replace('"pass"', '"inconclusive"'))
            with self.assertRaisesRegex(ValueError, "readiness did not pass"):
                runner.validate_admission(p, r, s, a, "frozen-revision")

    def test_reservations_are_independent_nonretryable_and_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); _, _, _, a, admission = self.fixture(root)
            out = root / "lanes"
            runner.reserve_lane(out, "small-u00030-s0", a, admission)
            with self.assertRaisesRegex(ValueError, "already reserved"):
                runner.reserve_lane(out, "small-u00030-s0", a, admission)
            runner.reserve_lane(out, "full-u00030-s0", a, admission)
            self.assertEqual(len(runner.read_json(out / "ledger.json")["lanes"]), 2)
            with runner.ledger_lock(out):
                with self.assertRaisesRegex(ValueError, "ledger is locked"):
                    runner.reserve_lane(out, "small-u00030-s0", a, admission)

    def test_candidates_and_commands_keep_task_recipe(self):
        protocol = runner.read_json(runner.DEFAULT_PROTOCOL)
        candidates = runner.candidates(protocol)
        self.assertEqual(len(candidates), 14)
        self.assertEqual(sum(row["updates"] for row in candidates.values()), 88860)
        command = runner.command(candidates["full-u30000-s0"], "/caller/bank", "/caller/out")
        self.assertEqual(command[command.index("--updates") + 1], "30000")
        self.assertEqual(command[command.index("--widths") + 1], "256,256,256,256")
        self.assertEqual(command[command.index("--checkpoint-selection") + 1], "final")
        self.assertFalse(any("oscar" in item for item in command))
