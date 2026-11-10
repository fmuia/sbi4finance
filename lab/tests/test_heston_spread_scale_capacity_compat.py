"""Serialization-only repair retains the frozen schedule implementation."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

from scripts import evaluate_heston_spread_scale_capacity_compat as compat


class CapacityCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.cls = compat.continuation.schedule_class()
        self.original_module = self.cls.__module__
        self.addCleanup(setattr, self.cls, "__module__", self.original_module)
        self.cls.__module__ = "scripts.continue_heston_spread_scale"
        self.expected = {"module": None, "class_name": "SpreadExposureCosine",
            "registered_name": "sbi4finance>SpreadExposureCosine",
            "config": {"train_rows": 100000, "batch_size": 512, "start_updates": 60000,
                "decay_passes": 400, "maximum": 5e-5, "minimum": 5e-6}}

    def test_only_module_attribution_changes_and_executable_rates_stay_exact(self):
        before = compat.code_hashes(self.cls)
        proof = compat.restore_cli_module(self.cls, self.expected)
        self.assertEqual(compat.code_hashes(self.cls), before)
        self.assertEqual(self.cls.__module__, "__main__")
        self.assertEqual(proof["serialization_after"], self.expected)
        self.assertTrue(proof["same_registered_class_object"])
        self.assertTrue(proof["numeric_rates_bitwise_unchanged"])
        self.assertEqual(proof["serialization_before"]["module"], "scripts.continue_heston_spread_scale")
        for entry in proof["rates"]:
            self.assertAlmostEqual(entry["actual"], entry["independent_formula"], delta=1e-11)
        self.assertGreater(proof["rates"][0]["actual"], proof["rates"][-1]["actual"])

    def test_other_serialization_changes_are_rejected(self):
        for key, value in (("class_name", "DifferentSchedule"),
                ("registered_name", "different>SpreadExposureCosine"), ("module", "different.module")):
            changed = copy.deepcopy(self.expected)
            changed[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "restricted"):
                compat.restore_cli_module(self.cls, changed)
        self.cls.__module__ = "different.module"
        with self.assertRaisesRegex(ValueError, "restricted"):
            compat.restore_cli_module(self.cls, self.expected)

    def test_archive_must_have_one_literal_exposure_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.keras"
            for schedules in ([], [self.expected, self.expected], [self.expected]):
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("config.json", json.dumps({"schedules": schedules}))
                if len(schedules) == 1:
                    self.assertEqual(compat.archive_schedule(path), self.expected)
                else:
                    with self.assertRaisesRegex(ValueError, "exactly one"):
                        compat.archive_schedule(path)

    def test_first_nested_output_can_write_proof_without_precreating_evaluator_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root/"checkpoint"
            checkpoint.mkdir()
            state = {group: [{"sha256": "fixture"}] for group in compat.fixed.GROUPS}
            record = {"optimizer_schedule": self.expected, "state": state,
                "actual_optimizer_updates": 64900}
            compat.fixed.write_json(checkpoint/"checkpoint.json", record)
            (checkpoint/"model.keras").write_bytes(b"opaque guarded-loader fixture")
            (checkpoint/"state.npz").write_bytes(b"opaque guarded-loader fixture")
            study, addendum = root/"study.json", root/"compatibility.json"
            compat.fixed.write_json(study, {})
            compat.fixed.write_json(addendum, {"bootstrap_source": {}, "frozen_sources": {}})
            optimizer = types.SimpleNamespace(get_config=lambda: {"learning_rate": self.expected},
                iterations=types.SimpleNamespace(numpy=lambda: 64900))
            workflow = types.SimpleNamespace(estimator=types.SimpleNamespace(optimizer=optimizer))
            out, payload = root/"new"/"nested"/"pass-0025", root/"payload"
            proof = out.parent/(out.name+".compatibility.json")
            def untouched_evaluator(*args):
                self.assertTrue(proof.is_file())
                self.assertFalse(out.exists())
                self.assertFalse(payload.exists())
                self.assertTrue(all(compat.read(proof)["all_saved_array_groups_exact"].values()))
                return {"status": "complete"}
            with ExitStack() as stack:
                stack.enter_context(patch.object(compat, "validate_addendum"))
                stack.enter_context(patch.object(compat, "archive_schedule", return_value=self.expected))
                stack.enter_context(patch.object(compat, "restore_cli_module", return_value={"fixture": "serialization proof"}))
                stack.enter_context(patch.object(compat.continuation, "load_checkpoint", return_value=(workflow, record)))
                stack.enter_context(patch.object(compat.fixed, "current_state", return_value=()))
                stack.enter_context(patch.object(compat.fixed, "state_summary", return_value=state))
                consumer = stack.enter_context(patch.object(compat.evaluator, "evaluate", side_effect=untouched_evaluator))
                result = compat.evaluate(checkpoint, study, out, payload, addendum)
            self.assertEqual(result["status"], "complete")
            consumer.assert_called_once_with(checkpoint, study, out, payload)


if __name__ == "__main__":
    unittest.main()
