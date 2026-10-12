"""Cheap synthetic/mocked checks of the one-shot held-out evaluation boundary.

These tests never open the actual reserved cohort or run an inference study.
"""

from copy import deepcopy
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_rough_test as runner


class RoughHeldoutTests(unittest.TestCase):
    def setUp(self):
        self.development = json.loads((ROOT / "configs/rough_reference8192_protocol.json").read_text())
        self.task = json.loads((ROOT / "configs/tasks/rough_inference_bank.json").read_text())["task"]
        self.protocol = {
            "experiment_id": "unit_test_only", "development_protocol": "development.json",
            "paths": {"bank": "bank", "reference": "reference", "checkpoint": "checkpoint",
                      "validation_report": "reference/neural-validation.json"},
            "split": "test", "conditional_ids": list(range(16)), "sbc_ids": [0, 256],
            "reference_gates": self.development["reference"]["gates"],
            "evaluation": dict(self.development["evaluation"], sample_seed=84034,
                               rank_randomization_seed=84035, prior_control_seed=84036),
            "target_pricing": dict(self.development["target_pricing"], neural_seed=84031),
            "targets": self.development["targets"], "observation_law": self.development["observation_law"],
        }
        rng = np.random.default_rng(49)
        bounds = np.asarray(self.task["prior_bounds"])
        self.split = {"example_id": np.array([f"test:{i}" for i in range(256)]),
                      "observations": rng.normal(3, 1, (256, 9, 1)),
                      "parameters": rng.uniform(bounds[:, 0], bounds[:, 1], (256, 2)),
                      "future_outcomes": np.full((256, 2), np.nan)}
        self.arrays = {"parameters": rng.uniform(bounds[:, 0], bounds[:, 1], (64, 2)),
                       "reference_prices": rng.normal(3, 1, (64, 9)),
                       "target_estimates": rng.normal(3, .2, (64, 2)),
                       "target_standard_errors": np.full((64, 2), .01)}

    def fixture_files(self, directory):
        """Create stand-in bytes; no real checkpoint or bank is read."""
        os.chdir(directory)
        paths = {name: Path(value) for name, value in self.protocol["paths"].items()}
        for folder in (paths["bank"], paths["reference"], paths["checkpoint"], Path("configs/tasks")):
            folder.mkdir(parents=True, exist_ok=True)
        files = [paths["bank"] / f"{split}.npz" for split in ("train", "validation", "test")]
        files += [paths["checkpoint"] / name for name in
                  ("model.keras", "training.json", "task.json", "protocol.json", "bank-manifest.json")]
        files += [paths["reference"] / name for name in
                  ("reference.json", "reference.npz", "reference-bank-manifest.json", "validation-checks.npz")]
        files += [Path(self.development[key]) for key in ("bank_config", "reference_config")]
        for filename in files:
            filename.write_text("stand-in bytes; never loaded as real data")
        manifest = {"splits": {split: {"file": f"{split}.npz"} for split in ("train", "validation", "test")}}
        (paths["bank"] / "manifest.json").write_text(json.dumps(manifest))
        development_path = Path(self.protocol["development_protocol"])
        development_path.write_text(json.dumps(self.development))
        validation = {
            "passed": True, "split": "validation", "reserved_test_evaluated": False,
            "experiment_id": self.development["experiment_id"],
            "thresholds": self.development["evaluation"]["gates"],
            "gates": {key: True for key in (*self.development["evaluation"]["gates"], "marginal_sbc")},
            "protocol_sha256": runner._sha256(development_path),
            "checkpoint_sha256": runner._sha256(paths["checkpoint"] / "model.keras"),
            "training_report_sha256": runner._sha256(paths["checkpoint"] / "training.json"),
            "reference_report_sha256": runner._sha256(paths["reference"] / "reference.json"),
            "bank_manifest_sha256": runner._sha256(paths["bank"] / "manifest.json"),
            "checks_sha256": runner._sha256(paths["reference"] / "validation-checks.npz"),
        }
        paths["validation_report"].write_text(json.dumps(validation))
        files += [development_path, paths["bank"] / "manifest.json", paths["validation_report"]]
        protocol = deepcopy(self.protocol)
        protocol["input_sha256"] = {str(filename): runner._sha256(filename) for filename in files}
        sources = ["scripts/evaluate_rough_test.py", "scripts/evaluate_rough.py", "scripts/train_rough.py",
                   "scripts/check_rough_reference.py", "src/sbi4finance/rough_inference.py",
                   "src/sbi4finance/rough_reference.py"]
        for filename in sources:
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("frozen source stand-in")
        protocol["frozen_source_sha256"] = {filename: runner._sha256(filename) for filename in sources}
        Path("protocol.json").write_text(json.dumps(protocol))
        return protocol, paths, validation

    def test_prerequisites_reject_before_any_reserved_array_loader(self):
        before = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                protocol, paths, validation = self.fixture_files(directory)
                runner.checked_prerequisites(Path("protocol.json"))
                variants = []
                mismatch = deepcopy(protocol)
                mismatch["input_sha256"]["bank/test.npz"] = "0" * 64
                variants.append(mismatch)
                omitted = deepcopy(protocol)
                del omitted["input_sha256"]["bank/test.npz"]
                variants.append(omitted)
                changed_source = deepcopy(protocol)
                changed_source["frozen_source_sha256"]["scripts/evaluate_rough.py"] = "0" * 64
                variants.append(changed_source)
                wrong_gate = deepcopy(protocol)
                wrong_gate["evaluation"]["gates"]["max_abs_correlation_error"] = 0.2
                variants.append(wrong_gate)
                for variant in variants:
                    Path("protocol.json").write_text(json.dumps(variant))
                    with patch.object(runner, "checked_inputs") as loader:
                        with self.assertRaises(ValueError):
                            runner.evaluate(Path("protocol.json"), Path("out"))
                        loader.assert_not_called()
                        self.assertFalse(Path("out").exists())
                # A consistently hash-bound but failed acceptance report still blocks access.
                validation["gates"]["marginal_sbc"] = False
                paths["validation_report"].write_text(json.dumps(validation))
                protocol["input_sha256"][str(paths["validation_report"])] = runner._sha256(paths["validation_report"])
                Path("protocol.json").write_text(json.dumps(protocol))
                with patch.object(runner, "checked_inputs") as loader:
                    with self.assertRaisesRegex(ValueError, "prerequisites"):
                        runner.evaluate(Path("protocol.json"), Path("out"))
                    loader.assert_not_called()
        finally:
            os.chdir(before)

    def test_fixed_split_and_original_id_order_are_required(self):
        before = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                protocol, _, _ = self.fixture_files(directory)
                for change in ({"split": "validation"}, {"conditional_ids": list(range(1, 17))},
                               {"sbc_ids": [0, 128]}):
                    Path("protocol.json").write_text(json.dumps({**protocol, **change}))
                    with self.assertRaisesRegex(ValueError, "fixed test"):
                        runner.checked_prerequisites(Path("protocol.json"))
        finally:
            os.chdir(before)
        case_ids, observations, truth = runner.selected_test_cases(self.split, self.protocol)
        assert_array_equal(case_ids, self.split["example_id"][:16])
        assert_array_equal(observations, self.split["observations"][:16])
        assert_array_equal(truth, self.split["parameters"][:16])
        for ids in (self.split["example_id"][::-1], np.array(["test:0"] * 256),
                    np.array([f"validation:{i}" for i in range(256)])):
            with self.assertRaisesRegex(ValueError, "original ordered"):
                runner.selected_test_cases({**self.split, "example_id": ids}, self.protocol)
        changed = deepcopy(self.split)
        changed["parameters"] = changed["parameters"][::-1]
        changed["future_outcomes"] = np.zeros((256, 2))
        _, changed_obs, changed_truth = runner.selected_test_cases(changed, self.protocol)
        assert_array_equal(changed_obs, observations)
        self.assertFalse(np.array_equal(changed_truth, truth))
        conditions = runner.inference.conditioning_data(changed_obs, self.task)
        self.assertEqual(set(conditions), {"observations"})
        assert_array_equal(conditions["observations"], observations[..., 0])

    def mock_run(self, directory, *, fail_reference):
        protocol, paths, _ = self.fixture_files(directory)
        fake_inputs = (self.development, {"task": self.task}, {"reference_bank_manifest_sha256": "refdigest"},
                       {}, {"test": self.split, "validation": {}}, {"checkpoint_sha256": "modeldigest"})
        calls = []

        def one_reference(parameters, prices, targets, observation, sigma, thresholds):
            index = len(calls)
            calls.append(observation.copy())
            passed = not (fail_reference and index == 3)
            return np.full(len(parameters), 1 / len(parameters)), {
                "passed": passed, "metrics": {key: 1000.0 if key == "min_ess" else 0.0 for key in thresholds},
                "gates": {key: passed for key in thresholds}}

        class Workflow:
            def __init__(self):
                self.conditions = []
                self.loaded = []

            def load_approximator(self, path):
                self.loaded.append(path)

            def sample(self, *, num_samples, conditions, batch_size, seed):
                self.conditions.append(deepcopy(conditions))
                rng = np.random.default_rng(seed)
                # Intentionally wrong narrow candidate makes scientific gates fail.
                values = rng.normal([.22, .9], [.002, .006], (len(conditions["observations"]), num_samples, 2))
                return {name: values[..., i:i + 1] for i, name in enumerate(runner.inference.PARAMETER_NAMES)}

        workflow = Workflow()
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, "checked_prerequisites", return_value=(protocol, paths)))
            stack.enter_context(patch.object(runner, "checked_inputs", return_value=fake_inputs))
            stack.enter_context(patch.object(runner, "checked_reference_arrays", return_value=self.arrays))
            reference_mock = stack.enter_context(patch.object(runner, "case_reference", side_effect=one_reference))
            build = stack.enter_context(patch.object(runner.inference, "build_workflow", return_value=workflow))
            price = stack.enter_context(patch.object(runner, "price_parameters", side_effect=lambda task, draws, *a, **k:
                                                    (np.column_stack((draws[:, 0] + 3, draws[:, 1] + 3)),
                                                     np.full((len(draws), 2), .01))))
            stack.enter_context(patch.object(runner, "version", return_value="mock-version"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            report = runner.evaluate(Path("protocol.json"), Path("out"))
            self.assertEqual(reference_mock.call_count, 16)
            assert_array_equal(np.asarray(calls), self.split["observations"][:16])
            if fail_reference:
                build.assert_not_called()
                price.assert_not_called()
                self.assertEqual(workflow.conditions, [])
            else:
                self.assertEqual(build.call_count, 1)
                self.assertEqual(price.call_count, 16)
            return report, workflow

    def test_any_reference_failure_records_all16_and_stops_before_neural_or_sbc(self):
        before = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                report, _ = self.mock_run(directory, fail_reference=True)
                self.assertEqual(report["status"], "blocked_by_reference")
                self.assertTrue(report["reserved_test_accessed"])
                self.assertFalse(report["neural_test_evaluated"])
                self.assertTrue(report["reserved_test_evaluated"])
                self.assertFalse(report["full_test_evaluated"])
                self.assertEqual(report["reference_evaluated_case_ids"], [f"test:{i}" for i in range(16)])
                self.assertEqual(report["sbc_evaluated_case_ids"], [])
                saved = json.loads(Path("out/reference.json").read_text())
                self.assertEqual(len(saved["cases"]), 16)
                self.assertFalse(saved["cases"][3]["passed"])
                self.assertTrue(Path("out/reference-checks.npz").exists())
                self.assertFalse(Path("out/neural-test.json").exists())
                self.assertFalse(Path("out/test-checks.npz").exists())
                with patch.object(runner, "checked_prerequisites") as check:
                    with self.assertRaises(FileExistsError):
                        runner.evaluate(Path("protocol.json"), Path("out"))
                    check.assert_not_called()
        finally:
            os.chdir(before)

    def test_failed_neural_gate_still_saves_all_cases_and_fully_reproducible_sbc(self):
        before = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                report, workflow = self.mock_run(directory, fail_reference=False)
                self.assertEqual(report["status"], "provisional")
                self.assertFalse(report["passed"])
                self.assertEqual(len(report["cases"]), 16)
                self.assertEqual(len(report["sbc_evaluated_case_ids"]), 256)
                self.assertTrue(report["full_test_evaluated"])
                self.assertEqual(len(workflow.conditions), 2)
                for conditions, count in zip(workflow.conditions, (16, 256), strict=True):
                    self.assertEqual(set(conditions), {"observations"})
                    assert_array_equal(conditions["observations"], self.split["observations"][:count, :, 0])
                with np.load("out/test-checks.npz", allow_pickle=False) as saved:
                    self.assertEqual(saved["conditional_draws"].shape, (16, 4096, 2))
                    self.assertEqual(saved["audit_draws"].shape, (256, 512, 2))
                    self.assertEqual(saved["prior_audit_draws"].shape, (256, 512, 2))
                    for prefix, draw_key, seed in (("", "audit_draws", 84035),
                                                   ("prior_", "prior_audit_draws", 84036)):
                        checked = runner.reference.randomized_sbc(saved[draw_key], saved["audit_truth"], seed=seed)
                        assert_array_equal(saved[prefix + "ranks"], checked["ranks"])
                        assert_array_equal(saved[prefix + "randomized_ranks"], checked["randomized_ranks"])
                        expected = report["sbc"] if not prefix else report["prior_baseline"]["sbc"]
                        self.assertEqual(expected["coverage_90"], checked["coverage_90"])
                self.assertAlmostEqual(report["sbc"]["rank_dkw_bound"],
                                       np.sqrt(np.log(4 / .05) / (2 * 256)))
        finally:
            os.chdir(before)


if __name__ == "__main__":
    unittest.main()
