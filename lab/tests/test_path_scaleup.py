"""New teaching-task safeguards and robust paired-result reporting."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from sbi4finance.path_inference import bank, inference, one_return, pmmh

LAB = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


comparison = module("path_scaleup_compare_tests", LAB / "scripts/compare_path_scaleup.py")
pilot = module("path_scaleup_pilot_tests", LAB / "scripts/check_path_scaleup.py")


class ScaleupTests(unittest.TestCase):
    def test_old_conditional_references_reject_the_new_observation_laws(self):
        with self.assertRaisesRegex(ValueError, "fixed drift"):
            one_return.ShortVariancePosterior(np.array([.01]), bank.teaching_task("rough-joint"))
        with self.assertRaisesRegex(ValueError, "four-parameter"):
            pmmh.validate_reference_task(bank.teaching_task("heston"))

    def test_contract_requires_same_observation_design_for_every_model(self):
        for name in bank.MODELS:
            task = bank.teaching_task(name)
            pilot.teaching_contract(task)
            physical = inference.prior_samples(task, 100, 20261031001)
            restored = inference.inverse_parameters(inference.transform_parameters(physical, task), task)
            np.testing.assert_allclose(restored, physical, rtol=2e-14, atol=1e-15)
            self.assertTrue(np.any(physical[:, -1] < 0))
            changed = bank.teaching_task(name, observations=252)
            with self.assertRaises(ValueError):
                pilot.teaching_contract(changed)

    def test_missing_lane_still_writes_an_explicit_inconclusive_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(bank=root / "missing", out=root / "out")
            report = comparison.comparison(args)
            self.assertEqual(report["status"], "inconclusive_missing_or_invalid_inputs")
            self.assertFalse(report["scientific_acceptance"])
            self.assertTrue((args.out / "comparison.json").exists())

    def test_checksum_mismatch_cannot_be_used_for_a_paired_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "case-summary.npz"
            np.savez(path, truth=np.zeros((1024, 4)))
            (root / "evaluation.json").write_text(json.dumps({"case_summary_file": path.name,
                "case_summary_sha256": "wrong"}))
            with self.assertRaisesRegex(ValueError, "hash"):
                comparison.load_evaluation(root)

    def test_successful_pair_reports_more_data_without_automatic_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = bank.teaching_task("rough-known")
            manifest = {"task": task, "task_sha256": bank.task_fingerprint(task)}
            bankdir = root / "bank"
            bankdir.mkdir()
            (bankdir / "manifest.json").write_text('{}')
            truth = inference.prior_samples(task, 1024, 42)
            identities = np.asarray([f"validation:{i}" for i in range(1024)])
            for label, count, epochs, error in (("small", 100000, 50, 1e-4), ("large", 1000000, 5, 0)):
                fit, evaluated = root / label, root / (label + "-evaluation")
                fit.mkdir()
                evaluated.mkdir()
                (fit / "model.keras").write_bytes(b"test-model-identity")
                checkpoint_hash = inference.sha256(fit / "model.keras")
                record = {"model_sha256": checkpoint_hash, "training_trajectories": count,
                    "same_seed_reload_exact": True, "epochs_completed": epochs,
                    "optimizer_steps": ((count + 127) // 128) * epochs,
                    "network": inference.DEFAULT_NETWORK, "seed": 20261031301, "batch_size": 128,
                    "prefix_policy": "uniform_log_mixture", "learning_rate": .0005,
                    "checkpoint_selection": "final"}
                (fit / "training.json").write_text(json.dumps(record))
                arrays = {"truth": truth, "trajectory_id": identities}
                lengths = [1, 2, 4, 8, 17, 21, 63, 126, 252, 504, 756, 1260]
                for n in lengths:
                    arrays.update({f"posterior_mean_{n}": truth + error,
                                   f"lower90_{n}": truth - .01, f"upper90_{n}": truth + .01})
                np.savez_compressed(evaluated / "case-summary.npz", **arrays)
                report = {"case_summary_file": "case-summary.npz",
                    "case_summary_sha256": inference.sha256(evaluated / "case-summary.npz"),
                    "split": "validation", "offset": 0, "reserved_test_evaluated": False,
                    "cases": 1024, "draws": 512, "targets": task["parameter_names"],
                    "checkpoint_sha256": checkpoint_hash, "task_sha256": bank.task_fingerprint(task),
                    "bank_manifest_sha256": inference.sha256(bankdir / "manifest.json"),
                    "case_identities_sha256": inference.fingerprint(identities.tolist()),
                    "lengths": {str(n): {"nonfinite_count": 0, "out_of_prior_fraction": [0] * 4} for n in lengths}}
                (evaluated / "evaluation.json").write_text(json.dumps(report))
            args = SimpleNamespace(bank=bankdir, small=root / "small", large=root / "large",
                small_evaluation=root / "small-evaluation", large_evaluation=root / "large-evaluation", out=root / "out")
            with patch.object(bank, "load_manifest", return_value=manifest), patch.object(comparison, "plot_histories"):
                report = comparison.comparison(args)
            self.assertEqual(report["status"], "development_comparison_complete")
            self.assertTrue(report["matched_training_exposure"])
            self.assertEqual(report["cohorts"]["new_validation"]["cases"], 768)
            self.assertTrue(all(x == 0 for x in report["cohorts"]["new_validation"]["lengths"]["1260"]["candidate_mse_ratio"]))
            self.assertFalse(report["scientific_acceptance"])


if __name__ == "__main__":
    unittest.main()
