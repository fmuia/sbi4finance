"""Frozen candidate identity, paired development reports and isolated stage failures."""

from argparse import Namespace
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import evaluate_heston_budget_candidate as evaluator
from scripts import run_heston_matrix_lane as worker
from scripts import run_heston_budget_study as study


class CandidateEvaluationTests(unittest.TestCase):
    def fixture(self):
        protocol = study.read_json(study.DEFAULT_PROTOCOL)
        candidate = protocol["screen_candidates"][0]
        fixed = protocol["fixed_training"]
        network = {key: fixed[key] for key in ("family", "encoder", "summary_dim", "flow_steps", "mean_steps")}
        network["widths"] = candidate["widths"]
        training = {"format": "full_surface_training_v1", "network": network,
            "checkpoint_selection": "endpoint", "same_seed_reload_exact": True,
            "reserved_test_evaluated": False, "task_fingerprint": protocol["bank"]["task_fingerprint"],
            "selected_update": candidate["updates"], "performance": {"work": {"optimizer_updates": candidate["updates"]}},
            "run_contract": {"format": "full_surface_update_budget_run_v1", "network": network,
                "dataset": {"manifest_sha256": protocol["bank"]["manifest_sha256"]},
                "settings": {"checkpoint_selection": "endpoint", "seed": 0, "batch_size": 256,
                             "updates": candidate["updates"], "fresh_quote_noise": True, "steps_per_execution": 1}}}
        report = {"protocol_sha256": study.digest(study.DEFAULT_PROTOCOL), "parameter_names": ["v0", "theta", "xi", "rho"],
            "status": "pass", "gates": {"conditional_all_cases": True}, "reserved_test_evaluated": False,
            "development": {"example_ids": [f"validation:{i}" for i in range(512)],
                "coverage": {key: {"coverage": [float(key)] * 4} for key in ("0.9", "0.95")}},
            "source_sha256": {}, "performance": {"hardware": {"system": "Linux", "machine": "x86_64", "devices": ["gpu:0"]}}}
        control = copy.deepcopy(report)
        control.update(format="heston_training_budget_readiness_v1", status="inconclusive",
            control_checkpoint_sha256=protocol["control"]["checkpoint_sha256"],
            bank_manifest_sha256=protocol["bank"]["manifest_sha256"])
        control["performance"]["hardware"] = {"system": "Darwin", "machine": "arm64", "devices": ["cpu:0"]}
        return protocol, candidate, training, report, control

    def test_identity_rejects_wrong_budget_architecture_seed_or_bank(self):
        protocol, candidate, training, _, _ = self.fixture()
        self.assertEqual(evaluator.candidate_identity(protocol, candidate["id"], training), candidate)
        for mutate in (lambda d: d.update(selected_update=100),
                       lambda d: d["network"].update(widths=[256] * 4),
                       lambda d: d["run_contract"]["settings"].update(seed=1),
                       lambda d: d["run_contract"]["dataset"].update(manifest_sha256="changed"),
                       lambda d: d.update(same_seed_reload_exact=False)):
            bad = copy.deepcopy(training)
            mutate(bad)
            with self.assertRaisesRegex(ValueError, "exact endpoint"):
                evaluator.candidate_identity(protocol, candidate["id"], bad)

    def test_coverage_keeps_inconclusive_control_and_device_difference(self):
        protocol, _, _, report, control = self.fixture()
        compared = evaluator.compare_control(report, control, protocol)
        self.assertTrue(compared["coverage_within_margin"])
        self.assertEqual(compared["control_status_retained"], "inconclusive")
        self.assertFalse(compared["same_reported_device_platform"])
        report["development"]["coverage"]["0.9"]["coverage"][0] = .84
        self.assertFalse(evaluator.compare_control(report, control, protocol)["coverage_within_margin"])
        report["development"]["example_ids"].reverse()
        with self.assertRaisesRegex(ValueError, "IDs/order"):
            evaluator.compare_control(report, control, protocol)

    def test_completed_candidate_never_claims_final_or_publication_admission(self):
        protocol, candidate, training, report, control = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "training.json").write_text(json.dumps(training))
            control_path = root / "control.json"
            control_path.write_text(json.dumps(control))
            out = root / "evaluation"
            out.mkdir()
            args = Namespace(protocol=study.DEFAULT_PROTOCOL, checkpoint=checkpoint,
                             candidate=candidate["id"], control_readiness=control_path, out=out)
            with patch.object(evaluator.common, "run", return_value=report) as run:
                evaluated = evaluator.run(args)
            self.assertEqual(run.call_args.kwargs, {"role": "candidate"})
            self.assertFalse(evaluated["eligible_for_final_confirmation"])
            self.assertFalse(evaluated["model_publication_allowed"])
            self.assertFalse(evaluated["scientific_acceptance"])
            self.assertEqual(evaluated["execution_status"], "completed")
            control["protocol_sha256"] = "different"
            control_path.write_text(json.dumps(control))
            with patch.object(evaluator.common, "run") as expensive:
                with self.assertRaisesRegex(ValueError, "saved-control"):
                    evaluator.run(args)
                expensive.assert_not_called()


class MatrixWorkerTests(unittest.TestCase):
    def fixture(self, root):
        protocol = study.read_json(study.DEFAULT_PROTOCOL)
        bank = root / "bank"
        bank.mkdir()
        (bank / "manifest.json").write_text("{}")
        protocol["bank"]["manifest_sha256"] = study.digest(bank / "manifest.json")
        protocol_path = root / "protocol.json"
        protocol_path.write_text(json.dumps(protocol))
        control = {"format": "heston_training_budget_readiness_v1", "status": "inconclusive",
            "protocol_sha256": study.digest(protocol_path), "bank_manifest_sha256": protocol["bank"]["manifest_sha256"],
            "control_checkpoint_sha256": protocol["control"]["checkpoint_sha256"], "reserved_test_evaluated": False,
            "development": {"example_ids": [f"validation:{i}" for i in range(512)]}}
        control_path = root / "control.json"
        control_path.write_text(json.dumps(control))
        return Namespace(protocol=protocol_path, bank=bank, out=root / "lane",
            candidate=protocol["screen_candidates"][0]["id"], control_readiness=control_path, wall_seconds=60.)

    def test_failed_training_preserves_journal_without_launching_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            with patch.object(worker.subprocess, "run", return_value=Namespace(returncode=1)) as launch:
                with self.assertRaisesRegex(RuntimeError, "training execution failed"):
                    worker.run(args)
            self.assertEqual(launch.call_count, 1)
            journal = study.read_json(args.out / "matrix-worker.json")
            self.assertEqual(journal["status"], "failed_execution")
            self.assertEqual(journal["stages"][1]["status"], "pending")
            self.assertFalse(journal["automatic_retry"])

    def test_failed_quality_checks_are_a_complete_development_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            def child(command, **kwargs):
                if "evaluate_heston_budget_candidate.py" in command[1]:
                    out = args.out / "evaluation"
                    out.mkdir()
                    (out / "evaluation.json").write_text(json.dumps({"execution_status": "completed", "status": "fail",
                        "reserved_test_evaluated": False, "model_publication_allowed": False}))
                return Namespace(returncode=0)
            with patch.object(worker.subprocess, "run", side_effect=child) as launch:
                result = worker.run(args)
            self.assertEqual(launch.call_count, 2)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["development_checks_status"], "fail")
            self.assertFalse(result["reserved_test_evaluated"])
            with self.assertRaisesRegex(ValueError, "fresh lane"):
                worker.run(args)

    def test_timeout_is_retained_and_never_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            with patch.object(worker.subprocess, "run", side_effect=subprocess.TimeoutExpired("fit", 1)) as launch:
                with self.assertRaises(subprocess.TimeoutExpired):
                    worker.run(args)
            self.assertEqual(launch.call_count, 1)
            journal = study.read_json(args.out / "matrix-worker.json")
            self.assertEqual(journal["stages"][0]["status"], "timeout")
            for bad in (float("nan"), float("inf"), True, 2.):
                args.wall_seconds = bad
                with self.assertRaisesRegex(ValueError, "finite lane"):
                    worker.run(args)


if __name__ == "__main__":
    unittest.main()
