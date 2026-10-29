"""The small runner retains failures and never authorizes downstream cases."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sbi4finance.calibration import bank_collection
from scripts import run_heston_reference_case as runner


class HestonCaseRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = argparse.Namespace(bank=self.root/"bank", checkpoint=self.root/"checkpoint",
                                       index=1, out=self.root/"run")

    def reports(self, *, product_pass=True):
        files = {
            "reference/diagnostics.json": {"mcmc_diagnostics_passed": True,
                "variance_functionals": {"mcmc_diagnostics_passed": product_pass}},
            "precision/precision.json": {"all_points_priced": True, "pricing_sources_match_reference": True},
            "comparison/comparison.json": {"status": "completed_development_comparison",
                "proposed_parameter_agreement": {"all_checks_passed": False},
                "variance_product_comparison": {"status": "descriptive_development_comparison"}},
        }
        for name, report in files.items():
            path = self.args.out/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report))

    def run_mocked(self, child):
        with (patch.dict("os.environ", {"SLURM_JOB_ID": "software-test", "SLURM_CPUS_PER_TASK": "4"}),
              patch.object(runner.metadata, "version", return_value="2.1.0"),
              patch.object(runner, "validate_inputs", return_value={"scope": "software fixture"}),
              patch.object(runner.subprocess, "check_output", return_value="test-source\n"),
              patch.object(runner.subprocess, "run", side_effect=child)):
            return runner.run(self.args)

    def test_reviewed_commands_keep_exact_seed_budget_and_existing_tools(self):
        stages = runner.commands(self.args.bank, self.args.checkpoint, 1, self.args.out)
        self.assertEqual([name for name, _ in stages], ["reference", "precision", "comparison"])
        reference = stages[0][1]
        for name, value in {"--tune": "4000", "--draws": "8000", "--chains": "4", "--cores": "4",
                            "--seed": "271027", "--initialization": "data-fit", "--fit-starts": "4",
                            "--fit-budget": "2000", "--mp-context": "spawn"}.items():
            self.assertEqual(reference[reference.index(name)+1], value)
        self.assertIn("16", stages[1][1])
        self.assertTrue(stages[2][1][1].endswith("compare_full_surface_reference.py"))
        with self.assertRaises(ValueError):
            runner.commands(self.args.bank, self.args.checkpoint, 3, self.args.out)

    def test_success_keeps_manual_review_pending_even_when_neural_checks_fail(self):
        def child(command, **kwargs):
            self.reports()
            return subprocess.CompletedProcess(command, 0)
        self.assertEqual(self.run_mocked(child), 0)
        state = json.loads((self.args.out/"run.json").read_text())
        self.assertEqual(state["status"], "ready_for_manual_review")
        self.assertEqual(state["trace_and_product_review"], "pending_manual_review")
        self.assertFalse(state["scientific_acceptance"])
        self.assertFalse(state["next_cases_authorized"])
        self.assertFalse(state["guards"]["proposed_neural_parameter_checks_passed"])
        self.assertEqual([row["status"] for row in state["stages"]], ["completed"]*3)

    def test_product_failure_is_independent_of_parameter_pass(self):
        def child(command, **kwargs):
            self.reports(product_pass=False)
            return subprocess.CompletedProcess(command, 0)
        self.assertEqual(self.run_mocked(child), 2)
        state = json.loads((self.args.out/"run.json").read_text())
        self.assertEqual(state["status"], "inconclusive_reference")
        self.assertTrue(state["guards"]["parameter_mcmc_passed"])
        self.assertFalse(state["guards"]["analytic_product_mcmc_passed"])
        self.assertFalse(state["next_cases_authorized"])

    def test_child_failure_retains_command_failure_and_pending_stages(self):
        with self.assertRaisesRegex(RuntimeError, "reference failed"):
            self.run_mocked(lambda command, **kwargs: subprocess.CompletedProcess(command, 7))
        state = json.loads((self.args.out/"run.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["stages"][0]["returncode"], 7)
        self.assertEqual(state["stages"][1]["status"], "pending")
        self.assertTrue((self.args.out/"reference.log").exists())
        self.assertFalse(state["sampler_resumable"])

    def test_refuses_login_compute_before_creating_outputs(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(ValueError, "allocation"):
            runner.run(self.args)
        self.assertFalse(self.args.out.exists())

    def test_preflight_rejects_changed_quotes_and_changed_task(self):
        anchor_path = runner.ROOT/"docs/pilot-results/2026-10-26/heston-reference-parallel/case-1/reference.json"
        anchor = json.loads(anchor_path.read_text())
        self.args.bank.mkdir()
        self.args.checkpoint.mkdir()
        (self.args.checkpoint/"task.json").write_text(json.dumps(anchor["task"]))
        (self.args.checkpoint/"model.keras").write_bytes(b"software fixture")
        training = {"network": {"family": "spline"}, "run_contract": {"settings": {"epochs": 300, "seed": 0}},
                    "checkpoint_sha256": runner.digest(self.args.checkpoint/"model.keras")}
        (self.args.checkpoint/"training.json").write_text(json.dumps(training))
        observations = np.broadcast_to(np.asarray(anchor["observed_prices"])[None, :, None], (3, 63, 1)).copy()
        manifest = {"configuration": {"task": copy.deepcopy(anchor["task"])}}
        data = {"validation": {"observations": observations, "example_id": ["validation:0", "validation:1", "validation:2"]}}
        real_digest = runner.digest
        def fixture_digest(path):
            return anchor["bank_manifest_sha256"] if path == self.args.bank/"manifest.json" else real_digest(path)
        with (patch.object(bank_collection, "load_bank", return_value=(manifest, data)),
              patch.object(runner, "digest", side_effect=fixture_digest)):
            self.assertEqual(runner.validate_inputs(self.args.bank, self.args.checkpoint, 1)["surface_id"], "validation:1")
            observations[1, 0, 0] += 1e-8
            with self.assertRaisesRegex(ValueError, "exactly"):
                runner.validate_inputs(self.args.bank, self.args.checkpoint, 1)
            observations[1, 0, 0] = anchor["observed_prices"][0]
            manifest["configuration"]["task"]["extra_changed_contract"] = True
            with self.assertRaisesRegex(ValueError, "exactly"):
                runner.validate_inputs(self.args.bank, self.args.checkpoint, 1)


if __name__ == "__main__":
    unittest.main()
