"""Identity, test-split isolation and bounded stopping for the two Heston fits."""

import argparse
import copy
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_heston_training_followup as runner


class FollowupGuards(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = argparse.Namespace(fit="spline-1000", bank=self.root/"bank", baseline=self.root/"baseline",
                                       references=self.root/"references", out=self.root/"run")
        self.recipe = json.loads((runner.ROOT/runner.RECIPES[self.args.fit]).read_text())

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def fixture(self):
        self.versions = {name: "fixture" for name in ("keras", "jax", "numpy", "scipy")}
        self.versions["bayesflow"] = "2.1.0"
        runtime = {"python": platform.python_version(), "keras_backend_env": "jax", **self.versions}
        task = {"scope": "metadata-only Heston fixture"}
        manifest = {"configuration": {"task": task, "splits": self.recipe["splits"]},
                    "task_fingerprint": self.recipe["task_fingerprint"]}
        self.write(self.args.bank/"manifest.json", manifest)
        self.recipe["bank_manifest_sha256"] = runner.digest(self.args.bank/"manifest.json")
        self.write(self.args.baseline/"task.json", task)
        (self.args.baseline/"model.keras").write_bytes(b"checkpoint fixture")
        self.recipe["baseline_checkpoint_sha256"] = runner.digest(self.args.baseline/"model.keras")
        for name in ("pyproject.toml", "pixi.lock"):
            (self.root/name).write_text("locked fixture")
        training = {"task_fingerprint": self.recipe["task_fingerprint"],
                    "checkpoint_sha256": self.recipe["baseline_checkpoint_sha256"],
                    "run_contract": {"dataset": {"manifest_sha256": self.recipe["bank_manifest_sha256"]},
                        "runtime": runtime, "settings": {"epochs": 300, "seed": 0},
                        "environment_sha256": {name: runner.digest(self.root/name) for name in ("pyproject.toml", "pixi.lock")}}}
        self.write(self.args.baseline/"training.json", training)
        for index in (0, 1, 2):
            case = self.args.references/f"case-{index}"
            self.write(case/"reference/reference.json", {"task": task, "surface_id": f"validation:{index}",
                "settings": {"seed": 271026+index}, "observed_prices": [index]})
            self.write(case/"reference/diagnostics.json", {"mcmc_diagnostics_passed": True,
                "variance_functionals": {"mcmc_diagnostics_passed": True}})
            (case/"reference/chains.npz").write_bytes(b"chain fixture")
            self.write(case/"precision/precision.json", {"all_points_priced": True, "pricing_sources_match_reference": True,
                "input_sha256": {name: runner.digest(case/"reference"/name)
                                  for name in ("reference.json", "diagnostics.json", "chains.npz")}})
            shutil.copytree(case, self.root/f"docs/pilot-results/2026-10-27/heston-reviewed/case-{index}")

    def validate(self):
        with (patch.object(runner, "ROOT", self.root), patch.dict("os.environ", {"KERAS_BACKEND": "jax"}),
              patch.object(runner.metadata, "version", side_effect=self.versions.__getitem__)):
            return runner.validate_inputs(self.args.bank, self.args.baseline, self.args.references, self.recipe)

    def test_metadata_preflight_never_requires_or_opens_test_arrays(self):
        self.fixture()
        record = self.validate()
        self.assertEqual(len(record["reference_sha256"]), 3)
        self.assertEqual(list(self.args.bank.glob("*.npz")), [])
        self.assertIn("no test arrays", record["test_access"])

    def test_changed_snapshot_or_environment_is_rejected(self):
        self.fixture()
        (self.root/"pixi.lock").write_text("changed lock")
        with self.assertRaisesRegex(ValueError, "environment bytes"):
            self.validate()
        (self.args.bank/"manifest.json").write_text((self.args.bank/"manifest.json").read_text()+" ")
        with self.assertRaisesRegex(ValueError, "original20k"):
            self.validate()

    def test_changed_observation_or_chain_payload_is_rejected(self):
        self.fixture()
        reference = self.args.references/"case-1/reference/reference.json"
        original = reference.read_text()
        changed = json.loads(original)
        changed["observed_prices"][0] += .01
        self.write(reference, changed)
        with self.assertRaisesRegex(ValueError, "case1"):
            self.validate()
        reference.write_text(original)
        (self.args.references/"case-1/reference/chains.npz").write_bytes(b"different retained draws")
        with self.assertRaisesRegex(ValueError, "case1"):
            self.validate()

    def test_failed_product_diagnostics_never_become_a_valid_reference(self):
        self.fixture()
        case = self.args.references/"case-2"
        diagnostics = json.loads((case/"reference/diagnostics.json").read_text())
        diagnostics["variance_functionals"]["mcmc_diagnostics_passed"] = False
        self.write(case/"reference/diagnostics.json", diagnostics)
        self.write(self.root/"docs/pilot-results/2026-10-27/heston-reviewed/case-2/reference/diagnostics.json", diagnostics)
        with self.assertRaisesRegex(ValueError, "case2"):
            self.validate()

    def test_two_recipes_share_data_limits_and_only_request_development_evaluation(self):
        for name, path in runner.RECIPES.items():
            recipe = json.loads((runner.ROOT/path).read_text())
            stages = runner.commands(self.args, recipe)
            self.assertEqual(sum(cap for _, _, cap in stages), 1740)
            self.assertLess(recipe["execution_budget"]["total_seconds"], recipe["resources"]["wall_seconds"])
            self.assertEqual(recipe["resources"]["cpus"], 4)
            self.assertEqual(recipe["resources"]["gpus"], 1)
            self.assertEqual(recipe["evaluation"]["split"], "validation")
            self.assertEqual(recipe["reference_comparison"]["case_indices"], [0, 1, 2])
            self.assertTrue(all("--split" not in command or command[command.index("--split")+1] == "validation"
                                for _, command, _ in stages))
            self.assertIn("--require-gpu", stages[0][1])
            self.assertEqual(recipe["training"]["epochs"], 1000 if name == "spline-1000" else 500)

    def test_stage_timeout_retains_failure_and_does_not_start_training(self):
        environment = {"SLURM_JOB_ID": "software-only", "SLURM_CPUS_PER_TASK": "4",
                       **{name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
        with (patch.dict("os.environ", environment), patch.object(runner, "validate_inputs", return_value={}),
              patch.object(runner.subprocess, "check_output", return_value="source-fixture\n"),
              patch.object(runner.subprocess, "run", side_effect=subprocess.TimeoutExpired("environment", 60)) as child):
            with self.assertRaises(subprocess.TimeoutExpired):
                runner.run(self.args)
        state = json.loads((self.args.out/"run.json").read_text())
        self.assertEqual(state["status"], "incomplete")
        self.assertEqual(state["stages"][0]["status"], "timeout")
        self.assertEqual(state["stages"][1]["status"], "pending")
        self.assertFalse(state["automatic_retry"])
        self.assertFalse(state["reserved_test_evaluated"])
        child.assert_called_once()

    def test_login_execution_is_rejected_before_output_creation(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(ValueError, "allocation"):
            runner.run(self.args)
        self.assertFalse(self.args.out.exists())


if __name__ == "__main__":
    unittest.main()
