"""Native defaults, immutable inputs and call-time overrides without model calls."""

import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scripts import check_cached_heston_native as native
from sbi4finance.calibration import tasks


class CachedNativeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.task = tasks.heston_task()
        self.bounds = np.asarray(self.task["prior_bounds"])

    def workflow(self, *, mutate=False):
        result = SimpleNamespace(inference_network=SimpleNamespace(integrate_kwargs={"method": "euler", "steps": 100}))
        values = np.stack([self.bounds.mean(axis=1), self.bounds[:, 0]*.25+self.bounds[:, 1]*.75])
        def sample(**kwargs):
            if mutate:
                result.inference_network.integrate_kwargs["steps"] = "adaptive"
            return {name: values[None, :, j:j+1] for j, name in enumerate(self.task["parameter_names"])}
        result.sample = Mock(side_effect=sample)
        return result

    def test_installed_native_defaults_are_read_without_loading_a_model(self):
        settings, sources = native.native_settings()
        self.assertEqual(settings, {"method": "tsit5", "steps": "adaptive", "min_steps": 50,
                                    "max_steps": 10000, "atol": 1e-6, "rtol": 1e-4})
        self.assertGreaterEqual(len(sources), 2)
        self.assertTrue(all(len(value) == 64 for value in sources.values()))

    def test_only_native_method_and_steps_override_the_stored_solver(self):
        workflow = self.workflow()
        original = copy.deepcopy(workflow.inference_network.integrate_kwargs)
        values = native.native_draws(workflow, self.task, np.zeros(63), 2, 271027)
        self.assertEqual(values.shape, (2, 4))
        kwargs = workflow.sample.call_args.kwargs
        self.assertEqual(kwargs["method"], "tsit5")
        self.assertEqual(kwargs["steps"], "adaptive")
        self.assertEqual(kwargs["seed"], 271027)
        self.assertIs(type(kwargs["seed"]), int)
        self.assertEqual(kwargs["sample_shape"], ())
        self.assertNotIn("atol", kwargs)
        self.assertNotIn("max_steps", kwargs)
        self.assertNotIn("integrate_kwargs", kwargs)
        self.assertEqual(workflow.inference_network.integrate_kwargs, original)

    def test_mutated_stored_solver_and_invalid_draws_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "mutated"):
            native.native_draws(self.workflow(mutate=True), self.task, np.zeros(63), 2, 271027)
        workflow = self.workflow()
        workflow.sample = Mock(return_value={name: np.full((1, 2, 1), np.nan) for name in self.task["parameter_names"]})
        with self.assertRaisesRegex(ValueError, "invalid physical"):
            native.native_draws(workflow, self.task, np.zeros(63), 2, 271027)

    def test_identical_draws_have_zero_paired_change_and_matching_summaries(self):
        rng = np.random.default_rng(427)
        chains = rng.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(4, 128, 4))
        draws = chains.reshape(-1, 4)
        diagnostics = {"variance_functionals": {"names": {name: {"r_hat": 1., "ess_bulk": 512, "ess_tail": 512}
            for name in ("variance_swap_1y", "forward_variance_0_5_to_1y")}}}
        report = native.score(self.task, draws, draws.copy(), chains, diagnostics)
        self.assertTrue(report["proposed_parameter_agreement"]["all_checks_passed"])
        for row in report["paired_parameter_changes_from_euler"].values():
            self.assertEqual(row["paired_rms_change_over_reference_sd"], 0.)
            self.assertEqual(row["signed_mean_change_over_reference_sd"], 0.)
        for row in report["paired_analytic_product_changes_from_euler"].values():
            self.assertEqual(row["paired_rms_change"], 0.)
            self.assertEqual(row["signed_q10_q90_change"], [0., 0.])

    def test_actual_prettyprinted_metadata_and_changed_euler_draw_guard(self):
        euler, references = self.root/"euler", self.root/"references"
        shutil.copytree(native.OLD, euler)
        shutil.copytree(native.REFERENCES, references)
        model = euler/"checkpoint/model.keras"
        model.write_bytes(b"software-only model placeholder; never loaded")
        recipe = json.loads(native.RECIPE.read_text())
        actual_digest = native.digest
        def placeholder_digest(path):
            return recipe["checkpoint_sha256"] if Path(path) == model else actual_digest(path)
        with patch.dict("os.environ", {"KERAS_BACKEND": "jax"}), patch.object(native, "digest", side_effect=placeholder_digest):
            checked = native.checked_inputs(euler, references, recipe)
            self.assertEqual([row["seed"] for row in checked["cases"]], [271026, 271027, 271028])
            task_path = euler/"checkpoint/task.json"
            original = task_path.read_bytes()
            task_path.write_text(json.dumps(json.loads(original), indent=1)+"\n")
            self.assertNotEqual(actual_digest(task_path), recipe["task_fingerprint"])
            reformatted = native.checked_inputs(euler, references, recipe)
            self.assertEqual(reformatted["task"], checked["task"])
            self.assertEqual(reformatted["sha256"][str(task_path)], actual_digest(task_path))
            (euler/"comparisons/case-1/neural_draws.npz").write_bytes(b"different Euler outputs")
            with self.assertRaisesRegex(ValueError, "case1"):
                native.checked_inputs(euler, references, recipe)

    def test_manifest_is_written_before_model_loading(self):
        args = argparse.Namespace(out=self.root/"out", euler_run=self.root/"euler", references=self.root/"refs")
        inputs = {"checkpoint": "frozen-checkpoint", "task": self.task, "cases": [], "sha256": {}}
        def stop_before_loading(path):
            frozen = json.loads((args.out/"run.json").read_text())
            self.assertEqual(frozen["status"], "inputs_frozen")
            self.assertEqual(frozen["call_time_overrides"], {"method": "tsit5", "steps": "adaptive"})
            raise ValueError("software stop before actual model loading")
        fake_jax = SimpleNamespace(devices=lambda: [SimpleNamespace(platform="gpu", device_kind="fake-software-device")])
        with (patch.dict("os.environ", {"SLURM_JOB_ID": "software", "SLURM_CPUS_PER_TASK": "4"}),
              patch.dict("sys.modules", {"jax": fake_jax}), patch.object(native, "checked_inputs", return_value=inputs),
              patch.object(native.subprocess, "check_output", return_value="software-source\n"),
              patch.object(native.signal, "signal"),
              patch.object(native.full_surface, "load_checkpoint", side_effect=stop_before_loading)):
            with self.assertRaisesRegex(ValueError, "software stop"):
                native.run(args)
        state = json.loads((args.out/"run.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertFalse(state["scientific_acceptance"])

    def test_no_login_execution_or_validation_panel(self):
        args = argparse.Namespace(out=self.root/"not-created")
        with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(ValueError, "allocation"):
            native.run(args)
        self.assertFalse(args.out.exists())
        recipe = json.loads(native.RECIPE.read_text())
        self.assertEqual(recipe["case_indices"], [0, 1, 2])
        self.assertEqual(recipe["resources"], {"cpus": 4, "gpus": 1, "memory_gib": 24, "wall_seconds": 900})


if __name__ == "__main__":
    unittest.main()
