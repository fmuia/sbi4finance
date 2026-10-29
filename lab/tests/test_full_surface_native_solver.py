"""Use upstream defaults for new flows without reinterpreting saved Euler runs."""

from contextlib import nullcontext, redirect_stdout
import copy
import hashlib
from importlib import metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from sbi4finance.calibration import full_surface, tasks
from sbi4finance.gbm_bank import _json

ROOT = Path(__file__).resolve().parents[1]
try:
    CORE_ENVIRONMENT = metadata.version("bayesflow") == "2.1.0"
except metadata.PackageNotFoundError:
    CORE_ENVIRONMENT = False


class NativeSolverConfigurationTests(unittest.TestCase):
    def test_default_is_native_and_explicit_old_step_count_is_preserved(self):
        self.assertIsNone(full_surface.configuration(family="flow_matching")["flow_steps"])
        self.assertEqual(full_surface.configuration(family="flow_matching", flow_steps=100)["flow_steps"], 100)
        for invalid in (True, 0, -1, "adaptive"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                full_surface.configuration(family="flow_matching", flow_steps=invalid)

    def test_training_cli_does_not_reintroduce_an_implicit_euler_override(self):
        spec = importlib.util.spec_from_file_location("native_solver_training_cli", ROOT / "scripts/train_full_surface.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for arguments, expected in (([], None), (["--flow-steps", "100"], 100)):
            with (self.subTest(arguments=arguments),
                  mock.patch("sys.argv", ["train_full_surface.py", "--bank", "/unused/bank",
                                          "--out", "/unused/run", "--family", "flow_matching", *arguments]),
                  mock.patch.object(module.recovery, "training_lock", return_value=nullcontext()),
                  mock.patch.object(module, "run_training", return_value={"selected_epoch": 1}) as training,
                  redirect_stdout(io.StringIO())):
                module.main()
                self.assertEqual(training.call_args.args[0].flow_steps, expected)


@unittest.skipUnless(CORE_ENVIRONMENT, "requires the locked BayesFlow2.1 core environment")
class ActualNativeSolverTests(unittest.TestCase):
    def test_actual_factory_matches_upstream_default_and_explicit_legacy_choice(self):
        os.environ.setdefault("KERAS_BACKEND", "jax")
        import bayesflow as bf
        task = tasks.heston_task()
        upstream = bf.networks.FlowMatching(subnet_kwargs={"widths": [8, 8]})
        native = full_surface.build_workflow(task, family="flow_matching", widths=(8, 8))
        legacy = full_surface.build_workflow(task, family="flow_matching", widths=(8, 8), flow_steps=100)
        self.assertEqual(native.inference_network.integrate_kwargs, upstream.integrate_kwargs)
        self.assertEqual(native.inference_network.integrate_kwargs, {"method": "tsit5", "steps": "adaptive"})
        self.assertEqual(legacy.inference_network.integrate_kwargs, {"method": "euler", "steps": 100})

    def test_native_and_legacy_checkpoint_reload_preserves_actual_sampler_and_draws(self):
        os.environ.setdefault("KERAS_BACKEND", "jax")
        import keras
        task = tasks.heston_task()
        bounds = np.asarray(task["prior_bounds"])
        rng = np.random.default_rng(438)
        split = {"observations": rng.normal(size=(16, 63, 1)) * .001,
                 "parameters": rng.uniform(bounds[:, 0], bounds[:, 1], size=(16, 4))}
        data = full_surface.training_data(split, task)
        # One epoch on dummy labels only builds the serializer/adapter; no
        # simulator, model-quality conclusion or experiment bank is involved.
        for flow_steps in (None, 4):
            with self.subTest(flow_steps=flow_steps), tempfile.TemporaryDirectory() as temporary:
                keras.utils.set_random_seed(438)
                directory = Path(temporary)
                settings = full_surface.configuration(family="flow_matching", widths=(8, 8), flow_steps=flow_steps)
                workflow = full_surface.build_workflow(task, **settings)
                workflow.fit_offline(data=data, validation_data=data, epochs=1, batch_size=8, verbose=0)
                before = full_surface.sample(workflow, task, split["observations"][:1], 4, seed=74)
                workflow.estimator.save(directory / "model.keras")
                model_bytes = (directory / "model.keras").read_bytes()
                record = {"format": "full_surface_training_v1", "network": settings,
                          "run_contract": {"network": settings},
                          "task_fingerprint": hashlib.sha256(_json(task).encode()).hexdigest(),
                          "checkpoint_sha256": hashlib.sha256(model_bytes).hexdigest()}
                if flow_steps is None:
                    record["flow_matching_integrate_kwargs"] = dict(workflow.inference_network.integrate_kwargs)
                # An old explicit-Euler record deliberately has no new field.
                (directory / "task.json").write_text(json.dumps(task))
                (directory / "training.json").write_text(json.dumps(record))
                _, restored, _ = full_surface.load_checkpoint(directory)
                self.assertEqual(restored.inference_network.integrate_kwargs, workflow.inference_network.integrate_kwargs)
                after = full_surface.sample(restored, task, split["observations"][:1], 4, seed=74)
                np.testing.assert_array_equal(before, after)
                self.assertEqual((directory / "model.keras").read_bytes(), model_bytes)
                altered = copy.deepcopy(record)
                changed_steps = 4 if flow_steps is None else None
                altered["network"]["flow_steps"] = changed_steps
                altered["run_contract"]["network"]["flow_steps"] = changed_steps
                (directory / "training.json").write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError, "sampler"):
                    full_surface.load_checkpoint(directory)


if __name__ == "__main__":
    unittest.main()
