"""Controlled fixed-grid compression and native custom-summary persistence."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from sbi4f_tutorial.heston_options import experimental_inference as inference


TASK = {"target": {"parameter_names": list(inference.PARAMETER_NAMES), "quote_count": 63}}


class ConfigurationTests(unittest.TestCase):
    def test_configuration_is_json_safe_and_rejects_unsupported_representations(self):
        config = inference.configuration("mlp", 8)
        self.assertEqual(json.loads(json.dumps(config)), config)
        for representation, dimension in [("set", 8), ("direct", 8), ("mlp", None), ("mlp", 0), ("mlp", True)]:
            with self.subTest(representation=representation, dimension=dimension):
                with self.assertRaises(ValueError):
                    inference.configuration(representation, dimension)

    def test_import_does_not_initialize_the_backend(self):
        code = """
import sys
from sbi4f_tutorial.heston_options import experimental_inference
experimental_inference.configuration('mlp', 8)
assert not {'bayesflow', 'keras', 'jax'}.intersection(sys.modules)
"""
        run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)


@unittest.skipUnless(importlib.util.find_spec("bayesflow"), "optional BayesFlow participant dependency")
class NativeInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("KERAS_BACKEND", "jax")

    def test_adapter_keeps_six_raw_targets_and_excludes_truth_from_conditions(self):
        data = {name: np.full((2, 1), index - 3., dtype=np.float32)
                for index, name in enumerate(inference.PARAMETER_NAMES)}
        data["observations"] = np.arange(126, dtype=np.float32).reshape(2, 63)
        data["clean_prices"] = data["observations"] + 1
        data["noise_realization"] = np.ones((2, 63))
        for representation, dim, key in [("direct", None, "inference_conditions"), ("mlp", 4, "summary_variables")]:
            with self.subTest(representation=representation):
                workflow = inference.build_workflow(TASK, inference.configuration(representation, dim))
                adapted = workflow.adapter(data)
                self.assertEqual(set(adapted), {"inference_variables", key})
                np.testing.assert_array_equal(adapted[key], data["observations"])
                np.testing.assert_array_equal(adapted["inference_variables"],
                    np.concatenate([data[name] for name in inference.PARAMETER_NAMES], axis=-1))
        with self.assertRaisesRegex(ValueError, "only observations"):
            inference.sample(workflow, {"observations": data["observations"], "c": data["c"]},
                             inference.PARAMETER_NAMES, 2)

    def test_small_joint_fit_and_fresh_process_summary_reload(self):
        rng = np.random.default_rng(184)
        observations = rng.normal(size=(32, 63)).astype(np.float32)
        train = {name: (observations[:, i:i+1] + i).copy()
                 for i, name in enumerate(inference.PARAMETER_NAMES)}
        train["observations"] = observations
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for representation, dimension in [("direct", None), ("mlp", 4)]:
                config = inference.configuration(representation, dimension, summary_widths=(8,),
                                                 posterior_widths=(8, 8))
                workflow = inference.build_workflow(TASK, config, seed=72)
                history = workflow.fit_offline(train, epochs=1, batch_size=16, verbose=0)
                self.assertTrue(np.isfinite(history.history["loss"]).all())
                draws = inference.sample(workflow, observations[:2], inference.PARAMETER_NAMES, 5, seed=91)
                self.assertEqual(draws.shape, (2, 5, 6))
                if representation == "direct":
                    continue
                self.assertEqual(tuple(workflow.summary_network(observations[:2]).shape), (2, 4))
                workflow.estimator.save(directory / "model.keras")
                np.save(directory / "conditions.npy", observations[:2])
                (directory / "config.json").write_text(json.dumps(config))
                code = """
import json, sys
from pathlib import Path
import numpy as np
from sbi4f_tutorial.heston_options import experimental_inference as inference
root = Path(sys.argv[1])
task = {'target': {'parameter_names': list(inference.PARAMETER_NAMES), 'quote_count': 63}}
workflow = inference.build_workflow(task, json.loads((root/'config.json').read_text()), seed=72)
workflow.load_estimator(root/'model.keras')
draws = inference.sample(workflow, np.load(root/'conditions.npy'), inference.PARAMETER_NAMES, 5, seed=91)
np.save(root/'reloaded.npy', draws)
"""
                run = subprocess.run([sys.executable, "-c", code, str(directory)], capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                np.testing.assert_allclose(np.load(directory / "reloaded.npy"), draws, rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
