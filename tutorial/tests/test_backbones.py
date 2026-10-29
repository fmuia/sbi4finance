"""New examples remain bound to their exact inputs; default reads initialize no neural backend."""
import ast
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from sbi4f_tutorial import backbones, full_surface
from sbi4f_tutorial.paths import ASSETS


class BackboneTests(unittest.TestCase):
    def test_all_current_development_bundles_validate_and_keep_known_joint_boundaries(self):
        self.assertEqual(len(backbones.examples()), 6)
        for name in backbones.examples():
            with self.subTest(name=name):
                task, report, arrays = backbones.load_example(name)
                full_surface.validate_task(task)
                self.assertFalse(report["scientific_acceptance"])
                data = full_surface.conditioning_data(arrays["observations"][None], task,
                    context_values=None if "context_values" not in arrays else arrays["context_values"][None])
                self.assertEqual("curve_context" in data, name.startswith("rough-known"))
                self.assertEqual(arrays["neural_draws"].shape[-1], len(task["parameter_names"]))
                self.assertEqual(len(backbones.parameter_rows(task, arrays)), len(task["parameter_names"]))

    def test_default_replay_imports_no_neural_backend_even_when_unavailable(self):
        program = '''
import importlib.abc, sys
class Offline(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'bayesflow','keras','jax','sbi4finance'}:
            raise AssertionError('cached import initialized neural backend: '+fullname)
sys.meta_path.insert(0,Offline())
from sbi4f_tutorial import backbones
for name in backbones.examples():
    backbones.load_example(name)
'''
        result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_live_loading_rejects_paths_outside_asset_cache_before_model_import(self):
        for path in (ASSETS, ASSETS / "checkpoints" / "heston", Path("/tmp")):
            with self.subTest(path=str(path)), self.assertRaisesRegex(ValueError, "local-checkpoints"):
                backbones.load_live("heston-final500", path)

    def test_no_notebook_uses_archived_checkpoint_loading_or_implicit_toy_training(self):
        notebooks = Path(__file__).resolve().parents[1] / "notebooks"
        for path in notebooks.glob("*.py"):
            text = path.read_text()
            ast.parse(text)
            self.assertNotIn("load_approximator", text, path.name)
            self.assertNotIn("teaching.load_network", text, path.name)
        toy = (notebooks / "slot2_amortized_bayes.py").read_text()
        self.assertNotIn("or not TOY_CHECKPOINT.exists()", toy)
        self.assertIn("mo.stop(not train_button.value", toy)


if __name__ == "__main__":
    unittest.main()
