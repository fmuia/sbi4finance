"""One editable runtime implementation, discoverable without the lab or a backend."""

import ast
import importlib
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from sbi4f_tutorial.trajectories import inference, lesson, simulation


RUNTIME = Path(__file__).resolve().parents[1] / "src" / "sbi4f_tutorial"


class RuntimeOwnershipTests(unittest.TestCase):
    def test_flat_compatibility_modules_define_no_second_implementation(self):
        for path in RUNTIME.glob("*.py"):
            with self.subTest(module=path.stem):
                nodes = ast.parse(path.read_text()).body
                self.assertFalse(any(isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign))
                                     for node in nodes), "edit the canonical application/shared module")

    def test_legacy_imports_are_the_same_functions_as_the_application_sources(self):
        checks = {
            "teaching.heston_illustrative_paths": "heston_options.lesson.heston_illustrative_paths",
            "teaching.heston_posterior_predictive": "heston_options.lesson.heston_posterior_predictive",
            "teaching.price_rough_smile": "rough_options.lesson.price_rough_smile",
            "hkde.european_price": "heston_options.model.european_price",
            "rough_bergomi.simulate_rough": "rough_options.simulator.simulate_rough",
            "full_surface.conditioning_data": "shared.full_surface.conditioning_data",
            "full_surface.load_checkpoint": "shared.full_surface.load_checkpoint",
            "trajectory.conditioning_data": "trajectories.inference.conditioning_data",
            "trajectory.sample": "trajectories.inference.sample",
            "trajectory.load_example": "trajectories.lesson.load_example",
        }
        for old, canonical in checks.items():
            with self.subTest(old=old):
                old_module, old_name = old.rsplit(".", 1)
                module, name = canonical.rsplit(".", 1)
                self.assertIs(getattr(importlib.import_module(f"sbi4f_tutorial.{old_module}"), old_name),
                              getattr(importlib.import_module(f"sbi4f_tutorial.{module}"), name))

    def test_complete_runtime_is_importable_without_the_lab_or_neural_backend(self):
        program = '''
import importlib, importlib.abc, pkgutil, sys
class Offline(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'bayesflow', 'keras', 'jax', 'numba', 'sbi4finance'}:
            raise AssertionError('runtime import initialized an optional dependency: ' + fullname)
sys.meta_path.insert(0, Offline())
import sbi4f_tutorial
for info in pkgutil.walk_packages(sbi4f_tutorial.__path__, sbi4f_tutorial.__name__ + '.'):
    importlib.import_module(info.name)
'''
        run = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)


class TrajectoryTrainingRecipeTests(unittest.TestCase):
    def test_training_targets_and_random_prefixes_preserve_the_shipped_task(self):
        for model in lesson.MODEL_LABELS:
            with self.subTest(model=model):
                task, _, arrays = lesson.load_example(model)
                pairs = {"observations": arrays["observations"][:2], "parameters": arrays["truth"][:2]}
                train = inference.training_data(pairs, task)
                np.testing.assert_allclose(inference.inverse_parameters(
                    inference.transform_parameters(pairs["parameters"], task), task), pairs["parameters"],
                    rtol=2e-14, atol=2e-14)
                augment = inference.random_prefixes(task, seed=2026, policy="uniform_log_mixture")
                replay = inference.random_prefixes(task, seed=2026, policy="uniform_log_mixture")
                result = augment(train)
                expected = replay(train)
                for name in result:
                    np.testing.assert_array_equal(result[name], expected[name])
                lengths = result["summary_mask"].sum(axis=1)
                self.assertTrue(np.all((lengths >= 1) & (lengths <= task["n_observations"])))
                np.testing.assert_array_equal(result["parameters"], train["parameters"])
                np.testing.assert_array_equal(result["observations"][~result["summary_mask"]], 0)
                np.testing.assert_array_equal(augment.exposure.sum(), len(pairs["parameters"]))


class ParticipantSimulationTests(unittest.TestCase):
    def test_task_law_matches_the_shipped_daily_examples(self):
        for model in simulation.MODELS:
            with self.subTest(model=model):
                published, _, _ = lesson.load_example(model)
                self.assertEqual(simulation.teaching_task(model), published)

    def test_numpy_pairs_are_chunk_invariant_and_hide_latent_states(self):
        for model in simulation.MODELS:
            with self.subTest(model=model):
                task = simulation.teaching_task(model, observations=29)
                whole = simulation.generate(task, "train", 0, 4, seed=2026, backend="numpy")
                tail = simulation.generate(task, "train", 2, 2, seed=2026, backend="numpy")
                self.assertEqual(set(whole), {"observations", "parameters", "trajectory_id"})
                for name, values in whole.items():
                    np.testing.assert_array_equal(values[2:], tail[name])
                self.assertEqual(whole["observations"].shape, (4, 29, 1))
                self.assertEqual(whole["parameters"].shape, (4, len(task["parameter_names"])))
                self.assertTrue(np.isfinite(whole["observations"]).all())
                self.assertEqual(len(set(whole["trajectory_id"])), 4)
                self.assertTrue(all(":train:" in name for name in whole["trajectory_id"]))
                lengths = np.asarray([1, 7, 13, 29])
                train = inference.training_data(whole, task, lengths=lengths)
                self.assertEqual(set(train), {"observations", "time_context", "summary_mask", "parameters"})
                self.assertEqual(train["observations"].shape, (4, 29, 2))
                np.testing.assert_array_equal(train["summary_mask"], np.arange(29) < lengths[:, None])
                np.testing.assert_array_equal(train["observations"][~train["summary_mask"]], 0)
                np.testing.assert_allclose(train["time_context"][:, 0], lengths / 29)
                np.testing.assert_array_equal(train["parameters"],
                                              inference.transform_parameters(whole["parameters"], task).astype(np.float32))

    def test_small_generation_needs_neither_numba_nor_a_neural_backend(self):
        program = '''
import importlib.abc, sys
class Offline(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'bayesflow', 'keras', 'jax', 'numba', 'sbi4finance'}:
            raise AssertionError('small NumPy simulation imported an optional dependency: ' + fullname)
sys.meta_path.insert(0, Offline())
from sbi4f_tutorial.trajectories import simulation, inference
for model in simulation.MODELS:
    task = simulation.teaching_task(model, observations=7)
    pairs = simulation.generate(task, 'train', 0, 2, seed=2026, backend='numpy')
    train = inference.training_data(pairs, task)
    assert train['observations'].shape == (2, 7, 2)
'''
        run = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == "__main__":
    unittest.main()
