"""Trajectory examples remain offline, asset-local and bound to exact prefixes."""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from sbi4f_tutorial.trajectories import lesson as trajectory


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def fixture_task(model):
    names = {"heston": ["kappa", "long_variance", "vol_of_vol", "rho"],
             "rough-known": ["hurst", "eta", "rho"],
             "rough-joint": ["hurst", "eta", "rho", "short_variance", "long_variance", "decay"]}[model]
    bounds = {"heston": [[.25, 5], [.01, .09], [.05, .6], [-.95, -.05]],
              "rough-known": [[.03, .4], [.2, 2.5], [-.95, -.05]],
              "rough-joint": [[.03, .4], [.2, 2.5], [-.95, -.05], [.01, .09], [.01, .09], [.25, 8]]}[model]
    kinds = {"heston": ["log_uniform", "log_uniform", "uniform", "uniform"],
             "rough-known": ["uniform"] * 3,
             "rough-joint": ["uniform"] * 3 + ["log_uniform"] * 3}[model]
    return {"task_id": f"physical_{model.replace('-', '_')}_daily_drift_paths_v2", "task_version": 2,
            "model": model, "measure": "P", "parameter_names": names + ["drift"],
            "prior_bounds": bounds + [[-.1, .2]], "prior_types": kinds + ["uniform"],
            "n_observations": 1260, "dt_years": 1 / 252, "history_length_range": [1, 1260],
            "prefix_lengths": [21, 63, 126, 252, 504, 756, 1260],
            "fixed": {"initial_variance": .04} if model == "heston" else {"variance_level": .04} if model == "rough-known" else {},
            "numerics": {"substeps": 8} if model == "heston" else {"observation_substeps": 4}}


def fixture_assets(root):
    folder = root / "trajectories"
    folder.mkdir()
    catalogue = {"format": "trajectory_tutorial_artifacts_v1", "scientific_acceptance": False, "artifacts": {}}
    rng = np.random.default_rng(83)
    for model in trajectory.MODEL_LABELS:
        cache = folder / model
        cache.mkdir()
        task = fixture_task(model)
        task_hash = trajectory.fingerprint(task)
        training = {"format": trajectory.FORMAT, "task_sha256": task_hash, "model_sha256": "a" * 64,
                    "bank_manifest_sha256": "b" * 64, "supported_history_lengths": [1, 1260],
                    "normalization": "fixed_prior_units_no_fitted_standardizer_v1",
                    "prefix_policy": "uniform_log_mixture",
                    "history_training": trajectory.PREFIX_POLICIES["uniform_log_mixture"]}
        evaluation = {"task_sha256": task_hash, "checkpoint_sha256": "a" * 64,
                      "bank_manifest_sha256": "b" * 64}
        bounds = np.asarray(task["prior_bounds"])
        cases, draws = 4, 8
        ids = np.asarray([f"{task_hash[:16]}:83:validation:{i}" for i in range(cases)])
        arrays = {"observations": rng.normal(size=(cases, 1260, 1)) * .01,
                  "truth": rng.uniform(bounds[:, 0], bounds[:, 1], (cases, len(bounds))),
                  "trajectory_id": ids}
        lengths = [1, 21, 63, 126, 252, 504, 756, 1260]
        arrays.update({f"draws_{n}": rng.uniform(bounds[:, 0], bounds[:, 1], (cases, draws, len(bounds))) for n in lengths})
        write_json(cache / "task.json", task)
        write_json(cache / "training.json", training)
        write_json(cache / "evaluation.json", evaluation)
        np.savez_compressed(cache / "posterior.npz", **arrays)
        bundle = {"format": "trajectory_teaching_replay_v1", "role": "cached_development_trajectory_replay",
                  "scientific_acceptance": False, "reserved_test_evaluated": False, "source_split": "validation",
                  "source_offset": 0, "model": model, "task_sha256": task_hash, "checkpoint_sha256": "a" * 64,
                  "cached_lengths": lengths, "case_ids": ids.tolist(), "limitations": "Synthetic fixture only.",
                  "files": {name: trajectory.sha256(cache / name) for name in trajectory.TRAJECTORY_FILES}}
        write_json(cache / "bundle.json", bundle)
        catalogue["artifacts"][model] = {"cache": model, "bundle_sha256": trajectory.sha256(cache / "bundle.json"),
                                          "checkpoint_sha256": "a" * 64, "task_sha256": task_hash,
                                          "model_source": "fixture native checkpoint"}
    write_json(folder / "artifacts.json", catalogue)
    return folder / "artifacts.json"


class TrajectoryReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "assets"
        self.root.mkdir()
        self.catalogue = fixture_assets(self.root)
        self.patch_assets = mock.patch.object(trajectory, "ASSETS", self.root)
        self.patch_catalogue = mock.patch.object(trajectory, "CATALOGUE", self.catalogue)
        self.patch_assets.start()
        self.patch_catalogue.start()

    def tearDown(self):
        self.patch_catalogue.stop()
        self.patch_assets.stop()
        self.temporary.cleanup()

    def rebind_bundle(self, model):
        cache = self.catalogue.parent / model
        bundle = json.loads((cache / "bundle.json").read_text())
        bundle["files"] = {name: trajectory.sha256(cache / name) for name in trajectory.TRAJECTORY_FILES}
        write_json(cache / "bundle.json", bundle)
        catalogue = json.loads(self.catalogue.read_text())
        catalogue["artifacts"][model]["bundle_sha256"] = trajectory.sha256(cache / "bundle.json")
        write_json(self.catalogue, catalogue)

    def test_models_share_grid_and_cached_past_prefix_is_explicit(self):
        for model in trajectory.MODEL_LABELS:
            task, report, arrays = trajectory.load_example(model)
            cached = trajectory.cached_posterior(task, arrays, 0, 137)
            self.assertEqual((cached["requested_length"], cached["length"], cached["exact"]), (137, 126, False))
            self.assertIn("nearest past", cached["label"])
            np.testing.assert_array_equal(cached["draws"], arrays["draws_126"][0])
            self.assertTrue(trajectory.cached_posterior(task, arrays, 0, 126)["exact"])
            self.assertEqual(len(trajectory.posterior_rows(task, cached["draws"])), len(task["parameter_names"]))
            self.assertFalse(report["reserved_test_evaluated"])
            self.assertNotIn("drift", task["fixed"])

    def test_saved_controls_have_an_exact_posterior_for_every_selectable_length(self):
        for model in trajectory.MODEL_LABELS:
            task, _, arrays = trajectory.load_example(model)
            lengths = trajectory.saved_lengths(arrays)
            self.assertEqual(lengths, [1, 21, 63, 126, 252, 504, 756, 1260])
            for length in lengths:
                view = trajectory.posterior_view(task, arrays, 0, length)
                self.assertEqual((view["source"], view["length"], view["exact"]), ("saved", length, True))

    def test_fresh_draws_cannot_follow_a_changed_model_history_or_prefix(self):
        task, _, arrays = trajectory.load_example("rough-known")
        draws = arrays["draws_252"][0].copy()
        live = {"selection": trajectory.selection_key(task, arrays, 0, 137), "draws": draws}
        fresh = trajectory.posterior_view(task, arrays, 0, 137, live=live)
        self.assertEqual((fresh["source"], fresh["length"], fresh["exact"]), ("live", 137, True))
        np.testing.assert_array_equal(fresh["draws"], draws)
        for case, length in ((1, 137), (0, 138), (1, 252)):
            view = trajectory.posterior_view(task, arrays, case, length, live=live)
            self.assertEqual(view["source"], "saved")
            expected = trajectory.cached_posterior(task, arrays, case, length)
            self.assertEqual(view["length"], expected["length"])
            np.testing.assert_array_equal(view["draws"], expected["draws"])
        other_task, _, other_arrays = trajectory.load_example("rough-joint")
        self.assertEqual(trajectory.posterior_view(other_task, other_arrays, 0, 137, live=live)["source"], "saved")
        changed_task = copy.deepcopy(task)
        changed_task["fixed"]["variance_level"] = .05
        self.assertEqual(trajectory.posterior_view(changed_task, arrays, 0, 137, live=live)["source"], "saved")

    def test_malformed_fresh_draws_cannot_be_displayed_as_a_completed_posterior(self):
        task, _, arrays = trajectory.load_example("rough-known")
        key = trajectory.selection_key(task, arrays, 0, 137)
        draws = arrays["draws_252"][0].copy()
        invalid = [draws[:1], draws[:, :-1], np.full_like(draws, np.nan), np.full_like(draws, 100)]
        for values in invalid:
            with self.subTest(shape=values.shape):
                with self.assertRaisesRegex(ValueError, "invalid fresh posterior"):
                    trajectory.posterior_view(task, arrays, 0, 137, live={"selection": key, "draws": values})

    def test_cached_loading_initializes_no_neural_backend(self):
        program = """
import importlib.abc, sys
class Offline(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'bayesflow','keras','jax','sbi4finance'}:
            raise AssertionError('cached trajectory imported backend: '+fullname)
sys.meta_path.insert(0,Offline())
from sbi4f_tutorial.trajectories import lesson as trajectory
for model in trajectory.MODEL_LABELS:
    trajectory.load_example(model)
"""
        run = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True,
                             env={**os.environ, "SBI4F_TUTORIAL_ASSETS": str(self.root)})
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_checksum_and_checkpoint_identity_changes_are_rejected(self):
        posterior = self.catalogue.parent / "heston" / "posterior.npz"
        posterior.write_bytes(posterior.read_bytes() + b"corruption")
        with self.assertRaisesRegex(ValueError, "checksum"):
            trajectory.load_example("heston")
        evaluation = self.catalogue.parent / "rough-joint" / "evaluation.json"
        value = json.loads(evaluation.read_text())
        value["checkpoint_sha256"] = "c" * 64
        write_json(evaluation, value)
        self.rebind_bundle("rough-joint")
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            trajectory.load_example("rough-joint")

    def test_paths_and_live_loading_are_restricted_to_asset_cache(self):
        value = json.loads(self.catalogue.read_text())
        value["artifacts"]["heston"]["cache"] = str(self.root.parent)
        write_json(self.catalogue, value)
        with self.assertRaisesRegex(ValueError, "tutorial/assets"):
            trajectory.load_example("heston")
        for folder in (self.root, self.root / "checkpoints", self.root.parent):
            with self.assertRaisesRegex(ValueError, "local-checkpoints|tutorial/assets"):
                trajectory.load_live("rough-known", folder)
        with self.assertRaisesRegex(FileNotFoundError, "Matching native checkpoint missing"):
            trajectory.load_live("rough-known")

    def test_symlinked_native_or_bundle_file_cannot_escape_asset_root(self):
        outside = self.root.parent / "outside.json"
        outside.write_text("{}")
        bundle = self.catalogue.parent / "rough-known" / "bundle.json"
        bundle.unlink()
        bundle.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "tutorial/assets"):
            trajectory.load_example("rough-known")
        local = self.root / "local-checkpoints" / "path-heston"
        local.mkdir(parents=True)
        (local / "model.keras").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "tutorial/assets"):
            trajectory.load_live("heston")

    def test_live_conditions_contain_observed_returns_and_design_only(self):
        task, _, arrays = trajectory.load_example("rough-joint")
        workflow = SimpleNamespace(sample=mock.Mock(return_value={"parameters": np.zeros((1, 8, len(task["parameter_names"]))) }))
        result = trajectory.sample(workflow, task, arrays["observations"][:1, :137], 8, seed=20261031001)
        self.assertEqual(result.shape, (1, 8, 7))
        conditions = workflow.sample.call_args.kwargs["conditions"]
        self.assertEqual(set(conditions), {"observations", "time_context", "summary_mask"})
        self.assertTrue(conditions["summary_mask"][0, :137].all())
        self.assertFalse(conditions["summary_mask"][0, 137:].any())
        np.testing.assert_array_equal(conditions["observations"][0, 137:], 0)

    def test_live_role_matches_the_complete_training_record_before_sampling(self):
        task, _, _ = trajectory.load_example("heston")
        record = json.loads((self.catalogue.parent / "heston" / "training.json").read_text())
        local = self.root / "local-checkpoints" / "path-heston"
        local.mkdir(parents=True)
        for filename in ("task.json", "training.json", "model.keras"):
            (local / filename).write_text("fixture")
        with mock.patch.object(trajectory, "load_checkpoint", return_value=(task, object(), record)):
            returned, _, _ = trajectory.load_live("heston")
            self.assertEqual(returned, task)
        changed = {**record, "bank_manifest_sha256": "d" * 64}
        with mock.patch.object(trajectory, "load_checkpoint", return_value=(task, object(), changed)):
            with self.assertRaisesRegex(ValueError, "native checkpoint"):
                trajectory.load_live("heston")

    def test_cache_scope_and_case_selection_are_explicit(self):
        model = "heston"
        cache = self.catalogue.parent / model
        bundle = json.loads((cache / "bundle.json").read_text())
        bundle["source_split"] = "test"
        write_json(cache / "bundle.json", bundle)
        self.rebind_bundle(model)
        with self.assertRaisesRegex(ValueError, "scope"):
            trajectory.load_example(model)


class ShippedTrajectoryTests(unittest.TestCase):
    def test_published_catalogue_validates_when_present(self):
        if not trajectory.CATALOGUE.exists():
            self.skipTest("trajectory assets have not been exported by the lab yet")
        for model in trajectory.catalogue()["artifacts"]:
            trajectory.load_example(model)


if __name__ == "__main__":
    unittest.main()
