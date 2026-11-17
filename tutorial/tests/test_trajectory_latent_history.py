"""Hidden-history illustrations must remain paired with the actual observed paths."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from sbi4f_tutorial.trajectories import latent_assets, lesson, simulation


class LatentCaptureTests(unittest.TestCase):
    def test_canonical_replay_capture_preserves_all_original_outputs_bitwise(self):
        for name in simulation.MODELS:
            with self.subTest(model=name):
                task = simulation.teaching_task(name, observations=37)
                pairs = simulation.generate(task, "validation", 7, 4, seed=73, backend="numpy")
                details = simulation.replay_details(task, "validation", 7, 4, seed=73)
                self.assertEqual(set(pairs), {"observations", "parameters", "trajectory_id"})
                for key in pairs:
                    np.testing.assert_array_equal(details[key], pairs[key])
                self.assertEqual(details["latent"]["variance"].shape, (4, 38))
                self.assertTrue(np.isfinite(details["latent"]["variance"]).all())
                self.assertTrue((details["latent"]["variance"] >= 0).all())

    def test_optional_simulator_details_keep_default_return_paths_unchanged(self):
        for name in simulation.MODELS:
            with self.subTest(model=name):
                task = simulation.teaching_task(name, observations=63)
                theta = np.mean(task["prior_bounds"], axis=1)[None]
                original = simulation.simulate_parameters(task, theta, seed=41)
                details = simulation.simulate_parameters(task, theta, seed=41, return_details=True)
                np.testing.assert_array_equal(original, details["observations"])
                self.assertEqual(original.shape, (1, 63, 1))
        with self.assertRaisesRegex(ValueError, "boolean"):
            simulation.simulate_parameters(task, theta, return_details=1)

    def test_rough_details_preserve_driver_curve_and_fixed_mean_compensator(self):
        task = simulation.teaching_task("rough-joint", observations=126)
        theta = np.mean(task["prior_bounds"], axis=1)
        eta_index = task["parameter_names"].index("eta")
        eta = theta[eta_index]
        epsilon = 1e-5
        paths = []
        for shift in (-epsilon, 0., epsilon):
            parameters = theta.copy()
            parameters[eta_index] += shift
            paths.append(simulation.simulate_parameters(task, parameters[None], seed=41, return_details=True)["latent"])
        low, middle, high = paths
        np.testing.assert_array_equal(low["driver"], high["driver"])
        np.testing.assert_array_equal(low["driver_variance"], high["driver_variance"])
        np.testing.assert_array_equal(low["mean_variance"], high["mean_variance"])
        expected = middle["mean_variance"]*np.exp(eta*middle["driver"]-.5*eta**2*middle["driver_variance"])
        np.testing.assert_array_equal(middle["variance"], expected)
        numerical = (np.log(high["variance"])-np.log(low["variance"]))/ (2*epsilon)
        np.testing.assert_allclose(numerical, middle["driver"]-eta*middle["driver_variance"], atol=8e-11)
        self.assertTrue(np.all(high["median_variance"][:, 1:] < low["median_variance"][:, 1:]))
        self.assertTrue(np.any(numerical[:, 1:] < 0))
        np.testing.assert_array_equal(middle["driver"][:, 0], [0.])
        np.testing.assert_array_equal(middle["driver_variance"][:, 0], [0.])
        np.testing.assert_array_equal(middle["variance"][:, 0], middle["mean_variance"][:, 0])

    def test_heston_capture_retains_signed_state_and_its_effective_variance(self):
        task = simulation.teaching_task("heston", observations=3)
        theta = np.mean(task["prior_bounds"], axis=1)[None]
        signed = np.array([[[.04], [-.01], [.02], [-.003]]])
        returns = np.array([[[-.01], [.02], [-.003]]])
        with mock.patch.object(simulation.heston, "simulate", return_value={
                "log_returns": returns, "variance_state": signed}):
            result = simulation.simulate_parameters(task, theta, seed=41, return_details=True)
        np.testing.assert_array_equal(result["observations"], returns)
        np.testing.assert_array_equal(result["latent"]["variance_state"], signed[:, :, 0])
        np.testing.assert_array_equal(result["latent"]["variance"], np.maximum(signed[:, :, 0], 0.))
        self.assertLess(result["latent"]["variance_state"][0, 1], 0.)


class PairedLatentAssetTests(unittest.TestCase):
    def test_all_twelve_saved_histories_have_paired_whole_horizon_latent_truth(self):
        for name in simulation.MODELS:
            task, _, arrays = lesson.load_example(name)
            for case in range(4):
                with self.subTest(model=name, case=case):
                    result = latent_assets.load_history(task, arrays, case)
                    np.testing.assert_array_equal(result["returns"], arrays["observations"][case, :, 0])
                    np.testing.assert_array_equal(result["parameters"], arrays["truth"][case])
                    np.testing.assert_array_equal(result["days"], np.arange(1261))
                    self.assertEqual(result["variance"].shape, (1261,))
                    self.assertEqual(result["time_years"][-1], 5.)
                    self.assertFalse(result["metadata"]["inference_input"])
                    self.assertFalse(result["metadata"]["state_estimate"])
                    self.assertLessEqual(result["metadata"]["replay_max_abs_return_error"], 2e-14)
                    if name == "heston":
                        self.assertIsNone(result["driver"])
                        np.testing.assert_array_equal(result["variance"], np.maximum(result["variance_state"], 0.))
                    else:
                        self.assertEqual(result["driver"][0], 0.)
                        self.assertEqual(result["variance"][0], result["mean_variance"][0])

    def setUp(self):
        self.task, _, self.arrays = lesson.load_example("rough-known")

    def test_different_observations_truth_case_order_or_task_never_receive_unpaired_latent_paths(self):
        for key in ("observations", "truth", "trajectory_id"):
            with self.subTest(key=key):
                altered = {k: v.copy() for k, v in self.arrays.items()}
                if key == "trajectory_id":
                    altered[key] = altered[key][::-1]
                else:
                    altered[key].flat[0] += .0001
                with self.assertRaises(ValueError):
                    latent_assets.load_history(self.task, altered, 0)
        altered_task = copy.deepcopy(self.task)
        altered_task["prior_bounds"][0][0] = .04
        with self.assertRaisesRegex(ValueError, "task"):
            latent_assets.load_history(altered_task, self.arrays, 0)

    def test_integrity_and_paired_driver_are_checked_before_any_result_is_returned(self):
        with tempfile.TemporaryDirectory() as temporary:
            assets = Path(temporary)
            target = assets / "trajectories" / latent_assets.ASSET_FOLDER
            source = latent_assets.ASSETS / "trajectories" / latent_assets.ASSET_FOLDER
            shutil.copytree(source, target)
            with mock.patch.object(latent_assets, "ASSETS", assets):
                with mock.patch.object(latent_assets, "BUNDLE_SHA256", "0"*64):
                    with self.assertRaisesRegex(ValueError, "pinned"):
                        latent_assets.load_history(self.task, self.arrays, 0)
                path = target / "rough-known.npz"
                with np.load(path, allow_pickle=False) as content:
                    changed = {key: content[key] for key in content.files}
                changed["variance"][0, 1] *= 2
                np.savez_compressed(path, **changed)
                # Refresh the test fixture's hashes to exercise the inner pairing
                # invariant rather than merely its outer integrity protection.
                bundle_path = target / "bundle.json"
                bundle = json.loads(bundle_path.read_text())
                bundle["files"]["rough-known.npz"] = hashlib.sha256(path.read_bytes()).hexdigest()
                bundle_path.write_text(json.dumps(bundle))
                pin = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
                with mock.patch.object(latent_assets, "BUNDLE_SHA256", pin):
                    with self.assertRaisesRegex(ValueError, "paired driver"):
                        latent_assets.load_history(self.task, self.arrays, 0)

    def test_latent_payload_symlink_cannot_escape_the_tutorial_asset_root(self):
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as outside:
            assets = Path(temporary)
            target = assets / "trajectories" / latent_assets.ASSET_FOLDER
            source = latent_assets.ASSETS / "trajectories" / latent_assets.ASSET_FOLDER
            shutil.copytree(source, target)
            path = target / "rough-known.npz"
            external = Path(outside) / path.name
            path.replace(external)
            path.symlink_to(external)
            with mock.patch.object(latent_assets, "ASSETS", assets):
                with self.assertRaisesRegex(ValueError, "remain under"):
                    latent_assets.load_history(self.task, self.arrays, 0)


if __name__ == "__main__":
    unittest.main()
