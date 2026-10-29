"""Rough replay identity, observed-curve boundary and offline tutorial dispatch."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from sbi4finance.calibration import full_surface_teaching as teaching

ROOT = Path(__file__).resolve().parents[1]


class RoughReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)

    def source(self, known):
        folder = self.root / ("known" if known else "joint"); folder.mkdir()
        name = "calibration_rough_known_backbone_20k.json" if known else "calibration_rough_joint_development_20k.json"
        task = json.loads((ROOT / "configs" / name).read_text())["task"]
        run = folder / "run"; (run / "posterior").mkdir(parents=True); (run / "validation").mkdir()
        (run / "validation-partial").mkdir()
        bounds = np.asarray(task["prior_bounds"]); truth = bounds.mean(axis=1)
        draws = truth + np.random.default_rng(4).uniform(-.1, .1, (12, len(bounds))) * np.diff(bounds).ravel()
        part = folder / "collection/parts/validation-0000000000-0000000001"; part.mkdir(parents=True)
        teaching.write(part / "manifest.json", {"scope": "synthetic integrity fixture"})
        np.savez_compressed(part / "arrays.npz", observations=np.arange(63)[None, :, None] * .0001,
                            parameters=truth[None], example_id=np.array(["validation:0"]),
                            context_values=np.full((1, 7), .04 if known else np.nan))
        snapshot = folder / "snapshot"; snapshot.mkdir()
        manifest = {"configuration": {"task": task}, "collection": {"path": "../collection"},
                    "parts": [{"split": "validation", "start": 0, "stop": 1,
                        "path": "../collection/parts/" + part.name,
                        "manifest_sha256": teaching.digest(part / "manifest.json"),
                        "array_sha256": teaching.digest(part / "arrays.npz")}]}
        teaching.write(snapshot / "manifest.json", manifest)
        training = {"format": "full_surface_training_v1", "task_fingerprint": teaching.fingerprint(task),
                    "checkpoint_sha256": "frozen-final-model", "selected_epoch": 500}
        teaching.write(run / "posterior/task.json", task); teaching.write(run / "posterior/training.json", training)
        evaluation = {"task_fingerprint": teaching.fingerprint(task), "checkpoint_sha256": "frozen-final-model",
                      "bank_manifest_sha256": teaching.digest(snapshot / "manifest.json"),
                      "evaluated_split": "validation", "reserved_test_evaluated": False, "example_ids": ["validation:0"]}
        teaching.write(run / "validation/evaluation.json", evaluation)
        np.savez_compressed(run / "validation/draws.npz", truth=truth[None])
        np.savez_compressed(run / "validation-partial/batch-0000.npz", draws=draws[None], first=0)
        teaching.write(run / "run.json", {"status": "completed", "frozen_inputs_verified": True,
            "reserved_test_evaluated": False, "scientific_acceptance": False,
            "posterior_sha256": "frozen-final-model", "bank": {"manifest_sha256": teaching.digest(snapshot / "manifest.json")}})
        out = folder / "cache"; teaching.prepare_rough_replay(run, snapshot, out)
        return task, run, snapshot, out

    def test_fixed_case_roundtrip_preserves_observation_draws_and_known_global_curve(self):
        for known in (True, False):
            with self.subTest(known=known):
                task, run, snapshot, out = self.source(known)
                actual, report, arrays = teaching.load_neural_replay(out, expected_task=task,
                    expected_checkpoint_sha256="frozen-final-model")
                self.assertEqual(actual, task); self.assertEqual(report["case_id"], "validation:0")
                self.assertEqual(arrays["neural_draws"].shape, (12, 3 if known else 6))
                self.assertEqual("context_values" in arrays, known)
                self.assertNotIn("reference_chains", arrays)
                self.assertFalse(report["scientific_acceptance"]); self.assertFalse(report["conditional_reference_available"])
                self.assertIn("possibly prior-truncated", report["limitations"])

    def test_rehashed_joint_curve_injection_and_mislabelled_epoch_are_rejected(self):
        _, _, _, out = self.source(False)
        with np.load(out / "posterior.npz", allow_pickle=False) as saved: arrays = dict(saved)
        arrays["context_values"] = np.ones(7)
        np.savez_compressed(out / "posterior.npz", **arrays)
        report = json.loads((out / "bundle.json").read_text())
        report["files"]["posterior.npz"] = teaching.digest(out / "posterior.npz")
        teaching.write(out / "bundle.json", report)
        with self.assertRaisesRegex(ValueError, "replay arrays"): teaching.load_neural_replay(out)
        report["selected_epoch"] = 402; teaching.write(out / "bundle.json", report)
        with self.assertRaisesRegex(ValueError, "identity differs"): teaching.load_neural_replay(out)

    def test_catalogue_pins_cache_and_rejects_changed_complete_task(self):
        task, _, _, out = self.source(True)
        catalogue = self.root / "artifacts.json"
        teaching.write(catalogue, {"format": "full_surface_tutorial_artifacts_v1", "scientific_acceptance": False,
            "artifacts": {"known": {"cache": str(out.relative_to(self.root)), "bundle_sha256": teaching.digest(out / "bundle.json"),
                "cache_format": "full_surface_neural_replay_v1", "checkpoint_sha256": "frozen-final-model",
                "task_fingerprint": teaching.fingerprint(task), "model_source": "oscar:/frozen/model"}}})
        _, report, _ = teaching.load_tutorial_artifact(catalogue, "known")
        self.assertEqual(report["artifact_role"], "cached")
        wrong = copy.deepcopy(task); wrong["context"]["contracts"][0]["strike"] *= 1.01
        with self.assertRaisesRegex(ValueError, "expected complete task"):
            teaching.load_neural_replay(out, expected_task=wrong)
        with (out / "posterior.npz").open("ab") as f: f.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"): teaching.load_tutorial_artifact(catalogue, "known")

    def test_builder_rejects_mismatched_truth_without_reading_test_or_running_inference(self):
        _, run, snapshot, _ = self.source(True)
        np.savez_compressed(run / "validation/draws.npz", truth=np.ones((1, 3)))
        with self.assertRaisesRegex(ValueError, "evaluation truth differs"):
            teaching.prepare_rough_replay(run, snapshot, self.root / "invalid")

    def additional_case(self, known, seed=0):
        task, run, snapshot, _ = self.source(known)
        bounds = np.asarray(task["prior_bounds"]); span = np.diff(bounds).ravel()
        truth = bounds.mean(axis=1) + np.array([.03, .07])[:, None] * span
        draws = truth[:, None, :] + np.random.default_rng(7).uniform(-.01, .01, (2, 12, len(bounds))) * span
        prices = np.arange(63)[None, :, None] * .0002 + np.array([.01, .02])[:, None, None]
        part = snapshot.parent / "collection/parts/validation-0000000025-0000000027"; part.mkdir()
        teaching.write(part / "manifest.json", {"scope": "synthetic integrity fixture"})
        np.savez_compressed(part / "arrays.npz", observations=prices, parameters=truth,
                            example_id=np.array(["validation:25", "validation:26"]),
                            context_values=np.full((2, 7), .05 if known else np.nan))
        manifest = json.loads((snapshot / "manifest.json").read_text())
        manifest["parts"].append({"split": "validation", "start": 25, "stop": 27,
            "path": "../collection/parts/" + part.name,
            "manifest_sha256": teaching.digest(part / "manifest.json"), "array_sha256": teaching.digest(part / "arrays.npz")})
        teaching.write(snapshot / "manifest.json", manifest)
        evaluation = json.loads((run / "validation/evaluation.json").read_text())
        evaluation.update(example_ids=[f"validation:{i}" for i in range(27)], seed=seed,
                          bank_manifest_sha256=teaching.digest(snapshot / "manifest.json"))
        teaching.write(run / "validation/evaluation.json", evaluation)
        record = json.loads((run / "run.json").read_text()); record["bank"]["manifest_sha256"] = teaching.digest(snapshot / "manifest.json")
        teaching.write(run / "run.json", record)
        all_truth = np.tile(bounds.mean(axis=1), (27, 1)); all_truth[25:27] = truth
        np.savez_compressed(run / "validation/draws.npz", truth=all_truth)
        batch = run / f"validation-partial/batch-{seed + 25:04d}.npz"
        np.savez_compressed(batch, draws=draws, first=seed + 25)
        return task, run, snapshot, batch, truth[1], prices[1, :, 0], draws[1]

    def test_explicit_case_selects_part_offset_and_sampling_seed_offset(self):
        for known in (True, False):
            with self.subTest(known=known):
                task, run, snapshot, _, truth, prices, draws = self.additional_case(known, seed=7)
                out = run.parent / "case26"
                report = teaching.prepare_rough_replay(run, snapshot, out, case_index=26)
                _, loaded, arrays = teaching.load_neural_replay(out, expected_task=task)
                self.assertEqual(loaded["case_id"], "validation:26")
                self.assertEqual(report["saved_batch_seed"], 32); self.assertEqual(report["saved_batch_row"], 1)
                self.assertTrue(report["combined_truth_crosscheck"])
                np.testing.assert_array_equal(arrays["observations"], prices)
                np.testing.assert_array_equal(arrays["truth"], truth)
                np.testing.assert_array_equal(arrays["neural_draws"], draws)
                self.assertEqual("context_values" in arrays, known)
                if known: np.testing.assert_array_equal(arrays["context_values"], np.full(7, .05))
                self.assertFalse(report["conditional_reference_available"])

    def test_explicit_case_rejects_absent_case_or_wrong_saved_batch(self):
        _, run, snapshot, batch, *_ = self.additional_case(False)
        with self.assertRaisesRegex(ValueError, "absent or ambiguous"):
            teaching.prepare_rough_replay(run, snapshot, self.root / "absent", case_index=27)
        with np.load(batch, allow_pickle=False) as saved: draws = saved["draws"]
        np.savez_compressed(batch, draws=draws, first=24)
        with self.assertRaisesRegex(ValueError, "different validation row"):
            teaching.prepare_rough_replay(run, snapshot, self.root / "wrong-batch", case_index=26)
        batch.unlink()
        with self.assertRaisesRegex(ValueError, "does not contain requested case"):
            teaching.prepare_rough_replay(run, snapshot, self.root / "missing-batch", case_index=26)

    def test_optional_combined_truth_and_case_label_binding(self):
        _, run, snapshot, _, *_ = self.additional_case(False)
        (run / "validation/draws.npz").unlink()
        out = self.root / "case26-without-combined"
        report = teaching.prepare_rough_replay(run, snapshot, out, case_index=26)
        self.assertFalse(report["combined_truth_crosscheck"])
        report["evaluation_row"] = 0; teaching.write(out / "bundle.json", report)
        with self.assertRaisesRegex(ValueError, "case identity differs"):
            teaching.load_neural_replay(out)

    def test_repository_catalogue_roundtrip_imports_no_backend_and_keeps_task_boundaries(self):
        probe = '''
import importlib.abc, sys
class Offline(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'bayesflow','keras','jax','scipy'}:
            raise AssertionError('Unexpected offline import: ' + name)
sys.meta_path.insert(0, Offline())
from sbi4finance.calibration.full_surface_teaching import load_tutorial_artifact
for name, dimension, reference, curve in [('heston-final500',4,True,False),('rough-known50k',3,False,True),('rough-joint20k',6,False,False)]:
    task, report, arrays = load_tutorial_artifact(sys.argv[1], name)
    assert arrays['neural_draws'].shape[1] == dimension
    assert arrays['truth'].shape == (dimension,)
    assert ('reference_chains' in arrays) == reference
    assert ('context_values' in arrays) == curve
    assert report['artifact_role'] == 'cached' and report['scientific_acceptance'] is False
'''
        result = subprocess.run([sys.executable, "-c", probe,
            str(ROOT / "data/synthetic/full-surface-backbones/artifacts.json")], capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__": unittest.main()
