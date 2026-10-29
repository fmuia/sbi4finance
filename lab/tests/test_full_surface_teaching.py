"""Integrity and task boundaries of the offline full-surface replay bundle."""

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


class FullSurfaceTeachingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.task = json.loads((ROOT / "configs/calibration_heston_backbone_100k.json").read_text())["task"]
        source = self.root / "source"; source.mkdir()
        (source / "model.keras").write_bytes(b"original minimum-loss checkpoint")
        (source / "last.keras").write_bytes(b"saved final500 checkpoint")
        teaching.write(source / "task.json", self.task)
        self.original = {"format": "full_surface_training_v1", "task_fingerprint": teaching.fingerprint(self.task),
                         "checkpoint_sha256": teaching.digest(source / "model.keras"), "selected_epoch": 402,
                         "selection": "minimum validation loss within this family", "scientific_validation": False,
                         "same_seed_reload_exact": True, "network": {"family": "flow_matching"}}
        teaching.write(source / "training.json", self.original)
        teaching.write(source / "resume.json", {"epochs_done": 500, "checkpoint_files": {"last": "last.keras"}})
        self.model_sha = teaching.digest(source / "last.keras")
        teaching.write(self.root / "gate.json", {"checkpoint_sha256": self.model_sha, "rank_screen_passed": True,
                       "reserved_test_evaluated": True, "scientific_acceptance": False})
        teaching.write(self.root / "products.json", {"candidate_sha256": self.model_sha,
                       "scientific_acceptance": False, "cases": [{"case": 0, "products": {}}]})
        self.checkpoint = self.root / "export"
        teaching.export_checkpoint(source, self.checkpoint, gate=self.root / "gate.json",
                                   product_checks=self.root / "products.json")
        reference = self.root / "reference"; reference.mkdir()
        observations = np.linspace(.001, .02, 63)
        teaching.write(reference / "reference.json", {"task": self.task, "surface_id": "validation:0",
                       "split": "validation", "observed_prices": observations.tolist()})
        teaching.write(reference / "diagnostics.json", {"mcmc_diagnostics_passed": True})
        bounds = np.asarray(self.task["prior_bounds"])
        rng = np.random.default_rng(1)
        def rows(shape):
            return bounds[:, 0] + (.2 + .6 * rng.random(shape)) * np.diff(bounds, axis=1).ravel()
        np.savez_compressed(reference / "chains.npz", parameters=rows((2, 12, 4)))
        neural = self.root / "neural.npz"
        np.savez_compressed(neural, final=rows((20, 4)))
        sources = [reference / n for n in ("reference.json", "diagnostics.json", "chains.npz")] + [neural]
        self.declaration = self.root / "declaration.json"
        teaching.write(self.declaration, {"candidate_sha256": self.model_sha,
                       "development_cases": [{"case": 0, "reference": str(reference), "neural_draws": str(neural)}],
                       "frozen_sha256": {str(p): teaching.digest(p) for p in sources}})
        self.cache = self.root / "cache"
        teaching.prepare_cache(self.checkpoint, self.declaration, 0, self.cache)

    def test_export_preserves_original_and_does_not_claim_original_selection_or_reload(self):
        record = json.loads((self.checkpoint / "training.json").read_text())
        self.assertEqual(json.loads((self.checkpoint / "origin-training.json").read_text()), self.original)
        self.assertEqual(record["selected_epoch"], 500)
        self.assertEqual(record["original_validation_selected_epoch"], 402)
        self.assertEqual(record["checkpoint_sha256"], self.model_sha)
        self.assertFalse(record["same_seed_reload_exact"])
        self.assertFalse(record["scientific_validation"])
        self.assertNotIn("minimum validation loss", record["selection"])

    def test_cache_roundtrip_preserves_chain_axes_and_parameter_product_pairing(self):
        task, report, arrays = teaching.load_cache(self.cache, expected_task=self.task,
                                                 expected_checkpoint_sha256=self.model_sha)
        self.assertEqual(task, self.task)
        self.assertEqual(arrays["reference_chains"].shape, (2, 12, 4))
        self.assertFalse(report["scientific_acceptance"])
        self.assertEqual(arrays["observations"].shape, (63,))

    def test_changed_task_context_and_checkpoint_are_rejected(self):
        changed = copy.deepcopy(self.task)
        changed["context"]["contracts"][0]["strike"] += .01
        with self.assertRaisesRegex(ValueError, "expected complete task"):
            teaching.load_cache(self.cache, expected_task=changed)
        with self.assertRaisesRegex(ValueError, "expected checkpoint"):
            teaching.load_cache(self.cache, expected_checkpoint_sha256="wrong")

    def test_corrupted_cache_fails_without_repair_or_inference(self):
        with (self.cache / "posterior.npz").open("ab") as output: output.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            teaching.load_cache(self.cache)

    def test_rehashed_product_mispairing_is_still_rejected(self):
        with np.load(self.cache / "posterior.npz", allow_pickle=False) as source: arrays = dict(source)
        arrays["neural_variance_swap_1y"] = arrays["neural_variance_swap_1y"][::-1]
        np.savez_compressed(self.cache / "posterior.npz", **arrays)
        report = json.loads((self.cache / "bundle.json").read_text())
        report["files"]["posterior.npz"] = teaching.digest(self.cache / "posterior.npz")
        teaching.write(self.cache / "bundle.json", report)
        with self.assertRaisesRegex(ValueError, "pairing changed"):
            teaching.load_cache(self.cache)

    def test_cannot_promote_cache_to_acceptance(self):
        report = json.loads((self.cache / "bundle.json").read_text()); report["scientific_acceptance"] = True
        teaching.write(self.cache / "bundle.json", report)
        with self.assertRaisesRegex(ValueError, "scope changed"):
            teaching.load_cache(self.cache)

    def test_offline_load_imports_neither_backend_nor_scipy(self):
        probe = '''
import importlib.abc, sys
class BlockBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'bayesflow', 'keras', 'jax', 'scipy'}:
            raise AssertionError('Offline cache imported backend: ' + name)
sys.meta_path.insert(0, BlockBackend())
from sbi4finance.calibration.full_surface_teaching import load_cache
load_cache(sys.argv[1])
'''
        result = subprocess.run([sys.executable, "-c", probe, str(self.cache)], capture_output=True, text=True,
                                env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def completed_interior(self):
        case = self.root / "interior"; (case / "reference").mkdir(parents=True); (case / "neural").mkdir()
        record = json.loads((self.root / "reference/reference.json").read_text())
        record.update(surface_id="development:midpoint", split="development")
        teaching.write(case / "reference/reference.json", record)
        teaching.write(case / "reference/diagnostics.json", {"mcmc_diagnostics_passed": True})
        teaching.write(case / "neural/run.json", {"candidate_sha256": self.model_sha})
        teaching.write(case / "interior-check.json", {"interior_passed": True})
        teaching.write(case / "precision.json", {"all_points_priced": True})
        with np.load(self.root / "reference/chains.npz", allow_pickle=False) as arrays:
            np.savez_compressed(case / "reference/chains.npz", parameters=arrays["parameters"])
        with np.load(self.root / "neural.npz", allow_pickle=False) as arrays:
            np.savez_compressed(case / "neural/draws.npz", parameters=arrays["final"])
        np.savez_compressed(case / "observation.npz", observed_prices=record["observed_prices"])
        teaching.write(case / "manifest.json", {"candidate_sha256": self.model_sha})
        paths = [p for p in case.rglob("*") if p.is_file() and p.name != "manifest.json"]
        teaching.write(case / "run.json", {"format": "heston_interior_development_run_v1",
            "status": "completed_development_case", "scientific_acceptance": False,
            "frozen_inputs_verified": True, "manifest_sha256": teaching.digest(case / "manifest.json"),
            "stages": {"precision": {"all_points_priced": True}},
            "outputs": {str(p.relative_to(case)): teaching.digest(p) for p in paths}})
        return case

    def test_new_interior_case_keeps_own_identity_and_does_not_inherit_old_product_gates(self):
        case = self.completed_interior(); out = self.root / "new-interior-cache"
        teaching.prepare_development_cache(self.checkpoint, case, out)
        _, report, arrays = teaching.load_cache(out, expected_task=self.task)
        self.assertEqual(report["case_id"], "development:midpoint")
        self.assertEqual(arrays["reference_chains"].shape, (2, 12, 4))
        products = json.loads((out / "products.json").read_text())
        self.assertIn("descriptive", products["scope"])
        self.assertNotIn("case", products)
        self.assertFalse(products["scientific_acceptance"])

    def test_unchecked_interior_case_cannot_be_packaged(self):
        case = self.completed_interior(); run = json.loads((case / "run.json").read_text())
        run["stages"]["precision"]["all_points_priced"] = False
        teaching.write(case / "run.json", run)
        with self.assertRaisesRegex(ValueError, "precision checks must pass"):
            teaching.prepare_development_cache(self.checkpoint, case, self.root / "invalid-interior-cache")


if __name__ == "__main__":
    unittest.main()
