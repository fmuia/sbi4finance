"""A cached posterior must stay bound to its exact observed panel and task."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from sbi4finance.datasets.hkde_teaching import load_teaching_example
from sbi4finance.datasets import load_artifact, write_artifact


BUNDLE = Path(__file__).resolve().parents[1] / "data/synthetic/hkde-walkthrough"


class TeachingExampleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "bundle"
        shutil.copytree(BUNDLE, self.path)

    def refresh_hashes(self):
        path = self.path / "bundle.json"
        report = json.loads(path.read_text())
        report["files"] = {name: hashlib.sha256((self.path / name).read_bytes()).hexdigest()
                           for name in report["files"]}
        path.write_text(json.dumps(report))

    def replace_panel(self, change):
        panel = load_artifact(self.path / "panel")
        change(panel)
        shutil.rmtree(self.path / "panel")
        write_artifact(self.path / "panel", panel.metadata, panel.rows)
        self.refresh_hashes()

    def test_published_failure_is_visible_and_target_is_not_an_input(self):
        panel, report, arrays = load_teaching_example(self.path)
        self.assertEqual(report["case_id"], "validation:257")
        self.assertFalse(report["scientific_validation_passed"])
        self.assertNotIn(report["target_contract"], panel.metadata["scientific_task"]["context"]["contracts"])
        error = np.max(np.abs(np.quantile(arrays["model_price_draws"], report["quantiles"])
                              - report["reference_price_quantiles"]))
        self.assertAlmostEqual(error, report["original_max_price_quantile_error_currency"])
        self.assertGreater(error, report["price_error_gate_currency"])

    def test_edited_observation_cannot_reuse_another_panels_cached_posterior(self):
        self.replace_panel(lambda panel: panel.rows[0].update(price=999.0))
        with self.assertRaisesRegex(ValueError, "different observed panel"):
            load_teaching_example(self.path)

    def test_changed_context_is_rejected_even_with_same_nine_prices(self):
        self.replace_panel(lambda panel: panel.metadata["scientific_task"]["context"].update(spot=101.0))
        with self.assertRaisesRegex(ValueError, "scientific_task differs"):
            load_teaching_example(self.path)

    def test_truth_can_be_removed_without_changing_inputs_or_cached_draws(self):
        _, _, before = load_teaching_example(self.path)
        self.replace_panel(lambda panel: panel.metadata.pop("truth"))
        panel, _, after = load_teaching_example(self.path)
        self.assertIsNone(panel.truth)
        for name in before:
            np.testing.assert_array_equal(before[name], after[name])

    def test_corrupted_cache_fails_instead_of_downloading_or_retraining(self):
        (self.path / "posterior.npz").write_bytes(b"incomplete")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_teaching_example(self.path)

    def test_invalid_parameter_draws_cannot_be_used_for_pricing(self):
        _, _, arrays = load_teaching_example(self.path)
        arrays["parameter_draws"][0, 0] = 20.0
        np.savez(self.path / "posterior.npz", **arrays)
        self.refresh_hashes()
        with self.assertRaisesRegex(ValueError, "outside the checkpoint prior"):
            load_teaching_example(self.path)


if __name__ == "__main__":
    unittest.main()
