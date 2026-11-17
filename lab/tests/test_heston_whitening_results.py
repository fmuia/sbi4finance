"""Historical whitening evidence stays matched and distinct from the lesson fit."""

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


LAB = Path(__file__).resolve().parents[1]
DATA = LAB / "evidence/pilot-results/2026-11-14/heston-whitening-lesson"
SPEC = importlib.util.spec_from_file_location("whitening_lesson_under_test", LAB / "notebooks/_heston_whitening_results.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HestonWhiteningResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = MODULE.load_results(DATA)

    def rejected(self, mutate):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = deepcopy(self.record)
            mutate(record)
            (root / "results.json").write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                MODULE.load_results(root)

    def test_saved_plot_values_and_metrics_load_without_private_payloads_or_writes(self):
        before = sha256((DATA / "results.json").read_bytes()).hexdigest()
        record = MODULE.load_results(DATA)
        self.assertEqual(len(record["population"]["c_truth"]), 512)
        self.assertEqual(record["training"]["rows"], 10000)
        self.assertFalse(record["scientific_acceptance"])
        self.assertEqual(sha256((DATA / "results.json").read_bytes()).hexdigest(), before)
        diagonal, zca = [record["arms"][mode]["metrics"] for mode in MODULE.MODES]
        self.assertAlmostEqual(diagonal["truth_median_correlation"], -.00116023, places=7)
        self.assertAlmostEqual(zca["truth_median_correlation"], .92327854, places=7)
        self.assertGreater(diagonal["median_width90"], 3 * zca["median_width90"])
        self.assertLess(zca["coverage90"], .75)

    def test_the_pilot_cannot_be_relabelled_as_the_current_task_or_a_sigma_model(self):
        for mutate in (
            lambda r: r["study"].update(quote_count=63),
            lambda r: r["study"]["architecture"].update(family="FlowMatching"),
            lambda r: r["study"]["observation_model"].update(sd="sigma"),
            lambda r: r["study"].update(distinct_from_current_63_quote_flow_matching=False),
            lambda r: r["study"]["prior_bounds"][-1].__setitem__(1, 50.),
            lambda r: r.update(scientific_acceptance=True),
        ):
            self.rejected(mutate)

    def test_training_budget_pairing_and_training_only_preprocessing_are_required(self):
        for mutate in (
            lambda r: r["training"].update(optimizer_updates=50000),
            lambda r: r["training"].update(noise="fresh noise every pass"),
            lambda r: r["pairing"].update(same_initial_state=False),
            lambda r: r["pairing"].update(same_evaluation_datasets_and_order=False),
            lambda r: r["arms"]["zca"]["preprocessing"].update(fit_split="validation"),
            lambda r: r["arms"]["diagonal"]["preprocessing"].update(mode="raw_without_any_scaling"),
        ):
            self.rejected(mutate)

    def test_invalid_quantiles_population_changes_and_metrics_drift_are_rejected(self):
        for mutate in (
            lambda r: r["population"]["c_truth"].reverse(),
            lambda r: r["arms"]["zca"]["c_q05_q50_q95"][1].reverse(),
            lambda r: r["arms"]["zca"]["c_q05_q50_q95"][0].__setitem__(0, 4.),
            lambda r: r["arms"]["zca"]["c_q05_q50_q95"][1].__setitem__(0, np.nan),
            lambda r: r["arms"]["diagonal"]["metrics"].update(median_rmse=.1),
        ):
            self.rejected(mutate)

    def test_original_calibration_failures_and_final_checkpoint_rule_remain_visible(self):
        for mutate in (
            lambda r: r["arms"]["zca"].update(sbc_passed=[True] * 6),
            lambda r: r["arms"]["zca"].update(physical_coverage90=[.9] * 5),
            lambda r: r["arms"]["zca"]["checkpoint"].update(epoch=83),
            lambda r: r["arms"]["zca"]["checkpoint"].update(validation_min_epoch=50),
            lambda r: r["arms"]["zca"]["checkpoint"].update(best_validation_weights_saved=True),
        ):
            self.rejected(mutate)

    def test_source_bindings_private_payloads_and_duplicate_keys_are_checked(self):
        for mutate in (
            lambda r: r["source_bindings"].update(protocol="g" * 64),
            lambda r: r.update(draws=[[1, 2]]),
            lambda r: r.update(source_path="/Users/example/private"),
        ):
            self.rejected(mutate)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = json.dumps(self.record)
            (root / "results.json").write_text('{"format":"duplicate",' + payload[1:])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                MODULE.load_results(root)


if __name__ == "__main__":
    unittest.main()
