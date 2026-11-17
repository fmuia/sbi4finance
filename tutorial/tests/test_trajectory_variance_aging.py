"""The aging illustration must keep its declared event and original examples."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from sbi4f_tutorial.trajectories import simulation, variance_aging


class VarianceAgingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.assets = Path(self.temporary.name)
        self.folder = self.assets / "diagnostics" / variance_aging.ASSET_FOLDER
        self.folder.mkdir(parents=True)
        source = variance_aging.ASSETS / "diagnostics" / variance_aging.ASSET_FOLDER
        for name in ("report.json", "variance-aging.png", "ACCEPTED.json"):
            shutil.copyfile(source / name, self.folder / name)
        self.report = json.loads((self.folder / "report.json").read_text())
        self.task = simulation.teaching_task("rough-known")
        self.patch = mock.patch.object(variance_aging, "ASSETS", self.assets)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temporary.cleanup()

    def load_rebound_fixture(self, task=None):
        # Rebinding the test pin lets the inner contracts reject individually
        # corrupted fields. Production keeps its immutable source-controlled pin.
        path = self.folder / "report.json"
        path.write_text(json.dumps(self.report))
        pin = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt_path = self.folder / "ACCEPTED.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["source_sha256"]["report.json"] = pin
        receipt_path.write_text(json.dumps(receipt))
        with mock.patch.object(variance_aging, "REPORT_SHA256", pin):
            return variance_aging.load(self.task if task is None else task)

    def test_published_counts_histogram_and_saved_markers_keep_all_histories(self):
        result = variance_aging.load(self.task)
        report = result["report"]
        self.assertEqual(result["figure_path"], (self.folder / "variance-aging.png").resolve())
        prior = report["prior_ensemble"]
        values = np.array(prior["final_year_mean_variance"])
        self.assertEqual(len(values), 512)
        self.assertTrue(np.all(values > 0))
        self.assertEqual(np.count_nonzero(values < .0004), 69)
        self.assertEqual(sum(prior["histogram_counts"]), 512)
        np.testing.assert_array_equal(np.histogram(np.log10(values), prior["log10_histogram_edges"])[0],
                                      prior["histogram_counts"])
        self.assertEqual([row["count"] for row in report["fixed_parameter_ensembles"]], [335, 109, 2, 449])
        self.assertTrue(all(row["final_year_mean_variance"] < .0004 for row in report["saved_cases"]))
        # Case four is an extreme original history, not discarded by the range.
        extreme = np.log10(report["saved_cases"][3]["final_year_mean_variance"])
        self.assertLess(extreme, -7)
        self.assertLessEqual(prior["log10_histogram_edges"][0], extreme)
        self.assertFalse(report["posterior_calibration_claim"])

    def test_report_pin_figure_hash_and_copy_receipt_are_separate_checks(self):
        with mock.patch.object(variance_aging, "REPORT_SHA256", "0"*64):
            with self.assertRaisesRegex(ValueError, "pinned identity"):
                variance_aging.load(self.task)
        figure = self.folder / "variance-aging.png"
        figure.write_bytes(figure.read_bytes()+b"changed")
        with self.assertRaisesRegex(ValueError, "figure.*file hash"):
            variance_aging.load(self.task)
        figure.write_bytes(figure.read_bytes()[:-len(b"changed")])
        receipt_path = self.folder / "ACCEPTED.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["changes"] = ["clipped to the displayed range"]
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "copy receipt"):
            variance_aging.load(self.task)

    def test_no_relabelling_of_task_checkpoint_scope_or_event(self):
        original = copy.deepcopy(self.report)
        mutations = [
            lambda r: r.update(posterior_calibration_claim=True),
            lambda r: r.update(prior_changed=True),
            lambda r: r.update(checkpoint_sha256="other-weights"),
            lambda r: r.update(source_replay_bundle_sha256="other-histories"),
            lambda r: r["event"].update(threshold_variance=.0025),
            lambda r: r["task"]["fixed"].update(variance_level=.09),
            lambda r: r["fixed_parameter_ensembles"][2].update(seed=123),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.report = copy.deepcopy(original)
                mutation(self.report)
                with self.assertRaises(ValueError):
                    self.load_rebound_fixture()
        for model in ("heston", "rough-joint"):
            with self.subTest(model=model):
                self.report = copy.deepcopy(original)
                with self.assertRaisesRegex(ValueError, "task differs"):
                    self.load_rebound_fixture(simulation.teaching_task(model))

    def test_counts_intervals_histogram_and_actual_truth_markers_cannot_disagree(self):
        original = copy.deepcopy(self.report)
        mutations = [
            lambda r: r["prior_ensemble"].update(count=70, fraction=70/512,
                                               wilson95=variance_aging._wilson(70, 512).tolist()),
            lambda r: r["fixed_parameter_ensembles"][0].update(fraction=.9),
            lambda r: r["fixed_parameter_ensembles"][1].update(wilson95=[0., 1.]),
            lambda r: r["prior_ensemble"]["histogram_counts"].__setitem__(0, 0),
            lambda r: r["saved_cases"][3].update(final_year_mean_variance=.0004),
            lambda r: r["saved_cases"][2]["parameters"].__setitem__(1, 2.),
            lambda r: r["saved_cases"][0].update(case_id="a-different-observation"),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.report = copy.deepcopy(original)
                mutation(self.report)
                with self.assertRaises(ValueError):
                    self.load_rebound_fixture()

    def test_symlinked_report_figure_or_receipt_cannot_escape_asset_root(self):
        with tempfile.TemporaryDirectory() as outside:
            for name in ("report.json", "variance-aging.png", "ACCEPTED.json"):
                with self.subTest(name=name):
                    local = self.folder / name
                    external = Path(outside) / name
                    local.replace(external)
                    local.symlink_to(external)
                    try:
                        with self.assertRaisesRegex(ValueError, "remain under"):
                            variance_aging.load(self.task)
                    finally:
                        local.unlink()
                        external.replace(local)


if __name__ == "__main__":
    unittest.main()
