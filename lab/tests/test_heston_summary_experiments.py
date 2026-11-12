"""The shipped summary notebook must work from compact evidence alone."""

import builtins
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


LAB = Path(__file__).resolve().parents[1]
NOTEBOOK = LAB / "notebooks/heston_summary_experiments.py"
EVIDENCE = LAB / "evidence/pilot-results/2026-11-12/heston-flexible-inputs"
DIMENSIONS = {2, 4, 8, 16, 32, 64}


def json_hashes(root):
    return {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*.json")}


@unittest.skipUnless(importlib.util.find_spec("marimo"), "the notebook uses the optional core marimo environment")
class SummaryNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_hashes = json_hashes(EVIDENCE)
        spec = importlib.util.spec_from_file_location("heston_summary_experiments_under_test", NOTEBOOK)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        original_import = builtins.__import__

        def compact_only_import(name, *args, **kwargs):
            if name.split(".")[0] in {"bayesflow", "keras", "jax", "sbi4finance", "sbi4f_tutorial"}:
                raise AssertionError("the compact-evidence notebook must not import inference or pricing runtimes")
            return original_import(name, *args, **kwargs)

        with (patch.dict(os.environ, {"HESTON_BUNDLE": "/unavailable/private-bank",
                                     "HESTON_EXPERIMENT_ROOT": "/unavailable/private-runs"}),
              patch("builtins.__import__", side_effect=compact_only_import)):
            spec.loader.exec_module(module)
            cls.outputs, cls.definitions = module.app.run()
        cls.read_study = staticmethod(cls.definitions["read_study"])
        cls.select_endpoints = staticmethod(cls.definitions["select_endpoints"])
        cls.all_runs = cls.definitions["all_runs"]

    @classmethod
    def tearDownClass(cls):
        import matplotlib.pyplot as plt
        plt.close("all")

    def copied_evidence(self, root):
        for source in EVIDENCE.rglob("*.json"):
            destination = root / source.relative_to(EVIDENCE)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
        return json.loads((root / "study.json").read_text())

    @staticmethod
    def save(path, record):
        path.write_text(json.dumps(record))

    def test_all_complete_endpoints_load_without_bank_or_inference_runtime(self):
        self.assertTrue(self.outputs)
        self.assertEqual(len(self.all_runs), 13)
        primary = self.select_endpoints(self.all_runs)
        secondary = self.select_endpoints(self.all_runs, endpoint="secondary_validation_minimum")
        self.assertEqual(len(primary), 7)
        self.assertEqual(len(secondary), 6)
        self.assertEqual({row["dimension"] for row in primary}, DIMENSIONS | {None})
        self.assertEqual({row["dimension"] for row in secondary}, DIMENSIONS)
        self.assertEqual(len({row["run"]["data_payload_sha256"] for row in self.all_runs}), 1)
        for key in ("truth_sha256", "observations_sha256"):
            self.assertEqual(len({row["evaluation"]["population"][key] for row in self.all_runs}), 1)
        for row in self.all_runs:
            self.assertFalse(row["run"]["smoke_test_only"])
            self.assertEqual(row["run"]["status"], "complete")
            self.assertFalse(row["evaluation"]["scientific_acceptance"])

    def test_ui_defaults_preserve_primary_conclusions_and_offer_secondary_discussion(self):
        self.assertEqual(self.definitions["target_selector"].value, "c")
        self.assertEqual(set(self.definitions["dimension_selector"].value), DIMENSIONS)
        self.assertEqual(self.definitions["checkpoint_dimension"].value, 8)
        self.assertEqual(len(self.definitions["primary_runs"]), 7)
        self.assertEqual(len(self.definitions["summary_runs"]), 6)
        self.assertTrue(all(row["endpoint"] == "primary_final" for row in self.definitions["primary_runs"]))

    def test_c_width_ratios_distinguish_informativeness_from_calibration(self):
        primary = {row["dimension"]: row["evaluation"] for row in self.select_endpoints(self.all_runs)}
        prior_width = primary[8]["population"]["recovery"]["prior_width90"][5]
        self.assertAlmostEqual(prior_width, .9 * (2. - .5))
        ratio = {dim: primary[dim]["parameters"]["c"]["median_width90"] / prior_width for dim in (2, 4, 8)}
        self.assertLess(ratio[8], .5)
        for dimension in (2, 4):
            self.assertGreater(ratio[dimension], .95)
            self.assertLess(ratio[dimension], 1.05)
            self.assertLess(primary[8]["parameters"]["c"]["rmse"], primary[dimension]["parameters"]["c"]["rmse"])

    def test_endpoint_selection_never_relabels_data_and_evidence_stays_unchanged(self):
        for endpoint in ("primary_final", "secondary_validation_minimum"):
            selected = self.select_endpoints(self.all_runs, endpoint=endpoint, dimensions=[8])
            self.assertTrue(selected)
            self.assertTrue(all(row["dimension"] in (None, 8) for row in selected))
            self.assertTrue(all(row["endpoint"] == endpoint for row in selected))
            self.assertEqual(sum(row["dimension"] == 8 for row in selected), 1)
        self.assertEqual(json_hashes(EVIDENCE), self.source_hashes)

    def test_loader_rejects_directory_escape_and_unsupported_dimension(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study = self.copied_evidence(root)
            study["runs"][0]["directory"] = "../outside-export"
            self.save(root / "study.json", study)
            with self.assertRaises(ValueError):
                self.read_study(root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study = self.copied_evidence(root)
            entry = next(entry for entry in study["runs"] if entry.get("summary_dim") == 8)
            run_path = root / entry["directory"] / "run.json"
            run = json.loads(run_path.read_text())
            run["configuration"]["summary_dim"] = 3
            entry["summary_dim"] = 3
            self.save(run_path, run)
            self.save(root / "study.json", study)
            with self.assertRaises(ValueError):
                self.read_study(root)

    def test_loader_rejects_mismatched_bank_or_population_pairing(self):
        for field in ("data_payload_sha256", "truth_sha256"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                study = self.copied_evidence(root)
                entry = next(entry for entry in study["runs"] if entry.get("summary_dim") == 8)
                path = root / entry["directory"] / ("run.json" if field == "data_payload_sha256" else "evaluation.json")
                record = json.loads(path.read_text())
                if field == "data_payload_sha256":
                    record[field] = "0" * 64
                else:
                    record["population"][field] = "0" * 64
                self.save(path, record)
                with self.assertRaises(ValueError):
                    self.read_study(root)


if __name__ == "__main__":
    unittest.main()
