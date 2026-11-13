"""The shipped summary notebook must work from compact evidence alone."""

import builtins
from hashlib import sha256
import importlib.util
import json
import os
import shutil
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


LAB = Path(__file__).resolve().parents[1]
NOTEBOOK = LAB / "notebooks/heston_summary_experiments.py"
EVIDENCE = LAB / "evidence/pilot-results/2026-11-12/heston-flexible-inputs"
REQUIRED_DIMENSIONS = {2, 4, 8, 16, 32, 64}
SUPPORTED_DIMENSIONS = REQUIRED_DIMENSIONS | {5, 6, 7}


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
        cls.read_recovery_figures = staticmethod(cls.definitions["read_recovery_figures"])
        cls.all_runs = cls.definitions["all_runs"]
        cls.dimensions = {row["dimension"] for row in cls.all_runs if row["dimension"] is not None}

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
        self.assertTrue(REQUIRED_DIMENSIONS.issubset(self.dimensions))
        self.assertTrue(self.dimensions.issubset(SUPPORTED_DIMENSIONS))
        expected_secondary = {row["dimension"] for row in self.all_runs
                              if row["dimension"] is not None and row["endpoint"] == "primary_final"
                              and (row["dimension"] in REQUIRED_DIMENSIONS or row["run"]["best_epoch"] != 300)}
        self.assertEqual(len(self.all_runs), 1 + len(self.dimensions) + len(expected_secondary))
        primary = self.select_endpoints(self.all_runs)
        secondary = self.select_endpoints(self.all_runs, endpoint="secondary_validation_minimum")
        self.assertEqual(len(primary), 1 + len(self.dimensions))
        self.assertEqual(len(secondary), len(expected_secondary))
        self.assertEqual({row["dimension"] for row in primary}, self.dimensions | {None})
        self.assertEqual({row["dimension"] for row in secondary}, expected_secondary)
        self.assertEqual(len({row["run"]["data_payload_sha256"] for row in self.all_runs}), 1)
        for key in ("truth_sha256", "observations_sha256"):
            self.assertEqual(len({row["evaluation"]["population"][key] for row in self.all_runs}), 1)
        for row in self.all_runs:
            self.assertFalse(row["run"]["smoke_test_only"])
            self.assertEqual(row["run"]["status"], "complete")
            self.assertFalse(row["evaluation"]["scientific_acceptance"])

    def test_ui_defaults_preserve_primary_conclusions_and_offer_secondary_discussion(self):
        self.assertEqual(self.definitions["target_selector"].value, "c")
        self.assertEqual(set(self.definitions["dimension_selector"].value), self.dimensions)
        self.assertEqual(self.definitions["checkpoint_dimension"].value, 8)
        self.assertEqual(self.definitions["recovery_model"].value, "mlp8")
        self.assertEqual(set(self.definitions["recovery_figures"]), self.dimensions | {None})
        self.assertEqual(len(self.definitions["primary_runs"]), 1 + len(self.dimensions))
        self.assertEqual(len(self.definitions["summary_runs"]), len(self.dimensions))
        self.assertTrue(all(row["endpoint"] == "primary_final" for row in self.definitions["primary_runs"]))
        backend = self.definitions["training_backend"]
        primary = {row["dimension"]: row for row in self.definitions["primary_runs"]}
        self.assertEqual(backend(primary[8]), "JAX CPU")
        self.assertEqual(backend(primary[None]), "Historical CPU (device metadata not stored)")
        # Hardware labels follow metadata, even when the dimension label is unchanged.
        self.assertEqual(backend({"dimension": 8, "run": {"devices": ["cuda:0"], "platform": "Linux"}}), "JAX GPU")
        self.assertEqual(backend({"dimension": 6, "run": {"devices": ["cpu:0"]}}), "JAX CPU")
        self.assertEqual(backend({"dimension": 5, "run": {}}), "Not recorded")

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

    def test_recovery_images_are_bound_to_primary_evidence_and_reject_tampering(self):
        self.assertEqual(set(self.read_recovery_figures(EVIDENCE, self.all_runs)), self.dimensions | {None})
        for corruption in ("image", "endpoint", "source_evaluation_sha256", "figure"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                shutil.copytree(EVIDENCE / "recovery", root / "recovery")
                path = root / "recovery/manifest.json"
                record = json.loads(path.read_text())
                entry = next(item for item in record["figures"] if item["dimension"] == 8)
                if corruption == "image":
                    with (root / entry["figure"]).open("ab") as image:
                        image.write(b"changed")
                else:
                    entry[corruption] = {"endpoint": "secondary_validation_minimum",
                                         "source_evaluation_sha256": "0" * 64,
                                         "figure": "../outside.png"}[corruption]
                    self.save(path, record)
                with self.assertRaises(ValueError):
                    self.read_recovery_figures(root, self.all_runs)

    def test_matching_intermediate_pairs_extend_the_screen_and_incomplete_pairs_are_rejected(self):
        # A structural fixture exercises optional-size loading without claiming that a fit ran.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study = self.copied_evidence(root)
            study["runs"] = [entry for entry in study["runs"]
                             if entry["summary_dim"] in REQUIRED_DIMENSIONS | {None}]
            originals = [entry for entry in study["runs"] if entry["summary_dim"] == 8]
            for original in originals:
                entry = dict(original, summary_dim=7, label="Test fixture only: intermediate 7",
                             directory=original["directory"].replace("mlp8-", "mlp7-"))
                source, destination = root / original["directory"], root / entry["directory"]
                shutil.copytree(source, destination, dirs_exist_ok=True)
                path = destination / "run.json"
                run = json.loads(path.read_text())
                run["configuration"]["summary_dim"] = 7
                self.save(path, run)
                study["runs"].append(entry)
            self.save(root / "study.json", study)
            runs = self.read_study(root)
            self.assertEqual({row["dimension"] for row in runs}, REQUIRED_DIMENSIONS | {7, None})
            self.assertEqual(len(runs), 15)
            study["runs"] = [entry for entry in study["runs"]
                             if not (entry["summary_dim"] == 7 and entry["endpoint"] == "secondary_validation_minimum")]
            self.save(root / "study.json", study)
            with self.assertRaisesRegex(ValueError, "distinct validation minima"):
                self.read_study(root)
            # If the final checkpoint is itself the validation minimum, no duplicate
            # secondary is exported or required for a newly added dimension.
            entry = next(entry for entry in study["runs"] if entry["summary_dim"] == 7)
            path = root / entry["directory"] / "run.json"
            run = json.loads(path.read_text())
            run["best_epoch"] = 300
            self.save(path, run)
            self.assertEqual(len(self.read_study(root)), 14)

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
