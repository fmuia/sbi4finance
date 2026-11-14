"""The compact Heston lesson renders from its shipped data without inference code."""

import builtins
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch


LAB = Path(__file__).resolve().parents[1]
NOTEBOOK = LAB / "notebooks/heston_summary_experiments.py"
DATA = LAB / "notebooks/data/heston_summary_dimensions"
DIMENSIONS = {64, 32, 16, 8, 7, 6, 5, 4, 2}


def file_hashes(root):
    return {path.name: sha256(path.read_bytes()).hexdigest() for path in root.iterdir() if path.is_file()}


@unittest.skipUnless(importlib.util.find_spec("marimo"), "requires the optional notebook environment")
class HestonSummaryLessonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_hashes = file_hashes(DATA)
        spec = importlib.util.spec_from_file_location("compact_heston_summary_lesson", NOTEBOOK)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        original_import = builtins.__import__

        def without_runtime(name, *args, **kwargs):
            if name.split(".")[0] in {"bayesflow", "keras", "jax", "sbi4finance", "sbi4f_tutorial"}:
                raise AssertionError("the lesson must not import pricing or inference implementations")
            return original_import(name, *args, **kwargs)

        with (patch.dict(os.environ, {"HESTON_BUNDLE": "/unavailable/private-bank",
                                      "HESTON_EXPERIMENT_ROOT": "/unavailable/private-runs"}),
              patch("builtins.__import__", side_effect=without_runtime)):
            spec.loader.exec_module(module)
            cls.outputs, cls.defs = module.app.run()
        cls.load_results = staticmethod(cls.defs["load_results"])
        cls.record = cls.defs["lesson_data"]

    @classmethod
    def tearDownClass(cls):
        import matplotlib.pyplot as plt
        plt.close("all")

    def assert_rejected(self, mutate, message=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(DATA, root, dirs_exist_ok=True)
            record = deepcopy(self.record)
            mutate(record, root)
            (root / "results.json").write_text(json.dumps(record))
            with self.assertRaises(ValueError, msg=message):
                self.load_results(root)

    def test_all_endpoints_render_without_private_bank_or_neural_runtime(self):
        self.assertTrue(self.outputs)
        self.assertEqual(len(self.defs["all_runs"]), 19)
        self.assertEqual(len(self.defs["primary_runs"]), 10)
        self.assertEqual(len(self.defs["summary_runs"]), 9)
        self.assertEqual({r["dimension"] for r in self.defs["primary_runs"]}, DIMENSIONS | {None})
        self.assertEqual(set(self.defs["recovery_figures"]), DIMENSIONS | {None})
        self.assertEqual(file_hashes(DATA), self.original_hashes)

    def test_one_history_per_fit_and_primary_secondary_roles_stay_distinct(self):
        self.assertEqual(sum("history" in fit for fit in self.record["fits"]), 9)
        self.assertEqual(set(self.original_hashes), {"README.md", "results.json", "direct-final.png"} |
                         {f"mlp{d}-final.png" for d in DIMENSIONS})
        for dimension in DIMENSIONS:
            pair = [r for r in self.defs["all_runs"] if r["dimension"] == dimension]
            self.assertEqual({r["endpoint"] for r in pair}, {"primary_final", "secondary_validation_minimum"})
            self.assertIs(pair[0]["history"], pair[1]["history"])
            self.assertEqual(pair[0]["history"]["epoch"], list(range(1, 301)))
            for row in pair:
                expected = 300 if row["endpoint"] == "primary_final" else row["run"]["best_epoch"]
                self.assertEqual(row["evaluation"]["epoch"], expected)
        self.assertEqual(len(self.record["producer"]["archive_commit"]), 40)

    def test_controls_and_c6_to5_result_match_the_saved_metrics(self):
        self.assertEqual(self.defs["target_selector"].value, "c")
        self.assertEqual(self.defs["checkpoint_dimension"].value, 8)
        self.assertEqual(self.defs["recovery_model"].value, "mlp8")
        self.assertEqual(set(self.defs["dimension_selector"].value), DIMENSIONS)
        runs = {r["dimension"]: r for r in self.defs["primary_runs"]}
        self.assertEqual(self.defs["training_backend"](runs[6]), "JAX GPU")
        self.assertEqual(self.defs["training_backend"](runs[5]), "JAX GPU")
        self.assertEqual(self.defs["training_backend"](runs[7]), "JAX CPU")
        self.assertIn("metadata not stored", self.defs["training_backend"](runs[None]))
        six, five = (runs[d]["evaluation"]["parameters"] for d in (6, 5))
        self.assertAlmostEqual(six["c"]["rmse"], .13396648214853743)
        self.assertAlmostEqual(five["c"]["rmse"], .4395404603686137)
        self.assertLess(six["c"]["median_width90"] / 1.35, .5)
        self.assertGreater(five["c"]["median_width90"] / 1.35, .99)
        direct = runs[None]["evaluation"]["parameters"]
        ratios = [five[n]["rmse"] / direct[n]["rmse"] for n in self.defs["names"] if n != "c"]
        self.assertEqual((round(min(ratios), 2), round(max(ratios), 2)), (.80, 1.11))

    def test_missing_duplicate_or_relabelled_endpoints_are_rejected(self):
        def missing(record, _):
            record["fits"][1]["endpoints"].pop("secondary_validation_minimum")
        def duplicate(record, _):
            record["fits"].append(deepcopy(record["fits"][1]))
        def relabelled(record, _):
            record["fits"][1]["endpoints"]["primary_final"]["epoch"] = 258
        for mutation in (missing, duplicate, relabelled):
            with self.subTest(mutation=mutation.__name__):
                self.assert_rejected(mutation)

    def test_mismatched_shared_contract_history_or_checkpoint_is_rejected(self):
        def identities(record, _):
            record["identities"]["bank_sha256"] = "0" * 64
        def budget(record, _):
            record["protocol"]["training"]["rows"] = 299999
        def history(record, _):
            record["fits"][1]["history"]["loss"][0] += 1
        def checkpoint(record, _):
            record["fits"][1]["training"]["best_epoch"] = 300
        for mutation in (identities, budget, history, checkpoint):
            with self.subTest(mutation=mutation.__name__):
                self.assert_rejected(mutation)

    def test_invalid_scientific_metrics_cannot_be_rendered(self):
        def nonfinite(record, _):
            record["fits"][1]["endpoints"]["primary_final"]["population"]["rmse"][0] = float("nan")
        def coverage(record, _):
            record["fits"][1]["endpoints"]["primary_final"]["population"]["coverage90"][0] = 1.1
        def ranks(record, _):
            flags = record["fits"][1]["endpoints"]["primary_final"]["population"]["rank_passed"]
            flags[0] = not flags[0]
        for mutation in (nonfinite, coverage, ranks):
            with self.subTest(mutation=mutation.__name__):
                self.assert_rejected(mutation)

    def test_changed_or_escaping_recovery_images_are_rejected(self):
        def changed(record, root):
            path = root / record["fits"][1]["recovery_figure"]["file"]
            path.write_bytes(path.read_bytes() + b"changed")
        def escape(record, _):
            record["fits"][1]["recovery_figure"]["file"] = "../outside.png"
        def draw_identity(record, _):
            record["fits"][1]["recovery_figure"]["source_draws_sha256"] = "0" * 64
        for mutation in (changed, escape, draw_identity):
            with self.subTest(mutation=mutation.__name__):
                self.assert_rejected(mutation)


if __name__ == "__main__":
    unittest.main()
