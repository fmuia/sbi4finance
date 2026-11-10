"""Held-out inputs retain their task/precision identity and never become labels in conditions."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from sbi4f_tutorial.rough_options import diagnostics
from sbi4f_tutorial.shared import full_surface, full_surface_teaching, paths, tasks


class RoughDiagnosticPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = json.loads((paths.ASSETS / "backbones" / "rough-known100k-validation0" / "task.json").read_text())

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.assets = self.root / "assets"
        self.assets.mkdir()
        self.patch = mock.patch.object(paths, "ASSETS", self.assets)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.directory = self.assets / "diagnostics" / "rough-known"
        self.directory.mkdir(parents=True)
        n = 3
        bounds = np.asarray(self.task["prior_bounds"])
        self.split = {
            "parameters": np.tile(bounds.mean(axis=1), (n, 1)),
            "observations": np.tile(np.linspace(.001, .03, 63), (n, 1)),
            "reference_prices": np.tile(np.linspace(.001, .03, 63), (n, 1)),
            "pricing_standard_errors": np.tile(.1 * tasks.sigma(self.task), (n, 1)),
            "context_values": np.full((n, 7), .04),
            "example_ids": np.array([17, 3, 29], dtype=np.int64),
            "pricing_paths": np.full((n, 3), self.task["numerics"]["paths_ladder"][0], dtype=np.int64),
        }
        self.split["observations"][0, 0] = -.0002
        self.metadata = {
            "format": "rough_diagnostic_panel_v1", "count": n,
            "task_fingerprint": full_surface_teaching.fingerprint(self.task),
            "parameter_names": self.task["parameter_names"],
            "scientific_acceptance": False, "reserved_test_evaluated": False,
            "generator": {"seed": 73, "settings": {
                "paths_ladder": self.task["numerics"]["paths_ladder"][:3],
                "max_mc_se_over_quote_sigma": .25,
                "max_mc_se_over_quote_sigma_last_rung": None,
            }},
        }
        self.write()

    def write(self, *, refresh_hashes=True):
        (self.directory / "task.json").write_text(json.dumps(self.task, indent=2) + "\n")
        np.savez_compressed(self.directory / "inputs.npz", **self.split)
        if refresh_hashes:
            self.metadata["files"] = {name: full_surface_teaching.digest(self.directory / name)
                                      for name in ("task.json", "inputs.npz")}
        (self.directory / "panel.json").write_text(json.dumps(self.metadata, indent=2) + "\n")

    def test_complete_task_inputs_load_without_checkpoint_or_clipping_or_reordering(self):
        self.split["observations"] = self.split["observations"][..., None]
        self.write()
        task, metadata, split = diagnostics.load_panel(self.directory, self.task)
        self.assertEqual(task, self.task)
        self.assertEqual(metadata["count"], 3)
        self.assertNotIn("checkpoint_sha256", metadata)
        self.assertEqual(split["observations"].shape, (3, 63))
        self.assertEqual(split["observations"][0, 0], -.0002)
        np.testing.assert_array_equal(split["example_ids"], [17, 3, 29])
        np.testing.assert_array_equal(split["parameters"], self.split["parameters"])

    def test_payload_and_task_tampering_and_changed_complete_expected_task_fail(self):
        self.split["observations"][0, 1] += .1
        self.write(refresh_hashes=False)
        with self.assertRaisesRegex(ValueError, "checksum"):
            diagnostics.load_panel(self.directory, self.task)
        self.write()
        for field in ("noise", "grid", "prior"):
            changed = copy.deepcopy(self.task)
            if field == "noise":
                changed["context"]["quote_noise_std"][0] *= 2
            elif field == "grid":
                changed["context"]["contracts"][0]["strike"] *= 1.001
            else:
                changed["prior_bounds"][0][0] += .001
            with self.subTest(field=field), self.assertRaises(ValueError):
                diagnostics.load_panel(self.directory, changed)

    def test_directory_and_every_file_cannot_escape_asset_and_panel_boundaries(self):
        outside = self.root / "outside"
        outside.mkdir()
        with self.assertRaisesRegex(ValueError, "tutorial/assets"):
            diagnostics.load_panel(outside, self.task)
        link = self.assets / "external-panel"
        link.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "tutorial/assets"):
            diagnostics.load_panel(link, self.task)
        for name in ("panel.json", "task.json", "inputs.npz"):
            with self.subTest(file=name):
                file = self.directory / name
                target = outside / name
                file.rename(target)
                file.symlink_to(target)
                with self.assertRaisesRegex(ValueError, "inside its panel"):
                    diagnostics.load_panel(self.directory, self.task)
                file.unlink()
                target.rename(file)

    def test_bad_shapes_nonfinite_labels_and_duplicate_ids_are_not_sliced_or_dropped(self):
        replacements = [
            ("observations", self.split["observations"][:2]),
            ("context_values", np.full((3, 7), np.nan)),
            ("context_values", np.zeros((3, 7))),
            ("parameters", np.zeros((3, 3))),
            ("example_ids", np.array([1, 1, 2])),
            ("example_ids", np.array([1., 2., 3.])),
            ("pricing_paths", np.zeros((3, 3), dtype=int)),
            ("pricing_paths", np.full((3, 3), 32768.)),
            ("pricing_standard_errors", np.full((3, 63), -1.)),
        ]
        original = copy.deepcopy(self.split)
        for name, value in replacements:
            self.split = copy.deepcopy(original)
            self.split[name] = value
            self.write()
            with self.subTest(field=name, shape=value.shape), self.assertRaises(ValueError):
                diagnostics.load_panel(self.directory, self.task)

    def test_relaxed_mc_precision_only_applies_to_that_blocks_actual_final_rung(self):
        settings = self.metadata["generator"]["settings"]
        settings["paths_ladder"] = self.task["numerics"]["paths_ladder"]
        settings["max_mc_se_over_quote_sigma_last_rung"] = .5
        # Only the first maturity block has a deliberately relaxed error budget.
        first = np.array([c["maturity"] <= self.task["numerics"]["blocks"][0]["horizon"] + 1e-12
                          for c in self.task["context"]["contracts"]])
        self.split["pricing_standard_errors"][0, first] = .4 * tasks.sigma(self.task)[first]
        self.split["pricing_paths"][0, 0] = settings["paths_ladder"][-1]
        self.write()
        diagnostics.load_panel(self.directory, self.task)
        self.split["pricing_paths"][0, 0] = settings["paths_ladder"][0]
        self.write()
        with self.assertRaisesRegex(ValueError, "precision budget"):
            diagnostics.load_panel(self.directory, self.task)
        settings["paths_ladder"] = settings["paths_ladder"][:3]
        self.write()
        with self.assertRaisesRegex(ValueError, "actual final rung"):
            diagnostics.load_panel(self.directory, self.task)

    def test_precision_policy_cannot_loosen_task_or_invent_paths(self):
        original = copy.deepcopy(self.metadata)
        for setting, value in (("max_mc_se_over_quote_sigma", .5),
                               ("paths_ladder", [12345]),
                               ("max_mc_se_over_quote_sigma_last_rung", .5)):
            self.metadata = copy.deepcopy(original)
            self.metadata["generator"]["settings"][setting] = value
            self.write()
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                diagnostics.load_panel(self.directory, self.task)

    def test_record_cannot_change_panel_count_parameter_order_or_scientific_scope(self):
        original = copy.deepcopy(self.metadata)
        for name, value in (("count", 4), ("count", True),
                            ("parameter_names", ["eta", "hurst", "rho"]),
                            ("scientific_acceptance", True), ("reserved_test_evaluated", True)):
            self.metadata = copy.deepcopy(original)
            self.metadata[name] = value
            self.write()
            with self.subTest(field=name, value=value), self.assertRaises(ValueError):
                diagnostics.load_panel(self.directory, self.task)

    def test_session_evaluation_rejects_masked_or_complex_inputs_without_calling_network(self):
        for name in ("parameters", "observations", "context_values"):
            for invalid in (np.ma.array(self.split[name], mask=np.ones_like(self.split[name])),
                            self.split[name].astype(complex)):
                changed = copy.deepcopy(self.split)
                changed[name] = invalid
                with self.subTest(field=name), mock.patch.object(full_surface, "sample") as sample:
                    with self.assertRaises(ValueError):
                        diagnostics.evaluate(object(), self.task, changed)
                    sample.assert_not_called()

    def test_evaluate_keeps_truth_out_of_conditions_and_returns_paired_physical_rows(self):
        workflow = object()
        returned = np.broadcast_to(self.split["parameters"][:, None, :], (3, 5, 3)).copy()
        with mock.patch.object(full_surface, "sample", return_value=returned) as sample:
            result = diagnostics.evaluate(workflow, self.task, self.split, draws=5, seed=19)
            np.testing.assert_array_equal(result, returned)
            args, kwargs = sample.call_args
            self.assertIs(args[0], workflow)
            np.testing.assert_array_equal(args[2], self.split["observations"])
            np.testing.assert_array_equal(kwargs["context_values"], self.split["context_values"])
            self.assertEqual(set(kwargs), {"num_samples", "context_values", "seed"})
            self.assertEqual(kwargs["seed"], 19)
            before = args[2].copy()
            changed = copy.deepcopy(self.split)
            changed["parameters"][:, 0] += .001
            diagnostics.evaluate(workflow, self.task, changed, draws=5, seed=19)
            np.testing.assert_array_equal(sample.call_args.args[2], before)

    def test_diagnostic_data_preserves_market_draw_and_parameter_pairing_in_physical_units(self):
        draws = np.broadcast_to(self.split["parameters"][:, None, :], (3, 5, 3)).copy()
        draws[:, :, 0] += np.arange(5) * .0001
        result = diagnostics.diagnostic_data(self.task, self.split, draws)
        for i, name in enumerate(self.task["parameter_names"]):
            np.testing.assert_array_equal(result["estimates"][name], draws[:, :, i:i + 1])
            np.testing.assert_array_equal(result["targets"][name], self.split["parameters"][:, i:i + 1])
        self.assertEqual(result["estimates"]["hurst"].shape, (3, 5, 1))
        with self.assertRaisesRegex(ValueError, "physical prior"):
            diagnostics.diagnostic_data(self.task, self.split, np.zeros((3, 5, 3)))
        with self.assertRaisesRegex(ValueError, "unmasked"):
            diagnostics.diagnostic_data(self.task, self.split, np.ma.array(draws, mask=np.ones_like(draws)))


if __name__ == "__main__":
    unittest.main()
