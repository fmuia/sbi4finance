"""Exercise the lesson's actual local-input cell without running inference."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import marimo as mo
import numpy as np
from numpy.testing import assert_array_equal

from sbi4finance.datasets import load_artifact, write_artifact
from sbi4finance.hkde_inference import conditioning_data


ROOT = Path(__file__).parents[1]
PANEL = Path("data/synthetic/hkde-walkthrough/panel")


class HKDETeachingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "hkde_teaching_tested", ROOT / "notebooks/10_hkde_corrected_workflow.py"
        )
        cls.lesson = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.lesson)
        cls.protocol = json.loads(
            (ROOT / "configs/hkde_correction_boundary_protocol.json").read_text()
        )
        cls.task = json.loads((ROOT / cls.protocol["task_config"]).read_text())["task"]
        with np.load(ROOT / "docs/results/hkde-correction/validation-checks.npz",
                     allow_pickle=False) as saved:
            cls.saved_checks = {"validation": {
                name: saved[name] for name in ("case_ids", "observations")
            }}
        cls.original = load_artifact(ROOT / PANEL)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.panel = self.root / PANEL
        self.checkpoint_task = self.root / self.protocol["checkpoint"] / "task.json"
        self.checkpoint_task.parent.mkdir(parents=True)
        self.checkpoint_task.write_text(json.dumps(self.task))
        self.write_panel(deepcopy(self.original))

    def write_panel(self, artifact):
        if self.panel.exists():
            shutil.rmtree(self.panel)
        write_artifact(self.panel, artifact.metadata, artifact.rows)

    def run_input(self):
        # Supply every reference: no ancestor cell, checkpoint or pricer runs.
        _, definitions = self.lesson.local_input.run(
            ROOT=self.root, json=json, mo=mo, np=np, task=self.task,
            protocol=self.protocol, saved_checks=self.saved_checks,
        )
        return definitions

    def test_local_panel_preserves_the_existing_checkpoint_conditions(self):
        result = self.run_input()
        index = result["local_case_index"]
        original = self.saved_checks["validation"]["observations"][index]
        self.assertEqual(self.saved_checks["validation"]["case_ids"][index],
                         "validation:257")
        self.assertEqual(result["local_observation"].shape, (9, 1))
        assert_array_equal(result["local_observation"], original)
        actual = conditioning_data(result["local_observation"][None], self.task)
        expected = conditioning_data(original[None], self.task)
        self.assertEqual(set(actual), {"observations"})
        assert_array_equal(actual["observations"], expected["observations"])

    def test_truth_and_auxiliary_values_are_neither_used_nor_required(self):
        expected = self.run_input()["local_observation"]
        artifact = deepcopy(self.original)
        artifact.metadata["truth"] = {"theta": 999.0, "jump_intensity": -999.0}
        artifact.metadata["columns"].append({
            "name": "future_price", "dtype": "float", "role": "auxiliary",
            "unit": "currency",
        })
        for row in artifact.rows:
            row.update(quote_time="2099-01-01T00:00:00Z", future_price=12345.0)
        self.write_panel(artifact)
        assert_array_equal(self.run_input()["local_observation"], expected)

        artifact.metadata.pop("truth")
        artifact.metadata["columns"] = [
            col for col in artifact.metadata["columns"] if col["role"] != "auxiliary"
        ]
        for row in artifact.rows:
            del row["quote_time"], row["future_price"]
        self.write_panel(artifact)
        result = self.run_input()
        self.assertIsNone(result["local_panel"].truth)
        assert_array_equal(result["local_observation"], expected)

    def test_contract_order_units_and_complete_task_are_checked(self):
        for mutation in ("order", "units", "context", "noise", "checkpoint"):
            with self.subTest(mutation=mutation):
                artifact = deepcopy(self.original)
                checkpoint_task = deepcopy(self.task)
                if mutation == "order":
                    artifact.rows = (artifact.rows[1], artifact.rows[0], *artifact.rows[2:])
                elif mutation == "units":
                    artifact.metadata["conventions"]["price_unit"] = "percent"
                elif mutation == "context":
                    artifact.metadata["scientific_task"]["context"]["spot"] += 1.0
                elif mutation == "noise":
                    artifact.metadata["scientific_task"]["context"]["quote_noise_std"] *= 2
                else:
                    checkpoint_task["prior_bounds"][0][1] += 0.01
                self.write_panel(artifact)
                self.checkpoint_task.write_text(json.dumps(checkpoint_task))
                with self.assertRaises(ValueError):
                    self.run_input()

    def test_cached_example_cannot_be_silently_replaced_or_relabelled(self):
        for mutation in ("price", "case_id", "origin", "purpose"):
            with self.subTest(mutation=mutation):
                artifact = deepcopy(self.original)
                if mutation == "price":
                    artifact.rows[0]["price"] += 0.01  # Finite, but not this cached panel.
                elif mutation == "case_id":
                    artifact.metadata["provenance"]["case_id"] = "validation:256"
                elif mutation == "origin":
                    artifact.metadata["data_origin"] = "mixed"
                else:
                    artifact.metadata["purpose"] = "conditioning"
                self.write_panel(artifact)
                with self.assertRaises(ValueError):
                    self.run_input()


if __name__ == "__main__":
    unittest.main()
