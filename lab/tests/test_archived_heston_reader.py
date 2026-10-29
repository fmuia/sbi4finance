"""Guard the archived runtime/task boundary without loading a neural network."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("archived_reader", ROOT / "scripts/compare_archived_heston.py")
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


class ArchivedHestonReaderTest(unittest.TestCase):
    def test_refuses_new_bayesflow_for_old_serialized_checkpoint(self):
        with patch.object(reader.metadata, "version", return_value="2.1.0"):
            with self.assertRaisesRegex(ValueError, "BayesFlow2.0.14"):
                reader.check_archive(ROOT.parent / "tutorial/assets/checkpoints/heston")

    def test_shipped_task_and_three_fixed_references_match(self):
        with patch.object(reader.metadata, "version", return_value="2.0.14"):
            task, training, accepted = reader.check_archive(ROOT.parent / "tutorial/assets/checkpoints/heston")
        self.assertNotEqual(training["checkpoint_sha256"], reader.MODEL_SHA)
        self.assertIn("optimizer state removed", accepted["changes"][0])
        for index in range(3):
            case = ROOT / f"evidence/pilot-results/2026-10-27/heston-reviewed/case-{index}"
            reference, _, _, chains, _ = reader.check_reference(case, task)
            self.assertEqual(reference["surface_id"], f"validation:{index}")
            self.assertEqual(chains.shape, (4, 8000, 4))

    def test_different_quote_law_cannot_be_called_same_task(self):
        task = json.loads((ROOT.parent / "tutorial/assets/checkpoints/heston/task.json").read_text())
        different = copy.deepcopy(task)
        different["fixed_parameters"]["kappa"] *= 2
        case = ROOT / "evidence/pilot-results/2026-10-27/heston-reviewed/case-0"
        with self.assertRaisesRegex(ValueError, "exact matching"):
            reader.check_reference(case, different)

    def test_changed_observation_is_rejected_by_precision_identity(self):
        original = ROOT / "evidence/pilot-results/2026-10-27/heston-reviewed/case-0"
        task = json.loads((ROOT.parent / "tutorial/assets/checkpoints/heston/task.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            (case / "reference").mkdir()
            (case / "precision").mkdir()
            for name in ("chains.npz", "diagnostics.json"):
                (case / "reference" / name).write_bytes((original / "reference" / name).read_bytes())
            (case / "precision/precision.json").write_bytes((original / "precision/precision.json").read_bytes())
            changed = json.loads((original / "reference/reference.json").read_text())
            changed["observed_prices"][0] += .001
            (case / "reference/reference.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "exact matching"):
                reader.check_reference(case, task)


if __name__ == "__main__":
    unittest.main()
