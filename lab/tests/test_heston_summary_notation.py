"""Notation-aligned figures preserve the frozen recovery study and provenance."""

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.export_heston_summary_notation import display_labels


LAB = Path(__file__).resolve().parents[1]
SOURCE = LAB / "notebooks/data/heston_summary_dimensions"
DATA = LAB / "notebooks/data/heston_summary_aligned_notation"
NOTEBOOK = LAB / "notebooks/heston_spx_calibration.py"
spec = importlib.util.spec_from_file_location("summary_notation_under_test", LAB / "notebooks/_heston_summary_notation.py")
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)


class HestonSummaryNotationTests(unittest.TestCase):
    def setUp(self):
        self.record = json.loads((SOURCE / "results.json").read_text())
        self.labels = display_labels(NOTEBOOK)

    def test_all_display_figures_bind_to_original_sources_and_current_notation(self):
        before = {p.name: sha256(p.read_bytes()).hexdigest() for p in SOURCE.iterdir() if p.is_file()}
        figures = loader.load_figures(DATA, SOURCE, self.record, self.labels)
        self.assertEqual(set(figures), {None, 64, 32, 16, 8, 7, 6, 5, 4, 2})
        self.assertTrue(all(p.parent == DATA and p.is_file() for p in figures.values()))
        self.assertEqual(self.labels[2:5], [r"long variance $L$", r"volatility of variance $\sigma$",
                                           r"leverage correlation $\rho$"])
        self.assertEqual(before, {p.name: sha256(p.read_bytes()).hexdigest() for p in SOURCE.iterdir() if p.is_file()})

    def test_changed_image_source_metric_label_or_path_is_rejected(self):
        original = json.loads((DATA / "manifest.json").read_text())
        for field in ("sha256", "source_archived_image_sha256", "source_draws_sha256", "truth_sha256",
                      "observations_sha256", "metric", "label", "file", "duplicate", "missing"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                shutil.copytree(DATA, root, dirs_exist_ok=True)
                record = deepcopy(original)
                if field == "metric":
                    record["figures"][0]["metrics"]["rmse"][0] += .01
                elif field == "label":
                    record["labels"][2] = r"long variance $\theta$"
                elif field == "file":
                    record["figures"][0]["file"] = "../outside.png"
                elif field == "duplicate":
                    record["figures"].append(record["figures"][0])
                elif field == "missing":
                    record["figures"].pop()
                else:
                    record["figures"][0][field] = "0" * 64
                (root / "manifest.json").write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    loader.load_figures(root, SOURCE, self.record, self.labels)


if __name__ == "__main__":
    unittest.main()
