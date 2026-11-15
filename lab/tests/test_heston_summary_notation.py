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
        self.assertEqual(self.labels[2:5], [r"long-run variance $v_\infty$", r"volatility of variance $\sigma$",
                                           r"leverage correlation $\rho$"])
        self.assertEqual(before, {p.name: sha256(p.read_bytes()).hexdigest() for p in SOURCE.iterdir() if p.is_file()})

    def test_relabelling_replaces_only_title_pixels(self):
        import matplotlib
        from matplotlib import ft2font
        import numpy as np
        from PIL import Image
        from scripts.relabel_heston_summary_notation import relabel

        recorded = json.loads((DATA / "manifest.json").read_text()).get("relabelled", [{}])[-1]
        if (recorded.get("matplotlib"), recorded.get("freetype")) != (matplotlib.__version__,
                                                                      ft2font.__freetype_version__):
            self.skipTest("exact text rendering needs the recorded matplotlib and FreeType versions")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(DATA, root / "figures")
            labels = list(self.labels)
            labels[2] = r"long-term variance $v_\infty$"
            (root / "notebook.py").write_text(f"MATH_LABELS = {labels!r}\n")
            manifest = relabel(root / "figures", SOURCE, root / "notebook.py")
            self.assertEqual(manifest["labels"], labels)
            self.assertEqual(manifest["relabelled"][-1]["previous_labels"], self.labels)
            self.assertEqual(set(loader.load_figures(root / "figures", SOURCE, self.record, labels)),
                             {None, 64, 32, 16, 8, 7, 6, 5, 4, 2})
            for item in manifest["figures"]:
                before = np.asarray(Image.open(DATA / item["file"]).convert("RGBA"))
                after = np.asarray(Image.open(root / "figures" / item["file"]).convert("RGBA"))
                rows, columns = np.nonzero(np.any(before != after, axis=-1))
                # Only the third panel's title, above the top-right axes, may change.
                self.assertTrue(rows.size and rows.max() < 100 and columns.min() > before.shape[1] * 2 / 3)
                self.assertEqual(rows.size, manifest["relabelled"][-1]["changed_pixels"][item["file"]]["pixels"])

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
