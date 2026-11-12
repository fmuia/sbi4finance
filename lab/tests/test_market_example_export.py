"""Fixed-window export preserves return units and rejects misleading inputs."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    "market_export", Path(__file__).resolve().parents[1] / "scripts" / "export_market_example.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class MarketExportTests(unittest.TestCase):
    def test_window_and_log_conversion_preserve_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = folder / "market.csv"
            source.write_text("date,VOO\n2020-01-02,0.01\n2020-01-03,-0.02\n2020-01-06,0.03\n2020-01-07,0.8\n")
            target = folder / "example"
            metadata = export.prepare(source, target, end="2020-01-06", count=2)
            with np.load(target / "returns.npz", allow_pickle=False) as values:
                np.testing.assert_array_equal(values["simple_returns"], [-.02, .03])
                np.testing.assert_array_equal(values["log_returns"], np.log1p([-.02, .03]))
                self.assertEqual(values["dates"].tolist(), ["2020-01-03", "2020-01-06"])
            self.assertEqual(metadata["n"], 2)
            self.assertEqual(json.loads((target / "market.json").read_text()), metadata)
            with self.assertRaises(FileExistsError):
                export.prepare(source, target, count=2)

    def test_duplicate_dates_and_inadmissible_returns_fail(self):
        for body in ("2020-01-02,0.1\n2020-01-02,0.2\n",
                     "2020-01-02,0.1\n2020-01-03,-1\n",
                     "2020-01-02,0.1\n2020-01-03,nan\n"):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                source = folder / "market.csv"
                source.write_text("date,VOO\n" + body)
                with self.assertRaises(ValueError):
                    export.prepare(source, folder / "example", count=2)
                self.assertFalse((folder / "example").exists())


if __name__ == "__main__":
    unittest.main()
