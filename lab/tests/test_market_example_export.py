"""Fixed-window export preserves return units and rejects misleading inputs."""

import importlib.util
import hashlib
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
    def chart_source(self, folder, prices=(100., 102., 99.)):
        source, receipt = folder / "chart.json", folder / "fetch.json"
        times = [1577975400, 1578061800, 1578321000]
        response = {"chart": {"error": None, "result": [{"meta": {"symbol": "VOO"},
            "timestamp": times, "indicators": {"adjclose": [{"adjclose": list(prices)}]}}]}}
        source.write_text(json.dumps(response))
        receipt.write_text(json.dumps({"raw_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "count": 3, "first_timestamp": times[0], "last_timestamp": times[-1],
            "url": "https://query1.finance.yahoo.com/v8/finance/chart/VOO", "fetched_at_utc": "2026-11-14T00:00:00Z"}))
        return source, receipt

    def test_downloaded_adjusted_prices_reproduce_every_return_and_bind_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source, receipt = self.chart_source(folder)
            metadata = export.prepare_chart(source, folder / "example", fetch_record=receipt)
            with np.load(folder / "example/returns.npz", allow_pickle=False) as values:
                np.testing.assert_array_equal(values["adjusted_close"], [100., 102., 99.])
                np.testing.assert_array_equal(values["simple_returns"], np.array([102/100-1, 99/102-1]))
                np.testing.assert_array_equal(values["log_returns"], np.log1p(values["simple_returns"]))
                self.assertEqual(values["dates"].tolist(), ["2020-01-03", "2020-01-06"])
            self.assertEqual(metadata["n"], 2)
            self.assertEqual(metadata["source_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(metadata["end_date"], "2020-01-06")

    def test_missing_price_or_modified_download_refuses_to_silently_drop_rows(self):
        for prices in ((100., None, 99.), (100., 0., 99.), (100., -2., 99.)):
            with self.subTest(prices=prices), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                source, receipt = self.chart_source(folder, prices)
                with self.assertRaises(ValueError):
                    export.prepare_chart(source, folder / "example", fetch_record=receipt)
                self.assertFalse((folder / "example").exists())
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source, receipt = self.chart_source(folder)
            source.write_text(source.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "receipt"):
                export.prepare_chart(source, folder / "example", fetch_record=receipt)

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
