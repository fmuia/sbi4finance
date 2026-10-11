"""Financial conversion and information-cutoff checks for local return windows."""

import copy
import math
from pathlib import Path
import tempfile
import unittest

from sbi4finance.datasets import load_artifact, prepare_return_window, write_artifact


METADATA = {
    "artifact_id": "explicit-return-window",
    "task_id": "example-returns",
    "measure": "P",
    "data_origin": "synthetic",
    "purpose": "parser_test",
    "provenance": {
        "source": "hand-constructed conversion example, not simulator truth",
        "series_id": "example-series",
    },
    "conventions": {
        "currency": "USD",
        "adjustment_policy": "none; synthetic",
        "calendar": "explicit test schedule",
        "time_basis": "supplied year fractions",
    },
}
BOUNDARIES = ["2026-01-02", "2026-01-05", "2026-01-06"]
DT = [1 / 252, 1 / 252]
PRICES = [
    {"timestamp": "2026-01-01", "value": "unused past value"},
    {"timestamp": "2026-01-02", "value": 100.0},
    {"timestamp": "2026-01-05", "value": 100.0},
    {"timestamp": "2026-01-06", "value": 110.0},
    {"timestamp": "2026-01-07", "value": "unused future value"},
]


class ReturnPreparationTests(unittest.TestCase):
    def prepare(self, records=None, **changes):
        arguments = {
            "metadata": copy.deepcopy(METADATA),
            "source_kind": "prices",
            "source_unit": "currency",
            "return_kind": "log",
            "boundaries": list(BOUNDARIES),
            "dt_years": list(DT),
            "cutoff": BOUNDARIES[-1],
        }
        arguments.update(changes)
        return prepare_return_window(
            copy.deepcopy(PRICES) if records is None else records, **arguments
        )

    def supplied(self, values):
        return [
            {"interval_start": start, "interval_end": end, "value": value}
            for start, end, value in zip(BOUNDARIES[:-1], BOUNDARIES[1:], values)
        ]

    def test_price_conversion_preserves_zero_and_explicit_model_time(self):
        log = self.prepare()
        simple = self.prepare(return_kind="simple")
        self.assertEqual(log.observations["return"][0], 0.0)
        self.assertAlmostEqual(log.observations["return"][1], math.log(1.1), places=14)
        self.assertEqual(simple.observations["return"][0], 0.0)
        self.assertAlmostEqual(simple.observations["return"][1], 0.1, places=14)
        # The Friday-to-Monday interval deliberately has the same model duration
        # as Monday-to-Tuesday. Calendar elapsed days cannot override this input.
        self.assertEqual(log.context, {"dt_years": tuple(DT)})
        self.assertEqual(log.metadata["representation"], "log_returns")
        self.assertEqual(simple.metadata["representation"], "simple_returns")
        self.assertEqual([row["interval_start"] for row in log.rows], BOUNDARIES[:-1])
        self.assertEqual([row["interval_end"] for row in log.rows], BOUNDARIES[1:])
        self.assertTrue(all(row["observed"] is True for row in log.rows))
        self.assertEqual([row["source_start"] for row in log.rows], [100.0, 100.0])
        self.assertEqual([row["source_end"] for row in log.rows], [100.0, 110.0])

    def test_supplied_returns_use_declared_scale_and_definition(self):
        cases = [
            ("simple_returns", "percent", [0.0, -50.0], "log", [0.0, -math.log(2)]),
            ("simple_returns", "decimal", [0.0, -0.5], "simple", [0.0, -0.5]),
            ("log_returns", "percent", [0.0, 100 * math.log(2)], "simple", [0.0, 1.0]),
            ("log_returns", "decimal", [0.0, -0.5], "log", [0.0, -0.5]),
        ]
        for kind, unit, values, output, expected in cases:
            with self.subTest(kind=kind, unit=unit, output=output):
                artifact = self.prepare(
                    self.supplied(values), source_kind=kind, source_unit=unit,
                    return_kind=output,
                )
                for actual, reference in zip(artifact.observations["return"], expected):
                    self.assertAlmostEqual(actual, reference, places=14)
                self.assertEqual([row["source_value"] for row in artifact.rows], values)

    def test_future_and_unselected_values_do_not_change_conditioning_or_lineage(self):
        original = self.prepare()
        records = copy.deepcopy(PRICES)
        records[0]["value"] = None
        records[-1]["value"] = float("nan")
        for row in records:
            row["future_return"] = 99999.0
            row["latent_variance"] = -42.0
        changed = self.prepare(records)
        self.assertEqual(original.rows, changed.rows)
        self.assertEqual(original.metadata, changed.metadata)
        self.assertEqual(set(changed.observations), {"return"})
        self.assertEqual(set(changed.context), {"dt_years"})
        self.assertNotIn("future_return", changed.rows[0])
        self.assertNotIn("latent_variance", changed.rows[0])
        report = changed.metadata["preparation"]
        self.assertEqual(report["input_rows"], 5)
        self.assertEqual(report["selected_rows"], 3)
        self.assertEqual(report["excluded_before_window"], 1)
        self.assertEqual(report["excluded_after_cutoff"], 1)
        self.assertEqual(len(report["selected_source_sha256"]), 64)
        records[3]["value"] = 120.0
        different = self.prepare(records)
        self.assertNotEqual(
            report["selected_source_sha256"],
            different.metadata["preparation"]["selected_source_sha256"],
        )

    def test_missing_or_invalid_selected_values_are_not_filled_or_dropped(self):
        for value in (None, float("nan"), float("inf"), 0.0, -1.0, "not a price"):
            with self.subTest(price=value):
                records = copy.deepcopy(PRICES)
                records[2]["value"] = value
                with self.assertRaises(ValueError):
                    self.prepare(records)
        for value in (None, float("nan"), float("inf"), -1.0, -1.1):
            with self.subTest(simple_return=value):
                with self.assertRaises(ValueError):
                    self.prepare(self.supplied([0.0, value]),
                                 source_kind="simple_returns", source_unit="decimal")

    def test_dates_and_intervals_must_match_the_explicit_schedule(self):
        missing = copy.deepcopy(PRICES)
        missing.pop(2)
        extra = copy.deepcopy(PRICES)
        extra.insert(2, {"timestamp": "2026-01-03", "value": 105.0})
        duplicate = copy.deepcopy(PRICES)
        duplicate.insert(2, copy.deepcopy(duplicate[1]))
        unordered = copy.deepcopy(PRICES)
        unordered[1], unordered[2] = unordered[2], unordered[1]
        timestamp = copy.deepcopy(PRICES)
        timestamp[2]["timestamp"] = "2026-01-05T16:00:00Z"
        for label, records in (("missing", missing), ("extra", extra),
                               ("duplicate", duplicate), ("unordered", unordered),
                               ("intraday", timestamp)):
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    self.prepare(records)
        returns = self.supplied([0.0, 0.1])
        returns[1]["interval_start"] = "2026-01-04"
        with self.assertRaises(ValueError):
            self.prepare(returns, source_kind="simple_returns", source_unit="decimal")

    def test_time_grid_cutoff_and_declared_units_are_checked(self):
        cases = [
            {"cutoff": "2026-01-05"},
            {"cutoff": "2026-01-07"},
            {"boundaries": ["2026-01-02", "2026-01-06", "2026-01-05"]},
            {"boundaries": ["2026-01-02", "2026-01-02", "2026-01-06"]},
            {"dt_years": [1 / 252]},
            {"dt_years": [0.0, 1 / 252]},
            {"dt_years": [1 / 252, float("nan")]},
            {"source_unit": "percent"},
            {"return_kind": "unknown"},
            {"source_kind": "guess_from_values"},
        ]
        for changes in cases:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.prepare(**changes)
        with self.assertRaises(ValueError):
            self.prepare(self.supplied([0.0, 0.1]), source_kind="log_returns",
                         source_unit="currency")

    def test_conflicting_metadata_and_market_truth_are_rejected(self):
        conflict = copy.deepcopy(METADATA)
        conflict["conventions"]["return_kind"] = "simple"
        representation = copy.deepcopy(METADATA)
        representation["representation"] = "simple_returns"
        market_truth = copy.deepcopy(METADATA)
        market_truth.update(data_origin="market", truth={"drift": 0.08})
        for meta in (conflict, representation, market_truth):
            with self.subTest(metadata=meta):
                with self.assertRaises(ValueError):
                    self.prepare(metadata=meta)
        market = copy.deepcopy(METADATA)
        market["data_origin"] = "market"
        self.assertIsNone(self.prepare(metadata=market).truth)

    def test_preparation_is_deterministic_nonmutating_and_roundtrips(self):
        records, metadata = copy.deepcopy(PRICES), copy.deepcopy(METADATA)
        originals = copy.deepcopy((records, metadata))
        first = self.prepare(records, metadata=metadata)
        second = self.prepare(records, metadata=metadata)
        self.assertEqual((records, metadata), originals)
        self.assertEqual(first.rows, second.rows)
        self.assertEqual(first.metadata, second.metadata)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "prepared"
            written = write_artifact(destination, first.metadata, first.rows)
            restored = load_artifact(destination)
            self.assertEqual(restored.rows, first.rows)
            self.assertEqual(restored.metadata, written.metadata)
            self.assertEqual(restored.observations, first.observations)
            self.assertEqual(restored.context, first.context)


if __name__ == "__main__":
    unittest.main()
