"""The fixed HKDE premium panel must preserve meaning, order and information."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from sbi4finance.datasets import load_artifact, write_artifact
from sbi4finance.datasets.options import hkde_panel_prices, prepare_hkde_panel


TASK = json.loads(
    (Path(__file__).parent / "fixtures/local/hkde_task.json").read_text()
)
METADATA = {
    "artifact_id": "hkde-panel-test",
    "task_id": TASK["task_id"],
    "measure": "Q",
    "data_origin": "synthetic",
    "purpose": "parser_test",
    "provenance": {"source": "hand-constructed interface test; not model prices"},
    "conventions": {
        "quote_form": "single_premium",
        "price_unit": "currency",
        "exercise": "European",
        "settlement": "cash",
        "premium_basis": "per_underlying_unit",
    },
}
PRICES = (0.0, -0.1, 10.3, 0.5, 5.1, 12.1, 2.0, 8.2, 14.7)


def records():
    return [
        {"contract_id": f"source-{i}", **contract, "price": price}
        for i, (contract, price) in enumerate(zip(TASK["context"]["contracts"], PRICES))
    ]


class OptionPreparationTests(unittest.TestCase):
    def prepare(self, rows=None, metadata=None, task=None):
        return prepare_hkde_panel(
            records() if rows is None else rows,
            metadata=copy.deepcopy(METADATA) if metadata is None else metadata,
            task=copy.deepcopy(TASK) if task is None else task,
        )

    def extract(self, artifact, task=None):
        return hkde_panel_prices(artifact, checkpoint_task=TASK if task is None else task)

    def test_preserves_single_premiums_order_and_unknown_timestamps(self):
        artifact = self.prepare()
        self.assertEqual(self.extract(artifact), PRICES)
        self.assertEqual(artifact.observations, {"price": PRICES})
        self.assertEqual(set(artifact.context), {"strike", "maturity", "option_type"})
        self.assertEqual(artifact.metadata["scientific_task"], TASK)
        self.assertEqual(artifact.metadata["representation"], "raw_quote")
        self.assertEqual([r["contract_id"] for r in artifact.rows],
                         [f"source-{i}" for i in range(9)])
        # A strike is not a contract identifier: it legitimately repeats at each expiry.
        self.assertEqual(artifact.context["strike"].count(100.0), 3)
        self.assertTrue(all(r["quote_time"] is None for r in artifact.rows))
        self.assertNotIn("bid", artifact.rows[0])
        self.assertNotIn("ask", artifact.rows[0])
        self.assertIsNone(artifact.truth)

    def test_preparation_roundtrip_is_nonmutating_and_truth_is_not_required(self):
        source, metadata, task = records(), copy.deepcopy(METADATA), copy.deepcopy(TASK)
        source[0]["quote_time"] = "2026-10-09T15:45:00Z"
        originals = copy.deepcopy((source, metadata, task))
        artifact = self.prepare(source, metadata, task)
        self.assertEqual((source, metadata, task), originals)
        self.assertEqual(artifact.rows[0]["quote_time"], source[0]["quote_time"])
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "panel"
            write_artifact(destination, artifact.metadata, artifact.rows)
            restored = load_artifact(destination)
            self.assertEqual(restored.rows, artifact.rows)
            self.assertIsNone(restored.truth)
            self.assertEqual(self.extract(restored), PRICES)
        # The prepared artifact must not alias mutable task or record inputs.
        task["context"]["spot"] = 999.0
        source[0]["price"] = 999.0
        metadata["conventions"]["exercise"] = "American"
        self.assertEqual(self.extract(artifact), PRICES)

    def test_truth_clean_prices_and_future_targets_do_not_enter_conditioning(self):
        source = records()
        for row in source:
            row.update(clean_price=12345.0, theta=0.04, future_payoff=99999.0)
        labelled = copy.deepcopy(METADATA)
        labelled["truth"] = {"theta": 0.04, "jump_intensity": 0.5}
        artifact = self.prepare(source, labelled)
        self.assertEqual(artifact.truth, labelled["truth"])
        self.assertEqual(artifact.rows, self.prepare().rows)
        self.assertEqual(self.extract(artifact), PRICES)
        # Explicitly declared auxiliary diagnostics may accompany a stored panel.
        artifact.metadata["columns"].append({
            "name": "future_payoff", "dtype": "float", "role": "auxiliary",
            "unit": "currency",
        })
        for row in artifact.rows:
            row["future_payoff"] = 99999.0
        self.assertEqual(self.extract(artifact), PRICES)
        self.assertNotIn("future_payoff", artifact.observations)
        self.assertNotIn("future_payoff", artifact.context)
        artifact.metadata.pop("truth")
        artifact.metadata["columns"] = [
            col for col in artifact.metadata["columns"] if col["role"] != "auxiliary"
        ]
        for row in artifact.rows:
            row.pop("future_payoff")
            row.pop("quote_time")
        self.assertEqual(self.extract(artifact), PRICES)

    def test_contract_count_coordinates_order_and_ids_cannot_be_guessed(self):
        missing, extra, swapped = records(), records(), records()
        missing.pop()
        extra.append({**extra[-1], "contract_id": "extra"})
        swapped[0], swapped[1] = swapped[1], swapped[0]
        duplicate_contract, duplicate_id = records(), records()
        duplicate_contract[1] = {**duplicate_contract[0], "contract_id": "different-id"}
        duplicate_id[1]["contract_id"] = duplicate_id[0]["contract_id"]
        changed_coordinate, changed_style = records(), records()
        changed_coordinate[0]["maturity"] = 0.26
        changed_style[0]["option_type"] = "call"
        for label, source in (
            ("missing", missing), ("extra", extra), ("permuted", swapped),
            ("duplicate contract", duplicate_contract), ("duplicate ID", duplicate_id),
            ("expiry", changed_coordinate), ("call", changed_style),
        ):
            with self.subTest(case=label), self.assertRaises(ValueError):
                self.prepare(source)

    def test_missing_and_nonfinite_values_are_not_imputed_or_filtered(self):
        for value in (None, float("nan"), float("inf"), -float("inf"), "invalid"):
            with self.subTest(price=value), self.assertRaises(ValueError):
                source = records()
                source[0]["price"] = value
                self.prepare(source)
        source = records()
        del source[0]["price"]
        with self.assertRaises(ValueError):
            self.prepare(source)

    def test_conflicting_metadata_cannot_silently_relabel_the_input(self):
        cases = []
        for key, value in (("task_id", "different-task"), ("measure", "P"),
                           ("representation", "processed_iv")):
            meta = copy.deepcopy(METADATA)
            meta[key] = value
            cases.append((key, meta))
        for key, value in (("quote_form", "bid_ask"), ("price_unit", "annualized_decimal"),
                           ("exercise", "American"), ("settlement", "physical"),
                           ("premium_basis", "per_contract")):
            meta = copy.deepcopy(METADATA)
            meta["conventions"][key] = value
            cases.append((key, meta))
        meta = copy.deepcopy(METADATA)
        meta["scientific_task"] = copy.deepcopy(TASK)
        meta["scientific_task"]["context"]["spot"] = 101.0
        cases.append(("existing scientific task", meta))
        for label, meta in cases:
            with self.subTest(case=label), self.assertRaises(ValueError):
                self.prepare(metadata=meta)

    def test_market_origin_does_not_create_or_accept_synthetic_labels(self):
        market = copy.deepcopy(METADATA)
        market["data_origin"] = "market"
        self.assertIsNone(self.prepare(metadata=market).truth)
        for truth in ({}, {"theta": 0.04}):
            with self.subTest(truth=truth), self.assertRaises(ValueError):
                self.prepare(metadata={**market, "truth": truth})

    def test_checkpoint_requires_the_entire_scientific_task_not_only_shape(self):
        artifact = self.prepare()
        mutations = (
            ("task_id", None, "other-task"), ("model", None, "bergomi"),
            ("measure", None, "P"), ("context", "spot", 101.0),
            ("context", "quote_noise_std", 0.5), ("context", "rate", 0.03),
            ("fixed_parameters", "rho", -0.5), ("model_context", "initial_variance", 0.05),
            ("numerics", "nodes", 1024),
            ("prior_bounds", None, [[0.025, 0.065], [0.25, 2.0]]),
            ("parameter_names", None, ["jump_intensity", "theta"]),
        )
        for section, key, value in mutations:
            changed = copy.deepcopy(TASK)
            if key is None:
                changed[section] = value
            else:
                changed[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                self.extract(artifact, changed)

    def test_checkpoint_gate_rechecks_stored_roles_conventions_and_rows(self):
        cases = []
        for key, value in (("representation", "processed_iv"), ("measure", "P"),
                           ("task_id", "different-task")):
            changed = self.prepare()
            changed.metadata[key] = value
            cases.append((key, changed))
        for key, value in (("price_unit", "annualized_decimal"), ("exercise", "American")):
            changed = self.prepare()
            changed.metadata["conventions"][key] = value
            cases.append((key, changed))
        for column, key, value in (("price", "unit", "percent"),
                                   ("price", "role", "auxiliary"),
                                   ("strike", "role", "auxiliary"),
                                   ("maturity", "unit", "days")):
            changed = self.prepare()
            next(c for c in changed.metadata["columns"] if c["name"] == column)[key] = value
            cases.append((f"{column}:{key}", changed))
        for role in ("observation", "context"):
            changed = self.prepare()
            changed.metadata["columns"].append({
                "name": "target", "dtype": "float", "role": role, "unit": "currency",
            })
            for row in changed.rows:
                row["target"] = 12345.0
            cases.append((f"additional {role}", changed))
        permuted = self.prepare()
        permuted.rows = (permuted.rows[1], permuted.rows[0], *permuted.rows[2:])
        cases.append(("stored order", permuted))
        missing = self.prepare()
        missing.rows = missing.rows[:-1]
        cases.append(("stored missing row", missing))
        nonfinite = self.prepare()
        nonfinite.rows[0]["price"] = float("nan")
        cases.append(("stored nonfinite price", nonfinite))
        for label, artifact in cases:
            with self.subTest(case=label), self.assertRaises(ValueError):
                self.extract(artifact)


if __name__ == "__main__":
    unittest.main()
