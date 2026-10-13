"""Task/unit/role boundaries and frozen-checkpoint input equivalence."""

import copy
from datetime import date, timedelta
from pathlib import Path
import unittest

import numpy as np

from sbi4finance.datasets import prepare_return_window
from sbi4finance.svj import adapters, bank, model


def prepared_example():
    task = bank.default_task()
    days = [(date(2020, 1, 1) + timedelta(days=i)).isoformat() for i in range(model.WINDOW + 1)]
    returns = model.simulate(1, seed=714)["returns"][0]
    metadata = {
        "artifact_id": "svj-synthetic-test", "task_id": task["task_id"], "scientific_task": task,
        "measure": "P", "data_origin": "synthetic", "purpose": "conditioning",
        "conventions": {"currency": "USD", "adjustment_policy": "synthetic", "calendar": "explicit_synthetic_daily_grid",
                        "time_basis": "252_trading_days_per_year"},
        "provenance": {"source": "seeded daily SVJ", "series_id": "test:714"},
        "truth": {"future_label": 123},
    }
    records = [{"interval_start": start, "interval_end": end, "value": value}
               for start, end, value in zip(days, days[1:], returns, strict=False)]
    return prepare_return_window(
        records, metadata=metadata, source_kind="log_returns", source_unit="percent", return_kind="log",
        boundaries=days, dt_years=[1 / 252] * model.WINDOW, cutoff=days[-1]), returns


class TestSVJAdapters(unittest.TestCase):
    def test_bank_conversion_preserves_native_features_and_paired_targets(self):
        sims = model.simulate(3, seed=913)
        split = {"observations": sims["returns"][..., None] / 100,
                 "parameters": sims["targets"][:, :8], "terminal_state": sims["variance"][:, -1:],
                 "dt_years": np.full(model.WINDOW, 1 / 252), "future_truth": np.full((3, 3), 999)}
        result = adapters.bank_batch(split)
        np.testing.assert_array_equal(result["features"], model.network_input(sims["returns"]))
        np.testing.assert_allclose(result["targets"], sims["targets"], rtol=0, atol=1e-15)
        altered = {**split, "parameters": split["parameters"] + 1, "terminal_state": split["terminal_state"] * 2}
        np.testing.assert_array_equal(adapters.bank_batch(altered)["features"], result["features"])

    def test_wrong_shapes_grids_and_state_fail(self):
        cfg = {"task": bank.default_task(), "root_seed": 1, "splits": {"train": 1, "validation": 1, "test": 1}}
        split = bank.generate_examples(cfg, "train", [0])
        with self.assertRaisesRegex(ValueError, "decimal bank adapter"):
            adapters.native_features(split["observations"])
        wrong_features = adapters.observations_to_features(split["observations"])
        wrong_features[..., 1] += 1
        with self.assertRaisesRegex(ValueError, "prepared feature channel"):
            adapters.native_features(wrong_features)
        for name, value in (("dt_years", np.full(1000, 1/365)), ("terminal_state", np.zeros((1, 1))),
                            ("parameters", np.zeros((1, 9))), ("observations", np.zeros((1, 999, 1)))):
            with self.subTest(name=name), self.assertRaises(ValueError):
                adapters.bank_batch({**split, name: value})

    def test_prepared_schedule_units_task_and_roles(self):
        artifact, percent = prepared_example()
        np.testing.assert_allclose(adapters.prepared_returns(artifact), percent, rtol=1e-15)
        expected = model.network_input(percent[None])
        np.testing.assert_array_equal(adapters.prepared_features(artifact), expected)
        artifact.metadata["truth"] = {"different_future": -999}
        np.testing.assert_array_equal(adapters.prepared_features(artifact), expected)
        for change in (lambda a: a.metadata.update(measure="Q"),
                       lambda a: a.metadata["conventions"].update(return_unit="percent"),
                       lambda a: a.metadata["scientific_task"]["prior"].update(drift_loc=50),
                       lambda a: a.rows[50].update(interval_start=a.rows[49]["interval_start"]),
                       lambda a: a.rows[0].update(observed=False),
                       lambda a: a.metadata["columns"][4].update(role="auxiliary")):
            broken = copy.deepcopy(artifact)
            change(broken)
            with self.assertRaises(ValueError):
                adapters.prepared_returns(broken)

    def test_frozen_checkpoint_same_seed_draws_agree_across_adapter(self):
        from sbi4finance.svj.amortized import AmortizedPosterior
        root = Path(__file__).resolve().parents[1]
        posterior = AmortizedPosterior.load(root / "checkpoints/svj")
        native = model.simulate(1, seed=17)["returns"]
        adapted = adapters.observations_to_features(native[..., None] / 100)
        np.testing.assert_array_equal(posterior.sample(native, num_samples=12, seed=13),
                                      posterior.sample(adapted, num_samples=12, seed=13))


if __name__ == "__main__":
    unittest.main()
