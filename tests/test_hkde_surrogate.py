"""Exact polynomial checks, context binding and plain-artifact round trips."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

from sbi4finance import hkde_surrogate as surrogate
from sbi4finance.hkde_reference import DEFAULT_TARGET_CONTRACTS


TASK = json.loads((Path(__file__).resolve().parents[1] / "configs/tasks/hkde_inference_bank.json").read_text())["task"]
CONTRACTS = list(DEFAULT_TARGET_CONTRACTS)


def rows_from_scaled(scaled):
    bounds = np.asarray(TASK["prior_bounds"])
    return bounds[:, 0] + (np.asarray(scaled) + 1) / 2 * np.diff(bounds)[:, 0]


def training_rows():
    # Rectangular grid, with more rows than the degree-three tensor basis.
    scaled = np.stack(np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-1, 1, 8), indexing="ij"), axis=-1).reshape(-1, 2)
    return scaled, rows_from_scaled(scaled)


def polynomial(scaled):
    x, y = np.asarray(scaled).T
    return np.column_stack([3 + .7*x + .4*y + .2*x*y + .3*x**2 - .1*y**3,
                            1 + .2*x - .1*y + .4*x**2*y**2])


class HKDESurrogateTests(unittest.TestCase):
    def test_nontrivial_mixed_polynomial_recovers_independent_rows(self):
        scaled, parameters = training_rows()
        model = surrogate.fit(TASK, CONTRACTS, parameters, polynomial(scaled), degree=3)
        independent = np.random.default_rng(561).uniform(-1, 1, size=(127, 2))
        result = surrogate.predict(model, rows_from_scaled(independent), task=TASK, contracts=CONTRACTS)
        assert_allclose(result, polynomial(independent), rtol=2e-14, atol=2e-14)
        self.assertEqual(model["rank"], 16)
        self.assertEqual(model["coefficients"].shape, (16, 2))
        self.assertEqual(model["coefficients"].dtype, np.float64)
        self.assertGreaterEqual(model["condition_number"], 1.)
        self.assertLess(model["condition_number"], 10.)
        assert_allclose(surrogate.predict(model, rows_from_scaled([[-1, -1], [1, 1]])),
                        polynomial([[-1, -1], [1, 1]]), atol=2e-14)

    def test_default_degree_six_recovers_high_order_mixed_terms(self):
        scaled, parameters = training_rows()
        def values(z):
            x, y = z.T
            t6x = 32*x**6 - 48*x**4 + 18*x**2 - 1
            t5y = 16*y**5 - 20*y**3 + 5*y
            return np.column_stack([3 + .1*t6x*t5y, 2 + .2*x**3 - .1*y**6])
        model = surrogate.fit(TASK, CONTRACTS, parameters, values(scaled))
        self.assertEqual(model["degree"], 6)
        self.assertEqual(model["rank"], 49)
        independent = np.random.default_rng(563).uniform(-1, 1, size=(51, 2))
        assert_allclose(surrogate.predict(model, rows_from_scaled(independent)),
                        values(independent), rtol=2e-14, atol=5e-14)

    def test_currency_output_scales_with_labels_without_output_normalization(self):
        scaled, parameters = training_rows()
        base = surrogate.fit(TASK, CONTRACTS, parameters, polynomial(scaled), degree=3)
        converted = surrogate.fit(TASK, CONTRACTS, parameters, 100*polynomial(scaled), degree=3)
        # Units of labels must be declared correctly by the producer. The
        # regression does not silently standardize or convert these numbers.
        assert_allclose(surrogate.predict(converted, parameters),
                        100*surrogate.predict(base, parameters), rtol=2e-14, atol=2e-12)

    def test_saved_reload_predictions_are_exact_and_archive_has_no_pickle(self):
        scaled, parameters = training_rows()
        model = surrogate.fit(TASK, CONTRACTS, parameters, polynomial(scaled), degree=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            surrogate.save_model(path, model)
            loaded = surrogate.load_model(path)
            self.assertTrue(np.array_equal(surrogate.predict(loaded, parameters),
                                           surrogate.predict(model, parameters)))
            self.assertTrue(np.array_equal(loaded["coefficients"], model["coefficients"]))
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), {"metadata", "coefficients"})
                self.assertEqual(saved["metadata"].shape, ())
                self.assertEqual(saved["coefficients"].dtype, np.float64)
            with self.assertRaises(FileExistsError):
                surrogate.save_model(path, model)

    def test_context_contract_order_and_raw_parameter_units_are_bound(self):
        scaled, parameters = training_rows()
        mutable_task, mutable_contracts = copy.deepcopy(TASK), copy.deepcopy(CONTRACTS)
        model = surrogate.fit(mutable_task, mutable_contracts, parameters, polynomial(scaled), degree=3)
        mutable_task["context"]["spot"] = 1.
        mutable_contracts[0]["strike"] = .95
        self.assertEqual(model["task"], TASK)
        self.assertEqual(model["contracts"], CONTRACTS)
        for section, key, value in (("context", "spot", 1.), ("context", "rate", .03),
                                    ("model_context", "initial_variance", .05),
                                    ("fixed_parameters", "rho", -.5)):
            changed = copy.deepcopy(TASK)
            changed[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                surrogate.predict(model, parameters, task=changed)
        with self.assertRaises(ValueError):
            surrogate.predict(model, parameters, contracts=CONTRACTS[::-1])
        with self.assertRaises(ValueError):
            surrogate.fit(TASK, CONTRACTS[::-1], parameters, polynomial(scaled), degree=3)
        wrong_units = parameters.copy()
        wrong_units[:, 0] *= 100  # Percent variance in place of decimal annual variance.
        with self.assertRaises(ValueError):
            surrogate.predict(model, wrong_units)
        swapped = copy.deepcopy(TASK)
        swapped["parameter_names"] = ["jump_intensity", "theta"]
        swapped["prior_bounds"] = swapped["prior_bounds"][::-1]
        with self.assertRaises(ValueError):
            surrogate.fit(swapped, CONTRACTS, parameters[:, ::-1], polynomial(scaled), degree=3)

    def test_no_out_of_domain_clipping_and_no_nonfinite_or_wrong_shape_inputs(self):
        scaled, parameters = training_rows()
        model = surrogate.fit(TASK, CONTRACTS, parameters, polynomial(scaled), degree=3)
        below = np.array(TASK["prior_bounds"])[:, 0]
        above = np.array(TASK["prior_bounds"])[:, 1]
        below[0] = np.nextafter(below[0], -np.inf)
        above[1] = np.nextafter(above[1], np.inf)
        for invalid in (below[None], above[None], np.zeros((2, 3)), np.zeros((0, 2)),
                        parameters[0], np.full((2, 2), np.nan), np.full((2, 2), np.inf)):
            with self.subTest(shape=np.shape(invalid)), self.assertRaises(ValueError):
                surrogate.predict(model, invalid)
        for prices in (np.ones((len(parameters), 1)), np.full((len(parameters), 2), np.nan)):
            with self.assertRaises(ValueError):
                surrogate.fit(TASK, CONTRACTS, parameters, prices, degree=3)
        for degree in (-1, True, 2.5):
            with self.assertRaises(ValueError):
                surrogate.fit(TASK, CONTRACTS, parameters, polynomial(scaled), degree=degree)

    def test_rank_deficient_or_underdetermined_design_raises_without_degree_change(self):
        parameter = np.asarray(TASK["prior_bounds"]).mean(axis=1)
        repeated = np.repeat(parameter[None], 64, axis=0)
        with self.assertRaisesRegex(ValueError, "rank-deficient"):
            surrogate.fit(TASK, CONTRACTS, repeated, np.ones((64, 2)), degree=3)
        scaled, parameters = training_rows()
        with self.assertRaisesRegex(ValueError, "too few"):
            surrogate.fit(TASK, CONTRACTS, parameters[:8], polynomial(scaled[:8]), degree=3)

    def test_constant_columns_are_valid_even_with_zero_prices(self):
        _, parameters = training_rows()
        prices = np.broadcast_to([0., 4.], (len(parameters), 2))
        model = surrogate.fit(TASK, CONTRACTS, parameters, prices, degree=3)
        independent = rows_from_scaled(np.random.default_rng(562).uniform(-1, 1, (21, 2)))
        assert_allclose(surrogate.predict(model, independent), np.broadcast_to([0., 4.], (21, 2)), atol=2e-14)
        constant = surrogate.fit(TASK, CONTRACTS, parameters[:1], prices[:1], degree=0)
        assert_allclose(surrogate.predict(constant, independent), np.broadcast_to([0., 4.], (21, 2)))

    def test_invalid_saved_metadata_coefficients_and_pickle_are_rejected(self):
        scaled, parameters = training_rows()
        model = surrogate.fit(TASK, CONTRACTS, parameters, polynomial(scaled), degree=3)
        metadata = {key: value for key, value in model.items() if key != "coefficients"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            for updates in ({"rank": 15}, {"condition_number": float("nan")},
                            {"price_units": "percent"}, {"training_rows": 8}, {"degree": True}):
                bad = dict(metadata, **updates)
                np.savez(path, coefficients=model["coefficients"], metadata=np.array(json.dumps(bad)))
                with self.subTest(updates=updates), self.assertRaises(ValueError):
                    surrogate.load_model(path)
            for coefficients in (model["coefficients"].astype(np.float32),
                                 np.full_like(model["coefficients"], np.inf), np.zeros((16, 1))):
                np.savez(path, coefficients=coefficients, metadata=np.array(json.dumps(metadata)))
                with self.assertRaises(ValueError):
                    surrogate.load_model(path)
            np.savez(path, coefficients=model["coefficients"], metadata=np.array(metadata, dtype=object))
            with self.assertRaises(ValueError):
                surrogate.load_model(path)


if __name__ == "__main__":
    unittest.main()
