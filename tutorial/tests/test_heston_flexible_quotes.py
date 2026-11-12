"""Quote-set contracts and information-exclusion tests; no fitting or pricing."""
from copy import deepcopy
import json
import unittest

import numpy as np

from sbi4f_tutorial.heston_options import flexible_quotes as quotes


def records():
    time = np.repeat(np.linspace(.02, .9, 7), 9)
    forward = 100 * np.exp(.04 * time)
    return {"forward": forward, "discount": np.exp(-.03 * time),
            "strike": forward * np.tile(np.linspace(.7, 1.2, 9), 7), "maturity": time,
            "option_type": np.tile(["put"] * 6 + ["call"] * 3, 7),
            "half_spread": np.linspace(.02, .08, 63), "mid": np.linspace(-.1, 4, 63)}


class FlexibleQuotesTests(unittest.TestCase):
    def test_currency_units_observed_schema_and_type(self):
        data = records()
        encoded = quotes.encode_quotes(data, np.stack((data["mid"], data["mid"] + .5)))
        factor = data["forward"] * data["discount"]
        self.assertEqual(encoded["tokens"].shape, (2, 63, 5))
        np.testing.assert_allclose(encoded["tokens"][0, :, 0] * factor, data["mid"])
        np.testing.assert_allclose(encoded["tokens"][0, :, 4] * factor, data["half_spread"])
        np.testing.assert_array_equal(encoded["tokens"][0, :, 3], np.tile([1] * 6 + [0] * 3, 7))
        np.testing.assert_array_equal(encoded["count"], [[63], [63]])
        self.assertLess(encoded["tokens"][0, 0, 0], 0)  # noisy prices need not satisfy price bounds
        quotes.validate_actual63_records(data)
        with self.assertRaisesRegex(ValueError, "63"):
            quotes.validate_actual63_records({key: value[:62] for key, value in data.items()})

    def test_mask_excludes_payload_before_normalization(self):
        data = records()
        mask = quotes.make_masks(data, 2, mode="random", seed=3, retained_counts=(32,))
        prices = np.repeat(data["mid"][None, :], 2, axis=0)
        expected = quotes.encode_quotes(data, prices, mask)
        changed = {key: np.repeat(value[None, :], 2, axis=0) for key, value in data.items()}
        for key in quotes.GEOMETRY_KEYS:
            changed[key][~mask] = "missing" if key == "option_type" else np.nan
        prices[~mask] = np.inf
        actual = quotes.encode_quotes(changed, prices, mask)
        np.testing.assert_array_equal(actual["tokens"], expected["tokens"])
        self.assertTrue(np.all(actual["tokens"][~mask] == 0))
        changed["strike"][mask] = 0
        with self.assertRaisesRegex(ValueError, "strike"):
            quotes.encode_quotes(changed, prices, mask)

    def test_scaler_uses_training_valid_records_only_and_preserves_geometry(self):
        encoded = quotes.encode_quotes(records(), records()["mid"])
        mask = encoded["mask"].copy(); mask[:, :4] = False
        contaminated = encoded["tokens"].copy(); contaminated[~mask] = np.nan
        scaler = quotes.fit_token_scaler(contaminated, mask)
        scaler = json.loads(json.dumps(scaler))
        self.assertEqual(scaler["valid_training_tokens"], 59)
        result = quotes.transform_tokens(contaminated, mask, scaler)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result[mask][:, [0, 1, 2, 4]].mean(axis=0), 0, atol=2e-15)
        np.testing.assert_array_equal(result[mask][:, 3], contaminated[mask][:, 3])
        self.assertEqual(len(np.unique(result[mask][:, 2])), 7)
        with self.assertRaisesRegex(ValueError, "training split"):
            quotes.fit_token_scaler(encoded["tokens"], mask, split="validation")
        corrupted = deepcopy(scaler); corrupted["scale"][0] = 0
        with self.assertRaises(ValueError): quotes.transform_tokens(contaminated, mask, corrupted)

    def test_padding_and_token_permutation_do_not_change_retained_inputs(self):
        data = records(); mask = quotes.make_masks(data, 1, mode="random", seed=4, retained_counts=(32,))
        original = quotes.encode_quotes(data, data["mid"], mask)
        scaler = quotes.fit_token_scaler(original["tokens"], mask)
        scaled = quotes.transform_tokens(original["tokens"], mask, scaler)
        permutation = np.random.default_rng(8).permutation(63)
        permuted = quotes.encode_quotes({key: value[permutation] for key, value in data.items()},
                                       data["mid"][permutation], mask[:, permutation])
        np.testing.assert_array_equal(quotes.transform_tokens(permuted["tokens"], permuted["mask"], scaler), scaled[:, permutation])
        padded = {key: np.concatenate((value, ["padding"] * 2 if key == "option_type" else [np.nan] * 2))
                  for key, value in data.items()}
        padmask = np.concatenate((mask, np.zeros((1, 2), dtype=bool)), axis=1)
        result = quotes.encode_quotes(padded, np.concatenate((data["mid"], [np.nan, np.inf])), padmask)
        result_scaled = quotes.transform_tokens(result["tokens"], result["mask"], scaler)
        np.testing.assert_array_equal(result_scaled[:, :63], scaled)
        np.testing.assert_array_equal(result_scaled[:, 63:], 0)
        np.testing.assert_array_equal(result["count"], original["count"])

    def test_masks_are_reproducible_geometry_only_and_nonempty(self):
        data = records(); alternate = deepcopy(data); alternate["mid"][:] = np.nan
        for mode in ("full", "random", "expiry", "put_wing", "call_wing"):
            first = quotes.make_masks(data, 8, mode=mode, seed=15)
            second = quotes.make_masks(alternate, 8, mode=mode, seed=15)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(first.dtype, bool)
            self.assertTrue(np.all(first.sum(axis=1) > 0))
        self.assertTrue(np.all(quotes.make_masks(data, 8, mode="expiry").sum(axis=1) == 54))
        random = quotes.make_masks(data, 30, mode="random", seed=2, retained_counts=(16, 32))
        self.assertEqual(set(random.sum(axis=1)), {16, 32})
        for counts in ((0,), (64,), (2.5,), (True,), ()):
            with self.assertRaises(ValueError): quotes.make_masks(data, 1, mode="random", retained_counts=counts)

    def test_mask_and_value_validation(self):
        data = records(); price = data["mid"]
        for invalid in (np.ones((1, 63)), np.ones(63, dtype=bool), np.zeros((1, 63), dtype=bool)):
            with self.assertRaises(ValueError): quotes.encode_quotes(data, price, invalid)
        bad = price.copy(); bad[0] = np.nan
        with self.assertRaisesRegex(ValueError, "observations"): quotes.encode_quotes(data, bad)
        bad = deepcopy(data); bad["option_type"][0] = "other"
        with self.assertRaisesRegex(ValueError, "option_type"): quotes.encode_quotes(bad, price)
        with self.assertRaises(ValueError): quotes.make_masks(data, 0)
        with self.assertRaises(ValueError): quotes.make_masks(data, 1, seed=-1)

    def test_carry_interpolation_is_loglinear_and_rejects_extrapolation(self):
        data = records(); time = np.array([.02, .5, .9])
        carry = quotes.interpolate_carry(data, time)
        np.testing.assert_allclose(carry["forward"], 100 * np.exp(.04 * time), rtol=1e-14)
        np.testing.assert_allclose(carry["discount"], np.exp(-.03 * time), rtol=1e-14)
        for request in ([.019], [.901], [np.nan]):
            with self.assertRaisesRegex(ValueError, "cannot extrapolate"): quotes.interpolate_carry(data, request)
        data["forward"][1] *= 1.001
        with self.assertRaisesRegex(ValueError, "shared"): quotes.interpolate_carry(data, [.3])

    def test_geometry_jitter_shared_expiries_slot_spreads_and_pricing_guard(self):
        data = records(); original = deepcopy(data)
        jittered = quotes.jitter_geometry(data, 4, seed=31)
        replay = quotes.jitter_geometry(data, 4, seed=31)
        self.assertNotIn("mid", jittered)
        for key in quotes.GEOMETRY_KEYS:
            np.testing.assert_array_equal(jittered[key], replay[key])
            np.testing.assert_array_equal(data[key], original[key])
        normalized = data["half_spread"] / (data["forward"] * data["discount"])
        np.testing.assert_allclose(jittered["half_spread"] / (jittered["forward"] * jittered["discount"]),
                                   np.repeat(normalized[None, :], 4, axis=0), rtol=5e-16)
        for time in np.unique(data["maturity"]):
            selected = data["maturity"] == time
            self.assertTrue(np.all(np.ptp(jittered["maturity"][:, selected], axis=1) == 0))
            self.assertTrue(np.all(np.diff(jittered["strike"][:, selected], axis=1) > 0))
            self.assertTrue(np.all(abs(jittered["maturity"][:, selected] / time - 1) <= .05 + 1e-15))
        self.assertTrue(np.all(jittered["maturity"] >= data["maturity"].min()))
        self.assertTrue(np.all(jittered["maturity"] <= data["maturity"].max()))
        # A fingerprint proves location binding, not that a supplied price is correct.
        with self.assertRaisesRegex(ValueError, "old prices cannot be relabelled"):
            quotes.encode_quotes(jittered, np.repeat(data["mid"][None, :], 4, axis=0))
        with self.assertRaisesRegex(ValueError, "old prices cannot be relabelled"):
            quotes.encode_quotes(jittered, np.zeros((4, 63)), priced_geometry=quotes.geometry_fingerprint(data))
        newly_priced_stub = np.ones((4, 63))  # contract test only; no numerical acceptance
        encoded = quotes.encode_quotes(jittered, newly_priced_stub, priced_geometry=quotes.geometry_fingerprint(jittered))
        self.assertEqual(encoded["tokens"].shape, (4, 63, 5))


if __name__ == "__main__":
    unittest.main()
