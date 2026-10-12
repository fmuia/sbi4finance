import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sbi4finance import option_bank
from sbi4finance.option_panel import noisy_panel

ROOT = Path(__file__).resolve().parents[1]


def config():
    c = json.loads((ROOT / "configs/tasks/hkde_option_bank.json").read_text())
    c["splits"] = dict(train=3, validation=2, test=2)
    return c


class OptionBankTests(unittest.TestCase):
    def test_ids_order_and_bank_extension_do_not_change_examples(self):
        c = config()
        a = option_bank.generate_examples(c, "train", [0, 1, 2])
        c["splits"]["train"] = 50
        b = option_bank.generate_examples(c, "train", [2, 0])
        for key in a:
            np.testing.assert_array_equal(a[key][[2, 0]], b[key])
        other = option_bank.generate_examples(c, "validation", [0])
        self.assertFalse(np.array_equal(a["parameters"][:1], other["parameters"]))
        self.assertEqual(a["observations"].shape, (3, 9, 1))

    def test_roundtrip_roles_and_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank"
            option_bank.write_bank(path, config())
            manifest, arrays = option_bank.load_bank(path)
            self.assertEqual(manifest["evidence_scope"], "S")
            self.assertFalse(manifest["numerical_scope"]["posterior_validated"])
            self.assertEqual(manifest["truth_available"], ["theta", "jump_intensity"])
            with self.assertRaises(FileExistsError):
                option_bank.write_bank(path, config())
            # An integrity-valid file with invalid MC covariance must still fail.
            train = arrays["train"]
            train["pricing_covariance"][0, 0, 0] = -1
            np.savez(path / "train.npz", **train)
            manifest["splits"]["train"]["sha256"] = option_bank._sha256(path / "train.npz")
            option_bank._write_json(path / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "covariance"):
                option_bank.load_bank(path)
            with (path / "train.npz").open("ab") as file:
                file.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                option_bank.load_bank(path)

    def test_numerical_settings_are_part_of_task_identity(self):
        c = config()
        changed = copy.deepcopy(c)
        changed["task"]["numerics"]["nodes"] += 1
        self.assertNotEqual(option_bank.task_fingerprint(c), option_bank.task_fingerprint(changed))
        changed = copy.deepcopy(c)
        changed["task"]["parameter_names"].reverse()
        changed["task"]["prior_bounds"].reverse()
        self.assertNotEqual(option_bank.task_fingerprint(c), option_bank.task_fingerprint(changed))

    def test_quote_noise_is_explicit_and_not_clipped(self):
        context = config()["task"]["context"]
        prices = np.zeros(9)
        actual = noisy_panel(prices, context, np.random.default_rng(12))[:, 0]
        expected = context["quote_noise_std"] * np.random.default_rng(12).standard_normal(9)
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.any(actual < 0))

    def test_reject_misdeclared_inputs(self):
        for change in (lambda c: c["task"].update(measure="P"),
                       lambda c: c["task"]["context"].update(quote_noise_std=0),
                       lambda c: c["task"]["prior_bounds"].__setitem__(0, [-1, .02])):
            c = config()
            change(c)
            with self.assertRaises(ValueError):
                option_bank.validate_config(c)

    def test_failed_precision_retains_original_id_without_redrawing(self):
        c = config()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failed"
            with patch.object(option_bank, "price_panel", return_value=(
                    np.ones(9), np.zeros(9), np.zeros((9, 9)), np.ones(9))) as pricing:
                with self.assertRaisesRegex(ValueError, "train:0.*refinement budget"):
                    option_bank.write_bank(path, c)
                self.assertEqual(pricing.call_count, 1)
            self.assertFalse((path / "manifest.json").exists())
            self.assertIn("train:0", json.loads((path / "failure.json").read_text())["error"])


if __name__ == "__main__":
    unittest.main()
