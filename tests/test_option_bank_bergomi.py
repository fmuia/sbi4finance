import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sbi4finance import option_bank

ROOT = Path(__file__).resolve().parents[1]


class BergomiBankTests(unittest.TestCase):
    def _check_roundtrip_and_order(self, filename):
        c = json.loads((ROOT / "configs/tasks" / filename).read_text())
        c["splits"] = dict(train=2, validation=1, test=1)
        # Interface fixture only: the scientific preset has much higher precision.
        c["task"]["numerics"].update(n_paths=256, n_steps=16, max_mc_se_over_quote_sigma=5.)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank"
            option_bank.write_bank(path, c)
            manifest, arrays = option_bank.load_bank(path)
            regenerated = option_bank.generate_examples(c, "train", [1, 0])
            for key in regenerated:
                np.testing.assert_array_equal(arrays["train"][key][[1, 0]], regenerated[key])
            self.assertFalse(manifest["numerical_scope"]["refinement_difference_available"])
            covariance = arrays["train"]["pricing_covariance"]
            self.assertTrue(np.any(covariance[:, 0, 1] != 0))
            np.testing.assert_allclose(np.diagonal(covariance, axis1=-2, axis2=-1),
                                       arrays["train"]["pricing_standard_errors"] ** 2)

    def test_one_factor_roundtrip_and_order(self):
        self._check_roundtrip_and_order("bergomi_option_bank.json")

    def test_rough_roundtrip_and_order(self):
        self._check_roundtrip_and_order("rough_bergomi_option_bank.json")


if __name__ == "__main__":
    unittest.main()
