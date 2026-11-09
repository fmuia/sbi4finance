"""Prior sensitivity must preserve observations and exact reference identities."""

from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import check_heston_pilot_kappa_prior as study


class KappaPriorBindingTests(unittest.TestCase):
    def test_wrong_design_is_rejected_before_loading_a_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            design = root / "design.json"
            design.write_text('{"records":{"mid":[1]}}\n')
            (root / "run.json").write_text(json.dumps({"design_sha256": "different-task"}))
            with self.assertRaisesRegex(ValueError, "exact declared design"):
                study.load_reference(root, design)

    def test_changed_quote_records_are_rejected_as_a_prior_sensitivity(self):
        first = {"records": {"mid": [1.]}, "products": {}, "short_mask": [True]}
        second = {"records": {"mid": [2.]}, "products": {}, "short_mask": [True]}
        args = Namespace(baseline=Path("first"), design=Path("first.json"),
                         variant=Path("second"), variant_design=Path("second.json"))
        with patch.object(study, "load_reference", side_effect=[({}, first, None, None),
                                                               ({}, second, None, None)]):
            with self.assertRaisesRegex(ValueError, "changed records"):
                study.study(args)


if __name__ == "__main__":
    unittest.main()
