"""Protect collaborator relocation, corruption handling and quote units."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scripts import heston_grid_collaborator as delivery


class HestonGridCollaboratorTests(unittest.TestCase):
    def test_member_escape_fails_before_model_deserialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);bundle=base/"bundle";bundle.mkdir();outside=base/"outside.bin";outside.write_bytes(b"x")
            (bundle/"bundle.json").write_text(json.dumps({"format":delivery.FORMAT,"scientific_acceptance":False,
                "files":{"../outside.bin":{"bytes":1,"sha256":delivery.digest(outside)}}}))
            with patch.object(delivery.producer,"build_workflow") as model:
                with self.assertRaisesRegex(ValueError,"missing/corrupt"): delivery.load_checkpoint(bundle)
                model.assert_not_called()

    def test_corrupt_state_fails_before_native_model_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle=Path(temporary);state=bundle/"state.npz";state.write_bytes(b"changed")
            (bundle/"bundle.json").write_text(json.dumps({"format":delivery.FORMAT,"scientific_acceptance":False,
                "files":{"state.npz":{"bytes":7,"sha256":"0"*64}}}))
            with patch.object(delivery.producer,"build_workflow") as model:
                with self.assertRaisesRegex(ValueError,"missing/corrupt"): delivery.load_checkpoint(bundle)
                model.assert_not_called()

    def test_training_uses_stored_y_over_halfspread_exactly_once(self):
        names=["initial_variance","kappa","theta","vol_of_vol","rho","c"]
        task={"target":{"parameter_names":names},"records":{"half_spread":[100.]*63}}
        arrays={}
        for split in ("train","validation"):
            arrays.update({f"{split}_{name}":np.ones((2,1)) for name in names})
            arrays[f"{split}_observations"]=np.full((2,63),2.) # Y/h already
        class FrozenWhitening:
            def transform(self,x): return x+3 # deterministic stand-in for stored affine transform
        with patch.object(delivery,"load_bank",return_value=(task,arrays,{})),patch.object(delivery,"_whitening",return_value=FrozenWhitening()):
            _,data=delivery.training_inputs("unused",rows=200000)
        np.testing.assert_array_equal(data["train"]["observations"],np.full((2,63),5.))
        self.assertEqual(set(data["train"]),set(names)|{"observations"})
        self.assertNotIn("historical",data)
