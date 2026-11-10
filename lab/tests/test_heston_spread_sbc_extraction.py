"""Validation-member isolation, payload provenance and exact-metric guards."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import extract_heston_spread_sbc as extraction


class SBCExtractionTests(unittest.TestCase):
    def test_only_original_named_validation_arrays_are_opened(self):
        with tempfile.TemporaryDirectory() as temporary:
            bank = Path(temporary)
            truth,clean = np.full((4,5),.5),np.ones((4,54))
            np.savez_compressed(bank/"bank.npz",validation_parameters=truth,validation_clean=clean,
                                validation_row_id=np.array(["unused"]*4),test_parameters=np.full((2,5),np.nan))
            bank_record = {"payload":"bank.npz","payload_sha256":extraction.trainer.digest(bank/"bank.npz"),
                           "identity":{"declared_split":"validation"}}
            (bank/"bank.json").write_text(json.dumps(bank_record))
            record = {"bank_record_sha256":extraction.trainer.digest(bank/"bank.json"),
                      "bank_payload_sha256":bank_record["payload_sha256"],"bank_identity":bank_record["identity"],
                      "task":{"target":{"prior_bounds":[[0.,1.]]*5,"quote_count":54}}}
            real_load,accesses = np.load,[]
            class Archive:
                def __init__(self,path,**kwargs):
                    self.archive = real_load(path,**kwargs)
                def __enter__(self):
                    return self
                def __exit__(self,*args):
                    self.archive.close()
                @property
                def files(self):
                    raise AssertionError("do not enumerate bank members")
                def __getitem__(self,key):
                    accesses.append(key)
                    if key not in ("validation_parameters","validation_clean"):
                        raise AssertionError("undeclared/reserved member opened")
                    return self.archive[key]
            with patch.object(extraction.np,"load",side_effect=Archive):
                actual_truth,actual_clean = extraction.load_validation(bank,record,rows=4)
            self.assertEqual(accesses,["validation_parameters","validation_clean"])
            np.testing.assert_array_equal(actual_truth,truth)
            np.testing.assert_array_equal(actual_clean,clean)
            changed = copy.deepcopy(record)
            changed["bank_payload_sha256"] = "0"*64
            with patch.object(extraction.np,"load",side_effect=AssertionError("archive opened")) as load:
                with self.assertRaisesRegex(ValueError,"payload differs"):
                    extraction.load_validation(bank,changed,rows=4)
                load.assert_not_called()

    def test_population_requires_exact_saved_ranks_support_and_coverage(self):
        rng = np.random.default_rng(14)
        draws,truth = rng.normal(size=(8,16,5)),rng.normal(size=(8,5))
        ranks = extraction.validation.sbc(draws,truth,seed=extraction.SEED+2000)
        ranks.pop("fractional_ranks")
        target = {"prior_bounds":[[-10.,10.]]*5}
        evaluation = {"task":target,"population":{"rank_dkw":ranks,
            "coverage":extraction.validation.coverage(draws,truth),
            "support":extraction.evaluator.support_report(draws,np.array(target["prior_bounds"]))}}
        fractional,*_ = extraction.exact_population(draws,truth,evaluation)
        self.assertEqual(np.shape(fractional),(8,5))
        for field in ("rank_dkw","coverage","support"):
            bad = copy.deepcopy(evaluation)
            if field == "rank_dkw":
                bad["population"][field]["rank_ks"][0] += 1e-12
            elif field == "coverage":
                bad["population"][field]["0.9"]["coverage"][0] += 1e-12
            else:
                bad["population"][field]["outside_value_count"] += 1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError,"exactly match"):
                extraction.exact_population(draws,truth,bad)

    def test_explicit_evaluation_payload_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            (checkpoint/"model.keras").write_bytes(b"model")
            (checkpoint/"state.npz").write_bytes(b"state")
            (checkpoint/"checkpoint.json").write_text("{}")
            task = {"target":{"infer_c":False,"fixed_c":1.}}
            record = {"actual_optimizer_updates":10000,"task":task,
                "task_fingerprint":extraction.trainer.fingerprint(task),
                "model_sha256":extraction.trainer.digest(checkpoint/"model.keras"),
                "state_sha256":extraction.trainer.digest(checkpoint/"state.npz"),"source_sha256":{}}
            evaluation = {"task":task["target"],"seed":extraction.SEED,
                "population":{"rows":512,"draws_per_case":512},"model_sha256":record["model_sha256"]}
            extraction.evaluation_binding(evaluation,record,checkpoint,10000)
            evaluation["model_sha256"] = "0"*64
            with self.assertRaisesRegex(ValueError,"provenance differs"):
                extraction.evaluation_binding(evaluation,record,checkpoint,10000)


if __name__ == "__main__":
    unittest.main()
