"""Offline dependence review uses bound draws and counts each pair once."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import review_heston_spread_whitening_replay as review


class ReplayReviewTests(unittest.TestCase):
    def test_dependence_matches_identical_draws_and_has_fifteen_unique_pairs(self):
        rng=np.random.default_rng(5401);values=rng.standard_normal((4,128,6))
        values[:,:,1]+=.7*values[:,:,0]
        result=review.dependence(values.reshape(-1,6),values)
        self.assertEqual(result["unique_parameter_pairs"],15)
        self.assertEqual(result["max_absolute_correlation_difference"],0.)
        self.assertEqual(result["median_absolute_correlation_difference"],0.)
        self.assertEqual(result["reference_chain_shape"],[4,128,6])
        np.testing.assert_allclose(result["reference_correlation"],np.corrcoef(values.reshape(-1,6),rowvar=False))

    def test_correlation_rejects_nonfinite_wrong_dimension_and_constant_targets(self):
        for values in (np.ones((128,6)),np.ones((128,5)),np.full((128,6),np.nan)):
            with self.subTest(shape=values.shape),self.assertRaises(ValueError):review.correlation(values)

    def test_text_corrections_require_exactly_one_frozen_phrase(self):
        self.assertEqual(review.strict_replace("a old b","old","new"),"a new b")
        for text in ("none","old old"):
            with self.assertRaisesRegex(ValueError,"replacement count"):review.strict_replace(text,"old","new")

    def test_metadata_compatibility_is_narrow_hash_checked_and_restored_on_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();files={}
            for i in range(6):
                path=root/f"original-{i}.json";path.write_text("{}\n");files[str(i)]=review.binding(path)
            protocol={"contract":{"original":{"files":files}}}
            inputs=review.shared.Artifacts(root/"manifest.json")
            original=review.frozen.selected.source_binding
            with review.original_metadata_bindings(protocol) as record:
                review.frozen.selected.source_binding(inputs,files["0"],"original metadata")
                self.assertEqual(inputs.sources[-1]["sha256"],files["0"]["sha256"])
                with self.assertRaisesRegex(ValueError,"metadata hash differs"):
                    review.frozen.selected.source_binding(inputs,{**files["0"],"sha256":"bad"},"changed metadata")
                other=root/"unknown.json";other.write_text("{}\n")
                with self.assertRaisesRegex(ValueError,"executed reference source"):
                    review.frozen.selected.source_binding(inputs,review.binding(other),"undeclared JSON")
                code=root/"code.py";code.write_text("# compact source\n")
                review.frozen.selected.source_binding(inputs,review.binding(code),"unchanged code source validation")
            self.assertIs(review.frozen.selected.source_binding,original)
            self.assertTrue(record["original_hook_restored"])
            with self.assertRaisesRegex(RuntimeError,"render failure"):
                with review.original_metadata_bindings(protocol):raise RuntimeError("render failure")
            self.assertIs(review.frozen.selected.source_binding,original)

    def fixture(self,root):
        rng=np.random.default_rng(5402)
        draws=rng.uniform(.1,.9,(4096,6));chains=rng.uniform(.1,.9,(2,128,6))
        neural_path=root/"conditional.npz";reference_path=root/"chains.npz"
        np.savez(neural_path,**{case:draws for case in review.shared.CASES})
        np.savez(reference_path,parameters=chains)
        run=root/"run.json";run.write_text("{}\n");diagnostics=root/"diagnostics.json";diagnostics.write_text("{}\n")
        target={"prior_bounds":[[0.,1.]]*6,"parameter_names":list(review.frozen.NAMES)}
        q=np.quantile(draws,[.05,.5,.95],axis=0).tolist()
        rq=np.pad(np.quantile(chains.reshape(-1,6),[.05,.5,.95],axis=0),((0,0),(0,2))).tolist()
        case_record={"observation_sha256":"unchanged-fixture","reference_ready":True,
            "reference_run_path":str(run),"reference_run_sha256":review.binding(run)["sha256"],
            "reference_diagnostics_path":str(diagnostics),"reference_diagnostics_sha256":review.binding(diagnostics)["sha256"],
            "reference_chain_payload_path":str(reference_path),"reference_chain_payload_sha256":review.binding(reference_path)["sha256"],
            "neural_quantiles_05_50_95":q,"comparison":{"reference_quantiles_05_50_95":rq}}
        specs=[]
        for epoch in review.EPOCHS:
            out=root/f"epoch{epoch}";out.mkdir()
            record={"status":"complete","actual_optimizer_updates":epoch*40,"task":target,
                "reserved_arrays_opened":False,"new_reference_sampling":False,
                "conditional_draws":{**review.binding(neural_path),"draws_per_case":4096},
                "cases":{case:copy.deepcopy(case_record) for case in review.shared.CASES}}
            (out/"evaluation.json").write_text(json.dumps(record))
            if epoch!=200:specs.append({"epoch":epoch,"output":str(out)})
        return {"task":{"target":target},"evaluations":specs,
            "reused_final_evaluation":review.binding(root/"epoch200/evaluation.json")}

    def test_saved_draws_are_verified_against_each_evaluation_before_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);protocol=self.fixture(root);protected={}
            values,evaluations=review.load_dependence(protocol,protected)
            self.assertEqual(set(evaluations),set(review.EPOCHS))
            self.assertEqual(set(values),set(review.shared.CASES))
            self.assertEqual(len(values[review.shared.CASES[0]]),5)
            self.assertIn(str((root/"conditional.npz").resolve()),protected)
            path=root/"epoch83/evaluation.json";record=json.loads(path.read_text())
            record["cases"][review.shared.CASES[0]]["neural_quantiles_05_50_95"][1][5]+=.01
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,"reproduce evaluation quantiles"):
                review.load_dependence(protocol,{})


if __name__=="__main__":unittest.main()
