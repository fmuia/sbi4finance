"""Scientific guard checks for the fixed-c development comparisons."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_heston_spread_baseline as evaluator
from scripts.run_heston_spread_reference import product_values


class EvaluationTests(unittest.TestCase):
    def test_reference_binding_rejects_a_different_observation_or_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"chains.npz"
            np.savez(path, parameters=np.ones((4,32,5)))
            target = {"fixed_c":1., "infer_c":False}
            y = np.array([1.,-2.,3.])
            diagnostic_path=Path(directory)/"diagnostics.json"
            diagnostic_path.write_text('{"valid":"diagnostics"}')
            products={"control":[0.,.1]}
            record = {"task":target,"design_sha256":"design","case_id":"case",
                      "observations":y.tolist(),"observation_sha256":evaluator.observations_digest(y),
                      "panel_sha256":"panel","products":products,"selected_attempt":0,
                      "attempts":[{"diagnostics_sha256":evaluator.digest(diagnostic_path)}]}
            diagnostics = {"chain_payload_sha256":evaluator.digest(path)}
            kwargs = dict(target=target,design_sha256="design",case_id="case",observations=y,
                          panel_sha256="panel",products=products,diagnostics_path=diagnostic_path)
            evaluator.check_reference_binding(record,diagnostics,path,**kwargs)
            with self.assertRaisesRegex(ValueError,"differs"):
                evaluator.check_reference_binding(record,diagnostics,path,**{**kwargs,"observations":y+1})
            changed = copy.deepcopy(record); changed["task"]={"fixed_c":2.,"infer_c":False}
            with self.assertRaisesRegex(ValueError,"differs"):
                evaluator.check_reference_binding(changed,diagnostics,path,**kwargs)
            with self.assertRaisesRegex(ValueError,"differs"):
                evaluator.check_reference_binding(record,diagnostics,path,**{**kwargs,"products":{"control":[0.,.2]}})
            with self.assertRaisesRegex(ValueError,"differs"):
                evaluator.check_reference_binding(record,diagnostics,path,**{**kwargs,"panel_sha256":"different"})
            with path.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError,"differs"):
                evaluator.check_reference_binding(record,diagnostics,path,**kwargs)

    def test_support_reports_rounding_without_clipping_material_values(self):
        bounds=np.array([[.01,.16],[.1,10.]])
        draws=np.array([[np.float32(.16),1.],[.17,2.]])
        saved=draws.copy()
        result=evaluator.support_report(draws,bounds)
        self.assertEqual(result["material_outside_value_count"],1)
        self.assertFalse(result["draws_clipped"])
        np.testing.assert_array_equal(draws,saved)
        with self.assertRaises(ValueError):
            evaluator.support_report(np.array([[np.nan,1.]]),bounds)

    def test_validation_arrays_reject_reshaping_nonfinite_or_wrong_physical_truths(self):
        bounds=np.array([[.01,.16],[.1,10.]])
        truth=np.array([[.05,1.],[.06,2.]])
        clean=np.ones((2,3))
        evaluator.validate_development_arrays(truth,clean,bounds,2,3)
        for bad in (truth[:1], np.array([[np.nan,1.],[.06,2.]]), np.array([[.2,1.],[.06,2.]])):
            with self.assertRaises(ValueError):
                evaluator.validate_development_arrays(bad,clean,bounds,2,3)

    def test_biased_neural_posterior_fails_and_unconverged_reference_cannot_pass(self):
        rng=np.random.default_rng(704)
        center=np.array([.06,2.,.04,.7,-.7])
        spread=np.array([.001,.02,.001,.01,.005])
        chains=center+spread*rng.standard_normal((4,512,5))
        neural=center+spread*rng.standard_normal((4096,5))
        names=["initial_variance","kappa","theta","vol_of_vol","rho"]
        products={"near_variance":[0.,.06],"forward_variance":[.22,.89]}
        all_values=np.concatenate([chains,product_values(chains,products)],axis=-1)
        all_names=[*names,*products]
        entries={name:{"ess_bulk":2048.,"ess_tail":2048.,
                       "mcse_mean":float(all_values[...,i].std()/np.sqrt(2048))}
                 for i,name in enumerate(all_names)}
        diagnostic={"parameters":{name:entries[name] for name in names},
                    "products":{"parameters":{name:entries[name] for name in products}},
                    "mcmc_diagnostics_passed":True}
        biased=neural.copy(); biased[:,0]+=.02
        result=evaluator.compare_draws(biased,chains,diagnostic,names,products,seed=21,replicates=100)
        self.assertEqual(result["status"],"fail")
        self.assertEqual(result["parameters"]["initial_variance"]["mean_status"],"fail")
        diagnostic["mcmc_diagnostics_passed"]=False
        result=evaluator.compare_draws(neural,chains,diagnostic,names,products,seed=21,replicates=100)
        self.assertEqual(result["status"],"inconclusive")


if __name__=="__main__":
    unittest.main()
