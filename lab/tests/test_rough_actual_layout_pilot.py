"""Pure preparation/gate tests; all numerical pricing entry points are guarded."""
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from sbi4finance.calibration import rough_actual_layout_pilot as pilot
from sbi4f_tutorial.rough_options import actual_layout as layout, simulator
from sbi4f_tutorial.shared import pricing

ROOT = Path(__file__).resolve().parents[2]
TIMES = [.019266330542871638,.03844441273465246,.08228002917300863,.16447180999492644,.2220060565702689,.47120696981227805,.8903850520040588]


def fixture(root):
    records = {"forward": [100.]*63, "discount": [.99]*63, "strike": np.tile(np.linspace(70.,120.,9),7).tolist(),
        "maturity": np.repeat(TIMES,9).tolist(), "option_type": np.tile(["put"]*6+["call"]*3,7).tolist(), "mid": [1.]*63,"half_spread": [.05]*63}
    design = root/"design.json"; pilot.write_new(design,{"records":records})
    heston = root/"heston.json"; pilot.write_new(heston,{"format":"heston_pr87_grid_task_v1","design_sha256":pilot.digest(design),"target":{"quote_count":63,"pricing_controls":{"fourier_representation":"black_control_variate_v1"}}})
    manifests=[]
    for name in ("rough-known100k-validation0","rough-joint100k-validation0"):
        base=json.loads((ROOT/"tutorial/assets/backbones"/name/"task.json").read_text())
        path=root/(name+".json");pilot.write_new(path,{"configuration":{"task":base},"task_fingerprint":pilot.bank.task_fingerprint({"task":base})})
        manifests.append(path)
    out=root/"prepared";work=pilot.prepare(design,heston,*manifests,out)
    return work,out/"work-order.json"


def completed_evaluation(target):
    support={"samples":4096,"outside_prior":{"sample_count":0},"nonphysical":{"sample_count":0},"nonfinite":{"value_count":0}}
    panel={"calibration_status":"defined","support":deepcopy(support),"coverage":{"0.9":{"wilson":[[.85,.95]]*6}},"rank_dkw":{"passed":[True]*6}}
    native=deepcopy(panel);native.update(rows=100,draws_per_case=300)
    population=deepcopy(panel);population.update(rows=512,draws_per_case=512)
    return {"format":"heston_pr87_grid_evaluation_v1","status":"complete","endpoint":"primary_final","native_epoch":300,
        "task":target,"protected_inputs_unchanged":True,"native_validation":native,"population":population,
        "cases":{name:{"reference_ready":True,"comparison":{"status":"defined","standardized_mean_error":[.1]*6,"width_ratio":[1.]*6},"support":deepcopy(support)}
            for name in ("synthetic-broad","synthetic-market-like","market")}}


def ready_condition(root,work,path):
    evaluation=root/"evaluation.json";pilot.write_new(evaluation,completed_evaluation(work["heston_target"]))
    condition=root/"condition.json";pilot.bind_heston(path,evaluation,condition)
    return condition


def fake_block(parameters, curve, contracts, **kwargs):
    n=len(contracts);eye=np.eye(n)*1e-14
    return {"controlled_prices":np.full((2,n),.03),"raw_prices":np.full((2,n),.031),
        "joint_covariance":(np.eye(4*n)*1e-14).reshape(4,n,4,n),"difference":np.zeros(n),"difference_covariance":eye,
        "moments":[{"bank_covariance":eye.tolist()} for _ in range(2)]}


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.guards=ExitStack();self.guards.enter_context(patch.object(pricing,"rough_block",side_effect=AssertionError("no MC")))
        self.guards.enter_context(patch.object(simulator,"simulate_rough",side_effect=AssertionError("no MC")));self.addCleanup(self.guards.close)

    def test_preparation_has_no_MC_and_stays_inside_existing_ceiling(self):
        with tempfile.TemporaryDirectory() as temporary:
            work,path=fixture(Path(temporary));pilot.validate(work)
            self.assertFalse(work["MC_executed"]);self.assertEqual(len(work["cases"]),16);self.assertEqual(len(work["components"]),48)
            self.assertEqual(work["grid_estimator_evaluations"],80)
            self.assertEqual(work["limits"]["allocated_cpu_seconds"],48*600)
            script=path.with_name("pilot.sbatch").read_text()
            self.assertIn("--array=0-47%2",script);self.assertIn("--time=00:10:00",script);self.assertNotIn("--gres=gpu",script)

    def test_replay_seeds_are_explicit_and_batches_are_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            work,_=fixture(Path(temporary));by_role={}
            for component in work["components"]:by_role[component["case_index"],component["replicate"],component["role"]]=component
            for case in work["initial_case_indices"]:
                seen=set()
                for rep in range(4):
                    pair=by_role[case,rep,"coupled_2_4"];replay=by_role[case,rep,"finest_replay"]
                    self.assertEqual(pair["streams"],replay["streams"])
                    seed=pair["streams"][0]["seed"];self.assertNotIn(seed,seen);seen.add(seed)

    def test_sources_priors_and_original_known_knots_are_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            work,_=fixture(Path(temporary));known=work["tasks"]["supplied_known"]
            self.assertEqual(known["base_task"]["context_prior"]["forward_variance"]["log_perturbation"],.2)
            self.assertEqual(known["base_task"]["model_context"]["forward_variance_knots"][-1],1.)
            changed=deepcopy(work);changed["source_bindings"]["actual_layout"]["sha256"]="0"*64
            with self.assertRaisesRegex(ValueError,"source differs"):pilot.validate(changed)
            changed=deepcopy(work);changed["limits"]["automatic_path_escalation"]=True
            with self.assertRaises(ValueError):pilot.validate(changed)
            changed=deepcopy(work);changed["components"][0]["paths"]*=2
            with self.assertRaisesRegex(ValueError,"recipe changed"):pilot.validate(changed)

    def test_heston_condition_rejects_incomplete_and_secondary_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);work,path=fixture(root);evaluation=root/"evaluation.json"
            for status,endpoint in (("started","primary_final"),("complete","secondary_best_validation")):
                evaluation.write_text(json.dumps({"format":"heston_pr87_grid_evaluation_v1","status":status,"endpoint":endpoint,
                    "native_epoch":300,"task":work["heston_target"],"protected_inputs_unchanged":True}))
                with self.assertRaisesRegex(ValueError,"not yet recorded"):pilot.bind_heston(path,evaluation,root/"condition.json")
            self.assertFalse((root/"condition.json").exists())

    def test_worker_refuses_unallocated_or_ungated_execution_before_MC(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);work,path=fixture(root)
            with patch.dict(os.environ,{},clear=True):
                with self.assertRaisesRegex(ValueError,"allocated Slurm"):pilot.run_component(path,root/"missing",0,root/"result",process_started=0.)
            condition=root/"condition.json";pilot.write_new(condition,{"format":"rough_actual_layout_heston_condition_v1","working_grid_condition_met":False})
            with patch.dict(os.environ,{"SLURM_JOB_ID":"unit-only","SLURM_CPUS_PER_TASK":"1"}):
                with self.assertRaisesRegex(ValueError,"matching Heston"):pilot.run_component(path,condition,0,root/"result",process_started=0.)
            self.assertFalse((root/"result").exists())

    def test_curve_construction_is_only_prior_evaluation_no_kernel(self):
        with tempfile.TemporaryDirectory() as temporary:
            work,_=fixture(Path(temporary))
            for index in (0,12):
                curve=pilot.curve_for_case(work,work["cases"][index])
                self.assertEqual(curve["times"][0],0.);self.assertTrue(np.all(np.asarray(curve["values"])>0))


    def test_failed_Heston_diagnostics_cannot_bind_and_worker_rechecks_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);work,path=fixture(root);evaluation=root/"evaluation.json"
            record=completed_evaluation(work["heston_target"]);record["population"]["rank_dkw"]["passed"][0]=False
            pilot.write_new(evaluation,record)
            with self.assertRaisesRegex(ValueError,"diagnostics fail"):pilot.bind_heston(path,evaluation,root/"condition.json")
            record=completed_evaluation(work["heston_target"]);evaluation.write_text(json.dumps(record))
            condition=root/"condition.json";pilot.bind_heston(path,evaluation,condition)
            record["cases"]["market"]["reference_ready"]=False;evaluation.write_text(json.dumps(record))
            with patch.dict(os.environ,{"SLURM_JOB_ID":"unit-only","SLURM_CPUS_PER_TASK":"1"}):
                with self.assertRaisesRegex(ValueError,"evaluation changed"):
                    pilot.run_component(path,condition,0,root/"result",process_started=time.monotonic())
            self.assertFalse((root/"result").exists())

    def test_mocked_worker_retains_blocks_covariance_and_DF_axes_without_MC(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);work,path=fixture(root);condition=ready_condition(root,work,path);out=root/"result"
            with patch.dict(os.environ,{"SLURM_JOB_ID":"unit-only","SLURM_CPUS_PER_TASK":"1"}),patch.object(pilot,"load_controlled_helper",return_value=SimpleNamespace(coupled_controlled_block=fake_block)):
                record=pilot.run_component(path,condition,0,out,process_started=time.monotonic())
            self.assertEqual(record["status"],"complete");self.assertEqual(record["blocks_completed"],7)
            self.assertEqual(len(record["block_payloads"]),7);self.assertFalse(record["numerical_target_established"])
            with np.load(out/"results.npz",allow_pickle=False) as arrays:
                self.assertEqual(arrays["joint_covariance_forward"].shape,(4,63,4,63))
                self.assertEqual(arrays["bank_covariance_nominal_forward"].shape,(2,63,63))
                np.testing.assert_allclose(arrays["controlled_raw_units"],99*arrays["controlled_forward"])
                np.testing.assert_allclose(arrays["bank_covariance_nominal_raw_units"],99**2*arrays["bank_covariance_nominal_forward"])
            self.assertLess(work["static_memory_audit"]["conservative_array_allowance_bytes"],24*1024**3)
            self.assertEqual(work["static_memory_audit"]["pair_chunk"],2048)

    def test_mocked_failure_preserves_completed_blocks_and_no_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);work,path=fixture(root);condition=ready_condition(root,work,path);out=root/"result";calls=[]
            def capped(*args,**kwargs):
                calls.append(1)
                if len(calls)>2:raise TimeoutError("mock cap; no numerical kernel")
                return fake_block(*args,**kwargs)
            with patch.dict(os.environ,{"SLURM_JOB_ID":"unit-only","SLURM_CPUS_PER_TASK":"1"}),patch.object(pilot,"load_controlled_helper",return_value=SimpleNamespace(coupled_controlled_block=capped)):
                with self.assertRaises(TimeoutError):pilot.run_component(path,condition,0,out,process_started=time.monotonic())
                with self.assertRaises(FileExistsError):pilot.run_component(path,condition,0,out,process_started=time.monotonic())
            record=pilot.read(out/"failure.json");self.assertEqual(record["blocks_completed"],2)
            self.assertEqual(len(record["block_payloads"]),2);self.assertTrue((out/"block-01.npz").is_file());self.assertFalse((out/"results.npz").exists())

    def test_mocked_negative_variance_cannot_complete_or_pass_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);work,path=fixture(root);condition=ready_condition(root,work,path);out=root/"result"
            def corrupt(*args,**kwargs):
                result=fake_block(*args,**kwargs);result["moments"][0]["bank_covariance"][0][0]=-1e-14;return result
            with patch.dict(os.environ,{"SLURM_JOB_ID":"unit-only","SLURM_CPUS_PER_TASK":"1"}),patch.object(pilot,"load_controlled_helper",return_value=SimpleNamespace(coupled_controlled_block=corrupt)):
                with self.assertRaisesRegex(ValueError,"negative covariance"):
                    pilot.run_component(path,condition,0,out,process_started=time.monotonic())
            record=pilot.read(out/"failure.json");self.assertEqual(record["status"],"failed");self.assertNotIn("strict_mc_target_met",record)
            self.assertFalse((out/"results.npz").exists())

    def test_portable_staged_evaluation_requires_identical_bytes_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);work,path=fixture(root);condition_path=ready_condition(root,work,path)
            condition=pilot.read(condition_path);staged=root/"staged.json";staged.write_bytes((root/"evaluation.json").read_bytes())
            (root/"evaluation.json").unlink()
            pilot.validate_condition(work,path,condition,staged)
            staged.write_text(staged.read_text()+" ")
            with self.assertRaisesRegex(ValueError,"evaluation changed"):pilot.validate_condition(work,path,condition,staged)

    def test_mocked_finest_replay_keeps_full_covariance_without_MC(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);work,path=fixture(root);condition=ready_condition(root,work,path);out=root/"result"
            def replay(parameters,curve,contracts,**kwargs):
                n=len(contracts);return np.full(n,.03),np.full(n,1e-7),np.eye(n)*1e-14
            with patch.dict(os.environ,{"SLURM_JOB_ID":"unit-only","SLURM_CPUS_PER_TASK":"1"}),patch.object(pilot,"load_controlled_helper",return_value=SimpleNamespace()),patch.object(pricing,"rough_block",side_effect=replay):
                record=pilot.run_component(path,condition,2,out,process_started=time.monotonic())
            self.assertEqual(record["component"]["role"],"finest_replay")
            with np.load(out/"results.npz",allow_pickle=False) as arrays:
                self.assertEqual(arrays["controlled_forward"].shape,(1,63))
                self.assertEqual(arrays["bank_covariance_nominal_forward"].shape,(1,63,63))

    def test_covariance_nonfinite_or_asymmetry_is_rejected(self):
        for matrix in (np.array([[1.,np.nan],[0.,1.]]),np.array([[1.,.2],[.1,1.]])):
            with self.assertRaises(ValueError):pilot.checked_covariance(matrix,2)


if __name__=="__main__":unittest.main()
