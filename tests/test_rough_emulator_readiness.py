"""Arithmetic checks for scientific diagnostics, without a training run."""
import unittest
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
from scripts import check_rough_emulator_readiness as diagnostic
from sbi4finance.calibration import correction, pricer_net, tasks
from sbi4finance.gbm_bank import _json


class EmulatorGradientTests(unittest.TestCase):
    def test_separate_consumer_does_not_replace_numerical_package_and_rejects_changed_binding(self):
        consumer=diagnostic.load_consumer(pricer_net.__file__)
        self.assertIsNot(consumer,pricer_net)
        self.assertIs(sys.modules['sbi4finance.calibration.pricer_net'],pricer_net)
        task=tasks.joint_rough_task()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'pricer'
            model=consumer.Pricer([(np.zeros((6,63)),np.ones(63))],np.zeros(6),np.ones(6),np.zeros(63),np.ones(63),activation='gelu_erf')
            model.save(root);(root/'task.json').write_text(json.dumps(task))
            record={'task_fingerprint':hashlib.sha256(_json(task).encode()).hexdigest(),'pricer_sha256':diagnostic.sha(root/'pricer.npz')}
            (root/'pricer.json').write_text(json.dumps(record))
            loaded,_=diagnostic.load_bound_pricer(root,task,consumer=consumer)
            self.assertEqual(loaded.activation,'gelu_erf')
            record['pricer_sha256']='0'*64;(root/'pricer.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'binding differs'):
                diagnostic.load_bound_pricer(root,task,consumer=consumer)

    def test_chain_rule_matches_both_saved_activation_definitions(self):
        rng = np.random.default_rng(17)
        for activation in ('gelu_erf', 'gelu_tanh'):
            pricer = pricer_net.Pricer([(rng.normal(size=(3,5)),rng.normal(size=5)),
                (rng.normal(size=(5,4)),rng.normal(size=4))],np.array([.1,.2,.3]),
                np.array([.7,1.1,1.7]),np.arange(4),np.array([.4,.8,1.2,1.6]),activation=activation)
            x = np.array([.3,-.7,1.2]); value,jac = diagnostic.value_and_jacobian(pricer,x)
            np.testing.assert_allclose(value,pricer(x)[0],rtol=1e-13,atol=1e-13)
            for i in range(3):
                delta = np.eye(3)[i]*1e-5
                numerical = (pricer(x+delta)[0]-pricer(x-delta)[0])/2e-5
                np.testing.assert_allclose(jac[i],numerical,rtol=1e-7,atol=1e-8)

    def test_physical_uniform_prior_keeps_logit_jacobian_gradient(self):
        task=tasks.joint_rough_task(); bounds=np.asarray(task['prior_bounds'])
        pricer=pricer_net.Pricer([(np.zeros((6,63)),np.ones(63))],np.zeros(6),np.ones(6),np.zeros(63),np.ones(63))
        theta=bounds[:,0]+.2*(bounds[:,1]-bounds[:,0])
        point={'case':0,'role':'truth','theta':theta.tolist(),'observed_prices':tasks.sigma(task).tolist(),
               'curve_levels':[float('nan')]*7}
        z=correction.to_unbounded(theta[None],bounds)[0]
        _,gradient,_=diagnostic.target_and_gradient(pricer,task,point,z)
        np.testing.assert_allclose(gradient,np.full(6,.6),rtol=0,atol=1e-14)
        report=diagnostic.check_gradient(pricer,task,point)
        self.assertTrue(report['finite_difference_passed']);self.assertTrue(report['value_parity_passed'])

    def test_reused_batches_keep_correlated_price_covariance_and_paired_grid_shift(self):
        task=tasks.rough_task();bounds=np.asarray(task['prior_bounds']);theta=bounds.mean(axis=1)
        pricer=pricer_net.Pricer([(np.zeros((10,63)),np.zeros(63))],np.zeros(10),np.ones(10),np.zeros(63),np.ones(63))
        plan={'task':task,'points':[{'case':0,'role':'truth','theta':theta.tolist(),
                                   'curve_levels':[.04]*7,'emulator_scaled_prices':[0]*63}]}
        noise=np.arange(8,dtype=float)-3.5
        batches=np.empty((1,8,2,63));sigma=tasks.sigma(task)
        batches[0,:,0]=noise[:,None]*sigma
        batches[0,:,1]=(noise[:,None]+.25)*sigma
        result=diagnostic.rescore_batches(pricer,plan,batches)[0]
        covariance=np.asarray(result['replicate_mean_price_covariance_quote_units_by_grid'][0])
        self.assertGreater(covariance[0,1],0)
        np.testing.assert_allclose(result['fine_minus_coarse_quote_sigma']['mean'],.25,atol=1e-14)
        np.testing.assert_allclose(result['fine_minus_coarse_quote_sigma']['mcse'],0,atol=1e-14)


if __name__=='__main__':unittest.main()
