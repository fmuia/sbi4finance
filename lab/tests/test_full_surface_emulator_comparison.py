"""Approximate reference comparisons preserve target identity and conditioning."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sbi4finance.calibration import pricer_net, reference, tasks
from sbi4finance.gbm_bank import _json

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/compare_full_surface_reference.py'
spec = importlib.util.spec_from_file_location('emulator_comparison_for_test', SCRIPT)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class EmulatorComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.checkpoint, self.reference, self.pricer = [self.root/n for n in ('checkpoint','reference','pricer')]
        self.checkpoint.mkdir(); self.reference.mkdir()
        for name in ('model.keras','task.json','training.json'):
            (self.checkpoint/name).write_text('mock posterior checkpoint')
        self._task(tasks.rough_task())

    def tearDown(self): self.temporary.cleanup()

    def _task(self, task):
        self.task = task
        fingerprint = hashlib.sha256(_json(task).encode()).hexdigest()
        inputs = len(task['parameter_names']) + (7 if task['task_id']=='rough_calibration_v1' else 0)
        if self.pricer.exists():
            for path in self.pricer.iterdir(): path.unlink()
            self.pricer.rmdir()
        pricer_net.Pricer([(np.zeros((inputs,63)),np.zeros(63))],np.zeros(inputs),np.ones(inputs),
            np.zeros(63),np.ones(63),activation='gelu_erf').save(self.pricer)
        (self.pricer/'task.json').write_text(json.dumps(task))
        (self.pricer/'pricer.json').write_text(json.dumps({'task_fingerprint':fingerprint,
            'pricer_sha256':comparison._digest(self.pricer/'pricer.npz')}))
        _, identity = reference.load_forward_pricer(self.pricer, task)
        self.training = {'task_fingerprint':fingerprint,'checkpoint_sha256':'posterior-sha',
            'network':{'family':'spline','encoder':'vector'},'run_contract':{'dataset':{'manifest_sha256':'bank-sha'}}}
        curve = np.linspace(.025,.05,7).tolist() if task['task_id']=='rough_calibration_v1' else None
        self.record = {'format':'full_surface_pymc_reference_v1','task':task,'task_fingerprint':fingerprint,
            'likelihood_kind':'forward-pricer-emulator', 'scientific_acceptance':False,
            'target':'independent Gaussian quote noise around deterministic forward-pricer emulator; approximate price target',
            'emulator':identity,'observed_prices':np.linspace(.01,.02,63).tolist(),'known_curve_levels':curve,
            'known_curve_sha256':None if curve is None else hashlib.sha256(_json(curve).encode()).hexdigest(),
            'surface_id':'validation:0','split':'validation','settings':{'chains':4,'draws':100}}
        self.record['observation_sha256'] = hashlib.sha256(_json(self.record['observed_prices']).encode()).hexdigest()
        def row(): return {'r_hat':1.001,'ess_bulk':600,'ess_tail':500,'mcse_mean':1e-5,'mcse_sd':2e-5}
        self.diagnostics = {'mcmc_diagnostics_passed':True,'parameters':{
            f'parameters[{name}]':row() for name in task['parameter_names']},
            'derived_quantities':{'mcmc_diagnostics_passed':True,'parameters':{'parameters[rho_eta]':row()}}}
        bounds=np.asarray(task['prior_bounds'])
        self.chains=np.random.default_rng(37).uniform(bounds[:,0],bounds[:,1],size=(4,100,len(bounds)))
        self._write()

    def _write(self):
        (self.reference/'reference.json').write_text(json.dumps(self.record))
        (self.reference/'diagnostics.json').write_text(json.dumps(self.diagnostics))
        np.savez(self.reference/'chains.npz',parameters=self.chains)

    def _compare(self, **kwargs):
        with patch.object(comparison.full_surface,'load_checkpoint',return_value=(self.task,object(),self.training)):
            return comparison.compare(self.checkpoint,self.reference,**kwargs)

    def test_known_curve_and_approximation_scope_reach_sampling(self):
        neural=self.chains.reshape(-1,3)
        with (patch.object(comparison.full_surface,'sample',return_value=neural[None]) as sample,
              patch.object(comparison,'_variance_product_comparison') as products):
            report,actual=self._compare(draws=400,seed=17)
        np.testing.assert_array_equal(sample.call_args.kwargs['context_values'],[self.record['known_curve_levels']])
        self.assertEqual(sample.call_args.kwargs['seed'],17);np.testing.assert_array_equal(actual,neural)
        self.assertEqual(report['status'],'completed_approximate_emulator_comparison')
        self.assertFalse(report['scientific_acceptance']);self.assertFalse(report['independent_pricing_assessed'])
        self.assertIn('Monte Carlo',report['approximation_scope']);self.assertIn('not be like-for-like',report['approximation_scope'])
        self.assertTrue(report['proposed_parameter_agreement']['all_checks_passed'])
        self.assertIn('not exact-model accuracy',report['proposed_parameter_agreement']['limitations'])
        derived=report['derived_parameter_comparison']
        self.assertEqual(derived['standardized_mean_error'],0);self.assertEqual(derived['width_ratio_90'],1)
        self.assertEqual(derived['reference_mcse_mean'],1e-5);self.assertFalse(derived['scientific_acceptance'])
        self.assertEqual(report['emulator']['input_sha256'],self.record['emulator']['input_sha256'])
        self.assertIsNone(report['variance_product_comparison']);products.assert_not_called()
        with np.load(self.reference/'chains.npz') as z:self.assertEqual(z['parameters'].shape,(4,100,3))

    def test_joint_sampling_has_no_auxiliary_curve_and_rejects_leakage(self):
        self._task(tasks.joint_rough_task());neural=self.chains.reshape(-1,6)
        with patch.object(comparison.full_surface,'sample',return_value=neural[None]) as sample:
            report,_=self._compare(draws=400)
        self.assertNotIn('context_values',sample.call_args.kwargs);self.assertIsNone(report['known_curve_levels'])
        self.record['known_curve_levels']=[.04]*7;self._write()
        with self.assertRaisesRegex(ValueError,'auxiliary curve'):self._compare()

    def test_parameter_and_derived_diagnostics_both_gate_sampling(self):
        original=copy.deepcopy(self.diagnostics)
        for mutate in (lambda d:d.__setitem__('mcmc_diagnostics_passed',False),
                       lambda d:d.__setitem__('parameter_mcmc_diagnostics_passed',False),
                       lambda d:d['parameters']['parameters[eta]'].__setitem__('ess_tail',20),
                       lambda d:d['derived_quantities'].__setitem__('mcmc_diagnostics_passed',False),
                       lambda d:d['derived_quantities']['parameters']['parameters[rho_eta]'].__setitem__('r_hat',1.1),
                       lambda d:d.pop('derived_quantities')):
            self.diagnostics=copy.deepcopy(original);mutate(self.diagnostics);self._write()
            with patch.object(comparison.full_surface,'sample') as sample:
                report,neural=self._compare()
            self.assertEqual(report['status'],'inconclusive_reference');self.assertIsNone(neural)
            self.assertIsNone(report['proposed_parameter_agreement']);self.assertIsNone(report['derived_parameter_comparison'])
            sample.assert_not_called()

    def test_task_kind_emulator_and_observation_bindings_are_required(self):
        original=copy.deepcopy(self.record)
        for mutate in (lambda r:r.__setitem__('task_fingerprint','wrong'),
                       lambda r:r.__setitem__('likelihood_kind','fourier'),
                       lambda r:r.__setitem__('target','exact numerical prices'),
                       lambda r:r['emulator']['input_sha256'].__setitem__(str(self.pricer/'pricer.npz'),'0'*64),
                       lambda r:r['emulator'].__setitem__('task_fingerprint','wrong'),
                       lambda r:r['observed_prices'].__setitem__(0,.9),
                       lambda r:r['known_curve_levels'].__setitem__(0,.09)):
            self.record=copy.deepcopy(original);mutate(self.record);self._write()
            with self.assertRaises(ValueError):self._compare()
        self.record=original;self._write();(self.pricer/'pricer.npz').write_bytes(b'changed fixed emulator')
        with self.assertRaises(ValueError):self._compare()

    def test_known_curve_must_be_one_positive_finite_seven_level_vector(self):
        for curve in (None,[],[.04]*6,[[.04]*7],[.04]*6+[float('nan')],[-.04]*7):
            self.record['known_curve_levels']=curve;self.record.pop('known_curve_sha256',None);self._write()
            with self.assertRaises(ValueError):self._compare()

    def test_heston_precision_is_rejected_for_rough_reference(self):
        with self.assertRaisesRegex(ValueError,'Heston precision'):
            self._compare(precision=self.root/'unused-heston-precision.json')

    def test_emulator_change_during_sampling_is_detected(self):
        def changed(*args,**kwargs):
            (self.pricer/'pricer.json').write_text('changed while sampling')
            return self.chains.reshape(1,-1,3)
        with patch.object(comparison.full_surface,'sample',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'emulator inputs changed'):self._compare(draws=400)

if __name__=='__main__':unittest.main()
