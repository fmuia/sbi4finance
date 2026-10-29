"""Approximate emulator likelihoods preserve physical priors and known curves."""
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sbi4finance.calibration import pricer_net, reference, tasks
from sbi4finance.gbm_bank import _json
from scripts import calibration_reference as runner

try:
    PYMC_AVAILABLE=metadata.version('pymc').startswith('6.')
except metadata.PackageNotFoundError:
    PYMC_AVAILABLE=False


def constant_pricer(task,value=2.):
    inputs=len(task['parameter_names'])+(tasks.context_size(task) if not tasks.is_joint_curve_task(task) else 0)
    return pricer_net.Pricer([(np.zeros((inputs,63)),np.full(63,value))],np.zeros(inputs),np.ones(inputs),
                             np.zeros(63),np.ones(63),activation='gelu_erf')


def bind_pricer(directory,task):
    constant_pricer(task).save(directory)
    (directory/'task.json').write_text(json.dumps(task))
    record={'task_fingerprint':hashlib.sha256(_json(task).encode()).hexdigest(),
            'pricer_sha256':hashlib.sha256((directory/'pricer.npz').read_bytes()).hexdigest()}
    (directory/'pricer.json').write_text(json.dumps(record))
    return record


class PricerLikelihoodTests(unittest.TestCase):
    def test_physical_normalization_units_and_no_prior_or_jacobian(self):
        for task in (tasks.rough_task(),tasks.joint_rough_task()):
            with self.subTest(task=task['task_id']):
                sigma=tasks.sigma(task);standardized_residual=np.linspace(-1,1,63)
                quotes=(2+standardized_residual)*sigma
                likelihood=reference.pricer_log_likelihood(task,quotes,constant_pricer(task),np.full(7,.04))
                expected=-.5*(standardized_residual@standardized_residual+np.log(2*np.pi*sigma**2).sum())
                bounds=np.asarray(task['prior_bounds']);center=bounds.mean(axis=1)
                self.assertAlmostEqual(likelihood(center),expected)
                # Constant predicted prices imply identical physical likelihood
                # at different points, although their logit Jacobians differ.
                other=bounds[:,0]+.2*(bounds[:,1]-bounds[:,0])
                self.assertAlmostEqual(likelihood(other),expected)
                for value in (bounds[:,0],bounds[:,1],bounds[:,0]-1):
                    self.assertEqual(likelihood(value),-np.inf)

    def test_known_curve_is_frozen_joint_curve_auxiliaries_are_ignored(self):
        known=tasks.rough_task();pricer=constant_pricer(known)
        pricer.weights[0][0][3,:]=.1
        curve=np.full(7,.04);quotes=np.zeros(63)
        fixed=reference.pricer_log_likelihood(known,quotes,pricer,curve)
        point=np.asarray(known['prior_bounds']).mean(axis=1);before=fixed(point)
        curve[:]=.09
        self.assertEqual(fixed(point),before)
        changed=reference.pricer_log_likelihood(known,quotes,pricer,curve)
        self.assertNotEqual(changed(point),before)
        with self.assertRaisesRegex(ValueError,'known curve'):
            reference.pricer_log_likelihood(known,quotes,pricer)
        joint=tasks.joint_rough_task();point=np.asarray(joint['prior_bounds']).mean(axis=1)
        first=reference.pricer_log_likelihood(joint,quotes,constant_pricer(joint),np.full(7,np.nan))
        second=reference.pricer_log_likelihood(joint,quotes,constant_pricer(joint))
        self.assertEqual(first(point),second(point))

    def test_invalid_input_or_emulator_failure_is_not_zero_density(self):
        task=tasks.joint_rough_task();point=np.asarray(task['prior_bounds']).mean(axis=1)
        for quotes in (np.zeros(62),np.full(63,np.nan),np.ones(63,dtype=complex),np.zeros((2,63))):
            with self.assertRaises(ValueError):reference.pricer_log_likelihood(task,quotes,constant_pricer(task))
        for bad in (np.zeros(5),np.full(6,np.nan),np.ones(6,dtype=complex)):
            likelihood=reference.pricer_log_likelihood(task,np.zeros(63),constant_pricer(task))
            with self.assertRaises(ValueError):likelihood(bad)
        calls=iter((np.zeros((1,63)),np.full((1,63),np.nan)))
        likelihood=reference.pricer_log_likelihood(task,np.zeros(63),lambda _:next(calls))
        with self.assertRaisesRegex(FloatingPointError,'pricer evaluation failed'):likelihood(point)
        with self.assertRaisesRegex(ValueError,'rough task'):
            reference.pricer_log_likelihood(tasks.heston_task(),np.zeros(63),constant_pricer(tasks.heston_task()))

    def test_bound_checkpoint_task_and_hash_are_required(self):
        task=tasks.rough_task()
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)/'pricer';record=bind_pricer(directory,task)
            pricer,identity=reference.load_forward_pricer(directory,task)
            self.assertEqual(identity['activation'],'gelu_erf');self.assertEqual(len(identity['input_sha256']),3)
            with self.assertRaisesRegex(ValueError,'binding differs'):
                reference.load_forward_pricer(directory,tasks.rough_task(level_bounds=(.15,.4)))
            record['pricer_sha256']='0'*64;(directory/'pricer.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'binding differs'):reference.load_forward_pricer(directory,task)

    @unittest.skipUnless(PYMC_AVAILABLE,'optional PyMC6 environment')
    def test_cli_labels_approximation_and_uses_data_fit_without_a_posterior_start(self):
        import xarray as xr
        for task in (tasks.rough_task(),tasks.joint_rough_task()):
            with self.subTest(task=task['task_id']),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);directory=root/'pricer';bind_pricer(directory,task)
                bank=root/'bank';bank.mkdir();manifest={'configuration':{'task':task,'root_seed':7,'splits':{'train':1,'validation':1,'test':1}},'source_sha256':{}}
                (bank/'manifest.json').write_text(json.dumps(manifest))
                quotes=(2*tasks.sigma(task))[None,:,None]
                data={'validation':{'observations':quotes,'example_id':np.array(['validation:0']),
                                    'context_values':np.full((1,7),.04 if not tasks.is_joint_curve_task(task) else np.nan)}}
                values=np.broadcast_to(np.asarray(task['prior_bounds']).mean(axis=1),(2,8,len(task['parameter_names']))).copy()
                tree=xr.DataTree.from_dict({'posterior':xr.Dataset({'parameters':(('chain','draw','parameter'),values)},coords={'parameter':task['parameter_names']}),
                                           'sample_stats':xr.Dataset({'accepted':(('chain','draw'),np.ones((2,8)))})})
                args=['reference','--bank',str(bank),'--pricer',str(directory),'--out',str(root/'out'),'--draws','8','--tune','8','--chains','2','--no-progress']
                initial={'kind':'data-fit','fixture':True}
                with (patch.object(sys,'argv',args),patch.object(runner.bank_collection,'load_bank',return_value=(manifest,data)) as loaded,
                      patch.object(reference,'data_fit_initialization',return_value=initial) as fitted,
                      patch.object(reference,'sample_deterministic_reference',return_value=tree) as sampled,
                      patch.object(reference,'chain_diagnostics',return_value={'mcmc_diagnostics_passed':True}),
                      patch.object(reference,'rough_derived_diagnostics',return_value={'mcmc_diagnostics_passed':True})):
                    runner.main()
                record=json.loads((root/'out/reference.json').read_text())
                self.assertEqual(record['likelihood_kind'],'forward-pricer-emulator')
                self.assertIn('approximate',record['target']);self.assertFalse(record['scientific_acceptance'])
                self.assertIn('not propagated',record['approximation_scope'])
                self.assertEqual(record['settings']['initialization'],'data-fit')
                self.assertEqual(sampled.call_args.kwargs['initialization'],initial);fitted.assert_called_once()
                expected_roles=['observations','example_id']+([] if tasks.is_joint_curve_task(task) else ['context_values'])
                self.assertEqual(loaded.call_args.kwargs['roles'],expected_roles)
                self.assertEqual(record['known_curve_levels'] is None,tasks.is_joint_curve_task(task))
                self.assertEqual(record['known_curve_sha256'] is None,tasks.is_joint_curve_task(task))
                self.assertIn('pricer_net.py',record['source_sha256'])
                with np.load(root/'out/chains.npz') as arrays:self.assertEqual(arrays['parameters'].shape,values.shape)

    @unittest.skipUnless(PYMC_AVAILABLE,'optional PyMC6 environment')
    def test_heston_cli_default_keeps_fourier_target_and_prior_starts(self):
        import xarray as xr
        task=tasks.heston_task()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);bank=root/'bank';bank.mkdir()
            manifest={'configuration':{'task':task,'root_seed':7,'splits':{'train':1,'validation':1,'test':1}},'source_sha256':{}}
            (bank/'manifest.json').write_text(json.dumps(manifest))
            data={'validation':{'observations':np.zeros((1,63,1)),'example_id':np.array(['validation:0'])}}
            values=np.broadcast_to(np.asarray(task['prior_bounds']).mean(axis=1),(2,8,4)).copy()
            tree=xr.DataTree.from_dict({'posterior':xr.Dataset({'parameters':(('chain','draw','parameter'),values)},coords={'parameter':task['parameter_names']}),
                                       'sample_stats':xr.Dataset({'accepted':(('chain','draw'),np.ones((2,8)))})})
            args=['reference','--bank',str(bank),'--out',str(root/'out'),'--draws','8','--tune','8','--chains','2','--no-progress']
            with (patch.object(sys,'argv',args),patch.object(runner.bank_collection,'load_bank',return_value=(manifest,data)) as loaded,
                  patch.object(reference,'heston_log_likelihood',return_value=lambda _:0.) as fourier,
                  patch.object(reference,'data_fit_initialization') as fitted,
                  patch.object(reference,'sample_deterministic_reference',return_value=tree),
                  patch.object(reference,'chain_diagnostics',return_value={'mcmc_diagnostics_passed':True}),
                  patch.object(reference,'variance_functional_diagnostics',return_value={'mcmc_diagnostics_passed':True})):
                runner.main()
            record=json.loads((root/'out/reference.json').read_text())
            self.assertEqual(record['target'],'independent Gaussian quote noise around deterministic refined Fourier prices')
            self.assertEqual(record['settings']['initialization'],'prior')
            self.assertNotIn('emulator',record);self.assertEqual(record['likelihood_kind'],'fourier')
            self.assertEqual(loaded.call_args.kwargs['roles'],['observations','example_id'])
            fitted.assert_not_called();fourier.assert_called_once()

    @unittest.skipUnless(PYMC_AVAILABLE,'optional PyMC6 environment')
    def test_tiny_real_pymc_emulator_sampling_and_partial_trace(self):
        task=tasks.rough_task();likelihood=reference.pricer_log_likelihood(task,2*tasks.sigma(task),constant_pricer(task),np.full(7,.04))
        initial=reference.data_fit_initialization(likelihood,task['prior_bounds'],2,19,fit_starts=2,fit_budget=120)
        with tempfile.TemporaryDirectory() as temporary:
            partial=runner.PartialDrawWriter(Path(temporary)/'partial',chains=2,tune=6,draws=6,every=3)
            data=reference.sample_deterministic_reference(likelihood,task['prior_bounds'],task['parameter_names'],
                    draws=6,tune=6,chains=2,cores=1,seed=19,progressbar=False,initialization=initial,callback=partial)
            actual=data['posterior']['parameters'].transpose('chain','draw','parameter').values
            self.assertEqual(actual.shape,(2,6,3));self.assertTrue(np.isfinite(actual).all())
            diagnostic=reference.rough_derived_diagnostics(data,task)
            self.assertIn('parameters[rho_eta]',diagnostic['parameters'])
            self.assertFalse(diagnostic['mcmc_diagnostics_passed'])
            for chain in range(2):
                with np.load(Path(temporary)/f'partial/chain-{chain:03d}.npz') as saved:
                    self.assertEqual(saved['parameters'].shape,(12,3));self.assertEqual(saved['tuning'].sum(),6)

if __name__=='__main__':unittest.main()
