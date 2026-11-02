"""Heston lesson calculations preserve the numerical law, paired rows and noise."""
import copy
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from sbi4finance import heston
from sbi4finance.calibration import teaching

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parent


class HestonLessonHelperTests(unittest.TestCase):
    def setUp(self):
        self.task = json.loads((REPO / 'tutorial/assets/backbones/heston-final500/task.json').read_text())
        bounds = np.asarray(self.task['prior_bounds'])
        self.rows = bounds[:, 0] + np.random.default_rng(9).uniform(.2, .8, (20, 4)) * np.diff(bounds).ravel()
        self.observed = np.linspace(-.001, .08, 63)

    def test_illustrative_paths_match_vetted_signed_simulator_with_prescribed_noise(self):
        row = np.array([.01, .01, 1.4, -.9])
        for substeps in (3, 16):
            found = teaching.heston_illustrative_paths(self.task, row, paths=3, days=18, substeps=substeps, seed=3)
            normals = np.random.default_rng(3).standard_normal((3, 18, substeps, 2))
            expected = heston.simulate({'drift':0., 'kappa':self.task['fixed_parameters']['kappa'],
                'theta':row[1], 'vol_of_vol':row[2], 'rho':row[3]}, np.full(18, 1/252), row[0],
                {'brownian':normals, 'jump_counts':np.zeros((3,18),dtype=np.int64), 'jump_normals':np.zeros((3,18))}, backend='numpy')
            assert_array_equal(found['log_returns'], expected['log_returns'][...,0])
            assert_array_equal(found['variance_states'], expected['variance_state'][...,0])
            assert_array_equal(found['variances'], np.maximum(expected['variance_state'][...,0],0))
            assert_allclose(np.diff(np.log(found['prices']),axis=1),found['log_returns'],atol=3e-16,rtol=1e-13)

    def test_path_initial_values_reproducibility_and_explicit_scope(self):
        a=teaching.heston_illustrative_paths(self.task)
        b=teaching.heston_illustrative_paths(self.task)
        assert_array_equal(a['prices'],b['prices'])
        assert_array_equal(a['prices'][:,0],np.ones(3))
        assert_array_equal(a['variances'][:,0],np.full(3,.04))
        self.assertEqual(a['prices'].shape,(3,127))
        self.assertEqual(a['times_years'][-1],.5)
        self.assertIn('not used for Fourier',a['scope'])
        self.assertGreater(2*a['kappa']*a['parameters'][1],a['parameters'][2]**2)

    def test_path_size_seed_and_parameter_guards(self):
        for options in ({'paths':True},{'paths':9},{'days':253},{'substeps':65},{'seed':-1},{'seed':True},{'parameters':[.04,.04,.25,0.]}, {'parameters':[.04]*3}):
            with self.subTest(options=options),self.assertRaises(ValueError):teaching.heston_illustrative_paths(self.task,**options)
        wrong=copy.deepcopy(self.task);wrong['measure']='P'
        with self.assertRaisesRegex(ValueError,'measure Q'):teaching.heston_illustrative_paths(wrong)

    def test_predictive_preserves_whole_rows_chain_indices_and_exact_observations(self):
        task_before=copy.deepcopy(self.task);rows_before=self.rows.copy();observed_before=self.observed.copy()
        chains=self.rows.reshape(2,10,4)
        with patch.object(teaching,'heston_smile',side_effect=lambda task,row:np.full(63,row[0]+2*row[1])):
            found=teaching.heston_posterior_predictive(self.task,{'neural':self.rows,'reference':chains},self.observed,truth=self.rows[3],draws_per_set=6)
        for label,original in [('neural',self.rows),('reference',chains)]:
            panel=found['draw_sets'][label];indices=np.linspace(0,19,6,dtype=int)
            assert_array_equal(panel['source_indices'],indices)
            assert_array_equal(panel['parameters'],original.reshape(-1,4)[indices])
            self.assertEqual(panel['source_shape'],original.shape)
            assert_array_equal(panel['mean_prices'][:,0],self.rows[indices,0]+2*self.rows[indices,1])
        assert_array_equal(found['observations'],self.observed)
        assert_array_equal(found['truth_prices'],np.full(63,self.rows[3,0]+2*self.rows[3,1]))
        self.assertEqual(self.task,task_before);assert_array_equal(self.rows,rows_before);assert_array_equal(self.observed,observed_before)
        self.assertFalse(np.shares_memory(found['observations'],self.observed))

    def test_predictive_noise_is_added_once_and_negative_quotes_are_retained(self):
        task=copy.deepcopy(self.task);task['context']['quote_noise_std']=[.1]*63
        with patch.object(teaching,'heston_smile',return_value=np.zeros(63)):
            found=teaching.heston_posterior_predictive(task,{'neural':self.rows},self.observed,draws_per_set=8,seed=12)
        panel=found['draw_sets']['neural'];z=np.random.default_rng(12).standard_normal((8,63))
        assert_array_equal(panel['standard_normal_noise'],z)
        assert_array_equal(panel['mean_prices'],np.zeros((8,63)))
        assert_array_equal(panel['replicated_quotes'],.1*z)
        self.assertTrue(np.any(panel['replicated_quotes']<0))
        assert_array_equal(found['observations'],self.observed)

    def test_truth_overlay_and_noise_seed_do_not_change_posterior_mean_prices(self):
        with patch.object(teaching,'heston_smile',side_effect=lambda task,row:np.full(63,row[0])):
            a=teaching.heston_posterior_predictive(self.task,{'neural':self.rows},self.observed,truth=self.rows[0],draws_per_set=4,seed=3)
            b=teaching.heston_posterior_predictive(self.task,{'neural':self.rows},self.observed,truth=self.rows[1],draws_per_set=4,seed=3)
            c=teaching.heston_posterior_predictive(self.task,{'neural':self.rows},self.observed,draws_per_set=4,seed=4)
        for name in ['parameters','mean_prices','standard_normal_noise','replicated_quotes']:
            assert_array_equal(a['draw_sets']['neural'][name],b['draw_sets']['neural'][name])
        assert_array_equal(a['draw_sets']['neural']['mean_prices'],c['draw_sets']['neural']['mean_prices'])
        self.assertFalse(np.array_equal(a['draw_sets']['neural']['replicated_quotes'],c['draw_sets']['neural']['replicated_quotes']))
        self.assertIsNone(c['truth_prices'])

    def test_predictive_limits_and_invalid_inputs_fail_before_pricing(self):
        cases=[({'a':self.rows},self.observed,{'draws_per_set':65}),
               ({'a':self.rows},self.observed[:62],{}),
               ({'a':np.full((2,4),np.nan)},self.observed,{}),
               ({'a':self.rows},self.observed,{'truth':[1.]*4}),
               ({'a':np.tile(self.rows,(4,1)),'b':np.tile(self.rows,(4,1)),'c':np.tile(self.rows,(4,1))},self.observed,{'draws_per_set':64})]
        with patch.object(teaching,'heston_smile') as price:
            for sets,observations,options in cases:
                with self.subTest(options=options),self.assertRaises(ValueError):teaching.heston_posterior_predictive(self.task,sets,observations,**options)
            price.assert_not_called()

    def test_current_synthetic_case_uses_the_existing_fourier_pricer(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('full_surface_teaching',LAB/'src/sbi4finance/calibration/full_surface_teaching.py')
        replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)
        task,_,arrays=replay.load_tutorial_artifact(REPO/'tutorial/assets/backbones/artifacts.json','heston-final500')
        found=teaching.heston_posterior_predictive(task,{'neural':arrays['neural_draws'],'PyMC':arrays['reference_chains']},arrays['observations'],truth=arrays['truth'],draws_per_set=2)
        for panel in found['draw_sets'].values():
            for row,mean in zip(panel['parameters'],panel['mean_prices']):assert_array_equal(mean,teaching.heston_smile(task,row))
            assert_array_equal(panel['replicated_quotes'],panel['mean_prices']+found['quote_noise_std']*panel['standard_normal_noise'])
        assert_array_equal(found['truth_prices'],teaching.heston_smile(task,arrays['truth']))

    def test_participant_helpers_work_without_lab_or_neural_backend_imports(self):
        program='''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'sbi4finance','bayesflow','keras','jax','tensorflow','torch'}:raise AssertionError('Unwanted import:'+name)
sys.meta_path.insert(0,Block())
from sbi4f_tutorial import backbones,teaching
task,report,arrays=backbones.load_example('heston-final500')
teaching.heston_illustrative_paths(task,paths=1,days=2)
teaching.heston_posterior_predictive(task,{'neural':arrays['neural_draws'],'PyMC':arrays['reference_chains']},arrays['observations'],truth=arrays['truth'],draws_per_set=2)
'''
        env={**os.environ,'PYTHONPATH':str(REPO/'tutorial/src')}
        process=subprocess.run([sys.executable,'-c',program],env=env,capture_output=True,text=True,timeout=20)
        self.assertEqual(process.returncode,0,process.stdout+process.stderr)

if __name__=='__main__':unittest.main()
