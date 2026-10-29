"""Same-path optional diagnostics and frozen independent-batch checks."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sbi4finance.calibration import pricing, tasks
from sbi4finance.calibration import rough_working_grid_pilot as base
from sbi4finance.calibration import rough_control_diagnostic as diagnostic


class OptionalControlDiagnosticsTests(unittest.TestCase):
    def test_enabled_disabled_prices_errors_and_covariance_are_bit_identical(self):
        task=tasks.joint_rough_task(days=(21,63),deltas=(-1,0,1),
                                  blocks=[{"horizon":.25,"n_steps":24}])
        for values in ([.1,1.5,-.7,.04,.09,2.], [.025,3.4,-.95,.04,.09,7.5], [.1,0.,0.,.04,.09,2.]):
            parameters=tasks.model_parameters(task,values);curve=tasks.joint_curve(task,values)
            settings=dict(horizon=.25,n_steps=24,n_pairs=128,seed=17)
            before=pricing.rough_block(parameters,curve,task['context']['contracts'],**settings)
            details={"capture_raw_control":True}
            after=pricing.rough_block(parameters,curve,task['context']['contracts'],diagnostics=details,**settings)
            for a,b in zip(before,after,strict=True):np.testing.assert_array_equal(a,b)
            raw=details['raw_control']
            self.assertTrue(np.all(raw['raw_put_mean']>=0))
            self.assertTrue(np.all(raw['raw_put_mean']<=raw['raw_put_upper_bound']))
            np.testing.assert_allclose(raw['raw_put_mean']-raw['control_mean']@raw['fitted_coefficients'],
                                       raw['controlled_put_mean'],rtol=1e-12,atol=1e-14)
            np.testing.assert_allclose(np.diag(raw['raw_put_covariance_of_mean']),details['plain_variance']/128,
                                       rtol=1e-12,atol=1e-20)


class DiagnosticDeclarationTests(unittest.TestCase):
    def test_exact_reviewed_52_calls_and_historical_actual_state_disjointness(self):
        reference=json.loads(diagnostic.REFERENCE.read_text())
        declaration=diagnostic.declare(reference,source_revision='test')
        self.assertEqual(declaration['root_seed'],271101)
        self.assertEqual(len(declaration['calls']),52)
        self.assertEqual(declaration['group_allocation_seconds']+declaration['declaration_allocation_seconds'],6300)
        self.assertEqual(declaration['allocated_cpu_ceiling_seconds'],7200)
        self.assertEqual(declaration['seed_audit']['new_pricing_streams'],156)
        self.assertEqual(declaration['seed_audit']['previous_initialized_pcg64_states'],724)
        self.assertTrue(all(c['total_paths']==32768 for c in declaration['calls']))
        for group,case in enumerate((0,4,6)):
            rows=[c for c in declaration['calls'] if c['group_index']==group]
            self.assertEqual(len(rows),16)
            expected=next(c['parameters'] for c in reference['components'] if c['mode']=='fixed_anchor' and c['point_id']==case)
            self.assertTrue(all(c['parameters']==expected and c['policy']=='working' for c in rows))
        timing=declaration['calls'][48:]
        self.assertEqual([c['policy'] for c in timing],['working','refined','refined','working'])
        self.assertTrue(all(not c['capture_raw_control'] for c in timing))
        self.assertTrue(all(c['parameters']==reference['prior_points'][6]['parameters'] for c in timing))
        diagnostic.validate(declaration,execution=True)
        changed=copy.deepcopy(declaration);changed['calls'][0]['pricing_seed']+=1
        with self.assertRaisesRegex(ValueError,'digest'):diagnostic.validate(changed)

    def test_collision_against_historical_initialized_states_fails(self):
        reference=json.loads(diagnostic.REFERENCE.read_text())
        original=base._stream;old_seeds={c['pricing_seed'] for c in reference['components']}
        old_state=reference['prior_points'][0]['pcg64_initial_state_sha256']
        def collide(seed,*args):
            result=original(seed,*args)
            if seed not in old_seeds:result['pcg64_initial_state_sha256']=old_state
            return result
        with patch.object(base,'_stream',side_effect=collide),self.assertRaisesRegex(ValueError,'collision'):
            diagnostic.declare(reference,source_revision='test')


class DiagnosticExecutionTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(diagnostic,'PATHS',64))
        self.temp=self.enterContext(tempfile.TemporaryDirectory());self.root=Path(self.temp)
        task=tasks.joint_rough_task(days=(21,63,252),deltas=(-1,0,1),
            blocks=[{"horizon":21/252,"n_steps":2},{"horizon":.25,"n_steps":6},{"horizon":1.,"n_steps":24}])
        reference=base.declare(task,root_seed=17,source_revision='tiny-reference')
        self.declaration=diagnostic.declare(reference,root_seed=18,source_revision='tiny')

    def test_all_52_real_tiny_calls_reduction_and_original_fixed_estimator(self):
        reports=[diagnostic.run_group(self.declaration,g,self.root/f'group-{g}') for g in range(4)]
        self.assertTrue(all(r['status']=='completed' for r in reports))
        self.assertEqual(sum(len(r['runs']) for r in reports),52)
        result=diagnostic.reduce_groups(self.declaration,list(reversed(reports)))
        self.assertEqual(result['status'],'completed');self.assertEqual(result['seed_audit']['observed_unique_streams'],156)
        row=reports[0]['runs'][0];call=row['call'];task=self.declaration['policies'][0]['task']
        original=pricing.rough_surface(tasks.model_parameters(task,call['parameters']),tasks.joint_curve(task,call['parameters']),
            task['context']['contracts'],task['numerics']['blocks'],[64],tasks.sigma(task),np.inf,call['pricing_seed'])
        np.testing.assert_array_equal(row['controlled_prices'],original[0])
        np.testing.assert_array_equal(row['controlled_covariance'],original[2])
        np.testing.assert_allclose(np.array(row['controlled_prices'])-row['raw_prices'],row['paired_difference'],atol=1e-14)
        self.assertIn('pointwise coverage is not verified',result['anchors'][0]['summary']['scope'])
        with self.assertRaises(FileExistsError):diagnostic.run_group(self.declaration,0,self.root/'group-0')
        with self.assertRaisesRegex(ValueError,'duplicate'):diagnostic.reduce_groups(self.declaration,reports+[reports[0]])
        self.assertEqual(diagnostic.reduce_groups(self.declaration,reports[:-1])['status'],'incomplete')

    def test_interruption_journal_retains_declared_missing_calls(self):
        with patch.object(pricing,'rough_block',side_effect=SystemExit('time limit')) as mocked:
            report=diagnostic.run_group(self.declaration,0,self.root/'interrupted')
        self.assertEqual(mocked.call_count,1);self.assertEqual(len(report['runs']),1)
        self.assertEqual(report['runs'][0]['status'],'interrupted')
        result=diagnostic.reduce_groups(self.declaration,[report],accounting=[{'group_index':0,'state':'TIMEOUT'}])
        self.assertEqual(result['group_statuses'][0]['status'],'censored')
        self.assertEqual(result['declared_calls'],52);self.assertEqual(result['reported_calls'],1)
        self.assertEqual(result['anchors'][0]['status'],'incomplete')

    def test_nonfinite_optional_summary_retains_clean_failure_and_default_return(self):
        original=pricing.rough_block
        def nonfinite(*args,**kwargs):
            result=original(*args,**kwargs)
            kwargs['diagnostics']['raw_control']['control_covariance_of_mean'][0,0]=np.inf
            return result
        with patch.object(pricing,'rough_block',side_effect=nonfinite):
            report=diagnostic.run_group(self.declaration,0,self.root/'nonfinite')
        self.assertEqual(report['status'],'incomplete')
        self.assertTrue(all(r['status']=='failed' for r in report['runs']))
        block=report['runs'][0]['blocks'][0]
        self.assertIn('prices',block);self.assertNotIn('diagnostics',block)
        self.assertIn('nonfinite optional',report['runs'][0]['error'])
        json.loads((self.root/'nonfinite/report.json').read_text())


class BatchAnalysisTests(unittest.TestCase):
    def test_paired_mean_covariance_and_nominal_sqrt_batch_scaling(self):
        rows=[]
        for i in range(16):
            value=(-1 if i%2 else 1)*.2
            rows.append({'controlled_prices':[value],'controlled_covariance':[[.04]],
                         'raw_prices':[value-.1],'raw_covariance':[[.09]],
                         'paired_difference':[.1],'paired_covariance':[[.01]]})
        result=diagnostic.batch_summary(rows)
        self.assertAlmostEqual(result['paired_controlled_minus_raw']['mean'][0],.1)
        self.assertAlmostEqual(result['controlled']['nominal_covariance_of_pooled_mean'][0][0],.04/16)
        pools=result['controlled']['pooling']
        self.assertEqual([p['batches_per_pool'] for p in pools],[1,2,4,8,16])
        for pool in pools:self.assertAlmostEqual(pool['rms_nominal_se'][0],.2/np.sqrt(pool['batches_per_pool']))
        self.assertIsNone(pools[-1]['empirical_sd_across_pools'])
        self.assertEqual(pools[1]['empirical_sd_across_pools'],[0.])


if __name__=='__main__':unittest.main()
