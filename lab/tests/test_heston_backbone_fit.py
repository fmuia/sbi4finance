"""Native sampling, validation isolation and fail-closed stage execution."""
import argparse
import json
from pathlib import Path
import subprocess
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_heston_backbone_fit as runner

class BackboneRunnerTests(unittest.TestCase):
    @staticmethod
    def archived_environment_digest(contract):
        """Replay the fixture's original environment, leaving production hash guards strict."""
        original = runner.digest

        def recorded(path):
            path = Path(path)
            if path.parent == runner.ROOT and path.name in contract['environment_sha256']:
                return contract['environment_sha256'][path.name]
            return original(path)

        return recorded

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.args = argparse.Namespace(bank=root/'bank', references=root/'references', out=root/'out')
        self.recipe = json.loads(runner.RECIPE.read_text())

    def test_commands_match_recipe_and_omit_legacy_solver_flag(self):
        stages = runner.commands(self.args, self.recipe)
        training = stages[1][1]
        self.assertNotIn('--flow-steps', training)
        self.assertEqual(training[training.index('--epochs')+1], '500')
        self.assertEqual(training[training.index('--batch-size')+1], '256')
        evaluation = stages[-1][1]
        self.assertEqual(evaluation[evaluation.index('--split')+1], 'validation')
        self.assertEqual(evaluation[evaluation.index('--surfaces')+1], '2000')
        self.assertEqual(evaluation[evaluation.index('--draws')+1], '1000')
        self.assertEqual(len(stages), 6)
        self.assertEqual([name for name, _, _ in stages], ["environment", "training", "comparison-0", "comparison-1", "comparison-2", "validation"])
        for i, (_, command, _) in enumerate(stages[2:5]):
            self.assertIn(str(self.args.references/f'case-{i}'/'reference'), command)
            self.assertEqual(command[command.index('--seed')+1], str(271026+i))
        caps = self.recipe['resources']
        exposure = caps['wall_seconds']+caps['scheduler_cleanup_seconds']
        self.assertLessEqual(exposure, 3600)
        self.assertLessEqual(exposure*caps['cpus'], 4*3600)
        self.assertTrue(all(cap > 0 for _, _, cap in stages))
        self.assertEqual(stages[-1][2],900)
        self.assertGreater(sum(cap for _, _, cap in stages), self.recipe['execution_budget']['total_seconds'])

    def test_real_snapshot_reference_preflight_rejects_changed_observation(self):
        self.args.bank.mkdir()
        snapshot = runner.ROOT/'evidence/pilot-results/2026-10-27/tutorial-backbone/heston-snapshots/snapshot-50k-manifest.json'
        shutil.copyfile(snapshot,self.args.bank/'manifest.json')
        shutil.copytree(runner.ANCHORS,self.args.references)
        contract=json.loads((runner.BASELINE/'training.json').read_text())['run_contract']
        versions={n:contract['runtime'][n] for n in ('bayesflow','keras','jax','numpy','scipy')}
        with (patch.dict('os.environ',{'KERAS_BACKEND':'jax'}),
              patch.object(runner, 'digest', side_effect=self.archived_environment_digest(contract)),
              patch.object(runner.platform,'python_version',return_value=contract['runtime']['python']),
              patch.object(runner.metadata,'version',side_effect=versions.__getitem__)):
            frozen=runner.validate_inputs(self.args.bank,self.args.references,self.recipe)
            self.assertEqual(len(frozen['cases']),3)
            path=self.args.references/'case-1/reference/reference.json'
            changed=json.loads(path.read_text());changed['observed_prices'][0]+=.001
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'case1'):
                runner.validate_inputs(self.args.bank,self.args.references,self.recipe)

    def test_100k_recipe_preserves_comparison_panel_and_declares_195500_updates(self):
        path=runner.ROOT/'configs/heston_backbone_100k_fit.json'
        recipe=json.loads(path.read_text());stages=runner.commands(self.args,recipe)
        self.assertEqual(recipe['splits'],{'train':100000,'validation':5000,'test':5000})
        self.assertEqual(recipe['training']['expected_updates'],195500)
        self.assertEqual(recipe['training']['epochs'],self.recipe['training']['epochs'])
        self.assertEqual(recipe['network'],self.recipe['network'])
        self.assertEqual(recipe['evaluation'],self.recipe['evaluation'])
        self.assertEqual(recipe['reference_comparison'],self.recipe['reference_comparison'])
        self.assertNotIn('--flow-steps',stages[1][1])
        self.assertEqual([name for name,_,_ in stages],[name for name,_,_ in runner.commands(self.args,self.recipe)])
        resources=recipe['resources'];exposure=resources['wall_seconds']+resources['scheduler_cleanup_seconds']
        self.assertEqual(exposure,2640);self.assertEqual(exposure*resources['cpus'],10560)
        self.assertLessEqual(exposure,recipe['allocation_ceiling']['gpu_hours']*3600)
        self.assertLessEqual(exposure*resources['cpus'],recipe['allocation_ceiling']['allocated_cpu_hours']*3600)
        self.assertEqual(recipe['execution_budget']['total_seconds'],2370)
        self.assertEqual(sum(cap for _,_,cap in stages),2340)

    def test_actual_100k_snapshot_and_update_metadata_are_checked(self):
        recipe=json.loads((runner.ROOT/'configs/heston_backbone_100k_fit.json').read_text())
        self.args.bank.mkdir()
        source=runner.ROOT/'evidence/pilot-results/2026-10-27/tutorial-backbone/heston-snapshots/snapshot-100k-manifest.json'
        shutil.copyfile(source,self.args.bank/'manifest.json');shutil.copytree(runner.ANCHORS,self.args.references)
        contract=json.loads((runner.BASELINE/'training.json').read_text())['run_contract']
        versions={n:contract['runtime'][n] for n in ('bayesflow','keras','jax','numpy','scipy')}
        with (patch.dict('os.environ',{'KERAS_BACKEND':'jax'}),
              patch.object(runner, 'digest', side_effect=self.archived_environment_digest(contract)),
              patch.object(runner.platform,'python_version',return_value=contract['runtime']['python']),
              patch.object(runner.metadata,'version',side_effect=versions.__getitem__)):
            frozen=runner.validate_inputs(self.args.bank,self.args.references,recipe)
            self.assertEqual(frozen['bank_manifest_sha256'],recipe['bank_manifest_sha256'])
            recipe['training']['expected_updates']-=1
            with self.assertRaisesRegex(ValueError,'update count'):
                runner.validate_inputs(self.args.bank,self.args.references,recipe)

    def test_recipe_option_records_selected_path_and_hash(self):
        self.args.recipe=runner.ROOT/'configs/heston_backbone_100k_fit.json'
        env={'SLURM_JOB_ID':'fixture','SLURM_CPUS_PER_TASK':'4',**{n:'1' for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
        with (patch.dict('os.environ',env),patch.object(runner.subprocess,'check_output',return_value='fixture'),
              patch.object(runner,'validate_inputs',side_effect=ValueError('preflight fixture'))):
            with self.assertRaisesRegex(ValueError,'preflight fixture'):runner.run(self.args)
        state=json.loads((self.args.out/'run.json').read_text())
        self.assertEqual(state['recipe_path'],'configs/heston_backbone_100k_fit.json')
        self.assertEqual(state['recipe']['training']['expected_updates'],195500)
        self.assertEqual(state['source_sha256'][state['recipe_path']],runner.digest(self.args.recipe))
        self.assertNotIn('configs/heston_backbone_50k_fit.json',state['source_sha256'])

    def test_no_login_execution(self):
        with patch.dict('os.environ', {}, clear=True), self.assertRaisesRegex(ValueError, 'allocation'):
            runner.run(self.args)
        self.assertFalse(self.args.out.exists())

    def test_timeout_keeps_journal_and_never_starts_training(self):
        env = {'SLURM_JOB_ID':'fixture', 'SLURM_CPUS_PER_TASK':'4', **{n:'1' for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
        with (patch.dict('os.environ',env), patch.object(runner,'validate_inputs',return_value={}),
              patch.object(runner.subprocess,'check_output',return_value='fixture'),
              patch.object(runner.subprocess,'run',side_effect=subprocess.TimeoutExpired('env',60)) as child):
            with self.assertRaises(subprocess.TimeoutExpired): runner.run(self.args)
        state = json.loads((self.args.out/'run.json').read_text())
        self.assertEqual(state['status'],'incomplete')
        self.assertEqual(state['stages'][0]['status'],'timeout')
        self.assertEqual(state['stages'][1]['status'],'pending')
        self.assertFalse(state['automatic_retry']);self.assertFalse(state['reserved_test_evaluated'])
        child.assert_called_once()

    def test_stage_maximum_is_clipped_to_remaining_whole_run_allowance(self):
        env = {'SLURM_JOB_ID':'fixture', 'SLURM_CPUS_PER_TASK':'4', **{n:'1' for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
        # Metadata/preflight consumes3210s, leaving30s of the hard3240s
        # allowance. The environment stage has a60s maximum but receives30s.
        with (patch.dict('os.environ',env), patch.object(runner,'validate_inputs',return_value={}),
              patch.object(runner.subprocess,'check_output',return_value='fixture'),
              patch.object(runner.time,'perf_counter',side_effect=[0,3210,3210,3240,3240]),
              patch.object(runner.subprocess,'run',side_effect=subprocess.TimeoutExpired('env',30)) as child):
            with self.assertRaises(subprocess.TimeoutExpired): runner.run(self.args)
        state=json.loads((self.args.out/'run.json').read_text())
        self.assertEqual(child.call_args.kwargs['timeout'],30)
        self.assertEqual(state['stages'][0]['effective_timeout_seconds'],30)
        self.assertEqual(state['stages'][0]['status'],'timeout')
        self.assertEqual(state['stages'][1]['status'],'pending')
        self.assertEqual(state['status'],'incomplete')
        child.assert_called_once()

    def test_preflight_failure_is_durable_before_gpu_work(self):
        env = {'SLURM_JOB_ID':'fixture','SLURM_CPUS_PER_TASK':'4',**{n:'1' for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
        with (patch.dict('os.environ',env),patch.object(runner.subprocess,'check_output',return_value='fixture'),
              patch.object(runner,'validate_inputs',side_effect=ValueError('changed snapshot')),
              patch.object(runner.subprocess,'run') as child):
            with self.assertRaisesRegex(ValueError,'changed snapshot'):runner.run(self.args)
        state=json.loads((self.args.out/'run.json').read_text());self.assertEqual(state['status'],'failed')
        self.assertEqual(state['failure']['type'],'ValueError');child.assert_not_called()

if __name__=='__main__':unittest.main()
