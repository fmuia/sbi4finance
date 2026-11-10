"""Exact user-requested batch sizes and bounded direct-child launch accounting."""
import copy
from pathlib import Path
import tempfile
import unittest
from scripts import run_heston_spread_scale_large_batch_campaign as campaign


class LargeBatchCampaignTests(unittest.TestCase):
    def fixture(self, root):
        parent, bank = root / 'parent', root / 'bank'
        parent.mkdir(); bank.mkdir()
        campaign.shared.write(parent / 'checkpoint.json', {})
        campaign.shared.write(bank / 'bank.json', {})
        trainer = root / 'trainer.py'
        trainer.write_text('''import json,sys\nfrom pathlib import Path\nout=Path(sys.argv[sys.argv.index("--out")+1]);out.mkdir()\n(out/"training.json").write_text(json.dumps({"status":"complete"}))\n''')
        identity = lambda p: {'path': str(p), 'sha256': campaign.shared.digest(p)}
        arms = []
        for batch in (1024, 4098):
            batches = (100000 + batch - 1) // batch
            arms.append({'batch_size': batch, 'output': str(root / f'batch-{batch}'),
                'log': str(root / f'batch-{batch}.log'), 'contract': {
                    'batch_size': batch, 'train_rows': 100000, 'batches_per_pass': batches,
                    'tail_batch_size': 100000 - (batches - 1) * batch,
                    'additional_bank_passes': 500, 'additional_presentations': 50000000,
                    'additional_optimizer_updates': 500 * batches,
                    'checkpoint_bank_passes': [25,100,250,500], 'parent_actual_optimizer_updates': 60000,
                    'max_wall_seconds': 5400, 'smoke_test_only': False}})
        return {'format': 'heston_spread_scale_large_batch_study_v1', 'arms': arms,
            'parent_checkpoint': {'record': identity(parent / 'checkpoint.json'), 'path': str(parent)},
            'bank': {'record': identity(bank / 'bank.json'), 'path': str(bank)},
            'sources': {'controller': identity(Path(campaign.__file__).resolve()), 'trainer': identity(trainer)},
            'campaign': {'worker_wall_seconds': 5400, 'whole_training_wall_seconds': 5700,
                'max_concurrent_workers': 2, 'automatic_retry': False, 'automatic_extension': False}}

    def test_batch4098_is_not_rounded_and_tail_exposure_is_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            study = self.fixture(Path(temporary)); campaign.validate(study)
            self.assertEqual([a['contract']['additional_optimizer_updates'] for a in study['arms']], [49000,12500])
            self.assertEqual([a['contract']['tail_batch_size'] for a in study['arms']], [672,1648])
            changed = copy.deepcopy(study); changed['arms'][1]['batch_size'] = 4096
            with self.assertRaisesRegex(ValueError, '1024/4098'):
                campaign.validate(changed)

    def test_changed_budget_or_parent_or_colliding_outputs_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            study = self.fixture(Path(temporary))
            for key, value in (('additional_presentations', 49999999), ('parent_actual_optimizer_updates', 109000),
                    ('tail_batch_size', 4098), ('smoke_test_only', True)):
                changed = copy.deepcopy(study); changed['arms'][1]['contract'][key] = value
                with self.assertRaisesRegex(ValueError, 'contract differs'):
                    campaign.validate(changed)
            changed = copy.deepcopy(study); changed['arms'][1]['output'] = changed['arms'][0]['output']
            with self.assertRaisesRegex(ValueError, 'distinct'):
                campaign.validate(changed)
            changed = copy.deepcopy(study); changed['campaign']['automatic_retry'] = True
            with self.assertRaisesRegex(ValueError, 'bounded'):
                campaign.validate(changed)

    def test_source_mutation_is_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            study = self.fixture(Path(temporary))
            Path(study['sources']['trainer']['path']).write_text('changed')
            with self.assertRaisesRegex(ValueError, 'source differs'):
                campaign.validate(study)

    def test_direct_workers_finish_once_and_claim_prevents_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); study = self.fixture(root)
            path, out = root / 'protocol.json', root / 'campaign.json'
            campaign.shared.write(path, study)
            result = campaign.run(path, out)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(set(result['arms']), {'1024','4098'})
            for key, value in result['arms'].items():
                self.assertEqual(value['status'], 'complete')
                command = value['command']; self.assertIn('--worker', command)
                self.assertEqual(command[command.index('--batch-size')+1], key)
            self.assertTrue(out.with_suffix('.claim.json').exists())
            with self.assertRaisesRegex(FileExistsError, 'must be new'):
                campaign.run(path, out)


if __name__ == '__main__': unittest.main()
