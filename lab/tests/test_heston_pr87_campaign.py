"""Notebook reproduction refuses reused outputs and altered provenance."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_heston_pr87_reproduction as controller


class NotebookCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        path=self.root/'task.json';path.write_text('{}\n');bound=controller.identity(path)
        self.protocol={'format':controller.FORMAT,'campaign':controller.LIMITS,
            'predeclared_before_fit':True,'scientific_acceptance':False,'fresh_noise_per_pass':False,
            'sources':{'controller':controller.identity(controller.__file__),'other':bound},
            'task':{},'task_path':str(path),'task_file':bound,'input_manifest':bound,
            'upstream_notebook':bound,'original_whitening_study':bound,'historical_population_inputs':bound,
            **{key:str(self.root/key) for key in ('output','training_log','evaluation_output',
                'evaluation_payload_output','evaluation_log')}}

    def test_changed_input_refuses_before_launch(self):
        controller.validate(self.protocol)
        Path(self.protocol['input_manifest']['path']).write_text('{"different":true}\n')
        with self.assertRaisesRegex(ValueError,'changed'):controller.validate(self.protocol)

    def test_output_alias_and_changed_budget_are_rejected(self):
        changed=copy.deepcopy(self.protocol);changed['evaluation_output']=changed['output']
        with self.assertRaisesRegex(ValueError,'distinct'):controller.validate(changed)
        changed=copy.deepcopy(self.protocol);changed['campaign']['automatic_retry']=True
        with self.assertRaisesRegex(ValueError,'bounded'):controller.validate(changed)

    def test_existing_output_is_not_an_automatic_retry(self):
        path=self.root/'protocol.json';path.write_text(json.dumps(self.protocol))
        Path(self.protocol['evaluation_payload_output']).mkdir()
        with patch.object(controller.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(FileExistsError,'no retries'):
                controller.run(path,self.root/'campaign.json')
            launch.assert_not_called()
        self.assertFalse((self.root/'campaign.claim.json').exists())


if __name__=='__main__':unittest.main()
