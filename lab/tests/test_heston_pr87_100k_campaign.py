"""The costly100k supervisor refuses altered provenance and reused outputs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_heston_pr87_100k as controller


class HundredKCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);task=self.root/'task.json';task.write_text('{}\n')
        bound=controller.identity(task)
        self.protocol={'format':controller.FORMAT,'campaign':controller.LIMITS,
            'predeclared_before_fit':True,'scientific_acceptance':False,
            'endpoint_policy':{'primary':'final_epoch200','secondary':'native minimum validation objective only if its epoch differs from200',
                'selection_used_posterior_diagnostics':False},
            'sources':{'controller':controller.identity(controller.__file__),'parent_evaluator':bound},
            'task':{},'task_path':str(task),'task_file':bound,'parent_study':bound,
            'parent_10k_evaluation':bound,'original_whitening_study':bound,'historical_population_inputs':bound,
            'output':str(self.root/'training'),'training_log':str(self.root/'training.log'),
            'evaluations':{role:{key:str(self.root/(role+'-'+key)) for key in ('output','payload_output','log')}
                for role in ('primary','secondary')}}

    def test_parent_source_change_refuses(self):
        controller.validate(self.protocol)
        Path(self.protocol['parent_study']['path']).write_text('{"changed":true}\n')
        with self.assertRaisesRegex(ValueError,'changed'):controller.validate(self.protocol)

    def test_selection_policy_and_duplicate_outputs_refuse(self):
        changed=copy.deepcopy(self.protocol);changed['endpoint_policy']['selection_used_posterior_diagnostics']=True
        with self.assertRaisesRegex(ValueError,'predeclared'):controller.validate(changed)
        changed=copy.deepcopy(self.protocol);changed['evaluations']['secondary']['output']=changed['evaluations']['primary']['output']
        with self.assertRaisesRegex(ValueError,'distinct'):controller.validate(changed)

    def test_existing_secondary_identity_prevents_launch(self):
        protocol=self.root/'protocol.json';protocol.write_text(json.dumps(self.protocol))
        Path(self.protocol['evaluations']['secondary']['payload_output']).mkdir()
        with patch.object(controller.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(FileExistsError,'no retries'):
                controller.run(protocol,self.root/'campaign.json')
            launch.assert_not_called()
        self.assertFalse((self.root/'campaign.claim.json').exists())


if __name__=='__main__':unittest.main()
