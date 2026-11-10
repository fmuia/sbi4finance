"""The replay supervisor fails before launching altered or reused identities."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_heston_spread_whitening_replay as controller


class ReplayCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = self.root/"source.json"
        source.write_text("{}\n")
        bound = controller.identity(source)
        self.protocol = {"format": controller.FORMAT, "campaign": controller.LIMITS,
            "checkpoint_selection_posthoc": True, "scientific_acceptance": False,
            "sources": {"controller": controller.identity(controller.__file__), "other": bound},
            "task_file": bound, "task_path": str(source), "task": {},
            "original": {"run": bound}, "original_study": bound,
            "reused_final_evaluation": bound, "reused_final_recovery": bound,
            "output": str(self.root/"training"), "log": str(self.root/"training.log"),
            "evaluations": [{"updates": step,
                "output": str(self.root/f"evidence-{step}"),
                "payload_output": str(self.root/f"payload-{step}"),
                "log": str(self.root/f"eval-{step}.log")} for step in controller.STEPS]}

    def test_changed_historical_input_is_rejected(self):
        controller.validate(self.protocol)
        Path(self.protocol["original_study"]["path"]).write_text('{"changed":true}\n')
        with self.assertRaisesRegex(ValueError, "changed"):
            controller.validate(self.protocol)

    def test_output_alias_and_undeclared_checkpoint_are_rejected(self):
        duplicate = copy.deepcopy(self.protocol)
        duplicate["evaluations"][0]["payload_output"] = duplicate["output"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            controller.validate(duplicate)
        extra = copy.deepcopy(self.protocol)
        extra["evaluations"][0]["updates"] = 1000
        with self.assertRaisesRegex(ValueError, "bounded"):
            controller.validate(extra)

    def test_existing_output_prevents_any_worker_launch(self):
        protocol_path = self.root/"protocol.json"
        protocol_path.write_text(json.dumps(self.protocol))
        Path(self.protocol["evaluations"][1]["output"]).mkdir()
        with patch.object(controller.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(FileExistsError, "no retries"):
                controller.run(protocol_path, self.root/"campaign.json")
            launch.assert_not_called()
        self.assertFalse((self.root/"campaign.claim.json").exists())


if __name__ == "__main__":
    unittest.main()
