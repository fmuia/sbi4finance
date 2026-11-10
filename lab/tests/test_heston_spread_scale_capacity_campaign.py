"""Direct-worker shutdown and prelaunch rejection for the capacity study."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts import run_heston_spread_scale_capacity as campaign


class CapacityCampaignTests(unittest.TestCase):
    def test_stop_reaps_an_actual_direct_child(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        campaign.stop(child)
        self.assertIsNotNone(child.poll())
        campaign.stop(child)

    def test_rejects_an_unbounded_study_before_output_or_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "protocol.json"
            campaign.write(path, {"format":"heston_spread_scale_capacity_study_v1",
                "arms":[{"batch_size":128},{"batch_size":512}],
                "campaign":{"worker_wall_seconds":5401,"whole_training_wall_seconds":5700,
                            "automatic_retry":False}})
            with self.assertRaisesRegex(ValueError,"bounded"):
                campaign.run(path, root/"campaign.json")
            self.assertFalse((root/"campaign.json").exists())


if __name__ == "__main__":
    unittest.main()
