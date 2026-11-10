"""Selection fairness and direct-worker environment/supervision checks."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from scripts import prepare_heston_spread_recovery as selection
from scripts import run_heston_spread_recovery_campaign as campaign


class RecoveryCampaignTests(unittest.TestCase):
    def test_selection_depends_only_on_generating_c_rank(self):
        rng = np.random.default_rng(82)
        truth = rng.normal(size=(512,6))
        truth[:, -1] = rng.permutation(np.linspace(.5,2,512))
        indices = selection.selected_rows(truth)
        np.testing.assert_array_equal(np.argsort(truth[:,-1],kind="stable")[list(selection.RANKS)],indices)
        changed = truth.copy(); changed[:,:5] *= -1234
        np.testing.assert_array_equal(selection.selected_rows(changed),indices)
        self.assertEqual(len(set(indices)),8)

    def test_worker_caches_use_writable_case_specific_base_and_compiled_directory(self):
        env = campaign.child_environment("validation-00000143")
        self.assertIn("base_compiledir=/private/tmp/",env["PYTENSOR_FLAGS"])
        self.assertIn("compiledir=/private/tmp/",env["PYTENSOR_FLAGS"])
        self.assertIn("validation-00000143",env["NUMBA_CACHE_DIR"])
        with self.assertRaisesRegex(ValueError,"safe"):
            campaign.child_environment("../bad")

    def test_direct_worker_stop_reaps_real_child(self):
        child = subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"])
        campaign.stop(child)
        self.assertIsNotNone(child.poll())

    def test_unbounded_or_wrong_case_count_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            campaign.write(root/"protocol.json",{"format":"heston_spread_recovery_reference_protocol_v1",
                "campaign":{},"jobs":[]})
            with self.assertRaisesRegex(ValueError,"bounded"):
                campaign.run(root/"protocol.json",root/"campaign.json",root/"logs")
            self.assertFalse((root/"campaign.json").exists())


if __name__ == "__main__":
    unittest.main()
