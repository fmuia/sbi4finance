"""A precision repair must retain the original training deadline."""
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from scripts import run_heston_spread_scale_precision_campaign as campaign


class PrecisionDeadlineTests(unittest.TestCase):
    def test_remaining_budget_comes_from_original_started_clock(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"original.json"
            campaign.shared.write(path,{"status":"incomplete","started_utc":"2026-11-10T18:13:49.590152+00:00"})
            protocol={"original_campaign":{"path":str(path),"sha256":campaign.shared.digest(path)},
                "campaign":{"absolute_deadline_utc":"2026-11-10T19:48:49.590152+00:00",
                    "worker_wall_seconds":5400,"automatic_retry":False}}
            now=datetime(2026,11,10,18,48,49,590152,tzinfo=timezone.utc)
            self.assertEqual(campaign.deadline_seconds(protocol,now=now),3600)
            protocol["campaign"]["absolute_deadline_utc"]="2026-11-10T19:48:50.590152+00:00"
            with self.assertRaisesRegex(ValueError,"cannot extend"):
                campaign.deadline_seconds(protocol,now=now)

    def test_mutated_original_campaign_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"original.json"
            path.write_text("{}")
            protocol={"original_campaign":{"path":str(path),"sha256":campaign.shared.digest(path)}}
            path.write_text('{"different":true}')
            with self.assertRaisesRegex(ValueError,"record differs"):
                campaign.deadline_seconds(protocol)


if __name__ == "__main__":
    unittest.main()
