"""Recovery preserves original deadlines and counts prior CPU before admission."""
from datetime import datetime,timezone
import os
import unittest
from unittest.mock import patch

from scripts import recover_heston_spread_scale_campaign as recovery
from test_heston_spread_scale_campaign import Clock,Worker


class ScaleRecoveryTests(unittest.TestCase):
    def test_prior_startup_cpu_reduces_reservations_and_completed_jobs_release_unused_cpu(self):
        self.assertFalse(recovery.cpu_admission(40.74+5*900,1))
        self.assertTrue(recovery.cpu_admission(40.74+5*100,1))
        clock,launches=Clock(),[]
        def launch(job):
            launches.append(job["id"]);return {"process":Worker(10+len(launches),clock,.1)}
        result=recovery.schedule([{"id":str(i)} for i in range(7)],launch,
            lambda *args:{"charged_inner_cpu_seconds":100.},lambda *args:None,10.,40.74,clock=clock,pause=clock.sleep)
        self.assertEqual(result,"completed");self.assertEqual(len(launches),7)
        launches.clear();clock.now=0
        result=recovery.schedule([{"id":str(i)} for i in range(7)],launch,
            lambda *args:{"charged_inner_cpu_seconds":900.},lambda *args:None,10.,40.74,clock=clock,pause=clock.sleep)
        self.assertEqual(result,"cpu_budget_limited");self.assertEqual(len(launches),6)

    def test_deadline_is_conservative_original_utc_plus3900_not_fresh3900(self):
        origin,deadline=recovery.original_deadline({"started_utc":"2026-11-10T15:15:41.563316+00:00","events":[{"seconds":.702818916}]})
        self.assertLess(origin,datetime(2026,11,10,15,15,41,tzinfo=timezone.utc))
        self.assertEqual((deadline-origin).total_seconds(),3900)
        self.assertAlmostEqual((deadline-datetime(2026,11,10,15,25,41,tzinfo=timezone.utc)).total_seconds(),3299.860497,places=5)

    def test_unique_cache_flag_is_set_before_launch_and_parent_environment_restored(self):
        class Consumer:__file__="consumer.py"
        with patch.dict(os.environ,{"PYTENSOR_FLAGS":"old=value"}),patch.object(recovery.base,"launch_consumer") as launch:
            launch.side_effect=lambda *args:{"seen":os.environ["PYTENSOR_FLAGS"]}
            result=recovery.launch_with_cache({"id":"one"},Consumer,"protocol.json","logs",{"PYTENSOR_FLAGS":"compiledir=/private/tmp/unique-one"})
            self.assertEqual(result["seen"],"compiledir=/private/tmp/unique-one")
            self.assertEqual(os.environ["PYTENSOR_FLAGS"],"old=value")


if __name__=="__main__":unittest.main()
