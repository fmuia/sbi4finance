"""Real direct children exercise completion/progress and shutdown without groups."""
import subprocess
import sys
import time
import unittest

from scripts import recover_heston_spread_scale_campaign_v2 as recovery


class ScaleRecoveryV2Tests(unittest.TestCase):
    def test_real_child_completion_updates_status_once_and_releases_cpu_reservation(self):
        progress={"a":{},"b":{}}
        def launch(job):
            p=subprocess.Popen([sys.executable,"-c","pass"])
            return {"process":recovery.DirectWorker(p,seconds=10)}
        def complete(job,entry,code):
            self.assertEqual(code,0)
            return {"status":"completed","reference_ready":False,"charged_inner_cpu_seconds":.01}
        def event(kind,job,details):
            if kind!="poll":recovery.update_job(progress[job["id"]],kind,details)
        status=recovery.old.schedule([{"id":"a"},{"id":"b"}],launch,complete,event,time.monotonic()+5.,46.056057,stopper=recovery.stop_direct)
        self.assertEqual(status,"completed")
        self.assertEqual([p["status"] for p in progress.values()],["completed","completed"])

    def test_real_direct_process_shutdown_needs_no_process_group_permission(self):
        p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"])
        worker=recovery.DirectWorker(p)
        recovery.stop_direct({"one":{"process":worker}},time.monotonic()+1.)
        self.assertIsNotNone(p.poll())
        self.assertLess(p.returncode,0)

    def test_direct_worker_outer_timeout_kills_a_real_stalled_child(self):
        p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"])
        worker=recovery.DirectWorker(p,seconds=.05)
        time.sleep(.08)
        self.assertIsNotNone(worker.poll())
        self.assertTrue(worker.timed_out)
        self.assertLess(p.returncode,0)


if __name__=="__main__":unittest.main()
