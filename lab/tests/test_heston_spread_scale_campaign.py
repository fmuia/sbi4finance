"""Virtual workers check scheduling, shutdown and process-session contracts."""
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_heston_spread_scale_campaign as campaign


class Clock:
    def __init__(self): self.now=0.
    def __call__(self): return self.now
    def sleep(self,seconds): self.now+=seconds


class Worker:
    def __init__(self,pid,clock,duration): self.pid,self.clock,self.end,self.code=pid,clock,clock()+duration,None
    def poll(self):
        if self.code is None and self.clock()>=self.end: self.code=0
        return self.code
    def wait(self): self.code=-signal.SIGKILL; return self.code


class ScaleCampaignTests(unittest.TestCase):
    def test_fifo_seven_workers_never_exceed_two_and_do_not_retry_failures(self):
        clock=Clock()
        jobs=[{"id":str(i)} for i in range(7)]
        launched,events=[],[]
        def launch(job):
            launched.append(job["id"])
            return {"process":Worker(100+len(launched),clock,.3)}
        def complete(job,entry,code):
            return {"status":"inconclusive" if job["id"]=="2" else "completed","reference_ready":False}
        finished=campaign.schedule(jobs,launch,complete,lambda kind,job,data:events.append((kind,data)),10.,clock=clock,pause=clock.sleep)
        self.assertEqual(launched,[str(i) for i in range(7)])
        self.assertEqual(len(finished),7)
        self.assertEqual(finished["2"]["status"],"inconclusive")
        self.assertLessEqual(max(data["active_jobs"] for kind,data in events if kind=="started"),2)

    def test_campaign_deadline_stops_all_running_groups_and_leaves_pending_unstarted(self):
        clock=Clock()
        launched,events,signals=[],[],[]
        def launch(job):
            launched.append(job["id"])
            return {"process":Worker(200+len(launched),clock,20.)}
        def stop(active,deadline,**kwargs):
            campaign.stop_groups(active,deadline,killpg=lambda pid,sig:signals.append((pid,sig)),**kwargs)
        with self.assertRaises(TimeoutError):
            campaign.schedule([{"id":str(i)} for i in range(7)],launch,lambda *args:{},
                lambda kind,job,data:events.append((kind,job)),.3,clock=clock,pause=clock.sleep,stopper=stop)
        self.assertEqual(launched,["0","1"])
        self.assertEqual([kind for kind,_ in events].count("stopped"),2)
        self.assertEqual([kind for kind,_ in events].count("not_started"),5)
        self.assertEqual(set(signals),{(201,signal.SIGTERM),(202,signal.SIGTERM),(201,signal.SIGKILL),(202,signal.SIGKILL)})
        self.assertEqual(clock(),.3)

    def test_interrupt_also_invokes_group_shutdown(self):
        clock=Clock()
        stopped=[]
        def pause(seconds): raise KeyboardInterrupt("test interruption")
        def stop(active,deadline,**kwargs): stopped.extend(active)
        with self.assertRaises(KeyboardInterrupt):
            campaign.schedule([{"id":"a"},{"id":"b"}],lambda job:{"process":Worker(9,clock,20)},
                lambda *args:{},lambda *args:None,100.,clock=clock,pause=pause,stopper=stop)
        self.assertEqual(stopped,["a","b"])

    def test_launch_uses_new_session_one_thread_environment_and_exclusive_log(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(campaign.subprocess,"Popen") as popen:
            root=Path(temporary)
            campaign.launch_consumer({"id":"case"},"consumer.py","protocol.json",root)
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            env=popen.call_args.kwargs["env"]
            self.assertTrue(all(env[key]=="1" for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")))
            with self.assertRaises(FileExistsError):
                campaign.launch_consumer({"id":"case"},"consumer.py","protocol.json",root)
            self.assertEqual(popen.call_count,1)


if __name__=="__main__": unittest.main()
