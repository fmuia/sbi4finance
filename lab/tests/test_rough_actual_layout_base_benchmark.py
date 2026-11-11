"""Metadata and fake-kernel tests for the tiny cost benchmark; no rough MC."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from scripts import benchmark_rough_actual_layout_base as benchmark


class BaseCostBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.parent=benchmark.bank.ROOT/"lab/evidence/pilot-results/2026-11-11/bergomi-observation-refresh/joint-clean-128/plan.json"
        self.path=self.root/"work-order.json";self.work=benchmark.prepare(self.parent,self.path)

    def test_four_fixed_rows_and_cost_profile(self):
        benchmark.validate(self.work)
        self.assertEqual([j["row_id"] for j in self.work["jobs"]],[0,1,63,127])
        self.assertEqual(self.work["paths"],32768);self.assertEqual(self.work["grid_multiplier"],1)
        self.assertEqual(self.work["work_comparison"]["old_three_block_sum_steps"],5312)
        self.assertEqual(self.work["work_comparison"]["new_seven_block_base_sum_steps"],8391)
        self.assertEqual(self.work["limits"]["total_allocated_cpu_seconds"],720)

    def test_changed_parameter_or_source_refuses(self):
        changed=copy.deepcopy(self.work);changed["jobs"][0]["spec"]["noise_scale"]+=.1
        with self.assertRaisesRegex(ValueError,"draws"):benchmark.validate(changed)
        changed=copy.deepcopy(self.work);changed["benchmark_source_sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"frozen"):benchmark.validate(changed)

    def test_fake_all_seven_base_blocks_and_no_training_claim(self):
        calls=[]
        def fake(parameters,curve,contracts,**kwargs):
            calls.append(kwargs);n=len(contracts);return np.full(n,.1),np.zeros(n),np.zeros((n,n))
        with patch.dict("os.environ",{"SLURM_JOB_ID":"fixture","SLURM_CPUS_PER_TASK":"1"}),patch.object(
                benchmark.bank.pilot.runtime_pricing,"rough_block",side_effect=fake):
            result=benchmark.run(self.path,0,self.root/"out",process_started=time.monotonic())
        self.assertEqual(result["status"],"complete");self.assertEqual(result["blocks_completed"],7)
        self.assertEqual([c["n_steps"] for c in calls],[b["n_steps"] for b in self.work["parent_bank_plan_value"]["tasks"]["joint_unknown"]["blocks"]])
        self.assertTrue(all(c["n_pairs"]==16384 for c in calls));self.assertFalse(result["training_usable"])
        self.assertFalse(result["continuous_bias_certified"])
        with np.load(self.root/"out/results.npz",allow_pickle=False) as data:
            self.assertEqual(data["physical_parameters"].shape,(6,));self.assertEqual(data["inference_parameters"].shape,(7,))


if __name__=="__main__":unittest.main()
