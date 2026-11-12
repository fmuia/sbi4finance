"""Pure explicit-authorization and unchanged-financial-worker checks; no MC."""
from contextlib import ExitStack
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from sbi4finance.calibration import rough_actual_layout_authorized as pilot
from sbi4f_tutorial.shared import pricing
from sbi4f_tutorial.rough_options import simulator
from test_rough_actual_layout_pilot import fake_block
WORK=pilot.ROOT/'lab/evidence/pilot-results/2026-11-10/bergomi-observation-refresh/preparation-portable/work-order.json'
class AuthorizedTests(unittest.TestCase):
 def setUp(self):
  guards=ExitStack();guards.enter_context(patch.object(pricing,'rough_block',side_effect=AssertionError('no MC')))
  guards.enter_context(patch.object(simulator,'simulate_rough',side_effect=AssertionError('no MC')));self.addCleanup(guards.close)
 def test_permission_separate_from_failed_Heston_and_old_limits(self):
  with tempfile.TemporaryDirectory() as tmp:
   a=pilot.prepare(WORK,Path(tmp)/'prepared');w=pilot.read(WORK)
   self.assertFalse(a['heston_diagnostics_passed']);self.assertFalse(a['scientific_acceptance'])
   self.assertTrue(a['explicit_user_authorization']);self.assertTrue(a['supersedes_heston_success_condition'])
   self.assertEqual(a['execution_policy']['maximum_concurrent_workers'],16);self.assertEqual(a['execution_policy']['maximum_allocated_cpu_seconds'],28800)
   self.assertEqual(w['limits']['maximum_concurrent_workers'],2);self.assertIn('--array=0-47%16',pilot.sbatch_text());pilot.validate_authorization(w,WORK,a)
 def test_missing_or_forged_permission_scientific_pass_resources_and_source_reject(self):
  with tempfile.TemporaryDirectory() as tmp:
   a=pilot.prepare(WORK,Path(tmp)/'prepared');w=pilot.read(WORK)
   for m in ('permission','Heston','instruction','policy','source'):
    b=deepcopy(a)
    if m=='permission':b['explicit_user_authorization']=False
    if m=='Heston':b['heston_diagnostics_passed']=True
    if m=='instruction':b['user_instruction']='Run anything'
    if m=='policy':b['execution_policy']['maximum_allocated_cpu_seconds']*=2
    if m=='source':b['source_bindings']['authorized_pilot']['sha256']='0'*64
    with self.assertRaises(ValueError):pilot.validate_authorization(w,WORK,b)
 def test_mocked_worker_original_panel_grid_paths_and_authorization_record(self):
  with tempfile.TemporaryDirectory() as tmp:
   r=Path(tmp);a=pilot.prepare(WORK,r/'prepared');seen=[]
   def capture(parameters,curve,contracts,**kw):seen.append(kw);return fake_block(parameters,curve,contracts,**kw)
   with patch.dict(os.environ,{'SLURM_JOB_ID':'unit-only','SLURM_CPUS_PER_TASK':'1'}),patch.object(pilot,'load_controlled_helper',return_value=SimpleNamespace(coupled_controlled_block=capture)):
    result=pilot.run_component(WORK,r/'prepared/authorization.json',0,r/'result',process_started=time.monotonic())
   self.assertEqual(len(seen),7);self.assertTrue(all(v['n_pairs']==16384 for v in seen));self.assertEqual(result['case'],pilot.read(WORK)['cases'][0])
   self.assertFalse(result['numerical_target_established']);self.assertEqual(result['execution_authorization'],pilot.binding(r/'prepared/authorization.json'));self.assertNotIn('condition',result)
 def test_unallocated_worker_rejects_before_simulation_or_output(self):
  with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True):
   out=Path(tmp)/'not-created'
   with self.assertRaisesRegex(ValueError,'allocated Slurm'):pilot.run_component(WORK,Path(tmp)/'missing',0,out,process_started=time.monotonic())
   self.assertFalse(out.exists())
if __name__=='__main__':unittest.main()
