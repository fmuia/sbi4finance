"""Pure ARM/x86 portability and immutable-panel orchestration guards; no MC."""
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from sbi4finance.calibration import rough_actual_layout_pilot as original
from sbi4finance.calibration import rough_actual_layout_pilot_portable as portable
from sbi4f_tutorial.shared import pricing
from sbi4f_tutorial.rough_options import simulator
from test_rough_actual_layout_pilot import completed_evaluation,fake_block
from rough_actual_layout_fixtures import current_portable,HISTORICAL_PORTABLE

PARENT=original.ROOT/'lab/evidence/pilot-results/2026-11-10/bergomi-observation-refresh/preparation/work-order.json'


def fixture(root):
    work,path=current_portable(root/'portable')
    path.with_name('pilot.sbatch').write_text(portable.sbatch_text())
    return work,path


class PortableTests(unittest.TestCase):
    def setUp(self):
        guards=ExitStack();guards.enter_context(patch.object(pricing,'rough_block',side_effect=AssertionError('no MC')))
        guards.enter_context(patch.object(simulator,'simulate_rough',side_effect=AssertionError('no MC')));self.addCleanup(guards.close)

    def test_current_source_fixture_preserves_authoritative_panel_without_new_draws(self):
        with tempfile.TemporaryDirectory() as tmp:
            work,path=fixture(Path(tmp));parent=original.read(PARENT)
            self.assertEqual(work['cases'],parent['cases']);self.assertEqual(work['tasks'],parent['tasks'])
            self.assertEqual(work['authoritative_panel_sha256'],portable.AUTHORITATIVE_PANEL_SHA256)
            self.assertEqual(work['parent_work_order']['sha256'],original.digest(PARENT));portable.validate(work)
            self.assertIn('run_rough_actual_layout_pilot_portable.py',path.with_name('pilot.sbatch').read_text())

    def test_historical_work_order_cannot_resume_with_a_changed_source(self):
        before={path:path.read_bytes() for path in (PARENT,HISTORICAL_PORTABLE)}
        source=original.ROOT/'tutorial/src/sbi4f_tutorial/shared/pricing.py'
        real=original.digest
        def changed(path):return '0'*64 if Path(path)==source else real(path)
        with tempfile.TemporaryDirectory() as tmp,patch.object(original,'digest',side_effect=changed):
            out=Path(tmp)/'not-created'
            with self.assertRaisesRegex(ValueError,'actual execution source differs'):
                portable.prepare(PARENT,out)
            self.assertFalse(out.exists())
        self.assertEqual(before,{path:path.read_bytes() for path in before})

    def test_one_ulp_host_math_variation_passes_without_replacing_stored_values(self):
        stored=original.read(PARENT)['cases'];before=deepcopy(stored);host=deepcopy(stored)
        for case in host[12:]:
            for field in portable.ROUNDING_POLICY['fields']:
                case[field]=np.nextafter(np.asarray(case[field]),np.inf).tolist()
        portable.check_panel(stored,host);self.assertEqual(stored,before)

    def test_only_declared_derived_known_fields_allow_roundoff(self):
        stored=original.read(PARENT)['cases']
        for case,field in ((0,'parameters'),(12,'parameters'),(12,'curve_draw_seed')):
            host=deepcopy(stored)
            if field=='parameters':host[case][field][0]=np.nextafter(host[case][field][0],np.inf)
            else:host[case][field]+=1
            with self.assertRaisesRegex(ValueError,'non-derived'):portable.check_panel(stored,host)
        for value in (float('nan'),-1.,.5):
            host=deepcopy(stored);host[12]['supplied_levels'][0]=value
            with self.assertRaises(ValueError):portable.check_panel(stored,host)

    def test_authoritative_stored_panel_cannot_change_even_by_one_ulp(self):
        with tempfile.TemporaryDirectory() as tmp:
            work,_=fixture(Path(tmp));work['cases'][12]['supplied_levels'][0]=np.nextafter(work['cases'][12]['supplied_levels'][0],np.inf)
            work['authoritative_panel_sha256']=portable.panel_fingerprint(work['cases'])
            with self.assertRaisesRegex(ValueError,'original-panel'):portable.validate(work)

    def test_exact_recipe_prior_and_source_tampering_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            work,_=fixture(Path(tmp))
            for kind in ('paths','prior','source','tolerance'):
                changed=deepcopy(work)
                if kind=='paths':changed['components'][0]['paths']*=2
                if kind=='prior':changed['tasks']['supplied_known']['prior_bounds'][0][0]=.03
                if kind=='source':changed['source_bindings']['portable_pilot']['sha256']='0'*64
                if kind=='tolerance':changed['validation_roundoff_policy']['known_derived_relative_tolerance']=1e-3
                with self.assertRaises(ValueError):portable.validate(changed)

    def test_failed_Heston_diagnostics_cannot_bind_portable_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work,path=fixture(root);evaluation=root/'evaluation.json'
            record=completed_evaluation(work['heston_target']);record['cases']['market']['reference_ready']=False
            original.write_new(evaluation,record)
            with self.assertRaisesRegex(ValueError,'diagnostics fail'):portable.bind_heston(path,evaluation,root/'condition.json')
            self.assertFalse((root/'condition.json').exists())

    def test_mocked_worker_executes_original_stored_curve_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work,path=fixture(root);evaluation=root/'evaluation.json';condition=root/'condition.json'
            original.write_new(evaluation,completed_evaluation(work['heston_target']));portable.bind_heston(path,evaluation,condition)
            expected=portable.curve_for_case(work,work['cases'][12]);seen=[]
            def capture(parameters,curve,contracts,**kwargs):
                self.assertEqual(curve,expected);seen.append(1);return fake_block(parameters,curve,contracts,**kwargs)
            # First supplied-known component:4 prior rows ordered16 components per row?12 per row.
            index=next(c['index'] for c in work['components'] if c['case_index']==12 and c['role']=='coupled_1_2')
            with patch.dict(os.environ,{'SLURM_JOB_ID':'unit-only','SLURM_CPUS_PER_TASK':'1'}),patch.object(portable,'load_controlled_helper',return_value=SimpleNamespace(coupled_controlled_block=capture)):
                result=portable.run_component(path,condition,index,root/'result',process_started=time.monotonic())
            self.assertEqual(len(seen),7);self.assertEqual(result['case'],original.read(PARENT)['cases'][12])
            self.assertFalse(result['numerical_target_established'])

    def test_preparer_gate_imports_current_readiness_from_outside_lab_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work,path=fixture(root);evaluation=root/'evaluation.json'
            record=completed_evaluation(work['heston_target']);record['cases']['market']['reference_ready']=False
            original.write_new(evaluation,record)
            env=os.environ.copy();env['PYTHONPATH']=str(original.ROOT/'lab/src')+':'+str(original.ROOT/'tutorial/src')
            run=subprocess.run([sys.executable,str(original.ROOT/'lab/scripts/prepare_rough_actual_layout_pilot_portable.py'),
                'bind-heston','--work-order',str(path),'--evaluation',str(evaluation),'--out',str(root/'condition.json')],cwd=root,env=env,capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0);self.assertIn('diagnostics fail',run.stderr)
            self.assertNotIn('ModuleNotFoundError',run.stderr);self.assertFalse((root/'condition.json').exists())

    def test_portable_condition_still_checks_exact_staged_evaluation_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work,path=fixture(root);evaluation=root/'evaluation.json';condition=root/'condition.json'
            original.write_new(evaluation,completed_evaluation(work['heston_target']));portable.bind_heston(path,evaluation,condition)
            record=original.read(condition);staged=root/'staged.json';staged.write_bytes(evaluation.read_bytes());evaluation.unlink()
            portable.validate_condition(work,path,record,staged)
            staged.write_text(staged.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'evaluation changed'):portable.validate_condition(work,path,record,staged)

if __name__=='__main__':unittest.main()
