"""Small no-submission checks for the one-off handoff."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

repo=Path(__file__).resolve().parents[4]
spec=importlib.util.spec_from_file_location('handoff',repo/'cluster/start_joint100k_20261029.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
plan=json.loads((Path(__file__).parent/'handoff-plan.json').read_text())
rows={j:[j,'COMPLETED','0:0',str(t),'4','cpu=4,gres/gpu=1'] for j,t in plan['pool_jobs'].items()}
with patch.object(h,'accounting',return_value=rows):
    cost=h.pool(plan);assert (cost['full_cumulative_cpu_seconds'],cost['full_cumulative_gpu_seconds'])==(30992,7748)
    rows['8060582'][1]='TIMEOUT'
    try:h.pool(plan)
    except ValueError:pass
    else:raise AssertionError('Failed pool allocation was accepted')
    rows['8060582'][1]='COMPLETED';del rows['8060582']
    try:h.pool(plan)
    except TimeoutError:pass
    else:raise AssertionError('Missing accounting was accepted')
with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary);p={**plan,'handoff_root':str(root/'waiter'),'training_root':str(root/'fit')}
    source=root/'source-plan.json';source.write_text(json.dumps(p))
    calls=[]
    def child(*args,**kwargs):
        assert (root/'waiter/launch-attempt.json').exists() and kwargs['start_new_session']
        calls.append(args);return SimpleNamespace(pid=123)
    with patch.object(h,'sources'),patch.object(h,'pool'),patch.object(h.subprocess,'Popen',side_effect=child),contextlib.redirect_stdout(io.StringIO()):
        h.detach(p,source)
        try:h.detach(p,source)
        except FileExistsError:pass
        else:raise AssertionError('Duplicate detached handoff accepted')
    assert len(calls)==1
    state=root/'native.json';p['bank_execution']=str(state)
    state.write_text(json.dumps({'state':'generation_active'}));assert h.native_ready(p) is None
    state.write_text(json.dumps({'state':'stopped_for_review'}))
    try:h.native_ready(p)
    except ValueError:pass
    else:raise AssertionError('Stopped native bank accepted')
print('PASS: pool caps, failed/missing accounting, pre-child receipt, duplicate guard, native wait/failure. No jobs submitted.')
