"""Export the combined tree's cached rough review with isolated inputs."""
import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
root, out = args.root.resolve(), args.out.resolve()
out.mkdir(parents=True, exist_ok=False)
isolated = out / 'isolated'
guard = out / 'guard'
guard.mkdir()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def git(*arguments):
    return subprocess.check_output(['git', '-C', str(root), *arguments], text=True).strip()

revision = {'commit': git('rev-parse', 'HEAD'), 'tree': git('rev-parse', 'HEAD^{tree}'),
            'dirty': bool(git('status', '--porcelain', '--untracked-files=normal'))}
historical = json.loads((root / 'docs/results/rough-heldout-test/notebook-verification.json').read_text())
paths = set(historical['source_and_artifact_sha256'])
paths.update(str(path.relative_to(root)) for path in (root / 'src').rglob('*.py'))
hashes = {}
for name in sorted(paths):
    destination = isolated / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / name, destination)
    hashes[name] = sha(destination)
assert not (isolated / 'data').exists()
assert not (isolated / 'checkpoints').exists()
assert not any(p.name in {'banks', 'bank'} for p in isolated.rglob('*'))

(guard / 'sitecustomize.py').write_text('''import atexit
import json
import os
from pathlib import Path
import sys

FORBIDDEN = {"bayesflow", "keras", "jax", "jaxlib", "tensorflow", "torch"}
class CachedOnly:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN:
            raise RuntimeError("cached review forbids framework: " + fullname)
sys.meta_path.insert(0, CachedOnly())
def local(address):
    return address in {None, "localhost", "127.0.0.1", "::1", ""} or (isinstance(address, str) and address.startswith("127."))
def offline(event, arguments):
    if event == "socket.getaddrinfo" and not local(arguments[0]):
        raise RuntimeError("external network disabled: " + event)
    if event in {"socket.connect", "socket.sendto"}:
        address = arguments[-1]
        if isinstance(address, tuple) and not local(address[0]):
            raise RuntimeError("external network disabled: " + event)
sys.addaudithook(offline)
Path(os.environ["ROUGH_GUARD_LOGS"]).mkdir(exist_ok=True)
Path(os.environ["ROUGH_GUARD_LOGS"], str(os.getpid()) + "-installed.json").write_text(json.dumps({"guard_installed": True}))
def record():
    modules = {name: str(module.__file__) for name, module in sys.modules.items()
               if name.startswith("sbi4finance") and getattr(module, "__file__", None)}
    data = {"package_modules": modules, "frameworks_imported": sorted(FORBIDDEN.intersection(sys.modules))}
    Path(os.environ["ROUGH_GUARD_LOGS"], str(os.getpid()) + "-exit.json").write_text(json.dumps(data))
atexit.register(record)
''')
env = dict(os.environ, PYTHONPATH=str(guard) + os.pathsep + str(isolated / 'src'),
           MPLBACKEND='Agg', MPLCONFIGDIR=str(out / 'matplotlib'),
           ROUGH_GUARD_LOGS=str(out / 'guard-logs'), KERAS_BACKEND='jax')
preflight = '''import importlib,json,pathlib,socket,sys
import sbi4finance
assert pathlib.Path(sbi4finance.__file__).resolve().is_relative_to(pathlib.Path.cwd() / 'src')
blocked=[]
for name in ('bayesflow','keras','jax'):
    try: importlib.import_module(name)
    except RuntimeError: blocked.append(name)
    else: raise AssertionError(name)
for method in ('connect','connect_ex'):
    with socket.socket() as probe:
        try: getattr(probe, method)(('203.0.113.1',443))
        except RuntimeError: blocked.append(method)
        else: raise AssertionError(method)
try: socket.getaddrinfo('example.com',443)
except RuntimeError: blocked.append('dns')
else: raise AssertionError('dns')
with socket.socket() as server:
    server.bind(('127.0.0.1',0));server.listen()
    with socket.socket() as client:
        client.connect(server.getsockname())
        accepted,_=server.accept();accepted.close()
print(json.dumps({'isolated_package':sbi4finance.__file__,'blocked':blocked,'local_ipc_allowed':True}))
'''

def run(arguments, logfile):
    start=time.perf_counter()
    with (out/logfile).open('w') as log:
        completed=subprocess.run([sys.executable,*arguments],cwd=isolated,env=env,stdout=log,stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f'{logfile}: command failed ({completed.returncode}); see {out/logfile}')
    return {'argv':[sys.executable,*arguments],'elapsed_seconds':time.perf_counter()-start,'exit_code':completed.returncode}

commands=[]
commands.append(run(['-c',preflight],'preflight.log'))
commands.append(run(['-m','marimo','check','--strict','notebooks/09_rough_inference.py'],'strict-check.log'))
commands.append(run(['-m','marimo','export','html','notebooks/09_rough_inference.py','-o',str(out/'rough-review.html')],'export.log'))
for name,digest in hashes.items():
    assert sha(isolated/name)==digest,name
for path in (out/'guard-logs').glob('*-exit.json'):
    report=json.loads(path.read_text())
    assert not report['frameworks_imported'],report
    assert all(Path(p).resolve().is_relative_to(isolated/'src') for p in report['package_modules'].values()),report
html=(out/'rough-review.html').read_text()
marker='value: Object.freeze('
config,_=json.JSONDecoder().raw_decode(html[html.index(marker,html.index('__MARIMO_MOUNT_CONFIG__'))+len(marker):].lstrip())
cells=config['session']['cells']
errors=[output for cell in cells for output in cell['outputs'] if output['type']=='error']
assert not errors,errors
rendered=json.dumps([cell['outputs'] for cell in cells])
images=re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',rendered)
assert len(images)==14,f'expected 14 figures, got {len(images)}'
(out/'plots').mkdir()
for index,image in enumerate(images,1):
    (out/'plots'/f'rough-{index:02d}.png').write_bytes(base64.b64decode(image))
result={
    'output_cells':len(cells),'error_outputs':len(errors),'rendered_figure_count':len(images),
    'source_revision':revision,'source_revision_scope':'Git tree is committed HEAD; copied-file hashes identify the executed working copy.',
    'notebook':'notebooks/09_rough_inference.py','strict_marimo_check':'passed',
    'commands':commands,'source_and_artifact_sha256':hashes,
    'environment':{'python':sys.version,'executable':sys.executable,'packages':{name:importlib.metadata.version(name) for name in ['bayesflow','keras','jax','numba','scipy','marimo','numpy']}},
    'external_connects_blocked':True,'external_dns_blocked':True,'localhost_ipc_allowed':True,
    'framework_imports_forbidden':['bayesflow','keras','jax','jaxlib','tensorflow','torch'],
    'market_data_directory_present':False,'checkpoint_directory_present':False,'raw_bank_directory_present':False,
    'scientific_acceptance':False,'reserved_cohort_rerun':False,
    'external_network_guard_sha256':sha(guard/'sitecustomize.py'),'helper_sha256':sha(Path(__file__)),
    'html_sha256':sha(out/'rough-review.html'),
}
(out/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'out':str(out),'revision':revision,'seconds':sum(c['elapsed_seconds'] for c in commands)},indent=2))
