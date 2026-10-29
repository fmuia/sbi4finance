import json
import os
from pathlib import Path
import subprocess
import time

repo = Path('/oscar/scratch/afengler/sbi4finance-overnight-20261028/repo')
root = Path('/oscar/scratch/afengler/sbi4finance-post-bank-baselines-20261028')
python = '/oscar/scratch/afengler/sbi4finance-full-core-20261026-wy2_j3o4/repo/.pixi/envs/core-gpu/bin/python'
operator = repo / 'cluster/submit_full_surface_backbones_20261028.py'
subprocess.run([python, str(operator), 'initialize', '--root', str(root), '--repo', str(repo),
                '--revision', '00faaefe946c8e8fef3b55ff3156c2d0db61ffb3', '--python', python], check=True)
with (root / 'operator.log').open('ab') as log:
    child = subprocess.Popen([python, str(operator), 'watch', '--root', str(root)], cwd=repo,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
receipt = {'format': 'post_bank_launch_v1', 'pid': child.pid, 'root': str(root),
           'source_revision': '00faaefe946c8e8fef3b55ff3156c2d0db61ffb3',
           'scope': 'known50k/joint20k intermediate posterior/pricer/validation; no reserved test'}
(root / 'launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
time.sleep(2)
os.kill(child.pid, 0)
receipt['process_alive'] = True
receipt['execution'] = json.loads((root / 'execution.json').read_text())
print(json.dumps(receipt, indent=2))
