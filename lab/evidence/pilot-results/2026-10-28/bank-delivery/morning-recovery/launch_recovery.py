"""Start the reviewed recovery controller exactly once, detached from SSH."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

root = Path('/oscar/scratch/afengler/sbi4finance-bank-delivery-20261028/morning-recovery-20261028')
path = root / 'recovery-plan.json'; plan = json.loads(path.read_text())
receipt = {'state': 'launch_attempted', 'attempted_utc': datetime.now(timezone.utc).isoformat(),
           'plan_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'automatic_retry': False}
with (root / 'launch.json').open('x') as f:
    json.dump(receipt, f, indent=2)
with (root / 'controller.log').open('ab') as log:
    child = subprocess.Popen([plan['python'], '-u', str(root / 'joint_recovery_controller.py'), '--plan', str(path)],
                             cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
receipt.update(state='launched', controller_pid=child.pid)
temporary = root / 'launch.json.tmp'; temporary.write_text(json.dumps(receipt, indent=2) + '\n')
temporary.replace(root / 'launch.json'); print(json.dumps(receipt), flush=True)
