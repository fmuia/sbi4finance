"""Launch the approved finite controller exactly once, detached from SSH."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/oscar/scratch/afengler/sbi4finance-bank-delivery-20261028')
plan_path = ROOT / 'plan.json'
plan = json.loads(plan_path.read_text())
receipt_path = ROOT / 'controller-launch.json'
receipt = {'state': 'launch_attempted', 'attempted_utc': datetime.now(timezone.utc).isoformat(),
           'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
           'automatic_retry': False}
with receipt_path.open('x') as f:
    json.dump(receipt, f, indent=2)
with (ROOT / 'controller.log').open('ab') as log:
    child = subprocess.Popen([plan['python'], '-u', str(ROOT / 'bank_delivery_controller.py'),
                              '--plan', str(plan_path)], cwd=ROOT, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
receipt.update(state='launched', pid=child.pid)
temporary = ROOT / 'controller-launch.json.tmp'
temporary.write_text(json.dumps(receipt, indent=2) + '\n')
temporary.replace(receipt_path)
print(json.dumps(receipt), flush=True)
