"""Read-only, host-specific detached waiter heartbeat evidence."""
from datetime import datetime, timezone
import json
from pathlib import Path
import socket

assert socket.gethostname() == "login010"
root = Path("/oscar/scratch/afengler/sbi4finance-overnight-20261028/followons/joint-tail-gate-20261028")
launch = json.loads((root / "waiter/launch.json").read_text())
run = json.loads((root / "waiter/execution.json").read_text())
pid = launch["pid"]
proc = Path("/proc") / str(pid)
assert run["pid"] == pid and proc.exists()
command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
assert str(root / "arm_bank_extensions.py") in command
assert not run["routes"]["joint"].get("attempt")
print(json.dumps({"checked_utc": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname(),
    "pid": pid, "ppid": int((proc / "stat").read_text().split()[3]), "command": command,
    "started_utc": run["started_utc"], "last_poll_utc": run.get("last_poll_utc"),
    "state": run["state"], "routes": run["routes"],
    "joint100k_campaign_exists": Path("/oscar/scratch/afengler/sbi4finance-overnight-20261028/followons/joint-100k").exists()}, indent=2))
