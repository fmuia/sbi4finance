"""One explicit, reviewed replacement of a metadata-only dependency waiter."""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def now():
    return datetime.now(timezone.utc).isoformat()


if socket.gethostname() != "login010":
    raise RuntimeError("wrong host; no process was stopped or launched")
base = Path("/oscar/scratch/afengler/sbi4finance-overnight-20261028/followons")
root = base / "joint-tail-gate-20261028"
plan_path = root / "joint_100k_tail_gate_20261028.json"
plan = json.loads(plan_path.read_text())
route = plan["routes"]["joint"]
config = json.loads(Path(route["campaign_config"]).read_text())
binding = route["native_recovery"]
assert set(plan["routes"]) == {"joint"}
assert config["cpu_hours"] == 3000 and config["prior_cpu_seconds"] == 0
assert config["max_workers"] == 200 and config["train"] == 100000
assert config["exclude_nodes"] == ["node2338", "node2339", "node2336", "node2344", "node2346"]
assert not Path(config["campaign_root"]).exists(), "Joint100k has artifacts: reconcile before launching"
assert binding["cpu_hours"] == 1500 and binding["prior_cpu_seconds"] == 3141589
assert all(sha(p) == digest for p, digest in binding["frozen_provenance"].items())
native_config = json.loads(Path(binding["campaign_config"]).read_text())
assert native_config["prior_cpu_seconds"] == binding["prior_cpu_seconds"]
assert native_config["cpu_hours"] == binding["cpu_hours"]
assert native_config["snapshot"] == binding["snapshot"]

oldroot = base / "joint-recovery-gate-20261028/waiter"
old = json.loads((oldroot / "execution.json").read_text())
assert old["pid"] == 2629343 and set(old["routes"]) == {"joint"}
assert old["routes"]["joint"]["state"] == "waiting"
assert not old["routes"]["joint"].get("attempt")
command = Path("/proc/2629343/cmdline").read_bytes().replace(b"\0", b" ").decode()
assert str(base / "joint-recovery-gate-20261028/arm_bank_extensions.py") in command
assert str(oldroot / "plan.json") in command

known_config = Path(route["after_campaign"]["config"])
known = json.loads(known_config.read_text())
known_root = Path(known["campaign_root"])
known_execution = json.loads((known_root / "execution.json").read_text())
known_manifest = Path(known["snapshot"]) / "manifest.json"
assert known_execution["state"] == "completed_checked_snapshot"
assert known_execution["snapshot_manifest_sha256"] == sha(known_manifest)
assert sha(known_manifest) == "f1bfa42f7bf2acba3daaffb263b5bdeb60b5b12ebdfc267c5c5cd6eb7e08c313"
untouched = {str(p): sha(p) for p in [known_config, known_root / "execution.json", known_manifest,
    base / "accounting-recovery-20261028/run_bank_campaign.py", known_root / "campaign.json"]}
for name in ["execution.json", "launch.json", "plan.json"]:
    with (root / ("replaced-waiter-" + name)).open("xb") as stream:
        stream.write((oldroot / name).read_bytes())

record = {"state": "metadata_replacement_prepared", "utc": now(), "old_pid": 2629343,
    "host": socket.gethostname(), "known_completed_files_sha256": untouched,
    "plan_sha256": sha(plan_path), "waiter_sha256": sha(root / "arm_bank_extensions.py"),
    "launcher_sha256": sha(plan["launcher"]), "campaign_config_sha256": sha(route["campaign_config"]),
    "native_config_sha256": sha(binding["campaign_config"]), "frozen_provenance": binding["frozen_provenance"],
    "new_numerical_jobs_by_deployer": 0, "automatic_retry": False, "budget_change": False}
create(root / "deployment-prospective.json", record)
os.kill(2629343, signal.SIGTERM)
for _ in range(40):
    stat = Path("/proc/2629343/stat")
    if not stat.exists() or stat.read_text().split()[2] == "Z":
        break
    time.sleep(0.1)
else:
    raise RuntimeError("Old metadata waiter still active; no replacement launched")
assert all(sha(p) == digest for p, digest in untouched.items())
create(root / "metadata-waiter-stop.json", {"state": "stopped_metadata_only", "utc": now(), "pid": 2629343})
argv = [plan["python"], str(root / "arm_bank_extensions.py"), "--plan", str(plan_path), "--detach"]
record.update(state="launch_attempted", argv=argv)
create(root / "launch-attempt.json", record)
try:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    record.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
    if result.returncode:
        raise RuntimeError("Metadata waiter launch failed; no automatic retry")
    receipt = json.loads((root / "waiter/launch.json").read_text())
    os.kill(receipt["pid"], 0)
    for _ in range(40):
        path = root / "waiter/execution.json"
        if path.exists():
            break
        time.sleep(0.25)
    state = json.loads(path.read_text())
    assert state["state"] == "armed" and state["routes"]["joint"]["state"] == "waiting"
    assert not state["routes"]["joint"].get("attempt")
    assert all(sha(p) == digest for p, digest in untouched.items())
    record.update(state="joint_tail_waiter_armed_verified_alive", pid=receipt["pid"],
        routes=state["routes"], launch_receipt_sha256=sha(root / "waiter/launch.json"), verified_utc=now())
except BaseException as error:
    record.update(state="unknown_or_failed_launch", error_type=type(error).__name__, error=str(error))
    create(root / "deployment-launch.json", record)
    raise
create(root / "deployment-launch.json", record)
print(json.dumps({k: record[k] for k in ["state", "pid", "host", "routes", "launcher_sha256"]}))
