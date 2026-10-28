#!/usr/bin/env python3
"""Ask Oscar what *this* user may actually schedule, and write it down.

What differs per person is which SLURM associations they hold, and therefore
how much they can run at once. The numbers that matter live in the QOS, not the
partition (`scontrol show partition` reports MaxTime=UNLIMITED for everything).

    pixi run python cluster/discover_cluster.py --ssh-host oscar   # from a laptop
    pixi run python cluster/discover_cluster.py                    # on a login node

Writes cluster/oscar.local.json (gitignored: it is YOUR allocation, not the lab's).

A "lane" is one (account, partition, QOS) triple you may submit to. Its cap is
MaxTRESPU on the QOS, which SLURM applies per QOS, so two lanes are two
independent budgets.

Adapted from LAN_pipeline_minimal/scripts/discover_cluster.py (MIT, A. Fengler).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "oscar.local.json"

# QOS that exist on every account but are not for real work: debug queues are
# time-boxed, vnc is for interactive desktops.
NON_COMPUTE_QOS = {"debug", "gpu-debug", "vnc"}

ASSOC_QUERY = "sacctmgr -nP show associations user=$USER format=Account,Partition,QOS"
QOS_QUERY = "sacctmgr -nP show qos where name={names} format=Name,Priority,MaxWall,MaxTRESPU,MaxSubmitJobsPU"


def run(command: str, ssh_host: str | None) -> str:
    """Run a read-only cluster query, locally or over ssh."""
    argv = ["ssh", "-o", "BatchMode=yes", ssh_host, command] if ssh_host else ["bash", "-lc", command]
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0:
        hint = (
            "Open one interactive `ssh oscar` first (password + Duo); the ControlMaster socket then serves this."
            if ssh_host
            else "Run this on a login node, or pass --ssh-host oscar."
        )
        raise RuntimeError(f"`{command}` failed ({result.returncode}): {result.stderr.strip()[:200]}\n{hint}")
    return result.stdout


def parse_cap(tres: str) -> dict:
    """`cpu=208,mem=2500G,gres/gpu=75` -> {'cores': 208, 'mem': '2500G', 'gpus': 75}."""
    cap: dict = {}
    for item in filter(None, tres.split(",")):
        key, _, value = item.partition("=")
        if key == "cpu":
            cap["cores"] = int(value)
        elif key == "mem":
            cap["mem"] = value
        elif key.startswith("gres/gpu"):
            cap["gpus"] = int(value)
    return cap


def parse_lanes(associations: str, qos_rows: str) -> list[dict]:
    """Join `sacctmgr show associations` rows with their QOS caps; best lane first."""
    lanes = []
    for line in associations.strip().splitlines():
        fields = line.split("|")
        if len(fields) < 3 or not fields[1]:
            continue  # rows without a partition are the account-level parent
        lanes.append({"account": fields[0], "partition": fields[1], "qos": fields[2]})

    qos_info = {}
    for line in qos_rows.strip().splitlines():
        f = (line.split("|") + [""] * 5)[:5]
        qos_info[f[0]] = {
            "priority": int(f[1]) if f[1].isdigit() else 0,
            "max_walltime": f[2] or None,
            "cap": parse_cap(f[3]),
            "max_submit_jobs": int(f[4]) if f[4].isdigit() else None,
        }

    for lane in lanes:
        info = qos_info.get(lane["qos"], {})
        lane["priority"] = info.get("priority", 0)
        lane["max_walltime"] = info.get("max_walltime")
        lane.update(info.get("cap", {}))
        if info.get("max_submit_jobs"):
            lane["max_submit_jobs"] = info["max_submit_jobs"]

    lanes.sort(key=lambda x: (-x["priority"], -x.get("cores", 0)))
    return lanes


def discover(ssh_host: str | None) -> list[dict]:
    associations = run(ASSOC_QUERY, ssh_host)
    qos_names = sorted({line.split("|")[2] for line in associations.strip().splitlines() if line.count("|") >= 2})
    if not qos_names:
        raise RuntimeError("No associations found. Is this the right cluster account?")
    return parse_lanes(associations, run(QOS_QUERY.format(names=",".join(qos_names)), ssh_host))


def suggest_job_defaults(lanes: list[dict]) -> dict:
    """Simulation is CPU array work on the best CPU lane; training wants one GPU on the best GPU lane."""
    usable = [x for x in lanes if x["qos"] not in NON_COMPUTE_QOS and x.get("cores")]
    cpu = [x for x in usable if not x.get("gpus")]
    gpu = [x for x in usable if x.get("gpus")]
    defaults = {}
    if cpu:
        defaults["simulate"] = {"account": cpu[0]["account"], "partition": cpu[0]["partition"], "cores": 1, "mem": "8G", "time": "02:00:00"}
    if gpu:
        defaults["train"] = {"account": gpu[0]["account"], "partition": gpu[0]["partition"], "gpus": 1, "cores": 8, "mem": "64G", "time": "04:00:00"}
    return defaults


def selftest() -> None:
    assoc = "carney-x-condo|batch|carney-condo\ncarney-x-condo|gpu-he|carney-gcondo\ndefault||normal\ndefault|debug|debug\n"
    qos = "carney-condo|10000||cpu=208,mem=2500G|\ncarney-gcondo|10000||cpu=160,mem=1920G,gres/gpu=75|1200\ndebug|0|01:00:00||\n"
    lanes = parse_lanes(assoc, qos)
    assert [x["partition"] for x in lanes] == ["batch", "gpu-he", "debug"], lanes
    assert lanes[1] == {"account": "carney-x-condo", "partition": "gpu-he", "qos": "carney-gcondo", "priority": 10000,
                        "max_walltime": None, "cores": 160, "mem": "1920G", "gpus": 75, "max_submit_jobs": 1200}, lanes[1]
    defaults = suggest_job_defaults(lanes)
    assert defaults["simulate"]["partition"] == "batch" and defaults["train"]["partition"] == "gpu-he", defaults
    print("ok: discover_cluster parsing")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ssh-host", help="Run the queries over ssh (e.g. oscar).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--print", action="store_true", help="Print instead of writing.")
    parser.add_argument("--selftest", action="store_true", help="Check the parsing on canned sacctmgr output.")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return 0

    try:
        lanes = discover(args.ssh_host)
    except RuntimeError as e:
        print(f"discovery failed: {e}", file=sys.stderr)
        return 1

    print(f"Found {len(lanes)} lanes:", file=sys.stderr)
    for lane in lanes:
        gpus = f", {lane['gpus']} gpus" if lane.get("gpus") else ""
        print(f"  {lane['account']:26s} {lane['partition']:10s} qos={lane['qos']:16s} "
              f"priority={lane['priority']:<6d} {lane.get('cores', '?')} cores{gpus}", file=sys.stderr)

    text = json.dumps({"lanes": lanes, "job_defaults": suggest_job_defaults(lanes)}, indent=2) + "\n"
    if args.print:
        print(text)
    else:
        args.output.write_text(text)
        print(f"wrote {args.output}; copy account/partition into cluster/train.sbatch or pass them to sbatch", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
