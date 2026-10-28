"""One config, one bounded Oscar bank campaign; native collection APIs do the work.

Run from the frozen producer checkout using its locked Python environment.
This operator is deliberately independent of the already-running campaigns.
"""

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
ACTIVE = {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED", "RESIZING",
          "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD", "SIGNALING", "STAGE_OUT"}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_config(path):
    path = Path(path).resolve()
    c = json.loads(path.read_text())
    c.setdefault("producer_repo", str(REPO))
    for name in ("task_config", "collection", "snapshot", "campaign_root", "producer_repo"):
        p = Path(c[name]).expanduser()
        c[name] = str(p if p.is_absolute() else (path.parent / p).resolve())
    for name in ("train", "chunk", "max_workers", "worker_wall_seconds", "memory_gib", "final_wall_seconds"):
        if type(c[name]) is not int or c[name] <= 0:
            raise ValueError(name + " must be a positive integer")
    if (not isinstance(c["cpu_hours"], (int, float)) or not math.isfinite(c["cpu_hours"])
            or c["cpu_hours"] <= 0 or c.get("seed_policy") not in ("native_collection", "certificate")):
        raise ValueError("positive CPU-hour ceiling and explicit seed policy required")
    c.setdefault("prior_cpu_seconds", 0)
    c.setdefault("cleanup_seconds", 240)
    c.setdefault("poll_seconds", 60)
    c.setdefault("controller_wall_hours", 48)
    c.setdefault("max_array_size", 1000)
    c.setdefault("python", sys.executable)
    if any(c[n] < 0 for n in ("prior_cpu_seconds", "cleanup_seconds")):
        raise ValueError("negative accounting allowance")
    if c["poll_seconds"] < 1 or c["controller_wall_hours"] <= 0:
        raise ValueError("positive polling/controller limits required")
    if type(c["max_array_size"]) is not int or c["max_array_size"] <= 0:
        raise ValueError("positive integer max_array_size required")
    if c["seed_policy"] == "certificate":
        certificate = Path(c["seed_certificate"]).expanduser()
        c["seed_certificate"] = str(certificate if certificate.is_absolute() else (path.parent / certificate).resolve())
    return c


def check_certificate(c, collection, target):
    """An external audit is optional and explicit; no historical scans are hidden."""
    if c["seed_policy"] == "native_collection":
        return {"scope": "native within-collection pricing-seed audit only; no cross-collection certificate requested"}
    p = Path(c["seed_certificate"])
    if sha(p) != c["seed_certificate_sha256"]:
        raise ValueError("seed certificate changed")
    x = json.loads(p.read_text())
    if (x.get("format") != "calibration_cross_collection_seed_certificate_v1" or x.get("passed") is not True
            or x.get("collection_id") != collection["collection_id"]
            or x.get("rng_protocol") != collection["rng_protocol"]
            or x.get("task_fingerprint") != collection["task_fingerprint"]
            or x.get("root_seed") != target["root_seed"]
            or any(x.get("covered_splits", {}).get(s, -1) < n for s, n in target["splits"].items())):
        raise ValueError("seed certificate does not cover this exact identity and requested IDs")
    return {"scope": "reused cross-collection certificate", "path": str(p), "sha256": sha(p)}


class Slurm:
    def __init__(self, repo=REPO):
        self.repo = repo

    def submit(self, argv):
        result = subprocess.run(argv, cwd=self.repo, capture_output=True, text=True, timeout=60)
        identifier = result.stdout.strip().split(";")[0]
        if result.returncode or not identifier.isdigit():
            raise RuntimeError("sbatch response is failed or ambiguous: " + result.stderr.strip())
        return identifier

    def accounting(self, attempts, started):
        if not attempts:
            return {}
        ids = ",".join(a["job_id"] for a in attempts)
        # Ledger times are UTC; sacct interprets date-only filters in cluster
        # local time. Include the previous date so evening local jobs whose
        # UTC timestamp crossed midnight are not silently omitted. Exact job
        # IDs still restrict the returned allocations.
        since = (datetime.fromisoformat(started).date() - timedelta(days=1)).isoformat()
        result = subprocess.run(["sacct", "-S", since, "-j", ids, "-n", "-P",
                                 "--format=JobID%64,State%40,ElapsedRaw,AllocCPUS,ExitCode"],
                                capture_output=True, text=True, timeout=45, check=True)
        rows = {}
        for line in result.stdout.splitlines():
            x = line.split("|")
            if len(x) >= 5 and x[0] and "." not in x[0]:
                rows[x[0]] = {"state": x[1].split()[0], "elapsed": int(x[2]),
                              "cpus": int(x[3]), "exit_code": x[4]}
        return rows


def task_ids(attempt):
    if attempt["role"] == "workers":
        return [attempt["job_id"] + "_" + str(i - attempt["array_offset"]) for i in attempt["indices"]]
    return [attempt["job_id"]]


def exposure(c, attempts, rows):
    actual, outstanding, workers = c["prior_cpu_seconds"], 0, 0
    for a in attempts:
        if a["state"] != "submitted":
            raise RuntimeError("unreconciled submission attempt; no automatic retry")
        for identifier in task_ids(a):
            r = rows.get(identifier)
            if r and r["state"] == "COMPLETED":
                if r["exit_code"] != "0:0" or r["cpus"] < 1:
                    raise RuntimeError("invalid completed accounting: " + identifier)
                actual += r["elapsed"] * r["cpus"]
            else:
                if r and r["state"] not in ACTIVE:
                    raise RuntimeError("failed or unrecognized task: " + identifier + " " + r["state"])
                outstanding += a["wall_seconds"] + c["cleanup_seconds"]
                workers += a["role"] == "workers"
    reserve = 0 if any(a["role"] == "final" for a in attempts) else c["final_wall_seconds"] + c["cleanup_seconds"]
    return {"actual_cpu_seconds": actual, "outstanding_cpu_seconds": outstanding,
            "final_reserve_cpu_seconds": reserve, "exposure_cpu_seconds": actual + outstanding + reserve,
            "nonterminal_workers": workers}


def submit(c, ledger, root, scheduler, rows, role, indices=None):
    repo = Path(c.get("producer_repo", REPO))
    b = exposure(c, ledger["attempts"], rows)
    count = len(indices) if indices is not None else 1
    wall = c["worker_wall_seconds"] if role == "workers" else c["final_wall_seconds"]
    extra = count * (wall + c["cleanup_seconds"]) if role == "workers" else 0  # final cost already reserved
    if b["exposure_cpu_seconds"] + extra > c["cpu_hours"] * 3600:
        raise RuntimeError("prospective campaign CPU-hour ceiling exceeded")
    if role == "workers":
        old = {i for a in ledger["attempts"] for i in a.get("indices", [])}
        if not indices or old.intersection(indices) or b["nonterminal_workers"] + count > c["max_workers"]:
            raise RuntimeError("duplicate indices or worker limit exceeded")
        if indices != list(range(indices[0], indices[0] + count)) or count > c["max_array_size"]:
            raise RuntimeError("a wave must be contiguous and fit the configured Slurm array size")
    elif any(a["role"] == "final" for a in ledger["attempts"]):
        raise RuntimeError("final publication already attempted")
    argv = ["sbatch", "--parsable", "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--no-requeue",
            "--mem=" + str(c["memory_gib"]) + "G",
            "--time=" + f"{wall//86400}-{wall%86400//3600:02}:{wall%3600//60:02}:{wall%60:02}",
            "--account=" + c["account"], "--partition=" + c["partition"], "--chdir=" + str(repo),
            "--output=" + str(root / "logs/%A_%a.out"), "--job-name=sbi4f-bank"]
    if role == "workers":
        argv += ["--array=0-" + str(count-1) + "%" + str(min(count, c["max_workers"])),
                 str(repo / "cluster/calibration_collection_array.sbatch"), c["python"], str(root / "jobs.json"), str(indices[0])]
    else:
        env = ["env", "PYTHONPATH=" + str(repo / "src"), "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
               "MKL_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1"]
        command = env + [c["python"], str(repo / "scripts/calibration_bank_collection.py"), "freeze",
                         "--collection", c["collection"], "--train", str(c["train"]), "--out", c["snapshot"]]
        argv += ["--wrap=" + shlex.join(command)]
    a = {"role": role, "state": "submission_attempted", "utc": now(), "wall_seconds": wall, "argv": argv}
    if indices is not None:
        a["indices"] = indices
        a["array_offset"] = indices[0]
    ledger["attempts"].append(a)
    save(root / "execution.json", ledger)  # Persist before touching the scheduler.
    try:
        a.update(job_id=scheduler.submit(argv), state="submitted")
    except BaseException:
        a["state"] = "unknown_submission"
        save(root / "execution.json", ledger)
        raise
    save(root / "execution.json", ledger)


def run(c, *, scheduler=None, prepared_root=False, sleep=time.sleep):
    if c.get("expected_operator_sha256") is not None and sha(__file__) != c["expected_operator_sha256"]:
        raise ValueError("detached operator changed before startup")
    repo = Path(c.get("producer_repo", REPO)).resolve()
    sys.path.insert(0, str(repo / "src"))
    from sbi4finance.calibration import bank, bank_collection as bc
    if not Path(bank.__file__).resolve().is_relative_to(repo / "src"):
        raise ValueError("Python already imported another producer checkout; run in a fresh process")
    scheduler = scheduler or Slurm(repo)
    root = Path(c["campaign_root"])
    if not prepared_root:
        root.mkdir(parents=True, exist_ok=False)
    lock = (root / "campaign.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lock.close()
        raise
    path = root / "execution.json"
    if path.exists():
        lock.close()
        raise FileExistsError("existing campaign requires explicit reconciliation; use a new campaign root")
    (root / "logs").mkdir(exist_ok=True)
    ledger = {"format": "calibration_bank_campaign_v1", "state": "preparing", "started_utc": now(),
              "pid": os.getpid(), "config": c, "attempts": [], "scientific_acceptance": False}
    save(path, ledger)
    started = time.monotonic()
    try:
        task_config = json.loads(Path(c["task_config"]).read_text())
        revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if c.get("expected_git_revision") is not None and revision != c["expected_git_revision"]:
            raise ValueError("producer Git revision differs from campaign declaration")
        collection = Path(c["collection"])
        if collection.exists():
            _, record = bc._read_collection(collection, current=True)
            bc._compatible(task_config, record)
        else:
            record = bc.create_collection(collection, task_config)
        if c.get("expected_runtime") is not None and bank.versions() != c["expected_runtime"]:
            raise ValueError("runtime differs from campaign declaration")
        plan = bc.plan_extension(collection, c["train"], c["chunk"])
        plan.update(collection=str(collection), collection_sha256=sha(collection / "collection.json"))
        plan["jobs"] = [{k: j[k] for k in ("split", "start", "stop", "job_sha256")} for j in plan["jobs"]]
        save(root / "jobs.json", plan)
        seed_scope = check_certificate(c, record, plan["configuration"])
        frozen = {str(p): sha(p) for p in (Path(__file__), Path(c["task_config"]), root / "jobs.json", collection / "collection.json")}
        if c["seed_policy"] == "certificate":
            frozen[c["seed_certificate"]] = sha(c["seed_certificate"])
        ledger.update(state="generation_active", collection_id=record["collection_id"], source_sha256=record["source_sha256"],
                      runtime=record["versions"], producer_git_revision=revision, producer_repo=str(repo),
                      seed_policy=seed_scope, frozen_sha256=frozen, planned_workers=len(plan["jobs"]))
        save(path, ledger)
        while time.monotonic() - started < c["controller_wall_hours"] * 3600:
            if any(sha(p) != digest for p, digest in frozen.items()):
                raise ValueError("frozen campaign input changed")
            rows = scheduler.accounting(ledger["attempts"], ledger["started_utc"])
            b = exposure(c, ledger["attempts"], rows)
            if b["exposure_cpu_seconds"] > c["cpu_hours"] * 3600:
                raise RuntimeError("campaign accounting exceeds its CPU-hour ceiling")
            submitted = {i for a in ledger["attempts"] for i in a.get("indices", [])}
            done = {i for a in ledger["attempts"] for i in a.get("indices", [])
                    if rows.get(a["job_id"] + "_" + str(i - a["array_offset"]), {}).get("state") == "COMPLETED"}
            missing = [i for i in range(len(plan["jobs"])) if i not in submitted]
            ledger.update(last_poll_utc=now(), budget=b, completed_workers=len(done), missing_workers=len(missing))
            if missing:
                available = math.floor((c["cpu_hours"] * 3600 - b["exposure_cpu_seconds"]) /
                                       (c["worker_wall_seconds"] + c["cleanup_seconds"]))
                n = min(len(missing), c["max_workers"] - b["nonterminal_workers"], c["max_array_size"], available)
                if n > 0:
                    submit(c, ledger, root, scheduler, rows, "workers", missing[:n])
                elif b["nonterminal_workers"] == 0:
                    ledger["state"] = "stopped_at_budget_ceiling"
                    break
            elif len(done) == len(plan["jobs"]):
                finals = [a for a in ledger["attempts"] if a["role"] == "final"]
                if not finals:
                    submit(c, ledger, root, scheduler, rows, "final")
                    ledger["state"] = "final_publication_submitted"
                elif rows.get(finals[0]["job_id"], {}).get("state") == "COMPLETED":
                    manifest = json.loads((Path(c["snapshot"]) / "manifest.json").read_text())
                    if manifest["collection_id"] != record["collection_id"] or manifest["configuration"] != plan["configuration"]:
                        raise ValueError("published snapshot identity differs")
                    ledger.update(state="completed_checked_snapshot", snapshot=c["snapshot"],
                                  snapshot_manifest_sha256=sha(Path(c["snapshot"]) / "manifest.json"))
                    break
            save(path, ledger)
            sleep(c["poll_seconds"])
        else:
            ledger["state"] = "stopped_at_controller_wall_limit"
        save(path, ledger)
    except BaseException as error:
        ledger.update(state="stopped_for_review", error_type=type(error).__name__, error=str(error), stopped_utc=now())
        save(path, ledger)
        raise
    finally:
        lock.close()
    return ledger


def detach(c, *, popen=subprocess.Popen):
    c = {**c, "expected_operator_sha256": sha(__file__)}
    root = Path(c["campaign_root"])
    root.mkdir(parents=True, exist_ok=False)
    config = root / "campaign.json"
    save(config, c)
    with (root / "operator.log").open("ab") as log:
        child = popen([sys.executable, str(Path(__file__).resolve()), "--config", str(config), "--run"],
                      cwd=c.get("producer_repo", REPO), stdin=subprocess.DEVNULL,
                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {"format": "calibration_bank_campaign_launch_v1", "state": "launched_not_yet_checked",
               "pid": child.pid, "utc": now(), "config_sha256": sha(config), "operator_sha256": sha(__file__),
               "execution": str(root / "execution.json"), "log": str(root / "operator.log")}
    save(root / "launch.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--run", action="store_true", help=argparse.SUPPRESS)
    a = parser.parse_args()
    if a.detach and a.run:
        parser.error("choose detached launcher or child, not both")
    c = read_config(a.config)
    result = detach(c) if a.detach else run(c, prepared_root=a.run)
    print(json.dumps({k: result[k] for k in ("state", "pid") if k in result}))


if __name__ == "__main__":
    main()
