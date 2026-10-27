"""Calibration banks from the command line: configs, job maps, shards, assembly, local runs.

    python scripts/calibration_bank.py config --model rough --curve-prior power_law_walk --train 100000 \
        --validation 5000 --test 7000 --seed 2611 --out configs/tasks/calibration_rough_bank.json
    python scripts/calibration_bank.py job-map --config CONFIG --chunk 50 --out jobs.json
    python scripts/calibration_bank.py shard --job-map jobs.json --task-index 0 --output-root /scratch/shards
    python scripts/calibration_bank.py assemble --config CONFIG --shards /scratch/shards --out /scratch/bank
    python scripts/calibration_bank.py local --config CONFIG --chunk 25 --workers 4 --out /tmp/bank

Every output directory must be new, except that `local --resume` keeps the
finished shards of an interrupted local run and redoes the rest, and that `shard`
is idempotent: a shard with a manifest is kept, a partial one is removed and
redone, so a requeued or resubmitted array task is safe. Shards are independent
and can run on any machine with the same sources; assembly checks identities,
coverage and arrays. The job map records the config's sha256 and task
fingerprint, and every shard refuses to run if the config file has changed;
each shard also writes run.json with the git commit of the checkout it ran in.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import shutil
import socket
import subprocess
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sbi4finance.calibration import bank, tasks  # noqa: E402


def _config(args):
    if args.model == "rough":
        prior = tasks.FULL_BANK_CURVE_PRIOR if args.curve_prior == "power_law_walk" else None
        task = tasks.rough_task(curve_prior=prior)
    else:
        task = tasks.heston_task()
    config = {"task": task, "root_seed": args.seed,
              "splits": {"train": args.train, "validation": args.validation, "test": args.test}}
    bank.validate_config(config)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(config, indent=1) + "\n")
    print(f"{out}: fingerprint {bank.task_fingerprint(config)}")


def _git_commit():
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": commit, "tracked_files_modified": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "tracked_files_modified": None}


def _job_map(args):
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    identity = {"config": str(config_path), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "task_fingerprint": bank.task_fingerprint(config)}
    jobs = [dict(job, **identity) for job in bank.job_map(config, args.chunk)]
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(f"{out}: {len(jobs)} jobs")


def _run_job(job, output_root):
    config_path = Path(job["config"])
    if "config_sha256" in job and hashlib.sha256(config_path.read_bytes()).hexdigest() != job["config_sha256"]:
        raise ValueError(f"{config_path} changed since the job map was written")
    config = json.loads(config_path.read_text())
    if "task_fingerprint" in job and bank.task_fingerprint(config) != job["task_fingerprint"]:
        raise ValueError("the config's task fingerprint differs from the job map's")
    destination = Path(output_root) / f"{job['split']}-{job['start']:07d}-{job['stop']:07d}"
    if (destination / "manifest.json").exists():  # finished by an earlier attempt: keep it
        record = json.loads((destination / "manifest.json").read_text())
        return destination, record["generation_cpu_seconds"], 0.0
    if destination.exists():  # a partial attempt (no manifest): redo it
        shutil.rmtree(destination)
    started = time.time()
    record = bank.write_shard(destination, config, job["split"], job["start"], job["stop"])
    (destination / "run.json").write_text(json.dumps({"git": _git_commit(), "host": socket.gethostname(),
                                                      "job": job, "wall_seconds": time.time() - started}) + "\n")
    return destination, record["generation_cpu_seconds"], time.time() - started


def _shard(args):
    jobs = json.loads(Path(args.job_map).read_text())
    destination, cpu, wall = _run_job(jobs[args.task_index], args.output_root)
    print(f"{destination}: {cpu:.1f} CPU-s, {wall:.1f} s")


def _assemble(args):
    config = json.loads(Path(args.config).read_text())
    shards = sorted(p for p in Path(args.shards).iterdir() if (p / "manifest.json").exists())
    manifest = bank.assemble_bank(args.out, config, shards)
    print(f"{args.out}: {sum(s['examples'] for s in manifest['splits'].values())} surfaces from {len(shards)} shards")


def _local(args):
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    out = Path(args.out)
    shard_root = out.with_name(out.name + "-shards")
    if out.exists() or (shard_root.exists() and not args.resume):
        raise FileExistsError(out)
    shard_root.mkdir(parents=True, exist_ok=True)
    jobs = [dict(job, config=str(config_path)) for job in bank.job_map(config, args.chunk)]
    started, cpu = time.time(), 0.0
    if args.resume:
        # A shard is finished when its manifest exists (written last); anything else is redone.
        pending = []
        for job in jobs:
            destination = shard_root / f"{job['split']}-{job['start']:07d}-{job['stop']:07d}"
            if (destination / "manifest.json").exists():
                cpu += json.loads((destination / "manifest.json").read_text())["generation_cpu_seconds"]
                continue
            if destination.exists():
                shutil.rmtree(destination)
            pending.append(job)
        print(f"resume: {len(jobs) - len(pending)} of {len(jobs)} shards already finished", flush=True)
        jobs = pending
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (_, seconds, _) in enumerate(pool.map(_run_job, jobs, [shard_root] * len(jobs)), 1):
            cpu += seconds
            if done % max(1, len(jobs) // 20) == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} shards, {time.time() - started:.0f} s wall, {cpu:.0f} CPU-s", flush=True)
    manifest = bank.assemble_bank(out, config, sorted(shard_root.iterdir()))
    total = sum(s["examples"] for s in manifest["splits"].values())
    print(f"{out}: {total} surfaces, {cpu / total:.3f} CPU-s per surface")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("config")
    p.add_argument("--model", choices=("rough", "heston"), required=True)
    p.add_argument("--curve-prior", choices=("linear_wobble", "power_law_walk"), default="linear_wobble",
                   help="rough only: the pilot's curve prior (default) or the full bank's")
    for split in ("train", "validation", "test"):
        p.add_argument(f"--{split}", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(run=_config)
    p = sub.add_parser("job-map")
    p.add_argument("--config", required=True)
    p.add_argument("--chunk", type=int, required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(run=_job_map)
    p = sub.add_parser("shard")
    p.add_argument("--job-map", required=True)
    p.add_argument("--task-index", type=int, required=True)
    p.add_argument("--output-root", required=True)
    p.set_defaults(run=_shard)
    p = sub.add_parser("assemble")
    p.add_argument("--config", required=True)
    p.add_argument("--shards", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(run=_assemble)
    p = sub.add_parser("local")
    p.add_argument("--config", required=True)
    p.add_argument("--chunk", type=int, default=25)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)
    p.add_argument("--resume", action="store_true", help="keep finished shards of an interrupted run")
    p.set_defaults(run=_local)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
