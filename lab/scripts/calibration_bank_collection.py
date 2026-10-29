"""Append-only calibration banks: initialize, adopt, extend, and freeze snapshots.

Run with --help for the command list. No command submits a cluster job.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sbi4finance.calibration import bank_collection  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("init", help="freeze a task, root seed, sources and runtime")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = commands.add_parser("adopt", help="validate and reuse existing finished legacy shards")
    p.add_argument("--collection", type=Path, required=True)
    p.add_argument("--shards", type=Path, required=True)
    p = commands.add_parser("plan", help="write jobs for missing IDs up to the requested training size")
    p.add_argument("--collection", type=Path, required=True)
    p.add_argument("--train", type=int, required=True)
    p.add_argument("--chunk", type=int, default=50)
    p.add_argument("--out", type=Path, required=True)
    p = commands.add_parser("shard", help="run one planned job; verified finished shards are reused")
    p.add_argument("--job-map", type=Path, required=True)
    p.add_argument("--task-index", type=int, required=True)
    p = commands.add_parser("freeze", help="publish an immutable snapshot without concatenating payloads")
    p.add_argument("--collection", type=Path, required=True)
    p.add_argument("--train", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "init":
        record = bank_collection.create_collection(args.out, json.loads(args.config.read_text()))
        print(f"{args.out}: collection {record['collection_id']}")
    elif args.command == "adopt":
        paths = sorted(p.parent for p in args.shards.glob("*/manifest.json"))
        if not paths:
            parser.error("--shards contains no finished shard manifests")
        bank_collection.adopt_shards(args.collection, paths)
        print(f"{args.collection}: checked {len(paths)} existing shards")
    elif args.command == "plan":
        if args.out.exists():
            raise FileExistsError(args.out)
        record = bank_collection.plan_extension(args.collection, args.train, args.chunk)
        record["collection"] = str(args.collection.resolve())
        record["collection_sha256"] = hashlib.sha256((args.collection / "collection.json").read_bytes()).hexdigest()
        # Keep the large task/config once per map, rather than once per shard.
        record["jobs"] = [{key: job[key] for key in ("split", "start", "stop", "job_sha256")}
                          for job in record["jobs"]]
        with args.out.open("x") as stream:
            stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
        print(f"{args.out}: {len(record['jobs'])} missing shards")
    elif args.command == "shard":
        record = json.loads(args.job_map.read_text())
        jobs = record["jobs"]
        if not 0 <= args.task_index < len(jobs):
            parser.error(f"--task-index must be in [0, {len(jobs)})")
        job = {key: record[key] for key in ("collection_id", "collection_sha256", "configuration")}
        job.update(jobs[args.task_index])
        bank_collection.run_job(record["collection"], job)
        print(f"completed or verified task {args.task_index}")
    else:
        record = bank_collection.freeze_snapshot(args.collection, args.train, args.out)
        print(f"{args.out}: frozen snapshot of collection {record['collection_id']}")


if __name__ == "__main__":
    main()
