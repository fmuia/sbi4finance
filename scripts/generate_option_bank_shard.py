"""Generate one fixed-ID option chunk directly or from a SLURM array job map."""

import argparse
import json
from pathlib import Path
import re

from sbi4finance.option_bank_shards import write_shard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--job-map", type=Path,
                        help="JSON list of config/split/start/stop/group; config paths relative to map")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    direct = (args.config, args.split, args.start, args.stop, args.out)
    if args.job_map is not None:
        if any(value is not None for value in direct) or args.task_index is None or args.output_root is None:
            parser.error("--job-map needs --task-index and --output-root; no direct arguments")
        records = json.loads(args.job_map.read_text())
        if not isinstance(records, list) or not 0 <= args.task_index < len(records):
            parser.error("task index outside the job-map list")
        record = records[args.task_index]
        if set(record) != {"config", "split", "start", "stop", "group"}:
            parser.error("job-map records require config, split, start, stop, group")
        group = record["group"]
        if not isinstance(group, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", group) is None:
            parser.error("group must be a single alphanumeric/underscore/hyphen name")
        config_path = args.job_map.parent / record["config"]
        split, start, stop = record["split"], record["start"], record["stop"]
        out = args.output_root / group / f"{split}-{start:06d}-{stop:06d}"
    else:
        if any(value is None for value in direct) or args.task_index is not None or args.output_root is not None:
            parser.error("provide --config, --split, --start, --stop, --out or use --job-map")
        config_path, split, start, stop, out = direct
    report = write_shard(out, json.loads(config_path.read_text()), split, start, stop)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
