"""Generate, benchmark or check a synthetic physical-measure return-path bank."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from sbi4finance.path_inference.bank import (
    MODELS, default_task, generate, load_manifest, load_split, task_fingerprint, write_bank,
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "benchmark"):
        command = commands.add_parser(name)
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--model", choices=MODELS)
        source.add_argument("--task", type=Path, help="explicit task JSON (including a smaller smoke design)")
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--seed", type=int, default=30102026)
        command.add_argument("--chunk", type=int, default=128)
        if name == "generate":
            command.add_argument("--train", type=int, default=20000)
            command.add_argument("--validation", type=int, default=2000)
            command.add_argument("--test", type=int, default=2000)
        else:
            command.add_argument("--count", type=int, default=128)
    check = commands.add_parser("check")
    check.add_argument("--bank", type=Path, required=True)
    check.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "check":
        manifest = load_manifest(args.bank)
        checked = {}
        for split in manifest["splits"]:
            arrays = load_split(args.bank, split)
            checked[split] = len(arrays["parameters"])
        report = {"passed": True, "task_sha256": manifest["task_sha256"], "counts": checked,
                  "scope": "byte/array/identity integrity; not scientific acceptance"}
        if args.out:
            _write(args.out, report)
    else:
        task = json.loads(args.task.read_text()) if args.task else default_task(args.model)
        if args.chunk < 1 or args.seed < 0:
            parser.error("chunk must be positive and seed nonnegative")
        if args.command == "generate":
            manifest = write_bank(task, args.out, args.train, args.validation, args.test,
                                  chunk=args.chunk, seed=args.seed)
            report = {"bank": str(args.out), "task_sha256": manifest["task_sha256"],
                      "counts": manifest["split_counts"], "timing_seconds": manifest["timing_seconds"]}
        else:
            if args.count < 1 or args.out.exists():
                parser.error("benchmark count must be positive and output must be new")
            started = perf_counter()
            squares = 0.0
            elements = 0
            for start in range(0, args.count, args.chunk):
                arrays = generate(task, "train", start, min(args.chunk, args.count - start), args.seed)
                squares += float(np.square(arrays["observations"]).sum())
                elements += arrays["observations"].size
            elapsed = perf_counter() - started
            report = {"model": task["model"], "task": task, "task_sha256": task_fingerprint(task),
                      "count": args.count, "seed": args.seed, "chunk": args.chunk,
                      "elapsed_seconds_including_compile": elapsed,
                      "seconds_per_trajectory_including_compile": elapsed / args.count,
                      "observed_return_rms": float(np.sqrt(squares / elements)),
                      "scope": "throughput/execution only; not identification or posterior validation"}
            _write(args.out, report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
