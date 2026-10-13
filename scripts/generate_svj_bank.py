"""Write and integrity-check a seeded daily SVJ bank in a new directory."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from sbi4finance.svj.bank import load_bank, write_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tasks/svj_daily_bank.json"))
    parser.add_argument("--out", type=Path, required=True, help="new output directory; never overwritten")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    started = perf_counter()
    manifest = write_bank(args.out, config)
    written = perf_counter()
    _, splits = load_bank(args.out)
    checked = perf_counter()
    print(json.dumps({"bank": str(args.out), "task_fingerprint": manifest["task_fingerprint"],
                      "generation_and_write_seconds": written - started,
                      "verified_load_seconds": checked - written,
                      "loaded_array_bytes": sum(a.nbytes for split in splits.values() for a in split.values()),
                      "evidence_scope": "S: bank engineering check only"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
