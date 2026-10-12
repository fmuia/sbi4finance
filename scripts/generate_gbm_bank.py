"""Write and verify a small synthetic bank; the same command can run on a CPU node."""

import argparse
import json
from pathlib import Path
import time

from sbi4finance.gbm_bank import load_bank, write_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tasks/gbm_bank.json"))
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    start = time.perf_counter()
    write_bank(args.out, config)
    written = time.perf_counter()
    _, splits = load_bank(args.out)
    checked = time.perf_counter()
    print(json.dumps({"bank": str(args.out), "generation_and_write_seconds": written - start,
                      "verified_load_seconds": checked - written,
                      "loaded_array_bytes": sum(a.nbytes for s in splits.values() for a in s.values())}, indent=2))


if __name__ == "__main__":
    main()
