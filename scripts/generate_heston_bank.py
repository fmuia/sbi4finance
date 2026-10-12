"""Write and verify the synthetic daily Heston theta/signed-state bank."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from sbi4finance.heston_bank import load_bank, write_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tasks/heston_inference.json"))
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    start = perf_counter()
    write_bank(args.out, config)
    written = perf_counter()
    _, splits = load_bank(args.out)
    checked = perf_counter()
    print(json.dumps({"bank": str(args.out), "generation_and_write_seconds": written - start,
                      "verified_load_seconds": checked - written,
                      "loaded_array_bytes": sum(a.nbytes for split in splits.values()
                                                for a in split.values())}, indent=2))


if __name__ == "__main__":
    main()
