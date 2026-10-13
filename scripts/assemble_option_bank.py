"""Assemble verified fixed-ID chunks into the existing option_bank_v1 format."""

import argparse
import json
from pathlib import Path

from sbi4finance.option_bank import load_bank
from sbi4finance.option_bank_shards import assemble_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--shards", required=True, type=Path,
                        help="directory whose immediate child directories are all expected chunks")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--base-bank", type=Path,
                        help="existing matching bank to reuse as immutable split prefixes")
    args = parser.parse_args()
    # Include incomplete child directories so a failed chunk cannot be silently
    # skipped. The assembler requires the full configured ID range regardless.
    chunks = sorted(path for path in args.shards.iterdir() if path.is_dir())
    manifest = assemble_bank(args.out, json.loads(args.config.read_text()), chunks,
                             base_bank=args.base_bank)
    load_bank(args.out)
    print(json.dumps({"format": manifest["format"], "splits": manifest["splits"],
                      "chunks": len(chunks), "integrity_verified": True}, indent=2))


if __name__ == "__main__":
    main()
