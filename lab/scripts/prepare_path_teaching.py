"""Prepare hash-bound trajectory examples for the existing accept.py replay route."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sbi4finance.path_inference import teaching


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=4)
    args = parser.parse_args()
    bundle = teaching.prepare(args.checkpoint, args.evaluation, args.out, cases=args.cases)
    print(f"prepared {bundle['model']}: {len(bundle['case_ids'])} cases, {len(bundle['cached_lengths'])} lengths")
