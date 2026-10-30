"""Run a bounded likelihood-noise panel at four saved five-year PMMH states."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--chains", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    for chain in range(4):
        destination = args.out / f"chain-{chain}"
        command = [sys.executable, str(Path(__file__).with_name("path_pmmh.py")),
            "--bank", str(args.bank), "--case-index", "0", "--length", "1260", "--chain", str(chain),
            "--noise-only", "--initial-checkpoint", str(args.chains / f"chain-{chain}"),
            "--noise-particle-counts", "512", "1024", "2048", "--noise-repeats", "8",
            "--phase-seed", "3", "--out", str(destination)]
        subprocess.run(command, check=True)
        records.append({"chain": chain, "summary": str(destination / "noise-summary.json")})
    (args.out / "panel.json").write_text(json.dumps({"points": records, "new_chains_started": False,
        "scope": "pointwise likelihood noise and runtime; no posterior or global particle adequacy certificate"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
