"""Select one predeclared pilot from a Slurm array; no shell evaluation.

Use through cluster/core_cpu.sbatch in its locked core/core-reference runtime.
Each task writes a distinct new output directory and retains semantic seeds.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def command(kind, index, out, bank=None):
    root = Path(__file__).resolve().parents[1]
    if index < 0:
        raise ValueError("nonnegative Slurm array index required")
    if kind == "rough-grid" and index < 8:
        return [sys.executable, str(root / "scripts/pilot_rough_numerics.py"), "--config",
                str(root / "configs/tasks/calibration_rough_joint_pilot.json"), "--case", str(index),
                "--out", str(out / f"case-{index}")]
    if kind == "curve" and index < 3:
        case = ("flat", "monotone", "hump")[index]
        return [sys.executable, str(root / "scripts/check_curve_adequacy.py"), "--case", case,
                "--out", str(out / case)]
    if kind == "heston" and index < 3 and bank is not None:
        return [sys.executable, str(root / "scripts/calibration_reference.py"), "--bank", str(bank),
                "--split", "validation", "--index", str(index), "--out", str(out / f"case-{index}"),
                "--draws", "2000", "--tune", "2000", "--chains", "4", "--seed", str(271026 + index),
                "--no-progress"]
    raise ValueError("unsupported pilot/index or missing Heston bank")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("rough-grid", "curve", "heston"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bank", type=Path)
    args = parser.parse_args()
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    subprocess.run(command(args.kind, index, args.out, args.bank), check=True)


if __name__ == "__main__":
    main()
