"""Select one predeclared pilot from a Slurm array; no shell evaluation.

Use through cluster/core_cpu.sbatch in its locked core/core-reference runtime.
Each task writes a distinct new output directory and retains semantic seeds.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def command(kind, index, out, bank=None, replay_report=None, references=None):
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
    if kind == "rough-fine" and index < 12:
        case = (4, 6, 2)[index // 4]
        grid = (4, 8)[(index % 4) // 2]
        replicate = index % 2
        paths = {4: 1048576, 6: 524288, 2: 131072}[case]
        return [sys.executable, str(root / "scripts/pilot_rough_numerics.py"), "--config",
                str(root / "configs/tasks/calibration_rough_joint_pilot.json"), "--case", str(case),
                "--paths", str(paths), "--grids", "4,8", "--only-grid", str(grid),
                "--only-replicate", str(replicate), "--out", str(out / f"case-{case}" / f"grid-{grid}-rep-{replicate}")]
    if kind == "curve-followup" and index < 3 and replay_report is not None:
        original = json.loads(replay_report.read_text())
        if original["case"]["name"] != "hump":
            raise ValueError("follow-up declaration requires the original hump observation")
        initial = original["fit"]["parameters"] if index < 2 else [.08, 2.0, -.8, .03, .06, 2.0]
        result = [sys.executable, str(root / "scripts/check_curve_adequacy.py"), "--case", "hump",
                  "--replay-report", str(replay_report), "--fit-paths", "8192", "--fit-evaluations", "160",
                  "--audit-paths", "32768", "--audit-replicates", "2", "--audit-namespace", "followup-20261026",
                  "--initial-parameters", *[str(x) for x in initial], "--out", str(out / ("A", "B", "C")[index])]
        return result + (["--decay-upper", "32"] if index else [])
    if kind == "heston-precision" and index < 3 and references is not None:
        return [sys.executable, str(root / "scripts/check_heston_reference_precision.py"),
                "--reference", str(references / f"case-{index}"), "--out", str(out / f"case-{index}")]
    raise ValueError("unsupported pilot/index or missing Heston bank")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("rough-grid", "curve", "heston", "rough-fine", "curve-followup", "heston-precision"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--replay-report", type=Path)
    parser.add_argument("--references", type=Path)
    args = parser.parse_args()
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    subprocess.run(command(args.kind, index, args.out, args.bank, args.replay_report, args.references), check=True)


if __name__ == "__main__":
    main()
