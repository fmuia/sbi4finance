"""Prepare reviewable rough numerical commands only; never submit jobs or price."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from sbi4finance.calibration import rough_actual_layout_pilot as pilot

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="role",required=True)
    prepare=sub.add_parser("prepare")
    for name in ("design","heston-task","known-manifest","joint-manifest","out"): prepare.add_argument("--"+name,type=Path,required=True)
    bind=sub.add_parser("bind-heston")
    for name in ("work-order","evaluation","out"): bind.add_argument("--"+name,type=Path,required=True)
    check=sub.add_parser("check-sources"); check.add_argument("--work-order",type=Path,required=True); check.add_argument("--repo-root",type=Path)
    args=parser.parse_args()
    if args.role=="prepare":
        result=pilot.prepare(args.design,args.heston_task,args.known_manifest,args.joint_manifest,args.out)
        print(json.dumps({"work_order":pilot.binding(args.out/"work-order.json"),"components":len(result["components"]),"MC_executed":False,"sbatch":str(args.out/"pilot.sbatch")}))
    elif args.role=="bind-heston": print(json.dumps(pilot.bind_heston(args.work_order,args.evaluation,args.out)))
    else:
        pilot.validate(pilot.read(args.work_order),args.repo_root); print("source/runtime identities verified; no MC executed")
