"""Run the authorized 48-batch + four-ABBA-timing rough diagnostic only.

Declare on Oscar from the frozen reviewed declaration. Run groups 0/1/2 for
anchors 0/4/6 and group 3 for all four same-allocation ABBA timings. Each group
has a 25-minute, one-CPU limit; declaration 5 minutes, total 105 allocated minutes.
No separate price warmup, adaptive escalation, replacement or repeat occurs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from sbi4finance.calibration import rough_control_diagnostic as diagnostic  # noqa: E402
from sbi4finance.gbm_bank import _write_json  # noqa: E402


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    roles=parser.add_subparsers(dest="role",required=True)
    declare=roles.add_parser("declare")
    declare.add_argument("--reference",type=Path,default=diagnostic.REFERENCE)
    declare.add_argument("--revision",required=True)
    declare.add_argument("--out",type=Path,required=True)
    run=roles.add_parser("run")
    run.add_argument("--declaration",type=Path,required=True)
    run.add_argument("--group-index",type=int,required=True)
    run.add_argument("--out",type=Path,required=True)
    reduce=roles.add_parser("reduce")
    reduce.add_argument("--declaration",type=Path,required=True)
    reduce.add_argument("--groups",type=Path,required=True,help="Parent of group-0 ... group-3.")
    reduce.add_argument("--accounting",type=Path)
    reduce.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.role=="declare":
        reference=json.loads(args.reference.read_text())
        if reference!=json.loads(diagnostic.REFERENCE.read_text()):
            raise ValueError("exact frozen reviewed Linux declaration required")
        declaration=diagnostic.declare(reference,source_revision=args.revision)
        args.out.mkdir(parents=True,exist_ok=False)
        _write_json(args.out/"declaration.json",declaration)
        print(json.dumps({"calls":len(declaration["calls"]),"seed_audit":declaration["seed_audit"],
                          "declaration_sha256":declaration["declaration_sha256"]}))
        return
    declaration=json.loads(args.declaration.read_text())
    if args.role=="run":
        def terminated(number,frame):
            raise SystemExit(f"received signal {number}; no automatic retry")
        previous=signal.signal(signal.SIGTERM,terminated)
        try:report=diagnostic.run_group(declaration,args.group_index,args.out)
        finally:signal.signal(signal.SIGTERM,previous)
        print(json.dumps({"group":args.group_index,"status":report["status"],"reported_calls":len(report["runs"])}))
        if report["status"]!="completed":raise SystemExit(2)
        return
    reports=[];provenance=[]
    for index in range(4):
        p=args.groups/f"group-{index}"/"report.json"
        if p.exists():
            reports.append(json.loads(p.read_text()))
            provenance.append({"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    accounting=[] if args.accounting is None else json.loads(args.accounting.read_text())
    result=diagnostic.reduce_groups(declaration,reports,accounting=accounting)
    result["input_artifacts"]=provenance
    args.out.mkdir(parents=True,exist_ok=False)
    _write_json(args.out/"report.json",result)
    print(json.dumps({"status":result["status"],"reported_calls":result["reported_calls"]}))


if __name__=="__main__":main()
