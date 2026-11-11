"""Prepare/check or execute explicitly local, source-bound failed-row repairs."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
STARTED=time.monotonic()
for variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[variable]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sbi4finance.calibration import rough_actual_layout_local_repair as repair

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest="command",required=True)
    prep=commands.add_parser("prepare");prep.add_argument("--manifest",type=Path,required=True);prep.add_argument("--out",type=Path,required=True)
    prep.add_argument("--workers",type=int,default=6);prep.add_argument("--row-seconds",type=int,default=2700);prep.add_argument("--expected-rows",type=int,default=87)
    prep.add_argument("--transport-archive",type=Path);prep.add_argument("--transport-sha256")
    check=commands.add_parser("check");check.add_argument("--plan",type=Path,required=True)
    run=commands.add_parser("worker");run.add_argument("--plan",type=Path,required=True);run.add_argument("--index",type=int,required=True);run.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="prepare":
        value=repair.prepare(args.manifest,args.out,workers=args.workers,row_seconds=args.row_seconds,expected_rows=args.expected_rows,
            transport_archive=args.transport_archive,transport_sha256=args.transport_sha256)
        print(json.dumps({"rows":len(value["rows"]),"plan":repair.binding(args.out),"prices_generated":0}))
    elif args.command=="check":repair.validate(repair.bank.pilot.read(args.plan));print("local source/environment/transport identities verified; no MC")
    else:print(json.dumps(repair.run_row(args.plan,args.index,args.out,process_started=STARTED)))
