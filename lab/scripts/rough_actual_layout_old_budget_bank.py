"""Prepare metadata or execute one explicitly admitted actual-layout bank shard."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
STARTED=time.monotonic()
for variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[variable]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sbi4finance.calibration import rough_actual_layout_old_budget_bank as bank

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="role",required=True)
    prep=sub.add_parser("prepare"); prep.add_argument("--parent-work-order",type=Path,required=True); prep.add_argument("--configuration",type=Path,required=True); prep.add_argument("--out",type=Path,required=True)
    run=sub.add_parser("worker")
    for name in ("plan","admission","out"): run.add_argument("--"+name,type=Path,required=True)
    run.add_argument("--index",type=int,required=True)
    check=sub.add_parser("check"); check.add_argument("--plan",type=Path,required=True)
    a=p.parse_args()
    if a.role=="prepare":
        result=bank.prepare(a.parent_work_order,a.out,**bank.pilot.read(a.configuration)); print(json.dumps({"jobs":len(result["jobs"]),"prices_generated":0,"plan":bank.pilot.binding(a.out)}))
    elif a.role=="check": bank.validate(bank.pilot.read(a.plan)); print("bank source/configuration identities verified; no MC executed")
    else: print(json.dumps(bank.run_shard(a.plan,a.admission,a.index,a.out,process_started=STARTED)))
