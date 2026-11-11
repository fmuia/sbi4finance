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
from sbi4finance.calibration import rough_actual_layout_overnight_bank as bank

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="role",required=True)
    prep=sub.add_parser("prepare"); prep.add_argument("--parent-work-order",type=Path,required=True); prep.add_argument("--configuration",type=Path,required=True); prep.add_argument("--out",type=Path,required=True)
    run=sub.add_parser("worker")
    for name in ("plan","admission","out"): run.add_argument("--"+name,type=Path,required=True)
    run.add_argument("--index",type=int,required=True)
    run.add_argument("--retry-failed",action="store_true");run.add_argument("--retry-reason");run.add_argument("--resume-from",type=Path);run.add_argument("--allocation-group",type=int)
    allocation=sub.add_parser("allocation")
    for name in ("plan","admission","out"):allocation.add_argument("--"+name,type=Path,required=True)
    allocation.add_argument("--index",type=int,required=True)
    export=sub.add_parser("export")
    for name in ("plan","journal","out"):export.add_argument("--"+name,type=Path,required=True)
    export.add_argument("--complete-prefix",action="store_true")
    check=sub.add_parser("check"); check.add_argument("--plan",type=Path,required=True)
    a=p.parse_args()
    if a.role=="prepare":
        result=bank.prepare(a.parent_work_order,a.out,**bank.pilot.read(a.configuration)); print(json.dumps({"jobs":len(result["jobs"]),"prices_generated":0,"plan":bank.pilot.binding(a.out)}))
    elif a.role=="check": bank.validate(bank.pilot.read(a.plan)); print("bank source/configuration identities verified; no MC executed")
    elif a.role=="allocation":print(json.dumps(bank.run_allocation(a.plan,a.admission,a.index,a.out)))
    elif a.role=="export":print(json.dumps(bank.export_completed(a.plan,a.journal,a.out,complete_prefix=a.complete_prefix)))
    else: print(json.dumps(bank.run_shard(a.plan,a.admission,a.index,a.out,process_started=STARTED,retry_failed=a.retry_failed,retry_reason=a.retry_reason,resume_from=a.resume_from,allocation_group=a.allocation_group)))
