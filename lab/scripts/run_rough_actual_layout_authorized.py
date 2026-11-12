"""Prepare/check or run one explicitly user-authorized rough component."""
import time
PROCESS_STARTED=time.monotonic()
import argparse
import json
import os
from pathlib import Path
import sys
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[name]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sbi4finance.calibration import rough_actual_layout_authorized as pilot
if __name__=="__main__":
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="role",required=True)
 prep=sub.add_parser("prepare");prep.add_argument("--work-order",type=Path,required=True);prep.add_argument("--out",type=Path,required=True)
 check=sub.add_parser("check");check.add_argument("--work-order",type=Path,required=True);check.add_argument("--authorization",type=Path,required=True)
 run=sub.add_parser("run")
 for name in ("work-order","authorization","out"):run.add_argument("--"+name,type=Path,required=True)
 run.add_argument("--index",type=int,required=True);a=p.parse_args()
 if a.role=="prepare":print(json.dumps(pilot.prepare(a.work_order,a.out)))
 elif a.role=="check":pilot.validate_authorization(pilot.read(a.work_order),a.work_order,pilot.read(a.authorization));print("25 source and explicit authorization identities verified; no pricing executed")
 else:
  r=pilot.run_component(a.work_order,a.authorization,a.index,a.out,process_started=PROCESS_STARTED)
  print(json.dumps({"status":r["status"],"index":a.index,"numerical_target_established":False}))
