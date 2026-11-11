"""Run exactly one declared, Heston-gated rough component inside Slurm."""
import time
PROCESS_STARTED=time.monotonic()
import argparse
import json
import os
from pathlib import Path
import sys
for variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[variable]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sbi4finance.calibration import rough_actual_layout_pilot_portable as pilot
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("work-order","condition","heston-evaluation","out"): p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--index",type=int,required=True);a=p.parse_args()
    result=pilot.run_component(a.work_order,a.condition,a.index,a.out,process_started=PROCESS_STARTED,heston_evaluation_path=a.heston_evaluation)
    print(json.dumps({"status":result["status"],"index":a.index,"numerical_target_established":False}))
