from pathlib import Path
import json, hashlib, time
import numpy as np
from sbi4finance.calibration.heston_spread import HestonSpreadPilot
from sbi4finance.calibration.heston_grid import HestonGridPilot
from scripts import run_heston_spread_reference as helper
from sbi4f_tutorial.heston_options import model,control_variate

e=Path("/Users/afengler/Projects/proj_sbi4finance/sbi4finance-heston-spread-baseline-20261109/lab/evidence/pilot-results/2026-11-09/heston-spread-baseline/revised-grid")
protocol=e/"cv-preparation/preparation-protocol.json"
p=json.loads(protocol.read_text());panel=json.loads(Path(p["old_panel"]["path"]).read_text())
design=json.loads(Path(panel["design_path"]).read_text())
old,new=HestonSpreadPilot(design["records"],infer_c=True),HestonGridPilot(design["records"],infer_c=True)
points=[];start=time.monotonic()
with helper.Budget(300) as budget:
 for i,phi in enumerate(p["pilot"]["physical_points"]):
  budget.check();theta=np.r_[phi,1.]
  row={"index":i}
  try: a=old.prices(theta)
  except (ValueError,FloatingPointError) as err:
   row.update(old_status="failed_numerical_gate",old_error=str(err))
  else:
   b=new.prices(theta);ratio=float(np.max(abs(a-b)/old.measurement_sigma))
   if ratio>old.numerical_fraction:raise FloatingPointError("representations differ beyond existing numerical budget")
   row.update(old_status="accepted",maximum_difference_over_minimum_sd=ratio)
  points.append(row)
def bind(path):
 path=Path(path).resolve();return {"path":str(path),"sha256":helper.digest(path)}
record={"format":"heston_pr87_grid_old54_representation_crosscheck_v1","status":"complete",
 "producer":bind(__file__),"declaration":bind(protocol),"old_design":bind(panel["design_path"]),
 "old_model_source":bind(model.__file__),"new_representation":bind(control_variate.__file__),
 "points":points,"accepted_old_points":sum(x["old_status"]=="accepted" for x in points),
 "maximum_accepted_difference_over_minimum_sd":max(x.get("maximum_difference_over_minimum_sd",0) for x in points),
 "seconds":time.monotonic()-start,"reserved_arrays_opened":False,"new_neural_sampling":False,"scientific_acceptance":False,
 "scope":"Same-CF numerical representation consistency on all predeclared pilot physical points; failed old numerical cases retained, not independent financial-model validation."}
helper.write_json(e/"old54-numerical-crosscheck.json",record)
print(json.dumps({k:record[k] for k in ["status","accepted_old_points","maximum_accepted_difference_over_minimum_sd","seconds"]}))
