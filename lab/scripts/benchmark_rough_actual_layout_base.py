"""Tiny old-cost-profile benchmark on the new actual quotes; no training claim."""
import time
STARTED = time.monotonic()
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
_repo = Path(os.environ["ROUGH_REPO_ROOT"]) if os.environ.get("ROUGH_REPO_ROOT") else Path(__file__).resolve().parents[2]
for _path in (_repo/"lab", _repo/"lab/src", _repo/"tutorial/src"):
    sys.path.insert(0, str(_path))
import numpy as np
from sbi4finance.calibration import rough_actual_layout_bank as bank

FORMAT = "rough_actual_layout_base_cost_benchmark_v1"
ROWS = (0, 1, 63, 127)
LIMITS = {"components": 4, "concurrent_workers": 4, "cpus_per_worker": 1,
    "memory_gib_per_worker": 4, "wall_seconds_per_component": 180,
    "total_allocated_cpu_seconds": 720, "automatic_retry": False}


def prepare(parent_path, out):
    parent = bank.pilot.read(parent_path); bank.validate(parent)
    if not parent.get("inferred_c_blueprint") or parent["active_ranges"]["joint_unknown"].get("train") != [0,128]:
        raise ValueError("exact main joint128 parent required")
    task = parent["tasks"]["joint_unknown"]
    old_steps = sum(b["n_steps"] for b in task["base_task"]["numerics"]["blocks"])
    new_steps = sum(b["n_steps"] for b in task["blocks"])
    result = {"format": FORMAT, "parent_bank_plan": bank.pilot.binding(parent_path),
        "parent_bank_plan_value": parent, "jobs": [{"index":i,"row_id":row,
            "spec":bank.row_spec(parent,"joint_unknown","train",row)} for i,row in enumerate(ROWS)],
        "limits": LIMITS, "paths":32768, "grid_multiplier":1,
        "benchmark_source_sha256":bank.pilot.digest(__file__), "financial_source_bindings":parent["source_bindings"],
        "work_comparison": {"old_three_block_sum_steps":old_steps,"new_seven_block_base_sum_steps":new_steps,
            "main_seven_block_fine_sum_steps":new_steps*4,"main_paths":262144,
            "benchmark_paths":32768,"main_over_benchmark_path_steps":32.,
            "main_over_old_first_rung_path_steps":new_steps*4*262144/(old_steps*32768)},
        "scientific_acceptance":False,"training_usable":False,
        "scope":"matched financial parameter rows and actual quotes; base-grid first-path-rung cost/MC diagnostic only; no continuous-price or posterior qualification"}
    Path(out).parent.mkdir(parents=True,exist_ok=True); bank.pilot.write_new(out,result)
    return result


def validate(work):
    if (work.get("format") != FORMAT or work.get("limits") != LIMITS or work.get("paths") != 32768
            or work.get("grid_multiplier") != 1 or work.get("scientific_acceptance") is not False
            or work.get("training_usable") is not False or work.get("benchmark_source_sha256") != bank.pilot.digest(__file__)):
        raise ValueError("frozen tiny base-cost benchmark required")
    parent = work["parent_bank_plan_value"]; bank.validate(parent)
    if work["financial_source_bindings"] != parent["source_bindings"]:
        raise ValueError("immutable financial source bindings changed")
    expected=[{"index":i,"row_id":row,"spec":bank.row_spec(parent,"joint_unknown","train",row)} for i,row in enumerate(ROWS)]
    if work["jobs"] != expected: raise ValueError("benchmark rows or joint-c draws changed")


def run(work_path, index, out, *, process_started=STARTED):
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CPUS_PER_TASK") != "1":
        raise ValueError("one allocated Slurm CPU required")
    work=bank.pilot.read(work_path); validate(work)
    if type(index) is not int or not 0 <= index < 4: raise ValueError("one declared component required")
    out=Path(out)
    if out.exists(): raise FileExistsError("new benchmark output identity required; no retry")
    out.mkdir(parents=True); job=work["jobs"][index];spec=job["spec"]
    task=work["parent_bank_plan_value"]["tasks"]["joint_unknown"]
    record={"format":"rough_actual_layout_base_cost_component_v1","status":"running","job":job,
        "work_order_sha256":bank.pilot.digest(work_path),"slurm_job_id":os.environ["SLURM_JOB_ID"],
        "blocks_completed":0,"block_payloads":[],"paths":32768,"grid_multiplier":1,
        "scientific_acceptance":False,"training_usable":False,"continuous_bias_certified":False}
    bank.pilot.write_new(out/"run.json",record)
    from scripts.run_heston_spread_reference import Budget
    prices=np.zeros(63); covariance=np.zeros((63,63));block_times=[]
    curve=bank.row_curve(task,spec);parameters=dict(zip(("hurst","eta","rho"),spec["parameters"][:3],strict=True))
    remaining=180-(time.monotonic()-process_started)-3
    try:
        if remaining <= 0: raise TimeoutError("benchmark setup exhausted total wall cap")
        with Budget(remaining) as budget:
            for block_index,block in enumerate(task["blocks"]):
                budget.check();columns=[i for i,c in enumerate(task["design"]["contracts"]) if c["maturity"]==block["horizon"]]
                contracts=[task["design"]["contracts"][i] for i in columns]
                entropy=[0x52424331,20261111,job["row_id"],block_index]
                diagnostic={"capture_raw_control":True};started=time.monotonic()
                mean,error,cov=bank.pilot.runtime_pricing.rough_block(parameters,curve,contracts,
                    horizon=block["horizon"],n_steps=block["n_steps"],n_pairs=16384,
                    seed=np.random.SeedSequence(entropy),diagnostics=diagnostic)
                bank.pilot.checked_covariance(cov,len(columns))
                if not np.isfinite(mean).all() or not np.isfinite(error).all():raise ValueError("nonfinite benchmark prices")
                prices[columns]=mean;covariance[np.ix_(columns,columns)]=cov;block_times.append(time.monotonic()-started)
                path=out/f"block-{block_index:02d}.npz"
                with path.open("xb") as stream: np.savez_compressed(stream,prices=mean,standard_errors=error,covariance=cov)
                record["blocks_completed"]+=1
                record["block_payloads"].append({"block":block_index,"columns":columns,"steps":block["n_steps"],
                    "entropy":entropy,"seconds":block_times[-1],**bank.pilot.binding(path)})
                (out/"run.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
        h=np.asarray(task["design"]["half_spread_forward"]);c=spec["noise_scale"];se=np.sqrt(np.diag(covariance))
        with (out/"results.npz").open("xb") as stream: np.savez_compressed(stream,
            physical_parameters=np.asarray(spec["parameters"]),inference_parameters=np.r_[spec["parameters"],c],
            noise_scale=np.asarray(c),reference_prices_forward=prices,pricing_covariance_forward=covariance,
            pricing_standard_errors_forward=se,mcse_over_h=se/h,mcse_over_observation_sd=se/(c*h))
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024)
        record.update(status="complete",seconds=time.monotonic()-process_started,process_cpu_seconds=time.process_time(),
            block_seconds=block_times,peak_rss_bytes=rss,max_mcse_over_h=float(np.max(se/h)),
            max_mcse_over_observation_sd=float(np.max(se/(c*h))),payload=bank.pilot.binding(out/"results.npz"))
        validate(work)
        (out/"run.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
        return record
    except BaseException as error:
        record.update(status="failed",error_type=type(error).__name__,error=str(error),seconds=time.monotonic()-process_started)
        (out/"run.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n");raise


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="role",required=True)
    prep=sub.add_parser("prepare");prep.add_argument("--parent-bank-plan",type=Path,required=True);prep.add_argument("--out",type=Path,required=True)
    check=sub.add_parser("check");check.add_argument("--work-order",type=Path,required=True)
    worker=sub.add_parser("worker");worker.add_argument("--work-order",type=Path,required=True);worker.add_argument("--index",type=int,required=True);worker.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.role=="prepare": print(json.dumps(prepare(a.parent_bank_plan,a.out)["work_comparison"]))
    elif a.role=="check":validate(bank.pilot.read(a.work_order));print("benchmark and immutable financial sources verified; no MC executed")
    else:
        r=run(a.work_order,a.index,a.out);print(json.dumps({"status":r["status"],"seconds":r["seconds"],"max_mcse_over_h":r["max_mcse_over_h"]}))
