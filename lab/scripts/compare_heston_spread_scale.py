"""Match saved neural draws to settled joint-c references, without resampling.

The expensive development population and three conditional neural samples are
exported once. This consumer adds the matched reference comparisons after the
independent reference campaign settles, preserving the original export.
"""
import argparse
import copy
from pathlib import Path
import shutil
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_heston_spread_scale as exported

fixed,baseline=exported.fixed,exported.baseline


def c_correlations(draws,names):
    values=np.asarray(draws).reshape(-1,6)
    sd=values.std(axis=0)
    return {name:float(np.corrcoef(values[:,index],values[:,-1])[0,1]) if sd[index]>0 and sd[-1]>0 else None
            for index,name in enumerate(names[:5])}


def verify_artifacts(values,protected):
    """Hash explicit artifact records; never interpret their text as code."""
    if isinstance(values,dict):
        if isinstance(values.get("path"),str) and isinstance(values.get("sha256"),str):
            path=str(Path(values["path"]).resolve())
            if fixed.digest(path)!=values["sha256"]:
                raise ValueError("comparison input artifact differs: "+path)
            protected[path]=values["sha256"]
        for value in values.values():verify_artifacts(value,protected)
    elif isinstance(values,list):
        for value in values:verify_artifacts(value,protected)


def compare(neural_path,study_path,recovery_path,out):
    started=time.monotonic()
    neural_path,study_path,recovery_path,out=map(Path,(neural_path,study_path,recovery_path,out))
    if out.exists():raise FileExistsError("matched comparison identity must be new")
    record=exported.read(neural_path)
    study=exported.read(study_path)
    exported.validate_study(study,record["task"])
    if (record["status"]!="complete" or record["study_protocol_sha256"]!=fixed.digest(study_path)
            or record["seed"]!=exported.SEED or record["family_checks"]!=144
            or record["actual_optimizer_updates"] not in study["joint_evaluation"]["checkpoints"]
            or record["conditional_draws"]["draws_per_case"]!=4096):
        raise ValueError("neural export is not the declared completed endpoint")
    ref_path,protected=exported.selected_reference_protocol(study,study_path,recovery_path)
    recovery,protocol=exported.read(recovery_path),exported.read(ref_path)
    campaign_path=Path(recovery["progress_path"])
    campaign=exported.read(campaign_path)
    if campaign["status"] in ("running","pending"):
        raise ValueError("reference campaign must settle before final comparison")
    exported.scale_reference.validate_protocol(protocol,protocol["jobs"][0]["id"])
    verify_artifacts(recovery,protected)
    verify_artifacts(record["source_sha256"],protected)
    for key,value in recovery.items():
        if "audit" in key and isinstance(value,dict) and "path" in value:
            verify_artifacts(exported.read(value["path"]),protected)
    payload=Path(record["conditional_draws"]["path"])
    if fixed.digest(payload)!=record["conditional_draws"]["sha256"]:
        raise ValueError("saved neural conditional payload differs")
    recovery_npz=neural_path.parent/"recovery.npz"
    if fixed.digest(recovery_npz)!=record["population"]["recovery_sha256"]:
        raise ValueError("saved recovery quantiles differ")
    checkpoint=Path(record["checkpoint"])
    metadata=exported.read(checkpoint/"checkpoint.json")
    exported.validate_training_record(metadata,study_path)
    if (metadata["task"]["target"]!=record["task"]
            or fixed.digest(checkpoint/"checkpoint.json")!=record["checkpoint_record_sha256"]
            or fixed.digest(checkpoint/"model.keras")!=record["model_sha256"]
            or fixed.digest(checkpoint/"state.npz")!=record["state_sha256"]):
        raise ValueError("source neural checkpoint differs")
    for path in (neural_path,study_path,payload,recovery_npz,campaign_path,Path(__file__),
            checkpoint/"checkpoint.json",checkpoint/"model.keras",checkpoint/"state.npz"):
        protected[str(path.resolve())]=fixed.digest(path)
    panel_path=Path(study["panel"]["path"])
    if fixed.digest(panel_path)!=study["panel"]["sha256"]:
        raise ValueError("reference panel differs")
    protected[str(panel_path.resolve())]=fixed.digest(panel_path)
    panel=exported.read(panel_path)
    result=copy.deepcopy(record)
    result.update(format="heston_spread_joint_scale_matched_comparison_v1",status="started",
        neural_export={"path":str(neural_path.resolve()),"sha256":fixed.digest(neural_path)},
        reference_recovery={"path":str(recovery_path.resolve()),"sha256":fixed.digest(recovery_path)},
        reference_protocol_path=str(ref_path.resolve()),reference_protocol_sha256=fixed.digest(ref_path),
        comparison_source={"path":str(Path(__file__).resolve()),"sha256":fixed.digest(__file__)},
        posterior_resampling=False,population_recomputed=False,scientific_acceptance=False)
    out.mkdir(parents=True)
    fixed.write_json(out/"run.json",result)
    try:
        with np.load(payload,allow_pickle=False) as saved:
            for index,case_id in enumerate(exported.CASES):
                y=np.asarray(panel["cases"][[item["id"] for item in panel["cases"]].index(case_id)]["observations"])
                neural=saved[case_id].copy()
                support=baseline.support_report(neural,np.asarray(record["task"]["prior_bounds"]))
                if neural.shape!=(4096,6) or support["material_outside_value_count"]:
                    raise ValueError("saved neural draw dimensions/support differ")
                original=record["cases"][case_id]
                if original["observation_sha256"]!=baseline.observations_digest(y):
                    raise ValueError("saved neural draws target other observations")
                selected,binding=exported.joint_reference(protocol,case_id,panel_path,record["task"],panel["products"],y)
                entry={key:value for key,value in original.items()
                    if key not in ("status","reason","comparison") and not key.startswith("reference_")}
                entry.update(binding)
                entry["neural_c_physical_correlations"]=c_correlations(neural,record["task"]["parameter_names"])
                for prefix in ("reference_run","reference_diagnostics","reference_chain_payload"):
                    if prefix+"_path" in binding:protected[binding[prefix+"_path"]]=binding[prefix+"_sha256"]
                if selected is not None:
                    chains,diagnostics=selected
                    flat=chains.reshape(-1,6)
                    entry["reference_c_marginal"]={"mean":float(flat[:,-1].mean()),"sd":float(flat[:,-1].std(ddof=1)),
                        "quantiles_05_50_95":np.quantile(flat[:,-1],[.05,.5,.95]).tolist(),
                        "probability_c_le_055":float(np.mean(flat[:,-1]<=.55)),"probability_c_ge_195":float(np.mean(flat[:,-1]>=1.95)),
                        "physical_parameter_c_correlations":c_correlations(flat,record["task"]["parameter_names"]),
                        "scope":"descriptive physical-coordinate dependence, not causality or copula acceptance"}
                    entry["comparison"]=baseline.compare_draws(neural,chains,diagnostics,record["task"]["parameter_names"],panel["products"],
                        seed=exported.SEED+100+index,replicates=400,family_checks=144)
                    entry["status"]=entry["comparison"]["status"]
                result["cases"][case_id]=entry
        combine=lambda values:"fail" if "fail" in values else "pass" if all(v=="pass" for v in values) else "inconclusive"
        result.update(status="complete",conditional_status=combine([entry["status"] for entry in result["cases"].values()]),
            synthetic_conditional_status=combine([result["cases"][case]["status"] for case in exported.CASES[:2]]),
            comparison_seconds=time.monotonic()-started)
        shutil.copyfile(recovery_npz,out/"recovery.npz")
        if fixed.digest(out/"recovery.npz")!=record["population"]["recovery_sha256"]:
            raise ValueError("recovery copy differs")
        if any(fixed.digest(path)!=value for path,value in protected.items()):
            raise ValueError("a protected comparison input changed")
        result["protected_comparison_inputs_unchanged"]=True
        fixed.write_json(out/"evaluation.json",result);fixed.write_json(out/"run.json",result)
        return result
    except BaseException as error:
        result.update(status="failed",error_type=type(error).__name__,error=str(error))
        fixed.write_json(out/"failure.json",result);fixed.write_json(out/"run.json",result)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("neural-evaluation","study","reference-recovery","out"):
        parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args()
    result=compare(args.neural_evaluation,args.study,args.reference_recovery,args.out)
    print(result["synthetic_conditional_status"])


if __name__=="__main__":main()
