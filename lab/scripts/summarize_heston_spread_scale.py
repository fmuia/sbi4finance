"""Bind compact noise-scale evidence for a presentation-only report.

No fitting, posterior sampling, option pricing or bank reads occur. Ready
references are revalidated before deriving descriptive fixed-target sensitivity.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_heston_spread_scale_reference as consumer
from scripts import train_heston_spread_baseline as fixed


def read(path):
    return json.loads(Path(path).read_text())


def artifact(path):
    path=Path(path)
    return {"path":str(path.resolve()),"sha256":fixed.digest(path)}


def ready_summary(protocol,spec):
    summary=consumer.summarize_run(protocol,spec)
    result={"case_id":spec["case_id"],"mode":spec["mode"],**summary,
        "sources":[artifact(Path(spec["evidence_out"])/"run.json")]}
    if not summary["reference_ready"]:
        return result
    chain_path=Path(summary["chain_payload_path"])
    diagnostic_path=Path(spec["evidence_out"])/"diagnostics.json"
    with np.load(chain_path,allow_pickle=False) as archive:
        chains=archive["parameters"].copy()
    derived=consumer.producer.product_values(chains,protocol["products"]).reshape(-1,len(protocol["products"]))
    result["products"]={name:{"interval":interval,"quantiles_05_50_95":np.quantile(derived[:,i],[.05,.5,.95]).tolist()}
        for i,(name,interval) in enumerate(protocol["products"].items())}
    posterior_path=Path(spec["evidence_out"])/"posterior.json"
    posterior=read(posterior_path)
    if posterior["products"]!=result["products"]:
        raise ValueError("derived reference product summaries differ from producer record")
    result["sources"].extend([artifact(diagnostic_path),artifact(posterior_path)])
    result["chain_payload_identity"]=artifact(chain_path)
    return result


def build(study_path,reference_path,out,*,recovery=None):
    study_path,reference_path,out=map(Path,(study_path,reference_path,out))
    study,protocol=read(study_path),read(reference_path)
    consumer.validate_protocol(protocol,protocol["jobs"][0]["id"])
    directory=study_path.parent
    if protocol["panel_sha256"]!=study["panel"]["sha256"]:
        raise ValueError("summary reference panel differs from scale study")
    refs=[]
    for case,record in protocol["fixed_c1_references"].items():
        posterior_path=Path(record["run_path"]).with_name("posterior.json")
        posterior=read(posterior_path)
        refs.append({"case_id":case,"mode":"one","status":"completed","reference_ready":True,
            **posterior,"sources":[artifact(record["run_path"]),artifact(record["diagnostics_path"]),artifact(posterior_path),
                                    ],"chain_payload_identity":artifact(record["chain_payload_path"])})
    for spec in protocol["jobs"]:
        consumer_path=Path(protocol["evidence_root"])/(spec["id"]+".consumer.json")
        if not (Path(spec["evidence_out"])/"run.json").exists():
            refs.append({"case_id":spec["case_id"],"mode":spec["mode"],"status":"inconclusive",
                "reference_ready":False,"reason":"no complete reference record","sources":[]})
            continue
        summary=ready_summary(protocol,spec)
        if consumer_path.exists():summary["sources"].append(artifact(consumer_path))
        refs.append(summary)
    evaluations={f"{step:06d}":artifact(directory/"evaluations"/f"{step:06d}"/"evaluation.json") for step in (10000,50000,60000)}
    artifacts={"study":artifact(study_path),"reference_protocol":artifact(reference_path),
        "panel":artifact(study["panel"]["path"]),"training_curve":artifact(directory/"training"/"learning-curve.json"),
        "training_record":artifact(directory/"training"/"run.json"),
        "training_protocol":artifact(directory/"training-protocol.json"),
        "trainer_source":artifact(Path(__file__).with_name("train_heston_spread_scale.py")),
        "fixed_helper_source":artifact(Path(fixed.__file__)),"evaluations":evaluations}
    if recovery is not None:artifacts["recovery"]=artifact(recovery)
    result={"format":"heston_spread_scale_report_summary_v1","artifacts":artifacts,"references":refs,
        "sources":{"summary_producer":artifact(__file__)},"scientific_acceptance":False,
        "scope":"Fixed-target sensitivity is descriptive; only same-target joint neural/reference comparisons have pass/fail checks. Prior-predictive data are development rows."}
    fixed.write_json(out,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("study","reference-protocol","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--recovery",type=Path)
    args=parser.parse_args()
    result=build(args.study,args.reference_protocol,args.out,recovery=args.recovery)
    print(json.dumps({"out":str(args.out),"references":len(result["references"])}))


if __name__=="__main__":main()
