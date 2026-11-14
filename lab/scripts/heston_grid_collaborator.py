"""Relocatable development export of the source-bound actual63 Heston study.

Original records/payload bytes are copied unchanged. This derived container
validates their identities without following historical owner-specific paths.
It permits a new offline fit or inference, not generation continuation or
scientific acceptance. Financial and native-network code stays canonical.
"""
import argparse
from importlib import metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import reproduce_heston_pr87_grid as producer

FORMAT = "heston_actual63_collaborator_bundle_v1"
AUTHORITATIVE_DATA = "47c7f1f25756162985d437e6a133c2651656e7c506bdc2b8aeb39b33fcb803c4"
AUTHORITATIVE_WHITENING = "1423b9b5bbb753a6ade0b82b83a489c6097faf5580f191ca427d03775c24c78d"
ROOT = Path(__file__).resolve().parents[2]
read = lambda p: json.loads(Path(p).read_text())
digest = producer.digest


def source_entries():
    result={}
    for key,spec in producer.source_bindings().items():
        path=Path(spec["path"]).resolve()
        if ROOT in path.parents:
            result[key]={"kind":"repository","relative_path":str(path.relative_to(ROOT)),"sha256":spec["sha256"]}
        else:
            parts=path.parts
            if "site-packages" not in parts: raise ValueError("unclassified external source: "+str(path))
            relative=Path(*parts[parts.index("site-packages")+1:]); distribution=relative.parts[0]
            result[key]={"kind":"installed_python_source","distribution":distribution,
                "version":metadata.version(distribution),"relative_path":str(relative),"sha256":spec["sha256"]}
    result["collaborator_loader"]={"kind":"repository","relative_path":str(Path(__file__).resolve().relative_to(ROOT)),"sha256":digest(__file__)}
    return result


def export(payload, evidence, out):
    payload, evidence, out = map(lambda p: Path(p).resolve(), (payload, evidence, out))
    if out.exists(): raise FileExistsError("new collaborator export identity required")
    bank = read(payload/"bank/bank.json")
    if (bank.get("format") != "heston_pr87_grid_bank_v1" or bank.get("status") != "complete"
            or bank["protocol"]["train_rows"] != 300000 or bank["task"]["target"]["quote_count"] != 63
            or digest(payload/"bank/data.npz") != bank["data_payload_sha256"]
            or bank["data_payload_sha256"] != AUTHORITATIVE_DATA
            or bank["whitening_sha256"] != AUTHORITATIVE_WHITENING):
        raise ValueError("completed300k actual63 bank required")
    out.mkdir(parents=True)
    files = {}
    def copy(source, name):
        destination = out/name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if digest(source) != digest(destination): raise ValueError("copy differs: "+name)
        files[name] = {"sha256": digest(destination), "bytes": destination.stat().st_size}
    for name in ("data.npz", "bank.json", "whitening.json"):
        copy(payload/"bank"/name, "bank/"+name)
    copy(Path(bank["protocol_path"]), "provenance/bank-protocol.json")
    producer.write_json(out/"task.json", bank["task"])
    files["task.json"] = {"sha256": digest(out/"task.json"), "bytes": (out/"task.json").stat().st_size}
    models = {}
    for run_id in ("200k-300", "300k-300"):
        training = payload/run_id/"training"; run = read(training/"training.json")
        if run.get("status") != "complete" or run["task"] != bank["task"]:
            raise ValueError("completed same-task fits required")
        copy(training/"training.json", f"provenance/{run_id}-training.json")
        copy(Path(run["protocol_path"]), f"provenance/{run_id}-protocol.json")
        copy(training/"native-epoch-history.json", f"provenance/{run_id}-history.json")
        for role, checkpoint in (("final300", "epoch-000300"), ("minimum", "best-validation")):
            label = f"{run_id}-{role}"; source = training/"checkpoints"/checkpoint
            record = read(source/"checkpoint.json")
            if record["task"] != bank["task"] or record["data_payload_sha256"] != bank["data_payload_sha256"]:
                raise ValueError("checkpoint task/bank identity differs")
            for name,key in (("model.keras","model_sha256"),("state.npz","state_sha256"),("whitening.json","whitening_sha256")):
                if digest(source/name) != record[key]: raise ValueError("origin checkpoint payload differs")
            for name in ("checkpoint.json", "model.keras", "state.npz", "whitening.json"):
                copy(source/name, f"checkpoints/{label}/{name}")
            models[label] = {"directory": f"checkpoints/{label}", "train_rows": run["train_rows"],
                "epoch": record["native_epoch"], "primary": role == "final300",
                "origin_training": f"provenance/{run_id}-training.json"}
    for name in ("report.html", "independent-audit.json", "reference-width-comparison.json"):
        copy(evidence/name, "evidence/"+name)
    copy(Path(__file__).resolve(), "support/heston_grid_collaborator.py")
    sources = source_entries()
    manifest = {"format": FORMAT, "source_commit": subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "source_branch": "feat/heston-spread-baseline-20261109", "files": files, "sources": sources,
        "task_fingerprint": producer.fingerprint(bank["task"]), "data_arrays": bank["data_arrays"], "models": models,
        "splits": {"train":300000,"validation":100,"historical":512}, "whitening_training_rows":10000,
        "scientific_acceptance": False, "resume_generation_supported": False,
        "scope": "Development-only offline training/inference export; original provenance bytes retained; no tutorial asset promotion."}
    producer.write_json(out/"bundle.json", manifest)
    verify(out)
    return manifest


def verify(directory, *, bank_payload=True):
    directory = Path(directory).resolve(); manifest = read(directory/"bundle.json")
    if manifest.get("format") != FORMAT or manifest.get("scientific_acceptance") is not False:
        raise ValueError("development actual63 bundle required")
    for name, spec in manifest["files"].items():
        if not bank_payload and name == "bank/data.npz": continue
        path = (directory/name).resolve()
        if directory not in path.parents or not path.is_file() or path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
            raise ValueError("bundle member missing/corrupt: "+name)
    for spec in manifest["sources"].values():
        if spec["kind"]=="repository": path=ROOT/spec["relative_path"]
        elif spec["kind"]=="installed_python_source":
            if metadata.version(spec["distribution"])!=spec["version"]: raise ValueError("installed source version differs")
            path=Path(metadata.distribution(spec["distribution"]).locate_file(spec["relative_path"]))
        else: raise ValueError("unknown source binding kind")
        if digest(path) != spec["sha256"]:
            raise ValueError("source closure differs: "+spec["relative_path"])
    task = producer.read_task(read(directory/"task.json"))
    bank = read(directory/"bank/bank.json")
    if (task != bank["task"] or producer.fingerprint(task) != manifest["task_fingerprint"]
            or bank["data_payload_sha256"] != AUTHORITATIVE_DATA
            or bank["whitening_sha256"] != AUTHORITATIVE_WHITENING
            or bank["data_payload_sha256"] != manifest["files"]["bank/data.npz"]["sha256"]
            or bank["whitening_sha256"] != manifest["files"]["bank/whitening.json"]["sha256"]):
        raise ValueError("exported original bank/task metadata differ")
    return directory, manifest, task, bank


def _whitening(path):
    """Restore the hash-pinned audited matrix; never recompute BLAS/eigenvectors.

    The original producer's exact reconstruction check is platform-specific.
    Here identical saved bytes and shape/finite checks preserve the authoritative
    first10k-only transform on a collaborator's CPU without refitting it.
    """
    if digest(path) != AUTHORITATIVE_WHITENING: raise ValueError("authoritative whitening bytes differ")
    record=read(path); w=producer.NotebookWhitening.__new__(producer.NotebookWhitening)
    for name,key in (("mean","mean"),("covariance","covariance"),("eigenvalues","eigenvalues"),
                     ("eigenvectors","eigenvectors"),("matrix","whitening_matrix")):
        setattr(w,name,np.asarray(record[key],dtype=float))
    w.floor,w.training_rows=record["eigenvalue_floor"],record["training_rows"]
    if (w.mean.shape!=(63,) or any(x.shape!=(63,63) for x in (w.covariance,w.eigenvectors,w.matrix))
            or w.eigenvalues.shape!=(63,) or not all(np.isfinite(x).all() for x in
                (w.mean,w.covariance,w.eigenvalues,w.eigenvectors,w.matrix))
            or w.training_rows!=10000 or w.to_dict()!=record):
        raise ValueError("authoritative whitening structure differs")
    return w


def load_bank(directory):
    directory, manifest, task, bank = verify(directory)
    with np.load(directory/"bank/data.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    producer._validate_arrays(arrays,task,bank["protocol"])
    if {name:producer.array_summary(value) for name,value in arrays.items()} != manifest["data_arrays"]:
        raise ValueError("bank array identities differ")
    return task, arrays, bank


def training_inputs(directory, rows=300000):
    if rows not in (200000,300000): raise ValueError("declared200k prefix or300k full bank required")
    task, arrays, bank = load_bank(directory)
    whitening = _whitening(Path(directory)/"bank/whitening.json")
    data = {}
    for split,n in (("train",rows),("validation",100)):
        data[split] = {name: arrays[f"{split}_{name}"][:n] for name in task["target"]["parameter_names"]}
        # Stored observations are Y/h already. Whiten once in host float64.
        data[split]["observations"] = whitening.transform(arrays[f"{split}_observations"][:n])
    return task,data


def load_checkpoint(directory, label="300k-300-final300"):
    directory, manifest, task, bank = verify(directory,bank_payload=False)
    if label not in manifest["models"]: raise ValueError("unknown exported checkpoint")
    entry = manifest["models"][label]; path=directory/entry["directory"]; record=read(path/"checkpoint.json")
    run=read(directory/entry["origin_training"])
    if (record["task"] != task or run["task"] != task or record["native_epoch"] != entry["epoch"]
            or record["train_rows"] != entry["train_rows"] or record["data_payload_sha256"] != bank["data_payload_sha256"]
            or record["whitening_sha256"] != bank["whitening_sha256"] or record["protocol"] != run["protocol"]
            or record["actual_optimizer_updates"] != entry["epoch"]*record["protocol"]["batches_per_epoch"]):
        raise ValueError("exported checkpoint/origin metadata differ")
    for name,key in (("model.keras","model_sha256"),("state.npz","state_sha256"),("whitening.json","whitening_sha256")):
        if digest(path/name) != record[key]: raise ValueError("checkpoint member identity differs")
    workflow=producer.build_workflow(task);workflow.load_estimator(path/"model.keras")
    with np.load(path/"state.npz",allow_pickle=False) as archive:
        keys={f"{group}_{i:04d}" for group in producer.GROUPS for i in range(len(record["state"][group]))}
        if set(archive.files)!=keys: raise ValueError("full state inventory differs")
        state=tuple([archive[f"{group}_{i:04d}"].copy() for i in range(len(record["state"][group]))] for group in producer.GROUPS)
    producer._assign_state(workflow.estimator,state)
    actual=producer._state_summary(workflow.estimator,producer._state(workflow.estimator))
    fields=lambda groups:{g:[{k:a[k] for k in ("shape","dtype","sha256")} for a in values] for g,values in groups.items()}
    if (fields(actual)!=fields(record["state"]) or int(workflow.estimator.optimizer.iterations.numpy())!=record["actual_optimizer_updates"]
            or producer._stable_config(__import__("keras").optimizers.serialize(workflow.estimator.optimizer))!=record["protocol"]["resolved_defaults"]["optimizer"]):
        raise ValueError("native full state differs after relocatable reload")
    whitening=_whitening(path/"whitening.json")
    return producer.RawQuoteWorkflow(workflow,task,whitening),record


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="role",required=True)
    create=sub.add_parser("export")
    for key in ("payload","evidence","out"): create.add_argument("--"+key,type=Path,required=True)
    check=sub.add_parser("verify");check.add_argument("--directory",type=Path,required=True)
    args=parser.parse_args()
    if args.role=="export":
        result=export(args.payload,args.evidence,args.out); print(json.dumps({"format":result["format"],"files":len(result["files"]),"scientific_acceptance":False}))
    else:
        _,result,_,_=verify(args.directory);print(json.dumps({"files_verified":len(result["files"]),"scientific_acceptance":False}))
