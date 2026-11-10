PROCESS_STARTED = __import__("time").monotonic()
# Explicit environment-bootstrap recovery; frozen statistical producers unchanged.
import argparse
import copy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_heston_spread_scale_campaign as base

FROZEN_CONTROLLER_SHA256="ed5a192df30aa0dd0774b1a5cddbe240724e4b75a7f09de73678a8ccb820aa0f"
ERROR_TEXT="Unable to create the compiledir directory"


def original_deadline(campaign):
    """A conservative UTC estimate of the original first-statement deadline."""
    stamped=datetime.fromisoformat(campaign["started_utc"])
    first=float(campaign["events"][0]["seconds"])
    origin=stamped-timedelta(seconds=first)
    return origin,origin+timedelta(seconds=base.CAMPAIGN_WALL_SECONDS)


def cpu_admission(spent,active_count,additional=900.):
    return spent+active_count*900.+additional<=base.MAX_INNER_CPU_BUDGET_SECONDS


def failure_audit(campaign_path,reference_protocol_path):
    campaign,protocol=base.read(campaign_path),base.read(reference_protocol_path)
    if (campaign["status"]!="completed" or len(campaign["jobs"])!=7
            or list(campaign["jobs"])!=[job["id"] for job in protocol["jobs"]]):
        raise ValueError("original campaign did not settle the exact seven startup failures")
    rows=[]
    for spec in protocol["jobs"]:
        item=campaign["jobs"][spec["id"]]
        consumed_path=Path(protocol["evidence_root"])/(spec["id"]+".consumer.json")
        run_path=Path(spec["evidence_out"])/"run.json"
        partial_path=Path(spec["payload_out"])/"attempt-0/partial/progress.json"
        log_path=Path(item["log_path"])
        consumed,run,partial=map(base.read,(consumed_path,run_path,partial_path))
        if (item["status"]!="inconclusive" or item["exit_code"]!=1 or consumed["status"]!="failed"
                or consumed["job"]!=spec or consumed["source_sha256"]!=protocol["source_sha256"]
                or run["status"]!="failed" or run["error_type"]!="ValueError" or ERROR_TEXT not in run["error"]
                or "Check the permissions." not in run["error"] or run["attempts"]!=[]
                or run["task"]!=spec["task"] or run["observation_sha256"]!=spec["observation_sha256"]
                or partial["counts_including_warmup"]!=[0]*4 or partial["posterior_draw_counts"]!=[0]*4
                or list(Path(spec["payload_out"]).glob("attempt-*/chains.npz"))
                or "Sequential sampling" in log_path.read_text()):
            raise ValueError("an original job is not a verified zero-sampling compiledir startup failure")
        rows.append({"id":spec["id"],"error":run["error"],"sampling_started":False,
            "callback_counts_including_warmup":partial["counts_including_warmup"],"posterior_draw_counts":partial["posterior_draw_counts"],
            "completed_sampler_attempts":0,"chain_payloads":0,"inner_cpu_seconds":run["usage"]["cpu_seconds"],
            "files":{name:{"path":str(path.resolve()),"sha256":base.digest(path)} for name,path in
                (("consumer",consumed_path),("run",run_path),("partial_progress",partial_path),("log",log_path))}})
    return {"format":"heston_spread_scale_bootstrap_failure_audit_v1","all_seven_zero_sampling":True,
        "failed_startup_inner_cpu_seconds":sum(row["inner_cpu_seconds"] for row in rows),"jobs":rows,
        "scope":"Initialization likelihood evaluations occurred, but all callback counts including warmup are zero and no chain payload exists. Not a statistical retry."}


def prepare(study_path,reference_path,campaign_path,new_evidence,new_payload,logs_root,progress_path,out):
    from scripts import run_heston_spread_scale_reference as consumer
    paths=list(map(Path,(study_path,reference_path,campaign_path,new_evidence,new_payload,logs_root,progress_path,out)))
    study_path,reference_path,campaign_path,new_evidence,new_payload,logs_root,progress_path,out=paths
    if (base.digest(base.__file__)!=FROZEN_CONTROLLER_SHA256 or base.digest(study_path)!=base.STUDY_PROTOCOL_SHA256
            or base.digest(reference_path)!=base.REFERENCE_PROTOCOL_SHA256):
        raise ValueError("original frozen controller/study/reference identity differs")
    if any(path.exists() for path in (new_evidence,new_payload,logs_root,progress_path,out,progress_path.with_suffix(".claim.json"))):
        raise FileExistsError("bootstrap recovery requires new output/log/claim identities")
    audit=failure_audit(campaign_path,reference_path)
    original=base.read(reference_path)
    recovered=copy.deepcopy(original)
    recovered.update(evidence_root=str(new_evidence.resolve()),payload_root=str(new_payload.resolve()))
    nonce=uuid.uuid4().hex[:12]
    caches={}
    for job in recovered["jobs"]:
        job.update(evidence_out=str((new_evidence/job["id"]).resolve()),payload_out=str((new_payload/job["id"]).resolve()))
        cache=Path("/private/tmp")/("sbi4f-noise-scale-"+nonce+"-"+job["id"])
        cache.mkdir(mode=0o700,exist_ok=False)
        probe=cache/".write-probe"
        probe.write_text("writable bootstrap cache\n");probe.unlink()
        caches[job["id"]]={"compiledir":str(cache),"PYTENSOR_FLAGS":"compiledir="+str(cache)}
    recovered["environment_bootstrap_recovery"]={"original_reference_protocol_sha256":base.digest(reference_path),
        "change":"Explicit unique writable PyTensor compiledir per job; data/target/sampler settings unchanged", "cache_environment":caches}
    out.parent.mkdir(parents=True,exist_ok=True)
    recovered_path=out.parent/"recovery-reference-protocol.json"
    audit_path=out.parent/"bootstrap-failure-audit.json"
    if recovered_path.exists() or audit_path.exists():
        raise FileExistsError("bootstrap recovery metadata identities already exist")
    base.write(recovered_path,recovered);base.write(audit_path,audit)
    for job in recovered["jobs"]:
        consumer.validate_protocol(recovered,job["id"])
    campaign=base.read(campaign_path)
    origin,deadline=original_deadline(campaign)
    remaining=(deadline-datetime.now(timezone.utc)).total_seconds()
    if remaining<=0:
        raise TimeoutError("original campaign deadline already passed; no extension authorized")
    identity=lambda p:{"path":str(p.resolve()),"sha256":base.digest(p)}
    addendum={"format":"heston_spread_scale_reference_recovery_v1",
        "original_study":identity(study_path),"original_reference_protocol":identity(reference_path),
        "original_campaign":identity(campaign_path),"startup_failure_audit":identity(audit_path),
        "recovery_reference_protocol":identity(recovered_path),"recovered_environment_cache":caches,
        "original_started_utc":campaign["started_utc"],"conservative_original_start_utc":origin.isoformat(),
        "deadline_utc":deadline.isoformat(),"remaining_seconds_at_prepare":remaining,
        "deadline_estimate":"Original campaign started_utc minus its first launch event elapsed time; conservative relative to its first-statement clock. Reconverted to monotonic at recovery launch; no new3900-second allowance.",
        "maximum_concurrent_jobs":2,"maximum_sum_inner_cpu_seconds":6300,
        "failed_startup_inner_cpu_seconds":audit["failed_startup_inner_cpu_seconds"],
        "cpu_admission_rule":"failed_startupCPU + completed_new_actual_innerCPU + active_jobs*900 + next900 <=6300; unused completed reservations are released; unreported CPU conservatively charged900",
        "logs_root":str(logs_root.resolve()),"progress_path":str(progress_path.resolve()),
        "sources":{"recovery_controller":identity(Path(__file__)),"original_controller":identity(Path(base.__file__)),
                   "consumer":identity(Path(consumer.__file__)),"producer":identity(Path(consumer.producer.__file__))},
        "modified_reference_fields":["evidence_root","payload_root","jobs[*].evidence_out","jobs[*].payload_out","environment_bootstrap_recovery"],
        "authorization":"Explicit environment correction after seven verified zero-sampling startup failures; no scientific rerun, prior change or automatic budget extension",
        "scientific_acceptance":False,"automatic_retry":False}
    base.write(out,addendum)
    return addendum


def validate(path):
    from scripts import run_heston_spread_scale_reference as consumer
    addendum=base.read(path)
    if addendum["format"]!="heston_spread_scale_reference_recovery_v1" or base.digest(base.__file__)!=FROZEN_CONTROLLER_SHA256:
        raise ValueError("unsupported recovery or changed frozen controller")
    for value in (*addendum["sources"].values(),addendum["original_study"],addendum["original_reference_protocol"],
                  addendum["original_campaign"],addendum["startup_failure_audit"],addendum["recovery_reference_protocol"]):
        if base.digest(value["path"])!=value["sha256"]:
            raise ValueError("recovery source/data/failure evidence changed")
    if failure_audit(addendum["original_campaign"]["path"],addendum["original_reference_protocol"]["path"])!=base.read(addendum["startup_failure_audit"]["path"]):
        raise ValueError("zero-sampling failure audit differs")
    old=base.read(addendum["original_reference_protocol"]["path"])
    new=base.read(addendum["recovery_reference_protocol"]["path"])
    expected=copy.deepcopy(old)
    expected.update(evidence_root=new["evidence_root"],payload_root=new["payload_root"])
    for job in expected["jobs"]:
        job.update(evidence_out=str(Path(new["evidence_root"])/job["id"]),payload_out=str(Path(new["payload_root"])/job["id"]))
    expected["environment_bootstrap_recovery"]={"original_reference_protocol_sha256":base.digest(addendum["original_reference_protocol"]["path"]),
        "change":"Explicit unique writable PyTensor compiledir per job; data/target/sampler settings unchanged", "cache_environment":addendum["recovered_environment_cache"]}
    if new!=expected or addendum["maximum_sum_inner_cpu_seconds"]!=6300 or addendum["maximum_concurrent_jobs"]!=2:
        raise ValueError("recovery changed more than output roots/cache environment")
    for job in new["jobs"]:
        consumer.validate_protocol(new,job["id"])
        cache=addendum["recovered_environment_cache"][job["id"]]
        if (Path(cache["compiledir"]).parent!=Path("/private/tmp") or not Path(cache["compiledir"]).is_dir()
                or cache["PYTENSOR_FLAGS"]!="compiledir="+cache["compiledir"]):
            raise ValueError("recovery cache path/environment differs")
    return addendum,new,consumer


def launch_with_cache(job,consumer,reference_path,logs_root,cache):
    old=os.environ.get("PYTENSOR_FLAGS")
    os.environ["PYTENSOR_FLAGS"]=cache["PYTENSOR_FLAGS"]
    try:
        return base.launch_consumer(job,consumer.__file__,reference_path,logs_root)
    finally:
        if old is None: os.environ.pop("PYTENSOR_FLAGS",None)
        else: os.environ["PYTENSOR_FLAGS"]=old


def schedule(jobs,launch,complete,event,deadline,spent,*,clock=time.monotonic,pause=time.sleep,stopper=base.stop_groups):
    pending,active=list(jobs),{}
    try:
        while pending or active:
            if clock()>=deadline: raise TimeoutError("original reference campaign wall deadline exhausted")
            while pending and len(active)<2 and cpu_admission(spent,len(active)):
                job=pending.pop(0);entry=launch(job);active[job["id"]]={"job":job,**entry}
                event("started",job,{"pid":entry["process"].pid,"spent_inner_cpu_seconds":spent,"reserved_active_inner_cpu_seconds":len(active)*900})
            if pending and not active:
                for job in pending:event("not_started",job,{"reason":"remaining original6300CPU budget cannot reserve another900-second child","reference_ready":False})
                return "cpu_budget_limited"
            for job_id,entry in list(active.items()):
                code=entry["process"].poll()
                if code is None:continue
                result=complete(entry["job"],entry,code)
                spent+=result["charged_inner_cpu_seconds"]
                active.pop(job_id)
                event("finished",entry["job"],{**result,"spent_inner_cpu_seconds":spent})
            if active:
                event("poll",None,{"active_job_ids":list(active),"pending_job_ids":[j["id"] for j in pending]})
                pause(min(.2,max(0.,deadline-clock())))
        return "completed"
    except BaseException as error:
        stopper(active,deadline,clock=clock,pause=pause)
        for entry in active.values():event("stopped",entry["job"],{"reason":type(error).__name__,"reference_ready":False})
        for job in pending:event("not_started",job,{"reason":type(error).__name__,"reference_ready":False})
        raise


def run(path):
    addendum,protocol,consumer=validate(path)
    remaining=(datetime.fromisoformat(addendum["deadline_utc"])-datetime.now(timezone.utc)).total_seconds()
    if remaining<=0:raise TimeoutError("original campaign deadline passed")
    deadline=time.monotonic()+remaining
    progress_path,logs_root=Path(addendum["progress_path"]),Path(addendum["logs_root"])
    if progress_path.exists() or logs_root.exists():raise FileExistsError("recovery campaign outputs already exist")
    for job in protocol["jobs"]:
        if any(p.exists() for p in (Path(job["evidence_out"]),Path(job["payload_out"]),Path(protocol["evidence_root"])/(job["id"]+".consumer.json"))):
            raise FileExistsError("recovery job already claimed; no retry")
    with progress_path.with_suffix(".claim.json").open("x") as stream:json.dump({"recovery_addendum_sha256":base.digest(path),"pid":os.getpid()},stream)
    logs_root.mkdir(parents=True,exist_ok=False)
    progress={"format":"heston_spread_scale_recovery_campaign_v1","status":"running","pid":os.getpid(),
        "recovery_addendum":{"path":str(Path(path).resolve()),"sha256":base.digest(path)},"deadline_utc":addendum["deadline_utc"],
        "failed_startup_inner_cpu_seconds":addendum["failed_startup_inner_cpu_seconds"],"events":[],"scientific_acceptance":False,
        "jobs":{j["id"]:{"status":"pending","reference_ready":False} for j in protocol["jobs"]}}
    last_write=0.
    def event(kind,job,details):
        nonlocal last_write
        seconds=time.monotonic()-PROCESS_STARTED
        progress["recovery_seconds"]=seconds
        progress["remaining_original_wall_seconds"]=max(0.,deadline-time.monotonic())
        if kind=="poll":
            progress["active_job_ids"]=details["active_job_ids"]
            if seconds-last_write<2:return
        else:
            progress["events"].append({"kind":kind,"job_id":job["id"],"recovery_seconds":seconds,**details})
            entry=progress["jobs"][job["id"]]
            entry.update(**details,status="running" if kind=="started" else details.get("status",kind))
            if kind=="started":entry["log_path"]=str(logs_root/(job["id"]+".log"))
        base.write(progress_path,progress);last_write=seconds
    def launch(job):
        return launch_with_cache(job,consumer,addendum["recovery_reference_protocol"]["path"],logs_root,addendum["recovered_environment_cache"][job["id"]])
    def complete(job,entry,code):
        for sig in (signal.SIGTERM,signal.SIGKILL):
            try:os.killpg(entry["process"].pid,sig)
            except ProcessLookupError:pass
        result={"status":"inconclusive","exit_code":code,"reference_ready":False,"charged_inner_cpu_seconds":900.,
                "log_path":entry["log_path"],"log_sha256":base.digest(entry["log_path"])}
        consumption_path=Path(protocol["evidence_root"])/(job["id"]+".consumer.json")
        if consumption_path.exists():
            consumed=base.read(consumption_path)
            if consumed["protocol_sha256"]!=addendum["recovery_reference_protocol"]["sha256"] or consumed["job"]!=job or consumed["source_sha256"]!=protocol["source_sha256"]:
                raise ValueError("recovered child target/source binding differs")
            result.update(consumer=consumed,consumer_result_path=str(consumption_path),consumer_result_sha256=base.digest(consumption_path))
            usage=consumed.get("result",{}).get("usage")
            if usage is not None:result["charged_inner_cpu_seconds"]=max(0.,float(usage["cpu_seconds"]))
            if code==0 and consumed["status"]=="completed":result.update(status="completed",reference_ready=bool(consumed["result"]["reference_ready"]))
        elif code==0:raise ValueError("successful recovery child has no bound consumer record")
        return result
    base.write(progress_path,progress)
    handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    def interrupted(signum,frame):raise KeyboardInterrupt(f"recovery campaign interrupted by signal{signum}")
    for sig in handlers:signal.signal(sig,interrupted)
    try:
        status=schedule(protocol["jobs"],launch,complete,event,deadline,addendum["failed_startup_inner_cpu_seconds"])
        validate(path)
        progress.update(status=status,recovery_seconds=time.monotonic()-PROCESS_STARTED,active_job_ids=[],
            ready_reference_job_ids=[j for j,v in progress["jobs"].items() if v["reference_ready"]])
        base.write(progress_path,progress);return progress
    except BaseException as error:
        progress.update(status="budget_exhausted" if isinstance(error,TimeoutError) else "interrupted" if isinstance(error,KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,error=str(error),recovery_seconds=time.monotonic()-PROCESS_STARTED,active_job_ids=[])
        base.write(progress_path,progress);raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser(description="Recover only verified zero-sampling PyTensor-cache startup failures under the original study budgets.")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare")
    for name in ("study","reference-protocol","original-campaign","evidence-root","payload-root","logs-root","progress","out"):
        p.add_argument("--"+name,type=Path,required=True)
    p=sub.add_parser("run");p.add_argument("--recovery",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="prepare":
        result=prepare(args.study,args.reference_protocol,args.original_campaign,args.evidence_root,args.payload_root,args.logs_root,args.progress,args.out)
        print(json.dumps({"addendum":str(args.out),"sha256":base.digest(args.out),"remaining_original_wall_seconds":result["remaining_seconds_at_prepare"]}))
    else:
        addendum=base.read(args.recovery)
        remaining=(datetime.fromisoformat(addendum["deadline_utc"])-datetime.now(timezone.utc)).total_seconds()
        if remaining<=0:raise TimeoutError("original campaign deadline passed")
        old=signal.getsignal(signal.SIGALRM)
        def exhausted(signum,frame):raise TimeoutError("original campaign wall deadline alarm exhausted")
        signal.signal(signal.SIGALRM,exhausted);signal.setitimer(signal.ITIMER_REAL,remaining)
        try:
            result=run(args.recovery)
            print(json.dumps({"status":result["status"],"ready_reference_job_ids":result["ready_reference_job_ids"]}))
        finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)


if __name__=="__main__":main()
