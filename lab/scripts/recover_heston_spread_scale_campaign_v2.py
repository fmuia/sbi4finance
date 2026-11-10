PROCESS_STARTED=__import__("time").monotonic()
# Cache-complete, direct-child bootstrap recovery; no frozen producer edits.
import argparse
import copy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import recover_heston_spread_scale_campaign as old
base=old.base
OLD_RECOVERY_SHA256="04ed9acd4b30873aa25c4c6be2ac24c77dfb6e805939c6aa4b133c7e49d195ab"


def identity(path):
    return {"path":str(Path(path).resolve()),"sha256":base.digest(path)}


def audit_extra_failures(addendum_path,campaign_path):
    addendum=base.read(addendum_path);protocol=base.read(addendum["recovery_reference_protocol"]["path"])
    campaign=base.read(campaign_path)
    rows=[]
    for spec in protocol["jobs"]:
        run_path=Path(spec["evidence_out"])/"run.json"
        if not run_path.exists():
            if (Path(protocol["evidence_root"])/(spec["id"]+".consumer.json")).exists():raise ValueError("unexpected partial recovery claim")
            continue
        consumed_path=Path(protocol["evidence_root"])/(spec["id"]+".consumer.json")
        partial_path=Path(spec["payload_out"])/"attempt-0/partial/progress.json"
        run,consumed,partial=map(base.read,(run_path,consumed_path,partial_path))
        log=Path(campaign["jobs"][spec["id"]]["log_path"])
        if (run["status"]!="failed" or run["error_type"]!="PermissionError" or "/.pytensor/numba/" not in run["error"]
                or run["attempts"]!=[] or partial["counts_including_warmup"]!=[0]*4 or partial["posterior_draw_counts"]!=[0]*4
                or list(Path(spec["payload_out"]).glob("attempt-*/chains.npz")) or "Sequential sampling" in log.read_text()
                or consumed["job"]!=spec or consumed["source_sha256"]!=protocol["source_sha256"]):
            raise ValueError("previous recovery is not a verified zero-sampling Numba-cache failure")
        rows.append({"id":spec["id"],"sampling_started":False,"callback_counts_including_warmup":[0]*4,
            "posterior_draw_counts":[0]*4,"inner_cpu_seconds":run["usage"]["cpu_seconds"],
            "files":{name:identity(path) for name,path in (("run",run_path),("consumer",consumed_path),("partial",partial_path),("log",log))}})
    if [row["id"] for row in rows]!=[job["id"] for job in protocol["jobs"][:2]]:
        raise ValueError("expected exactly two additional startup-only failures")
    return {"format":"heston_spread_scale_bootstrap_extra_failures_v2","jobs":rows,
        "inner_cpu_seconds":sum(row["inner_cpu_seconds"] for row in rows),"zero_mcmc_draws":True,
        "controller_errors_preserved":["duplicate status keyword during progress update","process-group signal denied by sandbox"],
        "process_groups_exited_verified":"read-only process inspection after failure found no remaining members of groups74846/74847"}


def prepare(previous_path,previous_campaign,preflight_path,new_evidence,new_payload,logs,progress,out):
    from scripts import run_heston_spread_scale_reference as consumer
    previous_path,previous_campaign,preflight_path,new_evidence,new_payload,logs,progress,out=map(Path,
        (previous_path,previous_campaign,preflight_path,new_evidence,new_payload,logs,progress,out))
    previous,_,_=old.validate(previous_path)
    if base.digest(old.__file__)!=OLD_RECOVERY_SHA256:raise ValueError("frozen previous recovery source differs")
    if any(path.exists() for path in (new_evidence,new_payload,logs,progress,out,progress.with_suffix(".claim.json"))):
        raise FileExistsError("v2 requires new outputs and claims")
    preflight=base.read(preflight_path)
    if (preflight["status"]!="passed" or preflight["mcmc_draws"]!=0 or preflight["heston_prices"]!=0
            or not preflight["result"]["symbolic_cvm_compilation"] or not preflight["result"]["symbolic_numba_compilation"]):
        raise ValueError("both actual symbolic C and Numba compilation preflights must pass first")
    extra=audit_extra_failures(previous_path,previous_campaign)
    original=base.read(previous["original_reference_protocol"]["path"])
    recovered=copy.deepcopy(original)
    recovered.update(evidence_root=str(new_evidence.resolve()),payload_root=str(new_payload.resolve()))
    nonce=uuid.uuid4().hex[:12];caches={}
    for job in recovered["jobs"]:
        job.update(evidence_out=str((new_evidence/job["id"]).resolve()),payload_out=str((new_payload/job["id"]).resolve()))
        cache=Path("/private/tmp")/("sbi4f-scale-v2-"+nonce+"-"+job["id"])
        cache.mkdir(mode=0o700,exist_ok=False)
        caches[job["id"]]={"base_compiledir":str(cache),"compiledir":str(cache/"compiled"),
            "PYTENSOR_FLAGS":f"base_compiledir={cache},compiledir={cache}/compiled","NUMBA_CACHE_DIR":str(cache/"numba-native")}
    recovered["environment_bootstrap_recovery"]={"original_reference_protocol_sha256":previous["original_reference_protocol"]["sha256"],
        "change":"Explicit writable base_compiledir,compiledir,NUMBA_CACHE_DIR per direct worker; statistical settings unchanged","cache_environment":caches}
    recovered_path=out.parent/"recovery-v2-reference-protocol.json"
    audit_path=out.parent/"bootstrap-extra-failure-audit-v2.json"
    if recovered_path.exists() or audit_path.exists():raise FileExistsError("v2 metadata identities already exist")
    base.write(recovered_path,recovered);base.write(audit_path,extra)
    for job in recovered["jobs"]:consumer.validate_protocol(recovered,job["id"])
    deadline=previous["deadline_utc"]
    remaining=(datetime.fromisoformat(deadline)-datetime.now(timezone.utc)).total_seconds()
    if remaining<=0:raise TimeoutError("original deadline passed")
    failed_cpu=previous["failed_startup_inner_cpu_seconds"]+extra["inner_cpu_seconds"]
    value={"format":"heston_spread_scale_reference_recovery_v1",
        "original_study":previous["original_study"],"original_reference_protocol":previous["original_reference_protocol"],
        "original_campaign":previous["original_campaign"],"startup_failure_audit":previous["startup_failure_audit"],
        "previous_recovery":identity(previous_path),"previous_recovery_campaign":identity(previous_campaign),
        "extra_startup_failure_audit":identity(audit_path),"symbolic_preflight":identity(preflight_path),
        "recovery_reference_protocol":identity(recovered_path),"recovered_environment_cache":caches,
        "original_started_utc":previous["original_started_utc"],"deadline_utc":deadline,"remaining_seconds_at_prepare":remaining,
        "failed_startup_inner_cpu_seconds":failed_cpu,"preflight_cpu_seconds":preflight["cpu_seconds"],
        "initial_charged_cpu_seconds":failed_cpu+preflight["cpu_seconds"],
        "maximum_sum_inner_cpu_seconds":6300,"maximum_concurrent_jobs":2,
        "cpu_admission_rule":"initialchargedCPU + completedactualinnerCPU + activeworkers*900 + next900 <=6300",
        "logs_root":str(logs.resolve()),"progress_path":str(progress.resolve()),
        "sources":{"recovery_controller":identity(Path(__file__)),"previous_recovery_controller":identity(Path(old.__file__)),
                   "original_controller":identity(Path(base.__file__)),"consumer":identity(Path(consumer.__file__)),"producer":identity(Path(consumer.producer.__file__))},
        "process_topology":"consumer --worker directly in one Popen; no nested consumer supervisor; native PyMC cores1. Controller owns950s outer timeout and signals direct Popen only.",
        "authorization":"Manual environment/controller correction after zero MCMC draws; original failures, deadline and CPU cost retained. No automatic statistical retry or budget extension.",
        "scientific_acceptance":False,"automatic_retry":False}
    base.write(out,value);return value


def validate(path):
    from scripts import run_heston_spread_scale_reference as consumer
    addendum=base.read(path)
    for key in ("original_study","original_reference_protocol","original_campaign","startup_failure_audit","previous_recovery",
                "previous_recovery_campaign","extra_startup_failure_audit","symbolic_preflight","recovery_reference_protocol"):
        item=addendum[key]
        if base.digest(item["path"])!=item["sha256"]:raise ValueError("v2 evidence/protocol changed")
    for item in addendum["sources"].values():
        if base.digest(item["path"])!=item["sha256"]:raise ValueError("v2 source changed")
    previous,_,_=old.validate(addendum["previous_recovery"]["path"])
    extra=audit_extra_failures(addendum["previous_recovery"]["path"],addendum["previous_recovery_campaign"]["path"])
    preflight=base.read(addendum["symbolic_preflight"]["path"])
    if extra!=base.read(addendum["extra_startup_failure_audit"]["path"]) or addendum["deadline_utc"]!=previous["deadline_utc"]:
        raise ValueError("v2 changed failure accounting or original deadline")
    failed_cpu=previous["failed_startup_inner_cpu_seconds"]+extra["inner_cpu_seconds"]
    if (addendum["failed_startup_inner_cpu_seconds"]!=failed_cpu or preflight["status"]!="passed"
            or addendum["preflight_cpu_seconds"]!=preflight["cpu_seconds"]
            or addendum["initial_charged_cpu_seconds"]!=failed_cpu+preflight["cpu_seconds"]
            or addendum["maximum_sum_inner_cpu_seconds"]!=6300 or addendum["maximum_concurrent_jobs"]!=2):
        raise ValueError("v2 changed the cumulative CPU budget or preflight charge")
    original=base.read(addendum["original_reference_protocol"]["path"])
    recovered=base.read(addendum["recovery_reference_protocol"]["path"])
    expected=copy.deepcopy(original);expected.update(evidence_root=recovered["evidence_root"],payload_root=recovered["payload_root"])
    for job in expected["jobs"]:job.update(evidence_out=str(Path(recovered["evidence_root"])/job["id"]),payload_out=str(Path(recovered["payload_root"])/job["id"]))
    expected["environment_bootstrap_recovery"]={"original_reference_protocol_sha256":addendum["original_reference_protocol"]["sha256"],
        "change":"Explicit writable base_compiledir,compiledir,NUMBA_CACHE_DIR per direct worker; statistical settings unchanged","cache_environment":addendum["recovered_environment_cache"]}
    if recovered!=expected:raise ValueError("v2 changed target/data/sampler settings")
    for job in recovered["jobs"]:
        consumer.validate_protocol(recovered,job["id"])
        cache=addendum["recovered_environment_cache"][job["id"]]
        root=Path(cache["base_compiledir"])
        if (root.parent!=Path("/private/tmp") or not root.is_dir() or cache["compiledir"]!=str(root/"compiled")
                or cache["NUMBA_CACHE_DIR"]!=str(root/"numba-native")
                or cache["PYTENSOR_FLAGS"]!=f"base_compiledir={root},compiledir={root}/compiled"):
            raise ValueError("v2 cache environment differs")
    return addendum,recovered,consumer


class DirectWorker:
    def __init__(self,process,seconds=950.,clock=time.monotonic):
        self.process,self.clock,self.deadline=process,clock,clock()+seconds
        self.pid=process.pid;self.timed_out=False
    def poll(self):
        code=self.process.poll()
        if code is None and self.clock()>=self.deadline:
            self.timed_out=True;self.process.terminate()
            try:self.process.wait(timeout=.5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
            code=self.process.returncode
        return code
    def wait(self):return self.process.wait()


def stop_direct(active,deadline,*,clock=time.monotonic,pause=time.sleep):
    for entry in active.values():
        process=entry["process"].process
        if process.poll() is None:process.terminate()
    for entry in active.values():
        process=entry["process"].process
        try:process.wait(timeout=min(.5,max(.001,deadline-clock())))
        except subprocess.TimeoutExpired:process.kill();process.wait()


def update_job(entry,kind,details):
    entry.update(details)
    entry["status"]="running" if kind=="started" else details.get("status",kind)


def run(path):
    addendum,protocol,consumer=validate(path)
    remaining=(datetime.fromisoformat(addendum["deadline_utc"])-datetime.now(timezone.utc)).total_seconds()
    if remaining<=0:raise TimeoutError("original campaign deadline passed")
    deadline=time.monotonic()+remaining
    progress_path,logs_root=Path(addendum["progress_path"]),Path(addendum["logs_root"])
    if progress_path.exists() or logs_root.exists():raise FileExistsError("v2 campaign outputs already exist")
    for job in protocol["jobs"]:
        if any(p.exists() for p in (Path(job["evidence_out"]),Path(job["payload_out"]),Path(protocol["evidence_root"])/(job["id"]+".consumer.json"))):raise FileExistsError("v2 job previously claimed")
    with progress_path.with_suffix(".claim.json").open("x") as stream:json.dump({"pid":os.getpid(),"recovery_sha256":base.digest(path)},stream)
    logs_root.mkdir(parents=True,exist_ok=False)
    progress={"format":"heston_spread_scale_recovery_v2_campaign_v1","status":"running","pid":os.getpid(),"recovery_addendum":identity(path),
        "deadline_utc":addendum["deadline_utc"],"initial_charged_cpu_seconds":addendum["initial_charged_cpu_seconds"],"events":[],
        "jobs":{j["id"]:{"status":"pending","reference_ready":False} for j in protocol["jobs"]},"scientific_acceptance":False}
    last_write=0.
    def event(kind,job,details):
        nonlocal last_write
        seconds=time.monotonic()-PROCESS_STARTED;progress["recovery_seconds"]=seconds
        progress["remaining_original_wall_seconds"]=max(0.,deadline-time.monotonic())
        if kind=="poll":
            progress["active_job_ids"]=details["active_job_ids"]
            if seconds-last_write<2:return
        else:
            progress["events"].append({"kind":kind,"job_id":job["id"],"seconds":seconds,**details})
            update_job(progress["jobs"][job["id"]],kind,details)
        base.write(progress_path,progress);last_write=seconds
    def launch(job):
        cache=addendum["recovered_environment_cache"][job["id"]]
        env=dict(os.environ,PYTENSOR_FLAGS=cache["PYTENSOR_FLAGS"],NUMBA_CACHE_DIR=cache["NUMBA_CACHE_DIR"],
                 OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
        log_path=logs_root/(job["id"]+".log")
        command=[sys.executable,consumer.__file__,"run","--protocol",addendum["recovery_reference_protocol"]["path"],"--job",job["id"],"--worker"]
        with log_path.open("x") as stream:process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        return {"process":DirectWorker(process),"log_path":str(log_path)}
    def complete(job,entry,code):
        result={"status":"inconclusive","exit_code":code,"reference_ready":False,"charged_inner_cpu_seconds":900.,
            "outer_timed_out":entry["process"].timed_out,"log_path":entry["log_path"],"log_sha256":base.digest(entry["log_path"])}
        p=Path(protocol["evidence_root"])/(job["id"]+".consumer.json")
        if p.exists():
            consumed=base.read(p)
            if consumed["protocol_sha256"]!=addendum["recovery_reference_protocol"]["sha256"] or consumed["job"]!=job or consumed["source_sha256"]!=protocol["source_sha256"]:raise ValueError("v2 result target/source changed")
            result.update(consumer=consumed,consumer_result_path=str(p),consumer_result_sha256=base.digest(p))
            usage=consumed.get("result",{}).get("usage")
            if usage is not None:result["charged_inner_cpu_seconds"]=max(0.,float(usage["cpu_seconds"]))
            if code==0 and consumed["status"]=="completed":result.update(status="completed",reference_ready=bool(consumed["result"]["reference_ready"]))
        elif code==0:raise ValueError("successful v2 worker has no consumer record")
        return result
    base.write(progress_path,progress)
    handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    def interrupted(signum,frame):raise KeyboardInterrupt(f"v2 campaign interrupted by signal{signum}")
    for sig in handlers:signal.signal(sig,interrupted)
    try:
        status=old.schedule(protocol["jobs"],launch,complete,event,deadline,addendum["initial_charged_cpu_seconds"],stopper=stop_direct)
        validate(path);progress.update(status=status,active_job_ids=[],ready_reference_job_ids=[j for j,v in progress["jobs"].items() if v["reference_ready"]])
        base.write(progress_path,progress);return progress
    except BaseException as error:
        progress.update(status="budget_exhausted" if isinstance(error,TimeoutError) else "failed",error_type=type(error).__name__,error=str(error),active_job_ids=[])
        base.write(progress_path,progress);raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser(description="Cache-complete direct-worker recovery under original Heston study budgets.")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare")
    for name in ("previous-recovery","previous-campaign","preflight","evidence-root","payload-root","logs-root","progress","out"):p.add_argument("--"+name,type=Path,required=True)
    p=sub.add_parser("run");p.add_argument("--recovery",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="prepare":
        value=prepare(args.previous_recovery,args.previous_campaign,args.preflight,args.evidence_root,args.payload_root,args.logs_root,args.progress,args.out)
        print(json.dumps({"addendum":str(args.out),"sha256":base.digest(args.out),"remaining_original_wall_seconds":value["remaining_seconds_at_prepare"]}))
    else:
        value=base.read(args.recovery);remaining=(datetime.fromisoformat(value["deadline_utc"])-datetime.now(timezone.utc)).total_seconds()
        if remaining<=0:raise TimeoutError("original campaign deadline passed")
        handler=signal.getsignal(signal.SIGALRM)
        def exhausted(signum,frame):raise TimeoutError("original campaign wall deadline exhausted")
        signal.signal(signal.SIGALRM,exhausted);signal.setitimer(signal.ITIMER_REAL,remaining)
        try:
            result=run(args.recovery);print(json.dumps({"status":result["status"],"ready_reference_job_ids":result["ready_reference_job_ids"]}))
        finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)


if __name__=="__main__":main()
