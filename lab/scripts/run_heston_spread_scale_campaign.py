PROCESS_STARTED = __import__("time").monotonic()
# The clock starts before all following imports and controller preparation.
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


MAX_CONCURRENT = 2
CAMPAIGN_WALL_SECONDS = 3900
MAX_INNER_CPU_BUDGET_SECONDS = 6300
FROZEN_CONSUMER_SHA256 = "191b0482fa73ec82f5e88fa3d8022c6fc502e5f9120a9b2dc37d25ceebed9eb3"
FROZEN_PRODUCER_SHA256 = "69b6267580b91c49b691a867326f8f64ce332dad445204e1fe00166ff6d2f575"
STUDY_PROTOCOL_SHA256 = "74ac585500dd0758a49113d1201c5a0119ffbc6970573c33ff61b12675639102"
REFERENCE_PROTOCOL_SHA256 = "a922b7bebb6dce2bef97d9ca539c59e18df54f7053ff57cb4947fd779cd912c2"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def checked_inputs(study_path, reference_path):
    from scripts import run_heston_spread_scale_reference as consumer
    if (digest(study_path) != STUDY_PROTOCOL_SHA256 or digest(reference_path) != REFERENCE_PROTOCOL_SHA256
            or digest(consumer.__file__) != FROZEN_CONSUMER_SHA256
            or digest(consumer.producer.__file__) != FROZEN_PRODUCER_SHA256):
        raise ValueError("frozen study/reference/source identity differs")
    study, reference = read(study_path), read(reference_path)
    limits = study["reference_protocol"]
    if (limits["sha256"] != digest(reference_path) or Path(limits["path"]).resolve() != Path(reference_path).resolve()
            or limits["new_targets"] != 7 or limits["max_concurrent_jobs"] != MAX_CONCURRENT
            or limits["whole_campaign_wall_seconds"] != CAMPAIGN_WALL_SECONDS
            or limits["max_sum_inner_cpu_seconds"] != MAX_INNER_CPU_BUDGET_SECONDS
            or len(reference["jobs"]) != 7 or consumer.INNER_BUDGET_SECONDS*len(reference["jobs"]) != MAX_INNER_CPU_BUDGET_SECONDS
            or consumer.OUTER_WALL_SECONDS != 950):
        raise ValueError("reference matrix or campaign budgets differ")
    for job in reference["jobs"]:
        consumer.validate_protocol(reference, job["id"])
    return reference, consumer


def prepare(study_path, reference_path, logs_root, progress_path, out):
    study_path, reference_path, logs_root, progress_path, out = map(Path, (study_path,reference_path,logs_root,progress_path,out))
    reference, consumer = checked_inputs(study_path, reference_path)
    if any(path.exists() for path in (logs_root,progress_path,progress_path.with_suffix(".claim.json"),out)):
        raise FileExistsError("campaign/log/progress identities must be new")
    for job in reference["jobs"]:
        if any(path.exists() for path in (Path(job["evidence_out"]),Path(job["payload_out"]),
                Path(reference["evidence_root"])/(job["id"]+".consumer.json"))):
            raise FileExistsError("a predeclared job already has an output or claim; no retries")
    declaration = {"format":"heston_spread_scale_campaign_protocol_v1",
        "study_protocol":{"path":str(study_path.resolve()),"sha256":digest(study_path)},
        "reference_protocol":{"path":str(reference_path.resolve()),"sha256":digest(reference_path)},
        "sources":{"controller":{"path":str(Path(__file__).resolve()),"sha256":digest(__file__)},
                   "consumer":{"path":str(Path(consumer.__file__).resolve()),"sha256":digest(consumer.__file__)},
                   "producer":{"path":str(Path(consumer.producer.__file__).resolve()),"sha256":digest(consumer.producer.__file__)}},
        "job_ids":[job["id"] for job in reference["jobs"]],"maximum_concurrent_jobs":MAX_CONCURRENT,
        "whole_campaign_wall_seconds":CAMPAIGN_WALL_SECONDS,"maximum_sum_inner_cpu_budget_seconds":MAX_INNER_CPU_BUDGET_SECONDS,
        "inner_budget_per_job_seconds":consumer.INNER_BUDGET_SECONDS,"outer_budget_per_job_seconds":consumer.OUTER_WALL_SECONDS,
        "logs_root":str(logs_root.resolve()),"progress_path":str(progress_path.resolve()),
        "clock_scope":"Earliest executable controller statement, before imports; includes controller validation, child startup and cleanup. Python interpreter startup before that statement is not measured.",
        "cpu_scope":"At most7 inner900-second CPU allocations. Per-process startup/controller CPU outside the inner budgets is not represented as part of that6300-second sum.",
        "process_groups":"Every consumer starts a new session; its reference worker inherits that process group. Whole-cap/interruption shutdown targets all such groups.",
        "automatic_retry":False,"scientific_acceptance":False}
    out.parent.mkdir(parents=True,exist_ok=True)
    write(out,declaration)
    return declaration


def validate_declaration(path):
    declaration = read(path)
    if (declaration["format"] != "heston_spread_scale_campaign_protocol_v1"
            or declaration["maximum_concurrent_jobs"] != MAX_CONCURRENT
            or declaration["whole_campaign_wall_seconds"] != CAMPAIGN_WALL_SECONDS
            or declaration["maximum_sum_inner_cpu_budget_seconds"] != MAX_INNER_CPU_BUDGET_SECONDS
            or declaration["sources"]["controller"]["sha256"] != digest(__file__)):
        raise ValueError("campaign controller declaration differs")
    for value in (*declaration["sources"].values(),declaration["study_protocol"],declaration["reference_protocol"]):
        if digest(value["path"]) != value["sha256"]:
            raise ValueError("campaign source/protocol changed")
    reference, consumer = checked_inputs(declaration["study_protocol"]["path"], declaration["reference_protocol"]["path"])
    if declaration["job_ids"] != [job["id"] for job in reference["jobs"]]:
        raise ValueError("campaign job order differs")
    return declaration,reference,consumer


def stop_groups(active, deadline, *, clock=time.monotonic, pause=time.sleep, killpg=os.killpg):
    """Stop descendant groups, including consumers whose direct child exited."""
    for entry in active.values():
        try:
            killpg(entry["process"].pid,signal.SIGTERM)
        except ProcessLookupError:
            pass
    grace_end = min(deadline,clock()+1.)
    while clock() < grace_end and any(entry["process"].poll() is None for entry in active.values()):
        pause(min(.05,max(0.,grace_end-clock())))
    for entry in active.values():
        try:
            killpg(entry["process"].pid,signal.SIGKILL)
        except ProcessLookupError:
            pass
        entry["process"].wait()


def schedule(jobs, launch, complete, event, deadline, *, clock=time.monotonic, pause=time.sleep, stopper=stop_groups):
    """Bounded FIFO scheduling; failure of one scientific run does not retry it."""
    pending,active,finished = list(jobs),{},{}
    try:
        while pending or active:
            if clock() >= deadline:
                raise TimeoutError("whole reference campaign wall limit exhausted")
            while pending and len(active) < MAX_CONCURRENT:
                if clock() >= deadline:
                    raise TimeoutError("whole reference campaign wall limit exhausted before launch")
                job=pending.pop(0)
                entry=launch(job)
                active[job["id"]]={"job":job,**entry}
                event("started",job,{"pid":entry["process"].pid,"active_jobs":len(active)})
            for job_id,entry in list(active.items()):
                code=entry["process"].poll()
                if code is None:
                    continue
                result=complete(entry["job"],entry,code)
                finished[job_id]=result
                active.pop(job_id)
                event("finished",entry["job"],result)
            if active:
                event("poll",None,{"active_job_ids":list(active),"pending_job_ids":[job["id"] for job in pending]})
                pause(min(.2,max(0.,deadline-clock())))
        return finished
    except BaseException as error:
        stopper(active,deadline,clock=clock,pause=pause)
        for entry in active.values():
            event("stopped",entry["job"],{"reason":type(error).__name__,"reference_ready":False})
        for job in pending:
            event("not_started",job,{"reason":type(error).__name__,"reference_ready":False})
        raise


def launch_consumer(job,consumer_path,reference_path,logs_root):
    log_path=Path(logs_root)/(job["id"]+".log")
    env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
    command=[sys.executable,str(consumer_path),"run","--protocol",str(reference_path),"--job",job["id"]]
    with log_path.open("x") as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    return {"process":process,"log_path":str(log_path.resolve()),"command":command}


def run(path):
    if time.monotonic()-PROCESS_STARTED>=CAMPAIGN_WALL_SECONDS:
        raise TimeoutError("controller startup exhausted the campaign wall limit")
    declaration,reference,consumer=validate_declaration(path)
    progress_path=Path(declaration["progress_path"])
    logs_root=Path(declaration["logs_root"])
    claim_path=progress_path.with_suffix(".claim.json")
    if progress_path.exists() or logs_root.exists():
        raise FileExistsError("campaign progress/log roots already exist")
    for job in reference["jobs"]:
        if any(p.exists() for p in (Path(job["evidence_out"]),Path(job["payload_out"]),
                Path(reference["evidence_root"])/(job["id"]+".consumer.json"))):
            raise FileExistsError("a campaign job was previously claimed; no retry")
    progress_path.parent.mkdir(parents=True,exist_ok=True)
    with claim_path.open("x") as stream:
        json.dump({"protocol_path":str(Path(path).resolve()),"protocol_sha256":digest(path),"controller_sha256":digest(__file__),
                   "started_utc":datetime.now(timezone.utc).isoformat()},stream,indent=2)
        stream.write("\n")
    logs_root.mkdir(parents=True,exist_ok=False)
    progress={"format":"heston_spread_scale_campaign_v1","status":"running","campaign_protocol_path":str(Path(path).resolve()),
        "campaign_protocol_sha256":digest(path),"sources":declaration["sources"],"started_utc":datetime.now(timezone.utc).isoformat(),
        "whole_campaign_wall_seconds":CAMPAIGN_WALL_SECONDS,"maximum_concurrent_jobs":MAX_CONCURRENT,
        "maximum_sum_inner_cpu_budget_seconds":MAX_INNER_CPU_BUDGET_SECONDS,"assigned_inner_cpu_budget_seconds":0,
        "reserved_arrays_opened":False,"automatic_retry":False,"scientific_acceptance":False,"events":[],
        "jobs":{job["id"]:{"status":"pending","reference_ready":False,
                 "reference_progress_path":str(Path(job["evidence_out"])/"progress.json"),
                 "consumer_result_path":str(Path(reference["evidence_root"])/(job["id"]+".consumer.json"))}
                for job in reference["jobs"]}}
    last_write=0.
    def event(kind,job,details):
        nonlocal last_write
        seconds=time.monotonic()-PROCESS_STARTED
        progress["seconds"]=seconds
        if kind=="poll":
            progress["active_job_ids"]=details["active_job_ids"]
            if seconds-last_write<2.:
                return
        else:
            progress["events"].append({"kind":kind,"job_id":job["id"],"seconds":seconds,**details})
            entry=progress["jobs"][job["id"]]
            if kind=="started":
                entry.update(status="running",pid=details["pid"],started_seconds=seconds,
                    log_path=str(logs_root/(job["id"]+".log")))
                progress["assigned_inner_cpu_budget_seconds"]+=consumer.INNER_BUDGET_SECONDS
                if progress["assigned_inner_cpu_budget_seconds"]>MAX_INNER_CPU_BUDGET_SECONDS:
                    raise RuntimeError("campaign assigned too much inner CPU budget")
            else:
                entry.update(**details,completed_seconds=seconds)
                if kind in ("stopped","not_started"):
                    entry["status"]=kind
        write(progress_path,progress)
        last_write=seconds
    def launch(job):
        return launch_consumer(job,consumer.__file__,declaration["reference_protocol"]["path"],logs_root)
    def complete(job,entry,code):
        # A failed consumer can exit before its worker. Reap any descendants
        # before releasing the scheduler slot or forgetting this process group.
        for sig in (signal.SIGTERM,signal.SIGKILL):
            try:
                os.killpg(entry["process"].pid,sig)
            except ProcessLookupError:
                pass
        result={"status":"inconclusive","exit_code":code,"reference_ready":False,
            "log_path":entry["log_path"],"log_sha256":digest(entry["log_path"])}
        consumer_path=Path(reference["evidence_root"])/(job["id"]+".consumer.json")
        if not consumer_path.exists():
            if code==0:
                raise ValueError("successful child has no source-bound consumer record")
            return {**result,"reason":"child failed before its consumer record was written"}
        consumed=read(consumer_path)
        if (consumed["protocol_sha256"]!=declaration["reference_protocol"]["sha256"] or consumed["job"]!=job
                or consumed["source_sha256"]!=reference["source_sha256"]):
            raise ValueError("child result source/target/protocol binding differs")
        result.update(consumer_result_path=str(consumer_path),consumer_result_sha256=digest(consumer_path),consumer=consumed)
        if code==0 and consumed["status"]=="completed":
            result.update(status="completed",reference_ready=bool(consumed["result"]["reference_ready"]))
        return result
    write(progress_path,progress)
    old_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f"campaign interrupted by signal{signum}")
    for sig in old_handlers:
        signal.signal(sig,interrupted)
    try:
        schedule(reference["jobs"],launch,complete,event,PROCESS_STARTED+CAMPAIGN_WALL_SECONDS)
        validate_declaration(path)
        if time.monotonic()-PROCESS_STARTED>=CAMPAIGN_WALL_SECONDS:
            raise TimeoutError("campaign final checks exhausted the whole wall limit")
        progress.update(status="completed",seconds=time.monotonic()-PROCESS_STARTED,active_job_ids=[],
            ready_reference_job_ids=[job_id for job_id,value in progress["jobs"].items() if value["reference_ready"]])
        write(progress_path,progress)
        return progress
    except BaseException as error:
        progress.update(status="budget_exhausted" if isinstance(error,TimeoutError) else "interrupted" if isinstance(error,KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,error=str(error),seconds=time.monotonic()-PROCESS_STARTED,active_job_ids=[])
        write(progress_path,progress)
        raise
    finally:
        for sig,handler in old_handlers.items():
            signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser(description="Run the frozen seven-job Heston scale-reference campaign, at most two processes at once.")
    commands=parser.add_subparsers(dest="command",required=True)
    prepare_parser=commands.add_parser("prepare")
    for name in ("study-protocol","reference-protocol","logs-root","progress","out"):
        prepare_parser.add_argument("--"+name,type=Path,required=True)
    run_parser=commands.add_parser("run")
    run_parser.add_argument("--protocol",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="prepare":
        result=prepare(args.study_protocol,args.reference_protocol,args.logs_root,args.progress,args.out)
        print(json.dumps({"protocol":str(args.out),"sha256":digest(args.out),"jobs":result["job_ids"]}))
    else:
        # Alarm also covers validation/imports and final checks outside the
        # polling loop. Its remaining allowance uses the first-statement clock.
        old_handler=signal.getsignal(signal.SIGALRM)
        old_timer=signal.getitimer(signal.ITIMER_REAL)
        def exhausted(signum,frame):
            raise TimeoutError("whole reference campaign wall alarm exhausted")
        signal.signal(signal.SIGALRM,exhausted)
        signal.setitimer(signal.ITIMER_REAL,max(.001,CAMPAIGN_WALL_SECONDS-(time.monotonic()-PROCESS_STARTED)))
        try:
            result=run(args.protocol)
            print(json.dumps({"status":result["status"],"seconds":result["seconds"],"ready_reference_job_ids":result["ready_reference_job_ids"]}))
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            signal.signal(signal.SIGALRM,old_handler)
            signal.setitimer(signal.ITIMER_REAL,*old_timer)


if __name__=="__main__":
    main()
