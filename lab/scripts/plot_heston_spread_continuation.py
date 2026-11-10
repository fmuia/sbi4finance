"""Package the completed zero-dropout 10k->50k development continuation.

Reads metadata/evaluations and hashes model/state files as bytes. It loads no
neural model, bank member, posterior draws or reserved arrays. Old evidence and
the frozen training/reference producers are never modified.
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.plot_heston_spread_development import case_summary, digest, read, write


def checkpoint_identity(directory, run, updates):
    record = read(directory/"checkpoint.json")
    if (record["actual_optimizer_updates"] != updates
            or record["task_fingerprint"] != run["task_fingerprint"]
            or record["task"] != run["task"] or record["architecture"] != run["architecture"]
            or record["source_sha256"] != run["source_sha256"]
            or digest(directory/"model.keras") != record["model_sha256"]
            or digest(directory/"state.npz") != record["state_sha256"]):
        raise ValueError("checkpoint task/source/model/state binding differs")
    return record


def evaluation_identity(evaluation, directory, checkpoint, *, require_explicit_hashes):
    declared = Path(evaluation["checkpoint"])
    if not declared.is_absolute():
        declared = Path(__file__).resolve().parents[1]/declared
    if (declared.resolve() != directory.resolve() or evaluation["task"] != checkpoint["task"]["target"]
            or evaluation["scientific_acceptance"] is not False
            or evaluation["reserved_test_arrays_opened"] is not False):
        raise ValueError("evaluation task/checkpoint/scope differs")
    if require_explicit_hashes and (
            evaluation["checkpoint_record_sha256"] != digest(directory/"checkpoint.json")
            or evaluation["model_sha256"] != checkpoint["model_sha256"]
            or evaluation["state_sha256"] != checkpoint["state_sha256"]
            or evaluation["actual_optimizer_updates"] != checkpoint["actual_optimizer_updates"]
            or evaluation["checkpoint_source_sha256"] != checkpoint["source_sha256"]):
        raise ValueError("evaluation source/model/state hashes differ")


def package(parent_run, continuation_run, evaluation_10k, evaluation_25k, evaluation_50k,
            boundary_reference, protocol, out):
    parent_run, continuation_run, boundary_reference, protocol, out = map(Path,
        (parent_run, continuation_run, boundary_reference, protocol, out))
    parent, run = read(parent_run/"run.json"), read(continuation_run/"run.json")
    training, resume = read(continuation_run/"training.json"), read(continuation_run/"resume-protocol.json")
    declared = read(protocol)
    lab = Path(__file__).resolve().parents[1]
    for name, filename in (("trainer", "train_heston_spread_baseline.py"), ("continuation", "continue_heston_spread_baseline.py")):
        if run["source_sha256"][name] != digest(lab/"scripts"/filename):
            raise ValueError("frozen producer source differs")
    shared = ("task_fingerprint", "task", "names", "bounds", "design_sha256", "bank_record_sha256",
              "bank_payload_sha256", "train_rows", "schedule_updates", "seed", "batch_size", "architecture",
              "optimizer_schedule", "numeric_configuration", "augmentation", "batch_plan", "probe_rows")
    if (any(parent[key] != run[key] for key in shared)
            or run["architecture"]["subnet_dropout"] != 0. or run["task"]["target"]["fixed_c"] != 1.
            or run["planned_updates"] != 50000 or run["checkpoint_updates"] != [25000,50000]
            or resume["start_updates"] != 10000 or resume["target_updates"] != 50000
            or resume["schedule_updates"] != 10000 or not resume["absolute_data_and_noise_steps"]
            or not resume["restoration"]["all_array_hashes_exact"] or resume["reserved_arrays_opened"]
            or digest(continuation_run/"resume-protocol.json") != run["resume_protocol_sha256"]
            or declared["start_absolute_update"] != 10000 or declared["final_absolute_update"] != 50000
            or training["status"] != "complete" or training["completed_updates"] != 50000
            or training["continued_updates"] != 40000 or not training["all_parent_array_hashes_restored"]
            or not training["same_seed_reload_exact"] or training["reserved_arrays_opened"]):
        raise ValueError("continuation changes task/data/schedule or lacks its completed restoration proof")
    if not np.allclose(list(resume["learning_rates"].values()), 5e-5, rtol=1e-7, atol=0):
        raise ValueError("continuation did not preserve the original schedule floor")
    parent_checkpoint = parent_run/"checkpoints/update-010000"
    if (resume["parent_checkpoint_sha256"] != digest(parent_checkpoint/"checkpoint.json")
            or resume["continuation_source_sha256"] != run["source_sha256"]["continuation"]):
        raise ValueError("parent/source resume binding differs")
    points = ((10000,parent_run,Path(evaluation_10k),parent),
              (25000,continuation_run,Path(evaluation_25k),run),
              (50000,continuation_run,Path(evaluation_50k),run))
    names, payloads, evaluations, source_files = run["names"], {}, {}, {}
    for updates, directory, path, identity in points:
        checkpoint_dir = directory/"checkpoints"/f"update-{updates:06d}"
        checkpoint = checkpoint_identity(checkpoint_dir,identity,updates)
        evaluation = read(path)
        evaluation_identity(evaluation,checkpoint_dir,checkpoint,require_explicit_hashes=updates>10000)
        if updates == 10000 and (resume["parent_model_sha256"] != checkpoint["model_sha256"]
                                or resume["parent_state_sha256"] != checkpoint["state_sha256"]):
            raise ValueError("restored parent model/state differs")
        payloads[str(updates)] = {key:checkpoint[key] for key in ("actual_optimizer_updates", "model_sha256",
            "state_sha256", "task_fingerprint", "architecture", "source_sha256", "probe_metrics", "resume_supported")}
        payloads[str(updates)].update(checkpoint_record_sha256=digest(checkpoint_dir/"checkpoint.json"),
            payload_bytes={name:(checkpoint_dir/name).stat().st_size for name in ("model.keras","state.npz")})
        evaluations[str(updates)] = {"conditional_status":evaluation["conditional_status"],
            "cases":case_summary(evaluation,names), "population":evaluation["population"],
            "explicit_checkpoint_hashes_in_evaluation":updates>10000}
        source_files[f"evaluation_{updates}"] = path
        source_files[f"checkpoint_record_{updates}"] = checkpoint_dir/"checkpoint.json"
    raw_evaluations = [read(path) for path in (evaluation_10k,evaluation_25k,evaluation_50k)]
    first = raw_evaluations[0]
    if any(value["panel_sha256"] != first["panel_sha256"] or value["seed"] != first["seed"]
           or value["population"]["rows"] != first["population"]["rows"]
           or value["population"]["draws_per_case"] != first["population"]["draws_per_case"] for value in raw_evaluations):
        raise ValueError("endpoint evaluation observations/population protocol differs")
    boundary = read(boundary_reference)
    if boundary["status"] != "failed" or boundary["panel_sha256"] != first["panel_sha256"]:
        raise ValueError("boundary reference failure/status binding changed")
    out.mkdir(parents=True,exist_ok=True)
    compact = {"format":"heston_spread_compact_continuation_run_v1", "target":run["task"]["target"],
        "original_run_sha256":digest(continuation_run/"run.json"), "original_run_path":str(continuation_run.resolve()),
        "completed_optimizer_updates":training["completed_updates"],
        **{key:run[key] for key in ("task_fingerprint","names","bounds","design_sha256","bank_record_sha256",
            "bank_payload_sha256","train_rows","planned_updates","schedule_updates","seed","batch_size",
            "checkpoint_updates","wall_seconds","architecture","source_sha256","environment_sha256","versions",
            "devices","optimizer_schedule","numeric_configuration","augmentation","batch_plan","probe_rows",
            "resume_protocol_sha256","parent_actual_optimizer_updates")},
        "scientific_acceptance":False,"reserved_arrays_opened":False,
        "scope":"compact completed continuation record; full checkpoint variable lists and payload arrays remain outside Git"}
    write(out/"run.json",compact)
    copied = {}
    for name in ("training.json","learning-curve.json","resume-protocol.json","initial-probes.json","first-batch.json"):
        temporary = out/(name+".tmp")
        temporary.write_bytes((continuation_run/name).read_bytes())
        temporary.replace(out/name)
        copied[name] = {"source_sha256":digest(continuation_run/name),"copied_sha256":digest(out/name)}
    write(out/"checkpoint-payloads.json",{"format":"heston_spread_continuation_payload_summary_v1",
        "checkpoints":payloads,"scope":"full models/state arrays remain ignored; no release asset accepted"})
    comparison = {"format":"heston_spread_continuation_comparison_v1","status":"no_accepted_endpoint",
        "scientific_acceptance":False,"parameter_names":names,"endpoints":evaluations,
        "boundary_reference":{"status":boundary["status"],"error_type":boundary.get("error_type"),
            "error":boundary.get("error"),"cpu_and_wall_limit_seconds":boundary["settings"]["cpu_and_wall_seconds"],
            "completed_attempts":len(boundary["attempts"]),"source_sha256":digest(boundary_reference),
            "interpretation":"reference failure leaves boundary neural comparison inconclusive, never accepted"},
        "limits":["Same512 development cases and seeds; reserved final pool remains unopened.",
            "10k evaluation predates explicit model/state hash fields; its declared immutable checkpoint path/payload is verified here. 25k/50k evaluations explicitly bind record/model/state/source hashes.",
            "Additional optimization modestly narrows broad-case intervals, but conditional mean errors and population calibration are not uniformly improved.",
            "Neither the lowest loss nor the best pooled rank count chooses an accepted checkpoint. Market-model adequacy remains separate."]}
    write(out/"comparison.json",comparison)
    old_curve, curve = read(parent_run/"learning-curve.json"),read(continuation_run/"learning-curve.json")
    all_curve = old_curve+curve
    endpoint_colors = {10000:"#8099a8",25000:"#c9782d",50000:"#176d9c"}
    fig, axes = plt.subplots(1,2,figsize=(11.2,4.3),layout="constrained")
    axes[0].plot([p["updates"]/1000 for p in all_curve],[p["validation_probe_loss"] for p in all_curve],color="#176d9c",lw=1.6)
    for updates in (10000,25000,50000):
        p = payloads[str(updates)]["probe_metrics"]
        axes[0].scatter(updates/1000,p["validation_probe_loss"],s=35,zorder=3,color=endpoint_colors[updates])
    axes[0].axvline(10,color=".5",ls=":",lw=1)
    axes[0].set(xlabel="Absolute optimizer updates (thousands)",ylabel="Fixed validation NLL (transformed target)",
                title="Same zero-dropout trajectory; original LR floor")
    axes[0].grid(alpha=.2)
    for offset,updates,color in ((-.2,10000,"#8099a8"),(0.,25000,"#c9782d"),(.2,50000,"#176d9c")):
        params = evaluations[str(updates)]["cases"]["synthetic-broad"]["parameters"]
        values = np.array([params[name]["width_ratio_90"] for name in names])
        intervals = np.array([params[name]["width_ratio_interval"] for name in names])
        axes[1].errorbar(np.arange(5)+offset,values,yerr=np.stack([values-intervals[:,0],intervals[:,1]-values]),
                         fmt="o",capsize=3,color=color,label=f"{updates//1000}k")
    axes[1].axhspan(.75,1.25,color="#3c9b55",alpha=.16,label="Declared width band")
    axes[1].axhline(1.,color="#426548",lw=.8)
    axes[1].set(xticks=np.arange(5),xticklabels=[r"$v_0$",r"$\kappa$",r"$\theta$",r"$\xi$",r"$\rho$"],
                ylabel="Neural / reference central-90% width",title="Broad case: narrower intervals still fail")
    axes[1].grid(axis="y",alpha=.2)
    axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle("Fixed-c Heston continuation: no accepted endpoint",fontsize=12)
    fig.savefig(out/"continuation-comparison.png",dpi=180)
    plt.close(fig)
    rows = ["| Parameter | 10k mean / width | 25k mean / width | 50k mean / width |",
            "| --- | ---: | ---: | ---: |"]
    for name in names:
        values = [evaluations[str(u)]["cases"]["synthetic-broad"]["parameters"][name] for u in (10000,25000,50000)]
        rows.append("| "+name+" | "+" | ".join(f"{v['signed_standardized_mean_difference']:+.2f} / {v['width_ratio_90']:.2f}" for v in values)+" |")
    metric_rows = ["| Update | Fixed validation NLL | 90% coverage by parameter | Rank checks passing |",
                   "| --- | ---: | --- | ---: |"]
    for updates in (10000,25000,50000):
        population = evaluations[str(updates)]["population"]
        coverage = ", ".join(f"{v:.3f}" for v in population["coverage"]["0.9"]["coverage"])
        metric_rows.append(f"| {updates//1000}k | {payloads[str(updates)]['probe_metrics']['validation_probe_loss']:.4f} | {coverage} | {sum(population['rank_dkw']['passed'])}/5 |")
    text = "\n".join(["# Zero-dropout Heston continuation to 50k", "",
        "**Completed bounded development experiment; no accepted endpoint.** The 10k zero-dropout model continued to the predeclared 25k/50k checkpoints with unchanged likelihood, prior, design, spreads, 20k-row bank prefix, architecture and absolute row/noise streams. Adam moments, standardization, metrics and saved RNG arrays were restored under exact array hashes. The original 10k cosine horizon remained at its approximately 5e-5 floor.", "",
        f"The continuation committed 40,000 updates in {training['seconds']:.2f} worker seconds, within its 900-second cap. It added {training['continued_row_presentations']:,} row presentations, for {training['total_row_presentations']:,} in total. Both new checkpoints preserve exact seeded inference reload. Full payloads remain outside Git; [resume proof](resume-protocol.json), [run identity](run.json) and [payload hashes](checkpoint-payloads.json) bind the compact evidence.", "",
        "The broad-case table reports signed mean error in reference SD, followed by the neural/reference central-90% width ratio. Existing MC-aware margins remain .3 reference SD and widths [.75,1.25]; the source comparisons include Monte Carlo uncertainty. Narrower intervals do not ensure better centering.", "", *rows, "",
        "Additional optimization narrows broad widths modestly, but every endpoint still fails its conditional comparison. Some mean errors improve and others worsen. The completed market-like and actual-market comparisons also fail at 25k and 50k; the market query's inferential error remains distinct from Heston/observation-model adequacy.", "",
        "Development coverage uses parameter order v0, kappa, theta, xi, rho on the same 512 held-out development rows and 512 posterior draws per row. These repeated checks are not fresh final acceptance evidence. All rank/coverage uncertainty remains in [comparison.json](comparison.json).", "", *metric_rows, "",
        "![Fixed endpoints and conditional widths](continuation-comparison.png)", "",
        "The boundary reference failed its 2700-second wall/CPU allocation after an unconverged first attempt and a bounded escalation. Its neural comparisons remain inconclusive. This reference failure is retained, not replaced by a successful case or counted as passing. Three other reference cases are available.", "",
        "Neither the slightly lower 50k loss nor the better 25k pooled rank count selects an accepted model. Further diagnostics or optimizer/data/capacity interventions require their own frozen comparison; these records authorize no additional fit, revised-grid labels, inferred-scale training or Bergomi production. The reserved test arrays remain unopened.", "",
        "Reproduce this package with `lab/scripts/plot_heston_spread_continuation.py` and the source paths/hashes in [manifest.json](manifest.json). The producer reads metadata and hashes payload files as bytes; it loads no model/state arrays, bank members or posterior draws. The 10k evaluation has the weaker historical path binding described in comparison.json; the new 25k/50k records explicitly bind checkpoint/source/model/state hashes.", ""])
    (out/"README.md").write_text(text)
    source_files.update({"producer":Path(__file__).resolve(),"shared_summary_producer":Path(__file__).with_name("plot_heston_spread_development.py"),
        "parent_run":parent_run/"run.json","parent_curve":parent_run/"learning-curve.json",
        "continuation_run":continuation_run/"run.json","declared_protocol":protocol,"boundary_reference":boundary_reference})
    for name in copied:
        source_files["original_"+name] = continuation_run/name
    manifest = {"format":"heston_spread_continuation_package_manifest_v1",
        "source_files":{name:{"path":str(path),"sha256":digest(path)} for name,path in source_files.items()},
        "copied_records":copied,"generated_files_sha256":{path.name:digest(path) for path in out.iterdir() if path.is_file() and path.name != "manifest.json"},
        "scientific_acceptance":False,"reserved_arrays_opened":False}
    write(out/"manifest.json",manifest)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-run","continuation-run","evaluation-10k","evaluation-25k","evaluation-50k","boundary-reference","protocol","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    package(**vars(args))
    print(json.dumps({"out":str(args.out),"status":"no_accepted_endpoint"}))


if __name__ == "__main__":
    main()
