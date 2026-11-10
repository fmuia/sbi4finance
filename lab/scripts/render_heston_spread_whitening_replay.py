"""Offline report for the exploratory exact-prefix ZCA checkpoint replay.

Only saved records, recovery quantiles and existing compact MCMC summaries are
read. Rendering never queries a network, generates prices or samples a chain.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import render_heston_spread_whitening as original
from scripts import render_heston_spread_mcmc_recovery as selected

shared = original.shared
import matplotlib.pyplot as plt

EPOCHS = (50, 75, 83, 100, 200)
COLORS = {50: "#b57920", 75: "#6f7c37", 83: "#246da0", 100: "#7952a3", 200: "#b44b37"}
NAMES, LABELS = shared.NAMES, shared.LABELS
esc, fmt, table, section, chart = shared.esc, shared.fmt, shared.table, shared.section, shared.chart


def read(path): return json.loads(Path(path).read_text())
def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": shared.digest(path)}


def collect(protocol_path):
    """Bind the completed replay and unchanged original final evaluation."""
    protocol_path = Path(protocol_path).resolve()
    protocol = read(protocol_path)
    if protocol.get("format") != "heston_spread_whitening_checkpoint_replay_study_v1":
        raise ValueError("checkpoint replay protocol required")
    base, payload = protocol_path.parent, Path(protocol["output"])
    training = read(payload/"training.json")
    if (training.get("status") != "complete" or training.get("completed_updates") != 4000
            or training.get("completed_bank_passes") != 100 or training.get("presentations") != 1000000
            or training.get("all_original_initial_state_hashes_exact") is not True
            or training.get("original_learning_curve_prefix_exact") is not True
            or training.get("original_curve_points_compared") != 100
            or training.get("schedule_updates") != 8000 or training.get("same_seed_reload_exact") is not True):
        raise ValueError("the original initialization, loss prefix, schedule or replay completion differs")
    exports = {}
    compact = base/"training"; compact.mkdir(parents=True, exist_ok=True)
    for name in ("run.json", "training.json", "learning-curve.json", "progress.json", "initial-probes.json"):
        source, target = payload/name, compact/name
        if target.exists() and shared.digest(source) != shared.digest(target):
            raise ValueError("existing compact replay export differs")
        if not target.exists(): shutil.copyfile(source, target)
        exports[name] = {"source": binding(source), "copy": binding(target)}
    arms = []
    for spec in protocol["evaluations"]:
        step, epoch = spec["updates"], spec["epoch"]
        checkpoint = payload/"checkpoints"/f"update-{step:06d}"
        path = Path(spec["output"])/"evaluation.json"
        record, metadata = read(path), read(checkpoint/"checkpoint.json")
        if (epoch not in EPOCHS[:-1] or step != 40*epoch
                or metadata["actual_optimizer_updates"] != step or metadata["training_protocol"] != protocol["contract"]
                or record["status"] != "complete" or record["actual_optimizer_updates"] != step
                or record["replay_protocol"] != binding(protocol_path)
                or record["checkpoint_record_sha256"] != shared.digest(checkpoint/"checkpoint.json")):
            raise ValueError("replay endpoint or evaluation binding differs")
        for name, key in (("model.keras", "model_sha256"), ("state.npz", "state_sha256")):
            if record[key] != shared.digest(checkpoint/name): raise ValueError("replay payload differs")
        if record["preprocessing"] != binding(checkpoint/"preprocessing.json"):
            raise ValueError("replay sidecar differs")
        arms.append({"epoch": epoch, "updates": step, "evaluation": binding(path),
            "recovery": binding(path.parent/"recovery.npz")})
    final_path = Path(protocol["reused_final_evaluation"]["path"])
    if binding(final_path) != protocol["reused_final_evaluation"]:
        raise ValueError("preserved final evaluation differs")
    arms.append({"epoch": 200, "updates": 8000, "evaluation": binding(final_path),
        "recovery": binding(final_path.parent/"recovery.npz")})
    if tuple(a["epoch"] for a in arms) != EPOCHS: raise ValueError("all five epochs are required in order")
    original_directory = Path(protocol["contract"]["original"]["directory"])
    manifest = {"format": "heston_spread_whitening_checkpoint_replay_report_v1",
        "protocol": binding(protocol_path), "original_study": protocol["original_study"],
        "original_curve": binding(original_directory/"learning-curve.json"),
        "original_run": binding(original_directory/"run.json"), "training_exports": exports,
        "arms": arms, "producer": binding(__file__), "scientific_acceptance": False}
    path = base/"report-manifest.json"
    if path.exists() and read(path) != manifest: raise ValueError("existing replay report identity differs")
    path.write_text(json.dumps(manifest, indent=2)+"\n")
    return path


def load_data(manifest_path):
    manifest_path = Path(manifest_path).resolve(); manifest = read(manifest_path)
    if manifest["format"] != "heston_spread_whitening_checkpoint_replay_report_v1":
        raise ValueError("unsupported replay report")
    inputs = shared.Artifacts(manifest_path)
    inputs.sources.append({"label": "Checkpoint replay report manifest", **binding(manifest_path)})
    protocol = inputs.json(manifest["protocol"], "Frozen exploratory checkpoint replay")
    study = inputs.json(manifest["original_study"], "Preserved whitening study")
    original_run = inputs.json(manifest["original_run"], "Original ZCA initialization and recipe")
    original_curve = inputs.json(manifest["original_curve"], "Original 200-epoch learning curve")
    exports = manifest["training_exports"]
    run = inputs.json(exports["run.json"]["copy"], "Exact-prefix replay run")
    training = inputs.json(exports["training.json"]["copy"], "Exact-prefix verification and completion")
    curve = inputs.json(exports["learning-curve.json"]["copy"], "Replayed 100-epoch learning curve")
    if len(curve) != 100 or len(original_curve) != 200:
        raise ValueError("complete original and replayed curves required")
    if (run["task"] != original_run["task"] or run["splits"] != original_run["splits"]
            or run["preprocessing_sha256"] != original_run["preprocessing_sha256"]):
        raise ValueError("raw observations, targets, split or fixed preprocessing differ")
    arms = {}
    reference_hashes = {}
    for spec in manifest["arms"]:
        epoch = spec["epoch"]; e = inputs.json(spec["evaluation"], f"Epoch {epoch} posterior diagnostics")
        if (e["status"] != "complete" or e["actual_optimizer_updates"] != spec["updates"]
                or e["task"] != study["joint_task"] or e["study_protocol"] != manifest["original_study"]
                or e["reserved_arrays_opened"] is not False or e["new_reference_sampling"] is not False
                or e["preprocessing_summary"]["validation_used_for_fit"] is not False
                or e["preprocessing_summary"]["fit_rows"] != 10000):
            raise ValueError("replay evaluation task, data boundary or preprocessing differs")
        if epoch != 200 and e["replay_protocol"] != manifest["protocol"]:
            raise ValueError("replay evaluation protocol differs")
        recovery = shared.recovery_data(spec["recovery"], inputs, e, spec["updates"])
        p = e["population"]
        population_binding = (p["observations_sha256"], p["truth_sha256"])
        if arms and population_binding != next(iter(arms.values()))["population_binding"]:
            raise ValueError("checkpoint populations differ")
        for case, value in e["cases"].items():
            current = (value["observation_sha256"], json.dumps(value["comparison"]["reference_quantiles_05_50_95"]))
            if case in reference_hashes and reference_hashes[case] != current:
                raise ValueError("checkpoint conditional observations or references differ")
            reference_hashes[case] = current
        for source in e["source_sha256"].values(): selected.source_binding(inputs, source, f"Epoch {epoch} executed source")
        arms[epoch] = {"evaluation": e, "recovery": recovery, "population_binding": population_binding}
    mcmc_path = inputs.path(study["matched_mcmc_report"], "Existing selected joint-MCMC references")
    mcmc = selected.load_data(mcmc_path)
    indices = mcmc["selection"]["selection"]["zero_based_population_indices"]
    for epoch, arm in arms.items():
        truth, q = arm["recovery"]
        if (not np.array_equal(truth[indices], mcmc["truth"])
                or arm["population_binding"][0] != mcmc["selection"]["population_observations_sha256"]):
            raise ValueError("selected MCMC rows differ")
        mcmc["quantiles"][f"epoch{epoch}"] = q[:, indices, :]
    inputs.sources.extend(mcmc["inputs"].sources)
    for name, source in protocol["sources"].items(): selected.source_binding(inputs, source, "Frozen replay source: "+name)
    for name, source in protocol["contract"]["original"]["files"].items():
        selected.source_binding(inputs, source, "Preserved original: "+name)
    return {"inputs": inputs, "protocol": protocol, "study": study, "training": training,
        "run": run, "curve": curve, "original_curve": original_curve,
        "arms": arms, "selected": mcmc, "target": study["joint_task"]}


def save(fig, base, name, title, description):
    return selected.save_figure(fig, base/name, title, description)


def plots(data, base):
    figures = {}; arms = data["arms"]
    fig, axis = plt.subplots(figsize=(10.7, 4), layout="constrained")
    for field, style, label in (("train_probe_loss", ":", "Training"), ("validation_probe_loss", "-", "Validation")):
        axis.plot([p["completed_bank_passes"] for p in data["original_curve"]],
            [p[field] for p in data["original_curve"]], style, color="#405562", label=label)
    for epoch in EPOCHS:
        point = data["original_curve"][epoch-1]
        axis.scatter(epoch, point["validation_probe_loss"], color=COLORS[epoch], label=f"Epoch {epoch}", zorder=4)
    axis.set(xlabel="Pass over the same fixed 10,000 noisy rows", ylabel="Fixed-probe negative log density",
        title="Original trajectory; replay matches the first 100 epochs exactly")
    axis.legend(fontsize=8, ncol=4)
    figures["learning-curves"] = save(fig, base, "learning-curves", "Saved checkpoints on the unchanged trajectory",
        "The cosine schedule still spans 8000 updates. Saving earlier weights does not shorten the decay schedule.")
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.3), layout="constrained")
    grid = np.linspace(0, 1, 201)
    for epoch, arm in arms.items():
        population = arm["evaluation"]["population"]; ranks = np.asarray(population["fractional_ranks"])
        for j, axis in enumerate(axes.flat):
            axis.plot(grid, np.searchsorted(np.sort(ranks[:, j]), grid, side="right")/512-grid,
                color=COLORS[epoch], label=f"Epoch {epoch}")
            if epoch == 50:
                axis.axhspan(-population["rank_dkw"]["dkw_bound"], population["rank_dkw"]["dkw_bound"], color="#dbece1", zorder=0)
                axis.axhline(0, color="#7e8790", lw=.6)
            axis.set(title=LABELS[NAMES[j]], xlabel="Fractional posterior rank", ylabel="ECDF−uniform CDF", xlim=(0,1))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    figures["sbc-ecdf"] = save(fig, base, "sbc-ecdf", "All six marginal calibration screens",
        "The DKW band adjusts over six parameters within an epoch. These posthoc epoch comparisons do not provide independent acceptance.")
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.3), layout="constrained")
    for j, axis in enumerate(axes.flat):
        axis.axhline(.9, color="#405562", ls="--", label="Nominal 90%")
        axis.plot(EPOCHS, [arms[e]["evaluation"]["population"]["coverage"]["0.9"]["coverage"][j] for e in EPOCHS], "o-", color="#246da0")
        axis.set(title=LABELS[NAMES[j]], xlabel="Epoch", ylabel="Coverage of central 90% interval", ylim=(0,1))
    figures["coverage"] = save(fig, base, "coverage", "Point recovery and uncertainty are separate questions",
        "Coverage uses the identical 512 full-prior development observations at each checkpoint.")
    bounds = np.asarray(data["target"]["prior_bounds"]); chosen = np.arange(0,512,16)
    for epoch, arm in arms.items():
        truth, q = arm["recovery"]; fig, axes = plt.subplots(2, 3, figsize=(11.6, 7.4), layout="constrained")
        for j, axis in enumerate(axes.flat):
            lo, hi = bounds[j]; margin = .025*(hi-lo)
            axis.plot([lo,hi],[lo,hi],"--",color="#405562",lw=1)
            axis.scatter(truth[:,j],q[1,:,j],s=9,alpha=.45,color=COLORS[epoch],edgecolors="none")
            axis.errorbar(truth[chosen,j],q[1,chosen,j],yerr=np.stack([q[1,chosen,j]-q[0,chosen,j],q[2,chosen,j]-q[1,chosen,j]]),
                fmt="o",ms=2,color=COLORS[epoch],capsize=2,elinewidth=.65)
            axis.set(title=LABELS[NAMES[j]],xlabel="Generating value",ylabel="Posterior median",
                xlim=(lo-margin,hi+margin),ylim=(lo-margin,hi+margin),aspect="equal")
        name = f"recovery-epoch{epoch:03d}"
        figures[name] = save(fig, base, name, f"Epoch {epoch}: all six parameters",
            "512 posterior medians; central 90% intervals for the same 32 fixed-index observations.")
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 7.5), layout="constrained")
    selected_data = data["selected"]; x = selected_data["truth"][:,-1]
    for axis, epoch in zip(axes.flat, EPOCHS, strict=False):
        q = selected_data["quantiles"][f"epoch{epoch}"][:,:,-1]
        ready = selected_data["ready"]; reference = selected_data["quantiles"]["mcmc"][:,ready,-1]
        axis.plot([.5,2],[.5,2],"--",color="#405562",lw=1)
        axis.errorbar(x,q[1],yerr=np.stack([q[1]-q[0],q[2]-q[1]]),fmt="o",color=COLORS[epoch],capsize=3,label=f"Epoch {epoch}")
        axis.errorbar(x[ready],reference[1],yerr=np.stack([reference[1]-reference[0],reference[2]-reference[1]]),
            fmt="x",color="#405562",capsize=2,label="Existing ready MCMC")
        axis.set(title=f"Epoch {epoch}",xlabel="Generating c",ylabel="Posterior median c",xlim=(.47,2.03),ylim=(.47,2.03),aspect="equal")
        axis.legend(fontsize=7)
    axes.flat[-1].axis("off")
    figures["selected-c-recovery"] = save(fig, base, "selected-c-recovery", "Same eight selected observations and seven ready references",
        "All eight neural cases remain visible; the unready MCMC case is excluded from reference intervals and matched statistics.")
    fig, axes = plt.subplots(3,1,figsize=(11,9),layout="constrained")
    for axis, case in zip(axes, shared.CASES, strict=True):
        reference = np.asarray(arms[50]["evaluation"]["cases"][case]["comparison"]["reference_quantiles_05_50_95"])[:,5]
        values = [("Existing joint MCMC", "#405562", reference)] + [(f"Epoch {e}",COLORS[e],np.asarray(arms[e]["evaluation"]["cases"][case]["neural_quantiles_05_50_95"])[:,5]) for e in EPOCHS]
        for i, (_, color, q) in enumerate(values):
            axis.errorbar(q[1],i,xerr=np.array([[q[1]-q[0]],[q[2]-q[1]]]),fmt="o",color=color,capsize=3)
        axis.set(title=shared.CASE_LABELS[case],xlabel="c: 5th percentile, median, 95th percentile",xlim=(.48,2.02),
            yticks=np.arange(len(values)),yticklabels=[v[0] for v in values]);axis.invert_yaxis()
    figures["conditional-c"] = save(fig,base,"conditional-c","Three unchanged fixed-observation checks",
        "Each network and MCMC reference infers all six targets jointly. The market observation is a model/prior-adequacy diagnostic.")
    return figures


def build(protocol_path, out):
    manifest_path = collect(protocol_path); data = load_data(manifest_path); inputs = data["inputs"]
    out = Path(out).resolve(); out.parent.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({"svg.fonttype":"none","svg.hashsalt":"heston-zca-checkpoint-replay-v1","font.family":"DejaVu Sans","font.size":9}):
        figures = plots(data,out.parent)
    arms = data["arms"]; stats = selected.statistics(data["selected"])
    csspath = Path(shared.__file__).with_name("render_heston_spread_report.py")
    inputs.sources.append({"label":"Shared stylesheet",**binding(csspath)})
    pieces = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Heston ZCA checkpoint replay</title><style>'+shared.stylesheet(csspath)+'</style></head><body><main>',
        '<header><p class="eyebrow">Exact-prefix replay · development diagnosis</p><h1>Does earlier stopping repair the ZCA posterior?</h1><p class="lead">Earlier checkpoints are compared with the same final epoch-200 model, on identical observations and references. This isolates checkpoint choice from changes to data, whitening, architecture and learning-rate schedule.</p><button onclick="window.print()">Print report</button></header>',
        '<nav><a href="#overview">Overview</a><a href="#training">Training</a><a href="#calibration">Calibration</a><a href="#recovery">Recovery</a><a href="#mcmc">MCMC</a><a href="#conditional">Conditional</a><a href="#sources">Sources</a></nav>']
    rows=[]
    for epoch,arm in arms.items():
        p=arm["evaluation"]["population"];m=p["recovery"];coverage=p["coverage"]["0.9"]["coverage"]
        rows.append([str(epoch),str(epoch*40),fmt(data["original_curve"][epoch-1]["validation_probe_loss"]),fmt(m["median_rmse"][-1]),
            f'{m["truth_posterior_median_correlation"][-1]:+.4f}',fmt(m["median_posterior_width90"][-1]),
            f'{100*coverage[-1]:.1f}%',f'{100*min(coverage[:5]):.1f}–{100*max(coverage[:5]):.1f}%',f'{sum(p["rank_dkw"]["passed"])}/6'])
    pieces.append(section("The five checkpoints on the same development observations",table(["Epoch","Update","Validation loss","c RMSE","c correlation","Median c 90% width","c 90% coverage","Physical 90% coverage range","SBC within band"],rows)+
        '<p>Nominal coverage is 90%. Lower validation loss and good point recovery are useful evidence, but neither alone certifies posterior uncertainty. The validation minimum at epoch 83 was already observed before this replay; these comparisons diagnose a development model and do not constitute independent acceptance.</p>',"overview"))
    snippet='''flow = bf.networks.CouplingFlow(
    transform="spline", depth=6,
    subnet_kwargs={"widths": [128, 128], "dropout": 0.0},
)
lr = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=5e-4, decay_steps=8000, alpha=0.1,
)
optimizer = keras.optimizers.Adam(lr)
workflow = bf.BasicWorkflow(
    adapter=bounded_target_adapter, inference_network=flow,
    standardize=["inference_variables"], optimizer=optimizer,
)'''
    pieces.append(section("What is replayed exactly",'<p>The replay uses the same 10,000 fixed noisy training observations, 100 fixed validation probes, frozen training-fitted ZCA transform, seed 9011, bounded six-target adapter, initialization, permutations, batch size 256 and 16-row final batch. Forty updates form one epoch. No fresh noise is introduced. The financial law remains Y=P(φ)+c h Z, with c∼Uniform(0.5,2) inferred jointly with five Heston parameters.</p><p>It stops after 100 epochs (4000 updates, 1m presentations), saving epochs 50/75/83/100. The learning-rate schedule retains its original 8000-update horizon; the final epoch-200 weights and evaluations are reused. Initialization and all 100 saved loss-probe records must match the original trajectory exactly before this report can render. The replay also verifies native checkpoint reloads with the same sampling seed.</p><pre><code>'+esc(snippet)+'</code></pre>'+chart(figures["learning-curves"],"Original fixed-probe losses with all evaluated checkpoints marked. Epoch 83 has the minimum observed validation loss; the previously reported model was epoch 200.")+ '<p>Dropout is explicitly zero. Whitening changes the numerical coordinates of the complete noisy quote vector; it does not add another noise source or change the posterior target.</p>',"training"))
    rows=[]
    for j,name in enumerate(NAMES):
        cells=[esc(LABELS[name])]
        for epoch in EPOCHS:
            p=arms[epoch]["evaluation"]["population"]
            cells.append(f'{100*p["coverage"]["0.9"]["coverage"][j]:.1f}% / '+("within" if p["rank_dkw"]["passed"][j] else "outside"))
        rows.append(cells)
    pieces.append(section("Coverage and SBC for every target",table(["Parameter"]+[f"Epoch {e}: coverage / SBC" for e in EPOCHS],rows)+
        chart(figures["coverage"],"Parameterwise coverage of nominal central 90% intervals on 512 identical full-prior development datasets.")+chart(figures["sbc-ecdf"],"Six marginal rank ECDF deviations. The band adjusts over six marginals within each epoch; it makes no simultaneous or fresh acceptance claim across these posthoc checkpoints."),"calibration"))
    recovery_body='<p>Every panel shows all 512 posterior medians, the unit line, and central 90% intervals for the same 32 fixed-index observations. Open each checkpoint to compare recovery with its coverage and SBC above.</p>'
    for epoch in EPOCHS:
        recovery_body+=f'<details'+(' open' if epoch==83 else '')+f'><summary>Epoch {epoch}: all six parameters</summary>'+chart(figures[f"recovery-epoch{epoch:03d}"],f"Epoch {epoch}: medians and a fixed subset of marginal posterior widths.")+'</details>'
    pieces.append(section("Basic parameter recovery",recovery_body,"recovery"))
    rows=[]
    for key,label in [("mcmc","Existing joint MCMC")]+[(f"epoch{e}",f"Epoch {e}") for e in EPOCHS]:
        s=stats[key]
        rows.append([label,str(s["cases"]),fmt(s["median_rmse"][-1]),fmt(s["truth_median_correlation"][-1]),fmt(s["median_width90"][-1])])
    pieces.append(section("Matched selected-case MCMC comparison",chart(figures["selected-c-recovery"],"Eight unchanged selected observations. Seven MCMC references are ready; the failed eighth reference remains visibly excluded from matched statistics.")+table(["Method","Ready matched cases","c median RMSE","c truth–median correlation","Median c 90% width"],rows)+'<p>These cases were selected by generating-c strata before reference sampling. Their recovery is descriptive and cannot replace population SBC. All references infer the full six-dimensional posterior; none fixes the physical parameters to their generating values. No new MCMC was run.</p>',"mcmc"))
    body=chart(figures["conditional-c"],"Exactly the same three observation vectors, saved joint references and neural sampling seeds.")+'<p>A conditional result means inference for one fixed observed quote vector. Each cell below is absolute posterior-mean error divided by the reference SD / neural central-90% width divided by reference width. Zero / one indicates agreement for these two summaries. These are descriptive checks, not a newly declared pass/fail family. The market case additionally tests model/prior adequacy.</p>'
    for case in shared.CASES:
        quantities=arms[50]["evaluation"]["cases"][case]["comparison"]["quantities"];rows=[]
        for j,name in enumerate(quantities):
            cells=[esc(LABELS.get(name,name))]
            for epoch in EPOCHS:
                comp=arms[epoch]["evaluation"]["cases"][case]["comparison"]
                cells.append(fmt(comp["standardized_mean_error"][j])+" / "+fmt(comp["width_ratio"][j]))
            rows.append(cells)
        body+='<h3>'+esc(shared.CASE_LABELS[case])+'</h3>'+table(["Quantity"]+[f"Epoch {e}" for e in EPOCHS],rows)
    pieces.append(section("Fixed-observation reference agreement",body,"conditional"))
    source_rows=[[esc(s["label"]),'<code>'+esc(s["path"])+'</code>','<code>'+esc(s["sha256"])+'</code>'] for s in inputs.sources]
    pieces.append(section("Data boundaries and reproducible evidence",'<p>The raw training and evaluation data, curve prefix, source identities, frozen preprocessing and model payloads are bound by SHA-256. Reserved arrays remain closed. No pricing labels, quotes, priors, architecture, MCMC chains, grid redesign or Bergomi training change in this checkpoint diagnosis.</p>'+table(["Input","Path","SHA256"],source_rows),"sources"))
    summary={"format":"heston_spread_whitening_checkpoint_replay_plot_summary_v1","protocol":binding(protocol_path),
        "manifest":binding(manifest_path),"epochs":list(EPOCHS),"population_rows":512,"selected_cases":8,
        "ready_selected_cases":int(data["selected"]["ready"].sum()),"selected_statistics":stats,
        "figures":{name+suffix:binding(out.parent/(name+suffix)) for name in figures for suffix in (".png",".svg")},
        "sources":inputs.sources,"scientific_acceptance":False,
        "result_scope":"Exact-prefix checkpoint diagnosis on reused development data; epoch83 selection is posthoc."}
    pieces.append('<script type="application/json" id="report-inventory">'+json.dumps(summary).replace("<","\\u003c")+'</script></main></body></html>')
    out.write_text("\n".join(pieces));summary["report"]=binding(out)
    out.with_name("plot-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,required=True);parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();print(build(args.protocol,args.out)["report"])
