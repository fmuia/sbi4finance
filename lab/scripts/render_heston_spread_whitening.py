"""Compact offline report for the completed fixed-input whitening pilot.

Read compact records and saved quantiles only. No learned models, banks,
native chains, pricing or posterior queries are used during presentation.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import render_heston_spread_scale_capacity_report as capacity
from scripts import render_heston_spread_mcmc_recovery as selected

shared = capacity.shared
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MODES = ("diagonal", "zca")
LABEL = {"diagonal": "Fixed diagonal", "zca": "Fixed ZCA", "mcmc": "Joint MCMC"}
COLOR = {"diagonal": "#b57920", "zca": "#246da0", "mcmc": "#b44b37"}
NAMES, LABELS = shared.NAMES, shared.LABELS
esc, fmt, table, section, chart = shared.esc, shared.fmt, shared.table, shared.section, shared.chart


def read(path): return json.loads(Path(path).read_text())
def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": shared.digest(path)}


def collect(study_path):
    """Validate complete arms, then copy only compact training records."""
    study_path = Path(study_path).resolve(); study = read(study_path); base = study_path.parent
    if study["format"] != "heston_spread_whitening_study_v1" or [a["mode"] for a in study["arms"]] != list(MODES):
        raise ValueError("the declared paired whitening study is required")
    records = []
    for arm in study["arms"]:
        mode, payload = arm["mode"], Path(arm["output"])
        training = read(payload/"training.json")
        checkpoint = payload/"checkpoints/update-008000"; metadata = read(checkpoint/"checkpoint.json")
        path = base/"evaluations"/mode/"evaluation.json"; evaluation = read(path)
        if (training["status"] != "complete" or training["completed_updates"] != 8000
                or training["completed_bank_passes"] != 200 or training["presentations"] != 2000000
                or training["same_seed_reload_exact"] is not True
                or training["paired_initial_trainable_hashes_exact"] is not True
                or metadata["training_protocol"] != arm["contract"]
                or evaluation["status"] != "complete" or evaluation["mode"] != mode
                or evaluation["study_protocol"] != binding(study_path)
                or evaluation["checkpoint_record_sha256"] != shared.digest(checkpoint/"checkpoint.json")
                or shared.digest(checkpoint/"model.keras") != evaluation["model_sha256"]
                or shared.digest(checkpoint/"state.npz") != evaluation["state_sha256"]
                or shared.digest(checkpoint/"preprocessing.json") != evaluation["preprocessing"]["sha256"]
                or shared.digest(path.parent/"recovery.npz") != evaluation["population"]["recovery_sha256"]):
            raise ValueError("whitening completion, checkpoint, sidecar or evaluation differs")
        with np.load(payload/"exposure.npz", allow_pickle=False) as arrays:
            if (arrays["presentations"].dtype != np.int64 or arrays["presentations"].shape != (10000,)
                    or not np.all(arrays["presentations"] == 200)
                    or not np.array_equal(arrays["row_id"], np.array([f"train:{i:08d}" for i in range(10000)]))):
                raise ValueError("fixed noisy training-row exposure ledger differs")
        records.append((arm, payload, evaluation, path))
    arms = []
    for arm, payload, evaluation, path in records:
        mode = arm["mode"]; compact = base/"training"/mode; compact.mkdir(parents=True, exist_ok=True)
        exports = {}
        for name in ("run.json", "training.json", "learning-curve.json", "progress.json", "initial-probes.json"):
            source, target = payload/name, compact/name
            if target.exists() and shared.digest(source) != shared.digest(target):
                raise ValueError("existing compact whitening export differs")
            if not target.exists(): shutil.copyfile(source, target)
            exports[name] = {"source": binding(source), "copy": binding(target)}
        arms.append({"mode": mode, "run": binding(compact/"run.json"), "training": binding(compact/"training.json"),
            "curve": binding(compact/"learning-curve.json"), "evaluation": binding(path),
            "recovery": binding(path.parent/"recovery.npz"), "exports": exports})
    manifest = {"format": "heston_spread_whitening_report_v1", "protocol": binding(study_path),
        "matched_mcmc_report": study["matched_mcmc_report"], "arms": arms,
        "producer": binding(__file__), "results_verification": binding(base/"results-verification.json"), "scientific_acceptance": False}
    shared_spec = base/"report-manifest.json"
    if shared_spec.exists() and read(shared_spec) != manifest:
        raise ValueError("existing whitening report manifest differs")
    shared_spec.write_text(json.dumps(manifest, indent=2)+"\n")
    return shared_spec


def load_data(manifest_path):
    manifest_path = Path(manifest_path).resolve(); manifest = read(manifest_path)
    if manifest["format"] != "heston_spread_whitening_report_v1": raise ValueError("unsupported whitening report")
    inputs = shared.Artifacts(manifest_path)
    inputs.sources.append({"label": "Whitening report manifest", **binding(manifest_path)})
    inputs.json(manifest["results_verification"], "Independent source/data/count/calibration audit")
    study = inputs.json(manifest["protocol"], "Predeclared paired whitening pilot")
    baseline = inputs.json(study["baseline_evaluation"], "Historical raw-observation population binding")
    target = study["joint_task"]; arms = {}
    for spec in manifest["arms"]:
        mode = spec["mode"]
        run = inputs.json(spec["run"], mode+" fresh training recipe")
        training = inputs.json(spec["training"], mode+" complete exposure ledger")
        curve = inputs.json(spec["curve"], mode+" fixed-probe learning curve")
        e = inputs.json(spec["evaluation"], mode+" final posterior diagnostics")
        if (mode not in MODES or mode in arms or run["task"] != study["task"]
                or run["training_protocol"] != next(a["contract"] for a in study["arms"] if a["mode"] == mode)
                or run["standardize"] != ["inference_variables"]
                or run["architecture"] != {"family": "spline", "depth": 6, "widths": [128,128], "dropout": 0.}
                or e["format"] != "heston_spread_whitening_evaluation_v1" or e["status"] != "complete"
                or e["task"] != target or e["study_protocol"] != manifest["protocol"]
                or e["preprocessing_summary"]["fit_split"] != "train"
                or e["preprocessing_summary"]["fit_rows"] != 10000
                or e["preprocessing_summary"]["validation_used_for_fit"] is not False
                or e["population"]["truth_sha256"] != baseline["population"]["truth_sha256"]
                or e["population"]["observations_sha256"] != baseline["population"]["observations_sha256"]):
            raise ValueError("pilot data, model or training-only preprocessing boundary differs")
        recovery = shared.recovery_data(spec["recovery"], inputs, e, 8000)
        if [p["completed_bank_passes"] for p in curve] != list(range(1, 201)):
            raise ValueError("all 200 saved pass-level loss probes are required")
        for source in e["source_sha256"].values(): selected.source_binding(inputs, source, mode+" executed evaluation source")
        arms[mode] = {"run": run, "training": training, "curve": curve, "evaluation": e,
            "recovery": recovery, "metrics": capacity.scale_metrics(*recovery)}
    if set(arms) != set(MODES): raise ValueError("both fixed transforms required")
    a, b = [arms[m] for m in MODES]
    if (a["run"]["splits"] != b["run"]["splits"]
            or a["run"]["initial_trainable_hashes"] != b["run"]["initial_trainable_hashes"]
            or a["run"]["versions"] != b["run"]["versions"]
            or a["run"]["numeric_configuration"] != b["run"]["numeric_configuration"]):
        raise ValueError("matched noisy data, initialization or software differs")
    mcmc_path = inputs.path(manifest["matched_mcmc_report"], "Unchanged 8-case joint-MCMC manifest")
    mcmc = selected.load_data(mcmc_path); indices = mcmc["selection"]["selection"]["zero_based_population_indices"]
    for mode, arm in arms.items():
        truth, q = arm["recovery"]
        if (not np.array_equal(truth[indices], mcmc["truth"])
                or arm["evaluation"]["population"]["observations_sha256"] != mcmc["selection"]["population_observations_sha256"]):
            raise ValueError("selected MCMC and pilot do not use identical observations/truths")
        mcmc["quantiles"][mode] = q[:, indices, :]
    inputs.sources.extend(mcmc["inputs"].sources)
    for source in study["sources"].values(): selected.source_binding(inputs, source, "Frozen pilot source")
    inputs.sources.append({"label": "Whitening presentation producer", **binding(__file__)})
    return {"study": study, "inputs": inputs, "arms": arms, "target": target, "selected": mcmc}


def figure(fig, base, name, title, description):
    return selected.save_figure(fig, base/name, title, description)


def plots(data, base):
    result = {}; arms = data["arms"]
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.1), layout="constrained")
    cov = arms["zca"]["evaluation"]["preprocessing_summary"]
    eigenvalues = np.asarray(cov["raw_covariance_eigenvalues"])
    axes[0].semilogy(np.arange(1,55), eigenvalues, color=COLOR["zca"])
    axes[0].axhline(eigenvalues[-1]*1e-8, ls=":", color="#7e8790", label="Declared relative floor")
    axes[0].set(xlabel="Ordered covariance eigenvalue", ylabel="Eigenvalue of raw noisy Y/h", title="Same 10k noisy training curves")
    axes[0].legend(fontsize=8)
    axes[1].bar(["Raw", "Diagonal", "ZCA"], [cov["raw_covariance_condition_number"],
        arms["diagonal"]["evaluation"]["preprocessing_summary"]["transformed_covariance_condition_number"], cov["transformed_covariance_condition_number"]],
        color=["#7e8790", COLOR["diagonal"], COLOR["zca"]])
    axes[1].set(yscale="log", ylabel="Covariance condition number", title="Full covariance after fixed preprocessing")
    result["input-covariance"] = figure(fig, base, "input-covariance", "What the input covariance measures",
        "Covariance across 54 quote features in noisy simulated training datasets. It describes both model-parameter variation and quote noise.")
    fig, axis = plt.subplots(figsize=(10.6,4), layout="constrained")
    for mode, arm in arms.items():
        for field, style in (("train_probe_loss", ":"), ("validation_probe_loss", "-")):
            axis.plot([p["completed_bank_passes"] for p in arm["curve"]], [p[field] for p in arm["curve"]], style,
                color=COLOR[mode], label=LABEL[mode]+(" train" if style == ":" else " validation"))
    axis.set(xlabel="Passes over fixed 10k noisy rows", ylabel="Fixed-probe joint negative log density", title="Training and validation losses")
    axis.legend(fontsize=8)
    result["learning-curves"] = figure(fig, base, "learning-curves", "Controlled fixed-data training",
        "Each arm has 8000 updates, batch 256 and 2m presentations. Losses use 512 fixed training and 100 fixed validation rows.")
    fig, axes = plt.subplots(2,3,figsize=(11.7,7.6),layout="constrained")
    bounds=np.asarray(data["target"]["prior_bounds"]);chosen=np.arange(0,512,16)
    for j, axis in enumerate(axes.flat):
        lo,hi=bounds[j];axis.plot([lo,hi],[lo,hi],"--",color="#405562",lw=1)
        for mode,arm in arms.items():
            truth,q=arm["recovery"];axis.scatter(truth[:,j],q[1,:,j],s=8,alpha=.35,color=COLOR[mode],edgecolors="none")
            axis.errorbar(truth[chosen,j],q[1,chosen,j],yerr=np.stack([q[1,chosen,j]-q[0,chosen,j],q[2,chosen,j]-q[1,chosen,j]]),
                fmt="o",ms=2,color=COLOR[mode],capsize=2,elinewidth=.6)
        margin=.025*(hi-lo);axis.set(title=LABELS[NAMES[j]],xlabel="Generating value",ylabel="Posterior median",
            xlim=(lo-margin,hi+margin),ylim=(lo-margin,hi+margin),aspect="equal")
    handles=[Line2D([],[],color=COLOR[m],marker="o",ls="none",label=LABEL[m])for m in MODES]
    fig.legend(handles=handles,loc="outside lower center",ncol=2,frameon=False)
    result["parameter-recovery"] = figure(fig,base,"parameter-recovery","All six parameters on the same 512 datasets",
        "Every median; 90% intervals for 32 fixed-index rows. Good unit-line recovery alone does not establish calibrated uncertainty.")
    fig,axes=plt.subplots(2,3,figsize=(11.7,6.3),layout="constrained");grid=np.linspace(0,1,201)
    for mode,arm in arms.items():
        p=arm["evaluation"]["population"];ranks=np.asarray(p["fractional_ranks"]);band=p["rank_dkw"]["dkw_bound"]
        for j,axis in enumerate(axes.flat):
            axis.plot(grid,np.searchsorted(np.sort(ranks[:,j]),grid,side="right")/512-grid,color=COLOR[mode],label=LABEL[mode])
            if mode==MODES[0]:axis.axhspan(-band,band,color="#dbece1",zorder=0);axis.axhline(0,color="#7e8790",lw=.6)
            axis.set(title=LABELS[NAMES[j]],xlabel="Fractional posterior rank",ylabel="ECDF−uniform CDF",xlim=(0,1))
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc="outside lower center",ncol=2,frameon=False)
    result["sbc-ecdf"] = figure(fig,base,"sbc-ecdf","Calibration screens for all six parameters",
        "The six-marginal DKW band applies within each arm. It does not certify a complete estimator or provide a selected-case coverage guarantee.")
    d=data["selected"];fig,axes=plt.subplots(2,2,figsize=(10.7,8),layout="constrained");x=d["truth"][:,-1]
    for axis,method in zip(axes.flat,("mcmc",*MODES),strict=False):
        mask=d["ready"]if method=="mcmc"else np.ones(8,dtype=bool);q=d["quantiles"][method][:,mask,-1]
        axis.plot([.5,2],[.5,2],"--",color="#405562",lw=1);axis.axhline(1.25,ls=":",color="#7e8790")
        axis.errorbar(x[mask],q[1],yerr=np.stack([q[1]-q[0],q[2]-q[1]]),fmt="o",color=COLOR[method],capsize=3)
        axis.set(title=LABEL[method],xlabel="Generating c",ylabel="Posterior median c",xlim=(.47,2.03),ylim=(.47,2.03),aspect="equal")
    axis=axes.flat[-1];axis.axhline(1.35,ls=":",color="#7e8790",label="Prior 90% width")
    for method in ("mcmc",*MODES):
        mask=d["ready"]if method=="mcmc"else np.ones(8,dtype=bool);q=d["quantiles"][method][:,mask,-1]
        axis.plot(x[mask],q[2]-q[0],"o-",color=COLOR[method],label=LABEL[method])
    axis.set(title="Remaining c uncertainty",xlabel="Generating c",ylabel="Central 90% width",ylim=(0,1.52));axis.set_box_aspect(1);axis.legend(fontsize=8)
    result["selected-c-recovery"] = figure(fig,base,"selected-c-recovery","The same selected observations and existing joint references",
        "All 8 pilot cases and only 7 ready MCMC references; the unready case remains excluded from matched-subset metrics.")
    fig,axes=plt.subplots(3,1,figsize=(10.7,7.2),layout="constrained")
    for axis,case in zip(axes,shared.CASES,strict=True):
        values=[("Uniform prior","#7e8790",np.array([.575,1.25,1.925]))]
        reference=None
        for mode,arm in arms.items():
            e=arm["evaluation"]["cases"][case];r=np.asarray(e["comparison"]["reference_quantiles_05_50_95"])[:,-3]
            # c is index5, followed by two recorded variance quantities.
            if reference is not None and not np.array_equal(reference,r):raise ValueError("conditional reference differs")
            reference=r
            values.append((LABEL[mode],COLOR[mode],np.asarray(e["neural_quantiles_05_50_95"])[:,-1]))
        values.insert(1,("Joint MCMC reference",COLOR["mcmc"],reference))
        for i,(_,color,q)in enumerate(values):axis.errorbar(q[1],i,xerr=np.array([[q[1]-q[0]],[q[2]-q[1]]]),fmt="o",color=color,capsize=3)
        axis.set(title=shared.CASE_LABELS[case],xlabel="c: q05, median, q95",xlim=(.48,2.02),yticks=np.arange(len(values)),yticklabels=[v[0]for v in values]);axis.invert_yaxis()
    result["conditional-c"] = figure(fig,base,"conditional-c","Three existing fixed-observation conditional checks",
        "All six parameters are inferred jointly. The market reference is a separate model/prior-adequacy diagnostic.")
    return result


def links(name):return f'<p><a href="{name}.png">PNG</a> · <a href="{name}.svg">SVG</a></p>'


def build(study_path,out):
    manifest_path=collect(study_path);data=load_data(manifest_path);inputs=data["inputs"];out=Path(out).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    arms=data["arms"];diag=arms["diagonal"]["evaluation"]["population"];zca=arms["zca"]["evaluation"]["population"]
    dm,zm=[arms[m]["evaluation"]["population"]["recovery"]for m in MODES];coverage=zca["coverage"]["0.9"]["coverage"]
    lead=(f"ZCA improves c median RMSE from {dm['median_rmse'][-1]:.4f} to {zm['median_rmse'][-1]:.4f} in this controlled pilot. "
        f"But all six ZCA SBC screens fail: nominal 90% intervals cover only {100*min(coverage[:5]):.1f}–{100*max(coverage[:5]):.1f}% of physical truths and {100*coverage[-1]:.1f}% of c truths. "
        "This small fixed-data estimator is not calibrated or accepted for use.")
    with plt.rc_context({"svg.fonttype":"none","svg.hashsalt":"heston-fixed-zca-v1","font.family":"DejaVu Sans","font.size":9}):figures=plots(data,out.parent)
    csspath=Path(shared.__file__).with_name("render_heston_spread_report.py");inputs.sources.append({"label":"Shared stylesheet",**binding(csspath)})
    pieces=['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Heston diagonal and ZCA pilot</title><style>'+shared.stylesheet(csspath)+'</style></head><body><main>',
        '<header><p class="eyebrow">Paired fixed-data preprocessing pilot</p><h1>Whitening reveals c information, but calibration remains incomplete</h1><p class="lead">'+esc(lead)+'</p><button onclick="window.print()">Print report</button></header>',
        '<nav><a href="#calibration">Calibration</a><a href="#covariance">Covariance</a><a href="#recipe">Training</a><a href="#recovery">Recovery</a><a href="#mcmc">Matched MCMC</a><a href="#conditional">Conditional checks</a><a href="#sources">Sources</a></nav>']
    rows=[]
    for i,name in enumerate(NAMES):
        cells=[esc(LABELS[name])]
        for mode in MODES:
            p=arms[mode]["evaluation"]["population"];d=p["rank_dkw"]
            cells.extend([f'{100*p["coverage"]["0.9"]["coverage"][i]:.1f}%',fmt(d["rank_ks"][i]),"Within band"if d["passed"][i]else"Outside band"])
        rows.append(cells)
    pieces.append(section("Uncertainty calibration is the limiting result",table(["Parameter","Diagonal 90% coverage","Diagonal rank KS","Diagonal SBC","ZCA 90% coverage","ZCA rank KS","ZCA SBC"],rows)+
        '<p>The nominal central credible level is 90%. The same 512 validation truths and noisy observations are used in both arms. Better point recovery and narrower intervals do not establish a calibrated posterior. These prior-predictive SBC screens adjust over six marginals within an arm; the pilot provides no release acceptance family.</p>'+chart(figures["sbc-ecdf"],"All six marginal rank ECDF deviations. Calibration problems are visible even when medians follow the unit line.")+links("sbc-ecdf"),"calibration"))
    metrics=[]
    for mode in MODES:
        e=arms[mode]["evaluation"];m=e["population"]["recovery"]
        metrics.append([LABEL[mode],fmt(m["median_rmse"][-1]),f'{m["truth_posterior_median_correlation"][-1]:+.4f}',fmt(m["median_posterior_width90"][-1]),f'{sum(e["population"]["rank_dkw"]["passed"])}/6'])
    pieces.append(section("What the input covariance means",'<div class="equation">xᵢ=Yᵢ/hᵢ; &nbsp; Σᵢⱼ=mean-training covariance of feature i with feature j</div><p>Each row is a simulated 54-quote surface; each column is one fixed option quote divided by its half-spread. A common change in volatility level, skew or maturity shape moves many columns together. Σ is estimated across the 10,000 noisy training rows and measures that shared variation, including variation from the physical-parameter prior and quote noise.</p><div class="equation">Yᵢ=Pᵢ(φ)+c hᵢ Zᵢ, &nbsp; Zᵢ∼N(0,1)</div><p>The observation model and priors stay fixed. Both networks infer all five Heston parameters and c jointly. The preprocessing changes their numerical input representation while retaining all 54 features.</p><p>Diagonal preprocessing subtracts the fitted mean and divides each feature by its training standard deviation. Full ZCA also uses off-diagonal covariance: if Σ=Q diag(λ) Qᵀ, its symmetric matrix is W=Q diag(max(λ,10⁻⁸λmax)⁻¹ᐟ²) Qᵀ and the network receives (x−mean)W. The eigenvalue floor bounds amplification; no eigenvector is discarded. The mean and matrix are fitted in float64 on training only, saved as an immutable sidecar, reused once for validation/inference, and then cast to float32.</p>'+chart(figures["input-covariance"],"Both arms use the same noisy training covariance. Diagonal scaling leaves strong dependence; ZCA brings the full fitted covariance close to identity. The declared floor clips "+str(arms["zca"]["evaluation"]["preprocessing_summary"]["floored_eigenvalues"])+" eigenvalues in this run.")+links("input-covariance"),"covariance"))
    pieces.append(section("Controlled training recipe",table(["Input transform","c median RMSE · 512","c truth–median correlation","Median c 90% width","SBC screens"],metrics)+
        '<p>Both fresh bounded-target BayesFlow 2.1 spline CouplingFlows use depth 6, 128×128 conditioners, zero dropout, identical trainable initialization, the same fixed10,000 noisy rows, batch 256 and 200 passes: 8000 updates and 2m presentations. Both use the same seeded permutations and unpadded 16-row tails. Adam follows CosineDecay(5e−4,8000,alpha=.1), ending at 5e−5. Only targets have BayesFlow running standardization; input transforms are frozen before training. All six targets retain their declared bounded transformations and priors.</p>'+chart(figures["learning-curves"],"The losses use 512 fixed training probes and 100 fixed validation probes. The separate final recovery/SBC population has 512 cases. A widening training/validation gap is evidence to examine before scaling or accepting this small fixed-data fit.")+links("learning-curves")+
        '<p>The source notebook also changes to FlowMatching with three 128-unit layers, dropout .1 and raw unconstrained targets. This pilot tests its whitening idea while holding the two new arms’ network and training recipe fixed. Historical 500-pass networks used a warm start, 100k physical rows and fresh c/noise on every pass with 50m examples; their saved results below are context, not matched controls for this small fresh fit.</p>',"recipe"))
    pieces.append(section("Parameterwise recovery",chart(figures["parameter-recovery"],"All 512 posterior medians and 90% intervals for 32 fixed-index rows per arm. Unit-line alignment measures point recovery; the coverage/SBC table above evaluates uncertainty.")+links("parameter-recovery"),"recovery"))
    stats=selected.statistics(data["selected"]);rows=[]
    for method,label in (("mcmc","Existing joint MCMC"),("diagonal",LABEL["diagonal"]),("zca",LABEL["zca"]),("nn128","Historical 50m · batch 128"),("nn512","Historical 50m · batch 512"),("prior","Uniform prior baseline")):
        s=stats[method];correlation=s["truth_median_correlation"][-1]
        rows.append([label,str(s["cases"]),fmt(s["median_rmse"][-1]),"Constant/undefined"if correlation is None else f'{correlation:+.4f}',fmt(s["median_width90"][-1])])
    pieces.append(section("The same selected data and existing joint references",chart(figures["selected-c-recovery"],"No new MCMC is run. The 8 cases were selected by generating-c rank before reference sampling; 7 references passed convergence and sampled price/likelihood gates. The unready case remains excluded from every matched-subset statistic.")+links("selected-c-recovery")+table(["Method","Matched ready cases","c median RMSE","c truth–median correlation","Median c 90% width"],rows)+
        '<p>This selected panel is descriptive recovery, not SBC or a population coverage estimate. Each reference infers all six parameters jointly; it is not a known-physical-parameter oracle. Old networks are represented only by their already saved quantiles. Full-population calibration takes priority over an attractive selected-case plot.</p>',"mcmc"))
    rows=[]
    for case in shared.CASES:
        for mode in MODES:
            c=arms[mode]["evaluation"]["cases"][case]["comparison"];index=c["quantities"].index("c")
            rows.append([esc(shared.CASE_LABELS[case]),LABEL[mode],fmt(c["standardized_mean_error"][index]),fmt(c["width_ratio"][index])])
    pieces.append(section("Existing fixed-observation conditional diagnostics",chart(figures["conditional-c"],"Each observation vector remains unchanged and all six parameters are inferred. The actual market case is a separate model/prior-adequacy diagnostic.")+links("conditional-c")+table(["Observation case","Input transform","c absolute mean error/refSD","c 90% width/refwidth"],rows)+
        '<p>These are descriptive point summaries of 4096 neural draws and the existing ready joint references. There is no newly declared simultaneous comparison family or pass/fail acceptance threshold in this pilot.</p>',"conditional"))
    source_rows=[[esc(s["label"]),'<code>'+esc(s["path"])+'</code>','<code>'+esc(s["sha256"])+'</code>']for s in inputs.sources]
    pieces.append(section("Data boundaries and source evidence",'<p>Both fit-condition hashes, target hashes and row hashes match. Preprocessing sees training rows only. The raw evaluation conditions and truths match the historical 512-case population exactly. Sidecars are bundled with each native checkpoint and checked by source/task/data hashes before sampling. No reserved data, new labels, new reference sampling, grid redesign or Bergomi changes enter this pilot.</p>'+table(["Input","Path","SHA256"],source_rows),"sources"))
    artifacts={name+suffix:binding(out.parent/(name+suffix))for name in figures for suffix in(".png",".svg")}
    summary={"format":"heston_spread_whitening_plot_summary_v1","protocol":binding(study_path),"manifest":binding(manifest_path),
        "modes":list(MODES),"population_rows":512,"selected_cases":8,"ready_selected_cases":int(data["selected"]["ready"].sum()),
        "selected_statistics":stats,"figures":artifacts,"sources":inputs.sources,"scientific_acceptance":False,
        "result_scope":"Controlled input representation pilot improves c recovery but is not a calibrated/accepted estimator; historical training recipes are contextual."}
    pieces.append('<script type="application/json" id="report-inventory">'+json.dumps(summary).replace("<","\\u003c")+'</script></main></body></html>')
    out.write_text("\n".join(pieces));summary["report"]=binding(out);out.with_name("plot-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--study",type=Path,required=True);parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();print(build(args.study,args.out)["report"])
