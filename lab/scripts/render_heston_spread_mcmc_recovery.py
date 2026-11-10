"""Plot selected joint-MCMC recovery against stored 500-pass neural recovery.

This presentation consumer never prices, samples or loads a learned model.
It reads hash-bound compact reference records and the two existing population
recovery NPZs. Selected truths remain separate from the sampler's quote panel.
Use --summary report-manifest.json --out report.html; figures are written beside
the new report as standalone SVG and PNG as well as embedded inline.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import render_heston_spread_scale_report as shared
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


NAMES, LABELS = shared.NAMES, shared.LABELS
METHODS = ("mcmc","nn128","nn512")
METHOD_LABELS = {"mcmc":"Joint MCMC", "nn128":"Network · batch 128", "nn512":"Network · batch 512"}
COLORS = {"mcmc":"#b44b37", "nn128":"#287b6a", "nn512":"#654a97", "prior":"#7e8790"}
MARKERS = {"mcmc":"o", "nn128":"^", "nn512":"s"}
esc, table, section, chart, fmt = shared.esc, shared.table, shared.section, shared.chart, shared.fmt


def digest_values(values):
    return hashlib.sha256(np.asarray(values,dtype="<f8").tobytes()).hexdigest()


def optional_record(inputs,spec,label):
    return None if spec is None else inputs.json(spec,label)


def source_binding(inputs,spec,label):
    """Hash executable/environment sources without opening scientific payloads."""
    path=Path(spec["path"])
    path=(path if path.is_absolute() else inputs.base/path).resolve()
    if path.suffix not in (".py",".toml",".lock") or shared.digest(path)!=spec["sha256"]:
        raise ValueError("executed reference source identity differs: "+label)
    if not any(s["path"]==str(path) and s["sha256"]==spec["sha256"] for s in inputs.sources):
        inputs.sources.append({"label":label,"path":str(path),"sha256":spec["sha256"]})


def load_population(inputs,spec,target):
    batch = spec["batch_size"]
    evaluation = inputs.json(spec["evaluation"],f"Final batch {batch} population evaluation")
    path = inputs.path(spec["recovery"],f"Final batch {batch} saved recovery quantiles",recovery=True)
    if path.name != "recovery.npz":
        raise ValueError("only the compact population recovery archive may be opened")
    p = evaluation["population"]
    if (batch not in (128,512) or evaluation.get("completed_bank_passes") != 500
            or evaluation.get("batch_size") != batch or evaluation.get("task") != target
            or p["parameter_names"] != list(NAMES) or p["rows"] != 512
            or p["recovery_sha256"] != shared.digest(path)):
        raise ValueError("network task, endpoint or population identity differs")
    with np.load(path,allow_pickle=False) as archive:
        truth, q, ids = archive["truth"].copy(), archive["quantiles"].astype(float), archive["row_id"].copy()
    if (truth.shape != (512,6) or q.shape != (3,512,6) or not np.isfinite(truth).all()
            or not np.isfinite(q).all() or np.any(np.diff(q,axis=0)<0)
            or digest_values(truth) != p["truth_sha256"]
            or not np.array_equal(ids,np.array([f"validation:{i:08d}" for i in range(512)]))):
        raise ValueError("saved neural recovery truths, quantiles or ordering differ")
    return {"evaluation":evaluation,"truth":truth,"quantiles":q,"row_id":ids,"spec":spec}


def reference_diagnostics(diagnostics):
    if diagnostics is None:
        return {"max_rhat":None,"min_bulk_ess":None,"min_tail_ess":None}
    rows = [*diagnostics.get("parameters",{}).values(),
            *diagnostics.get("products",{}).get("parameters",{}).values()]
    def numbers(key):
        result = []
        for row in rows:
            try:value=float(row[key])
            except (KeyError,TypeError,ValueError):continue
            if np.isfinite(value):result.append(value)
        return result
    rh,bulk,tail = numbers("r_hat"),numbers("ess_bulk"),numbers("ess_tail")
    return {"max_rhat":max(rh) if rh else None,
            "min_bulk_ess":min(bulk) if bulk else None,"min_tail_ess":min(tail) if tail else None}


def load_reference(inputs,spec,case,panel_spec,protocol_spec,target):
    ident=case["id"]
    run=optional_record(inputs,spec.get("run"),f"{ident} sampler run")
    diag=optional_record(inputs,spec.get("diagnostics"),f"{ident} sampler diagnostics")
    posterior=optional_record(inputs,spec.get("posterior"),f"{ident} compact posterior")
    if run is not None:
        if (run.get("format") != "heston_spread_recovery_reference_run_v1" or run["case_id"] != ident
                or run["panel_sha256"] != panel_spec["sha256"] or run["observation_sha256"] != case["observations_sha256"]
                or run["task"] != target or run.get("truth_available_to_sampler") is not False
                or run.get("names") != list(NAMES) or run.get("bounds") != target["prior_bounds"]
                or run.get("settings",{}).get("chains") != 4 or run.get("smoke") is not False):
            raise ValueError("reference target, observations or truth isolation differ")
        for name,value in run.get("source_sha256",{}).items():
            source_binding(inputs,value,f"{ident} executed source: {name}")
        # The protocol binding may be represented directly or by path/hash fields.
        bound = run.get("protocol",{})
        recorded_hash = run.get("protocol_sha256",run.get("study_protocol_sha256",bound.get("sha256")))
        if recorded_hash is not None and recorded_hash != protocol_spec["sha256"]:
            raise ValueError("reference run uses a different declared sampler protocol")
    stats=reference_diagnostics(diag)
    ready = (run is not None and run.get("status") in ("complete","completed")
        and run.get("reference_ready") is True and diag is not None and diag.get("reference_ready") is True
        and posterior is not None)
    q = None
    if ready:
        if (diag.get("parameter_mcmc_diagnostics_passed") is not True
                or diag.get("mcmc_diagnostics_passed") is not True
                or diag.get("pointwise_precision_passed") is not True
                or stats["max_rhat"] is None or stats["max_rhat"]>1.01
                or stats["min_bulk_ess"] is None or stats["min_bulk_ess"]<400
                or stats["min_tail_ess"] is None or stats["min_tail_ess"]<400):
            raise ValueError("ready reference contradicts its declared diagnostic gates")
        q=np.array([posterior["parameter_quantiles_05_50_95"][name] for name in NAMES],dtype=float).T
        bounds=np.asarray(target["prior_bounds"])
        if (q.shape != (3,6) or not np.isfinite(q).all() or np.any(np.diff(q,axis=0)<0)
                or np.any(q<bounds[:,0]) or np.any(q>bounds[:,1])):
            raise ValueError("invalid ready joint-MCMC quantiles")
        if (not isinstance(run.get("chain_payload_sha256"),str)
                or len(run["chain_payload_sha256"]) != 64
                or run.get("chain_payload_sha256") != diag.get("chain_payload_sha256")):
            raise ValueError("run and diagnostic refer to different native chain payloads")
    reason = "Passed declared convergence and sampled numerical gates" if ready else spec.get("reason")
    if not reason:
        reason = (run or {}).get("error",(run or {}).get("status",spec.get("status","Missing reference output")))
        if isinstance(reason,dict):reason=json.dumps(reason)
    if not ready and run is not None and run.get("status") in ("complete","completed"):
        issues=[]
        if diag is None:issues.append("Reference diagnostics unavailable")
        else:
            if diag.get("mcmc_diagnostics_passed") is not True:issues.append("Chain diagnostics failed")
            if diag.get("pointwise_precision_passed") is not True:issues.append("Sampled numerical-precision gates did not pass")
        if posterior is None:issues.append("Compact posterior unavailable")
        reason="; ".join(issues) or "Reference not marked ready"
    return {"case_id":ident,"ready":ready,"quantiles":q,"run":run,"diagnostics":diag,
            "stats":stats,"reason":str(reason)}


def load_data(summary_path):
    summary_path=Path(summary_path).resolve();summary=json.loads(summary_path.read_text())
    if summary.get("format") != "heston_spread_mcmc_recovery_report_v1":
        raise ValueError("unsupported selected-recovery report manifest")
    if type(summary.get("interim",False)) is not bool:
        raise ValueError("interim marker must be boolean")
    inputs=shared.Artifacts(summary_path)
    inputs.sources.append({"label":"Selected-recovery report manifest","path":str(summary_path),"sha256":shared.digest(summary_path)})
    protocol=inputs.json(summary["protocol"],"Predeclared reference protocol")
    panel=inputs.json(summary["panel"],"Observation-only sampler panel")
    selection=inputs.json(summary["truth_selection"],"Report-only truths and selection rule")
    if (protocol.get("format") != "heston_spread_recovery_reference_protocol_v1"
            or panel.get("format") != "heston_spread_recovery_observation_panel_v1"
            or selection.get("format") != "heston_spread_recovery_truth_selection_v1"
            or selection["panel"]["sha256"] != summary["panel"]["sha256"]
            or selection["parameter_names"] != list(NAMES)
            or panel["joint_task"]["parameter_names"] != list(NAMES)
            or selection["prior_bounds"] != panel["joint_task"]["prior_bounds"]
            or protocol.get("truth_available_to_sampler") is not False
            or protocol.get("panel_sha256") != summary["panel"]["sha256"]
            or protocol.get("design_sha256") != panel["design_sha256"]
            or selection.get("reserved_arrays_opened") is not False
            or selection.get("new_pricing_labels") is not False or selection.get("new_neural_sampling") is not False):
        raise ValueError("selection, sampler panel or full joint target differs")
    for name,value in protocol.get("source_sha256",{}).items():
        source_binding(inputs,value,"Reference protocol source: "+name)
    target=panel["joint_task"]
    if target.get("infer_c") is not True or target["prior_bounds"][-1] != [.5,2.] or target["quote_count"] != 54:
        raise ValueError("the unchanged54-quote joint-c posterior target is required")
    settings=protocol["settings"]
    if (settings["case_count"]!=8 or settings["chains"]!=4
            or settings["attempts"]!=[[2000,2000],[6000,12000]]
            or settings["smoke_test_only"]is not False or settings["automatic_retry_or_budget_extension"]is not False):
        raise ValueError("selected recovery report requires the declared full reference recipe")
    if selection["networks"] != summary["networks"]:
        raise ValueError("report substituted neural populations after selection")
    populations={spec["batch_size"]:load_population(inputs,spec,target) for spec in summary["networks"]}
    if set(populations)!={128,512}:
        raise ValueError("both predeclared final neural populations required")
    reference_population=populations[128];alltruth=reference_population["truth"]
    for population in populations.values():
        if (not np.array_equal(population["truth"],alltruth)
                or population["evaluation"]["population"]["truth_sha256"] != selection["population_truth_sha256"]
                or population["evaluation"]["population"]["observations_sha256"] != selection["population_observations_sha256"]):
            raise ValueError("methods did not use the same existing validation observations")
    rule=selection["selection"];ranks=rule["zero_based_c_ranks"];indices=rule["zero_based_population_indices"]
    if (rule["population_rows"] != 512 or rule["replacement_of_failed_cases"] is not False
            or ranks != list(range(32,512,64))
            or not np.array_equal(np.argsort(alltruth[:,-1],kind="stable")[ranks],indices)):
        raise ValueError("cases differ from the frozen generating-c rank selection")
    cases=selection["cases"]
    if (len(cases)!=len(indices) or len(set(x["id"] for x in cases))!=len(cases)
            or [c["id"] for c in cases]!=[c["id"] for c in panel["cases"]]):
        raise ValueError("truths and observation-only case identities differ")
    for i,(case,observed) in enumerate(zip(cases,panel["cases"],strict=True)):
        if (case["population_index"] != indices[i] or case["c_rank"] != ranks[i]
                or case["row_id"] != reference_population["row_id"][indices[i]]
                or not np.array_equal(case["truth"],alltruth[indices[i]])
                or case["observations_sha256"] != observed["observations_sha256"]
                or digest_values(observed["observations"]) != case["observations_sha256"]
                or any(key in observed for key in ("truth","clean","error","parameters"))):
            raise ValueError("selected truths, observations or no-truth sampler boundary differ")
    specs=summary["references"]
    if len(specs)!=len(cases) or set(s["case_id"] for s in specs)!=set(c["id"] for c in cases):
        raise ValueError("every selected case must retain an explicit reference status")
    references=[load_reference(inputs,next(s for s in specs if s["case_id"]==c["id"]),c,
        summary["panel"],summary["protocol"],target) for c in cases]
    campaign=optional_record(inputs,summary.get("campaign"),"Bounded reference campaign status")
    for key,spec in summary.get("sources",{}).items():inputs.path(spec,"Executed source: "+key)
    inputs.path(selection["source"],"Frozen selection producer source")
    truth=np.asarray([c["truth"] for c in cases]);quantiles={f"nn{bs}":p["quantiles"][:,indices,:] for bs,p in populations.items()}
    q=np.full((3,len(cases),6),np.nan)
    ready=np.array([r["ready"] for r in references])
    for i,r in enumerate(references):
        if r["ready"]:q[:,i,:]=r["quantiles"]
    quantiles["mcmc"]=q
    return {"summary":summary,"inputs":inputs,"protocol":protocol,"panel":panel,"selection":selection,
        "target":target,"cases":cases,"truth":truth,"quantiles":quantiles,"references":references,
        "ready":ready,"campaign":campaign}


def statistics(data):
    mask=data["ready"];truth=data["truth"][mask];n=int(mask.sum());result={}
    if not n:return result
    for method,q in data["quantiles"].items():
        q=q[:,mask,:];median=q[1]
        result[method]={"cases":n,"median_rmse":np.sqrt(np.mean((median-truth)**2,axis=0)).tolist(),
            "truth_median_correlation":[float(np.corrcoef(truth[:,i],median[:,i])[0,1])
                if n>=2 and np.ptp(truth[:,i])>0 and np.ptp(median[:,i])>0 else None for i in range(6)],
            "median_width90":np.median(q[2]-q[0],axis=0).tolist()}
    prior=np.asarray(data["target"]["prior_bounds"]);mid=prior.mean(axis=1)
    result["prior"]={"cases":n,"median_rmse":np.sqrt(np.mean((mid-truth)**2,axis=0)).tolist(),
        "truth_median_correlation":[None]*6,"median_width90":(.9*(prior[:,1]-prior[:,0])).tolist()}
    return result


def save_figure(fig,out,title,description):
    svg=shared._svg(fig,title,description)
    Path(str(out)+".svg").write_text(svg)
    fig.savefig(Path(str(out)+".png"),dpi=160,bbox_inches="tight")
    return svg


def parameter_recovery(data,out):
    fig,axes=plt.subplots(2,3,figsize=(12,7.8),layout="constrained")
    bounds=np.asarray(data["target"]["prior_bounds"])
    for j,axis in enumerate(axes.flat):
        lo,hi=bounds[j];axis.plot([lo,hi],[lo,hi],"--",color="#405562",lw=1)
        for method in ("nn512","nn128","mcmc"):
            mask=data["ready"] if method=="mcmc" else np.ones(len(data["cases"]),dtype=bool)
            q=data["quantiles"][method][:,mask,j];x=data["truth"][mask,j]
            if len(x):axis.errorbar(x,q[1],yerr=np.stack([q[1]-q[0],q[2]-q[1]]),
                fmt=MARKERS[method],color=COLORS[method],ms=4,alpha=.8,capsize=2,elinewidth=1,label=METHOD_LABELS[method])
        margin=.025*(hi-lo)
        axis.set(title=LABELS[NAMES[j]],xlabel="Generating value",ylabel="Posterior median",
            xlim=(lo-margin,hi+margin),ylim=(lo-margin,hi+margin),aspect="equal")
    handles=[Line2D([],[],color=COLORS[m],marker=MARKERS[m],ls="none",label=METHOD_LABELS[m]) for m in METHODS]
    fig.legend(handles=handles,loc="outside lower center",ncol=3,frameon=False)
    return save_figure(fig,out,"Selected-dataset recovery for all six parameters",
        "Every neural case and every ready joint-MCMC case is plotted with its central 90% credible interval. No generating parameter was fixed during reference inference.")


def scale_recovery(data,out):
    fig,axes=plt.subplots(2,2,figsize=(11.8,8),layout="constrained")
    lo,hi=data["target"]["prior_bounds"][-1];x=data["truth"][:,-1];mid=.5*(lo+hi)
    for axis,method in zip(axes.flat,METHODS,strict=False):
        axis.plot([lo,hi],[lo,hi],"--",color="#405562",lw=1,label="Unit line")
        axis.axhline(mid,color=COLORS["prior"],ls=":",label="Prior median")
        mask=data["ready"] if method=="mcmc" else np.ones(len(x),dtype=bool)
        q=data["quantiles"][method][:,mask,-1]
        if np.any(mask):axis.errorbar(x[mask],q[1],yerr=np.stack([q[1]-q[0],q[2]-q[1]]),
            fmt=MARKERS[method],color=COLORS[method],ms=5,capsize=3,elinewidth=1.1)
        if method=="mcmc" and not np.all(mask):
            missing=", ".join(str(i+1) for i,r in enumerate(mask) if not r)
            axis.text(.03,.97,"No ready MCMC: cases "+missing,transform=axis.transAxes,va="top",fontsize=8)
        axis.set(title=METHOD_LABELS[method],xlabel="Generating c",ylabel="Posterior median c",
            xlim=(lo-.03,hi+.03),ylim=(lo-.03,hi+.03),aspect="equal")
    axes.flat[0].legend(fontsize=8,loc="lower right")
    axis=axes.flat[3];axis.axhline(.9*(hi-lo),color=COLORS["prior"],ls=":",label="Uniform prior 90% width")
    for method in METHODS:
        mask=data["ready"] if method=="mcmc" else np.ones(len(x),dtype=bool)
        q=data["quantiles"][method][:,mask,-1]
        if np.any(mask):axis.plot(x[mask],q[2]-q[0],marker=MARKERS[method],color=COLORS[method],label=METHOD_LABELS[method])
    axis.set(title="Remaining c uncertainty",xlabel="Generating c",ylabel="Central 90% posterior width",xlim=(lo-.03,hi+.03),ylim=(0,1.05*.9*(hi-lo)))
    axis.set_box_aspect(1)
    handles,labels=axis.get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=2,frameon=False,fontsize=8)
    return save_figure(fig,out,"Joint-MCMC and neural recovery of the spread multiplier",
        "Three recovery panels retain every available ready case with its 90% posterior interval. The fourth compares interval widths with the fixed Uniform prior width.")


def build(summary_path,out):
    out=Path(out);data=load_data(summary_path);inputs=data["inputs"];stats=statistics(data)
    for label,path in (("Presentation producer",Path(__file__)),("Shared compact-report helpers",Path(shared.__file__)),
        ("Shared SVG helper",Path(shared.__file__).with_name("heston_spread_report_figures.py"))):
        inputs.sources.append({"label":label,"path":str(path.resolve()),"sha256":shared.digest(path)})
    csspath=Path(shared.__file__).with_name("render_heston_spread_report.py")
    inputs.sources.append({"label":"Shared stylesheet","path":str(csspath.resolve()),"sha256":shared.digest(csspath)})
    n=len(data["cases"]);ready=int(data["ready"].sum());out.parent.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({"svg.fonttype":"none","svg.hashsalt":"heston-selected-mcmc-recovery-v1","font.family":"DejaVu Sans","font.size":9}):
        allsvg=parameter_recovery(data,out.parent/"parameter-recovery")
        csvg=scale_recovery(data,out.parent/"c-recovery")
    if ready:
        lead=(f"{ready}/{n} joint-MCMC references passed the declared convergence and sampled numerical checks. "
            f"On those matched cases, c median RMSE is {stats['mcmc']['median_rmse'][-1]:.4f}, "
            f"versus {stats['nn128']['median_rmse'][-1]:.4f} and {stats['nn512']['median_rmse'][-1]:.4f} for the two networks. "
            "These are descriptive results on a small selected panel.")
    else:lead=f"None of the {n} selected references is ready. This panel cannot yet support a confident MCMC recovery comparison. All cases and their statuses remain visible."
    pieces=['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Joint MCMC and neural parameter recovery</title><style>'+shared.stylesheet(csspath)+'</style></head><body><main>',
        '<header><p class="eyebrow">Same synthetic quotes · all six parameters inferred</p><h1>Does joint MCMC recover c better?</h1><p class="lead">'+esc(lead)+'</p><button onclick="window.print()">Print report</button></header>',
        '<nav><a href="#selection">Selection</a><a href="#scale">Scale recovery</a><a href="#parameters">All parameters</a><a href="#diagnostics">MCMC status</a><a href="#metrics">Descriptive metrics</a><a href="#sources">Sources</a></nav>']
    if data["summary"].get("interim",False):
        pieces.append('<div class="callout"><strong>Interim development snapshot: '+esc(ready)+'/'+esc(n)+' references ready.</strong><p>Snapshot time: '+esc(data["summary"].get("snapshot_utc","recorded in the manifest"))+'. The full selected panel remains fixed. Pending cases supply no MCMC posterior points or matched-subset metrics yet; results can change as references finish.</p></div>')
    pieces.append(section("Same observations, separate truth and inference",'''<div class="equation">Yᵢ=Pᵢ(φ)+c hᵢ Zᵢ, &nbsp; Zᵢ∼N(0,1)</div><p>The known half-spreads hᵢ set quote-specific scales; c is one unknown multiplier shared by the quotes in a dataset. The sampler receives the 54 noisy option quotes and infers all five physical Heston parameters and c jointly. Generating parameters, clean prices and noise realizations are not available to it. This is not a fixed-physical oracle. The two networks use their saved 500-pass posterior quantiles for exactly the same existing validation rows; no network is queried again.</p>
        <p>Cases were fixed before sampling by stable-sorting the 512 generating c values and choosing one midpoint rank in each of eight equal rank strata. Selection uses no neural error or MCMC result, and failed cases are not replaced. This selected panel is not a new SBC or population coverage study.</p>
        <p>Every vertical bar is a central 90% posterior credible interval, not a Monte Carlo standard-error bar. Finite quote noise and the prior permit uncertainty and shrinkage: even correct posterior medians need not lie exactly on the unit line for every dataset.</p>''',"selection"))
    pieces.append(section("Recovery of the spread multiplier",chart(csvg,
        "All selected neural cases and all ready joint-MCMC cases are shown. The prior median is 1.25 and its 90% width is 1.35. Unready MCMC cases appear in the status table and do not supply confident posterior points or matched-subset metrics.")+
        '<p><a href="c-recovery.png">Download PNG</a> · <a href="c-recovery.svg">Download SVG</a></p>',"scale"))
    pieces.append(section("Basic parameterwise recovery",chart(allsvg,
        "Each method uses the same generating truths and observed quotes. All available 90% intervals are plotted; narrow physical-parameter intervals can be smaller than their markers, and methods can overlap. Ready MCMC is red, batch 128 green, batch 512 purple.")+
        '<p><a href="parameter-recovery.png">Download PNG</a> · <a href="parameter-recovery.svg">Download SVG</a></p>',"parameters"))
    rows=[];crows=[]
    for i,(case,ref) in enumerate(zip(data["cases"],data["references"],strict=True)):
        d=ref["stats"];render=lambda x:"Unavailable" if x is None else fmt(x)
        rows.append([str(i+1),esc(case["row_id"]),fmt(case["truth"][-1]),"Ready" if ref["ready"] else "NOT READY",
            render(d["max_rhat"]),render(d["min_bulk_ess"]),render(d["min_tail_ess"]),esc(ref["reason"])])
        cells=[str(i+1),fmt(case["truth"][-1])]
        for method in METHODS:
            q=data["quantiles"][method][:,i,-1]
            cells.append("Not ready" if not np.isfinite(q).all() else esc("["+", ".join(f"{x:.4f}" for x in q)+"]"))
        crows.append(cells)
    pieces.append(section("Convergence and case status",'<p>Native PyMC DEMetropolisZ uses four chains. The declared first attempt has 2k tuning and 2k retained draws per chain, with one 6k/12k escalation only if chain diagnostics fail. Ready requires the full joint reference to pass its declared chain diagnostics (R̂≤1.01, bulk/tail ESS≥400) and sampled pricing-refinement gates. The table reports the worst R̂ and minimum ESS across recorded parameters and products. These checks are practical diagnostics, not proof that every posterior feature is exact.</p>'+table(
        ["Case","Validation row","Generating c","Reference status","Max R̂","Min bulk ESS","Min tail ESS","Reason"],rows)+
        '<h3>c intervals on the same cases</h3>'+table(["Case","Generating c","Joint MCMC q05/median/q95","Network 128 q05/median/q95","Network 512 q05/median/q95"],crows),"diagnostics"))
    metricrows=[]
    for method in (*METHODS,"prior"):
        if method not in stats:continue
        s=stats[method];corr=s["truth_median_correlation"][-1]
        metricrows.append(["Uniform prior baseline" if method=="prior" else esc(METHOD_LABELS[method]),str(s["cases"]),
            fmt(s["median_rmse"][-1]),"Undefined / constant" if corr is None else f"{corr:+.4f}",fmt(s["median_width90"][-1])])
    allrows=[]
    for j,name in enumerate(NAMES):
        for method in METHODS:
            if method not in stats:continue
            s=stats[method];corr=s["truth_median_correlation"][j]
            allrows.append([esc(LABELS[name]),esc(METHOD_LABELS[method]),str(s["cases"]),fmt(s["median_rmse"][j]),
                "Undefined" if corr is None else f"{corr:+.4f}",fmt(s["median_width90"][j])])
    pieces.append(section("Descriptive recovery on the matched ready subset",'<p>RMSE compares each posterior median with its generating value across the matched cases: √mean[(median−truth)²]. Interval width is q95−q05; the table summarizes its median across cases.</p>'+table(
        ["Method","Matched cases","c median RMSE","c truth–median correlation","Median c 90% width"],metricrows)+
        '<p>All methods in this table use only the same ready-MCMC subset. RMSE and correlation summarize these few selected datasets; they are not population guarantees, posterior calibration tests or a coverage estimate. A smaller credible interval alone is insufficient evidence of correct inference. The scientific setup and existing network checkpoints remain unchanged.</p>'+
        '<details><summary>Descriptive metrics for all six parameters</summary>'+table(
            ["Parameter","Method","Matched cases","Median RMSE","Truth–median correlation","Median 90% width"],allrows)+'</details>',"metrics"))
    source_rows=[[esc(s["label"]),'<code>'+esc(s["path"])+'</code>','<code>'+esc(s["sha256"])+'</code>']for s in inputs.sources]
    pieces.append(section("Hash-bound source evidence",'<p>The report reads compact case records and saved recovery quantiles only. It loads no bank, network weights or MCMC chains and performs no pricing, training or posterior sampling. Standalone scientific figures accompany this offline HTML.</p>'+table(["Input","Path","SHA-256"],source_rows),"sources"))
    artifacts={name:{"path":str((out.parent/name).resolve()),"sha256":shared.digest(out.parent/name)} for name in
        ("parameter-recovery.png","parameter-recovery.svg","c-recovery.png","c-recovery.svg")}
    inventory={"format":"heston_spread_mcmc_recovery_report_inventory_v1","sources":inputs.sources,
        "case_count":n,"ready_count":ready,"case_ids":[c["id"] for c in data["cases"]],
        "interim":data["summary"].get("interim",False),"snapshot_utc":data["summary"].get("snapshot_utc"),
        "ready_case_ids":[c["id"] for c,r in zip(data["cases"],data["ready"],strict=True) if r],
        "statistics":stats,"figures":artifacts,"scientific_acceptance":False,
        "scope":"Selected-case recovery only; not SBC, population coverage or release acceptance."}
    pieces.append('<script type="application/json" id="report-inventory">'+json.dumps(inventory).replace("<","\\u003c")+'</script></main></body></html>')
    out.write_text("\n".join(pieces));summary_out=out.with_name("plot-summary.json")
    summary_out.write_text(json.dumps({**inventory,"report":{"path":str(out.resolve()),"sha256":shared.digest(out)}},indent=2)+'\n')
    return {"out":str(out.resolve()),"sha256":shared.digest(out),"case_count":n,"ready_count":ready,"sources":len(inputs.sources),"figures":artifacts,"statistics":stats}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--summary",type=Path,required=True);parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.summary,args.out),indent=2))


if __name__=="__main__":main()
