"""Read-only joint-dependence review of the frozen checkpoint replay report.

This consumer uses persisted neural draws and existing MCMC chains only. It
does not load a neural estimator, price an option or sample a reference chain.
The frozen renderer's original report stays preserved under a separate identity.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import render_heston_spread_whitening_replay as frozen

shared = frozen.shared
import matplotlib.pyplot as plt

EPOCHS = frozen.EPOCHS
read, binding = frozen.read, frozen.binding
esc, fmt, table, section, chart = frozen.esc, frozen.fmt, frozen.table, frozen.section, frozen.chart

WORDING = {
    "The market observation is a model/prior-adequacy diagnostic.":
        "The market observation compares conditional inference under the same assumed model and likelihood; model adequacy requires separate predictive or residual checks.",
    "The market case additionally tests model/prior adequacy.":
        "For the market case, this comparison tests neural inference under the assumed model and likelihood; market-model adequacy is a separate question requiring predictive or residual checks.",
}


def strict_replace(text, old, new, expected=1):
    if text.count(old) != expected:
        raise ValueError("frozen report text changed; expected replacement count differs: "+old)
    return text.replace(old, new)


@contextmanager
def original_metadata_bindings(protocol):
    """Route only the six declared original JSON records through JSON checks.

    The frozen report erroneously calls an executable-source-only validator for
    these metadata files. This scoped adapter retains its strict code-source
    validator unchanged and restores the shared hook even if rendering fails.
    """
    allowed={str(Path(spec["path"]).resolve()):spec
        for spec in protocol["contract"]["original"]["files"].values()}
    if len(allowed)!=6 or any(Path(path).suffix!=".json" for path in allowed):
        raise ValueError("exactly the six original declared JSON records are required")
    original=frozen.selected.source_binding
    record={"mode":"scoped original JSON metadata compatibility adapter",
        "allowed_metadata":list(allowed.values()),"original_executable_source_checks_unchanged":True,
        "original_hook_restored":False}
    def bound(inputs,spec,label):
        path=Path(spec["path"])
        path=(path if path.is_absolute() else inputs.base/path).resolve()
        if str(path) not in allowed: return original(inputs,spec,label)
        expected=allowed[str(path)]
        if spec["sha256"]!=expected["sha256"]:
            raise ValueError("declared original metadata hash differs")
        value=inputs.json(spec,label)
        if not isinstance(value,(dict,list)):
            raise ValueError("original JSON metadata must contain a record or curve")
    frozen.selected.source_binding=bound
    try:
        yield record
    finally:
        frozen.selected.source_binding=original
        record["original_hook_restored"]=frozen.selected.source_binding is original


def correlation(values):
    """Pearson dependence of physical posterior targets, retaining all six."""
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim not in (2,3) or values.shape[-1] != 6 or values.reshape(-1,6).shape[0] < 3
            or not np.isfinite(values).all()):
        raise ValueError("finite posterior samples with all six targets required")
    flat = values.reshape(-1,6)
    if np.any(flat.std(axis=0) <= 0):
        raise ValueError("posterior correlation is undefined for constant targets")
    result = np.corrcoef(flat, rowvar=False)
    if result.shape != (6,6) or not np.isfinite(result).all():
        raise ValueError("finite six-target posterior correlation required")
    return result


def dependence(neural, reference):
    a, b = correlation(neural), correlation(reference)
    difference = a-b
    unique = np.abs(difference[np.triu_indices(6,k=1)])
    return {"neural_correlation":a.tolist(),"reference_correlation":b.tolist(),
        "signed_correlation_difference":difference.tolist(),
        "unique_parameter_pairs":15,"max_absolute_correlation_difference":float(unique.max()),
        "median_absolute_correlation_difference":float(np.median(unique)),
        "neural_draws":int(np.asarray(neural).reshape(-1,6).shape[0]),
        "reference_draws":int(np.asarray(reference).reshape(-1,6).shape[0]),
        "reference_chain_shape":list(np.asarray(reference).shape),
        "scope":"Descriptive Pearson correlation only; no Monte Carlo acceptance threshold, nonlinear-dependence or full-joint equality claim."}


def checked_payload(spec, protected):
    path=Path(spec["path"]).resolve()
    if shared.digest(path) != spec["sha256"]: raise ValueError("saved posterior payload differs")
    protected[str(path)]=spec["sha256"]
    return path


def load_dependence(protocol, protected):
    """Verify saved neural/reference bindings before any summary is computed."""
    paths={spec["epoch"]:Path(spec["output"])/"evaluation.json" for spec in protocol["evaluations"]}
    paths[200]=checked_payload(protocol["reused_final_evaluation"],protected)
    result={}; evaluations={}; reference_bindings={}
    for epoch in EPOCHS:
        path=paths[epoch].resolve();protected[str(path)]=shared.digest(path);evaluation=read(path)
        if (evaluation["status"] != "complete" or evaluation["actual_optimizer_updates"] != epoch*40
                or evaluation["task"] != protocol["task"]["target"]
                or evaluation["reserved_arrays_opened"] is not False
                or evaluation["new_reference_sampling"] is not False):
            raise ValueError("saved conditional evaluation target, endpoint or data boundary differs")
        spec=evaluation["conditional_draws"]
        payload=checked_payload(spec,protected)
        if spec["draws_per_case"] != 4096: raise ValueError("same conditional posterior draw count required")
        with np.load(payload,allow_pickle=False) as archive:
            neural={case:archive[case].copy() for case in shared.CASES}
        evaluations[epoch]=evaluation
        for case in shared.CASES:
            entry=evaluation["cases"][case]
            if entry.get("reference_ready") is not True:
                raise ValueError("an existing ready conditional reference is required")
            ref_spec={"path":entry["reference_chain_payload_path"],"sha256":entry["reference_chain_payload_sha256"]}
            identity=(entry["observation_sha256"],ref_spec["path"],ref_spec["sha256"])
            if case in reference_bindings and reference_bindings[case] != identity:
                raise ValueError("conditional observations or joint reference changed between epochs")
            reference_bindings[case]=identity
            for prefix in ("reference_run","reference_diagnostics"):
                checked_payload({"path":entry[prefix+"_path"],"sha256":entry[prefix+"_sha256"]},protected)
            reference_path=checked_payload(ref_spec,protected)
            with np.load(reference_path,allow_pickle=False) as archive: chains=archive["parameters"].copy()
            values=neural[case]
            bounds=np.asarray(evaluation["task"]["prior_bounds"])
            if (values.shape != (4096,6) or chains.ndim != 3 or chains.shape[-1] != 6
                    or np.any(values<bounds[:,0]) or np.any(values>bounds[:,1])
                    or np.any(chains<bounds[:,0]) or np.any(chains>bounds[:,1])):
                raise ValueError("saved joint posterior dimensions or bounded support differ")
            q=np.quantile(values,[.05,.5,.95],axis=0)
            if not np.array_equal(q,np.asarray(entry["neural_quantiles_05_50_95"])):
                raise ValueError("saved conditional draws do not reproduce evaluation quantiles")
            reference_q=np.quantile(chains.reshape(-1,6),[.05,.5,.95],axis=0)
            expected=np.asarray(entry["comparison"]["reference_quantiles_05_50_95"])[:,:6]
            if not np.array_equal(reference_q,expected):
                raise ValueError("saved reference chains do not reproduce conditional quantiles")
            result.setdefault(case,{})[str(epoch)]=dependence(values,chains)
    return result,evaluations


def first_screen(evaluations):
    def values(epoch):
        p=evaluations[epoch]["population"]
        coverage=p["coverage"]["0.9"]["coverage"]
        return coverage,p["recovery"],sum(p["rank_dkw"]["passed"])
    early,em,passed=values(83);late,lm,_=values(200)
    return (f"At the previously observed validation minimum, epoch 83, nominal 90% intervals cover "
        f"{100*min(early[:5]):.1f}–{100*max(early[:5]):.1f}% of physical-parameter truths and {100*early[-1]:.1f}% of c truths. "
        f"At epoch 200 those figures are {100*min(late[:5]):.1f}–{100*max(late[:5]):.1f}% and {100*late[-1]:.1f}%. "
        f"The c median RMSE changes from {lm['median_rmse'][-1]:.4f} at epoch 200 to {em['median_rmse'][-1]:.4f} at epoch 83; "
        f"{passed}/6 marginal SBC screens are within their band at epoch 83. "
        "This is a diagnosis on reused development data, rather than independent acceptance of a selected estimator.")


def dependence_figures(records,base):
    result={}
    vlim=max(.1,max(abs(np.asarray(value["signed_correlation_difference"])).max()
        for case in records.values() for value in case.values()))
    labels=[frozen.LABELS[n] for n in frozen.NAMES]
    for case in shared.CASES:
        fig,axes=plt.subplots(2,3,figsize=(11.6,7.5),layout="constrained")
        for axis,epoch in zip(axes.flat,EPOCHS,strict=False):
            value=records[case][str(epoch)];matrix=np.asarray(value["signed_correlation_difference"])
            picture=axis.imshow(matrix,vmin=-vlim,vmax=vlim,cmap="RdBu_r")
            axis.set(title=f"Epoch {epoch}\nmax |Δ|={value['max_absolute_correlation_difference']:.3f}",
                xticks=range(6),xticklabels=labels,yticks=range(6),yticklabels=labels)
            axis.tick_params(axis="x",rotation=90)
        axes.flat[-1].axis("off")
        fig.colorbar(picture,ax=list(axes.flat),shrink=.75,label="Neural correlation − MCMC correlation")
        name="dependence-"+case
        result[name]=frozen.save(fig,base,name,shared.CASE_LABELS[case]+": posterior dependence",
            "Pearson correlation differences for all six jointly inferred targets; identical references across the five epochs.")
    return result


def build(protocol_path,raw_out,out):
    protocol_path,raw_out,out=map(lambda p:Path(p).resolve(),(protocol_path,raw_out,out))
    summary_path=out.with_name("review-summary.json")
    if raw_out.exists() or out.exists() or summary_path.exists():
        raise FileExistsError("raw and reviewed report identities must be new")
    protocol=read(protocol_path)
    if protocol["sources"]["renderer"] != binding(frozen.__file__):
        raise ValueError("original renderer differs from the pre-replay frozen source")
    protected={str(protocol_path):shared.digest(protocol_path),str(Path(__file__).resolve()):shared.digest(__file__)}
    with original_metadata_bindings(protocol) as compatibility:
        frozen.build(protocol_path,raw_out)
    original_html=raw_out.read_text();raw_summary=read(raw_out.with_name("plot-summary.json"))
    protected[str(raw_out)]=shared.digest(raw_out)
    protected[str(raw_out.with_name("plot-summary.json"))]=shared.digest(raw_out.with_name("plot-summary.json"))
    records,evaluations=load_dependence(protocol,protected)
    out.parent.mkdir(parents=True,exist_ok=True)
    exported={}
    for name,spec in raw_summary["figures"].items():
        source=checked_payload(spec,protected);target=out.parent/name
        if target.exists(): raise FileExistsError("review figure identity already exists: "+name)
        shutil.copyfile(source,target);exported[name]={"source":binding(source),"copy":binding(target)}
    compact_summary=out.with_name("plot-summary.json")
    if compact_summary.exists(): raise FileExistsError("compact plot summary identity exists")
    shutil.copyfile(raw_out.with_name("plot-summary.json"),compact_summary)
    with plt.rc_context({"svg.fonttype":"none","svg.hashsalt":"heston-zca-checkpoint-dependence-review-v1","font.family":"DejaVu Sans","font.size":8}):
        figures=dependence_figures(records,out.parent)
    body='<p>These checks use already saved neural posterior draws and existing ready joint MCMC chains. Every matrix retains all six targets. The table summarizes absolute differences across the 15 distinct off-diagonal Pearson correlations. Values near zero mean closer agreement for this linear-dependence summary. Pearson correlations do not characterize every nonlinear dependence or establish equality of full joint posteriors. MCMC draws are serially dependent; these summaries have no Monte Carlo acceptance threshold.</p>'
    for case in shared.CASES:
        rows=[]
        for epoch in EPOCHS:
            value=records[case][str(epoch)]
            rows.append([str(epoch),fmt(value["max_absolute_correlation_difference"]),fmt(value["median_absolute_correlation_difference"]),str(value["neural_draws"]),str(value["reference_draws"])])
        body+='<h3>'+esc(shared.CASE_LABELS[case])+'</h3>'+table(["Epoch","Largest |correlation difference|","Median |correlation difference|","Neural draws","Saved MCMC draws"],rows)+chart(figures["dependence-"+case],"Signed Pearson-correlation differences; the reference is unchanged at every epoch.")
    dependency_section=section("Does the joint parameter dependence agree?",body,"dependence")
    html=original_html
    for old,new in WORDING.items(): html=strict_replace(html,old,new)
    old_lead='<p class="lead">Earlier checkpoints are compared with the same final epoch-200 model, on identical observations and references. This isolates checkpoint choice from changes to data, whitening, architecture and learning-rate schedule.</p>'
    html=strict_replace(html,old_lead,'<p class="lead">'+esc(first_screen(evaluations))+'</p>')
    html=strict_replace(html,'<a href="#sources">Sources</a>','<a href="#dependence">Dependence</a><a href="#sources">Sources</a>')
    marker='<section id="sources">'
    # Shared section markup includes a CSS class; find the exact source section.
    if marker not in html: marker='<section id="sources" class="section">'
    if marker not in html:
        import re
        found=re.findall(r'<section\b[^>]*\bid="sources"[^>]*>',html)
        if len(found)!=1: raise ValueError("the frozen source section is not uniquely identifiable")
        marker=found[0]
    html=strict_replace(html,marker,dependency_section+marker)
    review={"format":"heston_spread_whitening_checkpoint_replay_review_v1","protocol":binding(protocol_path),
        "producer":binding(__file__),"frozen_renderer":binding(frozen.__file__),"raw_report":binding(raw_out),
        "raw_plot_summary":binding(raw_out.with_name("plot-summary.json")),"compact_plot_summary":binding(compact_summary),
        "exported_figures":exported,"dependence":records,
        "dependence_figures":{name+suffix:binding(out.parent/(name+suffix)) for name in figures for suffix in (".png",".svg")},
        "interpretation":first_screen(evaluations),"text_corrections":WORDING,
        "frozen_metadata_compatibility":compatibility,
        "protected_inputs":protected,"new_neural_queries":False,"new_pricing":False,"new_reference_sampling":False,
        "scientific_acceptance":False,"scope":"Offline posthoc dependence review; saved development observations and references only."}
    addition='<script type="application/json" id="review-inventory">'+json.dumps(review).replace("<","\\u003c")+'</script>'
    html=strict_replace(html,'</main></body></html>',addition+'</main></body></html>')
    if any(shared.digest(path)!=digest for path,digest in protected.items()):
        raise ValueError("a protected source, posterior payload or historical report changed")
    out.write_text(html);review["report"]=binding(out)
    summary_path.write_text(json.dumps(review,indent=2)+"\n")
    return review


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("protocol","raw-out","out"):parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args();print(build(args.protocol,args.raw_out,args.out)["report"])
