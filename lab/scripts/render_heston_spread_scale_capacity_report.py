"""Offline presentation of the larger-bank, batch-size continuation study.

Read only hash-bound compact JSON, sources, and recovery quantiles. Never load
a model, pricing bank, chain archive or posterior payload. The input manifest is
``heston_spread_scale_capacity_report_v1`` with protocol, baseline, arms, sources;
every artifact spec has path and sha256. Arm entries declare batch_size, run,
training, curve and evaluations [{epochs, evaluation, recovery}].
"""
import argparse
import ast
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import render_heston_spread_scale_report as shared
import matplotlib.pyplot as plt


EPOCHS = (25, 100, 250, 500)
COLORS = {128: "#287b6a", 512: "#654a97", "baseline": "#7e8790"}
NAMES, LABELS, CASES = shared.NAMES, shared.LABELS, shared.CASES
esc, fmt, table, section, chart = shared.esc, shared.fmt, shared.table, shared.section, shared.chart


def scale_metrics(truth, q):
    actual, prediction = truth[:, -1], q[1, :, -1]
    return {"rmse": float(np.sqrt(np.mean((prediction-actual)**2))),
            "prior_rmse": float(np.sqrt(np.mean((1.25-actual)**2))),
            "correlation": None if np.ptp(prediction) == 0 else float(np.corrcoef(actual, prediction)[0, 1]),
            "median_width90": float(np.median(q[2, :, -1]-q[0, :, -1])),
            "prior_width90": 1.35}


def code_excerpt(path, name):
    source = Path(path).read_text()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return ast.get_source_segment(source, node)
    raise ValueError(f"requested source definition {name} absent")


def resolved_spec(spec, base):
    path = Path(spec["path"])
    return (path if path.is_absolute() else Path(base)/path).resolve()


def implied_rate(passes, contract):
    """Next-step rate after complete passes, derived from the saved schedule."""
    schedule = contract["schedule"]
    if (schedule["class_name"] != "SpreadExposureCosine"
            or schedule["budget_axis"] != "exact cumulative additional presentations"):
        raise ValueError("unsupported recorded exposure schedule")
    maximum, minimum = float(schedule["maximum"]), float(schedule["minimum"])
    decay = int(schedule["decay_bank_passes"])
    if not 0 < minimum <= maximum or decay <= 0:
        raise ValueError("invalid recorded cosine rates")
    fraction = min(max(float(passes)/decay, 0.), 1.)
    return minimum+(maximum-minimum)*.5*(1.+np.cos(np.pi*fraction))


def check_counts(evaluation, cases):
    counts = {"pass":0, "fail":0, "inconclusive":0}
    for case in cases:
        comparison = evaluation["cases"].get(case, {}).get("comparison") or {}
        for record in comparison.get("parameters", {}).values():
            for key in ("mean_status", "width_status"):
                status = record[key]
                if status not in counts:
                    raise ValueError("unsupported conditional-check status")
                counts[status] += 1
    return counts


def count_text(counts):
    return f'{counts["pass"]} pass / {counts["fail"]} fail / {counts["inconclusive"]} inconclusive'


def recovery_plot(records, bounds, title, color="#654a97"):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7), layout="constrained")
    truth, q = records
    chosen = np.arange(0, 512, 16)
    for i, axis in enumerate(axes.flat):
        lo, hi = bounds[i]
        axis.scatter(truth[:, i], q[1, :, i], s=9, color="#617784", alpha=.4, edgecolors="none")
        axis.errorbar(truth[chosen, i], q[1, chosen, i],
            yerr=np.stack([q[1, chosen, i]-q[0, chosen, i], q[2, chosen, i]-q[1, chosen, i]]),
            fmt="o", ms=2.5, color=color, capsize=2, elinewidth=.75)
        axis.plot([lo, hi], [lo, hi], "--", color="#334e5c", lw=1)
        margin = .025*(hi-lo)
        axis.set(title=LABELS[NAMES[i]], xlabel="Generating value", ylabel="Posterior median",
                 xlim=(lo-margin, hi+margin), ylim=(lo-margin, hi+margin), aspect="equal")
    return chart(shared._svg(fig, title, "All 512 medians; marginal 90% intervals for 32 fixed index-selected cases."),
        "All 512 validation medians and 90% marginal posterior intervals for 32 cases selected by row index 0,16,…,496. Bars are posterior uncertainty, not Monte Carlo error. Unit-line alignment alone does not establish local posterior accuracy.")


def sbc_plot(final_records):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.3), layout="constrained", sharex=True)
    grid = np.linspace(0, 1, 201)
    rows = []
    for label, color, evaluation in final_records:
        p = evaluation["population"]
        _, ranks = shared.population_records({0: evaluation})[0]
        dkw = p["rank_dkw"]
        rows.append([esc(label), f'{sum(dkw["passed"])}/6', fmt(dkw["dkw_bound"]),
                     esc(", ".join(f'{name}: {p["coverage"]["0.9"]["coverage"][i]:.3f}' for i, name in enumerate(NAMES)))])
        for i, axis in enumerate(axes.flat):
            difference = np.searchsorted(np.sort(ranks[:, i]), grid, side="right")/len(ranks)-grid
            axis.plot(grid, difference, color=color, label=label, lw=1.3)
    bands = [float(r[2]["population"]["rank_dkw"]["dkw_bound"]) for r in final_records]
    if not np.allclose(bands, bands[0], rtol=0, atol=1e-14):
        raise ValueError("different marginal SBC bands require separate display")
    for i, axis in enumerate(axes.flat):
        axis.axhspan(-bands[0], bands[0], color="#dbece1", zorder=0)
        axis.axhline(0, color="#6d7d87", lw=.7)
        axis.set(title=LABELS[NAMES[i]], xlabel="Fractional posterior rank", ylabel="ECDF − uniform CDF", xlim=(0, 1))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(final_records), frameon=False)
    return table(["Estimator", "Marginal DKW screens", "Band", "90% coverage"], rows)+chart(
        shared._svg(fig, "SBC at the baseline and matched equal-exposure endpoints", "Six marginal rank ECDF deviations on identical 512 validation observations."),
        "These are the same 512 development observations at every endpoint, with c independently drawn from Uniform(.5,2). The green band adjusts across six parameters within each checkpoint; it is not simultaneous control across all fits/endpoints. Returning the prior can pass marginal SBC while ignoring information about c.")


def conditional_plot(final_records):
    fig, axes = plt.subplots(3, 1, figsize=(10.8, 8), layout="constrained")
    for axis, case in zip(axes, CASES, strict=True):
        neural = []
        reference = None
        for label, color, evaluation in final_records:
            entry = evaluation["cases"].get(case, {})
            values = entry.get("neural_quantiles_05_50_95")
            if values is not None:
                q = np.asarray(values)[:, -1]
                if q.shape != (3,) or not np.isfinite(q).all() or np.any(np.diff(q)<0):
                    raise ValueError("invalid neural conditional c quantiles")
                neural.append((label, color, q))
            candidate = entry.get("reference_c_marginal", {}).get("quantiles_05_50_95")
            if candidate is not None:
                if reference is not None and not np.array_equal(reference, candidate):
                    raise ValueError("conditional c reference changes across estimators")
                reference = np.asarray(candidate)
        values = [("Uniform prior", "#b1b8bd", np.array([.575, 1.25, 1.925]))]
        if reference is not None:
            values.append(("Joint MCMC reference", "#b44b37", reference))
        values += neural
        for y, (_, color, q) in enumerate(values):
            axis.errorbar(q[1], y, xerr=np.array([[q[1]-q[0]], [q[2]-q[1]]]), color=color, fmt="o", capsize=3)
        axis.set(yticks=np.arange(len(values)), yticklabels=[v[0] for v in values],
                 title=shared.CASE_LABELS[case], xlabel="c: q05, median, q95", xlim=(.48, 2.02))
        axis.invert_yaxis()
    return chart(shared._svg(fig, "Conditional c posteriors against joint MCMC", "Three fixed observation vectors; central 90% prior, joint-reference and neural intervals."),
        "Conditional means one fixed 54-quote observation vector at a time, with all six parameters inferred jointly. These are central 90% posterior intervals. The actual market reference presses against the upper c bound; that is model/prior tension, not a known true market-noise scale. The fixed-physical oracle is not used here.")


def conditional_table(evaluation):
    result = ""
    for case in CASES:
        entry = evaluation["cases"].get(case, {})
        comparison = entry.get("comparison") or {}
        rows = []
        for name, item in comparison.get("parameters", {}).items():
            rows.append([esc(LABELS.get(name, name)),
                fmt(item["absolute_standardized_mean_difference"]),
                esc(str([round(x, 4) for x in item["signed_standardized_mean_interval"]])),
                shared.badge(item["mean_status"]), fmt(item["width_ratio_90"]),
                esc(str([round(x, 4) for x in item["width_ratio_interval"]])), shared.badge(item["width_status"])])
        result += '<h4>'+esc(shared.CASE_LABELS[case])+" · "+shared.badge(entry.get("status", "unavailable"))+"</h4>"
        result += table(["Quantity", "|Mean error| / ref SD", "Signed-error MC interval", "Mean", "90% width / ref", "Width MC interval", "Width"], rows)
    return result


def precision_history(audit, protocol, summary, inputs, target, baseline):
    if (audit.get("format") != "heston_spread_scale_precision_failure_audit_v1"
            or audit.get("mathematical_target_unchanged") is not True
            or audit.get("clipping_or_resampling") is not False
            or audit.get("same_offending_raw_quote_vector_and_all_six_targets") is not True):
        raise ValueError("precision history does not establish the declared numerical repair")
    original = inputs.json(audit["original_protocol"], "Original capacity-study protocol")
    campaign = inputs.json(audit["original_campaign"], "Original failed paired campaign")
    if original["joint_task"] != target or protocol["joint_task"] != target:
        raise ValueError("precision replay changed the physical posterior target")
    if protocol["parent_checkpoint"] != original["parent_checkpoint"] or protocol["bank"] != original["bank"]:
        raise ValueError("precision replay changed parent checkpoint or clean bank")
    unchanged = ("train_rows", "batch_size", "batches_per_pass", "tail_batch_size", "additional_bank_passes",
        "checkpoint_bank_passes", "additional_optimizer_updates", "additional_presentations", "seed",
        "augmentation_namespace", "permutation_namespace", "schedule", "parent_model_sha256", "parent_state_sha256")
    for arm in protocol["arms"]:
        previous = next(x for x in original["arms"] if x["batch_size"] == arm["batch_size"])
        if any(previous["contract"][key] != arm["contract"][key] for key in unchanged):
            raise ValueError("precision replay changed a frozen scientific or paired-training setting")
    rows = []
    raw_values = []
    for arm in audit["arms"]:
        bs = arm["batch_size"]
        if not arm["all_saved_state_arrays_finite"] or arm["gradients_for_bad_batch_available"]:
            raise ValueError("failure history does not establish a preprocessing-only stop")
        compact = {name:inputs.json(arm["artifacts"][name], f"Original batch {bs} {name}")
            for name in ("run", "progress", "failure")}
        if (compact["run"]["task"]["target"] != target
                or compact["failure"]["completed_finite_updates"] != arm["completed_finite_updates"]
                or campaign["arms"][str(bs)]["status"] != "failed"):
            raise ValueError("original failure compact records disagree")
        raw_values.append(arm["raw_c"])
        rows.append([str(bs), f'{arm["complete_passes"]} complete + {arm["finite_batches_in_next_pass"]} batches',
            f'{arm["completed_finite_updates"]:,}', "25 passes", "All saved state arrays finite", "Before optimizer update"])
    if len(rows) != 2 or raw_values[0] != raw_values[1]:
        raise ValueError("precision audit does not cover the same offending draw in both arms")
    example = audit["arms"][0]
    body = '<p>The original pair completed 80 whole bank passes, then stopped in pass 81 on the same valid simulated example. Both saved their 25-pass checkpoint; neither reached the planned 100-pass checkpoint. The stop occurred while preparing targets, before computing a gradient for the offending batch. Saved network and optimizer arrays remained finite.</p>'
    body += table(["Original batch", "Finite training completed", "Absolute optimizer updates", "Last scheduled checkpoint", "State", "Failure stage"], rows)
    body += '<div class="equation">log((x−l)/(u−l)) − log(1−(x−l)/(u−l)) = log(x−l) − log(u−x)</div>'
    body += '<p>For c = '+esc(repr(example["raw_c"]))+', the normalized value was '+esc(repr(example["normalized_c_float64"]))+' in float64. The earlier logarithm path rounded it to '+esc(repr(example["normalized_c_float32"]))+' in float32 and produced an infinite logit. Evaluating the algebraically identical right-hand expression in float64 gives '+esc(fmt(example["stable_logit_float64"]))+'; the finite result is then converted to float32.</p>'
    body += '<p>The repair applies to all six bounded targets. It clips no values and redraws no examples. It preserves the priors, likelihood, original posterior inverse map, saved parent state, bank, raw c/noise streams, batch sizes and exposure schedule. Finite roundoff also changes, so the corrected pair replays from the same original 60k parent under new identities rather than claiming the old training trajectory is bitwise unchanged.</p>'
    body += '<p>The original failure artifacts and compute time remain recorded. The corrected campaign retains the original absolute deadline: <strong>'+esc(audit["original_absolute_deadline_utc"])+"</strong>. The shared preprocessing failure is distinct from the earlier serialization-only evaluation-loader startup issue.</p>"
    old_metrics = []
    for arm in summary.get("original_arms", []):
        bs = arm["batch_size"]
        evaluation = inputs.json(arm["evaluation"], f"Original batch {bs} partial 25-pass evaluation")
        if (evaluation.get("format") != "heston_spread_scale_capacity_evaluation_v1"
                or evaluation["batch_size"] != bs or evaluation["completed_bank_passes"] != 25
                or evaluation["study_protocol"]["sha256"] != audit["original_protocol"]["sha256"]
                or evaluation["task"] != target
                or evaluation["population"]["truth_sha256"] != baseline["population"]["truth_sha256"]
                or evaluation["population"]["observations_sha256"] != baseline["population"]["observations_sha256"]):
            raise ValueError("original partial result changed task, observations or endpoint")
        recovery = shared.recovery_data(arm["recovery"], inputs, evaluation, 25)
        old_metrics.append({"batch_size":bs, **scale_metrics(*recovery)})
    if old_metrics:
        if sorted(x["batch_size"] for x in old_metrics) != [128,512]:
            raise ValueError("original partial table must show both paired arms")
        body += '<h3>Retained original 25-pass results, before the numerical repair</h3>'
        body += table(["Original batch", "c median RMSE", "Prior-median RMSE", "Truth–median correlation", "Median c 90% width"],
            [[str(x["batch_size"]),fmt(x["rmse"]),fmt(x["prior_rmse"]),fmt(x["correlation"]),fmt(x["median_width90"])]for x in old_metrics])
        body += '<p>These are partial results from the failed original pair, outside the corrected replay comparison family. They are not 500-pass results and are not pooled with the corrected endpoints below.</p>'
    return body, old_metrics


def input_conditioning(summary, inputs, arms, target, protocol, precision_recovery):
    """Read a source-bound, training-only post hoc linear-geometry diagnostic."""
    names = ("input_conditioning", "input_conditioning_source")
    if not any(name in summary for name in names):
        return None, None
    if not all(name in summary for name in names):
        raise ValueError("input-conditioning evidence requires its exact producer copy")
    diagnostic = inputs.json(summary["input_conditioning"], "Exploratory training-input covariance diagnostic")
    source = inputs.path(summary["input_conditioning_source"], "Exact exploratory producer source copy")
    access, binding, method = diagnostic["data_access"], diagnostic["bindings"], diagnostic["method"]
    expected_protocol = precision_recovery["original_protocol"] if precision_recovery is not None else summary["protocol"]
    if (diagnostic.get("format") != "heston_input_conditioning_exploratory_v1"
            or shared.digest(source) != diagnostic["source"]["sha256"]
            or binding["protocol_sha256"] != expected_protocol["sha256"]
            or resolved_spec({"path":binding["protocol_path"]}, inputs.base) != resolved_spec(expected_protocol, inputs.base)
            or binding["parent_checkpoint_sha256"] != protocol["parent_checkpoint"]["record"]["sha256"]
            or binding["bank_record_sha256"] != protocol["bank"]["record"]["sha256"]
            or binding["bank_payload_sha256"] != protocol["bank"]["payload"]["sha256"]
            or method["c_prior"] != target["prior_bounds"][-1]
            or access != {"named_arrays_opened":["train_clean"], "train_rows":100000,
                "validation_opened":False, "reserved_opened":False, "neural_model_loaded":False,
                "posterior_queries":0, "fitting":False}):
        raise ValueError("exploratory input diagnostic source/task/prior/access binding differs")
    for arm in arms:
        if (binding["task_fingerprint"] != arm["run"]["task_fingerprint"]
                or binding["bank_payload_sha256"] != arm["run"]["bank_payload_sha256"]):
            raise ValueError("input diagnostic differs from the fitted task or bank")
    lo, hi = method["c_prior"]
    if not np.isclose(method["expected_c_squared"], (lo*lo+lo*hi+hi*hi)/3., rtol=0, atol=1e-14):
        raise ValueError("input diagnostic uses a different prior-expected noise variance")
    pcs = diagnostic["principal_components"]
    clean = np.array([p["clean_variance"] for p in pcs], dtype=float)
    noise = np.array([p["expected_noise_variance"] for p in pcs], dtype=float)
    cast_ratio = np.array([p["clean_float32_cast_error_sd_over_expected_noise_sd"] for p in pcs], dtype=float)
    if (len(pcs) != 54 or [p["pc"] for p in pcs] != list(range(1,55))
            or not np.isfinite(np.r_[clean,noise,cast_ratio]).all()
            or np.any(clean <= 0) or np.any(noise <= 0) or np.any(cast_ratio < 0)
            or np.any(np.diff(clean)>0)):
        raise ValueError("input diagnostic requires 54 ordered finite clean-PC projections")
    values = diagnostic["summary"]
    checks = ((clean[:3].sum()/clean.sum(), values["top3_clean_variance_fraction"]),
        (noise.sum()/clean.sum(), values["expected_noise_trace_over_clean_trace"]),
        (cast_ratio.max(), values["pc_rounding_sd_over_noise_sd_min_median_max"][-1]),
        (values["expected_noisy_covariance_max_eigenvalue"]/values["expected_noisy_covariance_min_eigenvalue"],
            values["expected_noisy_covariance_condition_number"]))
    if (any(not np.isclose(a,b,rtol=1e-10,atol=1e-14) for a,b in checks)
            or int(np.sum(noise>clean)) != values["noise_dominant_clean_PCs"]):
        raise ValueError("exploratory covariance summary disagrees with its compact projections")
    rows = [["Clean variance in the first 3 principal components", f'{100*values["top3_clean_variance_fraction"]:.2f}%'],
        ["Condition number: clean covariance + expected quote-noise covariance", f'{values["expected_noisy_covariance_condition_number"]:.3g}'],
        ["Clean-PC directions where expected quote noise exceeds clean variation", f'{values["noise_dominant_clean_PCs"]}/54'],
        ["Expected noise trace / clean trace", f'{values["expected_noise_trace_over_clean_trace"]:.3g}'],
        ["Largest across-PC clean-cast error SD / expected noise SD", f'{cast_ratio.max():.3g}']]
    body = '<p>This post hoc diagnostic uses only the 100k clean training surfaces. It measures linear covariance geometry across the training prior after dividing prices by half-spreads and standardizing each feature with training-only moments. Expected quote noise uses the unchanged c prior, with E[c²]=1.75. It does not measure local parameter identifiability or actual network activations.</p>'
    body += table(["Exploratory quantity", "Measured value"], rows)
    with plt.rc_context({"svg.fonttype":"none", "svg.hashsalt":"heston-scale-capacity-v1", "font.family":"DejaVu Sans", "font.size":9}):
        fig, axis = plt.subplots(figsize=(10.8,4),layout="constrained")
        axis.semilogy(np.arange(1,55),clean,color="#487f9f",label="Clean covariance eigenvalue")
        axis.semilogy(np.arange(1,55),noise,color="#b97725",label="Expected quote-noise variance in that clean-PC direction")
        axis.set(xlabel="Clean principal component, largest clean variance first",ylabel="Variance in standardized inputs",xlim=(1,54))
        axis.legend(fontsize=8)
        body += chart(shared._svg(fig,"Exploratory training covariance and projected quote noise",
            "Global training-prior clean covariance eigenvalues and projected expected quote-noise variances, from compact diagnostic JSON only."),
            "The noise curve shows projections into clean principal components. Those noise projections are generally correlated, so this is not a diagonalized likelihood or the eigenvalue spectrum of the total covariance.")
    body += '<p>A full-rank whitening using clean covariance plus the specified expected noise is an untested conditioning candidate. It would retain all 54 coordinates through an invertible linear transform. Clean-only whitening would amplify an almost singular covariance (condition number '+esc(f'{values["clean_covariance_condition_number"]:.3g}')+'). No whitening was fitted or applied, and no additional network was trained for this diagnostic.</p>'
    body += '<p>The cast-error figure covers only conversion of clean input prices to float32. It does not audit normalization or deep-network arithmetic. These descriptive measurements suggest a possible optimization direction; they do not establish the cause of failing to learn c or that whitening would improve inference. The reported posterior results and comparison criteria remain unchanged.</p>'
    return body, {"diagnostic":summary["input_conditioning"],"source_copy":summary["input_conditioning_source"],
        "scope":diagnostic["scope"],"training_only":True,"whitening_applied":False,"new_fit":False}


def build(summary_path, out):
    summary_path, out = Path(summary_path).resolve(), Path(out)
    summary = json.loads(summary_path.read_text())
    if summary.get("format") != "heston_spread_scale_capacity_report_v1":
        raise ValueError("unsupported capacity report manifest")
    fixture_only = summary.get("fixture_only", False)
    if type(fixture_only) is not bool:
        raise ValueError("fixture marker must be boolean")
    interim = summary.get("interim", False)
    if type(interim) is not bool:
        raise ValueError("interim marker must be boolean")
    inputs = shared.Artifacts(summary_path)
    inputs.sources.append({"label":"Report manifest", "path":str(summary_path), "sha256":shared.digest(summary_path)})
    protocol = inputs.json(summary["protocol"], "Predeclared capacity-study protocol")
    if protocol.get("format") not in ("heston_spread_scale_capacity_study_v1", "heston_spread_scale_precision_recovery_study_v1"):
        raise ValueError("capacity report requires the predeclared scientific study")
    precision_replay = protocol["format"] == "heston_spread_scale_precision_recovery_study_v1"
    precision_recovery = None
    if precision_replay:
        if (summary.get("recovery", {}).get("sha256") != protocol.get("precision_recovery", {}).get("sha256")
                or resolved_spec(summary["recovery"], inputs.base) != resolved_spec(protocol["precision_recovery"], inputs.base)):
            raise ValueError("precision replay requires the exact recorded failure/recovery audit")
        precision_recovery = inputs.json(summary["recovery"], "Original numerical failure and precision-replay audit")
    baseline = inputs.json(summary["baseline"]["evaluation"], "Original 60k evaluation")
    baseline_recovery = shared.recovery_data(summary["baseline"]["recovery"], inputs, baseline, 60000)
    target = baseline["task"]
    if protocol["joint_task"] != target or baseline.get("actual_optimizer_updates") != 60000:
        raise ValueError("study or original baseline target differs")
    history_body, original_metrics = (precision_history(precision_recovery, protocol, summary, inputs, target, baseline)
        if precision_replay else (None, []))
    arms = []
    for arm in summary["arms"]:
        bs = arm["batch_size"]
        if bs not in (128, 512) or any(a["batch_size"]==bs for a in arms):
            raise ValueError("capacity report requires unique128/512 arms")
        run = inputs.json(arm["run"], f"Batch {bs} run")
        training = inputs.json(arm["training"], f"Batch {bs} completion")
        curve = inputs.json(arm["curve"], f"Batch {bs} learning curve")
        if (run["task"]["target"] != target or run["batch_size"] != bs or run["train_rows"] != 100000
                or run.get("reserved_arrays_opened") is not False):
            raise ValueError("arm differs in frozen task,100k bank, batch or reserved-data scope")
        dropout = run.get("initialization_control", {})
        if (run.get("architecture", {}).get("effective_dropout") != 0.
                or len(dropout.get("rates_after", [])) != 24
                or any(rate != 0. for rate in dropout["rates_after"])
                or dropout.get("array_state_unchanged") is not True):
            raise ValueError("report requires the inherited24-layer effective-zero dropout control")
        contract = run["continuation_protocol"]
        if (contract["train_rows"] != 100000 or contract["batch_size"] != bs
                or contract["additional_bank_passes"] != 500 or contract["max_wall_seconds"] != 5400):
            raise ValueError("current continuation recipe or wall cap differs")
        if isinstance(curve, dict):
            curve = curve.get("curve", curve.get("learning_curve", []))
        if not isinstance(curve, list) or not curve:
            raise ValueError("arm must supply a nonempty saved learning curve")
        for point in curve:
            passes = point["completed_bank_passes"]
            if (type(passes) is not int or not 1 <= passes <= 500
                    or point["additional_presentations"] != passes*100000
                    or point["updates"] != 60000+passes*int(np.ceil(100000/bs))):
                raise ValueError("recorded learning-curve exposure accounting differs")
        if [p["completed_bank_passes"] for p in curve] != sorted(set(p["completed_bank_passes"] for p in curve)):
            raise ValueError("curve has duplicated or unordered passes")
        evaluations = []
        for item in arm["evaluations"]:
            epoch = item["epochs"]
            if epoch not in EPOCHS or any(e["epochs"]==epoch for e in evaluations):
                raise ValueError("duplicate or undeclared report endpoint")
            evaluation = inputs.json(item["evaluation"], f"Batch {bs}, {epoch} pass evaluation")
            expected_format = "heston_spread_scale_precision_evaluation_v1" if precision_replay else "heston_spread_scale_capacity_evaluation_v1"
            if (evaluation.get("format") != expected_format
                    or evaluation.get("batch_size") != bs or evaluation.get("completed_bank_passes") != epoch
                    or evaluation.get("actual_optimizer_updates") != 60000+epoch*int(np.ceil(100000/bs))
                    or evaluation.get("study_protocol", {}).get("sha256") != summary["protocol"]["sha256"]
                    or resolved_spec(evaluation["study_protocol"], inputs.base) != resolved_spec(summary["protocol"], inputs.base)
                    or evaluation.get("family_checks") != 384
                    or evaluation["task"] != target or evaluation["population"]["truth_sha256"] != baseline["population"]["truth_sha256"]
                    or evaluation["population"]["observations_sha256"] != baseline["population"]["observations_sha256"]
                    or evaluation.get("reserved_arrays_opened") is not False):
                raise ValueError("evaluation changed target, recovery population or reserved scope")
            if precision_replay and evaluation.get("precision_recovery") != protocol["precision_recovery"]:
                raise ValueError("evaluation does not bind the numerical-recovery audit")
            recovery = shared.recovery_data(item["recovery"], inputs, evaluation, epoch)
            evaluations.append({"epochs":epoch, "evaluation":evaluation, "recovery":recovery,
                                "metrics":scale_metrics(*recovery)})
        if not evaluations:
            raise ValueError("arm has no completed endpoints")
        arms.append({"batch_size":bs, "run":run, "training":training, "curve":curve,
                     "evaluations":sorted(evaluations, key=lambda x:x["epochs"])})
    if sorted(a["batch_size"] for a in arms) != [128,512]:
        raise ValueError("both declared batch arms are required")
    arms.sort(key=lambda a:a["batch_size"])
    common_passes = set.intersection(*[{item["epochs"] for item in arm["evaluations"]} for arm in arms])
    if not common_passes:
        raise ValueError("a paired report requires at least one common evaluated exposure budget")
    comparison_passes = max(common_passes)
    source_paths = {name:inputs.path(spec, "Executed source: "+name) for name,spec in summary.get("sources", {}).items()}
    for name in ("trainer", "parent_trainer"):
        if name not in source_paths:
            raise ValueError("report requires executed continuation and parent trainer sources")
    for arm in arms:
        source_record = arm["run"]["source_sha256"]
        parent_source_sha = source_record.get("trainer", source_record.get("frozen_scale"))
        if (arm["run"].get("continuation_source_sha256") != shared.digest(source_paths["trainer"])
                or parent_source_sha != shared.digest(source_paths["parent_trainer"])):
            raise ValueError("executed trainer bytes differ from report source")
        if ("fixed_helper" in source_paths
                and arm["run"]["continuation_protocol"]["source_sha256"]["frozen_fixed"] != shared.digest(source_paths["fixed_helper"])):
            raise ValueError("effective-dropout helper differs from the executed continuation binding")
    conditioning_body, conditioning_record = input_conditioning(summary,inputs,arms,target,protocol,precision_recovery)
    source_paths["renderer"] = Path(__file__)
    inputs.sources.append({"label":"Presentation producer", "path":str(Path(__file__).resolve()), "sha256":shared.digest(__file__)})
    for path in (Path(shared.__file__), Path(shared.__file__).with_name("heston_spread_report_figures.py")):
        inputs.sources.append({"label":"Shared presentation helper", "path":str(path.resolve()), "sha256":shared.digest(path)})
    css_path = Path(shared.__file__).with_name("render_heston_spread_report.py")
    inputs.sources.append({"label":"Shared CSS source", "path":str(css_path.resolve()), "sha256":shared.digest(css_path)})
    final_records = [("Original 20k / 60k updates", COLORS["baseline"], baseline)]
    for arm in arms:
        matched = next(item for item in arm["evaluations"] if item["epochs"]==comparison_passes)
        final_records.append((f'Batch {arm["batch_size"]} / {comparison_passes} passes', COLORS[arm["batch_size"]], matched["evaluation"]))
    metrics = [{"label":"Original 60k", "batch_size":128,"epochs":None,**scale_metrics(*baseline_recovery)}]
    for arm in arms:
        metrics += [{"label":f'Batch {arm["batch_size"]}', "batch_size":arm["batch_size"], "epochs":item["epochs"], **item["metrics"]} for item in arm["evaluations"]]
    lead = "The question is whether additional physical coverage and optimization improve the learned spread multiplier c while preserving physical-parameter inference."
    if not interim and not fixture_only and comparison_passes == 500:
        widths = {item["batch_size"]:item["median_width90"] for item in metrics if item["epochs"]==500}
        lead = ("Both continuations on the 100k bank completed 500 additional passes, or 50 million noisy examples each. "
            "Training loss decreased and physical parameter recovery remained strong. "
            f"The inferred c remains near its prior width: median 90% widths are {widths[128]:.4f} (batch 128) and {widths[512]:.4f} (batch 512), against the prior width 1.35. "
            "Pooled SBC does not establish conditional accuracy; checks against matched MCMC posteriors still fail. "
            "We stop here for reflection before grid changes or Bergomi.")
    pieces = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Heston: larger bank, batches and longer training</title><style>'+shared.stylesheet(css_path)+'</style></head><body><main>',
        '<header><p class="eyebrow">Fixed 54-contract Heston · joint spread-scale development</p><h1>Larger bank, two batch sizes and longer training</h1>',
        '<p class="lead">'+esc(lead)+'</p>',
        '<button onclick="window.print()">Print report</button></header>',
        '<nav><a href="#design">Design</a><a href="#endpoints">Endpoint overview</a><a href="#information">Learning c</a><a href="#training">Training</a><a href="#recovery">Recovery</a><a href="#sbc">SBC</a><a href="#conditional">Conditional checks</a>'+('<a href="#conditioning">Exploratory inputs</a>' if conditioning_body is not None else '')+'<a href="#code">Code</a><a href="#sources">Sources</a></nav>']
    if fixture_only:
        pieces.append('<div class="callout warn"><strong>Renderer QA fixture — no new fit results.</strong><p>This test report copies diagnostics from the earlier study and uses synthetic arm metadata solely to test presentation. It must not be interpreted as evidence from the larger-bank study.</p></div>')
    if interim:
        pieces.append('<div class="callout"><strong>Interim development snapshot.</strong><p>Training records were copied into immutable presentation snapshots at '+esc(summary.get("snapshot_utc","the recorded snapshot time"))+'. Only completed evaluated checkpoints appear below. The planned 500-pass study is still in progress; this report does not claim those final endpoints.</p></div>')
    if precision_replay:
        pieces.append('<div class="callout warn"><strong>Precision-corrected replay after a recorded preprocessing failure.</strong><p>The original paired runs stopped at the same valid near-boundary target. Their evidence remains separate. The corrected results below come from new runs starting at the same saved parent, with a mathematically equivalent target transform evaluated more accurately. This is not a continuation from the failed state.</p></div>')
        pieces.append(section("Original failure and numerical repair",history_body,"precision-history"))
    design_rows = []
    for arm in arms:
        bs = arm["batch_size"]
        last = arm["evaluations"][-1]["epochs"]
        updates_per_pass = int(np.ceil(100000/bs))
        design_rows.append([str(bs), "100,000", str(updates_per_pass), str(100000%bs),
            f"{last:,}", f"{last*100000:,}", f"{last*updates_per_pass:,}"])
    pieces.append(section("What this comparison changes", '''<div class="equation">Yᵢ=Pᵢ(φ)+c hᵢ Zᵢ, &nbsp; Zᵢ∼N(0,1), &nbsp; c∼Uniform(.5,2)</div>
        <p>The model, five physical priors, fixed 54 quoted contracts, half-spreads and six-target network remain the same. Inputs are the ordered 54 noisy prices divided by their frozen half-spreads. c is a posterior target, never an input. There is no extra τ component.</p>
        <p>Both arms begin from the same 60k model and full Adam/normalization state. The clean physical bank increases from 20k to 100k rows. Each additional pass uses fresh independent c and Gaussian noise; the arms share the same ordered noisy examples, grouped into different batch sizes. Reserved test arrays remain unopened.</p>
        <p>The primary comparison holds examples seen fixed. Batch 128 gets about four times as many parameter updates as batch 512. This assesses the two training recipes at equal data exposure; it does not isolate gradient variance from update count. Improvement over the old model combines a larger bank, a useful learning-rate restart and more optimization, so no separate causal claim is made for those changes.</p>'''+
        table(["Batch", "Clean rows", "Updates / pass", "Last batch / pass", "Latest passes", "Additional noisy examples", "Additional updates"], design_rows)+
        '<p>Planned checkpoints: 25,100,250,500 complete 100k-row passes. Each worker has a 5,400-second upper wall limit; the campaign deadline can stop it earlier. There is no automatic extension. An incomplete arm must be described at its latest completed checkpoint.</p>', "design"))
    campaign_limits = protocol.get("campaign", {})
    campaign_labels = {"worker_wall_seconds":"Maximum wall time per worker, seconds",
        "whole_training_wall_seconds":"Original training campaign budget, seconds",
        "original_total_training_campaign_wall_seconds":"Original training campaign budget, seconds",
        "absolute_deadline_utc":"Original campaign deadline, UTC", "max_concurrent_workers":"Maximum concurrent workers",
        "automatic_retry":"Automatic retries", "automatic_extension":"Automatic extensions"}
    campaign_rows = [[esc(campaign_labels.get(key,key.replace("_"," "))),
        esc("Yes" if value is True else "No" if value is False else value)] for key,value in campaign_limits.items()]
    pieces.append(section("Recorded campaign limits", table(["Control","Recorded limit"],campaign_rows)+
        '<p>The campaign deadline applies in addition to each worker limit. Original and replay compute remain subject to their recorded bounds; planned 500-pass endpoints are not claimed unless their completed artifacts are present.</p>', "limits"))
    status_rows, endpoint_rows = [], []
    endpoint_counts = []
    for arm in arms:
        training = arm["training"]
        completed = training.get("completed_bank_passes", arm["curve"][-1]["completed_bank_passes"])
        updates = training.get("completed_updates", arm["curve"][-1]["updates"])
        seconds = training.get("seconds")
        status_rows.append([str(arm["batch_size"]), esc(training.get("status", "not recorded")),
            f"{completed:,}", f"{updates:,}", "Not recorded" if seconds is None else f"{float(seconds):,.1f}",
            esc(", ".join(str(item["epochs"]) for item in arm["evaluations"]))])
        for item in arm["evaluations"]:
            evaluation = item["evaluation"]
            synthetic = check_counts(evaluation, CASES[:2])
            market = check_counts(evaluation, CASES[2:])
            endpoint_counts.append({"batch_size":arm["batch_size"],"passes":item["epochs"],"synthetic":synthetic,"market":market})
            endpoint_rows.append([str(arm["batch_size"]),str(item["epochs"]),fmt(item["metrics"]["rmse"]),
                f'{sum(evaluation["population"]["rank_dkw"]["passed"])}/6',esc(count_text(synthetic)),esc(count_text(market))])
    pieces.append(section("Completed training and evaluated endpoints",
        '<p>Training completion and posterior accuracy are separate. This snapshot shows the saved status and elapsed worker time, then the checkpoints whose posterior checks are available. A training status of complete does not mean that the neural posterior passed the checks.</p>'+
        '<p>The main recovery, SBC and conditional-interval figures compare both batches after <strong>'+esc(comparison_passes)+' complete passes</strong> ('+esc(f'{comparison_passes*100000:,}')+' additional examples each), the highest shared evaluated budget. Every other completed endpoint remains listed separately.</p>'+
        table(["Batch", "Training status", "Completed bank passes", "Absolute optimizer updates", "Elapsed seconds", "Evaluated bank passes"],status_rows)+
        table(["Batch", "Passes", "c median RMSE", "SBC screens", "Synthetic mean/width checks", "Market mean/width checks"],endpoint_rows)+
        '<p>Each endpoint contributes 32 synthetic and 16 market mean/width checks. The declared 384-check family covers both batches, four endpoints, three cases, eight quantities and two metrics. Conditional screens use approximate Monte Carlo intervals from 400 block-bootstrap replicates with the recorded family adjustment; this is not a rigorous simultaneous error guarantee. The original 60k baseline uses its earlier 144-check family and remains descriptive outside this new family. SBC screens adjust across six marginals within one endpoint.</p>',"endpoints"))
    metric_rows = [[esc(m["label"]), "Baseline" if m["epochs"] is None else str(m["epochs"]),fmt(m["rmse"]),fmt(m["prior_rmse"]),
        "Undefined" if m["correlation"] is None else f'{m["correlation"]:+.4f}',
        f'{m["median_width90"]:.4f} / 1.35 = {m["median_width90"]/1.35:.3f}×'] for m in metrics]
    info = '<p>A small loss or a good SBC curve does not by itself show that c uses the quotes. Read median recovery, interval contraction and fixed-observation references together. The constant prior-median predictor always returns 1.25 on these same truths.</p>'
    info += table(["Estimator", "Additional bank passes", "c median RMSE", "Prior-median RMSE", "Truth–median correlation", "Median 90% width / prior 90% width"], metric_rows)
    with plt.rc_context({"svg.fonttype":"none", "svg.hashsalt":"heston-scale-capacity-v1", "font.family":"DejaVu Sans", "font.size":9}):
        fig, axes = plt.subplots(1,3,figsize=(11.5,3.8),layout="constrained")
        for arm in arms:
            values = arm["evaluations"];x=[e["epochs"] for e in values];color=COLORS[arm["batch_size"]]
            for axis,key in zip(axes,("rmse","correlation","median_width90"),strict=True):
                axis.plot(x,[e["metrics"][key] for e in values],"o-",color=color,label=f'Batch {arm["batch_size"]}')
        axes[0].axhline(metrics[0]["prior_rmse"],ls="--",color="#7e8790",label="Prior-median baseline")
        axes[2].axhline(1.35,ls="--",color="#7e8790",label="Prior 90% width")
        for axis,label in zip(axes,("c median RMSE","Truth–median correlation","Median c 90% width"),strict=True):
            axis.set(xlabel="Additional complete 100k-row passes",ylabel=label)
        axes[0].legend(fontsize=8)
        info += chart(shared._svg(fig,"c recovery and contraction over longer training","Saved posterior quantiles only; same validation observations at every point."),
            "Recovery is descriptive, not a requirement that noisy posterior medians lie exactly on the unit line. A near-prior neural interval beside a narrow matched joint reference indicates missed information at that observation.")
        pieces.append(section("Does the estimator learn information about c?",info,"information"))
        fig,axes=plt.subplots(1,2,figsize=(11.5,4),layout="constrained")
        for arm in arms:
            curve=arm["curve"]
            if isinstance(curve,dict):curve=curve.get("curve",curve.get("learning_curve",[]))
            for key,style,label in (("train_probe_loss",":","Train probe"),("validation_probe_loss","-","Validation probe")):
                points=[]
                for p in curve:
                    epoch=p.get("completed_bank_passes",p.get("completed_epochs",p.get("epochs",p.get("completed_passes",p.get("passes")))))
                    updates=p.get("additional_updates",p.get("updates_since_parent"))
                    if epoch is None and updates is not None:epoch=updates/int(np.ceil(100000/arm["batch_size"]))
                    if epoch is not None and key in p:points.append((float(epoch),float(p[key])))
                if points:
                    x,y=np.asarray(points).T
                    if not np.isfinite(y).all():raise ValueError("nonfinite learning curve")
                    axes[0].plot(x,y,style,color=COLORS[arm["batch_size"]],label=f'Batch {arm["batch_size"]} {label}')
            lr_points=[(0., implied_rate(0., arm["run"]["continuation_protocol"]))]
            for p in curve:
                epoch=p.get("completed_bank_passes",p.get("completed_epochs",p.get("epochs",p.get("completed_passes",p.get("passes")))))
                if epoch is not None:
                    lr_points.append((float(epoch),implied_rate(epoch,arm["run"]["continuation_protocol"])))
            if lr_points:
                x,y=np.asarray(lr_points).T;axes[1].plot(x,y,color=COLORS[arm["batch_size"]],label=f'Batch {arm["batch_size"]}')
        axes[0].set(xlabel="Additional complete bank passes",ylabel="Fixed-probe loss")
        axes[1].set(xlabel="Additional complete bank passes",ylabel="Implied next-step learning rate",yscale="log")
        for axis in axes:
            handles,_=axis.get_legend_handles_labels()
            if handles:axis.legend(fontsize=8)
        recipe='<p>The old endpoint learning rate was 5×10⁻⁷. Both arms explicitly restart at 5×10⁻⁵, cosine-decay by example exposure to 5×10⁻⁶ over 400 additional passes, then hold 5×10⁻⁶ through 500. Adam iteration and moments are retained. No automatic linear learning-rate scaling is applied to the larger batch.</p>'
        if precision_replay:
            recipe+='<p>The current recipe and worker cap are recorded in <code>run.continuation_protocol</code> and the corrected top-level fields. The earlier parent recipe is retained separately in <code>run.parent_recipe</code>. The campaign deadline shown above may stop a worker before its individual limit.</p>'
        else:
            recipe+='<p>The current recipe and worker cap are recorded in <code>run.continuation_protocol</code>. The inherited top-level <code>recipe</code> and <code>max_wall_seconds</code> describe the earlier 20k/60k parent run; they are retained as parent provenance and are not the current continuation settings. The campaign deadline shown above may stop a worker before its individual limit.</p>'
        recipe+='<p>Network: BayesFlow 2.1 CouplingFlow, rational-quadratic spline, depth 6; twelve conditioners with two 128-unit hidden layers each. The original constructor uses dropout .05; its 24 existing Dropout layers have effective rate 0 throughout this continuation. This is a pointwise flow with running input/target standardization, not batch normalization. Architecture, prior and observation channels do not change.</p>'
        recipe+=chart(shared._svg(fig,"Larger bank learning curves and implied next-step rates","Compact fixed-probe metrics; next-step rates derived from the hash-bound exposure schedule at recorded complete passes."),
            "The rate curve is derived from the recorded schedule and completed presentation counts; these rates were not logged by the trainer. Probe losses measure the same six-target objective. Better aggregate likelihood does not establish c recovery; the scale-specific evidence appears above. Exact executing network and schedule definitions are reproduced below.")
        versions=arms[0]["run"].get("versions",{})
        recipe+=table(["Software","Recorded version"],[[esc(k),esc(v)]for k,v in versions.items()])
        pieces.append(section("Exact training recipe and learning curves",recipe,"training"))
        recovery_body=""
        for arm in arms:
            matched=next(item for item in arm["evaluations"] if item["epochs"]==comparison_passes)
            recovery_body+='<h3>'+esc(f'Batch {arm["batch_size"]}: {comparison_passes} additional passes')+'</h3>'
            recovery_body+=recovery_plot(matched["recovery"],np.asarray(target["prior_bounds"]),f'Batch {arm["batch_size"]} six-parameter recovery',COLORS[arm["batch_size"]])
        pieces.append(section("Parameterwise recovery",recovery_body,"recovery"))
        pieces.append(section("Simple marginal SBC plots",sbc_plot(final_records),"sbc"))
        conditional=conditional_plot(final_records)
    conditional+='<p>These checks compare each neural posterior with a ready joint MCMC posterior at exactly the same observation vector and likelihood. Mean tolerance: 0.3 reference SD. Central 90% width-ratio range: [.75,1.25]. These are predeclared Monte Carlo-aware comparison screens, using approximate block-bootstrap intervals and the recorded family adjustment. They do not provide a rigorous simultaneous 384-check error guarantee or certify extreme-tail accuracy. They are separate from market-model adequacy and marginal SBC. The quantities control/affected are expected average-variance functionals, not option prices or model-risk reserves.</p>'
    for arm in arms:
        for item in arm["evaluations"]:
            ev=item["evaluation"]
            conditional+='<details'+(' open' if item["epochs"]==comparison_passes else '')+'><summary>'+esc(f'Batch {arm["batch_size"]}, {item["epochs"]} passes · family {ev.get("family_checks", "recorded")}')+'</summary>'+conditional_table(ev)+'</details>'
    pieces.append(section("Conditional posterior accuracy",conditional,"conditional"))
    if conditioning_body is not None:
        pieces.append(section("Exploratory input conditioning: a candidate for reflection",conditioning_body,"conditioning"))
    code='<p>AST-extracted from hash-bound executing source; these functions are displayed without importing the training module.</p>'
    code+='<div class="callout"><strong>Effective dropout during every update: 0 on all 24 existing Dropout layers.</strong><p>The <code>build_workflow</code> constructor below declares <code>dropout=.05</code>. During restoration, <code>switch_dropout_zero</code> sets the existing layers to effective rate 0 while preserving their arrays, and the full saved parent state is restored. These runs inherit the trained zero-dropout model. The constructor value and the effective training value are recorded separately.</p></div>'
    code+='<details><summary>Unchanged network and Adam construction</summary><pre><code>'+esc(code_excerpt(source_paths["parent_trainer"],"build_workflow"))+'</code></pre></details>'
    if "fixed_helper" in source_paths:
        code+='<details open><summary>Executing effective-zero dropout helper</summary><pre><code>'+esc(code_excerpt(source_paths["fixed_helper"],"switch_dropout_zero"))+'</code></pre></details>'
    continuation_source=Path(source_paths["trainer"]).read_text()
    candidates=[node.name for node in ast.parse(continuation_source).body if isinstance(node,(ast.FunctionDef,ast.ClassDef))
        and any(key in node.name.lower() for key in ("schedule","restore","rate","cosine"))]
    for name in candidates:
        code+='<details><summary>'+esc(name)+'</summary><pre><code>'+esc(code_excerpt(source_paths["trainer"],name))+'</code></pre></details>'
    if precision_replay:
        code+='<details open><summary>Precision-safe forward target transform</summary><pre><code>'+esc(code_excerpt(source_paths["trainer"],"host_adapt_training"))+'</code></pre></details>'
        if "original_expansion" not in source_paths:
            raise ValueError("precision replay report requires the original exposure-schedule helper")
        code+='<details><summary>Unchanged exact exposure schedule</summary><pre><code>'+esc(code_excerpt(source_paths["original_expansion"],"schedule_class"))+'</code></pre></details>'
    if "full_state_helper" in source_paths:
        code+='<details><summary>Exact full-state restoration helper</summary><pre><code>'+esc(code_excerpt(source_paths["full_state_helper"],"restore_full_state"))+'</code></pre></details>'
    pieces.append(section("Executing network, state restoration and rate code",code,"code"))
    source_rows=[[esc(s["label"]),'<code>'+esc(s["path"])+'</code>','<code>'+esc(s["sha256"])+'</code>']for s in inputs.sources]
    pieces.append(section("Hash-bound evidence and reproducibility",'<p>This standalone report contains inline SVG and styles, with no external asset requests. It reads compact development evidence and recovery quantiles only. No bank, model, saved posterior draws or MCMC chains were opened by the renderer.</p>'+table(["Input","Path","SHA-256"],source_rows),"sources"))
    inventory={"format":"heston_spread_scale_capacity_report_inventory_v1","sources":inputs.sources,"scale_metrics":metrics,
               "fixture_only":fixture_only,"interim":interim,"snapshot_utc":summary.get("snapshot_utc"),
               "endpoint_counts":endpoint_counts,"precision_replay":precision_replay,
               "matched_comparison_bank_passes":comparison_passes,
               "input_conditioning":conditioning_record,
               "original_partial_metrics":original_metrics,
               "report_scope":"Larger bank continuation; equal-example batch recipes; no release acceptance or Bergomi work"}
    pieces.append('<script type="application/json" id="report-inventory">'+json.dumps(inventory).replace("<","\\u003c")+'</script></main></body></html>')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text("\n".join(pieces))
    return {"out":str(out.resolve()),"sha256":shared.digest(out),"sources":len(inputs.sources),"scale_metrics":metrics}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(build(args.summary,args.out),indent=2))


if __name__=="__main__":
    main()
