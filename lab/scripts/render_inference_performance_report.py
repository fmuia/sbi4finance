#!/usr/bin/env python3
"""Render the completed-model evidence; never imports an inference backend.

Requires numpy, matplotlib, reportlab, pypdf; PyMuPDF is used for optional QA.
Run from any directory. Large checkpoints, banks and new experiments are unused.
"""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evidence/pilot-results/2026-10-28"
REPORTS = ROOT / "evidence/reports"
FIGURES = REPORTS / "inference-performance-2026-10-28-figures"
OUTPUT = ROOT / "output/pdf/inference-performance-2026-10-28.pdf"
SHARED = REPORTS / OUTPUT.name
GH = "https://github.com/fmuia/sbi4finance/blob/feat/parallel-tutorial-delivery/"
INK, TEAL, BLUE, AMBER = "#193044", "#087f8c", "#3b6ea8", "#b45d16"
MUTED, PALE, LINE = "#536777", "#f0f5f7", "#d7e1e7"
LABELS = {
    "initial_variance": "Initial variance", "theta": "Long-run variance",
    "vol_of_vol": "Vol-of-vol", "hurst": "H", "eta": "eta", "rho": "rho",
    "short_variance": "Short variance", "long_variance": "Long variance",
    "decay": "Curve decay", "rho_eta": "rho x eta*",
}
SOURCES = [
    ("S1", "Heston frozen-candidate held-out evaluation", "evidence/pilot-results/2026-10-28/heston-candidate-acceptance/results/acceptance/evaluation.json"),
    ("S2", "Known-curve 50k validation", "evidence/pilot-results/2026-10-28/known50k-completed/evaluation.json"),
    ("S3", "Joint-curve 20k validation", "evidence/pilot-results/2026-10-28/joint20k-completed/evaluation.json"),
    ("S4", "Heston analytic product comparisons", "evidence/pilot-results/2026-10-28/heston-candidate-acceptance/results/products/products.json"),
    ("S5", "Heston interior PyMC / neural verification", "evidence/pilot-results/2026-10-28/interior-corners/heston-interior-verification.json"),
    ("S6", "Known-curve 50k pricing-emulator report", "evidence/pilot-results/2026-10-28/known50k-completed/pricer-run.json"),
    ("S7", "Joint-curve 20k pricing-emulator report", "evidence/pilot-results/2026-10-28/joint20k-completed/pricer-run.json"),
    ("S8", "Earlier known-curve independent pricing checks (20k model)", "plans/backbone-validation-2026-10-28.md"),
    ("S9", "Complete Heston interior corner provenance", "evidence/pilot-results/2026-10-28/interior-corners/heston-interior-complete-corner.json"),
    ("S10", "Known 50k fixed-point rescore and compiled gradient checks", "evidence/pilot-results/2026-10-28/rough-emulator-readiness/known50k/readiness.json"),
    ("S11", "Pinned offline backbone artifact catalogue", "evidence/full-surface-backbones/artifacts.json"),
    ("S12", "Joint 20k compiled gradient checks", "evidence/pilot-results/2026-10-28/rough-emulator-readiness/joint20k/readiness.json"),
    ("S13", "Current known 50k PyMC completion and fixed-case comparisons", "evidence/pilot-results/2026-10-28/known50k-current-pymc/terminal-summary.json"),
    ("S14", "Current known 50k reference target and Monte Carlo qualifications", "evidence/pilot-results/2026-10-28/known50k-current-pymc/README.md"),
    ("S15", "Matched known-curve 50k versus 100k learning comparison", "evidence/pilot-results/2026-10-28/known100k-completed-fit/comparison-summary.json"),
    ("S16", "Unchanged validation data and new model identity proof", "evidence/pilot-results/2026-10-28/known100k-completed-fit/bank-and-model-identity-proof.json"),
    ("S17", "100k pricing-emulator independent fixed-point rescore", "evidence/pilot-results/2026-10-28/rough-emulator-readiness/known100k/readiness.json"),
    ("S18", "50k and 100k neural estimators against fixed 50k PyMC chains", "evidence/pilot-results/2026-10-28/known100k-fixed50k-pymc/comparison.json"),
    ("S19", "Completed known-curve 100k validation evaluation", "evidence/pilot-results/2026-10-28/known100k-completed-fit/known100k/known/validation/evaluation.json"),
]


def read(path):
    return json.loads(Path(path).read_text())


def datasets():
    paths = [RESULTS / "heston-candidate-acceptance/results/acceptance/evaluation.json",
             RESULTS / "known50k-completed/evaluation.json",
             RESULTS / "joint20k-completed/evaluation.json"]
    result = []
    for name, short, train, split, color, p in zip(
        ["Heston", "Rough Bergomi / known curve", "Rough Bergomi / joint curve"],
        ["Heston", "Known curve", "Joint curve"], [100000, 50000, 20000],
        ["Held-out test", "Development validation", "Development validation"],
        [BLUE, TEAL, AMBER], paths,
    ):
        raw = read(p); diagnostic = raw.get("diagnostics_with_derived_targets", raw)
        assert diagnostic["sbc"]["surfaces"] == 2000
        assert diagnostic["sbc"]["draws_per_surface"] == 1000
        result.append(dict(name=name, short=short, train=train, split=split, color=color,
                           raw=raw, diagnostic=diagnostic, path=p))
    assert result[0]["raw"]["reserved_test_evaluated"] is True
    assert all(not d["raw"]["reserved_test_evaluated"] for d in result[1:])
    return result


def learning_data():
    summary=read(RESULTS/"known100k-completed-fit/comparison-summary.json")
    identity=read(RESULTS/"known100k-completed-fit/bank-and-model-identity-proof.json")
    prices=read(RESULTS/"rough-emulator-readiness/known100k/readiness.json")
    fixed=read(RESULTS/"known100k-fixed50k-pymc/comparison.json")
    evaluation=read(RESULTS/"known100k-completed-fit/known100k/known/validation/evaluation.json")
    assert summary["same_validation"]["example_ids_equal"]
    assert summary["same_validation"]["surfaces"]==2000
    assert summary["same_validation"]["draws_per_surface"]==1000
    assert identity["all_5000_validation_parts_identical"]
    assert identity["original50k_training_prefix_unchanged"]
    assert evaluation["checkpoint_sha256"]==identity["actual_posterior_sha256"]==summary["identities"]["posterior100k"]
    assert fixed["posterior_checkpoint_sha256"]==summary["identities"]["posterior100k"]
    assert not summary["cache_promoted"] and not summary["scientific_acceptance"]
    assert fixed["new_mcmc_runs"]==0 and not fixed["new_neural_sampling"]
    assert all(c["neural_draws_each_model"]==1000 and c["reference_chain_shape"]==[4,8000,3] for c in fixed["cases"])
    for model in ["known50k_neural","known100k_neural"]:
        assert sum(c[model]["proposed_parameter_agreement"]["all_checks_passed"] for c in fixed["cases"])==2
    return summary,identity,prices,fixed,evaluation


def style_plots():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.edgecolor": LINE, "grid.color": LINE,
                         "savefig.facecolor": "white"})


def savefig(fig, name):
    path = FIGURES / name
    fig.savefig(path, dpi=185, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_figures(data):
    style_plots(); FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), gridspec_kw={"width_ratios": [1, 1, 1.35]})
    for ax, d in zip(axes, data):
        dg = d["diagnostic"]; vals = np.asarray(dg["sbc"]["rank_ks"]) / dg["sbc"]["dkw_bound"]
        ys = np.arange(len(vals)); ax.barh(ys, vals, color=d["color"], height=.62)
        ax.axvline(1, color=INK, ls="--", lw=1)
        ax.set_yticks(ys, [LABELS[t] for t in dg["targets"]]); ax.invert_yaxis()
        ax.set_xlim(0, 1.14); ax.set_xticks([0, .5, 1]); ax.set_title(d["short"], loc="left", fontweight="bold")
        ax.set_xlabel("Rank KS / declared limit")
        for y, v in zip(ys, vals): ax.text(v + .025, y, f"{v:.2f}", va="center", fontsize=8)
    fig.tight_layout(w_pad=2.3); savefig(fig, "rank-screens.png")

    rows, labels, separators = [], [], []
    for d in data:
        dg = d["diagnostic"]
        for i, target in enumerate(dg["targets"]):
            rows.append([100 * (dg["coverage"][str(q)]["coverage"][i] - q) for q in [.5, .8, .9, .95]])
            labels.append(f"{d['short']} | {LABELS[target]}")
        separators.append(len(rows)-.5)
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    values = np.asarray(rows)
    im = ax.imshow(values, cmap="BrBG", norm=TwoSlopeNorm(vmin=-3.5, vcenter=0, vmax=3.5), aspect="auto")
    ax.set_xticks(range(4), ["50% interval", "80% interval", "90% interval", "95% interval"])
    ax.xaxis.tick_top(); ax.tick_params(length=0, pad=8); ax.set_yticks(range(len(labels)), labels)
    for cut in separators[:-1]: ax.axhline(cut, color="white", lw=3)
    for i in range(len(rows)):
        for j in range(4): ax.text(j, i, f"{values[i,j]:+.2f}", ha="center", va="center", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.04)
    cb.set_label("Observed minus nominal coverage (percentage points)")
    fig.tight_layout(); savefig(fig, "coverage-deviations.png")

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 4.9), sharey=True)
    labels, cs, recovery, contraction = [], [], [], []
    for d in data:
        for key in d["raw"]["recovery_correlation"]:
            labels.append(f"{d['short']} | {LABELS[key]}"); cs.append(d["color"])
            recovery.append(d["raw"]["recovery_correlation"][key]); contraction.append(d["raw"]["contraction_logit"][key])
    y = np.arange(len(labels))
    for ax, vals, title, lower in zip(axes, [recovery, contraction],
                                     ["Truth vs posterior-mean correlation", "Reported logit-space contraction"], [.7, .5]):
        ax.scatter(vals, y, c=cs, s=38, zorder=3); ax.set_xlim(lower, 1.01)
        ax.grid(axis="x"); ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
        for yy, v in zip(y, vals): ax.text(v-.012, yy-.16, f"{v:.3f}", ha="right", fontsize=7)
    axes[0].set_yticks(y, labels); axes[0].invert_yaxis(); fig.tight_layout(w_pad=2)
    savefig(fig, "identification.png")

    pricers = [read(RESULTS / f"{n}/pricer-run.json") for n in ["known50k-completed", "joint20k-completed"]]
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.15))
    for ax, key, title in zip(axes, ["absolute_quote_error_quantiles", "surface_rms_error_quantiles"],
                              ["Absolute quote error", "RMS error across 63 quotes"]):
        for p, d in zip(pricers, data[1:]):
            vals = [p["validation_errors"][key][q] for q in ["0.5", "0.9", "0.99", "1.0"]]
            ax.plot(range(4), vals, marker="o", color=d["color"], label=d["short"])
        ax.set_xticks(range(4), ["Median", "90th", "99th", "Maximum"]); ax.set_yscale("log")
        ax.set_ylabel("Quote-noise SD units"); ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
        ax.grid(axis="y", which="major"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(w_pad=2); savefig(fig, "pricer-tail-errors.png")

    products = read(RESULTS / "heston-candidate-acceptance/results/products/products.json")
    vals, labs = [], []
    for c in products["cases"]:
        for product, pr in c["products"].items():
            for q in pr["quantiles"]:
                assert q["status"] == "within_declared_tolerance"
                vals.append((q["absolute_error"] + q["combined_95_percent_half_width"]) / q["tolerance"])
                short = "Var swap" if product=="variance_swap_1y" else "Fwd var"
                labs.append(f"Case {c['case']} / {short} / q{int(q['probability']*100)}")
    fig, ax = plt.subplots(figsize=(8.6, 3.3));y=np.arange(len(vals));ax.barh(y,vals,color=BLUE,height=.65)
    ax.set_yticks(y,labs,fontsize=8);ax.invert_yaxis();ax.axvline(1,color=INK,ls="--",lw=1)
    ax.set_xlim(0,1.08);ax.set_xlabel("(Absolute error + uncertainty) / declared tolerance")
    for yy,v in zip(y,vals):ax.text(v+.015,yy,f"{v:.2f}",va="center",fontsize=7)
    fig.tight_layout();savefig(fig,"heston-products.png")


def make_styles():
    s=getSampleStyleSheet()
    s.add(ParagraphStyle(name="KickerR",fontName="Helvetica-Bold",fontSize=9,leading=12,textColor=colors.HexColor(TEAL),spaceAfter=8))
    s.add(ParagraphStyle(name="TitleR",fontName="Helvetica-Bold",fontSize=26,leading=30,textColor=colors.HexColor(INK),spaceAfter=12))
    s.add(ParagraphStyle(name="H1R",fontName="Helvetica-Bold",fontSize=20,leading=24,textColor=colors.HexColor(INK),spaceAfter=11))
    s.add(ParagraphStyle(name="H2R",fontName="Helvetica-Bold",fontSize=11,leading=15,textColor=colors.HexColor(TEAL),spaceBefore=9,spaceAfter=5))
    s.add(ParagraphStyle(name="BodyR",fontName="Helvetica",fontSize=9.4,leading=13.7,textColor=colors.HexColor(INK),spaceAfter=7))
    s.add(ParagraphStyle(name="SmallR",fontName="Helvetica",fontSize=7.7,leading=10.5,textColor=colors.HexColor(MUTED),spaceAfter=5))
    s.add(ParagraphStyle(name="CaptionR",fontName="Helvetica",fontSize=8.1,leading=11.3,textColor=colors.HexColor(MUTED),spaceAfter=8))
    s.add(ParagraphStyle(name="CellR",fontName="Helvetica",fontSize=8.4,leading=11.3,textColor=colors.HexColor(INK)))
    s.add(ParagraphStyle(name="CalloutR",fontName="Helvetica-Bold",fontSize=10.2,leading=14.6,textColor=colors.HexColor(INK),spaceAfter=7))
    return s


def render(data):
    styles=make_styles();story=[];width=A4[0]-84
    learning,learning_identity,learning_prices,fixed_reference,learning_evaluation=learning_data()
    def para(text,style="BodyR"):return Paragraph(text,styles[style])
    def add(text,style="BodyR"):story.append(para(text,style))
    def image(name,w=width,h=None):
        from PIL import Image as PI
        path=FIGURES/name
        with PI.open(path) as p:ow,oh=p.size
        return Image(str(path),width=w,height=(w*oh/ow if h is None else h))
    def table(rows,widths=None,pad=7):
        cells=[[para(str(v),"CellR") for v in row] for row in rows]
        t=Table(cells,colWidths=widths or [width/len(rows[0])]*len(rows[0]),hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(PALE)),("LINEBELOW",(0,0),(-1,0),.7,colors.HexColor(LINE)),
                              ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
                              ("TOPPADDING",(0,0),(-1,-1),pad),("BOTTOMPADDING",(0,0),(-1,-1),pad),
                              ("LINEBELOW",(0,1),(-1,-1),.3,colors.HexColor(LINE))]))
        return t
    def callout(text):
        t=Table([[para(text,"CalloutR")]],colWidths=[width]);t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor(PALE)),("BOX",(0,0),(-1,-1),.5,colors.HexColor(LINE)),("LEFTPADDING",(0,0),(-1,-1),11),("RIGHTPADDING",(0,0),(-1,-1),11),("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),5)]));return t
    def page(kicker,title):
        if story:story.append(PageBreak())
        add(kicker,"KickerR");add(title,"H1R")

    add("SBI4FINANCE / NUMERICAL EVIDENCE / 28 OCTOBER 2026","KickerR")
    add("Inference performance","TitleR")
    add("A readable review of the completed tutorial fits. Three baseline models are shown throughout; page 8 adds the newly completed known-curve 100k fit and its matched learning comparison.")
    story.append(table([["Trained model","Training surfaces","Evaluation panel","Rank screens"],
        ["Heston / final epoch 500","100,000","2,000 held-out test cases","4 / 4 pass"],
        ["Rough Bergomi / known curve","50,000","2,000 validation cases","4 / 4 pass*"],
        ["Rough Bergomi / inferred curve","20,000","2,000 validation cases","7 / 7 pass*"],
        ["Known curve / later fit (page 8)","100,000","Same 2,000 validation cases","4 / 4 pass*"],
    ],[159,81,156,115]));story.append(Spacer(1,10))
    story.append(callout("Calibration screens are encouraging for the baseline and later fits. Heston has the strongest independent reference evidence. Known-curve rough inference matches two of three approximate emulator-PyMC cases; full model-posterior, pricing and product acceptance remain unfinished."))
    add("Baseline rank calibration: every recorded target","H2R");story.append(image("rank-screens.png"))
    add("Bars show rank-KS distance divided by each model's declared simultaneous limit. Values below the dashed line pass that model's screen. Limits differ with the number of targets. [S1-S3]","CaptionR")
    add("* rho x eta is computed downstream from each joint posterior draw; it is not another inferred parameter. Heston has four inferred parameters with kappa fixed at 2; the known-curve rough model has three, and the joint-curve rough model has six.","SmallR")
    add("The figures on pages 1-7 retain Heston 100k, known-curve 50k and joint-curve 20k as explicit baselines. The completed known-curve 100k fit is additional development evidence, without cache promotion; its inference and price emulator are assessed separately on page 8. The joint 100k bank extension remains separate.","SmallR")

    page("02 / UNCERTAINTY CALIBRATION","Passing ranks is a starting point")
    add("Every panel uses 2,000 simulated observations and 1,000 posterior draws per observation. Heston's candidate and rules were frozen before opening these test outcomes. Both rough panels are development validation; their reserved tests remain unused for inference evaluation.")
    story.append(image("coverage-deviations.png"))
    add("Cell values are percentage-point deviations from nominal central-interval coverage. Brown is undercoverage; green is overcoverage. rho x eta* uses aligned joint samples. These are separate model-wise diagnostics, not a cross-model test of superiority. [S1-S3]","CaptionR")
    add("What requires attention","H2R")
    add("Heston's 50% intervals slightly overcover initial and long-run variance (52.25% / 52.30%). Known-curve rho x eta has 93.85% coverage for 95% intervals (pointwise Wilson upper 94.82%). Joint-curve decay has 46.75% coverage for 50% intervals; short-variance intervals overcover and eta's 95% coverage is 94.00%.")
    add("The retained Wilson intervals are pointwise and exploratory, without adjustment across parameters and coverage levels. These qualifications coexist with passing simultaneous rank screens. Neither result proves conditional or full joint-posterior calibration.","CaptionR")

    page("03 / IDENTIFICATION","What the observations identify")
    add("All three estimators recover substantial signal across the simulated prior. The joint curve's decay parameter is visibly less constrained than its short variance level, an interpretable weak-identification direction rather than an execution failure.")
    story.append(image("identification.png"))
    add("Left: correlation between generating parameters and posterior means across the evaluation panel. Right: the evaluator's logit-space contraction statistic; higher values indicate a larger posterior variance reduction relative to the transformed uniform prior. All primary inferred parameters are shown. Derived products are omitted from this contraction calculation. [S1-S3]","CaptionR")
    story.append(table([["Scenario","Inference target","Legitimate observed context"],
        ["Heston","Initial variance, long-run variance, vol-of-vol, rho","The fixed complete contract design; kappa = 2"],
        ["Known-curve rough","H, eta, rho","Seven known variance-curve levels vary between cases and are fixed within each inference"],
        ["Joint-curve rough","H, eta, rho, short variance, long variance, decay","Contracts and quotes; the generating curve must not enter the conditioning input"],
    ],[104,176,231]));story.append(Spacer(1,8))
    add("All models use BayesFlow 2.1 FlowMatching with vector conditions and four 256-unit layers, final epoch 500, and its native adaptive solver. These results do not yet compare affine/spline flows, MeanFlow, DeepSet or SetTransformer. Their comparisons are independent follow-on work, not an explanation already established by these runs.","SmallR")

    page("04 / INDEPENDENT HESTON REFERENCE","A complete interior posterior")
    add("This four-dimensional development case compares the same Heston 100k final-500 candidate against an independent Fourier-likelihood PyMC reference. All inferred parameters and every pair are visible; the reference's central 99% intervals lie comfortably inside the prior.")
    corner=RESULTS/"interior-corners/heston-interior-complete-corner.png"
    from PIL import Image as PI
    with PI.open(corner) as p: corner_w,corner_h=p.size
    story.append(Image(str(corner),width=width,height=width*corner_h/corner_w))
    add("Black/gray: four PyMC DEMetropolisZ chains, 8,000 retained draws each. Blue: 10,000 neural posterior draws. Green dashed lines: generating parameters. This example is mostly regular; it is not evidence of multimodality or a dramatic banana. [S5,S9]","CaptionR")
    story.append(table([["Reference / precision","Neural comparison"],
        ["Max R-hat 1.00297; min bulk ESS 1,908; min tail ESS 2,607. Largest tested Fourier price shift: 2.97e-6 quote SD.","Max mean discrepancy 0.1013 reference SD; 90% width ratios 1.016-1.040. Largest marginal CDF discrepancy 0.064 (rho)."],
    ],[width/2,width/2]))
    add("An ex-ante midpoint case and descriptive development comparison, not another untouched conditional acceptance collection. Earlier boundary examples and their failures remain preserved.","SmallR")

    page("05 / PRICING AND PRODUCTS","Where neural inference meets prices")
    add("The forward emulators approximate the numerical bank's conditional mean prices. Validation interpolation error mixes approximation error with label randomness; it does not measure continuous-model bias or validate an emulator likelihood.")
    story.append(image("pricer-tail-errors.png"))
    add("Current known 50k: RMSE 0.08287 quote SD, worst quote 1.311 SD. Current joint 20k: RMSE 0.10118, worst quote 3.888 SD and worst surface RMS 1.522 SD. Rare large errors matter for likelihood products across 63 quotes. [S6,S7]","CaptionR")
    add("New: the current 50k emulator improves the independent benchmark","H2R")
    rescore=read(RESULTS/"rough-emulator-readiness/known50k/readiness.json")
    assert rescore["new_pricing_paths"]==0 and not rescore["mcmc_run"]
    joint_gradient=read(RESULTS/"rough-emulator-readiness/joint20k/readiness.json")
    for panel, expected_count in [(rescore, 6), (joint_gradient, 4)]:
        assert panel["new_pricing_paths"]==0 and not panel["mcmc_run"]
        assert panel["all_gradient_checks_passed"]
        assert len(panel["gradient_checks"])==expected_count
        assert all(point["pytensor"]["passed"] for point in panel["gradient_checks"])
    rescore_rows=[["Fixed point","Old 20k coarse RMS","Current 50k coarse RMS","Current 50k refined RMS"]]
    for point in rescore["reused_price_panel"]:
        role={"truth":"generating vector","neural_mean":"old neural mean","reference_mean":"old PyMC mean"}[point["role"]]
        old=point["old_rms_error_by_grid"][0];new=point["new_rms_error_by_grid"]
        assert new[0]<old
        rescore_rows.append([f"Case {point['case']} / {role}",f"{old:.4f}",f"{new[0]:.4f}",f"{new[1]:.4f}"])
    story.append(table(rescore_rows,[170,112,113,116]));story.append(Spacer(1,8))
    add("All six current-emulator coarse-grid RMS errors are smaller at these fixed benchmark points. This rescore reused 48 saved independent batches with no new paths or MCMC. The neural/PyMC vectors belong to the earlier 20k study, not current 50k posterior means. RMS entries are point estimates in quote-noise SD units; the source retains per-quote t7 intervals and replicate covariance. Separate new PyMC results follow on page 6. [S10]","CaptionR")
    add("Refined-grid RMS remains 0.0526-0.1641 and the largest inspected refined-quote error is 0.5683 SD. The refined grid is a comparator, not exact truth. Grid shifts are unchanged by retraining; these six points cannot establish posterior-region or prior-wide accuracy, or attribute improvement solely to training-data count.","BodyR")

    page("06 / REFERENCES AND PRODUCTS","Evidence beyond marginal calibration")
    add("Heston has an independent Fourier-likelihood PyMC comparison and checks on propagated variance uncertainty. A new known-curve rough PyMC run improves agreement for its approximate emulator target; agreement with the full numerical model remains unfinished.")
    add("Heston analytic product uncertainty","H2R")
    story.append(image("heston-products.png",w=width*.90))
    add("All 12 development-product q10/q90 comparisons are inside the declared tolerance including chain-aware quantile uncertainty. The products are the one-year variance-swap rate and six-to-twelve-month forward variance. They test variance-curve uncertainty, not leverage, vol-of-vol or path-dependent payoffs. [S4]","CaptionR")
    add("New: current known-curve 50k PyMC comparison","H2R")
    reference=read(RESULTS/"known50k-current-pymc/terminal-summary.json")
    assert reference["state"]=="COMPLETED" and reference["exit_code"]=="0:0"
    assert all(c["mcmc_diagnostics_passed"] for c in reference["cases"])
    assert sum(c["parameter_comparison_passed"] for c in reference["cases"])==2
    add("Four native PyMC DEMetropolisZ chains x 8,000 retained draws per case pass all three chain/derived diagnostic screens. With the same observations and known curves, parameter checks pass cases 0 and 1 (2/3 versus old 1/3), using 0.3-SD mean and 25% relative-width rules. [S13,S14]","BodyR")
    challenge=reference["cases"][2]
    reference_rows=[["Case 2 diagnostic","Mean error / reference SD","Neural / PyMC 90% width"]]
    for key in ["hurst","eta","rho","rho_eta"]:
        values=challenge["rho_eta"] if key=="rho_eta" else challenge["parameters"][key]
        reference_rows.append([LABELS[key],f"{values['standardized_mean_error']:.3f}",f"{values['width_ratio_90']:.3f}"])
    story.append(table(reference_rows,[180,166,165]));story.append(Spacer(1,5))
    add("Case 2 still fails and lies near the eta prior cap: its eta mean discrepancy improves from 2.052 to 0.789 SD, while intervals remain too wide. Current comparisons reuse 1,000 saved neural draws versus 10,000 previously; neural Monte Carlo uncertainty and uncomputed reference-quantile MCSE qualify the width screens. Improvement cannot be attributed solely to more training data. [S8,S13,S14]","CaptionR")
    story.append(callout("A converged emulator-PyMC chain is a reference for that emulator target. It is not automatically a reference for the stochastic bank simulator or continuous-time model: numerical grid bias, emulator approximation and Monte Carlo label variability remain separate errors."))
    add("What remains for the rough references","H2R")
    add("Joint 20k has no completed conditional PyMC comparison. Ten local analytic/finite-difference/compiled PyTensor gradient checks pass; gradients alone do not establish likelihood or posterior accuracy. Next: independent prices around posterior mass, then qualified conditional and product comparisons. [S10,S12,S14]","BodyR")
    add("Scientific-acceptance flags remain false; reserved rough inference-test outcomes were not opened.","SmallR")

    page("07 / EVIDENCE AND NEXT DECISIONS","What is ready, and what comes next")
    story.append(callout("Usable substrate: three baseline estimators with offline replays, calibration evidence, an independently checked Heston example, and a later known-curve 100k fit. Remaining work concerns conditional rough inference, pricing tails, discretization, products and a full-100k joint fit."))
    add("Parallel priorities","H2R")
    for text in [
        "<b>Review the completed known-curve 100k fit before promotion.</b> The matched comparison on page 8 separates posterior behavior from a worse new price emulator. Preserve the 50k cache/reference and Heston's frozen test boundary.",
        "<b>Resolve the rough reference limitations.</b> Preserve the new known-curve 2/3 result and case-2 failure; check independent prices around posterior mass and prepare a joint-curve reference. Emulator agreement is one development screen. [S13,S14]",
        "<b>Prepare comparable architecture and summary experiments.</b> Separate inference-family changes from summary-network changes; match data, target, seeds and measured compute, and retain failed runs.",
        "<b>Review the prepared offline backbone notebook.</b> The new backbone_review.py replays pinned Heston, known 50k and joint 20k artifacts; it was statically checked, not executed. Cached examples are not fresh inference or new acceptance, and the joint fixture has prior-edge proximity. [S11]",
    ]:add(text)
    add("Model identities","H2R")
    for d in data:
        add(f"<b>{d['name']}</b> / {d['train']:,} training surfaces / checkpoint <font face='Courier'>{d['raw']['checkpoint_sha256']}</font>","SmallR")
    add("Reproducible evidence","H2R")
    source_cells=[]
    half=(len(SOURCES)+1)//2
    for i in range(half):
        row=[]
        for index in [i,i+half]:
            if index<len(SOURCES):
                code,title,path=SOURCES[index]
                row.append(para(f"<b>[{code}]</b> <link href='{GH+path}' color='{TEAL}'>{title}</link>","SmallR"))
            else:row.append("")
        source_cells.append(row)
    source_table=Table(source_cells,colWidths=[width/2,width/2],hAlign="LEFT")
    source_table.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),12),("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    story.append(source_table)
    add("The companion Markdown lists exact repository paths and full file hashes. Re-render with scripts/render_inference_performance_report.py. The renderer reads retained JSON and one existing figure; it performs no simulation, training, MCMC, notebook execution or new test evaluation.","SmallR")

    page("08 / MATCHED LEARNING COMPARISON","More data: what actually changed")
    story.append(callout("The known-curve 100k fit completed. Marginal calibration changes little; some conditional discrepancies shrink, but the new price emulator is worse. Bank growth is useful infrastructure, not a guarantee that every fitted component improves."))
    add("Same 2,000 validation cases x 1,000 draws; all 5,000 validation surfaces and the original 50k training prefix have identical hashes. Both fits use seed 0, the same architecture, default adaptive sampler and final epoch 500. Updates differ: 98,000 versus 195,500. This is one seed, not a replicated or fixed-compute learning curve. [S15,S16]","CaptionR")
    comparison_rows=[["Diagnostic target","Rank KS: 50k / 100k","90% coverage: 50k / 100k"]]
    for target in learning["posterior_comparison"]:
        comparison_rows.append([LABELS[target["target"]],f"{target['rank_ks_50k']:.5f} / {target['rank_ks_100k']:.5f}",f"{100*target['coverage90_50k']:.2f}% / {100*target['coverage90_100k']:.2f}%"])
    story.append(table(comparison_rows,[120,185,206],pad=4))
    add("Both four-target marginal rank families pass the same 0.03562 limit; rho x eta is derived downstream. Small KS changes and nearly unchanged coverage do not establish a material posterior improvement. [S15,S19]","CaptionR")
    add("The 100k price emulator regresses","H2R")
    pricer=learning["pricer_comparison"]
    story.append(table([["Validation label metric","50k pricer","100k pricer"],["RMSE / quote-noise SD",f"{pricer['rmse50k']:.5f}",f"{pricer['rmse100k']:.5f}"],["Epochs run / selected",f"{pricer['epochs_run50k']} / {pricer['selected_epoch50k']}",f"{pricer['epochs_run100k']} / {pricer['selected_epoch100k']}"]],[241,135,135],pad=4))
    old_prices=read(RESULTS/"rough-emulator-readiness/known50k/readiness.json")["reused_price_panel"]
    assert len(old_prices)==len(learning_prices["reused_price_panel"])==6
    for old,new in zip(old_prices,learning_prices["reused_price_panel"]):
        assert (old["case"],old["role"])==(new["case"],new["role"])
        assert all(n>o for o,n in zip(old["new_rms_error_by_grid"],new["new_rms_error_by_grid"]))
    add("Label RMSE worsens 18.8%, and independent coarse/refined RMS errors rise at all six reused benchmark points. Case-2 truth coarse RMS: 0.1052 to 0.2426 quote SD. Early stopping selected very different training stages: an optimization confound, not a proven explanation or evidence that more data is harmful. [S15,S17]","CaptionR")
    add("Hold the 50k PyMC reference fixed","H2R")
    fixed_rows=[["Case 2 target","Mean error / SD: 50k / 100k","90% width ratio: 50k / 100k"]]
    case=fixed_reference["cases"][2]
    for key in ["hurst","eta","rho","rho_eta"]:
        vals=[]
        for model in ["known50k_neural","known100k_neural"]:
            panel=case[model]
            vals.append(panel["derived_parameter_comparison"] if key=="rho_eta" else panel["proposed_parameter_agreement"]["parameters"][key])
        fixed_rows.append([LABELS[key],f"{vals[0]['standardized_mean_error']:.3f} / {vals[1]['standardized_mean_error']:.3f}",f"{vals[0]['width_ratio_90']:.3f} / {vals[1]['width_ratio_90']:.3f}"])
    story.append(table(fixed_rows,[120,185,206],pad=4))
    add("Both saved neural models pass 2/3 fixed-case parameter screens; case 2 still fails in H and eta. Each uses 1,000 draws against exactly the same four x 8,000 50k-emulator PyMC draws, observations and curves. No new MCMC; the worse 100k pricer is not the reference. Quantile/neural Monte Carlo errors remain unassessed; this approximate-target comparison is not acceptance. The 50k cache stays unchanged. [S18]","CaptionR")
    add(f"New 100k posterior: <font face='Courier'>{learning['identities']['posterior100k']}</font>","SmallR")

    def chrome(canvas,doc):
        canvas.saveState();w,h=A4
        canvas.setStrokeColor(colors.HexColor(LINE));canvas.line(42,h-30,w-42,h-30)
        canvas.setFont("Helvetica",7);canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(42,h-23,"SBI4FINANCE / INFERENCE PERFORMANCE / COMPLETED MODEL EVIDENCE")
        canvas.line(42,34,w-42,34);canvas.drawString(42,23,"28 October 2026 | Synthetic complete-surface scenarios | Development limitations retained")
        canvas.drawRightString(w-42,23,f"{doc.page}");canvas.restoreState()
    OUTPUT.parent.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(OUTPUT),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=46,bottomMargin=45,
                          title="SBI4Finance: inference performance, 28 October 2026",author="SBI4Finance collaborators")
    doc.build(story,onFirstPage=chrome,onLaterPages=chrome,canvasmaker=partial(Canvas,invariant=1))
    shutil.copyfile(OUTPUT,SHARED)


def companion(data):
    learning,learning_identity,learning_prices,fixed_reference,learning_evaluation=learning_data()
    rows=[]
    for d in data:
        dg=d["diagnostic"];vals=dg["coverage"]["0.95"]["coverage"]
        for t,ks,cv in zip(dg["targets"],dg["sbc"]["rank_ks"],vals):
            rows.append(f"| {d['short']} | {LABELS[t]} | {ks:.5f} | {dg['sbc']['dkw_bound']:.5f} | {100*cv:.2f}% |")
    text="""# Inference performance - 28 October 2026

[Read the rendered eight-page report](inference-performance-2026-10-28.pdf).

The main figures retain three completed baseline models: **Heston100k final500**, **known-curve rough50k**, and **joint-curve rough20k**. Page 8 adds the newly completed **known-curve rough 100k** fit. Each was evaluated on 2,000 cases with 1,000 posterior draws per case. Heston uses a frozen held-out test panel; the rough models use development validation, with reserved inference-test outcomes untouched. All recorded simultaneous rank screens pass. This does not imply perfect, conditional or full joint calibration.

The baseline known-curve 50k and later 100k fit have distinct model and price-emulator hashes. The new results do not overwrite the 50k cache or its current reference. rho x eta is a downstream product of aligned joint draws, not an additional inferred target.

## Recorded rank screens and 95% coverage

| Model | Diagnostic target | Rank KS | Declared limit | 95% coverage |
| --- | --- | ---: | ---: | ---: |
"""+"\n".join(rows)+"""

Pointwise coverage qualifications remain: Heston initial/long-run variance 50% coverage is52.25/52.30%; known-curve rho x eta95% coverage is93.85%; joint decay50% coverage is46.75%, eta95% is94.00%, and short variance80/90/95% coverage is82.3/92.0/96.35%. Retained Wilson intervals are exploratory pointwise checks, without multiplicity adjustment.

## Reference and pricing evidence

- Heston's complete interior4D PyMC/Fourier comparison is descriptive development evidence: maximum R-hat1.00297, minimum bulkESS1,908, maximum neural/reference mean difference.1013referenceSD and90%width ratios1.016-1.040. It is mostly regular, not evidence of dramatic non-Gaussian geometry.
- All12 Heston analytic product quantile comparisons pass their declared uncertainty-aware tolerances. These are variance-swap and forward-variance rates, not path-product acceptance.
- Current pricing-emulator validation-label RMSE is.08287quoteSD (known50k) and.10118 (joint20k). Worst quote errors are1.311 and3.888 respectively; worst joint surfaceRMS is1.522. Mean error alone does not validate an emulator-based likelihood.
- Earlier known 20k conditional comparisons passed 1/3 cases; independent price and grid checks detected up to .742 and .276 quote SD discrepancies. Current known 50k PyMC job 8059665 is complete: all three four-chain x 8,000-draw diagnostic screens pass, and parameter-agreement checks pass cases 0/1 and fail case 2 (2/3). The same observations, known curves, 0.3-reference-SD mean and 25% relative-width rules were reused. Case 2 mean errors for H/eta/rho are .822/.789/.290 reference SD and 90% width ratios 1.814/1.318/1.423; downstream rho x eta has mean error .103 SD and width ratio 1.451. Eta's mean discrepancy improves from 2.052 SD, but current neural comparisons reuse 1,000 draws versus 10,000 previously, reference-quantile MCSE is uncomputed, and no data-count-only causal claim is warranted. The reference remains a deterministic mean-emulator likelihood with independent Gaussian quote noise; MC label covariance, emulator uncertainty and grid bias are omitted. Joint conditional reference and full model/product/grid acceptance remain unfinished. [S13,S14]
- A new independent fixed-point rescore uses the current 50k pricer against the same 48 saved independent batches. Coarse RMS errors improve at all six fixed points: case 0 old/current pairs are .08165/.03811, .07635/.03588 and .07725/.03626; case 2 pairs are .25359/.10523, .16579/.05189 and .19455/.07426. Current refined RMS remains .0526-.1641, and the largest inspected refined-quote error is .5683 quote SD. Point roles are old neural/reference vectors, not current posterior means. This rescore ran no new Monte Carlo or MCMC; the separately completed current PyMC run is described above. No causal attribution solely to data count is established. Full pointwise t7 intervals and replicate covariance remain in S10.
- Ten local deterministic gradient checks pass: six known 50k and four joint 20k points agree across analytic derivatives, finite differences and compiled PyTensor log-density gradients. This verifies the checked implementation, not likelihood accuracy or conditional posterior agreement. Actual local platform/runtime metadata is retained in S10 and S12.

## Later known-curve 100k fit: matched learning comparison

The 100k fit completed. All 5,000 validation surfaces and the original 50k training prefix retain identical hashes; the same 2,000 validation cases, 1,000 draws and sampling seed are compared. Both fits use seed 0 and final epoch 500, but perform 98,000 versus 195,500 posterior updates, so this is neither a replicated nor fixed-compute learning curve. Both four-target rank families pass. Primary rank KS values become slightly smaller, while 90% coverage changes little; this does not establish a material posterior improvement. [S15,S16,S19]

The new price emulator regresses: label RMSE rises from 0.08287 to 0.09843 quote SD (18.8%), and independent coarse/refined RMS errors are higher at all six reused fixed points. It ran 116 epochs and selected 86, versus 300/298 for the 50k emulator. Different optimization histories are a confound, not a proven explanation or evidence that more data is harmful. [S15,S17]

Both saved neural estimators were also compared against exactly the same completed **50k-pricer** PyMC chains, observations and curves, with 1,000 neural draws each and no new MCMC. Both pass 2/3 fixed-case parameter screens; case 2 still fails H and eta. Case-2 H/eta/rho mean discrepancies change from .822/.789/.290 to .565/.662/.103 reference SD, with width ratios 1.814/1.318/1.423 to 1.587/1.293/1.169. Downstream rho x eta mean discrepancy changes .103 to .259 SD while its width ratio improves 1.451 to 1.188. This holds the approximate reference fixed; the new 100k pricer is not that reference. Neural and reference-quantile Monte Carlo errors remain unassessed. No scientific acceptance or cache promotion follows from these development results. [S18]

## Reproduce

Use a plotting/PDF environment with numpy, matplotlib, reportlab and pypdf (PyMuPDF optional for page rendering). Run `python scripts/render_inference_performance_report.py --source-ref feat/parallel-tutorial-delivery` from the repository, optionally replacing the ref with an immutable source commit. This reads retained evidence and generates figures, a PDF in `output/pdf/`, the shareable PDF here, this companion and source-hash inventory. It runs no notebooks, models, simulations or inference. Temporary dependencies used for the delivered rendering were isolated from the project's locked environments; exact renderer versions are recorded in the source inventory.

## Exact sources

"""
    for a,b in {"Heston100k":"Heston 100k","final500":"final 500","rough50k":"rough 50k","rough20k":"rough 20k","known-curve100k":"known-curve 100k","reported50k":"reported 50k","is52.25":"is 52.25","eta95%":"eta 95%","is93.85":"is 93.85","decay50%":"decay 50%","is46.75":"is 46.75","eta95%":"eta 95%","is94.00":"is 94.00","variance80/90/95%":"variance 80/90/95%","is82.3":"is 82.3","interior4D":"interior 4D","R-hat1.00297":"R-hat 1.00297","bulkESS1,908":"bulk ESS 1,908","difference.1013referenceSD":"difference .1013 reference SD","and90%width":"and 90% width","ratios1.016":"ratios 1.016","All12":"All 12","is.08287quoteSD":"is .08287 quote SD","known50k":"known 50k","and.10118":"and .10118","joint20k":"joint 20k","are1.311":"are 1.311","and3.888":"and 3.888","surfaceRMS":"surface RMS","is1.522":"is 1.522","known20k":"known 20k","passed1/3cases":"passed 1/3 cases","to.742":"to .742","and.276quoteSD":"and .276 quote SD","current50k":"current 50k"}.items():text=text.replace(a,b)
    for code,title,path in SOURCES:
        text+=f"- [{code}] [{title}](../../{path}).\n"
    (REPORTS/"inference-performance-2026-10-28.md").write_text(text)
    model_inventory=[{"model":d["name"],"training_surfaces":d["train"],"evaluated_split":d["raw"]["evaluated_split"],"checkpoint_sha256":d["raw"]["checkpoint_sha256"],"report_role":"baseline figures"} for d in data]
    model_inventory.append({"model":"Rough Bergomi / known curve","training_surfaces":100000,"evaluated_split":learning_evaluation["evaluated_split"],"checkpoint_sha256":learning_evaluation["checkpoint_sha256"],"report_role":"later fit / matched appendix comparison"})
    inventory={"report_date":"2026-10-28","source_link_base":GH,"renderer_versions":{n:version(n) for n in ["numpy","matplotlib","reportlab","pypdf"]},"models":model_inventory,
               "sources":{path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for _,_,path in SOURCES},
               "existing_figure":{"path":"evidence/pilot-results/2026-10-28/interior-corners/heston-interior-complete-corner.png","sha256":hashlib.sha256((RESULTS/"interior-corners/heston-interior-complete-corner.png").read_bytes()).hexdigest()},
               "generated_pdf_sha256":hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}
    (REPORTS/"inference-performance-2026-10-28-sources.json").write_text(json.dumps(inventory,indent=2)+"\n")


def main():
    global GH
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ref",default="feat/parallel-tutorial-delivery",help="GitHub source branch or immutable commit")
    args=parser.parse_args()
    GH=f"https://github.com/fmuia/sbi4finance/blob/{args.source_ref}/"
    data=datasets();make_figures(data);render(data);companion(data)
    from pypdf import PdfReader
    reader=PdfReader(OUTPUT)
    print(json.dumps({"pdf":str(OUTPUT),"shared":str(SHARED),"pages":len(reader.pages),
                      "page_text_characters":[len(p.extract_text()) for p in reader.pages]},indent=2))


if __name__=="__main__":main()
