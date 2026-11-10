#!/usr/bin/env python3
"""Create the companion proposal PDF; no task, model or inference run is changed.

Run from lab/ with reportlab, matplotlib, numpy and svglib available. Output
and source/QA sidecars are local generated files excluded from Git.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_heston_noise_report import (
    EVIDENCE, INK, LINE, MUTED, PAGE_H, PAGE_W, PALE, ROOT, TEAL, WIDTH,
    equation, fonts_and_styles, sha,
)
from reportlab.lib.pagesizes import A4
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


WEB_SOURCES = [
    ("1", "QuantLib: calibration residual definitions", "Price, relative-price and implied-volatility errors.",
     "https://github.com/lballabio/QuantLib/blob/master/ql/models/calibrationhelper.cpp"),
    ("2", "QuantLib: weighted calibration objective", "User-supplied weights in a sum of squared calibration errors.",
     "https://github.com/lballabio/QuantLib/blob/master/ql/models/model.cpp"),
    ("3", "OpenGamma Strata: SABR cap/floor calibrator", "Supplied errors are applied in price space in this calibrator.",
     "https://strata.opengamma.io/apidocs/com/opengamma/strata/pricer/capfloor/SabrIborCapletFloorletVolatilityCalibrator.html"),
    ("4", "Zeliade: Robust calibration and arbitrage-free interpolation of SSVI slices", "30 March 2018; sections 5.1 and 6.2.",
     "https://www.zeliade.com/wp-content/uploads/whitepapers/zwp-008-RobustNoArbSSVI.pdf"),
]


def build(output):
    styles = fonts_and_styles()
    sources = {
        "5a": ROOT / "lab/scripts/train_heston_pilot_campaign.py",
        "5b": ROOT / "lab/src/sbi4finance/calibration/heston_pilot.py",
        "6": EVIDENCE / "training-study/continuation-results.json",
        "7": EVIDENCE / "training-study/continuation-failure-audit.json",
    }
    current = json.loads(sources["6"].read_text())
    assert set(current["cells"]) == {"50000", "75000"}
    tau = current["cells"]["75000"]["market"]["parameters"]["tau"]
    assert abs(tau["reference_mean"]-.006091600653789715) < 1e-14
    assert abs(tau["neural_mean"]-.0014323803257571127) < 1e-14
    assert not current["scientific_acceptance"]

    story = []
    def p(text, style="body"):
        return Paragraph(text, styles[style])
    def add(text, style="body"):
        story.append(p(text, style))
    def title(n, text):
        add(f"{n:02d} / COMPANION PROPOSAL", "kicker")
        add(text, "h1")
    def heading(text):
        add(text, "h2")
    def box(text):
        t = Table([[p(text)]], colWidths=[WIDTH])
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),PALE),
            ("BOX",(0,0),(-1,-1),.55,LINE),
            ("LEFTPADDING",(0,0),(-1,-1),12), ("RIGHTPADDING",(0,0),(-1,-1),12),
            ("TOPPADDING",(0,0),(-1,-1),10), ("BOTTOMPADDING",(0,0),(-1,-1),5)]))
        story.append(t)
    def table(rows, widths):
        data = [[p(str(cell),"cellsmall") for cell in row] for row in rows]
        t = Table(data,colWidths=widths,repeatRows=1,hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),PALE),
            ("LINEBELOW",(0,0),(-1,0),.8,TEAL), ("LINEBELOW",(0,1),(-1,-1),.35,LINE),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("LEFTPADDING",(0,0),(-1,-1),8), ("RIGHTPADDING",(0,0),(-1,-1),8),
            ("TOPPADDING",(0,0),(-1,-1),6), ("BOTTOMPADDING",(0,0),(-1,-1),6)]))
        story.extend([t,Spacer(1,8)])
    def step(n, text, detail):
        add(f"<b>{n}. {text}</b> {detail}")

    # 1: recommendation and concrete contrast, with proposal status unmistakable.
    add("HESTON OPTIONS / TEACHING DESIGN / 9 NOVEMBER 2026", "kicker")
    add("A simpler observation-noise<br/>baseline", "title")
    add("A companion to <i>How we construct observation noise</i>. This report contrasts the current inferred-τ experiment with a proposed known-scale example, explains its practical calibration connection, and states the limits of that simplification.")
    box("<b>Recommendation:</b> infer the five Heston parameters using one Gaussian price-error model with a supplied, spread-based scale. Treat inferred discrepancy τ as a later extension. This is a proposal: the existing simulator, trained checkpoints and study evidence have not been changed.")
    heading("Current experiment: infer an additional discrepancy scale")
    story.append(equation(r"Y_i=P_i(\phi)+\sqrt{(c h_i)^2+(\tau\nu_i)^2}\,Z_i"))
    heading("Proposed baseline: supply the error scale")
    story.append(equation(r"Y_i=P_i(\phi)+s_iZ_i,\qquad s_i=c h_i,\qquad Z_i\sim\mathcal{N}(0,1)"))
    table([
        ["Aspect", "Current τ experiment", "Proposed teaching baseline"],
        ["Posterior targets", "Five physical parameters plus τ", "Five physical parameters only"],
        ["Observation SD", "Measurement and inferred discrepancy in quadrature", "Known c × quote half-spread"],
        ["Discrepancy scale", "τ ~ Uniform(0, 0.02), or 0-2 volatility points", "Additional discrepancy fixed to zero"],
        ["Role of c", "Fresh Uniform(0.5, 2), supplied as known", "Supplied as known; baseline 1, sensitivity 0.5 and 2"],
        ["Vega in error law", "Frozen market vega converts τ to price units", "Not required for the error law"],
        ["Main lesson", "Joint physical/discrepancy inference", "Amortized uncertainty under a declared observation scale"],
    ], [104,198,WIDTH-302])
    add("Notation: P is a clean Heston price; φ = (v₀, κ, θ, ξ, ρ); h is observed price half-spread; c is a known multiplier; ν is frozen price vega per unit decimal IV; τ is decimal-IV discrepancy SD. One volatility point is 0.01 decimal IV. The two formulas use the same fixed contract design and physical prior. [5a, 5b]")
    story.append(PageBreak())

    # 2: source-limited industry claims and the exact likelihood/optimization bridge.
    title(2,"The practical calibration connection")
    add("The defensible industry connection is to calibration losses, supplied weights and bid-ask fit tolerances. The checked sources do not establish one universal probability law for quote errors, nor a rule that half-spread equals one statistical SD.")
    heading("What the primary implementations demonstrate")
    add("<b>QuantLib:</b> its calibration helper supports price, relative-price and implied-volatility residuals. Its model calibration minimizes a weighted sum of squared residuals using supplied weights. This machinery does not require an inferred observation-noise parameter. [1, 2]")
    add("<b>OpenGamma Strata:</b> the cited SABR cap/floor calibrator accepts supplied errors and applies them in price space after converting volatility quotes to prices. This is evidence for error scales as calibration inputs; it is a rates example, not an equity-specific noise prescription. [3]")
    add("<b>Zeliade:</b> its eSSVI slice procedure describes absolute price differences as one suitable objective and assesses fitted surfaces against market bid-ask quotes, including reported exceptions. It illustrates a different loss and practical fit criterion, not a Gaussian likelihood requirement. [4]")
    heading("A Gaussian likelihood connects directly to least squares")
    add("Under the proposed law, with fixed positive s and a bounded Uniform physical prior, the physical-coordinate posterior mode minimizes:")
    story.append(equation(r"Q(\phi)=\sum_i\left[\frac{P_i(\phi)-y_i}{s_i}\right]^2,\qquad s_i=c h_i"))
    add("This is spread-weighted least squares. The probabilistic model also provides a distribution around the fit, conditional on the prior and assumed error scales. An informative prior would instead give a regularized objective. The physical-coordinate qualifier matters: a density mode can change under reparameterization.")
    heading("Relative weights do not determine absolute uncertainty")
    story.append(equation(r"Q_c(\phi)=\frac{1}{c^2}Q_1(\phi),\qquad\log L_c(\phi)=-\frac{Q_1(\phi)}{2c^2}+\mathrm{constant}"))
    add("For the same bounded Uniform prior, a global known c leaves the minimizing parameter set unchanged. It changes likelihood concentration and therefore the posterior. The same optimum does not imply the same posterior mean, median or interval. Exact proportional scaling of interval widths is not guaranteed in nonlinear or prior-truncated cases.")
    box("<b>The modelling choice remains explicit:</b> c = 1 is a transparent teaching convention, with c = 0.5 and 2 as sensitivity cases. Familiar calibration weights alone cannot certify the resulting Bayesian uncertainty.")
    story.append(PageBreak())

    # 3: why simplification is reasonable without a false diagnosis.
    title(3,"Why simplify, and what this does not prove")
    heading("The original Gaussian form was not the central weakness")
    add("A Gaussian working likelihood is a simple probabilistic version of least-squares calibration. The historical example's weaker choices were its fixed synthetic noise profile, fixed-20%-vega conversion and limited justification for the absolute scale. The proposal keeps a simple Gaussian form while making the price scale traceable to observed spreads.")
    heading("Two components do not automatically imply nonidentifiability")
    story.append(equation(r"s_i^2=(c h_i)^2+\tau^2\nu_i^2"))
    add("In the current experiment, h and ν are fixed and c is supplied. Only τ is an unknown noise scale. Thus we are not jointly estimating two unrestricted noise variances. If c and τ were both unknown, they could trade off; proportional spread and vega profiles would make their separation impossible from these variances alone.")
    add("The market reference posterior actually constrains τ reasonably tightly. Under the specified likelihood its mean is about 0.00609 and central 90% interval is [0.00516, 0.00722]. The 75k neural mean is about 0.00143. This is demonstrated neural approximation error for that target, not proof that τ is intrinsically unidentifiable. [6]")
    heading("The teaching and engineering argument for dropping τ")
    step(1,"One fewer posterior target.","The network can focus on physical parameter uncertainty under known observation scales.")
    step(2,"A direct calibration benchmark.","The same scales define an interpretable weighted-least-squares comparator.")
    step(3,"A narrower scientific claim.","We illustrate inference conditional on a declared quote-error model, without also claiming to estimate structural model discrepancy.")
    step(4,"An informative isolation experiment.","Holding the current contracts and physical prior fixed separates the extra scale-inference task from other changes made since the old example.")
    box("<b>Simpler does not guarantee easier optimization.</b> Removing positive τ reduces total SD at the same c and can sharpen physical-parameter posteriors. It does not establish identification of all five physical parameters, fix market mismatch, or explain the non-finite loss detected at 88k. That failure still needs its own diagnosis. [7]")
    story.append(PageBreak())

    # 4: specify a concrete bounded experiment without claiming it happened.
    title(4,"The proposed experiment")
    add("Keep the existing 54 European SPXW quote locations, carry conventions, frozen half-spreads and five-parameter prior. Use the current wide vol-of-vol support and numerical controls. Change the observation law and posterior target only. [5a, 5b]")
    heading("Simulation and amortized training")
    step(1,"Select a clean surface.","Use a five-parameter draw and its numerical prices from the preserved clean-price bank.")
    step(2,"Supply a known noise multiplier.","For a network covering specified scales, draw c from Uniform(0.5, 2) and provide it as conditioning information. It is not an unknown latent parameter with an inferred posterior.")
    step(3,"Add fresh Gaussian quote errors.","Use SD s<sub>i</sub> = c h<sub>i</sub>, with independent new standard-Normal errors for the 54 quotes on each exposure.")
    step(4,"Infer the five physical parameters.","Train a new checkpoint for p(φ | y, c, fixed design). No τ target or discrepancy-vega term is used in this baseline.")
    heading("Matched checks, in a bounded sequence")
    table([
        ["Check", "Purpose"],
        ["Synthetic Heston controls", "Known physical truth; measure ranks, coverage and parameter/product recovery under the stated simulator."],
        ["Reference posteriors", "Compare the neural posterior with the same five-target likelihood and prior at declared known c values."],
        ["Weighted least squares", "Verify the physical-coordinate mode/fit relationship, while keeping interval and mode checks distinct."],
        ["Full versus restricted quotes", "Withhold later smile quotes while retaining the same carry/design conditioning and surviving error scales."],
        ["Noise sensitivity", "Compare c = 0.5, 1 and 2; inspect inference quality and posterior sensitivity separately."],
        ["Real market application", "Inspect spread-unit residuals and held-out predictions; report model inadequacy if present."],
    ], [143,WIDTH-143])
    heading("Reuse prices, establish a new inference identity")
    add("The observation-law change alone needs no new Heston pricing labels if contracts, physical prior and accuracy requirements stay unchanged. Existing bank prices were gated against 0.5 h, independent of τ. Nevertheless, the proposed likelihood requires new matched reference posteriors, a distinct task identity and a newly trained five-target checkpoint.")
    box("Discarding τ coordinates from old six-target posterior draws is not this baseline: it marginalizes over τ rather than fixing its discrepancy contribution to zero. Preserve the current results and checkpoints as their original experiment. No implementation, training or reference run is performed by this report.")
    story.append(PageBreak())

    # 5: acceptance boundaries and short, clickable primary-source references.
    title(5,"Interpretation limits and sources")
    heading("A practical example without overstating its evidence")
    add("<b>First establish inference correctness on synthetic controls.</b> Realistic contracts and observed spread profiles make the design practical, while synthetic Heston prices provide known truth. This can demonstrate amortization, parameter dependence, the effect of missing quote information and uncertainty propagated to another product.")
    add("<b>Then assess the actual market separately.</b> The pilot already has substantial structured residuals in spread units. Removing τ changes the calibration weights, so refitting can improve a chosen residual metric. It does not enlarge Heston's attainable price family or guarantee an adequate fit. A narrow posterior around a poor fit must not be presented as calibrated real-world confidence or a model-risk reserve.")
    add("<b>Keep scale sensitivity visible.</b> Interpreting half-spread as Gaussian SD is a stated convention. Independent errors do not represent every timing, liquidity, carry or structural-model issue. If the application needs defensible real-world uncertainty beyond that convention, it needs additional data or a separately justified observation/discrepancy model.")
    add("<b>Revisit τ only for a defined extension.</b> Estimating additional discrepancy can be worthwhile once the simpler benchmark works and the teaching question requires it. Such an extension should address residual structure and validation, rather than use a larger error allowance to make posterior intervals look useful.")
    heading("Primary sources checked on 9 November 2026")
    for key,label,detail,url in WEB_SOURCES:
        add(f"<b>[{key}] <link href='{url}' color='#087e88'>{label}</link></b><br/>{detail}","small")
    add("The library and practitioner sources support calibration conventions, not a universal Bayesian noise prescription. The proposed Gaussian law, half-spread conversion and c sensitivity are the report's explicit teaching recommendations.","small")
    heading("Local evidence and reproducibility")
    for key,path in sources.items():
        label = {"5a":"Current task, priors and fresh-noise training", "5b":"Current price checks and likelihood",
                 "6":"Current 50k/75k development comparisons", "7":"Preserved numerical-failure audit"}[key]
        relative=str(path.relative_to(ROOT)).replace("/","/<wbr/>")
        add(f"<b>[{key}] {label}</b><br/>{relative}","tiny")
    add("Source/evidence base: study/heston-european-observations-20261109, through commit 52da9ee. The adjacent .sources.json records exact local source hashes. This companion leaves the earlier first-principles PDF intact.","tiny")

    output.parent.mkdir(parents=True,exist_ok=True)
    def decorate(canvas,doc):
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.line(48,PAGE_H-35,PAGE_W-48,PAGE_H-35)
        canvas.line(48,36,PAGE_W-48,36)
        canvas.setFont("Body",8);canvas.setFillColor(MUTED)
        canvas.drawString(48,PAGE_H-25,"HESTON / SIMPLER NOISE BASELINE")
        canvas.drawRightString(PAGE_W-48,PAGE_H-25,"Companion proposal - 9 November 2026")
        canvas.drawString(48,23,"Proposed specification | No new training or inference performed")
        canvas.drawRightString(PAGE_W-48,23,str(doc.page))
        canvas.restoreState()
    doc=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=55,bottomMargin=49,
        title="A simpler observation-noise baseline for Heston",author="SBI4Finance",
        subject="Contrast of inferred discrepancy and supplied spread-based noise for amortized inference")
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
    report={"format":"heston_simpler_noise_proposal_pdf_sources_v1","status":"proposal_not_implemented",
        "pdf_sha256":sha(output),"renderer_sha256":sha(__file__),
        "rendering_helper_sha256":sha(ROOT/"lab/scripts/render_heston_noise_report.py"),
        "local_inputs":{key:{"path":str(path.relative_to(ROOT)),"sha256":sha(path)} for key,path in sources.items()},
        "web_sources":[{"id":key,"title":label,"url":url,"checked":"2026-11-09"} for key,label,_,url in WEB_SOURCES]}
    output.with_suffix('.sources.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'output/pdf/heston-simpler-noise-baseline-proposal.pdf')
    args=parser.parse_args();report=build(args.out)
    print(json.dumps({"pdf":str(args.out),"sha256":report['pdf_sha256']}))


if __name__=='__main__':
    main()
