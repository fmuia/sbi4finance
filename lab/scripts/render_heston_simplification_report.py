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
    ("3", "OpenGamma Strata: RawOptionData", "Optional supplied measurement errors for least-squares calibration.",
     "https://strata.opengamma.io/apidocs/com/opengamma/strata/pricer/option/RawOptionData.html"),
    ("4", "OpenGamma Strata: SABR cap/floor calibrator", "Supplied errors are applied in price space in this rates calibrator.",
     "https://strata.opengamma.io/apidocs/com/opengamma/strata/pricer/capfloor/SabrIborCapletFloorletVolatilityCalibrator.html"),
    ("5", "QuantFlow: Heston calibration API", "ResidualKind and HestonCalibration: price/IV modes and capped price-mode weights.",
     "https://quantflow.quantmind.com/api/options/calibration/#quantflow.options.calibration.heston.HestonCalibration"),
    ("6", "LSEG: IPA Volatility Surfaces for Exchange-Traded Instruments", "Surface parameters: weights and applyVegaWeights.",
     "https://developers.lseg.com/en/api-catalog/refinitiv-data-platform/refinitiv-data-platform-apis/documentation/manuals-and-guides/ipa-volatility-surfaces/ipa-volatility-surfaces-eti"),
    ("7", "Zeliade: Robust calibration and arbitrage-free interpolation of SSVI slices", "30 March 2018; sections 5.1 and 6.2.",
     "https://www.zeliade.com/wp-content/uploads/whitepapers/zwp-008-RobustNoArbSSVI.pdf"),
]


def build(output):
    styles = fonts_and_styles()
    sources = {
        "A": ROOT / "lab/scripts/train_heston_pilot_campaign.py",
        "B": ROOT / "lab/src/sbi4finance/calibration/heston_pilot.py",
        "C": EVIDENCE / "training-study/continuation-results.json",
        "D": EVIDENCE / "training-study/continuation-failure-audit.json",
        "E": ROOT / "tutorial/src/sbi4f_tutorial/shared/surface.py",
        "F": ROOT / "tutorial/assets/backbones/heston-final500/task.json",
        "G": ROOT / "tutorial/src/sbi4f_tutorial/shared/full_surface.py",
    }
    current = json.loads(sources["C"].read_text())
    assert set(current["cells"]) == {"50000", "75000"}
    tau = current["cells"]["75000"]["market"]["parameters"]["tau"]
    assert abs(tau["reference_mean"]-.006091600653789715) < 1e-14
    assert abs(tau["neural_mean"]-.0014323803257571127) < 1e-14
    assert not current["scientific_acceptance"]
    old = json.loads(sources["F"].read_text())
    assert old["parameter_names"] == ["initial_variance", "theta", "vol_of_vol", "rho"]
    assert old["fixed_parameters"]["kappa"] == 2.
    assert len(old["context"]["quote_noise_std"]) == 63

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
    add("A companion to <i>How we construct observation noise</i>. This revised discussion note explains the return to the original known-noise model class, documents calibration choices in finance libraries, and proposes a paired example that estimates one global noise scale.")
    box("<b>Recommendation:</b> use a known, spread-based Gaussian scale as the core example. Add a controlled comparison that estimates a single global scale σ under the same error law. Keep the two-component τ model as a separate extension. These are proposals; no simulator, checkpoint or study evidence is changed by this report.")
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
    add("Notation: P is a clean Heston price; φ = (v₀, κ, θ, ξ, ρ); h is observed price half-spread; c is a known multiplier; ν is frozen price vega per unit decimal IV; τ is decimal-IV discrepancy SD. One volatility point is 0.01 decimal IV. The paired σ example on page 7 estimates a dimensionless multiplier of h, not a second additive component. [A, B]")
    story.append(PageBreak())

    # 2: explicitly answer how the proposal differs from the original lesson.
    title(2,"How this differs from the original example")
    box("<b>At the model-class level, the core proposal is a return to the original setup:</b> independent Gaussian price errors with known SDs. The substantive noise change is the rule used to choose those SDs. It is not a fundamentally new distribution for observation errors.")
    story.append(equation(r"Y_i=P_i(\phi)+s_iZ_i,\qquad Z_i\sim\mathcal{N}(0,1)"))
    table([
        ["Question", "Original published example", "Proposed core example"],
        ["Gaussian errors?", "Yes", "Yes"],
        ["SD known to inference?", "Yes", "Yes"],
        ["Infer noise scale?", "No", "No; estimated σ is a separate paired experiment"],
        ["How choose SD?", "Synthetic IV profile times vega at 20%", "Known c times observed price half-spread"],
        ["Fresh quote errors?", "Yes, with the fixed SD profile", "Yes, with the supplied SD profile"],
        ["Range of SD levels?", "One fixed profile for the task", "Optional known c across training; supplied with each case"],
    ],[110,195,WIDTH-305])
    heading("The exact change in the SD rule")
    story.append(equation(r"s_i^{\mathrm{old}}=\nu_i^{20\%}\left[0.003+0.007\frac{\min(|z_i|,2)}{2}\right]",16))
    story.append(equation(r"s_i^{\mathrm{proposed}}=c h_i"))
    add("The original profile prescribed 0.3 volatility points at ATM and 1.0 in the wings, converted using fixed-20% Black vega. z is standardized log-moneyness, not option delta. The proposed scale uses actual price spreads and a declared multiplier; it still does not establish that half-spread is an empirical Gaussian SD. [E-G]")
    heading("Other changes are separate from the noise simplification")
    add("The original task inferred four physical parameters with κ = 2 fixed and ξ ≤ 1.5, on 63 normalized-grid contracts. The revised pilot uses 54 actual European SPXW quote locations, infers κ and allows ξ ≤ 3. These changes can be retained independently of the observation-noise choice. Synthetic prices were computed directly on the old grid; interpolation applied to old market readings.","small")
    box("<b>Keep:</b> the original Gaussian mechanism. <b>Replace:</b> its prescribed SD profile with a spread-anchored convention. <b>Undo:</b> the additional inferred τ term in the core example. The proposed direct-σ exercise then tests learning one scale under that same simple error law.")
    story.append(PageBreak())

    # 3: document diverse choices without claiming they establish a noise law.
    title(3,"Finance libraries offer calibration choices")
    add("The primary sources below expose different residuals, weights and fit criteria. They do not identify a single obvious Gaussian noise law or an industry-wide conversion from bid-ask spread to statistical SD. They support making a documented choice and checking its consequences.")
    table([
        ["Implementation", "Documented choice", "What this supports"],
        ["QuantLib [1, 2]", "Price, relative-price or IV residuals; user-supplied squared-error weights", "Residual units and calibration weights are configurable."],
        ["OpenGamma Strata [3, 4]", "Optional supplied measurement errors; the cited SABR cap/floor calibrator applies errors in price space", "Error scales can be calibration inputs. This cited application is in rates."],
        ["QuantFlow Heston [5]", "Forward-price or Black-IV residuals; capped moneyness weights in price mode, omitted in IV mode", "Even a Heston library offers alternative deterministic calibration objectives."],
        ["LSEG SVI API [6]", "Weights by moneyness range, or optional dynamic vega weights; dynamic mode is off by default", "Commercial software also exposes weighting choices. The cited page does not specify its residual formula."],
        ["Zeliade eSSVI [7]", "Absolute price differences as one suitable objective; fit assessed against bid-ask quotes", "A practitioner method can use a different loss and a practical fit tolerance."],
    ],[108,220,WIDTH-328])
    heading("The scope of the evidence")
    add("QuantLib and QuantFlow document Heston-compatible calibration machinery. Strata's cited calibrator is for SABR caps/floors; LSEG and Zeliade concern SVI/eSSVI surfaces. These are relevant examples of calibration practice, not interchangeable model specifications or a survey proving what every desk does.")
    heading("Why the phrase 'vega weighting' needs units")
    add("Dividing a price residual by vega locally approximates an IV residual. Weighting an IV residual by vega has a different effect. A multiplier applied before squaring is also different from a weight on the squared residual. We therefore compare documented objectives and avoid translating an API's named weight into a Bayesian σ without its precise definition.")
    box("<b>There is no source-backed unique choice here.</b> That does not make every choice equally useful. We recommend a simple, observable price-spread anchor, explicit Gaussian assumptions and sensitivity checks. The absolute Bayesian uncertainty still requires justification and validation.")
    story.append(PageBreak())

    # 4: retain the exact likelihood/optimization bridge as its own readable page.
    title(4,"Calibration weights versus Bayesian uncertainty")
    add("The original Gaussian form was not itself the central weakness. A Gaussian working likelihood is a transparent probabilistic counterpart to least-squares calibration. The prescribed error magnitudes and fixed reference-vega conversion needed better justification.")
    heading("A Gaussian likelihood connects directly to least squares")
    add("Under the proposed law, with fixed positive s and a bounded Uniform physical prior, the physical-coordinate posterior mode minimizes:")
    story.append(equation(r"Q(\phi)=\sum_i\left[\frac{P_i(\phi)-y_i}{s_i}\right]^2,\qquad s_i=c h_i"))
    add("This is spread-weighted least squares. The probabilistic model also provides a distribution around the fit, conditional on the prior and assumed error scales. An informative prior would instead give a regularized objective. The physical-coordinate qualifier matters: a density mode can change under reparameterization.")
    heading("Relative weights do not determine absolute uncertainty")
    story.append(equation(r"Q_c(\phi)=\frac{1}{c^2}Q_1(\phi),\qquad\log L_c(\phi)=-\frac{Q_1(\phi)}{2c^2}+\mathrm{constant}"))
    add("For the same bounded Uniform prior, a global known c leaves the minimizing parameter set unchanged. It changes likelihood concentration and therefore the posterior. The same optimum does not imply the same posterior mean, median or interval. Exact proportional scaling of interval widths is not guaranteed in nonlinear or prior-truncated cases.")
    box("<b>The modelling choice remains explicit:</b> c = 1 is a transparent teaching convention, with c = 0.5 and 2 as sensitivity cases. Familiar calibration weights alone cannot certify the resulting Bayesian uncertainty.")
    story.append(PageBreak())

    # 5: simplification is a design choice, not an established failure diagnosis.
    title(5,"Why simplify, and what this does not prove")
    heading("Two components do not automatically imply nonidentifiability")
    story.append(equation(r"s_i^2=(c h_i)^2+\tau^2\nu_i^2"))
    add("In the current experiment, h and ν are fixed and c is supplied. Only τ is an unknown noise scale. Thus we are not jointly estimating two unrestricted noise variances. If c and τ were both unknown, they could trade off; proportional spread and vega profiles would make their separation impossible from these variances alone.")
    add("The market reference posterior actually constrains τ reasonably tightly. Under the specified likelihood its mean is about 0.00609 and central 90% interval is [0.00516, 0.00722]. The 75k neural mean is about 0.00143. This is demonstrated neural approximation error for that target, not proof that τ is intrinsically unidentifiable. [C]")
    heading("The teaching and engineering argument for dropping τ")
    step(1,"One fewer posterior target.","The network can focus on physical parameter uncertainty under known observation scales.")
    step(2,"A direct calibration benchmark.","The same scales define an interpretable weighted-least-squares comparator.")
    step(3,"A narrower scientific claim.","We illustrate inference conditional on a declared quote-error model, without also claiming to estimate structural model discrepancy.")
    step(4,"An informative isolation experiment.","Holding the current contracts and physical prior fixed separates the extra scale-inference task from other changes made since the old example.")
    add("The direct-σ experiment also has six targets, so it is not a dimension-reduction claim relative to the τ experiment. Its distinction is one unknown amplitude multiplying a fixed spread profile, rather than an additional vega-shaped component. It provides a cleaner comparison of known versus learned scale.")
    box("<b>Simpler does not guarantee easier optimization.</b> Removing positive τ reduces total SD at the same c and can sharpen physical-parameter posteriors. It does not establish identification of all five physical parameters, fix market mismatch, or explain the non-finite loss detected at 88k. That failure still needs its own diagnosis. [D]")
    story.append(PageBreak())

    # 6: retain a concrete bounded sequence, with the paired extension next.
    title(6,"The proposed core experiment")
    add("Keep the existing 54 European SPXW quote locations, carry conventions, frozen half-spreads and five-parameter prior. Use the current wide vol-of-vol support and numerical controls. Change the observation law and posterior target only. [A, B]")
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

    # 7: the requested single-scale estimation example, with no scale leakage.
    title(7,"Estimate σ, or vary an assumed σ?")
    add("Use the same one-component observation law in both exercises. Define a <b>dimensionless global scale σ</b> relative to the known half-spread profile:")
    story.append(equation(r"Y_i=P_i(\phi)+\sigma h_i Z_i,\qquad Z_i\sim\mathcal{N}(0,1)"))
    add("Each quote's price SD is s<sub>i</sub> = σ h<sub>i</sub>; after dividing by h<sub>i</sub>, its error is simply σ Z<sub>i</sub>. Thus σ is a common SD in spread units. It replaces c and is distinct from Heston's ξ or the IV-equivalent τ. The spread profile stays known.")
    table([
        ["Question", "Fixed-scale sensitivity", "Direct scale estimation"],
        ["Treatment of σ", "Supply 0.5, 1 or 2 as known", "Assign a prior and infer σ"],
        ["Posterior", "p(φ | y, σ)", "p(φ, σ | y), then marginalize σ"],
        ["Network conditions", "Prices, known spread/design and supplied σ", "Prices and known spread/design; hide the generating σ"],
        ["Meaning", "How conclusions change under specified error-scale assumptions", "What the assumed model and data say about the scale and physical parameters together"],
    ],[100,199,WIDTH-299])
    heading("A minimal controlled example")
    add("Use σ ~ Uniform(0.5, 2), independent of φ, matching the core network's scale range. This is a declared illustration prior, not an industry-measured distribution. Generate one synthetic dataset at σ* = 1 and hold its prices fixed. Compare the three conditional posteriors with the joint inferred-σ posterior against matched references. Fresh repeated datasets are a separate calibration check.")
    heading("How the data learn σ")
    story.append(equation(r"R(\phi)=\sum_i\left[\frac{y_i-P_i(\phi)}{h_i}\right]^2,\qquad\log L=-n\log\sigma-\frac{R(\phi)}{2\sigma^2}+\mathrm{constant}",15))
    add("Here n counts quotes (54 in the full panel). For fixed φ, the Uniform-σ conditional mode is √(R/n), clipped to [0.5, 2]. The log σ term penalizes larger scales. In the joint network, σ is a target only; neither σ nor log(σh) may be conditioning inputs.","small")
    box("Sensitivity compares separate conditional assumptions. Inferring σ updates a prior using the data; the physical marginal mixes p(φ | y, σ) with <i>posterior</i> weights p(σ | y), not an equal average of three sensitivity plots. Its intervals need not be wider than every fixed-σ case. Boundary mass can flag a restrictive prior or error/model mismatch; a large fitted scale is not automatically pure measurement noise.")
    story.append(PageBreak())

    # 8: concise acceptance limits plus all portable, clickable primary sources.
    title(8,"Interpretation limits and sources")
    add("Establish inference correctness first on synthetic Heston controls with realistic contracts and spread profiles. Then inspect the actual market residuals separately. Changing the error law can improve a chosen weighted fit, but does not enlarge Heston's attainable price family or guarantee model adequacy. A tight posterior around a poor fit is not calibrated real-world confidence or a model-risk reserve.","small")
    add("Known-scale sensitivity and inferred-scale uncertainty are both useful, but answer different questions. Keep the Gaussian assumption, prior and spread-to-SD convention explicit. Learning σ or τ does not validate those assumptions. No new training, reference computation or model implementation is reported here.","small")
    heading("Primary sources checked on 9 November 2026")
    for key,label,detail,url in WEB_SOURCES:
        add(f"<b>[{key}] <link href='{url}' color='#087e88'>{label}</link></b><br/>{detail}","small")
    add("The library and practitioner sources support calibration conventions, not a universal Bayesian noise prescription. The proposed Gaussian law, half-spread conversion and c sensitivity are the report's explicit teaching recommendations.","small")
    heading("Local evidence and reproducibility")
    for key,path in sources.items():
        label = {"A":"Current task, priors and fresh-noise training", "B":"Current price checks and likelihood",
                 "C":"Current 50k/75k development comparisons", "D":"Preserved numerical-failure audit",
                 "E":"Original synthetic noise profile", "F":"Original four-target Heston task",
                 "G":"Original known-noise conditioning and augmentation"}[key]
        relative=str(path.relative_to(ROOT)).replace("/","/<wbr/>")
        add(f"<b>[{key}] {label}</b><br/>{relative}","tiny")
    add("Source/evidence base: study/heston-european-observations-20261109, evidence through commit 1ae1c59. The adjacent .sources.json records exact local source hashes. This revision leaves the earlier first-principles PDF intact.","tiny")

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
        "web_sources":[{"id":key,"title":label,"url":url,"checked":"2026-11-09"} for key,label,_,url in WEB_SOURCES],
        "paired_scale_example":{"status":"proposal_not_run","sigma_units":"dimensionless multiplier of known half-spread",
            "proposed_prior":{"family":"Uniform","bounds":[.5,2.]},"fixed_sensitivity_values":[.5,1.,2.],
            "same_observations_across_comparison":True,"sigma_hidden_from_joint_network_conditions":True}}
    output.with_suffix('.sources.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'output/pdf/heston-simpler-noise-baseline-proposal.pdf')
    args=parser.parse_args();report=build(args.out)
    print(json.dumps({"pdf":str(args.out),"sha256":report['pdf_sha256']}))


if __name__=='__main__':
    main()
