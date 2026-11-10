#!/usr/bin/env python3
"""Create the companion proposal PDF; no task, model or inference run is changed.

Run from lab/ with reportlab, matplotlib, numpy and svglib available. Output
and source/QA sidecars are local generated files excluded from Git.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_heston_noise_report import (
    EVIDENCE, INK, LINE, MUTED, PAGE_H, PAGE_W, PALE, ROOT, TEAL, WIDTH,
    VectorArt, equation, fonts_and_styles, sha,
)
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from svglib.svglib import svg2rlg


PR_HEAD = "df2cfc442de07459a8b971f070e3a6f25b0b2020"
E_TARGETS = [-5, -3.5, -2.25, -1.25, -.5, 0, .5, 1, 2]


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
    ("8", "SBI4Finance PR #81: option application plan", "Collaborator proposal; reviewed at commit df2cfc4. The report adopts selected coverage ideas, not the full application scope.",
     "https://github.com/fmuia/sbi4finance/pull/81"),
    ("9", "PR #81: pinned observation and error-model plan", "Plan sections 3-4; written layout E uses fixed 16% reference volatility.",
     f"https://github.com/fmuia/sbi4finance/blob/{PR_HEAD}/lab/plans/OPTION-APPLICATION-PLAN.md"),
    ("10", "PR #81: pinned layout comparison", "layout_choice.py calls the supporting grid constructor; local information comparisons are preliminary.",
     f"https://github.com/fmuia/sbi4finance/blob/{PR_HEAD}/lab/evidence/pilot-results/2026-11-09/design-and-error/heston/layout_choice.py"),
    ("11", "PR #81: pinned grid and error-covariance constructor", "grid_vs_error_structure.py computes fitted ATM volatilities and uses them for its ATM-scaled candidates.",
     f"https://github.com/fmuia/sbi4finance/blob/{PR_HEAD}/lab/evidence/pilot-results/2026-11-09/design-and-error/heston/grid_vs_error_structure.py"),
]


def grid_comparison(old, design):
    """Show physical layouts in one coordinate; candidate points are only targets."""
    old_t = np.array([v["maturity"] for v in old["context"]["contracts"]])
    old_k = np.log([v["strike"] for v in old["context"]["contracts"]])
    records = design["records"]
    current_t = np.asarray(records["maturity"])
    current_k = np.log(np.asarray(records["strike"])/np.asarray(records["forward"]))
    candidate_t = np.repeat(np.array([5, 10, 21, 42, 63, 126, 252])/252, 9)
    candidate_z = np.tile(E_TARGETS, 7)
    panels = [
        (old_k/(.16*np.sqrt(old_t)), old_t, "Original: 63 contracts", "#526675", "o"),
        (current_k/(.16*np.sqrt(current_t)), current_t, "Current: 54 actual quotes", "#087e88", "o"),
        (candidate_z, candidate_t, "Proposed: nominal targets", "#87511c", "x"),
    ]
    with plt.rc_context({"font.family":"DejaVu Sans", "font.size":8,
                         "svg.fonttype":"path", "axes.spines.top":False,
                         "axes.spines.right":False}):
        fig, axes = plt.subplots(1, 3, figsize=(6.85, 2.85), sharex=True, sharey=True)
        for ax, (z, t, label, color, marker) in zip(axes, panels):
            ax.scatter(z, 365*t, s=14, marker=marker, color=color, linewidths=.8)
            ax.axvline(0, color="#b7c9ce", linewidth=.7)
            ax.set_yscale("log")
            ax.set(xlim=(-5.4, 2.9), ylim=(6, 440), title=label)
            ax.set_xticks([-5, -3, -1, 0, 2])
            ax.set_yticks([7, 14, 30, 60, 90, 180, 365],
                          labels=["7", "14", "30", "60", "90", "180", "365"])
            ax.minorticks_off()
            ax.grid(alpha=.16)
        axes[0].set_ylabel("Time to expiry (calendar-day equivalent)")
        fig.supxlabel("Common coordinate: log(K/F) / (0.16 × √T)", fontsize=8, y=.015)
        fig.tight_layout(pad=.75, rect=(0, .065, 1, 1))
        buffer = BytesIO()
        fig.savefig(buffer, format="svg")
        plt.close(fig)
    buffer.seek(0)
    return VectorArt(svg2rlg(buffer), extra=5)


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
        "H": EVIDENCE / "designs/put_heavy.json",
        "I": ROOT / "lab/plans/SPARSE-QUOTE-CASE-PLAN.md",
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
    design = json.loads(sources["H"].read_text())
    assert len(design["records"]["mid"]) == 54
    assert design["targets_z"] == [-4, -3, -2, -1.5, -1, -.5, 0, .5, 1]
    times = np.unique(design["records"]["maturity"])
    assert len(times) == 6
    assert 21 < 365*times[0] < 22 and 324 < 365*times[-1] < 326

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
    add("Proposal for collaborator review: retain a simple Gaussian error model, compare supplied versus inferred scale, and test a fixed grid with better maturity and strike coverage. Includes calibration-library evidence and a review of PR #81.")
    box("<b>Recommendation:</b> use known, spread-based Gaussian SDs for the core example, then estimate one global scale σ in a paired exercise. Test a modest fixed-grid coverage upgrade separately. Keep the two-component τ model and smooth discrepancy as later extensions. This report proposes work; it reports no new training or pricing study.")
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
    add("The original task inferred four physical parameters with κ = 2 fixed and ξ ≤ 1.5, on 63 normalized-grid contracts. The revised pilot uses 54 actual European SPXW quote locations, infers κ and allows ξ ≤ 3. These changes are separate from the noise choice. Synthetic prices were computed directly on the old grid; interpolation applied to old market readings. Pages 8-11 compare the grids and propose a staged follow-up.","small")
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
    title(6,"Stage A: isolate the noise change")
    add("First compare supplied and inferred scale on the preserved 54-contract SPXW design, using its prior, carry, spreads and numerical controls. Audit the revised grid separately before repeating the pair. [A, B]","small")
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
    add("A noise-law change alone needs no new prices when contracts, physical prior and accuracy requirements stay unchanged. The existing bank was gated against 0.5 h, independently of τ. New matched references, task identities and checkpoints are still required.","small")
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

    # 8: compare all three stages without conflating coordinate labels and contracts.
    title(8,"The grid: original, current, proposed")
    add("The observation grid specifies <b>which European contracts we observe</b>: their strikes and expiries. It is separate from the Fourier integration grid used to compute their prices. Changing quote selection and changing observation noise answer different questions.")
    story.append(equation(r"z=\frac{\log(K/F)}{a\sqrt{T}},\qquad K=F\exp(a z\sqrt{T})"))
    add("K is strike, F is the expiry's forward, T is time in years and a is a reference volatility. The coordinate scale a = 16% is unrelated to the dimensionless noise multiplier σ. Negative z selects puts below the forward; positive z selects calls above it. z is <b>not option delta</b> or a tail probability. At z = 0, K = F: ATM here means at the forward.","small")
    table([
        ["Aspect", "Published original", "Current Heston pilot", "Proposed candidate"],
        ["Contracts", "63 normalized contracts; 7 × 9", "54 actual SPXW quotes; 6 × 9", "About 63 actual SPXW quotes; 7 × 9 if supported"],
        ["Strike selection", "Symmetric targets −2 to +2 at fixed 20%", "Targets −4 to +1, scaled once by observed ATM IV", "Uneven targets −5 to +2 at fixed 16%"],
        ["Expiries", "5,10,21,42,63,126,252 divided by 252 years", "30 Nov 2026 to 30 Sep 2027; ACT/365F", "About 1 and 2 weeks; 1, 2, 3, 6 and 12 months; exact dates to select"],
        ["Market observation", "Interpolate the chain onto the fixed synthetic grid", "Select actual quotes and preserve spreads", "Select distinct actual quotes near declared targets"],
        ["During simulation", "Fixed normalized contracts", "Fixed selected contracts", "Fixed selected contracts"],
    ],[77,137,139,WIDTH-353])
    heading("Compare physical reach in a common coordinate")
    add("The original ±2 at 20% is ±2.5 when displayed at 16%. Thus the proposed −5 put target doubles the original negative log-moneyness reach at the same expiry. Its +2 call target has 80% of the original positive reach. Adding +2 calls improves the <i>current 54</i>, whose calls end near +1; it does not extend the original call wing. [F, H, 8, 9]","small")
    box("<b>Changing a label does not add information.</b> Relabelling the same K and T with 16% rather than 20% changes coordinates only. Selecting different strikes or expiries changes the observations and requires new prices for those contracts. ATM-based selection once, then freezing, also gives a valid fixed design.")
    story.append(PageBreak())

    # 9: actual coverage, literal candidate and a clearly marked target visualization.
    title(9,"A small coverage upgrade to test")
    add("The useful principle in PR #81 is to cover distinct information: deep puts, ATM, near calls, and short as well as later expiries. More points in an already covered region need not add as much information. This motivates an audit; it does not establish a universally optimal grid. [8, 9]")
    story.append(grid_comparison(old, design))
    add("All panels use the same fixed-16% display coordinate. The original and current panels use saved contract locations; crosses are <b>nominal proposed targets</b>, not selected market quotes. Candidate times use the PR's /252 labels for this sketch only; implementation retains exact ACT/365F expiry times.","tiny")
    table([
        ["Coverage audit", "Existing 54", "Proposed change"],
        ["Short expiries", "Starts at 21.03 calendar days; only one expiry within a month", "Target about one and two weeks as well as one month"],
        ["Later expiries", "Ends at 324.99 days, about 10.7 months", "Seek approximately one-year SPXW coverage"],
        ["Puts / ATM / calls", "Deepest puts −3.69 to −4.29; |ATM z| ≤ .011; furthest calls +.93 to +1.07", "Preserve puts and ATM; test added put reach and calls near +2"],
    ],[98,201,WIDTH-299])
    add("<b>Starting candidate:</b> z = [−5, −3.5, −2.25, −1.25, −0.5, 0, +0.5, +1, +2] at fixed 16%, with seven approximate expiry buckets. Fifty-four quotes could also support a useful redesigned layout; 63 is a candidate count, not an acceptance criterion.","small")
    box("Select distinct, usable actual quotes. Record target distances, half-spreads and unsupported locations. Revise the declared design when support is poor. Preserve SPXW settlement and exact timing; keep carry anchors disjoint from fitted/scored quotes and condition both designs on a common carry record. [H, I]")
    story.append(PageBreak())

    # 10: distinguish useful hypotheses and bounded source evidence from adoption.
    title(10,"What to take from PR #81")
    heading("Adopt the coverage principle and simple fixed inputs")
    add("Retain the current 54 as a control; test a single revised actual-quote layout with short expiries and a less truncated call side. Keep the selected contracts fixed and use one specification for simulation, least squares, reference inference and the neural network. An arbitrary-size set encoder is unnecessary for this first comparison. [8, 9]")
    heading("Treat layout E's numerical ranking as preliminary")
    add("The supporting layout_choice.py calls a grid built using fitted maturity-specific ATM volatilities, about 14.30%-17.25%. The written layout E uses fixed 16%. That changes the scale of its log-moneyness offsets by about −10.6% to +7.8%; these are not percentage changes in strikes. We should evaluate the literal candidate we intend to implement. [9-11]")
    add("The reported information calculations are local approximations at a fitted Heston point, with a fixed smooth-error covariance. They do not establish the ranking across the prior or under our simpler spread-based law. Layout E also worsens the local ρ width by about 9% or 31% in the two reported smooth-error scenarios. Better coverage has tradeoffs; there is no promise that every parameter interval becomes narrower. [9]")
    heading("Defer the new correlated-discrepancy model")
    add("The PR also proposes a smooth Gaussian-process error field across strike and maturity, changes the τ prior, profiles correlation lengths, and adds multi-day interpolation and a broader product study. Those are substantial additional choices. Residual smoothness is a reason to investigate such a model later, rather than make it a prerequisite for fixing coverage.")
    add("Our current network already fails conditional reference checks on observations from <i>its own</i> independent-error simulator. Market residual smoothness therefore does not establish the cause of that approximation failure. Improving average loss alone has not resolved it. First establish matched synthetic inference for the simpler targets. [C, I]")
    box("<b>Proposed scope for approval:</b> combine the PR's coverage goals with the single-component Gaussian examples. Keep GP discrepancy, the four-day application, broader exotic products and Bergomi for later decisions. This adopts useful ideas without requiring all of PR #81 to succeed as one project.")
    story.append(PageBreak())

    # 11: concrete development matrix, dependency order and review decisions.
    title(11,"The combined Heston study to approve")
    add("Use the same law, Y<sub>i</sub> = P<sub>i</sub>(φ) + σ h<sub>i</sub>Z<sub>i</sub>, in four development conditions. Keep the physical prior and numerical accuracy requirements matched. Supplied σ is a condition; inferred σ is a hidden target. [I]")
    table([
        ["Design", "Supplied scale", "One inferred scale"],
        ["Existing 54", "Five-target control; reuse clean prices", "Six-target paired scale example; reuse clean prices"],
        ["Revised fixed layout", "Test coverage under the core likelihood", "Combined coverage plus scale-inference candidate"],
    ],[113,193,WIDTH-306])
    heading("Execute in stages, with bounded runs")
    step(1,"Start with the reusable 54.","Add first-nonfinite loss/state capture and establish the supplied/inferred-σ pair against new matched references. Declare update/checkpoint and wall-time limits before launch.")
    step(2,"Audit one revised candidate.","Check actual quote support, a small representative parameter panel, price refinement and conditioning. Preserve ACT/365F, existing carry estimates where unchanged, and one common union-of-expiry carry record.")
    step(3,"Price changed contracts once.","Pair physical parameter rows across designs; reuse unchanged clean labels. Give new designs and likelihoods distinct identities, then repeat the Gaussian pair on the revised grid.")
    step(4,"Assess information and prediction.","Compare full versus later-smile-restricted observations; retain common carry conditions and fixed product horizons across grids. Predict withheld vanilla quotes and inspect one analytic forward-average-variance quantity plus its near-term control.")
    step(5,"Freeze, check and reflect.","Require converged references and MC-aware conditional agreement, then fresh synthetic calibration and market residual checks. Keep the reserved final pool closed during development. Stop for reflection before Bergomi or lesson promotion.")
    add("Reference gates: rank-normalized R-hat ≤ 1.01 and bulk/tail ESS ≥ 400. Conditional development margins include means within 0.3 reference SD and 90% width ratios in [0.75, 1.25], assessed with Monte Carlo uncertainty. A narrow posterior or falling loss alone is insufficient. Comparing 54 versus a changed layout tests a design package, not strike placement alone.","small")
    box("<b>Decisions requested from collaborators:</b> agree the one-component Gaussian baseline and paired σ prior/range; approve layout E as a candidate subject to actual support and precision; approve the staged four-condition Heston comparison and its reflection gate. Exact revised contracts are a phase-2 deliverable, not already selected results.")
    story.append(PageBreak())

    # 12: portable primary references and reproducible local evidence.
    title(12,"Interpretation limits and sources")
    add("Establish inference correctness first on synthetic Heston controls with realistic contracts and spread profiles. Then inspect the actual market residuals separately. Changing the error law can improve a chosen weighted fit, but does not enlarge Heston's attainable price family or guarantee model adequacy. A tight posterior around a poor fit is not calibrated real-world confidence or a model-risk reserve.","small")
    add("Known-scale sensitivity and inferred-scale uncertainty are both useful, but answer different questions. Keep the Gaussian assumption, prior and spread-to-SD convention explicit. Learning σ or τ does not validate those assumptions. No new training, reference computation or model implementation is reported here.","small")
    heading("Primary sources and collaborator proposal")
    for key,label,detail,url in WEB_SOURCES:
        add(f"<b>[{key}] <link href='{url}' color='#087e88'>{label}</link></b><br/>{detail}","small")
    add("The library and practitioner sources support calibration conventions, not a universal Bayesian noise prescription. The proposed Gaussian law, half-spread conversion and c sensitivity are the report's explicit teaching recommendations.","small")
    story.append(PageBreak())
    title(13,"Evidence and reproducibility")
    heading("Local evidence and reproducibility")
    for key,path in sources.items():
        label = {"A":"Current task, priors and fresh-noise training", "B":"Current price checks and likelihood",
                 "C":"Current 50k/75k development comparisons", "D":"Preserved numerical-failure audit",
                 "E":"Original synthetic noise profile", "F":"Original four-target Heston task",
                 "G":"Original known-noise conditioning and augmentation",
                 "H":"Actual 54-contract pilot design and quote provenance",
                 "I":"Combined Heston follow-up specification and review findings"}[key]
        relative=str(path.relative_to(ROOT)).replace("/","/<wbr/>")
        add(f"<b>[{key}] {label}</b><br/>{relative}","tiny")
    add("Sources checked 9 November 2026 local time. Evidence through commit 1ae1c59; combined plan builds on de4e1de; PR #81 pinned to df2cfc442de07459a8b971f070e3a6f25b0b2020. The adjacent .sources.json records exact local source hashes. GitHub proposal links require repository access. The earlier first-principles PDF remains unchanged.","tiny")

    output.parent.mkdir(parents=True,exist_ok=True)
    def decorate(canvas,doc):
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.line(48,PAGE_H-35,PAGE_W-48,PAGE_H-35)
        canvas.line(48,36,PAGE_W-48,36)
        canvas.setFont("Body",8);canvas.setFillColor(MUTED)
        canvas.drawString(48,PAGE_H-25,"HESTON / NOISE AND QUOTE COVERAGE")
        canvas.drawRightString(PAGE_W-48,PAGE_H-25,"Companion proposal - 9 November 2026")
        canvas.drawString(48,23,"Proposed specification | No new training or inference performed")
        canvas.drawRightString(PAGE_W-48,23,str(doc.page))
        canvas.restoreState()
    doc=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=55,bottomMargin=49,
        title="Heston: simpler noise and improved quote coverage",author="SBI4Finance",
        subject="Collaborator proposal: Gaussian error scales, fixed quote design and a staged Heston study")
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
    report={"format":"heston_simpler_noise_proposal_pdf_sources_v2","status":"proposal_not_implemented",
        "pdf_sha256":sha(output),"renderer_sha256":sha(__file__),
        "rendering_helper_sha256":sha(ROOT/"lab/scripts/render_heston_noise_report.py"),
        "local_inputs":{key:{"path":str(path.relative_to(ROOT)),"sha256":sha(path)} for key,path in sources.items()},
        "web_sources":[{"id":key,"title":label,"url":url,"checked":"2026-11-09"} for key,label,_,url in WEB_SOURCES],
        "paired_scale_example":{"status":"proposal_not_run","sigma_units":"dimensionless multiplier of known half-spread",
            "proposed_prior":{"family":"Uniform","bounds":[.5,2.]},"fixed_sensitivity_values":[.5,1.,2.],
            "same_observations_across_comparison":True,"sigma_hidden_from_joint_network_conditions":True},
        "grid_proposal":{"status":"candidate_not_selected_or_run", "pr":81,"pr_head":PR_HEAD,
            "reference_volatility":.16,"target_z":E_TARGETS,"nominal_business_days":[5,10,21,42,63,126,252],
            "actual_selected_expiries_use":"ACT/365F", "current_control_contracts":54,
            "comparison_stages":["existing54_known_vs_inferred_sigma","revised_grid_known_vs_inferred_sigma"],
            "defer_gp_discrepancy":True,"stop_before_bergomi":True}}
    output.with_suffix('.sources.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'output/pdf/heston-simpler-noise-baseline-proposal.pdf')
    args=parser.parse_args();report=build(args.out)
    print(json.dumps({"pdf":str(args.out),"sha256":report['pdf_sha256']}))


if __name__=='__main__':
    main()
