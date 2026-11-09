#!/usr/bin/env python3
"""Typeset the source-verified Heston noise explanation, without inference runs.

Report-only dependencies: reportlab, matplotlib, numpy and svglib.
Use pypdf and a PDF renderer for QA. No model, bank or tutorial asset is edited.
Run from lab/; default final output is ../output/pdf/.
"""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/sbi4f-matplotlib")
import matplotlib
matplotlib.use("Agg")
from matplotlib import mathtext
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
from reportlab.graphics import renderPDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)
from svglib.svglib import svg2rlg


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "lab/evidence/pilot-results/2026-11-09/heston-observations"
INK = colors.HexColor("#163044")
TEAL = colors.HexColor("#087e88")
MUTED = colors.HexColor("#526675")
PALE = colors.HexColor("#eef5f6")
LINE = colors.HexColor("#d7e2e7")
AMBER = colors.HexColor("#87511c")
PAGE_W, PAGE_H = A4
WIDTH = PAGE_W - 96


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_inputs():
    paths = {
        "S1": ROOT / "tutorial/src/sbi4f_tutorial/shared/surface.py",
        "S2": ROOT / "tutorial/assets/backbones/heston-final500/task.json",
        "S3": ROOT / "tutorial/src/sbi4f_tutorial/shared/full_surface.py",
        "S4": ROOT / "lab/src/sbi4finance/calibration/heston_pilot.py",
        "S5": ROOT / "lab/scripts/train_heston_pilot_campaign.py",
        "S6": ROOT / "lab/scripts/train_heston_pilot.py",
        "S7": EVIDENCE / "designs/put_heavy.json",
        "S8": EVIDENCE / "training-study/continuation-results.json",
        "S9": EVIDENCE / "training-study/reference-precision.json",
        "S10": EVIDENCE / "training-study/continuation-failure-audit.json",
    }
    old = json.loads(paths["S2"].read_text())
    assert old["parameter_names"] == ["initial_variance", "theta", "vol_of_vol", "rho"]
    assert old["fixed_parameters"]["kappa"] == 2.
    assert len(old["context"]["quote_noise_std"]) == 63
    design = json.loads(paths["S7"].read_text())
    assert len(design["records"]["mid"]) == 54
    results = json.loads(paths["S8"].read_text())
    current = results["cells"]["75000"]["market"]["parameters"]["tau"]
    assert abs(current["reference_mean"] - .006091600653789715) < 1e-14
    assert abs(current["neural_mean"] - .0014323803257571127) < 1e-14
    return paths, design, current


def fonts_and_styles():
    fontdir = Path(matplotlib.get_data_path()) / "fonts/ttf"
    for name, file in (("Body", "DejaVuSans.ttf"), ("BodyBold", "DejaVuSans-Bold.ttf"),
                       ("BodyItalic", "DejaVuSans-Oblique.ttf"),
                       ("BodyBoldItalic", "DejaVuSans-BoldOblique.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(fontdir / file)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold",
                                  italic="BodyItalic", boldItalic="BodyBoldItalic")
    base = dict(fontName="Body", fontSize=10.1, leading=14.3, textColor=INK,
                spaceAfter=7.5, allowWidows=0, allowOrphans=0)
    return {
        "body": ParagraphStyle("body", **base),
        "small": ParagraphStyle("small", **{**base, "fontSize": 8.6, "leading": 11.7, "spaceAfter": 5.5}),
        "tiny": ParagraphStyle("tiny", **{**base, "fontSize": 7.6, "leading": 10.1, "spaceAfter": 4}),
        "title": ParagraphStyle("title", fontName="BodyBold", fontSize=28, leading=32,
                                textColor=INK, spaceAfter=14),
        "h1": ParagraphStyle("h1", fontName="BodyBold", fontSize=21, leading=26,
                             textColor=INK, spaceAfter=12),
        "h2": ParagraphStyle("h2", fontName="BodyBold", fontSize=12.4, leading=17,
                             textColor=TEAL, spaceBefore=8, spaceAfter=6),
        "kicker": ParagraphStyle("kicker", fontName="BodyBold", fontSize=9, leading=12,
                                 textColor=TEAL, spaceAfter=8),
        "cell": ParagraphStyle("cell", **{**base, "fontSize": 9.0, "leading": 12.2, "spaceAfter": 0}),
        "cellsmall": ParagraphStyle("cellsmall", **{**base, "fontSize": 8.1, "leading": 11, "spaceAfter": 0}),
    }


class VectorArt(Flowable):
    def __init__(self, drawing, max_width=WIDTH, extra=8):
        super().__init__()
        factor = min(1., max_width / drawing.width)
        drawing.scale(factor, factor)
        self.drawing = drawing
        self.width = drawing.width * factor
        self.height = drawing.height * factor + extra
        self.extra = extra

    def draw(self):
        renderPDF.draw(self.drawing, self.canv, (WIDTH - self.width) / 2, self.extra / 2)


def equation(latex, size=17):
    data = BytesIO()
    with plt.rc_context({"mathtext.fontset": "stix", "svg.fonttype": "path"}):
        mathtext.math_to_image("$" + latex + "$", data, prop=FontProperties(size=size),
                              format="svg", color="#163044")
    data.seek(0)
    return VectorArt(svg2rlg(data), extra=13)


def noise_plot():
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8.7,
                         "svg.fonttype": "path", "axes.spines.top": False,
                         "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(6.7, 2.2))
        tau = np.linspace(0, .02, 400)
        sigma_iv_points = 100 * np.sqrt(.0004**2 + tau**2)
        ax.plot(100*tau, sigma_iv_points, color="#087e88", linewidth=2.2,
                label="Current total SD")
        ax.axhline(.04, color="#87511c", linestyle=":", label="Measurement SD alone")
        ax.axhline(.3, color="#526675", linestyle="--", label="Illustrative old ATM SD")
        ax.scatter([.6], [np.sqrt(.04**2+.6**2)], color="#163044", s=26, zorder=5)
        ax.set(xlim=(0, 2), ylim=(0, 2.12), xlabel="Additional scale τ (volatility points)",
               ylabel="Total SD (IV-equivalent points)")
        ax.legend(loc="upper left", frameon=False, fontsize=8)
        ax.grid(alpha=.16)
        fig.tight_layout(pad=.6)
        buffer = BytesIO()
        fig.savefig(buffer, format="svg")
        plt.close(fig)
    buffer.seek(0)
    return VectorArt(svg2rlg(buffer), extra=6)


def make_report(output):
    inputs, _, _ = verified_inputs()
    styles = fonts_and_styles()
    story = []
    def p(text, style="body"):
        return Paragraph(text, styles[style])
    def add(text, style="body"):
        story.append(p(text, style))
    def title(number, heading):
        add(f"{number:02d} / FIRST PRINCIPLES", "kicker")
        add(heading, "h1")
    def heading(text):
        add(text, "h2")
    def box(text):
        table = Table([[p(text)]], colWidths=[WIDTH])
        table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), PALE),
            ("BOX", (0,0), (-1,-1), .55, LINE), ("LEFTPADDING", (0,0), (-1,-1), 12),
            ("RIGHTPADDING", (0,0), (-1,-1), 12), ("TOPPADDING", (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5)]))
        story.extend([table, Spacer(1, 9)])
    def table(rows, widths, small=False):
        data = [[p(str(cell), "cellsmall" if small else "cell") for cell in row] for row in rows]
        t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,0), PALE),
            ("LINEBELOW",(0,0),(-1,0), .8, TEAL),
            ("LINEBELOW",(0,1),(-1,-1), .35, LINE),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("LEFTPADDING",(0,0),(-1,-1),8), ("RIGHTPADDING",(0,0),(-1,-1),8),
            ("TOPPADDING",(0,0),(-1,-1),6), ("BOTTOMPADDING",(0,0),(-1,-1),6)]))
        story.extend([t, Spacer(1, 8)])
    def step(n, label, text):
        add(f"<b>{n}. {label}</b> {text}")

    # Page 1: distinguish the forward price, the random shock and its scale.
    add("HESTON OPTIONS / OBSERVATION MODEL / 9 NOVEMBER 2026", "kicker")
    add("How we construct<br/>observation noise", "title")
    add("A first-principles explanation of the published Heston example and the current European-quote pilot.")
    box("<b>The central change:</b> we replaced a fixed, known price-noise profile with a spread-based measurement component <i>plus</i> an unknown additional discrepancy scale, τ. The total noise has not simply been made smaller.")
    heading("Start with the clean option price")
    add("Given Heston parameters and a contract, the model defines a discounted expected payoff. For a European call:")
    story.append(equation(r"P_i(\phi)=D_i\,\mathbb{E}^{Q}_{\phi}\!\left[(S_{T_i}-K_i)^{+}\right]"))
    add("K is the strike, T is time to expiry, S is the asset value at expiry and D is the discount factor. The asset and variance paths are random, but this expectation is deterministic once the parameters and contract are fixed. We compute it with a Fourier pricer, then add observation noise to represent uncertainty or discrepancy in the quoted price.")
    story.append(equation(r"Y_i=P_i(\phi)+\sigma_i Z_i,\qquad Z_i\sim\mathcal{N}(0,1)"))
    add("A standard Normal has mean zero and variance one. Multiplying by σ gives an error with SD σ: about 68% of draws lie within ±σ and 95% within ±1.96σ. This is a probabilistic scale, not a maximum error bound.", "small")
    table([
        ["Symbol", "Meaning and units"],
        ["P<sub>i</sub>, Y<sub>i</sub>", "Clean model price and simulated observed price; both in the same price units."],
        ["φ = (v₀, κ, θ, ξ, ρ)", "Initial variance, mean-reversion speed, long-run variance, volatility-of-volatility and spot/variance correlation."],
        ["σ<sub>i</sub>", "Observation standard deviation in price units. Its square is the price variance."],
        ["Z<sub>i</sub>", "Dimensionless random draw. Its distribution stays standard Normal."],
        ["τ", "Additional discrepancy SD in decimal-IV units, converted to prices through vega."],
    ], [135, WIDTH-135])
    add("Keep three quantities separate: the random draw Z, the conditional SD σ, and a prior over an unknown scale such as τ. Drawing fresh Z does not itself mean that σ varies across training cases.", "small")
    story.append(PageBreak())

    # Page 2: old implementation, including precise geometry and units.
    title(2, "Before: a fixed, known noise profile")
    add("The published option example used 63 contracts: seven maturities and nine standardized log-moneyness locations. This was a fixed synthetic training design. [S1-S3]")
    heading("1. Fix the design and its noise vector")
    story.append(equation(r"z_i=\frac{\log(K_i/F_i)}{0.20\sqrt{T_i}},\qquad -2\leq z_i\leq 2"))
    add("F is the forward price and T is measured in years. Here z is standardized log-moneyness, not option delta. The reference volatility 0.20 was a design convention. For each contract we assigned an IV-equivalent noise SD:")
    story.append(equation(r"a_i=0.003+0.007\,\frac{\min(|z_i|,2)}{2}"))
    add("Thus a is 0.003 at ATM and 0.010 at the wings: <b>0.3 and 1.0 volatility points</b>. We converted that profile into a fixed price SD using Black vega evaluated at 20% volatility:")
    story.append(equation(r"\sigma_i^{\mathrm{old}}=\nu_i^{20\%}\,a_i"))
    add("Old prices and vegas were in forward-normalized price units. Vega is the derivative of price with respect to decimal IV, so multiplying it by decimal-IV noise gives a price SD. This is a local conversion, not exact additive Gaussian noise in implied volatility.")
    heading("2. Generate a training surface")
    step(1, "Draw the physical parameters.", "Sample v₀, θ, ξ and ρ from their prior; κ is fixed at 2.")
    step(2, "Compute clean prices.", "Evaluate the Heston price on all 63 fixed contracts.")
    step(3, "Add fresh errors.", "Draw independent Z<sub>i</sub> and form Y<sub>i</sub> = P<sub>i</sub> + σ<sub>i</sub><sup>old</sup>Z<sub>i</sub>.")
    step(4, "Infer four physical parameters.", "The fixed noise vector is already specified; there is no inferred observation-noise parameter.")
    box("<b>What stayed fixed:</b> the 63 noise SDs. <b>What changed:</b> the physical parameters and the Gaussian error realizations. Fresh noise draws were already part of the old training procedure.")
    add("Historical market application: SPY chain readings were interpolated onto the old grid, and its synthetic noise floor remained. Synthetic bank prices themselves were computed directly on that grid. This report compares that published task with the current controlled campaign.", "small")
    story.append(PageBreak())

    # Page 3: the current error law, with a transparent independence argument.
    title(3, "Now: measurement plus discrepancy")
    add("The current pilot freezes 54 actual SPXW quote locations and their metadata: strikes, expiries, forwards, discounts, bid-ask half-spreads and market vegas. SPXW provides European exercise conventions. [S4-S7]")
    heading("1. Measurement component")
    story.append(equation(r"h_i=\frac{\mathrm{ask}_i-\mathrm{bid}_i}{2},\qquad\epsilon_i^{\mathrm{meas}}=c\,h_i Z_i"))
    add("h is in actual quote-price units. The known multiplier c scales this measurement allowance; the main market comparison uses c = 1. Interpreting a half-spread as one Gaussian SD is an explicit convention. A bid-ask interval alone does not establish an empirical Gaussian error distribution.")
    heading("2. Additional discrepancy component")
    story.append(equation(r"\epsilon_i^{\mathrm{disc}}=\tau\,\nu_i^{\mathrm{mkt}} U_i,\qquad U_i\sim\mathcal{N}(0,1)"))
    add("τ is an unknown nonnegative amplitude. In decimal-IV units, 0.20 denotes 20% volatility and 0.01 is one volatility percentage point. Thus τ = 0.006 means 0.6 volatility points. Market vega ν is price per unit decimal IV. Vegas are frozen for this design; they are not recomputed from each simulated Heston parameter vector.")
    heading("3. Combine the independent components")
    story.append(equation(r"Y_i=P_i(\phi)+c h_i Z_i+\tau\nu_i^{\mathrm{mkt}}U_i"))
    add("Assume Z and U are independent, both with zero mean and unit variance. Their variances add, giving:")
    story.append(equation(r"\sigma_i^2(\tau,c)=(c h_i)^2+(\tau\nu_i^{\mathrm{mkt}})^2"))
    story.append(equation(r"\sigma_i(\tau,c)=\sqrt{(c h_i)^2+(\tau\nu_i^{\mathrm{mkt}})^2}"))
    add("The code uses the equivalent, cheaper construction: one fresh standard Normal per quote multiplied by this total SD. The two-component representation above explains why we add <b>variances</b>, rather than adding SDs linearly.")
    box("One τ and one c apply to an entire simulated surface. Quote-level shocks are independent conditional on those scales. A shared τ is a shared error amplitude; it does not create a smooth, signed model-bias curve. At τ = 0, the positive measurement noise c h remains.")
    story.append(PageBreak())

    # Page 4: precise training protocol and target, not just a verbal prior.
    title(4, "Exactly what varies during training")
    add("The controlled campaign stores clean prices at 100,000 physical parameter vectors. Reusing a clean price does not freeze its observation noise. On each training exposure, τ, c and the quote errors are sampled again. [S5-S6]")
    table([
        ["Quantity", "Current training distribution", "Role"],
        ["v₀, θ", "Independent Uniform(0.01, 0.16)", "Inferred physical targets"],
        ["κ", "Uniform(0.1, 10)", "Inferred physical target"],
        ["ξ", "Uniform(0.1, 3)", "Inferred physical target"],
        ["ρ", "Uniform(-0.95, -0.1)", "Inferred physical target"],
        ["τ", "Uniform(0, 0.02): 0-2 vol points", "Fresh, inferred scale"],
        ["c", "Uniform(0.5, 2)", "Fresh, supplied as known input"],
        ["Z<sub>i</sub>", "Independent standard Normals", "Fresh per-quote errors"],
    ], [66, 238, WIDTH-304], small=True)
    heading("One training exposure, in order")
    step(1, "Select a clean surface.", "Read its five physical parameters and 54 computed prices from the bank.")
    step(2, "Draw scales.", "Sample a new τ and c, independently, for that surface.")
    step(3, "Construct observations.", "Compute all 54 total SDs, draw new Z values, and add the scaled errors to the clean prices.")
    step(4, "Provide conditions.", "Supply noisy prices and known noise metadata. The fixed-design implementation encodes prices/h and log(c h/(D F)), then standardizes them.")
    step(5, "Train the posterior.", "Predict the joint distribution of five physical parameters and τ. c is observed conditioning information, not a seventh inferred parameter.")
    story.append(equation(r"p\!\left(\phi,\tau\mid\mathbf{y},c,\mathrm{frozen\ design}\right)"))
    box("<b>Yes, we train across a noise-scale prior.</b> τ ranges from 0 to 2 volatility points; the measurement scale also varies by a factor of 0.5-2. A prior draw of τ defines one synthetic dataset. At inference, its posterior expresses uncertainty about the scale compatible with the observed dataset.")
    add("The half-spread vector, vegas and contract locations stay fixed within this task. This is not yet training across arbitrary quote chains or independently changing quote spreads. The physical prior remains broad; it has not been restricted to the market posterior. Negative simulated prices are retained under the stated Gaussian law, rather than clipped.", "small")
    story.append(PageBreak())

    # Page 5: all computations are explicitly illustrative and in consistent units.
    title(5, "A worked example in price units")
    add("Consider a hypothetical quote with clean price 100, half-spread h = 0.20, c = 1 and market vega ν = 500 price units per 1.00 decimal IV. Thus a one-volatility-point move has a local price effect of 5 units.")
    story.append(equation(r"\frac{h}{\nu}=\frac{0.20}{500}=0.0004\quad\text{decimal IV}=0.04\quad\text{vol points}"))
    add("At τ = 0.006, additional discrepancy SD is 500 × 0.006 = 3 price units. The total price SD is:")
    story.append(equation(r"\sigma=\sqrt{0.20^2+3^2}=3.00666\quad\text{price units}"))
    add("For the same chosen standard-Normal draw Z = +1.5, the simulated quote is 100 + 3.00666 × 1.5 = <b>104.510</b>. Measurement noise alone would give 100.300. The draw is the same here purely to make the scale effect visible.")
    rows = [["τ (decimal IV)", "τ (vol points)", "Total price SD", "Total SD / vega (vol points)"]]
    for t in (0., .001, .003, .006, .010, .020):
        s = float(np.hypot(.2, 500*t))
        rows.append([f"{t:.3f}", f"{100*t:.1f}", f"{s:.4f}", f"{100*s/500:.4f}"])
    table(rows, [112, 105, 115, WIDTH-332], small=True)
    story.append(noise_plot())
    add("For an illustrative old ATM comparison, also set the old reference vega to 500. Its SD would be 500 × 0.003 = 1.5 price units, or 0.3 IV-equivalent points. Equal vegas are an illustration only: the real historical and current panels, units and vegas differ.", "small")
    box("The smaller measurement floor does not imply smaller total noise. In the current market panel, the median half-spread/vega scale is about 0.039 vol points, while the reference τ mean is about 0.609 points. The extra discrepancy dominates there.")
    story.append(PageBreak())

    # Page 6: likelihood normalization, inference difficulty and numerical layers.
    title(6, "What the construction implies")
    heading("The likelihood learns the scale from residuals")
    add("For fixed physical parameters, residuals are y<sub>i</sub> - P<sub>i</sub>(φ). The normalized Gaussian log likelihood is:")
    story.append(equation(r"\log L(\phi,\tau)=-\frac{1}{2}\sum_i\left[\log\!\left(2\pi\sigma_i^2\right)+\frac{(y_i-P_i(\phi))^2}{\sigma_i^2}\right]", 16))
    add("A larger τ reduces the standardized residual penalty, but increases the log-variance penalty. Both terms matter: omitting the normalization would incorrectly reward ever larger noise. In the old fixed-noise example that normalization was constant in the inferred physical parameters; now it depends on τ. [S4]")
    add("With physical parameters known, divide residuals by frozen market vega:")
    story.append(equation(r"R_i=\frac{Y_i-P_i(\phi)}{\nu_i^{\mathrm{mkt}}}\sim\mathcal{N}\!\left(0,\left(\frac{c h_i}{\nu_i^{\mathrm{mkt}}}\right)^2+\tau^2\right)", 16))
    add("This is a one-dimensional scale-inference problem in τ. Near zero, measurement noise can hide a small discrepancy. In the current market case, however, the reference posterior constrains τ reasonably well: mean 0.00609 and central 90% interval [0.00516, 0.00722]. The 75k neural mean is 0.00143. That demonstrates approximation error for this specified target, rather than establishing intrinsic nonidentifiability. [S8]")
    heading("Small noise and numerics: three separate layers")
    step(1, "Clean pricing.", "Smaller measurement noise requires more accurate deterministic prices. Bank prices use float64 arithmetic and a refinement threshold of 0.02 × (0.5 h) = 0.01 h. Increasing τ never loosens that threshold.")
    step(2, "Statistical uncertainty.", "Smaller total SD can create sharper parameter posteriors. That is a property of the assumed likelihood, even with exact arithmetic.")
    step(3, "Neural optimization.", "A flow must represent those sharp densities. Float32 transforms, Jacobians, gradients or optimizer state can become unstable even when clean prices are accurate.")
    add("Pointwise reference checks found pricing differences and likelihood shifts well below their local gates; the six synthetic-reference audits covered 96 retained points, with worst absolute log-likelihood change 0.000871. These are checks, not a global proof. No Fourier repricing occurs during network updates. [S9]")
    box("The continuation detected a non-finite monitored loss at 88k; the last finite probe is 87k and the last saved, reload-checked checkpoint is 75k. That checkpoint still fails posterior agreement. The numerical failure's cause remains unresolved. Neither small noise nor inaccurate pricing has been established as the cause. [S8, S10]")
    story.append(PageBreak())

    # Page 7: compact comparison and a source ledger suitable for a portable PDF.
    title(7, "Comparison and implementation sources")
    table([
        ["Aspect", "Published Heston example", "Current controlled pilot"],
        ["Contracts", "63 fixed normalized-grid locations; old market readings interpolated onto them", "54 actual SPXW quote locations, with fixed carry and observed metadata"],
        ["Physical targets", "v₀, θ, ξ, ρ; κ fixed at 2", "v₀, κ, θ, ξ, ρ"],
        ["Vol-of-vol prior", "ξ in [0.1, 1.5]", "ξ in [0.1, 3]"],
        ["Measurement/error SD", "Fixed synthetic profile using 20% vega", "Spread-based component plus inferred vega-scaled discrepancy"],
        ["Unknown error scale", "None", "τ in [0, 0.02]"],
        ["Fresh randomization", "Gaussian quote errors", "τ, known c, and Gaussian quote errors"],
    ], [98, 199, WIDTH-297], small=True)
    add("The redesign changed the design, priors and target dimension as well as noise. It is therefore not an isolated experiment on reducing σ. A useful next diagnostic holds the current design and physical prior fixed, supplies τ as known, and tests inference at several noise levels.", "small")
    add("τ is discrepancy conditional on Heston, fixed carry/vegas, priors and an independent Gaussian error law. It cannot by itself distinguish structural model error from unmodelled data/convention problems, or validate the assumptions. Shared scales also mean residual magnitudes need not be independent after integrating over their prior.", "small")
    heading("Sources and reproducibility")
    add("Verified against the study/heston-european-observations-20261109 checkout, evidence through commit a1d5c60. Paths below are relative to the repository root. The PDF renderer reads source/evidence only; it runs no new inference or training.", "small")
    labels = {
        "S1": "Old IV-noise profile and 20% vega conversion",
        "S2": "Published four-target task and fixed noise vector",
        "S3": "Old price/noise conditioning and fresh Gaussian augmentation",
        "S4": "Current checked pricing, observation SD and normalized likelihood",
        "S5": "Current task, priors, bank and fresh τ/c/error augmentation",
        "S6": "Current network conditions and six posterior targets",
        "S7": "Frozen 54-quote design and market-vega metadata",
        "S8": "Source-bound 50k/75k conditional comparison",
        "S9": "Six synthetic-reference numerical audits",
        "S10": "Retained non-finite-loss failure and read-only audit",
    }
    for key, path in inputs.items():
        relative = str(path.relative_to(ROOT)).replace("/", "/<wbr/>")
        add(f"<b>[{key}] {labels[key]}</b><br/>{relative}", "tiny")

    output.parent.mkdir(parents=True, exist_ok=True)
    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.line(48, PAGE_H-35, PAGE_W-48, PAGE_H-35)
        canvas.setFont("Body", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(48, PAGE_H-25, "HESTON / OBSERVATION NOISE")
        canvas.drawRightString(PAGE_W-48, PAGE_H-25, "Technical explanation - 9 November 2026")
        canvas.line(48, 36, PAGE_W-48, 36)
        canvas.drawString(48, 23, "Fixed-design European-option pilot | Development specification")
        canvas.drawRightString(PAGE_W-48, 23, str(doc.page))
        canvas.restoreState()
    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=48, leftMargin=48,
        topMargin=55, bottomMargin=49, title="How we construct observation noise in Heston",
        author="SBI4Finance", subject="First-principles comparison of fixed and inferred observation-noise scales")
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    source_record = {"format": "heston_noise_pdf_sources_v1", "report_sha256": sha(output),
        "renderer_sha256": sha(__file__), "inputs": {key: {"path": str(path.relative_to(ROOT)),
        "sha256": sha(path)} for key, path in inputs.items()}, "worked_example": {
        "clean_price": 100., "half_spread": .2, "vega_per_decimal_iv": 500., "c": 1.,
        "tau": .006, "total_price_sd": float(np.hypot(.2, 500*.006))}}
    output.with_suffix(".sources.json").write_text(json.dumps(source_record, indent=2)+"\n")
    return source_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"output/pdf/heston-observation-noise-explained.pdf")
    args = parser.parse_args()
    result = make_report(args.out)
    print(json.dumps({"pdf": str(args.out), "sha256": result["report_sha256"]}))


if __name__ == "__main__":
    main()
