"""Build an offline HTML explanation from frozen Heston development records.

Presentation only: no fitting, option pricing, posterior sampling, or bank reads.
The optional old-noise comparison reuses the tutorial's existing noise function.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.heston_spread_report_figures import make_figures
from sbi4f_tutorial.shared import surface


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "lab/evidence/pilot-results/2026-11-09/heston-spread-baseline"
LABELS = {"initial_variance": "Initial variance v₀", "kappa": "Mean reversion κ",
          "theta": "Long-run variance θ", "vol_of_vol": "Volatility of volatility ξ",
          "rho": "Correlation ρ", "control": "Short-window expected variance",
          "affected": "Later-window expected variance"}
CASES = {"synthetic-broad": "Broad-prior synthetic example",
         "synthetic-market-like": "Market-like synthetic example", "market": "Actual market quotes"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def status(value):
    return f'<span class="badge {value}">{value}</span>'


def frame(title, body, ident):
    return f'<section id="{ident}"><h2>{title}</h2>{body}</section>'


def metric_cell(value, verdict, suffix=""):
    return f'<span class="number">{value:.2f}{suffix}</span> {status(verdict)}'


def conditional_table(entry, products=False):
    rows = []
    for name, p in entry["comparison"]["parameters"].items():
        if (name in ("control", "affected")) != products:
            continue
        mean = p["absolute_standardized_mean_difference"]
        width = p["width_ratio_90"]
        mi = p["signed_standardized_mean_interval"]
        wi = p["width_ratio_interval"]
        rows.append(f'''<tr data-status="{p['status']}"><th scope="row">{LABELS[name]}</th>
          <td>{metric_cell(mean,p['mean_status'],' SD')}<small>Signed uncertainty interval [{mi[0]:+.3f}, {mi[1]:+.3f}]</small></td>
          <td>{metric_cell(width,p['width_status'],'×')}<small>Uncertainty interval [{wi[0]:.3f}, {wi[1]:.3f}]</small></td>
          <td>{status(p['status'])}</td></tr>''')
    return '''<div class="table-scroll"><table><thead><tr><th>Quantity</th><th>Absolute mean error</th>
      <th>90% width ratio</th><th>Combined result</th></tr></thead><tbody>''' + "".join(rows) + "</tbody></table></div>"


CSS = r"""
:root{--ink:#152b3a;--muted:#536571;--line:#d8e1e6;--blue:#1b647c;--pale:#eef5f7;--pass:#176647;--fail:#a62d28;--inc:#815b09}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:#f2f5f6;font:16px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1120px;margin:auto;background:white;padding:48px 52px 64px}header{border-bottom:3px solid var(--blue);padding-bottom:30px}h1{font-size:clamp(30px,4vw,43px);line-height:1.17;max-width:950px;margin:14px 0 18px;letter-spacing:-.035em}h2{font-size:27px;line-height:1.25;margin:0 0 20px;letter-spacing:-.02em}h3{font-size:20px;line-height:1.3;margin:28px 0 12px}
p{margin:12px 0 17px}.eyebrow{text-transform:uppercase;letter-spacing:.12em;font-size:12px;font-weight:750;color:var(--blue)}.lead{font-size:19px;max-width:920px}.muted,small{color:var(--muted)}small{display:block;font-size:12px;line-height:1.45;margin-top:5px}.actions{display:flex;gap:12px;flex-wrap:wrap;margin:22px 0 0}button{border:1px solid var(--blue);border-radius:6px;padding:9px 14px;color:var(--blue);background:white;font:600 14px system-ui;cursor:pointer}button:hover{background:var(--pale)}a{color:var(--blue);text-underline-offset:3px}nav{display:flex;gap:8px 20px;flex-wrap:wrap;margin:24px 0;font-size:14px}section{padding-top:38px;margin-top:30px;border-top:1px solid var(--line);scroll-margin-top:15px}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:25px 0}.card{padding:18px;background:var(--pale);border-top:3px solid var(--blue);border-radius:4px}.card strong{display:block;font-size:18px;margin-bottom:7px}.card p{font-size:14px;margin:0}.callout{padding:18px 22px;border-left:4px solid var(--blue);background:var(--pale);margin:22px 0}.callout.warn{border-color:#bb7625;background:#fff8e9}.callout p:last-child{margin-bottom:0}.equation{font:18px/1.7 Georgia,serif;padding:16px 20px;background:#f4f7f8;border-radius:5px;overflow:auto;margin:15px 0}.equation small{font:13px/1.5 system-ui}
.table-scroll{overflow-x:auto;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.45}th,td{padding:12px 10px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}thead th{background:var(--pale);color:#254b5c;font-weight:750}tbody th{font-weight:650;min-width:140px}td{min-width:130px}.number{font-variant-numeric:tabular-nums;white-space:nowrap;font-weight:600}.badge{display:inline-block;white-space:nowrap;border-radius:4px;padding:2px 6px;font-size:11px;font-weight:750;text-transform:uppercase;letter-spacing:.02em}.pass{color:var(--pass);background:#e6f4ed}.fail{color:var(--fail);background:#fbecea}.inconclusive{color:var(--inc);background:#fff4d6}.chart{margin:25px 0 15px}.chart svg{width:100%;height:auto;display:block}.caption{font-size:13px;color:var(--muted);margin:8px 0 24px}.filter{font-size:14px;margin:20px 0;padding:10px 14px;border:1px solid var(--line);border-radius:4px}.filter input{margin-right:9px}details{border:1px solid var(--line);padding:14px 18px;margin:20px 0;border-radius:5px}summary{cursor:pointer;font-weight:700}ol,ul{padding-left:24px}li{margin:9px 0}.sources{font-size:12px}.sources td{min-width:0;word-break:break-word}.hash{font-family:ui-monospace,SFMono-Regular,monospace;font-size:10px;overflow-wrap:anywhere}footer{margin-top:38px;padding-top:20px;border-top:1px solid var(--line);font-size:13px;color:var(--muted)}code{font: .92em ui-monospace,SFMono-Regular,monospace;background:#f0f4f6;padding:1px 4px;border-radius:3px}.two{display:grid;grid-template-columns:1fr 1fr;gap:22px}.two>div{min-width:0}
@media(max-width:740px){main{padding:28px 20px 45px}.cards,.two{grid-template-columns:1fr}h2{font-size:24px}td,th{padding:10px 8px}.lead{font-size:17px}}
@media print{body{background:white;font-size:10.5pt}main{max-width:none;padding:0}h1{font-size:27pt}h2{font-size:19pt}h3{font-size:14pt}section{padding-top:20px;margin-top:20px}nav,.actions,.filter{display:none}.cards{grid-template-columns:repeat(3,1fr)}.card,.callout,thead th{-webkit-print-color-adjust:exact;print-color-adjust:exact}.table-scroll{overflow:visible}table{font-size:9pt}thead{display:table-header-group}tr,.card{break-inside:avoid}.chart{break-inside:avoid}.chart svg{max-height:650px}a{color:inherit}details:not([open])>:not(summary){display:block}tr[hidden]{display:table-row!important}.sources{font-size:8pt}.hash{font-size:7pt}@page{size:A4;margin:16mm}}
"""


def build(historical_root, out):
    historical_root = Path(historical_root).resolve()
    historical = historical_root / "lab/evidence/pilot-results/2026-11-09/heston-observations/README.md"
    sources = {
        "Current training protocol": EVIDENCE/"training-protocol.json",
        "Current frozen synthetic/market panel": EVIDENCE/"references/panel.json",
        "Latest 60k conditional evaluation": EVIDENCE/"annealing/evaluation-060000.json",
        "55k conditional evaluation": EVIDENCE/"annealing/evaluation-055000.json",
        "Annealing protocol": EVIDENCE/"annealing-protocol.json",
        "50k continuation evidence": EVIDENCE/"training-fixed-c1-dropout-zero-continuation/comparison.json",
        "Bank-resolution diagnostic": EVIDENCE/"diagnostics/bank-resolution.json",
        "Local likelihood curvature": EVIDENCE/"diagnostics/likelihood-curvature.json",
        "Original lesson task": ROOT/"tutorial/assets/backbones/heston-final500/task.json",
        "Original archive training metadata": ROOT/"tutorial/assets/checkpoints/heston/training.json",
        "100k original-task training metadata": ROOT/"lab/evidence/pilot-results/2026-10-27/backbone-runs/heston-100k/checkpoint/training.json",
        "Original replication audit": ROOT/"lab/plans/heston-replication-audit.md",
        "Earlier European-market pilot interpretation": historical,
        "Shared original noise function": ROOT/"tutorial/src/sbi4f_tutorial/shared/surface.py",
        "HTML report producer": Path(__file__).resolve(),
        "Report figure producer": Path(__file__).with_name("heston_spread_report_figures.py"),
    }
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    latest = read(sources["Latest 60k conditional evaluation"])
    panel = read(sources["Current frozen synthetic/market panel"])
    protocol = read(sources["Current training protocol"])
    original = read(sources["Original lesson task"])
    old_training = read(sources["100k original-task training metadata"])
    current = read(EVIDENCE/"training-fixed-c1/run.json")["task"]["records"]
    assert latest["actual_optimizer_updates"] == 60000
    assert original["fixed_parameters"]["kappa"] == 2. and len(original["parameter_names"]) == 4
    assert protocol["fixed_c"] == 1. and protocol["inferred_c_in_this_fit"] is False
    contracts = [dict(strike=k/f, maturity=t, option_type=o) for k,f,t,o in
        zip(current["strike"],current["forward"],current["maturity"],current["option_type"],strict=True)]
    old_price_sd = (np.asarray(current["discount"])*np.asarray(current["forward"])
                    * np.asarray(surface.quote_noise(contracts)))
    ratios = old_price_sd/np.asarray(current["half_spread"])
    figs = make_figures(EVIDENCE, ratios)
    svg = lambda name: '<div class="chart">'+figs[name][figs[name].index("<svg"):]+'</div>'
    old_rows = old_training["training_surfaces"]
    old_settings = old_training["run_contract"]["settings"]
    old_updates = int(np.ceil(old_rows/old_settings["batch_size"]))*old_settings["epochs"]
    snapshot = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    parts = ['''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
      <title>Heston development report — conditional accuracy and the original comparison</title><style>''',CSS,
      '''</style></head><body><main><header><div class="eyebrow">Heston development report · 10 November 2026</div>
      <h1>What improved, what still fails, and why the comparison changed</h1>
      <p class="lead">The simpler observation model is implemented. Neural accuracy improves substantially after removing dropout and lowering the learning rate, but the latest 60k checkpoint still fails conditional posterior checks. The study also changes several things beyond noise, so it cannot isolate the effect of smaller noise.</p>
      <div class="actions"><button onclick="window.print()">Print / save as PDF</button><button id="download">Download report data</button></div></header>
      <nav aria-label="Contents"><a href="#verdict">Main findings</a><a href="#checks">Conditional checks</a><a href="#training">Training progress</a><a href="#comparison">Original versus current</a><a href="#noise">Noise construction</a><a href="#xi">What ξ means</a><a href="#choices">Why changes happened</a><a href="#next">Controlled next steps</a><a href="#sources">Sources</a></nav>''']
    parts.append(frame("The findings to keep separate", '''
      <div class="cards"><div class="card"><strong>Observation law simplified</strong><p>One Gaussian error scale, c × half-spread. Current fits fix c=1; there is no inferred τ.</p></div>
      <div class="card"><strong>Inference still needs work</strong><p>Both completed synthetic cases still fail conditional agreement. Passing population rank checks does not resolve those failures.</p></div>
      <div class="card"><strong>Attribution is unresolved</strong><p>More parameters, wider prior support, different contracts, a different network and fewer distinct training surfaces changed together.</p></div></div>
      <div class="callout warn"><strong>The correction to the earlier interpretation</strong><p>Smaller noise is a plausible source of greater approximation difficulty. The current experiments do not establish that it caused the regression. This pilot is a broader redesign, not a noise-only adaptation of the original lesson.</p></div>
      <p>Numerical and likelihood checks passed at the sampled points. Three reference cases pass convergence and sampled pricing-refinement checks. The boundary reference reached its declared time cap and remains inconclusive. No new neural checkpoint or lesson is accepted; revised-grid production and Bergomi remain gated.</p>''',"verdict"))
    parts.append(frame("Which conditional posterior checks fail?", '''
      <p>For one fixed set of 54 quotes, compare the network’s draws with MCMC under the <strong>same prior, pricing model and Gaussian likelihood</strong>. These checks assess the network’s approximation to that posterior. They do not require its mean to equal a generating truth and do not establish that Heston describes actual market prices.</p>
      <div class="equation">Mean error = |network mean − reference mean| / reference SD ≤ 0.30<br>
      Width ratio = network central-90% width / reference central-90% width ∈ [0.75, 1.25]</div>
      <p>A <strong>pass</strong> places the sampling-uncertainty interval wholly inside the allowed range; a <strong>fail</strong> places it wholly outside; an <strong>inconclusive</strong> result overlaps a boundary. The evaluator accounts for correlated reference draws and finite neural draws, using approximate family-adjusted uncertainty intervals and two reference block lengths. A combined quantity passes only when both mean and width pass.</p>
      <p class="caption">All tables and the chart below use the predeclared <strong>60k endpoint</strong>. The 55k outcome is also retained; neither endpoint was chosen from the loss curve.</p>'''+svg("conditional")+'''
      <label class="filter"><input type="checkbox" id="hide-passes">Hide rows whose mean and width both pass</label>'''+
      "".join('<article class="case"><h3>'+CASES[cid]+'</h3><p>'+description+'</p>'+conditional_table(latest["cases"][cid])+conditional_table(latest["cases"][cid],True)+'</article>'
        for cid,description in (("synthetic-broad","A synthetic surface generated from a predeclared broad-prior parameter draw. “Broad” describes the source prior, not a broad posterior."),
          ("synthetic-market-like","Synthetic data generated from Heston at parameters near the earlier market fit. Failure here exists even when the assumed model generates the observations.")))+'''
      <p class="caption">The two downstream quantities are analytical Heston expected average variance: a short window from valuation to about day 21, and a later window from about day 81 to day 325. They are not forward-start option prices.</p>
      <details><summary>Actual market quotes: substantially larger neural/reference errors</summary>
      <p>All five physical-parameter mean checks fail. Their absolute discrepancies are about 7.2 SD (v₀), 3.7 SD (κ), 10.0 SD (θ), 20.9 SD (ξ) and 33.9 SD (ρ). This disagreement remains an inference problem to assess; market-model inadequacy is a separate issue and does not by itself excuse disagreement with the matched posterior.</p>'''+conditional_table(latest["cases"]["market"])+conditional_table(latest["cases"]["market"],True)+'''</details>
      <details><summary>Boundary example: unscored because its reference is unresolved</summary><p>The first MCMC attempt failed convergence. Its one escalation exhausted the 2,700-second wall limit. The failure and partial chains remain retained. We have no accepted reference for scoring this neural case; its status is inconclusive.</p></details>
      <p><strong>Why can a small mean error remain inconclusive?</strong> For broad-case ρ, the estimated error is only −0.024 reference SD, but its sampling-uncertainty interval is [−0.310,+0.263], slightly crossing −0.30. Finite neural-draw uncertainty dominates here. Its excessive central interval width is nevertheless a definite failure.</p>''',"checks"))
    stages = [("Default dropout .05",10000,EVIDENCE/"evaluation-update-010000-provisional.json"),
              ("Dropout zero",10000,EVIDENCE/"evaluation-dropout-zero-update-010000-provisional.json"),
              ("Same schedule floor",25000,EVIDENCE/"evaluation-dropout-zero-update-025000.json"),
              ("Same schedule floor",50000,EVIDENCE/"evaluation-dropout-zero-update-050000.json"),
              ("Lower-rate stage 1",55000,EVIDENCE/"annealing/evaluation-055000.json"),
              ("Lower-rate stage 2",60000,EVIDENCE/"annealing/evaluation-060000.json")]
    stage_rows=[]
    for label,steps,path in stages:
        r=read(path); entries=r["cases"]["synthetic-broad"]["comparison"]["parameters"]
        widths=[v["width_ratio_90"] for k,v in entries.items() if k not in ("control","affected")]
        ranks=r["population"]["rank_dkw"]["passed"]
        stage_rows.append(f'<tr><th>{label}</th><td>{steps:,}</td><td>{min(widths):.2f}–{max(widths):.2f}×</td><td>{sum(ranks)}/5</td><td>{status(r["conditional_status"])}</td></tr>')
    parts.append(frame("What training improved—and what it did not",svg("training")+'''
      <div class="table-scroll"><table><thead><tr><th>Recipe / endpoint</th><th>Updates</th><th>Broad physical width range</th><th>Development rank checks</th><th>Conditional result</th></tr></thead><tbody>'''+"".join(stage_rows)+'''</tbody></table></div>
      <p>Removing dropout changed a matched 10k control substantially. Continuing to 50k improved widths but left unstable centering. Eight discarded future updates showed that the original 5e-5 learning-rate floor could move broad-case means by .275–.818 reference SD RMS. That prompted one bounded schedule change: 5e-6 for 5k updates, then 5e-7 for 5k. Adam moments, iteration and all saved state were retained.</p>
      <p>The annealing fit plus both evaluations completed in 99.86 seconds. At 60k the population rank checks pass 5/5 and 90% coverage is .916–.934, yet both synthetic conditional panels fail. These are repeated development checks on 512 cases, not fresh acceptance evidence. Lower loss and acceptable average ranks can coexist with errors for individual quote surfaces.</p>''',"training"))
    parts.append(frame("What differs from the original example?",f'''
      <p>The current lesson’s saved original-task example uses the 100k, final-500 FlowMatching candidate. Its historical archive used 50k surfaces. Both retain the same original four-dimensional task.</p>
      <div class="table-scroll"><table><thead><tr><th>Aspect</th><th>Original lesson task / recipe</th><th>Current spread pilot</th></tr></thead><tbody>
      <tr><th>Inferred parameters</th><td>v₀, θ, ξ, ρ: four</td><td>v₀, κ, θ, ξ, ρ: five</td></tr>
      <tr><th>Mean reversion κ</th><td>Fixed at 2</td><td>Uniform [0.1,10]</td></tr>
      <tr><th>Vol-of-vol ξ prior</th><td>Uniform [0.1,1.5]</td><td>Uniform [0.1,3]</td></tr>
      <tr><th>Other physical priors</th><td>v₀ and θ [0.01,0.16]; ρ [−0.95,−0.1]</td><td>Same bounds</td></tr>
      <tr><th>Contracts</th><td>63 canonical quotes; symmetric standardized log-moneyness ±2; seven tenors through one year</td><td>54 actual SPXW contracts; put-heavy; six expiries about 21–325 calendar days</td></tr>
      <tr><th>Noise</th><td>Prescribed IV-scale rule converted at fixed 20% vega</td><td>Observed price half-spreads; fixed c=1</td></tr>
      <tr><th>Network</th><td>FlowMatching, four 256-unit blocks</td><td>Spline coupling flow, [128,128] conditioners</td></tr>
      <tr><th>Distinct clean training surfaces</th><td>{old_rows:,} in final-500 candidate; archive 50,000</td><td>20,000-row prefix of an available 100k bank</td></tr>
      <tr><th>Training updates / batch</th><td>{old_updates:,} / 256 in 100k recipe; archive about 98,000 / 256</td><td>60,000 / 128, including the declared continuation and annealing</td></tr>
      <tr><th>Checks</th><td>Original candidate passed three examined development conditional cases; historical archive was imperfect</td><td>Different predeclared cases; explicit sampling-uncertainty-aware verdicts</td></tr>
      </tbody></table></div>
      <div class="callout warn"><strong>The synthetic cases also changed physical regime.</strong><p>The current broad and market-like truths use ξ≈1.724 and ξ≈1.799. Both exceed the original prior ceiling of 1.5. Their κ values are about 3.083 and 4.336, rather than the original fixed 2. The difficult boundary case uses ξ=2.9 and κ=.12.</p></div>
      <p>There is therefore no isolated estimate here of how much smaller noise, free κ, the wider prior, geometry, network family or training budget contributed. A different family or smaller conditioner does not alone establish a capacity problem. The old model’s development success was scoped; it was not a guarantee of exact posteriors across its full domain.</p>''',"comparison"))
    parts.append(frame("How the noise changed, and why smaller noise can demand more precision",f'''
      <p>Both the original lesson and the current simplified pilot use independent Gaussian price errors:</p>
      <div class="equation">Y<sub>i</sub> = P<sub>i</sub>(parameters) + σ<sub>i</sub> ε<sub>i</sub>, &nbsp; ε<sub>i</sub> ∼ N(0,1)</div>
      <h3>Original synthetic lesson</h3>
      <div class="equation">z<sub>i</sub> = log(K<sub>i</sub>/F<sub>i</sub>) / (0.20 √T<sub>i</sub>)<br>
      s<sub>IV,i</sub> = 0.003 + 0.007 min(|z<sub>i</sub>|,2)/2<br>
      σ<sub>old,i</sub> = D<sub>i</sub> F<sub>i</sub> Vega<sub>forward</sub>(log(K<sub>i</sub>/F<sub>i</sub>),T<sub>i</sub>;0.20) s<sub>IV,i</sub>
      <small>Vega is the derivative of option price with respect to decimal volatility. Forward-normalized vega is √T φ(d₁). Original canonical tasks use F=D=1; D×F restores current currency-price units. One volatility point is .01 in decimal volatility.</small></div>
      <h3>Current simplified market-profile pilot</h3>
      <div class="equation">h<sub>i</sub> = (ask<sub>i</sub> − bid<sub>i</sub>)/2, &nbsp; σ<sub>i</sub> = c h<sub>i</sub>, &nbsp; c=1 in all fits reported here.</div>
      <p>Half-spread-as-standard-deviation is a simplifying convention, not a measured Gaussian-error law. The current fits do not infer c. A separate inferred-c likelihood exists, but its network has not been trained.</p>
      <details><summary>The intervening τ/vega model, for context</summary><div class="equation">σ<sub>i</sub> = √[(m h<sub>i</sub>)² + (τ Vega<sub>market,i</sub>)²]</div><p>This separated spread-based measurement uncertainty from an inferred discrepancy term in local IV units. It also used frozen market vegas. The current c×h target removes that second component. The new conditional failures therefore occur without jointly estimating two noise sources.</p></details>
      {svg('noise')}
      <p>For a controlled <em>counterfactual comparison on the same current 54 contracts</em>, evaluate the original rule in current currency units and divide it by h. The median ratio is <strong>{np.median(ratios):.2f}×</strong>; the range is {min(ratios):.2f}–{max(ratios):.2f}×. The original rule exceeds h on {np.sum(ratios>1)}/54 contracts. This is not a measured old-SPY/new-SPX spread ratio, and noise is not uniformly smaller at every deep-wing contract.</p>
      <p>With less observation error, informative quotes can produce a much sharper posterior. The inverse problem can become better identified while approximating its conditional distribution demands greater numerical and neural precision. Our tolerance is measured in posterior SD: an unchanged absolute mean error looks larger when that SD shrinks.</p>
      <div class="callout"><strong>A concrete scale in the broad case</strong><p>Reference v₀ mean≈0.1360258, network mean≈0.1360924. The absolute difference is only .0000666 in annualized variance, but the reference SD is .0000853. That is .78 SD, above the .30-SD tolerance.</p></div>
      <p>Fresh noise augmentation repeatedly perturbs 20k existing clean surfaces; it does not create new physical-parameter locations. The bank audit finds substantial interpolation distances in spread units, even for the available 100k bank. That motivates studying data density but does not prove that learning is impossible or that a larger bank will fix the errors. Sampled pricing refinements and local likelihood curvature agree closely with the reference calculation; they are not a global all-prior precision certificate.</p>''',"noise"))
    parts.append(frame("What is volatility of volatility—and is 1.5 unreasonable?",'''
      <p>Heston models instantaneous <strong>variance</strong> V, with instantaneous volatility √V. The parameter convention used here is:</p>
      <div class="equation">dV = κ(θ − V) dt + ξ √V dW<br><small>Time is measured in years; variance and volatility use decimal annualized units. ξ controls random fluctuations of variance and is commonly called volatility of volatility.</small></div>
      <p>There is a real economic counterpart: volatility fluctuates, and volatility derivatives carry information about those fluctuations. Heston’s ξ is nevertheless a model parameter inferred from prices. It is not a directly observed index value.</p>
      <p>A local interpretation follows by applying Itô’s rule to σ=√V at V&gt;0. The diffusion term of dσ is (ξ/2)dW. Over a short business-day increment, its local shock scale is approximately ξ/(2√252), in decimal annualized volatility:</p>
      <div class="table-scroll"><table><thead><tr><th>ξ</th><th>Local one-business-day volatility shock scale</th></tr></thead><tbody>'''+
      "".join(f'<tr><th>{xi:g}</th><td>{100*xi/(2*np.sqrt(252)):.2f} volatility percentage points</td></tr>' for xi in (.3,.6,1.5,1.8,3))+'''
      </tbody></table></div>
      <p class="caption">This is a local diffusion scale derived from the model, not a finite-day forecast, confidence interval or empirical estimate. Drift, mean reversion and boundary behavior matter, especially at large ξ and small V. For example, a 4.72-point change takes 20% annualized volatility to roughly 24.72% in this local illustration.</p>
      <p><strong>There is no universal mathematical ceiling at 1.5.</strong> It implies substantial volatility fluctuation under this convention and needs justification in a particular market/task. A posterior pushing above 1.5 is evidence about the model fit under its assumptions, not proof that actual volatility follows Heston with that ξ. In an option calibration we infer a pricing-measure model; interpreting ξ as a directly measured historical volatility statistic requires additional assumptions.</p>
      <p>A familiar market analogue is <a href="https://www.cboe.com/us/indices/dashboard/vvix/">Cboe’s VVIX</a>, derived from VIX options and describing expected volatility of the 30-day forward price of VIX. It is related to volatility-of-volatility risk, but cannot be read off as Heston ξ: the underlying quantity, units and horizon differ.</p>
      <p>For mathematical context, the sufficient strict-positivity condition is 2κθ ≥ ξ². Violating it does not make the usual nonnegative Heston variance process invalid; zero becomes attainable. It can matter for numerical schemes. The condition depends on κ and θ together with ξ, so it cannot justify a universal ξ cutoff. See <a href="https://www.econstor.eu/bitstream/10419/56653/1/637060245.pdf">Janek, Kluge, Weron and Wystup, §1.3.3</a>. This report does not establish a typical empirical SPX range for ξ.</p>''',"xi"))
    parts.append(frame("Why were the extra changes made?",'''
      <p>The current implementation reused the earlier European-market pilot, rather than starting from the original lesson task unchanged. The following distinguishes documented motivation from implementation choices and interpretation.</p>
      <div class="table-scroll"><table><thead><tr><th>Change</th><th>Recorded reason / origin</th><th>Implication for this comparison</th></tr></thead><tbody>
      <tr><th>ξ upper bound 1.5→3</th><td>The earlier market posterior accumulated at 1.5: median 1.479, 90% interval [1.420,1.499]. Widening moved full/short-panel medians to about 1.80/2.12 and changed downstream variance results.</td><td>A motivated prior-sensitivity diagnostic, later retained as an engineering benchmark. It should be tested separately from noise. It is not a scientifically final prior or independent validation of Heston.</td></tr>
      <tr><th>Free κ</th><td>Explicitly specified in the earlier five-parameter European pilot. No direct rationale for the particular [.1,10] bounds was found in the reviewed record.</td><td>Jointly learning mean reversion can be relevant to maturity extrapolation, but that is an interpretation, not a documented reason for these bounds. Free κ is not needed to test a noise change.</td></tr>
      <tr><th>54 actual put-heavy quotes</th><td>Inherited from the frozen European SPXW market-data pilot.</td><td>A legitimate separate observation-design choice. It changes which information is available; it is not the original canonical 63-quote task or the new PR81-style 63-contract candidate.</td></tr>
      <tr><th>Spline / 20k-row screen</th><td>A bounded development recipe selected for a quick local engineering screen.</td><td>It is not a matched reproduction of the original FlowMatching training/data budget. Failure cannot be assigned to noise alone.</td></tr>
      </tbody></table></div>
      <div class="callout"><strong>What widening ξ established</strong><p>The old 1.5 ceiling materially constrained that earlier market fit. It did not establish a universally appropriate ξ range, resolve market residual structure, or make a wider-prior neural task comparable with the original example. The earlier spread-only market fit still had RMS residual about 12.3 half-spreads; a higher fitted ξ can be a model's attempt to explain a surface it does not adequately describe. These extra choices should have been highlighted and isolated before interpreting the regression.</p></div>''',"choices"))
    parts.append(frame("A cleaner next comparison",'''
      <p>The immediate question should be whether lower noise alone strains the original amortized example. It requires a matched control, rather than another mixed redesign.</p>
      <ol><li><strong>Retain the original task and training recipe.</strong> Four targets, κ=2, ξ≤1.5, canonical 63-contract geometry, original FlowMatching family, and matched distinct-surface and update budgets.</li>
      <li><strong>Vary only a known synthetic noise amplitude.</strong> Compare σ=a×σ<sub>old</sub> at a small predeclared set of amplitudes, with separately matched likelihoods and networks. Use a for this synthetic multiplier and retain c for the market-spread model c×h. This tests shrinking noise; it does not claim to reproduce a market spread profile.</li>
      <li><strong>Check pricing precision before bank reuse.</strong> Tighter noise tightens the numerical requirement. Retain compatible clean labels only where refinement supports them; do not assume zero repricing.</li>
      <li><strong>Then introduce market geometry/profile and physical-prior changes separately.</strong> A 54-contract, κ=2, ξ≤1.5 control can test the actual-quote setup without the broader dynamics. It needs matching pricing labels: existing free-κ prices cannot simply be relabelled as κ=2.</li></ol>
      <p>This is a proposed attribution experiment, not a newly executed fit. The current 60k results remain useful engineering evidence under their own task identity. No training, sampler escalation or Bergomi production was launched to create this report.</p>''',"next"))
    source_rows=[]; source_data=[]
    for label,path in sources.items():
        external=path.is_relative_to(historical_root) and not path.is_relative_to(ROOT)
        rel=path.relative_to(historical_root if external else ROOT)
        commit="78cf9de" if external else snapshot
        link=f"https://github.com/fmuia/sbi4finance/blob/{commit}/{rel.as_posix()}"
        value=digest(path)
        source_rows.append(f'<tr><th>{html.escape(label)}</th><td><a href="{link}">{html.escape(str(rel))}</a></td><td class="hash">{value}</td></tr>')
        source_data.append(dict(label=label,path=str(rel),sha256=value,source_checkout="historical" if external else "current"))
    data={"format":"heston_spread_html_report_v1","date":"2026-11-10","input_snapshot_commit":snapshot,
          "latest_updates":60000,"scientific_acceptance":False,"sources":source_data,
          "same_54_counterfactual_old_sd_over_halfspread":{"median":float(np.median(ratios)),"min":float(min(ratios)),"max":float(max(ratios)),"count_above_one":int(np.sum(ratios>1)),"ratios":ratios.tolist()},
          "latest_case_statuses":{k:v["status"] for k,v in latest["cases"].items()},
          "scope":"Read-only presentation of frozen development records; no fitting, posterior draws, option pricing or bank-array access."}
    parts.append(frame("Sources, scope and reproducibility",'''
      <p>This file is self-contained: charts, styles, equations and tables render offline without a math service or JavaScript library. Source links are optional GitHub links to the private project. Neither source access nor scripts are required to read the report. Use the print button for a paper copy; the data button exports the source inventory and noise-comparison statistics.</p>
      <p>The main new evidence uses 4,096 neural draws per conditional query and 512 development cases with 512 draws each for population checks. The reserved test pool remains unopened. Conditional mean/width tests are useful screens, not a complete test of joint shape or tails. The original and current panels differ, so their pass counts are not a controlled head-to-head score.</p>
      <details><summary>Input files and SHA-256 identities</summary><div class="table-scroll"><table class="sources"><thead><tr><th>Source</th><th>Repository path</th><th>SHA-256</th></tr></thead><tbody>'''+"".join(source_rows)+'''</tbody></table></div></details>
      <p>Primary external reference for the market volatility-of-volatility analogue: <a href="https://www.cboe.com/us/indices/dashboard/vvix/">Cboe VVIX description and FAQ</a>. Model interpretation and local numerical scales above follow from the explicitly stated Heston SDE; the daily scale is a derivation, not an empirical calibration.</p>''',"sources"))
    payload=json.dumps(data,ensure_ascii=False,allow_nan=False).replace("<","\\u003c")
    parts.extend(['<footer>Prepared from completed development records. No accepted checkpoint. Report date: 10 November 2026.</footer>',
        '<script id="report-data" type="application/json">'+payload+'</script>',r'''<script>
      document.getElementById('hide-passes').addEventListener('change',function(){document.querySelectorAll('tr[data-status="pass"]').forEach(row=>row.hidden=this.checked)});
      document.getElementById('download').addEventListener('click',function(){const text=JSON.stringify(JSON.parse(document.getElementById('report-data').textContent),null,2);const url=URL.createObjectURL(new Blob([text],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='heston-report-data.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});
      </script></main></body></html>'''])
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text("".join(parts),encoding="utf-8")
    print(json.dumps({"report":str(out),"bytes":out.stat().st_size,"sha256":digest(out),"sources":len(sources),"scientific_acceptance":False}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=EVIDENCE/"report.html")
    args=parser.parse_args()
    build(args.historical_root,args.out)


if __name__=="__main__":
    main()
