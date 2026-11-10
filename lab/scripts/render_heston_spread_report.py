"""Build an offline HTML explanation from frozen Heston development records.

Presentation only: no fitting, option pricing, posterior sampling, or bank reads.
The optional old-noise comparison reuses the tutorial's existing noise function.
"""
import argparse
import ast
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.heston_spread_report_figures import make_figures, recovery_data
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


def source_function(path, name):
    text = Path(path).read_text()
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(text, node)
    raise ValueError(f"missing source function {name}")


def code_block(text):
    return '<pre><code>'+html.escape(text)+'</code></pre>'


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
pre{padding:18px 20px;background:#142d3b;color:#edf5f8;border-radius:5px;overflow-x:auto;font:12.5px/1.6 ui-monospace,SFMono-Regular,monospace;tab-size:4}pre code{font:inherit;background:transparent;padding:0;color:inherit}.flow{border:1px solid var(--line);padding:16px 20px;background:#f8fafb;white-space:pre-wrap;color:var(--ink);font-size:14px}.settings td:first-child{font-weight:650}
@media(max-width:740px){main{padding:28px 20px 45px}.cards,.two{grid-template-columns:1fr}h2{font-size:24px}td,th{padding:10px 8px}.lead{font-size:17px}}
@media print{body{background:white;font-size:10.5pt}main{max-width:none;padding:0}h1{font-size:27pt}h2{font-size:19pt}h3{font-size:14pt}section{padding-top:20px;margin-top:20px}nav,.actions,.filter{display:none}.cards{grid-template-columns:repeat(3,1fr)}.card,.callout,thead th{-webkit-print-color-adjust:exact;print-color-adjust:exact}.table-scroll{overflow:visible}table{font-size:9pt}thead{display:table-header-group}tr,.card{break-inside:avoid}.chart{break-inside:avoid}.chart svg{max-height:650px}a{color:inherit}details:not([open])>:not(summary){display:block}tr[hidden]{display:table-row!important}.sources{font-size:8pt}.hash{font-size:7pt}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:8pt;line-height:1.45;background:#eef3f5;color:var(--ink)}@page{size:A4;margin:16mm}}
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
        "SBC reproduction manifest": EVIDENCE/"sbc-development/manifest.json",
        "SBC default 10k ranks": EVIDENCE/"sbc-development/default-010000.json",
        "SBC zero-dropout 10k ranks": EVIDENCE/"sbc-development/zero-010000.json",
        "SBC zero-dropout 50k ranks": EVIDENCE/"sbc-development/zero-050000.json",
        "SBC lower-rate 60k ranks": EVIDENCE/"sbc-development/annealed-060000.json",
        "Recovery generating parameters and row IDs": EVIDENCE/"sbc-development/population.npz",
        "Recovery 60k marginal quantile summaries": EVIDENCE/"sbc-development/annealed-060000.quantiles.npz",
        "SBC extraction producer": ROOT/"lab/scripts/extract_heston_spread_sbc.py",
        "Resolved network and optimizer settings": EVIDENCE/"network-settings.json",
        "Actual fixed-c workflow and zero-dropout code": ROOT/"lab/scripts/train_heston_spread_baseline.py",
        "Actual original-floor continuation": ROOT/"lab/scripts/continue_heston_spread_baseline.py",
        "Actual lower-rate continuation": ROOT/"lab/scripts/anneal_heston_spread_baseline.py",
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
      <p class="lead">At the latest 60k checkpoint, most generating parameters are recovered close to the unit line, and all five marginal SBC rank screens stay within their reference band. There are some outlying recoveries and measurable network/MCMC differences on individual quote surfaces. Broad recovery is strong; precise per-dataset posterior agreement remains unresolved.</p>
      <p class="muted">This report covers the five-parameter, fixed-c=1 phase. The subsequent <a href="noise-scale/report.html">noise-scale study</a> examines fixed-scale sensitivity and a separate six-parameter network with inferred c.</p>
      <div class="actions"><button onclick="window.print()">Print / save as PDF</button><button id="download">Download report data</button></div></header>
      <nav aria-label="Contents"><a href="#verdict">How well does it work?</a><a href="#checks">Network versus MCMC</a><a href="#sbc">SBC plots</a><a href="#recovery">Parameter recovery</a><a href="#training">Training progress</a><a href="#settings">Exact network, dropout and rates</a><a href="#comparison">Original versus current</a><a href="#noise">Noise construction</a><a href="#xi">What ξ means</a><a href="#choices">Why changes happened</a><a href="#next">Controlled next steps</a><a href="#sources">Sources</a></nav>''']
    parts.append(frame("The findings to keep separate", '''
      <div class="cards"><div class="card"><strong>Parameter recovery</strong><p>Most posterior medians track their generating values closely across 512 cases. A few larger departures appear in θ and ρ. The new recovery plots show all points.</p></div>
      <div class="card"><strong>Population calibration</strong><p>5/5 marginal SBC rank screens do not reject at 60k. Nominal 90% intervals contain truths in 91.6–93.4% of cases across parameters: modest overcoverage.</p></div>
      <div class="card"><strong>Reference fidelity</strong><p>Neither completed synthetic reference case passes every mean/width check. Some means miss by .4–.8 reference SD, and some widths remain too broad.</p></div></div>
      <p><strong>The practical assessment:</strong> removing dropout and lowering the learning rate materially improved the approximation. Recovery and the latest ECDF curves are encouraging, alongside the remaining per-dataset precision errors. The real-market query has substantially larger neural/reference discrepancies and remains unvalidated. The two synthetic cases are a small development panel, not an estimate of a domain-wide failure rate. All fits here use five physical parameters, fixed c=1 and no τ.</p>
      <div class="callout warn"><strong>The correction to the earlier interpretation</strong><p>Smaller noise is a plausible source of greater approximation difficulty. The current experiments do not establish that it caused the regression. This pilot is a broader redesign, not a noise-only adaptation of the original lesson.</p></div>
      <p>Numerical and likelihood checks passed at the sampled points. Three reference cases pass convergence and sampled pricing-refinement checks. The boundary reference reached its declared time cap and remains inconclusive. No new neural checkpoint or lesson is accepted; revised-grid production and Bergomi remain gated.</p>''',"verdict"))
    parts.append(frame("What does “conditional result” mean?", '''
      <p><strong>Conditional means conditional on this one observed set of quotes.</strong> Hold all 54 quotes fixed and ask whether the fast network reproduces the posterior from our chosen Heston model. MCMC supplies a reference for that same dataset.</p>
      <pre class="flow">One fixed set of 54 option quotes
    ├─ Network → fast posterior draws for five Heston parameters
    └─ MCMC    → reference posterior draws for those same parameters
                    ↓
          Compare posterior means and 90% interval widths</pre>
      <p>For one fixed set of 54 quotes, compare the network’s draws with MCMC under the <strong>same prior, pricing model and Gaussian likelihood</strong>. These checks assess the network’s approximation to that posterior. They do not require its mean to equal a generating truth and do not establish that Heston describes actual market prices.</p>
      <div class="equation">Mean error = |network mean − reference mean| / reference SD ≤ 0.30<br>
      Width ratio = network central-90% width / reference central-90% width ∈ [0.75, 1.25]</div>
      <p>A <strong>pass</strong> places the sampling-uncertainty interval wholly inside the allowed range; a <strong>fail</strong> places it wholly outside; an <strong>inconclusive</strong> result overlaps a boundary. The evaluator accounts for correlated reference draws and finite neural draws, using approximate family-adjusted uncertainty intervals and two reference block lengths. A combined quantity passes only when both mean and width pass.</p>
      <div class="callout"><strong>Exactly how one case receives its label</strong><p>The network has five parameter targets. We also calculate two variance quantities from the same paired draws. That gives seven quantities × two checks = <strong>14 checks per quote surface</strong>. Any definite failure makes that case fail; all 14 must pass for the case to pass; otherwise it is inconclusive. “Fail” does not mean every parameter is wrong. An unavailable reference also makes a case inconclusive.</p></div>
      <p>For example, market-like κ and θ pass both checks, and the later-window expected variance passes too. The case still fails because ξ’s mean misses by .57 reference SD and ρ’s interval is 54% too wide. The result describes those specific disagreements, not a completely unsuccessful inverse fit.</p>
      <p class="caption">The following case tables and mean/width chart use the predeclared <strong>60k endpoint</strong>. The 55k outcome is also retained; neither endpoint was chosen from the loss curve.</p>'''+svg("conditional")+'''
      <label class="filter"><input type="checkbox" id="hide-passes">Hide rows whose mean and width both pass</label>'''+
      "".join('<article class="case"><h3>'+CASES[cid]+'</h3><p>'+description+'</p>'+conditional_table(latest["cases"][cid])+conditional_table(latest["cases"][cid],True)+'</article>'
        for cid,description in (("synthetic-broad","A synthetic surface generated from a predeclared broad-prior parameter draw. “Broad” describes the source prior, not a broad posterior."),
          ("synthetic-market-like","Synthetic data generated from Heston at parameters near the earlier market fit. Failure here exists even when the assumed model generates the observations.")))+'''
      <p class="caption">The two downstream quantities are analytical Heston expected average variance: a short window from valuation to about day 21, and a later window from about day 81 to day 325. They are not forward-start option prices.</p>
      <details><summary>Actual market quotes: substantially larger neural/reference errors</summary>
      <p>All five physical-parameter mean checks fail. Their absolute discrepancies are about 7.2 SD (v₀), 3.7 SD (κ), 10.0 SD (θ), 20.9 SD (ξ) and 33.9 SD (ρ). This disagreement remains an inference problem to assess; market-model inadequacy is a separate issue and does not by itself excuse disagreement with the matched posterior.</p>'''+conditional_table(latest["cases"]["market"])+conditional_table(latest["cases"]["market"],True)+'''</details>
      <details><summary>Boundary example: unscored because its reference is unresolved</summary><p>The first MCMC attempt failed convergence. Its one escalation exhausted the 2,700-second wall limit. The failure and partial chains remain retained. We have no accepted reference for scoring this neural case; its status is inconclusive.</p></details>
      <p><strong>Why can a small mean error remain inconclusive?</strong> For broad-case ρ, the estimated error is only −0.024 reference SD, but its sampling-uncertainty interval is [−0.310,+0.263], slightly crossing −0.30. Finite neural-draw uncertainty dominates here. Its excessive central interval width is nevertheless a definite failure.</p>''',"checks"))
    sbc_latest=read(EVIDENCE/"sbc-development/annealed-060000.json")
    sbc_rows="".join(f'<tr><th>{LABELS[name]}</th><td>{sbc_latest["rank_dkw"]["rank_ks"][i]:.4f}</td><td>{sbc_latest["rank_dkw"]["dkw_bound"]:.4f}</td><td>Does not reject</td><td>{100*sbc_latest["coverage"]["0.9"]["coverage"][i]:.1f}%</td></tr>' for i,name in enumerate(protocol["names"]))
    parts.append(frame("Simple SBC plots: how it behaves across simulated datasets",'''
      <p>SBC asks a different question. For each of 512 held-out development parameters drawn from the prior, generate a noisy quote surface and draw 512 samples from the network posterior. Rank the generating value among those posterior draws. Across repeated simulations, those ranks should be approximately uniform.</p>
      <pre class="flow">Repeat over 512 simulated datasets:
Prior parameter → Heston quotes + Gaussian error → 512 network posterior draws
                              ↓
             Rank each generating parameter among its draws
                              ↓
        Across datasets, the rank histograms should look approximately flat</pre>
      <p><strong>Why these plots were missing:</strong> the earlier evaluator computed SBC and saved KS/coverage summaries, but explicitly discarded the 512×5 fractional ranks before writing its JSON. The report initially showed those summaries alone. To add the plots, we repeated inference from the same four frozen checkpoints with the original validation rows, noise and sample seeds. The reproduced KS, support and coverage statistics match the saved evaluations exactly. No network was trained, no new synthetic dataset was selected and no reserved test arrays were opened.</p>'''+svg("sbc_histogram")+'''
      <p class="caption">These are the latest 60k ranks, with 10 bins and an expected uniform count of 51.2 per bin. Random departures from a flat histogram are normal. The dashed count is not a simultaneous confidence band.</p>'''+svg("sbc_ecdf")+'''
      <p>The ECDF plot makes the deviations easier to compare across runs: zero means uniform ranks. The green band is the saved simultaneous marginal DKW reference, ±.07193. At 60k, none of the five marginal rank screens rejects. It is not a claim that every conditional posterior is correct.</p>
      <p><strong>Our reading of the curves:</strong> the latest 60k ECDFs look reasonable, and the zero-dropout runs are markedly closer to uniform than the initial dropout run. The initial dropout run is the obvious poor case. The 50k θ curve also departs clearly from the band; smaller departures in other parameters explain why that checkpoint’s numerical screen is less favorable than its overall visual impression. These are encouraging population diagnostics, with the limitations below.</p>
      <div class="table-scroll"><table><thead><tr><th>Parameter</th><th>60k rank KS</th><th>DKW threshold</th><th>Rank screen</th><th>Truth inside nominal 90% interval</th></tr></thead><tbody>'''+sbc_rows+'''</tbody></table></div>
      <p>The coverage figures also show mild overcoverage. At the 90% level, the pointwise Wilson intervals for κ and θ are approximately [90.6%,95.0%] and [90.9%,95.2%], excluding 90%. Those intervals are not adjusted across parameters or coverage levels. Thus “5/5 rank screens do not reject” must not be read as “every calibration diagnostic passes.”</p>
      <div class="callout warn"><strong>Why SBC and the per-dataset failures can coexist</strong><p>SBC averages over prior-generated datasets; biases can vary across datasets and a finite screen has limited power. It does not establish joint dependence or real-market validity. As a hypothetical illustration, returning the prior for every dataset can yield uniform SBC ranks while ignoring the observations. This is not a description of our fitted network; it shows why SBC alone cannot establish accurate use of the quotes.</p></div>
      <p>SBC remains useful for detecting systematic rank distortion across many simulations without calculating an MCMC posterior for every dataset. The fixed-quote comparisons supply the complementary check of whether the network approximates the actual posterior on particular examples.</p>
      <p>The same development pool has been reused across checkpoints and interventions. These plots help diagnose this development round; they are not fresh acceptance evidence for the chosen recipe.</p>''',"sbc"))
    recovery_truth,recovery_quantiles,recovery_bounds=recovery_data(EVIDENCE)
    recovery_stats={}
    recovery_rows=[]
    for i,name in enumerate(protocol["names"]):
        error=recovery_quantiles[1,:,i]-recovery_truth[:,i]
        prior_width=recovery_bounds[i,1]-recovery_bounds[i,0]
        item={"median_absolute_error":float(np.median(np.abs(error))),
              "rmse_as_fraction_of_prior_range":float(np.sqrt(np.mean(error**2))/prior_width),
              "median_central90_width":float(np.median(recovery_quantiles[2,:,i]-recovery_quantiles[0,:,i]))}
        recovery_stats[name]=item
        recovery_rows.append(f'<tr><th>{LABELS[name]}</th><td>{item["median_absolute_error"]:.4g}</td><td>{100*item["rmse_as_fraction_of_prior_range"]:.3f}%</td><td>{item["median_central90_width"]:.4g}</td></tr>')
    parts.append(frame("Parameter-wise recovery against the unit line",'''
      <p>Each grey point represents one of the same 512 simulated development datasets at the latest 60k checkpoint: the generating parameter is on the x-axis, and its marginal posterior median is on the y-axis. The dashed line is y=x. Each panel uses identical x/y limits over its physical prior range.</p>'''+svg("recovery")+'''
      <p><strong>Purple vertical bars show marginal central 90% posterior intervals</strong> (5th–95th percentiles), not Monte Carlo uncertainty in the median. To keep the figure readable, bars appear for 32 cases only: validation indices 0,16,…,496, the same IDs in every panel. Selection depends only on index, not how good the recovery looks. All 512 medians remain visible.</p>
      <p>The bulk of points tracks the diagonal strongly. There are also a few sizeable departures, particularly in θ and ρ. A posterior median need not equal the generating truth on every noisy dataset; weakly informative observations can leave appreciable uncertainty. Reading recovery together with the intervals and SBC is more useful than treating every off-diagonal point as an approximation failure.</p>
      <div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Median absolute recovery error</th><th>RMSE / full prior range</th><th>Median marginal 90% width</th></tr></thead><tbody>'''+"".join(recovery_rows)+'''</tbody></table></div>
      <p class="caption">These descriptive point-recovery statistics use all 512 datasets and the saved quantile summaries. They compare posterior medians with generating truths, unlike the separate fixed-quote checks comparing network and MCMC posteriors. Some intervals are barely visible on the full prior scale.</p>
      <details><summary>Zoom in: errors and posterior widths for the same 32 interval cases</summary>'''+svg("recovery_errors")+'''
      <p class="caption">The vertical axis is posterior median minus truth, in each parameter’s physical units; vertical scales differ across panels. A bar crossing zero contains the truth. These are the same interval cases shown in purple above, not a new or favorably selected sample.</p></details>
      <div class="callout"><strong>How this fits with the other results</strong><p>The recovery plots and latest ECDF curves support a positive assessment of broad parameter recovery and average marginal calibration. The fixed-quote MCMC checks demand finer precision relative to very narrow posterior SDs. Strong diagonal tracking and reasonable pooled ranks can therefore coexist with the remaining mean/width discrepancies. Both findings should be reported.</p></div>
      <p>No new training or posterior sampling was needed: this figure uses the existing hash-verified truth array and 5th/50th/95th-percentile summaries from the reproduced 60k development inference.</p>''',"recovery"))
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
      <div class="table-scroll"><table><thead><tr><th>Recipe / endpoint</th><th>Updates</th><th>Broad physical width range</th><th>Development rank screens not rejecting</th><th>Network/MCMC agreement on checked quote sets</th></tr></thead><tbody>'''+"".join(stage_rows)+'''</tbody></table></div>
      <p class="caption">The last column is an aggregate label: any definite case failure makes that checkpoint’s checked-panel result fail. It is not a percentage of accurate posteriors. Case-level details and individual mean/width results are shown above.</p>
      <p>Removing dropout changed a matched 10k control substantially. Continuing to 50k improved widths but left unstable centering. Eight discarded future updates showed that the original 5e-5 learning-rate floor could move broad-case means by .275–.818 reference SD RMS. That prompted one bounded schedule change: 5e-6 for 5k updates, then 5e-7 for 5k. Adam moments, iteration and all saved state were retained.</p>
      <p>The annealing fit plus both evaluations completed in 99.86 seconds. At 60k the population rank checks pass 5/5 and 90% coverage is .916–.934, yet both synthetic conditional panels fail. These are repeated development checks on 512 cases, not fresh acceptance evidence. Lower loss and acceptable average ranks can coexist with errors for individual quote surfaces.</p>''',"training"))
    architecture=read(EVIDENCE/"network-settings.json")
    workflow_code=source_function(sources["Actual fixed-c workflow and zero-dropout code"],"build_workflow")
    zero_code=source_function(sources["Actual fixed-c workflow and zero-dropout code"],"switch_dropout_zero")
    rate_code=source_function(sources["Actual lower-rate continuation"],"make_schedule")
    parts.append(frame("Exactly which network, dropout and learning rates were used?",f'''
      <p><strong>There are two fresh runs, not one run with dropout removed at 10k.</strong> Branch A stops at 10k. Branch B starts afresh with matched initialization, rows and noise streams, effective dropout zero, and is then continued. The 60k result belongs entirely to Branch B.</p>
      <pre class="flow">Branch A: fresh matched construction → dropout .05 → 10k → STOP

Branch B: same construction; change existing dropout rates to 0 before training
          → fresh 0–10k fit → same Adam state to 50k
          → same Adam state, lower rates to 55k and 60k → STOP</pre>
      <h3>Network and preprocessing</h3>
      <div class="table-scroll"><table class="settings"><thead><tr><th>Setting</th><th>Actual value</th></tr></thead><tbody>
      <tr><td>Inference family</td><td>BayesFlow 2.1.0 spline CouplingFlow, Keras 3.15.1 / JAX 0.11.1</td></tr>
      <tr><td>Observations</td><td>54 complete ordered noisy prices divided by their fixed half-spreads, then learned workflow standardization. No variable locations, missing-quote masks or summary network.</td></tr>
      <tr><td>Posterior targets</td><td>Five physical Heston parameters: v₀, κ, θ, ξ, ρ. Bounded parameters use the adapter’s logit constraints and learned standardization. c=1 is fixed; no c or τ target is learned.</td></tr>
      <tr><td>Flow blocks</td><td>6; each contains ActNorm, a fixed random permutation and a dual coupling. Two conditional MLPs per block: 12 MLPs total.</td></tr>
      <tr><td>Conditioner MLP</td><td>Two hidden layers of 128 units each. Dense → hard_silu → Dropout at each hidden layer; orthogonal initialization, no residual connection or hidden normalization. A zero-initialized Dense projector produces spline parameters.</td></tr>
      <tr><td>Spline transformation</td><td>16 rational-quadratic bins; learned domain initialized at [−3,3] with affine tails. This is not a posterior support cutoff at ±3.</td></tr>
      <tr><td>Sampling</td><td>Draw a five-dimensional standard Normal latent and transform it conditional on the quotes, then map back to the physical bounds. Posterior draw randomness comes from this latent, not dropout.</td></tr>
      <tr><td>Size</td><td>{architecture['resolved_architecture']['trainable_scalar_count']:,} learned scalars in 84 trainable arrays. 24 Dropout layers were constructed; their rate is .05 in Branch A and zero in Branch B.</td></tr>
      <tr><td>Data and batches</td><td>20k distinct clean training surfaces; fresh independent Gaussian quote noise on each exposure; batch 128, with a final short batch of 32 in each pass; training seed 9011.</td></tr>
      </tbody></table></div>
      <p><strong>Actual construction code</strong>, extracted from the frozen trainer. The constructor visibly sets <code>dropout=.05</code>; the rest of the resolved defaults are listed in the table above.</p>'''+code_block(workflow_code)+'''
      <h3>Why was dropout present?</h3>
      <p>The spline pilot explicitly carried over the .05 coupling-MLP default. There was no financial reason to require dropout. During training, it sets each hidden activation to zero with probability 5% and scales retained activations by 1/.95. This is neural regularization; it does not remove quotes, add price noise or infer market uncertainty.</p>
      <p><strong>Validation and posterior inference disable dropout.</strong> We did not use Monte Carlo dropout. The original FlowMatching builder’s time-MLP default was dropout 0.0, so using .05 here was another change from that recipe. It should have been explicit in the comparison from the beginning.</p>
      <p>The matched zero-dropout control first constructs/builds the original .05 model, then changes the existing 24 rates to zero before its fresh training run. The switch is checked to leave all initialized arrays unchanged, and the original saved deterministic initial probes match. The original run did not save its full initial array archive, so a direct archive-to-archive initial-state comparison is unavailable. Directly constructing a new network with <code>dropout=0</code> is not claimed to be an exact replay of this control.</p>
      <p>The saved archives retain the original .05 constructor configuration. The strict checkpoint loader reapplies the recorded effective zero rates and verifies saved state and seeded inference. This explains why inspecting a constructor field alone can show .05 even for the zero-dropout control.</p>
      <details><summary>Actual zero-dropout switch and its state checks</summary>'''+code_block(zero_code)+'''</details>
      <h3>Optimizer and complete learning-rate timeline</h3>
      <p>Adam uses β₁=.9, β₂=.999, ε=1e−7, AMSGrad=False, no weight decay, no gradient clipping, no EMA and no gradient accumulation. The original lesson’s separate AdamW recipe is not used here.</p>
      <p>The first engineering run was budgeted for 10k updates with a 10%-of-initial-rate floor. The later 10k→50k continuation deliberately retained that schedule to test additional updates with the same recipe. The update-scale diagnostic then motivated the two lower-rate stages. Thus 60k is a sequence of separately declared development interventions, not one schedule chosen before the first fit. These are documented engineering choices, not established optimal settings.</p>
      <div class="table-scroll"><table><thead><tr><th>Branch</th><th>Pre-update iteration t</th><th>Learning rate</th><th>State handling</th></tr></thead><tbody>
      <tr><th>A and B fresh fits</th><td>0–9,999</td><td>Cosine decay from 5e−4 toward 5e−5; horizon 10,000; alpha=.1</td><td>Separate fresh fits; matched initial construction and data/noise schedule. A stops after update 10,000.</td></tr>
      <tr><th>B continuation</th><td>10,000–49,999</td><td>5e−5</td><td>Original cosine horizon remains 10k, so this is its constant floor. Adam moments/iteration retained.</td></tr>
      <tr><th>B lower-rate stage 1</th><td>50,000–54,999</td><td>5e−6</td><td>All saved state retained. Only the learning-rate schedule changes.</td></tr>
      <tr><th>B lower-rate stage 2</th><td>55,000–59,999</td><td>5e−7</td><td>Same stateful run. Stops after update 60,000.</td></tr>
      </tbody></table></div>
      <p class="caption">t is the zero-based optimizer iteration before an update; the update number is t+1. Thus the checkpoint named 55k has completed updates at t=0,…,54,999.</p>
      <div class="equation">For the first 10k: η(t)=5×10<sup>−5</sup> + ½(5×10<sup>−4</sup>−5×10<sup>−5</sup>)[1+cos(π min(t,10,000)/10,000)]</div>
      <p>The later schedule is created by the following actual function, with <code>BOUNDARY=54999</code> and <code>RATES=(5e-6,5e-7)</code>:</p>'''+code_block(rate_code)+'''
      <p>It is assigned using <code>model.optimizer.learning_rate = make_schedule()</code> after restoring the 50k checkpoint. Assignment verifies every saved array remains unchanged; Adam is not re-created. Existing historical producers and checkpoints remain unchanged. These snippets document the runs that produced the plots, not a newly executed training recipe.</p>''',"settings"))
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
      <tr><th>Hidden-layer dropout</th><td>Time-MLP default 0.0</td><td>Explicit .05 in initial branch; zero throughout matched control and its continuations</td></tr>
      <tr><th>Optimizer</th><td>AdamW; weight decay .005; per-variable clipnorm 1.5</td><td>Adam; no weight decay or clipping</td></tr>
      <tr><th>Learning-rate recipe</th><td>100k candidate: 9,775-update warm-up from 5e−5 to 5e−4, then 185,725-update cosine decay toward zero; scheduled zero at t=195,500</td><td>10k cosine 5e−4→5e−5; floor through 50k; 5e−6 to 55k; 5e−7 to 60k on the zero-dropout branch</td></tr>
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
      <p>Half-spread-as-standard-deviation is a simplifying convention, not a measured Gaussian-error law. The five-target fits reported here keep c fixed. The separate <a href="noise-scale/report.html">noise-scale study</a> now trains and examines an inferred-c target.</p>
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
          "network_settings":{key:architecture[key] for key in ("resolved_architecture","optimizer","learning_rate_timeline")},
          "recovery":{"checkpoint_updates":60000,"datasets":512,"interval_levels":[.05,.95],
                      "interval_case_indices":list(range(0,512,16)),"summary_by_parameter":recovery_stats,
                      "point_estimate":"posterior median","scope":"saved float32 quantile summaries; no new sampling"},
          "sbc":{ident:{key:record[key] for key in ("parameter_names","rows","draws_per_case","seed","fractional_ranks","rank_dkw","coverage","metrics_exact_match")}
                 for ident in ("default-010000","zero-010000","zero-050000","annealed-060000")
                 for record in [read(EVIDENCE/"sbc-development"/(ident+".json"))]},
          "scope":"Renderer reads frozen records only. SBC ranks separately reproduce existing checkpoint inference with exact original metrics; no fitting, option pricing or reserved-array access."}
    parts.append(frame("Sources, scope and reproducibility",'''
      <p>This file is self-contained: charts, styles, equations and tables render offline without a math service or JavaScript library. Source links are optional GitHub links to the private project. Neither source access nor scripts are required to read the report. Use the print button for a paper copy; the data button exports the source inventory, reproduced SBC ranks/statistics, resolved network/rate settings and noise-comparison statistics.</p>
      <p>The main new evidence uses 4,096 neural draws per conditional query and 512 development cases with 512 draws each for population checks. The SBC plot extraction repeats the four existing checkpoint population queries with the exact original observations and seeds and verifies their metrics exactly; it creates no new training results. The reserved test pool remains unopened. Conditional mean/width tests are useful screens, not a complete test of joint shape or tails. The original and current panels differ, so their pass counts are not a controlled head-to-head score.</p>
      <details><summary>Input files and SHA-256 identities</summary><div class="table-scroll"><table class="sources"><thead><tr><th>Source</th><th>Repository path</th><th>SHA-256</th></tr></thead><tbody>'''+"".join(source_rows)+'''</tbody></table></div></details>
      <p>Primary external reference for the market volatility-of-volatility analogue: <a href="https://www.cboe.com/us/indices/dashboard/vvix/">Cboe VVIX description and FAQ</a>. Model interpretation and local numerical scales above follow from the explicitly stated Heston SDE; the daily scale is a derivation, not an empirical calibration.</p>''',"sources"))
    payload=json.dumps(data,ensure_ascii=False,allow_nan=False).replace("<","\\u003c")
    parts.extend(['<footer>Prepared from completed development records. No accepted checkpoint. Report date: 10 November 2026.</footer>',
        '<script id="report-data" type="application/json">'+payload+'</script>',r'''<script>
      document.getElementById('hide-passes').addEventListener('change',function(){document.querySelectorAll('tr[data-status="pass"]').forEach(row=>row.hidden=this.checked)});
      document.getElementById('download').addEventListener('click',function(){const text=JSON.stringify(JSON.parse(document.getElementById('report-data').textContent),null,2);const url=URL.createObjectURL(new Blob([text],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='heston-report-data.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});
      </script></main></body></html>'''])
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    rendered = "\n".join(line.rstrip() for line in "".join(parts).splitlines()) + "\n"
    out.write_text(rendered,encoding="utf-8")
    print(json.dumps({"report":str(out),"bytes":out.stat().st_size,"sha256":digest(out),"sources":len(sources),"scientific_acceptance":False}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=EVIDENCE/"report.html")
    args=parser.parse_args()
    build(args.historical_root,args.out)


if __name__=="__main__":
    main()
