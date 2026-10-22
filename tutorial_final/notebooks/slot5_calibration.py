import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from sbi4finance.calibration import plots, teaching

    ROOT = mo.notebook_dir().parent
    RESULTS = ROOT / "results" / "calibration"
    CHECKPOINTS = ROOT / "checkpoints"
    plots.style()

    def pending(what):
        return mo.callout(mo.md(f"**Pending:** {what}"), kind="warn")

    def image(path, caption, width=None):
        found = teaching.existing(path)
        if found is None:
            return pending(f"`{path.relative_to(ROOT)}` is not in the results yet.")
        return mo.image(src=found, caption=caption, width=width)

    return CHECKPOINTS, RESULTS, image, mo, np, pending, plt, teaching, time


@app.cell
def _(mo):
    mo.md(r"""
    # Ten calibrations fit the smile. Which price do you mark?

    Every morning a derivatives desk calibrates a volatility model to the option smile and
    marks the products the vanillas do not price: forward-start options, volatility swaps,
    long-dated puts. Least squares returns one parameter vector. Several fit the smile within
    the bid–ask spread, and they price those products differently. The rules ask for exactly
    that range: a point at 90% confidence among plausible calibrations (Regulation (EU)
    2016/101, Art. 11(3)).

    This lesson computes the whole set of plausible calibrations, the **posterior**, for
    **rough Bergomi**, a model whose prices exist only by Monte Carlo. It does so in seconds
    per smile, checks the answer against exact posteriors, and turns it into a price range and
    a reserve.

    **Two claims.**

    1. Calibration has a posterior. The products the smile does not pin down carry a range and
       a reserve; the variance swap, which the smile pins down, carries none.
    2. For a simulation-only model the exact posterior costs hours per smile. The amortized
       posterior costs seconds, and it can be checked.

    The technical design document develops every step:
    [calibration design document](https://github.com/fmuia/sbi4finance/blob/main/docs/design/calibration-design-document.md).
    """)
    return


@app.cell
def _(RESULTS, mo, teaching):
    _checks = [
        ("Example surfaces (rough Bergomi, Heston)", RESULTS / "smile/example-rough.json"),
        ("Heston: SBC and recovery (S2)", RESULTS / "heston/s2/validation.json"),
        ("Heston: exact posterior check (S3)", RESULTS / "heston/s3/heston-check.json"),
        ("Heston: neural-pricer route (S4)", RESULTS / "heston/s4/pricer-route.json"),
        ("Heston: classical route (S6)", RESULTS / "heston/s6/classical.json"),
        ("Heston: products and reserves (S7)", RESULTS / "heston/s7/products.json"),
        ("Rough Bergomi pilot: SBC and recovery (S2)", RESULTS / "rough-pilot/s2/validation.json"),
        ("Rough Bergomi pilot: Monte Carlo audit (S5)", RESULTS / "rough-pilot/s5/audit.json"),
        ("Costs on one machine (S9)", RESULTS / "costs/costs.json"),
        ("Real SPY smile (S8)", RESULTS / "spy/surface.json"),
    ]
    _rows = [{"result": name, "status": "ready" if teaching.existing(path) else "pending"} for name, path in _checks]
    mo.vstack([
        mo.md("**What this notebook reads.** Every number below comes from an accepted study in "
              "`tutorial_final/results/calibration`; the rough-Bergomi results are a pilot (1,600 training "
              "surfaces) until the full bank of 100,000 has run on the cluster. Only the cells marked *live* "
              "compute anything."),
        mo.ui.table(_rows, selection=None, pagination=False),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. One volatility cannot fit the smile

    Under Black–Scholes the price follows a geometric Brownian motion under the pricing
    measure **Q**, and one volatility prices every strike and every maturity. Quote each
    option by the volatility that reproduces its price, its *implied volatility*, and the
    market's quotes form a **smile**: they slope down in strike (the *skew*), and the skew
    steepens as maturity shortens, close to a power law in maturity. For SPX the exponent is
    about −0.4.

    Slot 4's posterior forecast returns under the real-world measure **P**, from returns. This
    one prices, under **Q**, from prices. The data are now 63 option quotes: 7 maturities from
    one week to one year, 9 strikes each, placed at the same standardized distance from the
    forward at every maturity.
    """)
    return


@app.cell
def _(mo, teaching):
    maturity = mo.ui.dropdown(options={label: days for days, label in teaching.TENORS.items()}, value="1 month",
                              label="maturity")
    maturity
    return (maturity,)


@app.cell
def _(RESULTS, maturity, pending, plt, teaching):
    _rough = teaching.load_json(RESULTS / "smile/example-rough.json")
    _heston = teaching.load_json(RESULTS / "smile/example-heston.json")
    if _rough is None or _heston is None:
        _out = pending("the example surfaces (`tutorial_final/studies/smile_facts.py example`).")
    else:
        _fig, (_left, _right) = plt.subplots(1, 2, figsize=(10.5, 3.9), constrained_layout=True)
        teaching.flat_versus_smile(_rough, maturity.value, _left)
        _left.set_title(f"rough Bergomi at the centre of its prior, {teaching.TENORS[maturity.value]}", loc="left",
                        fontsize=9)
        teaching.skew_term_structures({"rough Bergomi": _rough, "Heston": _heston}, _right)
        _right.set_title("the skew's term structure, log–log", loc="left", fontsize=9)
        _out = _fig
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    Two models will explain the smile. **Heston** gives the variance its own diffusion,
    correlated with the price; its skew tends to a constant at short maturities (the orange
    line flattens). **Rough Bergomi** drives the log-variance with a fractional process of
    Hurst exponent $H$; its skew grows like $T^{H-1/2}$ as $T \to 0$, which is what index
    smiles do. At the centre of its prior, $H = 0.185$, the fitted exponent over one week to one
    year is −0.38, steeper than the limit $H - \tfrac12 = -0.32$: the power law is a
    short-maturity statement.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Heston: the model with an exact answer

    $$\frac{dS_t}{S_t} = \sqrt{V_t}\,dW^S_t, \qquad dV_t = \kappa(\theta - V_t)\,dt + \sigma_v\sqrt{V_t}\,dW^V_t,
    \qquad d\langle W^S, W^V\rangle_t = \rho\,dt .$$

    Heston is affine, so the characteristic function of the log price is known in closed form
    and each option price is one Fourier integral. The 63 prices of a smile cost about 35
    milliseconds, so the **likelihood is exact and cheap**:

    $$p(y \mid \theta) = \prod_{c=1}^{63} \mathcal{N}\!\big(y_c;\ P_c(\theta),\ \sigma_c^2\big),$$

    with $\sigma_c$ the quote noise: a third of a volatility point at the money, one point in
    the wings. The exact posterior can then be computed twice, by Markov chain Monte Carlo and
    by importance reweighting. This is the check: the slot-4 workflow (simulate a bank, train
    a flow-matching network, sample) with a new simulator, compared with the exact answer.
    """)
    return


@app.cell
def _(RESULTS, mo, pending, teaching):
    _report = teaching.load_json(RESULTS / "heston/s3/heston-check.json")
    if _report is None:
        _out = pending("study S3 (`tutorial_final/studies/heston_check.py`).")
    else:
        _s = _report["summary"]
        _out = mo.ui.table([
            {"check": "surfaces where network and exact posterior agree (means within 0.3 sd, 90% widths within 25%)",
             "value": f"{100 * _s['network_vs_reweighted_pass_rate']:.0f}% of {len(_report['surfaces'])}"},
            {"check": "median importance-sampling efficiency of the network's proposal",
             "value": f"{100 * _s['audit_efficiency_median']:.0f}%"},
            {"check": "surfaces with Pareto k̂ above 0.7", "value": str(_s["audit_khat_above_0_7"])},
            {"check": "exact chains: largest R̂, smallest effective sample size",
             "value": f"{_s['chain_rhat_max']:.3f}, {_s['chain_ess_min']:.0f}"},
            {"check": "network: seconds per smile for 1,000 draws", "value": f"{_s['network_seconds_per_surface']:.2f}"},
            {"check": "exact reweighting: CPU-seconds per smile", "value": f"{_s['audit_cpu_seconds_median']:.0f}"},
        ], selection=None, pagination=False)
    _out
    return


@app.cell
def _(RESULTS, image):
    image(RESULTS / "heston/s3/agreement.png",
          "Four Heston smiles: the network (blue) and the network reweighted with the exact likelihood (green) "
          "against the exact posterior (grey), standardized by the exact posterior; the star is the truth.")
    return


@app.cell
def _(RESULTS, mo, teaching):
    heston_draws = teaching.load_npz(RESULTS / "heston/s3/posterior-draws.npz")
    heston_rows = sorted(int(k.split("_")[1]) for k in (heston_draws or {}) if k.startswith("chain_"))
    heston_surface = mo.ui.dropdown(options=[str(r) for r in heston_rows], value=str(heston_rows[0]) if heston_rows else None,
                                    label="Heston test surface")
    heston_surface if heston_rows else mo.md("")
    return heston_draws, heston_rows, heston_surface


@app.cell
def _(heston_draws, heston_rows, heston_surface, pending, teaching):
    if not heston_rows:
        _out = pending("the exact and network draws of study S3.")
    else:
        _row = int(heston_surface.value)
        _labels = [teaching.LABELS[n] for n in ("initial_variance", "theta", "vol_of_vol", "rho")]
        _out = teaching.pairs({"exact posterior (MCMC)": heston_draws[f"chain_{_row}"],
                               "network": heston_draws[f"network_{_row}"]},
                              _labels, truth=heston_draws[f"truth_{_row}"])
    _out
    return


@app.cell
def _(mo):
    run_heston = mo.ui.run_button(label="Live: sample the Heston network on this smile")
    run_heston
    return (run_heston,)


@app.cell
def _(CHECKPOINTS, heston_draws, heston_surface, mo, pending, run_heston, teaching, time):
    mo.stop(not run_heston.value, mo.md("*Press the button to load the network and draw 1,000 posterior samples.*"))
    if teaching.existing(CHECKPOINTS / "heston" / "model.keras") is None or not heston_draws:
        _out = pending("the Heston checkpoint in `tutorial_final/checkpoints/heston`.")
    else:
        from sbi4finance.calibration import inference as _inference

        _task, _workflow = teaching.load_network(CHECKPOINTS / "heston")
        _row = int(heston_surface.value)
        _observation = heston_draws[f"observation_{_row}"][None, :]
        _started = time.perf_counter()
        _draws = _inference.sample(_workflow, _task, _observation, None, 1000, seed=1)[0]
        _seconds = time.perf_counter() - _started
        _labels = [teaching.LABELS[n] for n in _task["parameter_names"]]
        _out = mo.vstack([
            mo.md(f"1,000 posterior draws in **{_seconds:.2f} s** on this machine (including the first call's "
                  "compilation), against minutes for the exact chain."),
            teaching.pairs({"exact posterior (MCMC)": heston_draws[f"chain_{_row}"], "network, live": _draws},
                           _labels, truth=heston_draws[f"truth_{_row}"]),
        ])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Rough Bergomi: prices only by simulation

    $$V_t = \xi_0(t)\,\exp\!\Big(\eta\,Y_t - \tfrac12\eta^2 t^{2H}\Big), \qquad
    Y_t = \sqrt{2H}\int_0^t (t-s)^{H-\frac12}\,dW_s, \qquad
    \frac{dS_t}{S_t} = \sqrt{V_t}\,\big(\rho\,dW_t + \sqrt{1-\rho^2}\,dW^\perp_t\big).$$

    * $\xi_0(t)$ is today's forward-variance curve, read off the option chain; the model
      matches the variance-swap curve by construction, whatever $H, \eta, \rho$ are.
    * $H$ sets the roughness, $\eta$ the volatility of variance, $\rho$ the correlation.
    * The variance depends on the whole past of $W$: there is no finite state, so no Fourier
      or PDE pricer. **Every price is a Monte Carlo run.**

    One smile costs 6.4 CPU-seconds at $2^{15}$ paths per maturity block, with antithetic
    paths and two control variates. An exact posterior needs about $10^4$ likelihood
    evaluations, so roughly a CPU-day per smile, every morning, per underlying.
    """)
    return


@app.cell
def _(RESULTS, image):
    image(RESULTS / "smile/example-rough.png",
          "One simulated rough-Bergomi smile at the centre of the prior: the 63 noisy quotes and the model smile (left); "
          "the skew's term structure with the fitted power law and the short-maturity formula (right).")
    return


@app.cell
def _(RESULTS, mo, pending, teaching):
    _costs = teaching.load_json(RESULTS / "costs/costs.json")
    if _costs is None:
        _out = pending("the cost table of study S9 (`tutorial_final/studies/costs.py`).")
    else:
        _out = mo.vstack([mo.md(f"Costs on one machine ({_costs['cpu_count']} cores), in CPU time:"),
                          mo.ui.table(teaching.cost_rows(_costs), selection=None, pagination=False)])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. The amortized posterior

    Pay for inference once, in training. Draw parameters and a forward-variance curve from the
    prior, price the 63 contracts by Monte Carlo, add quote noise, and repeat: a bank of
    simulated smiles. A flow-matching network learns to carry Gaussian noise to the posterior,
    conditioned on the quotes and the curve. For a new smile, sampling the posterior is one
    pass of the network.

    What the smile identifies: the skew's level fixes the product $\rho\eta$; its term
    structure fixes $H$; $\eta$ and $\rho$ separately only through the curvature in the wings,
    where the quotes are noisiest. Expect a ridge along $\rho\eta = $ constant.
    """)
    return


@app.cell
def _(RESULTS, mo, teaching):
    pilot = teaching.load_npz(RESULTS / "rough-pilot/s2/draws.npz")
    pilot_surface = mo.ui.slider(0, (len(pilot["truth"]) - 1) if pilot else 0, value=0, label="pilot test surface",
                                 show_value=True)
    pilot_surface if pilot else mo.md("")
    return pilot, pilot_surface


@app.cell
def _(pending, pilot, pilot_surface, teaching):
    if not pilot:
        _out = pending("the rough-Bergomi pilot network's draws (study S2 on the pilot bank).")
    else:
        _i = pilot_surface.value
        _out = teaching.pairs({"network": pilot["draws"][_i]}, ["H", "η", "ρ", "ρη"], truth=pilot["truth"][_i],
                              figsize=(7.6, 7.6))
    _out
    return


@app.cell
def _(RESULTS, image):
    image(RESULTS / "rough-pilot/s2/recovery.png",
          "Recovery on held-out pilot surfaces: posterior median and 90% interval against the true value.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Is it right?

    A learned posterior can be wrong in ways the training loss does not show. Four checks,
    each against a known answer:

    1. **Simulation-based calibration.** On held-out simulated smiles the true parameters are
       known. If the posterior is right, the truth's rank among the posterior draws is uniform.
    2. **The exact Heston posterior** (section 2).
    3. **The audit**, on every smile: reweight the network's draws with the Monte Carlo
       likelihood. Nearly equal weights mean the network was right for this smile; a heavy
       weight tail (Pareto $\hat k > 0.7$) flags it.
    4. **A second simulation-based route**: a neural pricer trained on the same bank, with MCMC
       on its likelihood (deep calibration's pricer, sampled rather than optimized).
    """)
    return


@app.cell
def _(RESULTS, image, mo):
    mo.vstack([
        image(RESULTS / "heston/s2/sbc.png", "Heston: rank ECDF minus uniform on 2,000 held-out smiles, with the "
              "simultaneous 95% band."),
        image(RESULTS / "rough-pilot/s2/sbc.png", "Rough Bergomi, pilot network: the same check."),
    ])
    return


@app.cell
def _(RESULTS, mo, pending, teaching):
    _audit = teaching.load_json(RESULTS / "rough-pilot/s5/audit.json")
    if _audit is None:
        _out = pending("the Monte Carlo audit of the pilot network (study S5).")
    else:
        _s = _audit["summary"]
        _out = mo.ui.table([
            {"audit, rough Bergomi pilot": "surfaces audited", "value": str(_s["surfaces"])},
            {"audit, rough Bergomi pilot": "efficiency: 5%, median, 95%",
             "value": ", ".join(f"{100 * _s['efficiency_quantiles'][q]:.0f}%" for q in ("q05", "q50", "q95"))},
            {"audit, rough Bergomi pilot": "flagged by k̂ > 0.7", "value": f"{100 * _s['flag_rate_khat_above_0_7']:.0f}%"},
            {"audit, rough Bergomi pilot": "CPU-seconds per smile", "value": f"{_s['cpu_seconds_per_surface_median']:.0f}"},
        ], selection=None, pagination=False)
    _out
    return


@app.cell
def _(RESULTS, image):
    image(RESULTS / "heston/s4/agreement.png",
          "Heston: MCMC on a neural pricer against the amortized posterior; two simulation-based routes that share "
          "only the bank.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. The classical route on the same smiles

    The desk's tool is weighted least squares from several starting points, with standard
    errors from the Gauss–Newton approximation $(J^\top W J)^{-1}$. It returns a point and an
    ellipse. Where the objective has a valley the ellipse cannot follow it, and a
    regularized fit picks a point along the valley according to the penalty, not the data.
    """)
    return


@app.cell
def _(RESULTS, image, mo, pending, teaching):
    _report = teaching.load_json(RESULTS / "heston/s6/classical.json")
    if _report is None:
        _table = pending("study S6 (`tutorial_final/studies/classical_route.py`).")
    else:
        _s = _report["summary"]
        _names = _report["parameters"]
        _rows = []
        for _n, _label in zip(_names, _report["labels"]):
            _gn = (_s["gauss_newton_coverage_90"] or {}).get(_n)
            _rows.append({"parameter": _label,
                          "Gauss–Newton 90% interval covers the truth": "–" if _gn is None else f"{100 * _gn:.0f}%",
                          "posterior 90% interval covers the truth": f"{100 * _s['posterior_coverage_90'][_n]:.0f}%"})
        _table = mo.vstack([mo.md(f"Heston, {_s['surfaces']} smiles, {_s['starts']} starts each: "
                                  f"{100 * _s['best_on_bound_share']:.0f}% of best fits on a bound; "
                                  f"{_s['gauss_newton_singular']} singular Gauss–Newton matrices."),
                            mo.ui.table(_rows, selection=None, pagination=False)])
    mo.vstack([_table, image(RESULTS / "heston/s6/landscape.png",
                             "The least-squares objective over two parameters, the multi-start optima, the Gauss–Newton "
                             "90% region and the posterior draws, for one smile.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. From posterior to prices: the range and the reserve

    Each posterior draw prices each product by Monte Carlo, with the same random numbers for
    every draw, so the spread of prices is the parameters' and not the simulation's. The
    least-squares calibration prices them too. The **reserve** for a sold position is the
    distance from the least-squares price to the chosen quantile of the posterior prices.
    """)
    return


@app.cell
def _(RESULTS, mo, teaching):
    products = teaching.load_npz(RESULTS / "heston/s7/price-draws.npz")
    product_rows = sorted(int(k.split("_")[1]) for k in (products or {}) if k.startswith("prices_"))
    level = mo.ui.slider(0.80, 0.99, step=0.01, value=0.90, label="confidence level", show_value=True)
    product_surface = mo.ui.dropdown(options=[str(r) for r in product_rows],
                                     value=str(product_rows[0]) if product_rows else None, label="Heston test smile")
    mo.hstack([level, product_surface]) if product_rows else mo.md("")
    return level, product_rows, product_surface, products


@app.cell
def _(level, mo, pending, product_rows, product_surface, products, teaching):
    if not product_rows:
        _out = pending("the product prices of study S7.")
    else:
        _row = int(product_surface.value)
        _out = mo.vstack([
            mo.ui.table(teaching.reserve_rows(products, _row, level.value), selection=None, pagination=False),
            teaching.price_histograms(products, _row, level.value,
                                      f"Heston, test smile {_row}: product prices under the posterior"),
        ])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    Under Heston the variance swap carries a range, because Heston's forward-variance curve
    $\theta + (v_0 - \theta)e^{-\kappa t}$ depends on its parameters. Under rough Bergomi the
    curve is an input, $E[V_t] = \xi_0(t)$ for every $(H, \eta, \rho)$, so the variance swap's
    range is zero by construction: the control of the pipeline. The forward-start option
    depends on the forward smile, which the vanillas do not fix, and carries the largest
    reserve.

    ## 8. Within one model, and across models

    The posterior range is a statement within one model. Heston and rough Bergomi calibrated
    to the same smile price the forward-start differently; that gap is model risk in the
    other sense, and the posterior range is its lower bound, not its measure.

    **Exercises.** Double the quote noise and watch the reserve (a reweighting, no retraining).
    Drop the two shortest maturities and watch $H$. Calibrate Heston to a rough-Bergomi smile
    (the misspecification slot 6 takes up).
    """)
    return


if __name__ == "__main__":
    app.run()
