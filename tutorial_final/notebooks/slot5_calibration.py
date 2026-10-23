import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="Slot 5 · Rough Bergomi on the SPY smile",
)


@app.cell(hide_code=True)
def _():
    import os
    import time

    os.environ.setdefault("KERAS_BACKEND", "jax")

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from sbi4finance.calibration import plots, surface, teaching

    ROOT = mo.notebook_dir().parent
    RESULTS = ROOT / "results" / "calibration"
    CHECKPOINTS = ROOT / "checkpoints"
    CURRICULUM = ROOT / "curriculum"
    plots.style()


    def pending(what):
        return mo.callout(mo.md(f"**Pending:** {what}"), kind="warn")


    def pilot_banner(what):
        return mo.callout(mo.md(f"**Pilot result, not for the stage:** {what} The full study replaces it."), kind="neutral")


    def image(path, caption, width=None):
        found = teaching.existing(path)
        if found is None:
            return pending(f"`{path.relative_to(ROOT)}` is not in the results yet.")
        return mo.image(src=found, caption=caption, width=width)

    return (
        CHECKPOINTS,
        CURRICULUM,
        RESULTS,
        image,
        mo,
        np,
        pending,
        pilot_banner,
        plots,
        plt,
        surface,
        teaching,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Slot 5 · Rough Bergomi on the SPY smile: from the posterior to a reserve

    **The smile fits many calibrations. Which price do you mark?**

    Slot 4 calibrated Heston to the SPY smile and checked every step against the exact posterior. The
    posterior was precise, and it said Heston fits SPY only with a vol-of-vol about twice its Feller limit
    and variance swaps below the market. This notebook keeps the smile and the code and changes the model
    to one built for index smiles, **rough Bergomi**. That change removes the exact posterior, and it is
    where amortized inference pays.

    Three claims, each backed by evidence below:

    1. **Rough Bergomi is priced only by Monte Carlo**, so its exact posterior costs at least 17 CPU-hours
       per smile.
    2. **The amortized posterior takes 1.4 seconds**, and without an exact answer it is checked three ways:
       simulation-based calibration, a second method, and an audit of each smile.
    3. **Under the posterior, the products the smile does not quote carry a range**, and the forward-start's
       90% quantile against the least-squares mark is the reserve the rules ask for. The range is
       conditional on the prior, the noise model, the forward-variance curve and the model.
    """)
    return


@app.cell(hide_code=True)
def _(CHECKPOINTS, RESULTS, mo, teaching):
    _checks = [
        ("Real SPY smile (S8)", RESULTS / "spy/surface.json"),
        ("Costs on one machine (S9)", RESULTS / "costs/costs.json"),
        ("Rough Bergomi network, full bank", CHECKPOINTS / "rough/model.keras"),
        ("Rough Bergomi: SBC and recovery, full bank (S2)", RESULTS / "rough/s2/validation.json"),
        ("Rough Bergomi: audit, full bank (S5)", RESULTS / "rough/s5/audit.json"),
        ("Rough Bergomi: classical route, full bank (S6)", RESULTS / "rough/s6/classical.json"),
        ("Rough Bergomi: products and reserves, full bank (S7)", RESULTS / "rough/s7/products.json"),
        ("Rough Bergomi pilot: S2, S5, S6, S7, across models", RESULTS / "rough-pilot/s7-across/across-models.json"),
        ("Heston: neural-pricer route (S4)", RESULTS / "heston/s4/pricer-route.json"),
        ("Heston: classical route (S6)", RESULTS / "heston/s6/classical.json"),
        ("Heston: products and reserves (S7)", RESULTS / "heston/s7/products.json"),
    ]
    mo.accordion({"**What this notebook reads** (click to open): every result and its status": mo.vstack([
        mo.md("Every number below comes from an accepted study in `results/calibration`. The rough-Bergomi results "
              "are a pilot (1,600 training smiles) until the full bank of 100,000 has run on the cluster; pilot "
              "results are marked and are not quoted on stage."),
        mo.ui.table([{"result": n, "status": "ready" if teaching.existing(p) else "pending"} for n, p in _checks],
                    selection=None, pagination=False),
    ])})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | | Slot 4: Heston | Slot 5: rough Bergomi |
    |:---|:---|:---|
    | Data | the SPY smile: 63 prices | the same |
    | Unknowns | $v_0$, $\theta$, $\sigma_v$, $\rho$ | $H$, $\eta$, $\rho$, given the market's forward-variance curve $\xi_0$ |
    | One smile priced by | a Fourier integral, 20 ms | Monte Carlo, 5 CPU-seconds |
    | The bank | 60,000 smiles in 46 CPU-minutes | 112,000 smiles in about 370 CPU-hours, on a cluster |
    | The network | flow matching on the prices | the same, with $\xi_0$ as an extra condition |
    | Is it right? | the exact posterior by MCMC | SBC, a second method, and an audit of each smile |
    | The number at the end | a posterior, and what it says about Heston | a price range and a reserve |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "**The model in detail** (click to open): rough Bergomi, its prior and its likelihood": mo.md(r"""
        #### 1 · The model

        $$\frac{dS_t}{S_t} = \sqrt{V_t}\,\big(\rho\,dW_t + \sqrt{1-\rho^2}\,dW^\perp_t\big), \qquad
        V_t = \xi_0(t)\,\exp\!\Big(\eta\,Y_t - \tfrac12\eta^2 t^{2H}\Big), \qquad
        Y_t = \sqrt{2H}\int_0^t (t-s)^{H-\frac12}\,dW_s .$$

        - $\xi_0(t)$ is today's forward-variance curve, read off the option chain by replicating variance
          swaps. The model matches it by construction, $\mathbb E[V_t] = \xi_0(t)$, whatever $H$, $\eta$ and
          $\rho$ are.
        - $H$ is the roughness of the log variance; the at-the-money skew grows like $T^{H-1/2}$ as $T \to 0$.
        - $\eta$ is the volatility of variance and $\rho$ the correlation of the price with its variance.
        - The variance depends on the whole past of $W$, so there is no finite state, no characteristic
          function, and no Fourier or PDE pricer. **Every price is a Monte Carlo run**: the hybrid scheme for
          $Y$, the conditional Black–Scholes estimator for the options, antithetic paths and control variates.

        #### 2 · What the smile identifies

        The short skew sets the product $\rho\eta$. The skew's term structure sets $H$. $\eta$ is separated
        from $\rho$ mainly by the long-dated at-the-money level measured against $\xi_0$, so that separation
        is only as good as $\xi_0$, which the posterior treats as exact.

        #### 3 · The likelihood

        The 63 prices carry the design noise of slot 1. Monte Carlo prices carry their own error, of variance
        $s_c^2$, so the audit uses a Gaussian of variance $\sigma_c^2 - s_c^2$, which is unbiased for the exact
        likelihood if the Monte Carlo errors are independent across quotes; with paths shared within a maturity
        it is approximate (design document, Appendix G.3).
        """)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · The desk's question

    SPY on 21 October 2026: the 63 prices of slots 1 and 4, read from the chain onto the design grid. A
    desk marks products the smile does not quote: a forward-start call (one period of a cliquet) and a
    volatility swap. Least squares gives one parameter vector and one mark. The rules ask for the range
    and a 90% point in it: Regulation (EU) 2016/101, Art. 9 for the calibrated parameters and Art. 11 for
    the choice of model.
    """)
    return


@app.cell(hide_code=True)
def _(RESULTS, image):
    image(RESULTS / "spy" / "surface.png", "The SPY chain of 21 October 2026 read onto the design grid (study S8).")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Why rough Bergomi

    The at-the-money skew of SPY falls with maturity as a power law. Heston, at slot 4's posterior mean,
    cannot follow it: its skew tends to a constant as $T \to 0$. Rough Bergomi's skew goes like
    $T^{H-1/2}$ at short maturities, and the model takes $\xi_0$ from the market, so its variance swaps match
    SPY's by construction.
    """)
    return


@app.cell
def _(CHECKPOINTS, CURRICULUM, RESULTS, np, surface, teaching):
    spy = teaching.load_json(RESULTS / "spy" / "surface.json")
    _k, _maturity, _kind, _delta = teaching.design_arrays()
    spy_skew = np.abs(np.asarray(spy["atm_skew"]))
    power_law = spy["skew_power_law"]
    _heston_task = teaching.load_json(CHECKPOINTS / "heston" / "task.json")
    _heston_mean = np.load(CURRICULUM / "data" / "heston-spy-exact.npy").mean(axis=0)
    _heston_iv = teaching.implied_vol_grid(teaching.heston_smile(_heston_task, _heston_mean))
    heston_skew = np.abs(surface.smile_statistics(_heston_iv, _k, _maturity)[2])
    skew_days = np.asarray(surface.DESIGN_DAYS)
    return heston_skew, power_law, skew_days, spy_skew


@app.cell(hide_code=True)
def _(heston_skew, mo, plots, plt, power_law, skew_days, spy_skew, surface):
    _fig, _ax = plt.subplots(figsize=(7.4, 3.8), constrained_layout=True)
    _t = skew_days / surface.TRADING_DAYS
    _ax.loglog(skew_days, spy_skew, "o", color=plots.SERIES[1], label="SPY")
    _ax.loglog(skew_days, power_law["scale"] * _t ** power_law["exponent"], "--", color=plots.SERIES[1], linewidth=1.0,
               label=f"power law, exponent {power_law['exponent']:.2f}")
    _ax.loglog(skew_days, heston_skew, "s-", color=plots.SERIES[0], markersize=4, linewidth=1.0,
               label="Heston at slot 4's posterior mean")
    _ax.set_xticks(skew_days, ["1w", "2w", "1m", "2m", "3m", "6m", "1y"])
    _ax.minorticks_off()
    _ax.set_xlabel("maturity")
    _ax.set_ylabel("|at-the-money skew|")
    _ax.legend(fontsize=8)
    _ax.set_title("the skew's term structure, log–log", loc="left")
    mo.vstack([_fig, mo.md(f"""
    SPY's skew follows a power law with exponent **{power_law['exponent']:.2f}** from one week to one year (about
    −0.4 in the SPX literature). Under rough Bergomi with a large $\\eta$ the fitted exponent is steeper than
    $H - \\tfrac12$, so the exponent does not give $H$ directly.
    """)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Price a rough-Bergomi smile live.** Choose $H$, $\eta$ and $\rho$; the button prices the 63 options by
    Monte Carlo on a flat 20% forward-variance curve.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    rough_h = mo.ui.slider(0.02, 0.35, step=0.01, value=0.10, label="H", show_value=True)
    rough_eta = mo.ui.slider(0.5, 3.5, step=0.1, value=2.0, label="η", show_value=True)
    rough_rho = mo.ui.slider(-0.99, -0.2, step=0.01, value=-0.70, label="ρ", show_value=True)
    run_rough = mo.ui.run_button(label="Price this smile by Monte Carlo (a few seconds)")
    mo.vstack([mo.hstack([rough_h, rough_eta, rough_rho]), run_rough])
    return rough_eta, rough_h, rough_rho, run_rough


@app.cell
def _(mo, rough_eta, rough_h, rough_rho, run_rough, teaching):
    mo.stop(not run_rough.value, mo.md("*Choose H, η and ρ, then press the button: 63 prices from 8,192 paths per "
                                       "maturity block.*"))
    teaching.live_smile_figure(teaching.price_rough_smile(rough_h.value, rough_eta.value, rough_rho.value))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Why the exact posterior is out of reach

    One likelihood evaluation is one Monte Carlo surface. An exact chain needs about 12,300 evaluations for
    400 effective draws, so it costs at least 17 CPU-hours per smile. That is a lower bound: it assumes the
    cheapest pricing cost (the pilot bank's average gives 41 CPU-hours), and a chain on a noisy likelihood
    needs more evaluations than the exact Heston chains it is extrapolated from. Slot 4's exact check is not
    available here.
    """)
    return


@app.cell(hide_code=True)
def _(CURRICULUM, RESULTS, image, mo, teaching):
    _costs = teaching.load_json(RESULTS / "costs" / "costs.json")
    mo.vstack([
        image(CURRICULUM / "images" / "ladder.png", "One likelihood evaluation and one exact posterior under each model."),
        mo.ui.table(teaching.cost_rows(_costs), selection=None, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · The amortized posterior for SPY

    The training bank holds 100,000 rough-Bergomi smiles. Their parameters and forward-variance curves are
    drawn from the prior, they are priced by Monte Carlo on the cluster, and the design noise is added. The
    network is slot 4's, with $\xi_0$ as an extra condition. For a new smile it returns 1,000 posterior draws
    in 1.4 s.
    """)
    return


@app.cell(hide_code=True)
def _(CHECKPOINTS, mo, pending, teaching):
    _ready = teaching.existing(CHECKPOINTS / "rough" / "model.keras")
    pending("the rough-Bergomi network trained on the full bank (gates G0 to G2), and SPY's posterior from it "
            "(study S8).") if _ready is None else mo.md("The full-bank network is available.")
    return


@app.cell(hide_code=True)
def _(RESULTS, mo, pilot_banner, teaching):
    pilot = teaching.load_npz(RESULTS / "rough-pilot" / "s2" / "draws.npz")
    pilot_surface = mo.ui.slider(0, (len(pilot["truth"]) - 1) if pilot else 0, value=0, label="pilot test smile",
                                 show_value=True)
    mo.vstack([pilot_banner("the pilot network on held-out simulated smiles."), pilot_surface]) if pilot else mo.md("")
    return pilot, pilot_surface


@app.cell(hide_code=True)
def _(pending, pilot, pilot_surface, teaching):
    (teaching.pairs({"network (pilot)": pilot["draws"][pilot_surface.value]}, ["H", "η", "ρ", "ρη"],
                    truth=pilot["truth"][pilot_surface.value], figsize=(7.4, 7.4))
     if pilot else pending("the pilot network's draws."))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Is it right without an exact answer?

    Three checks replace slot 4's exact chain, each answering a different question.

    1. **Calibrated on average?** Simulation-based calibration on held-out simulated smiles.
    2. **Does an independent method agree?** MCMC on a neural pricer: deep calibration's network, sampled
       instead of optimized. On Heston it matched the exact-likelihood reference on all 50 smiles, so its
       agreement on rough smiles is evidence.
    3. **Right on this smile?** The audit is slot 4's importance-sampling correction with a Monte Carlo
       likelihood. It draws 256 parameter values from a widened fit to the network's posterior and reweights
       them with the likelihood of section 3. The efficiency and the Pareto shape $\hat k$ decide, with the
       threshold 0.58 that suits 256 draws: if the smile passes, the corrected posterior is used; if not, it
       goes to the exact route.
    """)
    return


@app.cell(hide_code=True)
def _(RESULTS, image, mo, pilot_banner):
    mo.vstack([
        image(RESULTS / "rough" / "s2" / "sbc.png", "Rough Bergomi, full bank: SBC on held-out smiles."),
        pilot_banner("the pilot network fails SBC for ρ and ρη."),
        image(RESULTS / "rough-pilot" / "s2" / "sbc.png", "Rough Bergomi, pilot network: rank ECDF minus uniform."),
    ])
    return


@app.cell(hide_code=True)
def _(RESULTS, image, mo, pending, teaching):
    _route = teaching.load_json(RESULTS / "heston" / "s4" / "pricer-route.json")
    _exact = teaching.load_json(RESULTS / "heston" / "s4-exact" / "pricer-route.json")
    _parts = [image(RESULTS / "heston" / "s4" / "agreement.png",
                    "Heston: MCMC on a neural pricer against the amortized posterior; two simulation-based routes that "
                    "share only the bank.")]
    if _route is not None:
        _error = _route["summary"]["pricer"]["held_out_rms_error_in_noise_sd"]
        _parts += [mo.md(f"The neural pricer's held-out error is {_error['median_over_contracts']:.3f} noise units (RMS, "
                         f"median contract). **The amortized network against MCMC on the pricer**, on "
                         f"{len(_route['surfaces'])} Heston smiles:"),
                   mo.ui.table(teaching.agreement_rows(_route, key="agreement"), selection=None, pagination=False)]
    if _exact is not None:
        _parts += [mo.md("**MCMC on the pricer against the exact-likelihood reference**: the check that this route can "
                         "stand in where no exact posterior is affordable."),
                   mo.ui.table(teaching.agreement_rows(_exact, key="pricer_vs_exact"), selection=None, pagination=False)]
    _parts.append(pending("the neural-pricer route on rough-Bergomi smiles (study S4)."))
    mo.vstack(_parts)
    return


@app.cell(hide_code=True)
def _(RESULTS, image, mo, pending, pilot_banner, teaching):
    _audit = teaching.load_json(RESULTS / "rough-pilot" / "s5" / "audit.json")
    if _audit is None:
        _out = pending("the audit (study S5).")
    else:
        _s = _audit["summary"]
        _out = mo.vstack([
            pilot_banner("computed with the likelihood that counted the Monte Carlo variance twice; it is rerun with "
                         "the corrected likelihood (gate G3)."),
            mo.ui.table([
                {"audit, rough-Bergomi pilot": "smiles audited", "value": str(_s["surfaces"])},
                {"audit, rough-Bergomi pilot": "efficiency: 5%, median, 95%",
                 "value": ", ".join(f"{100 * _s['efficiency_quantiles'][q]:.0f}%" for q in ("q05", "q50", "q95"))},
                {"audit, rough-Bergomi pilot": "flagged at k̂ > 0.7",
                 "value": f"{100 * _s['flag_rate_khat_above_0_7']:.1f}%"},
                {"audit, rough-Bergomi pilot": "CPU-seconds per smile", "value": f"{_s['cpu_seconds_per_surface_median']:.0f}"},
            ], selection=None, pagination=False),
            image(RESULTS / "rough-pilot" / "s5" / "audit.png", "The audit on 40 pilot smiles."),
        ])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · The classical route on the same smiles

    The desk's tool is weighted least squares from several starting points, with an interval from the
    Gauss–Newton approximation $(J^\top W J)^{-1}$. On Heston that interval covers the truth as often as the
    posterior does: it is the honest baseline, and good where the objective is close to a quadratic. What
    the posterior adds is the shape where the objective is not quadratic (near $\rho = -1$, at small $H$), a
    coverage statement tested on thousands of smiles, and prices for any product from the same draws without
    the product's gradient.
    """)
    return


@app.cell(hide_code=True)
def _(RESULTS, mo, teaching):
    _models = {name: key for name, key in (("rough Bergomi", "rough"), ("rough Bergomi (pilot)", "rough-pilot"),
                                           ("Heston", "heston"))
               if teaching.existing(RESULTS / key / "s6" / "classical.json")}
    classical_model = mo.ui.dropdown(options=_models, value=next(iter(_models)) if _models else None, label="model")
    classical_model if _models else mo.md("")
    return (classical_model,)


@app.cell(hide_code=True)
def _(RESULTS, classical_model, image, mo, pending, pilot_banner, teaching):
    if classical_model.value is None:
        _out = pending("study S6 (`tutorial_final/studies/classical_route.py`).")
    else:
        _report = teaching.load_json(RESULTS / classical_model.value / "s6" / "classical.json")
        _s = _report["summary"]
        _rows = []
        for _n, _label in zip(_report["parameters"], _report["labels"]):
            _gn = (_s["gauss_newton_coverage_90"] or {}).get(_n)
            _rows.append({"parameter": _label,
                          "Gauss–Newton 90% interval covers the truth": "–" if _gn is None else f"{100 * _gn:.0f}%",
                          "posterior 90% interval covers the truth": f"{100 * _s['posterior_coverage_90'][_n]:.0f}%"})
        _out = mo.vstack([
            pilot_banner("8 smiles.") if classical_model.value == "rough-pilot" else mo.md(""),
            mo.md(f"{classical_model.selected_key}, {_s['surfaces']} smiles, {_s['starts']} starts each. Distinct optima "
                  f"per smile: {_s['modes']['mean']:.1f} on average, two or more on "
                  f"{100 * _s['modes']['share_with_two_or_more']:.0f}% of smiles; best fit on a bound for "
                  f"{100 * _s['best_on_bound_share']:.0f}%; {_s['gauss_newton_singular']} singular Gauss–Newton matrices."),
            mo.ui.table(_rows, selection=None, pagination=False),
            image(RESULTS / classical_model.value / "s6" / "landscape.png",
                  "The least-squares objective over two parameters at the best values of the others, the multi-start "
                  "optima, the Gauss–Newton 90% region and the posterior draws, for one smile."),
        ])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · From posterior to prices: the range and the reserve

    Each posterior draw prices each path-dependent product by Monte Carlo, with the same random numbers for
    every draw, so the spread of prices comes from the parameters and not from the simulation. The
    least-squares calibration prices them too. The **reserve** for a sold position is the distance from the
    least-squares price to the chosen quantile of the posterior prices.

    The variance swap has zero range under rough Bergomi by construction: it is the integral of $\xi_0$,
    which fixes it under continuous monitoring. That is a consistency check of the curve input, not a
    finding; with daily monitoring the strike does depend on $(H, \eta, \rho)$ (Lecture 8). Under Heston the
    variance swap carries a range, because Heston's forward variance depends on its parameters.
    """)
    return


@app.cell(hide_code=True)
def _(RESULTS, mo, teaching):
    _models = {name: key for name, key in (("rough Bergomi", "rough"), ("rough Bergomi (pilot)", "rough-pilot"),
                                           ("Heston", "heston"))
               if teaching.existing(RESULTS / key / "s7" / "price-draws.npz")}
    product_model = mo.ui.dropdown(options=_models, value=next(iter(_models)) if _models else None, label="model")
    level = mo.ui.slider(0.80, 0.99, step=0.01, value=0.90, label="confidence level", show_value=True)
    mo.hstack([product_model, level]) if _models else mo.md("")
    return level, product_model


@app.cell(hide_code=True)
def _(RESULTS, mo, product_model, teaching):
    products = (teaching.load_npz(RESULTS / product_model.value / "s7" / "price-draws.npz") if product_model.value else None)
    product_rows = sorted(int(k.split("_")[1]) for k in (products or {}) if k.startswith("prices_"))
    product_surface = mo.ui.dropdown(options=[str(r) for r in product_rows],
                                     value=str(product_rows[0]) if product_rows else None, label="test smile")
    product_surface if product_rows else mo.md("")
    return product_rows, product_surface, products


@app.cell(hide_code=True)
def _(
    level,
    mo,
    pending,
    pilot_banner,
    product_model,
    product_rows,
    product_surface,
    products,
    teaching,
):
    if not product_rows:
        _out = pending("the product prices of study S7.")
    else:
        _row = int(product_surface.value)
        _out = mo.vstack([
            pilot_banner("the pilot network's intervals are 1.3 to 1.6 times too wide, so these ranges overstate the "
                         "exact ones.") if product_model.value == "rough-pilot" else mo.md(""),
            mo.ui.table([{_key: (float(f'{_value:.5g}') if isinstance(_value, float) else _value) for _key, _value in _r.items()}
                         for _r in teaching.reserve_rows(products, _row, level.value)], selection=None, pagination=False),
            teaching.price_histograms(products, _row, level.value,
                                      f"{product_model.selected_key}, test smile {_row}: product prices under the posterior"),
        ])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Heston against rough Bergomi on the same smile

    The posterior's range is the parameter part of model risk; the choice of model is the other part. Both
    models, calibrated to the same smile, price the forward-start differently, and the gap is judged smile
    by smile.
    """)
    return


@app.cell(hide_code=True)
def _(RESULTS, image, mo, pending, pilot_banner, teaching):
    _across = teaching.load_json(RESULTS / "rough-pilot" / "s7-across" / "across-models.json")
    if _across is None:
        _out = pending("the across-model study.")
    else:
        _rows = []
        for _s in _across["surfaces"]:
            _fs = _s["products"]["forward_start_call"]
            _rows.append({"pilot smile": _s["test_row"],
                          "Heston fit, χ² on 63 quotes": round(_s["heston_least_squares"]["chi2"], 1),
                          "Heston audit k̂": round(_s["heston_audit"]["khat"], 2),
                          "forward start, Heston minus rough (points)": f"{100 * _fs['gap_between_points_relative']:+.1f}%",
                          "90% ranges overlap": "yes" if _fs["ranges_overlap"] else "no"})
        _out = mo.vstack([
            pilot_banner("3 simulated rough-Bergomi smiles."),
            mo.ui.table(_rows, selection=None, pagination=False),
            image(RESULTS / "rough-pilot" / "s7-across" / "across-models.png",
                  "The forward-start price under each model's posterior, on the pilot smiles."),
            mo.md("A smile Heston cannot fit is reported as such: its posterior collapses and its audit flags it. "
                  "**On SPY**, with slot 4's Heston posterior and the full rough-Bergomi network:"),
            pending("the two models' forward-start distributions on SPY (study S7 on S8's posterior)."),
        ])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 · What the answer assumes, and what it takes on a desk

    The answer is conditional on four things: **the model**, **the prior**, **the quote-noise model**, and
    **$\xi_0$**, read off the same smile and treated as exact. Each is a lever; slot 6 pulls them.

    On a desk the chain is:

    1. the model, and a prior agreed with the desk;
    2. the noise model, from the chain's spreads;
    3. a simulated bank on a cluster, and the network;
    4. the validation: SBC, an exact sibling model, a second method;
    5. the audit of each smile, with a fallback to the exact route;
    6. the products, priced under the posterior with common random numbers;
    7. the reserve and its diagnostics, reported to model validation.

    Everything shown is in the repository. A model only simulation can price, or a validator asking for
    these ranges? Talk to the authors: ⟨contacts⟩.

    ## Takeaways

    1. **Rough Bergomi has no exact posterior worth computing every morning**: at least 17 CPU-hours per
       smile.
    2. **The amortized posterior takes seconds**, and the audit checks it smile by smile, at minutes.
    3. **The products the smile does not quote carry a range**, and the forward-start's reserve is the
       number the rules ask for.
    4. **The range is conditional**, and the gap across models is judged smile by smile.

    **Next, slot 6:** what changes when the prior or the model is wrong, and how to find out.

    *Evidence and code:* the [curriculum](../curriculum/05-slot5-rough-bergomi-to-a-reserve.md), the
    [design document](../../docs/design/calibration-design-document.md) and `src/sbi4finance/calibration/`.
    Numbers marked pilot are not quoted on stage until the full studies replace them.
    """)
    return


if __name__ == "__main__":
    app.run()
