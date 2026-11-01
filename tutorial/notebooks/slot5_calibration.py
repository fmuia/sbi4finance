import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="Slot 5 · Rough Bergomi to a reserve",
)


@app.cell(hide_code=True)
def _():
    import os
    import time

    os.environ.setdefault("KERAS_BACKEND", "jax")

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from sbi4f_tutorial import long_end, plots, surface, teaching

    ROOT = mo.notebook_dir().parent
    ASSETS = ROOT / "assets"
    RESULTS = ASSETS / "results" / "calibration"
    CHECKPOINTS = ASSETS / "checkpoints"
    CHECKS = ASSETS / "quick-checks"

    plots.style()


    def pending(what):
        return mo.callout(mo.md(f"**Pending:** {what}"), kind="warn")


    def pilot_banner(what):
        return mo.callout(mo.md(f"**Historical pilot result:** {what} This is separate from the current 100k estimators' evidence."), kind="neutral")


    def image(path, caption, width=None):
        found = teaching.existing(path)
        if found is None:
            return pending(f"`{path.relative_to(ROOT)}` is not in the results yet.")
        return mo.image(src=found, caption=caption, width=width)

    return (
        ASSETS,
        CHECKPOINTS,
        CHECKS,
        RESULTS,
        image,
        long_end,
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

    The new known-curve and joint-curve BayesFlow 2.1 development estimators can
    be inspected in [the cached/live backbone review](backbone_review.py).
    Those fixed synthetic replays have separate provenance and do not replace
    the historical SPY or pilot results below.

    **A long-dated mark rests on inputs no quote pins. The posterior prices them; amortization makes it checkable.**

    Slot 4 illustrated the workflow with a current synthetic Heston case and retained historical SPY
    references and model checks. Here the current known-curve and joint-curve examples make the inference
    inputs explicit. The historical SPY application below keeps the financial question and changes the model
    to one built for index smiles, **rough Bergomi**, a rough version of Bergomi's (2005) forward-variance
    model. That change makes the exact posterior expensive, and its validation far more so: that is where
    amortized inference pays.

    Three questions organize the application; each result below names its method and scope:

    1. **Rough Bergomi is priced only by Monte Carlo**, so its exact posterior costs at least 17 CPU-hours
       per smile by random-walk Metropolis. Importance sampling from a Gauss–Newton proposal takes about 85
       CPU-minutes on SPY's 63 quotes (1,024 evaluations at 5.0 CPU-seconds). Where that proposal works, one
       smile and a book are affordable; a validation on thousands of simulated smiles is not. The amortized
       timing of 1.4 seconds belongs to the historical benchmark. Calibration, independent-reference and
       per-observation audit checks remain different questions; a new fit does not inherit those results.
    2. **Which inputs does a smile constrain?** Historical local analysis on SPY examines $H$, $\eta$ and
       $\rho$ with supplied or perturbed curve levels. The current joint task instead infers three curve
       coefficients with the parameters; these are different conditioning assumptions.
    3. **Long-dated products depend on the forward-variance curve beyond the last quote.** Held flat, as in
       the tutorial's baseline, the curve gives a 3-year variance swap no range at all. With a stated prior and
       the long-dated quotes as data, the posterior gives it a 90% range of about a quarter of its price, and
       gives a 5-year put three times the baseline's range in the historical long-end experiment. This is
       Gaussian/Laplace propagation with a separate curve prior, not output of the new six-target estimator.
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Current numerical backbone · cached development example

    This section uses the completed 100,000-surface estimator. It opens without
    loading a neural model, generating prices or running MCMC. The historical
    SPY material below retains its original data, models and scientific scope.
    Fresh sampling is an explicit action in [the backbone review](backbone_review.py).
    """)
    return


@app.cell
def _(mo):
    from sbi4f_tutorial import backbones

    backbone_case = mo.ui.dropdown(options=backbones.examples("rough"),
                                    value="rough-known100k-validation0", label="Current cached example")
    backbone_case
    return backbone_case, backbones


@app.cell
def _(backbone_case, backbones, mo, np, plt):
    _task, _report, _arrays = backbones.load_example(backbone_case.value)
    _contracts = _task["context"]["contracts"]
    _noise = np.asarray(_task["context"]["quote_noise_std"])
    _maturities = np.asarray([c["maturity"] for c in _contracts])
    _strikes = np.asarray([c["strike"] for c in _contracts])
    _quote_fig, _quote_ax = plt.subplots(figsize=(6.8, 3.0), constrained_layout=True)
    for _maturity in np.unique(_maturities):
        _at = _maturities == _maturity
        _quote_ax.errorbar(_strikes[_at], _arrays["observations"][_at], yerr=_noise[_at],
                           fmt=".", capsize=2, label=f"{252 * _maturity:g} days")
    _quote_ax.set(xlabel="strike / forward", ylabel="observed price / forward",
                  title="This case's 63 observed option prices; bars are ±1 quote-noise SD")
    _quote_ax.legend(fontsize=7, ncol=4)
    _quote_rows = [{"maturity (years)": float(c["maturity"]), "strike / forward": float(c["strike"]),
                    "option": c["option_type"], "observed price / forward": float(y),
                    "quote-noise SD": float(noise)}
                   for c, y, noise in zip(_contracts, _arrays["observations"], _noise)]
    _quote_view = mo.vstack([
        _quote_fig,
        mo.accordion({"Exact observations, contract design and known quote noise": mo.vstack([
            mo.md("These are the saved noisy prices used for this posterior, in the task's spot-one, "
                  "zero-carry forward units. The quote-noise standard deviations define the observation "
                  "model; they are not posterior intervals. Prices are shown as stored, without implied-"
                  "volatility inversion or clipping."),
            mo.ui.table(_quote_rows, selection=None, pagination=True, page_size=9),
        ])}),
    ])
    _rates = backbones.replay.rough_variance_products(
        _task, _arrays["neural_draws"], context_values=_arrays.get("context_values"))
    _rate_rows = []
    for _quantity, _label in (("variance_swap_1y", "1-year mean variance rate"),
                              ("forward_variance_0_5_to_1y", "6-month to 1-year forward variance rate")):
        _q = np.quantile(_rates[_quantity], [.05, .5, .95])
        _rate_rows.append({"quantity": _label, "source": "neural candidate",
                           "median (variance/year)": float(_q[1]),
                           "central 90%": f"{_q[0]:.5g} to {_q[2]:.5g}"})
    _rate_scope = (
        "The known curve is an observed input: these rates are constant across the parameter draws "
        "by construction. Their zero posterior width does not mean the quotes identified the curve."
        if "context_values" in _arrays else
        "Each paired draw of the three exponential-curve coefficients gives one rate. This is the "
        "learned candidate's uncertainty propagation; a conditional reference is still unavailable.")
    _rate_view = mo.vstack([
        mo.md("**Carry the posterior to a derived quantity.** These two analytic rates average the "
              "continuous forward-variance curve over [0, 1] and [0.5, 1] years. They are annualized decimal "
              "variance rates, not realized payoffs or new Monte Carlo product prices."),
        mo.ui.table(_rate_rows, selection=None, pagination=False),
        mo.md(_rate_scope + " The known and joint examples have different observations and curve families; "
              "their widths do not isolate the effect of changing the conditioning assumption."),
        mo.md("**Interpret the rates.** Which curve inputs determine these quantities? In a known-curve "
              "case, could tighter intervals for H, eta or rho change these rates? In a joint case, why "
              "might a broad decay posterior coexist with a more concentrated curve average?"),
    ])
    if "context_values" in _arrays:
        _context = mo.vstack([
            mo.md("**Known curve first:** infer H, eta and rho from the quotes and seven supplied curve levels. "
                  "The known curve varies across training examples and stays fixed during each inference."),
            mo.ui.table([{"maturity (years)": float(t), "known forward variance": float(v)}
                         for t, v in zip(_task["model_context"]["forward_variance_knots"][1:],
                                         _arrays["context_values"])], selection=None, pagination=False),
        ])
    else:
        _notes = {"validation:1983": "Selected illustration of curved long-variance/decay trade-offs.",
                  "validation:1941": "Selected illustration of strong eta/rho correlation; correlation alone is not non-Gaussianity.",
                  "validation:1962": "Separate near-flat-curve illustration: decay can remain weakly informed; no interior claim."}
        _context = mo.md("**Joint calibration next:** infer H, eta, rho and three curve coefficients from the quotes. "
                         "The generating curve is excluded from the conditions. "
                         "xi(t) = long + (short - long) * exp(-decay * t). "
                         + _notes.get(_report["case_id"], "Validation:0 has no interior-selection claim."))
    mo.vstack([
        _context,
        mo.callout(mo.md(f"**Development example:** `{_report['case_id']}`; "
                         f"{len(_arrays['neural_draws']):,} paired joint draws. "
                         f"{'PyMC reference supplied' if 'reference_chains' in _arrays else 'conditional reference and Monte Carlo product checks pending'}. "
                         "The selected geometry examples illustrate trade-offs; they are not an accuracy study."), kind="neutral"),
        mo.md(f"Task SHA256: `{_report['task_fingerprint']}`  \nModel SHA256: `{_report['checkpoint_sha256']}`"),
        _quote_view,
        mo.ui.table(backbones.parameter_rows(_task, _arrays), selection=None, pagination=False),
        backbones.corner(_task, _arrays),
        _rate_view,
        mo.md("**Predict, then compare.** Before switching from known to joint curve, name which inputs "
              "become unknown. In the selected case, which parameter trade-off or broad interval can you "
              "explain? These cases have different synthetic observations: switching cases is not a matched "
              "experiment measuring the effect of curve uncertainty alone."),
    ])
    return




@app.cell(hide_code=True)
def _(CHECKPOINTS, RESULTS, backbones, mo, teaching):
    _checks = [
        ("Real SPY smile (S8)", RESULTS / "spy/surface.json"),
        ("Costs on one machine (S9)", RESULTS / "costs/costs.json"),
        ("Historical rough SPY route: checkpoint", CHECKPOINTS / "rough/model.keras"),
        ("Historical rough SPY route: SBC and recovery (S2)", RESULTS / "rough/s2/validation.json"),
        ("Historical rough SPY route: audit (S5)", RESULTS / "rough/s5/audit.json"),
        ("Historical rough SPY route: classical comparison (S6)", RESULTS / "rough/s6/classical.json"),
        ("Historical rough SPY route: products and reserves (S7)", RESULTS / "rough/s7/products.json"),
        ("Rough Bergomi pilot: S2, S5, S6, S7, across models", RESULTS / "rough-pilot/s7-across/across-models.json"),
        ("Heston: neural-pricer route (S4)", RESULTS / "heston/s4/pricer-route.json"),
        ("Heston: classical route (S6)", RESULTS / "heston/s6/classical.json"),
        ("Heston: products and reserves (S7)", RESULTS / "heston/s7/products.json"),
    ]
    mo.accordion({"**What this notebook reads** (click to open): every result and its status": mo.vstack([
        mo.md("The current 100,000-surface known-curve and joint-curve fits are complete and their development "
              "replays are packaged above. Their conditional references and downstream-product checks remain "
              "separate work. Historical studies, pilot results and quick checks below keep their own provenance; "
              "an absent historical export does not mean the new training bank is unfinished."),
        mo.ui.table([{"result": f"Current 2.1 replay: {name}", "status": "packaged development example"}
                     for name in backbones.examples("rough")] +
                    [{"result": n, "status": "historical asset available" if teaching.existing(p)
                      else "not packaged for this historical route"} for n, p in _checks],
                    selection=None, pagination=False),
    ])})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Historical SPY application and comparison experiments

    The following sections preserve the original SPY/pilot studies. In this route, the curve is supplied
    at short maturities. Section 8 uses a separate eleven-level long-end curve model. Neither route is
    the six-target exponential-curve inference task shown in the current development panel above.

    | | Slot 4: Heston | Slot 5: rough Bergomi |
    |:---|:---|:---|
    | Data | the SPY smile: 63 prices | the same |
    | Unknowns | $v_0$, $\theta$, $\sigma_v$, $\rho$ | $H$, $\eta$, $\rho$, given the market's forward-variance curve $\xi_0$ |
    | One smile priced by | a Fourier integral, 20 ms | Monte Carlo, 5 CPU-seconds |
    | The bank | 60,000 smiles in 46 CPU-minutes | 112,000 smiles in about 370 CPU-hours, on a cluster |
    | The network | flow matching on the prices | the same, with $\xi_0$ as an extra condition |
    | Is it right? | the exact posterior by MCMC | SBC, a second method, and an audit of each smile |
    | The number at the end | a posterior, and what it says about Heston | a price range and a reserve, out to the long end |
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
        from $\rho$ by the curvature of the wings and by the long-dated at-the-money level measured against
        $\xi_0$. On SPY the wings suffice: with each level of $\xi_0$ 20% uncertain, $\eta$'s local 90%
        interval stays within a tenth of its prior range (section 6). The network takes $\xi_0$ as given up
        to one year; beyond the last quote it is an assumption (section 8).

        #### 3 · The likelihood

        The 63 prices carry the design noise of slot 1. Monte Carlo prices carry their own error, of variance
        $s_c^2$, so the audit uses a Gaussian of variance $\sigma_c^2 - s_c^2$, which is unbiased for the exact
        likelihood if the Monte Carlo errors are independent across quotes; with paths shared within a maturity
        it is approximate (course, Lecture 8, Section 8.9).
        """)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · The question

    SPY on 21 October 2026: the 63 prices of slots 1 and 4, read from the chain onto the design grid. A
    bank marks products the smile does not quote: a forward-start call (one period of a cliquet), a
    volatility swap, and long-dated exotics such as autocallables and cliquets that run for years past the
    last liquid quote. Least squares gives one parameter vector and one mark. The rules ask for the range
    and a 90% point in it: Regulation (EU) 2016/101, Art. 9 for the calibrated parameters and Art. 11 for
    the choice of model, and IFRS 13 for the sensitivity of marks that rest on unobservable inputs.
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
def _(ASSETS, CHECKPOINTS, RESULTS, np, surface, teaching):
    spy = teaching.load_json(RESULTS / "spy" / "surface.json")
    _k, _maturity, _kind, _delta = teaching.design_arrays()
    spy_skew = np.abs(np.asarray(spy["atm_skew"]))
    power_law = spy["skew_power_law"]
    _heston_task = teaching.load_json(CHECKPOINTS / "heston" / "task.json")
    _heston_mean = np.load(ASSETS / "data" / "heston-spy-exact.npy").mean(axis=0)
    _heston_iv = teaching.implied_vol_grid(teaching.heston_smile(_heston_task, _heston_mean))
    heston_skew = np.abs(surface.smile_statistics(_heston_iv, _k, _maturity)[2])
    skew_days = np.asarray(surface.DESIGN_DAYS)
    return heston_skew, power_law, skew_days, spy, spy_skew


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
    Monte Carlo on a flat 20% forward-variance curve. Predict which feature of the smile changes before
    moving one control, then inspect the generated prices. This button runs the forward simulator;
    it does not update the current cached inverse posterior above.
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
    ## 3 · What the exact posterior costs

    One likelihood evaluation is one Monte Carlo surface. An exact chain by random-walk Metropolis needs about
    12,300 evaluations for 400 effective draws, so it costs at least 17 CPU-hours per smile. That is a lower
    bound: it assumes the cheapest pricing cost (the pilot bank's average gives 41 CPU-hours), and a chain on
    a noisy likelihood needs more evaluations than the exact Heston chains it is extrapolated from.
    Importance sampling from a Gauss–Newton proposal takes about 85 CPU-minutes on SPY's 63 quotes (1,024
    evaluations at 5.0 CPU-seconds). Where that proposal works, one smile and a book are affordable; a
    validation on thousands of simulated smiles is not. Slot 4's exact check is not available here.
    """)
    return


@app.cell(hide_code=True)
def _(ASSETS, RESULTS, image, mo, teaching):
    _costs = teaching.load_json(RESULTS / "costs" / "costs.json")
    mo.vstack([
        image(ASSETS / "images" / "ladder.png", "One likelihood evaluation and one exact posterior under each model "
                                                    "(the rough-Bergomi posterior, by random-walk Metropolis, is a lower bound)."),
        mo.ui.table(teaching.cost_rows(_costs), selection=None, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Historical SPY neural route · export status

    The current 100,000-surface estimators already have packaged synthetic replays above. A matching
    native SPY posterior and the historical S2–S7 exports are separate artifacts. In the known-curve
    training task, parameters and curves vary across simulations and the realized curve is supplied as
    an inference condition. In the joint task its three coefficients are targets instead.
    """)
    return


@app.cell(hide_code=True)
def _(CHECKPOINTS, mo, pending, teaching):
    _ready = teaching.existing(CHECKPOINTS / "rough" / "model.keras")
    pending("a matching native SPY posterior and historical-route export (study S8); the completed current "
            "100k fits are available above.") if _ready is None else mo.md(
                "A historical rough checkpoint file is present. File presence alone does not establish "
                "its compatibility or validate a current SPY posterior.")
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
    ## 5 · Is it right without an exact answer? · historical and pilot studies

    The original study design used three checks, each answering a different question.
    The saved historical results and explicit missing exports below do not certify the new 100k fits.

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

    The classical route is weighted least squares from several starting points, with an interval from the
    Gauss–Newton approximation $(J^\top W J)^{-1}$. On Heston that interval covers the truth as often as the
    posterior does (slot 1, section 3). For rough Bergomi on SPY, the local analysis below is the same
    approximation at SPY's least-squares fit, with Monte Carlo derivatives: the 63 quotes pin every
    parameter, and they still do when each level of $\xi_0$ is 20% uncertain. The quotes stop pinning the
    parameters only when they are few: on 15 quotes, $\rho$'s interval runs past the bound. On the full
    surface, then, the posterior's value is speed and a coverage check, not a different interval; it
    differs where the quotes are thin, and for the inputs they do not reach (section 8).
    """)
    return


@app.cell(hide_code=True)
def _(CHECKS, mo, np, plots, plt, teaching):
    _r = teaching.load_json(CHECKS / "rough_checks.json")
    _width = np.diff(np.asarray(_r["prior_bounds"]), axis=1)[:, 0]
    _sets = list(dict.fromkeys(s["quotes"] for s in _r["scenarios"]))
    _fig, _axes = plt.subplots(1, 3, figsize=(11.6, 3.2), constrained_layout=True, sharey=True)
    for _k, _q in enumerate(_sets):
        _rows = [s for s in _r["scenarios"] if s["quotes"] == _q]
        _x = [100 * s["xi0_log_sd"] for s in _rows]
        for _i, _name in enumerate(["H", "η", "ρ"]):
            _axes[_i].plot(_x, [2 * 1.645 * s["seed1"]["sd"][_i] / _width[_i] for s in _rows], marker="o",
                           color=plots.SERIES[_k], label=_q)
    for _i, _name in enumerate(["H", "η", "ρ"]):
        _axes[_i].axhline(1.0, color=plots.INK, linestyle="--", linewidth=0.8)
        _axes[_i].set_title(f"{_name}: local 90% interval / prior range", loc="left", fontsize=9)
        _axes[_i].set_xlabel("uncertainty of each ξ0 level (% sd)")
    _axes[0].set_ylim(0, 1.2)
    _axes[0].legend(fontsize=7, loc="upper left")
    _b = _r["best"]["theta"]
    mo.vstack([
        _fig,
        mo.md(f"""
        Rough Bergomi at SPY's least-squares fit ($H$ = {_b[0]:.3f}, $\\eta$ = {_b[1]:.2f}, $\\rho$ = {_b[2]:.2f};
        four starts reach the same optimum, χ² {_r['best']['chi2']:.0f} on 63 quotes). The dashed line is an
        interval as wide as the prior box. Local only: it shows where a direction is weak, not the posterior's
        shape. *Source:* `lab/checks/rough_checks.py`, a quick check.
        """),
    ])
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
        _out = pending("study S6 (`lab/studies/classical_route.py`).")
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
    ## 7 · From posterior to prices: the range and the reserve · stored studies

    Each posterior draw prices each path-dependent product by Monte Carlo, with the same random numbers for
    every draw, so the spread of prices comes from the parameters and not from the simulation. The
    least-squares calibration prices them too. The **reserve** for a sold position is the distance from the
    least-squares price to the chosen quantile of the posterior prices.

    The variance swap has zero range under rough Bergomi by construction: it is the integral of $\xi_0$,
    which fixes it under continuous monitoring. That is a consistency check of the curve input, not a
    finding; with daily monitoring the strike does depend on $(H, \eta, \rho)$ (Lecture 8). Under Heston the
    variance swap carries a range, because Heston's forward variance depends on its parameters.

    The model and test-smile controls below select stored historical product panels. They are independent
    of the current development-case selector. Before increasing the confidence level, predict which
    reported quantiles should move; afterward, explain why this changes the reported range without
    adding information to the posterior.
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
    ## 8 · The long end: where the reserve comes from · separate Gaussian/Laplace experiment

    This stored experiment has fourteen uncertain inputs: $H$, $\eta$, $\rho$ and eleven curve levels.
    Its sliders recompute Gaussian curve updates and delta-method product ranges from stored Monte Carlo
    gradients. It does not run the new six-target neural estimator, whose curve is exponential. Predict
    which product depends most on the unobserved long end, widen that prior, and explain the change.

    Long-dated exotics run past the last quote. An autocallable observes the index every year for three
    years; a cliquet sums capped annual returns; a five-year put depends on the whole variance path. Under
    rough Bergomi their prices depend on $(H, \eta, \rho)$ and on the forward-variance curve $\xi_0$ over
    their life. SPY's quotes pin $\xi_0$ to one year. Four LEAP expirations, from 1.2 to 2.2 years, give its
    replicated variance with a wider error. Beyond 2.2 years nothing is quoted.

    Extrapolation beyond the last quote has no single convention. The tutorial's baseline holds $\xi_0$ flat
    beyond the last liquid maturity and reports the range from the calibrated parameters alone. The posterior
    treats the curve beyond one year as unknown: a random walk in
    $\log \xi_0$ from the one-year level, flat on average (the baseline's assumption) with a stated width per
    square root of a year, updated by the LEAPs, and propagated with $(H, \eta, \rho)$ through the Monte
    Carlo pricer. The products: a 3-year autocallable (annual calls at 100%, an 8% coupon a year, a 60%
    barrier at maturity), a 3-year cliquet (annual returns capped at 8% and floored at 0), a forward start
    from 2 to 3 years, a 5-year 80% put and a 3-year variance swap.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    long_prior = mo.ui.slider(0.05, 0.40, step=0.05, value=0.20, label="prior width beyond the data (sd of log ξ0 per √year)",
                              show_value=True)
    long_noise = mo.ui.slider(0.25, 1.50, step=0.25, value=0.50, label="LEAP variance-swap noise (vol points)", show_value=True)
    mo.hstack([long_prior, long_noise], justify="start")
    return long_noise, long_prior


@app.cell
def _(CHECKS, long_end, long_noise, long_prior, np, teaching):
    long_report = teaching.load_json(CHECKS / "long_end" / "long_end.json")
    long_extra = teaching.load_json(CHECKS / "long_end" / "long_end_extra.json")
    long_desk, long_post, long_shares, _ = long_end.ranges(long_report, long_prior.value, long_noise.value / 100)
    long_mean, long_cov = long_end.posterior(np.log(np.asarray(long_report["xi_flat"])), long_prior.value,
                                             long_noise.value / 100)
    return (
        long_cov,
        long_desk,
        long_extra,
        long_mean,
        long_post,
        long_report,
        long_shares,
    )


@app.cell(hide_code=True)
def _(
    long_cov,
    long_desk,
    long_end,
    long_extra,
    long_mean,
    long_noise,
    long_post,
    long_report,
    long_shares,
    mo,
    np,
    plots,
    plt,
    spy,
):
    _tt = np.linspace(0.02, 5.0, 200)
    _knots = long_end.KNOTS
    _overlap = np.array([[long_end.overlap(t, _knots[j], _knots[j + 1]) for j in range(len(_knots) - 1)] for t in _tt])
    _draws = np.random.default_rng(1).multivariate_normal(long_mean[3:], long_cov[3:, 3:], 400)
    _curves = 100 * np.sqrt((np.exp(_draws) @ _overlap.T) / _tt)
    _lo, _hi = np.quantile(_curves, [0.05, 0.95], axis=0)
    _flat = 100 * np.sqrt((np.asarray(long_report["xi_flat"]) @ _overlap.T) / _tt)
    _fig, (_a, _b, _c) = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True,
                                      gridspec_kw={"width_ratios": [1.35, 1.25, 1.0]})
    _a.fill_between(_tt, _lo, _hi, color=plots.SERIES[0], alpha=0.2, label="rough Bergomi posterior, 90%")
    _a.plot(_tt, _flat, color=plots.INK, linestyle="--", linewidth=1.2, label="baseline: flat beyond one year")
    _T = np.asarray(long_extra["T"])
    _hq = np.asarray(long_extra["heston_kappa_free_vs_vol_q"])
    _a.fill_between(_T, _hq[0], _hq[2], color=plots.SERIES[1], alpha=0.35, label="Heston posterior (fitted to ≤ 1 year), 90%")
    _days = np.asarray(spy["design"]["days"]) / 252
    _wd = np.asarray(spy["diagnostics"]["design_total_variance"])
    _a.plot(_days, 100 * np.sqrt(_wd / _days), "o", color=plots.INK, markersize=3.5, label="SPY replicated, to one year")
    _lt = np.array([t for t, _ in long_end.LEAPS])
    _lv = np.array([100 * np.sqrt(w / t) for t, w in long_end.LEAPS])
    _a.errorbar(_lt, _lv, yerr=1.645 * long_noise.value, fmt="s", color=plots.SERIES[2], markersize=4, capsize=3,
                label="SPY replicated LEAPs, 90% noise")
    _a.axvspan(_lt.max(), 5.0, color=plots.BASELINE, alpha=0.12)
    _a.set_ylim(12, 28)
    _a.set_xlabel("maturity (years)")
    _a.set_ylabel("variance-swap volatility (%)")
    _a.set_title("SPY's variance term structure; shaded: no quotes", loc="left", fontsize=9)
    _a.legend(fontsize=6.3, loc="lower right")
    _names = ["autocallable 3y", "cliquet 3y", "forward start 2y→3y", "put 5y 80%", "variance swap 3y"]
    _x = np.arange(len(_names))
    _scale = np.array([10.0, 1, 1, 1, 1])
    _b.bar(_x - 0.2, 100 * long_desk * _scale, 0.38, color=plots.BASELINE, label="baseline: H, η, ρ only, ξ0 flat")
    _b.bar(_x + 0.2, 100 * long_post * _scale, 0.38, color=plots.SERIES[0], label="posterior: all inputs")
    _b.set_xticks(_x, [n.replace(" ", "\n", 1) for n in _names], fontsize=7)
    _b.set_ylabel("90% range, % of price (autocallable: bp of notional ÷ 10)")
    _b.set_title("the 90% range of each product", loc="left", fontsize=9)
    _b.legend(fontsize=7, loc="upper left")
    _bottom = np.zeros(len(_names))
    for (_g, _sh), _col in zip(long_shares.items(), [plots.SERIES[0], plots.SERIES[2], plots.INK]):
        _c.bar(_x, 100 * _sh, 0.6, bottom=_bottom, color=_col, alpha=0.85, label=_g)
        _bottom += 100 * _sh
    _c.set_xticks(_x, [n.split(" ")[0] for n in _names], fontsize=7, rotation=20)
    _c.set_ylim(0, 135)
    _c.set_ylabel("share of the posterior variance (%)")
    _c.set_title("where the range comes from", loc="left", fontsize=9)
    _c.legend(fontsize=6.3, loc="upper left")
    _rows = []
    for _j, _n in enumerate(_names):
        _fmt = (lambda v: f"{1e4 * v:.0f} bp of notional") if _j == 0 else (lambda v: f"{100 * v:.0f}%")
        _rows.append({"product": _n, "baseline 90% range": _fmt(long_desk[_j]), "posterior 90% range": _fmt(long_post[_j]),
                      **{f"share: {g}": f"{100 * s[_j]:.0f}%" for g, s in long_shares.items()}})
    _gap = [_lv[i] - _hq[2][list(np.round(_T, 2)).index(round(t, 2))] for i, t in enumerate(_lt)]
    _mc = np.asarray(long_report["range90_posterior_mc"]) / np.asarray(long_report["posterior_mc_quantiles"])[1]
    _mc_text = ", ".join([f"{1e4 * _mc[0]:.0f} bp"] + [f"{100 * v:.0f}%" for v in _mc[1:]])
    mo.vstack([
        _fig,
        mo.ui.table(_rows, selection=None, pagination=False),
        mo.md(f"*Delta-method ranges, recomputed live from the stored Monte Carlo gradients. At the default settings "
              f"(0.2 and 0.5), full Monte Carlo propagation through 400 posterior draws gives {_mc_text}.*"),
        mo.md(f"""
        **What this says.** For the products that live past the quotes (the forward start, the 5-year put, the
        variance swap), the baseline's range misses most of the uncertainty, and most of the posterior's comes
        from the curve beyond 2.2 years: {100 * long_shares['ξ0 2.25 to 5 years (prior only)'][2]:.0f}%,
        {100 * long_shares['ξ0 2.25 to 5 years (prior only)'][3]:.0f}% and
        {100 * long_shares['ξ0 2.25 to 5 years (prior only)'][4]:.0f}% of the variance, from an input no quote
        observes. For the autocallable and the cliquet, most comes from $(H, \\eta, \\rho)$ and the short end. The
        prior's width matters, and the slider shows how much; the baseline's implicit prior has width zero, which
        no rule accepts for an unobservable input. The LEAPs also move the prices: under the posterior's inputs
        the forward start, the put and the variance swap are 2–3% dearer than under the flat curve.

        **Heston is confidently wrong at the long end.** Fitted to the first year, its posterior puts the four
        LEAP variance-swap volatilities {min(_gap):.1f} to {max(_gap):.1f} points above the top of its 90% band: a
        parametric curve makes the long end look known.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Where amortized inference comes in.** Every number in this section needs the Monte Carlo pricer: the
    surface's likelihood, and the products. The posterior over the 14 inputs ($H$, $\eta$, $\rho$ and eleven
    levels of $\xi_0$) was approximated here by a Gaussian (Laplace) calculation at the least-squares fit,
    with product ranges from stored gradients. At the recorded settings, that propagation matched full
    Monte Carlo propagation through 400 posterior draws within 1–5%. A neural estimator for this
    fourteen-input task would need its own training and validation; the current three- and six-target
    fits do not provide it. Where the posterior stops being Gaussian, with sparse long-dated smiles instead of replicated
    variance swaps, or with a model whose vol-of-vol term structure the vanilla smile cannot pin, a
    Gauss–Newton proposal can fail, and the network's draws are the proposal that stays affordable (not
    tested here).

    *Source:* `lab/checks/long_end/` (a quick check of 23 October 2026, not yet an accepted study).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 · Heston against rough Bergomi on the same smile

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
    ## 10 · What the answer assumes, and what it takes

    The answer is conditional on four things: **the model**, **the prior**, **the quote-noise model**, and
    **$\xi_0$**, read off the same smile up to one year and treated as exact there, with its long end priced
    as in section 8. Each is a lever; slot 6 pulls them.

    The chain is:

    1. the model, and a stated prior;
    2. the noise model, from the chain's spreads;
    3. a simulated bank on a cluster, and the network;
    4. the validation: SBC, an exact sibling model, a second method;
    5. the audit of each smile, with a fallback to the exact route;
    6. the products, priced under the posterior with common random numbers;
    7. the reserve, reported with its diagnostics.

    Everything shown is in the repository. A model only simulation can price, or a range you must report?
    Talk to the authors: ⟨contacts⟩.

    ## Takeaways

    1. **Rough Bergomi's posterior is affordable for one smile, not for a validation**: at least 17 CPU-hours
       per smile by random-walk Metropolis, and about 85 CPU-minutes by importance sampling from a
       Gauss–Newton proposal on SPY's 63 quotes (1,024 evaluations at 5.0 CPU-seconds). Where that proposal
       works a book is affordable too; a validation on thousands of simulated smiles is not. The amortized posterior
       takes seconds, and the audit checks it smile by smile, at about 53 CPU-minutes on SPY's 63 quotes.
    2. **On SPY's full surface the classical interval is a good approximation**; the posterior differs where
       the quotes are thin, and for the inputs they do not reach.
    3. **Long-dated products rest on the curve beyond the quotes.** Held flat, as in the baseline, it gives a
       range of 0–10%; priced with a stated prior and the LEAPs, the posterior gives about a quarter to a
       third of the price for the forward start, the 5-year put and the variance swap.
    4. **The range is conditional on the model.** Heston's long end is confidently wrong, and the gap across
       models is judged smile by smile.

    **Next, slot 6:** what changes when the prior or the model is wrong, and how to find out.

    *Evidence and code:* the [curriculum](../curriculum/05-slot5-rough-bergomi-to-a-reserve.md), the
    [course](../../course/README.md) and `src/sbi4f_tutorial/`.
    Numbers marked pilot are not quoted on stage until the full studies replace them.
    """)
    return


if __name__ == "__main__":
    app.run()
