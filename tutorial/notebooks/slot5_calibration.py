import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Bayesian calibration of rough Bergomi")


@app.cell(hide_code=True)
def _():
    import os
    import time

    os.environ.setdefault("KERAS_BACKEND", "jax")
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from sbi4f_tutorial.rough_options import lesson as teaching, long_end
    from sbi4f_tutorial.shared import backbones, full_surface, plots, surface

    ROOT = mo.notebook_dir().parent
    ASSETS = ROOT / "assets"
    RESULTS = ASSETS / "results" / "calibration"
    CHECKPOINTS = ASSETS / "checkpoints"
    CHECKS = ASSETS / "quick-checks"
    plots.style()

    def pending(what):
        return mo.callout(mo.md(f"**Not included in this earlier study:** {what}"), kind="neutral")

    def pilot_banner(what):
        return mo.callout(mo.md(f"**Earlier pilot:** {what} This does not validate the current estimators."), kind="neutral")

    def image(path, caption, width=None):
        return (mo.image(src=path, caption=caption, width=width) if teaching.existing(path)
                else pending(f"`{path.relative_to(ROOT)}`"))

    def focused_posterior(task, arrays, pair=("eta", "rho")):
        names = task["parameter_names"]
        draws = arrays["neural_draws"]
        fig, axes = plt.subplots(1, 3, figsize=(10, 2.7), constrained_layout=True)
        for ax, name in zip(axes[:2], pair):
            j = names.index(name)
            ax.hist(draws[:, j], bins=35, density=True, histtype="step", color=plots.SERIES[0])
            ax.axvline(arrays["truth"][j], color=plots.INK, ls=":", label="generating truth")
            ax.set(xlabel=backbones.PARAMETER_LABELS[name][0], ylabel="posterior density")
            ax.set_yticks([])
        a, b = [names.index(name) for name in pair]
        ids = np.linspace(0, len(draws) - 1, min(800, len(draws)), dtype=int)
        axes[2].scatter(draws[ids, a], draws[ids, b], s=3, alpha=.15, color=plots.SERIES[0])
        axes[2].plot(arrays["truth"][a], arrays["truth"][b], "k+", ms=8)
        axes[2].set(xlabel=backbones.PARAMETER_LABELS[pair[0]][1],
                    ylabel=backbones.PARAMETER_LABELS[pair[1]][1], title="Keep paired draws together")
        axes[0].legend(fontsize=7)
        return fig

    return (ASSETS, CHECKPOINTS, CHECKS, RESULTS, backbones, focused_posterior,
            full_surface, image, long_end, mo, np, pending, pilot_banner,
            plots, plt, surface, teaching, time)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bayesian calibration of rough Bergomi

    We keep the option-calibration problem from Heston and change the volatility model.
    Our data are again **63 noisy option prices**: nine strikes at seven maturities.
    We first infer roughness $H$, variance amplitude $\eta$, and correlation $\rho$
    **given a forward-variance curve**. We then let the curve be unknown too.

    The main change is the forward calculation. Rough Bergomi prices options by
    Monte Carlo, so we reuse simulated data when training the network.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The rough Bergomi model

    Rough Bergomi makes variance depend on past shocks through a rough kernel.
    Under the pricing measure, with spot one and zero rates and dividends,

    $$\frac{dS_t}{S_t}=\sqrt{V_t}\left(\rho\,dW_t+\sqrt{1-\rho^2}\,dW_t^\perp\right),
    \qquad V_t=\xi_0(t)\exp\left(\eta Y_t-\tfrac12\eta^2t^{2H}\right),$$
    $$Y_t=\sqrt{2H}\int_0^t(t-s)^{H-1/2}\,dW_s.$$

    | Input | Meaning | In the first example |
    |:---|:---|:---|
    | $H$ | roughness of the variance process | infer it |
    | $\eta$ | amplitude of log-variance fluctuations | infer it |
    | $\rho$ | correlation of price and variance shocks | infer it |
    | $\xi_0(t)=\mathbb E[V_t]$ | initial forward-variance curve | supplied |
    | $V_t$ | instantaneous variance | hidden |

    Roughness describes how variance changes, not its level. A variance of 0.04
    means 20% volatility. These are pricing-measure parameters; fitting observed
    returns is a separate task in the trajectory notebook.
    """)
    return


@app.cell(hide_code=True)
def _(ASSETS, mo):
    mo.vstack([
        mo.md(r"""
        ## 2. Generating synthetic option data

        One training example is **one parameter vector, one curve, and 63 quotes**,
        with $\boldsymbol\vartheta=(H,\eta,\rho)$. Many simulated asset paths estimate each expected option payoff:

        $$P_{K,T}=\mathbb E^{\mathbb Q}[(S_T-K)_+],\qquad
        y_c=P_c(\boldsymbol\vartheta,\xi_0)+\sigma_c\epsilon_c,
        \quad \epsilon_c\sim\mathcal N(0,1).$$

        The Monte Carlo estimate $\widehat P_c$ replaces $P_c$ in the numerical
        simulator. We control its standard error before adding quote noise.
        The latent paths are never supplied to the option-calibration network.
        """),
        mo.image(ASSETS / "illustrations" / "application-workflows-20261101" / "rough-data-and-inference.png"),
        mo.md(r"""
        **Two sources of error.** Monte Carlo error belongs to the numerical
        price estimate. Quote noise represents an assumed tolerance between
        model prices and observed quotes. We keep them separate.

        As in Heston, quote SDs correspond to 0.3 volatility percentage points
        near the money and 1.0 point in the wings, converted to price SDs with
        reference vega. They are fixed model inputs, not measured bid–ask spreads.
        """),
        mo.accordion({"Numerical accuracy and the observation model": mo.md(r"""
        The pricer uses a hybrid time grid, conditional option expectations,
        antithetic paths and control variates. Its declared path ladder controls
        standard errors relative to quote SDs, with a separate final-rung cap.
        This does not establish the size of time-discretization bias.

        Shared paths can correlate numerical price errors. Quote errors remain
        independent Gaussians in this task, and quotes are not clipped at price
        bounds. Adding Monte Carlo variance to quote variance changes the target;
        it is not a free correction for an imprecise pricer.
        """)}),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Explore the forward model

    Before moving a control, predict which part of the smile will change.
    This illustration uses a flat curve at 20% volatility and a fixed simulation
    budget. It prices a new smile; it does not update the posterior below.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    rough_h = mo.ui.slider(0.02, 0.35, step=.01, value=.10, label="H", show_value=True)
    rough_eta = mo.ui.slider(.5, 3.5, step=.1, value=2., label="η", show_value=True)
    rough_rho = mo.ui.slider(-.99, -.2, step=.01, value=-.7, label="ρ", show_value=True)
    run_rough = mo.ui.run_button(label="Simulate this option smile")
    mo.vstack([mo.hstack([rough_h, rough_eta, rough_rho]), run_rough])
    return rough_eta, rough_h, rough_rho, run_rough


@app.cell
def _(mo, rough_eta, rough_h, rough_rho, run_rough, teaching):
    mo.stop(not run_rough.value, mo.md("*Choose parameters and press the button to run a fixed 8,192-path illustration.*"))
    _result = teaching.price_rough_smile(rough_h.value, rough_eta.value, rough_rho.value)
    mo.vstack([teaching.live_smile_figure(_result),
               mo.md(f"Largest estimated Monte Carlo SE / quote SD: **{_result['max_se_over_sigma']:.2f}**. "
                     "This fixed-budget illustration does not apply the training bank's adaptive accuracy policy.")])
    return


@app.cell(hide_code=True)
def _(backbones):
    known_task, known_report, known_arrays = backbones.load_example("rough-known100k-validation0")
    return known_arrays, known_report, known_task


@app.cell(hide_code=True)
def _(ASSETS, mo):
    mo.vstack([
        mo.md(r"""
        ## 3. Two neural approaches

        A **neural pricer** learns the forward map: parameters and a supplied
        curve go in, estimated expected option prices come out. It trains on
        Monte Carlo mean-price labels, rather than individual asset paths or
        noisy quotes.

        An **amortized posterior estimator** learns the inverse map: quotes and
        the supplied curve go in, joint parameter draws come out. Its training
        pairs include fresh quote noise.
        """),
        mo.image(ASSETS / "illustrations" / "heston-workflow-20261101" / "two-amortization-routes.png"),
        mo.md(r"""
        | | Neural pricer + MCMC | Amortized posterior |
        |:---|:---|:---|
        | Learn | parameters → expected prices | noisy quotes → parameter distribution |
        | For each new market | run a new sampler using the cheap pricer | condition the network and draw |
        | Main approximation | the learned prices and resulting likelihood | the learned posterior distribution |

        **The learned pricer makes a likelihood cheap; it does not supply the
        posterior.** With a fixed curve, its Gaussian likelihood is

        $$\widetilde L(\boldsymbol\vartheta;\mathbf y,\xi_0)
          =\prod_c\mathcal N\!\left(y_c;
             f_\phi(\boldsymbol\vartheta,\xi_0)_c,\sigma_c^2\right).$$

        MCMC then targets $p(\boldsymbol\vartheta)\widetilde L$. Comparing its
        draws with the amortized network can expose errors missed by aggregate
        calibration checks. Both routes still share simulation labels and model
        assumptions. Agreement does not establish accuracy of the Monte Carlo
        labels, the time grid or both neural approximations. We inspect three
        such comparisons below.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Building and training the posterior estimator

    The BayesFlow workflow is the same as in Heston: prepare the data, define an
    adapter and conditional network, train, then sample. The extra condition is
    the supplied curve. We divide prices by their quote SDs and take logs of the
    positive curve levels before standardizing the network inputs.

    The code below runs only when requested. It generates a **small fresh training
    set**, then fits a separate demonstration network. The selected cached
    observation is used only for inference. Eight markets are only enough to exercise the API; they do not give a
    useful calibrated estimator or reproduce the pretrained result.
    """)
    return


@app.cell(hide_code=True)
def _(full_surface, mo, teaching, time):
    def train_small_estimator(task, observed_prices, supplied_curve):
        started = time.perf_counter()
        import bayesflow as bf

        split, generation = teaching.generate_training_pairs(task, count=8, seed=2026, max_seconds=90)
        data = full_surface.training_data(split, task)

        adapter = bf.Adapter()
        for name, (lower, upper) in zip(task["parameter_names"], task["prior_bounds"], strict=True):
            adapter = adapter.constrain(name, lower=lower, upper=upper, inclusive="none")
        adapter = (adapter.convert_dtype("float64", "float32")
                   .concatenate(task["parameter_names"], into="inference_variables")
                   .concatenate(["observations", "curve_context"], into="inference_conditions"))
        network = bf.networks.FlowMatching(subnet_kwargs={"widths": [128, 128]})
        workflow = bf.BasicWorkflow(adapter=adapter, inference_network=network,
                                   standardize=["inference_variables", "inference_conditions"])
        fit_started = time.perf_counter()
        workflow.fit_offline(data=data, epochs=5, batch_size=4)
        fit_seconds = time.perf_counter() - fit_started

        conditions = full_surface.conditioning_data(observed_prices[None, :], task,
                                                     context_values=supplied_curve[None, :])
        posterior = workflow.sample(conditions=conditions, num_samples=128, seed=17)
        return posterior, {"generation": generation["runtime_seconds"], "fitting": fit_seconds,
                           "total": time.perf_counter() - started}

    mo.show_code()
    return (train_small_estimator,)


@app.cell(hide_code=True)
def _(mo):
    train_rough = mo.ui.run_button(label="Generate fresh examples and train a small network")
    mo.vstack([train_rough, mo.md("This action generates eight markets and makes ten optimizer updates. It may take a minute; results stay in memory.")])
    return (train_rough,)


@app.cell(hide_code=True)
def _(known_arrays, known_task, mo, np, train_rough, train_small_estimator):
    mo.stop(not train_rough.value, mo.md("*Press the button to run the displayed training workflow.*"))
    try:
        _draws, _timing = train_small_estimator(
            known_task, known_arrays["observations"], known_arrays["context_values"])
    except (RuntimeError, ValueError) as _error:
        mo.stop(True, mo.callout(mo.md(f"**Training run stopped:** {_error}. No partial or selectively accepted training sample is used."), kind="warn"))
    mo.vstack([
        mo.md(f"Generated 8 fresh markets; trained for 5 epochs (**10 optimizer updates**). "
              f"Generation: **{_timing['generation']:.1f} s**; fitting including compilation: **{_timing['fitting']:.1f} s**; "
              f"total including imports and sampling: **{_timing['total']:.1f} s**. These session draws are separate from the pretrained results below."),
        mo.ui.table([{ "parameter": name, "session median": float(np.median(_draws[name]))}
                     for name in known_task["parameter_names"]], selection=None, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"What is learned? And why train offline?": mo.md(r"""
    The network learns a joint conditional distribution over $H,\eta,\rho$.
    Its training targets are the generating parameters; its conditions are the
    63 quotes and seven supplied curve levels. New observations supply no target
    parameters. Curves vary between training examples and stay fixed within each
    inference problem.

    Offline training reuses expensive price simulations. A longer run can use a
    downloaded matching training bank through the same `training_data` API.
    The pretrained estimator used 100,000 training surfaces and a larger network;
    this action is a small demonstration of the API.
    """)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Infer parameters with the curve supplied

    We now use the pretrained estimator on one synthetic option market.
    The seven curve levels are known inputs; only $H,\eta,\rho$ are inferred.
    Their priors are independent uniforms on the ranges in the table. The plots show paired posterior draws in the original parameter units.
    Three checks against MCMC under a learned pricer follow below. References
    under the original rough-model likelihood remain outstanding.
    """)
    return


@app.cell(hide_code=True)
def _(backbones, focused_posterior, known_arrays, known_task, mo, np, plt):
    _contracts = known_task["context"]["contracts"]
    _t = np.asarray([c["maturity"] for c in _contracts])
    _k = np.asarray([c["strike"] for c in _contracts])
    _noise = np.asarray(known_task["context"]["quote_noise_std"])
    _fig, _axes = plt.subplots(1, 2, figsize=(9.5, 2.9), constrained_layout=True)
    for _maturity in np.unique(_t):
        _at = _t == _maturity
        _axes[0].errorbar(_k[_at], known_arrays["observations"][_at], yerr=_noise[_at],
                          fmt=".", label=f"{252 * _maturity:g} days")
    _axes[0].set(xlabel="strike / forward", ylabel="premium / forward", title="The observed option quotes")
    _axes[0].legend(fontsize=6, ncol=2)
    _knots = known_task["model_context"]["forward_variance_knots"]
    _axes[1].stairs(known_arrays["context_values"], _knots, color="#7755aa")
    _axes[1].set(xlabel="maturity (years)", ylabel="annualized forward variance", title="The supplied curve")
    mo.vstack([
        _fig,
        mo.md("Strike / forward is K/F; premium / forward is the option price divided by F. "
              "A premium of 0.02 means 2 when the forward is 100. Puts appear at/below the forward "
              "and calls above it. Bars show one quote-noise SD."),
        mo.ui.table(backbones.parameter_rows(known_task, known_arrays), selection=None, pagination=False,
                    format_mapping={"median": "{:.4g}".format}),
        focused_posterior(known_task, known_arrays),
        mo.md("The marginals summarize each parameter; the joint panel preserves dependence. "
              "Can you explain which combinations of variance amplitude and correlation remain plausible?"),
        mo.accordion({"All parameter relationships": backbones.corner(known_task, known_arrays)}),
    ])
    return


@app.cell(hide_code=True)
def _(RESULTS, backbones, known_report, known_task, mo, teaching):
    _comparison = teaching.load_json(RESULTS / "rough-known" / "surrogate-check" / "comparison.json")
    if _comparison is None:
        raise ValueError("The saved known-curve surrogate comparison is missing.")
    _pricer = _comparison["reference_pricer"]
    _training = _pricer["training_record"]
    if (_pricer["task_fingerprint"] != backbones.replay.fingerprint(known_task)
            or _pricer["task_fingerprint"] != known_report["task_fingerprint"]
            or _training["task_id"] != known_task["task_id"]
            or _comparison["posterior_checkpoint_sha256"] != known_report["checkpoint_sha256"]
            or _comparison["bank_manifest_sha256"] != known_report["bank_manifest_sha256"]):
        raise ValueError("The surrogate comparison does not match this task and posterior estimator.")
    _labels = {"hurst": "H", "eta": "η", "rho": "ρ"}
    _rows, _details = [], []
    for _i, _case in enumerate(_comparison["cases"]):
        _agreement = _case["known100k_neural"]["proposed_parameter_agreement"]
        _row = {"synthetic observation": "Worked example" if _case["surface_id"] == known_report["case_id"] else f"Market {_i + 1}"}
        for _name, _stats in _agreement["parameters"].items():
            _passed = (_stats["mean_error_within_0_3_reference_sd"]
                       and _stats["relative_width_error_within_25_percent"])
            _row[_labels[_name]] = "passed" if _passed else "failed"
            _details.append({"stored case": _case["surface_id"], "parameter": _labels[_name],
                             "mean gap / reference SD": f"{_stats['standardized_mean_error']:.3f}",
                             "90% width / reference width": f"{_stats['width_ratio_90']:.3f}"})
        _row["all parameter checks"] = "passed" if _agreement["all_checks_passed"] else "failed"
        _rows.append(_row)
    _rmse = _training["validation_errors"]["rmse"]
    mo.vstack([
        mo.md("### An alternate calculation on three synthetic markets"),
        mo.md(f"The pretrained **100,000-market posterior estimator** was compared with PyMC MCMC "
              f"under a **fixed pricer trained on {_training['training_surfaces']:,} markets**. "
              "The observation, prior, supplied curve and quote-noise assumptions match for each case. "
              "The reference pricer was held fixed; it was not replaced by a newer 100,000-market pricer."),
        mo.ui.table(_rows, selection=None, pagination=False),
        mo.md("Two cases meet the original comparison screens. In the third, **H and η fail both "
              "the mean and interval-width screens**; ρ passes. These differences remain visible even "
              "though aggregate calibration checks can look good."),
        mo.md(f"The pricer's validation RMSE was **{_rmse:.3f} quote-noise SDs** against numerical "
              "mean-price labels. This averages over contracts and markets; it does not bound the "
              "resulting posterior error. The reference targets the deterministic surrogate likelihood, "
              "with no correction for Monte Carlo-label uncertainty or pricing/grid bias. It is an "
              "approximate comparison, not an exact-model reference or validation of the neural posterior everywhere."),
        mo.accordion({"Comparison screens and recorded scope": mo.vstack([
            mo.md("The original descriptive screens require a mean gap below 0.3 reference SD "
                  "and a central 90% width within 25% of the reference width, for every parameter. "
                  "Each neural comparison uses 1,000 saved draws; each reference has four chains of "
                  "8,000 retained draws. Quantile MCSE and neural Monte Carlo error were not assessed. "
                  "These three validation observations are not reserved-test results."),
            mo.ui.table(_details, selection=None, pagination=False),
        ])}),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Return to the data: posterior predictive simulation

    Draw complete parameter rows, price the same contracts, then add fresh quote
    noise. This separates uncertainty about expected prices from variation in
    replicated observations:

    $$\boldsymbol\vartheta^{(m)}\sim q(\boldsymbol\vartheta\mid\mathbf y,\xi_0),
    \qquad \widetilde{\mathbf y}^{(m)}=
       \mathbf P(\boldsymbol\vartheta^{(m)},\xi_0)+\boldsymbol\sigma\odot\boldsymbol\epsilon^{(m)}.$$

    The curve stays at its supplied value. The action below uses a small number
    of paired draws and checks Monte Carlo precision. It is an illustration,
    not a stable estimate of tail probabilities or a proof of posterior accuracy.
    """)
    return


@app.cell(hide_code=True)
def _(mo, teaching):
    def simulate_predictive(task, case):
        return teaching.rough_posterior_predictive(
            task, case["neural_draws"], case["observations"],
            context_values=case["context_values"], draws=8, maturity=.25, seed=2026)

    mo.show_code()
    return (simulate_predictive,)


@app.cell(hide_code=True)
def _(mo):
    predictive_rough = mo.ui.run_button(label="Simulate a small posterior-predictive check")
    predictive_rough
    return (predictive_rough,)


@app.cell
def _(known_arrays, known_task, mo, np, plots, plt, predictive_rough, simulate_predictive):
    mo.stop(not predictive_rough.value, mo.md("*This optional action reprices posterior draws; opening the notebook does not run it.*"))
    try:
        _check = simulate_predictive(known_task, known_arrays)
    except (RuntimeError, ValueError) as _error:
        mo.stop(True, mo.callout(mo.md(f"**Predictive simulation stopped:** {_error}. No partial sample is presented as a posterior-predictive band."), kind="warn"))
    _k = _check["strikes"]
    _noise = _check["quote_noise_std"]
    _center = np.median(_check["reference_prices"], axis=0)
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True)
    for _values, _label, _color in [(_check["reference_prices"], "estimated expected-price draws", plots.SERIES[0]),
                                    (_check["replicated_quotes"], "replicated quotes", plots.SERIES[1])]:
        _lo, _hi = np.quantile(_values, [.1, .9], axis=0)
        _axes[0].fill_between(_k, _lo, _hi, alpha=.25, color=_color, label=_label)
        _axes[1].fill_between(_k, (_lo - _center) / _noise, (_hi - _center) / _noise,
                              alpha=.25, color=_color, label=_label)
    _axes[0].plot(_k, _check["observations"], "k.", label="observed")
    _axes[0].set(xlabel="strike / forward", ylabel="premium / forward", title="Three-month option prices")
    _axes[0].legend(fontsize=7)
    _axes[1].plot(_k, (_check["observations"] - _center) / _noise, "k.")
    _axes[1].set(xlabel="strike / forward", ylabel="price difference / quote SD",
                 title="The same uncertainty in quote-noise units")
    _meta = _check["metadata"]
    mo.vstack([_fig, mo.md("Shading shows empirical central 80% ranges from **8 paired parameter draws**, "
                          "with eight fresh quote-noise draws per parameter row. Tail estimates remain coarse. The right panel subtracts the posterior-median expected price at each strike. "
                          "Both bands retain monitored Monte Carlo pricing error; replicated quotes additionally include quote noise. Numerical variance is not added to the declared observation variance."),
               mo.md(f"Largest estimated Monte Carlo SE / quote SD: **{_meta['max_mc_se_over_sigma']:.2f}**. "
                     f"Wall time: **{_meta['runtime_seconds']:.1f} s**. This checks simulation precision, not time-grid bias."),
               mo.accordion({"Simulation precision and budget": mo.ui.table(
                   [{"setting": key, "value": str(value)} for key, value in _meta.items()],
                   selection=None, pagination=False)})])
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Extension: infer the curve too

    If the curve is uncertain, treating it as fixed removes part of the
    uncertainty. In the joint task, the three coefficients of
    $\xi_0(t)=L+(S-L)e^{-kt}$ join $H,\eta,\rho$ as targets.
    The generating curve is **not** an inference condition.

    These examples use different observations and a different curve family from
    the known-curve case. Switching between them is not a controlled comparison
    of interval widths.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    joint_case = mo.ui.dropdown(options={
        "Worked joint example": "rough-joint100k-validation0",
        "Curve level–decay trade-off": "rough-joint100k-validation1983",
        "Coupled volatility parameters": "rough-joint100k-validation1941",
        "Weakly informed decay": "rough-joint100k-validation1962",
    }, value="Worked joint example", label="Joint inference example")
    joint_case
    return (joint_case,)


@app.cell(hide_code=True)
def _(backbones, focused_posterior, joint_case, mo, np):
    _task, _report, _arrays = backbones.load_example(joint_case.value)
    _pair = ("eta", "rho") if joint_case.value.endswith("1941") else ("long_variance", "decay")
    _rates = backbones.replay.rough_variance_products(_task, _arrays["neural_draws"])
    _q = np.quantile(_rates["variance_swap_1y"], [.05, .5, .95])
    mo.vstack([
        mo.ui.table(backbones.parameter_rows(_task, _arrays), selection=None, pagination=False,
                    format_mapping={"median": "{:.4g}".format}),
        focused_posterior(_task, _arrays, pair=_pair),
        mo.md(f"**Propagate paired draws.** The one-year mean variance rate has median **{_q[1]:.4f}** "
              f"and central 90% interval **{_q[0]:.4f}–{_q[2]:.4f}**, in variance/year. "
              "This is the analytic curve average, not a new realized payoff or a predictive data set. "
              "With a supplied curve it would have zero parameter-induced width by construction."),
        mo.accordion({"All parameter relationships": backbones.corner(_task, _arrays),
                      "Case and model details": mo.md(f"Stored synthetic case `{_report['case_id']}`. "
                          "Task and checkpoint identities are verified when loading. These selected examples "
                          "illustrate geometry; they are not a conditional-accuracy study.")}),
        mo.md("Which curve combinations remain plausible? Why can weakly informed decay coexist with "
              "a better determined average variance rate?"),
    ])
    return


@app.cell(hide_code=True)
def _(known_report, mo):
    mo.md(r"""
    ## 8. What do the checks establish?

    Financial calibration estimates model parameters. Statistical calibration
    asks whether the uncertainty statements behave as advertised across new
    simulated data sets.

    | Check | Question |
    |:---|:---|
    | Recovery | Do posterior estimates track generating values? |
    | Coverage | Do 90% intervals contain truth about 90% of the time? |
    | Simulation-based calibration | Does truth behave like another posterior draw? |
    | Conditional reference | Does an independently checked calculation agree for this observation? |
    | Posterior predictive | Can the posterior generate plausible observations? |

    Each complete market is one evaluation case. Passing average calibration is
    not enough: a network that returns the prior can pass while learning nothing.
    The current synthetic fits have aggregate diagnostics, and three known-curve
    observations have the surrogate-likelihood comparisons shown above.
    Original-model likelihood references, joint-curve references and downstream
    Monte Carlo product checks remain unfinished. Earlier pilots do not supply
    that missing evidence.
    """)
    return


@app.cell(hide_code=True)
def _(ASSETS, backbones, known_report, mo):
    mo.accordion({"Diagnostic definitions and reproducibility": mo.md(r"""
    For $N$ independent simulated markets, coverage of a nominal 90% interval is
    $\widehat C_j=N^{-1}\sum_i\mathbf1\{\vartheta^*_{ij}\in C^{.9}_{ij}\}$.
    With $M$ posterior draws, the rank
    $R_{ij}=\sum_m\mathbf1\{\vartheta^{(m)}_{ij}<\vartheta^*_{ij}\}$ should be
    uniform on $\{0,\ldots,M\}$ under a correct continuous posterior.

    Finite case counts and dependent draws require care. These prior-averaged
    statements do not establish calibration at every observation, correct joint
    dependence, or adequacy for a real market.

    The current cases are stored under `assets/backbones/`; loaders verify
    their task and model identities. Full diagnostics and native checkpoint
    sampling are available in [the backbone review](backbone_review.py).
    """)})
    return


@app.cell(hide_code=True)
def _(ASSETS, mo, teaching):
    _rows = []
    for _label, _name in [("Supplied curve", "rough-known100k-validation0"),
                            ("Inferred curve", "rough-joint100k-validation0")]:
        _report = teaching.load_json(ASSETS / "backbones" / _name / "evaluation.json")
        for _j, _parameter in enumerate(_report["targets"]):
            _rows.append({"task": _label, "parameter": _parameter,
                          "coverage of 90% intervals": f"{100 * _report['coverage']['0.9']['coverage'][_j]:.1f}%",
                          "rank screen": "passed" if _report["sbc"]["passed"][_j] else "failed"})
    mo.accordion({"Saved aggregate diagnostics": mo.vstack([
        mo.md("Each task was evaluated on 2,000 synthetic validation markets, with 1,000 "
              "draws per market. These are prior-averaged checks on the current fits; "
              "they are not reserved-test results and do not themselves establish conditional agreement."),
        mo.ui.table(_rows, selection=None, pagination=False),
    ])})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    1. **The same workflow accommodates a different simulator.** Monte Carlo changes data-generation cost, not the Bayesian goal.
    2. **State which curve inputs are known.** Supplying a curve and inferring a curve define different posteriors.
    3. **Keep numerical error separate from observation noise.** Check pricing precision before interpreting the uncertainty.
    4. **Preserve paired draws.** Parameter and curve dependence matters for derived quantities.
    5. **Check the approximation.** A plausible posterior or predictive plot alone does not establish accuracy.

    *Code and evidence:* [rough-model curriculum](../curriculum/05-slot5-rough-bergomi-to-a-reserve.md),
    `src/sbi4f_tutorial/rough_options/`, and `assets/backbones/`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    archive_toggle = mo.ui.checkbox(value=False, label="Show earlier SPY examples and pilot experiments")
    mo.accordion({"Earlier SPY examples and pilot experiments": mo.vstack([
        mo.md("Some examples use real SPY quotes; others use synthetic markets. "
              "They use separate estimators and numerical experiments, so they are "
              "not outputs of the current worked example. Failed or incomplete checks remain labelled."),
        archive_toggle,
    ])})
    return (archive_toggle,)


@app.cell(hide_code=True)
def _(archive_toggle, mo):
    mo.stop(not archive_toggle.value)
    mo.md(r"""
    ## Earlier SPY examples and pilot experiments

    These examples combine real SPY quotes with separate synthetic pilot markets.
    Their estimators and numerical methods differ from the worked example above.
    Recorded costs describe those earlier workloads, not current software speed.
    The SPY route supplies the short-maturity curve; the long-end experiment uses
    eleven uncertain curve levels. Neither is the six-target exponential-curve
    estimator shown above.

    | | Slot 4: Heston | Slot 5: rough Bergomi |
    |:---|:---|:---|
    | Data | the SPY smile: 63 prices | the same |
    | Unknowns | $v_0$, $\theta$, $\sigma_v$, $\rho$ | $H$, $\eta$, $\rho$, given the market's forward-variance curve $\xi_0$ |
    | One smile priced by | a Fourier integral, 20 ms | Monte Carlo, 5 CPU-seconds |
    | The bank | 60,000 smiles in 46 CPU-minutes | 112,000 smiles in about 370 CPU-hours, on a cluster |
    | The network | flow matching on the prices | the same, with $\xi_0$ as an extra condition |
    | Checks in that study | MCMC under the Fourier likelihood | SBC, a second method, and an audit of each smile |
    | The number at the end | a posterior, and what it says about Heston | a price range and a reserve, out to the long end |
    """)
    return


@app.cell(hide_code=True)
def _(archive_toggle, mo):
    mo.stop(not archive_toggle.value)
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
        - The variance depends on the whole past of $W$. This implementation supplies no Heston-style
          Fourier pricer and uses Monte Carlo: the hybrid scheme for
          $Y$, the conditional Black–Scholes estimator for the options, antithetic paths and control variates.

        #### 2 · What the smile identifies

        The short skew sets the product $\rho\eta$. The skew's term structure sets $H$. $\eta$ is separated
        from $\rho$ by the curvature of the wings and by the long-dated at-the-money level measured against
        $\xi_0$. On SPY the wings suffice: with each level of $\xi_0$ 20% uncertain, $\eta$'s local 90%
        interval stays within a tenth of its prior range (section 6). The network takes $\xi_0$ as given up
        to one year; beyond the last quote it is an assumption (section 8).

        #### 3 · The likelihood

        The intended quote likelihood uses variance $\sigma_c^2$. A scalar Gaussian
        convolution identity can motivate subtracting numerical variance $s_c^2$
        inside a noisy likelihood estimator **only** for centered Gaussian Monte Carlo
        error with known variance, independent of quote noise, and $s_c^2<\sigma_c^2$.
        For joint prices with shared paths, the analogous correction needs the full
        covariance $\Sigma_{\rm quote}-\Sigma_{\rm MC}$ to be positive definite.
        Diagonal subtraction alone is not an unbiased joint-likelihood guarantee.
        Estimated covariances, non-Gaussian tails and grid bias add limitations.
        The historical audit is therefore an approximation under stated assumptions,
        not an exact rough posterior certificate (course, Lecture 8, Section 8.9).
        """)
    })
    return


@app.cell(hide_code=True)
def _(archive_toggle, mo):
    mo.stop(not archive_toggle.value)
    mo.md(r"""
    ### Earlier study 1 · The question

    SPY on 21 October 2026: the 63 prices of slots 1 and 4, read from the chain onto the design grid. A
    bank marks products the smile does not quote: a forward-start call (one period of a cliquet), a
    volatility swap, and long-dated exotics such as autocallables and cliquets that run for years past the
    last liquid quote. Least squares gives one parameter vector and one mark. The rules ask for the range
    and a 90% point in it: Regulation (EU) 2016/101, Art. 9 for the calibrated parameters and Art. 11 for
    the choice of model, and IFRS 13 for the sensitivity of marks that rest on unobservable inputs.
    """)
    return


@app.cell(hide_code=True)
def _(RESULTS, archive_toggle, image, mo):
    mo.stop(not archive_toggle.value)
    image(RESULTS / "spy" / "surface.png", "The SPY chain of 21 October 2026 read onto the design grid (study S8).")
    return


@app.cell(hide_code=True)
def _(archive_toggle, mo):
    mo.stop(not archive_toggle.value)
    mo.md(r"""
    ### Earlier study 2 · Why rough Bergomi

    The at-the-money skew of SPY falls with maturity as a power law. Heston, at slot 4's posterior mean,
    cannot follow it: its skew tends to a constant as $T \to 0$. Rough Bergomi's skew goes like
    $T^{H-1/2}$ at short maturities, and the model takes $\xi_0$ from the market, so its variance swaps match
    SPY's by construction.
    """)
    return


@app.cell
def _(ASSETS, CHECKPOINTS, RESULTS, archive_toggle, mo, np, surface, teaching):
    mo.stop(not archive_toggle.value)
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
def _(archive_toggle, heston_skew, mo, plots, plt, power_law, skew_days, spy_skew, surface):
    mo.stop(not archive_toggle.value)
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
def _(archive_toggle, mo):
    mo.stop(not archive_toggle.value)
    mo.md(r"""
    ### Earlier study 3 · What a numerical posterior would cost

    One numerical likelihood evaluation uses a Monte Carlo surface. The historical cost projection uses
    about 12,300 evaluations for 400 effective draws, giving at least 17 CPU-hours per smile. It extrapolates
    from the Heston random-walk chains at the cheapest recorded pricing cost (the pilot bank's average
    gives 41 CPU-hours). A noisy-likelihood method can have different mixing and precision costs; this is
    a workload estimate, not an achieved converged rough posterior or a guarantee for every method.
    Importance sampling from a Gauss–Newton proposal takes about 85 CPU-minutes on SPY's 63 quotes (1,024
    evaluations at 5.0 CPU-seconds). Where that proposal works, one smile and a book are affordable; a
    validation on thousands of simulated smiles is not. Slot 4's exact check is not available here.
    """)
    return


@app.cell(hide_code=True)
def _(ASSETS, RESULTS, archive_toggle, image, mo, teaching):
    mo.stop(not archive_toggle.value)
    _costs = teaching.load_json(RESULTS / "costs" / "costs.json")
    mo.vstack([
        image(ASSETS / "images" / "ladder.png", "One likelihood evaluation and estimated posterior workload under each model "
                                                    "(the rough-Bergomi random-walk workload is an extrapolated lower bound)."),
        mo.ui.table(teaching.cost_rows(_costs), selection=None, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(archive_toggle, mo):
    mo.stop(not archive_toggle.value)
    mo.md(r"""
    ### Earlier study 4 · SPY neural route · export status

    The current 100,000-surface estimators already have packaged synthetic replays above. A matching
    native SPY posterior and the historical S2–S7 exports are separate artifacts. In the known-curve
    training task, parameters and curves vary across simulations and the realized curve is supplied as
    an inference condition. In the joint task its three coefficients are targets instead.
    """)
    return


@app.cell(hide_code=True)
def _(CHECKPOINTS, archive_toggle, mo, pending, teaching):
    mo.stop(not archive_toggle.value)
    _ready = teaching.existing(CHECKPOINTS / "rough" / "model.keras")
    pending("a matching native SPY posterior and historical-route export (study S8); the completed current "
            "100k fits are available above.") if _ready is None else mo.md(
                "A historical rough checkpoint file is present. File presence alone does not establish "
                "its compatibility or validate a current SPY posterior.")
    return


@app.cell(hide_code=True)
def _(RESULTS, archive_toggle, mo, pilot_banner, teaching):
    mo.stop(not archive_toggle.value)
    pilot = teaching.load_npz(RESULTS / "rough-pilot" / "s2" / "draws.npz")
    pilot_surface = mo.ui.slider(0, (len(pilot["truth"]) - 1) if pilot else 0, value=0, label="pilot test smile",
                                 show_value=True)
    mo.vstack([pilot_banner("the pilot network on held-out simulated smiles."), pilot_surface]) if pilot else mo.md("")
    return pilot, pilot_surface


@app.cell(hide_code=True)
def _(archive_toggle, mo, pending, pilot, pilot_surface, teaching):
    mo.stop(not archive_toggle.value)
    (teaching.pairs({"network (pilot)": pilot["draws"][pilot_surface.value]}, ["H", "η", "ρ", "ρη"],
                    truth=pilot["truth"][pilot_surface.value], figsize=(7.4, 7.4))
     if pilot else pending("the pilot network's draws."))
    return


@app.cell(hide_code=True)
def _(archive_toggle, mo):
    mo.stop(not archive_toggle.value)
    mo.md(r"""
    ### Earlier study 5 · Is it right without an exact answer? · historical and pilot studies

    The original study design used three checks, each answering a different question.
    The saved historical results and explicit missing exports below do not certify the new 100k fits.

    1. **Calibrated on average?** Simulation-based calibration on held-out simulated smiles.
    2. **Does another calculation agree?** MCMC on a neural pricer samples a surrogate
       likelihood. Heston supplies a useful control against its Fourier reference.
       The three known-curve comparisons above use a separate current estimator;
       none provides an original-model rough-Bergomi likelihood reference.
    3. **Right on this smile?** This archived audit used importance weighting with a Monte Carlo
       likelihood. It draws 256 parameter values from a widened fit to the network's posterior and reweights
       them with the likelihood of section 3. The efficiency and the Pareto shape $\hat k$ decide, with the
       threshold 0.58 used for 256 draws in that study. A warning calls for a better proposal or
       an independently checked reference; neither route is certified merely by passing a weight diagnostic.
    """)
    return


@app.cell(hide_code=True)
def _(RESULTS, archive_toggle, image, mo, pilot_banner):
    mo.stop(not archive_toggle.value)
    mo.vstack([
        image(RESULTS / "rough" / "s2" / "sbc.png", "Rough Bergomi, full bank: SBC on held-out smiles."),
        pilot_banner("the pilot network fails SBC for ρ and ρη."),
        image(RESULTS / "rough-pilot" / "s2" / "sbc.png", "Rough Bergomi, pilot network: rank ECDF minus uniform."),
    ])
    return


@app.cell(hide_code=True)
def _(RESULTS, archive_toggle, image, mo, pending, teaching):
    mo.stop(not archive_toggle.value)
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
        _parts += [mo.md("**MCMC on the pricer against the Fourier-likelihood importance reference**: "
                         "this archived Heston comparison uses reweighted proposal summaries, not direct MCMC references for every case."),
                   mo.ui.table(teaching.agreement_rows(_exact, key="pricer_vs_exact"), selection=None, pagination=False)]
    _parts.append(mo.md("The three current known-curve surrogate comparisons are shown in the main lesson. "
                        "A matching neural-pricer posterior for these earlier rough SPY observations is not included."))
    mo.accordion({"Historical method comparison · technical diagnostic": mo.vstack([
        mo.md("These archived comparison panels use the original study’s plotting coordinates. "
              "The current synthetic example above is the primary posterior view, in physical parameter units. "
              "Heston agreement does not establish a rough-Bergomi reference."),
        *_parts,
    ])})
    return


@app.cell(hide_code=True)
def _(RESULTS, archive_toggle, image, mo, pending, pilot_banner, teaching):
    mo.stop(not archive_toggle.value)
    _audit = teaching.load_json(RESULTS / "rough-pilot" / "s5" / "audit.json")
    if _audit is None:
        _out = pending("the audit (study S5).")
    else:
        _s = _audit["summary"]
        _out = mo.vstack([
            pilot_banner("computed with a likelihood that counted Monte Carlo variance twice; this stored result does not show "
                         "completion of a corrected rerun (gate G3)."),
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
def _(archive_toggle, mo):
    mo.stop(not archive_toggle.value)
    mo.md(r"""
    ### Earlier study 6 · The classical route on the same smiles

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
def _(CHECKS, archive_toggle, mo, np, plots, plt, teaching):
    mo.stop(not archive_toggle.value)
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
def _(RESULTS, archive_toggle, mo, teaching):
    mo.stop(not archive_toggle.value)
    _models = {name: key for name, key in (("rough Bergomi", "rough"), ("rough Bergomi (pilot)", "rough-pilot"),
                                           ("Heston", "heston"))
               if teaching.existing(RESULTS / key / "s6" / "classical.json")}
    classical_model = mo.ui.dropdown(options=_models, value=next(iter(_models)) if _models else None, label="model")
    classical_model if _models else mo.md("")
    return (classical_model,)


@app.cell(hide_code=True)
def _(RESULTS, archive_toggle, classical_model, image, mo, pending, pilot_banner, teaching):
    mo.stop(not archive_toggle.value)
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
def _(archive_toggle, mo):
    mo.stop(not archive_toggle.value)
    mo.md(r"""
    ### Earlier study 7 · From posterior to prices: the range and the reserve · stored studies

    Each posterior draw prices each path-dependent product by Monte Carlo, with the same random numbers for
    every draw. This reduces pricing-noise contributions to differences between draws; numerical error still
    requires its own checks. The least-squares calibration prices them too. The **reserve** for a sold position is the distance from the
    least-squares price to the chosen quantile of the posterior prices.

    With the curve fixed, as in this historical baseline, the continuously monitored variance swap has
    zero parameter-induced range: its strike is the average of the supplied $\xi_0$. That is a consistency check of the curve input, not a
    finding; with daily monitoring the strike does depend on $(H, \eta, \rho)$ (Lecture 8). Under Heston the
    variance swap carries a range, because Heston's forward variance depends on its parameters.

    The model and test-smile controls below select stored historical product panels. They are independent
    of the current development-case selector. Before increasing the confidence level, predict which
    reported quantiles should move; afterward, explain why this changes the reported range without
    adding information to the posterior.
    """)
    return


@app.cell(hide_code=True)
def _(RESULTS, archive_toggle, mo, teaching):
    mo.stop(not archive_toggle.value)
    _models = {name: key for name, key in (("rough Bergomi", "rough"), ("rough Bergomi (pilot)", "rough-pilot"),
                                           ("Heston", "heston"))
               if teaching.existing(RESULTS / key / "s7" / "price-draws.npz")}
    product_model = mo.ui.dropdown(options=_models, value=next(iter(_models)) if _models else None, label="model")
    level = mo.ui.slider(0.80, 0.99, step=0.01, value=0.90, label="confidence level", show_value=True)
    mo.hstack([product_model, level]) if _models else mo.md("")
    return level, product_model


@app.cell(hide_code=True)
def _(RESULTS, archive_toggle, mo, product_model, teaching):
    mo.stop(not archive_toggle.value)
    products = (teaching.load_npz(RESULTS / product_model.value / "s7" / "price-draws.npz") if product_model.value else None)
    product_rows = sorted(int(k.split("_")[1]) for k in (products or {}) if k.startswith("prices_"))
    product_surface = mo.ui.dropdown(options={f"Synthetic market {i + 1}": str(row)
                                               for i, row in enumerate(product_rows)},
                                     value="Synthetic market 1" if product_rows else None,
                                     label="Stored synthetic market")
    product_surface if product_rows else mo.md("")
    return product_rows, product_surface, products


@app.cell(hide_code=True)
def _(archive_toggle, level, mo, pending, pilot_banner, product_model, product_rows, product_surface, products, teaching):
    mo.stop(not archive_toggle.value)
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
def _(archive_toggle, mo):
    mo.stop(not archive_toggle.value)
    mo.md(r"""
    ### Earlier study 8 · The long end: where the reserve comes from · separate Gaussian/Laplace experiment

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
def _(archive_toggle, mo):
    mo.stop(not archive_toggle.value)
    long_prior = mo.ui.slider(0.05, 0.40, step=0.05, value=0.20, label="prior width beyond the data (sd of log ξ0 per √year)",
                              show_value=True)
    long_noise = mo.ui.slider(0.25, 1.50, step=0.25, value=0.50, label="LEAP variance-swap noise (vol points)", show_value=True)
    mo.hstack([long_prior, long_noise], justify="start")
    return long_noise, long_prior


@app.cell
def _(CHECKS, archive_toggle, long_end, long_noise, long_prior, mo, np, teaching):
    mo.stop(not archive_toggle.value)
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
def _(archive_toggle, long_cov, long_desk, long_end, long_extra, long_mean, long_noise, long_post, long_report, long_shares, mo, np, plots, plt, spy):
    mo.stop(not archive_toggle.value)
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
def _(archive_toggle, mo):
    mo.stop(not archive_toggle.value)
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
def _(archive_toggle, mo):
    mo.stop(not archive_toggle.value)
    mo.md(r"""
    ### Earlier study 9 · Heston against rough Bergomi on the same smile

    The posterior's range is the parameter part of model risk; the choice of model is the other part. Both
    models, calibrated to the same smile, price the forward-start differently, and the gap is judged smile
    by smile.
    """)
    return


@app.cell(hide_code=True)
def _(RESULTS, archive_toggle, image, mo, pending, pilot_banner, teaching):
    mo.stop(not archive_toggle.value)
    _across = teaching.load_json(RESULTS / "rough-pilot" / "s7-across" / "across-models.json")
    _rough_prices = teaching.load_npz(RESULTS / "rough-pilot" / "s7" / "price-draws.npz")
    _heston_prices = teaching.load_npz(RESULTS / "rough-pilot" / "s7-across" / "heston-price-draws.npz")
    if _across is None or _rough_prices is None or _heston_prices is None:
        _out = pending("the across-model pilot report and price arrays.")
    else:
        _rows = [{"synthetic case": case["test_row"],
                  "Heston fit, χ² on 63 quotes": f"{case['heston_least_squares']['chi2']:.1f}",
                  "importance ESS / 2048": f"{case['heston_audit']['efficiency'] * _across['settings']['proposals']:.1f}",
                  "largest importance weight": f"{100 * case['heston_audit']['max_weight']:.3f}%"}
                 for case in _across["surfaces"]]
        _out = mo.vstack([
            pilot_banner("Three synthetic rough-Bergomi markets, compared using earlier estimators."),
            mo.ui.table(_rows, selection=None, pagination=False),
            teaching.product_comparison_figure(_across, _rough_prices, _heston_prices),
            mo.md("**Read the price distributions.** These show estimated expected product prices across "
                  "parameter draws, not distributions of future payoffs. The supplied rough-model curve "
                  "fixes the variance-swap price, so it appears as a line. That says nothing about "
                  "uncertainty in realized future variance. Dashed lines are least-squares price estimates. "
                  "The rough model receives supplied curve levels as well as quotes; Heston infers its "
                  "mean variance curve. Widths therefore reflect different conditioning assumptions and "
                  "product sensitivities, and need not be smaller under rough Bergomi."),
            mo.callout(mo.md("**The Heston approximation failed on case 291.** One proposal received "
                            "99.990% of the importance weight; the 400 saved parameter rows are identical. "
                            "The resulting price spikes have therefore been replaced by a sampling-failure "
                            "notice. They do not show a certain posterior. The other Heston panels pass "
                            "the displayed audit screen; that alone does not certify exact-model accuracy."), kind="warn"),
            mo.md("The earlier forward-start column uses the return-ratio payoff "
                  "E[(S₀.₅ / S₀.₂₅ − 1)⁺], rather than a spot-notional payoff. Pricing uses common "
                  "random numbers across parameter draws. Numerical price error and pilot calibration "
                  "limitations remain; these are not results from the current known- or joint-curve estimator."),
            mo.accordion({"Original unfiltered figure": mo.vstack([
                mo.md("**Preserved for comparison:** this original image labels the failed Heston "
                      "sample as a posterior and displays its duplicate draws as zero-width histograms. "
                      "Use the diagnosed rendering above when interpreting uncertainty."),
                image(RESULTS / "rough-pilot" / "s7-across" / "across-models.png",
                      "Original pilot figure, including the failed Heston importance sample."),
            ])}),
            pending("the two models' forward-start distributions for matching real SPY quotes."),
        ])
    _out
    return

@app.cell(hide_code=True)
def _(archive_toggle, mo):
    mo.stop(not archive_toggle.value)
    mo.md(r"""
    ### Earlier study 10 · What the answer assumes, and what it takes

    This historical SPY answer is conditional on four things: **the model**, **the prior**, **the quote-noise model**, and
    **$\xi_0$**, read off the same smile up to one year and treated as exact there, with its long end priced
    as in the long-end study. Each is an assumption to examine.

    The chain is:

    1. the model, and a stated prior;
    2. a declared quote-noise model, distinguished from numerical pricing error;
    3. a simulated bank on a cluster, and the network;
    4. validation: SBC, a Heston control and an independently assessed second method;
    5. a proposal audit for each smile, with a separately checked reference where feasible;
    6. the products, priced under the posterior with common random numbers;
    7. the reserve, reported with its diagnostics.

    The numerical records remain available for review. Missing panels and failed checks keep their original status.

    ### What these earlier examples illustrate

    1. **Rough Bergomi's posterior is affordable for one smile, not for a validation**: at least 17 CPU-hours
       per smile by random-walk Metropolis, and about 85 CPU-minutes by importance sampling from a
       Gauss–Newton proposal on SPY's 63 quotes (1,024 evaluations at 5.0 CPU-seconds). Where that proposal
       works a book is affordable too; a validation on thousands of simulated smiles is not. The amortized posterior
       was designed to amortize repeated inference. These cost estimates do not validate the archived audit: its pilot likelihood counted Monte Carlo variance twice, and a corrected rerun is not included.
    2. **On SPY's full surface the classical interval is a good approximation**; the posterior differs where
       the quotes are thin, and for the inputs they do not reach.
    3. **Long-dated products rest on the curve beyond the quotes.** Held flat, as in the baseline, it gives a
       range of 0–10%; priced with a stated prior and the LEAPs, the posterior gives about a quarter to a
       third of the price for the forward start, the 5-year put and the variance swap.
    4. **The range is conditional on the model.** Heston's long end is confidently wrong, and the gap across
       models is judged smile by smile.

    **Further discussion:** what changes when the prior or the model is wrong?

    *Evidence and code:* the [curriculum](../curriculum/05-slot5-rough-bergomi-to-a-reserve.md), the
    [course](../../course/README.md) and `src/sbi4f_tutorial/`.
    Pilot values remain historical development evidence, not validated results for the current candidates.
    """)
    return


if __name__ == "__main__":
    app.run()
