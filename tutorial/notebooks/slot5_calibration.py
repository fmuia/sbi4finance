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

    from sbi4f_tutorial.rough_options import lesson as teaching, context_checks, studies
    from sbi4f_tutorial.shared import backbones, full_surface, plots

    ROOT = mo.notebook_dir().parent
    ASSETS = ROOT / "assets"
    RESULTS = ASSETS / "results" / "calibration"
    plots.style()

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

    return (ASSETS, RESULTS, backbones, context_checks, focused_posterior,
            full_surface, mo, np, plots, plt, studies, teaching, time)


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
    surrogate_comparison = teaching.load_json(RESULTS / "rough-known" / "surrogate-check" / "comparison.json")
    if surrogate_comparison is None:
        raise ValueError("The saved known-curve surrogate comparison is missing.")
    _pricer = surrogate_comparison["reference_pricer"]
    _training = _pricer["training_record"]
    if (_pricer["task_fingerprint"] != backbones.replay.fingerprint(known_task)
            or _pricer["task_fingerprint"] != known_report["task_fingerprint"]
            or _training["task_id"] != known_task["task_id"]
            or surrogate_comparison["posterior_checkpoint_sha256"] != known_report["checkpoint_sha256"]
            or surrogate_comparison["bank_manifest_sha256"] != known_report["bank_manifest_sha256"]):
        raise ValueError("The surrogate comparison does not match this task and posterior estimator.")
    _labels = {"hurst": "H", "eta": "η", "rho": "ρ"}
    _rows, _details = [], []
    for _i, _case in enumerate(surrogate_comparison["cases"]):
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
    return (surrogate_comparison,)


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
    Monte Carlo product checks remain unfinished.
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
    _intro = mo.md(r"""
    ## Additional studies

    Each study adds one question to the core workflow. Make a prediction, change
    one assumption or representation, then explain the result. These studies use
    the current estimators and task definitions; the numerical limitations from
    the main lesson still apply.

    | Question | What you will learn |
    |:---|:---|
    | Can I use this estimator on a new market? | Check the simulation design before calibration. |
    | Why must posterior draws stay paired? | Dependence changes uncertainty in derived quantities. |
    | What if two inference routes disagree? | Read a conditional accuracy check. |
    | What lies beyond the last quote? | Separate information from an extrapolation assumption. |
    """)
    additional_study = mo.ui.dropdown(options={
        "Can I use this estimator on a new market?": "context",
        "Why must posterior draws stay paired?": "dependence",
        "What if two inference routes disagree?": "reference",
        "What lies beyond the last quote?": "tail",
    }, value="Can I use this estimator on a new market?", label="Additional study")
    mo.vstack([_intro, additional_study])
    return (additional_study,)


@app.cell(hide_code=True)
def _(additional_study, known_task, mo):
    mo.stop(additional_study.value != "context")
    _intro = mo.md(r"""
    ### 1. Check the simulation design before calibration

    **Question.** The quote grid matches. Is that enough to use the estimator?
    Here we use a fixed SPY market snapshot from 21 October 2026 and its supplied
    variance curve. Compare that curve with the current known-curve simulation prior.

    **Predict first:** does the estimator accept every possible seven-level curve,
    or only curves represented by its training design?
    """)
    context_width = mo.ui.slider(.05, .60, step=.05,
        value=float(known_task["context_prior"]["forward_variance"]["log_perturbation"]),
        label="Log-curve variation in a proposed simulation design", show_value=True)
    mo.vstack([_intro, context_width])
    return (context_width,)


@app.cell(hide_code=True)
def _(RESULTS, additional_study, context_checks, context_width, known_task, mo, teaching):
    mo.stop(additional_study.value != "context")
    import copy as _copy
    _spy = teaching.load_json(RESULTS / "spy" / "surface.json")
    _levels = _spy["forward_variance_levels"]
    _current = context_checks.context_support(known_task, _levels)
    _proposal = _copy.deepcopy(known_task)
    _proposal["context_prior"]["forward_variance"]["log_perturbation"] = context_width.value
    _check = context_checks.context_support(_proposal, _levels)
    mo.vstack([
        context_checks.context_support_figure(_proposal, _levels),
        mo.md(f"SPY's last/first variance-level ratio is **{_current['observed_end_ratio']:.2f}**. "
              f"The current training design permits at most **{_current['maximum_end_ratio']:.2f}**. "
              "This curve is outside that design's support. A network may still return numbers, "
              "but those numbers would be unsupported extrapolation."),
        mo.md("**Try it:** increase the proposed curve variation. The displayed envelope changes; "
              "the installed estimator does not. A ratio inside the envelope is only a necessary "
              "check, not proof that the entire curve is covered."),
        mo.md("Proposed design: **" + ("the curve still violates a necessary bound."
              if _check["violates_necessary_bounds"] else "this ratio check no longer rules the curve out.") + "**"),
        mo.accordion({"Why this is a support check": mo.md(r"""
        The simulation design draws
        $\xi_j=\ell^2(1+s m_j)e^{u_j}$, with $u_j\in[-w,w]$.
        Dividing by the first level removes the common scale $\ell^2$.
        The slope bounds and $e^{\pm2w}$ bound the remaining ratios.
        A violation proves the supplied curve cannot arise from this prior;
        passing does not check every joint constraint.

        Changing this prior is a new training design. Generate compatible pairs
        and train and validate an estimator for it before applying that model.
        The saved SPY quotes are a teaching case, not a live market feed.
        """)}),
        mo.md("**Explain:** what would need to change before you trusted a posterior for this market?"),
    ])
    return


@app.cell(hide_code=True)
def _(additional_study, mo):
    mo.stop(additional_study.value not in ("dependence", "tail"))
    study_case = mo.ui.dropdown(options={
        "Joint worked example": "rough-joint100k-validation0",
        "Curve level–decay trade-off": "rough-joint100k-validation1983",
        "Coupled volatility parameters": "rough-joint100k-validation1941",
        "Weakly informed decay": "rough-joint100k-validation1962",
    }, value="Joint worked example", label="Synthetic market for this study")
    study_case
    return (study_case,)


@app.cell(hide_code=True)
def _(additional_study, backbones, study_case):
    study_task, study_report, study_arrays = backbones.load_example(study_case.value)
    return study_arrays, study_report, study_task


@app.cell(hide_code=True)
def _(additional_study, mo):
    mo.stop(additional_study.value != "dependence")
    _intro = mo.md(r"""
    ### 2. Preserve dependence when carrying uncertainty to a price

    **Question.** Can the marginal distributions alone determine a product's uncertainty?
    For a continuously monitored variance swap, the fair annualized strike is

    $$K_{\rm var}(T)=\frac1T\int_0^T\xi_0(t)\,dt.$$

    In the joint task, each complete posterior row supplies its own curve.
    Compute the strike for each row, then summarize those strikes.

    **Predict first:** what happens if we keep every parameter's marginal
    distribution but destroy the relationship between the curve coefficients?
    """)
    shuffle_curves = mo.ui.checkbox(value=False, label="Compare independently shuffled curve coefficients")
    mo.vstack([_intro, shuffle_curves])
    return (shuffle_curves,)


@app.cell(hide_code=True)
def _(additional_study, known_arrays, known_task, mo, np, shuffle_curves, studies, study_arrays, study_task):
    mo.stop(additional_study.value != "dependence")
    _paired = study_arrays["neural_draws"]
    _other = None
    if shuffle_curves.value:
        _other = _paired.copy()
        _rng = np.random.default_rng(7)
        for _j in (3, 4, 5):
            _other[:, _j] = _rng.permutation(_other[:, _j])
    _rows = []
    for _label, _draws in [("Paired posterior", _paired)] + ([] if _other is None else [("Independent marginals (not posterior)", _other)]):
        for _name, _values in studies.curve_functionals(study_task, _draws).items():
            _q = np.quantile(_values, [.05, .5, .95])
            _rows.append({"representation": _label, "quantity": "3-month mean variance" if _name.endswith("3m") else "1-year mean variance",
                          "median": f"{_q[1]:.5f}", "central 90%": f"{_q[0]:.5f}–{_q[2]:.5f}"})
    _fixed = studies.supplied_curve_functionals(known_task, known_arrays["context_values"])
    mo.vstack([
        studies.curve_functional_figure(study_task, _paired, comparison_draws=_other),
        mo.ui.table(_rows, selection=None, pagination=False),
        mo.md("The shuffled columns retain each marginal but create artificial combinations. "
              "Their histogram is **not a posterior**. Shuffling can widen or narrow a derived range; "
              "the direction and size depend on the case."),
        mo.md(f"**If the curve is supplied exactly**, the same formula gives one value regardless "
              f"of H, η or ρ. In the main known-curve example, its one-year strike is {_fixed['variance_swap_1y']:.5f}. "
              "That is a different observation and conditioning assumption, not a matched width comparison. "
              "A fixed fair strike does not make future realized variance deterministic."),
        mo.accordion({"The calculation": mo.md(r"""```python
rates = studies.curve_functionals(task, paired_parameter_draws)
interval = np.quantile(rates["variance_swap_1y"], [.05, .5, .95])
```
For $\xi_0(t)=L+(S-L)e^{-kt}$,
$K_{\rm var}(T)=L+(S-L)(1-e^{-kT})/(kT)$.
Only the curve coefficients enter this mean. These are analytic rates under the
approximate parameter posterior, not simulated payoffs or a calibration verdict.
""")}),
        mo.md("**Explain:** why would multiplying independently chosen marginal medians or interval endpoints lose information?"),
    ])
    return


@app.cell(hide_code=True)
def _(additional_study, mo, surrogate_comparison):
    mo.stop(additional_study.value != "reference")
    _intro = mo.md(r"""
    ### 3. Read a disagreement between two inference routes

    **Question.** Can a small average pricing error coexist with a conditional
    posterior discrepancy? Revisit the matching current known-curve comparisons.
    The surrogate likelihood is an alternate approximate target; agreement is
    evidence to assess, not a certificate for either method.

    **Predict first:** which parameter checks fail in the third market?
    """)
    reference_case = mo.ui.dropdown(options={
        "Worked market": 0, "Second market": 1, "Third market": 2,
    }, value="Third market", label="Conditional comparison")
    mo.vstack([_intro, reference_case])
    return (reference_case,)


@app.cell(hide_code=True)
def _(additional_study, mo, np, plots, plt, reference_case, surrogate_comparison):
    mo.stop(additional_study.value != "reference")
    _case = surrogate_comparison["cases"][reference_case.value]
    _stats = _case["known100k_neural"]["proposed_parameter_agreement"]["parameters"]
    _names = ["hurst", "eta", "rho"]
    _mean = [_stats[n]["standardized_mean_error"] for n in _names]
    _width = [_stats[n]["width_ratio_90"] for n in _names]
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3), layout="constrained")
    _axes[0].bar(["H", "η", "ρ"], _mean, color=plots.SERIES[0])
    _axes[0].axhline(.3, ls="--", color="#64748b", label="Recorded screen: 0.3")
    _axes[0].set(ylabel="Absolute mean gap / reference SD", title="Location disagreement")
    _axes[0].legend(fontsize=8)
    _axes[1].axhspan(.75, 1.25, color="#94a3b8", alpha=.2, label="Recorded screen: 0.75–1.25")
    _axes[1].bar(["H", "η", "ρ"], _width, color=plots.SERIES[1])
    _axes[1].axhline(1, ls=":", color="#64748b")
    _axes[1].set(ylabel="Neural 90% width / reference 90% width", title="Spread disagreement")
    _axes[1].legend(fontsize=8)
    mo.vstack([
        _fig,
        mo.md("These are the original comparison screens for the current checkpoint and the fixed "
              "50,000-market pricer. They have not been retuned after inspecting these cases. "
              "The third market disagrees for H and η; the first two meet the screens."),
        mo.accordion({"Interpret the discrepancy": mo.md("Check MCMC convergence, pricer accuracy in the relevant "
            "parameter region, and the posterior estimator on further conditional cases. "
            "A pricing RMSE averaged over markets and 63 contracts cannot settle these questions. "
            "The reference still omits emulator uncertainty and grid/label bias, so disagreement "
            "does not by itself identify which approximation is responsible. More examples do not "
            "turn three cases into a population-wide validation." )}),
        mo.md("**Explain:** what can a passed comparison tell you, and what would you inspect before trusting a failed one?"),
    ])
    return


@app.cell(hide_code=True)
def _(additional_study, mo):
    mo.stop(additional_study.value != "tail")
    _intro = mo.md(r"""
    ### 4. Explore what the quoted maturities cannot identify

    **Question.** The latest option quote expires after one year. What identifies
    an extra curve parameter that acts only later?

    Keep the fitted curve unchanged through one year. Beyond that, multiply its
    continuation by $\exp[\alpha(t-1)]$, with a new independent prior
    $\alpha\sim\mathcal N(0,\tau^2)$. At $\tau=0$ we recover the original curve.

    **Predict first:** will the original quotes narrow this new prior?
    """)
    tail_width = mo.ui.slider(0, .5, step=.05, value=.20,
        label="Prior SD of the annual log slope after one year", show_value=True)
    mo.vstack([_intro, tail_width])
    return (tail_width,)


@app.cell(hide_code=True)
def _(additional_study, mo, np, studies, study_arrays, study_task, tail_width):
    mo.stop(additional_study.value != "tail")
    _draws = study_arrays["neural_draws"]
    _result = studies.tail_functionals(study_task, _draws, tail_width.value)
    _rows = []
    for _label, _name in [("0–3-year mean variance", "variance_swap"), ("1–3-year forward mean variance", "forward_variance")]:
        for _kind, _values in [("Original curve", _result["base_" + _name]), ("Additional tail prior", _result[_name])]:
            _q = np.quantile(_values, [.05, .5, .95])
            _rows.append({"quantity": _label, "assumption": _kind,
                          "median": f"{_q[1]:.5f}", "central 90%": f"{_q[0]:.5f}–{_q[2]:.5f}"})
    mo.vstack([
        studies.tail_figure(study_task, _draws, tail_width.value),
        mo.ui.table(_rows, selection=None, pagination=False),
        mo.md("Changing this control leaves every original quote likelihood unchanged. "
              "The new tail parameter therefore remains **prior-driven**. Changes in the ranges reflect "
              "the declared assumption, not information learned by the network. "
              "The plotted quantities are expected variance rates, not future realized variance, "
              "a continuation forecast or an exotic-product price."),
        mo.accordion({"Why the tail prior stays unchanged": mo.md(r"""
        With $\boldsymbol\vartheta$ the original parameter vector,

        $$\xi_{\boldsymbol\vartheta,\alpha}(t)=
        \begin{cases}\xi_{\boldsymbol\vartheta}(t),&t\le1,\\
        \xi_{\boldsymbol\vartheta}(t)e^{\alpha(t-1)},&t>1.\end{cases}$$

        The prior for $\alpha$ is independent, and all quotes expire by one year.
        Their likelihood is independent of $\alpha$, giving
        $p(\boldsymbol\vartheta,\alpha\mid\mathbf y)=
        p(\boldsymbol\vartheta\mid\mathbf y)p(\alpha)$.
        We use the existing approximate $q(\boldsymbol\vartheta\mid\mathbf y)$
        for the first factor and fresh independent normal draws for the second.
        The same random numbers are reused as the width changes.
        New long-dated observations would require a compatible new inference task.
        """)}),
        mo.md("**Explain:** when does a narrow credible interval say more about a restrictive model family than about the data?"),
    ])
    return


if __name__ == "__main__":
    app.run()
