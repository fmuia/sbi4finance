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

    from sbi4f_tutorial.rough_options import lesson as teaching, context_checks, studies, explorer, surrogate, comparison, data_story
    from sbi4f_tutorial.shared import backbones, full_surface, plots, tasks as option_tasks

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

    return (ASSETS, RESULTS, backbones, comparison, context_checks, data_story, explorer, focused_posterior,
            full_surface, mo, np, option_tasks, plots, plt, studies, surrogate, teaching, time)


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

    **Explore the paths → build a network → train a small example or load
    pretrained weights → infer and check.** Saved results are available without
    downloading a model or running training.
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
def _(mo):
    mo.accordion({
        "Read the simulator one component at a time": mo.md(r"""
        **1 · Draw the random increments.** Over a step Δt, draw independent
        standard normal numbers Z and Z⊥ and set ΔW = √Δt Z and
        ΔW⊥ = √Δt Z⊥. They represent the unpredictable part of the model's
        evolution, not a particular news event. The asset's Brownian increment
        is ΔB = ρ ΔW + √(1−ρ²) ΔW⊥. Negative ρ links an upward volatility-driver
        increment to a downward price increment.

        **2 · Filter past increments into Y.**
        $Y_t=\sqrt{2H}\int_0^t(t-s)^{H-1/2}dW_s$ weights recent increments
        most strongly. The same past random numbers receive different weights
        when H changes. The Gaussian driver has mean zero and variance t^(2H).
        It is not a variance or a return. H below 1/2 allows rougher short-scale
        movements than a Brownian variance driver.

        **3 · Turn Y into positive variance.** Write
        $V_t=\xi_0(t)M_t$ with
        $M_t=\exp(\eta Y_t-\tfrac12\eta^2\operatorname{Var}(Y_t))$.
        Because Y is Gaussian, E[Mₜ] = 1. The negative term compensates the
        exponential's upward mean shift; it is not mean reversion. Thus ξ₀
        sets mean variance and η controls random departures from it.

        **4 · Use variance to generate returns, then prices.** In this zero-carry
        example, $d\log S_t=-V_tdt/2+\sqrt{V_t}\,dB_t$. The finite-step scheme uses
        the variance at the start of the interval. The resulting log return rᵢ
        gives a relative price change ΔSᵢ/Sᵢ₋₁ = exp(rᵢ)−1, and accumulating log
        returns gives the price path. These daily increments approximate the
        continuous dynamics; they are not the instantaneous differential dS/S.

        **Follow the controls:** η changes V and returns, not Y; ρ changes the
        price increment, not Y or V; ξ₀ rescales variance. H changes the filtering
        and short-time scale of Y. These distinctions are visible in the
        component view below.
        """),
        "What exactly is the forward-variance curve ξ₀?": mo.md(r"""
        $$\xi_0(t)=\mathbb E^{\mathbb Q}[V_t\mid\mathcal F_0].$$

        **Today’s mean variance for a future instant.** It is a deterministic
        curve at calibration time, not the future realized variance path.
        ξ₀(t) = 0.04 means an annual variance rate of 0.04. Its square root is a
        20% volatility scale: √E[Vₜ], which generally differs from E[√Vₜ].
        It is also different from an option's Black–Scholes implied volatility.

        The average curve determines a continuously monitored variance-swap
        strike in this diffusion model:
        $K_{\rm var}(T)=T^{-1}\int_0^T\xi_0(t)dt$.
        An instantaneous forward rate and its average to expiry are different
        quantities. The swap's future realized payment is still random.

        The first task represents ξ₀ by **seven positive interval levels**, one
        between each pair of maturity knots. This piecewise-constant numerical
        representation explains the steps in its plot; the steps are not bins
        counting observations. Treating these levels as supplied is an explicit
        conditioning assumption. In the joint extension, the curve coefficients
        are unknown and are inferred together with H, η and ρ.
        """),
        "Grid approximation and the extra Gaussian channel": mo.md(r"""
        The hybrid scheme handles the newest, singular kernel interval with an
        additional independent Gaussian residual; older intervals form a
        convolution. This residual is a numerical component of the same rough
        driver, not a third market shock. The code keeps three independent
        normal channels: W, this residual, and W⊥.

        On the grid, the compensator uses the scheme's own Var(Yᵢ), rather than
        replacing it by the continuum formula tᵢ^(2H). That preserves E[Vᵢ] = ξ₀(tᵢ)
        for the Gaussian grid model. A small ensemble need not have sample mean
        exactly equal to ξ₀. This correction does not remove log-Euler time bias.
        """),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    path_h = mo.ui.slider(.02, .40, step=.01, value=.10, label="Path roughness H", show_value=True)
    path_eta = mo.ui.slider(.2, 3., step=.1, value=1.5, label="Path amplitude η", show_value=True)
    path_rho = mo.ui.slider(-.95, -.05, step=.05, value=-.70, label="Path correlation ρ", show_value=True)
    path_vol = mo.ui.slider(.10, .40, step=.01, value=.20, label="√ξ₀ (annualized)", show_value=True)
    path_seed = mo.ui.dropdown(options={"Realization A": 7, "Realization B": 17, "Realization C": 27},
                               value="Realization A", label="Simulated realization")
    path_view = mo.ui.radio(options={"Price and volatility": "paths", "Decompose the model": "components"},
                            value="Decompose the model", inline=True, label="Simulator view")
    path_scale = mo.ui.radio(options={"Standard ranges": "detail", "Wider ranges": "wide"},
                             value="Standard ranges", inline=True, label="Fixed display ranges")
    mo.vstack([
        mo.md("### Try the simulator\n\nPredict what changes when you increase η, then move the control. "
              "Keep the realization fixed to compare the same random numbers under different parameters. "
              "A new realization draws a different sample from the same normal distributions; it does not change the model."),
        mo.hstack([path_h, path_eta, path_rho], wrap=True),
        mo.hstack([path_vol, path_seed], wrap=True),
        path_view, path_scale,
    ])
    return path_eta, path_h, path_rho, path_scale, path_seed, path_view, path_vol


@app.cell(hide_code=True)
def _(explorer, mo, path_eta, path_h, path_rho, path_scale, path_seed, path_view, path_vol, simulate_model):
    path_example = simulate_model(
        hurst=path_h.value, eta=path_eta.value, rho=path_rho.value,
        volatility=path_vol.value, seed=path_seed.value)
    _figure = (explorer.component_figure(path_example, preset=path_scale.value)
               if path_view.value == "components" else explorer.path_figure(path_example, preset=path_scale.value))
    _ranges = explorer.axis_statistics(path_example, preset=path_scale.value, view=path_view.value)
    mo.vstack([
        _figure,
        mo.md("**One model, many possible histories.** The same path is highlighted across panels. "
              "The dashed reference is the supplied mean variance ξ₀ in the component view, "
              "or its square root in the volatility view. It is not a realized path or mean volatility. "
              "These are pricing-measure illustrations; option data use expected payoffs over many paths."),
        mo.md(f"**Axes stay fixed when parameters or the realization change.** "
              f"{_ranges['points_outside']:,} plotted coordinates are outside these display ranges; "
              "boundary markers show where paths leave the view. All simulated values are retained. "
              "Select wider ranges explicitly to inspect more of the tails."),
        mo.accordion({"What does this simulation approximate?": mo.md(
            "The illustration uses a flat supplied variance curve, one year and a finite hybrid grid. "
            "The discrete Gaussian variance replaces t^(2H) in the exponential compensator so "
            "the grid model retains E[Vₜ] = ξ₀(t). Twenty-four paths illustrate randomness; "
            "they do not establish Monte Carlo pricing accuracy. No latent paths enter the estimator.")}),
    ])
    return (path_example,)


@app.cell(hide_code=True)
def _(explorer, mo):
    def simulate_model(hurst=.10, eta=1.5, rho=-.70, volatility=.20, seed=7):
        # Fixed shocks make parameter changes easier to compare; these are Q paths,
        # not training observations. Option data require expected payoffs plus noise.
        return explorer.explore_paths(hurst=hurst, eta=eta, rho=rho,
                                      volatility=volatility, seed=seed)

    mo.show_code()
    return (simulate_model,)


@app.cell(hide_code=True)
def _(RESULTS, comparison, mo, path_h, teaching):
    _spy = teaching.load_json(RESULTS / "spy" / "surface.json")
    mo.vstack([
        mo.md(r"""
        ### Why move beyond Heston?

        **The extra freedom is in how volatility behaves across time scales.**
        In Heston, a current variance value summarizes its variance dynamics.
        Rough Bergomi weights the history of shocks with a power kernel;
        H controls their short-scale roughness separately from the variance level.

        The option-market signature is the **short-maturity skew**: the slope of
        implied volatility against log strike near the forward. Equity-index
        downside protection often becomes relatively more expensive as expiry
        shortens. We want a model that can reproduce that pattern across maturities,
        rather than matching just one smile.

        **Try it:** lower H in the path controls. Both shape guides equal one at
        one month. What happens to the one-week skew relative to one month?
        """),
        comparison.skew_shape_figure(path_h.value, _spy),
        mo.md("Dots summarize the interpolated SPY option grid from **21 October 2026**. "
              "The lines show normalized short-maturity shapes, not calibrated prices or a fit "
              "to those dots. A finite-maturity Heston skew can vary; the constant guide describes "
              "its limiting behavior near expiry. The pricing comparison in Section 2 evaluates "
              "both models at finite maturities."),
        mo.accordion({"The short-maturity relationship": mo.md(r"""
        Write $k=\log(K/F)$ and
        $\psi(T)=\partial_k\sigma_{\rm imp}(k,T)|_{k=0}$.
        With positive initial variance and nonzero leverage, standard Heston has
        a finite skew limit as $T\downarrow0$. For a regular, locally flat supplied
        curve, rough Bergomi's leading behavior is
        $|\psi(T)|\propto T^{H-1/2}$. Negative correlation gives a negative skew.

        H changes this rate; η and ρ also change its amplitude. The leading
        formula is not an exact relationship over the entire one-week to one-year
        grid, so estimating H by adding 1/2 to a fitted exponent is only a heuristic.
        This comparison shows modelling flexibility, not that either particular
        fit is accurate or that roughness is the only possible explanation.

        See [El Euch et al., *Short-term at-the-money asymptotics under stochastic volatility models*](https://arxiv.org/abs/1801.08675)
        for the asymptotics, [Bayer, Friz and Gatheral](https://doi.org/10.1080/14697688.2015.1099717)
        for rough option pricing, and [Gatheral, Jaisson and Rosenbaum](https://arxiv.org/abs/1410.3394)
        for the volatility evidence.
        """)}),
    ])
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
    mo.accordion({"A refresher: what does the pricing measure Q mean?": mo.md(r"""
    **Q supplies pricing probabilities, not a forecast of real-world returns.**
    Under the model's no-arbitrage pricing assumptions, an option value is its
    discounted expected payoff under Q. With rates r and dividend yield q, the
    asset drift under Q is r−q. Here r = q = 0 and the forward and initial spot
    are both one, so prices are simply Q expectations of the normalized payoffs.

    A zero drift in dS/S does not mean that each path stays flat. The Brownian
    increments and stochastic variance still produce random paths. The −V/2
    term belongs to log returns through Itô's formula.

    The **physical measure P** describes real-world time-series probabilities.
    That is the trajectory lesson's setting, where μ is inferred from returns.
    Its parameters need not equal their Q counterparts: volatility risk premia
    can matter. Fitting a Q model is not by itself a P return forecast.
    """)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build one synthetic training example

    **Parameters and a curve → random paths → expected payoff estimates → noisy quotes.**
    A training market starts with a draw from the parameter and supplied-curve
    priors. The generating parameters will be its labels. Select one such draw,
    an expiry and a path budget below. The figure makes the pricing and quote
    steps explicit; it is a small local simulation, not a training-bank build.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    dataset_market = mo.ui.dropdown(
        options={"Prior draw A": 2026, "Prior draw B": 2027, "Prior draw C": 2028},
        value="Prior draw A", label="Generating parameters and curve")
    dataset_paths = mo.ui.dropdown(options={"256 paths": 256, "1,024 paths": 1024, "4,096 paths": 4096},
                                   value="1,024 paths", label="Monte Carlo path budget")
    dataset_expiry = mo.ui.dropdown(options={"One week": 5/252, "One month": 21/252,
                                            "Three months": .25, "One year": 1.},
                                    value="Three months", label="Synthetic quote expiry")
    dataset_noise = mo.ui.slider(0., 4., step=.25, value=1., show_value=True,
                                 label="Quote-error scale (1 = declared task SD)")
    mo.vstack([mo.hstack([dataset_market, dataset_paths], wrap=True),
               mo.hstack([dataset_expiry, dataset_noise], wrap=True)])
    return dataset_expiry, dataset_market, dataset_noise, dataset_paths


@app.cell(hide_code=True)
def _(data_story, mo, np, option_tasks):
    def simulate_training_market(task, seed, paths, maturity, noise_scale):
        # Separate random streams for parameters, curve, paths and quote errors.
        streams = np.random.SeedSequence(seed).spawn(4)
        bounds = np.asarray(task["prior_bounds"])
        theta = np.random.default_rng(streams[0]).uniform(bounds[:, 0], bounds[:, 1])
        curve = option_tasks.draw_context(task, np.random.default_rng(streams[1]))[0]
        # One parameter row + one curve generate all nine contracts at this expiry.
        result = data_story.simulate_dataset(
            task, theta, context_values=curve, paths=paths, maturity=maturity,
            seed=int(streams[2].generate_state(1)[0]),
            noise_seed=int(streams[3].generate_state(1)[0]), noise_scale=noise_scale)
        return result, curve

    mo.show_code()
    return (simulate_training_market,)


@app.cell(hide_code=True)
def _(data_story, dataset_expiry, dataset_market, dataset_noise, dataset_paths, known_task, mo, simulate_training_market):
    try:
        dataset_example, dataset_curve = simulate_training_market(
            known_task, dataset_market.value, dataset_paths.value, dataset_expiry.value, dataset_noise.value)
    except (ValueError, RuntimeError, FloatingPointError) as _error:
        mo.stop(True, mo.callout(mo.md(f"**This small simulation stopped:** {_error}. No paths are silently replaced."), kind="warn"))
    _i = 4  # the at-the-money put, for a concrete arithmetic example
    _p, _sd = dataset_example["mean_prices"][_i], dataset_example["quote_noise_std"][_i]
    _error = dataset_example["quote_errors"][_i]
    _observed = dataset_example["observations"][_i]
    mo.vstack([
        mo.md("Generating parameters: " + ", ".join(
            f"**{_name} = {_value:.3g}**" for _name, _value in zip(
                known_task["parameter_names"], dataset_example["parameters"], strict=True))),
        data_story.dataset_figure(dataset_example),
        mo.md(f"**One quote, in numbers:** estimated ATM put premium **{_p:.5f}** "
              f"+ an independent quote error **{_error:+.5f}** = observed premium **{_observed:.5f}**. "
              f"Its assumed quote SD is **{_sd:.5f}**. The same curve and parameters generate "
              "all nine strikes; repeat at seven expiries for a 63-quote market."),
        mo.md("**Two different randomizations.** Increasing the path budget reduces uncertainty "
              "in an expected-price estimate. Changing quote-error scale changes the assumed "
              "observation tolerance. It does not improve the numerical price. The saved posterior "
              "later in the lesson still belongs to its own supplied observation."),
        mo.accordion({"How the numerical averaging works": mo.md(
            "Panel 1 shows raw realized put payoffs from simulated terminal prices. For the "
            "mean-price estimates in panels 2 and 3, the independent asset-price shock is "
            "integrated out analytically before averaging bounded conditional put values. "
            "Calls use put-call parity. These estimate the same Q expectations with less noise. "
            "The price bars are ±2 empirical Monte Carlo SEs; the quote bars are one assumed quote SD. "
            "The full bank additionally uses antithetic pairs, control variates and precision gates. "
            "This fixed-budget example does not certify the bank's accuracy threshold or time-grid bias.")}),
    ])
    return dataset_curve, dataset_example


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compare option smiles with Heston

    We now price both models on the same 63-contract grid, with a flat expected
    variance curve and the same correlation. Heston keeps κ = 2 and
    variance-shock amplitude σᵥ = 0.4. These are illustrative parameters, **not
    calibrations to SPY**. The rough simulator uses a fixed path budget.

    Use the **H, η, ρ and volatility-scale controls above**, predict which part of
    the smile changes, then run. Changing the expiry afterward only redraws the plot.
    """)
    return


@app.cell(hide_code=True)
def _(mo, path_eta, path_h, path_rho, path_seed, path_vol):
    run_rough = mo.ui.run_button(label="Compare rough Bergomi and Heston smiles")
    mo.vstack([
        mo.md(f"Selected rough model: **H = {path_h.value:.2f}, η = {path_eta.value:.1f}, "
              f"ρ = {path_rho.value:.2f}, √ξ₀ = {100*path_vol.value:.0f}%, seed = {path_seed.value}**."),
        run_rough,
    ])
    return (run_rough,)


@app.cell(hide_code=True)
def _(comparison, mo, path_eta, path_h, path_rho, path_seed, path_vol, run_rough, teaching):
    mo.stop(not run_rough.value, mo.md("*Press the button to compare Fourier prices with an 8,192-path rough-model illustration.*"))
    try:
        _rough = teaching.price_rough_smile(path_h.value, path_eta.value, path_rho.value,
                                            vol=path_vol.value, seed=path_seed.value)
        smile_comparison = comparison.compare_smiles(
            _rough, volatility=path_vol.value, sigma_v=.4, rho=path_rho.value)
    except (ValueError, RuntimeError) as _error:
        mo.stop(True, mo.callout(mo.md(f"**Pricing comparison stopped:** {_error}. Change the controls and try again."), kind="warn"))
    return (smile_comparison,)


@app.cell(hide_code=True)
def _(mo):
    smile_expiry = mo.ui.dropdown(
        options={"One week": 5, "One month": 21, "Three months": 63, "One year": 252},
        value="One week", label="Expiry for the smile comparison")
    smile_expiry
    return (smile_expiry,)


@app.cell(hide_code=True)
def _(comparison, mo, path_scale, smile_comparison, smile_expiry):
    mo.vstack([
        comparison.comparison_figure(smile_comparison, maturity_days=smile_expiry.value, scale=path_scale.value),
        mo.md("Left: δ = log(K/F)/(0.2√T); negative values are downside strikes. "
              "Right: absolute near-the-money skew versus expiry. Each skew comes from a quadratic "
              "fit to the five nearest strikes, rather than an exact derivative at k = 0. "
              "The rough prices contain Monte Carlo error, which can affect these slopes. "
              "This fixed-budget comparison does not apply the training bank's adaptive accuracy policy."),
    ])
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
        labels, the time grid or both neural approximations. The two methods
        learn different objects and need their own accuracy checks.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(explorer, mo):
    mo.vstack([
        mo.md("### What is inside each network?"),
        explorer.architecture_figure(pricer_label="Pretrained forward pricer"),
        mo.md(r"""
        **Pricer: a dense regression network.** Logit parameters and log curve levels
        give 10 inputs; 63 outputs estimate clean prices in quote-noise units.
        Training minimizes squared pricing errors. It is a deterministic forward map.

        **Posterior: a conditional flow-matching network.** The 63 scaled quotes
        and seven log curve levels form a fixed, ordered 70-value condition.
        A time-conditioned residual MLP learns a velocity for a noisy
        three-parameter state at time t. BayesFlow embeds time with Fourier
        features and supplies the market condition to its hidden blocks.
        Integrating that velocity transforms Gaussian draws into joint
        parameter draws; the adapter returns them in physical units.
        No recurrent or set encoder is needed for this fixed option grid.

        The live examples below use **two 32-unit dense layers** for the pricer
        and **two 128-unit residual blocks** for flow matching. The pretrained
        posterior has four 256-unit blocks; the pretrained pricer has four
        256-unit dense layers. These are
        separate fits with different losses and training budgets.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A working pricer-surrogate example

    Train on **clean expected-price estimates**, then compare predictions with
    separate validation markets. The same quote SDs scale the regression error
    and the Gaussian likelihood. No noisy quotes are used as price targets.

    This small fit shows the workflow. Two validation markets select an epoch;
    they are not an independent final accuracy test. We do not run MCMC on this
    tiny fit. The joint extension later loads a longer-fit forward pricer.

    Inside `surrogate.train`, the small pricer is an ordinary Keras network:

    ```python
    import keras

    # Three parameter logits + seven log curve levels enter the dense network.
    model = keras.Sequential([
        keras.Input((10,)),
        keras.layers.Dense(32, activation="gelu"),
        keras.layers.Dense(32, activation="gelu"),
        keras.layers.Dense(63),  # one standardized clean-price output per contract
    ])
    # The helper standardizes inputs/outputs and fits squared errors in quote-SD units.
    ```

    This network predicts prices; the BayesFlow flow below generates parameters.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, surrogate, teaching, time):
    def train_small_pricer(task):
        started = time.perf_counter()
        # Simulate clean mean prices once; the noisy observations are unused here.
        bank, generation = teaching.generate_training_pairs(task, count=8, seed=2026, max_seconds=120)
        # Validation selects an epoch. It is separate from these six fitting rows.
        train = {key: values[:6] for key, values in bank.items()}
        validation = {key: values[6:] for key, values in bank.items()}
        # Forward regression: parameter logits + log curve -> 63 scaled prices.
        pricer, fit = surrogate.train(task, train, validation, widths=(32, 32),
                                      epochs=5, batch_size=4, patience=5, seed=2026)
        # Keep the held-out rows for the physical-price comparison below.
        return pricer, validation, fit, {"generation": generation["runtime_seconds"],
                                      "total": time.perf_counter() - started}

    def surrogate_log_likelihood(pricer, task, parameters, prices, curve):
        from sbi4f_tutorial.shared.tasks import sigma

        # Supply candidate physical parameters and the observed curve, not quotes.
        inputs = surrogate.features(task, np.atleast_2d(parameters), np.atleast_2d(curve))
        mean_over_sd = pricer(inputs)
        # Independent Gaussian quote errors: log L = -sum((prediction-y)^2 / SD^2)/2.
        residual = mean_over_sd - prices / sigma(task)
        return -.5 * np.sum(residual**2, axis=-1)  # constant terms omitted

    mo.show_code()
    return surrogate_log_likelihood, train_small_pricer


@app.cell(hide_code=True)
def _(mo):
    train_pricer = mo.ui.run_button(label="Generate fresh examples and train a small pricer")
    mo.vstack([train_pricer, mo.md("Eight fresh markets: six for fitting, two for validation. "
                                  "Five epochs; results stay in memory. Pricing may take a minute.")])
    return (train_pricer,)


@app.cell(hide_code=True)
def _(known_task, mo, train_pricer, train_small_pricer):
    mo.stop(not train_pricer.value, mo.md("*Train the small forward network to inspect its predicted prices below.*"))
    try:
        session_pricer, session_pricer_validation, session_pricer_fit, session_pricer_timing = train_small_pricer(known_task)
    except (RuntimeError, ValueError) as _error:
        mo.stop(True, mo.callout(mo.md(f"**Pricer fit stopped:** {_error}. No partial bank is used."), kind="warn"))
    mo.md("**The small pricer has now fitted six markets.** The next figure compares its predictions "
          "with numerical expected-price labels on two different markets. It is a regression check, "
          "not a posterior or a calibration of a real asset.")
    return session_pricer, session_pricer_fit, session_pricer_timing, session_pricer_validation


@app.cell(hide_code=True)
def _(mo):
    pricer_market = mo.ui.dropdown(options={"Validation market A": 0, "Validation market B": 1},
                                   value="Validation market A", label="Market for the pricer check")
    pricer_expiry = mo.ui.dropdown(options={"One week": 5, "One month": 21, "Three months": 63, "One year": 252},
                                   value="Three months", label="Expiry for the pricer check")
    pricer_range = mo.ui.radio(options={"Standard ranges": "standard", "Wider ranges": "wide"},
                               value="Standard ranges", inline=True, label="Pricer-check display ranges")
    mo.vstack([mo.hstack([pricer_market, pricer_expiry], wrap=True), pricer_range])
    return pricer_expiry, pricer_market, pricer_range


@app.cell(hide_code=True)
def _(data_story, known_task, mo, np, pricer_expiry, pricer_market, pricer_range,
      session_pricer, session_pricer_fit, session_pricer_timing, session_pricer_validation, surrogate):
    _wide = pricer_range.value == "wide"
    _figure = data_story.pricer_validation_figure(
        session_pricer, known_task, session_pricer_validation,
        market=pricer_market.value, maturity_days=pricer_expiry.value,
        price_limits=(-.05, .30) if _wide else (-.02, .12),
        residual_limits=(-200, 200) if _wide else (-80, 80))
    _errors = surrogate.errors(session_pricer, known_task, session_pricer_validation)
    _rmse = float(np.sqrt(np.mean(_errors**2)))
    mo.vstack([
        _figure,
        mo.md(r"""
        **What are we checking?** At one set of validation parameters and its
        supplied curve, the expensive Monte Carlo calculation estimates nine
        expected prices (black). The small neural pricer tries to reproduce
        those same prices (purple). The left bars describe Monte Carlo uncertainty
        in the numerical labels; no quote noise is added to these labels.

        The right panel plots **(network prediction − numerical mean price) / quote SD**.
        A positive value means that the network overprices that contract.
        A value of 10 means an error ten times that contract's assumed observation
        noise. The shaded ±1 band supplies a scale, not an acceptance certificate.
        Axes stay fixed while you select markets and expiries.

        **The six-market fit is intentionally too small to be a useful pricer.**
        Large residuals show what has not been learned across the parameter space;
        they are not posterior uncertainty. A likelihood built from these inaccurate
        prices would be misleading. Before using a pricer for inference, expand its
        training data and check accuracy on genuinely independent cases.
        These two validation markets selected an epoch, so they are not a final test.
        """),
        mo.md(f"Overall validation RMSE across both 63-price surfaces: **{_rmse:.2f} quote SDs**. "
              f"Selected epoch: **{session_pricer_fit['selected_epoch']}**. "
              f"Generation: **{session_pricer_timing['generation']:.1f} s**; "
              f"total: **{session_pricer_timing['total']:.1f} s**. "
              "Changing the display reuses this fit; it does not train again. "
              "The likelihood code above explains the next step, but this tiny pricer is not used for MCMC."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Train a tutorial network or load pretrained weights

    The BayesFlow workflow is the same as in Heston: prepare the data, define an
    adapter and conditional network, train, then sample. The extra condition is
    the supplied curve. We divide prices by their quote SDs and take logs of the
    positive curve levels before standardizing the network inputs.

    **What changes from the Heston code?**

    | BayesFlow step | Heston live example | Rough Bergomi live example |
    |:---|:---|:---|
    | Training pairs | Fourier prices from a callable simulator | reuse a small Monte Carlo price bank |
    | Adapter targets | four bounded parameters | three bounded parameters; same constraint pattern |
    | Adapter conditions | 63 quotes, renamed | divide 63 prices by their quote SDs; append logs of seven supplied variance levels |
    | Inference network | `FlowMatching`, widths `[128, 128]` | **the same** |
    | Workflow | `BasicWorkflow`, standardize targets and conditions | **the same** |
    | Fit | `fit_online` generates new batches | `fit_offline` reuses the simulated pairs |
    | Infer | `sample(conditions=..., num_samples=...)` | **the same conditional-sampling pattern** |

    The supplied curve is an observed condition, not a target. In the joint-curve
    extension, its coefficients become targets and the curve is no longer supplied.

    | Route | What happens | What to use it for |
    |:---|:---|:---|
    | **Tutorial: train live** | generate eight markets and fit a small network | learn the API |
    | **Pretrained: load and infer** | load matching weights and draw for our worked example | reuse the longer fit |
    | **Saved results** | display bundled draws, no model needed | follow the lesson immediately |

    ### Tutorial version: train live

    The code below runs only when requested. It generates a **small fresh training
    set**, then fits a separate demonstration network. The selected cached
    observation is used only for inference. Eight markets are only enough to exercise the API; they do not give a
    useful calibrated estimator or reproduce the pretrained result.
    """)
    return


@app.cell(hide_code=True)
def _(known_arrays, known_task, mo, teaching):
    mo.vstack([
        mo.md(r"""
        ### What exactly goes into the network?

        **One market becomes one row of 70 numbers.** First divide each of its
        63 prices by that contract's quote-error SD: y₁/σ₁, …, y₆₃/σ₆₃.
        This expresses prices relative to their assumed observation-noise scale.
        Separately take the natural log of each positive supplied curve level.
        The seven logs describe one curve; they are not seven additional markets.

        **Concatenate means append**, not add: place the seven log levels after
        the 63 scaled quotes. We log the positive curve levels, not the option
        prices, which can be negative under the Gaussian observation model.
        BayesFlow then standardizes these features using statistics learned from
        the training examples. New markets use those same fitted statistics.
        """),
        teaching.conditioning_figure(known_task, known_arrays["observations"], known_arrays["context_values"]),
        mo.md("The generating H, η and ρ are separate training labels. They never appear "
              "in this condition vector. In the joint-curve extension, there are six labels "
              "and only the 63 quote features: no generating curve is supplied."),
    ])
    return


@app.cell(hide_code=True)
def _(full_surface, mo, teaching, time):
    def train_small_estimator(task, observed_prices, supplied_curve):
        started = time.perf_counter()
        import bayesflow as bf

        # Different from Heston: price a small bank first, then reuse it offline.
        split, generation = teaching.generate_training_pairs(task, count=8, seed=2026, max_seconds=90)
        # Scale quotes by their SDs; log the seven supplied positive curve levels.
        data = full_surface.training_data(split, task)

        # Same as Heston: map bounded parameters to unbounded network coordinates.
        adapter = bf.Adapter()
        for name, (lower, upper) in zip(task["parameter_names"], task["prior_bounds"], strict=True):
            adapter = adapter.constrain(name, lower=lower, upper=upper, inclusive="none")
        adapter = (adapter.convert_dtype("float64", "float32")
                   .concatenate(task["parameter_names"], into="inference_variables")
                   # Different: concatenate the curve as well as the quotes.
                   .concatenate(["observations", "curve_context"], into="inference_conditions"))
        # Same as Heston: a conditional flow with the same small TimeMLP widths.
        network = bf.networks.FlowMatching(subnet_kwargs={"widths": [128, 128]})
        workflow = bf.BasicWorkflow(adapter=adapter, inference_network=network,
                                   standardize=["inference_variables", "inference_conditions"])
        fit_started = time.perf_counter()
        # Different: fit_offline reuses prices rather than repricing every batch.
        workflow.fit_offline(data=data, epochs=5, batch_size=4)
        fit_seconds = time.perf_counter() - fit_started

        # Same inference pattern: condition and sample. No generating truth enters.
        # This helper applies exactly the same scaling/logs as training_data.
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
    ### Pretrained version: load and infer

    Download the **rough-known100k-final500** bundle using the
    [model guide](https://github.com/fmuia/sbi4finance/blob/main/lab/plans/training-data-guide.md).
    Keep `model.keras`, `task.json` and `training.json` together under
    `tutorial/assets/local-checkpoints/pretrained/rough-known100k-final500/`.

    This loads the longer fit and draws for the **same worked-example quotes
    and supplied curve**. It does not retrain. The loader checks model and task
    identities before inference. Loading weights does not remove the accuracy
    limitations discussed below.
    """)
    return


@app.cell(hide_code=True)
def _(backbones, full_surface, mo, time):
    def infer_with_pretrained(directory, prices, curve):
        started = time.perf_counter()
        # Rebuild the recorded workflow and verify task/weights; skip training.
        task, workflow, record = backbones.load_live("rough-known100k-validation0", directory)
        loaded = time.perf_counter()
        # Same observed inputs as the tutorial fit; return complete physical rows.
        draws = full_surface.sample(workflow, task, prices[None, :], num_samples=512,
                                    context_values=curve[None, :], seed=17)[0]
        return task, draws, {"load": loaded - started, "sampling": time.perf_counter() - loaded}

    mo.show_code()
    return (infer_with_pretrained,)


@app.cell(hide_code=True)
def _(ASSETS, mo):
    pretrained_folder = mo.ui.text(
        value=str(ASSETS / "local-checkpoints" / "pretrained" / "rough-known100k-final500"),
        placeholder="Folder containing model.keras",
        label="Matching pretrained posterior folder", full_width=True)
    pretrained_folder
    return (pretrained_folder,)


@app.cell(hide_code=True)
def _(mo, pretrained_folder):
    _directory = pretrained_folder.value.strip()
    load_pretrained = mo.ui.run_button(label="Load pretrained weights and draw this posterior")
    mo.vstack([load_pretrained, mo.md("Optional. The rest of the lesson uses saved draws and needs no download.")])
    return (load_pretrained,)


@app.cell(hide_code=True)
def _(focused_posterior, infer_with_pretrained, known_arrays, load_pretrained, mo, pretrained_folder):
    mo.stop(not load_pretrained.value, mo.md("*Download the matching bundle, then load it here to run inference yourself.*"))
    try:
        _task, _draws, _timing = infer_with_pretrained(
            pretrained_folder.value.strip(), known_arrays["observations"], known_arrays["context_values"])
    except (OSError, ValueError, RuntimeError) as _error:
        mo.stop(True, mo.callout(mo.md(f"**Model not loaded:** {_error}. "
                                      "Check the folder and matching bundle in the model guide."), kind="warn"))
    _case = dict(known_arrays, neural_draws=_draws)
    mo.vstack([
        mo.md(f"**Fresh inference from pretrained weights · 512 paired draws.** "
              f"Load including imports: **{_timing['load']:.1f} s**; sampling including compilation: "
              f"**{_timing['sampling']:.1f} s**. The saved checks below retain their original draws."),
        focused_posterior(_task, _case),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Infer parameters with the curve supplied

    We now inspect saved draws from the pretrained estimator on one synthetic option market.
    The seven curve levels are known inputs; only $H,\eta,\rho$ are inferred.
    Their priors are independent uniforms on the ranges in the table. The plots show paired posterior draws in the original parameter units.
    These are approximate draws. The predictive and calibration checks below
    help assess their implications and limits.
    """)
    return


@app.cell(hide_code=True)
def _(backbones, focused_posterior, known_arrays, known_task, mo, np, plt):
    _contracts = known_task["context"]["contracts"]
    _t = np.asarray([c["maturity"] for c in _contracts])
    _k = np.asarray([c["strike"] for c in _contracts])
    _noise = np.asarray(known_task["context"]["quote_noise_std"])
    _fig, _axes = plt.subplots(1, 2, figsize=(9.5, 2.9), constrained_layout=True)
    _maturities = np.unique(_t)
    _shades = plt.get_cmap("Greys")(np.linspace(.40, .90, len(_maturities)))
    for _maturity, _shade in zip(_maturities, _shades, strict=True):
        _at = np.flatnonzero(_t == _maturity)
        _at = _at[np.argsort(_k[_at])]
        _axes[0].errorbar(_k[_at], known_arrays["observations"][_at], yerr=_noise[_at],
                          fmt="o-", ms=3, lw=1, color=_shade, label=f"{252 * _maturity:g} days")
    _axes[0].set(xlabel="strike / forward", ylabel="premium / forward", title="The observed option quotes")
    _axes[0].legend(fontsize=6, ncol=2)
    _knots = known_task["model_context"]["forward_variance_knots"]
    # This is a step-function approximation to a curve, not a histogram.
    # Repeat the last level to avoid an artificial closing segment down to zero.
    _levels = known_arrays["context_values"]
    _axes[1].plot(_knots, np.r_[_levels, _levels[-1]], drawstyle="steps-post", color="#7755aa", lw=2)
    _axes[1].plot((np.asarray(_knots[:-1]) + _knots[1:]) / 2, _levels, "o", color="#7755aa", ms=3)
    _axes[1].set(xlabel="maturity (years)", ylabel="annualized forward variance", title="The supplied curve")
    mo.vstack([
        _fig,
        mo.md("Strike / forward is K/F; premium / forward is the option price divided by F. "
              "A premium of 0.02 means 2 when the forward is 100. Puts appear at/below the forward "
              "and calls above it. Lines join contracts at the same expiry; longer expiries are darker. "
              "Bars show one quote-noise SD. The right panel connects the seven supplied interval levels "
              "as a step line, without closing to zero; it is a mean-variance curve, not a frequency histogram."),
        mo.ui.table(backbones.parameter_rows(known_task, known_arrays), selection=None, pagination=False,
                    format_mapping={"median": "{:.4g}".format}),
        focused_posterior(known_task, known_arrays),
        mo.md("The marginals summarize each parameter; the joint panel preserves dependence. "
              "Can you explain which combinations of variance amplitude and correlation remain plausible?"),
        mo.accordion({"All parameter relationships": backbones.corner(known_task, known_arrays)}),
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
        # Keep H, eta and rho paired; reprice, then add independent quote noise.
        # The supplied curve stays fixed for this observation.
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
    joint_task, joint_report, joint_arrays = backbones.load_example(joint_case.value)
    _task, _report, _arrays = joint_task, joint_report, joint_arrays
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
    return joint_arrays, joint_report, joint_task


@app.cell(hide_code=True)
def _(joint_arrays, joint_task, mo, teaching):
    mo.vstack([
        mo.md(r"""
        ### The pricer counterpart when the curve is unknown

        **Yes: both neural routes can infer the curve.** Their inputs exchange roles:

        | Route | Inputs when the curve is unknown | Outputs |
        |:---|:---|:---|
        | Neural pricer | H, η, ρ, S, L, k | 63 expected option prices |
        | Posterior estimator | 63 noisy option quotes | joint draws of H, η, ρ, S, L, k |

        S, L and k determine ξ₀(t) = L + (S−L)exp(−kt) in the generative model.
        The learned pricer maps each complete six-parameter row directly to prices;
        there are **six parameter features and no supplied curve inputs**.
        To do Bayesian inference with it, combine the same Gaussian likelihood
        with the parameter prior and run a sampler for each market. We have a
        trained joint pricer, but no completed matching joint-curve MCMC reference.

        The posterior estimator has **63 market features**, since the seven
        curve levels are no longer known. S, L and k are labels during training
        and unknown outputs during inference.
        """),
        teaching.conditioning_figure(joint_task, joint_arrays["observations"]),
    ])
    return


@app.cell(hide_code=True)
def _(mo, option_tasks, surrogate):
    def predict_joint_prices(pricer, task, parameters):
        # All six entries, including the three curve coefficients, are candidates.
        # No context_values argument: S, L and k already encode the candidate curve.
        features = surrogate.features(task, parameters)
        # The exported predictor returns expected prices divided by quote SDs.
        return pricer(features) * option_tasks.sigma(task)

    def fit_joint_pricer(task, training_pairs, validation_pairs):
        # The same supervised API works with six parameter labels instead of three.
        # It learns clean Monte Carlo mean-price targets, not noisy observations.
        return surrogate.train(task, training_pairs, validation_pairs)

    mo.show_code()
    return fit_joint_pricer, predict_joint_prices


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. From vanilla calibration to another contract

    **Calibrate once; carry the uncertainty into the next price.**
    Suppose the vanilla quotes belong to one underlying. We now want the fair
    strike of a variance swap on that same underlying, which was not included
    among the calibration instruments.

    We use the **joint-curve case above**. With an exactly supplied ξ₀, this
    particular fair strike would be fixed by the curve, even if H, η and ρ
    remained uncertain.

    A least-squares point calibration chooses one parameter vector θ̂ and reports
    one plug-in strike Kvar(θ̂). Bayesian calibration retains plausible parameter
    combinations and maps every complete row through the same pricing formula:

    $$\vartheta^{(m)}\sim q(\vartheta\mid\text{vanilla quotes}),\qquad
      K_{\rm var}^{(m)}(T)=\frac1T\int_0^T\xi_0(t;\vartheta^{(m)})\,dt.$$

    The distribution describes **uncertainty about a fair price**, conditional
    on the chosen model and prior. It is not the distribution of a future
    realized payment. This transfers uncertainty between products on one
    underlying; sharing information between different underlyings would require
    an additional model connecting their parameters.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Optional: add an actual least-squares point fit

    The posterior propagation below works from saved draws. To add a point fit,
    download the pretrained archive in the
    [model guide](https://github.com/fmuia/sbi4finance/blob/main/lab/plans/training-data-guide.md)
    and use its `pricers/rough-joint100k` folder under `assets/local-checkpoints/`.

    The button loads that pricer and minimizes squared quote residuals for the
    **selected joint example**, in quote-SD units. Both routes use the same
    parameter ranges, observation model and underlying simulation bank. The
    point fit uses no truth labels and no posterior draws as starting values.
    It is a local multi-start calculation under a learned price approximation.
    """)
    return


@app.cell(hide_code=True)
def _(mo, studies, surrogate):
    def calibrate_with_joint_pricer(directory, task, observations):
        # Validate the supplied model's task and weights before evaluating it.
        _, pricer, record = surrogate.load_pricer(directory, expected_task=task)
        # A point fit: minimize sum(((predicted price - quote) / SD)**2).
        # Three independent starts, no posterior/prior-density or Jacobian term.
        fit = studies.point_calibration(pricer, task, observations,
                                        starts=3, seed=20261109, max_evaluations=500)
        return pricer, record, fit

    mo.show_code()
    return (calibrate_with_joint_pricer,)


@app.cell(hide_code=True)
def _(mo):
    joint_pricer_folder = mo.ui.text(
        value="local-checkpoints/pricers/rough-joint100k",
        placeholder="Folder containing pricer.npz", full_width=True,
        label="Joint pricer folder relative to tutorial/assets/")
    joint_pricer_folder
    return (joint_pricer_folder,)


@app.cell(hide_code=True)
def _(joint_arrays, joint_pricer_folder, mo):
    # Recreate this action whenever the selected observation or folder changes.
    _source = (joint_arrays["observations"].tobytes(), joint_pricer_folder.value.strip())
    run_point_calibration = mo.ui.run_button(label="Load the joint pricer and fit this market")
    run_point_calibration
    return (run_point_calibration,)


@app.cell(hide_code=True)
def _(calibrate_with_joint_pricer, joint_arrays, joint_pricer_folder, joint_report, joint_task,
      mo, np, predict_joint_prices, run_point_calibration):
    point_calibration = None
    point_calibration_error = None
    if run_point_calibration.value:
        try:
            _pricer, _record, _fit = calibrate_with_joint_pricer(
                joint_pricer_folder.value.strip(), joint_task, joint_arrays["observations"])
            if _record["bank_manifest_sha256"] != joint_report["bank_manifest_sha256"]:
                raise ValueError("the loaded pricer uses a different simulation bank")
            # The displayed prediction uses the same six-input API explained above.
            _prices = predict_joint_prices(_pricer, joint_task, _fit["parameters"])[0]
            if not np.allclose(_prices, _fit["fitted_prices"], rtol=1e-12, atol=1e-14):
                raise ValueError("point-fit predictions disagree with the displayed pricer API")
            point_calibration = _fit
        except (OSError, ValueError, RuntimeError) as _error:
            point_calibration_error = str(_error)
    if point_calibration_error is not None:
        _status = mo.callout(mo.md("**Point fit not available:** " + point_calibration_error
                         + ". The posterior-only propagation below is still available."), kind="warn")
    elif point_calibration is None:
        _status = mo.md("*Saved posterior propagation is shown below. Load a matching joint pricer to add the point-fit line.*")
    else:
        _status = mo.md("**The point-fit comparison is loaded for the selected observation.**")
    _status
    return point_calibration, point_calibration_error


@app.cell(hide_code=True)
def _(joint_arrays, joint_task, mo, np, point_calibration, studies):
    _effect = studies.calibration_effects(joint_task, joint_arrays["neural_draws"], point_calibration)
    _q = np.quantile(_effect["posterior_variance_swap_1y"], [.05, .5, .95])
    _rows = [{"calculation": "Paired approximate posterior", "one-year fair variance strike":
              f"median {_q[1]:.5f}; central 90% {_q[0]:.5f}–{_q[2]:.5f}"}]
    if point_calibration is not None:
        _rows.append({"calculation": "Local least-squares point fit", "one-year fair variance strike":
                      f"{_effect['point_variance_swap_1y']:.5f}; no uncertainty interval supplied"})
    mo.vstack([
        studies.calibration_effect_figure(joint_task, joint_arrays["neural_draws"], point_calibration),
        mo.ui.table(_rows, selection=None, pagination=False),
        mo.md("The same complete posterior rows determine the curve and the swap strike. "
              "Discarding their dependence would give a different distribution. "
              "We changed the contract we price; we did not observe or recalibrate to a swap quote. "
              "A point fit alone gives no interval. Frequentist methods can also quantify uncertainty, "
              "for example by an additional bootstrap calculation."),
        mo.accordion({"What this comparison can establish": mo.md(
            "The interval propagates the supplied approximate neural posterior; it is not an exact-model "
            "accuracy result. The point fit uses a learned pricer, three local starts and a finite "
            "logit search box of [-12,12] within the same parameter ranges. It has no density prior "
            "penalty. Different centers can reflect priors, finite samples, optimization and either "
            "neural approximation; this plot does not prove that one procedure is more accurate. "
            "The curve bands are pointwise 90% ranges, not a simultaneous band or a realized-variance forecast.")}),
    ])
    return


@app.cell(hide_code=True)
def _(known_report, mo):
    mo.md(r"""
    ## 9. What do the checks establish?

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
    The supplied synthetic fits have aggregate diagnostics. These checks do not
    certify the result for every market. Exact rough-model references and further
    joint-curve and Monte Carlo product checks remain unfinished.
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
    | What lies beyond the last quote? | Separate information from an extrapolation assumption. |
    """)
    additional_study = mo.ui.dropdown(options={
        "Can I use this estimator on a new market?": "context",
        "Why must posterior draws stay paired?": "dependence",
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
def _(additional_study, mo):
    mo.stop(additional_study.value != "tail")
    _intro = mo.md(r"""
    ### 3. Explore what the quoted maturities cannot identify

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
