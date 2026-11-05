import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="Heston: from option quotes to a posterior",
)


@app.cell(hide_code=True)
def _():
    import time

    import bayesflow as bf
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from sbi4f_tutorial.heston_options import lesson as teaching
    from sbi4f_tutorial.shared import plots, surface

    ROOT = mo.notebook_dir().parent
    ASSETS = ROOT / "assets"
    RESULTS = ASSETS / "results" / "calibration"
    CHECKPOINTS = ASSETS / "checkpoints"
    CHECKS = ASSETS / "quick-checks"
    plots.style()
    return (
        ASSETS,
        CHECKPOINTS,
        CHECKS,
        RESULTS,
        bf,
        mo,
        np,
        plots,
        plt,
        surface,
        teaching,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bayesian Calibration of the Heston Model

    The **Heston model** is a popular model in mathematical finance that describes how asset prices evolve over time [(Heston, 1993)](https://doi.org/10.1093/rfs/6.2.327). A key feature of the model is that it allows volatility to change randomly, capturing the observation that volatility in markets does not stay constant across time.

    We will use **option prices** to calibrate the model (i.e., estimate its parameters). Options are financial contracts whose prices depend on the future behavior of an underlying asset. By observing option prices across different strike prices and maturities, we can constrain the parameters compatible with the data.

    In this tutorial, our data consist of **63 option prices**, corresponding to nine strikes and seven maturities. Our goal is to infer (or *calibrate*) four parameters of the Heston model from these observations.

    Since we have only a finite number of observations, the parameters cannot be determined with certainty. We therefore take a **Bayesian approach**, estimating a posterior distribution over the model parameters rather than a single best-fitting set of values.

    We will use **amortized Bayesian inference** to learn this posterior from simulated data, allowing us to obtain parameter estimates and quantify their uncertainty for new sets of option prices in near-real time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The Heston Model

    The Heston model describes the evolution of an asset price using **stochastic volatility**, that is, assuming the asset's variance follows a separate stochastic process that can change over time. Importantly, changes in the asset price and its variance can be correlated. A negative correlation allows price declines to be associated with increases in volatility, a pattern frequently observed in financial markets. This flexibility enables the Heston model to reproduce features of option prices, such as the **volatility smile and skew**, that constant-volatility models cannot adequately capture.

    $$
    dS_t = S_t\sqrt{V_t}\,dW_t^S,\qquad
    dV_t = \kappa(\theta-V_t)\,dt+\sigma_v\sqrt{V_t}\,dW_t^V,
    \qquad d\langle W^S,W^V\rangle_t=\rho\,dt.
    $$

    | Symbol | Meaning | In this inference task |
    |:---|:---|:---|
    | $S_t$ | asset price, starting at 1 | a possible future path |
    | $V_t$ | instantaneous **variance**; volatility is $\sqrt{V_t}$ | an unobserved state |
    | $v_0$ | initial variance $V_0$ | infer it |
    | $\theta$ | long-run variance level | infer it |
    | $\sigma_v$ | strength of random variance moves | infer it |
    | $\rho$ | correlation between the two random shocks | infer it |
    | $\kappa$ | speed of mean reversion | fixed at 2 per year |

    These are **pricing-measure dynamics**: they describe the distribution used to value contingent payoffs. We use zero rates and dividends, so no drift appears in this equation. Estimating drift from observed returns is a different task, explored in the trajectory notebook.
    """)
    return


@app.cell(hide_code=True)
def _(heston_task, mo, plt, teaching):
    _paths = teaching.heston_illustrative_paths(heston_task, days=252)
    _fig, _axes = plt.subplots(1, 2, figsize=(9.6, 2.7), constrained_layout=True)
    _axes[0].plot(_paths["times_years"], _paths["prices"].T, lw=1.1)
    _axes[1].plot(_paths["times_years"], 100 * _paths["variances"].T ** .5, lw=1.1)
    _axes[0].set(xlabel="years", ylabel="asset price / initial price", title="Possible asset-price paths")
    _axes[1].set(xlabel="years", ylabel="instantaneous volatility (%)", title="Their hidden volatility paths")
    mo.vstack([
        mo.md("**A model you can simulate.** These paths illustrate the two coupled equations. "
              "An option price averages a payoff over the model's future uncertainty; it is not one simulated path."),
        _fig,
        mo.md("The path illustration uses a time-discretized approximation with fixed random seeds. "
              "The calibration data below use the **Fourier pricer**, which computes option expectations "
              "without these illustrated paths. Neither a realized path nor latent volatility is supplied to the estimator."),
        mo.md("""```python
    parameters = [0.04, 0.04, 0.25, -0.7]  # v0, long-run variance, sigma_v, rho
    paths = teaching.heston_illustrative_paths(heston_task, parameters, days=252, paths=3)
    prices = teaching.heston_smile(heston_task, parameters)  # expected option payoffs
    ```"""),
        mo.md(r"""
        **The model update, in code.**
        The illustration uses signed full-truncation Euler for the variance and
        a log update for the price. At each small step, draw independent
        standard normals $z_1,z_2$ and correlate the second shock:

        ```python
        v = np.maximum(variance_state, 0.0)
        z_variance = z1
        z_price = rho * z_variance + np.sqrt(1.0 - rho**2) * z2
        price *= np.exp(-0.5 * v * dt + np.sqrt(v * dt) * z_price)
        variance_state += kappa * (theta - v) * dt + sigma_v * np.sqrt(v * dt) * z_variance
        ```
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Generating Synthetic Data

    We generate synthetic data by sampling Heston parameters, computing the corresponding option prices, and adding observation noise.
    The four parameters of interest are $\boldsymbol\vartheta=(v_0,\theta,\sigma_v,\rho)$, where $v_0$ is the initial variance, $\theta$ is the long-run variance, $\sigma_v$ controls the volatility of variance, and $\rho$ is the correlation between asset returns and variance changes.

    For a given parameter vector, the theoretical price of a European call option with strike $K$ and maturity $T$ is the expected discounted payoff under the risk-neutral measure $\mathbb Q$:

    $$
    P_{K,T}(\boldsymbol\vartheta)=e^{-rT}\mathbb E^{\mathbb Q}_{\boldsymbol\vartheta}[(S_T-K)_+].
    $$

    We use a Fourier-based pricing method to compute the prices of all 63 options. To simulate observed market quotes, we add Gaussian observation noise:

    $$
    \boldsymbol\vartheta \sim p(\boldsymbol\vartheta),\qquad
    y_c=P_c(\boldsymbol\vartheta) + \sigma_c\epsilon_c,\qquad
    \epsilon_c \overset{\mathrm{iid}}{\sim} \mathcal N(0,1).
    $$

    Here, $p(\boldsymbol\vartheta)$ is the *prior distribution over Heston parameters*, $P_c(\boldsymbol\vartheta)$ is the theoretical price of contract $c$, and $\sigma_c$ specifies the standard deviation of its observation noise.
    Repeating this procedure produces a training dataset of parameter–observation pairs $(\boldsymbol\vartheta_i,\mathbf y_i)$. Each pair consists of one parameter vector and the corresponding **63 noisy option prices**.

    ### Observation Noise

    The observation noise accounts for discrepancies between theoretical model prices and observed quotes. It is a **statistical assumption**, separate from the stochastic dynamics of the Heston model and numerical errors in the pricing method.

    We specify the noise levels in terms of implied volatility: **0.3 volatility percentage points for near-the-money options** and **1.0 percentage point for options in the wings**. These values are converted into price standard deviations using Black–Scholes vega evaluated at a reference volatility of 20%.

    This produces contract-specific noise levels that depend on strike and maturity. For simplicity, we assume independent Gaussian errors with known standard deviations. In practice, market quote errors may be correlated, asymmetric, or otherwise more complex; amortized inference allows you to express any modeling assumption, as long as you can simulate it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Setting up a simulator in BayesFlow

    BayesFlow connects the prior and simulator through named parameters: the keys returned by the prior must match the simulator's argument names. The rest is handled automatically.


    ```python
    import bayesflow as bf

    rng = np.random.default_rng(4)
    parameter_names = heston_task["parameter_names"]
    bounds = np.asarray(heston_task["prior_bounds"])
    noise_sd = np.asarray(heston_task["context"]["quote_noise_std"])

    def prior():
        values = rng.uniform(bounds[:, 0], bounds[:, 1])
        return dict(zip(parameter_names, values))

    def observation_model(initial_variance, theta, vol_of_vol, rho):
        params = [initial_variance, theta, vol_of_vol, rho]
        prices = teaching.heston_smile(heston_task, params)
        noisy_prices = prices + rng.normal(0, noise_sd, size=prices.shape)
        return {"observations": noisy_prices}

    simulator = bf.make_simulator([prior, observation_model])
    batch = simulator.sample(1000)
    ```
    """)
    return


@app.cell(hide_code=True)
def _(ASSETS, mo):
    mo.vstack([
        mo.image(ASSETS / "illustrations" / "heston-workflow-20261101" / "simulation-and-observation.png"),
        mo.md("**One row of the bank:** four parameters → 63 clean prices → 63 noisy observations. "
              "Repeat with a new parameter draw and new quote errors."),
        mo.accordion({"The prior and observation model, formally": mo.md(r"""
        The four parameters have independent uniform priors over
        $v_0,\theta\in[0.01,0.16]$, $\sigma_v\in[0.1,1.5]$ and
        $\rho\in[-0.95,-0.1]$. Mean reversion is fixed at $\kappa=2$.

        $$p(\mathbf y\mid\boldsymbol\vartheta)
          =\prod_{c=1}^{63}\mathcal N(y_c;P_c(\boldsymbol\vartheta),\sigma_c^2),\qquad
          p(\boldsymbol\vartheta\mid\mathbf y)
          \propto p(\mathbf y\mid\boldsymbol\vartheta)p(\boldsymbol\vartheta).$$

        Larger $\sigma_c$ means a quote constrains the parameters less strongly.
        The task fixes $\sigma_c$ before training and inference. Gaussian noise
        is unbounded: synthetic observations can occasionally violate price
        bounds. We retain them rather than silently clip the assumed model.
        """)}),
    ])
    return


@app.cell(hide_code=True)
def _(backbone_case, backbones, mo, np, plt, teaching):
    _task, _, _a = backbones.load_example(backbone_case.value)
    _clean = teaching.heston_smile(_task, _a["truth"])
    _sd = np.asarray(_task["context"]["quote_noise_std"])
    _maturities = np.array([c["maturity"] for c in _task["context"]["contracts"]])
    _strikes = np.array([c["strike"] for c in _task["context"]["contracts"]])
    _at = np.isclose(_maturities, 63 / 252)
    _z = (_a["observations"] - _clean) / _sd

    _fig, _axes = plt.subplots(1, 3, figsize=(10.0, 2.7), constrained_layout=True)
    _axes[0].plot(_strikes[_at], _clean[_at], "o-", color="#7755aa", ms=3)
    _axes[0].set(xlabel="strike / forward", ylabel="premium / forward", title="Clean expected prices")
    _axes[1].axhline(0, color="#999999", lw=.7)
    _axes[1].bar(np.arange(9), _z[_at], color="#7755aa", alpha=.65)
    _axes[1].set(xlabel="contract within this maturity", ylabel="error / assumed SD", title="One quote-error draw")
    _axes[2].plot(_strikes[_at], _clean[_at], "--", color="#777777", lw=1, label="clean")
    _axes[2].errorbar(_strikes[_at], _a["observations"][_at], yerr=_sd[_at], fmt="o", ms=3,
                      color="#3a9874", label="observed ±1 SD")
    _axes[2].set(xlabel="strike / forward", ylabel="premium / forward", title="The observed quotes")
    _axes[2].legend(fontsize=7)

    mo.vstack([_fig, mo.md(
        "**One simulated market, viewed at three months.** The first panel prices the known generating "
        "parameters. The second shows the actual saved quote errors in SD units. Adding those errors "
        "gives the observations in the third panel. The complete data vector repeats this nine-contract "
        "design at seven maturities. Dots left of/at the forward are puts; dots to its right are calls.")])
    return


@app.cell(hide_code=True)
def _(ASSETS, mo):
    mo.vstack([
        mo.md(r"""
        ## 4 Two neural approaches: learn the price or learn the posterior

        A pricer surrogate solves the forward simulation. An amortized posterior solves the inverse problem.
        Both reuse simulations, but they learn different functions:

        | | Neural pricer + Bayesian sampler | Amortized posterior |
        |:---|:---|:---|
        | Network learns | from parameters to clean prices | from noisy prices to a distribution of parameters |
        | Training examples | $(\boldsymbol\vartheta_i,\mathbf P_i)$ | $(\boldsymbol\vartheta_i,\mathbf y_i)$ |
        | At a new observation | evaluate the cheap surrogate inside a new MCMC run | condition the network and draw parameters |
        | Main approximation to check | surrogate error, then sampler convergence | posterior approximation and sampling error |
        | Changing prior/noise | often reuse the pricer, then run a new chain | generally requires a newly compatible estimator |

        Neither route turns a posterior into a single best-fit parameter. A
        neural pricer does not itself return uncertainty; the Bayesian sampler
        supplies it. A posterior network learns a conditional distribution,
        including dependence between parameters.
        """),
        mo.image(ASSETS / "illustrations" / "heston-workflow-20261101" / "two-amortization-routes.png"),
        mo.accordion({"The two mathematical targets": mo.md(r"""
        **Surrogate route.** Fit $\widehat{\mathbf P}_{\psi}(\boldsymbol\vartheta)
        \approx\mathbf P(\boldsymbol\vartheta)$ to clean prices, then sample

        $$\widetilde p(\boldsymbol\vartheta\mid\mathbf y)
          \propto p(\boldsymbol\vartheta)\prod_c
          \mathcal N(y_c;\widehat P_{\psi,c}(\boldsymbol\vartheta),\sigma_c^2).$$

        This targets the **surrogate likelihood**, which must be checked against
        the Fourier likelihood. A small training error alone is insufficient.

        **Posterior route.** Fit $q_\phi(\boldsymbol\vartheta\mid\mathbf y)
        \approx p(\boldsymbol\vartheta\mid\mathbf y)$ from prior-predictive pairs.
        A conditional flow maps base noise to joint parameter draws. This lesson
        uses flow matching to train that map; posterior density estimation and
        a learned price function are different learning problems.
        """)}),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Building and training the posterior estimator

    ```python
    # The adapter performs necessary transformations before passing simulations through the networks
    adapter = bf.Adapter()

    # Here, we first declare the parameter bounds (similar to Stan or PyMC)
    for name, (lower, upper) in zip(parameter_names, bounds):
        adapter = adapter.constrain(
            name,
            lower=lower,
            upper=upper,
            inclusive="none",
        )

    # Next, we instruct BayesFlow which quantities should be inferred from which other quantities
    adapter = (
        adapter
        .concatenate(parameter_names, into="inference_variables")
        .rename("observations", "inference_conditions")
    )

    # Flow matching is one of the many inference networks available
    network = bf.networks.FlowMatching(
        subnet_kwargs={"widths": [128, 128]},
    )

    # The workflow object connects simulator, adapter, and networks
    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        inference_network=network,
        standardize="all"
    )

    # We can train with an unlimited stream of simulations or use pre-simulated
    # data sets with .fit_offline / .fit_disk
    history = workflow.fit_online(
        epochs=15,
        batch_size=64,
        num_batches_per_epoch=2,
    )

    # Once trained, the Heston model can be calibrated immediately to any options chain
    posterior = workflow.sample(
        conditions={"observations": observed_prices[None, :]},
        num_samples=1000,
    )
    ```

    The results below use cached draws from a larger network trained before the tutorial. The button later in the notebook builds and trains a smaller network during this session so that the complete workflow remains runnable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Infer parameters from one synthetic observation

    We use one synthetic set of 63 noisy prices. The neural draws come from the pretrained estimator, and a [PyMC](https://www.pymc.io/index.html) numerical reference targets the same observation, prior, and likelihood.

    ```python
    task, report, case = backbones.load_example("heston-final500")
    observed_prices = case["observations"]
    posterior_draws = case["neural_draws"]
    ```
    """)
    return


@app.cell
def _(mo):
    from sbi4f_tutorial.shared import backbones

    backbone_case = mo.ui.dropdown(options={"Synthetic option quotes": "heston-final500"},
                                    value="Synthetic option quotes", label="Observed data set")
    backbone_case
    return backbone_case, backbones


@app.cell(hide_code=True)
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
                  title="Observed option prices: bars show the assumed quote-noise SD")
    _quote_ax.legend(fontsize=7, ncol=4)
    _quote_rows = [{"maturity (years)": c["maturity"], "strike / forward": c["strike"],
                    "option": c["option_type"], "observed price / forward": y,
                    "quote-noise SD": noise}
                   for c, y, noise in zip(_contracts, _arrays["observations"], _noise)]
    _quote_view = mo.vstack([
        _quote_fig,
        mo.md(r"""The horizontal axis is strike divided by the forward price, $K/F$; 1 is at the money. The vertical axis is the option premium in the same normalized units. Here spot and forward both equal 1 because rates and dividends are zero, and colors identify expiry.

        We show **puts at or below $K/F=1$ and calls above it**. Each expiry contains nine out-of-the-money contracts. The bars are the assumed observation standard deviations, not posterior intervals."""),
        mo.accordion({"Exact observations, contract design and known quote noise": mo.vstack([
            mo.md("These are the saved noisy prices in spot-one, zero-carry forward units. The quote-noise standard deviations define the observation model; prices are shown without implied-volatility inversion or clipping."),
            mo.ui.table(_quote_rows, selection=None, pagination=True, page_size=9),
        ])}),
    ])
    _rate_rows = []
    for _quantity, _label in (("variance_swap_1y", "1-year mean variance rate"),
                              ("forward_variance_0_5_to_1y", "6-month to 1-year forward variance rate")):
        for _prefix, _source in (("neural_", "neural posterior"), ("reference_", "PyMC reference")):
            _q = np.quantile(_arrays[_prefix + _quantity], [.05, .5, .95])
            _rate_rows.append({"quantity": _label, "source": _source,
                               "median (variance/year)": _q[1],
                               "central 90%": f"{_q[0]:.5g} to {_q[2]:.5g}"})
    _rate_view = mo.vstack([
        mo.md("Each paired parameter draw is mapped to Heston's mean variance over [0, 1] and [0.5, 1] years at the fixed kappa. The results are annualized decimal variance rates; no Monte Carlo pricing is used."),
        mo.ui.table(_rate_rows, selection=None, pagination=False),
        mo.md("Compare the neural and PyMC intervals. Which rate is more uncertain? Agreement on these two quantities does not validate other products or observations."),
    ])
    mo.vstack([
        mo.callout(mo.md("**Synthetic worked case.** The generating parameters are known for teaching. The neural posterior is approximate; the numerical reference checks it for this observation."), kind="neutral"),
        _quote_view,
        mo.ui.table(backbones.parameter_rows(_task, _arrays), selection=None, pagination=False),
        backbones.corner(_task, _arrays),
        _rate_view,
        mo.accordion({"Source and reproducibility details": mo.md(f"Stored case: `{_report['case_id']}`  \nTask SHA-256: `{_report['task_fingerprint']}`  \nModel SHA-256: `{_report['checkpoint_sha256']}`. The cached neural draws and PyMC reference belong to this observation; the historical studies below evaluate a different checkpoint.")}),
        mo.md("Choose one parameter pair with visible dependence and compare the neural and PyMC shapes and marginal widths. What can this case establish, and what still requires many independent observations?"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Posterior predictive: simulate quotes from what we learned

    A posterior describes parameter uncertainty. A posterior predictive draws parameters from that posterior and simulates the quotes they could generate:

    $$\boldsymbol\vartheta^{(m)}\sim q(\boldsymbol\vartheta\mid\mathbf y),\qquad
      \widetilde{\mathbf y}^{(m)}=\mathbf P(\boldsymbol\vartheta^{(m)})
        +\boldsymbol\sigma\odot\boldsymbol\epsilon^{(m)}.$$

    ```python
    parameters = posterior_draws[m]              # keep all four coordinates paired
    model_prices = teaching.heston_smile(heston_task, parameters)
    replicated_quotes = model_prices + noise_sd * rng.standard_normal(63)
    ```

    The model-price spread reflects parameter uncertainty. The replicated-quote spread also includes observation noise. We compare both with the observed quotes and, because this example is synthetic, the clean prices at the generating parameters.
    """)
    return


@app.cell(hide_code=True)
def _(backbone_case, backbones, mo, np, plt, teaching):
    _task, _report, _a = backbones.load_example(backbone_case.value)
    _draw_sets = {"neural posterior": _a["neural_draws"],
                  "PyMC reference": _a["reference_chains"]}
    _predictive = teaching.heston_posterior_predictive(
        _task, _draw_sets, _a["observations"], truth=_a["truth"], draws_per_set=32)
    _at = np.isclose(_predictive["maturities_years"], 63 / 252)
    _x = _predictive["strikes"][_at]
    _fig, _axes = plt.subplots(2, 2, figsize=(10, 5.4), sharex=True, constrained_layout=True)
    _truth_prices = _predictive["truth_prices"][_at]
    _noise_sd = _predictive["quote_noise_std"][_at]
    for _col, (_name, _set) in enumerate(_predictive["draw_sets"].items()):
        _ax, _detail = _axes[:, _col]
        _model_q = np.quantile(_set["mean_prices"][:, _at], [.05, .5, .95], axis=0)
        _rep_q = np.quantile(_set["replicated_quotes"][:, _at], [.05, .5, .95], axis=0)
        _ax.fill_between(_x, _rep_q[0], _rep_q[2], color="#9bd3ba", alpha=.55,
                         label="replicated quote spread (90%)")
        _ax.fill_between(_x, _model_q[0], _model_q[2], color="#7f53ad", alpha=.5,
                         label="model-price spread (90%)")
        _ax.plot(_x, _truth_prices, "--", color="#333333", lw=1, label="clean generating prices")
        _ax.scatter(_x, _predictive["observations"][_at], color="#222222", s=22, zorder=4,
                    label="observed noisy quotes")
        _ax.set(title=_name)
        _detail.fill_between(_x, (_rep_q[0] - _truth_prices) / _noise_sd,
                             (_rep_q[2] - _truth_prices) / _noise_sd, color="#9bd3ba", alpha=.55)
        _detail.fill_between(_x, (_model_q[0] - _truth_prices) / _noise_sd,
                             (_model_q[2] - _truth_prices) / _noise_sd, color="#7f53ad", alpha=.5)
        _detail.axhline(0, color="#333333", ls="--", lw=1)
        _detail.scatter(_x, (_predictive["observations"][_at] - _truth_prices) / _noise_sd,
                        color="#222222", s=22, zorder=4)
        _detail.set(xlabel="strike / forward", title="Zoom: deviation from clean price, in quote SDs")
    _axes[0, 0].set_ylabel("option premium / forward")
    _axes[1, 0].set_ylabel("deviation / assumed quote SD")
    _axes[0, 0].legend(fontsize=7, loc="upper left")
    mo.vstack([_fig, mo.md("The upper row shows three-month option prices; the lower row subtracts the clean generating price and divides by the assumed quote-noise SD. The bands represent pointwise empirical 90% ranges, not simultaneous coverage bands. A systematic discrepancy across strikes would indicate model mismatch.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Calibration: do uncertainty statements work across repeated data sets?

    **Financial calibration** infers model parameters from quotes. **Statistical calibration** asks whether the resulting uncertainty statements behave as advertised across repeated simulations. A 90% interval should contain the truth in about 90% of repeated simulations. One posterior or a falling training loss cannot establish that, so we apply several distinct checks:

    | Check | Slogan | What to look for |
    |:---|:---|:---|
    | Prior predictive | Can our assumptions generate data of this kind? | simulated data and the observation occupy a comparable range |
    | Recovery | Do inferred values track the generating values? | posterior centers follow truth, with uncertainty rather than perfect equality |
    | Marginal coverage | Does “90%” behave like 90%? | truths inside central 90% intervals about 90% of the time |
    | Simulation-based calibration (SBC) | Does the truth look like another posterior draw? | approximately uniform ranks across simulated cases |
    | Numerical reference | Are we approximating this posterior faithfully? | agreement for the same prior, observation and likelihood |
    | Posterior predictive | Can learned parameters generate quotes like these? | compare replicated quotes with the observation |

    Calibration is not concentration: a narrow posterior can be wrong. A method that always returns the prior can also pass prior-averaged calibration while learning nothing, so recovery and information gain matter too.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Coverage and SBC, formally": mo.md(r"""
    Generate independent test cases $\boldsymbol\vartheta_i^*\sim p(\boldsymbol\vartheta)$ and $\mathbf y_i\sim p(\mathbf y\mid\boldsymbol\vartheta_i^*)$. For parameter $j$, let $C_{ij}^{.9}$ be its central 90% posterior interval:

    $$\widehat{\mathrm{coverage}}_j
      =\frac1N\sum_{i=1}^N\mathbf1\{\vartheta_{ij}^*\in C_{ij}^{.9}\}.$$

    For SBC, draw $M$ posterior samples for each case and count how many lie below the generating value:

    $$R_{ij}=\sum_{m=1}^M\mathbf1\{\vartheta_{ij}^{(m)}<\vartheta_{ij}^*\}.$$

    Under the correct continuous posterior with independent draws, ranks are uniform on $\{0,\ldots,M\}$. Systematic patterns can reveal bias or incorrect spread even before applying the model to analyze real prices from markets.
    """)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Historical SPY workflow and studies

    The sections below retain the SPY observation, archived training metadata, and study figures. The live simulator and small training run demonstrate the current runtime; they do not reproduce the archived fit or inherit its validation results.

    | | Slot 1: Black–Scholes | Slot 4: Heston |
    |:---|:---|:---|
    | Data | two SPY quotes | the SPY smile: 63 prices on the design grid |
    | Unknowns | the volatility σ | $v_0$, $\theta$, $\sigma_v$, $\rho$ ($\kappa = 2$ fixed) |
    | Pricer | closed form | a Fourier integral, 20 ms per smile |
    | Posterior | a grid | a flow-matching network, checked against MCMC with the Fourier likelihood |
    | BayesFlow pieces | none | `make_simulator`, `Adapter`, `FlowMatching`, `BasicWorkflow`, diagnostics |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "**The model in detail** (click to open): Heston, its prior and its likelihood": mo.md(r"""
        #### 1. The model

        Under the pricing measure, in forward units,

        $$\frac{dS_t}{S_t} = \sqrt{V_t}\,dW^S_t, \qquad dV_t = \kappa(\theta - V_t)\,dt + \sigma_v\sqrt{V_t}\,dW^V_t,
        \qquad d\langle W^S, W^V\rangle_t = \rho\,dt .$$

        The variance $V_t$ starts at $v_0$, reverts to $\theta$ at rate $\kappa$, has volatility $\sigma_v$, and moves against the price when $\rho<0$, producing the skew. Heston is affine, so the log-price characteristic function is known in closed form and each option price requires one Fourier integral (Lecture 6).

        #### 2. The unknowns and the prior

        | Parameter | Meaning | Prior |
        |:---|:---|:---|
        | $v_0$ | today's variance | uniform on 0.01 to 0.16 (10% to 40% volatility) |
        | $\theta$ | long-run variance | uniform on 0.01 to 0.16 |
        | $\sigma_v$ | volatility of variance | uniform on 0.1 to 1.5 |
        | $\rho$ | correlation of price and variance | uniform on −0.95 to −0.1 |
        | $\kappa$ | speed of mean reversion | fixed at 2 a year |

        The **Feller condition** $2\kappa\theta \ge \sigma_v^2$ keeps the variance away from zero. The prior does not impose it.

        #### 3. The likelihood

        Each of the 63 prices is the model price plus Gaussian noise with the design's standard deviation $\sigma_c$ (slot 1):

        $$p(y \mid \theta) = \prod_{c=1}^{63} \mathcal N\big(y_c;\ P_c(\theta),\ \sigma_c^2\big).$$

        Heston's Fourier likelihood makes a numerical posterior reference affordable. The posterior network trains from simulations; MCMC evaluates the likelihood only for validation.
        """)
    })
    return


@app.cell
def _(mo):
    prior_button = mo.ui.run_button(label="Simulate 100 Heston prior-predictive smiles")
    prior_button
    return (prior_button,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Historical study: simulator and prior predictive

    BayesFlow needs only functions that **draw**. `prior()` returns one parameter set from the box, and `observation_model(...)` returns the 63 noisy prices. `bf.make_simulator` chains them by matching the observation model's argument names to the keys returned by `prior()`.
    """)
    return


@app.cell
def _(CHECKPOINTS, bf, np, teaching):
    heston_task = teaching.load_json(CHECKPOINTS / "heston" / "task.json")
    NAMES = heston_task["parameter_names"]
    BOUNDS = np.asarray(heston_task["prior_bounds"])
    NOISE = np.asarray(heston_task["context"]["quote_noise_std"])
    _rng = np.random.default_rng(4)

    def prior():
        return dict(zip(NAMES, _rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1])))

    def observation_model(initial_variance, theta, vol_of_vol, rho):
        parameters = [initial_variance, theta, vol_of_vol, rho]
        prices = teaching.heston_smile(heston_task, parameters)
        return {"observations": prices + NOISE * _rng.standard_normal(len(NOISE))}

    simulator = bf.make_simulator([prior, observation_model])
    return BOUNDS, NAMES, NOISE, heston_task, simulator


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The prior predictive check comes first.** Draw smiles from the prior and look at them next to SPY's. If SPY's smile lay outside the cloud, the network would be asked to extrapolate.

    **Predict, then inspect:** before pressing the simulation button, name a feature of SPY's smile the prior should reproduce. Afterward, explain why being inside this cloud would still not prove that a fitted posterior is accurate.
    """)
    return


@app.cell
def _(RESULTS, np, prior_button, simulator, surface, teaching):
    if prior_button.value:
        _prior = simulator.sample(100)
        prior_iv = np.array([teaching.implied_vol_grid(p) for p in _prior["observations"]])
    else:
        prior_iv = np.empty((0, 63))

    spy = teaching.load_json(RESULTS / "spy" / "surface.json")
    spy_prices = np.asarray(spy["prices_forward_units"])
    spy_iv = teaching.implied_vol_grid(spy_prices)
    _k, _maturity, _kind, grid_delta = teaching.design_arrays()
    grid_days = np.rint(_maturity * surface.TRADING_DAYS)
    return grid_days, grid_delta, prior_iv, spy_iv, spy_prices


@app.cell(hide_code=True)
def _(grid_days, grid_delta, mo, np, plots, plt, prior_iv, spy_iv, teaching):
    _fig, _axes = plt.subplots(1, 3, figsize=(11.0, 3.4), sharey=True, constrained_layout=True)
    for _ax, _days in zip(_axes, (5, 21, 252)):
        _rows = np.flatnonzero(grid_days == _days)
        for _iv in prior_iv:
            _ax.plot(grid_delta[_rows], 100 * _iv[_rows], color=plots.SERIES[0], alpha=0.12, linewidth=0.8)
        _ax.plot(grid_delta[_rows], 100 * spy_iv[_rows], color=plots.SERIES[1], marker="o", markersize=3, linewidth=1.6,
                 label="SPY, 21 Oct 2026")
        _ax.set_title(teaching.TENORS[_days], loc="left")
        _ax.set_xlabel("standardized moneyness δ")
    _axes[0].set_ylabel("implied volatility (%)")
    _axes[0].set_ylim(0, 80)
    _axes[0].legend(fontsize=8, loc="upper right")
    mo.vstack([_fig, mo.md(f"{len(prior_iv)} explicitly simulated prior smiles (blue); SPY is cached (orange).")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Historical study: bank and network

    The archived estimator used 50,000 training, 5,000 validation, and 5,000 test smiles generated with the same pricer. It shares the bounded-adapter and flow-matching structure shown above, but preserves its original BayesFlow version, sampler settings, four 256-unit hidden layers, and training recipe. The live network is a separate, smaller demonstration.
    """)
    return


@app.cell(hide_code=True)
def _(CHECKPOINTS, RESULTS, mo, teaching):
    _bank = teaching.load_json(RESULTS / "heston" / "s1" / "bank-quality.json")
    _training = teaching.load_json(CHECKPOINTS / "heston" / "training.json")
    mo.md(f"""
    | | |
    |:---|:---|
    | Bank | {_bank['generation']['surfaces']:,} smiles in {_bank['generation']['cpu_seconds'] / 60:.0f} CPU-minutes |
    | Pricing error | below 0.1 noise sd on every quote ({100 * _bank['share_of_quotes_above_0_1']:.0f}% above) |
    | Training | {_training['training']['epochs']} epochs in {_training['timing_seconds']['training_including_compile'] / 60:.0f} minutes on {_training['cpu_count']} cores, best validation loss at epoch {_training['training']['selected_epoch']} |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Train a small network

    The cached results load immediately from the archived estimator. The button below constructs a separate, smaller workflow and trains it on fresh simulations during this session. Its loss curve demonstrates the native BayesFlow training path; it does not reproduce or validate the archived fit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    train_button = mo.ui.run_button(label="Train a small native BayesFlow 2.1 network explicitly")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def _(BOUNDS, NAMES, bf, mo, np, plots, plt, simulator, time, train_button):
    mo.stop(not train_button.value, mo.md("*Press the button to train a small network for 15 online epochs.*"))
    _adapter = bf.Adapter()
    for _name, (_lower, _upper) in zip(NAMES, BOUNDS, strict=True):
        _adapter = _adapter.constrain(
            _name,
            lower=_lower,
            upper=_upper,
            inclusive="none",
        )
    _adapter = (
        _adapter.convert_dtype("float64", "float32")
        .concatenate(NAMES, into="inference_variables")
        .rename("observations", "inference_conditions")
    )
    _network = bf.networks.FlowMatching(
        subnet_kwargs={"widths": [128, 128]},
    )
    _workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=_adapter,
        inference_network=_network,
        standardize=["inference_variables", "inference_conditions"],
    )
    _start = time.perf_counter()
    _history = _workflow.fit_online(
        epochs=15,
        batch_size=64,
        num_batches_per_epoch=2,
        verbose=0,
    )
    _fig, _ax = plt.subplots(figsize=(6.0, 3.0), constrained_layout=True)
    _loss = _history.history["loss"]
    _ax.plot(np.arange(1, len(_loss) + 1), _loss, color=plots.SERIES[0], marker="o", markersize=3)
    _ax.set_xlabel("epoch")
    _ax.set_ylabel("flow-matching loss")
    _ax.set_title(f"15 online epochs, {time.perf_counter() - _start:.0f} s", loc="left")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Historical study: calibration and reference checks

    A network can be **biased**, placing the posterior in the wrong region, or **miscalibrated**, making it too narrow or too wide. Study S2 tests the archived checkpoint on 2,000 held-out simulated smiles with known parameters. These results do not evaluate the live demonstration network.
    """)
    return


@app.cell(hide_code=True)
def _(RESULTS, mo, teaching):
    _v = teaching.load_json(RESULTS / "heston" / "s2" / "validation.json")
    _cover = _v["coverage"]["0.95"]["coverage"]
    mo.vstack([
        mo.image(src=RESULTS / "heston" / "s2" / "sbc.png",
                 caption="Simulation-based calibration: rank ECDF minus uniform, with the simultaneous 95% band."),
        mo.md(f"SBC passes for **{sum(_v['sbc']['passed'])} of {len(_v['sbc']['passed'])}** parameters on {_v['surfaces']:,} smiles, and the 95% intervals contain the truth on {100 * min(_cover):.1f}% to {100 * max(_cover):.1f}% of them. Logit-variance contraction after mapping each prior box to the real line: " + ", ".join(f"{teaching.LABELS[n]} {x:.2f}" for n, x in _v["contraction"].items()) + "."),
        mo.image(src=RESULTS / "heston" / "s2" / "recovery.png",
                 caption="Recovery: posterior median and 90% interval against the true value."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "**How to read the diagnostics** (click to open)": mo.md(r"""
        **Simulation-based calibration.** For each test smile, the rank is the fraction of posterior draws below the true value. If the posterior is calibrated, these ranks are uniform. A curve above zero indicates upward bias; a curve positive on the left and negative on the right indicates intervals that are too narrow.

        **Recovery.** Each dot compares the true value with the posterior median and interval for one test smile. Centers should track the truth across cases, but individual dots need not lie on the diagonal.

        **Logit-variance contraction.** Normalize each parameter to its prior box, apply the logit, and compare average posterior variance with the prior logit variance $\pi^2/3$. This measures spread reduction in one coordinate system, not inference accuracy by itself.

        SBC averages over the prior predictive distribution. A numerical reference checks particular posteriors under the stated model and has its own numerical limitations. Neither establishes market-model adequacy.
        """)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Numerical reference.** On 50 additional test smiles, importance sampling with the Fourier likelihood provides a reference; four cases also use MCMC. The comparison requires means within 0.3 posterior standard deviations and 90% interval widths within 25%.
    """)
    return


@app.cell(hide_code=True)
def _(RESULTS, mo, teaching):
    _report = teaching.load_json(RESULTS / "heston" / "s3" / "heston-check.json")
    _chains = _report["summary"]["chain_checks"]
    mo.vstack([
        mo.ui.table(teaching.agreement_rows(_report), selection=None, pagination=False),
        mo.md(f"The network meets the criterion for all four parameters on **{100 * _report['summary']['network_vs_reweighted_pass_rate']:.0f}%** of the smiles. After reweighting with the exact likelihood, it agrees with the exact chain on **{sum(c['reweighted_vs_chain'] for c in _chains)} of {len(_chains)}** smiles. Slot 5 develops that correction."),

    ])
    return


@app.cell(hide_code=True)
def _(mo):
    comparison_case = mo.ui.dropdown(
        options={"Low volatility-of-variance": 2000, "Strong negative price–variance correlation": 2001,
                 "High volatility-of-variance": 2002, "High long-run variance": 2003},
        value="Low volatility-of-variance", label="Synthetic data set for the reference comparison")
    mo.vstack([mo.md("### Compare posteriors in the parameters' own units"), comparison_case])
    return (comparison_case,)


@app.cell(hide_code=True)
def _(RESULTS, comparison_case, heston_task, mo, np, plt, teaching):
    _row = comparison_case.value
    _saved = teaching.load_npz(RESULTS / "heston" / "s3" / "posterior-draws.npz")
    _truth = _saved[f"truth_{_row}"]
    _paths = teaching.heston_illustrative_paths(heston_task, _truth, paths=2, days=252, seed=20261101)
    _overview, _axes = plt.subplots(1, 3, figsize=(10, 2.7), constrained_layout=True)
    _axes[0].plot(_paths["times_years"], _paths["prices"].T, lw=.8)
    _axes[0].set(xlabel="years", ylabel="price / initial price", title="Possible price paths")
    _axes[1].plot(_paths["times_years"], 100 * np.sqrt(_paths["variances"].T), lw=.8)
    _axes[1].set(xlabel="years", ylabel="volatility (%)", title="Their hidden volatility")
    _cs = heston_task["context"]["contracts"]
    _at = np.isclose([c["maturity"] for c in _cs], 63 / 252)
    _axes[2].errorbar(np.array([c["strike"] for c in _cs])[_at], _saved[f"observation_{_row}"][_at],
                      yerr=np.asarray(heston_task["context"]["quote_noise_std"])[_at],
                      fmt="o", color="#333333", ms=3)
    _axes[2].set(xlabel="strike / forward", ylabel="premium / forward", title="Observed three-month quotes")
    _fig, _axes2 = plt.subplots(1, 4, figsize=(10, 2.7), constrained_layout=True)
    _sets = {"Fourier-MCMC reference": (_saved[f"chain_{_row}"], "#555555"),
             "neural posterior": (_saved[f"network_{_row}"], "#397ec4"),
             "importance-corrected": (_saved[f"reweighted_{_row}"], "#35a478")}
    for _j, _ax in enumerate(_axes2):
        for _name, (_draws, _color) in _sets.items():
            _ax.hist(_draws[:, _j], bins=35, density=True, histtype="step", color=_color, label=_name)
        _ax.axvline(_truth[_j], color="#111111", ls=":", lw=1.5, label="generating truth")
        _ax.set_xlabel(["initial variance v₀", "long-run variance θ", "vol-of-variance σᵥ", "correlation ρ"][_j])
        _ax.set_yticks([])
    _axes2[0].set_ylabel("posterior density")
    _axes2[0].legend(fontsize=7, loc="upper left")
    _notice = (mo.callout(mo.md("**Case 2001 is a genuine interval miss.** Both the neural and numerical posteriors exclude the generating vol-of-variance and correlation from their 95% marginal intervals. The saved seed reproduces the case, whose likelihood favors a shifted, strongly correlated region. A credible interval need not contain the truth in every simulation, and one miss does not determine calibration. The importance correction is also unstable here."), kind="warn")
        if _row == 2001 else mo.md("The dotted line is the generating value. Compare the methods in physical units, and inspect joint dependence as well as these marginals."))
    mo.vstack([_overview, mo.md("The path panels illustrate the generating parameters; they are not recovered latent paths or calibration inputs. The third panel shows the saved Fourier-price observation with noise."), _fig, _notice,
        mo.accordion({"Numerical reference scope and source": mo.md(f"Archived case ID: `{_row}`. The reference uses the Fourier likelihood and four Metropolis chains initialized from neural draws. Mixing diagnostics are reassuring but cannot rule out an unvisited mode. The observations, truths, and posterior arrays are preserved; the original raw bank is unavailable. In case 2001, the truth is 3.03 and 3.33 reference SDs below the vol-of-variance and correlation means, their posterior correlation is 0.887, and the Pareto diagnostic k-hat is 0.738.")}),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Historical study: inference on SPY

    The saved SPY reference uses MCMC with the Fourier likelihood. Seven of eight optimizer starts reach the same optimum, and four dispersed chains agree. This notebook does not contain a current neural posterior for the same SPY quotes, so the remaining comparison cells stop explicitly.
    """)
    return


@app.cell
def _(ASSETS, mo, np):
    exact_spy = np.load(ASSETS / "data" / "heston-spy-exact.npy")
    network_spy = None
    network_seconds = None
    mo.md("**Historical market observation.** The saved Fourier-reference draws below concern SPY. A neural comparison for these same quotes is not included; the synthetic posterior above belongs to a different observation.")
    return exact_spy, network_seconds, network_spy


@app.cell
def _(CHECKS, teaching):
    heston_checks = teaching.load_json(CHECKS / "heston_checks.json")
    return (heston_checks,)


@app.cell(hide_code=True)
def _(NAMES, exact_spy, mo, network_seconds, network_spy, teaching):
    mo.stop(network_spy is None, mo.md("A neural posterior for these SPY quotes is not included in this lesson."))
    _labels = [teaching.LABELS[n] for n in NAMES]
    _rows = []
    for _j, _name in enumerate(_labels):
        _rows.append({
            "parameter": _name,
            "exact mean ± sd": f"{exact_spy[:, _j].mean():.4f} ± {exact_spy[:, _j].std():.4f}",
            "network mean ± sd": f"{network_spy[:, _j].mean():.4f} ± {network_spy[:, _j].std():.4f}",
            "mean gap, exact sd": round((network_spy[:, _j].mean() - exact_spy[:, _j].mean()) / exact_spy[:, _j].std(), 2),
            "width ratio": round(network_spy[:, _j].std() / exact_spy[:, _j].std(), 2),
        })
    mo.vstack([
        mo.md(f"2,000 network draws in **{network_seconds:.1f} s**; the exact chain took about 9 CPU-minutes."),
        mo.ui.table(_rows, selection=None, pagination=False),
        teaching.pairs({"exact posterior (MCMC)": exact_spy, "network": network_spy}, _labels, figsize=(7.0, 7.0)),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Classical comparison.** Least squares with a Gauss–Newton interval can agree with the posterior when its mass is interior, unimodal, and locally quadratic. Many liquid quotes do not guarantee those conditions, so the comparison must be checked.
    """)
    return


@app.cell(hide_code=True)
def _(NAMES, exact_spy, heston_checks, mo, network_spy, np, teaching):
    mo.stop(network_spy is None, mo.md("A neural posterior for these SPY quotes is not included in this lesson."))
    _gn = heston_checks["cases"]["full grid, kappa fixed"]["parameters"]
    _rows = []
    for _j, _name in enumerate(NAMES):
        _e = np.quantile(exact_spy[:, _j], [0.05, 0.95])
        _n = np.quantile(network_spy[:, _j], [0.05, 0.95])
        _g = _gn[_name]["gauss_newton_90"]
        _rows.append({"parameter": teaching.LABELS[_name], "exact posterior, 90%": f"{_e[0]:.4f} to {_e[1]:.4f}",
                      "network, 90%": f"{_n[0]:.4f} to {_n[1]:.4f}", "Gauss–Newton, 90%": f"{_g[0]:.4f} to {_g[1]:.4f}",
                      "Gauss–Newton width / exact": round((_g[1] - _g[0]) / (_e[1] - _e[0]), 2)})
    mo.vstack([
        mo.ui.table(_rows, selection=None, pagination=False),
        mo.md("""The methods agree here. The network's advantage is amortized speed, not a different target. They can disagree with fewer quotes or weakly identified inputs, as shown in slots 1 and 5.

        *Source of the Gauss–Newton column:* `lab/checks/heston_checks.py`."""),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Model-price check.** Re-price 200 posterior draws and compare them with SPY. This propagates parameter uncertainty without fresh quote errors; the posterior predictive above adds those errors to generate replicated observations.
    """)
    return


@app.cell
def _(NOISE, heston_task, mo, network_spy, np, spy_prices, teaching):
    mo.stop(network_spy is None, mo.md("A neural posterior for these SPY quotes is not included in this lesson."))
    _rows = np.random.default_rng(6).choice(len(network_spy), 200, replace=False)
    repriced = np.array([teaching.heston_smile(heston_task, network_spy[i]) for i in _rows])
    repriced_iv = np.array([teaching.implied_vol_grid(p) for p in repriced])
    residuals = (spy_prices - repriced.mean(axis=0)) / NOISE
    return repriced_iv, residuals


@app.cell(hide_code=True)
def _(
    grid_days,
    grid_delta,
    mo,
    np,
    plots,
    plt,
    repriced_iv,
    residuals,
    spy_iv,
    teaching,
):
    _fig, _axes = plt.subplots(1, 3, figsize=(11.0, 3.4), sharey=False, constrained_layout=True)
    for _ax, _days in zip(_axes, (5, 21, 252)):
        _r = np.flatnonzero(grid_days == _days)
        _lo, _hi = np.quantile(repriced_iv[:, _r], [0.05, 0.95], axis=0)
        _ax.fill_between(grid_delta[_r], 100 * _lo, 100 * _hi, color=plots.SERIES[0], alpha=0.35, label="posterior, 90%")
        _ax.plot(grid_delta[_r], 100 * spy_iv[_r], "o", color=plots.SERIES[1], markersize=4, label="SPY")
        _ax.set_title(teaching.TENORS[_days], loc="left")
        _ax.set_xlabel("standardized moneyness δ")
    _axes[0].set_ylabel("implied volatility (%)")
    _axes[0].legend(fontsize=8)
    mo.vstack([_fig, mo.md(f"Root-mean-square residual of the posterior-mean prices: **{np.sqrt(np.mean(residuals**2)):.2f} noise units**, with {np.sum(np.abs(residuals) <= 2)} of 63 quotes within two. The loose wing noise at SPY's low volatility makes the fit look better there than it is.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Historical study: model assumptions and derived quantities

    The posterior is precise. Three of its consequences deserve attention.
    """)
    return


@app.cell
def _(exact_spy, np, surface, teaching):
    feller_ratio = exact_spy[:, 2] ** 2 / (2 * 2.0 * exact_spy[:, 1])
    variance_swap_vol = teaching.heston_variance_swap_vol(exact_spy[:, 0], exact_spy[:, 1], 2.0, surface.DESIGN_DAYS)
    SPY_VARIANCE_SWAP_VOL = np.array([13.6, 15.0, 16.2, 17.6, 18.4, 20.1, 21.8])  # replicated from the chain (study S8)
    return SPY_VARIANCE_SWAP_VOL, feller_ratio, variance_swap_vol


@app.cell(hide_code=True)
def _(
    SPY_VARIANCE_SWAP_VOL,
    feller_ratio,
    mo,
    np,
    plots,
    plt,
    surface,
    variance_swap_vol,
):
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(11.0, 3.6), constrained_layout=True,
                                         gridspec_kw={"width_ratios": [1, 1.5]})
    _left.hist(feller_ratio, bins=50, color=plots.SERIES[0], alpha=0.8)
    _left.axvline(1.0, color=plots.INK, linestyle="--", label="Feller limit")
    _left.set_xlabel("σ_v² / (2κθ)")
    _left.set_yticks([])
    _left.legend(fontsize=8)
    _left.set_title("the Feller ratio over the posterior", loc="left")
    _q = np.quantile(variance_swap_vol, [0.05, 0.5, 0.95], axis=0)
    _days = surface.DESIGN_DAYS
    _right.fill_between(_days, _q[0], _q[2], color=plots.SERIES[0], alpha=0.3, label="Heston posterior, 90%")
    _right.plot(_days, _q[1], color=plots.SERIES[0])
    _right.plot(_days, SPY_VARIANCE_SWAP_VOL, "o", color=plots.SERIES[1], label="SPY, replicated from the chain")
    _right.set_xscale("log")
    _right.set_xticks(_days, ["1w", "2w", "1m", "2m", "3m", "6m", "1y"])
    _right.minorticks_off()
    _right.set_ylabel("variance-swap volatility (%)")
    _right.legend(fontsize=8, loc="upper left")
    _right.set_title("variance swaps: the posterior against the market", loc="left")
    _gaps = _q[1] - SPY_VARIANCE_SWAP_VOL
    mo.vstack([_fig, mo.md(f"""
    - **The Feller condition fails on every draw**: the ratio runs from {feller_ratio.min():.1f} to {feller_ratio.max():.1f}, so vol-of-vol is about {np.sqrt(np.median(feller_ratio)):.1f} times its Feller limit. The variance boundary is accessible, but violating this sufficient condition does not by itself invalidate Heston.

    - **Heston's variance swaps sit below the market** by {-_gaps[1:].max():.1f} to {-_gaps[1:].min():.1f} volatility points from two weeks to one year, while the posterior's 90% band is less than one point wide. The curve is tied to smile parameters, and the market replication uses strikes outside the 63-quote grid.

    Precise, and wrong for a product the smile does not pin down.
    """)])
    return


@app.cell(hide_code=True)
def _(heston_checks, mo, np, plots, plt):
    _free = heston_checks["cases"]["full grid, kappa free"]
    _fixed = heston_checks["cases"]["full grid, kappa fixed"]
    _kq = _free["parameters"]["kappa"]["posterior_q05_q50_q95"]
    _profile = heston_checks["kappa_profile"]
    _kap = [p["kappa"] for p in _profile]
    _chi2 = [p["chi2"] for p in _profile]
    _names = {"variance_swap": "63-day variance swap", "volatility_swap": "63-day volatility swap",
              "forward_start_call": "forward start, 3 to 6 months", "put_out_of_grid": "2-year 90% put"}
    _shift_rows = []
    for _p, _label in _names.items():
        _a = _fixed["products"][_p]["posterior_q05_q50_q95"]
        _b = _free["products"][_p]["posterior_q05_q50_q95"]
        _shift_rows.append({"product": _label, "κ = 2: median and 90% range": f"{_a[1]:.4f} ({_a[0]:.4f} to {_a[2]:.4f})",
                            "κ free: median and 90% range": f"{_b[1]:.4f} ({_b[0]:.4f} to {_b[2]:.4f})",
                            "shift of the median": f"{100 * (_b[1] / _a[1] - 1):+.1f}%",
                            "ranges overlap": "yes" if (_b[0] <= _a[2] and _a[0] <= _b[2]) else "no"})
    _fig, _ax = plt.subplots(figsize=(6.6, 3.2), constrained_layout=True)
    _ax.plot(_kap, _chi2, "o-", color=plots.INK)
    _ax.axvspan(_kq[0], _kq[2], color=plots.SERIES[0], alpha=0.25, label="posterior of κ, 90%")
    _ax.axvline(2.0, color=plots.SERIES[1], linestyle="--", label="κ fixed in slot 4")
    _ax.set_xscale("log")
    _ax.set_xticks(_kap, [f"{k:g}" for k in _kap])
    _ax.minorticks_off()
    _ax.set_xlabel("mean reversion κ (fixed in each least-squares fit)")
    _ax.set_ylabel("χ² on 63 quotes")
    _ax.legend(fontsize=8)
    _ax.set_title("the smile's verdict on κ", loc="left")
    mo.vstack([
        mo.md(f"**κ = 2 is a choice, and the smile disagrees with it.** When inferred, the exact posterior median is **{_kq[1]:.1f}**, with a 90% interval from {_kq[0]:.1f} to {_kq[2]:.1f}, and the fit improves from χ² {_fixed['best_chi2']:.1f} to {_free['best_chi2']:.1f}. Fixing κ is a zero-width prior, so every posterior and product price above is conditional on that choice."),
        _fig,
        mo.ui.table(_shift_rows, selection=None, pagination=False),
        mo.md("Product prices are in forward units under exact posteriors with κ fixed and free (`lab/checks/heston_checks.py`). Inferring κ requires a new task, simulation bank, and five-target estimator; editing the task file cannot change the existing four-target network. This stored check is independent of the unavailable SPY neural comparison."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    1. **State the inverse problem.** The observed quotes, unknown parameters, fixed assumptions, and quote noise define the posterior target.
    2. **Simulate, then learn.** BayesFlow connects the financial simulator to a reusable conditional distribution. The live demonstration and pretrained estimator are separate fits.
    3. **Know what the network learns.** A surrogate learns prices and still needs a Bayesian computation for each observation; an amortized estimator learns a conditional parameter distribution.
    4. **Check uncertainty at several levels.** Recovery, coverage, SBC, numerical references, and replicated quotes answer different questions. One successful case is not enough.
    5. **Preserve dependence and assumptions.** Carry paired posterior draws into derived quantities. Every result remains conditional on the model, prior, noise, and fixed parameters.

    **Next:** rough Bergomi keeps the option-inference problem but changes the variance dynamics and conditioning assumptions.

    *Evidence and code:* the [curriculum](../curriculum/04-slot4-heston-in-bayesflow.md); studies S1–S3 in `assets/results/calibration/heston/`; the historical SPY reference in `assets/data/`; and the Gauss–Newton and κ quick checks in `assets/quick-checks/`.
    """)
    return


if __name__ == "__main__":
    app.run()
