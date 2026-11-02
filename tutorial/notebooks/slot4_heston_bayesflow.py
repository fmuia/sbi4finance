import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Heston · from option quotes to a posterior")


@app.cell(hide_code=True)
def _():
    import os
    import time

    os.environ.setdefault("KERAS_BACKEND", "jax")

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from sbi4f_tutorial.heston_options import lesson as teaching
    from sbi4f_tutorial.shared import full_surface, plots, surface

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
        full_surface,
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
    # Heston · from option quotes to a posterior

    **Simulate and train once; infer uncertainty for each new set of quotes.**

    We observe prices of options: contracts that pay when an asset finishes above
    a strike (a call) or below it (a put). Different strikes and expiry dates give
    different information about the asset's possible future. Here one data set
    contains **63 option prices: nine strikes at each of seven maturities**.

    Our task is to infer four parameters of the Heston model from those prices.
    We will generate training data, build an amortized posterior estimator,
    compare it with a numerical reference, and check what its predictions imply.
    The main worked case is synthetic, so its generating parameters are known.
    A later section retains the separate historical SPY study.

    **The route:** model → simulated data → two neural approaches → posterior → checks → interpretation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · The model: the price moves, and its volatility moves too

    **Heston gives the asset a changing variance, correlated with its price.**
    A negative correlation lets downward price moves coincide with increases in
    variance. This produces different option prices across strikes, unlike a
    model with one constant volatility.

    $$
    dS_t = S_t\sqrt{V_t}\,dW_t^S,\qquad
    dV_t = \kappa(\theta-V_t)\,dt+\sigma_v\sqrt{V_t}\,dW_t^V,
    \qquad d\langle W^S,W^V\rangle_t=\rho\,dt.
    $$

    | Symbol | Meaning | In this inference task |
    |:---|:---|:---|
    | $S_t$ | asset price, starting at 1 | a possible future path; not our observed data |
    | $V_t$ | instantaneous **variance**; volatility is $\sqrt{V_t}$ | an unobserved state |
    | $v_0$ | initial variance $V_0$ | infer it |
    | $\theta$ | long-run variance level | infer it |
    | $\sigma_v$ | strength of random variance moves | infer it |
    | $\rho$ | correlation between the two random shocks | infer it |
    | $\kappa$ | speed of mean reversion | fixed at 2 per year |

    These are **pricing-measure dynamics**: they describe the distribution used
    to value contingent payoffs. We use zero rates and dividends, so no drift
    appears in this equation. Estimating drift from observed returns is a
    different task, explored in the trajectory notebook.
    """)
    return


@app.cell(hide_code=True)
def _(heston_task, mo, teaching, plt):
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

        The signed state may become negative; the next step uses its positive
        part. This numerical illustration is not a new calibrated process law.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · How one simulated data set comes about

    **Choose parameters → price the contracts → perturb the prices.**

    Write the four unknowns as $\boldsymbol\vartheta=(v_0,\theta,\sigma_v,\rho)$,
    reserving $\theta$ for Heston's long-run variance. A clean call price is an
    expected payoff; a put uses $(K-S_T)_+$ instead:

    $$P_{K,T}(\boldsymbol\vartheta)=\mathbb E^{\mathbb Q}_{\boldsymbol\vartheta}[(S_T-K)_+].$$

    The Fourier pricer returns all 63 expectations. The data simulator then adds
    the assumed quote errors:

    $$\boldsymbol\vartheta\sim p(\boldsymbol\vartheta),\qquad
    y_c=P_c(\boldsymbol\vartheta)+\sigma_c\epsilon_c,\qquad
    \epsilon_c\overset{\mathrm{iid}}\sim\mathcal N(0,1).$$

    Repeating this gives a training bank of pairs
    $(\boldsymbol\vartheta_i,\mathbf y_i)$. Each pair is **one hypothetical
    option market**, not one asset path and not one option.

    **What does the noise represent?** A transparent statistical tolerance for
    differences between ideal model prices and quoted prices. Here it is an
    imposed observation model, not the randomness of Heston paths, not Fourier
    numerical error, and not a measured bid–ask spread. The quoted SDs are fixed
    inputs; we do not estimate them in this lesson.

    The design starts from **0.3 volatility percentage points near the money**
    and **1.0 point in the wings**, then converts these to price SDs using
    Black–Scholes vega at a fixed 20% reference volatility. Longer-lived options
    generally have different price SDs. Independent Gaussian errors are a
    simplifying assumption; actual quote errors can be dependent or asymmetric.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · The complete data simulator

    **The financial calculation sits behind one function; the Bayesian simulator is short.**

    ```python
    import bayesflow as bf

    rng = np.random.default_rng(4)
    parameter_names = heston_task["parameter_names"]
    bounds = np.asarray(heston_task["prior_bounds"])
    noise_sd = np.asarray(heston_task["context"]["quote_noise_std"])

    def prior():
        values = rng.uniform(bounds[:, 0], bounds[:, 1])
        return dict(zip(parameter_names, values))

    def smile(initial_variance, theta, vol_of_vol, rho):
        parameters = [initial_variance, theta, vol_of_vol, rho]
        clean = teaching.heston_smile(heston_task, parameters)
        return {"observations": clean + noise_sd * rng.standard_normal(63)}

    simulator = bf.make_simulator([prior, smile])
    batch = simulator.sample(1000)
    ```

    Each row has four generating parameters and 63 observed prices. The network
    sees the prices as conditions; the known parameters are training targets.
    For a real market the targets are absent—that is precisely the inference problem.
    """)
    return


@app.cell(hide_code=True)
def _(ASSETS, mo):
    mo.vstack([
        mo.image(ASSETS / "illustrations" / "heston-workflow-20261101" / "simulation-and-observation.png"),
        mo.md("**One row of the bank:** four parameters → 63 clean prices → 63 noisy observations. "
              "Repeat with a new parameter draw and fresh quote errors. The parameter values label "
              "the simulation; they are never part of a new market's inference input."),
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
    _axes[0].set(xlabel="strike / forward", ylabel="premium / forward", title="1 · Clean expected prices")
    _axes[1].axhline(0, color="#999999", lw=.7)
    _axes[1].bar(np.arange(9), _z[_at], color="#7755aa", alpha=.65)
    _axes[1].set(xlabel="contract within this maturity", ylabel="error / assumed SD", title="2 · One quote-error draw")
    _axes[2].plot(_strikes[_at], _clean[_at], "--", color="#777777", lw=1, label="clean")
    _axes[2].errorbar(_strikes[_at], _a["observations"][_at], yerr=_sd[_at], fmt="o", ms=3,
                      color="#3a9874", label="observed ±1 SD")
    _axes[2].set(xlabel="strike / forward", ylabel="premium / forward", title="3 · The observed quotes")
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
        ## 4 · Two neural approaches: learn the price or learn the posterior

        **A pricer surrogate accelerates the forward calculation.
        An amortized posterior accelerates the inverse calculation.**

        Both reuse simulations, but they learn different functions:

        | | Neural pricer + Bayesian sampler | Amortized posterior |
        |:---|:---|:---|
        | Network learns | parameters → 63 clean prices | 63 noisy prices → a distribution of parameters |
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
    ### The core BayesFlow workflow, in view

    **Prepare pairs, build a conditional network, fit once, draw for new data.**
    These are the actual APIs used by the optional training cell below. The
    helper builds the adapter and flow-matching network; its short structure is
    exposed in the next panel.

    ```python
    batch = make_demo_simulator().sample(1000)
    split = {
        "parameters": np.column_stack([batch[name].ravel() for name in parameter_names]),
        "observations": batch["observations"],
    }
    data = full_surface.training_data(split, heston_task)
    workflow = full_surface.build_workflow(heston_task, family="flow_matching")
    history = workflow.fit_offline(data=data, epochs=15, batch_size=64)
    draws = full_surface.sample(workflow, heston_task, observed_prices[None, :], 1000)
    ```

    The adapter respects parameter bounds and puts inputs on useful numerical
    scales. Those transformations do not change the scientific meaning of the
    parameters. Training on 1,000 simulations demonstrates the API; the
    pretrained result uses a separate, much larger offline bank.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Inside the workflow: adapter + flow-matching network.**
    The reusable builder adds validation, task-dependent conditioning and native
    sampler settings. Its essential vector-input configuration is:

    ```python
    adapter = bf.Adapter()
    for name, (lower, upper) in zip(parameter_names, bounds):
        adapter = adapter.constrain(name, lower=lower, upper=upper, inclusive="none")
    adapter = (adapter.convert_dtype("float64", "float32")
               .concatenate(parameter_names, into="inference_variables")
               .concatenate(["observations"], into="inference_conditions"))
    network = bf.networks.FlowMatching(subnet_kwargs={"widths": [128, 128]})
    workflow = bf.BasicWorkflow(
        adapter=adapter, inference_network=network,
        standardize=["inference_variables", "inference_conditions"],
    )
    ```

    For this fixed quote design, a dense conditional network consumes all 63
    prices. The trajectory lesson instead summarizes a variable-length time
    series. A numerical likelihood is needed for the reference, not to train
    the posterior network.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5 · Infer parameters from one synthetic observation

    **A posterior must be attached to a particular observation.** We use one
    synthetic set of 63 noisy prices and a pretrained estimator. Cached draws
    let us inspect the result immediately. A numerical PyMC reference targets
    the same observation and prior; this one comparison is not an accuracy
    guarantee across all possible markets.

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
                  title="Observed option prices · bars show the assumed quote-noise SD")
    _quote_ax.legend(fontsize=7, ncol=4)
    _quote_rows = [{"maturity (years)": float(c["maturity"]), "strike / forward": float(c["strike"]),
                    "option": c["option_type"], "observed price / forward": float(y),
                    "quote-noise SD": float(noise)}
                   for c, y, noise in zip(_contracts, _arrays["observations"], _noise)]
    _quote_view = mo.vstack([
        _quote_fig,
        mo.md(r"""**Read the axes.** The horizontal axis is strike divided by the forward price, $K/F$.
        A strike of 0.8 means 80 when the forward is 100; 1 is at the money.
        The vertical axis is the option premium in the same normalized units:
        0.02 means a price of 2 when the forward is 100. Here spot and forward
        both equal 1 because rates and dividends are zero. Colors identify expiry.

        We show **puts at or below $K/F=1$ and calls above it**. The mound is a
        strip of out-of-the-money options, not the price of one call as its
        strike changes. Each expiry uses nine standardized-moneyness points,
        which is why its horizontal extent differs. The bars are assumed
        observation SDs; they are not posterior uncertainty intervals."""),
        mo.accordion({"Exact observations, contract design and known quote noise": mo.vstack([
            mo.md("These are the saved noisy prices used for this posterior, in the task's spot-one, "
                  "zero-carry forward units. The quote-noise standard deviations define the observation "
                  "model; they are not posterior intervals. Prices are shown as stored, without implied-"
                  "volatility inversion or clipping."),
            mo.ui.table(_quote_rows, selection=None, pagination=True, page_size=9),
        ])}),
    ])
    _rate_rows = []
    for _quantity, _label in (("variance_swap_1y", "1-year mean variance rate"),
                              ("forward_variance_0_5_to_1y", "6-month to 1-year forward variance rate")):
        for _prefix, _source in (("neural_", "neural posterior"), ("reference_", "PyMC reference")):
            _q = np.quantile(_arrays[_prefix + _quantity].reshape(-1), [.05, .5, .95])
            _rate_rows.append({"quantity": _label, "source": _source,
                               "median (variance/year)": float(_q[1]),
                               "central 90%": f"{_q[0]:.5g} to {_q[2]:.5g}"})
    _rate_view = mo.vstack([
        mo.md("**Carry each paired draw to a derived quantity.** These cached analytic quantities average "
              "Heston's mean variance over [0, 1] and [0.5, 1] years at the task's fixed kappa. They are "
              "annualized decimal variance rates; this table performs no new Monte Carlo pricing."),
        mo.ui.table(_rate_rows, selection=None, pagination=False),
        mo.md("**Interpret the rates.** Compare neural and PyMC intervals for the same observation. "
              "Which rate is more uncertain? Agreement on these two functionals would still leave "
              "other products and other observations to check."),
    ])
    mo.vstack([
        mo.callout(mo.md("**Synthetic worked case.** The generating parameters are known for teaching. "
                         "The neural posterior is approximate; the numerical reference checks it for this observation."), kind="neutral"),
        _quote_view,
        mo.ui.table(backbones.parameter_rows(_task, _arrays), selection=None, pagination=False),
        backbones.corner(_task, _arrays),
        _rate_view,
        mo.accordion({"Source and reproducibility details": mo.md(
            f"Stored case: `{_report['case_id']}`  \nTask SHA-256: `{_report['task_fingerprint']}`  \n"
            f"Model SHA-256: `{_report['checkpoint_sha256']}`. "
            "The cached neural draws and PyMC reference belong to this observation; "
            "the historical studies below evaluate a different checkpoint.")}),
        mo.md("**Interpret the current case.** Choose one parameter pair with visible dependence. "
              "Compare the neural and PyMC shapes as well as their marginal widths. What can this one "
              "comparison establish, and what still needs a collection of independent observations?"),
    ])
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Posterior predictive: simulate quotes from what we learned

    **Draw plausible parameters; generate another possible set of quotes.**
    A posterior describes parameter uncertainty. A posterior predictive describes
    what data those uncertain parameters could generate:

    $$\boldsymbol\vartheta^{(m)}\sim q(\boldsymbol\vartheta\mid\mathbf y),\qquad
      \widetilde{\mathbf y}^{(m)}=\mathbf P(\boldsymbol\vartheta^{(m)})
        +\boldsymbol\sigma\odot\boldsymbol\epsilon^{(m)}.$$

    ```python
    parameters = posterior_draws[m]              # keep all four coordinates paired
    model_prices = teaching.heston_smile(heston_task, parameters)
    replicated_quotes = model_prices + noise_sd * rng.standard_normal(63)
    ```

    The model-price spread reflects uncertainty about parameters. Replicated
    quotes add the assumed observation error. **These are different bands.**
    We can compare both with the observed synthetic quotes and, here, with the
    clean prices at the known generating parameters.
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
    mo.vstack([_fig, mo.md(
        "**Three-month options from the same synthetic observation.** The lower row subtracts the known "
        "clean generating price and divides by each quote's assumed SD: a value of 1 means one noise SD. "
        "This reveals uncertainty bands that are small on the full price axis. The generating truth is "
        "used only for this synthetic illustration, never as an inference input. Each column re-prices 32 saved "
        "joint parameter draws and adds fresh quote noise, using a fixed seed. Bands are pointwise "
        "empirical 90% ranges from a small illustrative sample—not simultaneous coverage bands. "
        "Puts are used at/below the forward and calls above it. "
        "Does adding observation noise widen the replicated-price spread? What would a systematic "
        "discrepancy across strikes suggest? This in-sample check can reveal mismatch; a good-looking "
        "fit alone does not certify the neural posterior or the financial model.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Calibration: do uncertainty statements work across repeated data sets?

    **Financial calibration** infers model parameters from quotes.
    **Statistical calibration** asks whether the resulting uncertainty statements
    behave as advertised across repeated simulations.

    **A 90% interval should contain the truth in about 90% of repeated simulations.**
    One attractive posterior or a falling training loss cannot establish that.
    Draw many new parameter–data pairs from the declared model, infer a posterior
    for each data set, and count successes. These are distinct checks:

    | Check | Slogan | What to look for |
    |:---|:---|:---|
    | Prior predictive | Can our assumptions generate data of this kind? | simulated data and the observation occupy a comparable range |
    | Recovery | Do inferred values track the generating values? | posterior centers follow truth, with uncertainty rather than perfect equality |
    | Marginal coverage | Does “90%” behave like 90%? | truths inside central 90% intervals about 90% of the time |
    | Simulation-based calibration (SBC) | Does the truth look like another posterior draw? | approximately uniform ranks across simulated cases |
    | Numerical reference | Are we approximating this posterior faithfully? | agreement for the same prior, observation and likelihood |
    | Posterior predictive | Can learned parameters generate quotes like these? | compare replicated quotes with the observation |

    **Calibration is not concentration.** A narrow posterior can be wrong. A
    method that always returns the prior can pass prior-averaged calibration
    while learning nothing; examine recovery and information gain as well.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Coverage and SBC, formally": mo.md(r"""
    Generate independent test cases
    $\boldsymbol\vartheta_i^*\sim p(\boldsymbol\vartheta)$,
    $\mathbf y_i\sim p(\mathbf y\mid\boldsymbol\vartheta_i^*)$.
    For parameter $j$, let $C_{ij}^{.9}$ be its central 90% posterior interval:

    $$\widehat{\mathrm{coverage}}_j
      =\frac1N\sum_{i=1}^N\mathbf1\{\vartheta_{ij}^*\in C_{ij}^{.9}\}.$$

    For SBC, draw $M$ posterior samples for each case and count how many lie
    below the corresponding generating value:

    $$R_{ij}=\sum_{m=1}^M\mathbf1\{\vartheta_{ij}^{(m)}<\vartheta_{ij}^*\}.$$

    Under the correct continuous posterior with independent draws, ranks are
    uniform on $\{0,\ldots,M\}$. Systematic rank patterns can reveal bias or
    incorrect spread. Dependence among posterior samples requires appropriate
    handling; simulation counts also carry sampling uncertainty.

    These checks average over the **declared prior predictive distribution**.
    Passing them does not prove conditional calibration at every parameter or
    observation, joint dependence accuracy, or adequacy for real market data.
    """)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Historical SPY workflow and studies

    The sections below retain the SPY observation, archived training metadata and study figures.
    Simulator and small-fit buttons use the current runtime as software demonstrations. They do not
    reproduce the archived full fit or transfer its validation results to the current synthetic example.

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
        #### 1 · The model

        Under the pricing measure, in forward units,

        $$\frac{dS_t}{S_t} = \sqrt{V_t}\,dW^S_t, \qquad dV_t = \kappa(\theta - V_t)\,dt + \sigma_v\sqrt{V_t}\,dW^V_t,
        \qquad d\langle W^S, W^V\rangle_t = \rho\,dt .$$

        The variance $V_t$ starts at $v_0$, reverts to $\theta$ at rate $\kappa$, has its own volatility
        $\sigma_v$, and moves against the price when $\rho < 0$, which produces the skew. Heston is affine,
        so the characteristic function of the log price is known in closed form, and each option price is
        one Fourier integral (Lecture 6).

        #### 2 · The unknowns and the prior

        | Parameter | Meaning | Prior |
        |:---|:---|:---|
        | $v_0$ | today's variance | uniform on 0.01 to 0.16 (10% to 40% volatility) |
        | $\theta$ | long-run variance | uniform on 0.01 to 0.16 |
        | $\sigma_v$ | volatility of variance | uniform on 0.1 to 1.5 |
        | $\rho$ | correlation of price and variance | uniform on −0.95 to −0.1 |
        | $\kappa$ | speed of mean reversion | fixed at 2 a year |

        The **Feller condition** $2\kappa\theta \ge \sigma_v^2$ keeps the variance away from zero. The prior
        does not impose it.

        #### 3 · The likelihood

        Each of the 63 prices is the model price plus Gaussian noise with the design's standard deviation
        $\sigma_c$ (slot 1):

        $$p(y \mid \theta) = \prod_{c=1}^{63} \mathcal N\big(y_c;\ P_c(\theta),\ \sigma_c^2\big).$$

        Heston's Fourier likelihood makes a numerical posterior reference affordable.
        The posterior network trains from simulations; MCMC evaluates the likelihood as a check.
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
    ### Historical study · simulator and prior predictive

    BayesFlow needs only functions that **draw**. `prior()` returns one parameter set from the box, and
    `smile(...)` returns the 63 noisy prices for it. `bf.make_simulator` chains them, matching the argument
    names to the keys `prior()` returns.
    """)
    return


@app.cell
def _(CHECKPOINTS, np, teaching):
    heston_task = teaching.load_json(CHECKPOINTS / "heston" / "task.json")
    NAMES = heston_task["parameter_names"]
    BOUNDS = np.asarray(heston_task["prior_bounds"])
    NOISE = np.asarray(heston_task["context"]["quote_noise_std"])

    def make_demo_simulator():
        import bayesflow as bf
        _rng = np.random.default_rng(4)

        def prior():
            return dict(zip(NAMES, _rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1])))

        def smile(initial_variance, theta, vol_of_vol, rho):
            prices = teaching.heston_smile(heston_task, [initial_variance, theta, vol_of_vol, rho])
            return {"observations": prices + NOISE * _rng.standard_normal(len(NOISE))}

        return bf.make_simulator([prior, smile])

    return NAMES, NOISE, heston_task, make_demo_simulator


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The prior predictive check comes first.** Draw smiles from the prior and look at them next to SPY's.
    If SPY's smile lay outside the cloud, the network would be asked to extrapolate.

    **Predict, then inspect:** before pressing the simulation button, name a feature of SPY's smile
    the prior should reproduce. Afterward, explain why being inside this cloud would still not prove
    that a fitted posterior is accurate.
    """)
    return


@app.cell
def _(RESULTS, make_demo_simulator, np, prior_button, surface, teaching, time):
    _start = time.perf_counter()
    if prior_button.value:
        _prior = make_demo_simulator().sample(100)
        prior_iv = np.array([teaching.implied_vol_grid(p) for p in _prior["observations"]])
    else:
        prior_iv = np.empty((0, 63))
    prior_seconds = time.perf_counter() - _start
    spy = teaching.load_json(RESULTS / "spy" / "surface.json")
    spy_prices = np.asarray(spy["prices_forward_units"])
    spy_iv = teaching.implied_vol_grid(spy_prices)
    _k, _maturity, _kind, grid_delta = teaching.design_arrays()
    grid_days = np.rint(_maturity * surface.TRADING_DAYS).astype(int)
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
    ### Historical study · bank and network

    The training bank was simulated once with the same pricer: 50,000 training, 5,000 validation and 5,000
    test smiles. The network is slot 3's configuration, built by one function: the adapter maps each
    parameter from its prior box to the real line and standardizes the prices, and the inference network
    is flow matching.

    ```python
    workflow = full_surface.build_workflow(heston_task, family="flow_matching", widths=(256, 256, 256, 256))   # slot 3, section 3
    ```
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
    | Noise | {_training['training']['quote_noise']} |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try a small training run

    Training is the one-off cost. The button trains a fresh network for 15 epochs on 1,000 smiles simulated
    now, to show the loss falling; the network used below is the one trained before the session on the
    full bank. This short software demonstration does not reproduce the baseline fit
    or establish inference quality.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    train_button = mo.ui.run_button(label="Train a small native BayesFlow 2.1 network explicitly")
    train_button
    return (train_button,)


@app.cell
def _(
    full_surface,
    heston_task,
    make_demo_simulator,
    mo,
    np,
    plots,
    plt,
    time,
    train_button,
):
    mo.stop(not train_button.value, mo.md("*Press the button to simulate 1,000 smiles and train for 15 epochs.*"))
    _bank_small = make_demo_simulator().sample(1000)
    _small = full_surface.build_workflow(heston_task, family="flow_matching", widths=(256, 256, 256, 256))
    _bank_small = full_surface.training_data({"observations": _bank_small["observations"],
        "parameters": np.column_stack([_bank_small[n].reshape(-1) for n in heston_task["parameter_names"]])}, heston_task)
    _start = time.perf_counter()
    _history = _small.fit_offline(data=_bank_small, epochs=15, batch_size=64, verbose=0)
    _fig, _ax = plt.subplots(figsize=(6.0, 3.0), constrained_layout=True)
    _ax.plot(np.arange(1, 16), _history.history["loss"], color=plots.SERIES[0], marker="o", markersize=3)
    _ax.set_xlabel("epoch")
    _ax.set_ylabel("flow-matching loss")
    _ax.set_title(f"15 epochs on 1,000 smiles, {time.perf_counter() - _start:.0f} s", loc="left")
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    The historical studies below evaluate their original estimator. Their diagnostics do not
    validate the synthetic example above or the small network trained during this session.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Historical study · calibration and reference checks

    A trained network can be wrong in two ways: **biased**, with the posterior in the wrong place, or
    **miscalibrated**, with a posterior too narrow or too wide. Both can be tested on simulated smiles whose
    parameters are known. These figures were computed for the archived checkpoint on 2,000 held-out
    test smiles (study S2); they are not a new evaluation of the current synthetic estimator above.
    """)
    return


@app.cell(hide_code=True)
def _(RESULTS, mo, teaching):
    _v = teaching.load_json(RESULTS / "heston" / "s2" / "validation.json")
    _cover = _v["coverage"]["0.95"]["coverage"]
    mo.vstack([
        mo.image(src=RESULTS / "heston" / "s2" / "sbc.png",
                 caption="Simulation-based calibration: rank ECDF minus uniform, with the simultaneous 95% band."),
        mo.md(f"""
        SBC passes for **{sum(_v['sbc']['passed'])} of {len(_v['sbc']['passed'])}** parameters on
        {_v['surfaces']:,} smiles, and the 95% intervals contain the truth on
        {100 * min(_cover):.1f}% to {100 * max(_cover):.1f}% of them. Logit-variance contraction, measured after mapping each parameter's
        prior box to the real line: """ + ", ".join(f"{teaching.LABELS[n]} {x:.2f}" for n, x in _v["contraction"].items()) + "."),
        mo.image(src=RESULTS / "heston" / "s2" / "recovery.png",
                 caption="Recovery: posterior median and 90% interval against the true value."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "**How to read the diagnostics** (click to open)": mo.md(r"""
        **Simulation-based calibration.** For each test smile, take the fraction of posterior draws below the
        true value: its rank. If the posterior is right, the truth is one more draw from it, so the ranks are
        uniform. A curve above zero means the posterior sits too high; a wave positive on the left and
        negative on the right means it is too narrow.

        **Recovery.** One dot per test smile: the true value against the posterior median, with its interval.
        Centers should track truth across cases, with the interval misses expected
        under their declared coverage; each dot need not lie on the diagonal.

        **Logit-variance contraction.** Normalize each parameter to its uniform prior box,
        apply the logit, then compare average within-case posterior variance with the prior
        logit variance $\pi^2/3$. This is a coordinate-specific measure of spread reduction,
        not a physical interval-width ratio or a standalone measure of inference accuracy.

        SBC averages over the prior predictive distribution. A numerical-reference comparison
        checks a particular posterior under the stated model, subject to the reference's own
        numerical and convergence limitations. Neither check establishes market-model adequacy.
        """)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The numerical reference check.** On 50 other test smiles the posterior was approximated using the Fourier likelihood
    (importance sampling on all 50, MCMC on 4), and the network compared with it. The criterion: means
    within 0.3 posterior standard deviations and 90% widths within 25%.
    """)
    return


@app.cell(hide_code=True)
def _(RESULTS, mo, teaching):
    _report = teaching.load_json(RESULTS / "heston" / "s3" / "heston-check.json")
    _chains = _report["summary"]["chain_checks"]
    mo.vstack([
        mo.ui.table(teaching.agreement_rows(_report), selection=None, pagination=False),
        mo.md(f"""
        The network alone meets the criterion for all four parameters on
        **{100 * _report['summary']['network_vs_reweighted_pass_rate']:.0f}%** of the smiles: close, but not
        exact. Reweighted by the exact likelihood, it agrees with the exact chain on
        **{sum(c['reweighted_vs_chain'] for c in _chains)} of {len(_chains)}** smiles. That correction is slot
        5's audit, with an exact likelihood.
        """),

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
    _notice = (mo.callout(mo.md(
        "**Why does the truth fall outside the posterior here?** Both the neural approximation and "
        "the numerical reference miss the generating vol-of-variance and correlation in their 95% "
        "marginal intervals. The saved parameters and quotes reproduce from the declared simulation "
        "seed. The quote errors have ordinary overall size, but the likelihood favors a shifted, "
        "strongly correlated parameter region. Thus the evidence is consistent with an unusual "
        "noise-induced miss, not a swapped truth marker. A 90% credible interval is not expected to "
        "contain the truth in every simulated case. This observation does not by itself explain "
        "the frequency of such misses or establish calibration. The importance correction is "
        "unstable for this case; it is not an independent guarantee of correctness."), kind="warn")
        if _row == 2001 else mo.md("**Read the comparison.** The dotted line is the known generating value; "
        "curves show uncertainty after observing this particular noisy data set. Compare methods in "
        "physical parameter units, and inspect the joint dependence as well as these marginals."))
    mo.vstack([_overview, mo.md(
        "The two path panels are new **illustrations at the generating parameters**, not recovered "
        "latent paths or inputs to the option calibration. Quotes were generated by Fourier "
        "expectations plus noise. The third panel shows the saved observation."), _fig, _notice,
        mo.accordion({"Numerical reference scope and source": mo.md(
            f"Archived synthetic case ID: `{_row}`. All four originally displayed cases remain available. "
            "The reference uses the Fourier likelihood and four Metropolis chains initialized from "
            "neural draws. Recorded mixing diagnostics are reassuring, but cannot exclude an "
            "unvisited mode. The original raw bank is unavailable; the published observations, truths "
            "and posterior arrays are preserved. For case 2001, truth is about 3.03 and 3.33 reference "
            "SDs below the vol-of-variance and correlation means; their posterior correlation is 0.887. "
            "The Pareto importance-weight diagnostic k-hat is 0.738, warranting caution.")}),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Historical study · inference on SPY

    The saved SPY reference was computed by MCMC on the Fourier
    likelihood: seven of eight optimizer starts over the prior box reach the same optimum, and four chains
    from dispersed starts agree. A matching current neural comparison and posterior re-pricing remain
    unavailable in this notebook; the cells below state that boundary explicitly.
    """)
    return


@app.cell
def _(ASSETS, mo, np):
    exact_spy = np.load(ASSETS / "data" / "heston-spy-exact.npy")
    network_spy = None
    network_seconds = None
    mo.md("**Historical market observation.** The saved Fourier-reference draws below concern SPY. "
          "A neural comparison for these same quotes is not included; the synthetic posterior above "
          "belongs to a different observation.")
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
            "mean gap, exact sd": round(float((network_spy[:, _j].mean() - exact_spy[:, _j].mean()) / exact_spy[:, _j].std()), 2),
            "width ratio": round(float(network_spy[:, _j].std() / exact_spy[:, _j].std()), 2),
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
    **And the classical route?** Least squares from several starts, with the Gauss–Newton interval at the
    optimum, on the same smile. A local Gaussian approximation can agree with the posterior
    when its mass is interior, unimodal and locally quadratic. Many liquid quotes alone do not
    guarantee those conditions; the comparison must be checked.
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
        mo.md("""
        Three routes, one answer. Here the network's value is not a different interval: it is the same interval
        for any smile in a second, with a check of its own calibration (SBC above). Where the answers part, with
        few quotes or inputs the quotes do not pin, is slot 1's first case and slot 5's long end.
        *Source of the Gauss–Newton column:* `lab/checks/heston_checks.py`.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Do the model prices fit the quotes?** Re-price 200 posterior draws and compare them with SPY's prices.
    This is a model-price check: it propagates parameter uncertainty without drawing fresh quote errors.
    The full posterior-predictive illustration above adds those errors to generate replicated observations.
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
    mo.vstack([_fig, mo.md(f"""
    Root-mean-square residual of the posterior-mean prices: **{np.sqrt(np.mean(residuals**2)):.2f} noise units**,
    with {int(np.sum(np.abs(residuals) <= 2))} of 63 quotes within two. The noise is loose in the wings at SPY's
    low volatility, so the fit looks better there than it is.
    """)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Historical study · model assumptions and derived quantities

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
    _days = np.asarray(surface.DESIGN_DAYS)
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
    - **The Feller condition fails on every draw**: the ratio runs from {feller_ratio.min():.1f} to
      {feller_ratio.max():.1f}, so the vol-of-vol is about {np.sqrt(np.median(feller_ratio)):.1f} times its
      Feller limit. The variance boundary is accessible; violating this sufficient positivity condition
      does not by itself invalidate Heston. Judge the product discrepancy and sensitivity checks separately.
    - **Heston's variance swaps sit below the market** by {-_gaps[1:].max():.1f} to {-_gaps[1:].min():.1f}
      volatility points from two weeks to one year, while the posterior's 90% band is less than one point wide.
      In Heston the variance-swap curve is tied to the smile parameters, and the replication uses strikes the
      63 quotes do not reach.

    Precise, and wrong for a product the smile does not pin down.
    """)])
    return


@app.cell(hide_code=True)
def _(heston_checks, mo, np, plots, plt):
    _free = heston_checks["cases"]["full grid, kappa free"]
    _fixed = heston_checks["cases"]["full grid, kappa fixed"]
    _kq = _free["parameters"]["kappa"]["posterior_q05_q50_q95"]
    _profile = heston_checks["kappa_profile"]
    _kap = np.array([p["kappa"] for p in _profile])
    _chi2 = np.array([p["chi2"] for p in _profile])
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
        mo.md(f"""
        **κ = 2 is a choice, and the smile disagrees with it.** Freed, the mean reversion is pinned by the
        smile: the exact posterior puts it at **{_kq[1]:.1f}**, from {_kq[0]:.1f} to {_kq[2]:.1f} (90%), and the fit
        improves from χ² {_fixed['best_chi2']:.1f} to {_free['best_chi2']:.1f} on 63 quotes. Fixing a parameter is a
        prior of zero width: the posterior above is conditional on it, and so is every product price.
        """),
        _fig,
        mo.ui.table(_shift_rows, selection=None, pagination=False),
        mo.md("Product prices in forward units, under the exact posteriors with κ fixed and free "
              "(`lab/checks/heston_checks.py`). The training bank of this notebook keeps κ = 2; a bank "
              "with κ inferred needs its own task, simulations and fitted estimator. Changing a task file "
              "alone does not make the existing four-target network infer κ. This stored check is independent "
              "of the unavailable SPY neural comparison."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    1. **State the inverse problem.** Observed quotes, unknown parameters, fixed
       assumptions and quote noise together define the posterior target.
    2. **Simulate pairs, then learn.** A few BayesFlow calls connect the financial
       simulator to a reusable conditional distribution. A small demonstration
       fit and the pretrained estimator are separate results.
    3. **Know what the neural network learns.** A surrogate learns prices and
       supports a new Bayesian computation for each observation; an amortized
       estimator learns to draw parameters conditional on observations.
    4. **Check uncertainty at several levels.** Recovery, coverage, SBC,
       numerical references and replicated quotes answer different questions.
       Agreement for one case does not establish accuracy for every observation.
    5. **Keep dependence and assumptions.** Carry paired posterior draws into
       derived quantities. A sharp answer is still conditional on the model,
       prior, noise and fixed parameters.

    **Next:** rough Bergomi keeps the option-inference problem and changes the
    variance dynamics. Its known-curve and joint-curve cases also make different
    conditioning assumptions; their selected synthetic observations are separate.

    *Evidence and code:* the [curriculum](../curriculum/04-slot4-heston-in-bayesflow.md); studies S1–S3
    in `assets/results/calibration/heston/`; the exact SPY posterior in `assets/data/` (a historical numerical reference); the Gauss–Newton and κ checks in `assets/quick-checks/` (quick checks, not yet accepted studies).
    """)
    return


if __name__ == "__main__":
    app.run()
