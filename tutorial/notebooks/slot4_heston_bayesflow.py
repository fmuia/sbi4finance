import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Slot 4 · Heston in BayesFlow")


@app.cell(hide_code=True)
def _():
    import os
    import time

    os.environ.setdefault("KERAS_BACKEND", "jax")

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from sbi4f_tutorial import full_surface, plots, surface, teaching

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
    # Slot 4 · A worked example in BayesFlow: Heston on the SPY smile

    For the new BayesFlow 2.1 estimator and checked synthetic interior example,
    open [the cached/live backbone review](backbone_review.py). That replay has
    its own observation and provenance; the historical SPY results below remain
    separate.

    **Follow the amortized workflow, then check its approximation against a numerical reference**

    Slot 1 showed why a bank wants the posterior, and that on a liquid surface the classical interval
    should agree with it. Black–Scholes cannot fit the smile, so the tutorial uses a model that can. This notebook
    follows **Heston** on the same 63-price design, step by step. Its Fourier pricer makes a likelihood-based
    numerical reference affordable. The current 2.1 example below is synthetic and has its own PyMC reference;
    the later SPY figures and studies retain their historical checkpoint identities. The matching 2.1 neural
    posterior for SPY has not been packaged here.

    Three claims, each backed by evidence below:

    1. **The BayesFlow workflow is a few objects**: a simulator, an adapter, a network, a training call,
       diagnostics. Five calls take a prior and a pricer to a posterior.
    2. **Checks answer different questions.** The current case permits a direct neural/reference comparison.
       The historical studies separately illustrate calibration across simulated smiles and correction with
       a likelihood. Their outcomes do not validate the new checkpoint or the small optional training run.
    3. **The posterior also judges the model.** On SPY, Heston needs a vol-of-vol about twice its Feller
       limit, prices variance swaps below the market, and prefers a mean reversion other than the fixed
       one. That is why slot 5 changes the model.
    """)
    return




@app.cell
def _(mo):
    prior_button = mo.ui.run_button(label="Simulate 100 Heston prior-predictive smiles")
    prior_button
    return (prior_button,)


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

    backbone_case = mo.ui.dropdown(options=backbones.examples("heston"),
                                    value="heston-final500", label="Current cached example")
    backbone_case
    return backbone_case, backbones


@app.cell
def _(backbone_case, backbones, mo):
    _task, _report, _arrays = backbones.load_example(backbone_case.value)
    mo.vstack([
        mo.callout(mo.md(f"**Development example:** `{_report['case_id']}`; "
                         f"{len(_arrays['neural_draws']):,} paired joint draws. "
                         f"{'PyMC reference supplied' if 'reference_chains' in _arrays else 'conditional reference and rough products pending'}. "
                         "The selected geometry examples illustrate trade-offs; they are not an accuracy study."), kind="neutral"),
        mo.md(f"Task SHA256: `{_report['task_fingerprint']}`  \nModel SHA256: `{_report['checkpoint_sha256']}`"),
        mo.ui.table(backbones.parameter_rows(_task, _arrays), selection=None, pagination=False),
        backbones.corner(_task, _arrays),
        mo.md("**Interpret the current case.** Choose one parameter pair with visible dependence. "
              "Compare the neural and PyMC shapes as well as their marginal widths. What can this one "
              "comparison establish, and what still needs a collection of independent observations?"),
    ])
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Historical SPY workflow and studies

    The sections below retain the SPY observation, archived training metadata and study figures.
    Simulator and small-fit buttons use the current runtime as software demonstrations. They do not
    reproduce the archived full fit or transfer its validation results to the current synthetic example.

    | | Slot 1: Black–Scholes | Slot 4: Heston |
    |:---|:---|:---|
    | Data | two SPY quotes | the SPY smile: 63 prices on the design grid |
    | Unknowns | the volatility σ | $v_0$, $\theta$, $\sigma_v$, $\rho$ ($\kappa = 2$ fixed) |
    | Pricer | closed form | a Fourier integral, 20 ms per smile |
    | Posterior | a grid | a flow-matching network, checked against an exact Markov chain |
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

        For Heston this costs 20 ms, so an exact posterior by MCMC takes minutes. The network never uses it;
        the exact chain does, as the check.
        """)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Step 1: a simulator that draws

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
    ## 2 · Step 2: the bank, the adapter, the network

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
    ## 3 · Step 3: train

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
    The archived SPY checkpoint was trained with BayesFlow 2.0. It is retained as evidence, and is not loaded in this BayesFlow 2.1 environment. The current Heston example above and backbone review use the native 2.1 model with its own exact observation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Step 4: check before you trust · historical studies

    A trained network can be wrong in two ways: **biased**, with the posterior in the wrong place, or
    **miscalibrated**, with a posterior too narrow or too wide. Both can be tested on simulated smiles whose
    parameters are known. These figures were computed for the archived checkpoint on 2,000 held-out
    test smiles (study S2); they are not a new evaluation of the 2.1 development checkpoint above.
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
        {100 * min(_cover):.1f}% to {100 * max(_cover):.1f}% of them. Contraction, the share of the prior's
        variance the smile removes: """ + ", ".join(f"{teaching.LABELS[n]} {x:.2f}" for n, x in _v["contraction"].items()) + "."),
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
        Accurate posteriors put the dots on the diagonal.

        **Contraction.** $1 - \text{posterior variance} / \text{prior variance}$: 0 means the smile taught
        nothing about the parameter, 1 that it pins it down.

        SBC holds on average over the prior. It does not prove that a single posterior is right, which is
        what the exact check below does.
        """)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The exact check.** On 50 other test smiles the exact posterior was computed with the exact likelihood
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
        mo.image(src=RESULTS / "heston" / "s3" / "agreement.png",
                 caption="Four test smiles: the network (blue) and the network reweighted with the exact likelihood (green) "
                         "against the exact posterior (grey), standardized by the exact posterior; the star is the truth."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Step 5: use it on SPY

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
    mo.md("**SPY neural update pending:** the saved Fourier-reference draws below remain historical. "
          "No new SPY neural posterior is fabricated from the synthetic development replay.")
    return exact_spy, network_seconds, network_spy


@app.cell
def _(CHECKS, teaching):
    heston_checks = teaching.load_json(CHECKS / "heston_checks.json")
    return (heston_checks,)


@app.cell(hide_code=True)
def _(NAMES, exact_spy, mo, network_seconds, network_spy, teaching):
    mo.stop(network_spy is None, mo.md("Historical SPY neural comparison awaits an explicitly validated BF2.1 export."))
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
    optimum, on the same smile. On a liquid surface it should agree with the posterior, because the
    likelihood of 63 informative quotes is close to Gaussian in the parameters (slot 1, section 3).
    """)
    return


@app.cell(hide_code=True)
def _(NAMES, exact_spy, heston_checks, mo, network_spy, np, teaching):
    mo.stop(network_spy is None, mo.md("Historical SPY neural comparison awaits an explicitly validated BF2.1 export."))
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
    **Does the posterior fit the quotes?** Re-price 200 posterior draws and compare them with SPY's prices:
    the posterior predictive check.
    """)
    return


@app.cell
def _(NOISE, heston_task, mo, network_spy, np, spy_prices, teaching):
    mo.stop(network_spy is None, mo.md("Historical SPY neural comparison awaits an explicitly validated BF2.1 export."))
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
    ## 6 · What the posterior says about Heston

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

    1. **The workflow is five objects**: a simulator that draws, an adapter, a flow-matching network, one
       training call, and diagnostics.
    2. **Keep evidence attached to its estimator.** Historical SBC and likelihood-correction studies retain
       their archived checkpoint identity. The current synthetic 2.1 case has a separate PyMC comparison;
       its SPY neural update remains pending.
    3. **Compare approximation and reference, then interpret the model.** Agreement on one development
       case does not certify all observations. The historical liquid-surface checks show where a local
       Gaussian approximation can agree with a numerical posterior.
    4. **The posterior judges the model.** Heston fits SPY only with a vol-of-vol twice its Feller limit,
       its variance swaps disagree with the market, and the smile prefers a mean reversion near 3.8 to the
       fixed 2.

    **Next, slot 5:** keep the data and the code, change the simulator.

    | | Slot 4 | Slot 5 |
    |:---|:---|:---|
    | Model | Heston: 4 parameters | rough Bergomi: $H$, $\eta$, $\rho$, with the market's forward-variance curve as input |
    | One smile priced by | a Fourier integral, 20 ms | Monte Carlo, 5 CPU-seconds |
    | The bank | 60,000 smiles in 46 CPU-minutes | 112,000 smiles in about 370 CPU-hours, on a cluster |
    | The final check | the exact posterior by MCMC | an audit of each smile with a Monte Carlo likelihood |
    | On SPY | extreme parameters, variance swaps below the market | a power-law skew, variance swaps that match by construction |

    *Evidence and code:* the [curriculum](../curriculum/04-slot4-heston-in-bayesflow.md); studies S1–S3
    in `assets/results/calibration/heston/`; the exact SPY posterior in `assets/data/` (a feasibility run, to be
    redone as study S8-H); the Gauss–Newton and κ checks in `assets/quick-checks/` (quick checks, not yet accepted studies).
    """)
    return


if __name__ == "__main__":
    app.run()
