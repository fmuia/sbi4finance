import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="Slot 4 · Heston on the SPY smile, in BayesFlow",
)


@app.cell(hide_code=True)
def _():
    import os
    import time

    os.environ.setdefault("KERAS_BACKEND", "jax")

    import bayesflow as bf
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from sbi4finance.calibration import inference, plots, surface, teaching

    ROOT = mo.notebook_dir().parent
    RESULTS = ROOT / "results" / "calibration"
    CHECKPOINTS = ROOT / "checkpoints"
    CURRICULUM = ROOT / "curriculum"
    plots.style()
    return (
        CHECKPOINTS,
        CURRICULUM,
        RESULTS,
        bf,
        inference,
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

    **The whole amortized workflow, on a model whose exact posterior can check every step**

    Slot 1 asked how sure a calibration is, and answered it for Black–Scholes on a grid. Black–Scholes
    cannot fit the smile, so desks use a model that can. This notebook calibrates **Heston** to the same
    63 SPY prices with BayesFlow, step by step. Heston is priced by a Fourier integral in 20 ms, so its
    exact posterior can be computed by MCMC and every step of the network can be compared with it.

    Three claims, each backed by evidence below:

    1. **The BayesFlow workflow is a few objects**: a simulator, an adapter, a network, a training call,
       diagnostics. Five calls take a prior and a pricer to a posterior.
    2. **The checks earn the trust.** The network passes simulation-based calibration on 2,000 held-out
       smiles and, on SPY, agrees with the exact posterior.
    3. **The posterior also judges the model.** On SPY, Heston needs a vol-of-vol about twice its Feller
       limit, and it prices variance swaps below the market. That is why slot 5 changes the model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
def _(CHECKPOINTS, bf, np, teaching):
    heston_task = teaching.load_json(CHECKPOINTS / "heston" / "task.json")
    NAMES = heston_task["parameter_names"]
    BOUNDS = np.asarray(heston_task["prior_bounds"])
    NOISE = np.asarray(heston_task["context"]["quote_noise_std"])
    _rng = np.random.default_rng(4)


    def prior():
        return dict(zip(NAMES, _rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1])))


    def smile(initial_variance, theta, vol_of_vol, rho):
        prices = teaching.heston_smile(heston_task, [initial_variance, theta, vol_of_vol, rho])
        return {"observations": prices + NOISE * _rng.standard_normal(len(NOISE))}


    simulator = bf.make_simulator([prior, smile])
    return NAMES, NOISE, heston_task, simulator


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The prior predictive check comes first.** Draw smiles from the prior and look at them next to SPY's.
    If SPY's smile lay outside the cloud, the network would be asked to extrapolate.
    """)
    return


@app.cell
def _(RESULTS, np, simulator, surface, teaching, time):
    _start = time.perf_counter()
    prior_draws = simulator.sample(100)
    prior_seconds = time.perf_counter() - _start
    prior_iv = np.array([teaching.implied_vol_grid(p) for p in prior_draws["observations"]])
    spy = teaching.load_json(RESULTS / "spy" / "surface.json")
    spy_prices = np.asarray(spy["prices_forward_units"])
    spy_iv = teaching.implied_vol_grid(spy_prices)
    _k, _maturity, _kind, grid_delta = teaching.design_arrays()
    grid_days = np.rint(_maturity * surface.TRADING_DAYS).astype(int)
    return (
        grid_days,
        grid_delta,
        prior_iv,
        prior_seconds,
        spy,
        spy_iv,
        spy_prices,
    )


@app.cell(hide_code=True)
def _(
    grid_days,
    grid_delta,
    mo,
    np,
    plots,
    plt,
    prior_iv,
    prior_seconds,
    spy_iv,
    teaching,
):
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
    mo.vstack([_fig, mo.md(f"100 prior smiles (blue) in {prior_seconds:.1f} s, and SPY (orange).")])
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
    workflow = inference.build_workflow(heston_task, time_power_law_alpha=-0.5)   # slot 3, section 3
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
    full bank.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    train_button = mo.ui.run_button(label="Train a small network live (about a minute)")
    train_button
    return (train_button,)


@app.cell
def _(
    heston_task,
    inference,
    mo,
    np,
    plots,
    plt,
    simulator,
    time,
    train_button,
):
    mo.stop(not train_button.value, mo.md("*Press the button to simulate 1,000 smiles and train for 15 epochs.*"))
    _bank_small = simulator.sample(1000)
    _small = inference.build_workflow(heston_task, time_power_law_alpha=-0.5)
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
def _(CHECKPOINTS, teaching):
    _task, heston_workflow = teaching.load_network(CHECKPOINTS / "heston")
    heston_workflow  # the network trained before the session, 500 epochs on 50,000 smiles
    return (heston_workflow,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Step 4: check before you trust

    A trained network can be wrong in two ways: **biased**, with the posterior in the wrong place, or
    **miscalibrated**, with a posterior too narrow or too wide. Both can be tested on simulated smiles whose
    parameters are known. The figures were computed on 2,000 held-out test smiles (study S2).
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

    One call gives SPY's posterior. The exact posterior beside it was computed by MCMC on the Fourier
    likelihood: seven of eight optimizer starts over the prior box reach the same optimum, and four chains
    from dispersed starts agree.
    """)
    return


@app.cell
def _(CURRICULUM, heston_task, heston_workflow, inference, np, spy, time):
    spy_conditions = np.asarray(spy["prices_forward_units"])[None]
    inference.sample(heston_workflow, heston_task, spy_conditions, None, 50, seed=0)  # compiles the sampler once
    _start = time.perf_counter()
    network_spy = inference.sample(heston_workflow, heston_task, spy_conditions, None, 2000, seed=1)[0]
    network_seconds = time.perf_counter() - _start
    exact_spy = np.load(CURRICULUM / "data" / "heston-spy-exact.npy")
    return exact_spy, network_seconds, network_spy


@app.cell(hide_code=True)
def _(NAMES, exact_spy, mo, network_seconds, network_spy, teaching):
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
    **Does the posterior fit the quotes?** Re-price 200 posterior draws and compare them with SPY's prices:
    the posterior predictive check.
    """)
    return


@app.cell
def _(NOISE, heston_task, network_spy, np, spy_prices, teaching):
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

    The posterior is precise. Two of its consequences are not what the market says.
    """)
    return


@app.cell
def _(exact_spy, np, surface, teaching):
    feller_ratio = exact_spy[:, 2] ** 2 / (2 * 2.0 * exact_spy[:, 1])
    variance_swap_vol = teaching.heston_variance_swap_vol(exact_spy[:, 0], exact_spy[:, 1], 2.0, surface.DESIGN_DAYS)
    SPY_VARIANCE_SWAP_VOL = np.array([13.6, 15.0, 16.2, 17.6, 18.4, 20.1, 21.8])  # replicated from the chain (design document, S8)
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
      Feller limit. Heston produces SPY's short skew only with a variance that keeps touching zero.
    - **Heston's variance swaps sit below the market** by {-_gaps[1:].max():.1f} to {-_gaps[1:].min():.1f}
      volatility points from two weeks to one year, while the posterior's 90% band is less than one point wide.
      In Heston the variance-swap curve is tied to the smile parameters, and the replication uses strikes the
      63 quotes do not reach.

    Precise, and wrong for a product the smile does not pin down.
    """)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    1. **The workflow is five objects**: a simulator that draws, an adapter, a flow-matching network, one
       training call, and diagnostics.
    2. **Check before you trust.** SBC on 2,000 smiles passes; smile by smile the network is close to the
       exact posterior, and on SPY it agrees with it.
    3. **The posterior judges the model.** Heston fits SPY only with a vol-of-vol twice its Feller limit,
       and its variance swaps disagree with the market.

    **Next, slot 5:** keep the data and the code, change the simulator.

    | | Slot 4 | Slot 5 |
    |:---|:---|:---|
    | Model | Heston: 4 parameters | rough Bergomi: $H$, $\eta$, $\rho$, with the market's forward-variance curve as input |
    | One smile priced by | a Fourier integral, 20 ms | Monte Carlo, 5 CPU-seconds |
    | The bank | 60,000 smiles in 46 CPU-minutes | 112,000 smiles in about 370 CPU-hours, on a cluster |
    | The final check | the exact posterior by MCMC | an audit of each smile with a Monte Carlo likelihood |
    | On SPY | extreme parameters, variance swaps below the market | a power-law skew, variance swaps that match by construction |

    *Evidence and code:* the [curriculum](../curriculum/04-slot4-heston-in-bayesflow.md); studies S1–S3
    in `results/calibration/heston/`; the exact SPY posterior in `curriculum/data/` (a feasibility run, to be
    redone as study S8-H).
    """)
    return


if __name__ == "__main__":
    app.run()
