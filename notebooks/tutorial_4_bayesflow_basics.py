import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Slot 4 · Your first amortized posterior")


@app.cell(hide_code=True)
def _():
    import json
    import os
    import time
    import warnings

    os.environ.setdefault("KERAS_BACKEND", "jax")

    import bayesflow as bf
    import keras
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from sbi4finance import gbm_volatility as gv
    from sbi4finance.svj import data, plots

    ROOT = mo.notebook_dir().parent
    CHECKPOINTS = ROOT / "checkpoints" / "tutorial_gbm"
    RETRAIN = os.environ.get("SBI4FINANCE_TUTORIAL_RETRAIN") == "1"  # regenerates the saved networks
    plots.style()
    return CHECKPOINTS, RETRAIN, ROOT, bf, data, gv, json, keras, mo, np, plots, plt, time, warnings


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Slot 4 · Your first amortized posterior

    **Volatility from 1,000 daily returns, in BayesFlow**

    Every morning a risk desk needs three things for each of thousands of names: a volatility estimate,
    how sure it is, and a value-at-risk that accounts for both. Bayes' rule delivers all three, but one
    name at a time: a fresh posterior computation for every name, every day. **Amortized inference**
    trains a network once, on simulated data, and from then on answers for any name in a single
    forward pass.

    This notebook is about the **software**. We use the simplest volatility model there is, constant
    volatility, because its exact posterior is known: every step of the BayesFlow workflow can be
    checked against the truth. In slot 5 the code stays, the model becomes realistic, and the exact
    answer disappears.

    | Step | BayesFlow | What it does |
    |---|---|---|
    | 1 | `make_simulator` | how the data are generated: no likelihood needed |
    | 2 | `Adapter` | which arrays are data and which are unknowns |
    | 3 | `FlowMatching` (+ a summary network) | turns noise into posterior draws, given the data |
    | 4 | `BasicWorkflow.fit_online` | trains once, on fresh simulations |
    | 5 | `plot_default_diagnostics` | checks before you trust |
    | 6 | `sample` | the posterior for any new series, in one pass |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "**The model in detail** (click to open): from prices to posterior and risk": mo.md(r"""
        #### 1 · From prices to returns

        The model is geometric Brownian motion, the price process behind Black–Scholes:

        $$\frac{dS_t}{S_t} = m\,dt + \sigma\,dW_t .$$

        By Itô's lemma the log price is a Brownian motion with drift $m - \sigma^2/2$, so log returns
        are normal and independent across days. Over one trading day (a year has 252), in percent:

        $$y_t = 100\,\log\frac{S_t}{S_{t-1}} = \frac{\mu}{252} + \frac{100\,\sigma}{\sqrt{252}}\,\varepsilon_t,
        \qquad \varepsilon_t \overset{\text{iid}}{\sim} \mathcal N(0, 1),$$

        where $\mu = 100\,(m - \sigma^2/2)$ is the drift of the **log** price, in percent a year. It is
        what returns measure directly, and it differs from $m$ by $\sigma^2/2$: about 3% a year at 25%
        volatility.

        #### 2 · The two unknowns

        | Symbol | In the code | Meaning | Units |
        |:---|:---|:---|:---|
        | $\mu$ | `drift` | expected log return | % a year |
        | $\log\sigma$ | `log_vol` | log of the annualized volatility | $\sigma = 0.25$ means 25% a year |

        One day's return has mean $\mu/252$ and standard deviation $100\,\sigma/\sqrt{252}$, about 1.6%
        at $\sigma = 25\%$. The network learns $\log\sigma$ rather than $\sigma$: it is unconstrained, its
        prior is symmetric, and its posterior is close to normal, which is what a flow learns most easily.

        #### 3 · The prior

        $$\mu \sim \mathcal N(8,\ 10^2), \qquad \log\sigma \sim \mathcal N(\log 0.25,\ 0.40^2),
        \qquad \text{independent.}$$

        A drift of 8% a year, 95% between −12% and +28%. A volatility with median 25%, 95% between 11%
        and 55%: wide enough for a quiet index and for a volatile single stock. These are the first two
        coordinates of slot 5's prior, so both notebooks ask the same question.

        #### 4 · The likelihood, and why two numbers are enough

        Write $m_d = \mu/252$ and $s_d = 100\,\sigma/\sqrt{252}$ for the daily mean and standard deviation.
        For $T = 1{,}000$ returns with sample mean $\bar y$ and sample standard deviation $\hat s$,

        $$\log p(y \mid \mu, \sigma) = -T\log s_d - \frac{T\,\hat s^{\,2} + T\,(\bar y - m_d)^2}{2\,s_d^2} + \text{const}.$$

        The 1,000 returns enter only through $(\bar y, \hat s)$. These two numbers are **sufficient**: a
        method that sees them and nothing else loses no information. Part A hands the network exactly
        these two numbers; Part B asks a network to find them in the raw series.

        #### 5 · The exact posterior

        $$p(\mu, \log\sigma \mid y) \;\propto\; p(y \mid \mu, \sigma)\; p(\mu)\; p(\log\sigma).$$

        With this prior there is no textbook closed form, and in two dimensions none is needed:
        `gbm_volatility.exact_posterior` evaluates the right-hand side on a 201 × 201 grid centered on the
        data and normalizes it. Every BayesFlow answer in this notebook is checked against that grid.

        #### 6 · What 1,000 days can and cannot tell you

        With this much data the posterior is close to normal, and its spread follows from the Fisher
        information:

        * **Volatility is pinned down.** $\operatorname{sd}(\log\sigma \mid y) \approx 1/\sqrt{2T} = 0.022$:
          volatility is known to about 2% of its value, 25% ± 0.6 points.
        * **The drift is not.** On their own, the data give $\operatorname{sd}(\mu) \approx
          100\,\sigma\sqrt{252/T}$, about 12.5% a year at 25% volatility. That is wider than the prior's
          10%, so the prior keeps about 60% of the weight and the posterior sd is about 7.8% a year. The
          precision of a drift estimate grows with the calendar span, as $100\,\sigma/\sqrt{\text{years}}$,
          not with the number of observations: sampling more often does not help (Merton, 1980).

        So expect tight volatility posteriors and drift posteriors close to the prior. That is the model
        speaking, not a weak network.

        #### 7 · From parameters to risk

        Given the parameters, the $h$-day log return is normal, $R_h \sim \mathcal N(h\,m_d,\ h\,s_d^2)$.
        Risk is quoted as a loss, $-R_h$, in percent:

        * **99% VaR**, the loss exceeded with probability 1%: $-q_{0.01}(R_h)$.
        * **97.5% ES**, the average loss in the worst 2.5% of outcomes:
          $-\mathbb E[R_h \mid R_h \le q_{0.025}(R_h)]$.

        For one normal, $\text{VaR} = -h\,m_d + 2.326\,\sqrt{h}\,s_d$ and
        $\text{ES} = -h\,m_d + 2.338\,\sqrt{h}\,s_d$. The two nearly coincide, which is why Basel's FRTB
        could replace 99% VaR with 97.5% ES.

        A **plug-in** forecast inserts one estimate of $(\mu, \sigma)$. The **predictive** forecast
        averages over the posterior:

        $$p(R_h \mid y) = \int \mathcal N\!\left(R_h;\ h\,m_d,\ h\,s_d^2\right) p(\mu, \sigma \mid y)\;
        d\mu\, d\sigma .$$

        This is a mixture of normals, with heavier tails than any one of them: uncertainty about the
        parameters is itself a risk. With 1,000 days of history the difference is small; with a few weeks
        it is not, as the slider in *From posterior to risk* shows.

        #### 8 · What the model leaves out, on purpose

        Real returns break the model in three ways. Volatility **clusters**: calm and turbulent periods
        alternate. Large moves arrive as **jumps**, so tails are fatter than normal. And volatility tends
        to rise when prices fall, the **leverage** effect. The VOO section at the end shows the first
        failure on real data. Slot 5 replaces this model with stochastic volatility and jumps, and keeps
        the BayesFlow code.

        #### 9 · What the simulator returns

        | Key | Content | Used by |
        |:---|:---|:---|
        | `returns` | the daily returns $y_1, \dots, y_{1000}$ | Part B's summary network |
        | `log_sq` | $\log(y_t^2 + 0.01)$ | Part B's summary network |
        | `stats` | $(\bar y,\ \log\hat s)$, the sufficient statistics | Part A's network |

        Why log squares: since $y_t \approx s_d\,\varepsilon_t$, we have
        $\log y_t^2 \approx 2\log s_d + \log\varepsilon_t^2$, so volatility becomes a shift in level that a
        network can read off. The 0.01 keeps the logarithm finite on days with almost no move.
        """),
    })
    return


@app.cell
def _(ROOT, data):
    voo = data.window(data.load_returns(ROOT / "data" / "SPMO_VOO_2020-01-01_2026-09-04.csv", "VOO"))
    return (voo,)


@app.cell(hide_code=True)
def _(mo, plots, plt, voo):
    _fig, _ax = plt.subplots(figsize=(9.6, 2.6), constrained_layout=True)
    _ax.plot(voo.index, voo.to_numpy(), color=plots.SERIES[0], linewidth=0.7)
    _ax.set_ylabel("daily log return (%)")
    _ax.set_title(f"VOO, an S&P 500 ETF: the last 1,000 trading days, to {voo.index[-1]:%d %b %Y}", loc="left")
    _ax.grid(axis="x", visible=False)
    mo.vstack([
        mo.md(r"""
        ## The question

        Here are 1,000 daily returns of a real fund. **What is its volatility, how sure can we be, and
        what is its 10-day 99% value-at-risk?** We build the answer step by step and return to VOO at
        the end.
        """),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1 · Write down how the data are generated

    The simplest volatility model: daily log returns, in percent, are independent normals with an
    annual drift $\mu$ and an annual volatility $\sigma$,

    $$y_t = \frac{\mu}{252} + \frac{100\,\sigma}{\sqrt{252}}\;\varepsilon_t,
    \qquad \varepsilon_t\sim\mathcal N(0,1),\quad t = 1,\dots,1000.$$

    The prior is stated in units a risk manager can judge, the same as slot 5's prior for these two
    quantities: a drift of 8% a year give or take 10%, and a volatility around 25%, with a 95% range of
    roughly 11% to 55%.

    BayesFlow never asks for a likelihood. It needs two plain functions that **draw**: one from the
    prior, one from the model given the parameters. The simulator may return more than the raw data.
    Here it also returns the log squared returns, where volatility is easier to see, and two summary
    numbers we will use first.
    """)
    return


@app.cell
def _(bf, np):
    def make_simulator(seed):
        rng = np.random.default_rng(seed)

        def prior():
            return {"drift": rng.normal(8.0, 10.0), "log_vol": rng.normal(np.log(0.25), 0.40)}

        def observation(drift, log_vol):
            returns = drift / 252 + 100 * np.exp(log_vol) / np.sqrt(252) * rng.standard_normal(1_000)
            return {
                "returns": returns,                                          # the data
                "log_sq": np.log(returns**2 + 0.01),                         # the same data, volatility made visible
                "stats": np.array([returns.mean(), np.log(returns.std())]),  # two numbers: mean and log sd
            }

        return bf.make_simulator([prior, observation])

    simulator = make_simulator(seed=2026)
    return make_simulator, simulator


@app.cell
def _(np):
    def conditions(returns):
        """Prepare any return series (simulated or real, in percent) exactly as the simulator does."""
        returns = np.atleast_2d(np.asarray(returns, dtype=float))
        return {
            "returns": returns,
            "log_sq": np.log(returns**2 + 0.01),
            "stats": np.stack([returns.mean(axis=1), np.log(returns.std(axis=1))], axis=1),
        }

    return (conditions,)


@app.cell
def _(make_simulator):
    prior_draws = make_simulator(seed=1).sample(2_000)
    return (prior_draws,)


@app.cell(hide_code=True)
def _(mo, np, plots, plt, prior_draws, voo):
    _realized = prior_draws["returns"].std(axis=1) * np.sqrt(252) / 100
    _voo_vol = voo.std() * np.sqrt(252) / 100
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(9.6, 3.0), constrained_layout=True,
                                         gridspec_kw={"width_ratios": [1.4, 1]})
    for _k in range(40):
        _left.plot(prior_draws["returns"][_k].cumsum(), color=plots.SERIES[0], alpha=0.35, linewidth=0.7)
    _left.set_xlabel("trading day")
    _left.set_ylabel("cumulative log return (%)")
    _left.set_title("40 simulated stocks from the prior", loc="left")
    _right.hist(100 * _realized, bins=np.linspace(0, 100, 51), color=plots.SERIES[0], alpha=0.85,
                edgecolor=plots.SURFACE, linewidth=0.5)
    _right.axvline(100 * _voo_vol, color=plots.SERIES[1], linewidth=1.5)
    _right.text(100 * _voo_vol + 2, 0.95, f"VOO {_voo_vol:.0%}", transform=_right.get_xaxis_transform(),
                color=plots.INK_2, fontsize=8, va="top")
    _right.set_xlabel("realized volatility (%)")
    _right.set_yticks([])
    _right.set_title("Realized volatility of 2,000 of them", loc="left")
    _right.grid(visible=False)
    mo.vstack([
        mo.md("**Check the prior before training: does it generate plausible stocks?**"),
        _fig,
        mo.md("Volatilities from about 10% to 60%, with VOO comfortably inside. The prior is wide, but not absurd."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Steps 2 and 3 · Tell BayesFlow what is data and what is unknown, and pick a network

    For this model two numbers carry **all** the information in the 1,000 returns: their mean and their
    standard deviation. Statisticians call them sufficient statistics. So the shortest path is to hand
    the network these two numbers directly, as conditions. The **adapter** routes each array to its role,
    and **flow matching** is the inference network: it learns to move standard normal noise to posterior
    draws, given the conditions.
    """)
    return


@app.cell
def _(bf, simulator):
    stats_adapter = (
        bf.Adapter()
        .drop(["returns", "log_sq"])                                      # this first network sees two numbers only
        .concatenate(["drift", "log_vol"], into="inference_variables")    # the unknowns
        .rename("stats", "inference_conditions")                          # what we condition on
        .convert_dtype("float64", "float32")
    )

    def make_stats_workflow():
        return bf.BasicWorkflow(
            simulator=simulator,
            adapter=stats_adapter,
            inference_network=bf.networks.FlowMatching(
                subnet_kwargs={"widths": (64, 64)}, integrate_kwargs={"method": "euler", "steps": 50}
            ),
            standardize=["inference_variables", "inference_conditions"],
        )

    stats_adapter
    return (make_stats_workflow,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4 · Train once, on fresh simulations

    `fit_online` draws a new batch from the simulator at every step, so the network never sees the same
    series twice and cannot overfit. 320,000 simulated stocks took about a minute on a 4-core machine.
    The network below was trained exactly this way before the session; press the button to train it live.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    train_stats_button = mo.ui.run_button(label="Train this network now (about a minute)")
    train_stats_button
    return (train_stats_button,)


@app.cell
def _(CHECKPOINTS, RETRAIN, gv, json, keras, make_stats_workflow, time, train_stats_button, warnings):
    stats_workflow = make_stats_workflow()
    if RETRAIN or train_stats_button.value:
        keras.utils.set_random_seed(2026)
        _start = time.time()
        _history = stats_workflow.fit_online(
            epochs=100, num_batches_per_epoch=50, batch_size=64, validation_data=512, verbose=0
        )
        stats_record = {"seconds": round(time.time() - _start, 1), "series": 100 * 50 * 64,
                        "loss": _history.history["loss"], "val_loss": _history.history["val_loss"]}
        if RETRAIN:
            CHECKPOINTS.mkdir(parents=True, exist_ok=True)
            gv.save_network(stats_workflow.approximator, CHECKPOINTS / "statistics.keras")
            (CHECKPOINTS / "statistics.json").write_text(json.dumps(stats_record, indent=1))
        stats_source = "trained just now"
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")
            stats_workflow.load_approximator(CHECKPOINTS / "statistics.keras")
        stats_record = json.loads((CHECKPOINTS / "statistics.json").read_text())
        stats_source = "trained before the session"
    return stats_record, stats_source, stats_workflow


@app.cell(hide_code=True)
def _(mo, np, plots, plt, stats_record, stats_source):
    _fig, _ax = plt.subplots(figsize=(9.6, 2.6), constrained_layout=True)
    _epochs = np.arange(1, len(stats_record["loss"]) + 1)
    _ax.plot(_epochs, stats_record["loss"], color=plots.SERIES[0], linewidth=1.2, label="training loss")
    _ax.plot(_epochs, stats_record["val_loss"], color=plots.SERIES[1], linewidth=1.2, label="validation loss")
    _ax.set_xlabel("epoch (3,200 fresh simulations each)")
    _ax.set_ylabel("flow-matching loss")
    _ax.set_title(f"Training: {stats_record['series']:,} simulated stocks in {stats_record['seconds']:.0f} s "
                  f"({stats_source})", loc="left")
    _ax.legend(loc="upper right", fontsize=8)
    mo.vstack([_fig, mo.md(
        "Training and validation losses agree: every batch is new, so there is nothing to overfit. "
        "The loss does not tell us whether the posterior is right. The next step does."
    )])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 5 · Check before you trust

    Simulate 300 new stocks whose drift and volatility we know, ask the network for each posterior, and
    compare. BayesFlow does this in one call:
    """)
    return


@app.cell
def _(make_simulator):
    test_set = make_simulator(seed=7).sample(300)
    return (test_set,)


@app.cell
def _(stats_workflow, test_set):
    stats_diagnostics = stats_workflow.plot_default_diagnostics(
        test_data=test_set, num_samples=500, variable_names=["drift (%/yr)", "log volatility"]
    )
    return (stats_diagnostics,)


@app.cell(hide_code=True)
def _(mo, stats_diagnostics):
    mo.vstack([
        mo.md("**Recovery:** posterior medians against the true values, with posterior spread."),
        stats_diagnostics["recovery"],
        mo.md(
            "**Calibration:** if the posterior is right, the true value's rank among the posterior draws is "
            "uniform, and the curve stays inside the grey band."
        ),
        stats_diagnostics["calibration_ecdf"],
        mo.md(
            "**Contraction:** how much of the prior's uncertainty the data remove (right is better), against "
            "how far the truth falls from the posterior mean, in posterior sd (centre is better)."
        ),
        stats_diagnostics["z_score_contraction"],
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two lessons in one picture. **Volatility is recovered almost perfectly**: 1,000 returns pin it down
    to within a few percent. **The drift is not**: the posterior is only somewhat narrower than the
    prior. Four years of returns say little about a stock's expected return, a classic result in
    finance (Merton, 1980). Both are correct answers, and the calibration panels say the network reports
    its uncertainty honestly.

    ### The ultimate check: the exact answer

    For this model we can compute the exact posterior on a grid, because the likelihood is known.
    Take a stock we invent, with a drift of 6% and a volatility of 22%:
    """)
    return


@app.cell
def _(conditions, gv, np, stats_workflow):
    stock_returns = gv.simulate_returns(6.0, np.log(0.22), 1_000, np.random.default_rng(42))
    stock_exact = gv.exact_posterior(stock_returns)
    _draws = stats_workflow.sample(conditions=conditions(stock_returns), num_samples=4_000)
    stock_stats_draws = np.concatenate([_draws["drift"][0], _draws["log_vol"][0]], axis=-1)
    return stock_exact, stock_returns, stock_stats_draws


@app.cell(hide_code=True)
def _(mo, np, plots, plt, stock_exact, stock_stats_draws):
    _w = stock_exact["weights"]
    _order = np.sort(_w.ravel())[::-1]
    _levels = [_order[np.searchsorted(np.cumsum(_order), _p)] for _p in (0.99, 0.9, 0.5)]
    _vol_axis = 100 * np.exp(stock_exact["log_vol"])
    _fig, _ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    _ax.scatter(stock_stats_draws[:, 0], 100 * np.exp(stock_stats_draws[:, 1]), s=4, alpha=0.25,
                color=plots.SERIES[0], linewidth=0, label="BayesFlow: 4,000 draws")
    _contours = _ax.contour(stock_exact["drift"], _vol_axis, _w.T, levels=_levels, colors=plots.INK, linewidths=1.0)
    _ax.clabel(_contours, fmt={_levels[0]: "99%", _levels[1]: "90%", _levels[2]: "50%"}, fontsize=7)
    _ax.plot([], [], color=plots.INK, linewidth=1.0, label="exact posterior")
    _mean, _sd = stock_exact["mean"], stock_exact["sd"]
    _ax.set_xlim(_mean[0] - 4.5 * _sd[0], _mean[0] + 4.5 * _sd[0])
    _ax.set_ylim(100 * np.exp(_mean[1] - 4.5 * _sd[1]), 100 * np.exp(_mean[1] + 4.5 * _sd[1]))
    _ax.set_xlabel("drift (% per year)")
    _ax.set_ylabel("volatility (%)")
    _ax.set_title("Network against the exact posterior", loc="left")
    _ax.legend(loc="upper right", fontsize=8)
    _rows = []
    for _j, _name in enumerate(("drift (%/yr)", "log volatility")):
        _diff = (stock_stats_draws[:, _j].mean() - stock_exact["mean"][_j]) / stock_exact["sd"][_j]
        _ratio = stock_stats_draws[:, _j].std() / stock_exact["sd"][_j]
        _rows.append(f"| {_name} | {stock_exact['mean'][_j]:.3f} ± {stock_exact['sd'][_j]:.3f} | "
                     f"{stock_stats_draws[:, _j].mean():.3f} ± {stock_stats_draws[:, _j].std():.3f} | "
                     f"{_diff:+.2f} | {_ratio:.2f} |")
    mo.hstack([
        _fig,
        mo.md("\n".join([
            "| | Exact: mean ± sd | Network: mean ± sd | Error in exact sd | Spread ratio |",
            "|---|---|---|---|---|",
            *_rows,
            "",
            "The criterion we will meet again in slot 5: mean error below **0.3 posterior sd** and a "
            "spread within 25% of the exact one. There, the exact answer comes from particle MCMC.",
        ])),
    ], widths=[1.1, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 6 · Use it: the amortization payoff

    Training is paid once. After that, a posterior costs one pass through the network, for any number
    of stocks at once:
    """)
    return


@app.cell
def _(conditions, make_simulator, stats_workflow, time):
    _many = make_simulator(seed=11).sample(100)["returns"]
    for _batch in (_many[:1], _many):  # compile once per batch shape, as a desk would at start-up
        stats_workflow.sample(conditions=conditions(_batch), num_samples=1_000)
    _start = time.time()
    stats_workflow.sample(conditions=conditions(_many[:1]), num_samples=1_000)
    one_name_seconds = time.time() - _start
    _start = time.time()
    stats_workflow.sample(conditions=conditions(_many), num_samples=1_000)
    many_names_seconds = time.time() - _start
    return many_names_seconds, one_name_seconds


@app.cell(hide_code=True)
def _(many_names_seconds, mo, one_name_seconds):
    mo.md(f"""
    | Task | Time on this machine |
    |---|---|
    | posterior of one stock, 1,000 draws | **{one_name_seconds:.2f} s** |
    | posteriors of 100 stocks at once, 1,000 draws each | **{many_names_seconds:.1f} s**, {many_names_seconds / 100:.2f} s per stock |

    Honesty check: here the exact grid is also fast. Enjoy it, because it is the last time. With
    stochastic volatility and jumps (slot 5), the exact posterior costs about **35 CPU-minutes per
    name**, and the network still answers in seconds.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From posterior to risk

    A forecast should average over what we do not know. Push each posterior draw forward 10 days and
    pool the outcomes: that is the **predictive** distribution. The shortcut, the **plug-in** forecast,
    fixes the parameters at a point estimate. Losses are percent log losses: minus the 10-day log return.
    """)
    return


@app.cell
def _(gv, np, stock_exact, stock_stats_draws):
    _median = np.median(stock_stats_draws, axis=0)
    stock_risk = {
        "predictive, network": gv.predictive_var_es(stock_stats_draws[:, 0], stock_stats_draws[:, 1], horizon=10),
        "predictive, exact": gv.exact_predictive_var_es(stock_exact, horizon=10),
        "plug-in at the posterior median": gv.predictive_var_es(_median[0], _median[1], horizon=10),
    }
    return (stock_risk,)


@app.cell(hide_code=True)
def _(mo, stock_risk):
    _rows = "\n".join(f"| {_name} | {_var:.2f}% | {_es:.2f}% |" for _name, (_var, _es) in stock_risk.items())
    mo.md("\n".join([
        "| 10-day forecast for the invented stock | 99% VaR | 97.5% ES |",
        "|---|---|---|",
        _rows,
        "",
        "Practically identical. With two parameters and 1,000 days, parameter uncertainty is a rounding "
        "error. **When does it matter?** When data are scarce. Move the slider to shorten the history: "
        "the exact posterior widens, and the plug-in forecast starts to understate the risk.",
    ]))
    return


@app.cell(hide_code=True)
def _(mo):
    window_slider = mo.ui.slider(steps=[20, 40, 60, 120, 250, 500, 1000], value=60, label="days of history")
    window_slider
    return (window_slider,)


@app.cell
def _(gv, np):
    window_lengths = (20, 40, 60, 120, 250, 500, 1000)
    _rng = np.random.default_rng(3)
    _gaps = {}
    for _n in window_lengths:
        _ratios = []
        for _ in range(25):
            _returns = gv.simulate_returns(6.0, np.log(0.22), _n, _rng)
            _post = gv.exact_posterior(_returns, n_grid=81)
            _, _es_pred = gv.exact_predictive_var_es(_post, horizon=10)
            _, _es_plug = gv.predictive_var_es(_post["mean"][0], _post["mean"][1], horizon=10)
            _ratios.append(_es_plug / _es_pred - 1)
        _gaps[_n] = 100 * np.mean(_ratios)
    es_gap_by_window = _gaps
    return es_gap_by_window, window_lengths


@app.cell(hide_code=True)
def _(es_gap_by_window, mo, plots, plt, window_lengths, window_slider):
    _fig, _ax = plt.subplots(figsize=(9.6, 2.8), constrained_layout=True)
    _ax.plot(window_lengths, [es_gap_by_window[_n] for _n in window_lengths], color=plots.SERIES[1],
             marker="o", markersize=5, linewidth=1.5)
    _ax.axhline(0, color=plots.BASELINE, linewidth=0.8)
    _ax.scatter([window_slider.value], [es_gap_by_window[window_slider.value]], s=90, color=plots.SERIES[1],
                edgecolor=plots.INK, zorder=3)
    _ax.set_xscale("log")
    _ax.set_xticks(window_lengths, [str(_n) for _n in window_lengths])
    _ax.set_xlabel("days of history (log scale)")
    _ax.set_ylabel("plug-in ES vs predictive ES (%)")
    _ax.set_title("How much the plug-in forecast understates the 10-day 97.5% ES (exact posterior, average of 25 histories)",
                  loc="left", fontsize=9)
    mo.vstack([_fig, mo.md(
        f"With **{window_slider.value} days** of history, the plug-in ES is **{-es_gap_by_window[window_slider.value]:.1f}% "
        f"lower** than the predictive ES. In slot 5 the jump parameters live permanently in this "
        "regime: four years of returns contain only a handful of jumps."
    )])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## When nobody tells you what matters: learned summary statistics

    We cheated a little: we knew that the mean and the standard deviation say it all. For stochastic
    volatility with jumps no finite set of numbers is sufficient, and nobody can tell us which ones to
    compute. So let a **summary network** read the raw 1,000 × 2 series, the returns and their log
    squares, and learn its own statistics. Two things change:
    """)
    return


@app.cell
def _(bf, simulator):
    series_adapter = (
        bf.Adapter()
        .drop("stats")                                                    # no hints this time
        .as_time_series(["returns", "log_sq"])                            # (batch, 1000) -> (batch, 1000, 1)
        .concatenate(["returns", "log_sq"], into="summary_variables")     # the raw series: (batch, 1000, 2)
        .concatenate(["drift", "log_vol"], into="inference_variables")
        .convert_dtype("float64", "float32")
    )

    def make_series_workflow():
        return bf.BasicWorkflow(
            simulator=simulator,
            adapter=series_adapter,
            summary_network=bf.networks.TimeSeriesNetwork(                # learns 8 statistics from the raw series
                summary_dim=8, filters=(16, 32), kernel_sizes=(8, 5), strides=(4, 2), recurrent_dim=32
            ),
            inference_network=bf.networks.FlowMatching(
                subnet_kwargs={"widths": (64, 64)}, integrate_kwargs={"method": "euler", "steps": 50}
            ),
            standardize="inference_variables",  # never rescale the returns: their scale *is* the volatility
        )

    return (make_series_workflow,)


@app.cell(hide_code=True)
def _(mo):
    train_series_button = mo.ui.run_button(label="Train the summary network now (about 5 minutes)")
    mo.vstack([
        mo.md(
            "Learning statistics from raw data is harder than being handed them. This network trained "
            "on 256,000 simulated stocks, about five minutes on the same machine, before the session."
        ),
        train_series_button,
    ])
    return (train_series_button,)


@app.cell
def _(CHECKPOINTS, RETRAIN, gv, json, keras, make_series_workflow, time, train_series_button, warnings):
    series_workflow = make_series_workflow()
    if RETRAIN or train_series_button.value:
        keras.utils.set_random_seed(2027)
        _start = time.time()
        _history = series_workflow.fit_online(
            epochs=40, num_batches_per_epoch=100, batch_size=64, validation_data=512, verbose=0
        )
        series_record = {"seconds": round(time.time() - _start, 1), "series": 40 * 100 * 64,
                         "loss": _history.history["loss"], "val_loss": _history.history["val_loss"]}
        if RETRAIN:
            CHECKPOINTS.mkdir(parents=True, exist_ok=True)
            gv.save_network(series_workflow.approximator, CHECKPOINTS / "summary_network.keras")
            (CHECKPOINTS / "summary_network.json").write_text(json.dumps(series_record, indent=1))
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")
            series_workflow.load_approximator(CHECKPOINTS / "summary_network.keras")
        series_record = json.loads((CHECKPOINTS / "summary_network.json").read_text())
    return series_record, series_workflow


@app.cell
def _(conditions, gv, np, series_workflow, stats_workflow, test_set):
    _n = 100
    _cond = conditions(test_set["returns"][:_n])
    _exact = [gv.exact_posterior(_r) for _r in test_set["returns"][:_n]]
    _exact_mean = np.array([_p["mean"] for _p in _exact])
    _exact_sd = np.array([_p["sd"] for _p in _exact])
    comparison = {}
    for _label, _workflow in (("two statistics", stats_workflow), ("learned summary", series_workflow)):
        _draws = _workflow.sample(conditions=_cond, num_samples=1_000)
        _draws = np.concatenate([_draws["drift"], _draws["log_vol"]], axis=-1)
        comparison[_label] = {
            "error": (_draws.mean(axis=1) - _exact_mean) / _exact_sd,
            "spread": _draws.std(axis=1) / _exact_sd,
        }
    return (comparison,)


@app.cell(hide_code=True)
def _(comparison, mo, np, plots, plt, series_record):
    _fig, _axes = plt.subplots(1, 2, figsize=(9.6, 3.0), constrained_layout=True, sharey=True)
    _bins = np.linspace(0, 3.5, 36)
    for _ax, _j, _name in zip(_axes, (0, 1), ("drift", "log volatility")):
        for _color, (_label, _result) in zip((plots.SERIES[0], plots.SERIES[1]), comparison.items()):
            _ax.hist(np.minimum(np.abs(_result["error"][:, _j]), 3.49), bins=_bins, histtype="step", linewidth=1.6,
                     color=_color, label=_label)
        _ax.axvline(0.3, color=plots.INK_2, linewidth=0.9, linestyle="--")
        _ax.set_title(f"{_name}: |network mean − exact mean| in exact sd", loc="left", fontsize=9)
        _ax.set_xlabel("error (posterior sd); dashed line at the 0.3 criterion")
    _axes[0].set_ylabel("test stocks (of 100)")
    _axes[0].legend(fontsize=8)
    _share = {_label: np.mean(np.abs(_r["error"][:, 1]) > 1) for _label, _r in comparison.items()}
    _median = {_label: np.median(np.abs(_r["error"][:, 1])) for _label, _r in comparison.items()}
    mo.vstack([
        mo.md(f"**Both networks against the exact posterior on 100 new test stocks** (summary network: "
              f"{series_record['series']:,} simulations, {series_record['seconds'] / 60:.0f} minutes):"),
        _fig,
        mo.md(f"""
        The learned summary is close on most stocks (median volatility error
        {_median['learned summary']:.2f} posterior sd, against {_median['two statistics']:.2f} with the two
        statistics), but on **{_share['learned summary']:.0%} of them it is off by more than one posterior
        sd**. Without the exact posterior you would not know which stocks those are.

        That is precisely the situation in slot 5: the model needs a learned summary, and no exact
        posterior exists to check single names against. The answer there is an **audit** that uses the
        exact likelihood, name by name.
        """),
    ])
    return


@app.cell
def _(conditions, gv, np, stats_workflow, voo):
    _draws = stats_workflow.sample(conditions=conditions(voo.to_numpy()), num_samples=4_000)
    voo_draws = np.concatenate([_draws["drift"][0], _draws["log_vol"][0]], axis=-1)
    voo_risk = gv.predictive_var_es(voo_draws[:, 0], voo_draws[:, 1], horizon=10)
    return voo_draws, voo_risk


@app.cell(hide_code=True)
def _(ROOT, json, mo, np, plots, plt, voo, voo_draws, voo_risk):
    _vol = 100 * np.exp(voo_draws[:, 1])
    _lo, _mid, _hi = np.quantile(_vol, [0.05, 0.5, 0.95])
    _drift_lo, _drift_hi = np.quantile(voo_draws[:, 0], [0.05, 0.95])
    _realized = voo.rolling(20).std() * np.sqrt(252)
    _svj = json.loads((ROOT / "docs" / "results" / "svj-daily" / "VOO_2026-09-03.json").read_text())
    _svj_spot = 100 * _svj["network_posterior"]["spot vol"]["median"]
    _fig, _ax = plt.subplots(figsize=(9.6, 3.0), constrained_layout=True)
    _ax.plot(_realized.index, _realized.to_numpy(), color=plots.INK_2, linewidth=0.9,
             label="realized volatility, last 20 days")
    _ax.axhspan(_lo, _hi, color=plots.SERIES[0], alpha=0.25, linewidth=0, label="constant-volatility posterior, 90%")
    _ax.axhline(_mid, color=plots.SERIES[0], linewidth=1.4)
    _ax.set_ylabel("annualized volatility (%)")
    _ax.set_title("VOO: a precise answer to the wrong question", loc="left")
    _ax.legend(loc="upper left", fontsize=8)
    _ax.grid(axis="x", visible=False)
    mo.vstack([
        mo.md(f"""
        ## Back to VOO

        The network answers the opening question in a fraction of a second. **Volatility {_mid:.1f}%
        (90% interval {_lo:.1f}% to {_hi:.1f}%)**, drift between {_drift_lo:.0f}% and {_drift_hi:.0f}% a
        year, and a **10-day 99% VaR of {voo_risk[0]:.1f}%** (97.5% ES {voo_risk[1]:.1f}%).

        Now look at the data the model was asked to explain:
        """),
        _fig,
        mo.md(f"""
        The posterior is tight, and the model is wrong. Over these four years VOO's 20-day volatility
        ranged from {_realized.min():.0f}% to {_realized.max():.0f}% (in {_realized.idxmax():%B %Y}). A
        constant-volatility model reports the same risk on calm and stormy days,
        and its narrow interval is certainty about an average that never happens. Slot 5's model, with
        moving volatility and jumps, puts VOO's volatility **today** at about **{_svj_spot:.0f}%**, not
        {_mid:.0f}%.

        - **Slot 5:** the same code, with a model where volatility moves and prices jump.
        - **Slot 6:** how to notice when the model, not the network, is wrong.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    Each takes a minute or two of training. Press a button to run it.

    1. **The normalization trap.** Many machine-learning pipelines rescale each input series to zero mean
       and unit variance. Do that to the returns, retrain the summary network, and look at the recovery
       of volatility.
    2. **The prior matters where the data are weak.** Widen the drift prior to $\mathcal N(0, 30^2)$,
       retrain the two-statistics network, and compare VOO's posterior for the drift and for the
       volatility. Which one moved? (More on priors in slot 6.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    normalization_button = mo.ui.run_button(label="Exercise 1: train on normalized returns (about a minute)")
    prior_button = mo.ui.run_button(label="Exercise 2: retrain with a wide drift prior (about a minute)")
    mo.hstack([normalization_button, prior_button])
    return normalization_button, prior_button


@app.cell
def _(bf, keras, make_simulator, mo, normalization_button, np):
    mo.stop(not normalization_button.value)
    _plain = make_simulator(seed=99)

    class NormalizedSimulator:
        """Wraps the simulator and rescales every series to mean 0 and sd 1, as a typical ML pipeline would."""

        def sample(self, batch_shape, **kwargs):
            batch = _plain.sample(batch_shape, **kwargs)
            _z = (batch["returns"] - batch["returns"].mean(axis=1, keepdims=True)) / batch["returns"].std(axis=1, keepdims=True)
            return {**batch, "returns": _z, "log_sq": np.log(_z**2 + 0.01)}

    _workflow = bf.BasicWorkflow(
        simulator=NormalizedSimulator(),
        adapter=bf.Adapter().drop("stats").as_time_series(["returns", "log_sq"])
        .concatenate(["returns", "log_sq"], into="summary_variables")
        .concatenate(["drift", "log_vol"], into="inference_variables").convert_dtype("float64", "float32"),
        summary_network=bf.networks.TimeSeriesNetwork(summary_dim=8, filters=(16, 32), kernel_sizes=(8, 5),
                                                      strides=(4, 2), recurrent_dim=32),
        inference_network=bf.networks.FlowMatching(subnet_kwargs={"widths": (64, 64)},
                                                   integrate_kwargs={"method": "euler", "steps": 50}),
    )
    keras.utils.set_random_seed(5)
    _workflow.fit_online(epochs=10, num_batches_per_epoch=50, batch_size=64, verbose=0)
    _figures = _workflow.plot_default_diagnostics(test_data=NormalizedSimulator().sample(200), num_samples=300,
                                                  variable_names=["drift (%/yr)", "log volatility"])
    mo.vstack([_figures["recovery"], mo.md(
        "Volatility can no longer be recovered: the posterior falls back to the prior, because the "
        "information was in the scale of the returns, and the normalization deleted it."
    )])
    return


@app.cell
def _(bf, conditions, keras, mo, np, prior_button, voo):
    mo.stop(not prior_button.value)
    _rng = np.random.default_rng(31)

    def _prior():
        return {"drift": _rng.normal(0.0, 30.0), "log_vol": _rng.normal(np.log(0.25), 0.40)}

    def _observation(drift, log_vol):
        _r = drift / 252 + 100 * np.exp(log_vol) / np.sqrt(252) * _rng.standard_normal(1_000)
        return {"returns": _r, "log_sq": np.log(_r**2 + 0.01), "stats": np.array([_r.mean(), np.log(_r.std())])}

    _workflow = bf.BasicWorkflow(
        simulator=bf.make_simulator([_prior, _observation]),
        adapter=bf.Adapter().drop(["returns", "log_sq"]).concatenate(["drift", "log_vol"], into="inference_variables")
        .rename("stats", "inference_conditions").convert_dtype("float64", "float32"),
        inference_network=bf.networks.FlowMatching(subnet_kwargs={"widths": (64, 64)},
                                                   integrate_kwargs={"method": "euler", "steps": 50}),
        standardize=["inference_variables", "inference_conditions"],
    )
    keras.utils.set_random_seed(6)
    _workflow.fit_online(epochs=100, num_batches_per_epoch=50, batch_size=64, verbose=0)
    _draws = _workflow.sample(conditions=conditions(voo.to_numpy()), num_samples=4_000)
    _drift = np.quantile(_draws["drift"][0, :, 0], [0.05, 0.5, 0.95])
    _vol = 100 * np.exp(np.quantile(_draws["log_vol"][0, :, 0], [0.05, 0.5, 0.95]))
    mo.md(f"""
    With the wide drift prior, VOO's drift posterior is **{_drift[1]:.0f}% a year (90%: {_drift[0]:.0f}% to
    {_drift[2]:.0f}%)**, and its volatility **{_vol[1]:.1f}% ({_vol[0]:.1f}% to {_vol[2]:.1f}%)**. Compare with
    the section above: the volatility barely moves, because 1,000 returns overwhelm any reasonable prior.
    The drift moves a lot, because the data say little about it and the prior fills the gap.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What you can now do, and what changes in slot 5

    You have built the full amortized workflow: simulator, adapter, networks, online training,
    diagnostics, exact check, and a forecast that carries parameter uncertainty into VaR and ES.

    | | This notebook | Slot 5 |
    |---|---|---|
    | Model | constant volatility | volatility that moves, plus jumps |
    | Unknowns | drift, volatility | 8 parameters and **today's volatility** |
    | Summary | two statistics, or a small time-series network | a larger time-series network (32 statistics) |
    | Training | 1 to 5 minutes | 79 minutes, once |
    | Is it right? | compare with the exact posterior | no exact posterior: the particle filter, particle MCMC and an audit |
    | Parameter uncertainty in VaR / ES | negligible with 1,000 days | matters in the tail, because of the jumps |

    The BayesFlow code you wrote today runs unchanged in slot 5. What changes is the simulator, and
    everything a harder model demands.
    """)
    return


if __name__ == "__main__":
    app.run()
