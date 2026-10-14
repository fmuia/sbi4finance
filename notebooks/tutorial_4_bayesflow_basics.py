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

    This notebook is about the **software**: what BayesFlow does, and the code that does it, shown in
    full. We use the simplest volatility model there is, constant volatility, because its exact
    posterior is known: every step of the workflow can be checked against the truth. In slot 5 the code
    stays, the model becomes realistic, and the exact answer disappears.

    | Step | BayesFlow | What it does |
    |:---|:---|:---|
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

        **VOO** is Vanguard's S&P 500 ETF, an exchange-traded fund that holds the roughly 500 large US
        companies of the S&P 500 index, weighted by market value. Its daily returns are, to a good
        approximation, those of the US stock market. The repository caches its daily returns, computed
        from adjusted closing prices so that dividends and splits are accounted for, in `data/`. We turn
        them into log returns in percent, $y_t = 100\,\log(P_t / P_{t-1})$, and keep the last 1,000
        trading days, about four years:
        """),
        _fig,
        mo.md(r"""
        **What is VOO's volatility, how sure can we be, and what is its 10-day 99% value-at-risk**, the
        loss over the next two weeks that is exceeded with only 1% probability? We build the answer step
        by step, and return to VOO at the end.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo, plots, plt):
    def _diagram():
        _fig, _ax = plt.subplots(figsize=(9.6, 4.0))
        _ax.set_xlim(0, 1)
        _ax.set_ylim(0, 1.12)
        _ax.axis("off")
        _train, _use = plots.SERIES[0], plots.SERIES[1]

        def _box(x, y, text, color):
            return _ax.text(x, y, text, ha="center", va="center", fontsize=9, color=plots.INK, linespacing=1.5,
                            bbox=dict(boxstyle="round,pad=0.55", fc=plots.SURFACE, ec=color, lw=1.3))

        def _arrow(a, b, label=None):
            _ax.annotate("", xy=(0, 0.5), xycoords=b.get_bbox_patch(), xytext=(1, 0.5),
                         textcoords=a.get_bbox_patch(), arrowprops=dict(arrowstyle="-|>", color=plots.INK_2, lw=1.2))
            if label:
                (_xa, _ya), (_xb, _) = a.get_position(), b.get_position()
                _ax.text((_xa + _xb) / 2, _ya + 0.035, label, ha="center", va="bottom", fontsize=9, color=plots.INK_2)

        _y_train, _y_use = 0.62, 0.2
        _ax.text(0.0, 1.1, "Train once, on simulated stocks", fontsize=10, fontweight="bold", color=_train, va="top")
        _ax.text(0.0, 0.43, "Use every day, on real data", fontsize=10, fontweight="bold", color=_use, va="top")
        _prior = _box(0.09, _y_train, "prior\n$p(\\theta)$", _train)
        _sim = _box(0.31, _y_train, "simulator\n$p(y \\mid \\theta)$", _train)
        _summary = _box(0.53, _y_train, "summary\n$s(y)$", _train)
        _net = _box(0.78, _y_train, "inference network\n$q_\\phi(\\theta \\mid s)$", _train)
        _arrow(_prior, _sim, "$\\theta$")
        _arrow(_sim, _summary, "$y$")
        _arrow(_summary, _net, "$s$")
        _top, _bracket = _y_train + 0.085, 0.86
        _ax.plot([0.09, 0.09, 0.78], [_top, _bracket, _bracket], color=_train, lw=1.1, linestyle=(0, (4, 3)))
        _ax.annotate("", xy=(0.78, _top), xytext=(0.78, _bracket),
                     arrowprops=dict(arrowstyle="-|>", color=_train, lw=1.1, linestyle=(0, (4, 3))))
        _ax.text(0.435, _bracket + 0.025, "the true $\\theta$: the target the network learns to reproduce",
                 ha="center", va="bottom", fontsize=8.5, color=_train)
        _data = _box(0.31, _y_use, "VOO's 1,000 returns\n$y_{\\mathrm{obs}}$", _use)
        _summary_obs = _box(0.53, _y_use, "summary\n$s(y_{\\mathrm{obs}})$", _use)
        _net_obs = _box(0.78, _y_use, "trained network\n$q_\\phi(\\theta \\mid s_{\\mathrm{obs}})$", _use)
        _arrow(_data, _summary_obs)
        _arrow(_summary_obs, _net_obs)
        _ax.annotate("", xy=(0.915, _y_use), xytext=(1, 0.5), textcoords=_net_obs.get_bbox_patch(),
                     arrowprops=dict(arrowstyle="-|>", color=plots.INK_2, lw=1.2))
        _ax.text(0.925, _y_use, "posterior draws\n$\\to$ VaR, ES", ha="left", va="center", fontsize=9,
                 color=plots.INK, linespacing=1.5)
        _ax.annotate("", xy=(0.78, _y_use + 0.085), xytext=(0.78, _y_train - 0.085),
                     arrowprops=dict(arrowstyle="-|>", color=plots.MUTED, lw=1.0, linestyle=(0, (2, 2))))
        _ax.text(0.79, (_y_use + _y_train) / 2, "same network,\ntrained once", ha="left", va="center",
                 fontsize=8, color=plots.MUTED, linespacing=1.4)
        return _fig

    _intro = mo.md(r"""
    ## How BayesFlow answers it

    Bayes' rule, $p(\theta \mid y) \propto p(y \mid \theta)\, p(\theta)$, answers for one data set at a
    time: a new stock, or a new day, means a new computation, here on a grid, in general with a Markov
    chain. **Amortized inference** splits the work in two:

    * **Train once.** Draw parameters $\theta$ from the prior, simulate data $y$ from the model, and
      train a neural network to produce the parameters from the data. It learns a whole conditional
      distribution $q_\phi(\theta \mid y)$, not a single best guess.
    * **Use many times.** For any new data set, simulated or real, one pass through the trained network
      returns thousands of posterior draws.
    """)
    _why = mo.md(r"""
    Why is the result the posterior? Training rewards the network for putting probability on the true
    $\theta$ of each simulated data set. Across all data sets, the distribution that does this best is
    the posterior itself, so a network with enough capacity and training converges to
    $p(\theta \mid y)$ for every $y$ at once. At no point does BayesFlow evaluate the likelihood
    $p(y \mid \theta)$: it only **simulates** from the model. That is what makes it work for models like
    slot 5's, whose likelihood nobody can write down.

    | Piece | In BayesFlow | Role |
    |:---|:---|:---|
    | Simulator | `bf.make_simulator` | draws pairs $(\theta, y)$ from the prior and the model |
    | Adapter | `bf.Adapter` | names the unknowns and the data, and shapes the arrays |
    | Summary network (optional) | e.g. `bf.networks.TimeSeriesNetwork` | compresses each data set into a few learned statistics |
    | Inference network | `bf.networks.FlowMatching` | the conditional distribution $q_\phi(\theta \mid y)$ |
    | Workflow | `bf.BasicWorkflow` | ties them together: training, diagnostics, sampling |
    """)
    _folds = mo.accordion({
        "**Why the trained network is the posterior** (the math)": mo.md(r"""
        BayesFlow trains on pairs drawn from the joint distribution $p(\theta, y) = p(\theta)\, p(y \mid \theta)$.
        For a network that outputs a density, the natural objective is the negative log probability it
        assigns to the true parameters:

        $$\mathcal L(\phi) = \mathbb E_{p(\theta, y)}\big[-\log q_\phi(\theta \mid y)\big] = \mathbb E_{p(y)}\Big[\mathrm{KL}\big(p(\theta \mid y)\,\big\|\,q_\phi(\theta \mid y)\big)\Big] + \text{const}.$$

        The constant does not depend on $\phi$, and the Kullback–Leibler divergence is zero only when the
        two distributions agree, so the best network reproduces the posterior for (almost) every data set
        the model can produce. This is neural posterior estimation; flow matching, in the next panel,
        reaches the same target through a different loss.

        Two consequences matter in practice. The network learns from data simulated by the model, so on
        real data it is only as good as the model (the VOO section at the end). And it is most accurate
        where simulations are plentiful, so the prior should cover the data you will meet (the prior
        check in step 1).
        """),
        "**How flow matching turns noise into posterior draws**": mo.md(r"""
        A flow moves a cloud of points. Start from standard normal noise $z_0 \sim \mathcal N(0, I)$, with
        one coordinate per unknown (here 2), and let it flow for one unit of time,
        $dz_t / dt = v_\phi(z_t, t, y)$. If the velocity field $v_\phi$ is right, the points arrive at
        $t = 1$ distributed as the posterior $p(\theta \mid y)$.

        **Training.** Take a simulated pair $(\theta, y)$, fresh noise $z_0$ and a random time $t$, and
        mix them: $z_t = t\,\theta + (1 - t)\,z_0$. Along this straight line from noise to the truth the
        velocity is constant, $\theta - z_0$, and the network learns to predict it by least squares:

        $$\mathcal L(\phi) = \mathbb E\,\big\|\,v_\phi(z_t, t, y) - (\theta - z_0)\,\big\|^2.$$

        No single straight line is the answer: many pairs pass through the same point $z_t$, and the best
        prediction averages their velocities. That average is exactly the velocity field whose flow
        carries the noise onto the posterior (Lipman et al., 2023; Wildberger et al., 2023, for
        simulation-based inference). BayesFlow draws the training times as $t = u^{1.5}$ with $u$
        uniform, which spends more effort near the noise end.

        **Sampling.** Draw $z_0$, integrate the flow from $t = 0$ to $t = 1$ (here with 50 Euler steps),
        and read off $\theta$: one posterior draw per noise draw, thousands in parallel. During training
        the unknowns are standardized with their simulation mean and sd, so the network works with
        numbers of order one, and BayesFlow undoes this on the way out.
        """),
    })
    mo.vstack([_intro, _diagram(), _why, _folds])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1 · Write down how the data are generated

    The simplest volatility model: daily log returns, in percent, are independent normals with an
    annual drift $\mu$ and an annual volatility $\sigma$ (the panel at the top derives it),

    $$y_t = \frac{\mu}{252} + \frac{100\,\sigma}{\sqrt{252}}\;\varepsilon_t, \qquad \varepsilon_t\sim\mathcal N(0,1),\quad t = 1,\dots,1000.$$

    The prior is stated in units a risk manager can judge, the same as slot 5's prior for these two
    quantities: a drift of 8% a year give or take 10%, and a volatility around 25%, with a 95% range of
    roughly 11% to 55%.

    BayesFlow needs two plain functions that **draw**. `prior()` returns one parameter set, and
    `observation(drift, log_vol)` returns one simulated data set for those parameters; its argument
    names match the keys `prior` returns, which is how `bf.make_simulator` chains them.
    `simulator.sample(n)` then returns a dictionary of arrays with one row per simulated stock: the
    parameters, and everything `observation` returns. Ours returns the data in three forms:

    * `returns`: the 1,000 daily returns themselves;
    * `log_sq`: their log squares, $\log(y_t^2 + 0.01)$, in which volatility shows up as a shift in level (used in Part B);
    * `stats`: two numbers, the mean and the log standard deviation of the returns (used first; step 2 explains why).
    """)
    return


@app.cell(hide_code=True)
def _(bf, mo, np):
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
    mo.show_code()
    return make_simulator, simulator


@app.cell(hide_code=True)
def _(mo, np):
    def conditions(returns):
        """Prepare any return series (simulated or real, in percent) exactly as the simulator does."""
        returns = np.atleast_2d(np.asarray(returns, dtype=float))
        return {
            "returns": returns,
            "log_sq": np.log(returns**2 + 0.01),
            "stats": np.stack([returns.mean(axis=1), np.log(returns.std(axis=1))], axis=1),
        }

    mo.show_code(mo.md(
        "Real data must reach the network in exactly the form the simulator produces. This helper prepares "
        "any return series, simulated or real, the same way:"
    ))
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
        mo.md(
            "**Check the prior before training.** A prior predictive check simulates stocks from the prior "
            "and the model before any training. If they looked absurd, say volatilities of 300%, the network "
            "would spend its effort on data that never occur, and where the data are weak the prior would "
            "pull the posterior toward nonsense."
        ),
        _fig,
        mo.md("Volatilities from about 10% to 60%, with VOO comfortably inside: wide, but plausible."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Steps 2 and 3 · What the network sees, and which network

    ### What the network conditions on: summary statistics

    The inference network needs a fixed-size input that carries the information about the unknowns.
    There are three options: the raw series (1,000 numbers per stock), a few **statistics we compute
    ourselves**, or statistics that a **summary network learns** from the raw series (Part B).

    For this model the second option is perfect. The likelihood depends on the 1,000 returns only
    through their mean $\bar y$ and their standard deviation $\hat s$ (section 4 of the panel at the top
    derives this), so two stocks with the same $\bar y$ and $\hat s$ have exactly the same posterior.
    Such statistics are called **sufficient**: conditioning on them loses nothing. The first network
    therefore sees two numbers per stock, $(\bar y, \log\hat s)$, and only has to learn how they map to
    a posterior. Part B takes this help away.

    ### The adapter and the inference network

    The **adapter** turns the simulator's dictionary into the three roles BayesFlow knows:

    | Role | Meaning | Here |
    |:---|:---|:---|
    | `inference_variables` | the unknowns, whose posterior we want | `drift`, `log_vol` |
    | `inference_conditions` | fixed-size inputs handed straight to the inference network | `stats` |
    | `summary_variables` | raw data for a summary network to compress | none yet (Part B) |

    The **inference network** is flow matching with a small velocity network, two hidden layers of 64
    units, and 50 Euler steps at sampling time. `standardize` rescales the unknowns and the conditions
    to mean 0 and sd 1 with simulation statistics, so the network works with numbers of order one.
    """)
    return


@app.cell(hide_code=True)
def _(bf, mo, simulator):
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

    mo.show_code()
    return (make_stats_workflow,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4 · Train once, on fresh simulations

    `fit_online` simulates new data at every training step. Each step draws a **batch** of 64 stocks
    from the simulator and nudges the network to reduce the flow-matching loss on them; 50 steps make an
    **epoch**, and 100 epochs see 320,000 different simulated stocks. Because every batch is new, the
    network cannot memorize its training data, and a validation set of 512 further simulations tracks the
    loss on stocks it never trained on. On a 4-core machine this takes about a minute. The network below
    was trained exactly this way before the session; press the button to train it live.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    train_stats_button = mo.ui.run_button(label="Train this network now (about a minute)")
    train_stats_button
    return (train_stats_button,)


@app.cell(hide_code=True)
def _(CHECKPOINTS, gv, json, warnings):
    def load_trained(workflow, name):
        """Load a network trained before the session, with its training record."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")
            workflow.load_approximator(CHECKPOINTS / f"{name}.keras")
        return json.loads((CHECKPOINTS / f"{name}.json").read_text()) | {"source": "trained before the session"}

    def record_training(workflow, name, history, seconds, series, save):
        """The training record; with save=True also store the network (pixi run tutorial-4-train)."""
        record = {"seconds": round(seconds, 1), "series": series,
                  "loss": history.history["loss"], "val_loss": history.history["val_loss"]}
        if save:
            CHECKPOINTS.mkdir(parents=True, exist_ok=True)
            gv.save_network(workflow.approximator, CHECKPOINTS / f"{name}.keras")
            (CHECKPOINTS / f"{name}.json").write_text(json.dumps(record, indent=1))
        return record | {"source": "trained just now"}

    return load_trained, record_training


@app.cell(hide_code=True)
def _(RETRAIN, keras, load_trained, make_stats_workflow, mo, record_training, time, train_stats_button):
    stats_workflow = make_stats_workflow()
    if train_stats_button.value or RETRAIN:          # train live: 100 epochs x 50 batches x 64 fresh stocks
        keras.utils.set_random_seed(2026)
        _start = time.time()
        _history = stats_workflow.fit_online(
            epochs=100, num_batches_per_epoch=50, batch_size=64, validation_data=512, verbose=0
        )
        stats_record = record_training(stats_workflow, "statistics", _history, time.time() - _start,
                                       series=100 * 50 * 64, save=RETRAIN)
    else:                                            # or load the same network, trained this way before the session
        stats_record = load_trained(stats_workflow, "statistics")
    mo.show_code()
    return stats_record, stats_workflow


@app.cell(hide_code=True)
def _(mo, np, plots, plt, stats_record):
    _fig, _ax = plt.subplots(figsize=(9.6, 2.6), constrained_layout=True)
    _epochs = np.arange(1, len(stats_record["loss"]) + 1)
    _ax.plot(_epochs, stats_record["loss"], color=plots.SERIES[0], linewidth=1.2, label="training loss")
    _ax.plot(_epochs, stats_record["val_loss"], color=plots.SERIES[1], linewidth=1.2, label="validation loss")
    _ax.set_xlabel("epoch (3,200 fresh simulations each)")
    _ax.set_ylabel("flow-matching loss")
    _ax.set_title(f"Training: {stats_record['series']:,} simulated stocks in {stats_record['seconds']:.0f} s "
                  f"({stats_record['source']})", loc="left")
    _ax.legend(loc="upper right", fontsize=8)
    mo.vstack([_fig, mo.md(
        "The loss falls quickly and levels off, and training and validation agree: with a new batch at every "
        "step there is nothing to overfit. Its value has no direct meaning (it is a regression error on "
        "velocities), and a low loss does not prove that the posterior is right. Step 5 checks that."
    )])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 5 · Check before you trust

    A trained network can be wrong in two ways: **biased**, with the posterior in the wrong place, or
    **miscalibrated**, with a posterior too narrow or too wide. Both can be tested without any exact
    posterior: simulate 300 new stocks whose parameters we know, ask the network for 500 posterior draws
    for each, and compare the draws with the truth. `plot_default_diagnostics` does this in one call
    and draws four standard plots:
    """)
    return


@app.cell(hide_code=True)
def _(make_simulator, mo, stats_workflow):
    test_set = make_simulator(seed=7).sample(300)    # 300 new stocks whose parameters we know
    stats_diagnostics = stats_workflow.plot_default_diagnostics(
        test_data=test_set, num_samples=500, variable_names=["drift (%/yr)", "log volatility"]
    )
    mo.show_code()
    return stats_diagnostics, test_set


@app.cell(hide_code=True)
def _(mo, stats_diagnostics):
    mo.vstack([
        mo.md(r"""
        **1 · Recovery.** One dot per test stock: the true value (horizontal) against the posterior
        median (vertical), with a bar for the 95% credible interval; $r$ is their correlation. Accurate
        posteriors put the dots on the diagonal. Volatility sits on it. The drift's medians are pulled
        toward the prior mean of 8%: with little information in the data the posterior leans on the
        prior, which is the correct Bayesian answer, not an error.
        """),
        stats_diagnostics["recovery"],
        mo.md(r"""
        **2 · Calibration** (simulation-based calibration). For each test stock, take the fraction of
        posterior draws that lie below the true value: its **rank**, between 0 and 1. If the posterior is
        right, the truth is statistically just one more draw from it, so the ranks are uniformly
        distributed. The plot shows the empirical distribution function of the ranks minus that of the
        uniform distribution. A flat line at zero is perfect, and the grey band is where the whole curve
        stays 95% of the time under perfect calibration. Reading the shapes: a curve above zero means the
        posterior sits too high (the truth ranks low), below zero too low; a wave that is positive on the
        left and negative on the right means the posterior is too narrow (truths fall in its tails), and
        the opposite wave too wide.
        """),
        stats_diagnostics["calibration_ecdf"],
        mo.md(r"""
        **3 · Coverage.** For each nominal level, the share of test stocks whose central credible
        interval of that level contains the truth. Calibrated posteriors follow the diagonal: 90%
        intervals contain the truth 90% of the time. Below the diagonal the intervals are too narrow,
        above it too wide; the grey ribbon shows the sampling uncertainty with 300 stocks.
        """),
        stats_diagnostics["coverage"],
        mo.md(r"""
        **4 · Contraction and z-scores.** One dot per test stock. Horizontally, the **posterior
        contraction**, $1 - \text{posterior variance} / \text{prior variance}$: 0 means the data taught
        nothing, 1 that they pin the parameter down. Vertically, the **z-score**,
        $(\text{posterior mean} - \text{truth}) / \text{posterior sd}$: how many posterior sds the truth
        sits from the posterior mean; calibrated posteriors keep most dots within ±2. The ideal is a cloud
        on the right, centered on zero.
        """),
        stats_diagnostics["z_score_contraction"],
        mo.accordion({"**Why the ranks are uniform when the posterior is right**": mo.md(r"""
        Draw $\theta^* \sim p(\theta)$, simulate $y \sim p(y \mid \theta^*)$, and draw
        $\theta^{(1)}, \dots, \theta^{(M)}$ from the posterior $p(\theta \mid y)$. Given $y$, the true
        $\theta^*$ is itself distributed as $p(\theta \mid y)$: that is Bayes' rule. So
        $\theta^*, \theta^{(1)}, \dots, \theta^{(M)}$ are $M + 1$ independent draws from one
        distribution, and the rank of $\theta^*$ among them is equally likely to take each of its
        $M + 1$ values. A network posterior that differs from $p(\theta \mid y)$ breaks this symmetry,
        and the direction of the break says how (Talts et al., 2018). The test looks at one parameter at
        a time and is necessary rather than sufficient for a correct posterior, which is why we also
        compare with the exact answer below.
        """)}),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What the four plots say.** Volatility is recovered almost perfectly: its contraction is close to
    1, because 1,000 returns pin it down to within a few percent. The drift is not: its contraction is
    about 0.4, so the data remove only about 40% of the prior's variance. Four years of returns say
    little about a stock's expected return, a classic result in finance (Merton, 1980), and section 6 of
    the panel at the top shows why. Both answers are correct, and the calibration and coverage plots say
    the network reports its uncertainty honestly.

    ### The ultimate check: the exact answer

    Calibration holds on average over many stocks; it does not prove that each single posterior is
    right. For this model we can compute the exact posterior of any one stock on a grid, because the
    likelihood is known. Take a stock we invent, with a drift of 6% and a volatility of 22%, and ask
    the trained network for its posterior with `sample`:
    """)
    return


@app.cell(hide_code=True)
def _(conditions, gv, mo, np, stats_workflow):
    stock_returns = gv.simulate_returns(6.0, np.log(0.22), 1_000, np.random.default_rng(42))  # the invented stock
    stock_exact = gv.exact_posterior(stock_returns)                                            # exact, on a grid
    _draws = stats_workflow.sample(conditions=conditions(stock_returns), num_samples=4_000)   # amortized
    stock_stats_draws = np.concatenate([_draws["drift"][0], _draws["log_vol"][0]], axis=-1)    # (4000, 2)
    mo.show_code()
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
            "|:---|:---|:---|:---|:---|",
            *_rows,
            "",
            "**Error**: (network mean − exact mean) / exact sd, how far off the network is in units of the "
            "posterior's own width. **Spread ratio**: network sd / exact sd, which is 1 when the network's "
            "uncertainty is right.",
            "",
            "The criterion we will meet again in slot 5: error below **0.3 posterior sd** and a spread "
            "within 25% of the exact one, so that the network's mistakes are small next to the uncertainty "
            "the posterior reports anyway. There, the exact answer comes from particle MCMC.",
        ])),
    ], widths=[1.1, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 6 · Use it: the amortization payoff

    Training is paid once. After that, a posterior costs one pass through the network, for any number
    of stocks at once. With one row of returns per stock,

    ```python
    draws = stats_workflow.sample(conditions=conditions(returns), num_samples=1_000)
    ```

    and `draws["drift"]` and `draws["log_vol"]` hold 1,000 posterior draws for every stock. Timed on
    this machine:
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
    |:---|:---|
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

    **Value-at-risk and expected shortfall.** Let $L$ be the loss over the next 10 trading days, in
    percent: minus the 10-day log return. The **99% VaR** is the loss exceeded with probability 1%. The
    **97.5% ES**, the expected shortfall, is the average loss in the worst 2.5% of outcomes, so it also
    measures how bad the bad days are. Banks report both; the ES is the Basel standard for market risk.

    **Two ways to forecast.** The **plug-in** forecast picks one value of the parameters, here the
    posterior median, and treats it as the truth. The **predictive** forecast averages over what we do
    not know: it takes the model's 10-day return distribution for every posterior draw and pools them.
    It is the Bayesian answer, and it is wider whenever the parameters are uncertain. For this model both
    have closed forms, a normal and a mixture of normals; in slot 5 they are simulated.
    """)
    return


@app.cell(hide_code=True)
def _(gv, mo, np, stock_exact, stock_stats_draws):
    _median = np.median(stock_stats_draws, axis=0)
    stock_risk = {                                                   # 10-day 99% VaR and 97.5% ES, in %
        "predictive, network": gv.predictive_var_es(stock_stats_draws[:, 0], stock_stats_draws[:, 1], horizon=10),
        "predictive, exact": gv.exact_predictive_var_es(stock_exact, horizon=10),
        "plug-in at the posterior median": gv.predictive_var_es(_median[0], _median[1], horizon=10),
    }
    mo.show_code()
    return (stock_risk,)


@app.cell(hide_code=True)
def _(mo, stock_risk):
    _rows = "\n".join(f"| {_name} | {_var:.2f}% | {_es:.2f}% |" for _name, (_var, _es) in stock_risk.items())
    mo.md("\n".join([
        "| 10-day forecast for the invented stock | 99% VaR | 97.5% ES |",
        "|:---|:---|:---|",
        _rows,
        "",
        "Practically identical. With two parameters and 1,000 days, parameter uncertainty is a rounding "
        "error. **When does it matter?** When data are scarce. The slider shortens the history: for each "
        "length we simulate 25 histories of the invented stock, compute each one's exact posterior, and "
        "compare the plug-in ES with the predictive ES.",
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
    ## Part B · When nobody tells you the sufficient statistics

    Part A had help: we knew that the mean and the standard deviation say everything. In slot 5's model
    no small set of numbers is sufficient. The information sits in how volatility moves through time and
    in rare jumps, and nobody can say which statistics to compute. The general solution is to **learn**
    them. A **summary network** $s = h_\psi(y)$ reads the raw series and outputs a few numbers, and the
    inference network conditions on those, $q_\phi(\theta \mid h_\psi(y))$. The two are trained together,
    on the same loss, so the summary network learns whatever features of the series help pin down the
    parameters.

    Our summary network, `TimeSeriesNetwork`, reads two channels per day, the return and its log square.
    Convolutions slide windows of 8 and then 5 days along the series to pick up local patterns, a
    recurrent layer (a GRU) reads the result from start to end, and a final layer outputs **8 learned
    statistics**. Three things change in the code:

    * the adapter sends the series to `summary_variables` and drops the two hand-made statistics;
    * the workflow gets a `summary_network`;
    * only the unknowns are standardized, never the returns: their scale is the volatility.
    """)
    return


@app.cell(hide_code=True)
def _(bf, mo, simulator):
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

    mo.show_code()
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


@app.cell(hide_code=True)
def _(RETRAIN, keras, load_trained, make_series_workflow, mo, record_training, time, train_series_button):
    series_workflow = make_series_workflow()
    if train_series_button.value or RETRAIN:         # train live: 40 epochs x 100 batches x 64 fresh stocks
        keras.utils.set_random_seed(2027)
        _start = time.time()
        _history = series_workflow.fit_online(
            epochs=40, num_batches_per_epoch=100, batch_size=64, validation_data=512, verbose=0
        )
        series_record = record_training(series_workflow, "summary_network", _history, time.time() - _start,
                                        series=40 * 100 * 64, save=RETRAIN)
    else:                                            # or load the same network, trained this way before the session
        series_record = load_trained(series_workflow, "summary_network")
    mo.show_code()
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
        mo.md(f"**How good is the learned summary?** For 100 test stocks we compare each network's posterior "
              f"mean with the exact posterior mean, in units of the exact posterior sd. The histograms count "
              f"stocks by that error; the dashed line marks the 0.3 criterion (summary network: "
              f"{series_record['series']:,} simulations, {series_record['seconds'] / 60:.0f} minutes):"),
        _fig,
        mo.md(f"""
        The learned summary is close on most stocks (median volatility error
        {_median['learned summary']:.2f} posterior sd, against {_median['two statistics']:.2f} with the two
        statistics), but on **{_share['learned summary']:.0%} of them it is off by more than one posterior
        sd**. Without the exact posterior you would not know which stocks those are.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(conditions, make_simulator, mo, np, series_workflow):
    probe_stocks = make_simulator(seed=5).sample(1_000)                                     # 1,000 new stocks
    learned_stats = series_workflow.approximator.summarize(conditions(probe_stocks["returns"]))  # (1000, 8)
    _design = np.column_stack([np.ones(len(learned_stats)), learned_stats])
    readout = {}
    for _j, _name in enumerate(("mean", "log sd")):       # best linear read-out of each sufficient statistic
        _target = probe_stocks["stats"][:, _j]
        _fit = _design @ np.linalg.lstsq(_design, _target, rcond=None)[0]
        readout[_name] = {"target": _target, "fit": _fit, "r2": 1 - np.var(_target - _fit) / np.var(_target)}
    mo.show_code()
    return (readout,)


@app.cell(hide_code=True)
def _(mo, np, plots, plt, readout):
    _fig, _axes = plt.subplots(1, 2, figsize=(9.6, 3.2), constrained_layout=True)
    for _ax, (_name, _label) in zip(_axes, (("log sd", "log standard deviation, $\\log\\hat s$"),
                                            ("mean", "mean return, $\\bar y$ (%)"))):
        _r = readout[_name]
        _ax.scatter(_r["target"], _r["fit"], s=5, alpha=0.4, color=plots.SERIES[0], linewidth=0)
        _lims = np.quantile(_r["target"], [0.001, 0.999])
        _ax.plot(_lims, _lims, color=plots.INK_2, linewidth=0.9)
        _ax.set_xlim(_lims)
        _ax.set_ylim(_lims)
        _ax.set_xlabel(f"true {_label}")
        _ax.set_ylabel("linear read-out of the 8 learned statistics")
        _ax.set_title(f"{_name}: $R^2$ = {_r['r2']:.3f}", loc="left")
    _resid = np.std(readout["log sd"]["target"] - readout["log sd"]["fit"])
    _post_sd = 1 / np.sqrt(2 * 1_000)
    mo.vstack([
        mo.md(r"""
        **What did the summary network learn?** We never told it about means or standard deviations.
        Feed it 1,000 new stocks and try to read the two sufficient statistics off its 8 outputs with a
        plain linear regression:
        """),
        _fig,
        mo.md(f"""
        It rediscovered the volatility statistic on its own ($R^2$ = {readout['log sd']['r2']:.3f}) and,
        roughly, the mean ($R^2$ = {readout['mean']['r2']:.2f}), which matters less here because the drift's
        posterior leans on the prior anyway.

        Why, then, the errors above? Because "almost" is judged against the posterior, and the posterior is
        narrow. The part of $\\log\\hat s$ that the linear read-out misses has a standard deviation of
        {_resid:.3f}, while the posterior sd of $\\log\\sigma$ is about {_post_sd:.3f}: the leftover is
        {_resid / _post_sd:.1f} posterior sds. The flow decodes the summaries non-linearly and does better
        than this read-out, but the scale of the problem is the same: **a learned summary must be accurate
        to a fraction of the posterior's width**, and that width shrinks as the data grow. More simulations
        and longer training close the gap; slot 5's network saw ten times as many.

        That is precisely the situation in slot 5: the model needs a learned summary, and no exact
        posterior exists to check single names against. The answer there is an **audit** that uses the
        exact likelihood, name by name.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(conditions, gv, mo, np, stats_workflow, voo):
    _draws = stats_workflow.sample(conditions=conditions(voo.to_numpy()), num_samples=4_000)
    voo_draws = np.concatenate([_draws["drift"][0], _draws["log_vol"][0]], axis=-1)
    voo_risk = {_h: gv.predictive_var_es(voo_draws[:, 0], voo_draws[:, 1], horizon=_h) for _h in (1, 10)}
    mo.show_code()
    return voo_draws, voo_risk


@app.cell(hide_code=True)
def _(ROOT, gv, json, mo, np, plots, plt, voo, voo_draws, voo_risk):
    _vol = 100 * np.exp(voo_draws[:, 1])
    _lo, _mid, _hi = np.quantile(_vol, [0.05, 0.5, 0.95])
    _drift_lo, _drift_mid, _drift_hi = np.quantile(voo_draws[:, 0], [0.05, 0.5, 0.95])
    _realized = voo.rolling(20).std() * np.sqrt(252)
    _now, _peak, _peak_day = _realized.iloc[-1], _realized.max(), _realized.idxmax()
    _var1, _var10 = voo_risk[1][0], voo_risk[10][0]
    _normal_var10 = {_k: gv.predictive_var_es(_drift_mid, np.log(_v / 100), horizon=10)[0]
                     for _k, _v in (("now", _now), ("peak", _peak))}
    _loss = -voo
    _breach = _loss > _var1
    _n_peak = int(_breach[(_breach.index.year == _peak_day.year) & (_breach.index.month == _peak_day.month)].sum())
    _n_first = int(_breach[_breach.index.year == voo.index[0].year].sum())
    _exact = json.loads((ROOT / "docs" / "results" / "svj-daily" / "VOO_2026-09-03_pmmh.json").read_text())
    _spot = {_k: 100 * _exact["posterior"]["spot vol"][_k] for _k in ("median", "q05", "q95")}

    _fig, (_top, _bottom) = plt.subplots(2, 1, figsize=(9.6, 5.2), sharex=True, constrained_layout=True)
    _top.plot(_realized.index, _realized.to_numpy(), color=plots.INK_2, linewidth=0.9,
              label="realized volatility, last 20 days")
    _top.axhspan(_lo, _hi, color=plots.SERIES[0], alpha=0.25, linewidth=0, label="constant-volatility posterior, 90%")
    _top.axhline(_mid, color=plots.SERIES[0], linewidth=1.4)
    _top.set_ylabel("annualized volatility (%)")
    _top.set_title("What the model assumes, against what VOO did", loc="left")
    _top.legend(loc="upper left", fontsize=8)
    _bottom.plot(_loss.index, _loss.to_numpy(), color=plots.MUTED, linewidth=0.6, label="daily loss")
    _bottom.axhline(_var1, color=plots.SERIES[0], linewidth=1.2, label=f"1-day 99% VaR, {_var1:.2f}%")
    _bottom.scatter(_loss.index[_breach], _loss[_breach], s=26, color=plots.SERIES[1], zorder=3,
                    edgecolor=plots.SURFACE, linewidth=0.6, label=f"breaches: {int(_breach.sum())}")
    _bottom.set_ylabel("daily loss (%)")
    _bottom.set_title("Breaches of the constant-volatility VaR come in clusters", loc="left")
    _bottom.legend(loc="upper left", fontsize=8, ncols=3)
    for _ax in (_top, _bottom):
        _ax.grid(axis="x", visible=False)
    mo.vstack([
        mo.md(f"""
        ## Back to VOO: the answer, and why it is not enough

        **The answer.** One pass through the two-statistics network gives VOO's posterior: volatility
        **{_mid:.1f}%** (90% interval {_lo:.1f}% to {_hi:.1f}%), drift between {_drift_lo:.0f}% and
        {_drift_hi:.0f}% a year, and a **10-day 99% VaR of {_var10:.1f}%** (97.5% ES
        {voo_risk[10][1]:.1f}%). The amortized part works.

        **Is the model right for VOO?** The posterior is conditional on the model, and the model says VOO
        had one volatility for four years. Here is that assumption next to the data:
        """),
        _fig,
        mo.md(f"""
        **What goes wrong.** VOO's 20-day volatility ranged from {_realized.min():.0f}% to {_peak:.0f}%
        (in {_peak_day:%B %Y}), and it is about {_now:.0f}% today. A constant volatility cannot follow
        that, so the risk numbers are wrong in both directions:

        * **Today**, at about {_now:.0f}% volatility, a normal 10-day VaR is about
          {_normal_var10['now']:.1f}%; the model's {_var10:.1f}% is {_var10 / _normal_var10['now']:.1f}
          times that.
        * **In {_peak_day:%B %Y}**, at {_peak:.0f}%, it would have been about
          {_normal_var10['peak']:.0f}%; the model understated it by a factor of
          {_normal_var10['peak'] / _var10:.1f}.
        * The 1-day 99% VaR was breached on {int(_breach.sum())} days where about
          {0.01 * len(voo):.0f} were expected, and the breaches cluster: {_n_peak} in
          {_peak_day:%B %Y} alone, and {_n_first} in the autumn-{voo.index[0].year} sell-off. Under a
          correct model they would fall at random times.

        **The lesson.** A posterior is only as good as the model behind it. Its narrow interval is
        uncertainty about a four-year average volatility **given** the model, and it cannot tell you that
        the model is wrong. Two remedies follow:

        * **Slot 5** keeps the code and changes the model: volatility moves, prices jump, and today's
          volatility is one of the unknowns. For VOO on {voo.index[-1].day} {voo.index[-1]:%B %Y} it puts today's
          volatility at about **{_spot['median']:.0f}%** (90%: {_spot['q05']:.0f}% to
          {_spot['q95']:.0f}%), not {_mid:.0f}%.
        * **Slot 6** shows how to notice when the model, not the network, is wrong.
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
    diagnostics, an exact check, and a forecast that carries parameter uncertainty into VaR and ES. You
    have also met its two limits: a learned summary must be accurate to a fraction of the posterior's
    width, and no network can repair a wrong model.

    | | This notebook | Slot 5 |
    |:---|:---|:---|
    | Model | constant volatility | volatility that moves, plus jumps |
    | Unknowns | drift, volatility | 8 parameters and **today's volatility** |
    | What the network sees | two sufficient statistics, or 8 learned ones | 32 learned statistics |
    | Training | 1 to 5 minutes | 79 minutes, once |
    | Is it right? | compare with the exact posterior | no exact posterior: the particle filter, particle MCMC and an audit |
    | Parameter uncertainty in VaR / ES | negligible with 1,000 days | matters in the tail, because of the jumps |

    The BayesFlow code you wrote today runs unchanged in slot 5. What changes is the simulator, and
    everything a harder model demands.
    """)
    return


if __name__ == "__main__":
    app.run()
