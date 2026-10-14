import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Slot 5 · When volatility moves and prices jump")


@app.cell(hide_code=True)
def _():
    import json
    import os
    import time

    os.environ.setdefault("KERAS_BACKEND", "jax")

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from sbi4finance.svj import amortized, audit, data, filtering, model, plots, risk, teaching, tutorial

    ROOT = mo.notebook_dir().parent
    RESULTS = ROOT / "docs" / "results"
    plots.style()
    return (
        RESULTS, ROOT, amortized, audit, data, filtering, json, mo, model, np, plots, plt, risk, teaching,
        time, tutorial,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Slot 5 · When volatility moves and prices jump

    **Amortized Bayesian inference for market risk, with a model nobody can solve exactly**

    In slot 4 we asked one question of 1,000 daily returns: what are the drift and the volatility, and
    what is the 10-day value-at-risk? Five BayesFlow calls answered it, and the exact posterior confirmed
    every step.

    Real returns do not behave like that model. Volatility clusters, with calm months followed by storms,
    and prices sometimes jump. This notebook keeps the question and the code and changes one thing:
    **the model**. That change removes the exact answer, and it is exactly what makes amortized inference
    worth having.

    Three claims, each backed by evidence below:

    1. **The exact Bayesian answer exists but costs about 35 CPU-minutes per name.** The fast classical
       shortcut hides precisely the uncertainty that drives the tail.
    2. **An amortized posterior gives the Bayesian answer in about 3 seconds per name**, and an audit with
       the exact likelihood says, name by name, when to trust it.
    3. **Parameter uncertainty moves the risk numbers where data are scarce: the jumps.** A forecast at a
       point estimate understates the expected shortfall; the predictive forecast does not.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | | Slot 4: constant volatility | Slot 5: stochastic volatility with jumps |
    |---|---|---|
    | Data | last 1,000 daily returns | the same |
    | Unknowns | drift, volatility | 8 parameters and **today's volatility** |
    | Simulator | three lines of NumPy | the daily SVJ model |
    | BayesFlow pieces | `Adapter`, `FlowMatching`, `BasicWorkflow`, then a small `TimeSeriesNetwork` | the same, with a larger summary network |
    | Training | one to five minutes on a 4-core machine | once, 79 minutes, 2.56 million simulated series |
    | Is the answer right? | compare with the exact posterior: the learned summary was off by more than 1 sd on some stocks | no exact posterior: particle filter, particle MCMC and an audit |
    | Risk forecast | simulate from the drift and volatility | simulate from the parameters **and today's volatility** |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "**The model in detail** (click to open): from prices to posterior and risk": mo.md(r"""
        #### 1 · From prices to returns

        In continuous time this is the Bates (1996) model: Heston's mean-reverting stochastic variance
        plus Merton's normal jumps in the price. We work with its one-day Euler form (Eraker, Johannes and
        Polson, 2003) and treat that form as the model itself: the simulator, the particle filter, the
        network and the forecast all use exactly these equations. With $y_t$ the daily log return in
        percent and $v_t$ the variance in %² per day,

        $$y_t = \mu + \sqrt{v_{t-1}}\,\varepsilon^1_t + B_t\,\xi_t, \qquad B_t \sim \text{Bernoulli}(\lambda), \quad \xi_t \sim \mathcal N(\mu_J, \sigma_J^2),$$

        $$v_t = \max\big(v_{t-1} + \kappa(\theta - v_{t-1}) + \sigma_v\sqrt{v_{t-1}}\,(\rho\,\varepsilon^1_t + \sqrt{1-\rho^2}\,\varepsilon^2_t),\ v_{\min}\big),$$

        with $\varepsilon^1_t$ and $\varepsilon^2_t$ independent standard normals. Read it in four parts:

        * **Return = drift + diffusion + jump.** Yesterday's variance sets the scale of today's move; on a
          fraction $\lambda$ of days a normal jump $\xi_t$ is added.
        * **The variance mean-reverts** to its long-run level $\theta$ at rate $\kappa$ a day, so a shock
          to it halves in $\ln 2/\kappa$ days.
        * **The variance is itself random**, with volatility of variance $\sigma_v$, and its shock shares
          $\varepsilon^1_t$ with the return: with $\rho < 0$, falling prices push volatility up
          (**leverage**).
        * **The floor** $v_{\min} = 10^{-3}$, a volatility of 0.5% a year, keeps the variance positive when
          a daily step would overshoot below zero.

        The window starts from $v_0 \sim \text{Gamma}(1/c^2,\ \theta c^2)$ with $c = \sigma_v/\sqrt{2\kappa\theta}$:
        the stationary law of the continuous-time variance, with mean $\theta$ and coefficient of
        variation $c$.

        #### 2 · The nine unknowns

        Eight parameters, each drawn and inferred in units a risk manager can judge and mapped to an
        unbounded scale for the network (logs and atanh), plus one hidden state:

        | Quantity | Definition | In the code |
        |:---|:---|:---|
        | Drift | $252\,\mu$, % a year | `drift` |
        | Long-run volatility | $\sqrt{252\,\theta}/100$ | `log_vol_lr` |
        | Half-life of variance shocks | $\ln 2/\kappa$, trading days | `log_half_life` |
        | CV of variance | $c = \sigma_v/\sqrt{2\kappa\theta}$ | `log_cv` |
        | Leverage | $\rho$ | `atanh_rho` |
        | Jump frequency | $252\,\lambda$, jumps a year | `log_jumps_per_year` |
        | Mean jump | $\mu_J$, % | `mean_jump` |
        | Jump size sd | $\sigma_J$, % | `log_jump_sd` |
        | **Today's volatility** | $\sigma_T = \sqrt{252\,v_T}/100$ | `log_spot_vol` |

        Two choices deserve a word. The **CV of variance** replaces $\sigma_v$ because it reads directly:
        $c = 0.6$ means the variance's typical deviation from its long-run level is 60% of that level,
        and $c < 1$ is exactly the Feller condition $2\kappa\theta > \sigma_v^2$, under which the
        continuous-time variance never touches zero. **Today's volatility** is not a parameter but the
        end of the hidden path; it is a target because every forecast starts from it (section 2).

        #### 3 · The prior

        | Quantity | Prior | 95% range | What it encodes |
        |:---|:---|:---|:---|
        | Drift | $\mathcal N(8,\ 10^2)$ | −12% to 28% a year | as in slot 4 |
        | Long-run volatility | log-normal, median 25%, log-sd 0.40 | 11% to 55% | as in slot 4 |
        | Half-life | log-normal, median 30 days, log-sd 0.70 | 8 to 118 days | shocks fade over weeks to months |
        | CV of variance | log-normal, median 0.6, log-sd 0.35 | 0.30 to 1.19 | volatility is volatile; above 1, Feller fails |
        | Leverage | $(\rho + 1)/2 \sim \text{Beta}(2, 4)$ | −0.89 to 0.43 | mostly negative, mean −1/3 |
        | Jump frequency | log-normal, median 2.5 a year, log-sd 0.75 | 0.6 to 11 a year | a few jumps a year |
        | Mean jump | $\mathcal N(-1,\ 2.5^2)$, % | −5.9% to 3.9% | on average down: crashes outnumber spikes |
        | Jump size sd | log-normal, median 3%, log-sd 0.45 | 1.2% to 7.3% | jumps of a few percent |

        The eight priors are independent, and the daily jump probability is capped at 0.5, a cap reached
        only far in the prior's tail. Today's volatility has no prior of its own: it is whatever the model
        produces after 1,000 days.

        #### 4 · Why the likelihood has no closed form

        The variance path is hidden, so the likelihood integrates it out:

        $$p(y_{1:T} \mid \vartheta) = \int p(v_0 \mid \vartheta) \prod_{t=1}^{T} p(y_t \mid v_{t-1}, \vartheta)\, p(v_t \mid v_{t-1}, y_t, \vartheta)\; dv_{0:T},$$

        a 1,000-dimensional integral with no closed form, where $\vartheta$ collects the eight parameters.
        Each day's factor is simple. Given yesterday's variance, today's return is a two-component mixture,

        $$p(y_t \mid v, \vartheta) = (1 - \lambda)\, \mathcal N(y_t;\ \mu,\ v) + \lambda\, \mathcal N(y_t;\ \mu + \mu_J,\ v + \sigma_J^2),$$

        and given the return and whether a jump occurred, today's variance is normal before the floor. On a
        day without a jump, $\sqrt{v_{t-1}}\,\varepsilon^1_t = y_t - \mu$ is known, so

        $$v_t \mid v_{t-1}, y_t \;\sim\; \mathcal N\big(v_{t-1} + \kappa(\theta - v_{t-1}) + \sigma_v \rho\, (y_t - \mu),\ \ \sigma_v^2 (1 - \rho^2)\, v_{t-1}\big):$$

        with $\rho < 0$, a down day raises tomorrow's variance. Two consequences:

        * **No sufficient statistics.** Every return matters, and so does their order. Slot 4 compressed
          1,000 returns into two numbers without loss; here a summary network learns 32.
        * **The particle filter estimates the integral** (section 3). Particles of $v$ are weighted by
          $p(y_t \mid v)$, resampled, and moved with the exact law above. The product of the daily average
          weights is an **unbiased** estimate of the likelihood: noisy, but never biased.

        #### 5 · The exact posterior, and what it costs

        $$p(\vartheta, v_T \mid y_{1:T}) = p(v_T \mid y_{1:T}, \vartheta)\; p(\vartheta \mid y_{1:T}), \qquad p(\vartheta \mid y_{1:T}) \propto p(y_{1:T} \mid \vartheta)\, p(\vartheta).$$

        **Particle MCMC** runs a Metropolis–Hastings chain on $\vartheta$ with a fresh particle-filter
        estimate in place of the likelihood at every proposal. Because that estimate is unbiased, the
        chain still targets the exact posterior (Andrieu, Doucet and Holenstein, 2010), and keeping one
        final particle of each accepted run adds $v_T$. It is exact but slow: about **35 CPU-minutes per
        name** (median of five windows, between 12 and 52 minutes), even with chains started at the
        network's draws. For 2,000 names that is about 1,150 CPU-hours a day.

        Between the network and particle MCMC sits the **audit** (section 7): importance sampling with the
        same unbiased likelihood, from a Student t fitted to the network's draws. Its efficiency, the
        effective sample size over the number of proposals, is the trust score; below 2% the name goes to
        particle MCMC. It costs about 75 CPU-seconds.

        #### 6 · What 1,000 days can and cannot tell you

        Contraction measures how much the data shrink the prior, $1 - \text{posterior variance} / \text{prior variance}$,
        averaged over held-out simulated series (section 6):

        | Quantity | Contraction |
        |:---|:---|
        | long-run volatility | 0.94 |
        | CV of variance | 0.82 |
        | leverage | 0.82 |
        | today's volatility | 0.76 |
        | mean jump | 0.58 |
        | half-life | 0.57 |
        | drift | 0.40 |
        | jump size sd | 0.39 |
        | jump frequency | 0.35 |

        The volatility process is learned well. The drift learns exactly as little as in slot 4: 0.40 in
        both. The jumps are learned least: four years at a few jumps a year give a handful of jump days,
        too few to pin down how often jumps come, how large they are and how much they vary. That is where
        the parameter uncertainty lives, and the tail of the forecast depends on exactly those parameters.

        #### 7 · From parameters to risk

        As in slot 4, risk is quoted as a loss in percent: the 99% VaR and the 97.5% ES of the $h$-day log
        return. Now no formula gives them. The predictive distribution

        $$p(R_{T,h} \mid y_{1:T}) = \int p(R_{T,h} \mid v_T, \vartheta)\; p(v_T, \vartheta \mid y_{1:T})\; dv_T\, d\vartheta$$

        is simulated: from every posterior draw $(\vartheta, v_T)$ the daily model runs $h$ days forward with
        fresh shocks and jumps, and the pooled paths are draws from the predictive law. The **plug-in**
        forecast fixes $\vartheta$ at the posterior median and keeps only the uncertainty about $v_T$,
        filtered at that point. Each posterior draw also gives a **conditional VaR**, the number an analyst
        would quote if $(\vartheta, v_T)$ were known; their spread shows how uncertain the risk number
        itself is.

        On 500 simulated series (section 8), the predictive forecast scores like a forecast made with the
        true parameters, while the plug-in breaches its VaR more often and understates the ES.

        #### 8 · What the model still leaves out

        Even this model simplifies. Its parameters are constant over the four-year window, the variance has
        a single factor, jump sizes are normal and arrive independently, volatility itself never jumps,
        and only daily closes are used. When real data break these assumptions, the network still returns
        a posterior, but for a model that does not describe the data. The misspecification test in
        section 10 asks whether VOO's learned summaries look like simulated ones, and slot 6 takes up
        priors and misspecification.

        #### 9 · What the network sees and returns

        | Piece | Content |
        |:---|:---|
        | Input | two channels per day for 1,000 days, as in slot 4: $y_t$ and $\log(y_t^2 + 0.01)$ |
        | Summary network | convolutions 8, 5 and 5 days wide (strides 4, 2, 2), read by forward and backward GRUs, plus a skip branch: 32 learned statistics |
        | Inference network | flow matching, integrated in 50 Euler steps |
        | Output | joint draws of the nine targets, in the code's coordinates |
        | Training | once, online, on 2.56 million simulated series (79 minutes) |

        A new name costs one pass of the summary network and 50 Euler steps: about 3 CPU-seconds for its
        posterior, against about 35 CPU-minutes for particle MCMC.
        """),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · The model: volatility moves, prices jump

    Daily log returns $y_t$ in percent, driven by a hidden variance $v_t$:

    $$y_t = \mu + \sqrt{v_{t-1}}\,\varepsilon^1_t + B_t\,\xi_t,\qquad
    B_t\sim\text{Bernoulli}(\lambda),\quad \xi_t\sim\mathcal N(\mu_J,\sigma_J^2),$$

    $$v_t = v_{t-1} + \kappa(\theta - v_{t-1}) + \sigma_v\sqrt{v_{t-1}}\,\big(\rho\,\varepsilon^1_t + \sqrt{1-\rho^2}\,\varepsilon^2_t\big).$$

    This is the Bates (1996) model in the daily form of Eraker, Johannes and Polson (2003). A risk manager
    reads it like this:

    - volatility is pulled back to a **long-run level** with a **half-life** of a few weeks;
    - volatility is itself volatile (the **CV of variance**), and it rises when prices fall (**leverage**, $\rho < 0$);
    - a few times a year a **jump** adds a large move of typical size $\mu_J \pm \sigma_J$.

    That makes eight parameters, plus the one number every forecast needs and nobody observes: **today's
    volatility**. The prior is stated in these units; its drift and long-run volatility are slot 4's prior.
    """)
    return


@app.cell
def _(model):
    model.prior_table().round(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Four years of a simulated stock

    We simulate 1,000 days with a long-run volatility of 18%, a half-life of 35 days, leverage of −0.6
    and two jumps a year of about −2.5% ± 3.5%. Because we simulated it, we know everything, including
    what is normally hidden.
    """)
    return


@app.cell
def _(model, teaching):
    _artifact, _simulation = teaching.synthetic_case()
    example_returns = _simulation["returns"][0]
    example_truth = _simulation["targets"][0]
    example_vol = model.spot_vol_from_variance(_simulation["variance"][0])
    example_jumps = _simulation["jumps"][0].astype(bool)
    return example_jumps, example_returns, example_truth, example_vol


@app.cell(hide_code=True)
def _(example_jumps, example_returns, example_truth, example_vol, model, np, plots, plt):
    _fig, (_top, _bottom) = plt.subplots(2, 1, figsize=(9.6, 4.8), sharex=True, constrained_layout=True)
    _days = np.arange(1, model.WINDOW + 1)
    _top.plot(_days, example_returns, color=plots.SERIES[0], linewidth=0.7)
    _top.scatter(_days[example_jumps], example_returns[example_jumps], s=28, color=plots.SERIES[1], zorder=3,
                 edgecolor=plots.SURFACE, linewidth=0.6, label="days with a jump (known only because we simulated)")
    _top.set_ylabel("daily log return (%)")
    _top.set_title("What a risk manager sees: the returns", loc="left")
    _top.legend(loc="lower left")
    _bottom.plot(_days, 100 * example_vol, color=plots.INK_2, linewidth=1.1, label="spot volatility")
    _bottom.axhline(100 * np.exp(example_truth[1]), color=plots.MUTED, linestyle="--", linewidth=0.9,
                    label="long-run level")
    _bottom.legend(loc="upper right", ncols=2, fontsize=8)
    _bottom.set_ylabel("spot volatility (%)")
    _bottom.set_xlabel("trading day")
    _bottom.set_title("What drives the risk, and is never observed: today's volatility", loc="left")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · The number that matters is hidden

    The forecast must average over everything we do not know on day $T$: the parameters $\vartheta$
    **and** today's variance $v_T$,

    $$p(R_{T,H}\mid y_{1:T})=\int \underbrace{p(R_{T,H}\mid v_T,\vartheta)}_{\text{simulate forward}}\;
    \underbrace{p(v_T,\vartheta\mid y_{1:T})}_{\text{the posterior}}\;dv_T\,d\vartheta.$$

    How much does today's volatility matter? Keep the parameters fixed at their true values and simulate
    the next 10 days from three values of today's volatility:
    """)
    return


@app.cell
def _(example_truth, model, np, risk):
    spot_vols = (0.08, 0.12, 0.24)
    _rng = np.random.default_rng(5)
    ten_day_returns = {}
    for _spot in spot_vols:
        _target = np.append(example_truth[: model.N_PARAMS], np.log(_spot))
        ten_day_returns[_spot] = model.forward_returns(_target[None], 10, 40_000, _rng)[0].sum(axis=-1)
    spot_risk = {_spot: risk.var_es(_returns) for _spot, _returns in ten_day_returns.items()}
    return spot_risk, spot_vols, ten_day_returns


@app.cell(hide_code=True)
def _(np, plots, plt, spot_risk, spot_vols, ten_day_returns):
    _fig, _ax = plt.subplots(figsize=(9.6, 3.4), constrained_layout=True)
    _bins = np.linspace(-22, 14, 145)
    for _color, _spot in zip(plots.SERIES, spot_vols):
        _var, _es = spot_risk[_spot]
        _ax.hist(ten_day_returns[_spot], bins=_bins, density=True, histtype="step", linewidth=1.6, color=_color,
                 label=f"today's volatility {_spot:.0%}: 10-day 99% VaR {_var:.1f}%, 97.5% ES {_es:.1f}%")
        _ax.axvline(-_var, color=_color, linewidth=0.9, linestyle="--")
    _ax.set_xlabel("10-day log return (%); dashed lines at minus the 99% VaR")
    _ax.set_ylabel("density")
    _ax.set_title("Same parameters, three values of today's volatility", loc="left")
    _ax.grid(axis="y", visible=False)
    _ax.legend(loc="upper left", fontsize=8)
    _fig
    return


@app.cell(hide_code=True)
def _(mo, spot_risk, spot_vols):
    _low, _high = spot_risk[spot_vols[0]][0], spot_risk[spot_vols[-1]][0]
    mo.md(f"""
    The 10-day VaR goes from **{_low:.1f}%** to **{_high:.1f}%**, a factor of {_high / _low:.1f}, with nothing
    changed but today's volatility. At low volatility the 1% tail is set by the jumps, so 8% and 12% give
    similar VaRs; at 24% the ordinary moves take over. On the real VOO window at the end of this notebook, today's
    volatility accounts for about **90% of the uncertainty of the 1-day risk number**. The forecast
    needs today's volatility, and it is hidden: it must be inferred from the returns, **jointly with
    the parameters**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Why there is no formula for the posterior

    Bayes' rule needs the likelihood $p(y_{1:T}\mid\vartheta)$. With a hidden variance it is an integral
    over the whole variance path, 1,001 dimensions with no closed form:

    $$p(y_{1:T}\mid\vartheta)=\int p(v_0\mid\vartheta)\prod_{t=1}^{T}
    p(y_t\mid v_{t-1},\vartheta)\,p(v_t\mid v_{t-1},y_t,\vartheta)\;dv_{0:T}.$$

    A **particle filter** estimates it by carrying a cloud of plausible variances through the 1,000 days,
    reweighting the cloud by how well each variance explains each day's return. Here it runs on the
    simulated stock at the true parameters. It sees only the returns.
    """)
    return


@app.cell
def _(example_returns, example_truth, model, tutorial):
    filtered_vol, jump_probability = tutorial.filtered_path(
        model.model_params(example_truth), example_returns, seed=0, n_particles=1_000
    )
    return filtered_vol, jump_probability


@app.cell(hide_code=True)
def _(example_jumps, example_vol, filtered_vol, jump_probability, model, np, plots, plt):
    _fig, (_top, _bottom) = plt.subplots(
        2, 1, figsize=(9.6, 4.8), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    _days = np.arange(1, model.WINDOW + 1)
    _top.fill_between(_days, 100 * filtered_vol[:, 0], 100 * filtered_vol[:, 2], color=plots.SERIES[0], alpha=0.18,
                      linewidth=0, label="filter: 90% band")
    _top.plot(_days, 100 * filtered_vol[:, 1], color=plots.SERIES[0], linewidth=1.2, label="filter: median")
    _top.plot(_days, 100 * example_vol, color=plots.INK, linewidth=0.8, label="true spot volatility (hidden)")
    _top.set_ylabel("spot volatility (%)")
    _top.legend(loc="upper right", ncols=3, fontsize=8)
    _top.set_title("The particle filter tracks the hidden volatility from the returns alone", loc="left")
    _bottom.vlines(_days, 0, jump_probability, color=plots.SERIES[0], linewidth=0.8)
    _bottom.scatter(_days[example_jumps], np.full(example_jumps.sum(), 1.08), marker="v", s=26,
                    color=plots.SERIES[1])
    _bottom.set_ylim(0, 1.18)
    _bottom.set_ylabel("P(jump today)")
    _bottom.set_xlabel("trading day")
    _bottom.set_title("The filter's probability of a jump on each day; triangles mark the true jump days",
                      loc="left", fontsize=9)
    _inside = np.mean((example_vol >= filtered_vol[:, 0]) & (example_vol <= filtered_vol[:, 2]))
    _fig.text(0.99, 0.01, f"true volatility inside the 90% band on {_inside:.0%} of days", ha="right",
              color=plots.INK_2, fontsize=8)
    _fig
    return


@app.cell
def _(example_returns, example_truth, filtering, model):
    loglik_noise = filtering.loglik_sd(
        model.model_params(example_truth), example_returns, seed=0, n_particles=100, n_replicates=32
    )
    return (loglik_noise,)


@app.cell(hide_code=True)
def _(RESULTS, json, loglik_noise, mo):
    _costs = json.loads((RESULTS / "svj-costs" / "costs.json").read_text())
    _pmmh = _costs["pmmh"]
    mo.md(f"""
    Big jumps are spotted, and the volatility band follows the truth. Small jumps hide among ordinary
    days. But each run of the filter gives only a **noisy estimate** of the likelihood: with 100 particles,
    the log-likelihood varies by **±{loglik_noise:.2f}** from run to run.

    **Particle MCMC** builds an exact posterior sampler on these noisy estimates (Andrieu, Doucet and
    Holenstein, 2010). It works, at a price. On four test series it needed
    **{_pmmh['range_cpu_s'][0]:,.0f} to {_pmmh['range_cpu_s'][1]:,.0f} CPU-seconds per 400 effective draws**
    (median {_pmmh['median_cpu_s']:,.0f}, about {_pmmh['median_cpu_s'] / 60:.0f} minutes). For 2,000 names
    every morning that is about **{_pmmh['cpu_hours_for_2000_names_at_median']:,.0f} CPU-hours a day**.
    Correct, and unaffordable.
    """)
    return


@app.cell(hide_code=True)
def _(RESULTS, json, mo):
    _classical = json.loads((RESULTS / "svj-classical" / "classical.json").read_text())
    _wald, _post = _classical["wald_coverage_95"], _classical["posterior_coverage_95"]
    _n = _classical["n_series"]
    mo.vstack([
        mo.md(f"""
        ## 4 · The fast classical answer, and what it hides

        The usual shortcut is to maximize the likelihood once (here with an accurate grid filter), read
        standard errors off the Hessian, and forecast at the estimate. On {_n} simulated series with known
        parameters, how often do the 95% intervals contain the truth?
        """),
        mo.image(RESULTS / "svj-classical" / "coverage.png", width=760),
        mo.md(f"""
        - Wald intervals contain the true **jump frequency** in {_wald['jumps per year'] * _n:.0f} of {_n}
          series and the true **mean jump** in {_wald['mean jump (%)'] * _n:.0f} (a 95% interval should
          manage about {0.95 * _n:.0f}). The posterior intervals: {_post['jumps per year'] * _n:.0f} and
          {_post['mean jump (%)'] * _n:.0f}.
        - {_classical['fits_with_a_parameter_on_a_bound']} of the {_n} fits end on a boundary, mostly a
          jump-size sd at its floor, and {_classical['hessians_not_positive_definite']} Hessians are not
          positive definite, so no interval exists at all.
        - Each fit costs about {_classical['median_cpu_s_per_fit']:,.0f} CPU-seconds, and returns one point,
          with no distribution to forecast from.

        Four years of returns contain a handful of jumps. Point estimates and Wald intervals behave as if
        the jump parameters were known. The posterior admits they are not.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Same BayesFlow code, new simulator

    The workflow is slot 4's, line for line. What changes is what the model demands:

    ```python
    simulator = model.SVJSimulator()                       # was: make_simulator(seed) with constant volatility
    adapter = (
        bf.Adapter()
        .convert_dtype("float64", "float32")
        .rename("targets", "inference_variables")          # 8 parameters + log of today's volatility
        .rename("features", "summary_variables")           # the same two channels: y and log(y² + 0.01)
    )
    summary_network = bf.networks.TimeSeriesNetwork(       # slot 4's class, larger: order now matters,
        summary_dim=32, filters=(32, 64, 64),              # because recent days reveal today's volatility
        kernel_sizes=(8, 5, 5), strides=(4, 2, 2), recurrent_dim=64,
    )
    inference_network = bf.networks.FlowMatching()         # the same class, at its default size
    workflow = bf.BasicWorkflow(simulator=simulator, adapter=adapter,
                                summary_network=summary_network, inference_network=inference_network,
                                standardize="inference_variables")
    workflow.fit_online(epochs=50, num_batches_per_epoch=400, batch_size=128)   # once: 79 minutes on 4 CPU cores
    ```

    That training ran once, before the tutorial, on 2.56 million simulated four-year histories. From
    now on every name costs one pass through the network:
    """)
    return


@app.cell
def _(ROOT, amortized, example_returns, time):
    _start = time.time()
    posterior = amortized.AmortizedPosterior.load(ROOT / "checkpoints" / "svj")
    load_seconds = time.time() - _start
    posterior.sample(example_returns[None], num_samples=10, seed=0)  # compiles the sampler once
    _start = time.time()
    example_draws = posterior.sample(example_returns[None], num_samples=2_000, seed=1)[0]
    sample_seconds = time.time() - _start
    return example_draws, load_seconds, posterior, sample_seconds


@app.cell
def _(model):
    prior_targets = model.simulate(4_000, seed=11)["targets"]
    return (prior_targets,)


@app.cell(hide_code=True)
def _(example_draws, example_truth, model, np, plots, plt, prior_targets, sample_seconds):
    _post, _prior, _truth = model.quantities(example_draws), model.quantities(prior_targets), model.quantities(example_truth)
    _fig, _axes = plt.subplots(3, 3, figsize=(9.6, 6.4), constrained_layout=True)
    for _j, _ax in enumerate(_axes.ravel()):
        _lo, _hi = np.quantile(_prior[:, _j], [0.01, 0.99])
        _lo, _hi = min(_lo, _truth[_j]), max(_hi, _truth[_j])
        _bins = np.linspace(_lo, _hi, 50)
        _ax.hist(np.clip(_prior[:, _j], _lo, _hi), bins=_bins, density=True, color=plots.GRID, label="prior")
        _ax.hist(np.clip(_post[:, _j], _lo, _hi), bins=_bins, density=True, color=plots.SERIES[0], alpha=0.85,
                 label="posterior")
        _ax.axvline(_truth[_j], color=plots.INK, linewidth=1.2, label="truth")
        _ax.set_title(model.QUANTITY_LABELS[_j], loc="left")
        _ax.set_yticks([])
        _ax.grid(visible=False)
    _axes[0, 0].legend(fontsize=7, loc="upper left")
    _fig.suptitle(f"The posterior for the simulated stock: 2,000 draws in {sample_seconds:.1f} s",
                  x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=plots.INK)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the panels against the prior (grey). The data pin down the volatility dynamics: the long-run
    level, how volatile volatility is and the leverage. The **jump frequency** barely moves. The jump
    size shifts, because this path holds two very large losses, but it stays wide. That is not a failure
    of the network: four years contain about eight jumps, and eight observations cannot pin down a
    frequency, a mean and a spread. **This is where parameter uncertainty lives, and where a point
    estimate misleads.**

    Today's volatility is centred near 16%, while the truth is 12%. That is the data speaking, not the
    network: the last weeks of returns are larger than 12% volatility would suggest, and the particle
    filter at the true parameters (previous section) ends in the same place. The posterior is right to
    leave the truth in its lower part.
    """)
    return


@app.cell(hide_code=True)
def _(RESULTS, json, mo, model):
    _validation = json.loads((RESULTS / "svj-validation" / "validation.json").read_text())
    _rows = []
    for _name, _label in zip(model.TARGET_NAMES, model.QUANTITY_LABELS):
        _target = _validation["targets"][_name]
        _rows.append(f"| {_label} | {_target['contraction']:.2f} | {_target['coverage']['90%']:.0%} |")
    _table = "\n".join(_rows)
    mo.vstack([
        mo.md(f"""
        ## 6 · Is the network right, on average?

        Before trusting a network on one series, check it on many where we know the truth. Over
        {_validation['n_series']:,} held-out simulated series, the posterior medians recover the truth
        where the data are informative, and fall back to the prior where they are not:
        """),
        mo.image(RESULTS / "svj-validation" / "recovery.png", width=760),
        mo.md("\n".join([
            "| Target | Share of prior variance removed | 90% intervals that contain the truth |",
            "|---|---|---|",
            _table,
            "",
            "Calibration holds on average: simulation-based calibration (SBC) keeps 8 of 9 targets inside "
            "its 95% bands, and all 9 on a fresh replicate. But \"on average\" is not \"for this name\".",
        ])),
    ])
    return


@app.cell(hide_code=True)
def _(RESULTS, json, mo):
    _study = json.loads((RESULTS / "svj-agreement" / "agreement.json").read_text())["series"]
    _rows = "\n".join(
        f"| {_s['series']} | {_s['network']['largest_mean_difference_sd']:.2f} sd | "
        f"{_s['importance_sampling']['largest_mean_difference_sd']:.2f} sd | "
        f"{_s['importance_sampling']['efficiency']:.1%}{' · **flagged**' if _s['importance_sampling']['efficiency'] < 0.02 else ''} | "
        f"{_s['pmmh']['cpu_s_per_400_ess']:,.0f} / {_s['importance_sampling']['cpu_s_per_400_effective_draws']:,.0f} |"
        for _s in _study
    )
    mo.vstack([
        mo.md(r"""
        ## 7 · Trust, but verify: the audit

        Four held-out series, each compared with the exact posterior from particle MCMC, target by
        target. The criterion: posterior means within 0.3 posterior sd, and 90% intervals within 25% of
        the exact width.
        """),
        mo.image(RESULTS / "svj-agreement" / "agreement.png", width=760),
        mo.md("\n".join([
            "| Series | Network alone: largest error | After the audit | Audit efficiency "
            "| CPU-s per 400 draws: particle MCMC / audit |",
            "|---|---|---|---|---|",
            _rows,
            "",
            "The network alone misses the criterion on every series, mostly in leverage: calibrated on "
            "average, off for individual names. The audit repairs three of the four, at 5 to 9 times less "
            "CPU than particle MCMC. On the fourth it scores below 2%, so the daily process sends that name "
            "to particle MCMC. **The audit flags exactly the series it cannot repair.**",
        ])),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How the audit works

    The network proposes; the exact likelihood decides. Fit a Student-t distribution to the network's
    draws, draw $M$ proposals $\vartheta^{(m)}$ from it, and weight each by what Bayes' rule would say,
    using the particle filter's likelihood estimate:

    $$w^{(m)} \propto \frac{p(\vartheta^{(m)})\,\hat L(\vartheta^{(m)})}{g(\vartheta^{(m)})},
    \qquad \text{efficiency} = \frac{1}{M}\,\frac{\big(\sum_m w^{(m)}\big)^2}{\sum_m \big(w^{(m)}\big)^2}.$$

    The reweighted draws are a consistent estimate of the exact posterior, whatever the network got
    wrong. The efficiency, the share of effective draws, is the score. If the network's posterior is off
    by $\delta$ (in posterior sd, jointly over the nine targets), the efficiency falls roughly like
    $e^{-\delta^2}$. A low score therefore means: do not trust the network for this name.
    """)
    return


@app.cell
def _(audit, example_draws, example_returns):
    example_audit = audit.audit(example_draws, example_returns, seed=0, n_proposals=1_000, n_particles=100)
    return (example_audit,)


@app.cell(hide_code=True)
def _(example_audit, example_draws, example_truth, mo, model, np, plots, plt):
    _picks = (4, 5, 8)
    _net, _checked = model.quantities(example_draws), model.quantities(example_audit.posterior)
    _truth = model.quantities(example_truth)
    _fig, _axes = plt.subplots(1, 3, figsize=(9.6, 2.7), constrained_layout=True)
    for _ax, _j in zip(_axes, _picks):
        _bins = np.linspace(*np.quantile(np.concatenate([_net[:, _j], _checked[:, _j]]), [0.002, 0.998]), 40)
        _ax.hist(_net[:, _j], bins=_bins, density=True, histtype="step", linewidth=1.5, color=plots.SERIES[0],
                 label="network")
        _ax.hist(_checked[:, _j], bins=_bins, density=True, histtype="step", linewidth=1.5, color=plots.SERIES[2],
                 label="after the audit")
        _ax.axvline(_truth[_j], color=plots.INK, linewidth=1.1, label="truth")
        _ax.set_title(model.QUANTITY_LABELS[_j], loc="left")
        _ax.set_yticks([])
        _ax.grid(visible=False)
    _axes[0].legend(fontsize=7, loc="upper left")
    _verdict = "flagged: send to particle MCMC" if example_audit.flagged() else "passes: the network's answer stands"
    mo.vstack([
        mo.md(f"""
        **Live audit of the simulated stock:** {example_audit.n_proposals:,} proposals, {example_audit.n_particles}
        particles each, {example_audit.seconds:.0f} CPU-seconds. Efficiency **{example_audit.efficiency:.1%}**
        ({example_audit.ess:.0f} effective draws), so it **{_verdict}**. The audited histograms are spiky
        because they resample about {example_audit.ess:.0f} effective draws. With 1,000 proposals the score is
        itself noisy; the daily process uses 3,000.
        """),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    hardest_audit_button = mo.ui.run_button(
        label="Audit the hardest test series live (frequent jumps, high vol-of-variance; about 30 s)"
    )
    hardest_audit_button
    return (hardest_audit_button,)


@app.cell
def _(audit, hardest_audit_button, mo, posterior, tutorial):
    mo.stop(not hardest_audit_button.value)
    _returns, _truth, _name = tutorial.agreement_series(3)
    _draws = posterior.sample(_returns[None], num_samples=1_000, seed=3)[0]
    _checked = audit.audit(_draws, _returns, seed=3, n_proposals=3_000)
    mo.md(f"""
    **{_name}:** efficiency **{_checked.efficiency:.2%}** with {_checked.n_proposals:,} proposals and
    {_checked.n_particles} particles ({_checked.seconds:.0f} CPU-seconds):
    **{'flagged, so this name goes to particle MCMC' if _checked.flagged() else 'passes'}**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Why it matters for the risk numbers

    Push every posterior draw forward: simulate 10 days from its parameters **and** its own value of
    today's volatility, and pool the paths. That is the predictive forecast. The shortcut fixes the
    parameters at a point estimate and keeps only today's volatility uncertain.
    """)
    return


@app.cell
def _(example_audit, example_draws, example_returns, risk):
    example_posterior = example_draws if example_audit.flagged() else example_audit.posterior
    example_forecast = risk.forecast(example_posterior, example_returns, seed=0, split=False)
    example_conditional_var = risk.conditional_var(example_posterior)
    return example_conditional_var, example_forecast


@app.cell(hide_code=True)
def _(example_conditional_var, example_forecast, mo, plots, plt):
    _pred, _plug = example_forecast["predictive"], example_forecast["plugin_at_posterior_median"]
    _rows = "\n".join(
        f"| {_h} day{'s' if _h > 1 else ''} | {_pred[_h]['var']:.2f}% | {_plug[_h]['var']:.2f}% | "
        f"{_pred[_h]['es']:.2f}% | {_plug[_h]['es']:.2f}% |"
        for _h in (1, 10)
    )
    _fig, _ax = plt.subplots(figsize=(9.6, 3.0), constrained_layout=True)
    _ax.hist(example_conditional_var, bins=40, color=plots.SERIES[0], alpha=0.85, edgecolor=plots.SURFACE, linewidth=0.5)
    _ax.axvline(_pred[1]["var"], color=plots.INK, linewidth=1.3, label=f"predictive VaR {_pred[1]['var']:.2f}%")
    _ax.axvline(_plug[1]["var"], color=plots.SERIES[1], linewidth=1.3, label=f"plug-in VaR {_plug[1]['var']:.2f}%")
    _ax.set_xlabel("1-day 99% VaR given the parameters and today's volatility (% log loss)")
    _ax.set_ylabel("posterior draws")
    _ax.set_title("The risk number is itself uncertain: one conditional VaR per posterior draw", loc="left")
    _ax.grid(axis="y", visible=False)
    _ax.legend(fontsize=8)
    mo.vstack([
        mo.md("\n".join([
            "| Horizon | 99% VaR, predictive | 99% VaR, plug-in | 97.5% ES, predictive | 97.5% ES, plug-in |",
            "|---|---|---|---|---|",
            _rows,
            "",
            "Losses are in percent log loss: minus the log return over the horizon.",
        ])),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(RESULTS, json, mo):
    _backtest = json.loads((RESULTS / "svj-backtest" / "backtest.json").read_text())
    _gap = {_h: _backtest["horizons"][_h]["plugin_minus_predictive"] for _h in ("1", "10")}
    mo.vstack([
        mo.md(f"""
        One series proves nothing about a forecasting method. On **{_backtest['n_series']} simulated
        series**, where the true parameters and today's true variance are known, each forecast is scored
        against 50,000 simulated futures: how often the loss breaches the 99% VaR, and the
        Acerbi–Szekely Z2 score of the 97.5% ES (0 when right, negative when the ES is too low).
        """),
        mo.image(RESULTS / "svj-backtest" / "backtest_paired.png", width=760),
        mo.md(f"""
        The predictive forecast is indistinguishable from a forecast made **with the true parameters**.
        The plug-in forecast breaches its VaR more often (+{_gap['1']['exceedance_pp']:.2f} and
        +{_gap['10']['exceedance_pp']:.2f} percentage points at 1 and 10 days) and understates the ES
        (Z2 {_gap['1']['z2']:+.2f} and {_gap['10']['z2']:+.2f}), more at 10 days than at 1 day. The gap is
        modest, and it is systematic, in the direction a regulator cares about.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(RESULTS, json, load_seconds, mo, sample_seconds):
    _costs = json.loads((RESULTS / "svj-costs" / "costs.json").read_text())
    mo.vstack([
        mo.md("## 9 · What it costs"),
        mo.image(RESULTS / "svj-costs" / "cost_per_name.png", width=760),
        mo.md(f"""
        On this machine the network answered in **{sample_seconds:.1f} s** (loading it took
        {load_seconds:.0f} s, once). At the medians, the amortized posterior is about
        **{_costs['ratios_at_the_median']['pmmh_over_network']:,.0f} times cheaper** than particle MCMC, and still
        **{_costs['ratios_at_the_median']['pmmh_over_network_plus_audit']:.0f} times cheaper** with an audit of
        every name. Training pays for itself within about ten names.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10 · Real data: VOO

    The same network, unchanged, on the last 1,000 trading days of VOO (an S&P 500 ETF), to
    3 September 2026. No truth is available now, so the audit and particle MCMC are the only checks.
    """)
    return


@app.cell
def _(ROOT, data, model, posterior):
    voo = data.window(data.load_returns(ROOT / "data" / "SPMO_VOO_2020-01-01_2026-09-04.csv", "VOO"))
    voo_draws = posterior.sample(voo.to_numpy()[None], num_samples=2_000, seed=7)[0]
    voo_quantities = model.quantities(voo_draws)
    return voo, voo_quantities


@app.cell(hide_code=True)
def _(RESULTS, json, mo, model, np, voo, voo_quantities):
    _audited = json.loads((RESULTS / "svj-daily" / "VOO_2026-09-03.json").read_text())
    _exact = json.loads((RESULTS / "svj-daily" / "VOO_2026-09-03_pmmh.json").read_text())
    _rows = []
    for _j, _label in ((4, "leverage rho"), (5, "jumps per year"), (8, "spot vol")):
        _live = np.median(voo_quantities[:, _j])
        _fmt = (lambda _x: f"{_x:.1%}") if _label == "spot vol" else (lambda _x: f"{_x:.2f}")
        _rows.append(f"| {_label} | {_fmt(_live)} | {_fmt(_audited['posterior'][_label]['median'])} | "
                     f"{_fmt(_exact['posterior'][_label]['median'])} |")
    for _h in ("1", "10"):
        _rows.append(
            f"| {_h}-day 99% VaR / 97.5% ES | {_audited['network_risk'][_h]['var']:.2f}% / {_audited['network_risk'][_h]['es']:.2f}% | "
            f"{_audited['risk']['predictive'][_h]['var']:.2f}% / {_audited['risk']['predictive'][_h]['es']:.2f}% | "
            f"{_exact['risk']['predictive'][_h]['var']:.2f}% / {_exact['risk']['predictive'][_h]['es']:.2f}% |"
        )
    mo.md("\n".join([
        f"| VOO, window to {voo.index[-1]:%d %b %Y} | Network | Network + audit | Particle MCMC (exact) |",
        "|---|---|---|---|",
        *_rows,
        "",
        "The network column's parameters are computed live; its risk numbers and the other two columns "
        "come from the recorded daily run. "
        f"The audit scored **{_audited['audit']['efficiency']:.1%}**, below the 2% threshold, so VOO's "
        f"window went to particle MCMC ({_exact['timings']['pmmh_cpu_s'] / 60:.0f} CPU-minutes). The exact "
        "posterior moves leverage and the jump frequency, and lowers the 1-day VaR from "
        f"{_audited['network_risk']['1']['var']:.2f}% to {_exact['risk']['predictive']['1']['var']:.2f}%, "
        "mostly through today's volatility. The misspecification test, which compares the window's learned "
        "summaries with simulated ones, finds nothing unusual "
        f"(p = {_audited['misspecification_pvalue']:.2f}).",
    ]))
    return


@app.cell(hide_code=True)
def _(RESULTS, mo):
    mo.vstack([
        mo.md(r"""
        Now run the network on every week of the last three years, 136 windows, one network pass each:
        """),
        mo.image(RESULTS / "svj-rolling" / "VOO_rolling.png", width=760),
        mo.md(r"""
        The posterior's spot volatility tracks realized volatility (correlation 0.73), and the jump
        frequency stays uncertain throughout, as the simulations predicted. Look at April 2025: realized
        volatility reaches 49% while the posterior peaks near 32%. The model attributes part of those days
        to jumps, because in this model jumps do not move the variance. Real markets also have jumps
        **in volatility**. The test does not flag these windows, which is exactly the question for the next
        slot: **how do we notice that the model, not the network, is wrong?**
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    1. **The model decides the difficulty.** One hidden volatility path makes the exact posterior cost
       about 35 CPU-minutes per name. The classical shortcut is fast but behaves as if the jump
       parameters were known.
    2. **Amortization makes the Bayesian answer affordable.** The slot-4 code with the SVJ simulator,
       trained once, returns the joint posterior of the parameters and today's volatility in seconds.
    3. **An audit makes it trustworthy, name by name.** The network is calibrated on average but can be
       off for a single series. Reweighting with the exact likelihood repairs it or flags it.
    4. **Parameter uncertainty belongs in the risk number.** The predictive forecast matches the truth;
       the plug-in understates the expected shortfall, most at the 10-day horizon.

    **Next, slot 6:** what if the prior is not the right one for your universe of names, and what if the
    model itself is wrong, as in April 2025?

    *Evidence and code:* the [design document](../docs/design/svj-design-document.md),
    the [results summary](../docs/svj-results-summary.md) and `src/sbi4finance/svj/`. All simulated
    results use the daily SVJ model and its prior; they do not certify the model on market data.
    """)
    return


if __name__ == "__main__":
    app.run()
