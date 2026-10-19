import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Slot 6 seed · When the prior does the work")


@app.cell(hide_code=True)
def _():
    import json

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    from sbi4finance import crash_risk as cr
    from sbi4finance.svj import data, plots

    ROOT = mo.notebook_dir().parent
    RESULTS = ROOT / "docs" / "results"
    plots.style()
    return RESULTS, ROOT, cr, data, json, mo, np, plots, plt, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Slot 6 seed · When the prior does the work

    **How likely is a crash day? Bayesian and classical answers, with no network and no sampler**

    > **Draft.** This notebook was first written as the slot-1 intro. The intro now uses betas
    > across many stocks, a simpler example, and this one is kept as the starting point for slot 6
    > (priors and misspecification). It is a case where the prior decides much of the answer: the
    > data cannot split the tail into jump frequency and jump size. It also measures what a flat
    > prior changes, where an informative prior hurts, and what happens when the tail model is the
    > wrong one. The sections below still read as an intro and will be reframed when slot 6 is
    > designed.

    Every morning a risk desk needs three things for each of thousands of names: a volatility estimate,
    how sure it is, and a risk number that accounts for both. This notebook asks four years of daily
    returns for one such number, where the second thing, how sure we are, decides the answer:

    > **How likely is a day with a loss of 7% or more in the next year?**

    We answer it with the tools a risk team would use, and with Bayes' rule. Everything is exact: the
    posterior is computed on a grid, with no neural network and no sampler. Slot 2 starts where that
    stops being possible.

    | Step | What it shows |
    |:---|:---|
    | 1 | The question, on four years of an S&P 500 ETF |
    | 2 | Bayes' rule on one page |
    | 3 | A model that allows crash days, with a prior a risk manager can read |
    | 4 | What four years of data can and cannot tell |
    | 5 | The classical answers, and every frequentist option we tried |
    | 6 | The Bayesian answer |
    | 7 | Hundreds of histories: which answers are calibrated? |
    | 8 | The catch: what the exact answer costs |
    """)
    return


@app.cell
def _(ROOT, data):
    voo_all = data.load_returns(ROOT / "data" / "SPMO_VOO_2020-01-01_2026-09-04.csv", "VOO")
    voo = data.window(voo_all)  # the last 1,000 trading days, daily log returns in %
    return voo, voo_all


@app.cell(hide_code=True)
def _(mo, np, plots, plt, stats, voo, voo_all):
    _crash_all = voo_all[voo_all <= -7]
    _normal_daily = stats.norm.cdf((-7 - voo.mean()) / voo.std())
    _years = 1 / (_normal_daily * 252)
    _fig, _ax = plt.subplots(figsize=(9.6, 2.8), constrained_layout=True)
    _ax.plot(voo_all.index, voo_all.to_numpy(), color=plots.SERIES[0], linewidth=0.6)
    _ax.axvspan(voo.index[0], voo.index[-1], color=plots.GRID, alpha=0.6, linewidth=0,
                label="the last 1,000 days")
    _ax.axhline(-7, color=plots.SERIES[1], linewidth=1.0, linestyle="--", label="a loss of 7%")
    _ax.scatter(_crash_all.index, _crash_all.to_numpy(), s=30, color=plots.SERIES[1], zorder=3,
                edgecolor=plots.SURFACE, linewidth=0.6)
    _ax.set_ylabel("daily log return (%)")
    _ax.set_title("VOO, an S&P 500 ETF, January 2020 to September 2026", loc="left")
    _ax.legend(loc="lower right", fontsize=8)
    _ax.grid(axis="x", visible=False)
    mo.vstack([
        mo.md(r"""
        ## 1 · The question

        **VOO** is Vanguard's S&P 500 ETF; its daily returns are, to a good approximation, those of the US
        stock market. The repository caches them in `data/`, and like slots 4 and 5 we use the last 1,000
        trading days, about four years, as daily log returns in percent.
        """),
        _fig,
        mo.md(f"""
        In those 1,000 days there is **no day with a loss of 7% or more**. So what is the probability of
        one in the next year?

        * **Count them:** zero days in four years, so the historical frequency says **0**.
        * **Fit a normal distribution:** with a daily volatility of {voo.std():.2f}%, a 7% loss is a
          {7 / voo.std():.1f}-sd day: about **once every {_years / 1e9:.0f} billion years**.
        * **Look further back:** March 2020 had **{len(_crash_all)} such days in six trading days**
          ({", ".join(f"{_v:.1f}%" for _v in _crash_all.to_numpy())}).

        Neither number belongs in a risk report. Four years of data hold the answer only weakly, and the
        honest answer has to say how weakly. That is what Bayes' rule is for.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Bayes' rule on one page

    A model with unknown parameters $\theta$ says how likely the data $y$ are, $p(y \mid \theta)$: the
    **likelihood**. A **prior** $p(\theta)$ says what is plausible before seeing the data. Bayes' rule
    combines them into the **posterior**, everything the data and the prior say about $\theta$:

    $$p(\theta \mid y) \;=\; \frac{p(y \mid \theta)\, p(\theta)}{p(y)} \;\propto\; p(y \mid \theta)\, p(\theta).$$

    A risk question is about the future, not about $\theta$. If $P(\theta)$ is the probability of a
    crash day next year when the parameters are $\theta$, the **predictive probability** averages it
    over everything the data allow:

    $$P(\text{crash day next year} \mid y) \;=\; \int P(\theta)\; p(\theta \mid y)\; d\theta .$$

    The usual shortcut, the **plug-in**, picks one value $\hat\theta$ and reports $P(\hat\theta)$, as if
    the parameters were known. The two agree when the data pin $\theta$ down. They disagree when they do
    not, which is exactly the case for tails.

    Two kinds of interval will appear. A 90% **confidence interval** comes from a rule that, applied to
    many data sets, covers the truth 90% of the time. A 90% **credible interval** holds 90% of the
    posterior probability, given these data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "**You already use Bayes** (click to open): where it hides in a quant's toolbox": mo.md(r"""
        * The **Kalman filter** is Bayes' rule for a hidden state, applied day after day. Slot 5's
          particle filter is its non-linear sibling.
        * **Black–Litterman** starts from a market-equilibrium prior and updates it with views.
        * **Shrinkage** estimators (James–Stein, Ledoit–Wolf) behave like a posterior mean under a prior
          centred on the shrinkage target.
        * **Bayesian VARs** with the Minnesota prior have been a forecasting staple since Litterman's
          work in the 1980s.
        * **Stochastic volatility** has been fitted by MCMC since Jacquier, Polson and Rossi (1994).
        """),
        "**The Bayesian toolbox**: how the posterior is computed, and what it costs": mo.md(r"""
        | Route | What it needs from the model | Cost per new data set | Examples |
        |:---|:---|:---|:---|
        | Closed form or grid | a likelihood; a handful of parameters | milliseconds to seconds | this notebook |
        | MCMC (HMC, NUTS) | a likelihood and its gradient | seconds to minutes | Stan, PyMC, NumPyro |
        | Particle MCMC | a simulator of the hidden state and an observation density | minutes to hours | slot 5's exact route |
        | ABC, synthetic likelihood | only a simulator | many simulations per data set | |
        | Amortized neural inference | only a simulator | seconds, after one training run | BayesFlow (slots 2 to 6) |
        """),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · A model that allows crash days

    A normal model cannot produce a 7% loss in a lifetime, so the model must allow jumps. The simplest
    one adds, on a few days a year, a jump to an otherwise normal day. With $y_t$ the daily log return in
    percent,

    $$y_t = \mu + \sigma\,\varepsilon_t + B_t\,\xi_t, \qquad \varepsilon_t \sim \mathcal N(0,1),\quad
    B_t \sim \text{Bernoulli}(\lambda / 252),\quad \xi_t \sim \mathcal N(\mu_J, \sigma_J^2).$$

    This is Merton's (1976) jump model in daily form: slot 5's model without stochastic volatility. A
    risk manager reads it as a volatility $\sigma$ on ordinary days and, about $\lambda$ times a year, a
    jump of typical size $\mu_J \pm \sigma_J$. Nothing marks which days were jumps. Given the parameters,
    the probability of a loss of $x\%$ or more on one day is a two-term mixture,

    $$q(\theta) = \Big(1 - \tfrac{\lambda}{252}\Big)\,\Phi\Big(\tfrac{-x-\mu}{\sigma}\Big)
    + \tfrac{\lambda}{252}\,\Phi\Big(\tfrac{-x-\mu-\mu_J}{\sqrt{\sigma^2+\sigma_J^2}}\Big),
    \qquad P(\theta) = 1 - \big(1 - q(\theta)\big)^{252},$$

    and the likelihood of 1,000 days is the product of the daily mixtures: exact, cheap, no integral.
    The prior is slot 5's, stated in units a risk manager can judge:

    | Quantity | Prior | 95% range |
    |:---|:---|:---|
    | Jumps per year $\lambda$ | log-normal, median 2.5, log-sd 0.75 | 0.6 to 11 |
    | Jump size sd $\sigma_J$ | log-normal, median 3%, log-sd 0.45 | 1.2% to 7.3% |
    | Mean jump $\mu_J$ | $\mathcal N(-1\%,\ 2.5\%^2)$ | −5.9% to 3.9% |
    | Volatility on ordinary days | log-normal, median 25% a year, log-sd 0.40 | 11% to 55% |

    The drift $\mu$ is set to the sample mean for every method below: a tail probability barely moves
    with it.
    """)
    return


@app.cell
def _(cr, np):
    model = cr.JumpModel()
    _rng = np.random.default_rng(11)
    _draws = [model.draw_truth(_rng) for _ in range(20_000)]  # (parameters, drift) pairs from the prior
    _params, _drift = np.array([_d[0] for _d in _draws]), np.array([_d[1] for _d in _draws])
    prior_crash = model.crash_probability(_params, 7.0, _drift)
    _params[:, 3] = np.log(100 * 0.16 / np.sqrt(252))  # the same draws at VOO's volatility, 16% a year
    prior_crash_16 = model.crash_probability(_params, 7.0, _drift)
    return model, prior_crash, prior_crash_16


@app.cell(hide_code=True)
def _(mo, np, plots, plt, prior_crash, prior_crash_16):
    _lo, _mid, _hi = np.quantile(prior_crash, [0.05, 0.5, 0.95])
    _fig, _ax = plt.subplots(figsize=(9.6, 2.4), constrained_layout=True)
    _ax.hist(100 * np.clip(prior_crash, 1e-6, 1), bins=np.logspace(-4, 2, 61), color=plots.SERIES[0],
             alpha=0.85, edgecolor=plots.SURFACE, linewidth=0.4)
    _ax.set_xscale("log")
    _ax.set_xlabel("probability of a day with a loss of 7% or more in the next year (%)")
    _ax.set_yticks([])
    _ax.grid(visible=False)
    _ax.set_title("Before any data: what the model and the prior say, over 20,000 draws", loc="left")
    mo.vstack([
        mo.md(
            "**Check the prior before using it.** Simulate parameters from the prior and compute, for each "
            "draw, the probability of a crash day next year: this is what the model believes before seeing "
            "any data."
        ),
        _fig,
        mo.md(
            f"Median **{100 * _mid:.0f}%**, 90% between {100 * _lo:.1f}% and {100 * _hi:.0f}%. The prior is wide on "
            "purpose, from calm names to volatile ones, and most of its spread is the volatility, which four "
            "years of data pin down well. At VOO's volatility of 16% a year, the same draws give a median of "
            f"**{100 * np.median(prior_crash_16):.0f}%**: crash days are possible, rare and uncertain, which is what "
            "a risk manager would sign off. A prior that put that median at 1-in-1,000 or at 90% would need "
            "defending."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · What four years of data can and cannot tell

    To judge the methods we need to know the right answer, so we simulate a stock: 1,000 days with a
    volatility of 16% a year and 2.5 jumps a year of −1% ± 3%. We chose this history because it shows
    the problem clearly. Section 7 asks how often that happens, over hundreds of histories.
    """)
    return


@app.cell
def _(model, np):
    truth = np.array([np.log(2.5), np.log(3.0), -1.0, np.log(1.0)])  # log jumps/yr, log jump sd, mean jump, log sd
    returns, jump_days = model.simulate(truth, 0.03, 1000, np.random.default_rng(1), with_jumps=True)
    true_crash = float(model.crash_probability(truth, 7.0, 0.03))
    return jump_days, returns, true_crash, truth


@app.cell(hide_code=True)
def _(jump_days, mo, np, plots, plt, returns, true_crash):
    _days = np.arange(1, len(returns) + 1)
    _fig, _ax = plt.subplots(figsize=(9.6, 2.6), constrained_layout=True)
    _ax.plot(_days, returns, color=plots.SERIES[0], linewidth=0.6)
    _ax.scatter(_days[jump_days], returns[jump_days], s=26, color=plots.SERIES[1], zorder=3,
                edgecolor=plots.SURFACE, linewidth=0.6, label="jump days (known only because we simulated)")
    _ax.set_xlabel("trading day")
    _ax.set_ylabel("daily log return (%)")
    _ax.legend(loc="lower left", fontsize=8)
    _ax.set_title("What a risk manager sees: the returns", loc="left")
    mo.vstack([_fig, mo.md(
        f"The history holds **{int(jump_days.sum())} jumps**, its worst day is {returns.min():.1f}%, and no day "
        f"reaches −7%. Because we know the parameters, we know the answer: the probability of a crash day "
        f"next year is **{100 * true_crash:.1f}%**. Each method below sees only the returns."
    )])
    return


@app.cell(hide_code=True)
def _(mo, model, np, returns):
    mu = float(np.mean(returns))  # the drift: the sample mean, for every method
    grid = model.grid(returns, mu)  # the exact log-likelihood and log-prior on a 45 x 33 x 43 x 11 grid
    mo.show_code()
    return grid, mu


@app.cell
def _(cr, grid, model, mu, np, returns):
    analysis = cr.analyse(model, returns, (5.0, 7.0), n_boot=100, rng=np.random.default_rng(2), grid=grid)
    return (analysis,)


@app.cell(hide_code=True)
def _(cr, grid, model, mu, returns, stats):
    plane = cr.jump_plane(model, returns, mu, grid, 7.0)  # profile likelihood, crash probability and posterior, per point
    ridge = cr.ridge_ends(plane)  # the lowest and highest crash probability inside the 95% likelihood region
    region_levels = {0.683: stats.chi2.ppf(0.683, 2), 0.95: stats.chi2.ppf(0.95, 2)}
    return plane, region_levels, ridge


@app.cell(hide_code=True)
def _(analysis, np, plane, plots, plt, region_levels, ridge, truth):
    def _hpd_levels(w, masses):
        _order = np.sort(w.ravel())[::-1]
        return [_order[np.searchsorted(np.cumsum(_order), _m)] for _m in masses]

    _fig, _axes = plt.subplots(1, 2, figsize=(9.6, 4.0), constrained_layout=True, sharey=True)
    _lr = 2 * (np.nanmax(plane["profile"]) - plane["profile"])
    _lr = np.where(np.isfinite(_lr), _lr, 1e6)
    _left, _right = _axes
    _left.contourf(plane["jumps_per_year"], plane["jump_sd"], _lr.T, levels=[0, region_levels[0.683], region_levels[0.95]],
                   colors=[plots.SERIES[0], "#9cc2ee"])
    _left.set_title("The likelihood: what the data alone allow", loc="left")
    _w = plane["posterior"].T
    _levels = _hpd_levels(plane["posterior"], [0.9, 0.5])
    _right.contourf(plane["jumps_per_year"], plane["jump_sd"], _w, levels=[_levels[0], _levels[1], _w.max() * 1.01],
                    colors=["#9cc2ee", plots.SERIES[0]])
    _right.set_title("The posterior: data and prior together", loc="left")
    for _ax in _axes:
        _cs = _ax.contour(plane["jumps_per_year"], plane["jump_sd"], plane["crash"].T, levels=[0.001, 0.01, 0.05, 0.2, 0.5],
                          colors=plots.INK_2, linewidths=0.8)
        _ax.clabel(_cs, fmt=lambda _v: f"{100 * _v:g}%", fontsize=7)
        _ax.scatter([np.exp(truth[0])], [np.exp(truth[1])], marker="*", s=140, color=plots.SERIES[1], zorder=4,
                    edgecolor=plots.INK, linewidth=0.5, label="truth")
        _ax.set_xscale("log")
        _ax.set_yscale("log")
        _ax.set_xlim(plane["jumps_per_year"][0], plane["jumps_per_year"][-1])
        _ax.set_ylim(plane["jump_sd"][0], plane["jump_sd"][-1])
        _ax.set_xlabel("jumps per year")
    _mle = np.exp(analysis["mle_x"][:2])
    _left.scatter([_mle[0]], [max(_mle[1], 1.05 * plane["jump_sd"][0])], s=40, color=plots.INK, zorder=4, label="maximum likelihood")
    for _end in ridge.values():  # the lowest and highest crash probability inside the 95% region
        _left.scatter([_end["jumps_per_year"]], [_end["jump_sd"]], s=46, facecolor="none", edgecolor=plots.INK, linewidth=1.2, zorder=4)
        _left.annotate(f"{100 * _end['crash']:.2g}%", (_end["jumps_per_year"], _end["jump_sd"]), xytext=(6, 4),
                       textcoords="offset points", fontsize=8, color=plots.INK)
    _left.set_ylabel("jump size sd (%)")
    _left.legend(loc="upper right", fontsize=7)
    _right.legend(loc="upper right", fontsize=7)
    _fig.text(0.01, -0.02, "Shaded: 68% and 95% likelihood regions (left), 50% and 90% posterior regions (right). "
              "Lines: the probability of a crash day next year at each point. Circles: its lowest and highest values "
              "inside the 95% likelihood region.", fontsize=7.5, color=plots.INK_2)
    crash_plane_figure = _fig
    return (crash_plane_figure,)


@app.cell(hide_code=True)
def _(analysis, crash_plane_figure, mo, np, ridge):
    _low, _high = ridge["low"], ridge["high"]
    _mle = analysis["mle_x"]
    mo.vstack([
        mo.md(r"""
        ### The likelihood is a ridge

        The left panel shows which pairs of jump frequency and jump size fit the history: the likelihood,
        with the mean jump and the volatility set to their best values at each point. The fitting region
        is a long **ridge**, running from *a few large jumps* to *many small ones*. Every point in the
        shaded region passes a likelihood-ratio test against the best fit at the 5% level, and the dark
        part fits almost as well as the best. The thin lines show what each point says about the future.
        """),
        crash_plane_figure,
        mo.md(f"""
        The ridge cuts across the lines. Inside the 95% region, the probability of a crash day next year runs
        from **{100 * _low["crash"]:.2g}%**, with {_low["jumps_per_year"]:.0f} small jumps a year of
        ±{_low["jump_sd"]:.1f}%, to **{100 * _high["crash"]:.2g}%**, with {_high["jumps_per_year"]:.1f} larger
        ones of {_high["mean_jump"]:.1f}% ± {_high["jump_sd"]:.1f}%: a factor of
        {_high["crash"] / _low["crash"]:,.0f}. The data cannot choose between these futures, because a 7% day
        is exactly the event four years of data rarely contain.

        Maximum likelihood picks a single point at the edge of the region: **{np.exp(_mle[0]):.2f} jumps a year of
        {_mle[2]:.1f}% ± {np.exp(_mle[1]):.1f}%**{", on the boundary of the parameter space" if analysis["mle_on_bound"] else ""}.
        That is a few jumps of almost fixed size, a story that fits the past well and puts the probability of
        a crash day next year at only {100 * analysis["by_threshold"]["7"]["point"]["mle"]:.2g}%. The posterior on
        the right keeps the whole region, weighted by the prior: jumps come a few times a year, with sizes of
        a few percent.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(analysis, mo, np, true_crash, truth):
    _b = analysis["by_threshold"]["7"]
    _pct = lambda _v: "0%" if _v == 0 else "below 0.01%" if _v < 1e-4 else f"{100 * _v:.2g}%"
    _iv = lambda _v: "none (Hessian not positive definite)" if not np.isfinite(_v[0]) else f"{_pct(_v[0])} to {_pct(_v[1])}"
    _f = analysis["jump_frequency_interval"]
    _fiv = lambda _v: "none" if not np.isfinite(_v[1]) else f"{np.exp(_v[0]):.2g} to {np.exp(_v[1]):.3g}"
    _covers = lambda _v: "yes" if (np.isfinite(_v[1]) and _v[0] <= truth[0] <= _v[1]) else "**no**"
    mo.md(f"""
    ## 5 · The classical answers

    Every method below sees the same 1,000 returns. The true probability is **{_pct(true_crash)}**.

    | Method | Probability of a crash day next year | 90% interval |
    |:---|:---|:---|
    | Historical frequency | {_pct(_b["point"]["empirical"])} | |
    | Peaks over threshold (a Pareto tail, model-free) | {_pct(_b["point"]["evt"])} | |
    | A Student-t fitted instead, plug-in | {_pct(_b["point"]["other_family"])} | |
    | Maximum likelihood, plug-in | {_pct(_b["point"]["mle"])} | Wald, delta method: {_iv(_b["interval"]["wald"])} |
    | Profile likelihood | | {_iv(_b["interval"]["profile"])} |
    | Bootstrap: refit on 100 resampled histories | {_pct(_b["point"]["bootstrap"])} (average) | {_iv(_b["interval"]["bootstrap"])} |
    | Parametric bootstrap: refit on 100 histories simulated from the fit | {_pct(_b["point"]["param_bootstrap"])} (average) | {_iv(_b["interval"]["param_bootstrap"])} |
    | Penalized likelihood: the posterior mode | {_pct(_b["point"]["map"])} | Laplace: {_iv(_b["interval"]["laplace"])} |

    And for the jump frequency itself (truth {np.exp(truth[0]):.1f} a year), 95% intervals:

    | Method | Jumps per year | Covers the truth? |
    |:---|:---|:---|
    | Wald, from the Hessian | {_fiv(_f["wald"])} | {_covers(_f["wald"])} |
    | Profile likelihood | {_fiv(_f["profile"])} | {_covers(_f["profile"])} |
    | Bootstrap percentile | {_fiv(_f["bootstrap"])} | {_covers(_f["bootstrap"])} |
    | Parametric bootstrap percentile | {_fiv(_f["param_bootstrap"])} | {_covers(_f["param_bootstrap"])} |
    | Flag days beyond 3 sd, then count | {_fiv(_f["threshold_count"])} | {_covers(_f["threshold_count"])} |

    The point forecasts disagree by orders of magnitude, and the ones a desk would use first (the count,
    the plug-in, the bootstrap) sit far below the truth. A Student-t, the other usual fat-tailed model,
    does worse still: rare jumps barely move its fit to the ordinary days. Most intervals cover the
    truth because they are wide: they span the ridge, from almost no crash risk to one chance in four.
    The bootstrap's does not: when the fit sits on a boundary, resampling the days rarely reaches the
    other end of the ridge. A frequentist who uses the profile likelihood or the penalized fit ends up
    close to the Bayesian answer; section 7 shows how close.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "**Every frequentist option we considered, and why it does or does not help** (click to open)": mo.md(r"""
        | Method | Used here | What happens |
        |:---|:---|:---|
        | Wald interval from the Hessian | yes | Assumes a parabola; the likelihood is a ridge with its maximum on a bound. |
        | Profile likelihood ratio | yes | Honest, and therefore wide; it gives an interval, not a probability to report. |
        | Boundary-corrected likelihood ratio (Chernoff 1954; Self and Liang 1987) | no | Built for testing at a known boundary; the χ² interval above is already the conservative choice. |
        | Nonparametric bootstrap | yes | Inconsistent when the estimate sits on a boundary (Andrews 2000); its intervals can exclude the truth. |
        | Parametric bootstrap | yes | Its average is the frequentist predictive distribution of Harris (1989); it shares the boundary problem. |
        | m-out-of-n bootstrap, subsampling | no | Valid at a boundary only with many effective observations; ten jump days are too few. |
        | Weak-identification-robust sets (Andrews and Cheng 2012) | no | Valid, hence wide or unbounded here (Dufour 1997, below). |
        | Flag jump days, then count | yes | With daily data, small jumps hide among ordinary days and large ordinary days look like jumps. |
        | Method of moments, GMM, EMM | no | The moments that identify jumps are dominated by the few jump days themselves. |
        | Penalized likelihood with a Laplace interval | yes | Adds the same prior, but keeps a single point and a Gaussian around it. |
        | Peaks over threshold (EVT) | yes | Model-free, but extrapolates from the largest 50 losses to a level none of them reached. The study also tries the largest 25 and 100. |
        | A different tail model (Student-t for jump data, jumps for Student-t data) | yes | What a practitioner who picked the other family would report. |
        | Historical frequency | yes | Zero, whenever four years contain no crash day. |

        **Why no method can do better without extra information.** When a parameter is unidentified on part
        of the parameter space, any confidence set valid at every truth must be unbounded with probability
        close to its level there; a method that always returns a bounded interval, such as Wald, then has
        confidence level zero (Dufour 1997). The jump model has this structure: as the jump frequency goes to
        zero the jump size drops out of the likelihood, and as the jump size goes to zero the frequency does.
        This is a property of the question and the data, not of a choice of method. The Bayesian answer does
        not escape it either: it adds information, the prior, and asks for calibration on average over that
        prior instead of coverage at every truth.
        """),
    })
    return


@app.cell(hide_code=True)
def _(grid, mo, model, mu, np):
    _ok = np.isfinite(grid["ll"]) & np.isfinite(grid["lp"])         # grid points the model allows
    _log_post = np.where(_ok, grid["ll"] + grid["lp"], -np.inf)     # log likelihood + log prior
    posterior_weights = np.exp(_log_post - _log_post[_ok].max())
    posterior_weights /= posterior_weights.sum()                    # Bayes' rule, normalized
    crash_on_grid = np.where(_ok, model.crash_probability(np.where(_ok[..., None], grid["x"], 0.0), 7.0, mu), 0.0)
    predictive = float(np.sum(posterior_weights * crash_on_grid))   # the posterior average of P(theta)
    mo.show_code()
    return crash_on_grid, posterior_weights, predictive


@app.cell(hide_code=True)
def _(analysis, crash_on_grid, mo, np, plots, plt, posterior_weights, predictive, true_crash):
    _b = analysis["by_threshold"]["7"]
    _fig, _ax = plt.subplots(figsize=(9.6, 3.0), constrained_layout=True)
    _bins = np.logspace(-4, 0, 61)
    _ax.hist(np.clip(crash_on_grid.ravel(), 1e-4, 1), bins=_bins, weights=posterior_weights.ravel(),
             color=plots.SERIES[0], alpha=0.85, edgecolor=plots.SURFACE, linewidth=0.4)
    _marks = [("truth", true_crash, plots.SERIES[1], "-"), ("Bayes, predictive", predictive, plots.INK, "-"),
              ("maximum likelihood", _b["point"]["mle"], plots.INK_2, "--"),
              ("bootstrap", _b["point"]["bootstrap"], plots.INK_2, ":"), ("peaks over threshold", _b["point"]["evt"],
              plots.MUTED, "-.")]
    for _label, _value, _color, _style in _marks:
        _ax.axvline(max(_value, 1e-4), color=_color, linestyle=_style, linewidth=1.4,
                    label=f"{_label}: {100 * _value:.2g}%")
    _ax.set_xscale("log")
    _ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1], ["0.01%", "0.1%", "1%", "10%", "100%"])
    _ax.set_yticks([])
    _ax.grid(visible=False)
    _ax.set_xlabel("probability of a crash day next year")
    _ax.set_title("The posterior distribution of the crash probability, and the single numbers", loc="left")
    _ax.legend(loc="upper left", fontsize=7.5)
    _lo, _hi = _b["interval"]["bayes"]
    mo.vstack([
        mo.md(r"""
        ## 6 · The Bayesian answer

        The posterior weighs every point of the grid by how well it fits the history *and* how plausible it
        was beforehand. Each point carries its own crash probability, so the posterior of the crash
        probability comes for free, and its average is the predictive probability:
        """),
        _fig,
        mo.md(f"""
        The predictive probability is **{100 * predictive:.1f}%** (90% credible interval {100 * _lo:.2g}% to
        {100 * _hi:.2g}%), against a truth of {100 * true_crash:.1f}%. The width is not a weakness: it is
        what four years of data can honestly say about a 7% day. The single numbers each come from one point
        of the ridge, and say nothing about the rest.

        One history proves nothing. Was this luck?
        """),
    ])
    return


@app.cell
def _(RESULTS, cr, json):
    _lines = (RESULTS / "crash-risk" / "records.jsonl").read_text().splitlines()
    _groups = {}  # "jump/prior", "jump/fixed/typical", ..., "student/prior", ...
    for _r in map(json.loads, _lines):
        if "error" not in _r:
            _groups.setdefault("/".join(filter(None, (_r["model"], _r["design"], _r["truth"]))), []).append(_r)
    study_scores = {_name: cr.score(_records, "7") for _name, _records in _groups.items()}
    jump_prior, jump_scores = _groups["jump/prior"], study_scores["jump/prior"]
    return jump_prior, jump_scores, study_scores


@app.cell(hide_code=True)
def _(jump_prior, jump_scores, mo, np, plots, plt, study_scores):
    _p = np.array([_r["p_true"]["7"] for _r in jump_prior])
    _methods = [("bayes", "Bayes, predictive", plots.SERIES[0], "o"), ("mle", "maximum likelihood", plots.SERIES[1], "s"),
                ("bootstrap", "bootstrap average", plots.SERIES[2], "^"), ("evt", "peaks over threshold", plots.MUTED, "D")]
    _edges = np.array([1e-4, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0])
    _fig, _ax = plt.subplots(figsize=(5.6, 4.2), constrained_layout=True)
    _ax.plot([1e-3, 1], [1e-3, 1], color=plots.BASELINE, linewidth=1.0)
    for _key, _label, _color, _marker in _methods:
        _q = np.clip([_r["result"]["by_threshold"]["7"]["point"][_key] for _r in jump_prior], 1e-4, 1)
        _bins = np.digitize(_q, _edges[1:-1])
        _xs = [np.mean(_q[_bins == _k]) for _k in range(len(_edges) - 1) if np.sum(_bins == _k) >= 10]
        _ys = [np.mean(_p[_bins == _k]) for _k in range(len(_edges) - 1) if np.sum(_bins == _k) >= 10]
        _ax.plot(_xs, _ys, marker=_marker, color=_color, linewidth=1.2, markersize=5, label=_label)
    _ax.set_xscale("log")
    _ax.set_yscale("log")
    _ax.set_xlabel("forecast probability (average within a bin)")
    _ax.set_ylabel("true probability (average within the bin)")
    _ax.set_title("Calibration: on the diagonal means right on average", loc="left")
    _ax.legend(fontsize=7.5, loc="upper left")
    _share = lambda _v: f"{100 * _v:.1f}%" if 0 < _v < 0.01 else f"{100 * _v:.0f}%"
    _rows = "\n".join(
        f"| {_label} | {jump_scores['point'][_key]['excess_log_loss']:.3f} | "
        f"{_share(jump_scores['point'][_key]['understated_10x'])} | {_share(jump_scores['point'][_key]['overstated_10x'])} |"
        for _key, _label in [("bayes", "Bayes, predictive"), ("map", "posterior mode, plug-in"),
                             ("param_bootstrap", "parametric bootstrap average"), ("bootstrap", "bootstrap average"),
                             ("mle", "maximum likelihood, plug-in"), ("other_family", "a Student-t, plug-in"),
                             ("evt", "peaks over threshold"), ("empirical", "historical frequency")])
    _irows = "\n".join(
        f"| {_label} | {100 * jump_scores['interval'][_key]['coverage']:.0f}% | {np.exp(jump_scores['interval'][_key]['median_log_width']):.0f}× |"
        for _key, _label in [("bayes", "Bayes, credible"), ("profile", "profile likelihood"), ("laplace", "Laplace at the mode"),
                             ("param_bootstrap", "parametric bootstrap percentile"), ("bootstrap", "bootstrap percentile"),
                             ("wald", "Wald, delta method")])
    mo.vstack([
        mo.md(f"""
        ## 7 · Hundreds of histories: which answers are calibrated?

        The study behind this section (`scripts/crash_risk_study.py`) simulates
        {sum(_s['n_series'] for _s in study_scores.values()):,} histories of 1,000 days and runs every method
        on each. In the main design, **{jump_scores['n_series']} histories** each come from their own truth,
        drawn from the prior. Because every truth is known, every forecast can be scored. The plot bins the
        forecasts and compares each bin's average forecast with its average true probability.
        """),
        mo.hstack([_fig, mo.md("\n".join([
            "| Point forecast | Excess log loss | Understated 10× or more | Overstated 10× or more |",
            "|:---|:---|:---|:---|",
            _rows,
            "",
            "**Excess log loss**: the average extra loss, in nats, of a forecaster quoting this probability "
            "instead of the true one. It is a proper score: lower is better, whatever one's philosophy. "
            "**Understated 10×**: the share of histories where the forecast was below a tenth of the true "
            "probability, the misses that hurt a risk manager; **overstated 10×**, the opposite.",
        ]))], widths=[1, 1.1]),
        mo.md("\n".join([
            "| 90% interval | Covers the truth | Median width (high / low) |",
            "|:---|:---|:---|",
            _irows,
        ])),
    ])
    return


@app.cell(hide_code=True)
def _(jump_scores, mo, study_scores):
    _p, _i, _f = jump_scores["point"], jump_scores["interval"], jump_scores["jump_frequency"]
    _pc = lambda _v: f"{100 * _v:.1f}%" if 0 < _v < 0.01 else f"{100 * _v:.0f}%"
    _ell = lambda _k: f"{_p[_k]['excess_log_loss']:.3f}"
    _under = lambda _k: _pc(_p[_k]["understated_10x"])
    _cell = lambda _k: f"{_ell(_k)}, {_under(_k)}"
    _freq = lambda _k: f"{_pc(_f[_k]['coverage'])} at a factor {_f[_k]['median_width_factor']:.0f}"
    _fixed = [(_name, study_scores.get(f"jump/fixed/{_name}")) for _name in
              ("typical", "rare_and_large", "frequent_and_small", "almost_none")]
    _label = {"typical": "2.5 a year of −1% ± 3%", "rare_and_large": "0.5 a year of −3% ± 6%",
              "frequent_and_small": "12 a year of −0.5% ± 1.5%", "almost_none": "0.1 a year of −1% ± 3%"}
    _rows = "\n".join(
        f"| {_label[_name]} | {100 * _s['mean_true_probability']:.2g}% | "
        + " | ".join(f"{_s['point'][_k]['excess_log_loss']:.3f}" for _k in ("bayes", "map", "mle", "bootstrap", "evt"))
        + f" | {_pc(_s['interval']['bayes']['coverage'])} / {_pc(_s['interval']['profile']['coverage'])} |"
        for _name, _s in _fixed if _s)
    _st = study_scores.get("student/prior")
    _st_rows = "" if not _st else "\n".join(
        f"| {_lab} | {_st['point'][_k]['excess_log_loss']:.3f} | {_pc(_st['point'][_k]['understated_10x'])} |"
        for _k, _lab in [("bayes", "Bayes, predictive"), ("map", "posterior mode, plug-in"),
                         ("param_bootstrap", "parametric bootstrap average"), ("bootstrap", "bootstrap average"),
                         ("mle", "maximum likelihood, plug-in"), ("other_family", "the jump model, plug-in"),
                         ("evt", "peaks over threshold"), ("empirical", "historical frequency")])
    mo.md(f"""
    **Reading the results.** On a typical history, the answers built on the likelihood do about equally
    well; the differences are in how badly they fail. Maximum likelihood understates the crash
    probability by a factor of 10 or more in **{_under('mle')}** of histories. The bootstrap, which averages
    refits around that fit, does so in {_under('bootstrap')}, and peaks over threshold in {_under('evt')}. The
    Bayesian predictive probability does so in **{_under('bayes')}**, and it is calibrated. That is the promise
    of Bayes' rule, and it holds exactly when the prior describes the world the histories come from. It
    pays by overstating more often ({_pc(_p['bayes']['overstated_10x'])} of histories, by 10× or more), on
    quiet stocks that the prior finds unusual.

    **Where the gain comes from.** The Bayesian answer differs from the plug-in in two ways: it averages
    over every parameter value the data allow, weighted by the likelihood, and it adds the prior. Switching
    them on one at a time (excess log loss, then the share understated 10× or more):

    | | One point | Averaged over what this history allows |
    |:---|:---|:---|
    | No prior information | maximum likelihood: {_cell('mle')} | flat prior on the grid's box: {_cell('bayes_flat')} |
    | Slot 5's prior | posterior mode: {_cell('map')} | **Bayes: {_cell('bayes')}** |

    The flat prior gives every point of the grid the same weight: jump frequencies from 0.03 to 126 a year
    and jump sizes from 0.2% to 45%, both on a log scale, and mean jumps from −12% to 9%. Either step alone
    recovers most of the gap, and together they give the calibrated answer. The two
    bootstraps average too ({_cell('bootstrap')} and, parametric, {_cell('param_bootstrap')}), but over
    histories resampled or simulated around one fit, not over what this history allows: when the fit sits
    at one end of the ridge, most refits stay there.

    **Intervals.** The profile likelihood covers the truth about as often as the Bayesian interval
    ({_pc(_i['profile']['coverage'])} against {_pc(_i['bayes']['coverage'])} for a 90% interval), at a similar
    width. For a frequentist it is the honest tool here, and it says what the posterior says: the answer is
    uncertain by an order of magnitude. What it does not give is one probability for the report. The
    bootstrap ({_pc(_i['bootstrap']['coverage'])}) and Wald ({_pc(_i['wald']['coverage'])}, with the same convention
    for a missing Hessian) intervals are too narrow.

    **The jump frequency itself** (95% intervals). The profile likelihood covers {_freq('profile')} at the
    median: valid and nearly useless, as Dufour's theorem predicts. A flat prior does no better
    ({_freq('bayes_flat')}), and neither do the bootstraps ({_freq('bootstrap')}; parametric
    {_freq('param_bootstrap')}). Slot 5's prior covers {_freq('bayes')}: the information that narrows the
    interval is the prior's. Wald covers {_pc(_f['wald']['coverage'])}, counting as misses the
    {_pc(_f['wald']['missing'])} of histories where the Hessian is not positive definite and there is no
    interval; flagging and counting jump days covers {_pc(_f['threshold_count']['coverage'])}. Maximum
    likelihood ends on a boundary in {_pc(jump_scores['mle_on_bound'])} of histories.

    **At a fixed truth**, the frequentist's criterion: four named truths, at the middle and the ends of the
    ridge, 100 histories each. Excess log loss, and the coverage of the 90% intervals:

    | Truth: jumps | True probability | Bayes | Posterior mode | Maximum likelihood | Bootstrap | Peaks over threshold | Coverage, Bayes / profile |
    |:---|:---|:---|:---|:---|:---|:---|:---|
    {_rows}

    Where the prior is roughly right, at the typical truth and the rare-and-large end, Bayes wins by far.
    At the two quiet truths, with a true probability of about 0.2%, it loses: the prior expects a few
    jumps a year of a few percent, and pulls the forecast up by a factor of 10 or more. The plug-in wins
    there on excess log loss even though it understates a tiny probability most of the time, because
    missing a tiny probability costs little. Calibration is a promise on average over the prior, not at
    every truth: that is the price of adding information, and slot 6 is about choosing and checking it.

    **Did we pick the wrong model?** No: every history here comes from the model being fitted, so the model
    is right by construction. Would a simpler fat-tailed model have saved the plug-in? Only if it were the
    right one. With Student-t returns and the Student-t model, whose two parameters four years pin down
    well, every answer built on the likelihood is about as good as Bayes, the plug-in included (truths
    drawn from that family's own prior; excess log loss, then the share understated 10× or more):

    | Method | Excess log loss | Understated 10× or more |
    |:---|:---|:---|
    {_st_rows}

    Fitted to the jump-model histories, the same Student-t is worse than every jump-model method
    ({_cell('other_family')}), and the jump model fitted to Student-t histories fails in the same way. So the
    single fit does not fail because the tail is fat. It fails when the parameters that decide the answer
    are weakly identified, as the split between jump frequency and jump size is here, and as many of slot
    5's parameters are.
    """)
    return


@app.cell
def _(cr, model, np, voo):
    _y = voo.to_numpy()
    voo_grid = model.grid(_y, float(_y.mean()))
    voo_analysis = cr.analyse(model, _y, (5.0, 7.0), n_boot=100, rng=np.random.default_rng(3), grid=voo_grid)
    voo_ridge = cr.ridge_ends(cr.jump_plane(model, _y, float(_y.mean()), voo_grid, 7.0))
    return voo_analysis, voo_ridge


@app.cell(hide_code=True)
def _(mo, np, voo_analysis, voo_ridge):
    _b = voo_analysis["by_threshold"]["7"]
    _x = voo_analysis["mle_x"]
    _pct = lambda _v: "0%" if _v == 0 else "below 0.01%" if _v < 1e-4 else f"{100 * _v:.2g}%"
    mo.md(f"""
    ## Back to VOO

    The same methods, on VOO's last 1,000 days:

    | Method | Probability of a crash day next year | 90% interval |
    |:---|:---|:---|
    | Historical frequency | {_pct(_b["point"]["empirical"])} | |
    | Peaks over threshold | {_pct(_b["point"]["evt"])} | |
    | Maximum likelihood, plug-in | {_pct(_b["point"]["mle"])} | profile: {_pct(_b["interval"]["profile"][0])} to {_pct(_b["interval"]["profile"][1])} |
    | Bootstrap average | {_pct(_b["point"]["bootstrap"])} | {_pct(_b["interval"]["bootstrap"][0])} to {_pct(_b["interval"]["bootstrap"][1])} |
    | Parametric bootstrap average | {_pct(_b["point"]["param_bootstrap"])} | {_pct(_b["interval"]["param_bootstrap"][0])} to {_pct(_b["interval"]["param_bootstrap"][1])} |
    | Posterior mode, plug-in | {_pct(_b["point"]["map"])} | Laplace: {_pct(_b["interval"]["laplace"][0])} to {_pct(_b["interval"]["laplace"][1])} |
    | Bayes, flat prior | {_pct(_b["point"]["bayes_flat"])} | {_pct(_b["interval"]["bayes_flat"][0])} to {_pct(_b["interval"]["bayes_flat"][1])} |
    | **Bayes, predictive** | **{_pct(_b["point"]["bayes"])}** | {_pct(_b["interval"]["bayes"][0])} to {_pct(_b["interval"]["bayes"][1])} |

    VOO has a ridge too. Inside its 95% likelihood region, the probability of a crash day next year runs
    from **{_pct(voo_ridge["low"]["crash"])}**, with {voo_ridge["low"]["jumps_per_year"]:.0f} small jumps a
    year, to **{_pct(voo_ridge["high"]["crash"])}**, with {voo_ridge["high"]["jumps_per_year"]:.0f} larger
    ones: both readings pass a likelihood-ratio test on the same four years.

    Maximum likelihood lands far from where it did for our simulated stock: **{np.exp(_x[0]):.0f} jumps a year** of
    {_x[2]:.1f}% ± {np.exp(_x[1]):.1f}%, on a volatility of {np.exp(_x[3]) * np.sqrt(252):.0f}%. Many small
    jumps are how a constant-volatility model explains calm months followed by turbulent ones. This model
    has no volatility clustering, so take the VOO numbers as an illustration of the method, not as a risk
    figure: **slot 5** lets volatility move, and **slot 6** asks how to notice when a model, or a prior,
    does not fit.
    """)
    return


@app.cell(hide_code=True)
def _(RESULTS, grid, json, mo, np):
    _costs = json.loads((RESULTS / "svj-costs" / "costs.json").read_text())
    _ok = np.isfinite(grid["ll"])
    mo.md(f"""
    ## 8 · The catch

    Everything above was exact because the likelihood is cheap and the model has four free parameters: the
    grid held **{int(_ok.sum()):,} points**, and one history took seconds. Two things break this:

    * **A hidden path.** Let volatility move, as real volatility does (slot 5), and the likelihood of
      1,000 days becomes an integral over the whole volatility path: a 1,000-dimensional integral with no
      closed form. A particle filter estimates it, and particle MCMC turns those estimates into the exact
      posterior, at about **{_costs['pmmh']['median_cpu_s'] / 60:.0f} CPU-minutes per name** in our runs.
    * **Many names, every day.** For 2,000 names that is about
      **{_costs['pmmh']['cpu_hours_for_2000_names_at_median']:,.0f} CPU-hours every morning.**

    Bayes' rule is not the obstacle; computing it, name by name, is. **What if inference were paid once?**
    Train a network on simulated markets, then answer for any name in seconds. That is slot 2.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. **The prior matters where the data are weak.** Replace the jump-frequency prior by a flat one on
       log scale over the whole grid, from 0.03 to 126 jumps a year. What happens to the predictive
       probability and its interval? (Slot 6 is about choosing priors.)
    2. **Hidden or rare?** Give the methods the jump days, as if they were labelled, and count them. Does
       the Wald interval for the jump frequency work now?
    3. **A milder crash.** Repeat section 5 for a loss of 5% instead of 7%. Which answers move closer
       together, and why?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    flat_prior_button = mo.ui.run_button(label="Exercise 1: use a flat prior on the jump frequency")
    labelled_button = mo.ui.run_button(label="Exercise 2: count the labelled jump days")
    milder_button = mo.ui.run_button(label="Exercise 3: a loss of 5%")
    mo.hstack([flat_prior_button, labelled_button, milder_button], wrap=True)
    return flat_prior_button, labelled_button, milder_button


@app.cell
def _(crash_on_grid, flat_prior_button, grid, mo, model, np, posterior_weights, predictive):
    mo.stop(not flat_prior_button.value)
    _ok = np.isfinite(grid["ll"]) & np.isfinite(grid["lp"])
    _m, _s = model.prior_mean[0], model.prior_sd[0]
    _flat_lp = grid["lp"] + 0.5 * ((grid["x"][..., 0] - _m) / _s) ** 2  # remove the log-normal prior on frequency
    _lp = np.where(_ok, grid["ll"] + _flat_lp, -np.inf)
    _w = np.exp(_lp - _lp[_ok].max())
    _w /= _w.sum()
    _flat = float(np.sum(_w * crash_on_grid))
    _order = np.argsort(crash_on_grid.ravel())
    _cdf = np.cumsum(_w.ravel()[_order])
    _lo, _hi = np.interp([0.05, 0.95], _cdf, crash_on_grid.ravel()[_order])
    _log_lam = np.where(_ok, grid["x"][..., 0], 0.0)
    _before, _after = np.exp(np.sum(posterior_weights * _log_lam)), np.exp(np.sum(_w * _log_lam))
    mo.md(f"""
    With a flat prior on the log jump frequency, the predictive probability becomes **{100 * _flat:.2g}%**
    (90% interval {100 * _lo:.2g}% to {100 * _hi:.2g}%), against {100 * predictive:.2g}% with slot 5's prior.
    The typical jump frequency in the posterior (its geometric mean) moves from {_before:.2g} to {_after:.2g}
    a year: without the information that jumps come a few times a year, the posterior spreads along the
    ridge toward {"rarer" if _after < _before else "more frequent"} jumps. The prior is doing real work here;
    it has to be stated, and checked.
    """)
    return


@app.cell
def _(jump_days, labelled_button, mo, np, stats, truth):
    mo.stop(not labelled_button.value)
    _count, _years = int(jump_days.sum()), len(jump_days) / 252
    _lo = stats.chi2.ppf(0.025, 2 * _count) / 2 / _years
    _hi = stats.chi2.ppf(0.975, 2 * _count + 2) / 2 / _years
    _rate = _count / _years
    _w = (_rate * np.exp(-1.96 / np.sqrt(_count)), _rate * np.exp(1.96 / np.sqrt(_count)))
    mo.md(f"""
    With the jump days labelled, the count is a Poisson variable: {_count} jumps in {_years:.1f} years. The
    Wald interval on the log scale is **{_w[0]:.2f} to {_w[1]:.2f}** jumps a year and the exact interval
    {_lo:.2f} to {_hi:.2f}, around a truth of {np.exp(truth[0]):.1f}. Both are ordinary, informative
    intervals. Rare jumps are not the problem; **hidden** jumps are.
    """)
    return


@app.cell
def _(analysis, milder_button, mo, model, np, truth):
    mo.stop(not milder_button.value)
    _b = analysis["by_threshold"]["5"]
    _true5 = float(model.crash_probability(truth, 5.0, 0.03))
    _pct = lambda _v: "0%" if _v == 0 else "below 0.01%" if _v < 1e-4 else f"{100 * _v:.2g}%"
    mo.md(f"""
    For a loss of 5% the truth is **{_pct(_true5)}**. Bayes {_pct(_b["point"]["bayes"])}, maximum likelihood
    {_pct(_b["point"]["mle"])}, bootstrap {_pct(_b["point"]["bootstrap"])}, peaks over threshold
    {_pct(_b["point"]["evt"])}, historical frequency {_pct(_b["point"]["empirical"])}. A milder event sits closer
    to what the data contain, so the methods agree more: the problem lives in the tail.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to take away

    1. **A point estimate hides the uncertainty that decides a tail number.** Four years of returns cannot
       tell a few large jumps from many small ones, and the two stories disagree about a crash day by orders
       of magnitude.
    2. **The standard tools inherit the problem when the tail is weakly identified.** The plug-in, the
       Wald interval and the bootstrap stay at one end of the ridge, and sometimes call a one-in-three
       event a one-in-a-million one; the historical count and the Pareto tail extrapolate from too
       little. With a well-identified model, such as a Student-t on Student-t data, the plug-in is fine.
    3. **Two ideas fix most of it: average over what the data allow, and say what you know.** Either one
       alone helps a lot; the profile likelihood and the penalized fit are their frequentist versions.
       Bayes does both at once and returns one calibrated probability, at the price of a stated prior.
    4. **Bayes' rule is easy; computing it is not.** Here a grid sufficed. With a hidden volatility path it
       costs about half an hour of CPU per name. That cost is what the rest of the tutorial removes.

    **Next, slot 2:** how a neural network learns to return a posterior in one pass.
    """)
    return


if __name__ == "__main__":
    app.run()
