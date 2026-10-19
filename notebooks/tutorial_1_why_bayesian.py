import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Slot 1 · One beta, then five hundred")


@app.cell(hide_code=True)
def _():
    import json

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats

    from sbi4finance import beta_shrinkage as bs
    from sbi4finance.svj import data, plots

    ROOT = mo.notebook_dir().parent
    RESULTS = ROOT / "docs" / "results"
    plots.style()
    return RESULTS, ROOT, bs, data, json, mo, np, pd, plots, plt, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Slot 1 · One beta, then five hundred

    **Bayes' rule on a risk desk, with no network and no sampler**

    A risk desk hedges the stocks it holds with the market, and for that it needs each stock's
    beta. This notebook estimates betas the usual way, one regression per stock, and then with
    Bayes' rule, which lets every estimate learn from the other stocks. Everything is exact and
    takes milliseconds. Slot 2 starts where that stops being true.

    | Step | What it shows |
    |:---|:---|
    | 1 | The question: why data vendors adjust every beta |
    | 2 | Bayes' rule on one page |
    | 3 | One stock: the Bayesian market model |
    | 4 | Five hundred stocks: the winner's curse, and where the prior comes from |
    | 5 | Bloomberg's rule against Bayes' rule, at three data frequencies |
    | 6 | What the estimates do to a hedge |
    | 7 | Intervals |
    | 8 | On real stocks: Blume's test |
    | 9 | The catch: the same idea for a volatility model, at scale |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · The question

    Look up a stock's beta on Bloomberg and you get two numbers: the raw beta, from a regression of
    the stock's returns on the market's, and the **adjusted beta**,

    $$\beta_{\text{adjusted}} = 0.67\,\hat\beta + 0.33 \times 1 .$$

    Every beta is moved a third of the way toward 1, and Value Line also adjusts its betas toward 1.
    The weights go back to Marshall Blume (1971), who found that betas estimated in one period drift
    toward 1 in the next. MSCI's Barra equity risk model applies a "Bayesian shrinkage" step to
    each stock's specific risk for a related reason.

    So data vendors deliberately bias every estimate they show you. **Why would that make the
    estimates better, and is a third the right amount?** This notebook answers both questions
    with Bayes' rule, first for one stock and then for five hundred.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Bayes' rule on one page

    A model with unknown parameters $\theta$ says how likely the data $y$ are, $p(y \mid \theta)$: the
    **likelihood**. A **prior** $p(\theta)$ says what is plausible before seeing the data. Bayes' rule
    combines them into the **posterior**, everything the data and the prior say about $\theta$:

    $$p(\theta \mid y) \;=\; \frac{p(y \mid \theta)\, p(\theta)}{p(y)} \;\propto\; p(y \mid \theta)\, p(\theta).$$

    A decision depends on the future, not on $\theta$ itself. The **predictive** distribution averages
    the model's forecast over everything the posterior allows, instead of plugging in one estimate.

    Two kinds of interval will appear. A 90% **confidence interval** comes from a rule that, applied to
    many data sets, covers the truth 90% of the time. A 90% **credible interval** holds 90% of the
    posterior probability, given these data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "**You already use Bayes** (click to open): where it hides in a quant's toolbox": mo.md(r"""
        * **Adjusted betas** (Bloomberg, Value Line) shrink every estimate toward 1. Vasicek (1973)
          showed that Bayes' rule gives this kind of adjustment, with the amount set by each estimate's
          precision.
        * **Risk models** shrink noisy inputs: MSCI's Barra USE4 applies Bayesian shrinkage to specific
          risk (Menchero, Orr and Wang 2011), and Ledoit–Wolf shrinkage is standard for covariance
          matrices.
        * The **Kalman filter** is Bayes' rule for a hidden state, applied day after day. Slot 5's
          particle filter is its non-linear sibling.
        * **Black–Litterman** starts from a market-equilibrium prior and updates it with views.
        * **Bayesian VARs** with the Minnesota prior have been a forecasting staple since Litterman's
          work in the 1980s, and **stochastic volatility** has been fitted by MCMC since Jacquier,
          Polson and Rossi (1994).
        """),
        "**The Bayesian toolbox**: how the posterior is computed, and what it costs": mo.md(r"""
        | Route | What it needs from the model | Cost per new data set | Examples |
        |:---|:---|:---|:---|
        | Closed form | a conjugate prior and likelihood | microseconds | this notebook |
        | Grid | a likelihood; a handful of parameters | milliseconds to seconds | slot 4's exact check |
        | MCMC (HMC, NUTS) | a likelihood and its gradient | seconds to minutes | Stan, PyMC, NumPyro |
        | Particle MCMC | a simulator of the hidden state and an observation density | minutes to hours | slot 5's exact route |
        | Amortized neural inference | only a simulator | seconds, after one training run | BayesFlow (slots 2 to 6) |
        """),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · One stock: the Bayesian market model

    With $r_t$ the stock's return and $m_t$ the market's in week $t$, the market model is

    $$r_t = \alpha + \beta\, m_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal N(0, \sigma^2).$$

    Least squares gives an estimate $\hat\beta$ with a standard error $s$; with a year or more of
    weekly data, the likelihood of $\beta$ is close to $\mathcal N(\hat\beta, s^2)$. Before seeing
    this stock's data, we have some idea of what betas look like: a value-weighted average of all
    stocks' betas is exactly 1, the market's own. For now we take the prior
    $\beta \sim \mathcal N(1, 0.3^2)$; section 4 estimates its centre and spread from the data
    instead of assuming them. The posterior is normal too, and its mean is a weighted average:

    $$E[\beta \mid \text{data}] = 1 + w\,(\hat\beta - 1), \qquad w = \frac{0.3^2}{0.3^2 + s^2}, \qquad
    \text{sd}[\beta \mid \text{data}] = \sqrt{w}\, s .$$

    A precise estimate ($s$ small) keeps almost all its weight. A noisy one moves toward 1. In code:
    """)
    return


@app.cell(hide_code=True)
def _(ROOT, bs, data, mo, np):
    _csv = ROOT / "data" / "SPMO_VOO_2020-01-01_2026-09-04.csv"
    _weekly = lambda _ticker: data.load_returns(_csv, _ticker).resample("W-FRI").sum() / 100  # weekly log returns
    _pair = _weekly("SPMO").to_frame("SPMO").join(_weekly("VOO").to_frame("VOO")).dropna().iloc[-104:]
    spmo_b, spmo_se = bs.least_squares(_pair["SPMO"].to_numpy(), _pair["VOO"].to_numpy())

    _rng = np.random.default_rng(42)  # a single stock: true beta 1.3, idiosyncratic volatility 40% a year
    _market = _rng.normal(0.0, bs.SETTINGS["weekly"].market_sd, 104)
    _stock = 1.3 * _market + _rng.normal(0.0, 0.40 / np.sqrt(52), 104)
    stock_b, stock_se = bs.least_squares(_stock, _market)

    prior_mean, prior_sd = 1.0, 0.3
    spmo_post, spmo_post_sd, spmo_w = bs.posterior(spmo_b, spmo_se, prior_mean, prior_sd)
    stock_post, stock_post_sd, stock_w = bs.posterior(stock_b, stock_se, prior_mean, prior_sd)
    mo.show_code()
    return (spmo_b, spmo_post, spmo_post_sd, spmo_se, spmo_w, stock_b, stock_post, stock_post_sd, stock_se,
            stock_w, prior_mean, prior_sd)


@app.cell(hide_code=True)
def _(mo, np, plots, plt, prior_mean, prior_sd, spmo_b, spmo_post, spmo_post_sd, spmo_se, spmo_w, stats,
      stock_b, stock_post, stock_post_sd, stock_se, stock_w):
    _grid = np.linspace(0.0, 2.4, 600)
    _fig, _axes = plt.subplots(1, 2, figsize=(9.6, 3.0), constrained_layout=True, sharey=True)
    for _ax, _title, _b, _se, _m, _sd in ((_axes[0], "SPMO against VOO, 2 years of weekly returns", spmo_b, spmo_se, spmo_post, spmo_post_sd),
                                           (_axes[1], "A simulated stock, 2 years of weekly returns", stock_b, stock_se, stock_post, stock_post_sd)):
        _ax.plot(_grid, stats.norm.pdf(_grid, prior_mean, prior_sd), color=plots.MUTED, linestyle="--", label="prior: what betas look like")
        _ax.plot(_grid, stats.norm.pdf(_grid, _b[0], _se[0]), color=plots.SERIES[1], label="likelihood: this stock's data")
        _ax.plot(_grid, stats.norm.pdf(_grid, _m[0], _sd[0]), color=plots.SERIES[0], linewidth=2, label="posterior")
        _ax.set_title(_title, loc="left", fontsize=9)
        _ax.set_xlabel("beta")
        _ax.set_yticks([])
    _axes[0].legend(fontsize=7.5, loc="upper left")
    mo.vstack([_fig, mo.md(f"""
    For **SPMO**, an S&P 500 momentum ETF, the estimate is precise: $\\hat\\beta$ = {spmo_b[0]:.2f} ± {spmo_se[0]:.2f}, so it
    keeps {100 * spmo_w[0]:.0f}% of its weight and the posterior mean is {spmo_post[0]:.2f}. The **simulated stock** has a
    volatile business: its estimate is {stock_b[0]:.2f} ± {stock_se[0]:.2f}, it keeps {100 * stock_w[0]:.0f}% of its weight, and
    the posterior mean is {stock_post[0]:.2f}, between the estimate and 1.

    We simulated that stock with a true beta of 1.3, so for this one stock the estimate was closer than the
    posterior mean. That will happen: Bayes' rule pulls every estimate toward the middle, including the
    ones that really are extreme. A prior centred at 1 with a spread of 0.3 is a claim about the
    population of stocks, so its pay-off is a claim about the population too. The next section measures
    it on five hundred stocks, and asks where the prior should come from.
    """)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Five hundred stocks: the winner's curse

    A desk does not estimate one beta; it estimates hundreds. We simulate a universe of 500 stocks
    whose true betas we know: drawn around 1 with a spread of 0.3, with a median idiosyncratic
    volatility of 25% a year, and two years of weekly returns for each. Then we estimate every beta
    by least squares, as a desk would.
    """)
    return


@app.cell
def _(bs, np):
    universe = bs.simulate(np.random.default_rng(7), bs.SETTINGS["weekly"], n_stocks=500)
    ls_beta, ls_se = bs.least_squares(universe["returns"], universe["market"])
    top50 = np.argsort(ls_beta)[-50:]  # the 50 stocks with the highest estimated betas
    return ls_beta, ls_se, top50, universe


@app.cell(hide_code=True)
def _(bs, ls_beta, ls_se, mo):
    eb = bs.empirical_bayes(ls_beta, ls_se)
    # inside empirical_bayes: the prior's mean is the average estimate, and its variance is the variance
    # of the estimates minus the average sampling variance, because each estimate adds its own noise
    #   prior_mean = mean(b);  prior_sd = sqrt(var(b) - mean(se**2));  w = prior_sd**2 / (prior_sd**2 + se**2)
    mo.show_code()
    return (eb,)


@app.cell(hide_code=True)
def _(eb, ls_beta, mo, np, plots, plt, top50, universe):
    _true = universe["beta"]
    _fig, _axes = plt.subplots(1, 2, figsize=(9.6, 4.0), constrained_layout=True, sharex=True, sharey=True)
    for _ax, _est, _title in ((_axes[0], ls_beta, "Least squares, one stock at a time"), (_axes[1], eb["mean"], "Bayes, with the prior learned from all 500")):
        _rest = np.setdiff1d(np.arange(len(_true)), top50)
        _ax.scatter(_true[_rest], _est[_rest], s=9, color=plots.BASELINE, label="all stocks")
        _ax.scatter(_true[top50], _est[top50], s=14, color=plots.SERIES[1], label="the 50 highest least-squares betas")
        _ax.plot([0, 2.2], [0, 2.2], color=plots.INK_2, linewidth=0.8)
        _ax.set_title(_title, loc="left", fontsize=9)
        _ax.set_xlabel("true beta")
    _axes[0].set_ylabel("estimated beta")
    _axes[0].legend(fontsize=7.5, loc="upper left")
    _ls_top, _true_top, _eb_top = ls_beta[top50].mean(), _true[top50].mean(), eb["mean"][top50].mean()
    mo.vstack([_fig, mo.md(f"""
    Each estimate on its own is unbiased: least squares is right on average for every stock. But the
    stocks a desk singles out are not average. The 50 highest estimates average **{_ls_top:.2f}**,
    while those stocks' true betas average **{_true_top:.2f}**. They made the top 50 partly because their
    noise was positive. This is the winner's curse, and it is what Blume saw in real data: betas
    estimated as extreme drift back toward the middle in the next period.

    The prior learned from the cross-section has mean {eb["prior_mean"]:.2f} and spread {eb["prior_sd"]:.2f}, close to the
    true 1.00 and 0.30. The raw estimates spread more, by {np.std(ls_beta):.2f}, because they include the noise;
    using their spread as the prior is a known mistake (Lally 1998). With this prior, the same 50
    stocks get Bayesian estimates averaging **{_eb_top:.2f}**, on target.

    **This is not a matter of taste.** Charles Stein proved in 1956 that when you estimate three or
    more means at once, estimating each one separately is beaten. James and Stein (1961) gave an
    estimator that shrinks every estimate toward a common point and has a lower total squared error,
    whatever the true values; with four or more, the common point can be the average of the estimates
    themselves (Lindley 1962). Efron and Morris (1973) showed that this is Bayes' rule with the prior
    estimated from the data. The theorem is frequentist, since it holds at every truth, and it is
    stated for Gaussian errors with known variances. Our estimates have estimated, unequal variances;
    the study below shows the same result for them.
    """)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Bloomberg's rule against Bayes' rule

    Bloomberg's adjustment shrinks every beta by the same third. Bayes' rule shrinks each one by its
    own noise, so the amount depends on the data. The study below repeats the experiment on 200
    simulated universes for each of three common choices of data: one year of daily returns, two
    years of weekly returns and five years of monthly returns. The oracle uses the true population as
    its prior, which empirical Bayes has to estimate.
    """)
    return


@app.cell
def _(bs):
    studies = {_key: bs.study(_setting, n_universes=200, seed=1, keep_draws=True) for _key, _setting in bs.SETTINGS.items()}
    return (studies,)


@app.cell(hide_code=True)
def _(bs, mo, studies):
    _names = {"least_squares": "least squares", "bloomberg": "Bloomberg's rule", "bayes": "Bayes, prior learned from the data",
              "oracle": "Bayes, true population as prior"}
    _rows = []
    for _key in ("daily", "weekly", "monthly"):
        _s = studies[_key]
        for _m in bs.POINT_METHODS:
            _bias = _s['top_bias'][_m] if abs(_s['top_bias'][_m]) >= 0.005 else 0.0
            _rows.append(f"| {_s['setting'] if _m == 'least_squares' else ''} | {_names[_m]} | {_s['mse_vs_least_squares'][_m]:.2f} | "
                         f"{_s['universes_beating_least_squares'][_m]} of {_s['n_universes']} | {_bias:+.2f} |")
    _w = {_k: studies[_k]["median_weight_on_estimate"] for _k in studies}
    mo.md("\n".join([
        "| Data | Estimator | Squared error, relative to least squares | Universes where it beats least squares | Bias of the 50 highest |",
        "|:---|:---|:---|:---|:---|",
        *_rows,
        "",
        f"With daily data the estimates are precise: Bayes keeps about {100 * _w['daily']:.0f}% of each one, and Bloomberg's fixed third "
        f"**over-shrinks**, so it does worse than least squares. With weekly data, Bloomberg's default setting, Bayes keeps about "
        f"{100 * _w['weekly']:.0f}% and the fixed rule is close to right. With monthly data Bayes keeps {100 * _w['monthly']:.0f}% and the "
        "fixed rule under-shrinks. Bayes' rule adapts to each stock and each frequency; a fixed rule can be right for one at most. "
        "And learning the prior from the 500 stocks costs almost nothing against knowing the true population.",
    ]))
    return


@app.cell(hide_code=True)
def _(mo, np, plots, plt, studies):
    _draws = studies["weekly"]["hedge_exposure_draws"]
    _fig, _ax = plt.subplots(figsize=(9.6, 2.8), constrained_layout=True)
    _bins = np.linspace(-14, 9, 47)
    for _m, _label, _color in (("least_squares", "least squares", plots.SERIES[1]), ("bloomberg", "Bloomberg's rule", plots.MUTED),
                               ("bayes", "Bayes", plots.SERIES[0])):
        _ax.hist(_draws[_m], bins=_bins, color=_color, histtype="step", linewidth=1.8, label=_label)
    _ax.axvline(0, color=plots.INK, linewidth=0.8)
    _ax.set_xlabel("residual market exposure of the hedged book, in positions (negative: net short the market)")
    _ax.set_yticks([])
    _ax.grid(visible=False)
    _ax.legend(fontsize=7.5, loc="upper left")
    _ax.set_title("200 universes: hold the 50 highest-beta stocks, sell each estimated beta in the market", loc="left", fontsize=9)
    _ls, _bb, _ba = (studies["weekly"]["hedge_exposure"][_m]["mean"] for _m in ("least_squares", "bloomberg", "bayes"))
    mo.vstack([mo.md(r"""
    ## 6 · What the estimates do to a hedge

    A desk buys \$1m of each of the 50 stocks with the highest estimated betas and hedges each one by
    selling its estimated beta in the market. The hedge is right only if the betas are. With two years
    of weekly data:
    """), _fig, mo.md(f"""
    With least-squares betas the book is over-hedged by **{-_ls:.1f} positions** on average, about
    **\\${-_ls:.0f}m net short the market** on a \\$50m book, in every one of the 200 universes. At a
    market volatility of 16% a year, that unintended position adds about \\${-_ls * 0.16:.1f}m a year of P&L
    volatility that the risk report does not show. Bloomberg's rule leaves {_bb:+.1f} positions and Bayes
    {_ba:+.1f}: the Bayesian hedge is right on average, with a spread that comes only from the
    remaining uncertainty about each beta.
    """)])
    return


@app.cell(hide_code=True)
def _(mo, studies):
    _rows = "\n".join(
        f"| {studies[_k]['setting']} | {100 * studies[_k]['coverage']['least_squares']:.0f}% | {studies[_k]['median_width']['least_squares']:.2f} | "
        f"{100 * studies[_k]['coverage']['bayes']:.0f}% | {studies[_k]['median_width']['bayes']:.2f} | "
        f"{100 * (1 - studies[_k]['median_width']['bayes'] / studies[_k]['median_width']['least_squares']):.0f}% |"
        for _k in ("daily", "weekly", "monthly"))
    mo.md(f"""
    ## 7 · Intervals

    | Data | Least squares: covers the truth | Width | Bayes: covers the truth | Width | Bayes narrower by |
    |:---|:---|:---|:---|:---|:---|
    {_rows}

    Both 90% intervals cover the true betas about 90% of the time, and they promise different things.
    The least-squares interval is exact for each stock, whatever its true beta. The Bayesian interval
    covers 90% on average over the stocks of the universe, and it is narrower because it also uses
    what the other 499 stocks say. The classical intervals do not fail here; the gain is the
    information the stock-by-stock view leaves on the table.
    """)
    return


@app.cell(hide_code=True)
def _(RESULTS, bs, mo, pd):
    _table = pd.read_csv(RESULTS / "intro-betas" / "window_betas.csv")
    _tests = {_w: bs.blume_test(_g.to_dict("records"), n_boot=2000, seed=0) for _w, _g in _table.groupby("window_weeks")}
    _pct = lambda _v: f"{100 * _v:+.0f}%"
    _rows = "\n".join(
        f"| {_w // 52} year{'s' if _w > 52 else ''} | {_t['window_pairs']} | "
        f"{_pct(_t['mse']['bayes'] / _t['mse']['least_squares'] - 1)} | {_pct(_t['relative_change_interval']['bayes'][0])} to {_pct(_t['relative_change_interval']['bayes'][1])} | "
        f"{_t['window_pairs_beating_least_squares']['bayes']} of {_t['window_pairs']} | "
        f"{_pct(_t['mse']['bloomberg'] / _t['mse']['least_squares'] - 1)} | {_pct(_t['relative_change_interval']['bloomberg'][0])} to {_pct(_t['relative_change_interval']['bloomberg'][1])} |"
        for _w, _t in sorted(_tests.items()))
    _two = _tests[104]
    mo.md(f"""
    ## 8 · On real stocks

    The simulations show the mechanism with betas that stay put. Real betas also drift, so the real test
    is Blume's: estimate every beta in one window, then check how well each estimate predicts the next
    window's. We ran it on weekly returns of {_table.stock.nunique()} large US stocks against SPY, from November 1999
    to October 2026. They are utilities, consumer staples, health care and telecoms, because our free
    data quota ran out there. `docs/results/intro-betas/window_betas.csv` holds every window's beta; the
    prices are not in the repository.

    | Windows | Pairs of windows | Bayes, change in error | 90% interval | Bayes better in | Bloomberg's rule, change in error | 90% interval |
    |:---|:---|:---|:---|:---|:---|:---|
    {_rows}

    Errors are squared errors against the next window's least-squares beta, relative to least squares;
    the intervals come from a bootstrap over pairs of windows. Bayes' rule predicts the next period's
    betas better. Bloomberg's rule does not help here: these defensive stocks have betas around
    {_table.beta.mean():.2f}, so it pulls nine estimates in ten upward, and that moved them away from the next
    window's beta more often than toward it, as in exercise 1. Over two
    years the betas also drift: next-window betas regress on current ones with a slope of
    {_two['slope_next_on_current']:.2f}, where estimation noise alone would give {_two['constant_beta_slope']:.2f}. That drift is why Levi and
    Welch (2017) shrink a second time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "**Is this Bayesian or frequentist?** (click to open) What a careful frequentist would say, and what remains": mo.md(r"""
        * **Shrinkage is not exclusively Bayesian.** The James–Stein estimator is a frequentist estimator, and
          its advantage is a frequentist theorem. Ledoit–Wolf shrinkage of covariance matrices is the same
          idea. What the Bayesian view adds is a reason for the amount, each estimate's own noise against the
          spread of the population, and a full distribution to carry into decisions.
        * **The prior is estimated, not assumed.** Here it comes from the cross-section of 500 stocks:
          empirical Bayes. The only care needed is to remove the sampling noise from the spread of the
          estimates (Lally 1998).
        * **Real betas drift.** SPMO's beta to VOO was about 0.85 in 2022–23 and 1.30 in 2024–26, against
          standard errors near 0.05. A forecast of next year's beta therefore needs more shrinkage than
          estimation noise alone would suggest; Levi and Welch (2017) recommend Vasicek-adjusted betas from
          one to four years of daily returns, shrunk again for this drift.
        * **What it does not fix.** If the market model or the population is wrong, both answers inherit the
          error. Slot 6 is about noticing when the model or the prior does not fit.
        """),
    })
    return


@app.cell(hide_code=True)
def _(RESULTS, json, mo):
    _costs = json.loads((RESULTS / "svj-costs" / "costs.json").read_text())
    mo.md(f"""
    ## 9 · The catch

    Everything above was a weighted average: the posterior of 500 betas takes microseconds, because
    the market model with a normal prior has a closed-form posterior. The same structure, one model
    per name and a prior that describes the population of names, carries over to the volatility
    models in slots 4 and 5. The closed form does not.

    * **A hidden path.** In a stochastic-volatility model with jumps (slot 5), the likelihood of
      1,000 days is an integral over the whole volatility path, with no closed form. Particle MCMC
      computes the exact posterior, at about **{_costs['pmmh']['median_cpu_s'] / 60:.0f} CPU-minutes per name** in our runs.
    * **Many names, every day.** For 2,000 names that is about
      **{_costs['pmmh']['cpu_hours_for_2000_names_at_median']:,.0f} CPU-hours every morning.**

    Amortized inference keeps the structure and drops the cost: train a network once on names
    simulated from the population prior, then get each real name's posterior in seconds. That is
    slot 2.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. **A utilities universe.** Utilities have betas around 0.6. Bloomberg's rule pulls them toward 1;
       Bayes' rule pulls them toward the universe's own average. Which hedges better?
    2. **Five stocks.** Stein's theorem needs only four stocks to shrink toward their own average.
       With five, the prior is learned from very little: does shrinkage still pay, and does it pay in
       every universe?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    utilities_button = mo.ui.run_button(label="Exercise 1: a utilities universe")
    five_button = mo.ui.run_button(label="Exercise 2: five stocks")
    mo.hstack([utilities_button, five_button], wrap=True)
    return five_button, utilities_button


@app.cell
def _(bs, mo, utilities_button):
    mo.stop(not utilities_button.value)
    _population = {"beta_mean": 0.6, "beta_sd": 0.2, "idio_dispersion": 0.35}
    _s = bs.study(bs.SETTINGS["weekly"], n_universes=200, n_stocks=60, top=10, seed=5, population=_population)
    mo.md(f"""
    Sixty utilities with betas around 0.6, two years of weekly data, 200 universes. Squared error relative
    to least squares: Bloomberg's rule **{_s['mse_vs_least_squares']['bloomberg']:.2f}**, Bayes
    **{_s['mse_vs_least_squares']['bayes']:.2f}**. The ten highest estimates are too high by
    {_s['top_bias']['least_squares']:+.2f} with least squares, {_s['top_bias']['bloomberg']:+.2f} with Bloomberg's
    rule and {_s['top_bias']['bayes']:+.2f} with Bayes. Bloomberg's rule assumes the average stock has a beta of 1;
    for utilities it pulls every estimate toward 1, which on average is the wrong way. The prior should
    describe the population you are estimating.
    """)
    return


@app.cell
def _(bs, five_button, mo):
    mo.stop(not five_button.value)
    _s = bs.study(bs.SETTINGS["weekly"], n_universes=400, n_stocks=5, top=1, seed=5)
    mo.md(f"""
    Five stocks, two years of weekly data, 400 universes. Squared error relative to least squares: Bayes
    with the prior learned from the five **{_s['mse_vs_least_squares']['bayes']:.2f}**, Bayes with the true population
    {_s['mse_vs_least_squares']['oracle']:.2f}. Bayes beats least squares in {_s['universes_beating_least_squares']['bayes']} of
    400 universes. Shrinkage still pays on average, but with five stocks the population is estimated from
    very little, so it does not pay every time. With 500 stocks it paid in all 200.

    Bloomberg's rule does better here ({_s['mse_vs_least_squares']['bloomberg']:.2f}): its target, 1, happens to be this
    population's true centre, and five stocks cannot estimate the centre as well as that. When data on
    the population are scarce, a prior from outside the data (here, the knowledge that betas centre on
    1) is worth more than one learned from the data. Exercise 1 showed the cost when that outside
    knowledge is wrong.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to take away

    1. **One regression per stock ignores what the other stocks tell you.** The extreme estimates are
       the least reliable: the highest betas are too high and the lowest too low.
    2. **Bayes' rule with a prior learned from the cross-section fixes this**, by each estimate's own
       noise. It is provably better in total (Stein), and it is what data vendors approximate with a
       fixed rule.
    3. **You get a distribution, not just a better number.** Carry it into the decision: here, the
       hedge.
    4. **Here the posterior is a weighted average. For the volatility models in slots 4 and 5 it is
       not,** and computing it for every name, every day, is the problem the rest of the tutorial
       solves.

    **Next, slot 2:** how a neural network learns to return a posterior in one pass.
    """)
    return


if __name__ == "__main__":
    app.run()
