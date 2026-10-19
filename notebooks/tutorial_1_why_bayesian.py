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
    | 7 | Intervals: which question each one answers |
    | 8 | When the prior is wrong |
    | 9 | On real stocks: Blume's test |
    | 10 | The catch: the same idea for a volatility model, at scale |
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

    So data vendors deliberately pull every estimate toward 1. **Why would that make the estimates
    better, and is a third the right amount?** This notebook answers both questions with Bayes' rule,
    first for one stock and then for five hundred, and then looks for the cases where the answer
    breaks.
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
    _daily = data.load_returns(_csv, "SPMO").to_frame("SPMO").join(data.load_returns(_csv, "VOO").to_frame("VOO"), how="inner")
    _weekly = _daily.resample("W-FRI").sum(min_count=1).dropna() / 100  # weekly log returns
    _pair = _weekly.iloc[-104:]
    spmo_b, spmo_se = bs.least_squares(_pair["SPMO"].to_numpy(), _pair["VOO"].to_numpy())
    _before = _weekly.iloc[-208:-104]  # the two years before, for the drift note further down
    _b, _se = bs.least_squares(_before["SPMO"].to_numpy(), _before["VOO"].to_numpy())
    spmo_drift = {"before": (float(_b[0]), float(_se[0]), _before.index[0].strftime("%B %Y"), _before.index[-1].strftime("%B %Y")),
                  "last": (float(spmo_b[0]), float(spmo_se[0]))}

    _rng = np.random.default_rng(42)  # a single stock: true beta 1.3, idiosyncratic volatility 40% a year
    _market = _rng.normal(0.0, bs.SETTINGS["weekly"].market_sd, 104)
    _stock = 1.3 * _market + _rng.normal(0.0, 0.40 / np.sqrt(52), 104)
    stock_b, stock_se = bs.least_squares(_stock, _market)

    prior_mean, prior_sd = 1.0, 0.3
    spmo_post, spmo_post_sd, spmo_w = bs.posterior(spmo_b, spmo_se, prior_mean, prior_sd)
    stock_post, stock_post_sd, stock_w = bs.posterior(stock_b, stock_se, prior_mean, prior_sd)
    mo.show_code()
    return (spmo_b, spmo_drift, spmo_post, spmo_post_sd, spmo_se, spmo_w, stock_b, stock_post, stock_post_sd, stock_se,
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
    population's 1.00 and 0.30. The raw estimates spread more, {np.std(ls_beta):.2f}, because each carries its own
    noise; using their spread for the prior would be a known mistake (Lally 1998). With this prior, the
    same 50 stocks get Bayesian estimates averaging **{_eb_top:.2f}**, on target.

    **Pooling is not a matter of taste.** Charles Stein proved in 1956 that when three or more means are
    estimated at once, each with normal noise of one common, known variance, estimating each one
    separately is beaten in expected total squared error, whatever the true values. James and Stein
    (1961) gave an estimator that shrinks every estimate toward a fixed point; with four or more, the
    point can be the estimates' own average (Lindley 1962). Efron and Morris (1973) showed that this is
    Bayes' rule with the prior estimated from the data. The theorem is frequentist, since it holds at
    every truth, and a frequentist reaches our estimator too, as the best linear unbiased predictor of a
    random-coefficients model (Robinson 1991). The contrast is one stock at a time against all stocks
    together, not Bayes against frequentism.

    One condition matters: our standard errors differ from stock to stock, and then the theorem no
    longer covers this estimator (Bock 1975 gives estimators it does cover). What it gains depends on
    its prior describing the stocks. Sections 5 to 7 measure the gain when it does; section 8 shows
    what happens when it does not.
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
        for _m in ("least_squares", "bloomberg", "bayes", "oracle"):
            _bias = _s['top_bias'][_m] if abs(_s['top_bias'][_m]) >= 0.005 else 0.0
            _rows.append(f"| {_s['setting'] if _m == 'least_squares' else ''} | {_names[_m]} | {_s['mse_vs_least_squares'][_m]:.2f} | "
                         f"{_s['universes_beating_least_squares'][_m]} of {_s['n_universes']} | {_bias:+.2f} |")
    _w = {_k: studies[_k]["best_fixed_weight"] for _k in studies}
    mo.md("\n".join([
        "| Data | Estimator | Squared error, relative to least squares | Universes where it beats least squares | Bias of the 50 highest |",
        "|:---|:---|:---|:---|:---|",
        *_rows,
        "",
        "How far to shrink depends on how noisy the estimates are against how much the true betas vary. The fixed weight on "
        "the estimate with the smallest expected error is $\\tau^2/(\\tau^2 + \\overline{s^2})$, with $\\tau$ the spread of the true betas "
        f"and $\\overline{{s^2}}$ the average squared standard error. In this population it is {100 * _w['daily']:.0f}% with daily data, "
        f"{100 * _w['weekly']:.0f}% with weekly and {100 * _w['monthly']:.0f}% with monthly. Bloomberg's rule keeps 67% everywhere: far too little "
        "with daily data, where it **over-shrinks** and does worse than least squares, a little too little with weekly data, and "
        "about the best a fixed weight can do with monthly. The frequency matters only through that ratio, so the same daily "
        "data need more shrinkage in a population whose betas are closer together. Bayes' rule does better than any fixed "
        "weight here because it gives each stock its own, and learning its prior from the 500 stocks costs almost nothing "
        "against knowing the true population.",
    ]))
    return


@app.cell(hide_code=True)
def _(bs, mo, np, plots, plt, studies):
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
    _s = studies["weekly"]
    _ls, _bb, _ba = (_s["hedge_exposure"][_m]["mean"] for _m in ("least_squares", "bloomberg", "bayes"))
    _idio, _market = _s["book_idio_risk"], -_ls * bs.SETTINGS["weekly"].market_vol
    mo.vstack([mo.md(r"""
    ## 6 · What the estimates do to a hedge

    A desk buys \$1m of each of the 50 stocks with the highest estimated betas and hedges each one by
    selling its estimated beta in the market. The hedge is right only if the betas are. With two years
    of weekly data:
    """), _fig, mo.md(f"""
    With least-squares betas the book is over-hedged by **{-_ls:.1f} positions** on average, about
    **\\${-_ls:.0f}m net short the market** on a \\$50m book, and short in {_s['hedge_exposure']['least_squares']['universes_short']} of the 200 universes.
    Pick the 50 stocks at random instead and the same hedge is right on average
    ({_s['random_hedge_exposure']['least_squares']['mean']:+.1f} positions): the bias comes from choosing stocks by their estimates.

    The risk report, which takes the book as hedged, shows its idiosyncratic risk, about \\${_idio:.1f}m a
    year. The hidden short is a market position worth \\${_market:.1f}m a year on its own, so the book's true P&L
    volatility is about \\${(_idio**2 + _market**2) ** 0.5:.1f}m, and it loses money whenever the market rallies.
    Bloomberg's rule leaves {_bb:+.1f} positions and Bayes {_ba:+.1f}: in this population the Bayesian hedge is right
    on average. Section 8 shows populations where it is not.
    """)])
    return


@app.cell(hide_code=True)
def _(mo, studies):
    _pct = lambda _k, _m, _subset: f"{100 * studies[_k]['coverage'][_m][_subset]:.0f}%"
    _rows = "\n".join(
        f"| {studies[_k]['setting']} | {_pct(_k, 'least_squares', 'all')} | {_pct(_k, 'least_squares', 'top')} | {_pct(_k, 'least_squares', 'extreme')} | "
        f"{_pct(_k, 'bayes', 'all')} | {_pct(_k, 'bayes', 'top')} | {_pct(_k, 'bayes', 'extreme')} | "
        f"{100 * (1 - studies[_k]['median_width']['bayes'] / studies[_k]['median_width']['least_squares']):.0f}% |"
        for _k in ("daily", "weekly", "monthly"))
    mo.md(f"""
    ## 7 · Intervals: which question each one answers

    The share of true betas inside each 90% interval, over all stocks, over the 50 with the highest
    estimates, and over the stocks whose true beta lies beyond 1 ± 0.45 (1.5 population sds, about one
    stock in eight):

    | Data | Least squares: all | 50 highest | beyond 1 ± 0.45 | Bayes: all | 50 highest | beyond 1 ± 0.45 | Bayes narrower by |
    |:---|:---|:---|:---|:---|:---|:---|:---|
    {_rows}

    Over all stocks, both intervals cover about 90%, and they answer different questions. The
    least-squares interval is exact for any stock fixed in advance, but the 50 stocks a desk picks for
    their high estimates are not fixed in advance, and it misses more of them. The Bayesian interval
    conditions on the data, so the selection does not hurt it. For a stock whose true beta is far from
    the centre the roles turn: the Bayesian interval pulls it toward the middle and misses more often,
    while the least-squares interval stays exact. Neither interval fails; each is calibrated for its own
    question, and the Bayesian one is narrower because it uses what the other 499 stocks say.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · When the prior is wrong

    Everything so far used a population that matches the prior: betas from one normal distribution,
    and noise that says nothing about a stock's beta. Real stocks need not match it. The cross-section
    of betas can be skewed, and a stock's noise can be tied to its beta: small, volatile companies are
    the noisiest, and their betas can be far from 1. The weekly study runs again below on four
    populations like that. One more estimator joins: Bayes with a separate prior for each fifth of the
    stocks, ranked by their standard error, so that the prior can see what the pooled one ignores.
    """)
    return


@app.cell
def _(bs):
    robustness = {_key: bs.study(bs.SETTINGS["weekly"], _population, n_universes=200, seed=2)
                  for _key, _population in bs.MISSPECIFIED.items()}
    return (robustness,)


@app.cell(hide_code=True)
def _(mo, robustness, studies):
    _rows = []
    for _s in (studies["weekly"], *robustness.values()):
        _r, _h = _s["mse_vs_least_squares"], _s["hedge_exposure"]
        _rows.append(f"| {_s['population']} | {_r['bloomberg']:.2f} | {_r['bayes']:.2f} | {_r['bayes_by_volatility']:.2f} | "
                     f"{_h['bayes']['mean']:+.1f} | {_h['bayes_by_volatility']['mean']:+.1f} |")
    _skew, _beta, _fringe = robustness["skewed"], robustness["vol_on_beta"], robustness["fringe"]
    mo.md("\n".join([
        "| Population | Bloomberg's rule | Bayes, one prior | Bayes, a prior per volatility group | Hedge, one prior | Hedge, a prior per group |",
        "|:---|:---|:---|:---|:---|:---|",
        *_rows,
        "",
        "Squared error relative to least squares, and the hedge's market exposure in positions, as in section 6.",
        "",
        f"* **Skewed betas** cost little. The normal prior's tails are too thin, so the highest betas are pulled a little "
        f"too far ({_skew['top_bias']['bayes']:+.2f}), but one prior still beats Bloomberg's rule.",
        f"* **When noise rises with beta, or with distance from 1,** one prior shrinks hardest exactly the stocks whose betas "
        f"are extreme. Bloomberg's rule, which shrinks every stock alike, now does as well or better, and the Bayesian hedge "
        f"is net long by up to {max(_beta['hedge_exposure']['bayes']['mean'], robustness['vol_on_distance']['hedge_exposure']['bayes']['mean']):.1f} "
        "positions.",
        f"* **A quiet core and a noisy fringe** (nine stocks in ten with betas near 1 and 15% volatility, one in ten at 0.4 "
        f"or 1.6 with 50%) is the worst case: least squares beats one prior in "
        f"{_fringe['n_universes'] - _fringe['universes_beating_least_squares']['bayes']} of {_fringe['n_universes']} universes. "
        "Stein's theorem does not forbid this; it assumes every estimate has the same noise.",
        "",
        f"A normal prior with the true population's mean and spread fails the same way ({_fringe['mse_vs_least_squares']['oracle']:.2f} with the "
        "noisy fringe), so the problem is the prior's shape, not learning it from the data. The repair is a better prior, not a "
        "fixed rule: a prior per volatility group beats Bloomberg's rule where noise rises with beta, matches it where noise "
        "rises with distance from 1, and comes close with the noisy fringe. Commercial betas condition on more than "
        "volatility; Rosenberg and Guy (1976) predicted them from company fundamentals. Noticing that a prior does not fit "
        "is the subject of slot 6.",
    ]))
    return


@app.cell(hide_code=True)
def _(RESULTS, bs, mo, pd):
    _table = pd.read_csv(RESULTS / "intro-betas" / "window_betas.csv")
    _tests = {_w: bs.blume_test(_g.to_dict("records"), n_boot=2000, seed=0) for _w, _g in _table.groupby("window_weeks")}
    _pct = lambda _v: f"{100 * _v:+.0f}%"
    _cell = lambda _t, _m: (f"{_pct(_t['mse'][_m] / _t['mse']['least_squares'] - 1)} "
                            f"({_pct(_t['relative_change_interval'][_m][0])} to {_pct(_t['relative_change_interval'][_m][1])})")
    _rows = "\n".join(
        f"| {_w // 52} year{'s' if _w > 52 else ''} | {_t['window_pairs']} | {_cell(_t, 'bayes')} | {_cell(_t, 'bloomberg')} | "
        f"{_cell(_t, 'cross_section_mean')} | {_t['window_pairs_beating_least_squares']['bayes']} of {_t['window_pairs']} |"
        for _w, _t in sorted(_tests.items()))
    _one, _two = _tests[52], _tests[104]
    mo.md(f"""
    ## 9 · On real stocks

    The simulations hold each beta fixed. Real betas also move, so the real test is Blume's: estimate
    every beta in one window, then check how well each estimate predicts the next window's. We ran it on
    weekly returns of {_table.stock.nunique()} large US stocks against SPY, from November 1999 to October 2026, in
    windows that end at the latest week. `docs/results/intro-betas/window_betas.csv` holds every window's
    beta; the prices are not in the repository.

    | Windows | Pairs of windows | Bayes | Bloomberg's rule | Every beta set to the average | Bayes better than least squares in |
    |:---|:---|:---|:---|:---|:---|
    {_rows}

    Each cell is the change in squared error against the next window's least-squares beta, relative to
    least squares, with a 90% interval from a bootstrap over blocks of consecutive pairs of windows
    (consecutive pairs share a window). With one-year windows, Bayes' rule predicts next year's betas
    clearly better. With two-year windows there are only {_two['window_pairs']} pairs: the gain is smaller and less
    certain, and setting every beta to the average does about as well.

    Bloomberg's rule makes no measurable difference. These are defensive stocks with betas around
    {_table.beta.mean():.2f}, so it pulls {_one['share_below_one']:.0%} of the estimates upward, and that moved them away from the
    next window's beta in {_one['bloomberg_moved_away']:.0%} of cases with one-year windows and {_two['bloomberg_moved_away']:.0%} with two-year
    windows, as in exercise 1.

    The betas also move between windows by more than their standard errors allow. Within each pair of
    windows, net of the pair's average, next-window betas regress on current ones with a slope of
    {_one['slope_next_on_current']:.2f} (one year) and {_two['slope_next_on_current']:.2f} (two years), where estimation noise alone would give
    {_one['constant_beta_slope']:.2f} and {_two['constant_beta_slope']:.2f}. Standard errors robust to changing volatility explain only a
    small part of the gap, so the betas drift; Levi and Welch (2017) shrink a second time for this.

    One caution: these are {_table.stock.nunique()} stocks from defensive sectors (utilities, consumer staples, health care
    and telecoms) that survived to 2026, the ones our free data quota reached. They illustrate the
    test; Blume (1975) and Levi and Welch (2017) ran it on thousands of stocks.
    """)
    return


@app.cell(hide_code=True)
def _(mo, spmo_drift):
    _b0, _se0, _from, _to = spmo_drift["before"]
    _b1, _se1 = spmo_drift["last"]
    mo.accordion({
        "**Is this Bayesian or frequentist?** (click to open) What a careful frequentist would say, and what remains": mo.md(f"""
        * **Shrinkage is not exclusively Bayesian.** The James–Stein estimator is a frequentist estimator, its
          advantage is a frequentist theorem, and our estimator is also the best linear unbiased predictor of a
          random-coefficients model. Ledoit–Wolf shrinkage of covariance matrices is the same idea. What the
          Bayesian view adds is a reason for the amount, each estimate's own noise against the spread of the
          population, and a full distribution to carry into decisions.
        * **The prior is estimated, not assumed.** Here it comes from the cross-section of 500 stocks:
          empirical Bayes. The care needed is to remove the sampling noise from the spread of the estimates
          (Lally 1998), and to check that the prior's picture of the stocks fits them (section 8).
        * **Real betas drift.** SPMO's beta to VOO was {_b0:.2f} ± {_se0:.2f} from {_from} to {_to} and {_b1:.2f} ± {_se1:.2f}
          in the two years after. A forecast of next year's beta therefore needs more shrinkage than
          estimation noise alone would suggest; Levi and Welch (2017) recommend Vasicek-adjusted betas from
          one to four years of daily returns, shrunk again for this drift.
        * **What it does not fix.** If the market model or the population is wrong, both answers inherit the
          error. Slot 6 is about noticing when the model or the prior does not fit.
        """),
    })
    return


@app.cell(hide_code=True)
def _(RESULTS, json, mo):
    _costs = json.loads((RESULTS / "svj-costs" / "costs.json").read_text())["pmmh"]
    _hours = _costs["cpu_hours_for_2000_names_at_median"]
    mo.md(f"""
    ## 10 · The catch

    Everything above was a weighted average: the posterior of 500 betas takes microseconds, because
    the market model with a normal prior has a closed-form posterior. The same structure, one model
    per name and a prior that describes the population of names, carries over to the volatility
    models in slots 4 and 5. The closed form does not.

    * **A hidden path.** In a stochastic-volatility model with jumps (slot 5), the likelihood of
      1,000 days is an integral over the whole volatility path, with no closed form. Particle MCMC
      computes the exact posterior; with our basic sampler it took **{_costs['range_cpu_s'][0] / 60:.0f} to
      {_costs['range_cpu_s'][1] / 60:.0f} CPU-minutes per name** (median {_costs['median_cpu_s'] / 60:.0f}).
    * **Many names, every day.** For 2,000 names at the median that is about
      **{round(_hours, -1):,.0f} CPU-hours every morning**, or {_hours / 24:.0f} cores running all day.

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
    2. **Five stocks.** With five, the prior's centre and spread are learned from five estimates. Does
       shrinkage still pay, does it pay in every universe, and how does Bloomberg's fixed rule do?
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
    _s = bs.study(bs.SETTINGS["weekly"], bs.UTILITIES, n_universes=200, n_stocks=60, top=10, seed=5)
    _h = {_m: _s["hedge_exposure"][_m]["mean"] for _m in ("least_squares", "bloomberg", "bayes")}
    mo.md(f"""
    Sixty utilities with betas around 0.6, two years of weekly data, 200 universes. Squared error relative
    to least squares: Bloomberg's rule **{_s['mse_vs_least_squares']['bloomberg']:.2f}**, Bayes
    **{_s['mse_vs_least_squares']['bayes']:.2f}**. The ten highest estimates are too high by
    {_s['top_bias']['least_squares']:+.2f} with least squares, {_s['top_bias']['bloomberg']:+.2f} with Bloomberg's
    rule and {_s['top_bias']['bayes']:+.2f} with Bayes, so hedging those ten leaves {_h['least_squares']:+.1f}, {_h['bloomberg']:+.1f}
    and {_h['bayes']:+.1f} positions of market exposure. Bloomberg's rule assumes the average stock has a beta of 1;
    for utilities it pulls every estimate toward 1, which on average is the wrong way. The prior should
    describe the population you are estimating.
    """)
    return


@app.cell
def _(bs, five_button, mo, studies):
    mo.stop(not five_button.value)
    _s = bs.study(bs.SETTINGS["weekly"], n_universes=400, n_stocks=5, top=1, seed=5)
    _r, _n = _s["mse_vs_least_squares"], _s["universes_beating_least_squares"]
    mo.md(f"""
    Five stocks, two years of weekly data, 400 universes. Squared error relative to least squares: Bayes
    with the prior learned from the five **{_r['bayes']:.2f}**, Bayes with the true population {_r['oracle']:.2f},
    Bloomberg's rule **{_r['bloomberg']:.2f}**. Bayes beats least squares in {_n['bayes']} of 400 universes, and even the
    true prior only in {_n['oracle']}: the gain holds on average, and five errors make a noisy total. With 500
    stocks, Bayes won in {studies['weekly']['universes_beating_least_squares']['bayes']} of 200.

    Bloomberg's rule beats the learned prior here because it needs nothing from the data: its centre, 1,
    is this population's true centre, and its amount is close to the best fixed amount for weekly data (it keeps
    67% of each estimate; the best fixed weight is {100 * studies['weekly']['best_fixed_weight']:.0f}%). Empirical Bayes has to learn
    both from five numbers, and the amount is the hard part: in {_s['prior_sd_at_floor']} of 400 universes the
    estimated spread of the true betas came out at zero, and all five estimates were pooled to their
    average. Its 90% intervals cover only {100 * _s['coverage']['bayes']['all']:.0f}% of the true betas, because they treat the
    estimated prior as exact. With few stocks, a prior from outside the data beats one learned from it,
    as long as the outside prior is right; exercise 1 showed what it costs when it is not.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to take away

    1. **One regression per stock ignores what the other stocks tell you.** The extreme estimates are
       the least reliable: the highest betas are too high and the lowest too low.
    2. **Bayes' rule with a prior learned from the cross-section fixes this**, shrinking each estimate
       by its own noise. Stein's theorem explains why pooling pays when every estimate has the same
       noise. With unequal noise the prior has to describe the stocks; when it does not, a fixed rule
       can do as well or better, and the remedy is a better prior.
    3. **You get a distribution, not just a better number.** Carry it into the decision: here, the
       hedge.
    4. **Here the posterior is a weighted average. For the volatility models in slots 4 and 5 it is
       not,** and computing it for every name, every day, is the problem the rest of the tutorial
       solves.

    **Next, slot 2:** how a neural network learns to return a posterior in seconds.
    """)
    return


if __name__ == "__main__":
    app.run()
