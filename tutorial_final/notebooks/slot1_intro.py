import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="Slot 1 · Bayes for volatility models",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from sbi4finance.calibration import plots, surface, teaching

    ROOT = mo.notebook_dir().parent
    RESULTS = ROOT / "results" / "calibration"
    DOCS = ROOT.parent / "docs"
    plots.style()
    return DOCS, RESULTS, ROOT, mo, np, plots, plt, surface, teaching


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Slot 1 · Bayes for volatility models

    **The range behind every calibrated number**

    Every morning a derivatives desk fits a volatility model to the option quotes of the day. It then uses
    the fitted model to price products nobody quotes, such as an option whose strike is set in a year's
    time. The quotes are noisy, so the fit is uncertain, and so is every price computed from it.

    This notebook makes three claims, each backed by a computation below:

    1. **A calibration is an estimate.** Several parameter sets fit the same quotes, they price unquoted
       products differently, and the rules ask banks to report that range.
    2. **The posterior is the range.** It is every parameter value consistent with the quotes at their
       precision, and pricing a product under each posterior draw turns it into a range of prices.
    3. **For the models desks use, the posterior is expensive.** It costs minutes to hours per smile,
       too slow for every smile every morning. Amortized inference removes that cost; the rest of the
       afternoon shows how, and how to check it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | | Slot 1: Black–Scholes | Slot 4: Heston | Slot 5: rough Bergomi |
    |:---|:---|:---|:---|
    | Data | two SPY quotes | the SPY smile, 63 prices | the same |
    | Parameters | the volatility σ | $v_0$, $\theta$, $\sigma_v$, $\rho$ | $H$, $\eta$, $\rho$, and the market's forward-variance curve |
    | One likelihood evaluation | about 10 µs per price | 20 ms per smile (one quadrature rung) | 5 CPU-seconds per smile |
    | One exact posterior | a grid, a fraction of a millisecond | about 4 CPU-minutes | at least 17 CPU-hours |
    | What it does on SPY | cannot fit two quotes | fits, with extreme parameters | built for this smile's shape |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "**The model in detail** (click to open): Black–Scholes, the quote noise and the posterior": mo.md(r"""
        #### 1 · Prices in forward units

        Divide an option's price by the discount factor and by the forward $F$. Then the forward is 1,
        interest rates and dividends disappear, and a call with log-moneyness $k = \ln(K/F)$ and maturity
        $T$ costs, under Black–Scholes with volatility $\sigma$,

        $$c(k, T; \sigma) = N(d_1) - e^{k} N(d_2), \qquad d_{1,2} = \frac{-k \pm \tfrac12 \sigma^2 T}{\sigma\sqrt T},$$

        where $N$ is the standard normal distribution function. A put follows by put–call parity,
        $p = c - 1 + e^{k}$. The **implied volatility** of a quote is the $\sigma$ that reproduces it.

        #### 2 · The design grid

        The tutorial reads every chain onto the same 63 points: 7 maturities (5, 10, 21, 42, 63, 126 and 252
        business days) times 9 standardized moneyness points $\delta = k / (\sigma_{\text{ref}}\sqrt T)$ from
        $-2$ to $2$, with $\sigma_{\text{ref}} = 20\%$. Out-of-the-money puts sit at $\delta < 0$, calls at
        $\delta > 0$.

        #### 3 · The quote noise

        Each price is the model price plus Gaussian noise with a known standard deviation,

        $$y_c = c_c(\sigma) + \sigma_c\,\varepsilon_c, \qquad \sigma_c = \text{vega}_c(\sigma_{\text{ref}}) \times s(\delta_c),$$

        where $s$ rises from a third of a volatility point at the money to one point at $|\delta| = 2$. On
        21 October 2026 this noise is wider than SPY's bid–ask at every one of the 63 points, so it is the
        tutorial's assumption, not the market's spread.

        #### 4 · The posterior on a grid

        With a flat prior on $[5\%, 60\%]$, the posterior of $\sigma$ given quotes $y_1, \dots, y_n$ is

        $$p(\sigma \mid y) \propto \prod_{c=1}^{n} \exp\!\Big(-\frac{(y_c - c_c(\sigma))^2}{2\sigma_c^2}\Big),$$

        computed here on 4,001 grid points. The posterior of a product's price is the law of that price
        when $\sigma$ is drawn from $p(\sigma \mid y)$.
        """)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · What a desk does every morning

    These are the 63 SPY prices from the close of 21 October 2026, read from the chain onto the design grid
    and shown as implied volatilities. A desk fits a model to them. The model then prices what is not on
    the grid: a forward-start option, a volatility swap, a barrier.
    """)
    return


@app.cell
def _(RESULTS, np, surface, teaching):
    spy = teaching.load_json(RESULTS / "spy" / "surface.json")
    spy_prices = np.asarray(spy["prices_forward_units"])
    spy_noise = np.asarray(spy["noise_sd"])
    spy_iv = teaching.implied_vol_grid(spy_prices).reshape(len(surface.DESIGN_DAYS), len(surface.DESIGN_DELTAS))
    return spy_iv, spy_noise, spy_prices


@app.cell(hide_code=True)
def _(plt, spy_iv, surface, teaching):
    _fig, _ax = plt.subplots(figsize=(8.4, 3.8), constrained_layout=True)
    for _j, _days in enumerate(surface.DESIGN_DAYS):
        _ax.plot(surface.DESIGN_DELTAS, 100 * spy_iv[_j], marker="o", markersize=3, linewidth=1.2,
                 color=plt.cm.viridis(_j / (len(surface.DESIGN_DAYS) - 1)), label=teaching.TENORS[_days])
    _ax.set_xlabel("standardized moneyness δ (puts left, calls right)")
    _ax.set_ylabel("implied volatility (%)")
    _ax.set_title("SPY, close of 21 October 2026: 63 prices on the design grid", loc="left")
    _ax.legend(ncols=4, fontsize=8, loc="upper right")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Two calibrations, two prices

    How uncertain can a calibrated price be? The figure comes from a study of 9 October on Heston's model
    with jumps, fitted to synthetic smiles whose true parameters are known. On the SHOP-like smile, thirty
    starts of the optimizer end at two parameter sets that fit the quotes almost equally well. They give a
    one-year variance-swap volatility 83% above and 16% below the true one. The blue shapes are the
    posterior: all parameter values consistent with the quotes, and the prices they give.
    """)
    return


@app.cell(hide_code=True)
def _(DOCS, mo):
    mo.image(src=DOCS / "design" / "images" / "01-hkde-option-prices.png",
             caption="HKDE study, 9 October 2026: exotic prices relative to the true-parameter price under the posterior "
                     "(violins, 90% interval as a bar) and at the two optimizer end points (dots).")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · The rules ask for the range

    | Rule | What it asks for |
    |:---|:---|
    | Regulation (EU) 2016/101, Art. 9 (market price uncertainty) | for inputs calibrated to market quotes, a range of plausible values and a point in it at 90% confidence |
    | Regulation (EU) 2016/101, Art. 11 (model risk) | the same for the choice among models and calibration approaches |
    | IFRS 13, §93(h) | the sensitivity of a Level 3 fair value to reasonably possible alternative inputs |
    | PRA SS1/23 | model uncertainty: "the inherent uncertainty in the parameter estimates and results of statistical models" |

    The difference between the fair value and the 90% point is a valuation adjustment, deducted from
    capital. Least squares returns one parameter vector and one price; the rules ask for a distribution of
    prices and a point in it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · One quote gives a posterior

    Black–Scholes has one parameter, the volatility $\sigma$. Take one quote: the at-the-money one-month
    price. Its noise is a third of a volatility point, so only volatilities close to its implied volatility
    are consistent with it. The curve below is the posterior.
    """)
    return


@app.cell
def _(np, spy_noise, spy_prices, surface, teaching):
    _k, _maturity, _kind, _delta = teaching.design_arrays()
    _one_month = np.flatnonzero(_maturity * surface.TRADING_DAYS == 21)
    _atm = _one_month[np.argmin(np.abs(_delta[_one_month]))]
    _put = _one_month[np.argmin(_delta[_one_month])]
    quotes = {
        name: {"k": float(_k[i]), "T": float(_maturity[i]), "kind": str(_kind[i]), "price": float(spy_prices[i]),
               "sd": float(spy_noise[i]), "strike": float(np.exp(_k[i]))}
        for name, i in (("at the money", _atm), ("put", _put))
    }
    sigma_grid = np.linspace(0.05, 0.60, 4001)


    def standardized_miss(quote, noise_multiple=1.0):
        """(quote - model) / noise for every sigma on the grid."""
        model = surface.bs_price(quote["k"], quote["T"], sigma_grid, quote["kind"])
        return (quote["price"] - model) / (noise_multiple * quote["sd"])


    def grid_posterior(log_likelihood):
        """Normalized posterior on sigma_grid under the flat prior."""
        weights = np.exp(log_likelihood - log_likelihood.max())
        return weights / np.trapezoid(weights, sigma_grid)


    one_quote = grid_posterior(-0.5 * standardized_miss(quotes["at the money"]) ** 2)
    one_quote_mean = np.trapezoid(sigma_grid * one_quote, sigma_grid)
    one_quote_sd = np.sqrt(np.trapezoid((sigma_grid - one_quote_mean) ** 2 * one_quote, sigma_grid))
    return (
        grid_posterior,
        one_quote,
        one_quote_mean,
        one_quote_sd,
        quotes,
        sigma_grid,
        standardized_miss,
    )


@app.cell(hide_code=True)
def _(mo, one_quote, one_quote_mean, one_quote_sd, plots, plt, sigma_grid):
    _fig, _ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    _ax.plot(100 * sigma_grid, one_quote, color=plots.SERIES[0])
    _ax.fill_between(100 * sigma_grid, one_quote, color=plots.SERIES[0], alpha=0.2)
    _ax.set_xlim(100 * (one_quote_mean - 6 * one_quote_sd), 100 * (one_quote_mean + 6 * one_quote_sd))
    _ax.set_xlabel("Black–Scholes volatility σ (%)")
    _ax.set_yticks([])
    _ax.set_title("posterior of σ from the at-the-money one-month quote", loc="left")
    mo.vstack([
        _fig,
        mo.md(f"Posterior: **σ = {100 * one_quote_mean:.2f}% ± {100 * one_quote_sd:.2f} volatility points** "
              "(mean ± standard deviation). The standard deviation is the quote's noise divided by its vega: "
              "the noise expressed as a volatility."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · The posterior prices another option

    To price another option, price it under every volatility the posterior allows. The spread of those
    prices is the range the rules ask for. Here the other option is the one-month put at the lowest strike
    of the grid.
    """)
    return


@app.cell
def _(np, one_quote, quotes, sigma_grid, surface):
    _cdf = np.cumsum(one_quote) / np.sum(one_quote)
    _low, _high = (sigma_grid[np.searchsorted(_cdf, q)] for q in (0.05, 0.95))
    put_range = [surface.bs_price(quotes["put"]["k"], quotes["put"]["T"], s, "put") for s in (_low, _high)]
    put_market_iv = float(surface.implied_vol(quotes["put"]["price"], quotes["put"]["k"], quotes["put"]["T"], "put"))
    return put_market_iv, put_range


@app.cell(hide_code=True)
def _(mo, put_market_iv, put_range, quotes):
    mo.md(f"""
    Under the posterior, the one-month put struck at **{100 * quotes['put']['strike']:.0f}% of the forward** is worth
    between **{1e4 * put_range[0]:.2f} and {1e4 * put_range[1]:.2f} basis points** of the forward (90% range).

    The market quotes it at **{1e4 * quotes['put']['price']:.1f} basis points**, an implied volatility of
    {100 * put_market_iv:.1f}%: about **{quotes['put']['price'] / put_range[1]:.0f} times** the top of the range.
    The range is honest about σ and silent about the model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Two quotes: the model is wrong

    Now use both quotes as data. If Black–Scholes were right, one volatility would fit both. The slider
    multiplies the assumed noise of both quotes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    noise_multiple = mo.ui.slider(1.0, 10.0, step=0.5, value=1.0, label="noise, multiple of the assumed noise",
                                  show_value=True)
    noise_multiple
    return (noise_multiple,)


@app.cell
def _(grid_posterior, noise_multiple, np, quotes, standardized_miss):
    _atm_miss = standardized_miss(quotes["at the money"], noise_multiple.value)
    _put_miss = standardized_miss(quotes["put"], noise_multiple.value)
    two_quotes = grid_posterior(-0.5 * (_atm_miss**2 + _put_miss**2))
    best = int(np.argmax(two_quotes))
    misses = {"at the money": float(_atm_miss[best]), "put": float(_put_miss[best])}
    likelihoods = {"at the money": np.exp(-0.5 * _atm_miss**2), "put": np.exp(-0.5 * _put_miss**2)}
    return best, likelihoods, misses, two_quotes


@app.cell(hide_code=True)
def _(best, likelihoods, misses, mo, plots, plt, sigma_grid, two_quotes):
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(10.4, 3.5), constrained_layout=True)
    for _name, _color in (("at the money", plots.SERIES[0]), ("put", plots.SERIES[1])):
        _left.plot(100 * sigma_grid, likelihoods[_name], color=_color, label=f"likelihood, {_name} quote")
    _left.plot(100 * sigma_grid, two_quotes / two_quotes.max(), color=plots.INK, linestyle="--",
               label="posterior from both (scaled)")
    _left.set_xlim(8, 32)
    _left.set_yticks([])
    _left.set_xlabel("Black–Scholes volatility σ (%)")
    _left.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.32, 1.0))
    _left.set_title("each quote alone, and both together", loc="left")
    _right.bar(list(misses), list(misses.values()), color=[plots.SERIES[0], plots.SERIES[1]])
    for _level in (-2, 2):
        _right.axhline(_level, color=plots.BASELINE, linestyle=":")
    _right.set_ylabel("quote minus model, in noise units")
    _right.set_title(f"the best single σ = {100 * sigma_grid[best]:.2f}%", loc="left")
    mo.vstack([
        _fig,
        mo.md(f"""
        The two likelihoods do not overlap. The posterior from both quotes stays at the at-the-money value,
        whose quote has the larger vega, and misses the put by **{misses['put']:.1f} noise units**. Move the
        slider: it takes about five times the assumed noise to bring the miss under two. That misfit is the
        **smile**. A posterior says which parameters fit only inside a model that can produce the data.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Richer models, costlier posteriors

    Desks therefore use models that produce a smile. Heston adds a random variance that moves against the
    price; it prices a smile by a Fourier integral (slot 4). Rough Bergomi makes that variance rough, which
    gives the skew the power-law shape SPY's has; it prices a smile only by Monte Carlo (slot 5). Each step
    up the ladder makes the posterior more expensive.
    """)
    return


@app.cell(hide_code=True)
def _(ROOT, mo):
    mo.image(src=ROOT / "curriculum" / "images" / "ladder.png",
             caption="One likelihood evaluation and one exact posterior under each model, CPU-seconds on a log scale "
                     "(Heston at one quadrature rung; the rough-Bergomi posterior is a lower bound).")
    return


@app.cell(hide_code=True)
def _(RESULTS, mo, teaching):
    costs = teaching.load_json(RESULTS / "costs" / "costs.json")
    mo.vstack([mo.md(f"The measured costs, on one {costs['cpu_count']}-core machine (study S9):"),
               mo.ui.table(teaching.cost_rows(costs), selection=None, pagination=False)])
    return (costs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Every smile, every morning

    One posterior is affordable for Heston. A desk needs one per underlying, every morning. Choose the
    number of smiles:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    smiles_per_morning = mo.ui.slider(1, 2000, step=1, value=500, label="smiles per morning", show_value=True)
    smiles_per_morning
    return (smiles_per_morning,)


@app.cell
def _(costs, mo, smiles_per_morning):
    _n = smiles_per_morning.value
    _heston_exact = costs["exact_heston_posterior"]["cpu_seconds"]
    _rough_exact = 3600 * costs["exact_rough_posterior"]["cpu_hours"]
    _network = costs["rough_amortized_posterior_1000_draws_cpu_s"]["median"]
    _audit = 425.0  # pilot value of study S5 (design document, S9); the full study replaces it
    morning = [
        {"route": "Heston, exact posterior (one quadrature rung)", "CPU-hours": _n * _heston_exact / 3600},
        {"route": "Rough Bergomi, exact posterior (lower bound)", "CPU-hours": _n * _rough_exact / 3600},
        {"route": "Rough Bergomi, amortized network", "CPU-hours": _n * _network / 3600},
        {"route": "Rough Bergomi, network plus audit (pilot cost)", "CPU-hours": _n * (_network + _audit) / 3600},
    ]
    for _row in morning:
        _row["cores for a four-hour morning"] = _row["CPU-hours"] / 4
        _row["CPU-hours"] = round(_row["CPU-hours"], 1)
        _row["cores for a four-hour morning"] = round(_row["cores for a four-hour morning"], 1)
    mo.ui.table(morning, selection=None, pagination=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 · Amortized inference: pay once, then seconds

    Amortized inference moves the cost into training. Parameters are drawn from the prior, smiles are
    simulated from them, and a network learns to return the posterior from a smile. A new smile then costs
    one pass through the network. The network is an approximation, so every slot that uses one also
    checks it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid("""
    flowchart LR
      P["prior over the parameters"] --> S["simulator: price a smile, add quote noise"]
      S --> B["a bank of (parameters, smile) pairs"]
      B --> N["train a network once"]
      Q["a new smile"] --> N
      N --> D["posterior draws in seconds"]
      D --> C["checks: an exact sibling model, SBC, an audit of each smile"]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    1. **A calibration is an estimate.** Two parameter sets can fit the same quotes and price an unquoted
       product far apart; the rules ask for the range and a 90% point.
    2. **The posterior is the range.** One quote gives a posterior for σ; pricing under each draw gives
       a price range.
    3. **The posterior is conditional on the model.** Black–Scholes cannot fit two SPY quotes, so its
       range for the put is precise and wrong.
    4. **Richer models make the posterior expensive**, from minutes for Heston to at least 17 CPU-hours
       for rough Bergomi, per smile, every morning.

    **Next, slot 2:** how a network trained on simulations returns the posterior, and when that pays.

    *Evidence and code:* the [curriculum](../curriculum/01-slot1-the-problem.md), the
    [design document](../../docs/design/calibration-design-document.md) and `src/sbi4finance/calibration/`.
    The returns and value-at-risk notebooks are in the take-home track.
    """)
    return


if __name__ == "__main__":
    app.run()
