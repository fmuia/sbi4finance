import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from sbi4finance.calibration import plots, surface, teaching

    ROOT = mo.notebook_dir().parent
    plots.style()
    return ROOT, mo, np, plots, plt, surface, teaching


@app.cell
def _(mo):
    mo.md(r"""
    # Intro, option E: one quote, then the smile

    *Draft storyboard for decision D1 (the plan, §10). It shows the computational core of
    option E; the framing slides around it are Francesco's.*

    A model, a market, and what the market does not tell you. A desk prices options with a
    model and calibrates the model to quotes. Bayes asks a plain question of every
    calibration: **which parameter values are consistent with the quotes, and what does that
    mean for the prices we did not observe?**

    The market here is one-month options on an index, simulated from the rough-Bergomi smile
    of slot 5 (the real SPY chain replaces it when the data are available). Prices are in
    units of the forward; each quote has a noise standard deviation from its bid–ask spread.
    """)
    return


@app.cell
def _(ROOT, np, surface, teaching):
    _example = teaching.load_json(ROOT / "results" / "calibration" / "smile" / "example-rough.json")
    _k, _maturity, _kind, _delta = teaching.design_arrays()
    _one_month = np.flatnonzero(np.isclose(_maturity * surface.TRADING_DAYS, 21))
    _atm = _one_month[np.argmin(np.abs(_delta[_one_month]))]
    _wing = _one_month[np.argmin(_delta[_one_month])]
    quotes = {
        name: {"k": float(_k[i]), "T": float(_maturity[i]), "kind": str(_kind[i]),
               "price": float(_example["noisy_quotes"][i]), "sd": float(_example["quote_noise_sd"][i]),
               "strike": float(np.exp(_k[i]))}
        for name, i in (("at the money", _atm), ("out-of-the-money put", _wing))
    }
    return (quotes,)


@app.cell
def _(mo):
    spread = mo.ui.slider(0.5, 10.0, step=0.5, value=1.0, label="bid–ask noise, multiple of the quoted sd",
                          show_value=True)
    spread
    return (spread,)


@app.cell
def _(np, quotes, spread, surface):
    sigma_grid = np.linspace(0.05, 0.60, 4001)

    def standardized(quote):
        model = surface.bs_price(quote["k"], quote["T"], sigma_grid, quote["kind"])
        return (quote["price"] - model) / (spread.value * quote["sd"])

    def posterior(log_likelihood):
        weights = np.exp(log_likelihood - log_likelihood.max())  # flat prior on the grid
        return weights / np.trapezoid(weights, sigma_grid)

    atm_z = standardized(quotes["at the money"])
    put_z = standardized(quotes["out-of-the-money put"])
    one_quote = posterior(-0.5 * atm_z**2)
    two_quotes = posterior(-0.5 * (atm_z**2 + put_z**2))
    return atm_z, one_quote, put_z, sigma_grid, two_quotes


@app.cell
def _(mo, np, one_quote, quotes, sigma_grid, surface):
    _mean = np.trapezoid(sigma_grid * one_quote, sigma_grid)
    _sd = np.sqrt(np.trapezoid((sigma_grid - _mean) ** 2 * one_quote, sigma_grid))
    _put = quotes["out-of-the-money put"]
    _cdf = np.cumsum(one_quote) / np.sum(one_quote)
    _lo, _hi = (sigma_grid[np.searchsorted(_cdf, q)] for q in (0.05, 0.95))
    _range = [surface.bs_price(_put["k"], _put["T"], s, "put") for s in (_lo, _hi)]
    _market_iv = float(surface.implied_vol(_put["price"], _put["k"], _put["T"], "put"))
    mo.md(f"""
    ## 1. One quote gives a posterior for the volatility

    Under Black–Scholes one volatility σ prices every option. The at-the-money quote alone
    gives the posterior for σ: **{100 * _mean:.2f}% ± {100 * _sd:.2f}%** (posterior mean and
    standard deviation, flat prior from 5% to 60%). The width is the quote's noise divided by
    its sensitivity to σ, its vega.

    ## 2. The posterior prices a second option, with a range

    Under that posterior the {100 * _put['strike']:.0f}% put is worth between
    **{1e4 * _range[0]:.1f} and {1e4 * _range[1]:.1f}** basis points of the forward (90% range).
    The market quotes it at **{1e4 * _put['price']:.1f}** basis points, an implied volatility of
    {100 * _market_iv:.1f}%. The range is honest about σ and silent about the model: it is
    far too narrow to contain the market's price.
    """)
    return


@app.cell
def _(atm_z, mo, np, one_quote, plots, plt, put_z, quotes, sigma_grid, two_quotes):
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(10.5, 3.6), constrained_layout=True)
    _atm_like = np.exp(-0.5 * atm_z**2)
    _put_like = np.exp(-0.5 * put_z**2)
    _left.plot(100 * sigma_grid, _atm_like, color=plots.SERIES[0], label="likelihood, at-the-money quote")
    _left.plot(100 * sigma_grid, _put_like, color=plots.SERIES[1],
               label=f"likelihood, {100 * quotes['out-of-the-money put']['strike']:.0f}% put")
    _left.plot(100 * sigma_grid, two_quotes / two_quotes.max(), color=plots.INK, linestyle="--",
               label="posterior from both (scaled)")
    _left.set_xlim(15, 40)
    _left.set_xlabel("Black–Scholes volatility σ (%)")
    _left.set_yticks([])
    _left.legend(fontsize=7.5, loc="upper right")
    _left.set_title("each quote alone, and both together", loc="left", fontsize=9)
    _best = np.argmax(np.exp(-0.5 * (atm_z**2 + put_z**2)))
    _right.bar(["at the money", f"{100 * quotes['out-of-the-money put']['strike']:.0f}% put"],
               [atm_z[_best], put_z[_best]], color=[plots.SERIES[0], plots.SERIES[1]])
    _right.axhline(2, color=plots.BASELINE, linestyle=":")
    _right.axhline(-2, color=plots.BASELINE, linestyle=":")
    _right.set_ylabel("quote minus model, in noise sd")
    _right.set_title(f"the best single σ = {100 * sigma_grid[_best]:.1f}% misses both quotes", loc="left", fontsize=9)
    mo.vstack([mo.md(r"""
    ## 3. Two quotes: no volatility fits both

    The two likelihoods barely overlap. The posterior from both quotes sits between them, at a
    volatility that misses each quote by many noise standard deviations. The data reject the
    model, and the shape of the rejection, puts dearer than calls, is the **smile**. Widen the
    noise with the slider: only an implausibly wide spread reconciles the two quotes.
    """), _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. P and Q, and when Bayes changes the decision

    | | Slot 4 | Slot 5 |
    |---|---|---|
    | Data | 1,000 daily returns | 63 option quotes: a smile |
    | Measure | **P**, the real world: forecast risk | **Q**, pricing: value products |
    | Model | Geometric Brownian motion | Heston, then rough Bergomi |
    | Does Bayes change the answer? | No: at 1,000 days the plug-in forecast equals the predictive one; the slot checks the workflow | Yes, where the product depends on what the smile does not pin down |
    | Is simulation-based inference needed? | No: the likelihood is exact | For rough Bergomi, yes: prices exist only by Monte Carlo |

    ## 5. When simulation is needed

    | | One posterior | Thousands of posteriors (every smile, every morning) |
    |---|---|---|
    | **Likelihood tractable** (Black–Scholes, Heston) | MCMC on the exact likelihood | MCMC per smile, or amortized for speed |
    | **Likelihood only by simulation** (rough Bergomi, path-dependent volatility) | MCMC with a Monte Carlo pricer inside: hours | **Amortized posterior: seconds, then checked** |

    The route: slot 4 under P, slot 5 under Q.
    """)
    return


if __name__ == "__main__":
    app.run()
