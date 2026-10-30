import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="Slot 1 · Bayes for volatility models",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from sbi4f_tutorial import backbones, plots, surface, teaching

    ROOT = mo.notebook_dir().parent
    ASSETS = ROOT / "assets"
    RESULTS = ASSETS / "results" / "calibration"

    plots.style()
    return ASSETS, RESULTS, backbones, mo, np, plots, plt, surface, teaching


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Slot 1 · Bayes for volatility models

    This first part sets up the problem the tutorial works on and introduces the objects of the next slots. It
    follows the first steps of an introduction to Bayesian inference, on one data set: the SPY option quotes at
    the close of 21 October 2026.

    ## 1 · The problem

    ### What a derivatives desk does
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.mermaid("""
    flowchart LR
      Q["Option quotes<br/>bid and ask for each<br/>strike and expiration"] --> C["Calibration<br/>choose the parameters<br/>that reproduce the quotes"]
      C --> M["Calibrated model"]
      M --> P["Prices of products<br/>that are not quoted:<br/>forward starts, cliquets,<br/>long-dated options"]
      P --> B["The book:<br/>valuations, hedges,<br/>reserves"]
    """),
        mo.md(r"""
    The quotes change every day, so the calibration is redone every day.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The question

    Calibration usually returns one set of parameters, the one whose prices come closest to the quotes. But the
    quotes are noisy, and nearby parameter values reproduce them almost as well. Every price computed with the
    model inherits this uncertainty, and valuation rules ask for it to be measured: a range for each mark, at a
    stated confidence level.

    Bayesian inference gives this range. It treats the parameters as unknowns and turns the quotes into a
    probability distribution over them, the posterior. Pricing a product under the posterior gives a distribution
    for its price.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · The data

    SPY options are quoted for many expirations, from one day to more than two years, each with its own strikes. To
    compare models, and later to train networks, we read these quotes onto a fixed grid of 63 points: 7 maturities
    from one week to one year, and 9 strikes per maturity.

    Strikes are placed by their standardized moneyness $\delta = \ln(K/F) / (0.2\sqrt T)$: the distance of the strike
    $K$ from the forward $F$, measured in standard deviations of a 20% volatility over the option's life $T$. At
    $\delta = -2$ a put is far below the money; at $\delta = 2$ a call is far above it.
    """)
    return


@app.cell
def _(RESULTS, backbones, np, surface, teaching):
    spy = teaching.load_json(RESULTS / "spy" / "surface.json")
    spy_prices = np.asarray(spy["prices_forward_units"])
    spy_noise = np.asarray(spy["noise_sd"])
    spy_iv = teaching.implied_vol_grid(spy_prices).reshape(len(surface.DESIGN_DAYS), len(surface.DESIGN_DELTAS))
    spy_expirations = spy["diagnostics"]["expirations"]  # the listed chain, after cleaning

    # the networks of slots 4 and 5 read the same 63 contracts, with the same noise
    _context = backbones.load_example("heston-final500")[0]["context"]
    _k, _maturity, _ = surface.contract_arrays(_context["contracts"])
    same_design = bool(np.allclose(_k, teaching.design_arrays()[0]) and np.allclose(_maturity, teaching.design_arrays()[1])
                       and np.allclose(_context["quote_noise_std"], spy_noise, rtol=1e-12))
    return same_design, spy, spy_expirations, spy_iv, spy_noise, spy_prices


@app.cell(hide_code=True)
def _(mo, np, plots, plt, spy, spy_expirations, spy_iv, surface, teaching):
    _days, _deltas = np.array(surface.DESIGN_DAYS), np.array(surface.DESIGN_DELTAS)
    _put_color, _call_color = "#087F8C", "#D97732"  # the course's colours for this figure
    _exp_days = np.array([_e["days"] for _e in spy_expirations], dtype=float)
    _exp_forward = np.array([_e["forward"] for _e in spy_expirations])
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(9.6, 3.9), constrained_layout=True,
                                         gridspec_kw={"width_ratios": [1, 1.15]})
    for _d in _days:
        _forward = float(np.interp(_d, _exp_days, _exp_forward))
        _strikes = _forward * np.exp(_deltas * surface.SIGMA_REF * np.sqrt(_d / surface.TRADING_DAYS))
        _left.scatter(np.full(len(_deltas), _d), _strikes, s=16, zorder=3,
                      color=[_put_color if _x <= 0 else _call_color for _x in _deltas])
    _left.plot(_exp_days, _exp_forward, color=plots.INK_2, linewidth=1, label="forward $F_T$")
    _left.scatter([], [], s=16, color=_put_color, label="put (strike below the forward)")
    _left.scatter([], [], s=16, color=_call_color, label="call (strike above the forward)")
    _left.set_xscale("log")
    _left.set_xticks(list(_days), [str(_d) for _d in _days])
    _left.set_xlabel("maturity, trading days")
    _left.set_ylabel("strike, US dollars")
    _left.set_title("the 63 contracts read from the SPY chain", loc="left")
    _left.legend(fontsize=7.5, loc="upper left")
    for _j, _d in enumerate(_days):
        _right.plot(_deltas, 100 * spy_iv[_j], marker="o", markersize=3, linewidth=1.2,
                    color=plt.cm.viridis(_j / (len(_days) - 1)), label=teaching.TENORS[_d])
    _right.set_xlabel("standardized moneyness δ")
    _right.set_ylabel("implied volatility (%)")
    _right.set_title("implied volatility at the 63 points", loc="left")
    _right.legend(ncols=4, fontsize=7, loc="upper right")
    _kept = sum(_e["quotes_kept"] for _e in spy_expirations)
    _inside = not any(spy["grid_points_outside_quoted_strikes"])
    mo.vstack([
        _fig,
        mo.md(f"""
    Each dot is one of the 63 contracts on the grid: a put when its strike is below the forward, a call when it is
    above. The strikes spread out with maturity because $\\delta$ measures the distance from the forward in units of
    $0.2\\sqrt T$. The line is SPY's forward for each listed expiration, from put–call parity. The grid's prices are
    read from {_kept:,} cleaned quotes on {len(spy_expirations)} expirations. {"Every point lies inside the strikes quoted for its maturity, so the grid interpolates the quotes and never extrapolates them." if _inside else "Some points lie outside the quoted strikes, where the grid extrapolates."}
    On the right are the implied volatilities at these points, which form the smile.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The noise

    A quote is not an exact number: the bid and the ask differ, and the grid's prices are read from neighbouring
    expirations. Each of the 63 prices therefore comes with a standard deviation $\sigma_c$, its noise. A fixed rule
    sets it: 0.3 volatility points at the money, rising to 1 point at $\delta = \pm 2$, turned into a price with the
    Black–Scholes vega at 20%. On 21 October SPY's half-spreads are below this noise at every point. The 63 prices
    and their noise are all the data that the next sections use.
    """)
    return


@app.cell
def _(np, spy_iv, spy_noise, spy_prices, surface, teaching):
    _k, _maturity, _kind, _ = teaching.design_arrays()
    # implied vols of each price minus and plus one noise sd (0 where the lower price is not positive)
    spy_iv_band = np.stack([
        np.nan_to_num(surface.implied_vol(spy_prices - spy_noise, _k, _maturity, _kind), nan=0.0),
        surface.implied_vol(spy_prices + spy_noise, _k, _maturity, _kind),
    ]).reshape(2, *spy_iv.shape)
    return (spy_iv_band,)


@app.cell(hide_code=True)
def _(mo, np, plots, plt, spy_iv, spy_iv_band, surface, teaching):
    _deltas = np.array(surface.DESIGN_DELTAS)
    _fig, _axes = plt.subplots(1, 2, figsize=(9.6, 3.3), constrained_layout=True, sharey=True)
    for _ax, _row in zip(_axes, (2, 6)):
        _ax.errorbar(_deltas, 100 * spy_iv[_row], yerr=100 * np.abs(spy_iv_band[:, _row] - spy_iv[_row]), fmt="o",
                     markersize=4, color=plots.SERIES[0], elinewidth=1.2, capsize=3,
                     label="SPY's quote, ± one noise standard deviation")
        _ax.set_xlabel("standardized moneyness δ")
        _ax.set_title(f"SPY at {teaching.TENORS[surface.DESIGN_DAYS[_row]]}", loc="left")
    _axes[0].set_ylabel("implied volatility (%)")
    _axes[0].set_ylim(0, 34)
    _axes[0].legend(fontsize=7.5, loc="upper right")
    for _name, _col, _offset in (("at the money", 4, (14, 26)), ("the put of sections 4 and 5", 0, (14, 10))):
        _axes[0].annotate(_name, (_deltas[_col], 100 * spy_iv[2, _col]), xytext=_offset, textcoords="offset points",
                          fontsize=7.5, color=plots.INK_2, arrowprops={"arrowstyle": "-", "color": plots.MUTED, "lw": 0.7})
    _half = 50 * (spy_iv_band[1] - spy_iv_band[0])  # half-length of each bar, in volatility points
    _open = " and ".join(f"{_d:g}" for _d, _low in zip(_deltas, spy_iv_band[0, 2]) if _low == 0)
    mo.vstack([
        _fig,
        mo.md(f"""
    Each bar runs from the implied volatility of the quote minus one noise standard deviation to that of the quote
    plus one. At the money, the one-month quote pins the volatility to about ±{_half[2, 4]:.1f} points. Below the money
    the bars stay short, about ±{_half[2, 0]:.1f} points at δ = −2, less than the rule's 1 point, because SPY's
    volatility there is above 20% and the option's price is more sensitive to it. Above the money SPY's volatility
    is low and the prices are small; at δ = {_open} the bars reach zero, so any lower volatility is consistent with
    those quotes. The section *The data in detail*, after the takeaways, describes how the grid is read from the
    chain and what the later slots take from it.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Bayes with one parameter

    ### A model turns parameters into prices

    A volatility model gives a price for every option on the grid once its parameters are fixed. Black–Scholes has
    a single parameter, the volatility $\sigma$, and gives every option the same implied volatility, whatever its
    strike. That cannot match SPY's smile, and section 5 shows how a check of the model detects it. With one
    parameter, though, each step of Bayesian inference fits in one figure, so we start with Black–Scholes and a
    single quote: the at-the-money option that expires in one month.

    Calibration goes backwards, from the quote to $\sigma$. Least squares, the usual method, chooses the $\sigma$
    that minimizes the squared miss in noise units, summed over the quotes used:

    $$\chi^2(\sigma) = \sum_c \frac{\big(y_c - P_c(\sigma)\big)^2}{\sigma_c^2}.$$
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
    sigma_grid = np.linspace(0.05, 0.60, 4001)  # the flat prior's range


    def standardized_miss(quote, noise_multiple=1.0):
        """(quote - model) / noise for every sigma on the grid."""
        model = surface.bs_price(quote["k"], quote["T"], sigma_grid, quote["kind"])
        return (quote["price"] - model) / (noise_multiple * quote["sd"])


    def grid_posterior(log_density):
        """Normalized density on sigma_grid from an unnormalized log density."""
        weights = np.exp(log_density - log_density.max())
        return weights / np.trapezoid(weights, sigma_grid)


    one_quote_loglik = -0.5 * standardized_miss(quotes["at the money"]) ** 2
    one_quote = grid_posterior(one_quote_loglik)  # flat prior: the normalized likelihood
    one_quote_mean = np.trapezoid(sigma_grid * one_quote, sigma_grid)
    one_quote_sd = np.sqrt(np.trapezoid((sigma_grid - one_quote_mean) ** 2 * one_quote, sigma_grid))
    return (
        grid_posterior,
        one_quote,
        one_quote_loglik,
        one_quote_mean,
        quotes,
        sigma_grid,
        standardized_miss,
    )


@app.cell(hide_code=True)
def _(mo, np, plots, plt, quotes, sigma_grid, standardized_miss, surface):
    _quote = quotes["at the money"]
    _price = surface.bs_price(_quote["k"], _quote["T"], sigma_grid, _quote["kind"])
    _chi2 = standardized_miss(_quote) ** 2
    _best = sigma_grid[np.argmin(_chi2)]
    _inside = sigma_grid[_chi2 <= 1]
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(9.6, 3.2), constrained_layout=True)
    _left.axhspan(1e4 * (_quote["price"] - _quote["sd"]), 1e4 * (_quote["price"] + _quote["sd"]), color=plots.SERIES[1],
                  alpha=0.25, linewidth=0, label="the quote ± one noise standard deviation")
    _left.axhline(1e4 * _quote["price"], color=plots.SERIES[1], linewidth=1)
    _left.axvspan(100 * _inside[0], 100 * _inside[-1], color=plots.SERIES[0], alpha=0.18, linewidth=0,
                  label="volatilities that reproduce it")
    _left.plot(100 * sigma_grid, 1e4 * _price, color=plots.INK, linewidth=1.5, label="Black–Scholes price for each σ")
    _left.set_xlim(11, 15)
    _left.set_ylim(1e4 * np.interp([0.11, 0.15], sigma_grid, _price))
    _left.set_xlabel("volatility σ (%)")
    _left.set_ylabel("price (basis points of the forward)")
    _left.set_title("the model maps σ to a price; calibration goes back", loc="left")
    _left.legend(fontsize=7.5, loc="upper left")
    _right.axvspan(100 * _inside[0], 100 * _inside[-1], color=plots.SERIES[0], alpha=0.18, linewidth=0, label="χ² ≤ 1")
    _right.plot(100 * sigma_grid, _chi2, color=plots.INK, linewidth=1.5, label="χ²(σ)")
    _right.axvline(100 * _best, color=plots.SERIES[1], linewidth=1, linestyle="--",
                   label=f"least squares: σ = {100 * _best:.2f}%")
    _right.set_xlim(11, 15)
    _right.set_ylim(0, 25)
    _right.set_xlabel("volatility σ (%)")
    _right.set_ylabel("squared miss, in noise units")
    _right.set_title("least squares: the squared miss χ²(σ)", loc="left")
    _right.legend(fontsize=7.5, loc="upper center")
    mo.vstack([
        _fig,
        mo.md(f"""
    On the left, the Black–Scholes price of the one-month at-the-money option rises with σ. The quote with its
    noise is a horizontal band, and the volatilities whose prices fall inside it reproduce the quote to within its
    noise: from {100 * _inside[0]:.2f}% to {100 * _inside[-1]:.2f}%. On the right, least squares measures the miss of
    each σ in noise units and picks the smallest, σ = {100 * _best:.2f}%. The volatilities with χ² ≤ 1 form the same
    interval as on the left: the usual error bar of a least-squares fit.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prior × likelihood → posterior

    If the noise is Gaussian, the probability of seeing the quote when the volatility is $\sigma$, the likelihood
    $p(y \mid \sigma)$, is proportional to $e^{-\chi^2(\sigma)/2}$. Least squares keeps the peak of this curve;
    Bayesian inference keeps all of it. Bayes' rule multiplies the likelihood by a prior $p(\sigma)$, which says
    which volatilities we consider possible before seeing the quote, and rescales the product to integrate to one:

    $$p(\sigma \mid y) = \frac{p(y \mid \sigma)\, p(\sigma)}{p(y)}.$$

    The result is the posterior. With one parameter it can be computed exactly, on a grid of 4,001 volatilities.
    The default prior gives every value between 5% and 60% the same probability; the buttons change it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    prior_choice = mo.ui.radio(
        options={"flat, 5% to 60%": None, "normal, 20% ± 2 points": 0.02, "normal, 20% ± 0.5 points": 0.005,
                 "normal, 20% ± 0.2 points": 0.002},
        value="flat, 5% to 60%", inline=True, label="prior for σ",
    )
    prior_choice
    return (prior_choice,)


@app.cell(hide_code=True)
def _(
    grid_posterior,
    mo,
    np,
    one_quote_loglik,
    one_quote_mean,
    plots,
    plt,
    prior_choice,
    sigma_grid,
):
    _sd = prior_choice.value
    _log_prior = np.zeros_like(sigma_grid) if _sd is None else -0.5 * ((sigma_grid - 0.20) / _sd) ** 2
    _prior = grid_posterior(_log_prior)
    _posterior = grid_posterior(one_quote_loglik + _log_prior)
    _mean = np.trapezoid(sigma_grid * _posterior, sigma_grid)
    _spread = np.sqrt(np.trapezoid((sigma_grid - _mean) ** 2 * _posterior, sigma_grid))
    _fig, _ax = plt.subplots(figsize=(9.6, 2.9), constrained_layout=True)
    _ax.fill_between(100 * sigma_grid, _prior / _prior.max(), color=plots.GRID, alpha=0.6, linewidth=0, label="prior")
    _ax.plot(100 * sigma_grid, np.exp(one_quote_loglik), color=plots.SERIES[0], linewidth=1.5,
             label="likelihood of the at-the-money quote")
    _ax.plot(100 * sigma_grid, _posterior / _posterior.max(), color=plots.SERIES[1], linewidth=1.5, linestyle="--",
             label="posterior")
    _ax.set_xlim(5, 30)
    _ax.set_ylim(0, 1.08)
    _ax.set_yticks([])
    _ax.set_xlabel("Black–Scholes volatility σ (%)")
    _ax.set_title("Prior × likelihood → posterior, each scaled to the same height", loc="left")
    _ax.legend(loc="upper right", fontsize=8)
    if _sd is None:
        _text = f"""
    With the flat prior the posterior is the likelihood rescaled to integrate to one. Its mean is
    σ = {100 * _mean:.2f}%, with a standard deviation of {100 * _spread:.2f} volatility points. The width equals the
    noise of the quote divided by the option's vega, which converts the price noise into volatility. Slot 2 trains a
    network to reproduce this posterior.
    """
    else:
        _text = f"""
    With this prior the posterior mean is σ = {100 * _mean:.2f}%, with a standard deviation of
    {100 * _spread:.2f} volatility points, compared with {100 * one_quote_mean:.2f}% under the flat prior. A prior that
    is much wider than the precision of the quote barely changes the result, while a narrower one pulls it toward
    20%. Penalizing a calibration for moving away from a reference value, such as yesterday's parameters, is
    equivalent to using a prior of this kind. Slot 6 changes the priors of the real models.
    """
    mo.vstack([_fig, mo.md(_text)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### More quotes, a narrower posterior

    Each new quote multiplies the likelihood by one more factor, and the posterior narrows. To see this on data the
    model can produce, we simulate quotes from Black–Scholes itself, with a volatility of 13%: one at-the-money
    quote for each of the seven maturities, each with its noise. Simulated data, whose true parameters are known,
    are also how the methods of the afternoon are tested.
    """)
    return


@app.cell
def _(grid_posterior, np, sigma_grid, spy_noise, surface, teaching):
    sim_sigma = 0.13  # the volatility the quotes are simulated with
    _k, _maturity, _kind, _delta = teaching.design_arrays()
    _atm = np.flatnonzero(np.isclose(_delta, 0))  # one at-the-money option per maturity, from 1 week to 1 year
    sim_quotes = (surface.bs_price(_k[_atm], _maturity[_atm], sim_sigma, _kind[_atm])
                  + spy_noise[_atm] * np.random.default_rng(4).standard_normal(len(_atm)))
    _model = surface.bs_price(_k[_atm, None], _maturity[_atm, None], sigma_grid, _kind[_atm, None])  # 7 x 4,001
    _misses = (sim_quotes[:, None] - _model) / spy_noise[_atm, None]
    sim_posteriors = [grid_posterior(-0.5 * np.sum(_misses[:_n] ** 2, axis=0)) for _n in range(1, len(_atm) + 1)]
    return sim_posteriors, sim_sigma


@app.cell(hide_code=True)
def _(
    mo,
    np,
    plots,
    plt,
    sigma_grid,
    sim_posteriors,
    sim_sigma,
    surface,
    teaching,
):
    _means = np.array([np.trapezoid(sigma_grid * _p, sigma_grid) for _p in sim_posteriors])
    _sd = np.array([np.sqrt(np.trapezoid((sigma_grid - _m) ** 2 * _p, sigma_grid))
                    for _m, _p in zip(_means, sim_posteriors)])
    _cdfs = [np.cumsum(_p) / np.sum(_p) for _p in sim_posteriors]
    _intervals = [sigma_grid[np.searchsorted(_c, [0.05, 0.95])] for _c in _cdfs]  # 90% intervals
    _covered = sum(_low <= sim_sigma <= _high for _low, _high in _intervals)
    _total = len(sim_posteriors)
    _which = f"all {_total}" if _covered == _total else f"{_covered} of the {_total}"
    _count = np.arange(1, len(sim_posteriors) + 1)
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(9.6, 3.2), constrained_layout=True,
                                         gridspec_kw={"width_ratios": [1.5, 1]})
    for _n, _color in zip((1, 2, 4, 7), plt.cm.viridis(np.linspace(0.15, 0.85, 4))):
        _last = teaching.TENORS[surface.DESIGN_DAYS[_n - 1]]
        _label = "1 quote, at 1 week" if _n == 1 else f"{_n} quotes, 1 week to {_last}"
        _left.plot(100 * sigma_grid, sim_posteriors[_n - 1], color=_color, linewidth=1.5, label=_label)
    _left.axvline(100 * sim_sigma, color=plots.INK, linestyle="--", linewidth=1, label="the true σ")
    _left.set_xlim(11.8, 14.2)
    _left.set_yticks([])
    _left.set_xlabel("volatility σ (%)")
    _left.set_title("posterior of σ from simulated at-the-money quotes", loc="left")
    _left.legend(fontsize=7.5, loc="upper left")
    _right.plot(_count, 100 * _sd, "o", color=plots.SERIES[0], label="posterior standard deviation")
    _right.plot(_count, 100 * _sd[0] / np.sqrt(_count), color=plots.INK_2, linewidth=1,
                label="first value / √(number of quotes)")
    _right.set_ylim(0, 0.35)
    _right.set_xticks(_count)
    _right.set_xlabel("number of quotes")
    _right.set_ylabel("volatility points")
    _right.set_title("the width falls like 1/√n", loc="left")
    _right.legend(fontsize=7.5, loc="upper right")
    mo.vstack([
        _fig,
        mo.md(f"""
    With the first quote alone, the posterior's standard deviation is {100 * _sd[0]:.2f} volatility points. With all
    seven quotes it is {100 * _sd[-1]:.2f}, close to {100 * _sd[0]:.2f} / √7 = {100 * _sd[0] / np.sqrt(7):.2f}.
    The true value, {100 * sim_sigma:.0f}%, lies inside the 90% interval of {_which} posteriors,
    because these quotes come from the model itself. On real quotes the model can be wrong, and then more quotes
    narrow the posterior around a wrong value: section 5 shows this on SPY.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Using the posterior

    The main use of the posterior is pricing. To price another option, we draw volatilities from the posterior and
    price the option under each draw. As an example we take the one-month put with its strike far below the money, at
    $\delta = -2$, and suppose for now that the market does not quote it, as is the case for the products a desk has
    to mark.
    """)
    return


@app.cell
def _(np, one_quote, quotes, sigma_grid, surface):
    _rng = np.random.default_rng(1)
    _cdf = np.cumsum(one_quote) / np.sum(one_quote)
    _index = np.minimum(np.searchsorted(_cdf, _rng.uniform(size=4_000)), len(sigma_grid) - 1)
    sigma_draws = sigma_grid[_index] + (_rng.uniform(size=_index.size) - 0.5) * (sigma_grid[1] - sigma_grid[0])
    put_draws = surface.bs_price(quotes["put"]["k"], quotes["put"]["T"], sigma_draws, "put")
    put_range = np.quantile(put_draws, [0.05, 0.95])
    return put_draws, put_range, sigma_draws


@app.cell(hide_code=True)
def _(mo, plots, plt, put_draws, put_range, quotes, sigma_draws):
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(9.6, 2.9), constrained_layout=True)
    _left.hist(100 * sigma_draws, bins=40, color=plots.SERIES[0], alpha=0.85, edgecolor=plots.SURFACE, linewidth=0.4)
    _left.set_xlabel("σ (%)")
    _left.set_yticks([])
    _left.set_title(f"{len(sigma_draws):,} draws from the posterior of σ", loc="left")
    _right.hist(1e4 * put_draws, bins=40, color=plots.SERIES[1], alpha=0.85, edgecolor=plots.SURFACE, linewidth=0.4)
    for _edge in put_range:
        _right.axvline(1e4 * _edge, color=plots.INK, linestyle="--", linewidth=0.9)
    _right.set_xlabel("put price (basis points of the forward)")
    _right.set_yticks([])
    _right.set_title("the put priced under each draw; dashed: the 90% range", loc="left")
    mo.vstack([
        _fig,
        mo.md(f"""
    The put's strike is {100 * quotes['put']['strike']:.0f}% of the forward. Under the posterior, its price lies
    between {1e4 * put_range[0]:.2f} and {1e4 * put_range[1]:.2f} basis points of the forward with 90% probability.
    The same procedure, a pricer for the product applied to draws from the posterior, gives a price range for any
    product the model can price, for example an option that starts in a year. Slot 5 does this with rough Bergomi.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Valuation rules ask for this kind of range, that is, the uncertainty of a mark at a stated confidence level.

    | Rule | What it asks for |
    |:---|:---|
    | Regulation (EU) 2016/101, Art. 9 (market price uncertainty) | for inputs calibrated to market quotes, a range of plausible values and a point in it at 90% confidence |
    | Regulation (EU) 2016/101, Art. 11 (model risk) | the same for the choice among models and calibration approaches |
    | IFRS 13, §93(h) | the sensitivity of a Level 3 fair value to reasonably possible alternative inputs |
    | PRA SS1/23 | model uncertainty: "the inherent uncertainty in the parameter estimates and results of statistical models" |

    A posterior range answers the first row for one model. A full valuation adjustment needs more than that.
    """)
    return


@app.cell(hide_code=True)
def _(mo, put_range, quotes):
    mo.md(f"""
    ## 5 · Checking the model

    The posterior is computed inside the model, so on its own it cannot show that the model is wrong. A posterior
    predictive check compares what the fitted model predicts with quotes it was not fitted to. The market does quote
    the put of section 4, at {1e4 * quotes['put']['price']:.1f} basis points: about
    {quotes['put']['price'] / put_range[1]:.0f} times the upper end of the predicted range.
    """)
    return


@app.cell(hide_code=True)
def _(
    mo,
    np,
    plots,
    plt,
    put_draws,
    quotes,
    sigma_draws,
    spy_iv,
    spy_iv_band,
    surface,
):
    _predicted = put_draws + quotes["put"]["sd"] * np.random.default_rng(2).standard_normal(put_draws.size)
    _distance = (quotes["put"]["price"] - _predicted.mean()) / _predicted.std()
    _deltas = np.array(surface.DESIGN_DELTAS)
    _band = np.quantile(sigma_draws, [0.05, 0.95])
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(9.6, 3.3), constrained_layout=True)
    _left.hist(1e4 * _predicted, bins=40, color=plots.SERIES[0], alpha=0.85, edgecolor=plots.SURFACE, linewidth=0.4)
    _left.axvline(1e4 * quotes["put"]["price"], color=plots.SERIES[1], linewidth=2)
    _top = _left.get_ylim()[1]
    _left.text(1e4 * quotes["put"]["price"] - 0.4, 0.95 * _top, "the market's quote", ha="right", va="top", fontsize=7.5,
               color=plots.SERIES[1])
    _left.text(2.8, 0.82 * _top, "predicted quotes: the price under\neach posterior draw, plus noise", ha="left",
               va="top", fontsize=7.5, color=plots.SERIES[0])
    _left.annotate("", xy=(1e4 * quotes["put"]["price"], 0.45 * _top), xytext=(1e4 * _predicted.mean(), 0.45 * _top),
                   arrowprops={"arrowstyle": "<->", "color": plots.INK_2, "lw": 0.9})
    _left.text(1e4 * (quotes["put"]["price"] + _predicted.mean()) / 2, 0.47 * _top, f"{_distance:.1f} standard deviations",
               ha="center", va="bottom", fontsize=7.5, color=plots.INK_2)
    _left.set_yticks([])
    _left.set_xlabel("put price (basis points of the forward)")
    _left.set_title("the put at δ = −2: prediction against the market", loc="left")
    _right.axhspan(100 * _band[0], 100 * _band[1], color=plots.SERIES[0], alpha=0.3, linewidth=0,
                   label="Black–Scholes posterior, 90%: one σ for every strike")
    _right.errorbar(_deltas, 100 * spy_iv[2], yerr=100 * np.abs(spy_iv_band[:, 2] - spy_iv[2]), fmt="o", markersize=4,
                    color=plots.SERIES[1], elinewidth=1.2, capsize=3, label="SPY's quotes, ± one noise standard deviation")
    _right.set_ylim(0, 34)
    _right.set_xlabel("standardized moneyness δ")
    _right.set_ylabel("implied volatility (%)")
    _right.set_title("the same check across the 1-month smile", loc="left")
    _right.legend(fontsize=7.5, loc="upper right")
    mo.vstack([
        _fig,
        mo.md(f"""
    On the left are the quotes the model predicts for the put: its price under each posterior draw, plus the quote's
    noise. The market's quote lies {_distance:.1f} standard deviations away from them. On the right is the same check
    across the 1-month smile. Black–Scholes predicts the same volatility for every strike, while SPY's implied
    volatility rises from {100 * spy_iv[2, 4]:.1f}% at the money to {100 * spy_iv[2, 0]:.1f}% at δ = −2.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To see what goes wrong, we now use both quotes as data. If Black–Scholes were the right model, a single
    volatility would fit both. Before moving the slider, try to guess how much noisier the two quotes would have to
    be for one volatility to fit both.
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
    best_sigma = int(np.argmax(two_quotes))
    misses = {"at the money": float(_atm_miss[best_sigma]), "put": float(_put_miss[best_sigma])}
    likelihoods = {"at the money": np.exp(-0.5 * _atm_miss**2), "put": np.exp(-0.5 * _put_miss**2)}


    def _largest_miss(multiple):
        a, p = standardized_miss(quotes["at the money"], multiple), standardized_miss(quotes["put"], multiple)
        i = int(np.argmax(-0.5 * (a**2 + p**2)))
        return max(abs(a[i]), abs(p[i]))


    noise_needed = next((m for m in np.arange(1.0, 20.01, 0.25) if _largest_miss(m) < 2), np.nan)
    return best_sigma, likelihoods, misses, noise_needed, two_quotes


@app.cell(hide_code=True)
def _(
    best_sigma,
    likelihoods,
    misses,
    mo,
    noise_needed,
    plots,
    plt,
    sigma_grid,
    two_quotes,
):
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(9.6, 3.2), constrained_layout=True)
    for _name, _label, _color in (("at the money", "at-the-money quote", plots.SERIES[0]),
                                  ("put", "put quote", plots.SERIES[1])):
        _left.plot(100 * sigma_grid, likelihoods[_name], color=_color, label=_label)
    _left.plot(100 * sigma_grid, two_quotes / two_quotes.max(), color=plots.INK, linestyle="--",
               label="posterior from both")
    _left.set_xlim(8, 32)
    _left.set_yticks([])
    _left.set_xlabel("Black–Scholes volatility σ (%)")
    _left.legend(fontsize=7.5, loc="center", bbox_to_anchor=(0.45, 0.62))
    _left.set_title("each quote's likelihood, and the posterior from both", loc="left")
    _right.bar(list(misses), list(misses.values()), color=[plots.SERIES[0], plots.SERIES[1]])
    for _level in (-2, 2):
        _right.axhline(_level, color=plots.BASELINE, linestyle=":")
    _right.set_ylabel("quote minus model, in noise units")
    _right.set_title(f"the best single σ = {100 * sigma_grid[best_sigma]:.2f}%", loc="left")
    mo.vstack([
        _fig,
        mo.md(f"""
    The two likelihoods do not overlap. The posterior from both quotes stays close to the at-the-money value,
    because that quote is the more precise of the two, and it misses the put by {misses['put']:.1f} noise units. Only
    about {noise_needed:.2g} times the assumed noise would bring both misses below two units. No single volatility
    prices both options, and this mismatch is what the volatility smile shows. The posterior is precise but wrong,
    because it is computed inside a model that cannot produce these quotes. A model has to be checked before its
    ranges are used. For this reason the rest of the afternoon uses models that produce a smile: Heston, with four
    parameters (section 6 and slot 4), and rough Bergomi, with six (slot 5).
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · A model with a smile: Heston

    Heston lets the variance of SPY move randomly: it reverts to a long-run level, and its moves are correlated with
    the price's. With four free parameters, the initial and long-run variance, the volatility of variance and the
    correlation, and the speed of mean reversion fixed at 2, it reproduces the shape of SPY's smile. Rough Bergomi,
    the model of slot 5, describes short maturities better still, but it can only be priced by Monte Carlo.
    """)
    return


@app.cell
def _(ASSETS, backbones, np, surface, teaching):
    heston_task = backbones.load_example("heston-final500")[0]  # the grid, the noise, and mean reversion fixed at 2
    _fit = teaching.load_json(ASSETS / "data" / "heston-spy-exact.json")["summary"]
    heston_spy_fit = np.array([_fit[_name]["mean"] for _name in ("initial_variance", "theta", "vol_of_vol", "rho")])
    heston_iv = {
        _rho: teaching.implied_vol_grid(teaching.heston_smile(heston_task, np.r_[heston_spy_fit[:3], _rho])).reshape(
            len(surface.DESIGN_DAYS), len(surface.DESIGN_DELTAS))
        for _rho in (-0.85, heston_spy_fit[3], -0.4)
    }
    return heston_iv, heston_spy_fit, heston_task


@app.cell(hide_code=True)
def _(
    heston_iv,
    heston_spy_fit,
    mo,
    np,
    plots,
    plt,
    spy_iv,
    surface,
    teaching,
):
    _deltas = np.array(surface.DESIGN_DELTAS)
    _fitted = heston_iv[heston_spy_fit[3]]
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(9.6, 3.5), constrained_layout=True)
    for _j, _color in ((2, plots.SERIES[0]), (5, plots.SERIES[1])):
        _tenor = teaching.TENORS[surface.DESIGN_DAYS[_j]]
        _left.scatter(_deltas, 100 * spy_iv[_j], s=16, color=_color, zorder=3, label=f"SPY, {_tenor}")
        _left.plot(_deltas, 100 * _fitted[_j], color=_color, linewidth=1.4, label=f"Heston, {_tenor}")
    _left.axhline(100 * spy_iv[2, 4], color=plots.INK_2, linestyle="--", linewidth=1, label="Black–Scholes, one σ")
    _left.set_xlabel("standardized moneyness δ")
    _left.set_ylabel("implied volatility (%)")
    _left.set_title("SPY's smile, Black–Scholes and Heston", loc="left")
    _left.legend(fontsize=7.5, loc="upper right", ncols=2)
    for _rho, _color in ((-0.85, plots.SERIES[1]), (heston_spy_fit[3], plots.INK), (-0.4, plots.SERIES[2])):
        _right.plot(_deltas, 100 * heston_iv[_rho][2], color=_color, linewidth=1.5,
                    label=f"ρ = {_rho:.2f}".replace("-", "−"))
    _right.scatter(_deltas, 100 * spy_iv[2], s=16, color=plots.SERIES[0], zorder=3, label="SPY, 1 month")
    _right.set_xlabel("standardized moneyness δ")
    _right.set_ylabel("implied volatility (%)")
    _right.set_title("Heston at one month, changing only the correlation ρ", loc="left")
    _right.legend(fontsize=7.5, loc="lower left")
    mo.vstack([
        _fig,
        mo.md(r"""
    The Heston curves use the posterior mean of an exact calibration to these quotes, with the mean reversion fixed
    at 2; slot 4 computes it. On the right only the correlation between the price and its variance changes. A more
    negative correlation makes puts below the money more expensive and calls above it cheaper.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The posterior of four parameters

    Sections 3 and 4 carry over unchanged. The likelihood is the same product over the quotes, now over all 63 and
    as a function of four parameters, and a product is priced under each posterior draw. With four parameters a
    grid of values is no longer practical. Markov chain Monte Carlo (MCMC) samples the posterior instead: a chain of
    parameter values that visits each region in proportion to its posterior probability.
    """)
    return


@app.cell
def _(ASSETS, heston_task, np, teaching):
    heston_spy_draws = np.load(ASSETS / "data" / "heston-spy-exact.npy")  # the exact posterior of slot 4
    swap_vol_1y = teaching.heston_variance_swap_vol(heston_spy_draws[:, 0], heston_spy_draws[:, 1],
                                                    heston_task["fixed_parameters"]["kappa"], [252])[:, 0]
    return heston_spy_draws, swap_vol_1y


@app.cell(hide_code=True)
def _(heston_spy_draws, mo, np, plots, plt, swap_vol_1y):
    _shown = heston_spy_draws[np.random.default_rng(0).choice(len(heston_spy_draws), 3_000, replace=False)]
    _range = np.quantile(swap_vol_1y, [0.05, 0.95])
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(9.6, 3.2), constrained_layout=True)
    _left.scatter(_shown[:, 2], _shown[:, 3], s=3, alpha=0.3, color=plots.SERIES[0], rasterized=True)
    _left.set_xlabel("volatility of variance")
    _left.set_ylabel("correlation ρ")
    _left.set_title("Heston parameters consistent with SPY's quotes", loc="left")
    _right.hist(swap_vol_1y, bins=35, color=plots.SERIES[1], alpha=0.85, edgecolor=plots.SURFACE, linewidth=0.4)
    for _edge in _range:
        _right.axvline(_edge, color=plots.INK, linestyle="--", linewidth=0.9)
    _right.set_yticks([])
    _right.set_xlabel("1-year variance swap, in volatility points")
    _right.set_title("a 1-year variance swap priced under each of them", loc="left")
    mo.vstack([
        _fig,
        mo.md(f"""
    Each dot on the left is one draw from the posterior, out of {len(heston_spy_draws):,} computed by MCMC in a
    feasibility run of 23 October that slot 4 also uses; it is not yet an accepted study. On the right, the 1-year
    variance swap is priced under each draw. The dashed lines bound the central 90%, from {_range[0]:.2f} to
    {_range[1]:.2f} volatility points. With 63 liquid quotes the range is narrow. The next subsection shows a case
    where the quotes leave a parameter undetermined.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### When the posterior differs from least squares

    With a model that fits the smile, the posterior is often close to the least-squares answer of section 3: the
    best fit, with an error ellipse computed from the curvature of $\chi^2$. The table compares the two approaches.

    | | Least squares | Bayesian calibration |
    |:---|:---|:---|
    | Output | one parameter vector $\hat\theta$ | a distribution $p(\theta \mid y)$ |
    | Uncertainty | the Gauss–Newton ellipse $(J^\top W J)^{-1}$ around $\hat\theta$ | the posterior, whatever its shape |
    | What is known beforehand | penalties, parameters fixed by hand | a stated prior |
    | The range of a product | the delta method, through the product's gradient | the product priced under each posterior draw |
    | What "90%" means | across repeated sets of quotes | given today's quotes and the prior |
    | Cost | one optimization | thousands of likelihood evaluations |

    The two panels below use synthetic smiles on SPY's grid, with the same noise. The left panel shows Heston's exact
    posterior, sampled by MCMC in PyMC. The right panel shows rough Bergomi's posterior for a smile whose variance
    curve is almost flat, drawn with the network of slot 5. In both panels the dashed ellipse is the Gaussian with the
    same mean and covariance as the draws.
    """)
    return


@app.cell
def _(RESULTS, backbones, np, teaching):
    heston_cache = backbones.load_example("heston-final500")[2]
    rough_task, rough_report, rough_cache = backbones.load_example("rough-joint100k-validation1962")
    rough_names = rough_task["parameter_names"]
    heston_pair = heston_cache["reference_chains"].reshape(-1, 4)[:, 2:]  # volatility of variance, correlation
    rough_pair = rough_cache["neural_draws"][:, [rough_names.index("long_variance"), rough_names.index("decay")]]
    decay_bounds = np.asarray(rough_task["prior_bounds"])[rough_names.index("decay")]
    heston_s6 = teaching.load_json(RESULTS / "heston" / "s6" / "classical.json")["summary"]


    def gaussian_twin(draws, size=100_000, seed=0):
        """Draws from the Gaussian with the same mean and covariance as `draws`."""
        return np.random.default_rng(seed).multivariate_normal(draws.mean(axis=0), np.cov(draws.T), size)


    def share_inside_90(draws):
        """Share of `draws` inside the 90% ellipse of that Gaussian."""
        centred = draws - draws.mean(axis=0)
        distance = np.einsum("ni,ij,nj->n", centred, np.linalg.inv(np.cov(draws.T)), centred)
        return float(np.mean(distance < 4.605))


    return (
        decay_bounds,
        gaussian_twin,
        heston_pair,
        heston_s6,
        rough_cache,
        rough_names,
        rough_pair,
        rough_report,
        rough_task,
        share_inside_90,
    )


@app.cell(hide_code=True)
def _(
    decay_bounds,
    gaussian_twin,
    heston_pair,
    heston_s6,
    mo,
    np,
    plots,
    plt,
    rough_cache,
    rough_names,
    rough_pair,
    rough_report,
    rough_task,
    share_inside_90,
):
    def _ellipse(draws):
        _values, _vectors = np.linalg.eigh(np.cov(draws.T))
        _angle = np.linspace(0, 2 * np.pi, 200)
        _circle = np.vstack([np.cos(_angle), np.sin(_angle)])
        return draws.mean(axis=0)[:, None] + _vectors @ (np.sqrt(4.605 * _values)[:, None] * _circle)


    _box = np.asarray(rough_task["prior_bounds"])[[rough_names.index("long_variance"), rough_names.index("decay")]]
    _twin = gaussian_twin(rough_pair)
    _outside = float(np.mean(np.any((_twin < _box[:, 0]) | (_twin > _box[:, 1]), axis=1)))
    _decay90 = np.quantile(rough_pair[:, 1], [0.05, 0.95])
    _truth = dict(zip(rough_names, rough_cache["truth"]))
    _shown = heston_pair[np.random.default_rng(0).choice(len(heston_pair), 3_000, replace=False)]
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(9.6, 3.6), constrained_layout=True)
    _left.scatter(_shown[:, 0], _shown[:, 1], s=3, alpha=0.3, color=plots.SERIES[0], rasterized=True)
    _left.plot(*_ellipse(heston_pair), color=plots.INK, linestyle="--", linewidth=1.2, label="Gaussian, 90%")
    _left.set_xlabel("volatility of variance")
    _left.set_ylabel("correlation ρ")
    _left.set_title("Heston, exact posterior: an ellipse", loc="left")
    _left.legend(fontsize=8, loc="lower right")
    _e = _ellipse(rough_pair)
    _low, _high = min(_e[1].min(), _box[1, 0]) - 0.4, max(_e[1].max(), _box[1, 1]) + 0.4
    _right.axhspan(_low, _box[1, 0], color=plots.GRID, linewidth=0)
    _right.axhspan(_box[1, 1], _high, color=plots.GRID, linewidth=0, label="outside the prior")
    _right.scatter(rough_pair[:, 0], rough_pair[:, 1], s=5, alpha=0.45, color=plots.SERIES[1], rasterized=True)
    _right.plot(*_e, color=plots.INK, linestyle="--", linewidth=1.2, label="Gaussian, 90%")
    _right.set_ylim(_low, _high)
    _right.set_xlabel("long level of the variance curve")
    _right.set_ylabel("decay (per year)")
    _right.set_title("Rough Bergomi, almost flat curve: not an ellipse", loc="left")
    _right.legend(fontsize=8, loc="center left")
    _gn = list(heston_s6["gauss_newton_coverage_90"].values())
    _post = list(heston_s6["posterior_coverage_90"].values())
    mo.vstack([
        _fig,
        mo.md(f"""
    In the Heston case, {share_inside_90(heston_pair):.0%} of the exact draws fall inside the Gaussian's 90%
    ellipse, as they would for a Gaussian. With 63 informative quotes the least-squares interval is a good
    approximation of the posterior. Over {heston_s6['surfaces']} synthetic Heston smiles, both kinds of 90% interval
    contain the true parameters about as often as they should: the Gauss–Newton intervals in {min(_gn):.0%} to
    {max(_gn):.0%} of cases and the posterior intervals in {min(_post):.0%} to {max(_post):.0%} (claim C11). For
    {heston_s6['gauss_newton_singular']} of these smiles no Gauss–Newton interval could be computed.

    In the rough Bergomi case the variance curve is almost flat, with short and long levels of
    {_truth['short_variance']:.3f} and {_truth['long_variance']:.3f}. How fast the curve moves from one level to the
    other, the decay, then has almost no effect on prices, and the quotes cannot determine it. Its 90% posterior
    interval runs from {_decay90[0]:.1f} to {_decay90[1]:.1f} per year and covers
    {np.ptp(_decay90) / np.ptp(decay_bounds):.0%} of the prior range. A Gaussian with the same mean and covariance
    would put {_outside:.0%} of its probability outside the prior box, on values the model does not allow. In this
    case the result depends on the prior, so the prior has to be stated.

    Status: the Heston example is a development case checked against PyMC. The rough Bergomi panel shows network
    draws, without an exact reference yet (`{rough_report['case_id']}`).
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A posterior is not needed for every volatility problem. On a liquid smile with a well-identified model, the
    least-squares interval is enough, as in the Heston panel. The posterior adds most in these situations:

    - the quotes leave some direction in parameter space undetermined, as in the rough Bergomi panel, and a product
      depends on it;
    - a range for a mark has to be reported at a stated confidence level (section 4);
    - prior knowledge should enter the calibration explicitly, as a prior, instead of through a penalty or a
      parameter fixed by hand;
    - the results have to be tested against the assumptions, by changing the prior or the noise model and
      recomputing the ranges (slot 6).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · What it costs, and amortized inference

    Computing a posterior becomes more expensive as the model becomes more realistic. The table lists four ways of
    doing it.

    | Method | Needs | In this tutorial |
    |:---|:---|:---|
    | A grid | one or two parameters | Black–Scholes, sections 3 to 5 |
    | A Gaussian approximation | a posterior close to an ellipse | Heston on a liquid smile, section 6 |
    | MCMC | thousands of likelihood evaluations | Heston's exact posterior: minutes. Rough Bergomi: at least 17 CPU-hours by a random-walk chain, or about 85 CPU-minutes by importance sampling from a Gauss–Newton proposal on SPY's 63 quotes |
    | Amortized inference | one training run on simulated smiles | seconds per smile, for any smile the prior can produce |

    The likelihood can be written down for every model here. The cost comes from the pricing inside it, which takes
    about 20 ms per smile for Heston's Fourier integral at one quadrature rung and 5 CPU-seconds per smile for rough
    Bergomi's Monte Carlo.
    """)
    return


@app.cell(hide_code=True)
def _(ASSETS, RESULTS, mo, teaching):
    costs = teaching.load_json(RESULTS / "costs" / "costs.json")
    mo.vstack([
        mo.image(src=ASSETS / "images" / "ladder.png", width=760,
                 caption="One likelihood evaluation and one exact posterior under each model, CPU-seconds on a log "
                         "scale (Heston at one quadrature rung; the rough-Bergomi posterior by random-walk Metropolis "
                         "is a lower bound)."),
        mo.md(f"The measured costs, on one {costs['cpu_count']}-core machine (study S9, claim C8):"),
        mo.ui.table(teaching.cost_rows(costs), selection=None, pagination=False),
    ])
    return (costs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A desk needs many posteriors: one for every underlying in a valuation run, for example at month-end, and
    thousands of simulated smiles to test a method. The slider sets the number of smiles in one run.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    smiles_per_run = mo.ui.slider(1, 2000, step=1, value=500, label="smiles in one valuation run", show_value=True)
    smiles_per_run
    return (smiles_per_run,)


@app.cell
def _(costs, smiles_per_run):
    _n = smiles_per_run.value
    _evaluation = costs["rough_likelihood_evaluation_cpu_s"]["median"]
    valuation_run = {  # CPU-hours for one posterior per smile, by route
        "Heston, exact posterior by MCMC": _n * costs["exact_heston_posterior"]["cpu_seconds"] / 3600,
        "rough Bergomi, exact posterior by MCMC (lower bound)": _n * costs["exact_rough_posterior"]["cpu_hours"],
        "rough Bergomi, importance sampling (1,024 evaluations)": _n * 1024 * _evaluation / 3600,
        "rough Bergomi, amortized network": _n * costs["rough_amortized_posterior_1000_draws_cpu_s"]["median"] / 3600,
    }
    return (valuation_run,)


@app.cell(hide_code=True)
def _(mo, np, plots, plt, smiles_per_run, valuation_run):
    _names, _hours = list(valuation_run), np.array(list(valuation_run.values()))
    _fig, _ax = plt.subplots(figsize=(9.6, 2.5), constrained_layout=True)
    _ax.barh(range(len(_hours)), _hours, height=0.6,
             color=[plots.SERIES[0], plots.SERIES[1], plots.SERIES[1], plots.SERIES[2]])
    for _row, _value in enumerate(_hours):
        _ax.text(_value * 1.2, _row, f"{_value:,.1f} CPU-hours", va="center", fontsize=7.5, color=plots.INK_2)
    _ax.set_yticks(range(len(_names)), _names)
    _ax.invert_yaxis()
    _ax.set_xscale("log")
    _ax.set_xlim(0.3 * _hours.min(), 300 * _hours.max())
    _ax.set_xlabel("CPU-hours for one valuation run (log scale)")
    _ax.set_title(f"the cost of {smiles_per_run.value:,} posteriors, one per smile", loc="left")
    _ax.grid(axis="y", visible=False)


    def _hours_text(value):
        return f"{value:,.0f}" if value >= 10 else f"{value:.2g}"


    mo.vstack([
        _fig,
        mo.md(f"""
    The exact routes grow with the number of smiles. For {smiles_per_run.value:,} smiles, the exact rough-Bergomi
    posterior by MCMC needs at least {_hours_text(_hours[1])} CPU-hours, or {_hours_text(_hours[1] / 4)} cores for a
    four-hour run. Importance sampling from a Gauss–Newton proposal needs about {_hours_text(_hours[2])} where that
    proposal works, Heston's chain about {_hours_text(_hours[0])}, and a trained network {_hours_text(_hours[3])}.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Amortized inference

    Amortized inference pays the cost of the posterior once. We draw parameters from the prior, simulate a smile for
    each, and train a neural network on these pairs to return the posterior for any smile. After that, a new smile
    takes one pass through the network: about 1.4 seconds for 1,000 posterior draws. The rest of the afternoon
    builds these networks and checks them before relying on them; slot 2 explains how they work and when they pay
    off.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid("""
    flowchart LR
      P["prior: plausible<br/>parameter values"] --> S["simulator: price a smile,<br/>add quote noise"]
      S --> B["a bank of<br/>(parameters, smile) pairs"]
      B --> N["train a network<br/>once"]
      Q["today's smile"] --> N
      N --> D["posterior draws<br/>in seconds"]
      D --> C["checks: exact posterior where possible,<br/>calibration on simulated smiles,<br/>an audit of each smile"]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The objects of the afternoon

    | Object | Here | Where it returns |
    |:---|:---|:---|
    | Quotes $y$ | 63 prices of one smile, each with its noise | every slot |
    | Model | a pricer plus quote noise; run forward, it simulates smiles | slot 2 trains on its simulations; slots 4 and 5 |
    | Parameters $\theta$ | $\sigma$; Heston's four; rough Bergomi's six | slots 4 and 5 |
    | Prior $p(\theta)$ | a box of plausible values | slot 6 changes it |
    | Posterior $p(\theta \mid y)$ | every $\theta$ the quotes allow | slot 2 learns it; slots 4 and 5 compute it |
    | A product's range | the product priced under each posterior draw | slot 5 |
    | Checks | predictive checks, coverage, SBC, an audit of each smile | slots 2, 4, 5 and 6 |
    | Network $q(\theta \mid y)$ | a posterior in seconds, after one training run | slots 2 to 5 |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    1. A calibration is an estimate. Least squares keeps the peak of the likelihood; the posterior keeps all of it,
       every parameter value consistent with the quotes and their noise. Pricing a product under the posterior gives
       a range for its price.
    2. A posterior is only as good as its model. On SPY, Black–Scholes gives a precise but wrong answer, so the model
       has to be checked before its ranges are used.
    3. With many informative quotes, the posterior is close to the least-squares result. When the quotes leave a
       parameter undetermined, the posterior is no longer an ellipse and the prior matters.
    4. Exact posteriors become expensive as models become more realistic, and many of them are needed. The rest of
       the afternoon shows how amortized inference reduces this cost.

    Slot 2 shows how a network trained on simulations returns the posterior, starting from the quote of section 3,
    and when this pays off.

    Sources: `CLAIMS.md` (claims C8 and C11), the cached examples in `assets/backbones/` (development examples, not
    accepted studies) and `src/sbi4f_tutorial/`. The SPY quotes come from study S8. Results on simulated smiles use
    the tutorial's models and priors, and do not show how well these models describe market data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The data in detail

    The argument above needs only the 63 prices and their noise. The later slots rely on more details of the data,
    collected here: how the grid is read from the chain, what a network reads, the variance curve and the skew, and
    how simulated smiles are produced.
    """)
    return


@app.cell(hide_code=True)
def _(mo, spy):
    _rates = spy["rate_curve"]["points_bond_equivalent"]
    _readings = spy["diagnostics"]["grid_point_readings"]
    _extrapolated = ("none is extrapolated" if _readings["extrapolated"] == 0
                     else f"{_readings['extrapolated']} are extrapolated")
    mo.md(f"""
    ### From the chain to the grid

    The 63 numbers are produced from the chain in five steps.

    1. Clean the quotes. We keep out-of-the-money options with a positive bid, an ask above the bid and a spread of
       at most half the mid. These are puts with strikes below the forward and calls with strikes above it.
    2. Find each expiration's discount factor and forward. The discount factor comes from US Treasury yields,
       {100 * _rates[0][1]:.2f}% at three months and {100 * _rates[1][1]:.2f}% at two years. The forward comes from
       put–call parity on the six strike pairs nearest the money.
    3. Divide each mid price by the discount factor and the forward, and convert it to an implied volatility.
    4. Read the grid. A smoothing spline runs through the implied volatilities of each expiration. Each grid point is
       read at its log-moneyness from the nearest expirations on either side whose quotes reach it, and the two
       readings are interpolated linearly in total implied variance, σ²T. Of the 63 points,
       {_readings['two expirations']} use two expirations and {_readings['one expiration']} use one; {_extrapolated}.
    5. Convert the 63 implied volatilities back into prices in units of the forward. These prices, with their
       noise, form the observation y.

    Reading every smile onto the same grid gives real and simulated smiles the same 63 inputs, so one network can
    process any of them. Slot 3 discusses set networks, which could read a chain's quotes directly and so remove the grid.
    """)
    return


@app.cell(hide_code=True)
def _(mo, same_design):
    mo.md(rf"""
    | One observation $y$ | |
    |:---|:---|
    | Contracts | 63, ordered by maturity and then by strike |
    | Maturities | 5, 10, 21, 42, 63, 126 and 252 trading days, from one week to one year; in the formulas $T$ = days / 252 |
    | Strikes | 9 per maturity, at $\delta = -2, -1.5, \dots, 2$: a put for $\delta \le 0$ and a call above |
    | Values | prices divided by the discount factor and the forward, so the models are priced with a forward of 1 and no rates or dividends |
    | Noise | one standard deviation per price, from the rule of section 2 |
    | Used by | slot 2's toy network, which reads one of these quotes, and the networks of slots 4 and 5, which read all 63 ({'the same contracts and noise, checked live' if same_design else 'a different grid: check the task files'}) |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What a network reads

    The noise rule of section 2 is fixed rather than read from each day's half-spreads, so real and simulated smiles
    carry exactly the same noise, and the networks of the later slots rely on that. For inference, what matters is
    each price relative to its noise. The networks read $y_c / \sigma_c$, the price in noise units. During training,
    each simulated smile gets a new draw of the noise every time it is used. Slot 6 changes the noise level to see
    how much the conclusions depend on it.
    """)
    return


@app.cell
def _(np, spy_noise, spy_prices, surface):
    noise_vol_points = 100 * surface.implied_vol_noise(np.array(surface.DESIGN_DELTAS))  # the rule, in volatility points
    noise_units = (spy_prices / spy_noise).reshape(len(surface.DESIGN_DAYS), len(surface.DESIGN_DELTAS))
    return noise_units, noise_vol_points


@app.cell(hide_code=True)
def _(
    mo,
    noise_units,
    noise_vol_points,
    np,
    plots,
    plt,
    spy_iv,
    surface,
    teaching,
):
    _deltas = np.array(surface.DESIGN_DELTAS)
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(9.6, 3.6), constrained_layout=True,
                                         gridspec_kw={"width_ratios": [1, 1.7]})
    _line = np.linspace(-2, 2, 81)
    _left.plot(_line, 100 * surface.implied_vol_noise(_line), color=plots.INK_2, linewidth=1.4)
    _left.scatter(_deltas, noise_vol_points, s=16, color=plots.SERIES[0], zorder=3)
    _left.set_ylim(0, 1.2)
    _left.set_xlabel("standardized moneyness δ")
    _left.set_ylabel("volatility points")
    _left.set_title("the noise rule, one standard deviation", loc="left")
    _right.imshow(np.log10(noise_units), cmap="viridis", aspect="auto")
    for (_row, _col), _value in np.ndenumerate(noise_units):
        _right.text(_col, _row, f"{_value:.1f}" if _value < 10 else f"{_value:.0f}", ha="center", va="center",
                    fontsize=7, color="white" if np.log10(_value) < 0.6 else plots.INK)
    _right.set_xticks(range(len(_deltas)), [f"{_x:g}" for _x in _deltas])
    _right.set_yticks(range(len(surface.DESIGN_DAYS)), [teaching.TENORS[_d] for _d in surface.DESIGN_DAYS])
    _right.set_xlabel("standardized moneyness δ")
    _right.set_title("SPY's prices in noise units, y / σ: what a network reads", loc="left")
    _right.grid(visible=False)
    _low = np.unravel_index(np.argmin(noise_units), noise_units.shape)
    _high = np.unravel_index(np.argmax(noise_units), noise_units.shape)
    _faint = noise_units.max(axis=0) < 1  # strikes below one noise standard deviation at every maturity
    _faint_iv = 100 * np.asarray(spy_iv).reshape(noise_units.shape)[:, _faint]
    _faint_deltas = " and ".join(f"{_x:g}" for _x in _deltas[_faint])
    mo.vstack([
        _fig,
        mo.md(f"""
    The left panel shows the rule. The right panel shows SPY's 63 prices divided by their noise. The option at
    δ = {_deltas[_low[1]]:g} with {teaching.TENORS[surface.DESIGN_DAYS[_low[0]]]} to expiry is worth only
    {noise_units[_low]:.1f} noise standard deviations, and the option at δ = {_deltas[_high[1]]:g} with
    {teaching.TENORS[surface.DESIGN_DAYS[_high[0]]]} to expiry is worth {noise_units[_high]:.0f}. The calls at
    δ = {_faint_deltas} are worth less than one noise standard deviation at every maturity. SPY's implied volatility
    there, {_faint_iv.min():.0f} to {_faint_iv.max():.0f}%, is well below the 20% at which the rule is set, so their
    prices are small compared with the noise. On their own, these quotes say little about the parameters.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The variance curve and the skew

    The same chain gives two more quantities that the later slots use.

    The first is the market's variance-swap curve. For each expiration whose quotes reach far enough into the wings,
    down to $\delta = -8$ for the puts and up to $\delta = 1.8$ for the calls, a log contract is replicated from the
    whole strip of out-of-the-money options. Its price is the total variance that a variance swap to that expiration
    pays. Read at the grid maturities and differenced, these values give seven forward-variance levels, one for each
    interval between grid maturities. Rough Bergomi takes this forward-variance curve as an input. In slot 5 the
    curve is either given, with these seven levels, or inferred together with the other parameters. When it is
    given, the network reads the seven levels alongside the 63 prices; the levels vary from one training smile to
    the next, so the same network works with any curve. In slot 4, Heston's variance swaps are compared with these
    market values.

    The second is the at-the-money skew, the slope of the smile at the money. On SPY it falls with maturity roughly
    like a power of the maturity. Heston cannot follow this: its skew tends to a constant at short maturities.
    Rough Bergomi's skew falls like a power of the maturity, which is one reason slot 5 uses it.
    """)
    return


@app.cell
def _(np, spy, surface):
    _knots = np.r_[0, surface.DESIGN_DAYS] / surface.TRADING_DAYS
    forward_variance = np.asarray(spy["forward_variance_levels"])  # one level per interval between grid maturities
    variance_swap_vol = 100 * np.sqrt(np.cumsum(forward_variance * np.diff(_knots)) / _knots[1:])
    atm_skew = np.abs(np.asarray(spy["atm_skew"]))  # |d sigma / d k| at the money, one per maturity
    skew_law = spy["skew_power_law"]  # |skew| = scale * T ** exponent, fitted from one week to one year
    return atm_skew, forward_variance, skew_law, variance_swap_vol


@app.cell(hide_code=True)
def _(
    atm_skew,
    forward_variance,
    mo,
    np,
    plots,
    plt,
    skew_law,
    spy_expirations,
    surface,
    variance_swap_vol,
):
    _days = np.array(surface.DESIGN_DAYS)
    _exponent = f"{skew_law['exponent']:.2f}".replace("-", "−")
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(9.6, 3.4), constrained_layout=True)
    _left.stairs(100 * np.sqrt(forward_variance), np.r_[0, _days], baseline=None, color=plots.SERIES[0], linewidth=1.6,
                 label="forward volatility on each interval")
    _left.set_ylim(0, 26)
    _left.plot(_days, variance_swap_vol, "o", color=plots.SERIES[1], markersize=5, label="variance swap to each maturity")
    _left.set_xlabel("maturity, trading days")
    _left.set_ylabel("volatility (%)")
    _left.set_title("SPY's forward-variance curve and variance swaps", loc="left")
    _left.legend(fontsize=7.5, loc="lower right")
    _t = _days / surface.TRADING_DAYS
    _fine = np.geomspace(_t[0], _t[-1], 50)
    _right.plot(_days, atm_skew, "o", color=plots.SERIES[0], label="SPY")
    _right.plot(_fine * surface.TRADING_DAYS, skew_law["scale"] * _fine ** skew_law["exponent"], color=plots.INK_2,
                linewidth=1.2, label=f"power law, exponent {_exponent}")
    _right.set_xscale("log")
    _right.set_yscale("log")
    _right.minorticks_off()
    _right.set_xticks(list(_days), [str(_d) for _d in _days])
    _right.set_yticks([0.25, 0.4, 0.6, 0.85], ["0.25", "0.4", "0.6", "0.85"])
    _right.set_xlabel("maturity, trading days")
    _right.set_ylabel("|∂σ/∂k| at the money")
    _right.set_title("the at-the-money skew falls like a power of maturity", loc="left")
    _right.legend(fontsize=7.5, loc="upper right")
    _covered = sum(_e["replication_covered"] for _e in spy_expirations)
    mo.vstack([
        _fig,
        mo.md(f"""
    On the left, the steps are the forward volatility on each interval between grid maturities, the square root of the
    forward variance. The dots are the variance-swap volatility to each maturity, the square root of the average
    forward variance up to it. They come from the {_covered} expirations, out of {len(spy_expirations)}, whose quotes
    reach far enough into the wings. On the right is the slope of the smile at the money for each maturity, with a
    power law fitted from one week to one year. Its exponent is {_exponent} (claim C1).
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simulated smiles and the task files

    The real smile is a single observation. Most of what follows also needs simulated smiles on the same grid: draw
    parameters from the prior, price the 63 options with the model, and add noise from the rule of section 2, as section 3 did for single quotes. The true
    parameters of a simulated smile are known, so simulated smiles are used to train the networks and to test any
    method.

    For each model, one file, the task file, fixes everything a simulation needs: the 63 contracts and their noise,
    the prior box of the parameters, the parameters held fixed (Heston's mean reversion, $\kappa = 2$), rough
    Bergomi's variance curve or its prior, and the accuracy required of the pricer. Simulated prices have to be
    accurate relative to the noise. Heston's Fourier prices are refined until they change by less than 0.1 noise
    standard deviations. Rough Bergomi's Monte Carlo uses between 32,768 and about 2 million paths, enough to bring
    the Monte Carlo error below 0.25 noise standard deviations (0.5 at the largest number of paths). The same file
    is used to simulate, to train and to compute posteriors, and its hash identifies it.

    | Data set | What it is | Where it is used |
    |:---|:---|:---|
    | SPY's 63 prices | the real smile of 21 October 2026 | slot 1 (Black–Scholes), slot 4 (Heston), slot 5 (rough Bergomi, products and reserves) |
    | SPY's forward-variance curve | seven levels from the same chain | slot 4 (Heston's variance swaps against the market), slot 5 (rough Bergomi's input curve) |
    | Training banks | for each model, 100,000 simulated smiles for training, 5,000 for validation, and 5,000 (Heston) or 7,000 (rough Bergomi) held back for tests; slot 4's SPY network comes from an earlier Heston bank of 50,000 | slots 3 to 5 |
    | Tests with known truth | simulated smiles whose parameters are known, used to check that posteriors are calibrated | slots 2, 4 and 5 |
    | Cached examples | single simulated smiles with their posteriors, stored with the tutorial | slots 1, 4 and 5 |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The formulas

    The panel collects the formulas behind sections 2 to 7.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "**The model in detail** (click to open): the data, the noise, Black–Scholes, Bayes' rule and the three models": mo.md(r"""
    #### 1 · One observation: 63 prices on a fixed grid

    Every smile in this tutorial, real or simulated, is read onto the same grid: 7 maturities (5, 10, 21, 42,
    63, 126 and 252 trading days) times 9 standardized moneyness points $\delta = k / (\sigma_{\text{ref}}\sqrt T)$
    from $-2$ to $2$, with $k = \ln(K/F)$ and $\sigma_{\text{ref}} = 20\%$. Puts sit below the money and calls
    above it. Prices are in forward units: divided by the discount factor and the forward, so the forward is 1
    and rates and dividends drop out. One observation $y$ is these 63 numbers.

    #### 2 · The noise

    Each price is the model price plus independent Gaussian noise:

    $$y_c = P_c(\theta) + \sigma_c\,\varepsilon_c, \qquad \varepsilon_c \sim \mathcal N(0, 1), \qquad c = 1, \dots, 63.$$

    The standard deviation $\sigma_c$ is the Black–Scholes vega at $\sigma_{\text{ref}}$ times a volatility noise
    that rises linearly from 0.3 volatility points at the money to 1 point at $|\delta| = 2$: the design's noise.
    On 21 October 2026 it is wider than SPY's bid–ask at all 63 points. Every network of the afternoon is trained
    with fresh draws of exactly this noise, and reads the prices in noise units, $y_c / \sigma_c$.

    #### 3 · Black–Scholes in forward units

    $$c(k, T; \sigma) = N(d_1) - e^{k} N(d_2), \qquad d_{1,2} = \frac{-k \pm \tfrac12 \sigma^2 T}{\sigma\sqrt T},$$

    where $N$ is the standard normal distribution function. A put follows from put–call parity,
    $p = c - 1 + e^{k}$. The **implied volatility** of a quote is the $\sigma$ that reproduces it. Black–Scholes
    prices all 63 options from one number, so its implied volatilities form a flat smile.

    #### 4 · Bayes' rule

    With a prior $p(\theta)$, the posterior given the quotes is

    $$p(\theta \mid y) = \frac{p(y \mid \theta)\,p(\theta)}{p(y)}, \qquad p(y \mid \theta) = \prod_{c=1}^{63} \frac{1}{\sqrt{2\pi}\,\sigma_c} \exp\Big(-\frac{(y_c - P_c(\theta))^2}{2\sigma_c^2}\Big).$$

    The evidence $p(y)$ does not depend on $\theta$. On a grid of $\theta$ values it is the sum that makes the
    posterior integrate to one: that is how section 3 computes it, on 4,001 volatilities.

    #### 5 · Propagation

    A product's price $f(\theta)$ inherits the posterior. Draw $\theta$ from $p(\theta \mid y)$, price the product
    under each draw, and read quantiles off the prices. The 90% range runs from the 5% to the 95% quantile; a
    one-sided 90% bound uses one tail.

    #### 6 · Least squares and its Gaussian approximation

    Least squares minimizes $\chi^2(\theta) = \sum_c (y_c - P_c(\theta))^2 / \sigma_c^2$, which is
    $-2 \log p(y \mid \theta)$ up to a constant. Expanding it to second order at the optimum $\hat\theta$, with
    $J = \partial P / \partial \theta$ and $W = \mathrm{diag}(\sigma_c^{-2})$, gives the Gauss–Newton approximation

    $$p(\theta \mid y) \approx \mathcal N\big(\hat\theta,\ (J^\top W J)^{-1}\big),$$

    valid under a flat prior away from its bounds. With many informative quotes the likelihood is close to
    Gaussian in the parameters and the two agree (the Bernstein–von Mises theorem). They part when it is not,
    or when the prior's bounds cut it.

    #### 7 · The three models of the afternoon

    | Model | Unknowns | Prices by | Slots |
    |:---|:---|:---|:---|
    | Black–Scholes | the volatility $\sigma$ | a closed form | 1, 2 |
    | Heston | initial and long-run variance, volatility of variance, correlation; mean reversion fixed at 2 | a Fourier integral | 3, 4, 6 |
    | Rough Bergomi | roughness $H$, volatility of volatility $\eta$, correlation $\rho$, and the variance curve's short level, long level and decay | Monte Carlo only | 5, 6 |

    Rough Bergomi's variance curve has Heston's shape, $\xi_0(t) = \xi_\ell + (\xi_s - \xi_\ell)\,e^{-\lambda t}$, with
    short level $\xi_s$, long level $\xi_\ell$ and decay $\lambda$. Every prior of the afternoon is uniform on a box of
    plausible values; the boxes are in the task files under `assets/backbones/`.

    #### 8 · What this slot leaves out

    Hierarchical models that pool information across names, calibration from return series, and quote chains
    with missing points. Each extends the same machinery; the afternoon does not need them.
    """),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What the study assumes

    The examples simplify the desk's problem in several ways. These are the main choices.

    | Topic | What we assume | What it leaves out |
    |:---|:---|:---|
    | Data | One underlying and one date: SPY options at the close of 21 October 2026, from Cboe's delayed quotes | Daily recalibration across many underlyings |
    | Quotes | Out-of-the-money options with a positive bid, an ask above the bid, and a spread of at most half the mid | In-the-money options and the desk's own filters |
    | Exercise | SPY options are American; we price them as European | The early-exercise premium of the puts |
    | Rates and forwards | Discount factors from US Treasury yields; each expiration's forward from put–call parity near the money | The desk's funding curve and dividend forecasts |
    | Time | Weekdays to expiration, over 252 | Holidays and intraday time |
    | Grid | 63 values read from a smoothed smile of the neighbouring expirations, interpolated in total variance | The grid is a reading of the quotes, not the quotes themselves |
    | Noise | Independent Gaussian errors: 0.3 volatility points at the money, rising to 1 point at $\delta = \pm 2$, wider than SPY's bid–ask that day | Correlated quote errors, and spreads that change from day to day |
    | Models | Black–Scholes; Heston with mean reversion fixed at 2; rough Bergomi with a three-parameter variance curve; parameters constant in time | Jumps, and parameters that change over time |
    | Priors | Uniform over a box of plausible values | Other priors: slot 6 changes them |
    | Truth | Some examples use smiles simulated from the models, so the true parameters are known | On market data the truth is unknown |
    """)
    return


if __name__ == "__main__":
    app.run()
