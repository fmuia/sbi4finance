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

    This first part sets up the problem the tutorial works on, introduces the objects used in the next slots, and
    shows where Bayesian and amortized methods help. All the examples use one data set: the SPY option quotes at
    the close of 21 October 2026.

    ## 1 · The problem and the main objects

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
    The quotes change every day, so the calibration is redone every day. Any price computed with the model depends
    on how well the quotes pin down its parameters.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The data

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
def _(mo, same_design):
    mo.md(rf"""
    | One observation $y$ | |
    |:---|:---|
    | Maturities | 5, 10, 21, 42, 63, 126 and 252 trading days, from one week to one year |
    | Strikes | 9 per maturity, from $\delta = -2$ to $2$ |
    | Values | 63 prices, divided by the discount factor and the forward |
    | Noise | one standard deviation per price: a third of a volatility point at the money, one point in the wings |
    | Used by | every slot and every network of the afternoon ({'the same 63 contracts and noise, checked live' if same_design else 'a different grid: check the task files'}) |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Volatility models

    A volatility model describes how the price of SPY and its volatility move over time. Given values for its
    parameters, it gives a price, and so an implied volatility, for every option on the grid. Black–Scholes has a
    single parameter, the volatility $\sigma$, and gives every option the same implied volatility: a flat line. The
    market's smile is not flat. Heston lets the variance move randomly, and against the price, and with four
    parameters it reproduces the shape of SPY's smile. Rough Bergomi, in slot 5, describes short maturities better
    still, but it can only be priced by Monte Carlo.
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
    ### Bayesian calibration

    The usual calibration, least squares, returns one set of parameters: the one whose prices come closest to the
    quotes. Because the quotes are noisy, nearby parameter values reproduce them almost as well. Bayesian
    calibration keeps all of them. Each value gets a weight from two sources: how well it explains the quotes, the
    likelihood, and how plausible it was before seeing them, the prior. The weighted set of values is the posterior
    distribution. Pricing a product under each value gives a distribution of prices, and its spread says how
    precisely the quotes determine that price.
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
    {_range[1]:.2f} volatility points. With 63 liquid quotes the range is narrow. Section 5 shows a case where the
    quotes leave a parameter undetermined.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Amortized inference

    Computing a posterior exactly means pricing the 63 quotes thousands of times. For Heston, priced by a Fourier
    integral, that takes a few CPU-minutes per smile. For rough Bergomi, priced by Monte Carlo, it takes at least 17
    CPU-hours with a random-walk chain, or about 85 CPU-minutes with importance sampling from a good starting
    approximation. A desk needs one posterior per underlying and per day, and testing a method needs thousands.

    Amortized inference pays this cost once. We draw parameters from the prior, simulate a smile for each, and
    train a neural network on these pairs to return the posterior for any smile. After that, a new smile takes one
    pass through the network: about 1.4 seconds for 1,000 posterior draws. Slot 2 explains how this works, and
    section 6 compares the costs.
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
    ### What the study assumes

    The examples simplify the desk's problem in several ways. These are the main choices.

    | Topic | What we assume | What it leaves out |
    |:---|:---|:---|
    | Data | One underlying and one date: SPY options at the close of 21 October 2026, from Cboe's delayed quotes | Daily recalibration across many underlyings |
    | Quotes | Out-of-the-money options with a positive bid, an ask above the bid, and a spread of at most half the mid | In-the-money options and the desk's own filters |
    | Exercise | SPY options are American; we price them as European | The early-exercise premium of the puts |
    | Rates and forwards | Discount factors from US Treasury yields; each expiration's forward from put–call parity near the money | The desk's funding curve and dividend forecasts |
    | Time | Weekdays to expiration, over 252 | Holidays and intraday time |
    | Grid | 63 values read from a smoothed smile of the neighbouring expirations, interpolated in total variance | The grid is a reading of the quotes, not the quotes themselves |
    | Noise | Independent Gaussian errors: a third of a volatility point at the money, one point in the wings, wider than SPY's bid–ask that day | Correlated quote errors, and spreads that change from day to day |
    | Models | Black–Scholes; Heston with mean reversion fixed at 2; rough Bergomi with a three-parameter variance curve; parameters constant in time | Jumps, and parameters that change over time |
    | Priors | Uniform over a box of plausible values | Other priors: slot 6 changes them |
    | Truth | Some examples use smiles simulated from the models, so the true parameters are known | On market data the truth is unknown |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The rest of this notebook

    The next sections work through these objects on the simplest case, Black–Scholes on SPY's quotes, then look
    at when the answer differs from least squares and at what it costs.

    | Section | What it covers |
    |:---|:---|
    | 2 · Bayes on one quote | prior, likelihood and posterior for $\sigma$ |
    | 3 · Using the posterior | the range of another option's price |
    | 4 · Checking the model | the posterior's prediction against a quote it has not seen |
    | 5 · When does Bayes change the answer? | posteriors that a Gaussian approximation misses |
    | 6 · What it costs | exact posteriors against amortized ones |

    The panel below collects the formulas.
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
    posterior integrate to one: that is how section 2 computes it, on 4,001 volatilities.

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
    ## 2 · Bayes on one quote

    **What does one quote say about $\sigma$?** Take the at-the-money one-month price, and combine three things:

    - **the prior** $p(\sigma)$: what we accept before seeing today's quotes. Here every volatility from 5% to 60%
      is equally plausible;
    - **the likelihood** $p(y \mid \sigma)$: how probable this quote is if $\sigma$ were the true volatility, given
      the quote's noise;
    - **the posterior** $p(\sigma \mid y) \propto p(y \mid \sigma)\,p(\sigma)$: the two multiplied and normalized.
      It holds every volatility consistent with the quote, weighted by how consistent it is.

    With one parameter, all three can be computed exactly on a grid of 4,001 volatilities.
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
    Posterior: **σ = {100 * _mean:.2f}% ± {100 * _spread:.2f} volatility points**. With a flat prior the posterior is
    the likelihood, normalized. Its width is the quote's noise divided by its vega: the noise, expressed as a
    volatility. Slot 2 trains a network to return exactly this posterior.
    """
    else:
        _text = f"""
    With this prior the posterior sits at **σ = {100 * _mean:.2f}% ± {100 * _spread:.2f} volatility points**, against
    {100 * one_quote_mean:.2f}% under the flat prior. A prior much wider than the quote's precision hardly moves the
    answer; a prior narrower than it overrides the quote. A penalty that pulls a calibration toward a reference
    value, such as yesterday's parameters, acts as such a prior. Slot 6 varies the priors of the real models.
    """
    mo.vstack([_fig, mo.md(_text)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Using the posterior: the price of another option

    **How sure is a price the model computes?** Draw volatilities from the posterior and price another option
    under each draw. Take the one-month put struck far below the money, at $\delta = -2$, and pretend for a moment
    that the market does not quote it, as for the products a desk must mark.
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
    For the put struck at {100 * quotes['put']['strike']:.0f}% of the forward, the 90% range is
    **{1e4 * put_range[0]:.2f} to {1e4 * put_range[1]:.2f} basis points** of the forward. Nothing more is needed than
    a pricer and the posterior. The same recipe gives the range of any product the model can price, such as an
    option that starts in a year; slot 5 applies it with rough Bergomi.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Why a range?** The rules on valuation ask for the uncertainty of a mark at a stated confidence:

    | Rule | What it asks for |
    |:---|:---|
    | Regulation (EU) 2016/101, Art. 9 (market price uncertainty) | for inputs calibrated to market quotes, a range of plausible values and a point in it at 90% confidence |
    | Regulation (EU) 2016/101, Art. 11 (model risk) | the same for the choice among models and calibration approaches |
    | IFRS 13, §93(h) | the sensitivity of a Level 3 fair value to reasonably possible alternative inputs |
    | PRA SS1/23 | model uncertainty: "the inherent uncertainty in the parameter estimates and results of statistical models" |

    A posterior range answers the first row within one model. It is an input to a valuation adjustment, not a
    complete procedure by itself.
    """)
    return


@app.cell(hide_code=True)
def _(mo, put_range, quotes):
    mo.md(f"""
    ## 4 · Checking the model

    **Does the model reproduce data it has not seen?** The market does quote that put, at
    **{1e4 * quotes['put']['price']:.1f} basis points**: about {quotes['put']['price'] / put_range[1]:.0f} times the top
    of the range the posterior predicted. Now use both quotes as data. If Black–Scholes were right, one
    volatility would fit both.

    Before moving the slider, predict: how much noisier would both quotes have to be for one volatility to fit them?
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
    The two likelihoods do not overlap. The posterior from both quotes stays at the at-the-money value, whose
    quote is the more precise, and misses the put by **{misses['put']:.1f} noise units**. It takes about
    {noise_needed:.2g} times the assumed noise to bring both misses under two. That misfit is **the smile**: one
    volatility cannot price both options. The posterior was precise and wrong, because **a posterior is
    conditional on its model**: check the model before trusting its ranges. The afternoon therefore moves to
    models that make a smile: Heston, with four parameters (slot 4), and rough Bergomi, with six (slot 5).
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · When does Bayes change the answer?

    **With a model that fits, is the posterior just least squares with error bars?** Often, yes. The classical
    route fits once and reads an ellipse off the curvature at the optimum. The Bayesian route keeps the whole
    distribution:

    | | Least squares | Bayesian calibration |
    |:---|:---|:---|
    | Output | one parameter vector $\hat\theta$ | a distribution $p(\theta \mid y)$ |
    | Uncertainty | the Gauss–Newton ellipse $(J^\top W J)^{-1}$ around $\hat\theta$ | the posterior, whatever its shape |
    | What is known beforehand | penalties, parameters fixed by hand | a stated prior |
    | The range of a product | the delta method, through the product's gradient | the product priced under each posterior draw |
    | What "90%" means | across repeated sets of quotes | given today's quotes and the prior |
    | Cost | one optimization | thousands of likelihood evaluations |

    Two synthetic smiles, on SPY's grid and noise, show both cases. On the left, Heston's exact posterior, sampled
    by MCMC in PyMC. On the right, rough Bergomi's posterior for a smile whose variance curve is almost flat,
    drawn by slot 5's network. The dashed ellipse is the Gaussian with the same mean and covariance as the draws.
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
    **Left, an ellipse.** Of Heston's exact draws, {share_inside_90(heston_pair):.0%} fall inside the Gaussian's 90%
    ellipse, as they would for a Gaussian. With 63 informative quotes, the classical interval is the posterior's
    Gaussian limit. Over {heston_s6['surfaces']} synthetic Heston smiles, both kinds of 90% interval cover the truth
    at about the stated rate: Gauss–Newton's in {min(_gn):.0%} to {max(_gn):.0%} of cases, the posterior's in
    {min(_post):.0%} to {max(_post):.0%} (claim C11), although no Gauss–Newton interval exists on
    {heston_s6['gauss_newton_singular']} of them. On a liquid smile, the classical toolkit is right.

    **Right, not an ellipse.** This smile's variance curve is almost flat: its short and long levels are
    {_truth['short_variance']:.3f} and {_truth['long_variance']:.3f}. How fast the curve moves from one to the
    other, the decay, then barely changes a price, so the quotes cannot pin it. Its 90% posterior interval, from
    {_decay90[0]:.1f} to {_decay90[1]:.1f} a year, spans {np.ptp(_decay90) / np.ptp(decay_bounds):.0%} of the
    prior's range. The Gaussian with the same mean and covariance puts {_outside:.0%} of its mass outside the prior
    box, on values the model excludes. **Where the quotes say nothing, the prior decides**, which is why it must
    be stated.

    *Status:* the left panel is a development example checked against PyMC. The right panel shows network draws,
    with no exact reference yet (`{rough_report['case_id']}`).
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **So when is a posterior worth having?** Not on every volatility problem: on a liquid smile with a
    well-identified model, the classical interval does the job, as on the left. A posterior earns its place when

    - the quotes leave a direction open, as on the right, and a product depends on it;
    - a range for a mark must be reported at a stated confidence (section 3);
    - prior knowledge should enter openly, as a stated prior, rather than as a penalty or a parameter fixed by hand;
    - the answer must be tested against its assumptions: change the prior or the noise and re-read the ranges
      (slot 6).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · What it costs, and the way around

    **Why not compute the posterior for every smile?** Because computing it gets expensive as the model gets
    realistic. There are four ways, each a rung up:

    | Method | Needs | In this tutorial |
    |:---|:---|:---|
    | A grid | one or two parameters | Black–Scholes, sections 2 to 4 |
    | A Gaussian approximation | a posterior close to an ellipse | Heston on a liquid smile, section 5 |
    | MCMC | thousands of likelihood evaluations | Heston's exact posterior: minutes. Rough Bergomi: at least 17 CPU-hours by a random-walk chain, or about 85 CPU-minutes by importance sampling from a Gauss–Newton proposal on SPY's 63 quotes |
    | Amortized inference | one training run on simulated smiles | seconds per smile, for any smile the prior can produce |

    The likelihood exists for every model here. What costs is the pricing inside it: about 20 ms per smile for
    Heston's Fourier integral at one quadrature rung, and 5 CPU-seconds per smile for rough Bergomi's Monte Carlo.
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
    smiles_per_run = mo.ui.slider(1, 2000, step=1, value=500, label="smiles in one valuation run", show_value=True)
    smiles_per_run
    return (smiles_per_run,)


@app.cell
def _(costs, mo, smiles_per_run):
    _n = smiles_per_run.value
    _evaluation = costs["rough_likelihood_evaluation_cpu_s"]["median"]
    valuation_run = [
        {"route": "Heston, exact posterior (one quadrature rung)",
         "CPU-hours": _n * costs["exact_heston_posterior"]["cpu_seconds"] / 3600},
        {"route": "Rough Bergomi, exact posterior by random-walk Metropolis (lower bound)",
         "CPU-hours": _n * costs["exact_rough_posterior"]["cpu_hours"]},
        {"route": "Rough Bergomi, importance sampling: 1,024 evaluations, where a Gauss–Newton proposal works",
         "CPU-hours": _n * 1024 * _evaluation / 3600},
        {"route": "Rough Bergomi, amortized network",
         "CPU-hours": _n * costs["rough_amortized_posterior_1000_draws_cpu_s"]["median"] / 3600},
    ]
    for _row in valuation_run:
        _row["cores for a four-hour run"] = round(_row["CPU-hours"] / 4, 1)
        _row["CPU-hours"] = round(_row["CPU-hours"], 1)
    mo.ui.table(valuation_run, selection=None, pagination=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Many posteriors are the common case: one for every name in a valuation run, for example at month-end, and
    thousands on simulated smiles to test a method before it is used. The amortized route of section 1 replaces
    each of them by one pass through a trained network. The training is paid once, and the rest of the afternoon
    checks the network before trusting it.
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

    1. **A calibration is an estimate.** The posterior is every parameter value consistent with the quotes at
       their precision, and pricing under it gives the range of any mark.
    2. **A posterior is conditional on its model.** Black–Scholes is precise and wrong on SPY; check the model
       before trusting its ranges.
    3. **Often the classical interval is the posterior's Gaussian limit, but not always.** Where the quotes leave
       a direction open, the posterior is not an ellipse, and the stated prior decides.
    4. **Exact posteriors get expensive as models get realistic, and many are needed**: every name in a valuation
       run, every smile of a validation. The rest of the afternoon amortizes them.

    **Next, slot 2:** how a network trained on simulations returns the posterior, starting with this slot's
    quote, and when that pays.

    *Evidence and code:* `CLAIMS.md` (claims C8 and C11), the cached examples in `assets/backbones/` (development
    examples, not accepted studies) and `src/sbi4f_tutorial/`. The SPY quotes come from study S8. Simulated
    results use the tutorial's models and priors; they do not certify the models on market data.
    """)
    return


if __name__ == "__main__":
    app.run()
