import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="Slot 1 · Bayes for volatility models",
)


@app.cell(hide_code=True)
def _():
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm

    from sbi4finance.calibration import plots, surface, teaching, why_bayes

    ROOT = mo.notebook_dir().parent
    RESULTS = ROOT / "results" / "calibration"
    CHECKS = ROOT / "checks"
    COURSE = ROOT.parent / "docs" / "course"
    sys.path.insert(0, str(CHECKS))
    import bs_hierarchy  # noqa: E402  (the sector case: tutorial_final/checks/bs_hierarchy.py)

    plots.style()
    return (
        CHECKS,
        COURSE,
        RESULTS,
        ROOT,
        bs_hierarchy,
        mo,
        norm,
        np,
        plots,
        plt,
        surface,
        teaching,
        why_bayes,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Slot 1 · Bayes for volatility models: the problem

    **Every mark rests on a calibration. How sure is it, and when does that change the number?**

    A derivatives desk fits a volatility model to the day's option quotes. It then uses the model to price
    what nobody quotes: a forward start, a cliquet, an autocallable, a long-dated put. The quotes are noisy
    and often thin, so the fit is an estimate, and so is every mark computed from it. Banks must report how
    uncertain those marks are, at a stated confidence.

    Three claims, each backed by a computation below:

    1. **Calibration is inference.** A quote is an observation with noise. The posterior is every parameter
       value consistent with the quotes, and pricing under it gives the range of any mark.
    2. **The classical interval is the posterior's Gaussian approximation.** On a liquid surface the two
       agree. They part when the quotes do not pin the parameters, and when a name must borrow information
       from its peers. Two synthetic cases show why; SPY data show where.
    3. **The posterior is rarely computed because it is expensive** under the models desks use: microseconds
       under Black–Scholes, hours under rough Bergomi. That cost is the problem this tutorial solves.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | | Least squares: what desks do | Bayesian calibration |
    |:---|:---|:---|
    | Output | one parameter vector $\hat\theta$ | a distribution $p(\theta \mid y)$ |
    | Uncertainty | the Gauss–Newton ellipse $(J^\top W J)^{-1}$ around $\hat\theta$ | the posterior, whatever its shape |
    | What is known beforehand | penalties, parameters fixed by hand | a stated prior |
    | The range of a product | the delta method, through the product's gradient | the product priced under each posterior draw |
    | What "90%" means | across repeated sets of quotes | given today's quotes and the prior |
    | Cost | one optimization | thousands of likelihood evaluations |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "**The model in detail** (click to open): Black–Scholes, the quote noise, the posterior and Gauss–Newton": mo.md(r"""
        #### 1 · Prices in forward units

        Divide an option's price by the discount factor and the forward $F$. Then the forward is 1, rates
        and dividends disappear, and a call with log-moneyness $k = \ln(K/F)$ and maturity $T$ costs, under
        Black–Scholes with volatility $\sigma$,

        $$c(k, T; \sigma) = N(d_1) - e^{k} N(d_2), \qquad d_{1,2} = \frac{-k \pm \tfrac12 \sigma^2 T}{\sigma\sqrt T},$$

        where $N$ is the standard normal distribution function. A put follows from put–call parity,
        $p = c - 1 + e^{k}$. The **implied volatility** of a quote is the $\sigma$ that reproduces it.

        #### 2 · The design grid and the quote noise

        Every chain is read onto 63 points: 7 maturities (5 to 252 business days) times 9 standardized
        moneyness points $\delta = k / (\sigma_{\text{ref}}\sqrt T)$ from $-2$ to $2$, with
        $\sigma_{\text{ref}} = 20\%$. Each price is the model price plus Gaussian noise,
        $y_c = c_c(\theta) + \sigma_c \varepsilon_c$, with $\sigma_c$ the vega at $\sigma_{\text{ref}}$ times
        a third of a volatility point at the money, rising to one point at $|\delta| = 2$. On 21 October 2026
        this is wider than SPY's bid–ask at all 63 points.

        #### 3 · The posterior

        With a prior $p(\theta)$, the posterior given the quotes $y$ is

        $$p(\theta \mid y) \propto p(\theta) \prod_{c} \exp\!\Big(-\frac{(y_c - P_c(\theta))^2}{2\sigma_c^2}\Big).$$

        The posterior of a product's price is the law of that price when $\theta$ is drawn from it.

        #### 4 · Gauss–Newton is the posterior's Gaussian approximation

        Least squares minimizes $\chi^2(\theta) = \sum_c (y_c - P_c(\theta))^2 / \sigma_c^2$, which is
        $-2 \log$ of the likelihood. Expanding it to second order at the optimum $\hat\theta$, with
        $J = \partial P / \partial \theta$ and $W = \mathrm{diag}(\sigma_c^{-2})$, gives

        $$p(\theta \mid y) \approx \mathcal N\big(\hat\theta,\ (J^\top W J)^{-1}\big),$$

        under a flat prior away from its bounds. When many informative quotes make the likelihood close to
        Gaussian, the two agree (the Bernstein–von Mises theorem). They differ when the likelihood is far from
        Gaussian, or when the prior's bounds cut it.
        """)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · A quote is an observation, so a calibration is an estimate

    These are the 63 SPY prices of the close of 21 October 2026, read from the chain onto the design grid
    and shown as implied volatilities. Each is the mid of a bid and an ask, so it is an observation of the
    model price, with noise.
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
    _fig, _ax = plt.subplots(figsize=(8.4, 3.6), constrained_layout=True)
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
    Under Black–Scholes one quote is enough for a posterior. Take the at-the-money one-month price. Its noise
    is a third of a volatility point, so only volatilities close to its implied volatility are consistent
    with it.
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
        """Normalized posterior on sigma_grid under a flat prior on [5%, 60%]."""
        weights = np.exp(log_likelihood - log_likelihood.max())
        return weights / np.trapezoid(weights, sigma_grid)


    one_quote = grid_posterior(-0.5 * standardized_miss(quotes["at the money"]) ** 2)
    one_quote_mean = np.trapezoid(sigma_grid * one_quote, sigma_grid)
    one_quote_sd = np.sqrt(np.trapezoid((sigma_grid - one_quote_mean) ** 2 * one_quote, sigma_grid))
    _cdf = np.cumsum(one_quote) / np.sum(one_quote)
    _low, _high = (sigma_grid[np.searchsorted(_cdf, q)] for q in (0.05, 0.95))
    put_range = [surface.bs_price(quotes["put"]["k"], quotes["put"]["T"], s, "put") for s in (_low, _high)]
    return (
        grid_posterior,
        one_quote,
        one_quote_mean,
        one_quote_sd,
        put_range,
        quotes,
        sigma_grid,
        standardized_miss,
    )


@app.cell(hide_code=True)
def _(
    mo,
    one_quote,
    one_quote_mean,
    one_quote_sd,
    plots,
    plt,
    put_range,
    quotes,
    sigma_grid,
):
    _fig, _ax = plt.subplots(figsize=(7.2, 3.0), constrained_layout=True)
    _ax.plot(100 * sigma_grid, one_quote, color=plots.SERIES[0])
    _ax.fill_between(100 * sigma_grid, one_quote, color=plots.SERIES[0], alpha=0.2)
    _ax.set_xlim(100 * (one_quote_mean - 6 * one_quote_sd), 100 * (one_quote_mean + 6 * one_quote_sd))
    _ax.set_xlabel("Black–Scholes volatility σ (%)")
    _ax.set_yticks([])
    _ax.set_title("posterior of σ from the at-the-money one-month quote", loc="left")
    mo.vstack([
        _fig,
        mo.md(f"""
        Posterior: **σ = {100 * one_quote_mean:.2f}% ± {100 * one_quote_sd:.2f} volatility points**. The width
        is the quote's noise divided by its vega: the noise expressed as a volatility. Pricing another option
        under every σ the posterior allows gives that option's range. For the one-month put struck at
        {100 * quotes['put']['strike']:.0f}% of the forward, the 90% range is **{1e4 * put_range[0]:.2f} to
        {1e4 * put_range[1]:.2f} basis points** of the forward.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The range is what the rules ask for.** A mark from a calibrated model is a valuation input, and the
    rules ask for its uncertainty at a stated confidence:

    | Rule | What it asks for |
    |:---|:---|
    | Regulation (EU) 2016/101, Art. 9 (market price uncertainty) | for inputs calibrated to market quotes, a range of plausible values and a point in it at 90% confidence |
    | Regulation (EU) 2016/101, Art. 11 (model risk) | the same for the choice among models and calibration approaches |
    | IFRS 13, §93(h) | the sensitivity of a Level 3 fair value to reasonably possible alternative inputs |
    | PRA SS1/23 | model uncertainty: "the inherent uncertainty in the parameter estimates and results of statistical models" |

    The gap between the fair value and the 90% point is a valuation adjustment, deducted from capital.
    Least squares returns one price; the rules ask for a distribution of prices and a point in it.
    """)
    return


@app.cell(hide_code=True)
def _(mo, put_range, quotes):
    mo.md(f"""
    ## 2 · The posterior also judges the model

    The market quotes that one-month put at **{1e4 * quotes['put']['price']:.1f} basis points**, about
    {quotes['put']['price'] / put_range[1]:.0f} times the top of the range above. Use both quotes as data. If
    Black–Scholes were right, one volatility would fit both. The slider multiplies the assumed noise of both
    quotes.
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
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(10.4, 3.4), constrained_layout=True)
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
    _right.set_title(f"the best single σ = {100 * sigma_grid[best_sigma]:.2f}%", loc="left")
    mo.vstack([
        _fig,
        mo.md(f"""
        The two likelihoods do not overlap. The posterior from both quotes stays at the at-the-money value and
        misses the put by **{misses['put']:.1f} noise units**. It takes {noise_needed:.2g} times the assumed noise
        to bring both misses under two. That misfit is the smile: the posterior is precise, and the model is wrong.
        Desks therefore use models that produce a smile, Heston (slot 4) and rough Bergomi (slot 5). The
        question for the rest of this notebook is what the posterior adds once the model fits.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · With a model that fits a liquid surface, the classical interval is the posterior

    Heston fits the SPY smile (slot 4). Its likelihood costs 20 ms, so the exact posterior can be computed
    by MCMC and set beside the Gauss–Newton interval of least squares. On the full grid of 63 quotes, with
    the mean reversion $\kappa$ fixed at 2 or left free, the two agree, and so do the 90% ranges of four
    products. That is the Bernstein–von Mises theorem at work, and it is good news: the classical toolkit is
    the posterior's Gaussian limit. Over 200 synthetic Heston smiles, both intervals cover the truth at the
    stated rate (study S6: 87–92% for Gauss–Newton against 87–90% for the posterior).

    Remove quotes and the two part. On a sparse grid of 15 quotes with $\kappa$ free, the smile no longer
    pins $\kappa$, $\theta$ and $\sigma_v$ separately.
    """)
    return


@app.cell
def _(CHECKS, np, teaching):
    heston_checks = teaching.load_json(CHECKS / "heston_checks.json")
    kappa_draws = {"full grid, 63 quotes": np.load(CHECKS / "heston_checks-full_grid_kappa_free.npy")[:, 4],
                   "3 x 5 grid, 15 quotes": np.load(CHECKS / "heston_checks-3_x_5_grid_kappa_free.npy")[:, 4]}
    kappa_gn = {"full grid, 63 quotes": heston_checks["cases"]["full grid, kappa free"]["parameters"]["kappa"],
                "3 x 5 grid, 15 quotes": heston_checks["cases"]["3 x 5 grid, kappa free"]["parameters"]["kappa"]}
    return heston_checks, kappa_draws, kappa_gn


@app.cell(hide_code=True)
def _(heston_checks, kappa_draws, kappa_gn, mo, norm, np, plots, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(10.6, 3.3), constrained_layout=True)
    _grid = np.linspace(-2, 9, 600)
    for _ax, (_name, _d) in zip(_axes, kappa_draws.items()):
        _e = kappa_gn[_name]
        _lo, _hi = _e["gauss_newton_90"]
        _sd = (_hi - _lo) / (2 * 1.645)
        _ax.hist(_d, bins=60, density=True, color=plots.SERIES[0], alpha=0.55, label="exact posterior (MCMC)")
        _ax.plot(_grid, norm.pdf(_grid, _e["best"], _sd), color=plots.SERIES[1], label="Gauss–Newton normal")
        for _edge in (0.25, 8.0):
            _ax.axvline(_edge, color=plots.INK, linestyle="--", linewidth=0.8)
        _ax.axvspan(-2, 0, color=plots.BASELINE, alpha=0.25)
        _ax.set_xlim(-2, 9)
        _ax.set_yticks([])
        _ax.set_xlabel("mean reversion κ (per year)")
        _ax.set_title(f"Heston on SPY, {_name}", loc="left")
    _axes[0].legend(fontsize=8, loc="upper left")
    _rows = []
    for _case, _c in heston_checks["cases"].items():
        _ratios = [e["width_ratio_gn_over_posterior"] for e in _c["parameters"].values() if "width_ratio_gn_over_posterior" in e]
        _prod = [e["delta_over_posterior_range"] for e in _c["products"].values() if "delta_over_posterior_range" in e]
        _rows.append({"case": _case, "quotes": _c["n_quotes"],
                      "parameter widths, Gauss–Newton / posterior": f"{min(_ratios):.2f} to {max(_ratios):.2f}",
                      "product ranges, delta method / posterior": f"{min(_prod):.2f} to {max(_prod):.2f}"})
    mo.vstack([
        _fig,
        mo.md(f"""
        With 63 quotes the smile pins $\\kappa$ at {kappa_gn['full grid, 63 quotes']['posterior_q05_q50_q95'][1]:.1f}
        and the Gauss–Newton normal sits on the posterior. With 15 quotes the posterior is skewed and runs from
        {kappa_gn['3 x 5 grid, 15 quotes']['posterior_q05_q50_q95'][0]:.1f} to
        {kappa_gn['3 x 5 grid, 15 quotes']['posterior_q05_q50_q95'][2]:.1f}, while the Gauss–Newton interval runs
        from **{kappa_gn['3 x 5 grid, 15 quotes']['gauss_newton_90'][0]:.1f}** to
        {kappa_gn['3 x 5 grid, 15 quotes']['gauss_newton_90'][1]:.1f}: it includes negative mean reversion (shaded),
        which the model excludes. All five cases on SPY:
        """),
        mo.ui.table(_rows, selection=None, pagination=False),
        mo.md("*Source:* `tutorial_final/checks/heston_checks.py`, a quick check on SPY's quotes, not yet an "
              "accepted study."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Why Bayes, case 1: a posterior that is not an ellipse

    A stochastic-volatility smile carries its parameters in two places. The skew, the odd part of the smile,
    carries the product of the correlation $\rho$ and the volatility of volatility $\eta$. The curvature,
    the even part, carries $\eta^2$, and it is largest in the wings, where quotes are fewest and noisiest.
    Heston's $(\sigma_v, \rho)$ and rough Bergomi's $(\eta, \rho)$ share this structure. The course's toy
    (Lecture 9.7) keeps only these two pieces: nine quotes at strikes $x_j \in \{-1, -0.75, \dots, 1\}$,

    $$y_j = \rho\,\eta\,x_j + 0.05\,\eta^2 x_j^2 + s_j\,\varepsilon_j, \qquad s_j = 0.01 + 0.10\,w\,x_j^2,
    \qquad \eta \in (0.5, 3.5), \quad \rho \in (-0.99, -0.2).$$

    At $w = 1$ the skew is twelve times its noise at $x = 0.5$ and the curvature half its noise at $x = 1$.
    The posterior below is exact, on a grid. The last quantity is a quote at $x = 2$, a strike beyond the
    data, which depends on $\eta$ separately from $\rho\eta$: a product the quotes do not pin.
    """)
    return


@app.cell(hide_code=True)
def _(mo, why_bayes):
    toy_truth = mo.ui.dropdown(options=why_bayes.TRUTHS, value=list(why_bayes.TRUTHS)[0], label="truth")
    toy_wing = mo.ui.slider(0.1, 2.0, step=0.1, value=1.0, label="wing noise w", show_value=True)
    toy_seed = mo.ui.number(start=0, stop=999, step=1, value=7, label="quote draw")
    mo.hstack([toy_truth, toy_wing, toy_seed], justify="start")
    return toy_seed, toy_truth, toy_wing


@app.cell
def _(np, toy_seed, toy_truth, toy_wing, why_bayes):
    toy_y = why_bayes.data(toy_truth.value, np.random.default_rng(int(toy_seed.value)), toy_wing.value)
    toy = why_bayes.summary(toy_y, toy_wing.value)
    return (toy,)


@app.cell(hide_code=True)
def _(mo, np, plots, plt, toy, toy_truth, why_bayes):
    _eta, _rho, _p = toy["eta"], toy["rho"], toy["p"]
    _flat = _p.ravel()
    _order = np.sort(_flat)[::-1]
    _level = _order[np.searchsorted(np.cumsum(_order), 0.90)]
    _fig, _axes = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.3, 1, 1]})
    _ax = _axes[0]
    _ax.contourf(_eta, _rho, _p, levels=[_level, _p.max() * 2], colors=[plots.SERIES[0]], alpha=0.35)
    _vals, _vecs = np.linalg.eigh(toy["cov"])
    _t = np.linspace(0, 2 * np.pi, 200)
    _ell = toy["best"][:, None] + _vecs @ (np.sqrt(_vals)[:, None] * np.vstack([np.cos(_t), np.sin(_t)])) * np.sqrt(4.605)
    _ax.plot(_ell[0], _ell[1], color=plots.SERIES[1], linestyle="--", linewidth=1.4, label="Gauss–Newton 90% ellipse")
    _ax.plot([], [], color=plots.SERIES[0], linewidth=8, alpha=0.4, label="posterior 90% region")
    _ax.plot(*toy_truth.value, "*", color=plots.INK, markersize=11, label="truth")
    _ax.axhline(why_bayes.RHO_BOX[0], color=plots.INK, linestyle=":", linewidth=1)
    _ax.axvline(why_bayes.ETA_BOX[0], color=plots.INK, linestyle=":", linewidth=1)
    _ax.set_xlim(0.0, 3.5)
    _ax.set_ylim(-1.5, -0.2)
    _ax.set_xlabel("η")
    _ax.set_ylabel("ρ")
    _ax.set_title("the posterior follows the valley; dotted: the prior box", loc="left", fontsize=9)
    _ax.legend(fontsize=7.5, loc="lower right")
    _ax = _axes[1]
    _pe = _p.sum(axis=0)
    _ax.plot(_eta[0], _pe / _pe.max(), color=plots.SERIES[0], label="posterior of η")
    _sd = np.sqrt(toy["cov"][0, 0])
    _g = np.linspace(0, 3.5, 400)
    _ax.plot(_g, np.exp(-0.5 * ((_g - toy["best"][0]) / _sd) ** 2), color=plots.SERIES[1], linestyle="--",
             label="Gauss–Newton normal")
    _ax.axvline(toy_truth.value[0], color=plots.INK, linestyle=":", linewidth=1)
    _ax.axvspan(0, why_bayes.ETA_BOX[0], color=plots.BASELINE, alpha=0.25)
    _ax.set_xlim(0, 3.5)
    _ax.set_yticks([])
    _ax.set_xlabel("η")
    _ax.set_title("η alone", loc="left", fontsize=9)
    _ax.legend(fontsize=7.5)
    _ax = _axes[2]
    _q2 = why_bayes.model(_eta, _rho, np.array([2.0]))[..., 0].ravel()
    _row = toy["rows"][3]
    _span = [min(_row["posterior_90"][0], _row["gauss_newton_90"][0]), max(_row["posterior_90"][1], _row["gauss_newton_90"][1])]
    _pad = 0.6 * (_span[1] - _span[0])
    _h, _edges = np.histogram(_q2, bins=np.linspace(_span[0] - _pad, _span[1] + _pad, 70), weights=_flat, density=True)
    _mid = 0.5 * (_edges[1:] + _edges[:-1])
    _ax.plot(_mid, _h / _h.max(), color=plots.SERIES[0], label="posterior")
    _c = _row["least_squares"]
    _s2 = (_row["gauss_newton_90"][1] - _row["gauss_newton_90"][0]) / (2 * 1.645)
    _ax.plot(_mid, np.exp(-0.5 * ((_mid - _c) / _s2) ** 2), color=plots.SERIES[1], linestyle="--", label="delta method")
    _ax.axvline(float(why_bayes.model(*toy_truth.value, np.array([2.0]))[0]), color=plots.INK, linestyle=":", linewidth=1)
    _ax.set_xlim(_span[0] - _pad, _span[1] + _pad)
    _ax.set_yticks([])
    _ax.set_xlabel("quote at x = 2")
    _ax.set_title("a quote beyond the data", loc="left", fontsize=9)
    _ax.legend(fontsize=7.5)
    _table = []
    for _r in toy["rows"]:
        _lo, _hi = _r["gauss_newton_90"]
        _plo, _phi = _r["posterior_90"]
        _leaves = {"η": _lo < why_bayes.ETA_BOX[0] or _hi > why_bayes.ETA_BOX[1],
                   "ρ": _lo < why_bayes.RHO_BOX[0] or _hi > why_bayes.RHO_BOX[1]}.get(_r["quantity"])
        _table.append({"quantity": _r["quantity"], "posterior 90%": f"{_plo:.3f} to {_phi:.3f}",
                       "Gauss–Newton 90%": f"{_lo:.3f} to {_hi:.3f}",
                       "width, Gauss–Newton / posterior": round((_hi - _lo) / (_phi - _plo), 2),
                       "Gauss–Newton leaves the prior box": "–" if _leaves is None else ("yes" if _leaves else "no")})
    mo.vstack([_fig, mo.ui.table(_table, selection=None, pagination=False)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **How to read it.** The product $\rho\eta$ is pinned by the skew, so both methods agree on it. The split
    between $\rho$ and $\eta$ is not: the data allow a curved valley along the curve $\rho\eta = c$, cut by
    the bound $\rho = -0.99$. The posterior follows the valley and stops at the bound. The Gauss–Newton ellipse is
    straight and symmetric, so it runs past the box into values the model excludes, and it misplaces the
    range of the quote beyond the data. Move $w$ down to 0.3: the wings pin $\eta^2$, the valley closes, and
    the two methods agree again.

    **Is the classical interval wrong on average?** Not here. Over truths drawn from the prior, both 90%
    intervals cover the truth about 90% of the time. Near the bound, the Gauss–Newton interval for $\rho$
    covers 84% against the posterior's 95% (Lecture 10, below). The failure is one of **shape**: an interval
    with impossible values cannot be priced, and a range in the wrong place gives the wrong reserve.
    """)
    return


@app.cell(hide_code=True)
def _(COURSE, mo):
    mo.image(src=COURSE / "images" / "l10-credible-vs-confidence.png",
             caption="Lecture 10: coverage of posterior and Gauss–Newton 90% intervals in the toy, 2,000 data sets per case.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Where this happens on real data.** Not on SPY's full surface, whose wings sit 5 to 7 volatility points
    above the money against one point of noise. It happens when the wings or the maturities are few:

    | Data | What least squares returns | What the posterior returns |
    |:---|:---|:---|
    | Heston, SPY reduced to 15 quotes, κ free (section 3) | κ from −0.6 to 5.8; the 2-year put's range 21% wider | κ from 0.7 to 6.3, skewed |
    | Rough Bergomi, SPY reduced to 15 quotes | ρ's local interval past the bound −0.99 | a posterior cut by the bound |

    *Source:* `tutorial_final/checks/` (quick checks of 23 October 2026).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Why Bayes, case 2: names that must borrow from their peers

    A valuation desk marks implied volatilities for every name in a sector. A few names are liquid, with a
    full smile. Most quote one wide option, if any. Desks then **proxy**: they mark the illiquid name at a
    peer average. A hierarchical model makes the proxy precise: each name's volatility is drawn from a sector
    distribution, $\log \sigma_i \sim \mathcal N(\mu, \tau^2)$, whose centre and spread are learned from all
    names at once. Each name's posterior then combines its own quote with what the sector says, weighted by
    how informative each is.

    The synthetic sector below uses Black–Scholes per name. The liquid names quote the 63-point grid. The
    illiquid names quote one one-month at-the-money call, with the noise set by the slider. Four methods mark
    the illiquid names:

    | Method | Mark | 90% interval |
    |:---|:---|:---|
    | per name | least squares on the name's own quote | Gauss–Newton |
    | proxy | the mean of the liquid names' marks | ±1.645 times their cross-sectional spread |
    | empirical Bayes (classical random effects) | the hierarchical model with $(\mu, \tau)$ fixed at their estimates | conditional on those estimates |
    | hierarchical Bayes | the same model, with $(\mu, \tau)$ integrated out | the posterior |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    sector_size = mo.ui.dropdown(options={"8 liquid + 32 illiquid": (8, 32), "2 liquid + 6 illiquid": (2, 6)},
                                 value="8 liquid + 32 illiquid", label="sector")
    sector_noise = mo.ui.slider(1.0, 6.0, step=0.5, value=3.0, label="illiquid quote noise (vol points)", show_value=True)
    sector_spread = mo.ui.slider(1.0, 6.0, step=0.5, value=3.0, label="spread of true vols across names (vol points)",
                                 show_value=True)
    sector_run = mo.ui.run_button(label="Simulate 100 sectors with these settings (about 10 s)")
    mo.vstack([mo.hstack([sector_size, sector_noise, sector_spread], justify="start"), sector_run])
    return sector_noise, sector_run, sector_size, sector_spread


@app.cell
def _(
    CHECKS,
    bs_hierarchy,
    sector_noise,
    sector_run,
    sector_size,
    sector_spread,
    teaching,
):
    sector_cached = teaching.load_json(CHECKS / "bs_hierarchy.json")
    if sector_run.value:
        sector_result = bs_hierarchy.run(*sector_size.value, 100, 2026, tau=sector_spread.value / 30,
                                         ill_vol_noise=sector_noise.value / 100)
        sector_source = (f"100 sectors simulated now: quote noise {sector_noise.value:g} and spread {sector_spread.value:g} "
                         "volatility points.")
    else:
        sector_result = sector_cached["sectors"][sector_size.selected_key]
        sector_source = (f"{sector_result['replications']} sectors, computed before the session with quote noise and spread "
                         "of 3 volatility points each; press the button to rerun with the sliders' values.")
    return sector_result, sector_source


@app.cell(hide_code=True)
def _(mo, np, plots, plt, sector_result, sector_source):
    _ex = sector_result["example"]
    _truth = np.array(_ex["truth"])
    _order = np.argsort(_truth)
    _x = np.arange(len(_truth))
    _proxy = np.array(_ex["proxy"])[0]
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(12.0, 3.6), constrained_layout=True,
                                         gridspec_kw={"width_ratios": [1.7, 1]})
    _left.axhspan(100 * _proxy[1], 100 * _proxy[2], color=plots.SERIES[2], alpha=0.15, label="proxy 90% band")
    _left.axhline(100 * _proxy[0], color=plots.SERIES[2], linewidth=1)
    for _m, _off, _c in (("per name", -0.18, plots.BASELINE), ("hierarchical Bayes", 0.18, plots.SERIES[0])):
        _e = 100 * np.array(_ex[_m])[_order]
        _left.vlines(_x + _off, _e[:, 1], _e[:, 2], color=_c, linewidth=2, label=f"{_m} 90%")
    _left.scatter(_x, 100 * _truth[_order], s=14, color=plots.INK, zorder=5, label="true volatility")
    _left.set_xlabel("illiquid names, sorted by true volatility")
    _left.set_ylabel("implied volatility (%)")
    _left.set_title("one sector: each illiquid name's 90% interval", loc="left", fontsize=9)
    _left.legend(fontsize=7, loc="upper left", ncol=2)
    _scores = sector_result["scores"]
    _methods = list(_scores)
    _colors = [plots.BASELINE, plots.SERIES[2], plots.SERIES[1], plots.SERIES[0]]
    _right.bar(_methods, [100 * _scores[m]["coverage_90"] for m in _methods], color=_colors)
    _right.axhline(90, color=plots.INK, linestyle="--", linewidth=0.8)
    _right.set_ylim(40, 100)
    _right.set_ylabel("90% intervals covering the truth (%)")
    _right.set_xticks(range(len(_methods)), [m.replace(" ", "\n", 1) for m in _methods], fontsize=7.5)
    _right.set_title("coverage over the sectors", loc="left", fontsize=9)
    mo.vstack([
        _fig,
        mo.ui.table([{"method": m, "error of the mark (RMSE, vol points)": round(s["rmse_vol_points"], 2),
                      "90% coverage": f"{100 * s['coverage_90']:.0f}%", "mean 90% width (vol points)": round(s["mean_width_vol_points"], 1),
                      "error on a 6-month call nobody quotes": f"{s['product_rmse_percent']:.1f}%"}
                     for m, s in _scores.items()], selection=None, pagination=False),
        mo.md(sector_source),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **How to read it.** Pooling in any form cuts the error of the mark by about a quarter, because a name's
    own wide quote and the sector's centre carry independent information. The proxy ignores the name's own
    quote and is biased for names unlike their peers. The per-name interval is honest but wide. With eight
    liquid peers, empirical Bayes and hierarchical Bayes agree. **With two, only the full posterior keeps 90%
    coverage** (88% against 70% for empirical Bayes and 60% for the proxy): the classical version fixes the
    sector's spread at an estimate from a handful of names, and its intervals inherit that estimate's
    error as if it were known.

    The truth here is drawn from the model's own sector distribution, which flatters the hierarchy. A name
    that really differs from its peers is pulled too far toward them, so on a desk the pooled marks come
    with a check of each name against the sector.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · What the posterior buys a desk

    Both cases end in the same object: a range for every mark, at a stated confidence, from one set of
    draws. The reserve for a sold position is the distance from the mark to the 90% point of that range.

    The same logic decides the long end of an index surface, where long-dated exotics live. Listed SPY quotes
    stop at about two years; the forward variance beyond is an assumption. A desk holds it flat and reports
    the range from the calibrated parameters alone. The posterior makes the assumption a stated prior,
    narrows it with the long-dated quotes, and reports what follows (slot 5):
    """)
    return


@app.cell(hide_code=True)
def _(CHECKS, mo, np, teaching):
    _le = teaching.load_json(CHECKS / "long_end" / "long_end.json")
    _lx = teaching.load_json(CHECKS / "long_end" / "long_end_extra.json")
    _sens = np.array(list(_lx["sensitivity_range90_percent"].values()))
    _desk = 100 * np.array(_le["range90_desk_delta"]) / np.array(_le["price_desk_point"])
    _post = 100 * np.array(_le["range90_posterior_mc"]) / np.array(_le["posterior_mc_quantiles"])[1]
    _rows = []
    for _j, _name in enumerate(_le["products"]):
        _unit = (lambda v: f"{100 * v:.0f} bp of notional") if _name.startswith("autocallable") else (lambda v: f"{v:.0f}%")
        _rows.append({"product on SPY, rough Bergomi": _name, "desk: calibrated parameters, flat long end": _unit(_desk[_j]),
                      "posterior: all inputs": _unit(_post[_j]),
                      "posterior, prior width 0.1 to 0.3": f"{_unit(_sens[:, _j].min())} to {_unit(_sens[:, _j].max())}"})
    mo.vstack([mo.ui.table(_rows, selection=None, pagination=False),
               mo.md("90% ranges as a share of the price. *Source:* `tutorial_final/checks/long_end/`, a quick check.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Why desks do not compute the posterior

    The posterior needs the likelihood thousands of times. Under Black–Scholes that is a grid of closed-form
    prices. Under Heston each smile is a Fourier integral, so the exact posterior takes minutes, and slot 4
    uses it to check a network. Under rough Bergomi each smile is a Monte Carlo run, so the exact posterior
    takes at least 17 CPU-hours: no desk computes it every morning, for every underlying.
    """)
    return


@app.cell(hide_code=True)
def _(RESULTS, ROOT, mo, teaching):
    costs = teaching.load_json(RESULTS / "costs" / "costs.json")
    mo.vstack([
        mo.image(src=ROOT / "curriculum" / "images" / "ladder.png",
                 caption="One likelihood evaluation and one exact posterior under each model, CPU-seconds on a log scale "
                         "(Heston at one quadrature rung; the rough-Bergomi posterior is a lower bound)."),
        mo.md(f"The measured costs, on one {costs['cpu_count']}-core machine (study S9):"),
        mo.ui.table(teaching.cost_rows(costs), selection=None, pagination=False),
    ])
    return (costs,)


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
    morning = [
        {"route": "Heston, exact posterior (one quadrature rung)", "CPU-hours": _n * _heston_exact / 3600},
        {"route": "Rough Bergomi, exact posterior (lower bound)", "CPU-hours": _n * _rough_exact / 3600},
        {"route": "Rough Bergomi, amortized network", "CPU-hours": _n * _network / 3600},
    ]
    for _row in morning:
        _row["cores for a four-hour morning"] = round(_row["CPU-hours"] / 4, 1)
        _row["CPU-hours"] = round(_row["CPU-hours"], 1)
    mo.ui.table(morning, selection=None, pagination=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Amortized inference moves the cost into training.** Parameters are drawn from the prior, smiles are
    simulated from them, and a network learns to return the posterior from a smile. A new smile then costs
    one pass through the network, and the network is checked before it is trusted.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid("""
    flowchart LR
      P["prior over the parameters"] --> S["simulator: price a smile, add quote noise"]
      S --> B["a bank of (parameters, smile) pairs"]
      B --> N["train a network once"]
      Q["today's smile"] --> N
      N --> D["posterior draws in seconds"]
      D --> C["checks: an exact sibling model, SBC, an audit of each smile"]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    1. **Calibration is inference.** A quote is an observation with noise, the posterior is every parameter
       value consistent with it, and pricing under the posterior gives the range the rules ask for.
    2. **The posterior judges the model first.** Black–Scholes is precise and wrong on SPY; Heston and rough
       Bergomi fit.
    3. **On a liquid surface, the classical interval is the posterior's Gaussian limit.** On SPY's 63 quotes
       the two agree; with 15 quotes they part.
    4. **Two reasons to want the posterior:** quotes that do not pin the parameters, where the posterior
       follows a shape no ellipse has, and names that must borrow from their peers, where only the full
       posterior keeps its coverage with few peers.
    5. **The posterior is expensive** under the models desks use for exotics, which is why the rest of the
       afternoon amortizes it.

    **Next, slot 2:** how a network trained on simulations returns the posterior, and when that pays.

    *Evidence and code:* `tutorial_final/checks/` (quick checks of 23 October 2026, not yet accepted studies),
    Lectures 9 and 10 of the [course](../../docs/course/README.md), and `src/sbi4finance/calibration/`.
    """)
    return


if __name__ == "__main__":
    app.run()
