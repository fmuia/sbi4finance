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
    from scipy import stats

    from sbi4f_tutorial import backbones, plots, surface, teaching

    ROOT = mo.notebook_dir().parent
    ASSETS = ROOT / "assets"
    RESULTS = ASSETS / "results" / "calibration"
    CHECKS = ASSETS / "quick-checks"  # outputs of the lab's quick checks, not accepted studies

    plots.style()

    return (
        ASSETS,
        CHECKS,
        RESULTS,
        backbones,
        mo,
        np,
        plots,
        plt,
        stats,
        surface,
        teaching,
    )


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
def _(mo, uncertainty):
    _liquid = uncertainty["Heston, 63 SPY quotes: 90% range"]
    _thin = uncertainty["Heston, 15 SPY quotes, κ free: 90% range"]
    _models = uncertainty["two models that fit one smile: gap between marks"]
    mo.vstack([
        mo.md(r"""
    ### Why the uncertainty matters

    A calibration returns one set of parameters, the one whose prices come closest to the quotes. The quotes are
    noisy, so other parameter values reproduce them almost as well, and they give different prices to the products
    the smile does not quote: exotic options, and vanillas beyond the quoted strikes and maturities.
    """),
        mo.callout(mo.md(f"""
    **In one line.** On SPY's 63 liquid quotes, Heston's products move by {min(_liquid):.0f} to {max(_liquid):.0f}% of
    their price. With 15 quotes the 2-year put moves by {_thin[3]:.0f}%, and a second model that fits a smile as well
    marks the forward-start {_models[2]:.0f}% apart. The rules ask banks to measure this range and to hold capital
    against its 90% point.
    """), kind="info"),
    ])

    return


@app.cell
def _(CHECKS, RESULTS, np, teaching):
    heston_checks = teaching.load_json(CHECKS / "heston_checks.json")  # exact posterior against Gauss-Newton on SPY (lab quick check)
    _across = teaching.load_json(RESULTS / "rough-pilot" / "s7-across" / "across-models.json")
    PRODUCTS = {"variance_swap": "variance swap, 3 months", "volatility_swap": "volatility swap, 3 months",
                "forward_start_call": "forward-start call, 3 to 6 months", "put_out_of_grid": "2-year 90% put"}
    _liquid = heston_checks["cases"]["full grid, kappa fixed"]["products"]
    _thin = heston_checks["cases"]["3 x 5 grid, kappa free"]["products"]
    _smile = next(_s for _s in _across["surfaces"] if _s["test_row"] == 290)["products"]
    uncertainty = {  # % of each product's price
        "Heston, 63 SPY quotes: 90% range": [100 * _liquid[_p]["posterior_range_90_relative"] for _p in PRODUCTS],
        "Heston, 15 SPY quotes, κ free: 90% range": [100 * _thin[_p]["posterior_range_90_relative"] for _p in PRODUCTS],
        "two models that fit one smile: gap between marks": [100 * abs(_smile[_p]["gap_between_points_relative"]) for _p in PRODUCTS],
    }
    put_money = {}  # the 2-year put on 100 million of notional: its mark and its reserve for a sold position
    for _name, _case in (("63 quotes", "full grid, kappa fixed"), ("15 quotes", "3 x 5 grid, kappa free")):
        _summary = heston_checks["cases"][_case]
        _mark = _summary["products"]["put_out_of_grid"]["least_squares_point"]
        _q90 = np.quantile(np.asarray(_summary["posterior_product_draws"])[:, list(PRODUCTS).index("put_out_of_grid")], 0.9)
        put_money[_name] = {"mark": 100e6 * _mark, "reserve": 100e6 * (_q90 - _mark)}
    put_money["second model"] = {"gap": 100e6 * abs(_smile["put_out_of_grid"]["heston"]["point"] - _smile["put_out_of_grid"]["rough"]["point"])}

    return PRODUCTS, heston_checks, put_money, uncertainty


@app.cell(hide_code=True)
def _(PRODUCTS, mo, np, plots, plt, put_money, uncertainty):
    _fig, _ax = plt.subplots(figsize=(9.6, 3.4), constrained_layout=True)
    _y = np.arange(len(PRODUCTS))
    for _i, ((_label, _values), _color) in enumerate(zip(uncertainty.items(), plots.SERIES)):
        _bars = _ax.barh(_y + (_i - 1) * 0.27, _values, height=0.25, color=_color, label=_label)
        for _bar, _value in zip(_bars, _values):
            _ax.text(_value + 0.4, _bar.get_y() + _bar.get_height() / 2, f"{_value:.1f}%", va="center", fontsize=7,
                     color=plots.INK_2)
    _ax.set_yticks(_y, list(PRODUCTS.values()))
    _ax.invert_yaxis()
    _ax.set_xlabel("% of the product's price")
    _ax.set_xlim(0, 36)
    _ax.grid(axis="y", visible=False)
    _ax.legend(fontsize=7.5, loc="upper right")
    _ax.set_title("how far a product's price moves while the model still fits the quotes", loc="left")
    mo.vstack([
        _fig,
        mo.hstack([
            mo.stat(value=f"{put_money['63 quotes']['mark'] / 1e6:.1f} m", label="2-year 90% put: the mark",
                    caption="on 100 million of notional", bordered=True),
            mo.stat(value=f"{put_money['63 quotes']['reserve'] / 1e3:,.0f} k", label="reserve, 63 quotes",
                    caption="90% point minus the mark", bordered=True),
            mo.stat(value=f"{put_money['15 quotes']['reserve'] / 1e3:,.0f} k", label="reserve, 15 quotes",
                    caption="same, κ free", bordered=True),
            mo.stat(value=f"{put_money['second model']['gap'] / 1e3:,.0f} k", label="gap to a second model",
                    caption="same smile, both fit", bordered=True),
        ], widths="equal"),
        mo.md(r"""
    - **Blue:** Heston's exact posterior on SPY's 63 quotes, with the mean reversion fixed. The range runs from the
      5% to the 95% point of each product's price over 300 posterior draws.
    - **Orange:** the same model on 15 of those quotes, at 1, 3 and 6 months and $\delta$ from −1 to 1, with the mean
      reversion free. The 2-year put depends on what happens beyond six months, which these quotes do not reach.
    - **Green:** a smile generated by rough Bergomi and fitted by both Heston and rough Bergomi, nearly equally well;
      the bars are the gaps between the two models' least-squares prices (study S7).
    - The cards translate the 2-year put into money. Each case uses its own mark; the 15-quote and second-model
      marks are 6.4 and 6.7 million.

    Sources: `assets/quick-checks/heston_checks.json` (a lab quick check, not an accepted study) and
    `assets/results/calibration/rough-pilot/s7-across/` (pilot networks).
    """),
    ])

    return


@app.cell(hide_code=True)
def _(mo):
    _card = lambda text: mo.callout(mo.md(text), kind="neutral")  # noqa: E731
    mo.vstack([
        mo.md("**Why it matters**"),
        mo.hstack([
            _card("**Capital.** Under EU prudent valuation (Regulation 2016/101), a bank deducts additional valuation "
                  "adjustments (AVAs) from its capital. For calibrated parameters (Art. 9) and for the choice of model "
                  "(Art. 11), the AVA is the distance from the mark to the 90% point of the range of plausible values."),
            _card("**Accounts.** IFRS 13 §93(h): for fair values that rest on unobservable inputs (Level 3), the bank "
                  "discloses how the value changes under reasonably possible alternative inputs."),
        ], widths="equal"),
        mo.hstack([
            _card("**Model risk.** SR 26-2 in the US, which replaces SR 11-7, and PRA SS1/23 in the UK ask banks to "
                  "understand the uncertainty of parameter estimates and of model choices."),
            _card("**P&L.** Refitting to the next day's quotes moves the parameters within this range, and the marks "
                  "with them, even when the market has not moved."),
        ], widths="equal"),
        mo.md("**What is measured**"),
        mo.hstack([
            _card("**The range.** The 5% to 95% points of a product's price over the calibrations the quotes allow: "
                  "the bars above."),
            _card("**The 90% point and the reserve.** For a position the bank has sold, the 90% point is the prudent "
                  "value and the reserve (the AVA) is that point minus the mark; for a position it holds, the 10% point. "
                  "It is a quantile, like a VaR, but of today's valuation, not of tomorrow's P&L."),
            _card("**Coverage.** On simulated smiles whose true parameters are known, a method's 90% ranges should "
                  "contain the truth 90% of the time. This checks the method, not one mark."),
        ], widths="equal"),
        mo.md("Market-risk measures, VaR and expected shortfall, describe how the value may change over a horizon. "
              "They do not measure how uncertain today's mark is: that is the range and its 90% point."),
    ])

    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("## 2 · The data"),
        mo.callout(mo.md(r"""
    **In one line.** One observation is 63 SPY option prices on a fixed grid of maturities and strikes, each with a
    stated noise. The grid covers the centre of the smile, where the quotes are informative; prices in the far wings and
    beyond one year come from the model, not from the data.
    """), kind="info"),
        mo.md(r"""
    SPY options are quoted for many expirations, from one day to more than two years, each with its own strikes. We read
    these quotes onto a fixed grid: 7 maturities from one week to one year, and 9 strikes per maturity. Strikes are
    placed by their standardized moneyness $\delta = \ln(K/F) / (0.2\sqrt T)$, the distance of the strike $K$ from the
    forward $F$ in standard deviations of a 20% volatility over the option's life $T$. At $\delta = -2$ a put is far
    below the money; at $\delta = 2$ a call is far above it.
    """),
    ])

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
def _(mo, spy):
    mo.md(f"""
    ### What the smile tells: level, skew and curvature

    Three numbers describe a smile near the money. The models are judged on how they reproduce each, at every maturity.

    - **Level**, the at-the-money volatility: how expensive options are overall. SPY's rises from
      {100 * spy['atm_implied_vol'][0]:.1f}% at one week to {100 * spy['atm_implied_vol'][-1]:.1f}% at one year.
    - **Skew**, the slope at the money: puts are dearer than calls, because the market prices large falls and
      volatility tends to rise when the price falls. In Heston it comes from the correlation ρ times the volatility
      of variance.
    - **Curvature**: both wings above a straight line, the market pricing fat tails. In Heston it comes from the
      volatility of variance.

    Skew and curvature grow without bound as the maturity shortens, each at its own rate. Heston's skew tends to a
    constant at short maturities, rough Bergomi's follows a power law: this is why slot 5 uses rough Bergomi.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, plots, plt, spy, spy_iv, surface):
    _days = np.array(surface.DESIGN_DAYS)
    _t = _days / surface.TRADING_DAYS
    _k = np.array(surface.DESIGN_DELTAS) * surface.SIGMA_REF * np.sqrt(_t[2])  # log-moneyness at one month
    _coef = np.polyfit(_k[2:7], spy_iv[2, 2:7], 2)  # a quadratic through the five quotes nearest the money
    _level, _skew, _curvature = _coef[2], _coef[1], 2 * _coef[0]
    _fig, (_left, _middle, _right) = plt.subplots(1, 3, figsize=(9.6, 3.2), constrained_layout=True,
                                                  gridspec_kw={"width_ratios": [1.3, 1, 1]})
    _fine = np.linspace(_k[2], _k[6], 60)
    _left.scatter(100 * _k, 100 * spy_iv[2], s=16, color=plots.SERIES[0], zorder=3, label="SPY, 1 month")
    _left.plot(100 * _fine, 100 * np.polyval(_coef, _fine), color=plots.INK_2, linewidth=1.2, label="quadratic near the money")
    _left.plot(100 * _fine, 100 * (_level + _skew * _fine), color=plots.SERIES[1], linewidth=1, linestyle="--",
               label="slope at the money")
    _left.set_xlim(100 * _k[0] - 1, 100 * _k[-1] + 1)
    _left.set_ylim(8, 27)
    _left.set_xlabel("log-moneyness k = ln(K/F), %")
    _left.set_ylabel("implied volatility (%)")
    _left.set_title("one smile: level, skew, curvature", loc="left")
    _left.legend(fontsize=7, loc="upper right")
    _left.annotate(f"level {100 * _level:.1f}%", (0, 100 * _level), xytext=(4, -26), textcoords="offset points",
                   fontsize=7.5, color=plots.INK_2, arrowprops={"arrowstyle": "-", "color": plots.MUTED, "lw": 0.7})
    _middle.plot(_days, 100 * np.asarray(spy["atm_implied_vol"]), "o-", color=plots.SERIES[0], markersize=4)
    _middle.set_xscale("log")
    _middle.minorticks_off()
    _middle.set_xticks(list(_days), [str(_d) for _d in _days])
    _middle.set_xlabel("maturity, trading days")
    _middle.set_ylabel("at-the-money volatility (%)")
    _middle.set_title("level by maturity", loc="left")
    _skews, _curvatures = np.abs(np.asarray(spy["atm_skew"])), np.asarray(spy["atm_curvature"])
    _fit = np.polyfit(np.log(_t), np.log(_curvatures), 1)
    _exponent = f"{spy['skew_power_law']['exponent']:.2f}".replace("-", "−")
    _right.loglog(_days, _skews, "o", color=plots.SERIES[1], markersize=4, label=f"|skew|, exponent {_exponent}")
    _right.loglog(_days, spy["skew_power_law"]["scale"] * _t ** spy["skew_power_law"]["exponent"], color=plots.SERIES[1],
                  linewidth=1)
    _right.loglog(_days, _curvatures, "s", color=plots.SERIES[2], markersize=4,
                  label=f"curvature, exponent {_fit[0]:.2f}".replace("-", "−"))
    _right.loglog(_days, np.exp(np.polyval(_fit, np.log(_t))), color=plots.SERIES[2], linewidth=1)
    _right.minorticks_off()
    _right.set_xticks(list(_days), [str(_d) for _d in _days])
    _right.set_xlabel("maturity, trading days")
    _right.set_title("skew and curvature by maturity", loc="left")
    _right.legend(fontsize=7, loc="upper right")
    mo.vstack([
        _fig,
        mo.md(f"""
    - **Left:** the one-month smile against log-moneyness, with the quadratic through the five quotes nearest the
      money. Its value, slope and curvature at $k = 0$ are the level ({100 * _level:.1f}%), the skew
      ({_skew:.2f}) and the curvature ({_curvature:.1f}).
    - **Right:** the absolute skew and the curvature on log–log axes, where a power law is a straight line. The skew's
      exponent is {_exponent} (claim C1).
    """.replace("-0.", "−0.")),
    ])

    return


@app.cell(hide_code=True)
def _(mo, np, spy_expirations):
    _reach = np.array([_e["strip_deltas"] for _e in spy_expirations])
    mo.md(f"""
    ### Why the grid stops at δ = ±2

    - **The calls run out.** On the call side the cleaned quotes of {int(np.sum(_reach[:, 1] < 2))} of the
      {len(_reach)} expirations stop before δ = 2, and none reaches beyond {_reach[:, 1].max():.1f}. Past that point
      the chain lists no strikes, or their quotes fail the cleaning rules: a zero bid, or a spread above half the mid.
    - **The far calls say little.** At δ = 1.5 and 2 the call prices are already below their noise (section 2's last
      figure, and *What a network reads*).
    - **Every point is read, none extrapolated.** At δ = ±2 every grid point is read from quoted strikes.
    - **The centre holds what the models must fit:** level, skew and curvature.

    The puts reach much further, to δ = {np.median(_reach[:, 0]):.1f} for the median expiration. That part of the
    chain gives the variance-swap curve (*The data in detail*), but it is not calibration data. What lies outside the
    grid, the far wings and maturities beyond one year, is priced by the model alone: there the uncertainty of
    section 1 matters most.
    """.replace("δ = -", "δ = −"))

    return


@app.cell(hide_code=True)
def _(np, plots, plt, spy_expirations, surface):
    _days = np.array([_e["days"] for _e in spy_expirations], dtype=float)
    _low = np.array([_e["strip_deltas"][0] for _e in spy_expirations])
    _high = np.array([_e["strip_deltas"][1] for _e in spy_expirations])
    _short = _high < 2
    _fig, _ax = plt.subplots(figsize=(9.6, 3.1), constrained_layout=True)
    _ax.axhspan(-2, 2, color=plots.SERIES[0], alpha=0.12, linewidth=0, label="the grid, δ from −2 to 2")
    _ax.vlines(_days[~_short], _low[~_short], _high[~_short], color=plots.INK_2, linewidth=2,
               label="cleaned quotes of one expiration")
    _ax.vlines(_days[_short], _low[_short], _high[_short], color=plots.SERIES[1], linewidth=2,
               label="calls stop before δ = 2")
    for _x in surface.DESIGN_DAYS:
        _ax.axvline(_x, color=plots.BASELINE, linewidth=0.6, linestyle=":")
    _ax.set_xscale("log")
    _ax.minorticks_off()
    _ticks = [1, 5, 10, 21, 42, 63, 126, 252, int(_days.max())]
    _ax.set_xticks(_ticks, [str(_x) for _x in _ticks])
    _ax.set_ylim(-15, 4)
    _ax.set_xlabel("expiration, trading days (dotted: the grid's maturities)")
    _ax.set_ylabel("standardized moneyness δ")
    _ax.set_title("how far each expiration's cleaned quotes reach", loc="left")
    _ax.legend(fontsize=7.5, loc="lower left")
    _fig

    return


@app.cell(hide_code=True)
def _(mo):
    _card = lambda text: mo.callout(mo.md(text), kind="neutral")  # noqa: E731
    mo.vstack([
        mo.md(r"""
    ### The noise

    Each grid price is the model's price plus an error, $y_c = P_c(\theta) + \sigma_c\,\varepsilon_c$, with independent
    standard normal errors $\varepsilon_c$. The standard deviation $\sigma_c$, the noise, is set in two steps:

    $$\sigma_c = \text{vega}_{20\%}(k_c, T_c) \times s(\delta_c), \qquad s(\delta) = 0.3 + 0.35\,|\delta| \text{ volatility points}.$$
    """),
        mo.hstack([
            _card("**Why in volatility points.** Market makers quote in volatility, and a quote's half-spread is "
                  "roughly constant in volatility: a fraction of a point at the money, about a point in the wings. In "
                  "price it is not."),
            _card("**Why the vega at 20%.** The vega turns a volatility error into a price error. Taking it at a fixed "
                  "20%, not at each smile's own volatility, gives every smile, real or simulated, exactly the same noise."),
        ], widths="equal"),
        mo.hstack([
            _card("**How it compares with SPY.** On 21 October SPY's own half-spreads are below this rule at every grid "
                  "point: the rule is a floor, wider than the market's spread."),
            _card(r"**What it does.** It is the yardstick of the calibration. Least squares weights each miss by "
                  r"$1/\sigma_c^2$, and the posterior's width scales with it. Slot 6 changes it."),
        ], widths="equal"),
    ])

    return


@app.cell(hide_code=True)
def _(mo, np, plots, plt, spy_noise, surface, teaching):
    _deltas = np.array(surface.DESIGN_DELTAS)
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(9.6, 3.2), constrained_layout=True)
    _line = np.linspace(-2, 2, 81)
    _left.plot(_line, 100 * surface.implied_vol_noise(_line), color=plots.INK_2, linewidth=1.4)
    _left.scatter(_deltas, 100 * surface.implied_vol_noise(_deltas), s=16, color=plots.SERIES[0], zorder=3)
    _left.set_ylim(0, 1.2)
    _left.set_xlabel("standardized moneyness δ")
    _left.set_ylabel("volatility points")
    _left.set_title("step 1: the error in volatility, s(δ)", loc="left")
    _noise = spy_noise.reshape(len(surface.DESIGN_DAYS), len(_deltas))
    for _row, _day in enumerate(surface.DESIGN_DAYS):
        _right.plot(_deltas, 1e4 * _noise[_row], "o-", markersize=3, linewidth=1.1, color=plt.cm.viridis(_row / 6),
                    label=teaching.TENORS[_day])
    _right.set_yscale("log")
    _right.set_xlabel("standardized moneyness δ")
    _right.set_ylabel("basis points of the forward")
    _right.set_title("step 2: times the vega at 20%, the price noise σ_c", loc="left")
    _right.legend(fontsize=7, ncols=2, loc="lower center")
    mo.vstack([
        _fig,
        mo.md(f"""
    The price noise runs from {1e4 * spy_noise.min():.1f} to {1e4 * spy_noise.max():.0f} basis points of the forward.
    It grows with maturity, like the vega, and peaks a little away from the money, where the error in volatility has
    grown and the vega has not yet fallen much.
    """),
    ])

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
    mo.vstack([
        mo.md("## 3 · Bayes with one parameter"),
        mo.callout(mo.md(r"""
    **In one line.** Least squares and Bayes start from the same curve, the squared miss $\chi^2(\sigma)$. Least squares
    keeps its minimum; Bayes keeps all of it, weighted by a prior. With one liquid quote they give the same answer.
    """), kind="info"),
        mo.md(r"""
    ### A model turns parameters into prices

    - Black–Scholes has one parameter, the volatility $\sigma$, and gives every option the same implied volatility.
      It cannot match SPY's smile, and section 5 shows how a check of the model detects this. But with one parameter
      each step of Bayesian inference fits in one figure.
    - We start with one quote: the at-the-money option that expires in one month.
    - Calibration goes backwards, from the quote to $\sigma$. Least squares chooses the $\sigma$ that minimizes the
      squared miss in noise units, summed over the quotes used:

    $$\chi^2(\sigma) = \sum_c \frac{\big(y_c - P_c(\sigma)\big)^2}{\sigma_c^2}.$$
    """),
    ])

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

    - **Likelihood.** If the noise is Gaussian, the probability of the quote given $\sigma$ is proportional to
      $e^{-\chi^2(\sigma)/2}$: the least-squares curve, read as a probability.
    - **Prior.** Which volatilities we consider possible before the quote. By default every value from 5% to 60% is
      equally likely; the buttons change it.
    - **Posterior.** Their product, rescaled to integrate to one, computed exactly on a grid of 4,001 volatilities:

    $$p(\sigma \mid y) = \frac{p(y \mid \sigma)\, p(\sigma)}{p(y)}.$$
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

    Each quote adds a factor to the likelihood. To see the effect where the model is right, we simulate one
    at-the-money quote per maturity from Black–Scholes itself, at a known $\sigma$ = 13%, with the same noise. Simulated
    data with a known truth is also how every method of the afternoon is tested.
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
    mo.vstack([
        mo.md("## 4 · Using the posterior: the range of a price"),
        mo.callout(mo.md(r"""
    **In one line.** Price the product under each posterior draw and read off its quantiles. Here $\sigma$'s posterior
    is exactly the Gauss–Newton normal, but the put's price is not normal: linearizing the price, the delta method,
    misses its skew and understates its 90% point.
    """), kind="info"),
        mo.md(r"""
    We take the one-month put at $\delta = -2$, far below the money, and suppose the market does not quote it, as for
    the products a desk has to mark.

    - **The classical route** has two steps: the Gauss–Newton normal for $\sigma$, centred at the least-squares fit
      with standard deviation equal to the noise divided by the vega; then the delta method for the price, a normal
      whose standard deviation is the price's slope in $\sigma$ times that of $\sigma$.
    - **The Bayesian route** prices the put at each posterior draw of $\sigma$.
    """),
    ])

    return


@app.cell
def _(np, one_quote, quotes, sigma_grid, standardized_miss, surface):
    _rng = np.random.default_rng(1)
    _cdf = np.cumsum(one_quote) / np.sum(one_quote)
    _index = np.minimum(np.searchsorted(_cdf, _rng.uniform(size=4_000)), len(sigma_grid) - 1)
    sigma_draws = sigma_grid[_index] + (_rng.uniform(size=_index.size) - 0.5) * (sigma_grid[1] - sigma_grid[0])
    put_draws = surface.bs_price(quotes["put"]["k"], quotes["put"]["T"], sigma_draws, "put")
    put_range = np.quantile(put_draws, [0.05, 0.95])

    # the classical route: Gauss-Newton for sigma, the delta method for the put
    _atm, _put = quotes["at the money"], quotes["put"]
    sigma_hat = sigma_grid[np.argmin(standardized_miss(_atm) ** 2)]  # least squares
    sigma_gn_sd = _atm["sd"] / float(surface.bs_vega(_atm["k"], _atm["T"], sigma_hat))  # the noise over the vega
    put_mark = float(surface.bs_price(_put["k"], _put["T"], sigma_hat, "put"))
    put_delta_sd = float(surface.bs_vega(_put["k"], _put["T"], sigma_hat)) * sigma_gn_sd  # slope in sigma times its sd
    put_gn_draws = surface.bs_price(_put["k"], _put["T"], np.random.default_rng(3).normal(sigma_hat, sigma_gn_sd, 4_000),
                                    "put")

    return (
        put_delta_sd,
        put_draws,
        put_gn_draws,
        put_mark,
        put_range,
        sigma_draws,
        sigma_gn_sd,
        sigma_hat,
    )


@app.cell(hide_code=True)
def _(
    mo,
    np,
    plots,
    plt,
    put_delta_sd,
    put_draws,
    put_gn_draws,
    put_mark,
    put_range,
    sigma_draws,
    sigma_gn_sd,
    sigma_hat,
    stats,
):
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(9.6, 3.2), constrained_layout=True)
    _left.hist(100 * sigma_draws, bins=40, density=True, color=plots.SERIES[0], alpha=0.8, edgecolor=plots.SURFACE,
               linewidth=0.4, label="posterior draws")
    _x = np.linspace(sigma_hat - 4 * sigma_gn_sd, sigma_hat + 4 * sigma_gn_sd, 200)
    _left.plot(100 * _x, stats.norm.pdf(_x, sigma_hat, sigma_gn_sd) / 100, color=plots.INK, linestyle="--", linewidth=1.3,
               label="Gauss–Newton normal")
    _left.set_yticks([])
    _left.set_xlabel("σ (%)")
    _left.set_title("σ: the posterior and Gauss–Newton coincide", loc="left")
    _left.legend(fontsize=7.5, loc="upper right")
    _bins = np.linspace(0, 0.25, 51)
    _right.hist(1e4 * put_draws, bins=_bins, density=True, color=plots.SERIES[1], alpha=0.8, edgecolor=plots.SURFACE,
                linewidth=0.4, label="priced at posterior draws")
    _right.hist(1e4 * put_gn_draws, bins=_bins, density=True, histtype="step", color=plots.INK_2, linewidth=1.1,
                label="priced at Gauss–Newton draws")
    _y = np.linspace(1e4 * (put_mark - 4 * put_delta_sd), 1e4 * (put_mark + 4 * put_delta_sd), 200)
    _right.plot(_y, stats.norm.pdf(_y, 1e4 * put_mark, 1e4 * put_delta_sd), color=plots.INK, linestyle="--", linewidth=1.3,
                label="delta method")
    _q90 = np.quantile(put_draws, 0.9)
    _d90 = put_mark + 1.2816 * put_delta_sd
    _right.axvline(1e4 * _q90, color=plots.SERIES[1], linewidth=1.2, label="90% point, posterior")
    _right.axvline(1e4 * _d90, color=plots.INK, linewidth=1.2, linestyle=":", label="90% point, delta method")
    _right.set_yticks([])
    _right.set_xlim(0, 0.22)
    _right.set_xlabel("put price (basis points of the forward)")
    _right.set_title("the put: skewed, which the delta method misses", loc="left")
    _right.legend(fontsize=7, loc="upper right")
    mo.vstack([
        _fig,
        mo.hstack([
            mo.stat(value=f"{100 * sigma_draws.mean():.2f}% ± {100 * sigma_draws.std():.2f}", label="σ, posterior",
                    caption="mean ± standard deviation", bordered=True),
            mo.stat(value=f"{100 * sigma_hat:.2f}% ± {100 * sigma_gn_sd:.2f}", label="σ, Gauss–Newton",
                    caption="least squares ± noise / vega", bordered=True),
            mo.stat(value=f"{100 * (_q90 / put_mark - 1):.0f}%", label="put's reserve, posterior",
                    caption="90% point over the mark, sold put", bordered=True),
            mo.stat(value=f"{100 * (_d90 / put_mark - 1):.0f}%", label="put's reserve, delta method",
                    caption="the same, linearized", bordered=True),
        ], widths="equal"),
        mo.md(f"""
    - The put's price is convex in σ: a draw above the centre raises it more than a draw below lowers it. Its
      distribution is skewed to the right, whether σ comes from the posterior or from the Gauss–Newton normal (outline).
    - The delta method makes the price normal. Its 90% point, {1e4 * _d90:.3f} bp, is below the posterior's,
      {1e4 * _q90:.3f} bp: for a bank that has sold the put, the reserve would be too small.
    - Under the posterior the put's 90% range is {1e4 * put_range[0]:.2f} to {1e4 * put_range[1]:.2f} basis points
      of the forward. Section 6 compares the two routes on Heston and four exotic products.
    """),
    ])

    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **What the rules ask for.** This range, and its 90% point, are what EU prudent valuation asks a bank to estimate for
    each position: Art. 9 for calibrated parameters, Art. 11 for the choice of model (section 1).
    """), kind="neutral")

    return


@app.cell(hide_code=True)
def _(mo, put_range, quotes):
    mo.vstack([
        mo.md("## 5 · Checking the model"),
        mo.callout(mo.md(f"""
    **In one line.** A posterior is computed inside its model, so it cannot show that the model is wrong. A predictive
    check can: the market quotes the put at {1e4 * quotes['put']['price']:.1f} basis points, about
    {quotes['put']['price'] / put_range[1]:.0f} times the top of the predicted range. Black–Scholes cannot produce a smile.
    """), kind="warn"),
        mo.md("A posterior predictive check compares what the fitted model predicts with quotes it was not fitted to."),
    ])

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
    - The two likelihoods do not overlap. The posterior from both quotes stays near the at-the-money value, the more
      precise quote, and misses the put by {misses['put']:.1f} noise units.
    - Only about {noise_needed:.2g} times the assumed noise would bring both misses below two units.
    - No single volatility prices both options: this mismatch is the smile. The posterior is precise but wrong,
      because the model cannot produce these quotes. A model has to be checked before its ranges are used, which is
      why the afternoon moves to models with a smile: Heston (section 6 and slot 4) and rough Bergomi (slot 5).
    """),
    ])
    return


@app.cell(hide_code=True)
def _(gn_reserves, mo):
    _thin = [_k for _k in gn_reserves if _k[0] == "15 quotes, κ free"]
    _worst = max(abs(gn_reserves[_k]["delta method"] / gn_reserves[_k]["sold"] - 1) for _k in _thin)
    mo.vstack([
        mo.md("## 6 · A model with a smile: Heston"),
        mo.callout(mo.md(f"""
    **In one line.** On SPY's 63 quotes, Heston's posterior is Gaussian down to its tails, and the classical route,
    Gauss–Newton and the delta method, gives the same reserves. With 15 quotes the posterior bends, a tail grows, and
    the classical reserves differ from the posterior's by up to {100 * _worst:.0f}%. The test is the reserve, not the
    shape of a scatter plot.
    """), kind="info"),
        mo.md(r"""
    Heston lets the variance of SPY move randomly: it reverts to a long-run level and moves against the price. Its
    parameters map onto the smile's features of section 2.

    | Parameter | What it sets |
    |:---|:---|
    | initial variance $v_0$ | the level at short maturities |
    | long-run variance $\theta$ | the level at long maturities |
    | mean reversion $\kappa$, fixed at 2 | how fast the level moves from $v_0$ to $\theta$ |
    | correlation $\rho$, times the volatility of variance | the skew |
    | volatility of variance $\sigma_v$ | the curvature |

    Rough Bergomi, the model of slot 5, describes short maturities better still, but it can only be priced by Monte Carlo.
    """),
    ])

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

    - Sections 3 and 4 carry over: the likelihood is the same product, now over all 63 quotes and four parameters, and
      a product is priced at each posterior draw.
    - With four parameters a grid of values is no longer practical. Markov chain Monte Carlo (MCMC) samples the
      posterior instead: a chain of parameter values that visits each region in proportion to its posterior
      probability.
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
def _(heston_checks, mo):
    _widths = [_p["width_ratio_gn_over_posterior"] for _p in heston_checks["cases"]["full grid, kappa fixed"]["parameters"].values()]
    _card = lambda text: mo.callout(mo.md(text), kind="neutral")  # noqa: E731
    mo.vstack([
        mo.md(r"""
    ### Gauss–Newton or the posterior?

    The classical route approximates the posterior by a normal centred at the least-squares fit, with the covariance
    $(J^\top W J)^{-1}$ given by the curvature of $\chi^2$, and each product's range by the delta method. An elliptical
    scatter is not enough to trust it. Three checks, with the exact posterior on SPY's quotes:
    """),
        mo.hstack([
            _card(f"**1. Centre and width.** Do the Gauss–Newton intervals match the posterior's? On 63 quotes their "
                  f"widths are {min(_widths):.2f} to {max(_widths):.2f} times the posterior's."),
            _card("**2. Tails.** Do the quantiles match out to the 1% and 99% points, not only near the centre? "
                  "The next figure compares them, parameter by parameter."),
            _card("**3. The reserve.** Do the products' 90% points match? This is the number the rules ask for, so it "
                  "decides whether a difference matters."),
        ], widths="equal"),
    ])

    return


@app.cell
def _(CHECKS, PRODUCTS, heston_checks, heston_spy_draws, np, stats):
    gn_cases = {  # the exact posterior's draws, and the lab quick check's Gauss-Newton summary of the same quotes
        "63 quotes, κ = 2": (heston_spy_draws, heston_checks["cases"]["full grid, kappa fixed"]),
        "15 quotes, κ free": (np.load(CHECKS / "heston_checks-3_x_5_grid_kappa_free.npy"),
                              heston_checks["cases"]["3 x 5 grid, kappa free"]),
    }
    PARAMETER_LABELS = {"initial_variance": "v₀", "theta": "θ", "vol_of_vol": "σ_v", "rho": "ρ", "kappa": "κ"}
    _p = np.linspace(0.01, 0.99, 99)
    gn_quantiles, gn_shape = {}, {}  # quantiles in Gauss-Newton sd from the least-squares fit; skewness and kurtosis
    for _case, (_draws, _summary) in gn_cases.items():
        for _i, _name in enumerate(_summary["names"]):
            _parameter = _summary["parameters"][_name]
            _sd = np.ptp(_parameter["gauss_newton_90"]) / (2 * 1.645)
            gn_quantiles[_case, _name] = (np.quantile(_draws[:, _i], _p) - _parameter["best"]) / _sd
            gn_shape[_case, _name] = (float(stats.skew(_draws[:, _i])), float(stats.kurtosis(_draws[:, _i])))
    gn_reserves, gn_precision = {}, {}  # reserves as shares of the mark, and the bootstrap precision of the posterior's
    _rng = np.random.default_rng(0)
    for _case, (_, _summary) in gn_cases.items():
        _prices = np.asarray(_summary["posterior_product_draws"])  # 300 draws, columns in the order of PRODUCTS
        for _j, _product in enumerate(PRODUCTS):
            _info = _summary["products"][_product]
            _x = _prices[:, _j] / _info["least_squares_point"]
            gn_reserves[_case, _product] = {"sold": np.quantile(_x, 0.9) - 1, "held": 1 - np.quantile(_x, 0.1),
                                            "delta method": 1.2816 * _info["delta_range_90_relative"] / (2 * 1.645)}
            _boot = np.quantile(_x[_rng.integers(0, len(_x), (2000, len(_x)))], [0.1, 0.9], axis=1)
            gn_precision[_case, _product] = float(max(np.std(_boot[0]), np.std(_boot[1])))

    return (
        PARAMETER_LABELS,
        gn_cases,
        gn_precision,
        gn_quantiles,
        gn_reserves,
        gn_shape,
    )


@app.cell(hide_code=True)
def _(
    PARAMETER_LABELS,
    gn_cases,
    gn_quantiles,
    gn_shape,
    heston_checks,
    mo,
    np,
    plots,
    plt,
    stats,
):
    _z = stats.norm.ppf(np.linspace(0.01, 0.99, 99))
    _colors = list(plots.SERIES) + [plots.INK_2, plots.CRITICAL]
    _fig, _axes = plt.subplots(1, 2, figsize=(9.6, 3.4), constrained_layout=True, sharey=True)
    for _ax, (_case, (_, _summary)) in zip(_axes, gn_cases.items()):
        _ax.plot([-3, 3], [-3, 3], color=plots.BASELINE, linewidth=1)
        for _name, _color in zip(_summary["names"], _colors):
            _ax.plot(_z, gn_quantiles[_case, _name], color=_color, linewidth=1.4, label=PARAMETER_LABELS[_name])
        _ax.set_xlim(-2.6, 2.6)
        _ax.set_ylim(-3.2, 5.5)
        _ax.set_xlabel("quantile of the Gauss–Newton normal (sd)")
        _ax.set_title(f"Heston on SPY, {_case}", loc="left")
        _ax.legend(fontsize=7.5, loc="upper left")
    _axes[0].set_ylabel("quantile of the exact posterior (sd)")
    _liquid = max(max(abs(_a), abs(_b)) for (_case, _), (_a, _b) in gn_shape.items() if _case == "63 quotes, κ = 2")
    _theta = gn_shape["15 quotes, κ free", "theta"]
    _kappa = heston_checks["cases"]["3 x 5 grid, kappa free"]["parameters"]["kappa"]["gauss_newton_90"]
    mo.vstack([
        _fig,
        mo.md(f"""
    Each line compares one parameter's posterior quantiles, from the 1% to the 99% point, with the Gauss–Newton normal's.
    On the diagonal the two agree, tails included.

    - **63 quotes, κ = 2:** every line lies on the diagonal, and every skewness and excess kurtosis is within
      ±{_liquid:.2f}. The posterior is the Gauss–Newton normal, out to the tails.
    - **15 quotes, κ free:** θ's right tail reaches {gn_quantiles['15 quotes, κ free', 'theta'][-1]:.1f} Gauss–Newton
      standard deviations at the 99% point, with skewness {_theta[0]:.1f}. κ is cut by the prior's lower bound, while
      its Gauss–Newton interval reaches {_kappa[0]:.2f}, a negative mean reversion that the model does not allow.
    """.replace("reaches -", "reaches −")),
    ])

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A curved posterior, and why

    In Heston the expected variance at time $t$ is $\theta + (v_0 - \theta)\,e^{-\kappa t}$: it starts at $v_0$ and moves
    to the level $\theta$ at the speed $\kappa$. Fifteen quotes up to six months pin its **average up to six months**,
    but not how that average splits between the level and the speed. Every pair on the dashed curve gives the same
    average, and the exact posterior lies along it (left). Gauss–Newton replaces the curve by a box that misses its
    upper arm and reaches negative $\kappa$.

    Rough Bergomi's variance curve has the same form, with a short level, a long level and a decay. When the two levels
    nearly coincide, the decay barely changes the curve, and the network's posterior follows the same kind of dashed
    curve (right).
    """)
    return


@app.cell
def _(ASSETS, RESULTS, backbones, np, teaching):
    _task, rough_report, rough_cache = backbones.load_example("rough-joint100k-validation1962")
    rough_names = _task["parameter_names"]
    _evaluation = teaching.load_json(ASSETS / "backbones" / "rough-joint100k-validation1962" / "evaluation.json")
    rough_coverage_90 = dict(zip(_evaluation["targets"], _evaluation["coverage"]["0.9"]["coverage"]))  # 2,000 smiles
    rough_decay_cases = []  # how much of its prior the decay's 90% interval covers, by the ratio of the curve's levels
    for _case in ("rough-joint100k-validation0", "rough-joint100k-validation1941", "rough-joint100k-validation1962",
                  "rough-joint100k-validation1983"):
        _t, _, _c = backbones.load_example(_case)
        _n = _t["parameter_names"]
        _truth = dict(zip(_n, _c["truth"]))
        _q = np.quantile(_c["neural_draws"][:, _n.index("decay")], [0.05, 0.95])
        rough_decay_cases.append({"levels": _truth["short_variance"] / _truth["long_variance"],
                                  "share": np.ptp(_q) / np.ptp(np.asarray(_t["prior_bounds"])[_n.index("decay")])})
    heston_s6 = teaching.load_json(RESULTS / "heston" / "s6" / "classical.json")["summary"]

    return (
        heston_s6,
        rough_cache,
        rough_coverage_90,
        rough_decay_cases,
        rough_names,
    )


@app.cell(hide_code=True)
def _(
    gn_cases,
    mo,
    np,
    plots,
    plt,
    rough_cache,
    rough_coverage_90,
    rough_decay_cases,
    rough_names,
):
    def _average(start, level, decay, horizon):
        """The variance curve start + (level - start)(1 - e^(-decay t)), averaged from 0 to horizon."""
        _g = (1 - np.exp(-decay * horizon)) / (decay * horizon)
        return level + (start - level) * _g


    def _same_average(start, target, decay, horizon):
        """The level that gives this average for each decay, the start held fixed."""
        _g = (1 - np.exp(-decay * horizon)) / (decay * horizon)
        return (target - start * _g) / (1 - _g)


    _draws, _summary = gn_cases["15 quotes, κ free"]
    _v0, _theta, _kappa = _draws[:, 0], _draws[:, 1], _draws[:, 4]
    _speeds = np.linspace(0.25, 8, 200)
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(9.6, 3.6), constrained_layout=True)
    _left.scatter(_kappa, _theta, s=3, alpha=0.25, color=plots.SERIES[0], rasterized=True, label="exact posterior (MCMC)")
    _left.plot(_speeds, _same_average(_v0.mean(), _average(_v0, _theta, _kappa, 0.5).mean(), _speeds, 0.5),
               color=plots.INK, linewidth=1.3, linestyle="--", label="same average variance to 6 months")
    _gk = _summary["parameters"]["kappa"]["gauss_newton_90"]
    _gt = _summary["parameters"]["theta"]["gauss_newton_90"]
    _left.add_patch(plt.Rectangle((max(_gk[0], 0), _gt[0]), _gk[1] - max(_gk[0], 0), _gt[1] - _gt[0], fill=False, edgecolor=plots.SERIES[1],
                                  linewidth=1.3, label=f"Gauss–Newton 90% box (κ from {_gk[0]:.2f})".replace("-", "−")))
    _left.set_xlim(0, 8.2)
    _left.set_ylim(0, 0.17)
    _left.set_xlabel("mean reversion κ (the speed)")
    _left.set_ylabel("long-run variance θ (the level)")
    _left.set_title("Heston, 15 SPY quotes: exact posterior", loc="left")
    _left.legend(fontsize=7.2, loc="upper right")
    _short, _long, _decay = (rough_cache["neural_draws"][:, rough_names.index(_n)]
                             for _n in ("short_variance", "long_variance", "decay"))
    _right.scatter(_decay, _long, s=4, alpha=0.4, color=plots.SERIES[1], rasterized=True, label="network posterior")
    _right.plot(_speeds, _same_average(_short.mean(), _average(_short, _long, _decay, 1.0).mean(), _speeds, 1.0),
                color=plots.INK, linewidth=1.3, linestyle="--", label="same average variance to 1 year")
    _right.set_xlim(0, 8.2)
    _right.set_ylim(0.04, 0.1)
    _right.set_xlabel("decay (the speed, per year)")
    _right.set_ylabel("long level of the variance curve")
    _right.set_title("rough Bergomi, almost flat curve: network", loc="left")
    _right.legend(fontsize=7.2, loc="lower right")
    _pinned = [_c for _c in rough_decay_cases if abs(np.log(_c["levels"])) > 0.3]
    mo.vstack([
        _fig,
        mo.callout(mo.md(f"""
    **Is the rough Bergomi shape real?** The dashed curves come from the model's formula, not from the networks or the
    sampler, and both posteriors follow them: the shape is the model's geometry. The Heston posterior is exact
    (MCMC on real SPY quotes). The rough Bergomi draws are the network's, and two further facts support them: in the
    cached cases whose two levels differ, the network pins the decay to {min(_c['share'] for _c in _pinned):.0%} to
    {max(_c['share'] for _c in _pinned):.0%} of its prior range, and on 2,000 simulated smiles its 90% intervals for the
    decay contain the truth {rough_coverage_90['decay']:.0%} of the time. What is missing is an exact sampler on this
    smile (pending, J2).
    """), kind="neutral"),
    ])

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What it does to a reserve

    The rules ask for the 90% point of each product's price, so the two routes have to agree there. Below, two of the
    four products, under the exact posterior (bars) and under the classical route (dashed). The solid and dashed
    vertical lines are each route's 10% and 90% points: the reserve for a position held and for one sold is the
    distance from the mark (0) to these lines.
    """)
    return


@app.cell(hide_code=True)
def _(
    PRODUCTS,
    gn_cases,
    gn_precision,
    gn_reserves,
    mo,
    np,
    plots,
    plt,
    stats,
):
    _fig, _axes = plt.subplots(2, 2, figsize=(9.6, 4.6), constrained_layout=True)
    for _row, (_case, (_, _summary)) in enumerate(gn_cases.items()):
        _prices = np.asarray(_summary["posterior_product_draws"])
        for _col, _product in enumerate(("forward_start_call", "put_out_of_grid")):
            _ax = _axes[_row, _col]
            _info = _summary["products"][_product]
            _relative = 100 * (_prices[:, list(PRODUCTS).index(_product)] / _info["least_squares_point"] - 1)
            _sd = 100 * _info["delta_range_90_relative"] / (2 * 1.645)
            _ax.hist(_relative, bins=30, density=True, color=plots.SERIES[0], alpha=0.75, edgecolor=plots.SURFACE,
                     linewidth=0.4, label="exact posterior")
            _x = np.linspace(-4 * _sd, 4 * _sd, 200)
            _ax.plot(_x, stats.norm.pdf(_x, 0, _sd), color=plots.INK, linestyle="--", linewidth=1.2,
                     label="Gauss–Newton and the delta method")
            for _q in np.quantile(_relative, [0.1, 0.9]):
                _ax.axvline(_q, color=plots.SERIES[0], linewidth=1.5)
            for _q in (-1.2816 * _sd, 1.2816 * _sd):
                _ax.axvline(_q, color=plots.INK, linewidth=1.2, linestyle="--")
            _ax.axvline(0, color=plots.MUTED, linewidth=0.8)
            _ax.set_yticks([])
            _ax.set_title(f"{PRODUCTS[_product]}, {_case}", loc="left", fontsize=8.5)
            if _row == 1:
                _ax.set_xlabel("price relative to the least-squares mark (%)")
    _fig.legend(*_axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncols=2, fontsize=7.5)


    def _cell(case, product, side):
        _r = gn_reserves[case, product]
        return f"{100 * _r[side]:.1f}% / {100 * _r['delta method']:.1f}%"


    _rows = "\n".join(f"| {PRODUCTS[_p]} | " + " | ".join(_cell(_c, _p, _s) for _c in gn_cases for _s in ("sold", "held")) + " |"
                      for _p in PRODUCTS)
    mo.vstack([
        _fig,
        mo.md(f"""
    | Reserve, posterior / delta method | 63 quotes, sold | 63 quotes, held | 15 quotes, sold | 15 quotes, held |
    |:---|:---|:---|:---|:---|
    {_rows}

    Shares of the least-squares mark. The posterior's reserves come from 300 draws, which fix them to about
    ±{100 * min(gn_precision.values()):.1f} to ±{100 * max(gn_precision.values()):.1f} points (bootstrap).
    """),
    ])

    return


@app.cell(hide_code=True)
def _(PRODUCTS, gn_precision, gn_reserves, heston_checks, heston_s6, mo):
    _liquid = [_k for _k in gn_reserves if _k[0] == "63 quotes, κ = 2"]
    _z = max(abs(gn_reserves[_k][_side] - gn_reserves[_k]["delta method"]) / gn_precision[_k]
             for _k in _liquid for _side in ("sold", "held"))
    _thin = [_k for _k in gn_reserves if _k[0] == "15 quotes, κ free"]
    _gaps = {_k: gn_reserves[_k]["delta method"] / gn_reserves[_k]["sold"] - 1 for _k in _thin}
    _low, _high = min(_gaps, key=_gaps.get), max(_gaps, key=_gaps.get)
    _chains = heston_checks["cases"]["3 x 5 grid, kappa free"]["parameters"]
    _ess = min(_chains[_n]["rhat_ess"][1] for _n in ("theta", "kappa"))
    _coverage = heston_s6["gauss_newton_coverage_90"].values()
    mo.callout(mo.md(f"""
    **When does a difference matter?** Compare reserves, not shapes. A difference matters when it is larger than the
    precision of the reserve itself and larger than the bank's materiality threshold, once added up over the book.

    - **63 quotes:** for every product and both sides, the two routes differ by at most {_z:.1f} times the precision of
      the posterior's reserve: no real difference. Gauss–Newton is enough here, as on 200 simulated Heston smiles, where
      its 90% intervals contain the truth {min(_coverage):.0%} to {max(_coverage):.0%} of the time (claim C11).
    - **15 quotes:** for a sold {PRODUCTS[_low[1]].split(",")[0]} the delta method's reserve is {-100 * _gaps[_low]:.0f}%
      too small, the unsafe direction; for a sold {PRODUCTS[_high[1]].split(",")[0]} it is {100 * _gaps[_high]:.0f}% too
      large. The posterior's prices are skewed, and the delta method makes every product symmetric.

    Status: the 63-quote posterior is the feasibility run of 23 October (not an accepted study; S8-H replaces it). The
    15-quote posterior and the classical reserves come from a lab quick check (`lab/checks/heston_checks.py`), whose
    chains for θ and κ carry about {_ess:.0f} effective draws, so its reserves are less precise than the bootstrap says.
    """), kind="warn")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **When the posterior adds most.** On a liquid smile with a well-identified model, the classical route is enough. The
    posterior adds most when

    - the quotes leave a direction open, as in the curved posteriors above, and a product depends on it;
    - a product is priced beyond the quotes, or its price is far from linear in the parameters (section 4's put);
    - a range has to be reported at a stated confidence level, as the rules ask (section 1);
    - prior knowledge should enter openly, as a prior, instead of as a penalty or a parameter fixed by hand;
    - the result has to be tested by changing the prior or the noise model (slot 6).
    """), kind="success")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("## 7 · What it costs, and amortized inference"),
        mo.callout(mo.md(r"""
    **In one line.** Exact posteriors get expensive as models get realistic, and a desk needs many of them. Amortized
    inference pays once, in training, and then returns a posterior in about a second.
    """), kind="info"),
        mo.md(r"""
    | Method | Needs | In this tutorial |
    |:---|:---|:---|
    | A grid | one or two parameters | Black–Scholes, sections 3 to 5 |
    | A Gaussian approximation | a posterior close to a normal | Heston on a liquid smile, section 6 |
    | MCMC | thousands of likelihood evaluations | Heston's exact posterior: minutes. Rough Bergomi: at least 17 CPU-hours by a random-walk chain, or about 85 CPU-minutes by importance sampling from a Gauss–Newton proposal on SPY's 63 quotes |
    | Amortized inference | one training run on simulated smiles | seconds per smile, for any smile the prior can produce |

    The likelihood can be written down for every model here. The cost is the pricing inside it: about 20 ms per smile
    for Heston's Fourier integral at one quadrature rung, and 5 CPU-seconds per smile for rough Bergomi's Monte Carlo.
    """),
    ])

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

    1. **A calibration is an estimate.** Least squares keeps the peak of the likelihood; the posterior keeps all of it.
       Pricing a product under the posterior gives a range: a few percent of its price on a liquid smile, much more
       with few quotes or beyond them. The rules ask for its 90% point.
    2. **A posterior is only as good as its model.** On SPY, Black–Scholes gives a precise but wrong answer, so the
       model has to be checked before its ranges are used.
    3. **The classical route is right when the quotes inform every direction.** Test it on the reserve, not on the shape
       of a scatter plot: on 63 quotes Gauss–Newton and the posterior agree, on 15 they do not.
    4. **Exact posteriors become expensive** as models become realistic, and many are needed. The rest of the afternoon
       shows how amortized inference reduces this cost.

    Slot 2 shows how a network trained on simulations returns the posterior, starting from the quote of section 3, and
    when this pays off.

    Sources: `CLAIMS.md` (claims C1, C8 and C11), the cached examples in `assets/backbones/` (development examples,
    not accepted studies), the lab quick checks in `assets/quick-checks/` (not accepted studies) and
    `src/sbi4f_tutorial/`. The SPY quotes come from study S8. Results on simulated smiles use the tutorial's models and
    priors, and do not show how well these models describe market data.
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
    return (noise_units,)


@app.cell(hide_code=True)
def _(mo, noise_units, np, plots, plt, spy_iv, surface, teaching):
    _deltas = np.array(surface.DESIGN_DELTAS)
    _fig, _ax = plt.subplots(figsize=(9.6, 3.4), constrained_layout=True)
    _ax.imshow(np.log10(noise_units), cmap="viridis", aspect="auto")
    for (_row, _col), _value in np.ndenumerate(noise_units):
        _ax.text(_col, _row, f"{_value:.1f}" if _value < 10 else f"{_value:.0f}", ha="center", va="center", fontsize=7.5,
                 color="white" if np.log10(_value) < 0.6 else plots.INK)
    _ax.set_xticks(range(len(_deltas)), [f"{_x:g}" for _x in _deltas])
    _ax.set_yticks(range(len(surface.DESIGN_DAYS)), [teaching.TENORS[_d] for _d in surface.DESIGN_DAYS])
    _ax.set_xlabel("standardized moneyness δ")
    _ax.set_title("SPY's prices in noise units, y / σ: what a network reads", loc="left")
    _ax.grid(visible=False)
    _low = np.unravel_index(np.argmin(noise_units), noise_units.shape)
    _high = np.unravel_index(np.argmax(noise_units), noise_units.shape)
    _faint = noise_units.max(axis=0) < 1  # strikes below one noise standard deviation at every maturity
    _faint_iv = 100 * np.asarray(spy_iv).reshape(noise_units.shape)[:, _faint]
    _faint_deltas = " and ".join(f"{_x:g}" for _x in _deltas[_faint])
    mo.vstack([
        _fig,
        mo.md(f"""
    - The option at δ = {_deltas[_low[1]]:g} with {teaching.TENORS[surface.DESIGN_DAYS[_low[0]]]} to expiry is worth
      only {noise_units[_low]:.1f} noise standard deviations; the one at δ = {_deltas[_high[1]]:g} with
      {teaching.TENORS[surface.DESIGN_DAYS[_high[0]]]} to expiry is worth {noise_units[_high]:.0f}.
    - The calls at δ = {_faint_deltas} are worth less than one noise standard deviation at every maturity. SPY's
      implied volatility there, {_faint_iv.min():.0f} to {_faint_iv.max():.0f}%, is well below the 20% at which the rule
      is set, so their prices are small compared with the noise. On their own, these quotes say little.
    """),
    ])

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The variance curve

    The same chain gives the market's variance-swap curve, which the later slots use.

    - For each expiration whose quotes reach far enough into the wings, down to $\delta = -8$ for the puts and up to
      $\delta = 1.8$ for the calls, a log contract is replicated from the whole strip of out-of-the-money options. Its
      price is the total variance that a variance swap to that expiration pays.
    - Read at the grid maturities and differenced, these values give seven forward-variance levels, one for each
      interval between grid maturities.
    - Rough Bergomi takes this curve as an input. In slot 5 it is either given, with these seven levels, or inferred
      with the other parameters. When it is given, the network reads the seven levels alongside the 63 prices; the
      levels vary from one training smile to the next, so the same network works with any curve.
    - In slot 4, Heston's variance swaps are compared with these market values.
    """)

    return


@app.cell
def _(np, spy, surface):
    _knots = np.r_[0, surface.DESIGN_DAYS] / surface.TRADING_DAYS
    forward_variance = np.asarray(spy["forward_variance_levels"])  # one level per interval between grid maturities
    variance_swap_vol = 100 * np.sqrt(np.cumsum(forward_variance * np.diff(_knots)) / _knots[1:])

    return forward_variance, variance_swap_vol


@app.cell(hide_code=True)
def _(
    forward_variance,
    mo,
    np,
    plots,
    plt,
    spy_expirations,
    surface,
    variance_swap_vol,
):
    _days = np.array(surface.DESIGN_DAYS)
    _fig, _ax = plt.subplots(figsize=(9.6, 3.2), constrained_layout=True)
    _ax.stairs(100 * np.sqrt(forward_variance), np.r_[0, _days], baseline=None, color=plots.SERIES[0], linewidth=1.6,
               label="forward volatility on each interval")
    _ax.set_ylim(0, 26)
    _ax.plot(_days, variance_swap_vol, "o", color=plots.SERIES[1], markersize=5, label="variance swap to each maturity")
    _ax.set_xlabel("maturity, trading days")
    _ax.set_ylabel("volatility (%)")
    _ax.set_title("SPY's forward-variance curve and variance swaps", loc="left")
    _ax.legend(fontsize=7.5, loc="lower right")
    _covered = sum(_e["replication_covered"] for _e in spy_expirations)
    mo.vstack([
        _fig,
        mo.md(f"""
    The steps are the forward volatility on each interval between grid maturities, the square root of the forward
    variance. The dots are the variance-swap volatility to each maturity, the square root of the average forward
    variance up to it. They come from the {_covered} expirations, out of {len(spy_expirations)}, whose quotes reach far
    enough into the wings.
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
