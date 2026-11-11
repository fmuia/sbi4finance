import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="Bayesian calibration of the Heston model on SPX options",
)


@app.cell(hide_code=True)
def _():
    import hashlib
    import json
    import logging
    import os
    import sys
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm

    LAB = mo.notebook_dir().parent
    if str(LAB) not in sys.path:
        sys.path.insert(0, str(LAB))  # lab notebooks import lab scripts; tutorial code never does this

    logging.disable(logging.INFO)  # keep backend start-up messages out of the rendered notebook
    import bayesflow as bf
    import keras

    from sbi4f_tutorial.shared import plots, surface
    from sbi4f_tutorial.shared.teaching import pairs
    from sbi4f_tutorial.shared.whitening import fit_whitening
    from sbi4finance.calibration.heston_grid import HestonGridPilot
    from scripts import audit_heston_grid_coverage as grid_rules  # the contract-selection rules of the grid
    from scripts import heston_grid_collaborator as delivery
    from scripts import reproduce_heston_pr87_grid as recipe

    _threads = {_key: os.environ.get(_key) for _key in
                ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    from scripts.run_heston_spread_reference import expected_variance  # this import pins BLAS threads for MCMC workers
    for _key, _value in _threads.items():  # restore the notebook's own thread settings
        if _value is None:
            os.environ.pop(_key, None)
        else:
            os.environ[_key] = _value
    logging.disable(logging.NOTSET)

    EVIDENCE = LAB / "evidence" / "pilot-results"
    GRID = EVIDENCE / "2026-11-09" / "heston-spread-baseline" / "revised-grid"
    EVALUATION = GRID / "recovery-bank-independent" / "300k-300" / "evaluation"
    REFERENCES = EVIDENCE / "2026-11-11" / "heston-actual63-notebook"
    ESTIMATOR = "300k-300-final300"  # trained for 300 epochs on 300,000 simulated data sets
    BUNDLE = os.environ.get("HESTON_BUNDLE") or None
    # Seeds and draw counts of the saved evaluation, so that fresh draws are directly comparable with it.
    EVAL_SEED, CONDITIONAL_DRAWS, PANEL_DRAWS, PANEL_BATCH = 261112, 4096, 512, 16

    BUNDLE_HELP = mo.callout(mo.md(
        "**The pretrained estimator is not loaded.** Set `HESTON_BUNDLE` to the folder that contains `bundle.json` "
        "and restart marimo; `lab/plans/training-data-guide.md` explains how to obtain and check it. Until then, "
        "results that need the trained network are left out and the PyMC reference is shown on its own."), kind="warn")

    STAGES = (
        ("Model", "Heston dynamics, a prior over six parameters and quote noise set by the bid–ask spread"),
        ("Simulate", "draw parameters, price the 63 options and add quote noise, 300,000 times"),
        ("Train", "a flow-matching network learns the posterior of the parameters given the quotes"),
        ("Infer", "condition on observed quotes and draw from the posterior in seconds"),
        ("Check", "exact reference, posterior predictive, calibration across simulations, real data"),
    )

    def workflow(active=None):
        """The five workflow stages as boxes; `active` highlights one stage, `None` shows all with descriptions."""
        boxes = []
        for _number, (_name, _text) in enumerate(STAGES, start=1):
            _current = _name == active
            _border = "2px solid #2a78d6" if _current else ("1px solid #2a78d6" if active is None else "1px solid #d9d8d0")
            _fill = "rgba(42,120,214,0.12)" if _current else ("rgba(42,120,214,0.05)" if active is None else "transparent")
            _color = "inherit" if active is None or _current else "#9a9890"
            _detail = (f'<div style="font-size:0.8em;line-height:1.35;margin-top:5px">{_text}</div>'
                       if active is None else "")
            boxes.append(f'<div style="flex:1;min-width:118px;border:{_border};border-radius:10px;padding:9px 10px;'
                         f'background:{_fill};color:{_color};text-align:center">'
                         f'<div style="font-weight:600;letter-spacing:0.02em">{_number}. {_name}</div>{_detail}</div>')
        _arrow = '<div style="align-self:center;color:#9a9890;font-size:1.3em;padding:0 1px">&#8594;</div>'
        return mo.Html('<div style="display:flex;gap:5px;align-items:stretch;margin:4px 0 6px 0">'
                       + _arrow.join(boxes) + "</div>")

    plots.style()
    return (
        BUNDLE,
        BUNDLE_HELP,
        CONDITIONAL_DRAWS,
        ESTIMATOR,
        EVALUATION,
        EVAL_SEED,
        GRID,
        HestonGridPilot,
        PANEL_BATCH,
        PANEL_DRAWS,
        REFERENCES,
        bf,
        delivery,
        expected_variance,
        fit_whitening,
        grid_rules,
        hashlib,
        json,
        keras,
        mo,
        norm,
        np,
        pairs,
        plots,
        plt,
        recipe,
        surface,
        time,
        workflow,
    )


@app.cell(hide_code=True)
def _(mo, workflow):
    mo.vstack([
        mo.md(r"""
        # Bayesian Calibration of the Heston Model on SPX Options

        In this notebook we calibrate the **Heston stochastic-volatility model** to **63 SPX option quotes**. Rather than a single best-fitting parameter vector, we compute a **posterior distribution**: the parameter values consistent with the quotes, given the model and the size of the quote errors. We write $\boldsymbol\theta=(v_0,\kappa,v_\infty,\sigma_v,\rho,c)$ for the six unknown parameters, introduced in Section 1.

        The posterior is learned with **amortized simulation-based inference** in BayesFlow. A neural network is trained once on simulated quotes; afterwards it returns the posterior for any new set of 63 quotes in seconds. We follow five steps, and at the end we check the network against an exact numerical reference, across many simulated markets and on the real quotes.
        """),
        workflow(),
    ])
    return


@app.cell(hide_code=True)
def _(GRID, HestonGridPilot, json, np, plt, recipe, surface):
    task = recipe.read_task(GRID / "cv-preparation" / "task.json")
    pilot = HestonGridPilot(task["records"], infer_c=True, c_bounds=(0.5, 2.0))
    pilot.validate_task(task["target"])
    NAMES = list(pilot.names)
    BOUNDS = np.asarray(pilot.bounds)
    H = np.asarray(pilot.records["half_spread"])
    LABELS = [r"$v_0$", r"$\kappa$", r"$v_\infty$", r"$\sigma_v$", r"$\rho$", r"$c$"]
    TEXT_LABELS = ["initial variance v₀", "mean reversion κ", "long-run variance v∞",
                   "volatility of variance σᵥ", "correlation ρ", "noise multiplier c"]

    design = json.loads((GRID / "cv-preparation" / "design.json").read_text())
    QUOTES = design["selected_quotes"]
    _records = pilot.records
    FORWARD, STRIKE, DISCOUNT, MATURITY = (np.asarray(_records[_k]) for _k in ("forward", "strike", "discount", "maturity"))
    KIND = np.asarray(_records["option_type"])
    MID = np.asarray(_records["mid"])
    # Standardized moneyness on the scale the 63 contracts were selected on (the grid's design record).
    REF_VOL = float(design["reference_volatility"])
    DELTA = np.log(STRIKE / FORWARD) / (REF_VOL * np.sqrt(MATURITY))
    if not np.allclose(DELTA, [_q["z_fixed16"] for _q in QUOTES], rtol=0, atol=1e-9):
        raise ValueError("standardized moneyness differs from the grid's design record")
    EXPIRY_DATES = sorted({_q["expiration"] for _q in QUOTES})
    EXPIRY = np.asarray([EXPIRY_DATES.index(_q["expiration"]) for _q in QUOTES])
    EXPIRY_DAYS = [int(round(365 * MATURITY[EXPIRY == _e][0])) for _e in range(len(EXPIRY_DATES))]
    EXPIRY_COLORS = list(plt.cm.viridis(np.linspace(0, 0.9, len(EXPIRY_DATES))))

    def implied_vols(prices):
        """Black-76 implied volatility for rows of 63 prices, using each quote's F, D, K and T."""
        rows = np.atleast_2d(np.asarray(prices, dtype=float))
        return surface.implied_vol(rows / (DISCOUNT * FORWARD), np.log(STRIKE / FORWARD), MATURITY, KIND)

    def vol_point_noise(sd_prices, implied):
        """Convert a price SD into implied-volatility points with the quote's own Black vega."""
        vega = DISCOUNT * FORWARD * surface.bs_vega(np.log(STRIKE / FORWARD), MATURITY, implied)
        return 100 * sd_prices / vega

    MARKET_IV = implied_vols(MID)[0]
    if recipe.fingerprint(task) != "c43458baf37617d5c2c7bf53b0dc8e3760d0c91efc2eb307c3141670a29fa077":
        raise ValueError("the contract list differs from the one the pretrained estimator was trained on")
    return (
        BOUNDS,
        DELTA,
        DISCOUNT,
        EXPIRY,
        EXPIRY_COLORS,
        EXPIRY_DATES,
        EXPIRY_DAYS,
        FORWARD,
        H,
        KIND,
        LABELS,
        MARKET_IV,
        MATURITY,
        MID,
        NAMES,
        QUOTES,
        REF_VOL,
        TEXT_LABELS,
        design,
        implied_vols,
        pilot,
        task,
        vol_point_noise,
    )


@app.cell(hide_code=True)
def _(GRID, json, np):
    _panel = json.loads((GRID / "cv-preparation" / "conditional-panel.json").read_text())
    _audit = json.loads((GRID / "cv-preparation" / "conditional-truth-audit.json").read_text())
    CASES = {_case["id"]: np.asarray(_case["observations"], dtype=float) for _case in _panel["cases"]}
    TRUTHS = {_case_id: None for _case_id in CASES}
    CLEAN = {}
    ERRORS = {}
    for _case in _audit["cases"]:
        TRUTHS[_case["id"]] = np.r_[_case["physical_truth"], _case["generating_c"]]
        CLEAN[_case["id"]] = np.asarray(_case["clean_prices"], dtype=float)
        ERRORS[_case["id"]] = np.asarray(_case["standard_errors"], dtype=float)
    CASE_TITLES = {"synthetic-market-like": "Simulated quotes, market-like parameters",
                   "synthetic-broad": "Simulated quotes, stressed market",
                   "market": "Market quotes, 9 November 2026"}
    PRODUCTS = _panel["products"]
    PRODUCT_TITLES = {"control": f"average variance, now to {365 * PRODUCTS['control'][1]:.0f} days",
                      "affected": f"forward variance, {365 * PRODUCTS['affected'][0]:.0f} to {365 * PRODUCTS['affected'][1]:.0f} days"}
    return CASES, CASE_TITLES, CLEAN, ERRORS, PRODUCTS, PRODUCT_TITLES, TRUTHS


@app.cell(hide_code=True)
def _(DELTA, KIND, REF_VOL, design, mo):
    _puts = sorted({_q["target_z"] for _q in design["selected_quotes"] if _q["option_type"] == "put"})
    _calls = [_t for _t in design["targets_z"] if _t not in _puts]
    _fmt = lambda _values: ", ".join(f"{_v:g}" for _v in _values)
    mo.md(rf"""
    ## The data: 63 SPX option quotes

    We use one snapshot of SPX index options, taken at 20:13 UTC on 9 November 2026: **63 European, PM-settled SPXW contracts** across seven expiries, from one week to about eleven months, with nine strikes per expiry. For every contract the market shows a **bid** and an **ask**. We record the mid price and the **half-spread** $h_i=(\text{{ask}}_i-\text{{bid}}_i)/2$, and for every expiry the forward price $F$ and discount factor $D$, estimated from put–call parity on the same snapshot. Prices are in index points.

    To compare strikes across expiries we use the **standardized moneyness**

    $$\delta=\frac{{\ln(K/F)}}{{\sigma_{{\text{{ref}}}}\sqrt{{T}}}},\qquad \sigma_{{\text{{ref}}}}={100 * REF_VOL:.0f}\%,$$

    with $T$ the time to expiry in years (actual/365, as in the pricer). It measures the distance from the forward in units of a typical move of the log price over the option's life: a strike 5% below the forward is far out of the money at one week but close to the money at one year, and $\delta$ puts both on one scale. The contracts were selected on this scale: at every expiry, the puts nearest to $\delta={_fmt(_puts)}$ and the calls nearest to $\delta={_fmt(_calls)}$. They run from $\delta={DELTA.min():.1f}$ to $\delta={DELTA.max():.1f}$; {int((KIND == "put").sum())} of the 63 are puts.
    """)
    return


@app.cell(hide_code=True)
def _(
    DELTA,
    DISCOUNT,
    EXPIRY,
    EXPIRY_COLORS,
    EXPIRY_DATES,
    EXPIRY_DAYS,
    FORWARD,
    H,
    KIND,
    MARKET_IV,
    MATURITY,
    MID,
    QUOTES,
    REF_VOL,
    design,
    grid_rules,
    mo,
    np,
    plots,
    plt,
):
    if grid_rules.SIGMA_REF != REF_VOL or list(grid_rules.TARGET_Z) != design["targets_z"]:
        raise ValueError("the selection rules differ from the grid's design record")
    _fig, _axes = plt.subplots(1, 3, figsize=(11.4, 3.3), constrained_layout=True)
    for _kind, _marker in (("put", "v"), ("call", "^")):
        _at = KIND == _kind
        _axes[0].scatter(365 * MATURITY[_at], DELTA[_at], marker=_marker, s=22, color=plots.SERIES[0 if _kind == "put" else 1])
    _axes[0].set(xscale="log", xlabel="days to expiry", ylabel="standardized moneyness δ",
                 title=f"The 63 contracts: {(KIND == 'put').sum()} puts (▼), {(KIND == 'call').sum()} calls (▲)")
    _axes[0].title.set_fontsize(9)
    _axes[0].axhline(0, color=plots.BASELINE, lw=.8)
    for _e, _date in enumerate(EXPIRY_DATES):
        _at = EXPIRY == _e
        _axes[1].plot(DELTA[_at], 100 * H[_at] / MID[_at], "o-", ms=3, lw=1, color=EXPIRY_COLORS[_e])
        _axes[2].plot(DELTA[_at], 100 * MARKET_IV[_at], "o-", ms=3, lw=1, color=EXPIRY_COLORS[_e], label=f"{_date} ({EXPIRY_DAYS[_e]} d)")
    _axes[1].set(yscale="log", xlabel="standardized moneyness δ", ylabel="half-spread / mid price (%)",
                 title="Relative half-spread")
    _ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    _axes[1].set_yticks(_ticks, labels=[f"{_t:g}" for _t in _ticks])
    _axes[1].minorticks_off()
    _axes[2].set(xlabel="standardized moneyness δ", ylabel="implied volatility (%)", title="Market smile at the mid prices")
    _axes[2].legend(fontsize=6.5, loc="upper right")
    _atm = [100 * MARKET_IV[EXPIRY == _e][np.argmin(np.abs(DELTA[EXPIRY == _e]))] for _e in range(len(EXPIRY_DATES))]
    _wing = [100 * MARKET_IV[EXPIRY == _e][np.argmin(DELTA[EXPIRY == _e])] for _e in range(len(EXPIRY_DATES))]
    _rows = [{"expiry": _q["expiration"], "days": int(round(365 * _t)), "type": _k, "strike": _q["strike"],
              "forward": round(float(_f), 2), "discount": round(float(_d), 5), "bid": _q["bid"], "ask": _q["ask"],
              "mid": float(_m), "half-spread h": round(float(_h), 3), "δ": round(float(_x), 2),
              "mid IV (%)": round(100 * float(_iv), 2)}
             for _q, _t, _k, _f, _d, _m, _h, _x, _iv in zip(QUOTES, MATURITY, KIND, FORWARD, DISCOUNT, MID, H, DELTA, MARKET_IV)]
    _relative = 100 * H / MID
    _override = design["target_overrides"][0]
    _distance = max(abs(_q["target_distance"]) for _q in QUOTES)
    _days = ", ".join(str(_d) for _d in grid_rules.TARGET_DAYS[:-1]) + f" and {grid_rules.TARGET_DAYS[-1]}"
    mo.vstack([
        _fig,
        mo.md(f"""
        **What the figure shows.** *Left:* every expiry has nine contracts at similar values of $\\delta$, most of them puts. *Middle:* the half-spread as a share of the price. Near the money quotes are precise to well under 1% ({_relative.min():.2f}% at best). Deep out-of-the-money options cost little, {MID.min():.2f} index points for the cheapest, and their half-spread is up to {_relative.max():.0f}% of the price. *Right:* the **volatility smile**. Implied volatility rises steeply for low strikes, from {min(_atm):.0f}–{max(_atm):.0f}% at the money to {min(_wing):.0f}–{max(_wing):.0f}% for the deepest puts: the market charges much more for crash protection than a constant-volatility model would. In these units the seven smiles are nearly parallel, so the skew per unit of $\\delta$ is about the same at one week as at eleven months. This skew is the main feature the model must reproduce.

        Implied volatilities are used for display only; all computations use prices.
        """),
        mo.accordion({
            "How the 63 contracts were chosen": mo.md(rf"""
            **Expiries.** For each target maturity of {_days} business days (divided by 252 to give years) we took the nearest eligible SPXW expiry. For the one-year target this is {EXPIRY_DATES[-1]}, {EXPIRY_DAYS[-1]} days away.

            **Strikes.** At each expiry we took the out-of-the-money contract nearest to each of the nine targets in $\delta$. A contract was eligible only if its half-spread was at most {100 * grid_rules.MAX_RELATIVE_HALFSPREAD:.0f}% of its mid price and it was not among the quotes used to estimate the forwards and discount factors or held out for scoring. A target counts as supported when the nearest eligible contract lies within {grid_rules.MAX_Z_DISTANCE:g} of it. At the {_override['expiration']} expiry no eligible call lies near $\delta={_override['original_target_z']:g}$, so that target was declared at $\delta={_override['revised_target_z']:g}$ instead. Every selected contract lies within {_distance:.2f} of its target.

            The reference volatility only sets the units of the axis: the pricer uses each contract's actual strike, forward, discount factor and expiry, and the network sees prices.
            """),
            "All 63 quotes": mo.ui.table(_rows, selection=None, pagination=True, page_size=9),
        }),
    ])
    return


@app.cell(hide_code=True)
def _(mo, workflow):
    mo.vstack([workflow("Model"), mo.md(r"""
    ## 1. Model

    ### Heston dynamics

    The Heston model lets the variance of returns move randomly over time. Price and variance shocks are correlated; a negative correlation makes volatility rise when prices fall, which produces the **skew** seen in the smile.

    $$
    \frac{dF_t}{F_t} = \sqrt{V_t}\,dW_t^F,\qquad
    dV_t = \kappa(v_\infty-V_t)\,dt+\sigma_v\sqrt{V_t}\,dW_t^V,
    \qquad d\langle W^F,W^V\rangle_t=\rho\,dt .
    $$

    | Symbol | Meaning | Prior |
    |:---|:---|:---|
    | $v_0$ | initial variance $V_0$ | uniform on 0.01 to 0.16 (10% to 40% volatility) |
    | $\kappa$ | speed of mean reversion, per year | uniform on 0.1 to 10 |
    | $v_\infty$ | long-run variance, the level $V_t$ reverts to | uniform on 0.01 to 0.16 |
    | $\sigma_v$ | volatility of variance | uniform on 0.1 to 3 |
    | $\rho$ | correlation of price and variance shocks | uniform on −0.95 to −0.1 |
    | $c$ | quote-noise multiplier (below) | uniform on 0.5 to 2 |

    The independent uniform priors cover calm as well as stressed markets. $F_t$ is the forward price for the option's expiry; the box below explains why its equation has no drift term.
    """)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Why there is no drift: the pricing measure and forward prices": mo.md(r"""
        Option prices are computed under the **pricing measure** $\mathbb Q$, under which discounted traded prices are martingales: their expected future value equals today's value. The **forward price** for expiry $T$, $F_t=S_t\,e^{(r-q)(T-t)}$ with interest rate $r$ and dividend yield $q$, is therefore a martingale itself, so its dynamics have no drift. At expiry $F_T=S_T$, and a European call is worth

        $$C=D\;\mathbb E^{\mathbb Q}\big[(F_T-K)^+\big],\qquad D=e^{-rT}.$$

        Rates and dividends enter only through $D$ and $F$, which we read from the market (put–call parity), so the model needs no assumption about them. Dividing by $D\,F$ gives prices in **forward units**, which depend only on the log-moneyness $\ln(K/F)$, the expiry $T$ and the Heston parameters. This is why the same model can price every expiry with a single set of parameters, even though each expiry has its own forward and discount factor.

        The expected return of the index in the real world plays no role here. Estimating it requires a time series of prices and a different, physical-measure model.
        """),
    })
    return


@app.cell(hide_code=True)
def _(TRUTHS, mo, np, plt):
    def illustrative_paths(parameters, *, days=252, paths=3, substeps=16, seed=20261101):
        """Signed full-truncation Euler variance and log-price steps; for illustration only."""
        v0, kappa, v_inf, xi, rho = np.asarray(parameters, dtype=float)[:5]
        step = (1 / 252) / substeps
        normals = np.random.default_rng(seed).standard_normal((paths, days, substeps, 2))
        variance = np.full(paths, v0)
        states, log_price = np.empty((paths, days + 1)), np.zeros((paths, days + 1))
        states[:, 0] = v0
        for day in range(days):
            total = np.zeros(paths)
            for sub in range(substeps):
                positive = np.maximum(variance, 0.0)
                scale = np.sqrt(positive * step)
                z_variance, z_other = normals[:, day, sub, :].T
                total += -0.5 * positive * step + scale * (rho * z_variance + np.sqrt(1 - rho**2) * z_other)
                variance = variance + kappa * (v_inf - positive) * step + xi * scale * z_variance
            log_price[:, day + 1] = log_price[:, day] + total
            states[:, day + 1] = variance
        return np.arange(days + 1) / 252, np.exp(log_price), np.maximum(states, 0)

    _truth = TRUTHS["synthetic-market-like"]
    _t, _prices, _variances = illustrative_paths(_truth)
    _fig, _axes = plt.subplots(1, 2, figsize=(9.6, 2.7), constrained_layout=True)
    _axes[0].plot(_t, _prices.T, lw=1.1)
    _axes[1].plot(_t, 100 * np.sqrt(_variances.T), lw=1.1)
    _axes[1].axhline(100 * np.sqrt(_truth[2]), color="#888888", ls="--", lw=1)
    _axes[0].set(xlabel="years", ylabel="forward / initial forward", title="Possible forward-price paths")
    _axes[1].set(xlabel="years", ylabel="instantaneous volatility (%)", title="Their hidden volatility")
    mo.vstack([
        _fig,
        mo.md(
            f"Three simulated paths at $v_0={_truth[0]:.4f}$, $\\kappa={_truth[1]:.2f}$, $v_\\infty={_truth[2]:.4f}$, "
            f"$\\sigma_v={_truth[3]:.2f}$ and $\\rho={_truth[4]:.2f}$, values within 15% of the market posterior medians of Section 5.4. "
            f"**What to see:** volatility wanders far from its starting level of {100 * np.sqrt(_truth[0]):.0f}% and is pulled "
            f"back towards the long-run level $\\sqrt{{v_\\infty}}={100 * np.sqrt(_truth[2]):.0f}\\%$ (dashed), with a half-life "
            f"of about {12 * np.log(2) / _truth[1]:.0f} months; falls in the price coincide with volatility spikes because "
            "$\\rho<0$. These paths only illustrate the dynamics: option prices are computed exactly (box below), and neither "
            "paths nor volatility are ever observed."),
    ])
    return


@app.cell(hide_code=True)
def _(NAMES, TRUTHS, mo, reference_summary):
    _truth = TRUTHS["synthetic-market-like"]
    _market = {_n: reference_summary["cases"]["market"]["parameters"][_n]["regenerated_q05_q50_q95"][1] for _n in NAMES}
    _paths = (2 * _truth[1] * _truth[2], _truth[3] ** 2)
    _fit = (2 * _market["kappa"] * _market["theta"], _market["vol_of_vol"] ** 2)
    mo.accordion({
        "How option prices are computed: the Fourier pricer, and the Feller condition": mo.md(rf"""
        **Fourier pricing.** Heston is an *affine* model: the characteristic function of the log forward at expiry, $\varphi(u)=\mathbb E^{{\mathbb Q}}\big[e^{{iu\ln(F_T/F_0)}}\big]$, is known in closed form, so a European option needs a single one-dimensional integral (Lewis, 2001). The pricer uses it in a *control-variate* form: it starts from the Black–Scholes price $C_{{\mathrm{{BS}}}}$ with the same expected average variance and adds the integral of the difference between the two characteristic functions,

        $$C=C_{{\mathrm{{BS}}}}+D\,\frac{{\sqrt{{FK}}}}{{\pi}}\int_0^\infty\operatorname{{Re}}\!\left[e^{{iu\ln(F/K)}}\Big(\varphi_{{\mathrm{{BS}}}}\big(u-\tfrac i2\big)-\varphi\big(u-\tfrac i2\big)\Big)\right]\frac{{du}}{{u^2+\tfrac14}},$$

        and puts get the same correction. For very cheap options the plain formula is a small difference of two large numbers; the control-variate form avoids that cancellation. The integral is evaluated by Gauss–Legendre quadrature on a finite interval. The number of nodes and the interval are doubled step by step until two successive steps agree to within 2% of the smallest noise standard deviation the prior allows for that contract ($c=0.5$, that is 1% of its half-spread), so numerical error is negligible next to quote noise. Pricing all 63 options takes milliseconds; the prior predictive check below times 1,000 simulated data sets.

        **The Feller condition.** If $2\kappa v_\infty\ge\sigma_v^2$, the variance process never reaches zero. When the condition fails, $V_t$ can touch zero and is immediately pushed back up by the drift term. The model and the Fourier formula remain valid either way, and the prior does not impose the condition. The parameters of the paths above violate it ($\sigma_v^2={_paths[1]:.1f}$ against $2\kappa v_\infty={_paths[0]:.2f}$), and so do the market posterior medians of Section 5.4 ($\sigma_v^2={_fit[1]:.1f}$ against $2\kappa v_\infty={_fit[0]:.2f}$): the posterior fits the SPX skew with a volatility of variance far above the Feller limit.
        """),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Observation model: noise set by the bid–ask spread

    The quotes are noisy measurements of the model prices:

    $$
    y_i = P_i(v_0,\kappa,v_\infty,\sigma_v,\rho) + c\,h_i\,\epsilon_i,\qquad
    \epsilon_i \overset{\mathrm{iid}}{\sim}\mathcal N(0,1),\qquad i=1,\dots,63,
    $$

    where $y_i$ is the observed mid price of contract $i$ and $P_i$ its Heston price. The three ingredients of the noise term play different roles:

    * $h_i$ is **data**: half the bid–ask spread of contract $i$, read from the snapshot. Like the strike and the expiry it is known and fixed, and it is the same in every simulation. Tight spreads mean precise quotes.
    * $\epsilon_i$ is a fresh standard-normal draw for every contract and every simulated data set.
    * $c$ is a **parameter**: one multiplier, inferred with the Heston parameters, that scales all 63 noise levels together. Treating one half-spread as one standard deviation is a convention, and $c$ lets the data correct it.

    **What this observation model leaves out.** It contains a single source of error, quote noise. It assumes that the Heston model, with the right parameters, prices every contract exactly, and that the mid prices differ from those prices only by noise of size $c\,h_i$. There is no separate term for **model error**, a systematic difference between Heston and the market. If such a difference exists, the only way the model can accommodate it is a larger $c$. Section 5.4 shows that this is exactly what happens with the market quotes.
    """)
    return


@app.cell(hide_code=True)
def _(mo, workflow):
    mo.vstack([workflow("Simulate"), mo.md(r"""
    ## 2. Simulate

    A simulator draws parameters from the prior and returns one synthetic data set: **one number per contract**, a noisy mid price. It does not simulate bids and asks; the observed spreads only set the noise scale. In BayesFlow the simulator is assembled from two functions connected by argument names: the prior returns named parameters, and the observation model receives them, prices the 63 contracts with the Fourier pricer and adds the noise. It returns every quote **divided by its half-spread**, so that all network inputs are measured in noise units.
    """)])
    return


@app.cell
def _(BOUNDS, H, NAMES, bf, np, pilot):
    def make_simulator(seed):
        rng = np.random.default_rng(seed)

        def prior():
            return dict(zip(NAMES, rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1])))

        def observation_model(initial_variance, kappa, theta, vol_of_vol, rho, c):
            # `theta` is the long-run variance v_inf (the name used by the pricing code)
            parameters = np.array([initial_variance, kappa, theta, vol_of_vol, rho, c])
            quotes = pilot.prices(parameters) + c * H * rng.standard_normal(H.size)
            return {"observations": quotes / H}

        return bf.make_simulator([prior, observation_model])

    return (make_simulator,)


@app.cell(hide_code=True)
def _(
    CASES,
    CLEAN,
    DELTA,
    ERRORS,
    EXPIRY,
    EXPIRY_COLORS,
    EXPIRY_DATES,
    H,
    MARKET_IV,
    TRUTHS,
    implied_vols,
    mo,
    np,
    plots,
    plt,
    vol_point_noise,
):
    _case = "synthetic-market-like"
    _c = TRUTHS[_case][-1]
    _observed = CLEAN[_case] + _c * H * ERRORS[_case]
    if not np.allclose(_observed, CASES[_case], rtol=0, atol=1e-9):
        raise ValueError("saved clean prices and errors do not reproduce the saved observation")
    _e = 4
    _at = EXPIRY == _e
    _clean_iv = implied_vols(CLEAN[_case])[0]
    _obs_iv = implied_vols(_observed)[0]
    _sd_points = vol_point_noise(_c * H, _clean_iv)
    _noise_points = vol_point_noise(H, MARKET_IV)
    _fig, _axes = plt.subplots(1, 3, figsize=(11.2, 3.0), constrained_layout=True)
    _axes[0].plot(DELTA[_at], 100 * _clean_iv[_at], "-", color=plots.MUTED, lw=1.2, label="Heston price")
    _axes[0].scatter(DELTA[_at], 100 * _obs_iv[_at], s=18, color=plots.SERIES[2], zorder=3, label="simulated quote")
    _axes[0].set(xlabel="standardized moneyness δ", ylabel="implied volatility (%)", title=f"Simulated quotes, {EXPIRY_DATES[_e]}")
    _axes[0].legend(fontsize=7)
    _axes[1].fill_between(DELTA[_at], -_sd_points[_at], _sd_points[_at], color=plots.SERIES[2], alpha=.18, lw=0, label="±1 noise SD")
    _axes[1].axhline(0, color=plots.MUTED, lw=1)
    _axes[1].plot(DELTA[_at], 100 * (_obs_iv - _clean_iv)[_at], "o", ms=4, color=plots.SERIES[2], label="quote − Heston price")
    _axes[1].set(xlabel="standardized moneyness δ", ylabel="implied-volatility points", title="The same errors, magnified")
    _axes[1].legend(fontsize=7)
    for _x, _date in enumerate(EXPIRY_DATES):
        _axes[2].plot(DELTA[EXPIRY == _x], _noise_points[EXPIRY == _x], "o-", ms=2.5, lw=.8, color=EXPIRY_COLORS[_x])
    _axes[2].set(yscale="log", xlabel="standardized moneyness δ", ylabel="implied-volatility points",
                 title="One half-spread, all 63 contracts")
    _ticks = [0.02, 0.05, 0.1, 0.2]
    _axes[2].set_yticks(_ticks, labels=[f"{_t:g}" for _t in _ticks])
    _axes[2].minorticks_off()
    _smile_range = 100 * (np.nanmax(MARKET_IV) - np.nanmin(MARKET_IV))
    _largest = float(np.nanmax(np.abs(100 * (_obs_iv - _clean_iv)[_at])))
    _share = 100 * float(np.mean(np.abs(ERRORS[_case]) <= 1))
    mo.vstack([
        _fig,
        mo.md(f"""
        **One simulated data set**, generated at the market-like parameters of Section 4 with $c=1$, shown at one expiry. *Left:* on the scale of the smile the quote errors are invisible; the simulated quotes sit on the Heston curve. *Middle:* the same errors magnified. None exceeds {_largest:.2f} volatility points. Over all 63 contracts of this data set, {_share:.0f}% of the errors lie within one noise standard deviation, close to the two thirds expected for normal noise. *Right:* the noise scale of every contract: one half-spread, converted into implied-volatility points and coloured by expiry as above. It ranges from {np.nanmin(_noise_points):.2f} to {np.nanmax(_noise_points):.2f} points, against a smile that spans about {_smile_range:.0f} points.

        **Why this matters.** Liquid index options are quoted very precisely relative to how much the smile moves when the parameters change. The 63 quotes therefore pin the parameters down tightly, and the posteriors in Section 4 are narrow. The flip side: any systematic difference between model and market, even a fraction of a volatility point, is large compared with the assumed noise. Section 5.4 shows this on the market quotes.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prior predictive check

    Before training, we simulate 1,000 data sets from the prior and compare them with the market quotes. If the market lay outside the simulated range, the network would have to extrapolate, and the prior or the model would need rethinking.
    """)
    return


@app.cell(hide_code=True)
def _(DELTA, EXPIRY, EXPIRY_DATES, H, MARKET_IV, implied_vols, make_simulator, mo, np, plots, plt, time):
    _started = time.perf_counter()
    prior_batch = make_simulator(20261111).sample(1000)
    _seconds = time.perf_counter() - _started
    _iv = implied_vols(prior_batch["observations"] * H)
    _fig, _axes = plt.subplots(1, 3, figsize=(11.0, 3.2), sharey=True, constrained_layout=True)
    for _ax, _e in zip(_axes, (0, 4, 6)):
        _at = EXPIRY == _e
        for _row in _iv[:150]:
            _ax.plot(DELTA[_at], 100 * _row[_at], color=plots.SERIES[0], alpha=.12, lw=.8)
        _ax.plot(DELTA[_at], 100 * MARKET_IV[_at], "o-", color=plots.SERIES[1], ms=3, lw=1.5, label="market")
        _ax.set(title=f"{EXPIRY_DATES[_e]}", xlabel="standardized moneyness δ")
    _axes[0].set(ylabel="implied volatility (%)", ylim=(0, 70))
    _axes[0].legend(fontsize=7)
    mo.vstack([
        _fig,
        mo.md(f"150 of 1,000 prior-predictive smiles (blue) at three expiries, with the market (orange); simulating all "
              f"1,000 data sets took {_seconds:.0f} s. **What to see:** the prior generates calm and stressed markets, flat and "
              "steep skews, and the market lies inside this range, towards its low-volatility side at the shortest expiry. "
              "The check only shows that the prior does not rule out the data; it says nothing yet about whether the model "
              f"fits them. {100 * np.isnan(_iv).mean():.1f}% of simulated quotes have no implied volatility because noise "
              "pushed a very cheap option below its intrinsic value; they remain valid data."),
    ])
    return (prior_batch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Preparing the network input: whitening

    The network receives the 63 numbers $y_i/h_i$. In this raw form they are hard to learn from, for two reasons:

    1. **They are extremely correlated.** All 63 quotes move together when a parameter changes: a higher $v_0$ raises every short-dated quote at once. A few combinations of the quotes, such as the overall level of volatility and the slope of the skew, vary enormously across simulations, while most other combinations hardly vary.
    2. **The information about the noise level hides in the small directions.** The size of the quote errors, and hence $c$, shows up mainly in combinations of quotes that the smooth Heston smile cannot change. These combinations vary about a million times less than the dominant ones, so a network sees them as tiny ripples on top of large swings.

    **Whitening** is a fixed linear change of coordinates that removes both problems. It rotates the 63 inputs onto their principal directions, rescales every direction to unit variance and rotates back. After whitening the inputs are uncorrelated, every direction has the same weight, and no information is lost because the transformation is invertible. It is fitted once on simulated data (10,000 simulations for the pretrained estimator), then frozen and applied unchanged to every later input, including the real quotes. The figure shows the effect on the 1,000 prior simulations above.
    """)
    return


@app.cell(hide_code=True)
def _(fit_whitening, mo, np, plots, plt, prior_batch):
    _x = prior_batch["observations"]
    _c = prior_batch["c"][:, 0]
    _fit = fit_whitening(_x[:700])
    _corr_before = np.corrcoef(_x[700:], rowvar=False)
    _corr_after = np.corrcoef(_fit.transform(_x[700:]), rowvar=False)
    _values, _vectors = np.linalg.eigh(np.cov(_x[:700], rowvar=False))
    _order = np.argsort(_values)[::-1]
    _values, _vectors = _values[_order], _vectors[:, _order]
    _scores = (_x[700:] - _x[:700].mean(axis=0)) @ _vectors / np.sqrt(_values)
    _link = np.array([abs(np.corrcoef(np.abs(_scores[:, _j]), _c[700:])[0, 1]) for _j in range(63)])
    _mantissa, _exponent = f"{_values[0] / _values[-1]:.0e}".split("e")
    _fig, _axes = plt.subplots(1, 4, figsize=(13.0, 3.0), constrained_layout=True)
    _axes[0].semilogy(np.arange(1, 64), _values, "o", ms=2.5, color=plots.SERIES[0])
    _axes[0].set(xlabel="direction, by variance", ylabel="variance", title="Variance per direction")
    for _ax, _corr, _title in ((_axes[1], _corr_before, "|correlation| before"),
                               (_axes[2], _corr_after, "|correlation| after whitening")):
        _image = _ax.imshow(np.abs(_corr), vmin=0, vmax=1, cmap="Blues")
        _ax.set(title=_title, xlabel="quote", ylabel="quote")
    _fig.colorbar(_image, ax=_axes[2], shrink=.8)
    _axes[3].bar(np.arange(1, 64), _link, color=plots.SERIES[1], width=.8)
    _axes[3].set(xlabel="direction, by variance", ylabel="|correlation with c|", title="Where the information about c is")
    mo.vstack([
        _fig,
        mo.md(f"""
        **How to read the figure.** *First panel:* the variance of the 63 inputs along their principal directions. The first direction varies about ${_mantissa}\\times10^{{{int(_exponent)}}}$ times more than the last. *Second and third panels:* correlations between the 63 inputs on new simulations, before and after whitening. Almost every pair is strongly correlated before; afterwards the matrix is nearly diagonal. *Last panel:* how strongly the size of each component reveals the noise multiplier $c$. The information about $c$ sits in the low-variance directions (average {np.mean(_link[-30:]):.2f} over the last 30, against {np.mean(_link[:10]):.2f} over the first 10). Before whitening these directions are invisible next to the dominant ones; after whitening they carry the same weight as the rest.
        """),
        mo.accordion({"Whitening, formally": mo.md(r"""
        Let $x\in\mathbb R^{63}$ be the vector of inputs $y_i/h_i$, and let $\mu$ and $\Sigma$ be its mean and covariance, estimated from $N$ simulated training data sets. Write $\Sigma=U\Lambda U^\top$, with orthonormal eigenvectors $U$ and eigenvalues $\Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_{63})$. The whitened input is

        $$\tilde x = W(x-\mu),\qquad W=U\Lambda^{-1/2}U^\top=\Sigma^{-1/2},$$

        so that $\operatorname{Cov}(\tilde x)=W\Sigma W^\top=I$. This choice of $W$ is **ZCA whitening**. Among all transformations with identity covariance, it keeps $\tilde x$ closest to $x$, so each whitened input still corresponds roughly to one quote. The alternative $\Lambda^{-1/2}U^\top$ (PCA whitening) would mix them. Eigenvalues below $10^{-8}\lambda_{\max}$ are floored for numerical stability.

        The ratio $\lambda_{\max}/\lambda_{\min}$, the **condition number** of $\Sigma$, measures how badly scaled the raw inputs are. Training a network by gradient descent on badly scaled inputs is slow and unreliable, because a step size that suits the dominant directions is far too small for the others. Whitening sets the condition number to one. $\mu$ and $W$ come from training simulations only and are stored with the estimator, so the same transformation is applied at training and at inference.
        """)}),
    ])
    return


@app.cell(hide_code=True)
def _(mo, workflow):
    mo.vstack([workflow("Train"), mo.md(r"""
    ## 3. Train

    Simulations can be used in two ways:

    | | Neural pricer + Bayesian sampler | Amortized posterior (this notebook) |
    |:---|:---|:---|
    | Network learns | parameters → clean prices | noisy quotes → distribution of parameters |
    | At a new observation | evaluate the surrogate inside a new MCMC run | condition the network and draw parameters |
    | Main approximation to check | surrogate error, then sampler convergence | posterior approximation error |
    | Changing the prior, noise or contract list | often reuse the pricer and run a new chain | requires a new estimator |

    ### The posterior estimator: flow matching

    The estimator turns random noise into posterior draws. It starts from a draw $z_1$ of a six-dimensional standard normal distribution at time $t=1$ and moves it continuously back to $t=0$ by following a **velocity field**:

    $$\frac{dz_t}{dt}=v_\phi\big(z_t,\,t,\,\tilde x\big).$$

    The end point $z_0$ is a draw from the posterior given the whitened quotes $\tilde x$, in standardized units; undoing the standardization gives $(v_0,\kappa,v_\infty,\sigma_v,\rho,c)$. The velocity $v_\phi$ is a neural network with weights $\phi$. Because the quotes are an input, one network serves every possible set of quotes: that is what makes the posterior **amortized**.
    """)])
    return


@app.cell(hide_code=True)
def _(mo):
    def _box(title, body, fill="rgba(42,120,214,0.06)", border="#2a78d6"):
        return (f'<div style="border:1px solid {border};border-radius:10px;padding:8px 10px;background:{fill};'
                f'text-align:center;min-width:120px"><div style="font-weight:600">{title}</div>'
                f'<div style="font-size:0.8em;line-height:1.35;margin-top:3px">{body}</div></div>')

    _arrow = '<div style="align-self:center;color:#9a9890;font-size:1.4em;padding:0 4px">&#8594;</div>'
    _inputs = ('<div style="display:flex;flex-direction:column;gap:6px">'
               + _box("Whitened quotes x̃", "63 numbers: the observed data")
               + _box("Current point z<sub>t</sub>", "6 numbers: standardized parameters")
               + _box("Time t", "1 number between 0 and 1") + "</div>")
    _network = _box("Neural network v<sub>φ</sub>", "3 hidden layers of 128 units, dropout 0.1",
                    fill="rgba(235,104,52,0.08)", border="#eb6834")
    _output = _box("Velocity", "6 numbers: the direction in which z<sub>t</sub> moves")
    _sampling = ('<div style="margin-top:10px;font-size:0.9em;line-height:1.55">'
                 '<b>Sampling:</b> draw z<sub>1</sub> ~ N(0, I) &#8594; follow the velocity from t = 1 back to t = 0 '
                 '(adaptive Runge–Kutta solver) &#8594; z<sub>0</sub> is a posterior draw. Repeat with new z<sub>1</sub> for '
                 'more draws.<br>'
                 '<b>Training:</b> take a simulated pair (θ, x̃), with θ standardized; draw noise z<sub>1</sub> ~ N(0, I) '
                 'independently of θ and a time t, more often near t = 0; form the point z<sub>t</sub> = (1 − t) θ + t '
                 'z<sub>1</sub> on the straight line from θ to z<sub>1</sub>; and teach the network that the velocity there is '
                 'z<sub>1</sub> − θ (squared-error loss).</div>')
    mo.Html('<div style="display:flex;align-items:center;gap:4px;flex-wrap:wrap">' + _inputs + _arrow + _network + _arrow
            + _output + "</div>" + _sampling)
    return


@app.cell(hide_code=True)
def _(mo, np, plots, plt):
    _rng = np.random.default_rng(3)
    _noise = _rng.standard_normal((400, 2))
    _a = 0.55 * _rng.standard_normal(400) + 1.6
    _theta = np.column_stack([_a, 0.9 * (_a - 1.6) ** 2 + 0.25 * _rng.standard_normal(400) + 1.0])  # a curved toy posterior
    _fig, _axes = plt.subplots(1, 4, figsize=(12.4, 3.0), sharex=True, sharey=True, constrained_layout=True)
    for _ax, _t in zip(_axes[:3], (1.0, 0.5, 0.0)):
        _points = (1 - _t) * _theta + _t * _noise
        _ax.scatter(*_points.T, s=4, color=plots.SERIES[1] if _t == 0 else plots.SERIES[0], alpha=.6)
        _ax.set_title(f"t = {_t:g}" + (": noise" if _t == 1 else ": posterior" if _t == 0 else ""))
    for _k in range(25):
        _axes[3].plot([_theta[_k, 0], _noise[_k, 0]], [_theta[_k, 1], _noise[_k, 1]], color=plots.MUTED, lw=.7)
    _axes[3].scatter(*_noise[:25].T, s=8, color=plots.SERIES[0], zorder=3)
    _axes[3].scatter(*_theta[:25].T, s=8, color=plots.SERIES[1], zorder=3)
    _axes[3].set_title("training: straight paths")
    for _ax in _axes:
        _ax.set(xlabel="parameter 1 (standardized)")
    _axes[0].set_ylabel("parameter 2 (standardized)")
    mo.vstack([
        _fig,
        mo.md("**Flow matching in two dimensions.** A toy posterior with curved dependence between two parameters. *First three "
              "panels:* the cloud of points at $t=1$ (standard normal noise), halfway, and at $t=0$ (the posterior). *Last panel:* "
              "during training each simulated parameter (orange) is paired with an independent noise draw (blue), and the "
              "network learns the velocity along the straight line between them. Averaged over many pairs, these velocities "
              "define a flow that, integrated from $t=1$ back to $t=0$, carries the whole noise cloud onto the posterior. Our "
              "estimator does the same in six dimensions, with the whitened quotes as an extra input that tells the flow which "
              "posterior to produce."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | Object | Used here | Dimension |
    |:---|:---|---:|
    | Observed data $y$ | 63 quotes in index points | 63 |
    | Network conditions $\tilde x$ | $y/h$, then the frozen whitening | 63 |
    | Summary network | none; the fixed vector enters directly | n/a |
    | Flow state $z_t$ | $v_0,\kappa,v_\infty,\sigma_v,\rho,c$, standardized | 6 |
    | Flow time $t$ | scalar time | 1 |

    The parameters are standardized but not mapped onto their bounded prior ranges, so a small fraction of posterior draws can fall just outside the prior. We report them rather than clip them.

    **Training.** The pretrained estimator is trained on 300,000 simulated data sets for 300 epochs with batch size 256, using BayesFlow's default optimizer for offline training. That is AdamW with weight decay 0.005 and gradient clipping at 1.5, with a learning rate that warms up from 5·10⁻⁵ to 5·10⁻⁴ and then decays to zero on a cosine schedule. Each simulated data set keeps its noise draw throughout training. The fit ran on the CPU of an Apple-silicon Mac and finished within its one-hour limit. To sample, the flow is integrated with an adaptive Runge–Kutta solver (Tsit5).

    ```python
    task, data = delivery.training_inputs(HESTON_BUNDLE, rows=300_000)  # y/h with the frozen whitening applied
    keras.utils.set_random_seed(9011)
    workflow = recipe.build_workflow(task)  # FlowMatching([128, 128, 128], dropout=0.1), standardized parameters
    history = workflow.fit_offline(data=data["train"], validation_data=data["validation"],
                                   epochs=300, batch_size=256)
    ```

    The button below runs the whole loop at a small scale. It simulates 3,000 data sets, fits the whitening, trains a network of the same architecture for 40 epochs and checks it on new simulations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    live_train_button = mo.ui.run_button(label="Simulate 3,000 data sets and train a small network (about a minute on a CPU)")
    live_train_button
    return (live_train_button,)


@app.cell
def _(
    EVALUATION,
    LABELS,
    NAMES,
    TEXT_LABELS,
    bf,
    fit_whitening,
    json,
    keras,
    live_train_button,
    make_simulator,
    mo,
    np,
    plots,
    recipe,
    task,
    time,
):
    mo.stop(not live_train_button.value, mo.md("*Press the button above to simulate and train.*"))
    _started = time.perf_counter()
    live_simulator = make_simulator(4)  # its own seeded generator
    _train = live_simulator.sample(3000)
    _validation = live_simulator.sample(300)
    _simulated = time.perf_counter() - _started

    live_whitening = fit_whitening(_train["observations"])  # fitted on training rows only, then frozen
    _train["observations"] = live_whitening.transform(_train["observations"])
    _validation["observations"] = live_whitening.transform(_validation["observations"])

    keras.utils.set_random_seed(4)
    live_workflow = recipe.build_workflow(task)
    live_history = live_workflow.fit_offline(data=_train, validation_data=_validation, epochs=40, batch_size=64, verbose=0)
    _trained = time.perf_counter() - _started - _simulated

    # Check the network just trained on 100 new simulations.
    _held = live_simulator.sample(100)
    _sampling_started = time.perf_counter()
    _draws = live_workflow.sample(conditions={"observations": live_whitening.transform(_held["observations"])},
                                  num_samples=200)
    _sampled = time.perf_counter() - _sampling_started

    # Everything said about this run is computed from it.
    _loss, _val = live_history.history["loss"][-1], live_history.history.get("val_loss", [np.nan])[-1]
    _fit = ("the validation loss ends close to the training loss, so the network is not overfitting"
            if _val <= 1.1 * _loss else "the validation loss ends clearly above the training loss, so the network has "
            "started to overfit")
    _corr = [float(np.corrcoef(np.median(_draws[_n][..., 0], axis=1), _held[_n][:, 0])[0, 1]) for _n in NAMES]
    _pretrained = json.loads((EVALUATION / "evaluation.json").read_text())["population"]["mean_posterior_sd"]
    _width = [float(np.mean(np.std(_draws[_n][..., 0], axis=1))) / _pretrained[_j] for _j, _n in enumerate(NAMES)]
    _list = lambda _values, _fmt: ", ".join(f"{_l} {_fmt.format(_v)}" for _l, _v in zip(TEXT_LABELS, _values))
    mo.vstack([
        mo.md(f"Simulated 3,300 data sets in {_simulated:.0f} s and trained for 40 epochs in {_trained:.0f} s. Final "
              f"training loss {_loss:.3f}, validation loss {_val:.3f}: {_fit}."),
        bf.diagnostics.loss(live_history),
        mo.md(f"**Check the network just trained.** 100 new simulated data sets, 200 posterior draws each, in {_sampled:.1f} s. "
              "In the recovery plot each dot compares a posterior median with the true value; the correlation between the "
              f"two is {_list(_corr, '{:.2f}')}. In the calibration plot each curve should stay inside its grey band "
              "(Section 5.2 explains how to read it). This network saw 1% of the training data of the pretrained estimator; "
              f"its average posterior standard deviation, relative to the pretrained one, is {_list(_width, '{:.1f}×')}."),
        bf.diagnostics.recovery(estimates=_draws, targets=_held, variable_keys=NAMES, variable_names=LABELS,
                                uncertainty_agg_kwargs={"prob": 0.68}, color=plots.SERIES[0]),
        bf.diagnostics.calibration_ecdf(estimates=_draws, targets=_held, variable_keys=NAMES, variable_names=LABELS,
                                        difference=True, rank_ecdf_color=plots.SERIES[0], fill_color=plots.BASELINE),
    ])
    return live_simulator, live_whitening, live_workflow


@app.cell(hide_code=True)
def _(mo, workflow):
    mo.vstack([workflow("Infer"), mo.md(r"""
    ## 4. Infer

    We now load the pretrained estimator and apply it to three sets of 63 quotes:

    * **Simulated quotes, market-like parameters**: generated with $c=1$ at parameters within 15% of the market posterior medians (Section 5.4).
    * **Simulated quotes, stressed market**: generated with $c=1$ at a high-volatility parameter vector drawn from the prior. Volatility starts at 37% and reverts towards 29%, so short-dated options are more expensive than long-dated ones (an inverted term structure), and the skew is steep. Markets look like this in a crisis.
    * **Market quotes**: the 63 mid prices of 9 November 2026, for which the true parameters are unknown.

    For each data set, a **PyMC reference** samples the same posterior with the exact Fourier likelihood: four DEMetropolisZ chains with 12,000 draws each. It takes minutes per data set rather than seconds (run times below), and serves as a numerical ground truth.

    ```python
    model, checkpoint = delivery.load_checkpoint(HESTON_BUNDLE, "300k-300-final300")
    draws = model.sample(quotes[None, :], count=4096, seed=261113)[0]   # (4096, 6) posterior draws
    ```

    The estimator takes raw quotes in index points; it divides them by their half-spreads and applies the frozen whitening itself.
    """)])
    return


@app.cell(hide_code=True)
def _(BUNDLE, BUNDLE_HELP, ESTIMATOR, delivery, mo, recipe, task, time):
    estimator = None
    if BUNDLE is None:
        _view = BUNDLE_HELP
    else:
        _started = time.perf_counter()
        estimator, _record = delivery.load_checkpoint(BUNDLE, ESTIMATOR)
        if recipe.fingerprint(estimator.task) != recipe.fingerprint(task):
            raise ValueError("the pretrained estimator was trained for a different contract list")
        _view = mo.md(f"Pretrained estimator loaded in {time.perf_counter() - _started:.1f} s "
                      f"({_record['train_rows']:,} simulated data sets, {_record['native_epoch']} epochs); "
                      "all of its files passed their SHA-256 checks.")
    _view
    return (estimator,)


@app.cell(hide_code=True)
def _(CASES, CONDITIONAL_DRAWS, EVAL_SEED, estimator, mo, np, time):
    neural_draws = {}
    if estimator is None:
        _view = mo.md("")
    else:
        _started = time.perf_counter()
        for _index, _case_id in enumerate(CASES):
            neural_draws[_case_id] = np.asarray(estimator.sample(CASES[_case_id][None], CONDITIONAL_DRAWS,
                                                                 seed=EVAL_SEED + _index)[0])
        _view = mo.md(f"{CONDITIONAL_DRAWS:,} posterior draws for each of the three data sets in "
                      f"{time.perf_counter() - _started:.1f} s.")
    _view
    return (neural_draws,)


@app.cell(hide_code=True)
def _(CASES, CASE_TITLES, REFERENCES, hashlib, json, mo, np):
    reference_summary = json.loads((REFERENCES / "reference-comparison.json").read_text())
    _display = REFERENCES / "reference-display-draws.npz"
    if hashlib.sha256(_display.read_bytes()).hexdigest() != reference_summary["display_cache"]["sha256"]:
        raise ValueError("the reference draws differ from their recorded identity")
    with np.load(_display, allow_pickle=False) as _archive:
        reference_draws = {_case_id: _archive[f"{_case_id}__parameters"].reshape(-1, 6) for _case_id in CASES}
        reference_products = {_case_id: _archive[f"{_case_id}__products"].reshape(-1, 2) for _case_id in CASES}
    _times = "; ".join(f"{CASE_TITLES[_case_id][0].lower() + CASE_TITLES[_case_id][1:]}: "
                       f"{reference_summary['cases'][_case_id]['usage']['wall_seconds'] / 60:.0f} min" for _case_id in CASES)
    mo.md(f"PyMC reference run times, one CPU core each: {_times}.")
    return reference_draws, reference_products, reference_summary


@app.cell(hide_code=True)
def _(CASE_TITLES, mo):
    case_choice = mo.ui.dropdown(options={_title: _case_id for _case_id, _title in CASE_TITLES.items()},
                                 value=CASE_TITLES["synthetic-market-like"], label="Data set")
    case_choice
    return (case_choice,)


@app.cell(hide_code=True)
def _(
    CASES,
    CASE_TITLES,
    DELTA,
    EXPIRY,
    EXPIRY_COLORS,
    EXPIRY_DATES,
    EXPIRY_DAYS,
    LABELS,
    NAMES,
    PRODUCTS,
    PRODUCT_TITLES,
    TEXT_LABELS,
    TRUTHS,
    case_choice,
    expected_variance,
    implied_vols,
    mo,
    neural_draws,
    np,
    pairs,
    plt,
    reference_draws,
    reference_products,
    reference_summary,
):
    _case = case_choice.value
    _neural = neural_draws.get(_case)
    _truth = TRUTHS[_case]
    _iv = implied_vols(CASES[_case])[0]
    _fig, _ax = plt.subplots(figsize=(7.2, 3.0), constrained_layout=True)
    for _e, _date in enumerate(EXPIRY_DATES):
        _at = EXPIRY == _e
        _ax.plot(DELTA[_at], 100 * _iv[_at], "o-", ms=3, lw=1, color=EXPIRY_COLORS[_e], label=f"{EXPIRY_DAYS[_e]} d")
    _ax.set(xlabel="standardized moneyness δ", ylabel="implied volatility (%)", title=f"{CASE_TITLES[_case]}")
    _ax.legend(fontsize=7, ncol=2)

    _rows = []
    _ref = reference_summary["cases"][_case]["parameters"]
    for _j, _name in enumerate(NAMES):
        _rq = _ref[_name]["regenerated_q05_q50_q95"]
        _row = {"parameter": TEXT_LABELS[_j], "truth": "unknown" if _truth is None else f"{_truth[_j]:.5g}"}
        if _neural is not None:
            _q = np.quantile(_neural[:, _j], [.05, .5, .95])
            _row["network: median [90% interval]"] = f"{_q[1]:.5g} [{_q[0]:.5g}, {_q[2]:.5g}]"
        _row["PyMC reference: median [90% interval]"] = f"{_rq[1]:.5g} [{_rq[0]:.5g}, {_rq[2]:.5g}]"
        _rows.append(_row)

    _product_rows = []
    for _k, (_key, _interval) in enumerate(PRODUCTS.items()):
        _values = {"PyMC reference": reference_products[_case][:, _k]}
        if _neural is not None:
            _values = {"network": expected_variance(_neural[_neural[:, 1] > 0], _interval), **_values}
        for _label, _value in _values.items():
            _pq = np.quantile(_value, [.05, .5, .95])
            _product_rows.append({"quantity": PRODUCT_TITLES[_key], "source": _label,
                                  "median, volatility (%)": round(100 * np.sqrt(_pq[1]), 3),
                                  "90% interval, volatility (%)": f"{100 * np.sqrt(_pq[0]):.3f} to {100 * np.sqrt(_pq[2]):.3f}"})
    _sets = {"PyMC reference": reference_draws[_case]}
    if _neural is not None:
        _sets = {"network posterior": _neural, **_sets}
    _corner = pairs(_sets, LABELS, truth=_truth, figsize=(8.6, 8.6), points=1200)
    _first = _corner.axes[0]
    _handles, _names = _first.get_legend_handles_labels()
    if _first.get_legend() is not None:  # a panel-anchored legend squeezes the first column of a 6 x 6 grid
        _first.get_legend().remove()
        _corner.legend(_handles, _names, loc="upper right", fontsize=8, frameon=False)
    for _axis in _corner.axes:
        _axis.tick_params(labelsize=6.5)
    for _axis in _corner.axes[-len(LABELS):]:
        for _tick in _axis.get_xticklabels():
            _tick.set_rotation(35)

    _reference = reference_draws[_case]
    _corr = np.corrcoef(_reference, rowvar=False)
    _i, _j = max(((_a, _b) for _a in range(6) for _b in range(_a)), key=lambda _p: abs(_corr[_p]))
    _relative = {_name: (_ref[_name]["regenerated_q05_q50_q95"][2] - _ref[_name]["regenerated_q05_q50_q95"][0])
                 / (2 * abs(_ref[_name]["regenerated_q05_q50_q95"][1])) for _name in NAMES[:5]}
    _sharpest = min(_relative, key=_relative.get)

    if _truth is None:
        _note = ("There is no known truth for the market quotes. Note the noise multiplier: its posterior is pressed against "
                 "the upper end of its prior range, $c=2$. The data ask for larger quote errors than the model allows, because "
                 "no Heston smile comes close enough to these quotes. Section 5.4 explains what this means and why the network "
                 "and the reference disagree here.")
    else:
        # Where does the truth sit in the reference posterior? Marginally (in posterior SDs) and jointly (Mahalanobis).
        _centre, _spread = _reference.mean(axis=0), _reference.std(axis=0, ddof=1)
        _z = (_truth - _centre) / _spread
        _precision = np.linalg.inv(np.cov(_reference, rowvar=False))
        _distance = lambda _x: np.einsum("...i,ij,...j->...", _x - _centre, _precision, _x - _centre)
        _joint = 100 * np.mean(_distance(_reference) < _distance(_truth))
        _outside = [TEXT_LABELS[_k] for _k in range(6) if abs(_z[_k]) > 1.645]
        if _outside:
            _listed = _outside[0] if len(_outside) == 1 else ", ".join(_outside[:-1]) + " and " + _outside[-1]
            _where = (f"The true values are marked by stars and lines. Not all of them lie inside the 90% intervals: "
                      f"{_listed} {'is' if len(_outside) == 1 else 'are'} up to {np.max(np.abs(_z)):.1f} "
                      "posterior standard deviations from the posterior mean. This is what one noisy data set can do: "
                      "the noise shifts the posterior, and strongly correlated parameters shift together. ")
        else:
            _where = "The true values, marked by stars and lines, lie inside the 90% intervals. "
        _note = (_where + f"Taking all six parameters and their correlations together, the truth is farther from the "
                 f"posterior centre than {_joint:.0f}% of the posterior draws. If the posterior is correct, this share is "
                 f"equally likely to be anything from 0 to 100%, so a single value of {_joint:.0f}% is no warning sign. "
                 "Whether intervals contain the truth as often as they claim can only be judged over many data sets "
                 "(Section 5.2).")
        if _case == "synthetic-broad":
            _first, _last = (EXPIRY == 0), (EXPIRY == len(EXPIRY_DATES) - 1)
            _atm = [100 * _iv[_at][np.argmin(np.abs(DELTA[_at]))] for _at in (_first, _last)]
            _deep = [100 * _iv[EXPIRY == _e][np.argmin(DELTA[EXPIRY == _e])] for _e in range(len(EXPIRY_DATES))]
            _note = (f"**A stressed market.** At the money, short-dated options are more expensive than long-dated ones: "
                     f"{_atm[0]:.0f}% at {EXPIRY_DAYS[0]} days against {_atm[1]:.0f}% at {EXPIRY_DAYS[-1]} days, an inverted "
                     f"term structure. Deep puts trade at {min(_deep):.0f}–{max(_deep):.0f}%. Index options look like this "
                     "after a sell-off, when current volatility is far above its long-run level. One feature is specific to "
                     "Heston: in standardized-moneyness units its short-dated smiles are flatter than its long-dated ones, "
                     "so the curves cross. Real markets keep a steep skew at short expiries (in the first figure of this "
                     "notebook the seven market smiles are nearly parallel); Heston cannot, and Section 5.4 shows the "
                     "consequence on the market quotes. As a test of the inference the case is fully valid: it is a draw from the model and the prior.\n\n"
                     + _note)
    mo.vstack([
        _fig,
        mo.md(_note),
        mo.ui.table(_rows, selection=None, pagination=False),
        _corner,
        mo.md(f"""
        **How to read the corner plot.** The diagonal shows the posterior of each parameter; each panel below it shows the joint posterior of a pair. The posteriors are very narrow: {TEXT_LABELS[NAMES.index(_sharpest)]} is determined to within about ±{100 * _relative[_sharpest]:.1f}% of its value, and even the least precise Heston parameter to within ±{100 * max(_relative.values()):.0f}%. Several pairs are strongly dependent; the strongest is {TEXT_LABELS[_j]} with {TEXT_LABELS[_i]} (correlation {_corr[_i, _j]:+.2f}). The quotes determine certain **combinations** of parameters more precisely than each parameter alone, because different combinations produce almost the same smile. Uncertainty statements about any quantity that depends on several parameters must therefore use the draws jointly, never the marginals separately.
        """),
        mo.md("Below, each posterior draw is mapped to the expected average variance over a fixed horizon, shown as a "
              "volatility. Such quantities, the fair values of variance swaps, are what a trader would price with the "
              "calibrated model. They inherit the precision of the quotes at those horizons."),
        mo.ui.table(_product_rows, selection=None, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo, workflow):
    mo.vstack([workflow("Check"), mo.md(r"""
    ## 5. Check

    A posterior is only useful if its uncertainty statements can be trusted. We check the network in four ways:

    | Check | Question | What to look for |
    |:---|:---|:---|
    | Posterior predictive | Can the inferred parameters regenerate the quotes? | the quotes fall within the replicated range |
    | Calibration across simulations | Does a 90% interval contain the truth 90% of the time? | coverage near nominal, uniform ranks |
    | Exact reference | Does the network reproduce the posterior for one data set? | agreement with the PyMC reference |
    | Real data | Can the model itself describe the market? | residuals within the assumed noise |

    ### 5.1 Posterior predictive

    Draw parameters from the posterior, price the 63 contracts and add fresh quote noise with each draw's own $c$:

    $$\boldsymbol\theta^{(m)}\sim q(\boldsymbol\theta\mid\mathbf y),\qquad
      \widetilde y^{(m)}_i=P_i(\boldsymbol\theta^{(m)})+c^{(m)}h_i\,\epsilon^{(m)}_i .$$

    The model-price band shows what the parameter uncertainty implies for prices; the replicated-quote band adds the observation noise. If the model and the posterior are right, the observed quotes should look like one more replicated data set.
    """)])
    return


@app.cell(hide_code=True)
def _(EXPIRY_DATES, EXPIRY_DAYS, mo):
    expiry_choice = mo.ui.dropdown(options={f"{_date} ({_days} days)": _e for _e, (_date, _days) in enumerate(zip(EXPIRY_DATES, EXPIRY_DAYS))},
                                   value=f"{EXPIRY_DATES[4]} ({EXPIRY_DAYS[4]} days)", label="Expiry")
    mo.hstack([expiry_choice, mo.md("*Data set: as selected in Section 4.*")], justify="start", gap=1.5)
    return (expiry_choice,)


@app.cell(hide_code=True)
def _(BOUNDS, CASES, CASE_TITLES, CLEAN, DELTA, EXPIRY, H, case_choice, expiry_choice, implied_vols, mo, neural_draws, np, pilot, plt, reference_draws):
    _case = case_choice.value
    _rng = np.random.default_rng(20261112)
    _sets = {}
    _note = ""
    if _case in neural_draws:
        _all = neural_draws[_case]
        _inside = _all[np.all((_all >= BOUNDS[:, 0]) & (_all <= BOUNDS[:, 1]), axis=1)]
        if len(_inside) >= 2:
            _sets["network posterior"] = _inside[_rng.choice(len(_inside), min(40, len(_inside)), replace=False)]
        _note = (f"Network draws outside the prior range ({len(_all) - len(_inside)} of {len(_all):,}) are left out here, "
                 "because the pricer is checked on the prior's range.")
    _sets["PyMC reference"] = reference_draws[_case][_rng.choice(len(reference_draws[_case]), 40, replace=False)]
    _y = CASES[_case]
    _at = EXPIRY == expiry_choice.value
    _fig, _axes = plt.subplots(2, len(_sets), figsize=(5.2 * len(_sets), 5.6), sharex=True, squeeze=False,
                               constrained_layout=True)
    _misfit, _typical_c = None, None
    for _col, (_name, _draws) in enumerate(_sets.items()):
        _prices = np.array([pilot.prices(_row) for _row in _draws])
        _replicated = _prices + _draws[:, -1:] * H * _rng.standard_normal(_prices.shape)
        _model_iv = np.nanquantile(implied_vols(_prices)[:, _at], [.05, .5, .95], axis=0)
        _rep_iv = np.nanquantile(implied_vols(_replicated)[:, _at], [.05, .5, .95], axis=0)
        _top, _bottom = _axes[:, _col]
        _top.fill_between(DELTA[_at], 100 * _rep_iv[0], 100 * _rep_iv[2], color="#9bd3ba", alpha=.55, label="replicated quotes (90%)")
        _top.fill_between(DELTA[_at], 100 * _model_iv[0], 100 * _model_iv[2], color="#7f53ad", alpha=.5, label="model prices (90%)")
        _top.scatter(DELTA[_at], 100 * implied_vols(_y)[0][_at], color="#222222", s=20, zorder=4, label="observed quotes")
        if _case in CLEAN:
            _top.plot(DELTA[_at], 100 * implied_vols(CLEAN[_case])[0][_at], "--", color="#333333", lw=1, label="true Heston prices")
        _top.set(title=_name, ylabel="implied volatility (%)" if _col == 0 else None)
        _model_dev = np.quantile((_prices - _y)[:, _at] / H[_at], [.05, .95], axis=0)
        _rep_dev = np.quantile((_replicated - _y)[:, _at] / H[_at], [.05, .95], axis=0)
        _bottom.fill_between(DELTA[_at], *_rep_dev, color="#9bd3ba", alpha=.55)
        _bottom.fill_between(DELTA[_at], *_model_dev, color="#7f53ad", alpha=.5)
        _bottom.axhline(0, color="#222222", lw=1)
        if _case in CLEAN:
            _bottom.plot(DELTA[_at], ((CLEAN[_case] - _y) / H)[_at], "--", color="#333333", lw=1)
        _bottom.set(xlabel="standardized moneyness δ", ylabel="model − observed, in half-spreads" if _col == 0 else None)
        if _name == "PyMC reference":
            _misfit = float(np.sqrt(np.mean((((_prices.mean(axis=0) - _y) / H)[_at]) ** 2)))
            _typical_c = float(np.median(_draws[:, -1]))
    _axes[0, 0].legend(fontsize=6.5, loc="upper right")
    _verdict = (f"Here the posterior-mean prices miss the quotes by {_misfit:.1f} half-spreads (root mean square), in line with "
                f"the inferred noise scale $c\\approx{_typical_c:.2f}$: the quotes look like one more replicated data set, as they "
                "should when the model is right."
                if _misfit < 2.5 * _typical_c else
                f"Here the posterior-mean prices miss the quotes by {_misfit:.1f} half-spreads (root mean square), far more than "
                f"the inferred noise scale $c\\approx{_typical_c:.2f}$ allows, and the miss has a systematic pattern across "
                "strikes: the model cannot reproduce these quotes (Section 5.4).")
    _denoise = ("" if _case not in CLEAN else
                " The dashed line marks the true, noise-free prices. The purple band follows it closely: the posterior "
                "recovers the true prices more precisely than any single quote measures them.")
    mo.vstack([_fig, mo.md(
        f"{CASE_TITLES[_case]}, one expiry; bands are pointwise 90% ranges over 40 posterior draws. *Top:* implied "
        "volatilities, where the bands are so narrow that they hide behind the points. *Bottom:* the same quantities "
        "minus the observed quotes, in half-spreads, where the differences become visible. Each observed quote contains "
        "its own noise, so the model prices (purple) differ from it by about one noise standard deviation, $c$ "
        "half-spreads, in either direction. The replicated quotes (green) add fresh noise, so the zero line, which marks "
        f"the observed quotes, should lie inside the green band.{_denoise} **{_verdict}** {_note}")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.2 Calibration across simulated data sets

    One posterior can look plausible and still be miscalibrated. To test the uncertainty statements themselves we need many data sets with known truth, so we use **512 simulated data sets**, drawing 512 posterior samples for each, and ask two questions.

    **Coverage.** How often does the central 90% interval contain the truth? It should be close to 90%: much more means the intervals are too wide, much less that they are too narrow.

    **Simulation-based calibration (SBC).** For each data set, compute the **rank** of the true value among the posterior draws: the fraction of draws that fall below it. If the posterior is correct, the truth is statistically indistinguishable from one more posterior draw, so its rank is equally likely to be anywhere between 0 and 1. Across 512 data sets the ranks should then be **uniformly distributed**. We plot the empirical CDF of the ranks minus the uniform CDF: a calibrated posterior gives a curve that wanders around zero within the grey band, and systematic shapes reveal specific problems.

    The four synthetic examples below show the typical shapes.
    """)
    return


@app.cell(hide_code=True)
def _(mo, norm, np, plots, plt):
    _rng = np.random.default_rng(7)
    _n = 512
    _truth = _rng.standard_normal(_n)
    _data = _truth + 0.5 * _rng.standard_normal(_n)
    _mean, _sd = 0.8 * _data, np.sqrt(0.2)  # exact posterior for a N(0, 1) prior and N(truth, 0.5^2) data
    _cases = {"calibrated": (0.0, 1.0), "biased upwards": (0.6, 1.0), "too narrow (overconfident)": (0.0, 0.6),
              "too wide (underconfident)": (0.0, 1.7)}
    _grid = np.linspace(0, 1, 201)
    _band = np.sqrt(np.log(2 / 0.05) / (2 * _n))
    _fig, _axes = plt.subplots(1, 4, figsize=(12.4, 2.6), sharey=True, constrained_layout=True)
    for _ax, (_title, (_shift, _scale)) in zip(_axes, _cases.items()):
        _ranks = norm.cdf((_truth - _mean - _shift * _sd) / (_scale * _sd))
        _ecdf = (np.sort(_ranks)[None, :] <= _grid[:, None]).mean(axis=1)
        _ax.fill_between(_grid, -_band, _band, color=plots.BASELINE, alpha=.45, lw=0)
        _ax.axhline(0, color=plots.INK, lw=.6)
        _ax.plot(_grid, _ecdf - _grid, color=plots.SERIES[0] if _title == "calibrated" else plots.CRITICAL, lw=1.5)
        _ax.set(title=_title, xlabel="rank of the truth")
    _axes[0].set_ylabel("ECDF − uniform")
    mo.vstack([
        _fig,
        mo.md("**How to read rank plots.** *Calibrated:* the curve stays inside the band. *Biased upwards:* posterior draws "
              "tend to lie above the truth, so ranks pile up near zero and the curve bulges **above** zero; a downward bias "
              "gives the mirror image. *Too narrow:* the truth often falls in the tails of the posterior, so ranks pile up "
              "near 0 and 1; the curve rises above zero on the left and falls below on the right. *Too wide:* the truth "
              "almost always sits in the middle of the posterior, ranks pile up around 0.5, and the curve dips **below** "
              "zero on the left and rises above it on the right."),
    ])
    return


@app.cell(hide_code=True)
def _(BOUNDS, EVALUATION, LABELS, NAMES, TEXT_LABELS, json, mo, np, plots, plt):
    _population = json.loads((EVALUATION / "evaluation.json").read_text())["population"]
    _ranks = np.asarray(_population["fractional_ranks"])
    _band = _population["rank_dkw"]["dkw_bound"]
    _passed = _population["rank_dkw"]["passed"]
    with np.load(EVALUATION / "recovery.npz", allow_pickle=False) as _archive:
        _truth, _quantiles = _archive["truth"], _archive["quantiles"].astype(float)  # 5%, 50%, 95% per case
    _grid = np.linspace(0, 1, 201)
    _fig, _axes = plt.subplots(2, 3, figsize=(10.4, 4.9), sharex=True, sharey=True, constrained_layout=True)
    for _j, _ax in enumerate(_axes.flat):
        _ecdf = (np.sort(_ranks[:, _j])[None, :] <= _grid[:, None]).mean(axis=1)
        _ax.fill_between(_grid, -_band, _band, color=plots.BASELINE, alpha=.45, lw=0)
        _ax.axhline(0, color=plots.INK, lw=.6)
        _ax.plot(_grid, _ecdf - _grid, color=plots.SERIES[0] if _passed[_j] else plots.CRITICAL, lw=1.5)
        _ax.set_title(LABELS[_j] + ("" if _passed[_j] else "  (leaves the band)"), fontsize=10)
        if _j >= 3:
            _ax.set_xlabel("rank of the truth")
    _axes[0, 0].set_ylabel("ECDF − uniform")
    _axes[1, 0].set_ylabel("ECDF − uniform")

    _shown = np.arange(0, len(_truth), 8)
    _fig2, _axes2 = plt.subplots(2, 3, figsize=(10.4, 5.4), constrained_layout=True)
    for _j, _ax in enumerate(_axes2.flat):
        _lo, _hi = BOUNDS[_j]
        _ax.plot([_lo, _hi], [_lo, _hi], "--", color=plots.MUTED, lw=1)
        _ax.scatter(_truth[:, _j], _quantiles[1, :, _j], s=5, color=plots.SERIES[0], alpha=.55)
        _ax.errorbar(_truth[_shown, _j], _quantiles[1, _shown, _j],
                     yerr=[_quantiles[1, _shown, _j] - _quantiles[0, _shown, _j], _quantiles[2, _shown, _j] - _quantiles[1, _shown, _j]],
                     fmt="none", ecolor=plots.SERIES[0], alpha=.6, lw=.8)
        _ax.set_title(LABELS[_j], fontsize=10)
        _ax.set(xlabel="true value", ylabel="posterior median" if _j % 3 == 0 else None)

    _coverage = _population["coverage"]
    _support = _population["support"]
    _rows = []
    for _j, _label in enumerate(TEXT_LABELS):
        _ci = _coverage["0.9"]["wilson"][_j]
        _rows.append({
            "parameter": _label,
            "90% coverage [95% CI]": f"{100 * _coverage['0.9']['coverage'][_j]:.1f}% [{100 * _ci[0]:.1f}, {100 * _ci[1]:.1f}]",
            "50 / 80 / 95% coverage": " / ".join(f"{100 * _coverage[_level]['coverage'][_j]:.0f}%" for _level in ("0.5", "0.8", "0.95")),
            "rank plot": "within band" if _passed[_j] else "leaves band",
            "draws outside prior": f"{100 * _support['outside_prior']['per_dimension_count'][_j] / _support['samples']:.2f}%",
        })
    # Everything below is read off the saved evaluation.
    _r = [float(np.corrcoef(_quantiles[1, :, _j], _truth[:, _j])[0, 1]) for _j in range(len(NAMES))]
    _hardest = int(np.argmin(_r))
    _recovery_text = ("**Recovery.** Posterior medians against the true values for all 512 data sets, with 90% intervals for "
                      "every eighth. Points on the dashed line mean perfect recovery. Correlation between posterior median "
                      "and truth: " + ", ".join(f"{_l} {_v:.3f}" for _l, _v in zip(TEXT_LABELS, _r)) + ". "
                      f"The hardest to pin down is {TEXT_LABELS[_hardest]}"
                      + (", which is identified only through the size of the quote errors." if NAMES[_hardest] == "c" else "."))
    _cis = _coverage["0.9"]["wilson"]
    _good = [_j for _j in range(len(NAMES)) if _passed[_j] and _cis[_j][0] <= 0.9 <= _cis[_j][1]]
    _bad = [_j for _j in range(len(NAMES)) if _j not in _good]
    _lesson = (f"**What we learn.** {len(_good)} of the {len(NAMES)} parameters are well calibrated: the 95% confidence "
               "interval of their 90% coverage includes 90%, and their rank curves stay inside the band.")
    for _j in _bad:
        _diff = lambda _x: np.mean(_ranks[:, _j] <= _x) - _x
        _low, _high = _diff(0.25), _diff(0.75)
        _shape = ("the *too wide* shape: below zero on the left, above zero on the right" if _low < 0 < _high else
                  "the *too narrow* shape: above zero on the left, below zero on the right" if _high < 0 < _low else
                  "the shape of a *bias*: the curve stays on one side of zero")
        _cov = _coverage["0.9"]["coverage"][_j]
        _lesson += (f" The exception is **{TEXT_LABELS[_j]}**: its 90% intervals contain the truth {100 * _cov:.1f}% of "
                    f"the time, and its rank curve has {_shape}.")
        if _cov > 0.9 and _low < 0 < _high:
            _lesson += " The network is **conservative** about this parameter rather than overconfident."
        if NAMES[_j] == "initial_variance":
            _lesson += (" A plausible explanation is precision: the shortest-dated quotes pin the initial variance down to "
                        f"an average posterior standard deviation of {_population['mean_posterior_sd'][_j]:.5f}, on a "
                        f"prior range of {BOUNDS[_j, 1] - BOUNDS[_j, 0]:.2f}, which may be finer than the network resolves.")
    _counts = np.asarray(_support["outside_prior"]["per_dimension_count"])
    _top = [LABELS[_k] for _k in np.argsort(_counts)[::-1][:2]]
    _lesson += (f" {100 * _support['outside_prior']['sample_fraction']:.1f}% of raw draws fall slightly outside the prior "
                f"range, most often in {_top[0]} and {_top[1]}; they are kept in every check.")
    mo.vstack([
        mo.md("**Results for the pretrained estimator** on 512 simulated data sets with 512 posterior draws each:"),
        mo.ui.table(_rows, selection=None, pagination=False),
        _fig,
        mo.md("**Rank plots** for the six parameters, read as above. The grey band is the simultaneous 95% band for six "
              "parameters."),
        _fig2,
        mo.md(_recovery_text),
        mo.callout(mo.md(_lesson), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    panel_button = mo.ui.run_button(label="Resample the 512 data sets with the pretrained estimator (a few minutes)")
    panel_button
    return (panel_button,)


@app.cell(hide_code=True)
def _(BUNDLE, BUNDLE_HELP, EVAL_SEED, LABELS, NAMES, PANEL_BATCH, PANEL_DRAWS, bf, delivery, estimator, mo, np, panel_button, plots, time):
    mo.stop(not panel_button.value, mo.md("*The figures above can be reproduced here: press the button to draw "
                                         "512 × 512 fresh posterior samples and recompute the diagnostics with BayesFlow.*"))
    mo.stop(BUNDLE is None or estimator is None, BUNDLE_HELP)
    _started = time.perf_counter()
    _, _arrays, _ = delivery.load_bank(BUNDLE)
    _quotes = _arrays["historical_quotes"]
    _draws = np.concatenate([np.asarray(estimator.sample(_quotes[_start:_start + PANEL_BATCH], PANEL_DRAWS,
                                                         seed=EVAL_SEED + 1000 + _start))
                             for _start in range(0, len(_quotes), PANEL_BATCH)])
    _truth = np.column_stack([_arrays[f"historical_{_name}"][:, 0] for _name in NAMES])
    _estimates = {_name: _draws[..., _j:_j + 1] for _j, _name in enumerate(NAMES)}
    _targets = {_name: _truth[:, _j:_j + 1] for _j, _name in enumerate(NAMES)}
    mo.vstack([
        mo.md(f"{len(_quotes)} data sets × {PANEL_DRAWS} posterior draws in {time.perf_counter() - _started:.0f} s."),
        bf.diagnostics.calibration_ecdf(estimates=_estimates, targets=_targets, variable_keys=NAMES, variable_names=LABELS,
                                        difference=True, rank_ecdf_color=plots.SERIES[0], fill_color=plots.BASELINE),
        bf.diagnostics.recovery(estimates=_estimates, targets=_targets, variable_keys=NAMES, variable_names=LABELS,
                                uncertainty_agg_kwargs={"prob": 0.9}, color=plots.SERIES[0]),
    ])
    return


@app.cell(hide_code=True)
def _(BOUNDS, CASES, NAMES, TEXT_LABELS, mo, neural_draws, np, reference_summary):
    mo.stop(not neural_draws, mo.vstack([
        mo.md("### 5.3 One data set at a time: comparison with the exact reference"),
        mo.md("*This comparison needs the pretrained estimator.*")]))
    _columns = {"synthetic-market-like": "market-like simulation", "synthetic-broad": "stressed simulation", "market": "market quotes"}
    _rows, _gaps, _ratios, _outside = [], {}, {}, {}
    for _j, _name in enumerate(NAMES):
        _row = {"parameter": TEXT_LABELS[_j]}
        for _case_id in CASES:
            _draws = neural_draws[_case_id][:, _j]
            _ref = reference_summary["cases"][_case_id]["parameters"][_name]
            _rq = _ref["regenerated_q05_q50_q95"]
            _q = np.quantile(_draws, [.05, .95])
            _gap = (_draws.mean() - _ref["regenerated_mean"]) / _ref["regenerated_sd"]
            _ratio = (_q[1] - _q[0]) / (_rq[2] - _rq[0])
            _gaps.setdefault(_case_id, []).append(abs(_gap))
            _ratios.setdefault(_case_id, []).append(_ratio)
            _row[_columns[_case_id]] = f"{_gap:+.2f} / {_ratio:.2f}"
        _rows.append(_row)
    for _case_id in CASES:
        _all = neural_draws[_case_id]
        _outside[_case_id] = np.mean(np.any((_all < BOUNDS[:, 0]) | (_all > BOUNDS[:, 1]), axis=1))
    _synthetic = ("synthetic-market-like", "synthetic-broad")
    _all_gaps = np.concatenate([_gaps[_c] for _c in _synthetic])
    _all_ratios = np.concatenate([_ratios[_c] for _c in _synthetic])
    _agree = int(np.sum((_all_gaps < 0.3) & (np.abs(_all_ratios - 1) < 0.25)))
    _wider, _narrower = int(np.sum(_all_ratios > 1.25)), int(np.sum(_all_ratios < 1))
    _conservative = ("The network is **conservative** here: it does not claim more precision than the data support, "
                     "but it does not extract all the information in the quotes either. " if _narrower <= 2 else "")
    mo.vstack([
        mo.md("### 5.3 One data set at a time: comparison with the exact reference\n\n"
              "Each entry is *(network mean − reference mean) / reference SD* followed by *network 90% width / reference "
              "90% width*. Perfect agreement would be 0 and 1; we read differences below 0.3 SD and width ratios between "
              "0.75 and 1.25 as agreement."),
        mo.ui.table(_rows, selection=None, pagination=False),
        mo.md(f"**What we learn.** On the two simulated data sets, {_agree} of the 12 parameter posteriors meet both "
              f"criteria. The centres are mostly close to the reference (median difference {np.median(_all_gaps):.2f} SD, "
              f"largest {_all_gaps.max():.2f} SD), but the intervals are wider: {_wider} of the 12 are more than 25% wider "
              f"than the exact ones, up to {_all_ratios.max():.1f} times, and {_narrower} "
              f"{'is' if _narrower == 1 else 'are'} narrower. {_conservative}This is compatible with the good calibration "
              "of Section 5.2, because calibration is necessary but not sufficient: a posterior equal to the prior is "
              "perfectly calibrated and useless. Only a comparison with an exact posterior shows how much sharpness is "
              f"lost. On the market quotes the network departs far from the reference, and {100 * _outside['market']:.0f}% "
              "of its draws leave the prior range. The next section explains why."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.4 The market quotes

    All the checks so far assume that the model is right: the data were simulated from it. On real quotes the first question is whether **any** Heston parameters reproduce the data within the assumed noise.
    """)
    return


@app.cell(hide_code=True)
def _(BOUNDS, CASES, DELTA, EXPIRY, EXPIRY_DATES, EXPIRY_DAYS, H, MARKET_IV, implied_vols, mo, neural_draws, np, pilot, plots, plt, reference_draws):
    _y = CASES["market"]
    _rng = np.random.default_rng(20261113)
    _reference = reference_draws["market"]
    _sets = {"PyMC reference": (_reference, _reference[_rng.choice(len(_reference), 40, replace=False)], plots.SERIES[1])}
    _neural = neural_draws.get("market")
    if _neural is not None:
        _inside = _neural[np.all((_neural >= BOUNDS[:, 0]) & (_neural <= BOUNDS[:, 1]), axis=1)]
        if len(_inside) >= 2:
            _sets["network posterior"] = (_neural, _inside[_rng.choice(len(_inside), min(40, len(_inside)), replace=False)],
                                          plots.SERIES[0])
    _prices = {_name: np.array([pilot.prices(_row) for _row in _sample]) for _name, (_, _sample, _) in _sets.items()}
    _fig, _axes = plt.subplots(2, 4, figsize=(11.6, 5.4), constrained_layout=True)
    for _e, _ax in enumerate(_axes.flat[:7]):
        _at = EXPIRY == _e
        for _name, (_, _, _color) in _sets.items():
            _band = np.nanquantile(implied_vols(_prices[_name])[:, _at], [.05, .95], axis=0)
            _ax.fill_between(DELTA[_at], 100 * _band[0], 100 * _band[1], color=_color, alpha=.35, label=f"{_name} (90%)")
        _ax.scatter(DELTA[_at], 100 * MARKET_IV[_at], color="#222222", s=14, zorder=4, label="market")
        _ax.set_title(f"{EXPIRY_DATES[_e]} ({EXPIRY_DAYS[_e]} d)", fontsize=8.5)
    _axes[0, 0].legend(fontsize=6, loc="upper right")
    _last = _axes.flat[7]
    for _name, (_all, _, _color) in _sets.items():
        _last.hist(_all[:, -1], bins=40, density=True, histtype="step", color=_color, lw=1.3, label=_name)
    _last.axvline(2.0, color=plots.INK, ls="--", lw=1, label="prior bound")
    _last.set(title="noise multiplier c", xlabel="c", yticks=[])
    _last.legend(fontsize=6)
    for _ax in _axes[:, 0]:
        _ax.set_ylabel("implied volatility (%)")
    for _ax in list(_axes[1, :3]) + [_axes[0, 3]]:
        _ax.set_xlabel("standardized moneyness δ")
    _residual = (_y - _prices["PyMC reference"].mean(axis=0)) / H
    _rms = float(np.sqrt(np.mean(_residual ** 2)))
    _rows = [{"expiry": f"{_date} ({EXPIRY_DAYS[_e]} d)",
              "RMS residual (half-spreads)": round(float(np.sqrt(np.mean(_residual[EXPIRY == _e] ** 2))), 1),
              "largest |residual| (half-spreads)": round(float(np.max(np.abs(_residual[EXPIRY == _e]))), 1),
              "at moneyness δ": round(float(DELTA[EXPIRY == _e][np.argmax(np.abs(_residual[EXPIRY == _e]))]), 2)}
             for _e, _date in enumerate(EXPIRY_DATES)]
    _wing = sum(DELTA[EXPIRY == _e][np.argmax(np.abs(_residual[EXPIRY == _e]))] < -4 for _e in range(len(EXPIRY_DATES)))
    _network_text = ("" if _neural is None else
                     f" Its posterior is shifted and much wider than the reference, and {100 * np.mean(_neural[:, -1] > 2):.0f}% "
                     "of its draws have $c>2$, outside the prior range.")
    mo.vstack([
        _fig,
        mo.md("*Bands:* implied volatilities of the model prices from 40 posterior draws; *points:* market mid quotes; *last "
              "panel:* the posterior of the noise multiplier $c$."),
        mo.ui.table(_rows, selection=None, pagination=False),
        mo.md(f"""
        **What the reference posterior says.** The PyMC posterior is the exact answer to the question we asked: which Heston parameters and which noise multiplier are consistent with these quotes, under our prior. Its answer is that **no Heston smile fits the quotes within the assumed noise**. The posterior-mean prices miss the quotes by **{_rms:.1f} half-spreads** on average (root mean square, table above), while the prior allows a noise standard deviation of at most two half-spreads ($c\\le2$). The likelihood would favour an even larger $c$; the posterior piles up against the bound $c=2$, and the Heston parameters settle on the best available compromise. At {_wing} of {len(EXPIRY_DATES)} expiries the largest residual is at the deepest put ($\\delta\\approx-5$). There the market's implied volatility lies up to several points above every posterior Heston smile, most at the shortest expiries: Heston cannot produce a short-dated skew this steep together with the rest of the surface.

        **Why the network disagrees with the reference.** The network learned from simulations in which the quotes scatter around a Heston smile with a noise standard deviation of at most two half-spreads. These market quotes are {_rms:.0f} half-spreads away on average, a situation it never saw in training. Like any learned function, it is only reliable on inputs that resemble its training data; here it extrapolates.{_network_text} The disagreement is therefore a symptom of **model misspecification**, not a failure of training. It is also a useful alarm: an amortized estimator should only be trusted on data its simulator could have produced, which is exactly what the posterior predictive and the comparison with an exact reference test.

        **What to do about it** is a modelling decision. One could widen or reshape the noise model, add an explicit model-error term, or use a richer model, for example with jumps or rough volatility. The simulation checks measure how well the network approximates the posterior; only checks on real data can tell whether that posterior describes the market.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    1. **Simulation-based inference turns a pricer into a posterior.** Prior, pricer and noise model define a simulator; one round of training gives an estimator that returns the posterior for any 63 quotes in seconds.
    2. **Liquid quotes are very informative.** Bid–ask spreads of a few hundredths to a few tenths of a volatility point pin the Heston parameters down tightly, and make any model error stand out.
    3. **Prepare the inputs.** Dividing quotes by their half-spreads and whitening them puts all 63 inputs on a common, uncorrelated scale. It also exposes the low-variance directions that carry the information about the noise level.
    4. **Check calibration across many simulations.** Coverage and rank plots show where the uncertainty can be trusted. Here five of six parameters are well calibrated, and the initial-variance intervals are too wide rather than too narrow.
    5. **Check single data sets against an exact reference.** Good calibration across simulations does not mean sharp individual posteriors: here the network's intervals can be several times wider than the exact ones.
    6. **Check the model, not only the network.** On the market quotes the noise multiplier hits its bound and the posterior Heston smiles miss by many half-spreads, mostly in the deep put wing. An amortized estimator is only reliable on data its simulator could have produced.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Running this notebook": mo.md(r"""
    The pretrained estimator and its training data are distributed as a separate 465 MB archive; `lab/plans/training-data-guide.md` explains how to download and verify it. From `lab/`, start marimo with `HESTON_BUNDLE` set to the extracted folder (the one that contains `bundle.json`):

    ```sh
    HESTON_BUNDLE=/path/to/bundle pixi run --locked -e core-reference marimo edit notebooks/heston_spx_calibration.py
    ```

    To render a static copy with every output:

    ```sh
    HESTON_BUNDLE=/path/to/bundle pixi run --locked -e core-reference marimo export html notebooks/heston_spx_calibration.py -o heston_spx_calibration.html
    ```

    Without `HESTON_BUNDLE`, results that need the network are left out and the PyMC reference is shown alone. The reference posteriors and their run records are in `lab/evidence/pilot-results/2026-11-11/heston-actual63-notebook/`. Posterior draws use fixed seeds, so the figures are reproducible.
    """)})
    return


if __name__ == "__main__":
    app.run()
