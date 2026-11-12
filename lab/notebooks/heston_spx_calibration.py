import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="Bayesian calibration of the Heston model on SPX options",
    css_file="heston_spx_calibration.css",
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
        ("Use", "price contracts the market does not quote, with a range instead of a number"),
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
            boxes.append(f'<div style="flex:1;min-width:96px;border:{_border};border-radius:10px;padding:9px 10px;'
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


@app.cell
def _(mo, workflow):
    mo.vstack([
        mo.md(r"""
        # Bayesian Calibration of the Heston Model on SPX Options

        We calibrate the **Heston stochastic-volatility model** to **63 SPX option quotes**. Instead of one best-fitting parameter vector, we compute a **posterior distribution** over the six unknowns $\boldsymbol\theta=(v_0,\kappa,v_\infty,\sigma_v,\rho,c)$.

        The posterior is learned with **amortized simulation-based inference** in **BayesFlow**: a neural network is trained once on simulated quotes, then returns the posterior for any new set of 63 quotes in seconds. The workflow that we will explore is described as follows:
        """),
        workflow(),
        mo.md(
        r"""In this notebook, we explicitly show where and how **BayesFlow** is used, including code snippets where useful."""
        )
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
                   "market": "SPX market quotes, 9 November 2026"}
    PRODUCTS = _panel["products"]
    PRODUCT_TITLES = {"control": f"average variance, now to {365 * PRODUCTS['control'][1]:.0f} days",
                      "affected": f"forward variance, {365 * PRODUCTS['affected'][0]:.0f} to {365 * PRODUCTS['affected'][1]:.0f} days"}
    return CASES, CASE_TITLES, CLEAN, PRODUCTS, PRODUCT_TITLES, TRUTHS


@app.cell
def _(DELTA, KIND, REF_VOL, design, mo):
    _puts = sorted({_q["target_z"] for _q in design["selected_quotes"] if _q["option_type"] == "put"})
    _calls = [_t for _t in design["targets_z"] if _t not in _puts]
    _fmt = lambda _values: ", ".join(f"{_v:g}" for _v in _values)
    mo.md(rf"""
    ## The data: 63 SPX option quotes

    - **Snapshot:** SPX index options at 20:13 UTC on 9 November 2026: **63 European, PM-settled SPXW contracts**, seven expiries from one week to about eleven months, nine strikes each.
    - **Per contract:** the bid and the ask give the **mid price** $y_i$ and the **half-spread** $h_i=(\text{{ask}}_i-\text{{bid}}_i)/2$, in index points.
    - **Per expiry:** the forward price $F$ and the discount factor $D$, estimated from put–call parity on the same snapshot.

    Strikes of different expiries are compared through the **standardized moneyness**

    $$\delta=\frac{{\ln(K/F)}}{{\sigma_{{\text{{ref}}}}\sqrt{{T}}}},\qquad \sigma_{{\text{{ref}}}}={100 * REF_VOL:.0f}\%,$$

    the distance from the forward in units of a typical move over the option's life ($T$ in years, actual/365).[^sigma] The contracts were selected on this scale: at every expiry, the puts nearest to $\delta={_fmt(_puts)}$ and the calls nearest to $\delta={_fmt(_calls)}$. They run from $\delta={DELTA.min():.1f}$ to ${DELTA.max():.1f}$ ({int((KIND == 'put').sum())} puts, {int((KIND == 'call').sum())} calls).

    [^sigma]: 16% is close to the at-the-money implied volatility of SPX on this snapshot (15–17% across the seven expiries), so one unit of δ is roughly one standard deviation of the log price at expiry.
    """)
    return


@app.cell
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
        - **Left:** every expiry has nine contracts at similar values of $\\delta$, two thirds of them puts.
        - **Middle:** the half-spread as a share of the price: {_relative.min():.2f}% at best near the money, up to {_relative.max():.0f}% for the cheapest puts ({MID.min():.2f} index points).
        - **Right:** the **volatility smile**, from {min(_atm):.0f}–{max(_atm):.0f}% at the money to {min(_wing):.0f}–{max(_wing):.0f}% for the deepest puts. The seven smiles are nearly parallel in $\\delta$: the skew per unit of $\\delta$ is about the same at one week as at eleven months. This is the feature the model must reproduce.

        **Note**: Implied volatilities are for display only and all computations use prices.
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


@app.cell
def _(mo, workflow):
    mo.vstack([workflow("Model"), mo.md(r"""
    ## 1. Model

    ### Heston dynamics

    The variance $V_t$ follows its own random process, correlated with the price. With $\rho<0$, volatility rises when prices fall, which produces the **skew** of the smile.

    $$
    \frac{dF_t}{F_t} = \sqrt{V_t}\,dW_t^F,\qquad
    dV_t = \kappa(v_\infty-V_t)\,dt+\sigma_v\sqrt{V_t}\,dW_t^V,
    \qquad d\langle W^F,W^V\rangle_t=\rho\,dt .
    $$
    where $W^F$ and $W^V$ are two Brownian motions with correlation $\rho$, and:


    | Symbol | Meaning |
    |:---|:---|
    | $v_0$ | initial variance $V_0$ |
    | $\kappa$ | speed of mean reversion, per year |
    | $v_\infty$ | long-run variance, the level $V_t$ reverts to |
    | $\sigma_v$ | volatility of variance |
    | $\rho$ | correlation of price and variance shocks |
    | $c$ | quote-noise multiplier (observation model below) |

    $F_t$ is the forward price for the option's expiry. The priors are discussed with the prior predictive check in Section 2.
    """)])
    return


@app.cell(hide_code=True)
def _(NAMES, TRUTHS, mo, reference_summary):
    _truth = TRUTHS["synthetic-market-like"]
    _market = {_n: reference_summary["cases"]["market"]["parameters"][_n]["regenerated_q05_q50_q95"][1] for _n in NAMES}
    _like = (2 * _truth[1] * _truth[2], _truth[3] ** 2)
    _fit = (2 * _market["kappa"] * _market["theta"], _market["vol_of_vol"] ** 2)
    mo.accordion({
        "Why there is no drift: the pricing measure and forward prices": mo.md(r"""
        - Prices are computed under the **pricing measure** $\mathbb Q$, under which discounted traded prices are martingales.
        - The **forward price** $F_t=S_t\,e^{(r-q)(T-t)}$ (interest rate $r$, dividend yield $q$) is then a martingale: no drift. At expiry $F_T=S_T$, so a call is worth $C=D\,\mathbb E^{\mathbb Q}\big[(F_T-K)^+\big]$ with $D=e^{-rT}$.
        - Rates and dividends enter only through $D$ and $F$, read from the market by put–call parity. Divided by $D\,F$, prices are in **forward units**: they depend only on $\ln(K/F)$, $T$ and the Heston parameters, so one parameter set prices every expiry.
        - The index's real-world expected return plays no role. Estimating it would need a time series of prices and a physical-measure model.
        """),
        "How option prices are computed: the Fourier pricer": mo.md(r"""
        - Heston is **affine**: the characteristic function $\varphi(u)=\mathbb E^{\mathbb Q}\big[e^{iu\ln(F_T/F_0)}\big]$ is known in closed form, so a European option needs one one-dimensional integral (Lewis, 2001).
        - The pricer uses a **control-variate** form: the Black–Scholes price $C_{\mathrm{BS}}$ with the same expected average variance, plus the integral of the difference of the two characteristic functions (puts get the same correction):

        $$C=C_{\mathrm{BS}}+D\,\frac{\sqrt{FK}}{\pi}\int_0^\infty\operatorname{Re}\!\left[e^{iu\ln(F/K)}\Big(\varphi_{\mathrm{BS}}\big(u-\tfrac i2\big)-\varphi\big(u-\tfrac i2\big)\Big)\right]\frac{du}{u^2+\tfrac14}.$$

        - **Why the control variate:** for very cheap options the plain formula is a small difference of two large numbers; this form avoids the cancellation.
        - **Accuracy:** Gauss–Legendre quadrature, with nodes and interval doubled until two successive steps agree to within 2% of the smallest noise standard deviation the prior allows for that contract ($c=0.5$, that is 1% of its half-spread).
        - **Speed:** milliseconds for all 63 options. The prior predictive check in Section 2 times 1,000 simulated data sets.
        """),
        "The Feller condition": mo.md(rf"""
        - If $2\kappa v_\infty\ge\sigma_v^2$, the variance never reaches zero. Otherwise it can touch zero and is pushed straight back up by the drift. The model and the pricer are valid either way, and the prior does not impose the condition.
        - The market-like parameters of Section 4 violate it ($\sigma_v^2={_like[1]:.1f}$ against $2\kappa v_\infty={_like[0]:.2f}$), and so do the SPX market posterior medians ($\sigma_v^2={_fit[1]:.1f}$ against $2\kappa v_\infty={_fit[0]:.2f}$): the posterior fits the SPX skew with a volatility of variance far above the Feller limit.
        """),
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Observation model: noise set by the bid–ask spread

    $$
    y_i = P_i(v_0,\kappa,v_\infty,\sigma_v,\rho) + c\,h_i\,\epsilon_i,\qquad
    \epsilon_i \overset{\mathrm{iid}}{\sim}\mathcal N(0,1),\qquad i=1,\dots,63
    $$

    - $y_i$: the quote (mid price) of contract $i$. $P_i$: its Heston price.
    - $h_i$: **data**, the observed half-spread. Fixed, and identical in every simulation.
    - $\epsilon_i$: a fresh standard-normal draw for every contract and every simulated data set.
    - $c$: a **parameter**, inferred with the others. It scales all 63 noise levels together: "one half-spread is one standard deviation" is a convention that $c$ can correct.

    **Note:** a **model-error term** is left out. Heston is assumed to price every contract exactly, so a systematic gap between Heston and the market can only be absorbed by a larger $c$ (Section 5.4).
    """)
    return


@app.cell(hide_code=True)
def _(
    DELTA,
    EXPIRY,
    EXPIRY_COLORS,
    EXPIRY_DATES,
    H,
    MARKET_IV,
    mo,
    np,
    plt,
    vol_point_noise,
):
    _points = vol_point_noise(H, MARKET_IV)
    _fig, _ax = plt.subplots(figsize=(6.4, 2.9), constrained_layout=True)
    for _e, _date in enumerate(EXPIRY_DATES):
        _at = EXPIRY == _e
        _ax.plot(DELTA[_at], _points[_at], "o-", ms=2.5, lw=.8, color=EXPIRY_COLORS[_e], label=_date)
    _ax.set(yscale="log", xlabel="standardized moneyness δ", ylabel="implied-volatility points",
            title="One half-spread, in volatility points")
    _ticks = [0.02, 0.05, 0.1, 0.2]
    _ax.set_yticks(_ticks, labels=[f"{_t:g}" for _t in _ticks])
    _ax.minorticks_off()
    _ax.legend(fontsize=6, ncol=2)
    _range = 100 * (np.nanmax(MARKET_IV) - np.nanmin(MARKET_IV))
    mo.vstack([_fig, mo.md(
        f"- One half-spread is **{np.nanmin(_points):.2f}–{np.nanmax(_points):.2f} volatility points**, against a smile "
        f"that spans about {_range:.0f} points.\n"
        "- The quotes are therefore very informative, and the posteriors in Section 4 are narrow.\n"
        "- The flip side: a systematic model error of even a fraction of a volatility point is large compared with "
        "the assumed noise.")])
    return


@app.cell
def _(mo, workflow):
    mo.vstack([workflow("Simulate"), mo.md(r"""
    ## 2. Simulate

    One simulation corresponds to a **forward pass** through the pricer plus the addition of noise, and produces one hypothetical market:

    1. draw $\boldsymbol\theta=(v_0,\kappa,v_\infty,\sigma_v,\rho,c)$ from the prior,
    2. price the 63 contracts with the Fourier pricer,
    3. add the quote noise $c\,h_i\,\epsilon_i$.

    The output is a vector of **63 numbers**: in this toy-model only the noisy mid prices are simulated per each contract, while bids and asks are not simulated. The half-spreads $h_i$ keep their observed values and set the size of the noise. Move the sliders to see what the model produces.
    """)])
    return


@app.cell(hide_code=True)
def _(BOUNDS, EXPIRY_DATES, EXPIRY_DAYS, TRUTHS, mo, np):
    _start = TRUTHS["synthetic-market-like"]
    _steps = (0.001, 0.1, 0.001, 0.05, 0.01, 0.05)
    _digits = (3, 1, 3, 2, 2, 2)
    _labels = ("v₀ initial variance", "κ mean reversion", "v∞ long-run variance",
               "σᵥ volatility of variance", "ρ correlation", "c noise multiplier")
    sim_params = mo.ui.array([
        mo.ui.slider(start=float(BOUNDS[_j, 0]), stop=float(BOUNDS[_j, 1]), step=_steps[_j],
                     value=round(float(np.clip(_start[_j], BOUNDS[_j, 0], BOUNDS[_j, 1])), _digits[_j]),
                     label=_labels[_j], show_value=True)
        for _j in range(6)])
    sim_expiry = mo.ui.dropdown(
        options={f"{_date} ({_days} days)": _e for _e, (_date, _days) in enumerate(zip(EXPIRY_DATES, EXPIRY_DAYS))},
        value=f"{EXPIRY_DATES[4]} ({EXPIRY_DAYS[4]} days)", label="Quotes at expiry")
    sim_seed = mo.ui.number(start=0, stop=9999, step=1, value=1, label="Random seed")
    mo.vstack([
        mo.md("**Heston simulator.** The sliders start at the market-like parameters of Section 4."),
        mo.hstack([mo.vstack([sim_params[_j] for _j in range(3)]), mo.vstack([sim_params[_j] for _j in range(3, 6)])],
                  justify="start", gap=2),
        mo.hstack([sim_expiry, sim_seed], justify="start", gap=2),
    ])
    return sim_expiry, sim_params, sim_seed


@app.cell(hide_code=True)
def _(
    DELTA,
    EXPIRY,
    EXPIRY_DATES,
    H,
    MARKET_IV,
    implied_vols,
    mo,
    np,
    pilot,
    plots,
    plt,
    sim_expiry,
    sim_params,
    sim_seed,
    vol_point_noise,
):
    from scipy.stats import gamma as _gamma

    _theta = np.asarray(sim_params.value, dtype=float)
    _seed = int(sim_seed.value or 0)
    _v0, _kappa, _v_inf, _xi, _rho, _c = _theta
    _shape = 2 * _kappa * _v_inf / _xi ** 2  # the long-run law of V_t is a Gamma with this shape and mean v∞
    _median_vol = 100 * np.sqrt(_gamma.median(_shape, scale=_v_inf / _shape))

    # Illustrative paths: full-truncation Euler steps for the variance and the log forward. Pricing never uses paths.
    _days, _count, _sub = 252, 3, 16
    _dt = 1 / 252 / _sub
    _normals = np.random.default_rng(_seed).standard_normal((_count, _days, _sub, 2))
    _variance = np.full(_count, _v0)
    _log_forward = np.zeros((_count, _days + 1))
    _vol = np.empty((_count, _days + 1))
    _vol[:, 0] = np.sqrt(_v0)
    for _day in range(_days):
        _step = np.zeros(_count)
        for _k in range(_sub):
            _positive = np.maximum(_variance, 0.0)
            _scale = np.sqrt(_positive * _dt)
            _z_var, _z_other = _normals[:, _day, _k, :].T
            _step = _step - 0.5 * _positive * _dt + _scale * (_rho * _z_var + np.sqrt(1 - _rho ** 2) * _z_other)
            _variance = _variance + _kappa * (_v_inf - _positive) * _dt + _xi * _scale * _z_var
        _log_forward[:, _day + 1] = _log_forward[:, _day] + _step
        _vol[:, _day + 1] = np.sqrt(np.maximum(_variance, 0.0))
    _years = np.arange(_days + 1) / 252

    _failure = None
    try:
        _clean = pilot.prices(_theta)
    except (FloatingPointError, ValueError) as _error:
        _clean, _failure = None, str(_error)

    _at = EXPIRY == sim_expiry.value
    _fig, _axes = plt.subplots(2, 2, figsize=(10.8, 6.0), constrained_layout=True)
    _axes[0, 0].plot(_years, np.exp(_log_forward).T, lw=1)
    _axes[0, 0].set(xlabel="years", ylabel="forward / initial forward", title="Forward-price paths")
    _axes[0, 1].plot(_years, 100 * _vol.T, lw=1)
    _axes[0, 1].axhline(100 * np.sqrt(_v_inf), color="#888888", ls="--", lw=1, label=r"$\sqrt{v_\infty}$")
    _axes[0, 1].axhline(_median_vol, color="#888888", ls=":", lw=1, label="long-run median")
    _axes[0, 1].set(xlabel="years", ylabel="volatility (%)", title="Their hidden volatility")
    _axes[0, 1].legend(fontsize=7)
    if _clean is not None:
        _quotes = _clean + _c * H * np.random.default_rng(_seed + 1).standard_normal(H.size)
        _clean_iv, _quote_iv = implied_vols(_clean)[0], implied_vols(_quotes)[0]
        _sd = vol_point_noise(_c * H, _clean_iv)
        _axes[1, 0].scatter(DELTA[_at], 100 * MARKET_IV[_at], s=16, color="#bbbbbb", label="SPX market, 9 Nov 2026")
        _axes[1, 0].plot(DELTA[_at], 100 * _clean_iv[_at], "-", color=plots.MUTED, lw=1.3, label="Heston price")
        _axes[1, 0].scatter(DELTA[_at], 100 * _quote_iv[_at], s=18, color=plots.SERIES[2], zorder=3, label="simulated quote")
        _axes[1, 1].fill_between(DELTA[_at], -_sd[_at], _sd[_at], color=plots.SERIES[2], alpha=.18, lw=0, label="±1 noise SD")
        _axes[1, 1].axhline(0, color=plots.MUTED, lw=1)
        _axes[1, 1].plot(DELTA[_at], 100 * (_quote_iv - _clean_iv)[_at], "o", ms=4, color=plots.SERIES[2],
                         label="quote − Heston price")
        _axes[1, 0].legend(fontsize=7)
        _axes[1, 1].legend(fontsize=7)
    _axes[1, 0].set(xlabel="standardized moneyness δ", ylabel="implied volatility (%)",
                    title=f"Quotes at {EXPIRY_DATES[sim_expiry.value]}")
    _axes[1, 1].set(xlabel="standardized moneyness δ", ylabel="implied-volatility points", title="The quote errors, magnified")

    if _clean is None:
        _text = f"The pricer did not reach its accuracy target for these parameters ({_failure}); move a slider."
    else:
        _atm = 100 * _clean_iv[_at][np.argmin(np.abs(DELTA[_at]))]
        _deep = 100 * _clean_iv[_at][np.argmin(DELTA[_at])]
        _errors = np.abs(100 * (_quote_iv - _clean_iv))[_at]
        _feller = (
            f" A single path does not settle there: $\\sigma_v^2={_xi ** 2:.1f}$ exceeds $2\\kappa v_\\infty={2 * _kappa * _v_inf:.2f}$ "
            f"(Feller condition, Section 1), so in the long run volatility is below {_median_vol:.2g}% half the time "
            "(dotted line): short spikes carry the average, and in between the price barely moves." if _shape < 1 else "")
        _text = (
            f"- **Volatility:** starts at {100 * np.sqrt(_v0):.0f}%. The expected variance reverts to $v_\\infty$ with a "
            f"half-life of {12 * np.log(2) / _kappa:.1f} months (dashed line: $\\sqrt{{v_\\infty}}={100 * np.sqrt(_v_inf):.0f}\\%$)."
            f"{_feller}\n"
            "- **Price:** with $\\rho<0$, falls come with volatility spikes. The paths only illustrate the dynamics: "
            "the simulator prices options exactly and never draws paths.\n"
            f"- **Quotes:** at this expiry the Heston smile runs from {_atm:.1f}% at the money to {_deep:.1f}% at the "
            f"deepest put. The quote errors are at most {np.nanmax(_errors):.2f} volatility points"
            f"{': invisible on the smile' if np.nanmax(_errors) < 0.01 * np.ptp(_axes[1, 0].get_ylim()) else ''}.\n"
            "- **Try:** raise $\\sigma_v$ or make $\\rho$ more negative for a steeper skew; set $v_0$ above $v_\\infty$ for "
            "short expiries above long ones; lower $\\kappa$ to slow the return to $v_\\infty$; raise $c$ for larger "
            "quote errors. How close can you get to the SPX market (grey)?")
    mo.vstack([_fig, mo.md(_text)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The simulator in BayesFlow.** `bf.make_simulator` chains two functions by argument name: the prior returns named parameters, and the observation model receives them, prices the 63 contracts and adds the noise. It returns every quote **divided by its half-spread**, so that the network sees the data in noise units.
    """)
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
def _(BOUNDS, mo, np):
    _lo, _hi = BOUNDS[:, 0], BOUNDS[:, 1]
    _n = lambda _x: f"{_x:g}".replace("-", "−")
    _days = lambda _kappa: np.log(2) / _kappa * 365
    mo.md(rf"""
    ### Prior predictive check

    The **prior predictive distribution** is the distribution of quotes the model produces *before* seeing data: draw $\boldsymbol\theta$ from the prior, then simulate. Checking it answers two questions:

    - **Does it cover the observed quotes?** If not, the network would have to extrapolate, and the prior or the model would need rethinking.
    - **Is it not needlessly wide?** Simulations of implausible markets cost training effort and tell the network little.

    The prior is **independent and uniform** in each parameter's natural units:

    | Symbol | Prior | What the range means |
    |:---|:---|:---|
    | $v_0$ | uniform on {_n(_lo[0])} to {_n(_hi[0])} | initial volatility $\sqrt{{v_0}}$ between {100 * np.sqrt(_lo[0]):.0f}% and {100 * np.sqrt(_hi[0]):.0f}% |
    | $\kappa$ | uniform on {_n(_lo[1])} to {_n(_hi[1])} | half-life of a variance shock, $\ln 2/\kappa$, between {_days(_hi[1]):.0f} days and {_days(_lo[1]) / 365:.0f} years |
    | $v_\infty$ | uniform on {_n(_lo[2])} to {_n(_hi[2])} | long-run volatility between {100 * np.sqrt(_lo[2]):.0f}% and {100 * np.sqrt(_hi[2]):.0f}% |
    | $\sigma_v$ | uniform on {_n(_lo[3])} to {_n(_hi[3])} | large values allowed: the Feller condition is not imposed |
    | $\rho$ | uniform on {_n(_lo[4])} to {_n(_hi[4])} | negative only: volatility rises when prices fall |
    | $c$ | uniform on {_n(_lo[5])} to {_n(_hi[5])} | quote-noise SD between {_n(_lo[5])} and {_n(_hi[5])} half-spreads |

    The figure compares 1,000 simulated data sets with the SPX quotes.
    """)
    return


@app.cell(hide_code=True)
def _(
    DELTA,
    EXPIRY,
    EXPIRY_DATES,
    H,
    MARKET_IV,
    MID,
    implied_vols,
    make_simulator,
    mo,
    np,
    plots,
    plt,
    time,
):
    _started = time.perf_counter()
    prior_batch = make_simulator(20261111).sample(1000)
    _seconds = time.perf_counter() - _started
    _iv = implied_vols(prior_batch["observations"] * H)
    _fig, _axes = plt.subplots(1, 3, figsize=(11.0, 3.2), sharey=True, constrained_layout=True)
    for _ax, _e in zip(_axes, (0, 4, 6)):
        _at = EXPIRY == _e
        for _row in _iv[:150]:
            _ax.plot(DELTA[_at], 100 * _row[_at], color=plots.SERIES[0], alpha=.12, lw=.8)
        _ax.plot(DELTA[_at], 100 * MARKET_IV[_at], "o-", color=plots.SERIES[1], ms=3, lw=1.5, label="SPX market")
        _ax.set(title=f"{EXPIRY_DATES[_e]}", xlabel="standardized moneyness δ")
    _axes[0].set(ylabel="implied volatility (%)", ylim=(0, 70))
    _axes[0].legend(fontsize=7)
    _pct = 100 * np.mean(prior_batch["observations"] * H < MID, axis=0)   # the market quote's percentile, per contract
    _edge = int(np.sum((_pct == 0) | (_pct == 100)))
    _low = int(np.argmin(_pct))
    _cover = (f"**The market lies inside the simulated range at every contract:** its percentile among the simulated "
              f"prices runs from {_pct.min():.0f}% to {_pct.max():.0f}% (lowest at {EXPIRY_DATES[EXPIRY[_low]]}, "
              f"δ = {DELTA[_low]:.1f})." if _edge == 0 else
              f"**The market falls outside the simulated range at {_edge} of 63 contracts.**")
    mo.vstack([
        _fig,
        mo.md(f"- 150 of 1,000 prior-predictive smiles (blue) at three expiries, with the SPX market (orange). "
              f"Simulating all 1,000 data sets took {_seconds:.0f} s.\n"
              f"- The prior produces calm and stressed markets, flat and steep skews. {_cover}\n"
              "- The check only shows that the prior does not rule the data out; whether the model fits them is "
              "Section 5.4.\n"
              f"- {100 * np.isnan(_iv).mean():.1f}% of simulated quotes have no implied volatility, because noise pushed "
              "a very cheap option below its intrinsic value; they remain valid data."),
    ])
    return (prior_batch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Preparing the network input: whitening

    The network receives the 63 numbers $y_i/h_i$, the quotes in noise units. In this raw form they are hard to learn from:

    - **strongly correlated:** a change in any parameter moves all 63 together;
    - **badly scaled:** a few directions (overall level, skew) vary enormously across simulations, while the directions that reveal the noise level $c$ vary about a million times less.

    **Whitening** is a fixed linear change of coordinates that makes the inputs uncorrelated with unit variance, without losing information. It is fitted once on simulations (the first 10,000 training rows for the pretrained estimator), frozen, and applied unchanged to every later input, the SPX quotes included.
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
        - **Variance per direction:** the first principal direction of the inputs varies about ${_mantissa}\\times10^{{{int(_exponent)}}}$ times more than the last.
        - **Correlations** between the 63 inputs on new simulations: almost every pair is strongly correlated before whitening; afterwards the matrix is nearly diagonal.
        - **Where the information about $c$ is:** in the low-variance directions (average {np.mean(_link[-30:]):.2f} over the last 30, against {np.mean(_link[:10]):.2f} over the first 10). Whitening gives these directions the same weight as the dominant ones.
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

    The network learns the posterior $q_\phi(\boldsymbol\theta\mid\mathbf y)$ from the simulated pairs $(\boldsymbol\theta,\mathbf y)$. Each pair is a draw from the joint distribution of parameters and quotes, so among the simulations that produced quotes like $\mathbf y$, the parameters are distributed according to the posterior. The network learns that conditional distribution for every $\mathbf y$ at once. No likelihood is ever evaluated.
    """)])
    return


@app.cell(hide_code=True)
def _(mo):
    def _box(title, body, fill="rgba(42,120,214,0.06)", border="#2a78d6"):
        return (f'<div style="border:1px solid {border};border-radius:10px;padding:8px 10px;background:{fill};'
                f'text-align:center;flex:1;min-width:130px"><div style="font-weight:600">{title}</div>'
                f'<div style="font-size:0.8em;line-height:1.35;margin-top:3px">{body}</div></div>')

    _arrow = '<div style="align-self:center;color:#9a9890;font-size:1.4em;padding:0 2px">&#8594;</div>'
    _chain = [
        _box("Quotes y", "63 mid prices: simulated during training, observed at inference"),
        _box("Preprocessing", "y/h and the frozen whitening give x̃, 63 numbers; the BayesFlow adapter stacks θ, 6 numbers, "
                              "which the workflow standardizes"),
        _box("Summary network", "none: the 63 numbers enter directly", fill="transparent", border="#bbb9b0"),
        _box("Inference network", "FlowMatching: an MLP of 3 × 128 units, dropout 0.1",
             fill="rgba(235,104,52,0.08)", border="#eb6834"),
        _box("Posterior draws", "θ = (v₀, κ, v∞, σᵥ, ρ, c), as many as needed"),
    ]
    mo.vstack([
        mo.md("**The SBI network: what goes in, what comes out**"),
        mo.Html('<div style="display:flex;gap:4px;align-items:stretch;flex-wrap:wrap">' + _arrow.join(_chain) + "</div>"),
        mo.md(r"""
        - **Training** shows the network simulated pairs $(\boldsymbol\theta,\tilde{\mathbf x})$: 300,000 for the pretrained estimator.
        - **Inference** gives it one observed $\tilde{\mathbf x}$ and returns as many posterior draws as wanted.

        | Design choice | Here | Alternatives in BayesFlow |
        |:---|:---|:---|
        | What is learned | the posterior $q(\boldsymbol\theta\mid\mathbf y)$ | a neural pricer, a likelihood (NLE) or a likelihood ratio (NRE), each used inside a new MCMC run per data set |
        | Summary network | none: a fixed vector of 63 whitened quotes | `DeepSet` or `SetTransformer` for quote sets of varying size and layout; `TimeSeriesNetwork` for price histories |
        | Inference network | `FlowMatching` | `CouplingFlow` (exact density, one-pass sampling), `DiffusionModel`, `ConsistencyModel` (few sampling steps) |
        | Parameter space | standardized, unbounded: a few draws can leave the prior | bounded transforms in the adapter (`.constrain`), so that every draw respects the prior bounds |
        | Training data | a fixed bank (`fit_offline`), one noise draw per row | fresh simulations in every batch (`fit_online`) |
        | Amortized over | quotes for these 63 contracts and these spreads | contract terms and spreads as extra inputs, to reuse one network across days |
        """),
    ])
    return


@app.cell
def _(NAMES, bf):
    def build_workflow():
        """The BayesFlow workflow of the pretrained estimator (scripts/reproduce_heston_pr87_grid.py)."""
        adapter = (
            bf.Adapter()
            .concatenate(NAMES, into="inference_variables")   # θ: the six parameters
            .rename("observations", "inference_conditions")    # x̃: the whitened quotes
        )
        network = bf.networks.FlowMatching(subnet_kwargs={"widths": (128,) * 3, "dropout": 0.1})
        return bf.BasicWorkflow(adapter=adapter, inference_network=network, standardize="inference_variables")

    return (build_workflow,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Training the pretrained estimator**, as recorded in its training protocol:

    - **Data:** 300,000 simulated pairs; each keeps its noise draw throughout training.
    - **Schedule:** 300 epochs with batch size 256, 351,600 updates; seed 9011.
    - **Optimizer:** BayesFlow's default for offline training: AdamW with weight decay 0.005 and gradient clipping at 1.5; the learning rate warms up from $5\cdot10^{-5}$ to $5\cdot10^{-4}$, then decays to zero on a cosine schedule.
    - **Hardware:** the CPU of an Apple-silicon Mac, within the run's one-hour limit.

    ```python
    task, data = delivery.training_inputs(HESTON_BUNDLE, rows=300_000)   # (θ, x̃) pairs, x̃ already whitened
    keras.utils.set_random_seed(9011)
    workflow = build_workflow()
    history = workflow.fit_offline(data=data["train"], validation_data=data["validation"], epochs=300, batch_size=256)
    ```

    The parameters are standardized but not mapped onto their bounded prior ranges, so a small fraction of posterior draws falls just outside the prior; they are kept, not clipped. The button below runs the same loop at a small scale: 3,000 simulations, 40 epochs, then BayesFlow's own diagnostics.
    """)
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
    mo.accordion({"How flow matching works": mo.vstack([
        mo.md(r"""
        The inference network is a **velocity field** $v_\phi(\mathbf z,t,\tilde{\mathbf x})$ on the six-dimensional parameter space, for times $t\in[0,1]$.

        - **Training.** For a simulated pair $(\boldsymbol\theta,\tilde{\mathbf x})$, with $\boldsymbol\theta$ standardized, draw noise $\mathbf z_1\sim\mathcal N(0,I)$ independently of $\boldsymbol\theta$ and a time $t$, more often near $t=0$. Form the point $\mathbf z_t=(1-t)\,\boldsymbol\theta+t\,\mathbf z_1$ on the straight line from $\boldsymbol\theta$ to $\mathbf z_1$, and fit $v_\phi(\mathbf z_t,t,\tilde{\mathbf x})$ to the velocity $\mathbf z_1-\boldsymbol\theta$ with a squared-error loss.
        - **Why it works.** Many pairs pass through the same point, so the network learns the *average* velocity there. Following this averaged field moves probability mass so that its distribution at each time $t$ is that of $\mathbf z_t$: noise at $t=1$, the posterior at $t=0$.
        - **Sampling.** Draw $\mathbf z_1\sim\mathcal N(0,I)$ and integrate $d\mathbf z/dt=v_\phi(\mathbf z,t,\tilde{\mathbf x})$ from $t=1$ back to $t=0$ with an adaptive Runge–Kutta solver (Tsit5). The end point is one posterior draw; undoing the standardization gives $\boldsymbol\theta$.
        - **Network inputs.** $\mathbf z_t$ and $\tilde{\mathbf x}$ enter an MLP of three layers of 128 units; $t$ enters through a Fourier embedding that modulates every layer.
        """),
        _fig,
        mo.md("A toy posterior in two dimensions: the cloud at $t=1$ (noise), halfway, and at $t=0$ (posterior). The last "
              "panel shows straight training paths between simulated parameters (orange) and independent noise draws "
              "(blue)."),
    ])})
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
    build_workflow,
    fit_whitening,
    json,
    keras,
    live_train_button,
    make_simulator,
    mo,
    np,
    plots,
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
    live_workflow = build_workflow()   # the BayesFlow workflow defined above
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
    return


@app.cell(hide_code=True)
def _(mo, workflow):
    mo.vstack([workflow("Infer"), mo.md(r"""
    ## 4. Infer

    We apply the pretrained estimator to three sets of 63 quotes:

    - **Simulated, market-like:** generated with $c=1$ at parameters within 15% of the SPX market posterior medians.
    - **Simulated, stressed market:** generated with $c=1$ at a high-volatility draw from the prior; volatility starts at 37% and reverts to 29%.
    - **SPX market quotes** of 9 November 2026, for which the true parameters are unknown.

    Inference is one BayesFlow call, `workflow.sample`. The loader wraps it so that it takes raw quotes:

    ```python
    model, checkpoint = delivery.load_checkpoint(HESTON_BUNDLE, "300k-300-final300")
    draws = model.sample(quotes[None, :], 4096, seed=261113)[0]      # (4096, 6) posterior draws
    # inside: x_tilde = whitening.transform(quotes / half_spread)
    #         workflow.sample(conditions={"observations": x_tilde}, num_samples=4096)
    ```

    Nothing is fitted at this stage: each draw integrates the learned flow from one noise vector. As a benchmark, a **PyMC reference** samples the same posterior by MCMC with the exact likelihood (box below).
    """)])
    return


@app.cell(hide_code=True)
def _(CASES, CASE_TITLES, REFERENCES, json, mo):
    _rows = []
    for _case_id in CASES:
        _run = json.loads((REFERENCES / "references" / _case_id / "run.json").read_text())
        _stats = json.loads((REFERENCES / "references" / _case_id / "diagnostics.json").read_text())["parameters"]
        _attempt = _run["attempts"][_run["selected_attempt"]]
        _rows.append({
            "data set": CASE_TITLES[_case_id],
            "chains × draws": f"{_run['settings']['chains']} × {_attempt['draws']:,} after {_attempt['tune']:,} tuning steps",
            "largest R-hat": f"{max(_s['r_hat'] for _s in _stats.values()):.4f}",
            "smallest bulk ESS": f"{min(_s['ess_bulk'] for _s in _stats.values()):,.0f}",
            "likelihood evaluations": f"{_attempt['likelihood_evaluations']:,}",
            "run time, one core": f"{_attempt['wall_seconds'] / 60:.0f} min",
        })
    mo.accordion({"How the PyMC reference is computed": mo.vstack([
        mo.md(r"""
        - **Same posterior as the network:** the same uniform prior and the exact likelihood $y_i\sim\mathcal N\big(P_i(\boldsymbol\theta),(c\,h_i)^2\big)$, with prices from the Fourier pricer of Section 1.
        - **Sampler:** PyMC's `DEMetropolisZ`, a gradient-free Metropolis sampler whose proposals are differences of past draws. The likelihood calls the pricer as a black box without gradients, which rules out NUTS.
        - **Runs:** four chains, started from maximum-likelihood (Powell) fits and dispersed points, never from the true parameters or the network. A short run of 2,000 tuning steps and 2,000 draws comes first; when it fails the checks, a long run of 6,000 and 12,000 follows.
        - **Checks:** R-hat at most 1.01 and bulk and tail effective sample sizes of at least 400 for every parameter; prices accurate to 2% of the smallest noise standard deviation.
        """),
        mo.ui.table(_rows, selection=None, pagination=False),
        mo.md("The plots use a thinned subset of 500 draws per chain."),
    ])})
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
def _(CASES, REFERENCES, hashlib, json, np):
    reference_summary = json.loads((REFERENCES / "reference-comparison.json").read_text())
    _display = REFERENCES / "reference-display-draws.npz"
    if hashlib.sha256(_display.read_bytes()).hexdigest() != reference_summary["display_cache"]["sha256"]:
        raise ValueError("the reference draws differ from their recorded identity")
    with np.load(_display, allow_pickle=False) as _archive:
        reference_draws = {_case_id: _archive[f"{_case_id}__parameters"].reshape(-1, 6) for _case_id in CASES}
        reference_products = {_case_id: _archive[f"{_case_id}__products"].reshape(-1, 2) for _case_id in CASES}
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
        _note = ("- **No known truth** for the SPX quotes.\n"
                 "- The noise multiplier's posterior is pressed against the top of its prior range, $c=2$: the quotes ask "
                 "for larger errors than the model allows, because no Heston smile comes close enough. Section 5.4 "
                 "explains this, and why the network and the reference disagree here.")
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
            _note = (f"- **True values** (stars and lines): {_listed} {'lies' if len(_outside) == 1 else 'lie'} up to "
                     f"{np.max(np.abs(_z)):.1f} posterior SDs from the posterior mean, outside the 90% intervals. One noisy "
                     "data set can do this: the noise shifts the posterior, and strongly correlated parameters shift "
                     "together.\n")
        else:
            _note = "- **True values** (stars and lines) lie inside the 90% intervals.\n"
        _note += (f"- **Jointly**, the truth is farther from the posterior centre than {_joint:.0f}% of the draws. For a "
                  f"correct posterior this share is uniform between 0 and 100%, so a single value of {_joint:.0f}% is no "
                  "warning sign; coverage is judged over many data sets in Section 5.2.")
        if _case == "synthetic-broad":
            _first, _last = (EXPIRY == 0), (EXPIRY == len(EXPIRY_DATES) - 1)
            _atm = [100 * _iv[_at][np.argmin(np.abs(DELTA[_at]))] for _at in (_first, _last)]
            _deep = [100 * _iv[EXPIRY == _e][np.argmin(DELTA[EXPIRY == _e])] for _e in range(len(EXPIRY_DATES))]
            _note = (f"- **A stressed market:** at the money, {_atm[0]:.0f}% at {EXPIRY_DAYS[0]} days against {_atm[1]:.0f}% "
                     f"at {EXPIRY_DAYS[-1]} days, an inverted term structure; deep puts at {min(_deep):.0f}–{max(_deep):.0f}%. "
                     "Index options look like this after a sell-off.\n"
                     "- **A Heston signature:** in δ units its short-dated smiles are flatter than the long-dated ones, so the "
                     "curves cross. Real markets keep a steep short-dated skew (first figure of the notebook); Section 5.4 "
                     "shows the consequence on the SPX quotes. As a test of the inference the case is fully valid: it is a "
                     "draw from the model and the prior.\n" + _note)
    mo.vstack([
        _fig,
        mo.md(_note),
        mo.ui.table(_rows, selection=None, pagination=False),
        _corner,
        mo.md(f"""
        - **Reading the corner plot:** the diagonal shows each parameter's posterior; the panels below show pairs.
        - **Narrow posteriors:** {TEXT_LABELS[NAMES.index(_sharpest)]} is pinned to about ±{100 * _relative[_sharpest]:.1f}% of its value, and even the least precise Heston parameter to ±{100 * max(_relative.values()):.0f}%.
        - **Strong dependence:** the strongest pair is {TEXT_LABELS[_j]} with {TEXT_LABELS[_i]} (correlation {_corr[_i, _j]:+.2f}). The quotes pin down some *combinations* of parameters better than each parameter alone, so a quantity that depends on several parameters must be computed from the joint draws, not from the marginals.
        """),
        mo.md("**Derived quantities.** Each posterior draw gives an expected average variance over a horizon, the fair "
              "value of a variance swap, shown here as a volatility:"),
        mo.ui.table(_product_rows, selection=None, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo, workflow):
    mo.vstack([workflow("Check"), mo.md(r"""
    ## 5. Check

    | Check | Question | What to look for |
    |:---|:---|:---|
    | 5.1 Posterior predictive | Can the inferred parameters regenerate the quotes? | quotes inside the replicated range |
    | 5.2 Calibration over simulations | Does a 90% interval contain the truth 90% of the time? | coverage near nominal, uniform ranks |
    | 5.3 Exact reference | Does the network reproduce the posterior of one data set? | agreement with PyMC |
    | 5.4 Real SPX quotes | Can Heston describe the market at all? | residuals within the assumed noise |

    ### 5.1 Posterior predictive

    Draw parameters from the posterior, price the 63 contracts and add fresh noise with each draw's own $c$:
    $\boldsymbol\theta^{(m)}\sim q(\boldsymbol\theta\mid\mathbf y)$ and $\widetilde y^{(m)}_i=P_i(\boldsymbol\theta^{(m)})+c^{(m)}h_i\,\epsilon^{(m)}_i$. If the model and the posterior are right, the observed quotes look like one more replicated data set.
    """)])
    return


@app.cell(hide_code=True)
def _(EXPIRY_DATES, EXPIRY_DAYS, mo):
    expiry_choice = mo.ui.dropdown(options={f"{_date} ({_days} days)": _e for _e, (_date, _days) in enumerate(zip(EXPIRY_DATES, EXPIRY_DAYS))},
                                   value=f"{EXPIRY_DATES[4]} ({EXPIRY_DAYS[4]} days)", label="Expiry")
    mo.hstack([expiry_choice, mo.md("*Data set: as selected in Section 4.*")], justify="start", gap=1.5)
    return (expiry_choice,)


@app.cell(hide_code=True)
def _(
    BOUNDS,
    CASES,
    CASE_TITLES,
    CLEAN,
    DELTA,
    EXPIRY,
    H,
    case_choice,
    expiry_choice,
    implied_vols,
    mo,
    neural_draws,
    np,
    pilot,
    plt,
    reference_draws,
):
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
                "\n- **Dashed line:** the true, noise-free prices. The purple band follows it closely: the posterior "
                "recovers the true prices more precisely than any single quote measures them.")
    mo.vstack([_fig, mo.md(
        f"{CASE_TITLES[_case]}, one expiry; bands are pointwise 90% ranges over 40 posterior draws.\n\n"
        "- **Top:** implied volatilities; the bands are so narrow that they hide behind the points.\n"
        "- **Bottom:** the same minus the observed quotes, in half-spreads. Each quote carries its own noise, so the "
        "model prices (purple) differ from it by about $c$ half-spreads; the replicated quotes (green) add fresh noise "
        "and should cover the zero line." + _denoise + "\n"
        f"- **Verdict:** {_verdict}" + (f"\n- {_note}" if _note else ""))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.2 Calibration across 512 simulated data sets

    One posterior can look plausible and still be miscalibrated. Data sets with a known truth test the uncertainty statements themselves:

    - **Coverage:** a 90% interval should contain the truth in about 90% of the data sets. Much more means too wide, much less too narrow.
    - **Simulation-based calibration (SBC):** if the posterior is correct, the truth is just one more posterior draw, so its **rank** among the draws (the fraction of draws below it) is uniform between 0 and 1. We plot the empirical CDF of the ranks minus the uniform CDF; the four synthetic examples below show how to read it.
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
        mo.md("- **Calibrated:** the curve stays inside the grey band.\n"
              "- **Biased upwards:** posterior draws lie above the truth, ranks pile up near 0, and the curve bulges "
              "**above** zero; a downward bias gives the mirror image.\n"
              "- **Too narrow:** the truth often falls in the posterior's tails, so ranks pile up near 0 and 1: above zero "
              "on the left, below on the right.\n"
              "- **Too wide:** the truth sits near the middle of the posterior, so ranks pile up around 0.5: **below** zero "
              "on the left, above on the right."),
    ])
    return


@app.cell(hide_code=True)
def _(
    BOUNDS,
    EVALUATION,
    LABELS,
    NAMES,
    TEXT_LABELS,
    json,
    mo,
    np,
    plots,
    plt,
):
    _population = json.loads((EVALUATION / "evaluation.json").read_text())["population"]
    _ranks = np.asarray(_population["fractional_ranks"])
    _band = _population["rank_dkw"]["dkw_bound"]
    _passed = _population["rank_dkw"]["passed"]
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
    _cis = _coverage["0.9"]["wilson"]
    _good = [_j for _j in range(len(NAMES)) if _passed[_j] and _cis[_j][0] <= 0.9 <= _cis[_j][1]]
    _bad = [_j for _j in range(len(NAMES)) if _j not in _good]
    _lesson = (f"- **{len(_good)} of the {len(NAMES)} parameters are well calibrated:** the 95% confidence interval of "
               "their 90% coverage includes 90%, and their rank curves stay inside the band.\n")
    for _j in _bad:
        _diff = lambda _x: np.mean(_ranks[:, _j] <= _x) - _x
        _low, _high = _diff(0.25), _diff(0.75)
        _shape = ("the *too wide* shape: below zero on the left, above zero on the right" if _low < 0 < _high else
                  "the *too narrow* shape: above zero on the left, below zero on the right" if _high < 0 < _low else
                  "the shape of a *bias*: the curve stays on one side of zero")
        _cov = _coverage["0.9"]["coverage"][_j]
        _lesson += (f"- **The exception is {TEXT_LABELS[_j]}:** its 90% intervals contain the truth {100 * _cov:.1f}% of "
                    f"the time, and its rank curve has {_shape}.")
        if _cov > 0.9 and _low < 0 < _high:
            _lesson += " The network is **conservative** about it rather than overconfident."
        if NAMES[_j] == "initial_variance":
            _lesson += (" A plausible explanation is precision: the shortest-dated quotes pin the initial variance down to "
                        f"an average posterior SD of {_population['mean_posterior_sd'][_j]:.5f} on a prior range of "
                        f"{BOUNDS[_j, 1] - BOUNDS[_j, 0]:.2f}, which may be finer than the network resolves.")
        _lesson += "\n"
    _counts = np.asarray(_support["outside_prior"]["per_dimension_count"])
    _top = [LABELS[_k] for _k in np.argsort(_counts)[::-1][:2]]
    _lesson += (f"- **{100 * _support['outside_prior']['sample_fraction']:.1f}% of raw draws** fall slightly outside the "
                f"prior range, most often in {_top[0]} and {_top[1]}; they are kept in every check.")
    mo.vstack([
        mo.md("**Results for the pretrained estimator** on 512 simulated data sets with 512 posterior draws each, from "
              "its recorded evaluation:"),
        mo.ui.table(_rows, selection=None, pagination=False),
        _fig,
        mo.md("**Rank plots** for the six parameters, read as above. The grey band is the simultaneous 95% band for six "
              "parameters."),
        mo.callout(mo.md("**What we learn**\n\n" + _lesson), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(EVALUATION, np):
    with np.load(EVALUATION / "recovery.npz", allow_pickle=False) as _archive:
        recovery_truth = _archive["truth"].astype(float)       # (512, 6) true parameters
        recovery_q = _archive["quantiles"].astype(float)       # (3, 512, 6): 5%, 50% and 95% posterior quantiles
    return recovery_q, recovery_truth


@app.cell(hide_code=True)
def _(TEXT_LABELS, mo):
    recovery_parameter = mo.ui.dropdown(options={_label: _j for _j, _label in enumerate(TEXT_LABELS)},
                                        value=TEXT_LABELS[-1], label="Parameter")
    mo.vstack([
        mo.md("### Recovery: pick data sets and look at their posteriors\n\n"
              "Each point is one of the 512 simulated data sets: its true value against the posterior median, with the "
              "90% interval. **Drag a box around points** to see the posterior of one of them."),
        recovery_parameter,
    ])
    return (recovery_parameter,)


@app.cell(hide_code=True)
def _(
    BOUNDS,
    LABELS,
    mo,
    plots,
    plt,
    recovery_parameter,
    recovery_q,
    recovery_truth,
):
    _j = recovery_parameter.value
    _fig, _ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    _lo, _hi = BOUNDS[_j]
    _ax.plot([_lo, _hi], [_lo, _hi], "--", color=plots.MUTED, lw=1)
    _ax.vlines(recovery_truth[:, _j], recovery_q[0, :, _j], recovery_q[2, :, _j], color=plots.SERIES[0], alpha=.18, lw=.8)
    _ax.scatter(recovery_truth[:, _j], recovery_q[1, :, _j], s=9, color=plots.SERIES[0], alpha=.75)
    _ax.set(xlabel=f"true value of {LABELS[_j]}", ylabel=f"posterior median of {LABELS[_j]}",
            title="512 simulated data sets: truth against posterior")
    recovery_picker = mo.ui.matplotlib(_ax, debounce=True)
    recovery_picker
    return (recovery_picker,)


@app.cell(hide_code=True)
def _(BUNDLE, NAMES, delivery, estimator, np, recovery_truth):
    held_out_quotes = None
    if estimator is not None:
        _, _arrays, _ = delivery.load_bank(BUNDLE)
        _truth = np.column_stack([_arrays[f"historical_{_name}"][:, 0] for _name in NAMES])
        if not np.allclose(_truth, recovery_truth, rtol=0, atol=1e-12):
            raise ValueError("the bank's held-out data sets differ from the saved evaluation")
        held_out_quotes = _arrays["historical_quotes"]   # raw quotes in index points, one row per data set
    return (held_out_quotes,)


@app.cell(hide_code=True)
def _(
    EVAL_SEED,
    LABELS,
    NAMES,
    TEXT_LABELS,
    bf,
    estimator,
    held_out_quotes,
    mo,
    np,
    recovery_parameter,
    recovery_picker,
    recovery_q,
    recovery_truth,
):
    _j = recovery_parameter.value
    _x, _y = recovery_truth[:, _j], recovery_q[1, :, _j]
    _surprise = np.abs(_y - _x) / (recovery_q[2, :, _j] - recovery_q[0, :, _j])   # error in units of the 90% width
    _selected = np.flatnonzero(recovery_picker.value.get_mask(_x, _y)) if recovery_picker.value else np.array([], int)
    _pool = _selected if len(_selected) else np.arange(len(_x))
    _case = int(_pool[np.argmax(_surprise[_pool])])
    _how = (f"{len(_selected)} data sets selected; showing the one whose median is farthest from the truth, relative to "
            "its interval." if len(_selected) else
            "No selection yet: showing the data set whose median is farthest from the truth, relative to its interval.")
    _rows = [{"parameter": TEXT_LABELS[_k], "truth": f"{recovery_truth[_case, _k]:.5g}",
              "posterior median [90% interval]": f"{recovery_q[1, _case, _k]:.5g} [{recovery_q[0, _case, _k]:.5g}, "
                                                 f"{recovery_q[2, _case, _k]:.5g}]",
              "truth inside": "yes" if recovery_q[0, _case, _k] <= recovery_truth[_case, _k] <= recovery_q[2, _case, _k] else "no"}
             for _k in range(len(NAMES))]
    _parts = [mo.md(f"{_how} **Data set {_case}**, from the recorded evaluation:"),
              mo.ui.table(_rows, selection=None, pagination=False)]
    if estimator is not None and held_out_quotes is not None:
        _draws = np.asarray(estimator.sample(held_out_quotes[_case][None], 2000, seed=EVAL_SEED + 5000 + _case))[0]
        _grid = bf.diagnostics.pairs_posterior(_draws, targets=recovery_truth[_case], variable_names=LABELS, height=1.45,
                                               label_fontsize=10, tick_fontsize=7, legend_fontsize=9)
        _parts += [mo.md("Its posterior, 2,000 fresh draws, drawn with `bf.diagnostics.pairs_posterior`; the true values "
                         "are marked in red. Shapes differ from one data set to the next: compare cases near the edges of "
                         "the prior with cases in its middle."),
                   getattr(_grid, "figure", None) or _grid.fig]
    else:
        _parts.append(mo.md("*The posterior pair plot of the selected data set needs the pretrained estimator "
                            "(`HESTON_BUNDLE`).*"))
    mo.vstack(_parts)
    return


@app.cell(hide_code=True)
def _(mo):
    panel_button = mo.ui.run_button(label="Resample the 512 data sets with the pretrained estimator (a few minutes)")
    panel_button
    return (panel_button,)


@app.cell(hide_code=True)
def _(
    BUNDLE,
    BUNDLE_HELP,
    EVAL_SEED,
    LABELS,
    NAMES,
    PANEL_BATCH,
    PANEL_DRAWS,
    bf,
    delivery,
    estimator,
    mo,
    np,
    panel_button,
    plots,
    time,
):
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
        bf.diagnostics.coverage(estimates=_estimates, targets=_targets, variable_keys=NAMES, variable_names=LABELS,
                                difference=True, color=plots.SERIES[0]),
    ])
    return


@app.cell(hide_code=True)
def _(
    BOUNDS,
    CASES,
    NAMES,
    TEXT_LABELS,
    mo,
    neural_draws,
    np,
    reference_summary,
):
    mo.stop(not neural_draws, mo.vstack([
        mo.md("### 5.3 One data set at a time: comparison with the exact reference"),
        mo.md("*This comparison needs the pretrained estimator.*")]))
    _columns = {"synthetic-market-like": "market-like simulation", "synthetic-broad": "stressed simulation", "market": "SPX quotes"}
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
        mo.md(f"- **Simulated data sets:** {_agree} of 12 parameter posteriors agree on both counts. The centres are "
              f"mostly close (median difference {np.median(_all_gaps):.2f} SD, largest {_all_gaps.max():.2f} SD); the "
              f"intervals are wider: {_wider} of 12 more than 25% wider, up to {_all_ratios.max():.1f} times, and "
              f"{_narrower} narrower.\n"
              f"- {_conservative}This fits the good calibration of 5.2: calibration is necessary, not sufficient (a "
              "posterior equal to the prior is perfectly calibrated and useless). Only an exact reference shows how much "
              "sharpness is lost.\n"
              f"- **SPX quotes:** the network departs far from the reference, and {100 * _outside['market']:.0f}% of its "
              "draws leave the prior range (Section 5.4)."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.4 Real SPX quotes: does Heston fit the market?

    All checks so far used data simulated from the model itself. On the **SPX quotes of 9 November 2026** the first question is whether **any** Heston parameters reproduce the market within the assumed noise.
    """)
    return


@app.cell(hide_code=True)
def _(
    BOUNDS,
    CASES,
    DELTA,
    EXPIRY,
    EXPIRY_DATES,
    EXPIRY_DAYS,
    H,
    MARKET_IV,
    implied_vols,
    mo,
    neural_draws,
    np,
    pilot,
    plots,
    plt,
    reference_draws,
):
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
        _ax.scatter(DELTA[_at], 100 * MARKET_IV[_at], color="#222222", s=14, zorder=4, label="SPX market")
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
        mo.md("*Bands:* implied volatilities of the model prices from 40 posterior draws; *points:* SPX mid quotes; *last "
              "panel:* the posterior of the noise multiplier $c$."),
        mo.ui.table(_rows, selection=None, pagination=False),
        mo.md(f"""
        **What the PyMC posterior says**, the exact answer for our model and prior:

        - **No Heston smile fits the SPX quotes within the assumed noise.** The posterior-mean prices miss them by **{_rms:.1f} half-spreads** (root mean square), while the prior caps the noise SD at two half-spreads ($c\\le2$).
        - The likelihood would favour an even larger $c$, so the posterior piles up at the bound $c=2$, and the Heston parameters settle on the best compromise.
        - At {_wing} of {len(EXPIRY_DATES)} expiries the largest miss is at the deepest put ($\\delta\\approx-5$), where the SPX smile lies up to several points above every posterior Heston smile, most at short expiries: Heston cannot produce a short-dated skew this steep together with the rest of the surface.

        **Why the network disagrees with PyMC:**

        - It was trained on quotes that scatter around a Heston smile with a noise SD of at most two half-spreads. These quotes are {_rms:.0f} half-spreads away on average, outside anything it saw, so it extrapolates.{_network_text}
        - This is **model misspecification**, not a training failure, and a useful alarm: an amortized estimator should only be trusted on data its simulator could have produced.

        **What to do about it** is a modelling decision: widen or reshape the noise model, add an explicit model-error term, or use a richer model, for example with jumps or rough volatility.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo, workflow):
    mo.vstack([workflow("Use"), mo.md(r"""
    ## 6. Use: one price or a range?

    Calibration is usually a means to an end: pricing contracts the market does not quote. With a posterior, every draw of $\boldsymbol\theta$ gives a price, so the price comes with a range. A point estimate gives one number and hides how uncertain it is. How much does this matter here?

    We price three contracts that the SPX quotes do not include, in forward units, and compare a **point estimate** (the posterior-mean parameters) with the **posterior range** (300 posterior draws per data set).
    """)])
    return


@app.cell(hide_code=True)
def _(
    BOUNDS,
    CASES,
    CASE_TITLES,
    DELTA,
    EXPIRY,
    EXPIRY_DATES,
    HestonGridPilot,
    MARKET_IV,
    REF_VOL,
    implied_vols,
    mo,
    neural_draws,
    np,
    pilot,
    reference_draws,
    surface,
):
    _products = {"2-year at the money": (2.0, 0.0, "call"),
                 "2-year put at δ = −2": (2.0, -2.0, "put"),
                 "5-year at the money": (5.0, 0.0, "call")}
    _T = np.array([_v[0] for _v in _products.values()])
    _k = np.array([_v[1] * REF_VOL * np.sqrt(_v[0]) for _v in _products.values()])
    _kind = np.array([_v[2] for _v in _products.values()])
    _pricer = HestonGridPilot({"forward": np.ones(3), "strike": np.exp(_k), "discount": np.ones(3), "maturity": _T,
                               "option_type": _kind, "mid": np.full(3, 0.1), "half_spread": np.full(3, 1e-4)},
                              infer_c=True, c_bounds=(0.5, 2.0))
    _vol = lambda _theta: 100 * surface.implied_vol(_pricer.prices(_theta)[None], _k, _T, _kind)[0]
    _rng = np.random.default_rng(20261114)
    _rows, _widths, _gaps, _network_widths = [], [], [], []
    for _case_id in CASES:
        _sources = {"PyMC reference": reference_draws[_case_id]}
        if _case_id in neural_draws:
            _all = neural_draws[_case_id]
            _inside = _all[np.all((_all >= BOUNDS[:, 0]) & (_all <= BOUNDS[:, 1]), axis=1)]
            if len(_inside) >= 20:
                _sources["network"] = _inside
        _point = _vol(reference_draws[_case_id].mean(axis=0))
        _ranges = {}
        for _source, _draws in _sources.items():
            _sample = _draws[_rng.choice(len(_draws), min(300, len(_draws)), replace=False)]
            _ranges[_source] = np.nanquantile(np.array([_vol(_theta) for _theta in _sample]), [.05, .5, .95], axis=0)
        for _p, _name in enumerate(_products):
            _q = _ranges["PyMC reference"][:, _p]
            _row = {"data set": CASE_TITLES[_case_id], "contract": _name, "point estimate (%)": f"{_point[_p]:.2f}",
                    "PyMC posterior, 90% range (%)": f"{_q[0]:.2f} to {_q[2]:.2f}"}
            if "network" in _ranges:
                _n = _ranges["network"][:, _p]
                _row["network posterior, 90% range (%)"] = f"{_n[0]:.2f} to {_n[2]:.2f}"
                _network_widths.append(_n[2] - _n[0])
            _rows.append(_row)
            _widths.append(_q[2] - _q[0])
            _gaps.append(abs(_point[_p] - _q[1]))
    _widths = np.asarray(_widths).reshape(len(CASES), len(_products))
    # Model error on the SPX quotes: the point-estimate Heston smile against the market.
    _fit = implied_vols(pilot.prices(reference_draws["market"].mean(axis=0)))[0]
    _miss = 100 * np.abs(_fit - MARKET_IV)
    _worst = int(np.nanargmax(_miss))
    _network_text = ("" if not _network_widths else
                     f" The network's ranges are {min(_network_widths):.2f}–{max(_network_widths):.2f} points wide: "
                     "wider, as in Section 5.3, but still small.")
    _text = (
        f"- **Point estimate against posterior:** the point estimate lies within {max(_gaps):.3f} volatility points of "
        "the posterior median price.\n"
        f"- **Parameter uncertainty is small here:** the PyMC 90% ranges are {_widths.min():.2f}–{_widths.max():.2f} "
        "volatility points wide. With 63 liquid quotes, ignoring parameter uncertainty costs little for these "
        f"contracts.{_network_text}\n"
        f"- **The 5-year option is priced about as tightly as the 2-year one** (widest ranges {_widths[:, 2].max():.2f} "
        f"against {_widths[:, 0].max():.2f} points), although no quote reaches beyond eleven months. Heston ties every "
        "maturity to the same five parameters, so its extrapolation looks certain; that certainty comes from the model, "
        "not from the quotes.\n"
        f"- **Model error is far larger:** on the SPX quotes, the point-estimate Heston smile misses the market by up to "
        f"{_miss[_worst]:.1f} volatility points ({EXPIRY_DATES[EXPIRY[_worst]]}, δ = {DELTA[_worst]:.1f}), "
        f"{_miss[_worst] / _widths.max():.0f} times the widest posterior range. No parameter posterior captures this.\n"
        "- **When parameter uncertainty does matter downstream:** when the quotes do not pin down what a contract "
        "depends on: few or wide quotes, or a model flexible enough that the quotes leave some directions open. Then "
        "the range, not the point estimate, is the honest answer.")
    mo.vstack([mo.ui.table(_rows, selection=None, pagination=False), mo.md(_text)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    1. **SBI turns a pricer into a posterior.** Prior, pricer and noise model form a simulator; one training run gives an estimator that returns the posterior for any 63 quotes in seconds.
    2. **Liquid quotes are very informative.** Spreads of a few hundredths to a few tenths of a volatility point pin the Heston parameters down tightly and make model error stand out.
    3. **Prepare the inputs.** Quotes in noise units and whitening put all 63 inputs on one uncorrelated scale.
    4. **Check calibration over many simulations.** Here five of six parameters are well calibrated; the initial-variance intervals are too wide rather than too narrow.
    5. **Check single data sets against an exact reference.** Good calibration does not mean sharp posteriors: the network's intervals can be several times wider than the exact ones.
    6. **Check the model, not only the network.** On the SPX quotes the noise multiplier hits its bound and Heston misses by many half-spreads, mostly in the deep put wing. An amortized estimator is only reliable on data its simulator could have produced.
    7. **Downstream, a posterior gives a range instead of a number.** Here parameter uncertainty barely moves the prices of unquoted contracts, while model error moves them by volatility points.
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
