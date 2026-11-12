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
    # BayesFlow logs a line for every fit and every sampling call, some with local paths. The notebook reports
    # these timings itself, so only warnings and errors reach the page.
    logging.getLogger("bayesflow").setLevel(logging.WARNING)

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
        "and restart marimo. `lab/plans/training-data-guide.md` explains how to obtain and check the bundle. Until "
        "then, results that need the trained network are left out and the PyMC reference is shown on its own."),
        kind="warn")

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
    )


@app.cell(hide_code=True)
def _(mo):
    # The notebook's page elements. Each returns HTML with a class that heston_spx_calibration.css styles, so
    # the text below states content and the stylesheet decides how it looks.
    from inspect import cleandoc as _cleandoc

    STAGES = (
        ("Model", "Heston dynamics, a prior over six parameters and quote noise set by the bid–ask spread"),
        ("Simulate", "draw parameters, price the 63 options and add quote noise, 300,000 times"),
        ("Train", "a flow-matching network learns the posterior of the parameters given the quotes"),
        ("Infer", "condition on observed quotes and draw from the posterior in seconds"),
        ("Check", "posterior predictive, calibration across simulations, exact reference, real data"),
        ("Use", "price contracts the market does not quote, with a range instead of a number"),
    )

    def _html(content):
        """Markdown text, or a marimo object, as an HTML string."""
        return content.text if hasattr(content, "text") else mo.md(content).text

    def _blocks(items):
        """`items` maps a label to its content, or lists contents without labels."""
        pairs = items.items() if isinstance(items, dict) else [(None, content) for content in items]
        return "".join("<div>" + (f'<div class="nb-label">{_html(label)}</div>' if label else "") + _html(content)
                       + "</div>" for label, content in pairs)

    def workflow_map():
        """The six stages and what each one does, shown once under the title."""
        steps = "".join(
            f'<div class="nb-map-step"><div class="nb-map-name"><span>{number}</span>{name}</div>'
            f'<div class="nb-map-text">{text}</div></div>'
            for number, (name, text) in enumerate(STAGES, start=1))
        return mo.Html(f'<div class="nb-map">{steps}</div>')

    def section(title, lead, stage=None, kicker=None):
        """The banner that opens a section: its place in the workflow, its title and its one-sentence summary."""
        if stage is None:
            top, number = f'<div class="nb-kicker">{kicker}</div>', ""
        else:
            rail = []
            for index, (name, _) in enumerate(STAGES, start=1):
                state = "is-done" if index < stage else "is-current" if index == stage else ""
                if index > 1:
                    rail.append(f'<span class="nb-rail-arrow {"is-done" if index <= stage else ""}">&#8594;</span>')
                rail.append(f'<span class="nb-rail-step {state}"><span class="nb-rail-number">{index}</span>{name}</span>')
            top, number = f'<div class="nb-rail">{"".join(rail)}</div>', f'<span class="nb-section-number">{stage}</span>'
        return mo.Html(
            f'<div class="nb-section">{top}<div class="nb-section-head">{number}<div>'
            f'<h2 id="{title.lower().replace(" ", "-")}">{title}</h2>'
            f'<div class="nb-standfirst">{_html(lead)}</div></div></div></div>')

    def key(text):
        """The message to take away from what was just shown."""
        return mo.md("\n".join("> " + line for line in _cleandoc(text).splitlines()))

    def note(text, label="Note"):
        """A caveat or a technical remark: a detail, set smaller."""
        return mo.Html(f'<div class="nb-note"><span class="nb-note-label">{label}</span><div>{_html(text)}</div></div>')

    def caption(text):
        """What a figure or a table shows: a detail, set smaller."""
        return mo.Html(f'<div class="nb-caption">{_html(text)}</div>')

    def grid(items, columns=None, kind="plain", widths=None):
        """Blocks side by side. With kind="reading" they sit under a figure, one column per panel; `widths` then
        gives the columns the relative widths of the panels."""
        layout = (f'--nb-columns:{columns or len(items)}' if widths is None else
                  "grid-template-columns:" + " ".join(f"minmax(0,{width}fr)" for width in widths))
        return mo.Html(f'<div class="nb-grid is-{kind}" style="{layout}">{_blocks(items)}</div>')

    def stack(items):
        """Labelled blocks under one another, to fill one column of a grid."""
        return mo.Html(f'<div class="nb-stack">{_blocks(items)}</div>')

    def cards(items, columns=2, numbered=False):
        """Framed blocks side by side: `items` are (label, title, content) triples."""
        blocks = "".join(f'<div><div class="nb-label">{label}</div><div class="nb-card-title">{_html(title)}</div>'
                         f'{_html(content)}</div>' for label, title, content in items)
        return mo.Html(f'<div class="nb-grid is-cards{" is-numbered" if numbered else ""}" '
                       f'style="--nb-columns:{columns}">{blocks}</div>')

    def controls(*elements, hint=""):
        """A frame around interactive elements, with a line on what to do with them."""
        body = mo.vstack(elements).text if len(elements) > 1 else mo.as_html(elements[0]).text
        return mo.Html(f'<div class="nb-controls"><div class="nb-controls-head"><span class="nb-label">Try it</span>'
                       f'<span class="nb-hint">{_html(hint) if hint else ""}</span></div>{body}</div>')

    def bayesflow(text):
        """Text that introduces BayesFlow code."""
        return mo.md('<span class="nb-chip">BayesFlow</span> ' + _cleandoc(text))

    def table(rows, align=None):
        """A static table from a list of dicts. The keys are the column headings; `align` holds l, c or r per column."""
        columns = list(rows[0])
        rule = {"l": ":---", "c": ":---:", "r": "---:"}
        lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(rule[a] for a in align or "l" * len(columns)) + "|"]
        lines += ["| " + " | ".join(str(row[column]) for column in columns) + " |" for row in rows]
        return mo.md("\n".join(lines))

    return (
        bayesflow,
        caption,
        cards,
        controls,
        grid,
        key,
        note,
        section,
        stack,
        table,
        workflow_map,
    )


@app.cell(hide_code=True)
def _(bayesflow, grid, key, mo, note, workflow_map):
    mo.vstack([
        mo.md(r"""
        # Bayesian calibration of the Heston model on SPX options

        We calibrate the **Heston stochastic-volatility model** to **63 SPX option quotes**. Instead of one best-fitting parameter vector, we compute a **posterior distribution** over the six unknowns $\boldsymbol\theta=(v_0,\kappa,v_\infty,\sigma_v,\rho,c)$.

        The posterior is learned with **amortized simulation-based inference** in **BayesFlow**. A neural network is trained once on simulated quotes. It then returns the posterior for any new set of 63 quotes in seconds.

        The workflow that we will explore is described as follows:
        """),
        workflow_map(),
        bayesflow("In this notebook, we explicitly show where and how **BayesFlow** is used, including code snippets "
                  "where useful. Those places carry this mark."),
        mo.md("#### How to read this notebook"),
        grid([
            key("**Key messages** are set off like this. Read them first."),
            note("Notes and captions are set small. They document a figure or add a caveat."),
            mo.accordion({"Foldable sections hold method details": mo.md(
                "Nothing in a foldable section is needed to follow the rest of the notebook.")}),
        ]),
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
    # The parameters by name: in plain text for menus, with typeset symbols for the page.
    TEXT_LABELS = ["initial variance v₀", "mean reversion κ", "long-run variance v∞",
                   "volatility of variance σᵥ", "correlation ρ", "noise multiplier c"]
    MATH_LABELS = [r"initial variance $v_0$", r"mean reversion $\kappa$", r"long-run variance $v_\infty$",
                   r"volatility of variance $\sigma_v$", r"correlation $\rho$", r"noise multiplier $c$"]

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

    def forward_unit_pricer(log_moneyness, maturity, kind):
        """A Heston pricer for contracts the market does not quote, in forward units (F = D = 1)."""
        _n = np.size(log_moneyness)
        return HestonGridPilot({"forward": np.ones(_n), "strike": np.exp(log_moneyness), "discount": np.ones(_n),
                                "maturity": np.asarray(maturity, dtype=float), "option_type": np.asarray(kind),
                                "mid": np.full(_n, 0.01), "half_spread": np.full(_n, 1e-4)},
                               infer_c=True, c_bounds=(0.5, 2.0))

    def atm_shape(log_moneyness, vols):
        """At-the-money skew and curvature (Lecture 5): a quadratic in log-moneyness through the five nearest strikes."""
        _near = np.argsort(np.abs(log_moneyness))[:5]
        _c2, _c1, _ = np.polyfit(log_moneyness[_near], vols[_near], 2)
        return _c1, 2 * _c2

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
        MATH_LABELS,
        MATURITY,
        MID,
        NAMES,
        QUOTES,
        REF_VOL,
        TEXT_LABELS,
        atm_shape,
        design,
        forward_unit_pricer,
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


@app.cell(hide_code=True)
def _(
    DELTA,
    EXPIRY,
    EXPIRY_DATES,
    KIND,
    MARKET_IV,
    REF_VOL,
    design,
    grid,
    mo,
    note,
    np,
    section,
):
    _puts = sorted({_q["target_z"] for _q in design["selected_quotes"] if _q["option_type"] == "put"})
    _calls = [_t for _t in design["targets_z"] if _t not in _puts]
    _fmt = lambda _values: ", ".join(f"{_v:g}".replace("-", "−") for _v in _values)
    _atm = [100 * MARKET_IV[EXPIRY == _e][np.argmin(np.abs(DELTA[EXPIRY == _e]))] for _e in range(len(EXPIRY_DATES))]
    mo.vstack([
        section("The data", "63 SPX option quotes from one snapshot: seven expiries from one week to about eleven "
                "months, nine strikes each.", kicker="Before the workflow"),
        grid({
            "Snapshot": "SPX index options at 20:13 UTC on 9 November 2026. All 63 contracts are European, "
                        "PM-settled SPXW options.",
            "Per contract": r"The bid and the ask give the **mid price** $y_i$ and the **half-spread** "
                            r"$h_i=(\text{ask}_i-\text{bid}_i)/2$, in index points.",
            "Per expiry": "The forward price $F$ and the discount factor $D$, estimated from put–call parity on "
                          "the same snapshot.",
        }, kind="reading"),
        mo.md(rf"""
        ### A common scale for strikes

        Strikes of different expiries are compared through the **standardized moneyness** $\delta$, the distance from the forward in units of a typical move over the option's life ($T$ in years, actual/365).

        $$\delta=\frac{{\ln(K/F)}}{{\sigma_{{\text{{ref}}}}\sqrt{{T}}}},\qquad \sigma_{{\text{{ref}}}}={100 * REF_VOL:.0f}\%$$
        """),
        note(f"{100 * REF_VOL:.0f}% is close to the at-the-money implied volatility of SPX on this snapshot "
             f"({min(_atm):.0f}–{max(_atm):.0f}% across the seven expiries), so one unit of $\\delta$ is roughly one "
             "standard deviation of the log price at expiry."),
        mo.md(rf"""
        The contracts were selected on this scale. At every expiry we took the puts nearest to $\delta$ = {_fmt(_puts)} and the calls nearest to $\delta$ = {_fmt(_calls)}. They run from $\delta={DELTA.min():.1f}$ to ${DELTA.max():.1f}$ ({int((KIND == 'put').sum())} puts, {int((KIND == 'call').sum())} calls).
        """),
    ])
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
    caption,
    design,
    grid,
    grid_rules,
    key,
    mo,
    note,
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
        mo.md("### The 63 quotes"),
        _fig,
        caption("Left: the selected contracts, by days to expiry and standardized moneyness. Middle: the half-spread "
                "as a share of the mid price. Right: implied volatility at the mid prices. Each colour is one expiry."),
        key(r"""
        **The seven smiles are nearly parallel in $\delta$.** The skew per unit of $\delta$ is about the same at one week as at eleven months. This is the feature the model must reproduce.
        """),
        grid({
            "The contracts": "Every expiry has nine contracts at similar values of $\\delta$. Two thirds of them are puts.",
            "The spreads": f"The half-spread is {_relative.min():.2f}% of the price at best, near the money. It reaches "
                           f"{_relative.max():.0f}% for the cheapest puts ({MID.min():.2f} index points).",
            "The smile": f"Implied volatility runs from {min(_atm):.0f}–{max(_atm):.0f}% at the money to "
                         f"{min(_wing):.0f}–{max(_wing):.0f}% for the deepest puts.",
        }, kind="reading"),
        note("Implied volatilities are for display only and all computations use prices."),
        mo.accordion({
            "How the 63 contracts were chosen": mo.vstack([
                grid({
                    "Expiries": f"For each target of {_days} business days (divided by 252 to give years) we took the "
                                f"nearest eligible SPXW expiry. For the one-year target this is {EXPIRY_DATES[-1]}, "
                                f"{EXPIRY_DAYS[-1]} days away.",
                    "Strikes": "At each expiry we took the out-of-the-money contract nearest to each of the nine "
                               f"targets in $\\delta$. Every selected contract lies within {_distance:.2f} of its target.",
                    "Eligibility": "A contract was eligible only if its half-spread was at most "
                                   f"{100 * grid_rules.MAX_RELATIVE_HALFSPREAD:.0f}% of its mid price, and if it was "
                                   "not among the quotes used to estimate the forwards and discount factors or held "
                                   "out for scoring. A target counts as supported when the nearest eligible contract "
                                   f"lies within {grid_rules.MAX_Z_DISTANCE:g} of it.",
                    "One exception": f"At the {_override['expiration']} expiry no eligible call lies near "
                                     f"$\\delta={_override['original_target_z']:g}$, so that target was declared at "
                                     f"$\\delta={_override['revised_target_z']:g}$ instead.",
                }, columns=2),
                note("The reference volatility only sets the units of the axis. The pricer uses each contract's "
                     "actual strike, forward, discount factor and expiry, and the network sees prices."),
            ]),
            "All 63 quotes": mo.ui.table(_rows, selection=None, pagination=True, page_size=9),
        }),
    ])
    return


@app.cell(hide_code=True)
def _(grid, mo, note, section):
    mo.vstack([
        section("Model", "Heston dynamics for the price, a prior over six parameters, and quote noise set by the "
                "bid–ask spread.", stage=1),
        mo.md(r"""
        ### 1.1 Heston dynamics

        The variance $V_t$ follows its own random process, correlated with the price.

        $$
        \frac{dF_t}{F_t} = \sqrt{V_t}\,dW_t^F,\qquad
        dV_t = \kappa(v_\infty-V_t)\,dt+\sigma_v\sqrt{V_t}\,dW_t^V,
        \qquad d\langle W^F,W^V\rangle_t=\rho\,dt
        $$

        Here $W^F$ and $W^V$ are two Brownian motions with correlation $\rho$, and $F_t$ is the forward price for the option's expiry. The six unknowns are:

        | Symbol | Meaning |
        |:---|:---|
        | $v_0$ | initial variance $V_0$ |
        | $\kappa$ | speed of mean reversion, per year |
        | $v_\infty$ | long-run variance, the level $V_t$ reverts to |
        | $\sigma_v$ | volatility of variance |
        | $\rho$ | correlation of price and variance shocks |
        | $c$ | quote-noise multiplier (Section 1.3) |
        """),
        note("The priors are given with the prior predictive check in Section 2.2."),
        mo.accordion({"Why there is no drift: the pricing measure and forward prices": grid({
            "The pricing measure": r"Prices are computed under the pricing measure $\mathbb Q$, under which "
                                   "discounted traded prices are martingales.",
            "The forward price": r"$F_t=S_t\,e^{(r-q)(T-t)}$, with interest rate $r$ and dividend yield $q$, is then "
                                 r"a martingale, so it has no drift. At expiry $F_T=S_T$, so a call is worth "
                                 r"$C=D\,\mathbb E^{\mathbb Q}\big[(F_T-K)^+\big]$ with $D=e^{-rT}$.",
            "Forward units": r"Rates and dividends enter only through $D$ and $F$, read from the market by put–call "
                             r"parity. Divided by $D\,F$, prices depend only on $\ln(K/F)$, $T$ and the Heston "
                             "parameters, so one parameter set prices every expiry.",
            "The real-world return": "The index's real-world expected return plays no role. Estimating it would "
                                     "need a time series of prices and a physical-measure model.",
        }, columns=2)}),
    ])
    return


@app.cell(hide_code=True)
def _(cards, mo):
    mo.vstack([
        mo.md(r"""
        ### 1.2 Four features to keep in mind

        Four properties of Heston come back later in the notebook.
        """),
        cards([
            ("Feature 1", "Mean reversion sets the term structure", r"""
            $$\mathbb E[V_t]=v_\infty+(v_0-v_\infty)\,e^{-\kappa t}$$

            - $v_0$ sets short expiries, $v_\infty$ long ones.
            - $\kappa$ sets the speed. A shock halves in $\ln 2/\kappa$ years.
            """),
            ("Feature 2", r"$\rho$ tilts the smile, $\sigma_v$ curves it", r"""
            - With $\rho<0$, volatility rises when prices fall, so puts cost more. This is the **skew**.
            - A larger $\sigma_v$ makes the variance more random, which fattens both tails. This is the **curvature**, and it also steepens the skew.
            """),
            ("Feature 3", "The short-expiry skew has a ceiling", r"""
            $$\psi(T)=\frac{\partial\,\sigma_{\mathrm{imp}}(k,T)}{\partial k}\bigg|_{k=0},\qquad k=\ln(K/F)$$

            - $\psi(T)$ is the **at-the-money skew**: how fast implied volatility changes with log-moneyness at the money, for expiry $T$.
            - As $T$ shrinks, $|\psi(T)|$ grows, but only up to a ceiling: $|\psi(T)|\to|\rho|\,\sigma_v/(4\sqrt{v_0})$ as $T\to0$.
                - Over a few days the variance barely moves from $v_0$, so the skew depends only on $\rho$, $\sigma_v$ and $v_0$.
                - For expiries well below the reversion time $1/\kappa$ the skew is at the ceiling and grows no further.
            """),
            ("Feature 4", "Prices in closed form", r"""
            - The model is affine and $(F_t,V_t)$ is a Markov state, so the characteristic function is known exactly.
            - All 63 prices take milliseconds, so the **exact posterior** is affordable.
            """),
        ]),
    ])
    return


@app.cell(hide_code=True)
def _(
    BOUNDS,
    DELTA,
    EXPIRY_DAYS,
    REF_VOL,
    TRUTHS,
    atm_shape,
    caption,
    forward_unit_pricer,
    grid,
    mo,
    np,
    plots,
    plt,
    surface,
):
    _base = TRUTHS["synthetic-market-like"]  # (v0, kappa, v_inf, sigma_v, rho, c)
    _v0, _kappa, _v_inf, _xi, _rho, _ = _base
    _styles = [dict(color=plots.SERIES[0]), dict(color=plots.INK, lw=1.6), dict(color=plots.SERIES[1])]

    def _vary(_index, _values):
        """The market-like parameters with one entry replaced by each value in turn."""
        _rows = np.tile(_base, (len(_values), 1))
        _rows[:, _index] = _values
        return _rows

    # (a) Feature 1: the expected variance, in closed form.
    _days = np.geomspace(1, 730, 200)
    _starts = [_v0, _v_inf, min(2 * _v_inf, BOUNDS[0, 1])]
    _expected = [_v_inf + (_start - _v_inf) * np.exp(-_kappa * _days / 365) for _start in _starts]
    _half_life = 365 * np.log(2) / _kappa

    # (b, c) Feature 2: the 30-day smile across the quoted range of moneyness.
    _smile_t = 30 / 365
    _d = np.linspace(DELTA.min(), DELTA.max(), 40)
    _k = _d * REF_VOL * np.sqrt(_smile_t)
    _kind = np.where(_d <= 0, "put", "call")
    _smile_pricer = forward_unit_pricer(_k, np.full(_d.size, _smile_t), _kind)
    _smile = lambda _rows: [100 * surface.implied_vol(_smile_pricer.prices(_row)[None], _k, np.full(_d.size, _smile_t),
                                                      _kind)[0] for _row in _rows]
    _rhos = [-0.3, _rho, -0.9]
    _xis = [_xi / 2, _xi, min(1.5 * _xi, BOUNDS[3, 1])]
    _rho_smiles, _xi_smiles = _smile(_vary(4, _rhos)), _smile(_vary(3, _xis))

    # (d) Feature 3: the at-the-money skew from one day to one year, measured as in Lecture 5.
    _grid_t = np.geomspace(1 / 365, 1.0, 36)
    _grid_d = np.array([-1.25, -0.5, 0.0, 0.5, 1.0])
    _tt, _dd = np.meshgrid(_grid_t, _grid_d, indexing="ij")
    _gk, _gt, _gkind = (_dd * REF_VOL * np.sqrt(_tt)).ravel(), _tt.ravel(), np.where(_dd.ravel() <= 0, "put", "call")
    _grid_iv = surface.implied_vol(forward_unit_pricer(_gk, _gt, _gkind).prices(_base)[None], _gk, _gt,
                                   _gkind)[0].reshape(_tt.shape)
    _skew = np.abs([atm_shape(_grid_d * REF_VOL * np.sqrt(_t), _grid_iv[_i])[0] for _i, _t in enumerate(_grid_t)])
    _ceiling = abs(_rho * _xi / (4 * np.sqrt(_v0)))

    _fig, _axes = plt.subplots(2, 2, figsize=(10.8, 6.2), constrained_layout=True)
    _ax = _axes[0, 0]
    for _start, _curve, _style in zip(_starts, _expected, _styles):
        _ax.plot(_days, 100 * np.sqrt(_curve), label=f"$\\sqrt{{v_0}}$ = {100 * np.sqrt(_start):.0f}%", **_style)
    _ax.axhline(100 * np.sqrt(_v_inf), color=plots.MUTED, ls="--", lw=1)
    _ax.axvline(_half_life, color=plots.MUTED, ls=":", lw=1)
    _ax.text(_half_life * 1.08, 100 * np.sqrt(_starts[-1]), "half-life", fontsize=7, color=plots.MUTED, va="top")
    _ax.set(xscale="log", xlabel="days ahead", ylabel="volatility (%)",
            title=f"(a) Expected variance, as a volatility: towards $\\sqrt{{v_\\infty}}$ = {100 * np.sqrt(_v_inf):.0f}%")
    for _ax, _values, _curves, _symbol, _title in (
            (_axes[0, 1], _rhos, _rho_smiles, r"\rho", r"(b) $\rho$ tilts the 30-day smile"),
            (_axes[1, 0], _xis, _xi_smiles, r"\sigma_v", r"(c) $\sigma_v$ curves the 30-day smile")):
        for _value, _curve, _style in zip(_values, _curves, _styles):
            _ax.plot(_d, _curve, label=f"${_symbol}$ = {_value:.2f}".replace("-", "−"), **_style)
        _ax.set(xlabel="standardized moneyness δ", ylabel="implied volatility (%)", title=_title)
    _ax = _axes[1, 1]
    _ax.axvspan(min(EXPIRY_DAYS), max(EXPIRY_DAYS), color=plots.GRID, lw=0)
    _ax.text(np.sqrt(min(EXPIRY_DAYS) * max(EXPIRY_DAYS)), 1.12 * _ceiling, "quoted expiries", fontsize=7,
             color=plots.MUTED, ha="center")
    _ax.plot(365 * _grid_t, _skew, **_styles[1], label="market-like parameters")
    _ax.axhline(_ceiling, color=plots.MUTED, ls=":", lw=1, label=r"ceiling $|\rho|\sigma_v/(4\sqrt{v_0})$")
    _ax.set(xscale="log", yscale="log", xlabel="days to expiry", ylabel="|at-the-money skew|", ylim=(None, 1.35 * _ceiling),
            title="(d) The skew levels off at short expiries")
    for _ax in _axes.flat:
        _ax.title.set_fontsize(9)
        _ax.legend(fontsize=7)
    for _ax in (_axes[0, 0], _axes[1, 1]):
        _ax.set_xticks([1, 7, 30, 90, 365] + ([730] if _ax is _axes[0, 0] else []))
        _ax.set_xticklabels([str(_t) for _t in _ax.get_xticks()])
        _ax.minorticks_off()
    _axes[1, 1].set_yticks([0.3, 0.5, 1, 2], labels=["0.3", "0.5", "1", "2"])

    _deep = lambda _curves: (_curves[0][0], _curves[-1][0])
    _call = lambda _curves: (_curves[0][-1], _curves[-1][-1])
    _atm = lambda _curves: tuple(float(np.interp(0.0, _d, _c)) for _c in (_curves[0], _curves[-1]))
    _at = lambda _day: float(np.interp(np.log(_day), np.log(365 * _grid_t), _skew))
    _near = 365 * _grid_t[_skew >= 0.9 * _ceiling].max()
    _minus = lambda _x: f"{_x:.2f}".replace("-", "−")
    _wings = f"the put at $\\delta={_d[0]:.1f}$", f"the call at $\\delta={_d[-1]:.1f}$"
    mo.vstack([
        _fig,
        caption("Features 1 to 3. Black curves use the market-like parameters of Section 4.1. In each panel one "
                "parameter changes."),
        grid({
            "(a) Mean reversion":
                f"The expected variance moves from $v_0$ to $v_\\infty$ and covers half the distance in "
                f"{_half_life:.0f} days ($\\ln 2/\\kappa$). Implied volatilities sit below these curves, because a "
                "random variance lowers at-the-money volatility (Lecture 10, Figure 10.6).",
            "(b) Correlation":
                f"From $\\rho={_minus(_rhos[0])}$ to ${_minus(_rhos[-1])}$, {_wings[0]} rises from "
                f"{_deep(_rho_smiles)[0]:.1f}% to {_deep(_rho_smiles)[1]:.1f}% and {_wings[1]} falls from "
                f"{_call(_rho_smiles)[0]:.1f}% to {_call(_rho_smiles)[1]:.1f}%.",
            "(c) Volatility of variance":
                f"From $\\sigma_v={_xis[0]:.2f}$ to ${_xis[-1]:.2f}$, {_wings[0]} rises from "
                f"{_deep(_xi_smiles)[0]:.1f}% to {_deep(_xi_smiles)[1]:.1f}% and the at-the-money level falls from "
                f"{_atm(_xi_smiles)[0]:.1f}% to {_atm(_xi_smiles)[1]:.1f}%.",
            "(d) Short-expiry skew":
                f"The skew is {_at(365):.2f} at one year, {_at(30):.2f} at 30 days and {_at(7):.2f} at 7 days. Below "
                f"{_near:.0f} days it stays within 10% of the ceiling, {_ceiling:.2f}.",
        }, columns=2, kind="reading"),
    ])
    return


@app.cell(hide_code=True)
def _(NAMES, TRUTHS, grid, mo, reference_summary, stack):
    _truth = TRUTHS["synthetic-market-like"]
    _market = {_n: reference_summary["cases"]["market"]["parameters"][_n]["regenerated_q05_q50_q95"][1] for _n in NAMES}
    _like = (2 * _truth[1] * _truth[2], _truth[3] ** 2)
    _fit = (2 * _market["kappa"] * _market["theta"], _market["vol_of_vol"] ** 2)
    mo.accordion({
        "Feature 2 in depth: a large σᵥ and the Feller condition": grid({
            "The condition": r"A steep equity skew needs a large $\sigma_v$ (feature 2), and a large $\sigma_v$ "
                             r"breaks the **Feller condition** $2\kappa v_\infty\ge\sigma_v^2$.",
            "What it means": "If the condition holds, the variance never reaches zero. Otherwise the variance can "
                             "touch zero, and the drift then moves it back up. The model and the pricer are valid "
                             "either way, and the prior does not impose the condition.",
            "In this notebook": rf"The market-like parameters of Section 4.1 violate it ($\sigma_v^2={_like[1]:.1f}$ "
                                rf"against $2\kappa v_\infty={_like[0]:.2f}$). So do the SPX market posterior "
                                rf"medians ($\sigma_v^2={_fit[1]:.1f}$ against $2\kappa v_\infty={_fit[0]:.2f}$). "
                                "The posterior fits the SPX skew with a volatility of variance far above the "
                                "Feller limit.",
        }),
        "Feature 4 in depth: the Fourier pricer": grid([
            r"""
            Heston is **affine**. Its characteristic function $\varphi(u)=\mathbb E^{\mathbb Q}\big[e^{iu\ln(F_T/F_0)}\big]$ is known in closed form, so a European option needs one one-dimensional integral (Lewis, 2001).

            The pricer uses a **control-variate** form: the Black–Scholes price $C_{\mathrm{BS}}$ with the same expected average variance, plus the integral of the difference of the two characteristic functions. Puts get the same correction.

            $$C=C_{\mathrm{BS}}+D\,\frac{\sqrt{FK}}{\pi}\int_0^\infty\operatorname{Re}\!\left[e^{iu\ln(F/K)}\Big(\varphi_{\mathrm{BS}}\big(u-\tfrac i2\big)-\varphi\big(u-\tfrac i2\big)\Big)\right]\frac{du}{u^2+\tfrac14}$$
            """,
            stack({
                "Why the control variate": "For very cheap options the plain formula is a small difference of two "
                                           "large numbers. This form avoids the cancellation.",
                "Accuracy": "Gauss–Legendre quadrature. Nodes and interval are doubled until two successive steps "
                            "agree to within 2% of the smallest noise standard deviation the prior allows for that "
                            "contract ($c=0.5$, that is 1% of its half-spread).",
                "Speed": "Milliseconds for all 63 options. The prior predictive check in Section 2.2 times 1,000 "
                         "simulated data sets.",
            }),
        ]),
    })
    return


@app.cell(hide_code=True)
def _(mo, note):
    mo.vstack([
        mo.md(r"""
        ### 1.3 Observation model: noise set by the bid–ask spread

        Each quote is its Heston price plus Gaussian noise. The standard deviation of the noise is $c$ times the quote's half-spread.

        $$
        y_i = P_i(v_0,\kappa,v_\infty,\sigma_v,\rho) + c\,h_i\,\epsilon_i,\qquad
        \epsilon_i \overset{\mathrm{iid}}{\sim}\mathcal N(0,1),\qquad i=1,\dots,63
        $$

        | Symbol | Role | Meaning |
        |:---|:---|:---|
        | $y_i$ | data | the quote (mid price) of contract $i$ |
        | $P_i$ | model | its Heston price |
        | $h_i$ | data | the observed half-spread, fixed and identical in every simulation |
        | $\epsilon_i$ | noise | a fresh standard-normal draw for every contract and every simulated data set |
        | $c$ | parameter | scales all 63 noise levels together, inferred with the other five |

        "One half-spread is one standard deviation" is a convention. Inferring $c$ lets the data correct it.
        """),
        note("A **model-error term** is left out. Heston is assumed to price every contract exactly, so a systematic "
             "gap between Heston and the market can only be absorbed by a larger $c$ (Section 5.4)."),
    ])
    return


@app.cell(hide_code=True)
def _(
    DELTA,
    EXPIRY,
    EXPIRY_COLORS,
    EXPIRY_DATES,
    H,
    MARKET_IV,
    key,
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
    mo.hstack([
        _fig,
        mo.vstack([
            key(f"**One half-spread is {np.nanmin(_points):.2f}–{np.nanmax(_points):.2f} volatility points**, "
                f"against a smile that spans about {_range:.0f} points."),
            mo.md("- The quotes are therefore very informative, and the posteriors in Section 4.2 are narrow.\n"
                  "- For the same reason, a systematic model error of even a fraction of a volatility point is "
                  "large compared with the assumed noise."),
        ]),
    ], widths=[3, 2], gap=2, align="center")
    return


@app.cell(hide_code=True)
def _(mo, note, section):
    mo.vstack([
        section("Simulate", "One simulation corresponds to a **forward pass** through the pricer plus the addition "
                "of noise. It produces one hypothetical market.", stage=2),
        mo.md(r"""
        ### 2.1 The simulator

        A simulation takes three steps.

        1. Draw $\boldsymbol\theta=(v_0,\kappa,v_\infty,\sigma_v,\rho,c)$ from the prior.
        2. Price the 63 contracts with the Fourier pricer.
        3. Add the quote noise $c\,h_i\,\epsilon_i$.

        The output is a vector of **63 numbers**, one noisy mid price per contract.
        """),
        note("In this toy model only the mid prices are simulated. Bids and asks are not. The half-spreads $h_i$ "
             "keep their observed values and set the size of the noise."),
    ])
    return


@app.cell(hide_code=True)
def _(BOUNDS, EXPIRY_DATES, EXPIRY_DAYS, TRUTHS, controls, mo, np):
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
    controls(
        mo.hstack([mo.vstack([sim_params[_j] for _j in range(3)]), mo.vstack([sim_params[_j] for _j in range(3, 6)])],
                  justify="start", gap=2),
        mo.hstack([sim_expiry, sim_seed], justify="start", gap=2),
        hint="Move the sliders. They start at the market-like parameters of Section 4.1.",
    )
    return sim_expiry, sim_params, sim_seed


@app.cell(hide_code=True)
def _(
    DELTA,
    EXPIRY,
    EXPIRY_DATES,
    H,
    MARKET_IV,
    caption,
    grid,
    implied_vols,
    mo,
    note,
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

    _view = [_fig, caption("Top: three forward-price paths over one year, and the volatility behind them. Bottom: the "
                           "quotes at the selected expiry, and their errors against the Heston price. The paths only "
                           "illustrate the dynamics. The simulator prices options exactly and never draws paths.")]
    if _clean is None:
        _view.append(note(f"The pricer did not reach its accuracy target for these parameters ({_failure}). "
                          "Move a slider."))
    else:
        _atm = 100 * _clean_iv[_at][np.argmin(np.abs(DELTA[_at]))]
        _deep = 100 * _clean_iv[_at][np.argmin(DELTA[_at])]
        _errors = np.abs(100 * (_quote_iv - _clean_iv))[_at]
        _feller = (
            f"\n\n<small>A single path does not settle there. $\\sigma_v^2={_xi ** 2:.1f}$ exceeds "
            f"$2\\kappa v_\\infty={2 * _kappa * _v_inf:.2f}$ (Feller condition, Section 1.2), so in the long run "
            f"volatility is below {_median_vol:.2g}% half the time (dotted line). Short spikes carry the average, and "
            "in between the price barely moves.</small>" if _shape < 1 else "")
        _invisible = np.nanmax(_errors) < 0.01 * np.ptp(_axes[1, 0].get_ylim())
        _view.append(grid({
            "Volatility": f"Volatility starts at {100 * np.sqrt(_v0):.0f}%. The expected variance reverts to "
                          f"$v_\\infty$ with a half-life of {12 * np.log(2) / _kappa:.1f} months. The dashed line "
                          f"marks $\\sqrt{{v_\\infty}}={100 * np.sqrt(_v_inf):.0f}\\%$.{_feller}",
            "Price": "With $\\rho<0$, falls in the price come with spikes in volatility.",
            "Quotes": f"At this expiry the Heston smile runs from {_atm:.1f}% at the money to {_deep:.1f}% at the "
                      f"deepest put. The quote errors are at most {np.nanmax(_errors):.2f} volatility points"
                      f"{', invisible on the smile' if _invisible else ''}.",
        }, kind="reading"))
    _view.append(mo.md(r"""
    #### What to try

    Go through the features of Section 1.2, then see how close you can get to the SPX market (grey).

    | Move | What you see | Feature |
    |:---|:---|:---:|
    | set $v_0$ above $v_\infty$, then switch expiry | short expiries above long ones | 1 |
    | lower $\kappa$ | a slower return to $v_\infty$ | 1 |
    | make $\rho$ more negative | a steeper skew | 2 |
    | raise $\sigma_v$ | more curvature | 2 |
    | raise $v_0$ at the 7-day expiry | a flatter skew, because its ceiling falls like $1/\sqrt{v_0}$ | 3 |
    | raise $c$ | larger quote errors | noise |
    """))
    mo.vstack(_view)
    return


@app.cell(hide_code=True)
def _(bayesflow):
    bayesflow("""
    `bf.make_simulator` chains two functions by argument name. The prior returns named parameters. The observation model receives them, prices the 63 contracts and adds the noise. It returns every quote **divided by its half-spread**, so that the network sees the data in noise units.
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
def _(BOUNDS, grid, mo, np):
    _lo, _hi = BOUNDS[:, 0], BOUNDS[:, 1]
    _n = lambda _x: f"{_x:g}".replace("-", "−")
    _days = lambda _kappa: np.log(2) / _kappa * 365
    mo.vstack([
        mo.md(r"""
        ### 2.2 Prior predictive check

        The **prior predictive distribution** is the distribution of quotes the model produces before seeing data: draw $\boldsymbol\theta$ from the prior, then simulate.

        Checking it answers two questions.
        """),
        grid({
            "Does it cover the observed quotes?": "If not, the network would have to extrapolate, and the prior or "
                                                  "the model would need rethinking.",
            "Is it needlessly wide?": "Simulations of implausible markets cost training effort and tell the network "
                                      "little.",
        }),
        mo.md(rf"""
        The prior is **independent and uniform** in each parameter's natural units.

        | Symbol | Uniform on | What the range means |
        |:---|:---|:---|
        | $v_0$ | {_n(_lo[0])} to {_n(_hi[0])} | initial volatility $\sqrt{{v_0}}$ between {100 * np.sqrt(_lo[0]):.0f}% and {100 * np.sqrt(_hi[0]):.0f}% |
        | $\kappa$ | {_n(_lo[1])} to {_n(_hi[1])} | half-life of a variance shock, $\ln 2/\kappa$, between {_days(_hi[1]):.0f} days and {_days(_lo[1]) / 365:.0f} years |
        | $v_\infty$ | {_n(_lo[2])} to {_n(_hi[2])} | long-run volatility between {100 * np.sqrt(_lo[2]):.0f}% and {100 * np.sqrt(_hi[2]):.0f}% |
        | $\sigma_v$ | {_n(_lo[3])} to {_n(_hi[3])} | large values are allowed, and the Feller condition is not imposed |
        | $\rho$ | {_n(_lo[4])} to {_n(_hi[4])} | negative only, so volatility rises when prices fall |
        | $c$ | {_n(_lo[5])} to {_n(_hi[5])} | quote-noise SD between {_n(_lo[5])} and {_n(_hi[5])} half-spreads |
        """),
    ])
    return


@app.cell(hide_code=True)
def _(
    DELTA,
    EXPIRY,
    EXPIRY_DATES,
    H,
    MARKET_IV,
    MID,
    caption,
    implied_vols,
    key,
    make_simulator,
    mo,
    note,
    np,
    plots,
    plt,
    time,
):
    _started = time.perf_counter()
    prior_batch = make_simulator(20261111).sample(1000)
    prior_seconds = time.perf_counter() - _started
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
    _cover = (f"**The market lies inside the simulated range at every contract.** Its percentile among the simulated "
              f"prices runs from {_pct.min():.0f}% to {_pct.max():.0f}% (lowest at {EXPIRY_DATES[EXPIRY[_low]]}, "
              f"$\\delta={DELTA[_low]:.1f}$)." if _edge == 0 else
              f"**The market falls outside the simulated range at {_edge} of 63 contracts.**")
    mo.vstack([
        _fig,
        caption("150 of 1,000 prior-predictive smiles (blue) at three expiries, with the SPX market (orange). "
                f"Simulating all 1,000 data sets took {prior_seconds:.0f} s."),
        key(_cover),
        mo.md("- The prior produces calm and stressed markets, flat and steep skews.\n"
              "- The check only shows that the prior does not rule the data out. Whether the model fits them is "
              "the question of Section 5.4."),
        note(f"{100 * np.isnan(_iv).mean():.1f}% of simulated quotes have no implied volatility, because noise "
             "pushed a very cheap option below its intrinsic value. They remain valid data."),
    ])
    return prior_batch, prior_seconds


@app.cell(hide_code=True)
def _(grid, mo, note):
    mo.vstack([
        mo.md(r"""
        ### 2.3 Preparing the network input: whitening

        The network receives the 63 numbers $y_i/h_i$, the quotes in noise units. In this raw form they are hard to learn from, for two reasons.
        """),
        grid({
            "Strongly correlated": "A change in any parameter moves all 63 inputs together.",
            "Badly scaled": "A few directions, such as the overall level and the skew, vary enormously across "
                            "simulations. The directions that reveal the noise level $c$ vary about a million "
                            "times less.",
        }),
        mo.md(r"""
        **Whitening** is a fixed linear change of coordinates. It makes the inputs uncorrelated with unit variance, without losing information. It is fitted once on simulations, frozen, and applied unchanged to every later input, the SPX quotes included.
        """),
        note("The pretrained estimator's whitening was fitted on its first 10,000 training rows."),
    ])
    return


@app.cell(hide_code=True)
def _(caption, fit_whitening, grid, mo, np, plots, plt, prior_batch, stack):
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
        caption("Computed on the 1,000 simulations of Section 2.2. The whitening is fitted on 700 of them and shown "
                "on the other 300."),
        grid({
            "Variance per direction": f"The first principal direction of the inputs varies about "
                                      f"${_mantissa}\\times10^{{{int(_exponent)}}}$ times more than the last.",
            "Correlations": "On new simulations almost every pair of inputs is strongly correlated before "
                            "whitening. Afterwards the matrix is nearly diagonal.",
            "Information about $c$": "The information about $c$ sits in the low-variance directions "
                                                  f"(average {np.mean(_link[-30:]):.2f} over the last 30, against "
                                                  f"{np.mean(_link[:10]):.2f} over the first 10). Whitening gives "
                                                  "these directions the same weight as the dominant ones.",
        }, kind="reading", widths=(1, 2, 1)),
        mo.accordion({"Whitening, formally": grid([
            r"""
            Let $x\in\mathbb R^{63}$ be the vector of inputs $y_i/h_i$, with mean $\mu$ and covariance $\Sigma$ estimated from $N$ simulated training data sets. Write $\Sigma=U\Lambda U^\top$, with orthonormal eigenvectors $U$ and eigenvalues $\Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_{63})$. The whitened input is

            $$\tilde x = W(x-\mu),\qquad W=U\Lambda^{-1/2}U^\top=\Sigma^{-1/2}$$

            so that $\operatorname{Cov}(\tilde x)=W\Sigma W^\top=I$.
            """,
            stack({
                "Why this W": r"This choice of $W$ is **ZCA whitening**. Among all transformations with identity "
                              r"covariance, it keeps $\tilde x$ closest to $x$, so each whitened input still "
                              r"corresponds roughly to one quote. The alternative $\Lambda^{-1/2}U^\top$ (PCA "
                              r"whitening) would mix them. Eigenvalues below $10^{-8}\lambda_{\max}$ are floored "
                              "for numerical stability.",
                "Why it helps training": r"The ratio $\lambda_{\max}/\lambda_{\min}$, the **condition number** of "
                                         r"$\Sigma$, measures how badly scaled the raw inputs are. Gradient descent "
                                         "on badly scaled inputs is slow and unreliable, because a step size that "
                                         "suits the dominant directions is far too small for the others. Whitening "
                                         "sets the condition number to one.",
                "Fitted once": r"$\mu$ and $W$ come from training simulations only and are stored with the "
                               "estimator, so the same transformation is applied at training and at inference.",
            }),
        ])}),
    ])
    return


@app.cell(hide_code=True)
def _(mo, section):
    mo.vstack([
        section("Train", r"The network learns the posterior $q_\phi(\boldsymbol\theta\mid\mathbf y)$ from simulated "
                r"pairs $(\boldsymbol\theta,\mathbf y)$. No likelihood is ever evaluated.", stage=3),
        mo.md(r"""
        Each pair is a draw from the joint distribution of parameters and quotes. So among the simulations that produced quotes like $\mathbf y$, the parameters are distributed according to the posterior. The network learns that conditional distribution for every $\mathbf y$ at once.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(bayesflow, grid, mo):
    _steps = (
        ("Quotes y", "63 mid prices, simulated during training and observed at inference", ""),
        ("Preprocessing", "y/h and the frozen whitening give x̃, 63 numbers. The BayesFlow adapter stacks θ, 6 "
                          "numbers, which the workflow standardizes.", ""),
        ("Summary network", "none, the 63 numbers enter directly", "is-absent"),
        ("Inference network", "FlowMatching, an MLP of 3 × 128 units with dropout 0.1", "is-network"),
        ("Posterior draws", "θ = (v₀, κ, v∞, σᵥ, ρ, c), as many as needed", ""),
    )
    _chain = '<div class="nb-flow-arrow">&#8594;</div>'.join(
        f'<div class="nb-flow-step {_kind}"><div class="nb-flow-name">{_name}</div>'
        f'<div class="nb-flow-text">{_text}</div></div>' for _name, _text, _kind in _steps)
    mo.vstack([
        mo.md(r"""
        ### 3.1 The network

        The quotes go in on the left. Draws from the posterior come out on the right.
        """),
        mo.Html(f'<div class="nb-flow">{_chain}</div>'),
        grid({
            "In training": r"The network sees simulated pairs $(\boldsymbol\theta,\tilde{\mathbf x})$: 300,000 for "
                           "the pretrained estimator.",
            "At inference": r"The network receives one observed $\tilde{\mathbf x}$ and returns as many posterior "
                            "draws as wanted.",
        }),
        mo.accordion({"Design choices and their alternatives in BayesFlow": mo.md(r"""
        | Design choice | Here | Alternatives in BayesFlow |
        |:---|:---|:---|
        | What is learned | the posterior $q(\boldsymbol\theta\mid\mathbf y)$ | a neural pricer, a likelihood (NLE) or a likelihood ratio (NRE), each used inside a new MCMC run per data set |
        | Summary network | none, the input is a fixed vector of 63 whitened quotes | `DeepSet` or `SetTransformer` for quote sets of varying size and layout, `TimeSeriesNetwork` for price histories |
        | Inference network | `FlowMatching` | `CouplingFlow` (exact density, one-pass sampling), `DiffusionModel`, `ConsistencyModel` (few sampling steps) |
        | Parameter space | standardized and unbounded, so a few draws can leave the prior | bounded transforms in the adapter (`.constrain`), so that every draw respects the prior bounds |
        | Training data | a fixed bank (`fit_offline`), one noise draw per row | fresh simulations in every batch (`fit_online`) |
        | Amortized over | quotes for these 63 contracts and these spreads | contract terms and spreads as extra inputs, to reuse one network across days |
        """)}),
        bayesflow("""
        The workflow of the pretrained estimator has three parts. The adapter names the network's inputs. `FlowMatching` is the inference network. `BasicWorkflow` ties them together, standardizes the parameters and runs the training.
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
def _(bayesflow, mo, note):
    mo.vstack([
        mo.md(r"""
        ### 3.2 Training the pretrained estimator

        The pretrained estimator was trained once. Its training protocol records the settings.

        | Setting | Value |
        |:---|:---|
        | Data | 300,000 simulated pairs. Each keeps its noise draw throughout training. |
        | Schedule | 300 epochs with batch size 256, 351,600 updates, seed 9011 |
        | Optimizer | BayesFlow's default for offline training: AdamW with weight decay 0.005 and gradient clipping at 1.5. The learning rate warms up from $5\cdot10^{-5}$ to $5\cdot10^{-4}$, then decays to zero on a cosine schedule. |
        | Hardware | the CPU of an Apple-silicon Mac, within the run's one-hour limit |
        """),
        bayesflow("Training is one call, `workflow.fit_offline`, on the simulated pairs."),
        mo.md(r"""
        ```python
        task, data = delivery.training_inputs(HESTON_BUNDLE, rows=300_000)   # (θ, x̃) pairs, x̃ already whitened
        keras.utils.set_random_seed(9011)
        workflow = build_workflow()
        history = workflow.fit_offline(data=data["train"], validation_data=data["validation"], epochs=300, batch_size=256)
        ```
        """),
        note("The parameters are standardized but not mapped onto their bounded prior ranges, so a small fraction "
             "of posterior draws falls just outside the prior. Such draws are kept, not clipped."),
    ])
    return


@app.cell(hide_code=True)
def _(caption, grid, mo, np, plots, plt):
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
    _axes[3].set_title("Training: straight paths")
    for _ax in _axes:
        _ax.set(xlabel="parameter 1 (standardized)")
    _axes[0].set_ylabel("parameter 2 (standardized)")
    mo.accordion({"How flow matching works": mo.vstack([
        mo.md(r"""
        The inference network is a **velocity field** $v_\phi(\mathbf z,t,\tilde{\mathbf x})$ on the six-dimensional parameter space, for times $t\in[0,1]$.
        """),
        grid({
            "Training": r"Take a simulated pair $(\boldsymbol\theta,\tilde{\mathbf x})$, with $\boldsymbol\theta$ "
                        r"standardized. Draw noise $\mathbf z_1\sim\mathcal N(0,I)$ independently of "
                        r"$\boldsymbol\theta$, and a time $t$, more often near $t=0$. Form the point "
                        r"$\mathbf z_t=(1-t)\,\boldsymbol\theta+t\,\mathbf z_1$ on the straight line from "
                        r"$\boldsymbol\theta$ to $\mathbf z_1$. Fit $v_\phi(\mathbf z_t,t,\tilde{\mathbf x})$ to the "
                        r"velocity $\mathbf z_1-\boldsymbol\theta$ with a squared-error loss.",
            "Why it works": "Many pairs pass through the same point, so the network learns the average velocity "
                            "there. Following this averaged field moves probability mass so that its distribution "
                            r"at each time $t$ is that of $\mathbf z_t$: noise at $t=1$, the posterior at $t=0$.",
            "Sampling": r"Draw $\mathbf z_1\sim\mathcal N(0,I)$ and integrate "
                        r"$d\mathbf z/dt=v_\phi(\mathbf z,t,\tilde{\mathbf x})$ from $t=1$ back to $t=0$ with an "
                        "adaptive Runge–Kutta solver (Tsit5). The end point is one posterior draw. Undoing the "
                        r"standardization gives $\boldsymbol\theta$.",
            "Network inputs": r"$\mathbf z_t$ and $\tilde{\mathbf x}$ enter an MLP of three layers of 128 units. "
                              "$t$ enters through a Fourier embedding that modulates every layer.",
        }, columns=2),
        _fig,
        caption("A toy posterior in two dimensions: the cloud at $t=1$ (noise), halfway, and at $t=0$ (posterior). "
                "The last panel shows straight training paths between simulated parameters (orange) and independent "
                "noise draws (blue)."),
    ])})
    return


@app.cell(hide_code=True)
def _(bayesflow, controls, mo):
    live_train_button = mo.ui.run_button(label="Simulate 3,000 data sets and train a small network (about a minute on a CPU)")
    mo.vstack([
        mo.md(r"""
        ### 3.3 Train a small network yourself

        The button runs the same loop at a small scale: 3,000 simulations, 40 epochs, then BayesFlow's own diagnostics.
        """),
        controls(live_train_button, hint="The result appears in the next cell."),
        bayesflow("""
        The cell below is the whole loop: simulate, whiten, train with `fit_offline`, sample for 100 new data sets, and plot `bf.diagnostics`.
        """),
    ])
    return (live_train_button,)


@app.cell(hide_code=True)
def _(EVALUATION, LABELS, NAMES, caption, json, key, mo, note, np, table):
    def live_summary(train, validation, held_out, draws, history, seconds):
        """The text about one small training run. Every number in it is computed from that run."""
        loss, val = history.history["loss"][-1], history.history.get("val_loss", [np.nan])[-1]
        fit = ("The validation loss ends close to the training loss, so the network is not overfitting."
               if val <= 1.1 * loss else
               "The validation loss ends clearly above the training loss, so the network has started to overfit.")
        corr = [float(np.corrcoef(np.median(draws[name][..., 0], axis=1), held_out[name][:, 0])[0, 1]) for name in NAMES]
        evaluation = json.loads((EVALUATION / "evaluation.json").read_text())  # the pretrained estimator's record
        pretrained = evaluation["population"]["mean_posterior_sd"]
        width = [float(np.mean(np.std(draws[name][..., 0], axis=1))) / pretrained[j] for j, name in enumerate(NAMES)]
        n_train, n_val, n_test = (len(data["observations"]) for data in (train, validation, held_out))
        epochs = len(history.history["loss"])
        ratio = lambda w: f"{w:.0f}×" if w >= 10 else f"{w:.1f}×"
        rows = [{"": "Recovery: correlation of median and truth", **{label: f"{r:.2f}" for label, r in zip(LABELS, corr)}},
                {"": "Posterior SD, relative to the pretrained estimator",
                 **{label: ratio(w) for label, w in zip(LABELS, width)}}]
        return {
            "training": mo.vstack([
                mo.md(f"#### Training\n\nSimulated {n_train:,} training and {n_val} validation data sets in "
                      f"{seconds['simulate']:.0f} s, then trained for {epochs} epochs in {seconds['train']:.0f} s. "
                      f"The final training loss is {loss:.3f} and the validation loss {val:.3f}. {fit}"),
                note("The loss is the flow-matching error, not a likelihood. Only its trend and the gap between the "
                     "two curves matter."),
            ]),
            "check": mo.md(f"#### Check on {n_test} new data sets\n\nThe network just trained draws "
                           f"{draws[NAMES[0]].shape[1]} posterior samples for each of {n_test} new simulated data "
                           f"sets, not used in training ({seconds['sample']:.0f} s)."),
            "recovery": caption("Recovery. One dot per data set: the true value against the posterior median, with "
                                "a bar for the central 68% interval. Dots near the diagonal mean the parameter is "
                                "recovered."),
            "calibration": caption("Calibration. If the network's uncertainty is right, each curve stays inside its "
                                   "grey band. The foldable section below explains how to read the plot."),
            "comparison": mo.vstack([
                key(f"**Fewer simulations, wider posteriors.** This network saw "
                    f"{100 * n_train / evaluation['training_rows']:.0f}% of the pretrained estimator's training data "
                    f"and trained for {epochs} epochs instead of {evaluation['native_epoch']}. Its posteriors are "
                    f"{min(width):.1f} to {max(width):.0f} times wider."),
                table(rows, align="l" + "r" * len(LABELS)),
            ]),
        }

    return (live_summary,)


@app.cell
def _(
    LABELS,
    NAMES,
    bf,
    build_workflow,
    ecdf_guide,
    fit_whitening,
    keras,
    live_summary,
    live_train_button,
    make_simulator,
    mo,
    np,
    plots,
    time,
):
    mo.stop(not live_train_button.value, mo.md("*Press the button above to simulate and train.*"))
    _clock = [time.perf_counter()]
    live_simulator = make_simulator(4)  # its own seeded generator
    _train = live_simulator.sample(3000)
    _validation = live_simulator.sample(300)
    _clock.append(time.perf_counter())

    live_whitening = fit_whitening(_train["observations"])  # fitted on training rows only, then frozen
    _train["observations"] = live_whitening.transform(_train["observations"])
    _validation["observations"] = live_whitening.transform(_validation["observations"])

    keras.utils.set_random_seed(4)
    live_workflow = build_workflow()   # the BayesFlow workflow of Section 3.1
    live_history = live_workflow.fit_offline(data=_train, validation_data=_validation, epochs=40, batch_size=64, verbose=0)
    _clock.append(time.perf_counter())

    # Check the network just trained on 100 new simulations.
    _held = live_simulator.sample(100)
    _clock.append(time.perf_counter())
    _draws = live_workflow.sample(conditions={"observations": live_whitening.transform(_held["observations"])},
                                  num_samples=200)
    _clock.append(time.perf_counter())

    # The sentences about this run come from a hidden cell above. BayesFlow draws the three diagnostic plots.
    _text = live_summary(_train, _validation, _held, _draws, live_history,
                         dict(zip(("simulate", "train", "hold out", "sample"), np.diff(_clock))))
    mo.vstack([
        _text["training"],
        bf.diagnostics.loss(live_history),
        _text["check"],
        bf.diagnostics.recovery(estimates=_draws, targets=_held, variable_keys=NAMES, variable_names=LABELS,
                                uncertainty_agg_kwargs={"prob": 0.68}, color=plots.SERIES[0]),
        _text["recovery"],
        bf.diagnostics.calibration_ecdf(estimates=_draws, targets=_held, variable_keys=NAMES, variable_names=LABELS,
                                        difference=True, rank_ecdf_color=plots.SERIES[0], fill_color=plots.BASELINE),
        _text["calibration"],
        _text["comparison"],
        ecdf_guide,
    ])
    return


@app.cell(hide_code=True)
def _(TRUTHS, bayesflow, cards, mo, np, section):
    _stressed = TRUTHS["synthetic-broad"]
    mo.vstack([
        section("Infer", "The pretrained estimator returns posterior draws for observed quotes in seconds. A PyMC "
                "run with the exact likelihood is the benchmark.", stage=4),
        mo.md(r"""
        ### 4.1 Three data sets and two samplers

        We apply the pretrained estimator to three sets of 63 quotes.
        """),
        cards([
            ("Simulated", "Market-like parameters",
             "Generated with $c=1$ at parameters within 15% of the SPX market posterior medians."),
            ("Simulated", "Stressed market",
             "Generated with $c=1$ at a high-volatility draw from the prior. Volatility starts at "
             f"{100 * np.sqrt(_stressed[0]):.0f}% and reverts to {100 * np.sqrt(_stressed[2]):.0f}%."),
            ("Real", "SPX market quotes", "The quotes of 9 November 2026. The true parameters are unknown."),
        ], columns=3),
        bayesflow("Inference is one call, `workflow.sample`. The loader wraps it so that it takes raw quotes."),
        mo.md(r"""
        ```python
        model, checkpoint = delivery.load_checkpoint(HESTON_BUNDLE, "300k-300-final300")
        draws = model.sample(quotes[None, :], 4096, seed=261113)[0]      # (4096, 6) posterior draws
        # inside: x_tilde = whitening.transform(quotes / half_spread)
        #         workflow.sample(conditions={"observations": x_tilde}, num_samples=4096)
        ```

        Nothing is fitted at this stage. Each draw integrates the learned flow from one noise vector.

        As a benchmark, a **PyMC reference** samples the same posterior by MCMC with the exact likelihood.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(CASE_TITLES, REFERENCES, grid, json, mo, note, table):
    _rows = []
    for _case_id in CASE_TITLES:  # the order of the data-set menu
        _run = json.loads((REFERENCES / "references" / _case_id / "run.json").read_text())
        _stats = json.loads((REFERENCES / "references" / _case_id / "diagnostics.json").read_text())["parameters"]
        _attempt = _run["attempts"][_run["selected_attempt"]]
        _rows.append({
            "Data set": CASE_TITLES[_case_id],
            "Chains × draws": f"{_run['settings']['chains']} × {_attempt['draws']:,} after {_attempt['tune']:,} tuning steps",
            "Largest R-hat": f"{max(_s['r_hat'] for _s in _stats.values()):.4f}",
            "Smallest bulk ESS": f"{min(_s['ess_bulk'] for _s in _stats.values()):,.0f}",
            "Likelihood evaluations": f"{_attempt['likelihood_evaluations']:,}",
            "Run time, one core": f"{_attempt['wall_seconds'] / 60:.0f} min",
        })
    mo.accordion({"How the PyMC reference is computed": mo.vstack([
        grid({
            "Same posterior as the network": r"The same uniform prior and the exact likelihood "
                                             r"$y_i\sim\mathcal N\big(P_i(\boldsymbol\theta),(c\,h_i)^2\big)$, with "
                                             "prices from the Fourier pricer of Section 1.2 (feature 4).",
            "Sampler": "PyMC's `DEMetropolisZ`, a gradient-free Metropolis sampler whose proposals are differences "
                       "of past draws. The likelihood calls the pricer as a black box without gradients, which "
                       "rules out NUTS.",
            "Runs": "Four chains, started from maximum-likelihood (Powell) fits and dispersed points, never from "
                    "the true parameters or the network. A short run of 2,000 tuning steps and 2,000 draws comes "
                    "first. When it fails the checks, a long run of 6,000 and 12,000 follows.",
            "Checks": "R-hat at most 1.01, and bulk and tail effective sample sizes of at least 400, for every "
                      "parameter. Prices are accurate to 2% of the smallest noise standard deviation.",
        }, columns=2),
        table(_rows, align="llrrrr"),
        note("The plots use a thinned subset of 500 draws per chain."),
    ])})
    return


@app.cell
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
                      f"({_record['train_rows']:,} simulated data sets, {_record['native_epoch']} epochs).")
    _view
    return (estimator,)


@app.cell(hide_code=True)
def _(CASES, CONDITIONAL_DRAWS, EVAL_SEED, estimator, mo, np, time):
    neural_draws = {}
    _view = None  # nothing to show without the pretrained estimator
    if estimator is not None:
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
def _(CASE_TITLES, controls, mo):
    case_choice = mo.ui.dropdown(options={_title: _case_id for _case_id, _title in CASE_TITLES.items()},
                                 value=CASE_TITLES["synthetic-market-like"], label="Data set")
    mo.vstack([
        mo.md(r"""
        ### 4.2 The posterior, one data set at a time

        For the data set chosen here, this section shows the quotes, the posterior of the six parameters and two derived quantities. Section 5.1 uses the same choice.
        """),
        controls(case_choice, hint="Choose a data set."),
    ])
    return (case_choice,)


@app.cell(hide_code=True)
def _(
    BOUNDS,
    CASES,
    CASE_TITLES,
    DELTA,
    EXPIRY,
    EXPIRY_COLORS,
    EXPIRY_DATES,
    EXPIRY_DAYS,
    LABELS,
    MARKET_IV,
    MATH_LABELS,
    NAMES,
    PRODUCTS,
    PRODUCT_TITLES,
    TRUTHS,
    caption,
    case_choice,
    expected_variance,
    grid,
    implied_vols,
    key,
    mo,
    neural_draws,
    np,
    pairs,
    plt,
    reference_draws,
    reference_products,
    reference_summary,
    table,
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

    _g = lambda _x: f"{_x:.5g}".replace("-", "−")
    _rows = []
    _ref = reference_summary["cases"][_case]["parameters"]
    for _j, _name in enumerate(NAMES):
        _rq = _ref[_name]["regenerated_q05_q50_q95"]
        _row = {"Parameter": MATH_LABELS[_j]}
        if _truth is not None:  # the SPX quotes have no known truth
            _row["Truth"] = _g(_truth[_j])
        if _neural is not None:
            _q = np.quantile(_neural[:, _j], [.05, .5, .95])
            _row["Network: median [90% interval]"] = f"{_g(_q[1])} [{_g(_q[0])}, {_g(_q[2])}]"
        _row["PyMC reference: median [90% interval]"] = f"{_g(_rq[1])} [{_g(_rq[0])}, {_g(_rq[2])}]"
        _rows.append(_row)

    _product_rows = []
    for _k, (_key, _interval) in enumerate(PRODUCTS.items()):
        _values = {"PyMC reference": reference_products[_case][:, _k]}
        if _neural is not None:
            _values = {"network": expected_variance(_neural[_neural[:, 1] > 0], _interval), **_values}
        for _source, _value in _values.items():
            _pq = np.quantile(_value, [.05, .5, .95])
            _product_rows.append({"Quantity": PRODUCT_TITLES[_key], "Source": _source,
                                  "Median, volatility (%)": f"{100 * np.sqrt(_pq[1]):.3f}",
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
    _sharpest, _loosest = min(_relative, key=_relative.get), max(_relative, key=_relative.get)
    _named = lambda _name: MATH_LABELS[NAMES.index(_name)]
    _capital = lambda _text: _text[0].upper() + _text[1:]
    _signed = lambda _x: f"${_x:+.2f}$"

    # The data set, described next to its smile.
    _detail = None
    if _truth is None:
        _scenario = ("These are the **real SPX quotes** shown at the start of the notebook. There is no known truth, and "
                     "the model itself may be wrong (Section 5.4).")
    elif _case == "synthetic-broad":
        _first, _last = (EXPIRY == 0), (EXPIRY == len(EXPIRY_DATES) - 1)
        _atm = [100 * _iv[_at][np.argmin(np.abs(DELTA[_at]))] for _at in (_first, _last)]
        _deep = [100 * _iv[EXPIRY == _e][np.argmin(DELTA[EXPIRY == _e])] for _e in range(len(EXPIRY_DATES))]
        # Rise of the smile from the money to the deepest put, at the first and last expiry.
        _rise = lambda _vols: [100 * (_vols[_at][np.argmin(DELTA[_at])] - _vols[_at][np.argmin(np.abs(DELTA[_at]))])
                               for _at in (_first, _last)]
        _case_rise, _spx_rise = _rise(_iv), _rise(MARKET_IV)
        _v0, _v_inf, _xi, _rho = _truth[[0, 2, 3, 4]]
        _scenario = ("These quotes are **simulated from Heston** with $c=1$, in a stressed market. At the money, "
                     f"implied volatility is {_atm[0]:.0f}% at {EXPIRY_DAYS[0]} days against {_atm[1]:.0f}% at "
                     f"{EXPIRY_DAYS[-1]} days. The term structure is inverted, as after a sell-off. Deep puts are at "
                     f"{min(_deep):.0f}–{max(_deep):.0f}%.")
        _detail = grid({
            "Why the term structure is inverted":
                f"The true spot volatility $\\sqrt{{v_0}}={100 * np.sqrt(_v0):.0f}\\%$ is above the long-run "
                f"$\\sqrt{{v_\\infty}}={100 * np.sqrt(_v_inf):.0f}\\%$. The large $\\sigma_v={_xi:.2f}$ pulls "
                "long-dated at-the-money volatility further down.",
            "Why the curves cross":
                "Heston's short-dated skew per unit of log-moneyness tends to a constant, "
                f"$\\rho\\sigma_v/(4\\sqrt{{v_0}})={_rho * _xi / (4 * np.sqrt(_v0)):.2f}$ (feature 3, Section 1.2). "
                "$\\delta$ rescales log-moneyness by $\\sqrt T$, so short-dated smiles are flat in $\\delta$. From the "
                f"money to the deepest put they rise {_case_rise[0]:.1f} points at {EXPIRY_DAYS[0]} days and "
                f"{_case_rise[1]:.1f} at {EXPIRY_DAYS[-1]} days. The short smiles are high and flat, the long ones "
                "low and steep.",
            "Unlike the SPX quotes":
                f"On SPX the same rise is {_spx_rise[0]:.1f} and {_spx_rise[1]:.1f} points, much closer to parallel. "
                "Section 6.4 analyses this limitation. As a test of the inference the case is still valid, because "
                "it is a draw from the model and the prior.",
        }, kind="reading")
    else:
        _scenario = ("These quotes are **simulated from Heston** with $c=1$. The model is right by construction, so "
                     "this data set tests the inference alone.")

    # The corner plot, read in up to three columns: the exact posterior, where the truth sits, the network against PyMC.
    _reading = ("The diagonal shows each parameter's posterior"
                + (", the network in blue and PyMC in orange" if _neural is not None else " from PyMC")
                + ". The panels below show pairs of parameters."
                + (" Black lines and stars mark the true values." if _truth is not None else ""))
    _exact = [
        f"- {_capital(_named(_sharpest))} is determined to within ±{100 * _relative[_sharpest]:.2g}% of its value, "
        f"and the least precise Heston parameter, {_named(_loosest)}, to within ±{100 * _relative[_loosest]:.2g}%.",
        f"- The strongest dependence is between {MATH_LABELS[_j]} and {MATH_LABELS[_i]} (correlation "
        f"{_signed(_corr[_i, _j])}).",
        "    - The quotes determine combinations of parameters better than each one alone.",
        "    - A quantity that depends on several parameters must be computed from the joint draws, not from the "
        "marginals.",
    ]
    _columns = {"The exact posterior (PyMC)": _exact}
    if _truth is None:
        _exact += [
            f"- $c$ sits at the top of its prior range (median {_ref['c']['regenerated_q05_q50_q95'][1]:.4f}, bound "
            f"{BOUNDS[-1, 1]:g}).",
            "    - The quotes ask for more noise than the model allows, because no Heston smile comes close enough "
            "(Section 5.4).",
        ]
    else:
        # Where does the truth sit in the reference posterior? Marginally (in posterior SDs) and jointly (Mahalanobis).
        _centre, _spread = _reference.mean(axis=0), _reference.std(axis=0, ddof=1)
        _z = (_truth - _centre) / _spread
        _precision = np.linalg.inv(np.cov(_reference, rowvar=False))
        _distance = lambda _x: np.einsum("...i,ij,...j->...", _x - _centre, _precision, _x - _centre)
        _joint = 100 * np.mean(_distance(_reference) < _distance(_truth))
        _outside = [MATH_LABELS[_k] for _k in range(6) if abs(_z[_k]) > 1.645]
        if _outside:
            _listed = _outside[0] if len(_outside) == 1 else ", ".join(_outside[:-1]) + " and " + _outside[-1]
            _where = [
                f"- {_capital(_listed)} {'lies' if len(_outside) == 1 else 'lie'} up to {np.max(np.abs(_z)):.1f} "
                "posterior SDs from the posterior mean, outside the 90% intervals.",
                "    - One noisy data set can do this. The noise shifts the posterior, and strongly correlated "
                "parameters shift together.",
            ]
        else:
            _where = ["- Every true value lies inside its 90% interval."]
        _where += [
            f"- Jointly, the truth is farther from the posterior centre than {_joint:.0f}% of the draws.",
            f"    - For a correct posterior this share is uniform between 0 and 100%, so a single value of "
            f"{_joint:.0f}% is no warning sign.",
            "    - Coverage is judged over many data sets in Section 5.2.",
        ]
        _columns["Where the truth sits"] = _where
    _verdict = None
    if _neural is None:
        _columns["The network against PyMC"] = ["The pretrained network is not loaded. With the bundle, its posterior "
                                                "appears in blue."]
    else:
        # The same measures as the table of Section 5.3: centre gap in PyMC SDs, and the ratio of 90% widths.
        _gaps = np.array([(_neural[:, _k].mean() - _ref[_n]["regenerated_mean"]) / _ref[_n]["regenerated_sd"]
                          for _k, _n in enumerate(NAMES)])
        _widths = np.array([np.subtract(*np.quantile(_neural[:, _k], [.95, .05]))
                            / (_ref[_n]["regenerated_q05_q50_q95"][2] - _ref[_n]["regenerated_q05_q50_q95"][0])
                            for _k, _n in enumerate(NAMES)])
        _far, _wide = int(np.argmax(np.abs(_gaps))), int(np.argmax(_widths))
        _columns["The network against PyMC"] = [
            f"- The network's centres differ from PyMC's by up to {np.max(np.abs(_gaps)):.2g} PyMC standard "
            f"deviations, most for {MATH_LABELS[_far]}.",
            f"- Its 90% intervals are {_widths.min():.2g} to {_widths.max():.2g} times as wide as PyMC's, widest for "
            f"{MATH_LABELS[_wide]}.",
            f"- For {MATH_LABELS[_j]} with {MATH_LABELS[_i]}, its correlation is "
            f"{_signed(np.corrcoef(_neural, rowvar=False)[_i, _j])}, against PyMC's {_signed(_corr[_i, _j])}.",
            "- Section 5.3 compares the three data sets.",
        ]
        if _truth is None:
            _verdict = ("**The network and PyMC do not agree.** The SPX quotes lie outside anything the network saw "
                        "in training, so it extrapolates (Section 5.4). The disagreement is a warning about the "
                        "model, not a training failure.")
        elif np.max(np.abs(_gaps)) < 1:
            _verdict = ("**The network is close to PyMC, but wider.** One network that serves every data set is less "
                        "sharp than MCMC run on this one. It is cautious rather than wrong, consistent with the "
                        "calibration checks of Section 5.2.")
        else:
            _verdict = ("**The network is shifted against PyMC.** Its centre misses PyMC's by more than one standard "
                        f"deviation for {MATH_LABELS[_far]}.")

    _view = [mo.hstack([_fig, mo.md(_scenario)], widths=[3, 2], gap=2, align="center")]
    if _detail is not None:
        _view.append(_detail)
    _view += [
        mo.md("#### The six parameters"),
        table(_rows, align="l" + "r" * (len(_rows[0]) - 1)),
        mo.md("#### The joint posterior"),
        _corner,
        caption(_reading),
    ]
    if _verdict is not None:
        _view.append(key(_verdict))
    _view += [
        grid({_title: "\n".join(_lines) for _title, _lines in _columns.items()}, kind="reading"),
        mo.md("#### Derived quantities\n\nEach posterior draw gives an expected average variance over a horizon, the "
              "fair value of a variance swap, shown here as a volatility. This is feature 1 of Section 1.2, averaged "
              "over the horizon."),
        table(_product_rows, align="llrr"),
    ]
    mo.vstack(_view)
    return


@app.cell(hide_code=True)
def _(
    CASES,
    EVALUATION,
    GRID,
    REFERENCES,
    caption,
    json,
    key,
    mo,
    note,
    np,
    plots,
    plt,
    prior_batch,
    prior_seconds,
):
    # Every number below is read from a run record or timed in this notebook.
    _recovery = GRID / "recovery-bank-independent"
    _rows = json.loads((_recovery / "bank-protocol.json").read_text())["contract"]["train_rows"]
    _train = next(_job["seconds"] for _job in json.loads((_recovery / "campaign.json").read_text())["jobs"]
                  if _job["id"] == "300k-300_training")
    _pymc = np.array([sum(_a["wall_seconds"] for _a in json.loads(
        (REFERENCES / "references" / _case / "run.json").read_text())["attempts"]) for _case in CASES])
    # The recorded evaluation of this network: every posterior it drew, with loading and diagnostics included.
    _evaluation = json.loads((EVALUATION / "evaluation.json").read_text())
    _posteriors = {"Section 5.2": _evaluation["population"]["raw_draws_used"] // _evaluation["population"]["draws_per_case"],
                   "validation": (_evaluation["native_validation"]["raw_draws_used"]
                                  // _evaluation["native_validation"]["draws_per_case"]),
                   "Section 4": len(_evaluation["cases"])}
    _net = _evaluation["seconds"] / sum(_posteriors.values())  # an upper bound per posterior
    _sim = prior_seconds / len(prior_batch["observations"]) * _rows  # Section 2's timing, scaled
    _once = _sim + _train

    def _duration(_seconds):
        if _seconds < 60:
            return f"{_seconds:.1f} s" if _seconds < 10 else f"{_seconds:.0f} s"
        if _seconds < 3600:
            return f"{_seconds / 60:.0f} min"
        return f"{_seconds / 3600:.1f} h" if _seconds < 36000 else f"{_seconds / 3600:,.0f} h"

    _net_color, _pymc_color = plots.SERIES[0], plots.SERIES[1]
    _bars = [  # (label, low, high, color, paid once)
        (f"simulate {_rows:,} data sets (estimate)", _sim, _sim, _net_color, True),
        ("train the network", _train, _train, _net_color, True),
        ("one posterior: network", _net, _net, _net_color, False),
        ("one posterior: PyMC", _pymc.min(), _pymc.max(), _pymc_color, False),
    ]
    _fig, _axes = plt.subplots(1, 2, figsize=(11.6, 3.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.2, 1]})
    _ax = _axes[0]
    for _y, (_label, _low, _high, _color, _once_only) in enumerate(_bars):
        _style = dict(color=_color, height=0.6, hatch="///" if _once_only else None, edgecolor="white", lw=0)
        _ax.barh(_y, _low, **_style)
        if _high > _low:
            _ax.barh(_y, _high - _low, left=_low, alpha=0.4, **_style)
        _text = _duration(_low) if _high == _low else f"{_duration(_low)} – {_duration(_high)}"
        _ax.text(_high * 1.4, _y, ("≤ " if _label == "one posterior: network" else "") + _text, va="center", fontsize=8)
    _ax.set_yticks(range(len(_bars)), [_b[0] for _b in _bars])
    _ax.invert_yaxis()
    _ticks = {"0.1 s": 0.1, "1 s": 1, "1 min": 60, "1 h": 3600}
    _ax.set_xscale("log")
    _ax.set_xticks(list(_ticks.values()), list(_ticks))
    _ax.minorticks_off()
    _ax.set_xlim(0.05, 4e4)
    _ax.grid(axis="y", visible=False)
    _ax.set_title("What each step costs (hatched: paid once)", fontsize=9)

    _n = np.geomspace(1, 2000, 200)
    _check = _posteriors["Section 5.2"]
    _ax = _axes[1]
    _ax.plot(_n, (_once + _n * _net) / 3600, color=_net_color, label="network: simulate and train, then sample")
    _ax.plot(_n, _n * _pymc.mean() / 3600, color=_pymc_color, label="PyMC: one run per data set")
    _ax.axvline(_check, color=plots.MUTED, ls=":", lw=1)
    _ax.text(_check * 0.9, 0.12, "Section 5.2", fontsize=7, color=plots.MUTED, ha="right")
    _ax.set(xscale="log", yscale="log", xlabel="number of data sets", ylabel="total time (hours)",
            title="Total for many data sets")
    _ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
    _hours = [0.1, 1, 10, 100, 1000]
    _ax.set_yticks(_hours, [f"{_h:,g}" for _h in _hours])
    _ax.set_ylim(0.1, None)
    _ax.minorticks_off()
    _ax.title.set_fontsize(9)
    _ax.legend(fontsize=7, loc="upper left")

    mo.vstack([
        mo.md(r"""
        ### 4.3 What it costs

        The network is paid for once. PyMC is paid again for every data set.
        """),
        _fig,
        caption("Left: the time of each step, on a logarithmic scale. Hatched bars are paid once. Right: the total "
                "time for many data sets."),
        key(f"**Simulating and training take about {_duration(_once)}, as long as {_once / _pymc.mean():.0f} PyMC "
            f"runs.** After that, a posterior takes under {_duration(_net)} instead of {_duration(_pymc.min())} to "
            f"{_duration(_pymc.max())}."),
        mo.md(f"For the {_check} data sets of Section 5.2, the network's recorded evaluation took "
              f"{_evaluation['seconds']:.0f} s, including {sum(_posteriors.values()) - _check} further posteriors. "
              f"PyMC would need about {_duration(_check * _pymc.mean())}."),
        mo.accordion({"Where the numbers come from": mo.vstack([
            mo.md(f"""
            | Step | Source |
            |:---|:---|
            | Simulation | Section 2.2 timed {len(prior_batch['observations']):,} simulations on one core of this machine. The figure scales that time to {_rows:,}. The cost per simulation varies with the parameters drawn, so this is approximate. |
            | Training | The training job's record: {_duration(_train)} of wall time on an Apple-silicon CPU, loading the data and saving the network included. |
            | Network posterior | This network's recorded evaluation: {sum(_posteriors.values())} posteriors ({_posteriors['Section 5.2']} for Section 5.2, {_posteriors['validation']} for validation, {_posteriors['Section 4']} for Section 4.2) in {_evaluation['seconds']:.0f} s, diagnostics included. So each takes at most {_net:.2f} s. |
            | PyMC posterior | The three reference runs of Section 4.1: {", ".join(_duration(_s) for _s in _pymc)} on one core, the short and the long run together. |
            """),
            note("The steps ran on different machines. Compare orders of magnitude, not exact ratios."),
        ])}),
    ])
    return


@app.cell(hide_code=True)
def _(mo, section):
    mo.vstack([
        section("Check", "Four checks. The first three test the network's posterior. The last one tests the model "
                "itself.", stage=5),
        mo.md(r"""
        | Check | Question | What to look for |
        |:---|:---|:---|
        | 5.1 Posterior predictive | Can the inferred parameters regenerate the quotes? | quotes inside the replicated range |
        | 5.2 Calibration across simulations | Does a 90% interval contain the truth 90% of the time? | coverage near nominal, uniform ranks |
        | 5.3 Exact reference | Does the network reproduce the posterior of one data set? | agreement with PyMC |
        | 5.4 Real SPX quotes | Can Heston describe the market at all? | residuals within the assumed noise |

        ### 5.1 Posterior predictive

        Draw parameters from the posterior, price the 63 contracts, and add fresh noise with each draw's own $c$.

        $$\boldsymbol\theta^{(m)}\sim q(\boldsymbol\theta\mid\mathbf y),\qquad \widetilde y^{(m)}_i=P_i(\boldsymbol\theta^{(m)})+c^{(m)}h_i\,\epsilon^{(m)}_i$$

        If the model and the posterior are right, the observed quotes look like one more replicated data set.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(EXPIRY_DATES, EXPIRY_DAYS, controls, mo):
    expiry_choice = mo.ui.dropdown(options={f"{_date} ({_days} days)": _e for _e, (_date, _days) in enumerate(zip(EXPIRY_DATES, EXPIRY_DAYS))},
                                   value=f"{EXPIRY_DATES[4]} ({EXPIRY_DAYS[4]} days)", label="Expiry")
    controls(expiry_choice, hint="Choose an expiry. The data set is the one chosen in Section 4.2.")
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
    caption,
    case_choice,
    expiry_choice,
    grid,
    implied_vols,
    key,
    mo,
    neural_draws,
    note,
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
        if len(_inside) < len(_all):
            _note = (f"Network draws outside the prior range ({len(_all) - len(_inside)} of {len(_all):,}) are left "
                     "out here, because the pricer is checked on the prior's range.")
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
    _verdict = ("**The quotes look like one more replicated data set, as they should when the model is right.** The "
                f"posterior-mean prices miss them by {_misfit:.1f} half-spreads (root mean square), in line with the "
                f"inferred noise scale $c\\approx{_typical_c:.2f}$."
                if _misfit < 2.5 * _typical_c else
                "**The model cannot reproduce these quotes.** The posterior-mean prices miss them by "
                f"{_misfit:.1f} half-spreads (root mean square), far more than the inferred noise scale "
                f"$c\\approx{_typical_c:.2f}$ allows, and the miss has a systematic pattern across strikes "
                "(Section 5.4).")
    _columns = {
        "Top row": "Implied volatilities. The bands are so narrow that they hide behind the points.",
        "Bottom row": "The same, minus the observed quotes, in half-spreads. Each quote carries its own noise, so "
                      "the model prices (purple) differ from it by about $c$ half-spreads. The replicated quotes "
                      "(green) add fresh noise and should cover the zero line.",
    }
    if _case in CLEAN:
        _columns["Dashed line"] = ("The true, noise-free prices. The purple band follows them closely. The posterior "
                                   "recovers the true prices more precisely than any single quote measures them.")
    _view = [
        _fig,
        caption(f"{CASE_TITLES[_case]}, one expiry. Bands are pointwise 90% ranges over 40 posterior draws."),
        key(_verdict),
        grid(_columns, kind="reading"),
    ]
    if _note:
        _view.append(note(_note))
    mo.vstack(_view)
    return


@app.cell(hide_code=True)
def _(grid, mo):
    mo.vstack([
        mo.md(r"""
        ### 5.2 Calibration across 512 simulated data sets

        One posterior can look plausible and still be miscalibrated. Data sets with a known truth test the uncertainty statements themselves, in two ways.
        """),
        grid({
            "Coverage": "A 90% interval should contain the truth in about 90% of the data sets. Much more means the "
                        "intervals are too wide, much less that they are too narrow.",
            "Simulation-based calibration (SBC)": "If the posterior is correct, the truth is one more posterior "
                                                  "draw. Its **rank** among the draws, the fraction of draws below "
                                                  "it, is then uniform between 0 and 1. We plot the empirical CDF "
                                                  "of the ranks minus the uniform CDF.",
        }),
    ])
    return


@app.cell(hide_code=True)
def _(np):
    import functools as _functools

    from scipy.stats import binom as _binom

    @_functools.lru_cache(maxsize=4)
    def _extremes(n, simulations, seed):
        """For simulated samples of n uniform ranks: each curve's most extreme point, as a two-sided binomial
        tail probability. It does not depend on the band's level, so every level reuses it."""
        z = np.linspace(0, 1, min(n, 1000) + 2)[1:-1]
        u = np.random.default_rng(seed).uniform(size=(simulations, n))
        first = np.searchsorted(z, u)  # a rank is at or below z[k] exactly when first <= k
        rows = np.arange(simulations)[:, None] * (z.size + 1)
        counts = np.bincount((rows + first).ravel(), minlength=simulations * (z.size + 1))
        counts = counts.reshape(simulations, z.size + 1).cumsum(axis=1)[:, :z.size]
        tail = np.minimum(_binom.cdf(counts, n, z), _binom.sf(counts - 1, n, z))
        return z, 2 * tail.min(axis=1)

    def ecdf_band(n, level=0.95, simulations=10_000, seed=20261112):
        """Simultaneous band for "ECDF minus uniform" of n ranks (Säilynoja, Bürkner and Vehtari 2022, Section 2.2),
        the band BayesFlow draws: the whole curve of n uniform ranks stays inside it with probability `level`.
        It is narrow at 0 and 1, where a correct curve varies least, and widest at 0.5."""
        z, extremes = _extremes(n, simulations, seed)
        gamma = np.quantile(extremes, 1 - level)
        return z, _binom.ppf(gamma / 2, n, z) / n - z, _binom.ppf(1 - gamma / 2, n, z) / n - z

    return (ecdf_band,)


@app.cell(hide_code=True)
def _(caption, ecdf_band, grid, mo, norm, np, plots, plt):
    _rng = np.random.default_rng(7)
    _n = 512
    _truth = _rng.standard_normal(_n)
    _data = _truth + 0.5 * _rng.standard_normal(_n)
    _mean, _sd = 0.8 * _data, np.sqrt(0.2)  # exact posterior for a N(0, 1) prior and N(truth, 0.5^2) data
    _cases = {"calibrated": (0.0, 1.0), "biased upwards": (0.6, 1.0), "too narrow (overconfident)": (0.0, 0.6),
              "too wide (cautious)": (0.0, 1.7)}
    _z, _low, _high = ecdf_band(_n)  # one curve per panel: a 95% band for each
    _fig, _axes = plt.subplots(1, 4, figsize=(12.4, 2.6), sharey=True, constrained_layout=True)
    for _ax, (_title, (_shift, _scale)) in zip(_axes, _cases.items()):
        _ranks = norm.cdf((_truth - _mean - _shift * _sd) / (_scale * _sd))
        _curve = np.searchsorted(np.sort(_ranks), _z, side="right") / _n - _z
        _ax.fill_between(_z, _low, _high, color=plots.BASELINE, alpha=.45, lw=0)
        _ax.axhline(0, color=plots.INK, lw=.6)
        _ax.plot(_z, _curve, color=plots.SERIES[0] if _title == "calibrated" else plots.CRITICAL, lw=1.5)
        _ax.set(title=_title.capitalize(), xlabel="rank of the truth")
    _axes[0].set_ylabel("ECDF − uniform")
    mo.vstack([
        _fig,
        caption("Four synthetic examples of the rank plot, each from 512 data sets. A calibrated curve stays inside "
                "the grey band 95% of the time."),
        grid({
            "Calibrated": "The curve stays inside the grey band.",
            "Biased upwards": "Posterior draws lie above the truth, so ranks pile up near 0 and the curve bulges "
                              "above zero. A downward bias gives the mirror image.",
            "Too narrow": "The truth often falls in the posterior's tails, so ranks pile up near 0 and 1. The curve "
                          "is above zero on the left and below on the right.",
            "Too wide": "The truth sits near the middle of the posterior, so ranks pile up around 0.5. The curve is "
                        "below zero on the left and above on the right.",
        }, kind="reading"),
    ])
    return


@app.cell(hide_code=True)
def _(grid, mo):
    ecdf_guide = mo.accordion({"How to read the calibration (ECDF) plots": grid({
        "The question": """
        When the network gives a 90% interval, does it contain the truth 90% of the time? In other words, is the width of its posterior right? We can only check this where the truth is known, on simulated data.
        """,
        "One check, step by step": r"""
        For one parameter, say $v_0$:

        1. Draw true parameters from the prior and simulate 63 quotes from them.
        2. Ask the network for 200 posterior draws of $v_0$ given those quotes.
        3. Count the draws below the true $v_0$, say 140 of 200. The **rank** of the truth is $140/200 = 0.70$.
        """,
        "Why the rank should be uniform": """
        If the network's posterior is right, the true value is just one more draw from it. Nothing sets it apart from the 200 draws, so it is equally likely to land anywhere among them. Its rank is below 0.1 one time in ten, below 0.5 half the time, and so on.
        """,
        "The plot": r"""
        Repeat the check on many data sets (100 in the live check, 512 in Section 5.2) and collect the ranks. At each point $z$ between 0 and 1, the curve shows

        $$(\text{share of ranks below } z) - z$$

        If the ranks are uniform, the share below $z$ is close to $z$, and the curve stays near zero.
        """,
        "The grey band": """
        The band shows how far chance alone moves the curve of a correct network with this many data sets. A correct network stays inside it 95% of the time.

        - In the live check and after the resampling button, each of the six panels has its own band, so one curve may touch its band by chance.
        - In the saved results of Section 5.2, one band covers all six panels together.
        """,
        "Reading the shapes": """
        | Shape of the curve | What it means |
        |:---|:---|
        | inside the band | The uncertainty is right. |
        | up on the left, down on the right | Ranks pile up near 0 and 1, so the truth often falls outside the bulk of the draws. The posterior is **too narrow** (overconfident). |
        | down on the left, up on the right | Ranks pile up around 0.5, so the truth sits in the middle too often. The posterior is **too wide** (cautious). |
        | an arch above zero | Ranks pile up near 0, so most draws lie above the truth. The posterior is **biased upwards**. An arch below zero means biased downwards. |
        """,
        "What it cannot tell you": """
        - **Whether the posterior is sharp.** A network that ignored the quotes and returned the prior would pass. The recovery plot and the exact reference (Section 5.3) check sharpness.
        - **Whether the model is right.** The test data come from the same simulator. Sections 5.4 and 6 test the model on the SPX quotes.
        """,
    }, columns=2)})
    ecdf_guide
    return (ecdf_guide,)


@app.cell(hide_code=True)
def _(
    BOUNDS,
    EVALUATION,
    LABELS,
    MATH_LABELS,
    NAMES,
    caption,
    ecdf_band,
    grid,
    json,
    key,
    mo,
    np,
    plots,
    plt,
    table,
):
    _population = json.loads((EVALUATION / "evaluation.json").read_text())["population"]
    _ranks = np.asarray(_population["fractional_ranks"])
    # One band for all six curves together: each panel gets level 1 - 0.05/6, so all six stay inside with 95%.
    _z, _low, _high = ecdf_band(len(_ranks), level=1 - 0.05 / len(NAMES))
    _curves = [np.searchsorted(np.sort(_ranks[:, _j]), _z, side="right") / len(_ranks) - _z for _j in range(len(NAMES))]
    _passed = [bool(np.all((_c >= _low) & (_c <= _high))) for _c in _curves]
    _fig, _axes = plt.subplots(2, 3, figsize=(10.4, 4.9), sharex=True, sharey=True, constrained_layout=True)
    for _j, _ax in enumerate(_axes.flat):
        _ax.fill_between(_z, _low, _high, color=plots.BASELINE, alpha=.45, lw=0)
        _ax.axhline(0, color=plots.INK, lw=.6)
        _ax.plot(_z, _curves[_j], color=plots.SERIES[0] if _passed[_j] else plots.CRITICAL, lw=1.5)
        _ax.set_title(LABELS[_j] + ("" if _passed[_j] else "  (leaves the band)"), fontsize=10)
        if _j >= 3:
            _ax.set_xlabel("rank of the truth")
    _axes[0, 0].set_ylabel("ECDF − uniform")
    _axes[1, 0].set_ylabel("ECDF − uniform")

    _coverage = _population["coverage"]
    _support = _population["support"]
    _rows = []
    for _j, _label in enumerate(MATH_LABELS):
        _ci = _coverage["0.9"]["wilson"][_j]
        _rows.append({
            "Parameter": _label,
            "90% coverage [95% CI]": f"{100 * _coverage['0.9']['coverage'][_j]:.1f}% [{100 * _ci[0]:.1f}, {100 * _ci[1]:.1f}]",
            "50 / 80 / 95% coverage": " / ".join(f"{100 * _coverage[_level]['coverage'][_j]:.0f}%" for _level in ("0.5", "0.8", "0.95")),
            "Rank plot": "within band" if _passed[_j] else "leaves band",
            "Draws outside prior": f"{100 * _support['outside_prior']['per_dimension_count'][_j] / _support['samples']:.2f}%",
        })
    # Everything below is read off the saved evaluation.
    _cis = _coverage["0.9"]["wilson"]
    _good = [_j for _j in range(len(NAMES)) if _passed[_j] and _cis[_j][0] <= 0.9 <= _cis[_j][1]]
    _bad = [_j for _j in range(len(NAMES)) if _j not in _good]
    _lesson = {}
    for _j in _bad:
        _diff = lambda _x: np.mean(_ranks[:, _j] <= _x) - _x
        _low, _high = _diff(0.25), _diff(0.75)
        _shape = ("the too-wide shape, below zero on the left and above zero on the right" if _low < 0 < _high else
                  "the too-narrow shape, above zero on the left and below zero on the right" if _high < 0 < _low else
                  "the shape of a bias, on one side of zero throughout")
        _cov = _coverage["0.9"]["coverage"][_j]
        _text = (f"Its 90% intervals contain the truth {100 * _cov:.1f}% of the time, and its rank curve has "
                 f"{_shape}.")
        if _cov > 0.9 and _low < 0 < _high:
            _text += " The network is cautious about this parameter rather than overconfident."
        if NAMES[_j] == "initial_variance":
            _text += ("\n\n<small>A plausible explanation is precision. The shortest-dated quotes determine the initial "
                      f"variance to an average posterior SD of {_population['mean_posterior_sd'][_j]:.5f} on a prior "
                      f"range of {BOUNDS[_j, 1] - BOUNDS[_j, 0]:.2f}, which may be finer than the network "
                      "resolves.</small>")
        _lesson[f"The exception: {MATH_LABELS[_j]}"] = _text
    _counts = np.asarray(_support["outside_prior"]["per_dimension_count"])
    _top = [LABELS[_k] for _k in np.argsort(_counts)[::-1][:2]]
    _lesson["Draws outside the prior"] = (
        f"{100 * _support['outside_prior']['sample_fraction']:.1f}% of raw draws fall slightly outside the prior "
        f"range, most often in {_top[0]} and {_top[1]}. They are kept in every check.")
    mo.vstack([
        mo.md("#### Results for the pretrained estimator\n\n512 simulated data sets with 512 posterior draws each, "
              "from the estimator's recorded evaluation."),
        table(_rows, align="lrrlr"),
        _fig,
        caption("Rank plots for the six parameters. The grey band holds for all six together: a correct network "
                "keeps all six curves inside it 95% of the time. The band narrows at 0 and 1, where a correct curve "
                "varies least."),
        key(f"**{len(_good)} of the {len(NAMES)} parameters are well calibrated.** The 95% confidence interval of "
            "their 90% coverage includes 90%, and their rank curves stay inside the band."),
        grid(_lesson, columns=max(2, len(_lesson))),
    ])
    return


@app.cell(hide_code=True)
def _(EVALUATION, np):
    with np.load(EVALUATION / "recovery.npz", allow_pickle=False) as _archive:
        recovery_truth = _archive["truth"].astype(float)       # (512, 6) true parameters
        recovery_q = _archive["quantiles"].astype(float)       # (3, 512, 6): 5%, 50% and 95% posterior quantiles
    return recovery_q, recovery_truth


@app.cell(hide_code=True)
def _(TEXT_LABELS, controls, mo):
    recovery_parameter = mo.ui.dropdown(options={_label: _j for _j, _label in enumerate(TEXT_LABELS)},
                                        value=TEXT_LABELS[-1], label="Parameter")
    mo.vstack([
        mo.md("#### Recovery, one data set at a time\n\n"
              "Each point is one of the 512 simulated data sets: its true value against the posterior median, with "
              "the 90% interval."),
        controls(recovery_parameter, hint="Choose a parameter. Then drag a box around points in the plot to see the "
                                          "posterior of one of them."),
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
    MATH_LABELS,
    NAMES,
    bf,
    caption,
    estimator,
    held_out_quotes,
    mo,
    note,
    np,
    recovery_parameter,
    recovery_picker,
    recovery_q,
    recovery_truth,
    table,
):
    _j = recovery_parameter.value
    _x, _y = recovery_truth[:, _j], recovery_q[1, :, _j]
    _surprise = np.abs(_y - _x) / (recovery_q[2, :, _j] - recovery_q[0, :, _j])   # error in units of the 90% width
    _selected = np.flatnonzero(recovery_picker.value.get_mask(_x, _y)) if recovery_picker.value else np.array([], int)
    _pool = _selected if len(_selected) else np.arange(len(_x))
    _case = int(_pool[np.argmax(_surprise[_pool])])
    _how = (f"{len(_selected)} data sets are selected. Shown is the one whose median is farthest from the truth, "
            "relative to its interval." if len(_selected) else
            "Nothing is selected yet. Shown is the data set whose median is farthest from the truth, relative to "
            "its interval.")
    _g = lambda _x: f"{_x:.5g}".replace("-", "−")
    _rows = [{"Parameter": MATH_LABELS[_k], "Truth": _g(recovery_truth[_case, _k]),
              "Posterior median [90% interval]": f"{_g(recovery_q[1, _case, _k])} [{_g(recovery_q[0, _case, _k])}, "
                                                 f"{_g(recovery_q[2, _case, _k])}]",
              "Truth inside": "yes" if recovery_q[0, _case, _k] <= recovery_truth[_case, _k] <= recovery_q[2, _case, _k] else "no"}
             for _k in range(len(NAMES))]
    _parts = [mo.md(f"**Data set {_case}**, from the recorded evaluation. {_how}"),
              table(_rows, align="lrrl")]
    if estimator is not None and held_out_quotes is not None:
        _draws = np.asarray(estimator.sample(held_out_quotes[_case][None], 2000, seed=EVAL_SEED + 5000 + _case))[0]
        _grid = bf.diagnostics.pairs_posterior(_draws, targets=recovery_truth[_case], variable_names=LABELS, height=1.45,
                                               label_fontsize=10, tick_fontsize=7, legend_fontsize=9)
        _parts += [getattr(_grid, "figure", None) or _grid.fig,
                   caption("The posterior of this data set, from 2,000 fresh draws, plotted with "
                           "`bf.diagnostics.pairs_posterior`. The true values are marked in red. Shapes differ from "
                           "one data set to the next: compare cases near the edges of the prior with cases in its "
                           "middle.")]
    else:
        _parts.append(note("The posterior pair plot of the selected data set needs the pretrained estimator "
                           "(`HESTON_BUNDLE`)."))
    mo.vstack(_parts)
    return


@app.cell(hide_code=True)
def _(controls, mo):
    panel_button = mo.ui.run_button(label="Resample the 512 data sets with the pretrained estimator (a few minutes)")
    mo.vstack([
        mo.md("#### Reproduce the results\n\n"
              "The tables and plots of this section come from the estimator's recorded evaluation. The button draws "
              "512 × 512 fresh posterior samples and recomputes the diagnostics with `bf.diagnostics`."),
        controls(panel_button, hint="The pretrained estimator must be loaded."),
    ])
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
    caption,
    delivery,
    estimator,
    mo,
    np,
    panel_button,
    plots,
    time,
):
    mo.stop(not panel_button.value)
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
        caption(f"{len(_quotes)} data sets × {PANEL_DRAWS} posterior draws in {time.perf_counter() - _started:.0f} s. "
                "From top to bottom: rank plots, recovery with 90% intervals, and coverage at every level."),
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
    MATH_LABELS,
    NAMES,
    caption,
    grid,
    key,
    mo,
    neural_draws,
    note,
    np,
    reference_summary,
    table,
):
    _heading = mo.md(r"""
    ### 5.3 Comparison with the exact reference

    Calibration across many data sets does not show how sharp a single posterior is. The PyMC reference does, for the three data sets of Section 4.1.
    """)
    mo.stop(not neural_draws, mo.vstack([_heading, note("This comparison needs the pretrained estimator.")]))
    _columns = {"synthetic-market-like": "Market-like", "synthetic-broad": "Stressed", "market": "SPX"}
    _rows, _gaps, _ratios, _outside = [], {}, {}, {}
    for _j, _name in enumerate(NAMES):
        _row = {"Parameter": MATH_LABELS[_j]}
        for _case_id in _columns:
            _draws = neural_draws[_case_id][:, _j]
            _ref = reference_summary["cases"][_case_id]["parameters"][_name]
            _rq = _ref["regenerated_q05_q50_q95"]
            _q = np.quantile(_draws, [.05, .95])
            _gap = (_draws.mean() - _ref["regenerated_mean"]) / _ref["regenerated_sd"]
            _ratio = (_q[1] - _q[0]) / (_rq[2] - _rq[0])
            _gaps.setdefault(_case_id, []).append(abs(_gap))
            _ratios.setdefault(_case_id, []).append(_ratio)
            _row[f"{_columns[_case_id]}: centre gap"] = f"{_gap:+.2f}".replace("-", "−")
            _row[f"{_columns[_case_id]}: width ratio"] = f"{_ratio:.2f}"
        _rows.append(_row)
    for _case_id in CASES:
        _all = neural_draws[_case_id]
        _outside[_case_id] = np.mean(np.any((_all < BOUNDS[:, 0]) | (_all > BOUNDS[:, 1]), axis=1))
    _synthetic = ("synthetic-market-like", "synthetic-broad")
    _all_gaps = np.concatenate([_gaps[_c] for _c in _synthetic])
    _all_ratios = np.concatenate([_ratios[_c] for _c in _synthetic])
    _agree = int(np.sum((_all_gaps < 0.3) & (np.abs(_all_ratios - 1) < 0.25)))
    _wider, _narrower = int(np.sum(_all_ratios > 1.25)), int(np.sum(_all_ratios < 1))
    _cautious = (" The network is cautious here. It does not claim more precision than the data support, but "
                     "it does not extract all the information in the quotes either." if _narrower <= 2 else "")
    mo.vstack([
        _heading,
        table(_rows, align="l" + "r" * (len(_rows[0]) - 1)),
        caption("Centre gap: the network's mean minus the reference mean, in reference standard deviations. Width "
                "ratio: the network's 90% interval divided by the reference's. Perfect agreement is a gap of 0 and a "
                "ratio of 1. We count gaps below 0.3 and ratios between 0.75 and 1.25 as agreement."),
        key(f"**On the simulated data sets, {_agree} of 12 parameter posteriors agree with the reference on both "
            f"counts.** The median centre gap is {np.median(_all_gaps):.2f} SD, and the network's intervals are up to "
            f"{_all_ratios.max():.1f} times as wide as the reference's."),
        grid({
            "Simulated data sets":
                f"The largest centre gap is {_all_gaps.max():.2f} SD. {_wider} of 12 intervals are more than 25% "
                f"wider than the reference's, and {_narrower} are narrower.{_cautious}",
            "Calibrated is not the same as sharp":
                "This fits the good calibration of Section 5.2. Calibration is necessary, not sufficient: a posterior "
                "equal to the prior is perfectly calibrated and useless. Only an exact reference shows how much "
                "sharpness is lost.",
            "SPX quotes":
                f"The network departs far from the reference, and {100 * _outside['market']:.0f}% of its draws leave "
                "the prior range (Section 5.4).",
        }),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.4 Real SPX quotes: does Heston fit the market?

    All checks so far used data simulated from the model itself. On the **SPX quotes of 9 November 2026** the first question is whether any Heston parameters reproduce the market within the assumed noise.
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
    caption,
    grid,
    implied_vols,
    key,
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
    # The best Heston fit: prices at the posterior-mean parameters, and its misses in half-spreads.
    market_fit = pilot.prices(_reference.mean(axis=0))
    market_residual = (_y - market_fit) / H
    _rms = float(np.sqrt(np.mean(market_residual ** 2)))
    _network_text = ("" if _neural is None else
                     f" Its posterior is shifted and much wider than the reference, and {100 * np.mean(_neural[:, -1] > 2):.0f}% "
                     "of its draws have $c>2$, outside the prior range.")
    mo.vstack([
        _fig,
        caption("Bands: implied volatilities of the model prices from 40 posterior draws. Points: SPX mid quotes. "
                "Last panel: the posterior of the noise multiplier $c$."),
        key(f"""
        **No Heston smile fits the SPX quotes within the assumed noise.** The best fit, at the posterior-mean parameters, misses them by **{_rms:.1f} half-spreads** (root mean square). The prior caps the noise SD at two half-spreads ($c\\le2$).
        """),
        grid({
            "What the PyMC posterior says":
                "PyMC gives the exact answer for our model and prior. $c$ piles up at its bound (last panel), and "
                "the Heston parameters settle on the best compromise. Sections 6.2 to 6.5 show where Heston fails, "
                "why, and what can be done about it.",
            "Why the network disagrees with PyMC":
                "The network was trained on quotes that scatter around a Heston smile with a noise SD of at most two "
                f"half-spreads. These quotes are {_rms:.0f} half-spreads away on average, outside anything it saw, so "
                f"it extrapolates.{_network_text}",
        }),
        key("""
        **This is model misspecification, not a training failure.** An amortized estimator should only be trusted on data its simulator could have produced.
        """),
    ])
    return market_fit, market_residual


@app.cell(hide_code=True)
def _(grid, mo, section):
    mo.vstack([
        section("Use", "A desk holds products that nobody quotes, and must still value them every day. A posterior "
                "gives each value a range.", stage=6),
        mo.md(r"""
        ### 6.1 Pricing a product the market does not quote

        The example is a **forward-start call**. Bought today, it fixes its strike in 3 months at the index level on that day. At 6 months it pays the percentage rise above that level, if positive.

        $$\max\!\left(\frac{S_{6\mathrm{m}}}{S_{3\mathrm{m}}}-1,\;0\right)\quad\text{per unit of notional}$$
        """),
        grid({
            "Why a model": "The value depends on how volatile the index will be between month 3 and month 6, which "
                           "today's quotes do not state directly.",
            "The recipe": "Fit a model to the quoted options, then price the product with it. One fit gives one "
                          "price. A posterior gives a range, by pricing the product under each posterior draw.",
            "What the range is for": "Judging how reliable the value is, holding a reserve against being wrong, and "
                                     "setting the bid–ask.",
        }),
    ])
    return


@app.cell(hide_code=True)
def _(
    BOUNDS,
    CASES,
    DELTA,
    EXPIRY,
    EXPIRY_DAYS,
    H,
    TRUTHS,
    caption,
    grid,
    key,
    mo,
    neural_draws,
    np,
    pilot,
    plots,
    plt,
    reference_draws,
):
    # Price a forward-start call and a volatility swap under posterior draws, and under Heston refitted to SPX subsets.
    from scipy.optimize import least_squares as _least_squares

    from sbi4finance.calibration import products as _products

    # Monte Carlo prices under one parameter draw, with the same random numbers for every draw, so that differences
    # between draws are due to the parameters, not to Monte Carlo noise.
    def _price(_theta):
        _v0, _kappa, _v_inf, _xi, _rho = _theta[:5]
        _paths = _products.heston_paths(dict(kappa=_kappa, theta=_v_inf, vol_of_vol=_xi), _v0, 0.5, 252, 4096,
                                        seed=20261115)
        return (100 * _products.forward_start_call(_paths, _rho, 0.25, 0.5, 1.0)[0],  # % of notional
                100 * _products.volatility_swap(_paths, 0.25)[0])  # 3-month volatility swap, % per year

    def _posterior(_draws, _seed, _count=40):
        _inside = _draws[np.all((_draws >= BOUNDS[:, 0]) & (_draws <= BOUNDS[:, 1]), axis=1)]
        if len(_inside) < 20:
            return None
        _pick = np.random.default_rng(_seed).choice(len(_inside), min(_count, len(_inside)), replace=False)
        return np.array([_price(_t) for _t in _inside[_pick]])

    _sets = {}
    for _k, _case in enumerate(("synthetic-market-like", "market")):
        _sets[_case] = {"PyMC": _posterior(reference_draws[_case], [20261115, _k, 0])}
        _network_prices = _posterior(neural_draws[_case], [20261115, _k, 1]) if _case in neural_draws else None
        if _network_prices is not None:
            _sets[_case]["network"] = _network_prices
    _truth = np.array(_price(TRUTHS["synthetic-market-like"]))

    # Refit Heston to four subsets of the SPX quotes, by least squares in half-spreads, and price the same products.
    _subsets = {
        "all 63 quotes": np.ones(len(H), bool),
        f"the 3 shortest expiries ({EXPIRY_DAYS[0]}–{EXPIRY_DAYS[2]} d)": EXPIRY < 3,
        f"the 4 longest expiries ({EXPIRY_DAYS[3]}–{EXPIRY_DAYS[-1]} d)": EXPIRY >= 3,
        "near the money, |δ| ≤ 1.3": np.abs(DELTA) <= 1.3,
    }
    _start = reference_draws["market"].mean(axis=0)
    _refits = {}
    for _name, _mask in _subsets.items():
        def _misfit(_p, _mask=_mask):
            try:
                return ((CASES["market"] - pilot.prices(np.r_[_p, 1.0])) / H)[_mask]
            except (FloatingPointError, ValueError):
                return np.full(int(_mask.sum()), 1e3)
        _fit = _least_squares(_misfit, _start[:5], bounds=(BOUNDS[:5, 0], BOUNDS[:5, 1]), x_scale="jac", max_nfev=400)
        _refits[_name] = (_fit.x, np.array(_price(_fit.x)))

    _colors = {"PyMC": plots.SERIES[1], "network": plots.SERIES[0]}
    _fig, _axes = plt.subplots(1, 2, figsize=(11.8, 3.3), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.35]})
    for _ax, _case in zip(_axes, ("synthetic-market-like", "market")):
        _labels = []
        for _source, _values in _sets[_case].items():
            _q = np.quantile(_values[:, 0], [.05, .95])
            _ax.plot(_q, [len(_labels)] * 2, color=_colors[_source], lw=7, solid_capstyle="butt", alpha=.75)
            _ax.scatter(_values[:, 0], np.full(len(_values), len(_labels)), color=_colors[_source], s=7, zorder=3)
            _labels.append(f"{_source} posterior, 90% range")
        if _case == "market":
            for _name, (_, _prices) in _refits.items():
                _ax.scatter([_prices[0]], [len(_labels)], color=plots.INK, marker="D", s=26, zorder=3)
                _labels.append(f"refit on {_name}")
        else:
            _ax.axvline(_truth[0], color=plots.INK, ls="--", lw=1)
            _ax.text(_truth[0], len(_labels) - 0.5, " true price", fontsize=7, color=plots.INK, va="center")
        _ax.set_yticks(range(len(_labels)), _labels, fontsize=8)
        _ax.set_ylim(len(_labels) - 0.4, -0.6)
        _ax.set_xlabel("forward-start call, % of notional")
    _axes[0].set_title("Simulated market-like quotes: Heston is right", fontsize=10)
    _axes[1].set_title("SPX quotes: the model, not its parameters", fontsize=10)

    # Everything said below is computed here.
    _like, _spx = (np.quantile(_sets[_c]["PyMC"], [.05, .95], axis=0) for _c in ("synthetic-market-like", "market"))
    _refit_prices = np.array([_p for _, _p in _refits.values()])
    _low, _high = _refit_prices.min(axis=0), _refit_prices.max(axis=0)
    _ratio = (_high - _low) / (_spx[1] - _spx[0])
    _kappas, _xis = [_x[1] for _x, _ in _refits.values()], [_x[3] for _x, _ in _refits.values()]
    _covered = [_like[0, _j] <= _truth[_j] <= _like[1, _j] for _j in range(2)]
    _network = {_c: np.quantile(_sets[_c]["network"][:, 0], [.05, .95]) for _c in _sets if "network" in _sets[_c]}
    _panels = {
        "Where Heston is right":
            f"On the market-like simulation the PyMC 90% range is {_like[0, 0]:.3f}–{_like[1, 0]:.3f}% of notional, "
            f"±{50 * (_like[1, 0] - _like[0, 0]) / np.mean(_like[:, 0]):.1f}% of the price. The true price, "
            f"{_truth[0]:.3f}%, lies {'inside' if _covered[0] else 'outside'} it."
            + (f" The network's range is {_network['synthetic-market-like'][0]:.3f}–"
               f"{_network['synthetic-market-like'][1]:.3f}%." if "synthetic-market-like" in _network else "")
            + (" So the use works where the model is right: the unquoted product gets a range that contains its "
               "true price." if _covered[0] else " One noisy data set can shift a posterior (Section 4.2)."),
        "On the SPX quotes":
            f"The posterior range is just as narrow, {_spx[0, 0]:.3f}–{_spx[1, 0]:.3f}%. But Heston does not fit these "
            "quotes, so its parameters depend on which quotes it is fitted to. Four reasonable choices give "
            f"{_low[0]:.3f}% to {_high[0]:.3f}%, about {_ratio[0]:.0f} times the posterior range. Across them "
            f"$\\kappa$ runs from {min(_kappas):.1f} to {max(_kappas):.1f}"
            + (" (its upper bound)" if np.isclose(max(_kappas), BOUNDS[1, 1]) else "")
            + f" and $\\sigma_v$ from {min(_xis):.2f} to {max(_xis):.2f}."
            + (f" The network's range, {_network['market'][0]:.3f}–{_network['market'][1]:.3f}%, differs from PyMC's "
               "because these quotes lie outside its training data (Section 5.4)." if "market" in _network else ""),
    }
    mo.vstack([
        _fig,
        caption("Bars: 90% ranges over 40 posterior draws, each draw a dot. Diamonds: Heston refitted to part of the "
                "SPX quotes. Prices are Monte Carlo with 4,096 paths and the same random numbers for every draw. The "
                f"left panel is zoomed in about {np.ptp(_axes[1].get_xlim()) / np.ptp(_axes[0].get_xlim()):.0f} "
                "times more than the right one."),
        grid(_panels, kind="reading", widths=(1, 1.35)),
        key("""
        **A posterior measures uncertainty within the model.** On SPX the model is wrong. That error is many times larger, and the posterior cannot see it.
        """),
        grid({
            "The same for a volatility swap":
                f"For a 3-month volatility swap the posterior range is {_spx[0, 1]:.2f}–{_spx[1, 1]:.2f}% and the "
                f"refits give {_low[1]:.2f}–{_high[1]:.2f}%, about {_ratio[1]:.0f} times wider. On the simulation its "
                f"true strike, {_truth[1]:.2f}%, lies {'inside' if _covered[1] else 'outside'} the range "
                f"{_like[0, 1]:.2f}–{_like[1, 1]:.2f}%.",
            "What a desk would do":
                "Hold a reserve sized to the refit spread, add a model-error term, or use a better model. Sections "
                "6.2 to 6.5 show where Heston misses and what can be done.",
            "When the posterior range matters":
                "When the quotes leave the price open: few or wide quotes, or a model flexible enough that the "
                "quotes do not determine what the product depends on.",
        }),
    ])
    return


@app.cell(hide_code=True)
def _(
    BOUNDS,
    DELTA,
    EXPIRY,
    EXPIRY_DATES,
    EXPIRY_DAYS,
    H,
    MARKET_IV,
    NAMES,
    QUOTES,
    cards,
    grid,
    implied_vols,
    key,
    market_fit,
    market_residual,
    mo,
    np,
    plots,
    plt,
    reference_draws,
    table,
    vol_point_noise,
):
    _fit_iv = 100 * implied_vols(market_fit)[0]
    _misfit = 100 * MARKET_IV - _fit_iv  # SPX minus the best Heston fit, in volatility points
    _rows = [np.flatnonzero(EXPIRY == _e)[np.argsort(DELTA[EXPIRY == _e])] for _e in range(len(EXPIRY_DATES))]
    _grid = np.array([_misfit[_r] for _r in _rows])  # one row per expiry, deepest put first
    _days = np.asarray(EXPIRY_DAYS)
    _deep = np.array([_r[0] for _r in _rows])
    _atm = np.array([_r[np.argmin(np.abs(DELTA[_r]))] for _r in _rows])
    _skew_market = 100 * (MARKET_IV[_deep] - MARKET_IV[_atm])
    _skew_heston = _fit_iv[_deep] - _fit_iv[_atm]
    # Likelihood of c at the best fit: c scales every noise SD, so it peaks where c equals the RMS misfit.
    _n, _squares = market_residual.size, float(np.sum(market_residual ** 2))
    _c_hat = np.sqrt(_squares / _n)
    _c = np.linspace(0.05, max(20.0, 1.5 * _c_hat), 800)
    _likelihood = np.exp(-_n * np.log(_c / _c_hat) - 0.5 * _squares * (1 / _c ** 2 - 1 / _c_hat ** 2))

    _fig, _axes = plt.subplots(1, 3, figsize=(11.6, 3.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.5, 1, 1]})
    from matplotlib.colors import LinearSegmentedColormap as _Colormap

    _lim = float(np.max(np.abs(_grid)))
    _cmap = _Colormap.from_list("misfit", [plots.SERIES[0], "#f0efec", "#e34948"])  # blue below, red above the fit
    _image = _axes[0].imshow(_grid, cmap=_cmap, vmin=-_lim, vmax=_lim, aspect="auto")
    for (_i, _j), _value in np.ndenumerate(_grid):
        _axes[0].text(_j, _i, "0.0" if abs(_value) < 0.05 else f"{_value:+.1f}".replace("-", "−"),
                      ha="center", va="center", fontsize=6.5, color=plots.INK)
    _targets = [sorted({QUOTES[_r[_j]]["target_z"] for _r in _rows}, reverse=True) for _j in range(_grid.shape[1])]
    _axes[0].set_xticks(range(_grid.shape[1]), ["/".join(f"{_t:g}" for _t in _ts) for _ts in _targets])
    _axes[0].set_yticks(range(len(_days)), [f"{_d} d" for _d in _days])
    _axes[0].grid(False)
    _axes[0].set(xlabel="standardized moneyness δ (target)", ylabel="expiry",
                 title="SPX minus best Heston fit, volatility points")
    _fig.colorbar(_image, ax=_axes[0], shrink=0.9)

    _axes[1].plot(_days, _skew_market, "o-", color="#222222", lw=1.3, ms=4, label="SPX market")
    _axes[1].plot(_days, _skew_heston, "o-", color=plots.SERIES[1], lw=1.3, ms=4, label="best Heston fit")
    _axes[1].set_xscale("log")
    _axes[1].set_xticks(_days, [str(_d) for _d in _days], fontsize=7)
    _axes[1].minorticks_off()
    _axes[1].set(xlabel="days to expiry", ylabel="implied-volatility points", title="Put wing: deepest put minus at the money")
    _axes[1].legend(fontsize=7)

    _low, _high = BOUNDS[-1]
    _axes[2].axvspan(_low, _high, color=plots.GRID, lw=0)
    _axes[2].text(_high + 0.4, 0.97, "prior range", fontsize=7, color=plots.MUTED, va="top")
    _axes[2].plot(_c, _likelihood, color=plots.INK, lw=1.3)
    _axes[2].axvline(_c_hat, color=plots.MUTED, ls="--", lw=1)
    _axes[2].text(_c_hat + 0.4, 1.02, f"c = {_c_hat:.1f}", fontsize=7, color=plots.MUTED)
    _axes[2].set(xlim=(0, _c[-1]), ylim=(0, 1.12), xlabel="noise multiplier c", ylabel="likelihood, scaled to its maximum",
                 title="Likelihood of c at the best fit")

    _vol_of_vol = float(reference_draws["market"].mean(axis=0)[NAMES.index("vol_of_vol")])
    _points = vol_point_noise(H, MARKET_IV)
    _wing = DELTA < -3
    _share = float(np.sum(market_residual[_wing] ** 2) / _squares)
    _rest = _misfit[~_wing]
    _gap = _skew_market - _skew_heston
    _under = int(np.sum(_grid[:, 0] > 0))
    _where = "every expiry" if _under == len(_days) else f"{_under} of {len(_days)} expiries"
    _rms = [float(np.sqrt(np.mean(market_residual[_r] ** 2))) for _r in _rows]
    _number = lambda _x, _digits: f"{_x:.{_digits}f}".replace("-", "−")
    _table = [{"Expiry": f"{_date} ({_days[_e]} d)",
               "RMS misfit (half-spreads)": _number(_rms[_e], 1),
               "Largest misfit (half-spreads)": _number(market_residual[_r][np.argmax(np.abs(market_residual[_r]))], 1),
               "At moneyness $\\delta$": _number(DELTA[_r][np.argmax(np.abs(market_residual[_r]))], 2)}
              for _e, (_date, _r) in enumerate(zip(EXPIRY_DATES, _rows))]
    _theta = reference_draws["market"].mean(axis=0)  # the best Heston fit: the posterior-mean parameters
    mo.vstack([
        mo.md(r"""
        ### 6.2 Four signs of model misspecification

        The exact posterior shows all four signs, so they are not the network's doing.
        """),
        cards([
            ("Sign 1", "The noise multiplier sits at its bound",
             f"Its posterior median is {np.median(reference_draws['market'][:, -1]):.4f}. The bound is "
             f"{BOUNDS[-1, 1]:g}."),
            ("Sign 2", "Every expiry misses",
             f"The misfit per expiry is {min(_rms):.1f} to {max(_rms):.1f} half-spreads (root mean square)."),
            ("Sign 3", "The misses are systematic",
             f"The deepest put is underpriced at {_where}, by {_grid[:, 0].min():.1f} to {_grid[:, 0].max():.1f} "
             "volatility points."),
            ("Sign 4", "The fit needs extreme parameters",
             f"$\\sigma_v={_vol_of_vol:.2f}$, so $\\sigma_v^2={_vol_of_vol ** 2:.1f}$ against $2\\kappa v_\\infty="
             f"{2 * _theta[NAMES.index('kappa')] * _theta[NAMES.index('theta')]:.2f}$ (Feller condition, "
             "Section 1.2)."),
        ], columns=4),
        key("""
        **No Heston parameters and no better inference remove these signs.** The model lacks a feature of the data.
        """),
        mo.md(r"""
        ### 6.3 Where Heston misses

        Each panel compares the SPX quotes with the best Heston fit, the prices at the posterior-mean parameters.
        """),
        _fig,
        grid({
            "Largest in the deep put wing":
                f"The SPX quotes lie above the best Heston smile at the deepest put of {_where}: {_grid[0, 0]:.1f}, "
                f"{_grid[1, 0]:.1f} and {_grid[2, 0]:.1f} volatility points at {_days[0]}, {_days[1]} and {_days[2]} "
                f"days, {_grid[3:, 0].min():.1f}–{_grid[3:, 0].max():.1f} beyond. The {int(_wing.sum())} puts with "
                f"$\\delta<-3$ give {100 * _share:.0f}% of the squared misfit.",
            "A put wing too flat at short expiries":
                f"From the money to the deepest put, SPX implied volatility rises {_skew_market[0]:.1f} points at "
                f"{_days[0]} days, the best Heston smile only {_skew_heston[0]:.1f}. The shortfall is {_gap[0]:.1f}, "
                f"{_gap[1]:.1f} and {_gap[2]:.1f} points at {_days[0]}, {_days[1]} and {_days[2]} days, and "
                f"{_gap[3:].min():.1f}–{_gap[3:].max():.1f} beyond.",
            "Why $c$ sits at 2":
                f"A half-spread is only {np.nanmin(_points):.3f}–{np.nanmax(_points):.2f} volatility points, so these "
                f"misses amount to **{_c_hat:.1f} half-spreads** (root mean square). $c$ scales the noise of every "
                "quote by the same factor, so it cannot improve the fit, only measure it. The likelihood peaks at "
                f"$c={_c_hat:.1f}$, far outside the prior range.",
        }, kind="reading", widths=(1.5, 1, 1)),
        mo.md(f"Elsewhere the fit is a compromise. Across the rest of the smile it misses by up to "
              f"{np.max(np.abs(_rest)):.1f} points{', in both directions' if _rest.min() < 0 < _rest.max() else ''}."),
        mo.accordion({"Misfit by expiry, in half-spreads": table(_table, align="lrrr")}),
    ])
    return


@app.cell(hide_code=True)
def _(
    DELTA,
    EXPIRY,
    EXPIRY_DAYS,
    MARKET_IV,
    MATURITY,
    REF_VOL,
    atm_shape,
    caption,
    forward_unit_pricer,
    grid,
    implied_vols,
    key,
    market_fit,
    mo,
    np,
    plots,
    plt,
    reference_draws,
    surface,
):
    _theta = reference_draws["market"].mean(axis=0)  # the best Heston fit: the posterior-mean parameters
    _v0, _kappa, _v_inf, _xi, _rho, _c = _theta
    _days = np.asarray(EXPIRY_DAYS)
    _k = DELTA * REF_VOL * np.sqrt(MATURITY)  # log-moneyness ln(K/F)
    _fit_iv = implied_vols(market_fit)[0]
    _spx = np.array([atm_shape(_k[EXPIRY == _e], MARKET_IV[EXPIRY == _e]) for _e in range(len(_days))])
    _fit = np.array([atm_shape(_k[EXPIRY == _e], _fit_iv[EXPIRY == _e]) for _e in range(len(_days))])

    # The same fit at expiries the market does not quote: one day to one year, five strikes around the money.
    _grid_t = np.geomspace(1 / 365, 1.0, 36)
    _grid_d = np.array([-1.25, -0.5, 0.0, 0.5, 1.0])
    _tt, _dd = np.meshgrid(_grid_t, _grid_d, indexing="ij")
    _gk, _gt = (_dd * REF_VOL * np.sqrt(_tt)).ravel(), _tt.ravel()
    _gkind = np.where(_dd.ravel() <= 0, "put", "call")
    _grid_iv = surface.implied_vol(forward_unit_pricer(_gk, _gt, _gkind).prices(_theta)[None], _gk, _gt,
                                   _gkind)[0].reshape(_tt.shape)
    _curve = np.array([atm_shape(_grid_d * REF_VOL * np.sqrt(_t), _grid_iv[_i]) for _i, _t in enumerate(_grid_t)])
    _ceiling = abs(_rho * _xi / (4 * np.sqrt(_v0)))  # Heston's short-expiry skew limit (Lecture 10, Section 10.8)
    _power = lambda _x: np.polyfit(np.log(_days / 365), np.log(np.abs(_x)), 1)

    _fig, _axes = plt.subplots(1, 3, figsize=(11.8, 3.7), constrained_layout=True)
    _spx_style = dict(color="#222222", s=24, zorder=3, label="SPX quotes")
    _fit_style = dict(color=plots.SERIES[1], s=22, marker="s", zorder=3, label="best Heston fit")
    _b, _a = _power(_spx[:, 0])
    _axes[0].plot(365 * _grid_t, np.abs(_curve[:, 0]), color=plots.SERIES[1], lw=1.2, label="Heston, any expiry")
    _axes[0].axhline(_ceiling, color=plots.MUTED, ls=":", lw=1, label="Heston's ceiling")
    _axes[0].plot(_days, np.exp(_a) * (_days / 365) ** _b, color="#222222", ls="--", lw=1,
                  label=f"SPX power law, $T^{{{_b:.2f}}}$")
    _axes[0].scatter(_days, np.abs(_spx[:, 0]), **_spx_style)
    _axes[0].scatter(_days, np.abs(_fit[:, 0]), **_fit_style)
    _axes[0].set(xscale="log", yscale="log", xlabel="days to expiry", ylabel="|at-the-money skew|",
                 title="Skew: Heston levels off")
    from matplotlib.ticker import FuncFormatter as _Formatter

    _axes[0].yaxis.set_major_formatter(_Formatter(lambda _v, _p: f"{_v:g}"))
    _axes[0].yaxis.set_minor_formatter(_Formatter(lambda _v, _p: f"{_v:g}"))
    _axes[1].plot(365 * _grid_t, _curve[:, 1], color=plots.SERIES[1], lw=1.2, label="Heston, any expiry")
    _axes[1].axhline(0, color=plots.MUTED, lw=0.6)
    _axes[1].scatter(_days, _spx[:, 1], **_spx_style)
    _axes[1].scatter(_days, _fit[:, 1], **_fit_style)
    _axes[1].set(xscale="log", xlabel="days to expiry", ylabel="at-the-money curvature",
                 title="Curvature: Heston turns down")
    for _ax, _where in zip(_axes[:2], ("lower left", "lower right")):
        _ax.set_xticks([1, 7, 30, 90, 365], ["1", "7", "30", "90", "365"])
        _ax.legend(fontsize=7, loc=_where)
    _short = EXPIRY == 0
    _order = np.argsort(DELTA[_short])
    _axes[2].plot(DELTA[_short][_order], 100 * _fit_iv[_short][_order], "-s", color=plots.SERIES[1], ms=4, lw=1.2,
                  label="best Heston fit")
    _axes[2].scatter(DELTA[_short], 100 * MARKET_IV[_short], color="#222222", s=24, zorder=3, label="SPX quotes")
    _axes[2].set(xlabel="standardized moneyness δ", ylabel="implied volatility (%)",
                 title=f"The {_days[0]}-day smile")
    _axes[2].legend(fontsize=7)

    # Everything said below is computed here.
    _deep = np.array([100 * (MARKET_IV - _fit_iv)[EXPIRY == _e][np.argmin(DELTA[EXPIRY == _e])] for _e in range(len(_days))])
    _near = _short & (np.abs(DELTA) < 1.3)
    _peak = int(np.argmax(_fit[:, 1]))
    _minus = lambda _x: f"{_x:.2f}".replace("-", "−")
    _negative = 365 * _grid_t[_curve[:, 1] < 0]
    mo.vstack([
        mo.md(r"""
        ### 6.4 Which feature is missing

        The at-the-money skew and curvature of each smile show what Heston lacks at short expiries.
        """),
        _fig,
        caption("Skew and curvature are measured as in Lecture 5, by a quadratic in log-moneyness through the five "
                "quotes nearest the money. Right: the shortest smile."),
        grid({
            "Skew":
                "At the money, the fit follows the SPX skew at every expiry, within "
                f"{100 * np.max(np.abs(_fit[:, 0] / _spx[:, 0] - 1)):.0f}%, with nearly the same power law "
                f"({_minus(_power(_fit[:, 0])[0])} against {_minus(_b)}). But its {_days[0]}-day skew is already "
                f"{100 * abs(_fit[0, 0]) / _ceiling:.0f}% of Heston's ceiling "
                f"$|\\rho\\sigma_v|/(4\\sqrt{{v_0}})={_ceiling:.2f}$. Shorter expiries can add at most "
                f"{100 - 100 * abs(_fit[0, 0]) / _ceiling:.0f}% (feature 3, Section 1.2), while a power law keeps "
                "growing.",
            "Curvature":
                f"The SPX smile gets more curved as the expiry shortens: {_spx[2, 1]:.1f}, {_spx[1, 1]:.1f} and "
                f"{_spx[0, 1]:.1f} at {_days[2]}, {_days[1]} and {_days[0]} days. The fit's curvature peaks at "
                f"{_days[_peak]} days ({_fit[_peak, 1]:.1f}) and falls to {_fit[0, 1]:.1f} at {_days[0]} days."
                + (f" At shorter expiries it falls below zero (under {_negative.max():.0f} days)."
                   if _negative.size else ""),
            "The shortest smile":
                "Near the money the fit is within "
                f"{np.max(np.abs(100 * (MARKET_IV - _fit_iv)[_near])):.1f} volatility points. At the deepest put it "
                f"is {_deep[0]:.1f} points too low. With a single diffusive variance factor, the fit cannot price "
                f"that much crash risk into {_days[0]} days without breaking the rest of the smile.",
        }, kind="reading"),
        mo.md(r"""
        ### 6.5 What can be done within Heston

        Change the noise model: widen it, reshape it, or add an explicit model-error term. The posterior then admits the misfit instead of pushing $c$ against its bound.
        """),
        key("**The misfit itself stays.** Removing it needs a different model."),
    ])
    return


@app.cell(hide_code=True)
def _(cards, mo, section):
    mo.vstack([
        section("Takeaways", "Eight points, in the order of the notebook.", kicker="Summary"),
        cards([
            ("1", "SBI turns a pricer into a posterior",
             "Prior, pricer and noise model form a simulator. One training run gives an estimator that returns the "
             "posterior for any 63 quotes in seconds."),
            ("2", "Liquid quotes are very informative",
             "Spreads of a few hundredths to a few tenths of a volatility point determine the Heston parameters "
             "tightly and make model error stand out."),
            ("3", "Prepare the inputs",
             "Quotes in noise units and whitening put all 63 inputs on one uncorrelated scale."),
            ("4", "Check calibration over many simulations",
             "Here five of six parameters are well calibrated. The initial-variance intervals are too wide rather "
             "than too narrow."),
            ("5", "Check single data sets against an exact reference",
             "Good calibration does not mean sharp posteriors. The network's intervals can be several times wider "
             "than the exact ones."),
            ("6", "Check the model, not only the network",
             "On the SPX quotes the noise multiplier hits its bound and Heston misses by many half-spreads. Its "
             "smile is too flat in the deep put wing, most of all at short expiries. An amortized estimator is only "
             "reliable on data its simulator could have produced."),
            ("7", "A posterior prices an unquoted product with a range",
             "Where the model is right, the range covers the true price. On SPX the range is tiny, but refitting "
             "Heston to different quotes moves the price many times more. The model, not its parameters, dominates "
             "the uncertainty."),
            ("8", "Misspecification shows what the next model needs",
             "Heston follows the SPX skew at the money but not the short-dated curvature and put wing."),
        ], numbered=True),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Running this notebook": mo.md(r"""
    The pretrained estimator and its training data are distributed as a separate 465 MB archive. `lab/plans/training-data-guide.md` explains how to download and verify it. From `lab/`, start marimo with `HESTON_BUNDLE` set to the extracted folder (the one that contains `bundle.json`):

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
