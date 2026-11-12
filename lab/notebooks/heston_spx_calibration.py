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
def _(EVALUATION, json):
    # What was recorded when the pretrained estimator was trained and evaluated. The text quotes these records
    # instead of repeating their numbers.
    _run = EVALUATION.parent
    _contract = json.loads((_run / "protocol.json").read_text())["contract"]
    _optimizer = _contract["resolved_defaults"]["optimizer"]
    _rate = _optimizer["config"]["learning_rate"]["config"]
    _jobs = json.loads((_run.parent / "campaign.json").read_text())["jobs"]
    TRAINING = {
        "rows": _contract["train_rows"],
        "validation_rows": _contract["validation_rows"],
        "epochs": _contract["epochs"],
        "batch_size": _contract["batch_size"],
        "updates": _contract["expected_optimizer_updates"],
        "seed": _contract["seed"],
        "optimizer": _optimizer["class_name"],
        "weight_decay": _optimizer["config"]["weight_decay"],
        "clipnorm": _optimizer["config"]["clipnorm"],
        "rate_start": _rate["initial_learning_rate"],
        "rate_peak": _rate["warmup_target"],
        "warmup_updates": _rate["warmup_steps"],
        "whitening_rows": _contract["bank_contract"]["whitening_rows"],
        "fresh_noise": _contract["bank_contract"]["fresh_noise_per_pass"],
        "platform": _contract["platform"],
        "seconds": next(_job["seconds"] for _job in _jobs if _job["id"] == f"{_run.name}_training"),
    }
    EVALUATED = json.loads((EVALUATION / "evaluation.json").read_text())
    return EVALUATED, TRAINING


@app.cell(hide_code=True)
def _(TRAINING, mo):
    # The notebook's page elements. Each returns HTML with a class that heston_spx_calibration.css styles, so
    # the text below states content and the stylesheet decides how it looks.
    import html as _markup
    import math as _math
    import re as _re
    from inspect import cleandoc as _cleandoc
    from itertools import groupby as _groupby

    STAGES = (
        ("Model", "Specify Heston dynamics, a prior over six parameters and quote noise set by the bid–ask spread"),
        ("Simulate", f"Draw parameters, price the 63 contracts and add quote noise, {TRAINING['rows']:,} times"),
        ("Train", "Fit a flow-matching network to the posterior of the parameters given the quotes"),
        ("Infer", "Condition on observed quotes and draw from the posterior in seconds"),
        ("Check", "Test the posterior on simulated data and against PyMC, and the model on the SPX quotes"),
        ("Use", "Price a product the market does not quote, with a range instead of one number"),
    )

    # ---- text: section numbers become anchors and links

    def _anchor(number):
        """The id of a numbered section ("3") or subsection ("3.2")."""
        return "s" + str(number).replace(".", "-")

    _NUMBER = r"\d(?!\d)(?:\.\d(?!\d))?"
    _REFERENCE = _re.compile(rf"\bSections? {_NUMBER}(?:(?:, | and | to ){_NUMBER})*")
    _HEADING = _re.compile(r"^(\s*)### (\d\.\d) (.+)$", flags=_re.MULTILINE)

    def _link(match):
        """`Section 5.2` or `Sections 5.1 to 5.3`, with every number a link to its heading."""
        word, numbers = match[0].split(" ", 1)
        prefix = [word + " "]
        return _re.sub(_NUMBER, lambda number: f'<a class="nb-ref" href="#{_anchor(number[0])}">'
                                               f'{prefix.pop() if prefix else ""}{number[0]}</a>', numbers)

    def md(text):
        """Markdown. A numbered subsection heading gets an anchor, and a reference to a section links to it."""
        text = _HEADING.sub(lambda m: f'{m[1]}<h3 id="{_anchor(m[2])}"><span class="nb-number">{m[2]}</span> {m[3]}</h3>',
                            text)
        return mo.md(_REFERENCE.sub(_link, text))

    def _html(content):
        """Markdown text, or anything marimo can display, as an HTML string."""
        return md(content).text if isinstance(content, str) else mo.as_html(content).text

    # ---- numbers in text

    def number_word(number):
        """A small count as a word."""
        words = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")
        return words[number] if 0 <= number < len(words) else f"{number:,}"

    def decimals(width):
        """The number of decimals that shows an interval of this width to two significant digits."""
        return max(0, 1 - _math.floor(_math.log10(width)))

    def fixed(value, digits):
        """A number with a fixed number of decimals and a true minus sign. Zero carries no sign."""
        text = f"{value:.{digits}f}"
        return text.lstrip("-") if float(text) == 0 else text.replace("-", "−")

    def times(ratio, digits=1):
        """A ratio for running text: `digits` decimals below ten, a whole number above."""
        return f"{ratio:.{digits}f}" if abs(ratio) < 9.95 else f"{ratio:,.0f}"

    # ---- page elements

    def _blocks(items):
        """`items` maps a label to its content, or lists contents without labels."""
        labelled = items.items() if isinstance(items, dict) else [(None, content) for content in items]
        return "".join("<div>" + (f'<div class="nb-label">{_html(label)}</div>' if label else "") + _html(content)
                       + "</div>" for label, content in labelled)

    def workflow_map():
        """The six stages and what each one does, shown once under the title. Each stage links to its section."""
        steps = "".join(
            f'<a class="nb-map-step" href="#{_anchor(number)}"><div class="nb-map-name"><span>{number}</span>{name}</div>'
            f'<div class="nb-map-text">{text}</div></a>'
            for number, (name, text) in enumerate(STAGES, start=1))
        return mo.Html(f'<div class="nb-map">{steps}</div>')

    def section(title, lead, stage=None, kicker=None):
        """The banner that opens a section: its place in the workflow, its title and its summary."""
        if stage is None:
            top, number, outline = f'<div class="nb-eyebrow">{kicker}</div>', "", ""
            name = title.lower().replace(" ", "-")
        else:
            rail = []
            for index, (stage_name, _) in enumerate(STAGES, start=1):
                state = "is-done" if index < stage else "is-current" if index == stage else ""
                if index > 1:
                    rail.append(f'<span class="nb-rail-arrow {"is-done" if index <= stage else ""}">&#8594;</span>')
                rail.append(f'<a class="nb-rail-step {state}" href="#{_anchor(index)}">'
                            f'<span class="nb-rail-number">{index}</span>{stage_name}</a>')
            top, number = f'<div class="nb-rail">{"".join(rail)}</div>', f'<span class="nb-section-number">{stage}</span>'
            outline, name = f'<span class="nb-outline-only">{stage} </span>', _anchor(stage)
        return mo.Html(
            f'<div class="nb-section" id="{name}">{top}<div class="nb-section-head">{number}<div>'
            f'<h2 id="{name}-title">{outline}{title}</h2>'
            f'<div class="nb-standfirst">{_html(lead)}</div></div></div></div>')

    def key(text):
        """The message to take away from what was just shown."""
        return md("\n".join("> " + line for line in _cleandoc(text).splitlines()))

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

    def cards(items, columns=2):
        """Framed blocks side by side: `items` are (eyebrow, title, content) triples."""
        blocks = "".join(f'<div><div class="nb-eyebrow">{_html(eyebrow)}</div><div class="nb-card-title">{_html(title)}</div>'
                         f'{_html(content)}</div>' for eyebrow, title, content in items)
        return mo.Html(f'<div class="nb-grid is-cards" style="--nb-columns:{columns}">{blocks}</div>')

    def controls(*elements, hint=""):
        """A frame around interactive elements, with a line on what to do with them."""
        body = "" if not elements else mo.vstack(elements).text if len(elements) > 1 else mo.as_html(elements[0]).text
        return mo.Html(f'<div class="nb-controls"><div class="nb-controls-head"><span class="nb-eyebrow">Try it</span>'
                       f'<span class="nb-hint">{_html(hint) if hint else ""}</span></div>{body}</div>')

    def fields(items):
        """Controls with their labels in one column, so that the controls line up: `items` maps a label to a control."""
        rows = "".join(f'<div class="nb-field-label">{_html(label)}</div><div>{mo.as_html(element).text}</div>'
                       for label, element in items.items())
        return mo.Html(f'<div class="nb-fields">{rows}</div>')

    def bayesflow(text):
        """Text that introduces BayesFlow code."""
        return md('<span class="nb-chip">BayesFlow</span> ' + _cleandoc(text))

    def table(rows, align=None, nowrap=False):
        """A table from a list of dicts. The keys are the column headings and `align` holds l, c or r per column.
        A key (group, heading) puts neighbouring columns under a shared heading. `nowrap` keeps the first column
        on one line."""
        if not rows:
            return mo.Html("")
        columns = list(rows[0])
        side = dict(zip(columns, ({"l": "", "c": ' class="is-centre"', "r": ' class="is-right"'}[letter]
                                  for letter in align or "l" * len(columns))))
        cell = lambda value: (_html(value) if isinstance(value, str) and _re.search(r"[$*`\[<]|Section", value)
                              else _markup.escape(str(value)))
        if any(isinstance(column, tuple) for column in columns):
            top, bottom = [], []
            for group, members in _groupby(columns, key=lambda column: column[0] if isinstance(column, tuple) else None):
                members = list(members)
                if group is None:
                    top += [f'<th rowspan="2"{side[column]}>{cell(column)}</th>' for column in members]
                else:
                    top.append(f'<th colspan="{len(members)}" class="is-group"><span>{cell(group)}</span></th>')
                    bottom += [f"<th{side[column]}>{cell(column[1])}</th>" for column in members]
            head = f'<tr>{"".join(top)}</tr><tr>{"".join(bottom)}</tr>'
        else:
            head = "<tr>" + "".join(f"<th{side[column]}>{cell(column)}</th>" for column in columns) + "</tr>"
        body = "".join("<tr>" + "".join(f"<td{side[column]}>{cell(row[column])}</td>" for column in columns) + "</tr>"
                       for row in rows)
        return mo.Html(f'<div class="nb-table{" is-nowrap" if nowrap else ""}"><table><thead>{head}</thead>'
                       f'<tbody>{body}</tbody></table></div>')

    return (
        bayesflow,
        caption,
        cards,
        controls,
        decimals,
        fields,
        fixed,
        grid,
        key,
        md,
        note,
        number_word,
        section,
        stack,
        table,
        times,
        workflow_map,
    )


@app.cell(hide_code=True)
def _(bayesflow, controls, grid, key, md, mo, note, workflow_map):
    mo.vstack([
        md(r"""
        # Bayesian calibration of the Heston model on SPX options

        We calibrate the **Heston stochastic-volatility model** to 63 SPX option quotes. Instead of one best-fitting parameter vector, we compute a **posterior distribution** over the six unknowns $\boldsymbol\theta=(v_0,\kappa,v_\infty,\sigma_v,\rho,c)$.

        The posterior is learned with **amortized simulation-based inference** in **BayesFlow** (Radev et al., 2023). A neural network is trained once on simulated quotes. It then returns the posterior for any new set of 63 quotes in seconds.

        The workflow that we will explore is described as follows:
        """),
        workflow_map(),
        bayesflow("In this notebook, we explicitly show where and how **BayesFlow** is used, including code snippets "
                  "where useful. Those places carry this mark."),
        md("#### How to read this notebook"),
        grid([
            key("**Key messages** are set off like this. Read them first."),
            note("Notes and captions are set small. They document a figure or add a caveat."),
            mo.accordion({"Foldable sections hold method details": md(
                "Nothing in a foldable section is needed to follow the rest of the notebook.")}),
            controls(hint="These blocks are interactive in the live notebook. A static copy shows their initial state."),
        ], columns=2),
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
    # One half-spread in volatility points, for every quote, and its range as text.
    SPREAD_POINTS = vol_point_noise(H, MARKET_IV)
    SPREAD_RANGE = f"{np.nanmin(SPREAD_POINTS):.2g}–{np.nanmax(SPREAD_POINTS):.2g}"
    from datetime import datetime as _datetime

    _snapshot = _datetime.fromisoformat(design["valuation_time_utc"])
    SNAPSHOT_DAY = f"{_snapshot.day} {_snapshot:%B %Y}"
    SNAPSHOT_TIME = f"{_snapshot:%H:%M} UTC on {SNAPSHOT_DAY}"
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
        SNAPSHOT_DAY,
        SNAPSHOT_TIME,
        SPREAD_POINTS,
        SPREAD_RANGE,
        STRIKE,
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
    # The three data sets, in the order in which the notebook shows them.
    CASE_TITLES = {"synthetic-market-like": "Market-like simulation",
                   "synthetic-broad": "Stressed simulation",
                   "market": "SPX quotes"}
    PRODUCTS = _panel["products"]  # horizons in years, for the expected average variance
    PRODUCT_TITLES = {"control": f"Now to {365 * PRODUCTS['control'][1]:.0f} days",
                      "affected": f"{365 * PRODUCTS['affected'][0]:.0f} to {365 * PRODUCTS['affected'][1]:.0f} days"}
    return CASES, CASE_TITLES, CLEAN, PRODUCTS, PRODUCT_TITLES, TRUTHS


@app.cell(hide_code=True)
def _(
    DELTA,
    EXPIRY,
    EXPIRY_DATES,
    KIND,
    MARKET_IV,
    REF_VOL,
    SNAPSHOT_TIME,
    design,
    fixed,
    grid,
    md,
    mo,
    note,
    np,
    section,
):
    _puts = sorted({_q["target_z"] for _q in design["selected_quotes"] if _q["option_type"] == "put"})
    _calls = [_t for _t in design["targets_z"] if _t not in _puts]
    _n = lambda _x: f"{_x:g}".replace("-", "−")
    _atm = [100 * MARKET_IV[EXPIRY == _e][np.argmin(np.abs(DELTA[EXPIRY == _e]))] for _e in range(len(EXPIRY_DATES))]
    mo.vstack([
        section("The data", "The data are 63 SPX option quotes from one snapshot. They cover seven expiries, from one "
                "week to about eleven months, with nine strikes each.", kicker="Before the workflow"),
        grid({
            "Snapshot": f"The quotes are SPX index options, observed at {SNAPSHOT_TIME}. All 63 contracts are "
                        "European, PM-settled SPXW options.",
            "Each contract": r"The bid and the ask give the **mid price** $y_i$ and the **half-spread** "
                             r"$h_i=(\text{ask}_i-\text{bid}_i)/2$, in index points.",
            "Each expiry": "Each expiry has a forward price $F$ and a discount factor $D$. Both are estimated from "
                           "put–call parity on the same snapshot.",
        }),
        md(rf"""
        ### A common scale for strikes

        Strikes of different expiries are compared through the **standardized moneyness** $\delta$, the distance from the forward in units of a typical move over the option's life ($T$ in years, actual/365).

        $$\delta=\frac{{\ln(K/F)}}{{\sigma_{{\text{{ref}}}}\sqrt{{T}}}},\qquad \sigma_{{\text{{ref}}}}={100 * REF_VOL:.0f}\%$$

        The contracts were selected on this scale. At every expiry we took the puts nearest to $\delta$&nbsp;=&nbsp;{", ".join(_n(_v) for _v in _puts)} and the calls nearest to $\delta$&nbsp;=&nbsp;{", ".join(_n(_v) for _v in _calls)}. The selected contracts run from $\delta$&nbsp;=&nbsp;{fixed(DELTA.min(), 1)} to {fixed(DELTA.max(), 1)} ({int((KIND == 'put').sum())} puts, {int((KIND == 'call').sum())} calls).
        """),
        note(f"The reference volatility of {100 * REF_VOL:.0f}% is close to the at-the-money implied volatility of SPX "
             f"on this snapshot ({min(_atm):.0f}–{max(_atm):.0f}% across the seven expiries). One unit of $\\delta$ "
             "is therefore roughly one standard deviation of the log price at expiry."),
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
    STRIKE,
    caption,
    design,
    fixed,
    grid,
    grid_rules,
    key,
    md,
    mo,
    note,
    np,
    plots,
    plt,
    table,
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
    _axes[0].set_xticks([7, 30, 90, 365], labels=["7", "30", "90", "365"])
    _axes[0].minorticks_off()
    _axes[0].axhline(0, color=plots.BASELINE, lw=.8)
    for _e, _date in enumerate(EXPIRY_DATES):
        _at = EXPIRY == _e
        _axes[1].plot(DELTA[_at], 100 * H[_at] / MID[_at], "o-", ms=3, lw=1, color=EXPIRY_COLORS[_e])
        _axes[2].plot(DELTA[_at], 100 * MARKET_IV[_at], "o-", ms=3, lw=1, color=EXPIRY_COLORS[_e], label=f"{EXPIRY_DAYS[_e]} days")
    _axes[1].set(yscale="log", xlabel="standardized moneyness δ", ylabel="half-spread / mid price (%)",
                 title="Relative half-spread")
    _ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    _axes[1].set_yticks(_ticks, labels=[f"{_t:g}" for _t in _ticks])
    _axes[1].minorticks_off()
    _axes[2].set(xlabel="standardized moneyness δ", ylabel="implied volatility (%)", title="Market smile at the mid prices")
    _axes[2].legend(fontsize=6.5, loc="upper right")
    _atm = [100 * MARKET_IV[EXPIRY == _e][np.argmin(np.abs(DELTA[EXPIRY == _e]))] for _e in range(len(EXPIRY_DATES))]
    _wing = [100 * MARKET_IV[EXPIRY == _e][np.argmin(DELTA[EXPIRY == _e])] for _e in range(len(EXPIRY_DATES))]
    _relative = 100 * H / MID
    _worst = int(np.argmax(_relative))
    _override = design["target_overrides"][0]
    _distance = max(abs(_q["target_distance"]) for _q in QUOTES)
    _days = ", ".join(str(_d) for _d in grid_rules.TARGET_DAYS[:-1]) + f" and {grid_rules.TARGET_DAYS[-1]}"

    # The foldable table of all quotes: one row per expiry, then one row per contract, by expiry and strike.
    _expiries = [{"Expiry": _date, "Days to expiry": EXPIRY_DAYS[_e], "Forward $F$": f"{FORWARD[EXPIRY == _e][0]:,.2f}",
                  "Discount factor $D$": f"{DISCOUNT[EXPIRY == _e][0]:.5f}"} for _e, _date in enumerate(EXPIRY_DATES)]
    _contracts, _last = [], None
    for _i in np.lexsort((STRIKE, EXPIRY)):
        _quote = QUOTES[_i]
        _contracts.append({
            "Days to expiry": "" if EXPIRY[_i] == _last else EXPIRY_DAYS[EXPIRY[_i]],
            "Type": KIND[_i], "Strike $K$": f"{_quote['strike']:,.0f}", r"$\delta$": fixed(DELTA[_i], 2),
            "Bid": f"{_quote['bid']:,.2f}", "Ask": f"{_quote['ask']:,.2f}", "Mid $y$": f"{MID[_i]:,.3f}",
            "Half-spread $h$": f"{H[_i]:.3f}", "Implied volatility (%)": f"{100 * MARKET_IV[_i]:.2f}"})
        _last = EXPIRY[_i]

    mo.vstack([
        md("### The 63 quotes"),
        _fig,
        caption("The left panel shows the selected contracts by days to expiry and standardized moneyness, puts in "
                "blue and calls in orange. The middle panel shows the half-spread as a share of the mid price, and "
                "the right panel the implied volatility at the mid prices. In these two panels each colour is one "
                "expiry."),
        key(r"""
        **The seven smiles are nearly parallel in $\delta$.** The skew per unit of $\delta$ is about the same at one week as at eleven months. This is the feature the model must reproduce.
        """),
        grid({
            "Contracts": "Every expiry has nine contracts at similar values of $\\delta$. Of the 63 contracts, "
                         f"{int((KIND == 'put').sum())} are puts.",
            "Spreads": f"The half-spread is {_relative.min():.2f}% of the price at best, near the money. It reaches "
                       f"{_relative.max():.0f}% for a {KIND[_worst]} worth {MID[_worst]:.2f} index points"
                       + (", the cheapest contract." if _worst == int(np.argmin(MID)) else "."),
            "Smile": "The curve of implied volatility against strike is the **volatility smile**. It runs from "
                     f"{min(_atm):.0f}–{max(_atm):.0f}% at the money to {min(_wing):.0f}–{max(_wing):.0f}% for the "
                     "deepest puts.",
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
                    "Eligibility": "A contract was eligible if its half-spread was at most "
                                   f"{100 * grid_rules.MAX_RELATIVE_HALFSPREAD:.0f}% of its mid price. Quotes used to "
                                   "estimate the forwards and discount factors were excluded, and so were quotes set "
                                   "aside for later tests. A target was used only if an eligible contract lay within "
                                   f"{grid_rules.MAX_Z_DISTANCE:g} of it.",
                    "One exception": f"At the {EXPIRY_DAYS[EXPIRY_DATES.index(_override['expiration'])]}-day expiry no "
                                     f"eligible call lies near $\\delta={_override['original_target_z']:g}$, so that "
                                     f"target was moved to $\\delta={_override['revised_target_z']:g}$.",
                }, columns=2),
                note("The reference volatility only sets the units of the axis. The pricer uses each contract's "
                     "actual strike, forward, discount factor and expiry, and the network sees prices."),
            ]),
            "All 63 quotes": mo.vstack([
                md("Each expiry has one forward price and one discount factor."),
                table(_expiries, align="lrrr"),
                md("The contracts are listed by expiry and strike. Prices are in index points."),
                table(_contracts, align="rlrrrrrrr"),
            ]),
        }),
    ])
    return


@app.cell(hide_code=True)
def _(grid, md, mo, note, section):
    mo.vstack([
        section("Model", "The model combines Heston dynamics for the price, a prior over six parameters and quote "
                "noise set by the bid–ask spread.", stage=1),
        md(r"""
        ### 1.1 Heston dynamics

        The variance $V_t$ follows its own random process, correlated with the price (Heston, 1993).

        $$
        \frac{dF_t}{F_t} = \sqrt{V_t}\,dW_t^F,\qquad
        dV_t = \kappa(v_\infty-V_t)\,dt+\sigma_v\sqrt{V_t}\,dW_t^V,
        \qquad d\langle W^F,W^V\rangle_t=\rho\,dt
        $$

        Here $W^F$ and $W^V$ are two Brownian motions with correlation $\rho$, and $F_t$ is the forward price for the option's expiry. The model has six unknowns.

        | Symbol | Meaning |
        |:---|:---|
        | $v_0$ | initial variance $V_0$ |
        | $\kappa$ | speed of mean reversion, per year |
        | $v_\infty$ | long-run variance, the level $V_t$ reverts to |
        | $\sigma_v$ | volatility of variance |
        | $\rho$ | correlation of price and variance shocks |
        | $c$ | noise multiplier (Section 1.3) |
        """),
        note("The priors are given with the prior predictive check in Section 2.2."),
        mo.accordion({"Why the dynamics have no drift": grid({
            "Pricing measure": r"Prices are computed under the pricing measure $\mathbb Q$. Under it, the discounted "
                               "prices of traded assets are martingales.",
            "Forward price": r"The forward price $F_t=S_t\,e^{(r-q)(T-t)}$, with interest rate $r$ and dividend "
                             r"yield $q$, is a martingale under $\mathbb Q$. It has no drift. At expiry $F_T=S_T$, "
                             r"so a call is worth $C=D\,\mathbb E^{\mathbb Q}\big[(F_T-K)^+\big]$ with $D=e^{-rT}$.",
            "Forward units": r"Rates and dividends enter only through $D$ and $F$, which are read from the market "
                             r"by put–call parity. Divided by $D\,F$, prices depend only on $\ln(K/F)$, $T$ and the "
                             "Heston parameters, so one parameter set prices every expiry.",
            "Real-world return": "The index's real-world expected return plays no role. Estimating it would need a "
                                 "time series of prices and a model under the physical measure.",
        }, columns=2)}),
    ])
    return


@app.cell(hide_code=True)
def _(cards, md, mo):
    mo.vstack([
        md(r"""
        ### 1.2 Four features to keep in mind

        Four features of Heston shape every smile in this notebook.
        """),
        cards([
            ("Feature 1", "Mean reversion sets the term structure", r"""
            $$\mathbb E[V_t]=v_\infty+(v_0-v_\infty)\,e^{-\kappa t}$$

            - The initial variance $v_0$ sets short expiries, the long-run variance $v_\infty$ long ones.
            - The speed $\kappa$ sets how fast the variance reverts. A shock halves in $\ln 2/\kappa$ years.
            """),
            ("Feature 2", r"$\rho$ tilts the smile, $\sigma_v$ curves it", r"""
            - With $\rho<0$, volatility rises when prices fall, so puts cost more. This is the **skew**.
            - A larger $\sigma_v$ makes the variance more random, which fattens both tails. This is the **curvature**, and it also steepens the skew.
            - A large $\sigma_v$ breaks the **Feller condition** $2\kappa v_\infty\ge\sigma_v^2$. The variance can then touch zero.
            """),
            ("Feature 3", "The short-expiry skew has a ceiling", r"""
            $$\psi(T)=\frac{\partial\,\sigma_{\mathrm{imp}}(k,T)}{\partial k}\bigg|_{k=0},\qquad k=\ln(K/F)$$

            - The **at-the-money skew** $\psi(T)$ measures how fast implied volatility changes with log-moneyness at the money, for expiry $T$.
            - As $T$ shrinks, $|\psi(T)|$ grows, but only up to a ceiling: $|\psi(T)|\to|\rho|\,\sigma_v/(4\sqrt{v_0})$ as $T\to0$.
            - Over a few days the variance barely moves from $v_0$, so the skew depends only on $\rho$, $\sigma_v$ and $v_0$.
            - For expiries well below the reversion time $1/\kappa$ the skew is at the ceiling and grows no further.
            """),
            ("Feature 4", "Prices in closed form", r"""
            - The model is affine and $(F_t,V_t)$ is a Markov state, so the characteristic function is known exactly.
            - Each price is one Fourier integral of the characteristic function. We call this the **Fourier pricer**.
            - All 63 prices take milliseconds, so MCMC with the exact likelihood is affordable (Section 4.1).
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
        caption("The black curves use the market-like parameters, a parameter set close to the fit to the SPX quotes "
                "(Section 4.1). In each panel one parameter changes."),
        grid({
            "(a) Mean reversion":
                f"The expected variance moves from $v_0$ to $v_\\infty$ and covers half the distance in "
                f"{_half_life:.0f} days, which is $\\ln 2/\\kappa$ years. Implied volatilities sit below these "
                "curves, because a random variance lowers at-the-money volatility (Lecture 10, Figure 10.6).",
            "(b) Correlation":
                f"From $\\rho={_minus(_rhos[0])}$ to ${_minus(_rhos[-1])}$, the implied volatility of {_wings[0]} "
                f"rises from {_deep(_rho_smiles)[0]:.1f}% to {_deep(_rho_smiles)[1]:.1f}% and that of {_wings[1]} "
                f"falls from {_call(_rho_smiles)[0]:.1f}% to {_call(_rho_smiles)[1]:.1f}%.",
            "(c) Volatility of variance":
                f"From $\\sigma_v={_xis[0]:.2f}$ to ${_xis[-1]:.2f}$, the implied volatility of {_wings[0]} rises "
                f"from {_deep(_xi_smiles)[0]:.1f}% to {_deep(_xi_smiles)[1]:.1f}% and the at-the-money level falls "
                f"from {_atm(_xi_smiles)[0]:.1f}% to {_atm(_xi_smiles)[1]:.1f}%.",
            "(d) Short-expiry skew":
                f"The skew is {_at(365):.2f} at one year, {_at(30):.2f} at 30 days and {_at(7):.2f} at 7 days. Below "
                f"{_near:.0f} days it stays within 10% of the ceiling, {_ceiling:.2f}.",
        }, columns=2, kind="reading"),
    ])
    return


@app.cell(hide_code=True)
def _(BOUNDS, NAMES, TRUTHS, grid, md, mo, reference_summary):
    _truth = TRUTHS["synthetic-market-like"]
    _market = {_n: reference_summary["cases"]["market"]["parameters"][_n]["regenerated_q05_q50_q95"][1] for _n in NAMES}
    _like = (2 * _truth[1] * _truth[2], _truth[3] ** 2)
    _fit = (2 * _market["kappa"] * _market["theta"], _market["vol_of_vol"] ** 2)
    mo.accordion({
        r"Feature 2 in depth: a large $\sigma_v$ and the Feller condition": grid({
            "Condition": r"A steep equity skew needs a large $\sigma_v$ (feature 2), and a large $\sigma_v$ breaks "
                         r"the **Feller condition** $2\kappa v_\infty\ge\sigma_v^2$.",
            "Meaning": "If the condition holds, the variance never reaches zero. Otherwise the variance can touch "
                       "zero, and the drift then moves it back up. The model and the pricer are valid either way, "
                       "and the prior does not impose the condition.",
            "In this notebook": rf"The market-like parameters violate the condition ($\sigma_v^2={_like[1]:.1f}$ "
                                rf"against $2\kappa v_\infty={_like[0]:.2f}$). So do the posterior medians for the "
                                rf"SPX quotes ($\sigma_v^2={_fit[1]:.1f}$ against $2\kappa v_\infty={_fit[0]:.2f}$). "
                                rf"The posterior fits the SPX skew with a $\sigma_v^2$ of {_fit[1] / _fit[0]:.0f} "
                                "times the Feller limit.",
        }),
        "Feature 4 in depth: the Fourier pricer": mo.vstack([
            md(r"""
            Heston is **affine**. Its characteristic function $\varphi(u)=\mathbb E^{\mathbb Q}\big[e^{iu\ln(F_T/F_0)}\big]$ is known in closed form, so a European option needs a single one-dimensional integral (Lewis, 2001).

            The pricer uses a **control-variate** form: the Black–Scholes price $C_{\mathrm{BS}}$ with the same expected average variance, plus the integral of the difference of the two characteristic functions. Puts get the same correction.

            $$C=C_{\mathrm{BS}}+D\,\frac{\sqrt{FK}}{\pi}\int_0^\infty\operatorname{Re}\!\left[e^{iu\ln(F/K)}\Big(\varphi_{\mathrm{BS}}\big(u-\tfrac i2\big)-\varphi\big(u-\tfrac i2\big)\Big)\right]\frac{du}{u^2+\tfrac14}$$
            """),
            grid({
                "Control variate": "For very cheap options the plain formula is a small difference of two large "
                                   "numbers. The control-variate form avoids the cancellation.",
                "Accuracy": "The integral uses Gauss–Legendre quadrature. The nodes and the interval are doubled "
                            "until two successive results agree to within 1% of the contract's half-spread. That is "
                            f"2% of the smallest noise standard deviation the prior allows ($c={BOUNDS[-1, 0]:g}$).",
                "Speed": "All 63 prices take milliseconds. The prior predictive check in Section 2.2 reports the "
                         "time for its simulations.",
            }),
        ]),
    })
    return


@app.cell(hide_code=True)
def _(key, md, mo):
    mo.vstack([
        md(r"""
        ### 1.3 Observation model

        Each quote is its Heston price plus Gaussian noise. The standard deviation (SD) of the noise is $c$ times the quote's half-spread.

        $$
        y_i = P_i(v_0,\kappa,v_\infty,\sigma_v,\rho) + c\,h_i\,\epsilon_i,\qquad
        \epsilon_i \overset{\mathrm{iid}}{\sim}\mathcal N(0,1),\qquad i=1,\dots,63
        $$

        The table lists the terms of this equation.

        | Symbol | Role | Meaning |
        |:---|:---|:---|
        | $y_i$ | data | the quote (mid price) of contract $i$ |
        | $P_i$ | model | its Heston price |
        | $h_i$ | data | the observed half-spread, fixed and identical in every simulation |
        | $\epsilon_i$ | noise | a fresh standard-normal draw for every contract and every simulated data set |
        | $c$ | parameter | the noise multiplier, which scales all 63 noise levels together and is inferred with the other five |

        “One half-spread is one standard deviation” is a convention. Inferring $c$ lets the data correct it.
        """),
        key(r"""
        **The model has no model-error term.** Heston is assumed to price every contract exactly. A systematic gap between Heston and the market can then only be absorbed by a larger $c$ (Section 5.5).
        """),
    ])
    return


@app.cell(hide_code=True)
def _(
    DELTA,
    EXPIRY,
    EXPIRY_COLORS,
    EXPIRY_DAYS,
    MARKET_IV,
    SPREAD_POINTS,
    SPREAD_RANGE,
    caption,
    grid,
    key,
    mo,
    np,
    plt,
):
    _fig, _ax = plt.subplots(figsize=(6.4, 2.9), constrained_layout=True)
    for _e, _days in enumerate(EXPIRY_DAYS):
        _at = EXPIRY == _e
        _ax.plot(DELTA[_at], SPREAD_POINTS[_at], "o-", ms=2.5, lw=.8, color=EXPIRY_COLORS[_e], label=f"{_days} days")
    _ax.set(yscale="log", xlabel="standardized moneyness δ", ylabel="volatility points",
            title="One half-spread, in volatility points")
    _ticks = [0.02, 0.05, 0.1, 0.2]
    _ax.set_yticks(_ticks, labels=[f"{_t:g}" for _t in _ticks])
    _ax.minorticks_off()
    _ax.legend(fontsize=6, ncol=2)
    _range = 100 * (np.nanmax(MARKET_IV) - np.nanmin(MARKET_IV))
    mo.hstack([
        mo.vstack([
            _fig,
            caption("The half-spread of every quote, converted to volatility points with the quote's own vega. Each "
                    "colour is one expiry."),
        ]),
        mo.vstack([
            key(f"**One half-spread is {SPREAD_RANGE} volatility points.** The smile spans about {_range:.0f} "
                "points."),
            grid({
                "Informative quotes": "The quotes determine the parameters tightly, and the posteriors in Section "
                                      "4.2 are narrow.",
                "Visible model error": "A systematic model error of even a fraction of a volatility point is large "
                                       "compared with the assumed noise.",
            }, columns=1),
        ]),
    ], widths=[3, 2], gap=2, align="center")
    return


@app.cell(hide_code=True)
def _(md, mo, note, section):
    mo.vstack([
        section("Simulate", "One simulation corresponds to a **forward pass** through the pricer plus the addition "
                "of noise. It produces one hypothetical market.", stage=2),
        md(r"""
        ### 2.1 The simulator

        A simulation takes three steps.

        1. Draw $\boldsymbol\theta=(v_0,\kappa,v_\infty,\sigma_v,\rho,c)$ from the prior.
        2. Price the 63 contracts with the Fourier pricer (feature 4).
        3. Add the quote noise $c\,h_i\,\epsilon_i$.

        The output is a vector of 63 numbers, one noisy mid price per contract.
        """),
        note("In this toy model only the mid prices are simulated. Bids and asks are not. The half-spreads $h_i$ "
             "keep their observed values and set the size of the noise."),
    ])
    return


@app.cell(hide_code=True)
def _(
    BOUNDS,
    EXPIRY_DATES,
    EXPIRY_DAYS,
    MATH_LABELS,
    TRUTHS,
    controls,
    fields,
    mo,
    np,
):
    _start = TRUTHS["synthetic-market-like"]
    _steps = (0.001, 0.1, 0.001, 0.05, 0.01, 0.05)
    _digits = (3, 1, 3, 2, 2, 2)
    _names = [_label[0].upper() + _label[1:] for _label in MATH_LABELS]
    sim_params = mo.ui.array([
        mo.ui.slider(start=float(BOUNDS[_j, 0]), stop=float(BOUNDS[_j, 1]), step=_steps[_j],
                     value=round(float(np.clip(_start[_j], BOUNDS[_j, 0], BOUNDS[_j, 1])), _digits[_j]),
                     show_value=True)
        for _j in range(6)])
    sim_expiry = mo.ui.dropdown(
        options={f"{_days} days ({_date})": _e for _e, (_date, _days) in enumerate(zip(EXPIRY_DATES, EXPIRY_DAYS))},
        value=f"{EXPIRY_DAYS[4]} days ({EXPIRY_DATES[4]})")
    sim_seed = mo.ui.number(start=0, stop=9999, step=1, value=1)
    controls(
        mo.hstack([
            fields({**{_names[_j]: sim_params[_j] for _j in range(3)}, "Quotes at expiry": sim_expiry}),
            fields({**{_names[_j]: sim_params[_j] for _j in range(3, 6)}, "Random seed": sim_seed}),
        ], justify="start", gap=3),
        hint="Move the sliders. They start at the market-like parameters.",
    )
    return sim_expiry, sim_params, sim_seed


@app.cell(hide_code=True)
def _(
    DELTA,
    EXPIRY,
    EXPIRY_DAYS,
    H,
    MARKET_IV,
    caption,
    grid,
    implied_vols,
    md,
    mo,
    note,
    np,
    number_word,
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

    try:
        _clean = pilot.prices(_theta)
    except (FloatingPointError, ValueError):  # the pricer did not reach its accuracy target
        _clean = None

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
        _axes[1, 0].scatter(DELTA[_at], 100 * MARKET_IV[_at], s=16, color="#bbbbbb", label="SPX quotes")
        _axes[1, 0].plot(DELTA[_at], 100 * _clean_iv[_at], "-", color=plots.MUTED, lw=1.3, label="Heston price")
        _axes[1, 0].scatter(DELTA[_at], 100 * _quote_iv[_at], s=18, color=plots.SERIES[2], zorder=3, label="simulated quote")
        _axes[1, 1].fill_between(DELTA[_at], -_sd[_at], _sd[_at], color=plots.SERIES[2], alpha=.18, lw=0, label="±1 noise SD")
        _axes[1, 1].axhline(0, color=plots.MUTED, lw=1)
        _axes[1, 1].plot(DELTA[_at], 100 * (_quote_iv - _clean_iv)[_at], "o", ms=4, color=plots.SERIES[2],
                         label="quote − Heston price")
        _axes[1, 0].legend(fontsize=7)
        _axes[1, 1].legend(fontsize=7)
    _axes[1, 0].set(xlabel="standardized moneyness δ", ylabel="implied volatility (%)",
                    title=f"Quotes at {EXPIRY_DAYS[sim_expiry.value]} days")
    _axes[1, 1].set(xlabel="standardized moneyness δ", ylabel="volatility points", title="The quote errors, magnified")

    _view = [_fig, caption(f"The top row shows {number_word(_count)} forward-price paths over {number_word(_days // 252)} "
                           "year and the volatility behind them. The bottom row shows the quotes at the selected "
                           "expiry and their errors against the Heston price.")]
    if _clean is None:
        _view.append(note("The pricer did not reach its accuracy target for these parameters. Move a slider."))
    else:
        _atm = 100 * _clean_iv[_at][np.argmin(np.abs(DELTA[_at]))]
        _deep = 100 * _clean_iv[_at][np.argmin(DELTA[_at])]
        _errors = np.abs(100 * (_quote_iv - _clean_iv))[_at]
        _invisible = np.nanmax(_errors) < 0.01 * np.ptp(_axes[1, 0].get_ylim())
        _view.append(grid({
            "Price and volatility paths":
                f"Volatility starts at {100 * np.sqrt(_v0):.0f}%, and the expected variance reverts to $v_\\infty$ "
                f"with a half-life of {365 * np.log(2) / _kappa:.0f} days. The dashed line marks "
                f"$\\sqrt{{v_\\infty}}={100 * np.sqrt(_v_inf):.0f}\\%$. With $\\rho<0$, falls in the price come with "
                "spikes in volatility.",
            "Quotes and their errors":
                f"At this expiry the Heston smile runs from {_atm:.1f}% at the money to {_deep:.1f}% at the deepest "
                f"put. The quote errors are at most {np.nanmax(_errors):.2f} volatility points"
                f"{', invisible on the smile' if _invisible else ''}.",
        }, kind="reading"))
        _view.append(note("The paths only illustrate the dynamics. The simulator prices options exactly and never "
                          "draws paths."))
        if _shape < 1:
            _view.append(note(
                f"A single path does not settle at the dashed line. Here $\\sigma_v^2={_xi ** 2:.1f}$ exceeds "
                f"$2\\kappa v_\\infty={2 * _kappa * _v_inf:.2f}$, so the Feller condition fails (feature 2). In the "
                f"long run volatility is below {_median_vol:.2g}% half the time (dotted line). Short spikes carry "
                "the average, and in between the price barely moves."))
    _view.append(md(r"""
    #### What to try

    Go through the features of Section 1.2, then see how close you can get to the SPX quotes (grey).

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
    `bf.make_simulator` chains two functions by argument name. The prior returns named parameters. The observation model receives them, prices the 63 contracts and adds the noise. It returns every quote divided by its half-spread, so that the network sees the data in noise units.
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
def _(BOUNDS, grid, md, mo, np):
    _lo, _hi = BOUNDS[:, 0], BOUNDS[:, 1]
    _n = lambda _x: f"{_x:g}".replace("-", "−")
    _days = lambda _kappa: np.log(2) / _kappa * 365
    mo.vstack([
        md(r"""
        ### 2.2 Prior predictive check

        The **prior predictive distribution** is the distribution of quotes the model produces before seeing data: draw $\boldsymbol\theta$ from the prior, then simulate.

        The check asks two things of it.
        """),
        grid({
            "Coverage of the data": "The simulated quotes should cover the observed ones. If they do not, the "
                                    "network has to extrapolate, and the prior or the model needs rethinking.",
            "Width": "The simulated range should not be needlessly wide. Simulations of implausible markets cost "
                     "training effort and tell the network little.",
        }),
        md(rf"""
        The prior is independent and uniform in each parameter's natural units.

        | Symbol | Uniform on | Meaning of the range |
        |:---|:---|:---|
        | $v_0$ | {_n(_lo[0])} to {_n(_hi[0])} | initial volatility $\sqrt{{v_0}}$ between {100 * np.sqrt(_lo[0]):.0f}% and {100 * np.sqrt(_hi[0]):.0f}% |
        | $\kappa$ | {_n(_lo[1])} to {_n(_hi[1])} | half-life of a variance shock, $\ln 2/\kappa$, between {_days(_hi[1]):.0f} days and {_days(_lo[1]) / 365:.0f} years |
        | $v_\infty$ | {_n(_lo[2])} to {_n(_hi[2])} | long-run volatility between {100 * np.sqrt(_lo[2]):.0f}% and {100 * np.sqrt(_hi[2]):.0f}% |
        | $\sigma_v$ | {_n(_lo[3])} to {_n(_hi[3])} | large values are allowed, and the Feller condition is not imposed |
        | $\rho$ | {_n(_lo[4])} to {_n(_hi[4])} | negative only, so volatility rises when prices fall |
        | $c$ | {_n(_lo[5])} to {_n(_hi[5])} | noise SD between {_n(_lo[5])} and {_n(_hi[5])} half-spreads |
        """),
    ])
    return


@app.cell(hide_code=True)
def _(
    DELTA,
    EXPIRY,
    EXPIRY_DAYS,
    H,
    KIND,
    MARKET_IV,
    MID,
    caption,
    grid,
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
    _count, _shown = 1000, 150  # simulated data sets, and how many of their smiles are drawn
    _started = time.perf_counter()
    prior_batch = make_simulator(20261111).sample(_count)
    prior_seconds = time.perf_counter() - _started
    _iv = implied_vols(prior_batch["observations"] * H)
    _fig, _axes = plt.subplots(1, 3, figsize=(11.0, 3.2), sharey=True, constrained_layout=True)
    for _ax, _e in zip(_axes, (0, 4, 6)):
        _at = EXPIRY == _e
        for _row in _iv[:_shown]:
            _ax.plot(DELTA[_at], 100 * _row[_at], color=plots.SERIES[0], alpha=.12, lw=.8)
        _ax.plot(DELTA[_at], 100 * MARKET_IV[_at], "o-", color=plots.SERIES[1], ms=3, lw=1.5, label="SPX quotes")
        _ax.set(title=f"{EXPIRY_DAYS[_e]} days", xlabel="standardized moneyness δ")
    _axes[0].set(ylabel="implied volatility (%)", ylim=(0, 70))
    _axes[0].legend(fontsize=7)
    _pct = 100 * np.mean(prior_batch["observations"] * H < MID, axis=0)   # the market quote's percentile, per contract
    _edge = int(np.sum((_pct == 0) | (_pct == 100)))
    _low = int(np.argmin(_pct))
    _cover = (f"**The SPX quotes lie inside the simulated range at every contract.** Their percentiles among the "
              f"simulated prices run from {_pct.min():.0f}% to {_pct.max():.0f}%. The lowest belongs to the "
              f"{EXPIRY_DAYS[EXPIRY[_low]]}-day {KIND[_low]} at $\\delta={DELTA[_low]:.1f}$." if _edge == 0 else
              f"**The SPX quotes fall outside the simulated range at {_edge} of 63 contracts.**")
    mo.vstack([
        _fig,
        caption(f"The blue curves are {_shown} of the {_count:,} prior-predictive smiles at three expiries, and the "
                f"orange curve is the SPX smile. Simulating all {_count:,} data sets took {prior_seconds:.0f} s."),
        key(_cover),
        grid({
            "Range of the prior": "The prior produces calm and stressed markets, flat and steep skews.",
            "Limit of the check": "The check only shows that the prior does not rule the data out. Whether the "
                                  "model fits them is the question of Section 5.5.",
        }, kind="reading"),
        note(f"Of the simulated quotes, {100 * np.isnan(_iv).mean():.1f}% have no implied volatility, because noise "
             "pushed a very cheap option below its intrinsic value. They remain valid data."),
    ])
    return prior_batch, prior_seconds


@app.cell(hide_code=True)
def _(
    TRAINING,
    caption,
    fit_whitening,
    grid,
    key,
    md,
    mo,
    note,
    np,
    number_word,
    plots,
    plt,
    prior_batch,
):
    _x = prior_batch["observations"]
    _c = prior_batch["c"][:, 0]
    _split = int(0.7 * len(_x))  # the whitening is fitted on the first part and shown on the rest
    _fit = fit_whitening(_x[:_split])
    _corr_before = np.corrcoef(_x[_split:], rowvar=False)
    _corr_after = np.corrcoef(_fit.transform(_x[_split:]), rowvar=False)
    _values, _vectors = np.linalg.eigh(np.cov(_x[:_split], rowvar=False))
    _order = np.argsort(_values)[::-1]
    _values, _vectors = _values[_order], _vectors[:, _order]
    _scores = (_x[_split:] - _x[:_split].mean(axis=0)) @ _vectors / np.sqrt(_values)
    _link = np.array([abs(np.corrcoef(np.abs(_scores[:, _j]), _c[_split:])[0, 1]) for _j in range(63)])
    _mantissa, _exponent = f"{_values[0] / _values[-1]:.0e}".split("e")
    _ratio = f"${_mantissa}\\times10^{{{int(_exponent)}}}$"
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
        md(r"""
        ### 2.3 Whitening the network input

        The network receives the 63 numbers $y_i/h_i$, the quotes in noise units. In this raw form they are hard to learn from, for two reasons.
        """),
        grid({
            "Strong correlation": "A change in any parameter moves all 63 inputs together.",
            "Bad scaling": f"A few directions, such as the overall level and the skew, vary up to {_ratio} times "
                           "more across simulations than the directions that reveal the noise multiplier $c$.",
        }),
        md(r"""
        **Whitening** is a fixed linear change of coordinates. It makes the inputs uncorrelated with unit variance, without losing information. It is fitted once on simulations, frozen, and applied unchanged to every later input, the SPX quotes included.
        """),
        _fig,
        caption(f"The figure uses the {len(_x):,} simulations of Section 2.2. The whitening is fitted on {_split:,} of "
                f"them and shown on the other {len(_x) - _split:,}."),
        key("**Whitening gives every direction of the input the same weight.** Before it, the information about $c$ "
            "sits in the directions that vary least. Afterwards these directions count as much as the dominant "
            "ones."),
        grid({
            "Variance per direction": "The variance falls from the first principal direction of the inputs to the "
                                      f"last by {number_word(int(_exponent))} orders of magnitude.",
            "Correlations": "On new simulations almost every pair of inputs is strongly correlated before "
                            "whitening. Afterwards the matrix is nearly diagonal.",
            "Information about $c$": f"The correlation with $c$ averages {np.mean(_link[-30:]):.2f} over the 30 "
                                     f"directions of lowest variance, against {np.mean(_link[:10]):.2f} over the "
                                     "first 10.",
        }, kind="reading", widths=(1, 2, 1)),
        note(f"The pretrained estimator's whitening was fitted on its first {TRAINING['whitening_rows']:,} training "
             "pairs."),
        mo.accordion({"Whitening, formally": mo.vstack([
            md(r"""
            Let $x\in\mathbb R^{63}$ be the vector of inputs $y_i/h_i$, with mean $\mu$ and covariance $\Sigma$ estimated from $N$ simulated training data sets. Write $\Sigma=U\Lambda U^\top$, with orthonormal eigenvectors $U$ and eigenvalues $\Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_{63})$. The whitened input is

            $$\tilde x = W(x-\mu),\qquad W=U\Lambda^{-1/2}U^\top=\Sigma^{-1/2}$$

            so that $\operatorname{Cov}(\tilde x)=W\Sigma W^\top=I$.
            """),
            grid({
                "Choice of $W$": r"This choice of $W$ is **ZCA whitening**. Among all transformations with identity "
                                 r"covariance, it keeps $\tilde x$ closest to $x$, so each whitened input still "
                                 r"corresponds roughly to one quote. The alternative $\Lambda^{-1/2}U^\top$ (PCA "
                                 r"whitening) would mix them. Eigenvalues below $10^{-8}\lambda_{\max}$ are floored "
                                 "for numerical stability.",
                "Effect on training": r"The ratio $\lambda_{\max}/\lambda_{\min}$, the **condition number** of "
                                      r"$\Sigma$, measures how badly scaled the raw inputs are. Gradient descent on "
                                      "badly scaled inputs is slow and unreliable, because a step size that suits "
                                      "the dominant directions is far too small for the others. Whitening sets the "
                                      "condition number to one.",
                "Fitted once": r"The mean $\mu$ and the matrix $W$ come from training simulations only and are "
                               "stored with the estimator, so the same transformation is applied at training and at "
                               "inference.",
            }),
        ])}),
    ])
    return


@app.cell(hide_code=True)
def _(TRAINING, bayesflow, grid, key, md, mo, section, table):
    _steps = (
        (r"Quotes $\mathbf y$", "63 mid prices, simulated in training and observed at inference", ""),
        ("Preprocessing", r"Division by the half-spreads and the frozen whitening give $\tilde{\mathbf x}$, 63 numbers", ""),
        ("Summary network", "None: the 63 numbers enter directly", "is-absent"),
        ("Inference network", "`FlowMatching` with an MLP of 3 × 128 units and dropout 0.1", "is-network"),
        ("Posterior draws", r"As many draws of $\boldsymbol\theta=(v_0,\kappa,v_\infty,\sigma_v,\rho,c)$ as needed", ""),
    )
    _chain = '<div class="nb-flow-arrow">&#8594;</div>'.join(
        f'<div class="nb-flow-step {_kind}"><div class="nb-flow-name">{md(_name).text}</div>'
        f'<div class="nb-flow-text">{md(_text).text}</div></div>' for _name, _text, _kind in _steps)
    mo.vstack([
        section("Train", r"The network learns the posterior $q_\phi(\boldsymbol\theta\mid\mathbf y)$ from simulated "
                r"pairs $(\boldsymbol\theta,\mathbf y)$. No likelihood is ever evaluated.", stage=3),
        md(r"""
        ### 3.1 The network

        The network turns the 63 quotes into draws from the posterior of the six parameters.
        """),
        mo.Html(f'<div class="nb-flow">{_chain}</div>'),
        grid({
            "In training": rf"The network sees simulated pairs $(\boldsymbol\theta,\tilde{{\mathbf x}})$. The "
                           rf"pretrained estimator saw {TRAINING['rows']:,} of them.",
            "At inference": r"The network receives one observed $\tilde{\mathbf x}$ and returns as many posterior "
                            "draws as wanted.",
        }, kind="reading"),
        key(r"""
        **Simulated pairs are draws from the joint distribution of parameters and quotes.** Among the simulations that produced quotes like $\mathbf y$, the parameters follow the posterior. The network learns this conditional distribution for every $\mathbf y$ at once.
        """),
        mo.accordion({"Design choices and their alternatives in BayesFlow": table([
            {"Design choice": "What is learned",
             "Here": r"the posterior $q_\phi(\boldsymbol\theta\mid\mathbf y)$",
             "Alternatives in BayesFlow": "a neural pricer, a likelihood (NLE) or a likelihood ratio (NRE), each "
                                          "used inside a new MCMC run per data set"},
            {"Design choice": "Summary network",
             "Here": "none, the input is a fixed vector of 63 whitened quotes",
             "Alternatives in BayesFlow": "`DeepSet` or `SetTransformer` for quote sets of varying size and layout, "
                                          "`TimeSeriesNetwork` for price histories"},
            {"Design choice": "Inference network",
             "Here": "`FlowMatching`",
             "Alternatives in BayesFlow": "`CouplingFlow` (exact density, one-pass sampling), `DiffusionModel`, "
                                          "`ConsistencyModel` (few sampling steps)"},
            {"Design choice": "Parameter space",
             "Here": "standardized and unbounded, so a few draws can leave the prior",
             "Alternatives in BayesFlow": "bounded transforms in the adapter (`.constrain`), so that every draw "
                                          "respects the prior bounds"},
            {"Design choice": "Training data",
             "Here": "a fixed set of simulations (`fit_offline`), one noise draw per pair",
             "Alternatives in BayesFlow": "fresh simulations in every batch (`fit_online`)"},
            {"Design choice": "Amortized over",
             "Here": "quotes for these 63 contracts and these spreads",
             "Alternatives in BayesFlow": "contract terms and spreads as extra inputs, to reuse one network across "
                                          "days"},
        ], nowrap=True)}),
        bayesflow("""
        The workflow of the pretrained estimator has three parts. The adapter names the network's inputs. `FlowMatching` is the inference network. `BasicWorkflow` ties them together, standardizes the parameters and runs the training.
        """),
    ])
    return


@app.cell
def _(NAMES, bf):
    def build_workflow():
        """The BayesFlow workflow of the pretrained estimator."""
        adapter = (
            bf.Adapter()
            .concatenate(NAMES, into="inference_variables")   # θ: the six parameters
            .rename("observations", "inference_conditions")    # x̃: the whitened quotes
        )
        network = bf.networks.FlowMatching(subnet_kwargs={"widths": (128,) * 3, "dropout": 0.1})
        return bf.BasicWorkflow(adapter=adapter, inference_network=network, standardize="inference_variables")

    return (build_workflow,)


@app.cell(hide_code=True)
def _(EVALUATED, TRAINING, bayesflow, caption, md, mo, note):
    _power = lambda _x: (lambda _m, _e: rf"${_m}\cdot10^{{{int(_e)}}}$")(*f"{_x:.0e}".split("e"))
    _machine = ("an Apple-silicon Mac" if TRAINING["platform"].startswith("macOS") and "arm64" in TRAINING["platform"]
                else TRAINING["platform"])
    _noise = ("each with fresh noise in every epoch" if TRAINING["fresh_noise"] else
              "each with one noise draw kept throughout training")
    _evaluated = EVALUATED["population"]
    mo.vstack([
        md(rf"""
        ### 3.2 Training the pretrained estimator

        The pretrained estimator was trained once, with the settings of its training record.

        | Setting | Value |
        |:---|:---|
        | Training data | {TRAINING['rows']:,} simulated pairs, {_noise} |
        | Validation data | {TRAINING['validation_rows']:,} simulated pairs |
        | Schedule | {TRAINING['epochs']} epochs with batch size {TRAINING['batch_size']}, {TRAINING['updates']:,} updates |
        | Optimizer | {TRAINING['optimizer']} with weight decay {TRAINING['weight_decay']:g} and gradient clipping at {TRAINING['clipnorm']:g}, BayesFlow's default for offline training |
        | Learning rate | warm-up from {_power(TRAINING['rate_start'])} to {_power(TRAINING['rate_peak'])} over the first {TRAINING['warmup_updates']:,} updates, then cosine decay to zero |
        | Random seed | {TRAINING['seed']} |
        | Hardware and time | the CPU of {_machine}, {TRAINING['seconds'] / 60:.0f} min |
        """),
        bayesflow("Training is one call, `workflow.fit_offline`, on the simulated pairs."),
        md(rf"""
        ```python
        task, data = delivery.training_inputs(BUNDLE, rows={TRAINING['rows']:_})   # (θ, x̃) pairs, x̃ already whitened
        keras.utils.set_random_seed({TRAINING['seed']})
        workflow = build_workflow()
        history = workflow.fit_offline(data=data["train"], validation_data=data["validation"], epochs={TRAINING['epochs']}, batch_size={TRAINING['batch_size']})
        ```
        """),
        caption("This call trained the pretrained estimator. It does not run in this notebook."),
        md(f"""
        After training, the estimator was evaluated once on {_evaluated['rows']} simulated data sets that it had not seen, with {_evaluated['draws_per_case']} posterior draws each. This **recorded evaluation** supplies the results of Sections 5.2 and 5.3.
        """),
        note("The parameters are standardized but not mapped onto their bounded prior ranges, so a small fraction "
             "of posterior draws falls just outside the prior. Such draws are kept, not clipped."),
    ])
    return


@app.cell(hide_code=True)
def _(caption, grid, md, mo, np, plots, plt):
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
        md(r"""
        The inference network is a **velocity field** $v_\phi(\mathbf z,t,\tilde{\mathbf x})$ on the six-dimensional parameter space, for times $t\in[0,1]$ (Lipman et al., 2023).
        """),
        grid({
            "Training": r"""
            1. Take a simulated pair $(\boldsymbol\theta,\tilde{\mathbf x})$, with $\boldsymbol\theta$ standardized.
            2. Draw noise $\mathbf z_1\sim\mathcal N(0,I)$ independently of $\boldsymbol\theta$, and a time $t$, more often near $t=0$.
            3. Form the point $\mathbf z_t=(1-t)\,\boldsymbol\theta+t\,\mathbf z_1$ on the straight line from $\boldsymbol\theta$ to $\mathbf z_1$.
            4. Fit $v_\phi(\mathbf z_t,t,\tilde{\mathbf x})$ to the velocity $\mathbf z_1-\boldsymbol\theta$ with a squared-error loss.
            """,
            "Sampling": r"""
            1. Draw $\mathbf z_1\sim\mathcal N(0,I)$.
            2. Integrate $d\mathbf z/dt=v_\phi(\mathbf z,t,\tilde{\mathbf x})$ from $t=1$ back to $t=0$ with an adaptive Runge–Kutta solver (Tsit5).
            3. Undo the standardization. The end point is one posterior draw of $\boldsymbol\theta$.
            """,
            "Why it works": "Many pairs pass through the same point, so the network learns the average velocity "
                            "there. Following this averaged field moves probability mass so that its distribution "
                            r"at each time $t$ is that of $\mathbf z_t$: noise at $t=1$, the posterior at $t=0$.",
            "Network inputs": r"The point $\mathbf z_t$ and the whitened quotes $\tilde{\mathbf x}$ enter an MLP of "
                              "three layers of 128 units. The time $t$ enters through a Fourier embedding that "
                              "modulates every layer.",
        }, columns=2),
        _fig,
        caption("The figure shows a toy posterior in two dimensions: the cloud at $t=1$ (noise), halfway, and at "
                "$t=0$ (posterior). The last panel shows straight training paths between simulated parameters "
                "(orange) and independent noise draws (blue)."),
    ])})
    return


@app.cell(hide_code=True)
def _(
    EVALUATED,
    MATH_LABELS,
    NAMES,
    caption,
    key,
    md,
    mo,
    note,
    np,
    table,
    times,
):
    def live_summary(train, validation, held_out, draws, history, seconds):
        """The text about one small training run. Every number in it is computed from that run."""
        loss, val = history.history["loss"][-1], history.history.get("val_loss", [np.nan])[-1]
        fit = ("The validation loss ends close to the training loss, so the network is not overfitting."
               if val <= 1.1 * loss else
               "The validation loss ends clearly above the training loss, so the network has started to overfit.")
        corr = [float(np.corrcoef(np.median(draws[name][..., 0], axis=1), held_out[name][:, 0])[0, 1]) for name in NAMES]
        pretrained = EVALUATED["population"]["mean_posterior_sd"]
        width = [float(np.mean(np.std(draws[name][..., 0], axis=1))) / pretrained[j] for j, name in enumerate(NAMES)]
        n_train, n_val, n_test = (len(data["observations"]) for data in (train, validation, held_out))
        epochs = len(history.history["loss"])
        rows = [{"Parameter": label, "Correlation of median and truth": f"{r:.2f}",
                 "Posterior SD, as a multiple of the pretrained estimator's": times(w)}
                for label, r, w in zip(MATH_LABELS, corr, width)]
        return {
            "training": md(f"#### Training\n\nThe run simulated {n_train:,} training and {n_val:,} validation data "
                           f"sets in {seconds['simulate']:.0f} s and trained for {epochs} epochs in "
                           f"{seconds['train']:.0f} s. The final training loss is {loss:.3f} and the validation loss "
                           f"{val:.3f}. {fit}"),
            "loss": note("The loss is the flow-matching error, not a likelihood. Only its trend and the gap between "
                         "the two curves matter."),
            "check": md(f"#### Check on {n_test} new data sets\n\nThe new network returns "
                        f"{draws[NAMES[0]].shape[1]} posterior draws for each of {n_test} simulated data sets that it "
                        f"has not seen ({seconds['sample']:.0f} s)."),
            "recovery": caption("Each dot is one data set: the true value against the posterior median, with a bar "
                                "for the central 68% interval. Dots near the diagonal mean that the parameter is "
                                "recovered."),
            "calibration": caption("If the network's uncertainty is right, each curve stays inside its grey band. "
                                   "The foldable section below explains how to read the plot."),
            "comparison": mo.vstack([
                key(f"**With {100 * n_train / EVALUATED['training_rows']:.0f}% of the training data, the posteriors "
                    f"are {times(min(width))} to {times(max(width))} times as wide.** This network saw {n_train:,} "
                    f"simulated pairs instead of {EVALUATED['training_rows']:,} and trained for {epochs} epochs "
                    f"instead of {EVALUATED['native_epoch']}."),
                table(rows, align="lrr"),
            ]),
        }

    return (live_summary,)


@app.cell(hide_code=True)
def _(bayesflow, controls, md, mo):
    live_train_button = mo.ui.run_button(label="Simulate 3,000 data sets and train a small network")
    mo.vstack([
        md(r"""
        ### 3.3 Train a small network yourself

        The button runs the same steps at a small scale, as a live check: it simulates, trains and tests the new network on fresh simulations.
        """),
        controls(live_train_button, hint="Press the button. The run takes about a minute on a CPU."),
        bayesflow("""
        The cell below is the whole loop: simulate, whiten, train with `fit_offline`, sample for new data sets and plot `bf.diagnostics`.
        """),
    ])
    return (live_train_button,)


@app.cell
def _(
    LABELS,
    NAMES,
    bf,
    build_workflow,
    caption,
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
    mo.stop(not live_train_button.value, caption("Press the button above. The run and its checks appear here."))
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
        _text["loss"],
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
def _(
    CASE_TITLES,
    CONDITIONAL_DRAWS,
    EVAL_SEED,
    NAMES,
    SNAPSHOT_DAY,
    TRUTHS,
    bayesflow,
    cards,
    md,
    mo,
    np,
    reference_summary,
    section,
):
    _like, _stressed = TRUTHS["synthetic-market-like"], TRUTHS["synthetic-broad"]
    _medians = [reference_summary["cases"]["market"]["parameters"][_n]["regenerated_q05_q50_q95"][1] for _n in NAMES[:5]]
    _apart = max(abs(_like[_j] / _medians[_j] - 1) for _j in range(5))  # distance from the SPX posterior medians
    mo.vstack([
        section("Infer", "The pretrained estimator returns posterior draws for observed quotes in seconds. A PyMC "
                "run with the exact likelihood provides the reference.", stage=4),
        md(r"""
        ### 4.1 Three data sets and two samplers

        We apply the pretrained estimator to three sets of 63 quotes.
        """),
        cards([
            ("Simulated", CASE_TITLES["synthetic-market-like"],
             f"The quotes are generated with $c={_like[5]:g}$ at parameters within {np.ceil(100 * _apart):.0f}% of the "
             "posterior medians for the SPX quotes."),
            ("Simulated", CASE_TITLES["synthetic-broad"],
             f"The quotes are generated with $c={_stressed[5]:g}$ at a high-volatility draw from the prior. Volatility "
             f"starts at {100 * np.sqrt(_stressed[0]):.0f}% and reverts to {100 * np.sqrt(_stressed[2]):.0f}%."),
            ("Real", CASE_TITLES["market"],
             f"These are the quotes of {SNAPSHOT_DAY}. The true parameters are unknown."),
        ], columns=3),
        bayesflow("Inference is one call, `workflow.sample`. The loader wraps it so that it takes raw quotes."),
        md(rf"""
        ```python
        estimator, record = delivery.load_checkpoint(BUNDLE, ESTIMATOR)
        draws = estimator.sample(quotes[None, :], {CONDITIONAL_DRAWS}, seed={EVAL_SEED})[0]   # ({CONDITIONAL_DRAWS}, 6) posterior draws
        # inside: x_tilde = whitening.transform(quotes / half_spread)
        #         workflow.sample(conditions={{"observations": x_tilde}}, num_samples={CONDITIONAL_DRAWS})
        ```

        Nothing is fitted at this stage. Each draw integrates the learned flow from one noise vector.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(BUNDLE, BUNDLE_HELP, ESTIMATOR, delivery, recipe, task, time):
    estimator, estimator_record, load_seconds = None, None, None
    _view = None
    if BUNDLE is None:
        _view = BUNDLE_HELP
    else:
        _started = time.perf_counter()
        estimator, estimator_record = delivery.load_checkpoint(BUNDLE, ESTIMATOR)
        load_seconds = time.perf_counter() - _started
        if recipe.fingerprint(estimator.task) != recipe.fingerprint(task):
            raise ValueError("the pretrained estimator was trained for a different contract list")
    _view
    return estimator, estimator_record, load_seconds


@app.cell(hide_code=True)
def _(
    CASES,
    CONDITIONAL_DRAWS,
    EVAL_SEED,
    TRAINING,
    estimator,
    estimator_record,
    load_seconds,
    mo,
    note,
    np,
    number_word,
    time,
):
    neural_draws = {}
    _view = None  # nothing to show without the pretrained estimator
    if estimator is not None:
        _seconds = lambda _t: f"{_t:.1f} s" if _t < 10 else f"{_t:.0f} s"
        _started = time.perf_counter()
        for _index, _case_id in enumerate(CASES):
            neural_draws[_case_id] = np.asarray(estimator.sample(CASES[_case_id][None], CONDITIONAL_DRAWS,
                                                                 seed=EVAL_SEED + _index)[0])
        _view = [note(f"The pretrained estimator loaded in {_seconds(load_seconds)}. It returned "
                      f"{CONDITIONAL_DRAWS:,} posterior draws for each of the {number_word(len(CASES))} data sets in "
                      f"{_seconds(time.perf_counter() - _started)}.", label="This run")]
        _trained = (estimator_record["train_rows"], estimator_record["native_epoch"])
        if _trained != (TRAINING["rows"], TRAINING["epochs"]):
            _view.append(note(f"The loaded estimator was trained on {_trained[0]:,} simulated pairs for {_trained[1]} "
                              f"epochs. The training record of Section 3.2 describes {TRAINING['rows']:,} pairs and "
                              f"{TRAINING['epochs']} epochs.", label="Check"))
        _view = mo.vstack(_view)
    _view
    return (neural_draws,)


@app.cell(hide_code=True)
def _(
    CASE_TITLES,
    REFERENCES,
    grid,
    json,
    md,
    mo,
    note,
    number_word,
    reference_draws,
    table,
):
    _limits = json.loads((REFERENCES / "reference-protocol.json").read_text())["settings"]
    _rows, _selected = [], []
    for _case_id in CASE_TITLES:
        _run = json.loads((REFERENCES / "references" / _case_id / "run.json").read_text())
        _stats = json.loads((REFERENCES / "references" / _case_id / "diagnostics.json").read_text())["parameters"]
        _attempt = _run["attempts"][_run["selected_attempt"]]
        _selected.append(_run["selected_attempt"])
        _rows.append({
            "Data set": CASE_TITLES[_case_id],
            "Chains × draws": f"{_run['settings']['chains']} × {_attempt['draws']:,} ({_attempt['tune']:,} tuning)",
            "Largest R-hat": f"{max(_s['r_hat'] for _s in _stats.values()):.4f}",
            "Smallest bulk ESS": f"{min(_s['ess_bulk'] for _s in _stats.values()):,.0f}",
            "Likelihood evaluations": f"{sum(_a['likelihood_evaluations'] for _a in _run['attempts']):,}",
            "Run time, one core": f"{sum(_a['wall_seconds'] for _a in _run['attempts']) / 60:.0f} min",
        })
    _short, _long = _run["settings"]["attempts"]  # (tuning steps, draws) of the short and of the long run
    _chains = _run["settings"]["chains"]
    _total = number_word(len(_selected))
    _needed = (f"All {_total} data sets needed the long run." if all(_s == 1 for _s in _selected) else
               f"{number_word(sum(_s == 1 for _s in _selected)).capitalize()} of the {_total} data sets needed the "
               "long run.")
    mo.vstack([
        md("A **PyMC reference** samples the same posterior by MCMC, with the exact likelihood. The network is judged "
           "against it in Sections 4.2 and 5.4."),
        mo.accordion({"How the PyMC reference is computed": mo.vstack([
            grid({
                "Target": r"PyMC uses the same uniform prior and the exact likelihood "
                          r"$y_i\sim\mathcal N\big(P_i(\boldsymbol\theta),(c\,h_i)^2\big)$. The prices come from the "
                          "Fourier pricer (feature 4).",
                "Sampler": "The sampler is PyMC's `DEMetropolisZ`, a gradient-free Metropolis sampler whose "
                           "proposals are differences of past draws (ter Braak and Vrugt, 2008). The likelihood "
                           "calls the pricer as a black box without gradients, which rules out NUTS.",
                "Runs": f"{number_word(_chains).capitalize()} chains start from maximum-likelihood (Powell) fits and "
                        "from dispersed points. None starts from the true parameters or from the network's draws. A "
                        f"short run of {_short[0]:,} tuning steps and {_short[1]:,} draws comes first. If it fails "
                        f"the checks, a long run of {_long[0]:,} tuning steps and {_long[1]:,} draws follows.",
                "Checks": f"Every parameter must reach an R-hat of at most {_limits['r_hat_max']:g} and bulk and "
                          f"tail effective sample sizes (ESS) of at least {_limits['ess_bulk_min']:,}. Prices are "
                          "accurate to 2% of the smallest noise standard deviation.",
            }, columns=2),
            md(f"The table reports the run that passed the checks, for each data set. {_needed}"),
            table(_rows, align="llrrrr", nowrap=True),
            note("Likelihood evaluations and run times include the short run. The plots use a thinned subset of "
                 f"{len(reference_draws['market']) // _chains:,} draws per chain."),
        ])}),
    ])
    return


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
def _(
    BOUNDS,
    CASES,
    CASE_TITLES,
    DELTA,
    EXPIRY,
    EXPIRY_COLORS,
    EXPIRY_DAYS,
    LABELS,
    MARKET_IV,
    MATH_LABELS,
    NAMES,
    PRODUCTS,
    PRODUCT_TITLES,
    TRUTHS,
    caption,
    decimals,
    expected_variance,
    fixed,
    grid,
    implied_vols,
    key,
    md,
    mo,
    neural_draws,
    np,
    pairs,
    plots,
    plt,
    reference_draws,
    reference_products,
    reference_summary,
    table,
    times,
):
    def _posterior_view(case):
        """Everything Section 4.2 shows for one data set."""
        neural, truth, reference = neural_draws.get(case), TRUTHS[case], reference_draws[case]
        summary = reference_summary["cases"][case]["parameters"]
        iv = implied_vols(CASES[case])[0]
        fig, ax = plt.subplots(figsize=(7.2, 3.0), constrained_layout=True)
        for e, days in enumerate(EXPIRY_DAYS):
            at = EXPIRY == e
            ax.plot(DELTA[at], 100 * iv[at], "o-", ms=3, lw=1, color=EXPIRY_COLORS[e], label=f"{days} days")
        ax.set(xlabel="standardized moneyness δ", ylabel="implied volatility (%)", title=CASE_TITLES[case])
        ax.legend(fontsize=7, ncol=2)

        # The six parameters: median and 90% interval, with the decimals that resolve the PyMC interval.
        rows = []
        for j, name in enumerate(NAMES):
            rq = summary[name]["regenerated_q05_q50_q95"]
            digits = decimals(rq[2] - rq[0])
            interval = lambda q: f"{fixed(q[1], digits)} [{fixed(q[0], digits)}, {fixed(q[2], digits)}]"
            row = {"Parameter": MATH_LABELS[j]}
            if truth is not None:  # the SPX quotes have no known truth
                row["Truth"] = fixed(truth[j], digits)
            if neural is not None:
                row["Network, median [90% interval]"] = interval(np.quantile(neural[:, j], [.05, .5, .95]))
            row["PyMC reference, median [90% interval]"] = interval(rq)
            rows.append(row)

        # Two derived quantities: the expected average variance over a horizon, as a volatility in %.
        horizons, spread, covered = [], [], []
        for k, (product, horizon) in enumerate(PRODUCTS.items()):
            pymc = 100 * np.sqrt(np.quantile(reference_products[case][:, k], [.05, .5, .95]))
            digits = decimals(pymc[2] - pymc[0])
            interval = lambda q: f"{fixed(q[1], digits)} [{fixed(q[0], digits)}, {fixed(q[2], digits)}]"
            row = {"Horizon": PRODUCT_TITLES[product]}
            if truth is not None:
                value = 100 * np.sqrt(expected_variance(truth, horizon))
                row["Truth"] = fixed(value, digits)
                covered.append(pymc[0] <= value <= pymc[2])
            if neural is not None:
                network = 100 * np.sqrt(np.quantile(expected_variance(neural[neural[:, 1] > 0], horizon), [.05, .5, .95]))
                row["Network, median [90% interval]"] = interval(network)
                spread.append((network[2] - network[0]) / (pymc[2] - pymc[0]))
            row["PyMC reference, median [90% interval]"] = interval(pymc)
            horizons.append(row)
        learned = []
        if spread:
            learned.append(f"The network's 90% intervals for the two horizons are {times(spread[0])} and "
                           f"{times(spread[1])} times as wide as PyMC's.")
        if covered:
            learned.append("Both true values lie inside PyMC's intervals." if all(covered) else
                           "Neither true value lies inside PyMC's interval." if not any(covered) else
                           "One of the two true values lies outside PyMC's interval.")

        sets = {"PyMC reference": reference}
        if neural is not None:
            sets = {"network posterior": neural, **sets}
        corner = pairs(sets, LABELS, truth=truth, figsize=(8.6, 8.6), points=1200)
        if neural is None:  # shown alone, PyMC keeps the colour it has next to the network
            for axis in corner.axes:
                for artist in list(axis.patches) + list(axis.collections[:1]):
                    artist.set_color(plots.SERIES[1])
        first_axis = corner.axes[0]
        handles, legend_labels = first_axis.get_legend_handles_labels()
        if first_axis.get_legend() is not None:  # a panel-anchored legend squeezes the first column of a 6 x 6 grid
            first_axis.get_legend().remove()
            corner.legend(handles, legend_labels, loc="upper right", fontsize=8, frameon=False)
        for axis in corner.axes:
            axis.tick_params(labelsize=6.5)
        for axis in corner.axes[-len(LABELS):]:
            for tick in axis.get_xticklabels():
                tick.set_rotation(35)

        corr = np.corrcoef(reference, rowvar=False)
        hi, lo = max(((i, j) for i in range(6) for j in range(i)), key=lambda pair: abs(corr[pair]))
        width90 = lambda name: summary[name]["regenerated_q05_q50_q95"][2] - summary[name]["regenerated_q05_q50_q95"][0]
        relative = {name: width90(name) / (2 * abs(summary[name]["regenerated_q05_q50_q95"][1])) for name in NAMES[:5]}
        sharpest, loosest = min(relative, key=relative.get), max(relative, key=relative.get)
        named = lambda name: MATH_LABELS[NAMES.index(name)]
        capital = lambda text: text[0].upper() + text[1:]
        signed = lambda x: f"${x:+.2f}$"

        # The data set, described next to its smile.
        detail = None
        if truth is None:
            scenario = ("These are the SPX quotes shown at the start of the notebook. There is no known truth, and "
                        "the model itself may be wrong (Section 5.5).")
        elif case == "synthetic-broad":
            shortest, longest = (EXPIRY == 0), (EXPIRY == len(EXPIRY_DAYS) - 1)
            atm = [100 * iv[at][np.argmin(np.abs(DELTA[at]))] for at in (shortest, longest)]
            deep = [100 * iv[EXPIRY == e][np.argmin(DELTA[EXPIRY == e])] for e in range(len(EXPIRY_DAYS))]
            # Rise of the smile from the money to the deepest put, at the first and last expiry.
            rise = lambda vols: [100 * (vols[at][np.argmin(DELTA[at])] - vols[at][np.argmin(np.abs(DELTA[at]))])
                                 for at in (shortest, longest)]
            case_rise, spx_rise = rise(iv), rise(MARKET_IV)
            v0, v_inf, xi, rho = truth[[0, 2, 3, 4]]
            scenario = (f"These quotes are simulated from Heston with $c={truth[5]:g}$, in a stressed market. At the "
                        f"money, implied volatility is {atm[0]:.0f}% at {EXPIRY_DAYS[0]} days against {atm[1]:.0f}% at "
                        f"{EXPIRY_DAYS[-1]} days. The term structure is inverted, as after a sell-off. Deep puts are "
                        f"at {min(deep):.0f}–{max(deep):.0f}%.")
            detail = grid({
                "Inverted term structure":
                    f"The true initial volatility $\\sqrt{{v_0}}={100 * np.sqrt(v0):.0f}\\%$ is above the long-run "
                    f"$\\sqrt{{v_\\infty}}={100 * np.sqrt(v_inf):.0f}\\%$. The large $\\sigma_v={xi:.2f}$ pulls "
                    "long-expiry at-the-money volatility further down.",
                "Crossing curves":
                    "Heston's short-expiry skew per unit of log-moneyness tends to the ceiling of feature 3, "
                    f"$|\\rho|\\sigma_v/(4\\sqrt{{v_0}})={abs(rho) * xi / (4 * np.sqrt(v0)):.2f}$. Standardized "
                    "moneyness rescales log-moneyness by $\\sqrt T$, so short-expiry smiles are flat in $\\delta$. "
                    f"From the money to the deepest put they rise {case_rise[0]:.1f} points at {EXPIRY_DAYS[0]} days "
                    f"and {case_rise[1]:.1f} at {EXPIRY_DAYS[-1]} days. The short smiles are high and flat, the long "
                    "ones low and steep.",
                "Contrast with SPX":
                    f"On SPX the same rise is {spx_rise[0]:.1f} and {spx_rise[1]:.1f} points, much closer to parallel. "
                    'The <a class="nb-ref" href="#what-heston-misses">closing section</a> analyses this limit of '
                    "Heston. As a test of the inference the data set is still valid, because it is a draw from the "
                    "model and the prior.",
            }, kind="reading")
        else:
            scenario = (f"These quotes are simulated from Heston with $c={truth[5]:g}$. The model is right by "
                        "construction, so this data set tests the inference alone.")

        # The corner plot, read in up to three columns: the PyMC reference, the truth, the network against PyMC.
        reading = ("The diagonal shows each parameter's posterior"
                   + (", the network in blue and PyMC in orange" if neural is not None else " from PyMC")
                   + ". The panels below show pairs of parameters."
                   + (" Black lines and stars mark the true values." if truth is not None else ""))
        columns = {"PyMC reference": [
            f"- {capital(named(sharpest))} is determined to within ±{100 * relative[sharpest]:.2g}% of its value, "
            f"and the least precise Heston parameter, {named(loosest)}, to within ±{100 * relative[loosest]:.2g}%.",
            f"- The strongest dependence is between {MATH_LABELS[lo]} and {MATH_LABELS[hi]} (correlation "
            f"{signed(corr[hi, lo])}).",
        ]}
        if truth is None:
            columns["PyMC reference"].append(
                f"- The noise multiplier $c$ sits at the top of its prior range (median "
                f"{summary['c']['regenerated_q05_q50_q95'][1]:.4f}, bound {BOUNDS[-1, 1]:g}). The quotes ask for more "
                "noise than the model allows, because no Heston smile comes close enough (Section 5.5).")
        else:
            # Where does the truth sit in the reference posterior? Marginally (in posterior SDs) and jointly (Mahalanobis).
            centre, sd = reference.mean(axis=0), reference.std(axis=0, ddof=1)
            z = (truth - centre) / sd
            precision = np.linalg.inv(np.cov(reference, rowvar=False))
            distance = lambda x: np.einsum("...i,ij,...j->...", x - centre, precision, x - centre)
            joint = 100 * np.mean(distance(reference) < distance(truth))
            outside = [MATH_LABELS[k] for k in range(6) if abs(z[k]) > 1.645]
            if len(outside) > 1:
                where = (f"- The true values of {', '.join(outside[:-1])} and {outside[-1]} lie outside their 90% "
                         f"intervals, up to {np.max(np.abs(z)):.1f} posterior SDs from the posterior mean.")
            elif outside:
                where = (f"- The true value of {outside[0]} lies outside its 90% interval, {np.max(np.abs(z)):.1f} "
                         "posterior SDs from the posterior mean.")
            else:
                where = "- Every true value lies inside its 90% interval."
            columns["Truth"] = [where, f"- Jointly, the truth is farther from the posterior centre than {joint:.0f}% "
                                       "of the draws."]
        verdict = None
        if neural is not None:
            # The same measures as the table of Section 5.4: centre gap in PyMC SDs, and the ratio of 90% widths.
            gaps = np.array([(neural[:, k].mean() - summary[name]["regenerated_mean"]) / summary[name]["regenerated_sd"]
                             for k, name in enumerate(NAMES)])
            widths = np.array([np.subtract(*np.quantile(neural[:, k], [.95, .05])) / width90(name)
                               for k, name in enumerate(NAMES)])
            far, wide = int(np.argmax(np.abs(gaps))), int(np.argmax(widths))
            columns["Network against PyMC"] = [
                f"- The network's centres differ from PyMC's by up to {times(np.max(np.abs(gaps)), 2)} PyMC SDs, most "
                f"for {MATH_LABELS[far]}.",
                f"- Its 90% intervals are {times(widths.min())} to {times(widths.max())} times as wide as PyMC's, "
                f"widest for {MATH_LABELS[wide]}.",
                f"- For {MATH_LABELS[lo]} with {MATH_LABELS[hi]}, its correlation is "
                f"{signed(np.corrcoef(neural, rowvar=False)[hi, lo])}, against PyMC's {signed(corr[hi, lo])}.",
            ]
            if truth is None:
                verdict = ("**The network and PyMC do not agree.** The SPX quotes lie outside anything the network "
                           "saw in training, so it extrapolates. Section 5.5 traces the disagreement to the model.")
            elif np.max(np.abs(gaps)) < 1 and widths.min() >= 1:
                verdict = ("**The network is close to PyMC, but wider.** One network that serves every data set is "
                           "less sharp than MCMC run on this one. Its wider intervals err on the cautious side, in "
                           "line with the calibration checks of Section 5.2.")
            elif np.max(np.abs(gaps)) < 1:
                verdict = ("**The network is close to PyMC.** Every centre lies within one PyMC SD of the "
                           f"reference's, and its intervals are {times(widths.min())} to {times(widths.max())} times "
                           "as wide.")
            else:
                verdict = ("**The network's posterior is shifted relative to PyMC's.** For "
                           f"{MATH_LABELS[far]} its centre lies {times(np.max(np.abs(gaps)), 2)} PyMC SDs away.")

        parts = [mo.hstack([
            mo.vstack([fig, caption("The 63 quotes of this data set as implied volatilities. Each colour is one "
                                    "expiry.")]),
            md(scenario),
        ], widths=[3, 2], gap=2, align="center")]
        if detail is not None:
            parts.append(detail)
        parts += [
            md("#### The six parameters\n\nThe table gives the posterior median and the 90% interval of each "
               "parameter."),
            table(rows, align="l" + "r" * (len(rows[0]) - 1)),
            md("#### The joint posterior"),
            corner,
            caption(reading),
        ]
        if verdict is not None:
            parts.append(key(verdict))
        parts += [
            grid({title: "\n".join(lines) for title, lines in columns.items()}, kind="reading",
                 columns=max(2, len(columns))),
            md("#### Derived quantities\n\nEach posterior draw gives an expected average variance over a horizon, the "
               "fair rate of a variance swap. The table shows it as a volatility in %, for two horizons. This is "
               "feature 1 averaged over the horizon."),
            table(horizons, align="l" + "r" * (len(horizons[0]) - 1)),
        ]
        if learned:
            parts.append(md(" ".join(learned)))
        return mo.vstack(parts)

    _lead = ("Each data set gets two posteriors, one from the network and one from PyMC." if neural_draws else
             "Each data set gets its PyMC posterior. The network's posterior needs the pretrained estimator.")
    mo.vstack([
        md(f"""
        ### 4.2 The posterior, one data set at a time

        {_lead} The tabs switch the data set.
        """),
        mo.ui.tabs({CASE_TITLES[_case]: _posterior_view(_case) for _case in CASE_TITLES}),
        mo.accordion({"How to read the joint posterior": grid({
            "Combinations of parameters": "The quotes determine combinations of parameters better than each one "
                                          "alone. A quantity that depends on several parameters must be computed "
                                          "from the joint draws, not from the marginals.",
            "A truth outside its interval": "One noisy data set can put a true value outside its 90% interval. The "
                                            "noise shifts the posterior, and strongly correlated parameters shift "
                                            "together.",
            "Joint distance of the truth": "For a correct posterior, the share of draws closer to the centre than "
                                           "the truth is uniform between 0 and 100%. A single large share is "
                                           "therefore no warning sign. Coverage is judged over many data sets in "
                                           "Section 5.2.",
        })}),
    ])
    return


@app.cell(hide_code=True)
def _(
    CASE_TITLES,
    EVALUATED,
    REFERENCES,
    TRAINING,
    caption,
    json,
    key,
    md,
    mo,
    note,
    np,
    plots,
    plt,
    prior_batch,
    prior_seconds,
    table,
):
    # Every number below is read from a run record or timed in this notebook.
    _rows, _train = TRAINING["rows"], TRAINING["seconds"]
    # One entry per data set, in the order of the table of Section 4.1: the short and the long run together.
    _pymc = np.array([sum(_a["wall_seconds"] for _a in json.loads(
        (REFERENCES / "references" / _case / "run.json").read_text())["attempts"]) for _case in CASE_TITLES])
    # The recorded evaluation of the pretrained estimator: every posterior it drew, with loading and diagnostics included.
    _evaluation = EVALUATED
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

    _total = sum(_posteriors.values())
    mo.vstack([
        md(r"""
        ### 4.3 What it costs

        The network is paid for once. PyMC is paid again for every data set.
        """),
        _fig,
        caption("The left panel shows the time of each step on a logarithmic scale. Hatched bars are paid once. The "
                "right panel shows the total time for many data sets."),
        key(f"**Simulating and training take about {_duration(_once)}, once.** One PyMC run takes "
            f"{_duration(_pymc.min())} to {_duration(_pymc.max())}. After training, a network posterior takes under "
            f"{np.ceil(10 * _net) / 10:.1f} s."),
        md(f"The estimator's recorded evaluation drew {_total} posteriors in {_evaluation['seconds']:.0f} s, {_check} "
           f"of them for Section 5.2 with {_evaluation['population']['draws_per_case']} draws each. PyMC would need "
           f"about {_duration(_check * _pymc.mean())} for those {_check} data sets."),
        mo.accordion({"Where the numbers come from": mo.vstack([
            table([
                {"Step": "Simulation",
                 "Source": f"Section 2.2 timed {len(prior_batch['observations']):,} simulations on one core of this "
                           f"machine. The figure scales that time to {_rows:,}. The cost per simulation varies with "
                           "the parameters drawn, so this is approximate."},
                {"Step": "Training",
                 "Source": f"The training record gives {_duration(_train)} of wall time on one CPU, loading the data "
                           "and saving the network included."},
                {"Step": "Network posterior",
                 "Source": f"The recorded evaluation drew {_total} posteriors in {_evaluation['seconds']:.0f} s, "
                           f"diagnostics included: {_posteriors['Section 5.2']} for Section 5.2, "
                           f"{_posteriors['validation']} for validation and {_posteriors['Section 4']} for Section "
                           f"4.2. That is {_net:.2f} s each on average. A posterior with more draws takes longer, as "
                           "the timing in Section 4.1 shows."},
                {"Step": "PyMC posterior",
                 "Source": f"The reference runs of Section 4.1 took {', '.join(_duration(_s) for _s in _pymc)} on "
                           "one core, in the order of its table."},
            ], nowrap=True),
            note("The steps ran on different machines. Compare orders of magnitude, not exact ratios."),
        ])}),
    ])
    return


@app.cell(hide_code=True)
def _(EXPIRY_DATES, EXPIRY_DAYS, controls, md, mo, section):
    expiry_choice = mo.ui.dropdown(
        options={f"{_days} days ({_date})": _e for _e, (_date, _days) in enumerate(zip(EXPIRY_DATES, EXPIRY_DAYS))},
        value=f"{EXPIRY_DAYS[4]} days ({EXPIRY_DATES[4]})", label="Expiry")
    mo.vstack([
        section("Check", "Five checks. The first asks whether model and posterior together regenerate the quotes. "
                "The next three test the network's posterior. The last tests the model on the SPX quotes.", stage=5),
        md(r"""
        | Check | Question | What to look for |
        |:---|:---|:---|
        | <a class="nb-ref" href="#s5-1">5.1 Posterior predictive check</a> | Can the inferred parameters regenerate the quotes? | quotes inside the replicated range |
        | <a class="nb-ref" href="#s5-2">5.2 Calibration across simulated data sets</a> | Does a 90% interval contain the truth 90% of the time? | coverage near nominal, uniform ranks |
        | <a class="nb-ref" href="#s5-3">5.3 Recovery across simulated data sets</a> | Does the posterior median find the true parameters? | points on the diagonal, short intervals |
        | <a class="nb-ref" href="#s5-4">5.4 Comparison with the PyMC reference</a> | Does the network reproduce the posterior of one data set? | agreement with PyMC |
        | <a class="nb-ref" href="#s5-5">5.5 Heston against the SPX quotes</a> | Can Heston describe the market at all? | residuals within the assumed noise |

        ### 5.1 Posterior predictive check

        Draw parameters from the posterior, price the 63 contracts and add fresh noise with each draw's own $c$.

        $$\boldsymbol\theta^{(m)}\sim q_\phi(\boldsymbol\theta\mid\mathbf y),\qquad \widetilde y^{(m)}_i=P_i(\boldsymbol\theta^{(m)})+c^{(m)}h_i\,\epsilon^{(m)}_i$$

        If the model and the posterior are right, the observed quotes look like one more replicated data set. The PyMC draws go through the same recipe.
        """),
        controls(expiry_choice, hint="Choose an expiry. The tabs below switch the data set."),
    ])
    return (expiry_choice,)


@app.cell(hide_code=True)
def _(
    BOUNDS,
    CASES,
    CASE_TITLES,
    CLEAN,
    DELTA,
    EXPIRY,
    EXPIRY_DAYS,
    H,
    caption,
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
    _shown = 40  # posterior draws priced for each band

    def _predictive_view(case):
        """The posterior predictive check of one data set, at the chosen expiry."""
        rng = np.random.default_rng(20261112)
        sets, dropped = {}, 0
        if case in neural_draws:
            everything = neural_draws[case]
            inside = everything[np.all((everything >= BOUNDS[:, 0]) & (everything <= BOUNDS[:, 1]), axis=1)]
            if len(inside) >= 2:
                sets["network posterior"] = inside[rng.choice(len(inside), min(_shown, len(inside)), replace=False)]
            dropped = len(everything) - len(inside)
        sets["PyMC reference"] = reference_draws[case][rng.choice(len(reference_draws[case]), _shown, replace=False)]
        y = CASES[case]
        at = EXPIRY == expiry_choice.value
        fig, axes = plt.subplots(2, len(sets), figsize=(5.2 * len(sets), 5.6), sharex=True, squeeze=False,
                                 constrained_layout=True)
        misfit, typical_c = None, None
        for col, (name, draws) in enumerate(sets.items()):
            prices = np.array([pilot.prices(row) for row in draws])
            replicated = prices + draws[:, -1:] * H * rng.standard_normal(prices.shape)
            model_iv = np.nanquantile(implied_vols(prices)[:, at], [.05, .5, .95], axis=0)
            rep_iv = np.nanquantile(implied_vols(replicated)[:, at], [.05, .5, .95], axis=0)
            top, bottom = axes[:, col]
            top.fill_between(DELTA[at], 100 * rep_iv[0], 100 * rep_iv[2], color="#9bd3ba", alpha=.55, label="replicated quotes (90%)")
            top.fill_between(DELTA[at], 100 * model_iv[0], 100 * model_iv[2], color="#7f53ad", alpha=.5, label="model prices (90%)")
            top.scatter(DELTA[at], 100 * implied_vols(y)[0][at], color="#222222", s=20, zorder=4, label="observed quotes")
            if case in CLEAN:
                top.plot(DELTA[at], 100 * implied_vols(CLEAN[case])[0][at], "--", color="#333333", lw=1, label="true Heston prices")
            top.set(title=name, ylabel="implied volatility (%)" if col == 0 else None)
            model_dev = np.quantile((prices - y)[:, at] / H[at], [.05, .95], axis=0)
            rep_dev = np.quantile((replicated - y)[:, at] / H[at], [.05, .95], axis=0)
            bottom.fill_between(DELTA[at], *rep_dev, color="#9bd3ba", alpha=.55)
            bottom.fill_between(DELTA[at], *model_dev, color="#7f53ad", alpha=.5)
            bottom.axhline(0, color="#222222", lw=1)
            if case in CLEAN:
                bottom.plot(DELTA[at], ((CLEAN[case] - y) / H)[at], "--", color="#333333", lw=1)
            bottom.set(xlabel="standardized moneyness δ", ylabel="model − observed, in half-spreads" if col == 0 else None)
            if name == "PyMC reference":
                misfit = float(np.sqrt(np.mean((((prices.mean(axis=0) - y) / H)[at]) ** 2)))
                typical_c = float(np.median(draws[:, -1]))
        axes[0, 0].legend(fontsize=6.5, loc="upper right")
        verdict = ("**The quotes look like one more replicated data set, as they should when the model is right.** The "
                   f"posterior-mean prices miss them by {misfit:.1f} half-spreads (root mean square, RMS), in line "
                   f"with the inferred noise multiplier $c\\approx{typical_c:.2f}$."
                   if misfit < 2.5 * typical_c else
                   "**The model cannot reproduce these quotes.** The posterior-mean prices miss them by "
                   f"{misfit:.1f} half-spreads (root mean square, RMS). The inferred noise multiplier "
                   f"$c\\approx{typical_c:.2f}$ allows far less, and the miss has a systematic pattern across "
                   "strikes (Section 5.5).")
        columns = {
            "Implied volatilities": "The top row shows implied volatilities. The bands are so narrow that they "
                                    "hide behind the points.",
            "Residuals in half-spreads": "The bottom row shows the same prices minus the observed quotes, in "
                                         "half-spreads. Each quote carries its own noise, so the model prices "
                                         "(purple) differ from it by about $c$ half-spreads. The replicated quotes "
                                         "(green) add fresh noise and should cover the zero line.",
        }
        if case in CLEAN:
            columns["True prices"] = ("The dashed line marks the true, noise-free prices. The purple band follows "
                                      "them closely. The posterior recovers the true prices more precisely than any "
                                      "single quote measures them.")
        parts = [
            fig,
            caption(f"The figure shows the {CASE_TITLES[case] if case == 'market' else CASE_TITLES[case].lower()} at "
                    f"the {EXPIRY_DAYS[expiry_choice.value]}-day expiry. The bands are pointwise 90% ranges over "
                    f"{_shown} posterior draws."
                    + (" Network draws outside the prior range are left out, because the pricer is checked on the "
                       "prior's range." if case in neural_draws else "")),
            key(verdict),
            grid(columns, kind="reading"),
        ]
        if dropped:
            parts.append(note(f"Of the {len(neural_draws[case]):,} network draws for this data set, {dropped:,} lie "
                              "outside the prior range."))
        return mo.vstack(parts)

    mo.ui.tabs({CASE_TITLES[_case]: _predictive_view(_case) for _case in CASE_TITLES})
    return


@app.cell(hide_code=True)
def _(grid, md, mo):
    mo.vstack([
        md(r"""
        ### 5.2 Calibration across simulated data sets

        One posterior can look plausible and still be miscalibrated. Data sets with a known truth test the uncertainty statements themselves, in two ways.
        """),
        grid({
            "Coverage": "A 90% interval should contain the truth in about 90% of the data sets. Much more means the "
                        "intervals are too wide, much less that they are too narrow.",
            "Simulation-based calibration": "If the posterior is correct, the truth is one more posterior draw "
                                            "(Talts et al., 2018). Its **rank** among the draws, the fraction of "
                                            "draws below it, is then uniform between 0 and 1. The calibration plot "
                                            "shows the empirical CDF of the ranks minus the uniform CDF.",
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
def _(EVALUATED, caption, ecdf_band, grid, mo, norm, np, plots, plt):
    _rng = np.random.default_rng(7)
    _n = EVALUATED["population"]["rows"]  # as many data sets as the recorded evaluation
    _level = 0.95
    _truth = _rng.standard_normal(_n)
    _data = _truth + 0.5 * _rng.standard_normal(_n)
    _mean, _sd = 0.8 * _data, np.sqrt(0.2)  # exact posterior for a N(0, 1) prior and N(truth, 0.5^2) data
    _cases = {"calibrated": (0.0, 1.0), "biased upwards": (0.6, 1.0), "too narrow (overconfident)": (0.0, 0.6),
              "too wide (cautious)": (0.0, 1.7)}
    _z, _low, _high = ecdf_band(_n, level=_level)  # one curve per panel: a band for each
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
        caption(f"The figure shows four synthetic examples of the calibration plot, each from {_n} data sets. A "
                f"calibrated curve stays inside the grey band {100 * _level:.0f}% of the time."),
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
def _(grid, mo, stack, table):
    ecdf_guide = mo.accordion({"How to read the calibration (ECDF) plots": mo.vstack([
        grid([
            stack({
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
            }),
            stack({
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
            }),
        ]),
        grid({"Reading the shapes": table([
            {"Shape of the curve": "inside the band", "What it means": "The uncertainty is right."},
            {"Shape of the curve": "up on the left, down on the right",
             "What it means": "Ranks pile up near 0 and 1, so the truth often falls outside the bulk of the draws. "
                              "The posterior is **too narrow** (overconfident)."},
            {"Shape of the curve": "down on the left, up on the right",
             "What it means": "Ranks pile up around 0.5, so the truth sits in the middle too often. The posterior "
                              "is **too wide** (cautious)."},
            {"Shape of the curve": "an arch above zero",
             "What it means": "Ranks pile up near 0, so most draws lie above the truth. The posterior is **biased "
                              "upwards**. An arch below zero means biased downwards."},
        ], nowrap=True)}, columns=1),
        grid({"What it cannot tell you": grid([
            "**Whether the posterior is sharp.** A network that ignored the quotes and returned the prior would "
            "pass. The recovery plot (Section 5.3) and the PyMC reference (Section 5.4) check sharpness.",
            "**Whether the model is right.** The test data come from the same simulator. Section 5.5 and the "
            '<a class="nb-ref" href="#what-heston-misses">closing section</a> test the model on the SPX quotes.',
        ])}, columns=1),
    ])})
    ecdf_guide
    return (ecdf_guide,)


@app.cell(hide_code=True)
def _(
    BOUNDS,
    EVALUATED,
    LABELS,
    MATH_LABELS,
    NAMES,
    caption,
    ecdf_band,
    grid,
    key,
    md,
    mo,
    note,
    np,
    number_word,
    plots,
    plt,
    table,
):
    _population = EVALUATED["population"]
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
    _capital = lambda _text: _text[0].upper() + _text[1:]
    _rows = []
    for _j, _label in enumerate(MATH_LABELS):
        _ci = _coverage["0.9"]["wilson"][_j]
        _rows.append({
            "Parameter": _label,
            "90% coverage [95% CI]": f"{100 * _coverage['0.9']['coverage'][_j]:.1f}% [{100 * _ci[0]:.1f}, {100 * _ci[1]:.1f}]",
            "50 / 80 / 95% coverage": " / ".join(f"{100 * _coverage[_level]['coverage'][_j]:.0f}%" for _level in ("0.5", "0.8", "0.95")),
            "Calibration plot": "inside the band" if _passed[_j] else "leaves the band",
            "Draws outside the prior": f"{100 * _support['outside_prior']['per_dimension_count'][_j] / _support['samples']:.2f}%",
        })
    # Everything below is read off the saved evaluation.
    _cis = _coverage["0.9"]["wilson"]
    _good = [_j for _j in range(len(NAMES)) if _passed[_j] and _cis[_j][0] <= 0.9 <= _cis[_j][1]]
    _bad = [_j for _j in range(len(NAMES)) if _j not in _good]
    _lesson, _exceptions = {}, []
    for _j in _bad:
        _diff = lambda _x: np.mean(_ranks[:, _j] <= _x) - _x
        _left, _right = _diff(0.25), _diff(0.75)
        _kind = "too wide" if _left < 0 < _right else "too narrow" if _right < 0 < _left else "biased"
        _shape = {"too wide": "the too-wide shape, below zero on the left and above zero on the right",
                  "too narrow": "the too-narrow shape, above zero on the left and below zero on the right",
                  "biased": "the shape of a bias, on one side of zero throughout"}[_kind]
        _cov = _coverage["0.9"]["coverage"][_j]
        _text = (f"Its 90% intervals contain the truth {100 * _cov:.1f}% of the time, and its calibration curve has "
                 f"{_shape}.")
        if _cov > 0.9 and _kind == "too wide":
            _text += " The network is too cautious about this parameter."
        if NAMES[_j] == "initial_variance":
            _text += (" It may not resolve the parameter finely enough. The shortest quotes determine the initial "
                      f"variance to an average posterior SD of {_population['mean_posterior_sd'][_j]:.5f}, on a prior "
                      f"range of {BOUNDS[_j, 1] - BOUNDS[_j, 0]:.2f}.")
        _lesson[_capital(MATH_LABELS[_j])] = _text
        _exceptions.append((MATH_LABELS[_j], _kind))
    # What the takeaways say about calibration.
    calibration_summary = {"good": len(_good), "exceptions": _exceptions}
    _counts = np.asarray(_support["outside_prior"]["per_dimension_count"])
    _top = [MATH_LABELS[_k] for _k in np.argsort(_counts)[::-1][:2]]
    _view = [
        md(f"#### Results for the pretrained estimator\n\nThe estimator's recorded evaluation covers {len(_ranks)} "
           f"simulated data sets, with {_population['draws_per_case']} posterior draws each."),
        table(_rows, align="lrrlr"),
        _fig,
        caption("The figure shows the calibration plots of the six parameters. The grey band holds for all six "
                "together: a correct network keeps all six curves inside it 95% of the time (Säilynoja et al., "
                "2022). The band narrows at 0 and 1, where a correct curve varies least."),
        key(f"**{_capital(number_word(len(_good)))} of the {number_word(len(NAMES))} parameters are well "
            "calibrated.** The 95% confidence interval of their 90% coverage includes 90%, and their calibration "
            "curves stay inside the band."),
    ]
    if _lesson:
        _view.append(grid(_lesson, kind="reading", columns=max(2, len(_lesson))))
    _view.append(note(f"Of all posterior draws, {100 * _support['outside_prior']['sample_fraction']:.1f}% fall "
                      f"slightly outside the prior range, most often in {_top[0]} and {_top[1]}. They are kept in "
                      "every check."))
    mo.vstack(_view)
    return (calibration_summary,)


@app.cell(hide_code=True)
def _(EVALUATION, np):
    with np.load(EVALUATION / "recovery.npz", allow_pickle=False) as _archive:
        recovery_truth = _archive["truth"].astype(float)       # (512, 6) true parameters
        recovery_q = _archive["quantiles"].astype(float)       # (3, 512, 6): 5%, 50% and 95% posterior quantiles
    return recovery_q, recovery_truth


@app.cell(hide_code=True)
def _(
    BOUNDS,
    MATH_LABELS,
    NAMES,
    TEXT_LABELS,
    controls,
    key,
    md,
    mo,
    np,
    number_word,
    recovery_q,
    recovery_truth,
    table,
):
    recovery_parameter = mo.ui.dropdown(options={_label: _j for _j, _label in enumerate(TEXT_LABELS)},
                                        value=TEXT_LABELS[-1], label="Parameter")
    # How well each parameter is recovered: the median against the truth, and the 90% interval against the prior range.
    _corr = [float(np.corrcoef(recovery_q[1, :, _j], recovery_truth[:, _j])[0, 1]) for _j in range(len(NAMES))]
    _share = [float(100 * np.mean(recovery_q[2, :, _j] - recovery_q[0, :, _j]) / (BOUNDS[_j, 1] - BOUNDS[_j, 0]))
              for _j in range(len(NAMES))]
    _rows = [{"Parameter": MATH_LABELS[_j], "Correlation of median and truth": f"{_corr[_j]:.3f}",
              "Average 90% interval, share of the prior range": f"{_share[_j]:.1f}%"} for _j in range(len(NAMES))]
    _tight = [_j for _j in range(len(NAMES)) if _corr[_j] >= 0.99]
    _loose = [_j for _j in range(len(NAMES)) if _j not in _tight]
    _message = (f"**The posterior medians recover {'all' if not _loose else number_word(len(_tight))} "
                f"{'six' if not _loose else 'of the six'} parameters almost exactly.** For these the correlation "
                f"with the truth is at least {min(_corr[_j] for _j in _tight):.3f}, and the 90% intervals span at "
                f"most {max(_share[_j] for _j in _tight):.1f}% of the prior range."
                if _tight else
                "**No parameter is recovered almost exactly.** The correlation of posterior median and truth "
                f"stays below 0.99, at {min(_corr):.2f} to {max(_corr):.2f}.")
    for _j in _loose if _tight else []:
        _message += (f" For the {MATH_LABELS[_j]} the correlation is {_corr[_j]:.2f}, and the intervals span "
                     f"{_share[_j]:.0f}% of the prior range.")
    mo.vstack([
        md(f"""
        ### 5.3 Recovery across simulated data sets

        Recovery asks how close the posterior median comes to the true value, on the same {len(recovery_truth)} simulated data sets.
        """),
        table(_rows, align="lrr"),
        key(_message),
        controls(recovery_parameter, hint="Choose a parameter. Then drag a box around points in the plot to see the "
                                          "posterior of one of them."),
    ])
    return (recovery_parameter,)


@app.cell(hide_code=True)
def _(
    BOUNDS,
    LABELS,
    caption,
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
            title=f"{len(recovery_truth)} simulated data sets: truth against posterior")
    recovery_picker = mo.ui.matplotlib(_ax, debounce=True)
    mo.vstack([
        recovery_picker,
        caption("Each point is one simulated data set: its true value against the posterior median, with the 90% "
                "interval as a bar. A point on the dashed diagonal is recovered exactly."),
    ])
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
    bayesflow,
    bf,
    caption,
    decimals,
    estimator,
    fixed,
    held_out_quotes,
    md,
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
    _fresh = 2000  # posterior draws for the pair plot of the selected data set
    _x, _y = recovery_truth[:, _j], recovery_q[1, :, _j]
    _surprise = np.abs(_y - _x) / (recovery_q[2, :, _j] - recovery_q[0, :, _j])   # error in units of the 90% width
    _selected = np.flatnonzero(recovery_picker.value.get_mask(_x, _y)) if recovery_picker.value else np.array([], int)
    _pool = _selected if len(_selected) else np.arange(len(_x))
    _case = int(_pool[np.argmax(_surprise[_pool])])
    _how = (f"The box selects {len(_selected)} data sets. The table shows the one whose median is farthest from "
            "the truth, relative to its interval." if len(_selected) else
            "Nothing is selected yet. The table shows the data set whose median is farthest from the truth, "
            "relative to its interval.")
    _rows = []
    for _k in range(len(NAMES)):
        _digits = decimals(recovery_q[2, _case, _k] - recovery_q[0, _case, _k])
        _rows.append({
            "Parameter": MATH_LABELS[_k], "Truth": fixed(recovery_truth[_case, _k], _digits),
            "Posterior median [90% interval]": f"{fixed(recovery_q[1, _case, _k], _digits)} "
                                               f"[{fixed(recovery_q[0, _case, _k], _digits)}, "
                                               f"{fixed(recovery_q[2, _case, _k], _digits)}]",
            "Truth inside": "yes" if recovery_q[0, _case, _k] <= recovery_truth[_case, _k] <= recovery_q[2, _case, _k] else "no"})
    _parts = [md(f"#### One data set\n\nThis is data set {_case} of the recorded evaluation. {_how}"),
              table(_rows, align="lrrl")]
    if estimator is not None and held_out_quotes is not None:
        _draws = np.asarray(estimator.sample(held_out_quotes[_case][None], _fresh, seed=EVAL_SEED + 5000 + _case))[0]
        _grid = bf.diagnostics.pairs_posterior(_draws, targets=recovery_truth[_case], variable_names=LABELS, height=1.45,
                                               label_fontsize=10, tick_fontsize=7, legend_fontsize=9)
        _parts += [bayesflow(f"The pair plot below is `bf.diagnostics.pairs_posterior`, on {_fresh:,} fresh posterior "
                             "draws for this data set."),
                   getattr(_grid, "figure", None) or _grid.fig,
                   caption("The true values are marked in red and labelled Targets. Posterior shapes differ from "
                           "one data set to the next: compare data sets near the edges of the prior with data sets "
                           "in its middle.")]
    else:
        _parts.append(note("The pair plot of the selected data set needs the pretrained estimator."))
    mo.vstack(_parts)
    return


@app.cell(hide_code=True)
def _(PANEL_DRAWS, controls, md, mo, recovery_truth):
    panel_button = mo.ui.run_button(label=f"Resample the {len(recovery_truth)} data sets with the pretrained estimator")
    mo.vstack([
        md(f"#### Reproduce the results\n\nThe coverage table and the calibration plots of Section 5.2 and the "
           f"recovery plot above come from the estimator's recorded evaluation. The resampling button draws "
           f"{len(recovery_truth)} × {PANEL_DRAWS} fresh posterior draws and recomputes them."),
        controls(panel_button, hint="Press the button. Resampling takes a few minutes and needs the pretrained "
                                    "estimator."),
    ])
    return (panel_button,)


@app.cell(hide_code=True)
def _(
    BUNDLE,
    BUNDLE_HELP,
    EVALUATED,
    EVAL_SEED,
    LABELS,
    NAMES,
    PANEL_BATCH,
    PANEL_DRAWS,
    bayesflow,
    bf,
    caption,
    delivery,
    estimator,
    key,
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
    # Coverage of the fresh 90% intervals, against the coverage in the recorded evaluation.
    _bounds = np.quantile(_draws, [.05, .95], axis=1)
    _cover = 100 * np.mean((_bounds[0] <= _truth) & (_truth <= _bounds[1]), axis=0)
    _recorded = 100 * np.asarray(EVALUATED["population"]["coverage"]["0.9"]["coverage"])
    mo.vstack([
        bayesflow(f"The three plots are `bf.diagnostics.calibration_ecdf`, `recovery` and `coverage`, on "
                  f"{len(_quotes)} data sets × {PANEL_DRAWS} fresh posterior draws ({time.perf_counter() - _started:.0f} s)."),
        key(f"**The fresh draws give a 90% coverage of {_cover.min():.1f}% to {_cover.max():.1f}% across the six "
            f"parameters.** The recorded evaluation gave {_recorded.min():.1f}% to {_recorded.max():.1f}%, and the "
            f"largest difference for one parameter is {np.max(np.abs(_cover - _recorded)):.1f} points."),
        bf.diagnostics.calibration_ecdf(estimates=_estimates, targets=_targets, variable_keys=NAMES, variable_names=LABELS,
                                        difference=True, rank_ecdf_color=plots.SERIES[0], fill_color=plots.BASELINE),
        caption("Calibration: each curve should stay inside its grey band. Here every panel has its own band."),
        bf.diagnostics.recovery(estimates=_estimates, targets=_targets, variable_keys=NAMES, variable_names=LABELS,
                                uncertainty_agg_kwargs={"prob": 0.9}, color=plots.SERIES[0]),
        caption("Recovery: the true value against the posterior median, with the 90% interval as a bar."),
        bf.diagnostics.coverage(estimates=_estimates, targets=_targets, variable_keys=NAMES, variable_names=LABELS,
                                difference=True, color=plots.SERIES[0]),
        caption("Coverage at every level, as the difference from the nominal level. A calibrated network stays "
                "near zero."),
    ])
    return


@app.cell(hide_code=True)
def _(
    BOUNDS,
    CASE_TITLES,
    MATH_LABELS,
    NAMES,
    grid,
    key,
    md,
    mo,
    neural_draws,
    note,
    np,
    reference_summary,
    table,
    times,
):
    _heading = md(r"""
    ### 5.4 Comparison with the PyMC reference

    Calibration across many data sets does not show how sharp a single posterior is. The PyMC reference does, for the three data sets of Section 4.1. Two measures compare each network posterior with it.
    """)
    _measures = grid({
        "Centre gap": "The centre gap is the network's mean minus the reference mean, in reference SDs. Perfect "
                      "agreement is a gap of 0, and we count a gap below 0.3 as agreement.",
        "Width ratio": "The width ratio is the network's 90% interval divided by the reference's. Perfect agreement "
                       "is a ratio of 1, and we count a ratio between 0.75 and 1.25 as agreement.",
    })
    comparison = None  # what the takeaways say about this section, once the network is loaded
    if not neural_draws:
        _view = [_heading, _measures, note("This comparison needs the pretrained estimator.")]
    else:
        _number = lambda _x, _sign="": (f"{_x:{_sign}.2f}" if abs(_x) < 9.95 else f"{_x:{_sign},.0f}").replace("-", "−")
        _rows, _gaps, _ratios = [], {}, {}
        for _j, _name in enumerate(NAMES):
            _row = {"Parameter": MATH_LABELS[_j]}
            for _case_id in CASE_TITLES:
                _draws = neural_draws[_case_id][:, _j]
                _ref = reference_summary["cases"][_case_id]["parameters"][_name]
                _rq = _ref["regenerated_q05_q50_q95"]
                _q = np.quantile(_draws, [.05, .95])
                _gap = (_draws.mean() - _ref["regenerated_mean"]) / _ref["regenerated_sd"]
                _ratio = (_q[1] - _q[0]) / (_rq[2] - _rq[0])
                _gaps.setdefault(_case_id, []).append(abs(_gap))
                _ratios.setdefault(_case_id, []).append(_ratio)
                _row[(CASE_TITLES[_case_id], "centre gap")] = _number(_gap, "+")
                _row[(CASE_TITLES[_case_id], "width ratio")] = _number(_ratio)
            _rows.append(_row)
        _market = neural_draws["market"]
        _outside = np.mean(np.any((_market < BOUNDS[:, 0]) | (_market > BOUNDS[:, 1]), axis=1))
        _synthetic = ("synthetic-market-like", "synthetic-broad")
        _all_gaps = np.concatenate([_gaps[_c] for _c in _synthetic])
        _all_ratios = np.concatenate([_ratios[_c] for _c in _synthetic])
        _agree = int(np.sum((_all_gaps < 0.3) & (np.abs(_all_ratios - 1) < 0.25)))
        _wider, _narrower = int(np.sum(_all_ratios > 1.25)), int(np.sum(_all_ratios < 1))
        # "Cautious" means intervals at least as wide as the reference's around about the same centres.
        _cautious = (" The network is cautious here. It does not claim more precision than the data support, but "
                     "it does not extract all the information in the quotes either."
                     if _narrower == 0 and _all_gaps.max() < 1 else "")
        comparison = {"widest": float(_all_ratios.max())}
        _view = [
            _heading,
            _measures,
            table(_rows, align="l" + "r" * (len(_rows[0]) - 1)),
            key(f"**On the simulated data sets, {_agree} of {len(_all_gaps)} parameter posteriors "
                f"{'agrees' if _agree == 1 else 'agree'} with the reference on both counts.** The median centre gap is {np.median(_all_gaps):.2f} SD, and the "
                f"network's intervals are up to {times(_all_ratios.max())} times as wide as the reference's."),
            key("**A calibrated posterior can still be too wide.** A posterior equal to the prior is perfectly "
                "calibrated and useless. Only an exact reference shows how much sharpness is lost."),
            grid({
                "Simulated data sets":
                    f"The largest centre gap is {_all_gaps.max():.2f} SD. Of the {len(_all_ratios)} intervals, "
                    f"{_wider} are more than 25% wider than the reference's and "
                    f"{'none is' if _narrower == 0 else '1 is' if _narrower == 1 else f'{_narrower} are'} narrower."
                    f"{_cautious}",
                "SPX quotes":
                    f"On the SPX quotes the network's centres lie up to {times(max(_gaps['market']))} reference SDs "
                    f"from the reference's, and {100 * _outside:.0f}% of its draws fall outside the prior range "
                    "(Section 5.5).",
            }, kind="reading"),
        ]
    mo.vstack(_view)
    return (comparison,)


@app.cell(hide_code=True)
def _(
    BOUNDS,
    CASES,
    DELTA,
    EXPIRY,
    EXPIRY_DAYS,
    H,
    MARKET_IV,
    caption,
    grid,
    implied_vols,
    key,
    md,
    mo,
    neural_draws,
    np,
    number_word,
    pilot,
    plots,
    plt,
    reference_draws,
    times,
):
    _y = CASES["market"]
    _rng = np.random.default_rng(20261113)
    _shown = 40  # posterior draws priced for each band
    _reference = reference_draws["market"]
    _sets = {"PyMC reference": (_reference, _reference[_rng.choice(len(_reference), _shown, replace=False)], plots.SERIES[1])}
    _neural = neural_draws.get("market")
    if _neural is not None:
        _inside = _neural[np.all((_neural >= BOUNDS[:, 0]) & (_neural <= BOUNDS[:, 1]), axis=1)]
        if len(_inside) >= 2:
            _sets["network posterior"] = (_neural, _inside[_rng.choice(len(_inside), min(_shown, len(_inside)), replace=False)],
                                          plots.SERIES[0])
    _prices = {_name: np.array([pilot.prices(_row) for _row in _sample]) for _name, (_, _sample, _) in _sets.items()}
    _fig, _axes = plt.subplots(2, 4, figsize=(11.6, 5.4), constrained_layout=True)
    for _e, _ax in enumerate(_axes.flat[:7]):
        _at = EXPIRY == _e
        for _name, (_, _, _color) in _sets.items():
            _band = np.nanquantile(implied_vols(_prices[_name])[:, _at], [.05, .95], axis=0)
            _ax.fill_between(DELTA[_at], 100 * _band[0], 100 * _band[1], color=_color, alpha=.35, label=f"{_name} (90%)")
        _ax.scatter(DELTA[_at], 100 * MARKET_IV[_at], color="#222222", s=14, zorder=4, label="SPX quotes")
        _ax.set_title(f"{EXPIRY_DAYS[_e]} days", fontsize=8.5)
    _axes[0, 0].legend(fontsize=6, loc="upper right")
    _last = _axes.flat[7]
    for _name, (_all, _, _color) in _sets.items():
        _last.hist(_all[:, -1], bins=40, density=True, histtype="step", color=_color, lw=1.3, label=_name)
    _last.axvline(BOUNDS[-1, 1], color=plots.INK, ls="--", lw=1, label="prior bound")
    _last.set(title="noise multiplier c", xlabel="c", yticks=[])
    _last.legend(fontsize=6)
    for _ax in _axes[:, 0]:
        _ax.set_ylabel("implied volatility (%)")
    for _ax in list(_axes[1, :3]) + [_axes[0, 3]]:
        _ax.set_xlabel("standardized moneyness δ")
    # The best Heston fit: prices at the posterior-mean parameters, and its misses in half-spreads.
    market_fit = pilot.prices(_reference.mean(axis=0))
    market_residual = (_y - market_fit) / H
    market_rms = float(np.sqrt(np.mean(market_residual ** 2)))
    _bound = float(BOUNDS[-1, 1])
    _cap = number_word(int(_bound)) if _bound.is_integer() else f"{_bound:g}"
    _columns = {
        "PyMC posterior":
            "PyMC samples the exact posterior of our model and prior. The noise multiplier $c$ piles up at its bound "
            "(last panel), and the Heston parameters settle on the best compromise. The "
            '<a class="nb-ref" href="#what-heston-misses">closing section</a> shows where the fit misses and which '
            "feature is missing.",
    }
    if _neural is not None:
        _width = lambda _draws: np.subtract(*np.quantile(_draws[:, -1], [.95, .05]))
        _columns["Network posterior"] = (
            "The network was trained on quotes that scatter around a Heston smile with a noise SD of at most "
            f"{_cap} half-spreads. The SPX quotes are about {market_rms:.0f} half-spreads away (RMS), outside "
            "anything it saw, so it extrapolates. Its 90% interval for $c$ is "
            f"{times(_width(_neural) / _width(_reference))} times as wide as PyMC's, and "
            f"{100 * np.mean(_neural[:, -1] > _bound):.0f}% of its draws have $c>{_bound:g}$, outside the prior range.")
    mo.vstack([
        md(r"""
        ### 5.5 Heston against the SPX quotes

        Sections 5.2 to 5.4 judged the network. On the SPX quotes the first question is about the model: can any Heston parameters reproduce the market within the assumed noise?
        """),
        _fig,
        caption(f"The bands are implied volatilities of the model prices from {_shown} posterior draws, and the "
                "points are the SPX mid quotes. The last panel shows the posterior of the noise multiplier $c$."),
        key(f"""
        **No Heston smile fits the SPX quotes within the assumed noise.** The best fit, at the posterior-mean parameters, misses them by {market_rms:.1f} half-spreads (RMS). The prior caps the noise SD at {_cap} half-spreads ($c\\le{_bound:g}$).
        """),
        grid(_columns, kind="reading", columns=2),
        key("**This is model misspecification, not a training failure.** The disagreement is a useful alarm: an "
            "amortized estimator should only be trusted on data its simulator could have produced."
            if _neural is not None else
            "**This is model misspecification.** An amortized estimator should only be trusted on data its "
            "simulator could have produced."),
    ])
    return market_fit, market_residual, market_rms


@app.cell(hide_code=True)
def _(grid, md, mo, section):
    FORWARD_START = (0.25, 0.5)  # in years: the strike is fixed after three months, the option expires after six
    _fix, _end = (f"{12 * _t:.0f}" for _t in FORWARD_START)
    mo.vstack([
        section("Use", "A desk holds products that nobody quotes, and must still value them every day. A posterior "
                "gives each value a range.", stage=6),
        md(rf"""
        ### 6.1 Pricing a product the market does not quote

        The example is a **forward-start call**. Bought today, it fixes its strike in {_fix} months at the index level on that day. At {_end} months it pays the percentage rise above that level, if positive.

        $$\max\!\left(\frac{{S_{{{_end}\mathrm{{m}}}}}}{{S_{{{_fix}\mathrm{{m}}}}}}-1,\;0\right)\quad\text{{per unit of notional}}$$
        """),
        grid({
            "Need for a model": "The value depends on how volatile the index will be between the two dates. "
                                "Today's quotes do not state that directly.",
            "Recipe": "Fit a model to the quoted options, then price the product with it. One fit gives one price. "
                      "A posterior gives a range, by pricing the product under each posterior draw.",
            "Use of the range": "The range shows how reliable the value is. It sizes the reserve against being "
                                "wrong and the bid–ask spread.",
        }),
    ])
    return (FORWARD_START,)


@app.cell(hide_code=True)
def _(
    BOUNDS,
    CASES,
    DELTA,
    EXPIRY,
    EXPIRY_DAYS,
    FORWARD_START,
    H,
    TRUTHS,
    caption,
    grid,
    key,
    md,
    mo,
    neural_draws,
    note,
    np,
    number_word,
    pilot,
    plots,
    plt,
    reference_draws,
    table,
):
    # Price a forward-start call and a volatility swap under posterior draws, and under Heston refitted to SPX subsets.
    _n_paths, _shown = 4096, 40  # Monte Carlo paths per price, and posterior draws priced
    from scipy.optimize import least_squares as _least_squares

    from sbi4finance.calibration import products as _products

    # Monte Carlo prices under one parameter draw, with the same random numbers for every draw, so that differences
    # between draws are due to the parameters, not to Monte Carlo noise.
    def _price(_theta):
        _v0, _kappa, _v_inf, _xi, _rho = _theta[:5]
        _paths = _products.heston_paths(dict(kappa=_kappa, theta=_v_inf, vol_of_vol=_xi), _v0, FORWARD_START[1], 252, _n_paths,
                                        seed=20261115)
        return (100 * _products.forward_start_call(_paths, _rho, *FORWARD_START, 1.0)[0],  # % of notional
                100 * _products.volatility_swap(_paths, FORWARD_START[0])[0])  # 3-month volatility swap, % per year

    def _posterior(_draws, _seed, _count=_shown):
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
        f"the 3 shortest expiries ({EXPIRY_DAYS[0]}–{EXPIRY_DAYS[2]} days)": EXPIRY < 3,
        f"the 4 longest expiries ({EXPIRY_DAYS[3]}–{EXPIRY_DAYS[-1]} days)": EXPIRY >= 3,
        "quotes near the money, |δ| ≤ 1.3": np.abs(DELTA) <= 1.3,
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

    def _ranges(case, height, title):
        """The forward-start call for one data set: posterior ranges, with the true price or the refits."""
        fig, ax = plt.subplots(figsize=(9.6, height), constrained_layout=True)
        labels = []
        for source, values in _sets[case].items():
            q = np.quantile(values[:, 0], [.05, .95])
            ax.plot(q, [len(labels)] * 2, color=_colors[source], lw=7, solid_capstyle="butt", alpha=.75)
            ax.scatter(values[:, 0], np.full(len(values), len(labels)), color=_colors[source], s=7, zorder=3)
            labels.append(f"{source} posterior, 90% range")
        if case == "market":
            for name, (_, prices) in _refits.items():
                ax.scatter([prices[0]], [len(labels)], color=plots.INK, marker="D", s=26, zorder=3)
                labels.append(f"refit to {name}")
        else:
            ax.axvline(_truth[0], color=plots.INK, ls="--", lw=1)
            ax.text(_truth[0], len(labels) - 0.5, " true price", fontsize=7, color=plots.INK, va="center")
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_ylim(len(labels) - 0.4, -0.6)
        ax.set_xlabel("forward-start call, % of notional")
        ax.set_title(title, fontsize=10)
        return fig

    # Everything said below is computed here.
    _like, _spx = (np.quantile(_sets[_c]["PyMC"], [.05, .95], axis=0) for _c in ("synthetic-market-like", "market"))
    _refit_prices = np.array([_p for _, _p in _refits.values()])
    _low, _high = _refit_prices.min(axis=0), _refit_prices.max(axis=0)
    _ratio = (_high - _low) / (_spx[1] - _spx[0])
    _covered = [bool(_like[0, _j] <= _truth[_j] <= _like[1, _j]) for _j in range(2)]
    _network = {_c: np.quantile(_sets[_c]["network"], [.05, .95], axis=0) for _c in _sets if "network" in _sets[_c]}
    use_summary = {"ratio": float(_ratio[0]), "covered": _covered[0]}  # for the takeaways
    _heads = ("Forward-start call (% of notional)", "Volatility swap (%)")
    _range = lambda _q: {_heads[0]: f"[{_q[0, 0]:.3f}, {_q[1, 0]:.3f}]", _heads[1]: f"[{_q[0, 1]:.2f}, {_q[1, 1]:.2f}]"}
    _months = f"{12 * FORWARD_START[0]:.0f}"

    _simulated = [{"Source": "True value", _heads[0]: f"{_truth[0]:.3f}", _heads[1]: f"{_truth[1]:.2f}"},
                  {"Source": "PyMC reference, 90% range", **_range(_like)}]
    if "synthetic-market-like" in _network:
        _simulated.append({"Source": "Network, 90% range", **_range(_network["synthetic-market-like"])})
    _first = (f"**Where the model is right, the range contains the true price.** The PyMC range is "
              f"{_like[0, 0]:.3f}–{_like[1, 0]:.3f}% of notional, ±"
              f"{50 * (_like[1, 0] - _like[0, 0]) / np.mean(_like[:, 0]):.1f}% of the price, and the true price is "
              f"{_truth[0]:.3f}%." if _covered[0] else
              f"**On this data set the range misses the true price.** The PyMC range is {_like[0, 0]:.3f}–"
              f"{_like[1, 0]:.3f}% of notional and the true price is {_truth[0]:.3f}%. One noisy data set can shift "
              "a posterior (Section 4.2).")
    _swap = ("The true rate of the volatility swap lies inside its range too." if all(_covered) else
             "The true rate of the volatility swap lies outside its range." if _covered[0] else
             "The true rate of the volatility swap lies inside its range." if _covered[1] else
             "The true rate of the volatility swap lies outside its range too.")

    _median = lambda _draws, _j: f"{np.median(_draws[:, _j]):.2f}"
    _kappa_text = lambda _x: f"{_x:.2f}" + (" (its upper bound)" if np.isclose(_x, BOUNDS[1, 1]) else "")
    _fits = [{"Fit": "PyMC reference, 90% range", **_range(_spx), r"$\kappa$": _median(reference_draws["market"], 1),
              r"$\sigma_v$": _median(reference_draws["market"], 3)}]
    if "market" in _network:
        _fits.append({"Fit": "Network, 90% range", **_range(_network["market"]),
                      r"$\kappa$": _median(neural_draws["market"], 1), r"$\sigma_v$": _median(neural_draws["market"], 3)})
    for _name, (_parameters, _prices) in _refits.items():
        _fits.append({"Fit": f"Refit to {_name}", _heads[0]: f"{_prices[0]:.3f}", _heads[1]: f"{_prices[1]:.2f}",
                      r"$\kappa$": _kappa_text(_parameters[1]), r"$\sigma_v$": f"{_parameters[3]:.2f}"})
    _second = [
        md(f"""
        ### 6.2 The same product on the SPX quotes

        Heston does not fit the SPX quotes (Section 5.5), so its parameters depend on which quotes it is fitted to. Besides the posterior, Heston is therefore refitted to {number_word(len(_refits))} subsets of the quotes, and the product is priced again.
        """),
        _ranges("market", 3.0, "SPX quotes: posterior ranges and refits"),
        caption("The bars are 90% posterior ranges, as in Section 6.1. Each diamond is the price after Heston is "
                "refitted by least squares to part of the SPX quotes."),
        key(f"**A posterior measures uncertainty within the model.** On SPX the model is wrong: refitting it to "
            f"different quotes moves the price {_ratio[0]:.0f} times more than the posterior range, and the posterior "
            "cannot see that."),
        md(f"The table lists the posterior ranges and the {number_word(len(_refits))} refits. For the posteriors it "
           r"gives 90% ranges of the prices and medians of $\kappa$ and $\sigma_v$."),
        table(_fits, align="lrrrr"),
        grid({
            "Volatility swap": f"For the volatility swap the refits spread {_ratio[1]:.0f} times wider than the "
                               "posterior range.",
            "Desk practice": "A desk would hold a reserve sized to the refit spread, add a model-error term or use "
                             "a better model.",
            "Value of the posterior range": "The posterior range matters when the quotes leave the price open. "
                                            "That happens with few or wide quotes, or with a model so flexible "
                                            "that the quotes do not determine what the product depends on.",
        }),
    ]
    if "market" in _network:
        _second.append(note("The network's ranges differ from PyMC's because the SPX quotes lie outside its "
                            "training data (Section 5.5)."))
    mo.vstack([
        _ranges("synthetic-market-like", 1.9, "Market-like simulation: Heston is right"),
        caption(f"The bars are 90% ranges over {_shown} posterior draws, and each dot is one draw. The prices are "
                f"Monte Carlo estimates with {_n_paths:,} paths and the same random numbers for every draw."),
        key(_first),
        md(f"The table adds a {_months}-month volatility swap, priced in the same way. {_swap}"),
        table(_simulated, align="lrr"),
        *_second,
    ])
    return (use_summary,)


@app.cell(hide_code=True)
def _(
    BOUNDS,
    DELTA,
    EXPIRY,
    EXPIRY_DATES,
    EXPIRY_DAYS,
    MARKET_IV,
    NAMES,
    QUOTES,
    SPREAD_RANGE,
    caption,
    cards,
    grid,
    implied_vols,
    key,
    market_fit,
    market_residual,
    md,
    mo,
    np,
    plots,
    plt,
    reference_draws,
    section,
    table,
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
    _axes[0].set_yticks(range(len(_days)), [f"{_d} days" for _d in _days])
    _axes[0].grid(False)
    _axes[0].set(xlabel="standardized moneyness δ (target)", ylabel="expiry",
                 title="SPX minus best Heston fit, volatility points")
    _fig.colorbar(_image, ax=_axes[0], shrink=0.9)

    _axes[1].plot(_days, _skew_market, "o-", color="#222222", lw=1.3, ms=4, label="SPX quotes")
    _axes[1].plot(_days, _skew_heston, "o-", color=plots.SERIES[1], lw=1.3, ms=4, label="best Heston fit")
    _axes[1].set_xscale("log")
    _axes[1].set_xticks(_days, [str(_d) for _d in _days], fontsize=7)
    _axes[1].minorticks_off()
    _axes[1].set(xlabel="days to expiry", ylabel="volatility points", title="Put wing: deepest put minus at the money")
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
    _wing = DELTA < -3
    _share = float(np.sum(market_residual[_wing] ** 2) / _squares)
    _rest = _misfit[~_wing]
    _gap = _skew_market - _skew_heston
    _under = int(np.sum(_grid[:, 0] > 0))
    _where = "every expiry" if _under == len(_days) else f"{_under} of {len(_days)} expiries"
    _rms = [float(np.sqrt(np.mean(market_residual[_r] ** 2))) for _r in _rows]
    _number = lambda _x, _digits: f"{_x:.{_digits}f}".replace("-", "−")
    _table = [{"Days to expiry": int(_days[_e]),
               "RMS misfit (half-spreads)": _number(_rms[_e], 1),
               "Largest misfit (half-spreads)": _number(market_residual[_r][np.argmax(np.abs(market_residual[_r]))], 1),
               r"$\delta$ of the largest misfit": _number(DELTA[_r][np.argmax(np.abs(market_residual[_r]))], 2)}
              for _e, _r in enumerate(_rows)]
    _theta = reference_draws["market"].mean(axis=0)  # the best Heston fit: the posterior-mean parameters
    _bound = BOUNDS[-1, 1]
    _feller = 2 * _theta[NAMES.index("kappa")] * _theta[NAMES.index("theta")]
    _missed = sum(_r > _bound for _r in _rms)  # expiries whose misfit exceeds the largest noise SD the prior allows
    _every = "every expiry" if _missed == len(_days) else f"{_missed} of {len(_days)} expiries"
    _message = ("**The misfit sits in the deep put wing and is largest at short expiries.**"
                if np.argmax(_grid[:, 0]) == 0 else "**The misfit sits in the deep put wing.**")
    mo.vstack([
        section("What Heston misses", "On the SPX quotes the model is wrong. This section shows the signs, locates "
                "the misfit and names the missing feature.", kicker="After the workflow"),
        md(r"""
        ### Four signs of model misspecification

        The PyMC reference for the SPX quotes shows four signs of misspecification. The network plays no part in them.
        """),
        cards([
            ("Sign 1", "The noise multiplier sits at its bound",
             f"Its posterior median is {np.median(reference_draws['market'][:, -1]):.4f}. The bound is {_bound:g}."),
            ("Sign 2", f"The fit misses at {_every}",
             f"The RMS misfit per expiry is {min(_rms):.1f} to {max(_rms):.1f} half-spreads. The prior allows a "
             f"noise SD of at most {_bound:g} half-spreads."),
            ("Sign 3", "The misses are systematic",
             f"The deepest put is underpriced at {_where}, by {_grid[:, 0].min():.1f} to {_grid[:, 0].max():.1f} "
             "volatility points."),
            ("Sign 4", "The fit needs extreme parameters",
             f"The fit has $\\sigma_v={_vol_of_vol:.2f}$, so $\\sigma_v^2={_vol_of_vol ** 2:.1f}$ against "
             f"$2\\kappa v_\\infty={_feller:.2f}$. The Feller condition fails by a factor of "
             f"{_vol_of_vol ** 2 / _feller:.0f} (feature 2)."),
        ], columns=4),
        key("""
        **No Heston parameters and no better inference remove these signs.** The model lacks a feature of the data.
        """),
        md(r"""
        ### Where the fit misses

        The misfit is largest for deep puts and at short expiries.
        """),
        _fig,
        caption("Each panel compares the SPX quotes with the best Heston fit, the prices at the posterior-mean "
                "parameters. The left panel shows the difference for every quote. The middle panel shows how far "
                "each smile rises from the money to the deepest put. The right panel shows the likelihood of the "
                "noise multiplier at the best fit."),
        key(f"{_message} The {int(_wing.sum())} puts with $\\delta<-3$ give {100 * _share:.0f}% of the squared "
            "misfit."),
        grid({
            "Deep put wing":
                f"The SPX quotes lie above the best Heston smile at the deepest put of {_where}, by "
                f"{_grid[0, 0]:.1f} points at {_days[0]} days and {_grid[3:, 0].min():.1f} to "
                f"{_grid[3:, 0].max():.1f} points from {_days[3]} days on. Elsewhere the fit is a compromise, the "
                f"best that five parameters can do: it misses by up to {np.max(np.abs(_rest)):.1f} points"
                f"{', in both directions' if _rest.min() < 0 < _rest.max() else ''}.",
            "Put-wing slope":
                f"From the money to the deepest put, SPX implied volatility rises {_skew_market[0]:.1f} points at "
                f"{_days[0]} days, the best Heston smile only {_skew_heston[0]:.1f}. From {_days[3]} days on the "
                f"shortfall is {_gap[3:].min():.1f} to {_gap[3:].max():.1f} points.",
            "Likelihood of $c$":
                f"A half-spread is only {SPREAD_RANGE} volatility points, so these misses amount to {_c_hat:.1f} "
                "half-spreads (RMS). The multiplier $c$ scales the noise of every quote by the same factor, so it "
                f"cannot improve the fit, only measure it. The likelihood peaks at $c={_c_hat:.1f}$, against a "
                f"prior bound of {_bound:g}.",
        }, kind="reading", widths=(1.5, 1, 1)),
        mo.accordion({"Misfit by expiry, in half-spreads": table(_table, align="rrrr")}),
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
    md,
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
        md(r"""
        ### Which feature is missing

        The at-the-money skew and curvature of each smile show what Heston lacks at short expiries.
        """),
        _fig,
        caption("Skew and curvature are measured as in Lecture 5, by a quadratic in log-moneyness through the five "
                "quotes nearest the money. The right panel shows the shortest smile."),
        key(f"**The missing feature is a short-expiry smile that keeps steepening.** Heston's skew stops at its "
            f"ceiling and its curvature turns down below {_days[_peak]} days. The SPX skew follows a power law and "
            "keeps growing."),
        grid({
            "Skew":
                "At the money, the fit follows the SPX skew at every expiry, within "
                f"{100 * np.max(np.abs(_fit[:, 0] / _spx[:, 0] - 1)):.0f}%, with nearly the same power law (exponents "
                f"{_minus(_power(_fit[:, 0])[0])} and {_minus(_b)}). Its {_days[0]}-day skew is already "
                f"{100 * abs(_fit[0, 0]) / _ceiling:.0f}% of Heston's ceiling "
                f"$|\\rho|\\sigma_v/(4\\sqrt{{v_0}})={_ceiling:.2f}$, so shorter expiries can add at most "
                f"{100 - 100 * abs(_fit[0, 0]) / _ceiling:.0f}% (feature 3).",
            "Curvature":
                f"The SPX smile gets more curved as the expiry shortens: {_spx[2, 1]:.1f}, {_spx[1, 1]:.1f} and "
                f"{_spx[0, 1]:.1f} at {_days[2]}, {_days[1]} and {_days[0]} days. The fit's curvature peaks at "
                f"{_days[_peak]} days ({_fit[_peak, 1]:.1f}) and falls to {_fit[0, 1]:.1f} at {_days[0]} days."
                + (f" At shorter expiries it falls below zero (under {_negative.max():.0f} days)."
                   if _negative.size else ""),
            "Shortest smile":
                "Near the money the fit is within "
                f"{np.max(np.abs(100 * (MARKET_IV - _fit_iv)[_near])):.1f} volatility points. At the deepest put it "
                f"is {_deep[0]:.1f} points too low. With a single diffusive variance factor, the fit cannot price "
                f"that much crash risk into {_days[0]} days without breaking the rest of the smile.",
        }, kind="reading"),
        md(r"""
        ### What can be done within Heston

        Change the noise model: widen it or add an explicit model-error term. The posterior then admits the misfit instead of pushing $c$ against its bound.
        """),
        key("**The misfit itself stays.** Removing it needs a different model."),
    ])
    return


@app.cell(hide_code=True)
def _(
    SPREAD_RANGE,
    calibration_summary,
    cards,
    comparison,
    market_rms,
    mo,
    number_word,
    section,
    times,
    use_summary,
):
    # The takeaways quote what the sections above computed.
    _exceptions = calibration_summary["exceptions"]
    _calibration = ("All six parameters are well calibrated." if not _exceptions else
                    f"{number_word(calibration_summary['good']).capitalize()} of the six parameters are well "
                    "calibrated." + "".join(
                        f" The posterior of the {_label} is biased." if _kind == "biased" else
                        f" The intervals for the {_label} are {_kind}." for _label, _kind in _exceptions))
    _sharpness = ("Only an exact reference shows how much sharpness the network loses." if comparison is None else
                  f"On the simulated data sets the network's 90% intervals are up to {times(comparison['widest'])} "
                  "times as wide as PyMC's.")
    _range = (("Where the model is right, the range contains the true price. " if use_summary["covered"] else "")
              + f"On SPX, refitting Heston to different quotes moves the price {use_summary['ratio']:.0f} times more "
              "than the posterior range. Model error dominates parameter uncertainty.")
    mo.vstack([
        section("Takeaways", "Amortized inference returns a posterior in seconds. The checks show where it can be "
                "trusted.", kicker="Summary"),
        cards([
            ("Sections 1 to 4", "Simulation-based inference turns a pricer into a posterior",
             "Prior, pricer and noise model form a simulator. One training run gives an estimator that returns the "
             "posterior for any quotes of these 63 contracts in seconds."),
            ("Section 1.3", "Liquid quotes determine the parameters tightly",
             f"A half-spread is {SPREAD_RANGE} volatility points. The posteriors are narrow, and model error "
             "stands out."),
            ("Section 2.3", "The network needs well-scaled inputs",
             "Quotes in noise units and whitening put all 63 inputs on one uncorrelated scale."),
            ("Section 5.2", "Calibration is judged across many simulations", _calibration),
            ("Section 5.4", "A calibrated posterior can still be too wide", _sharpness),
            ("Section 5.5", "The model needs checking as well as the network",
             f"On the SPX quotes the noise multiplier sits at its bound and Heston misses by {market_rms:.1f} "
             "half-spreads (RMS). An amortized estimator is only reliable on data its simulator could have "
             "produced."),
            ("Section 6", "A posterior gives an unquoted product a price range", _range),
            ('<a class="nb-ref" href="#what-heston-misses">What Heston misses</a>',
             "Misspecification shows what the next model needs",
             "Heston follows the SPX skew at the money. It misses the short-expiry curvature and the deep put "
             "wing."),
        ]),
    ])
    return


@app.cell(hide_code=True)
def _(grid, md, mo, note):
    mo.accordion({
        "References": grid([
            r"""
            - Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. *The Review of Financial Studies*, 6(2), 327–343.
            - Lewis, A. L. (2001). A simple option formula for general jump-diffusion and other exponential Lévy processes. Working paper, Envision Financial Systems and OptionCity.net.
            - Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., and Le, M. (2023). Flow matching for generative modeling. *International Conference on Learning Representations*.
            - Radev, S. T., Schmitt, M., Schumacher, L., Elsemüller, L., Pratz, V., Schälte, Y., Köthe, U., and Bürkner, P.-C. (2023). BayesFlow: Amortized Bayesian workflows with neural networks. *Journal of Open Source Software*, 8(89), 5702.
            """,
            r"""
            - Säilynoja, T., Bürkner, P.-C., and Vehtari, A. (2022). Graphical test for discrete uniformity and its applications in goodness-of-fit evaluation and multiple sample comparison. *Statistics and Computing*, 32, 32.
            - Talts, S., Betancourt, M., Simpson, D., Vehtari, A., and Gelman, A. (2018). Validating Bayesian inference algorithms with simulation-based calibration. arXiv:1804.06788.
            - ter Braak, C. J. F., and Vrugt, J. A. (2008). Differential Evolution Markov Chain with snooker updater and fewer chains. *Statistics and Computing*, 18(4), 435–446.
            """,
        ]),
        "Running this notebook": mo.vstack([
            md("The pretrained estimator and its training data are distributed as a separate 465 MB archive. "
               "`lab/plans/training-data-guide.md` explains how to download and verify it."),
            grid({
                "Live notebook": r"""
                From `lab/`, start marimo with `HESTON_BUNDLE` set to the extracted folder, the one that contains `bundle.json`.

                ```text
                HESTON_BUNDLE=/path/to/bundle \
                  pixi run --locked -e core-reference \
                  marimo edit notebooks/heston_spx_calibration.py
                ```
                """,
                "Static copy": r"""
                The same setting renders a static copy with every output. In it the tabs and the foldable sections still work, and the other controls show their initial state.

                ```text
                HESTON_BUNDLE=/path/to/bundle \
                  pixi run --locked -e core-reference \
                  marimo export html notebooks/heston_spx_calibration.py \
                  -o heston_spx_calibration.html
                ```
                """,
            }),
            note("Without `HESTON_BUNDLE`, results that need the network are left out and the PyMC reference is "
                 "shown alone. The reference posteriors and their run records are in "
                 "`lab/evidence/pilot-results/2026-11-11/heston-actual63-notebook/`. Posterior draws use fixed "
                 "seeds, so the figures are reproducible."),
        ]),
    })
    return


if __name__ == "__main__":
    app.run()
