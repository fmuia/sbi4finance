import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="Learn volatility dynamics from observed paths",
    css_file="trajectory_inference.css",
)


@app.cell(hide_code=True)
def _():
    import json
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from sbi4f_tutorial.shared import plots
    from sbi4f_tutorial.trajectories import lesson as trajectory
    from sbi4f_tutorial.trajectories import diagnostics as path_checks, simulation as path_simulation, model_checks
    from sbi4f_tutorial.trajectories import predictive_visuals
    plots.style()
    return (
        json,
        mo,
        model_checks,
        np,
        path_checks,
        path_simulation,
        plt,
        predictive_visuals,
        trajectory,
    )


@app.cell(hide_code=True)
def _(mo):
    # Same section, type and navigation contract as the Heston/SPX notebook.
    import html as _markup
    import re as _re

    _stages = (
        ("Model", "Parameters and physical return law"),
        ("Simulate", "Histories before seeing data"),
        ("Train", "Learn a conditional density"),
        ("Infer", "Condition on observed returns"),
        ("Check", "Replicate, recover, calibrate"),
        ("Use", "What a continuation forecast requires"),
    )

    def _anchor(number):
        return "s" + str(number).replace(".", "-")

    def lesson_md(text):
        # Leave displayed code intact while giving numbered headings anchors.
        pieces = _re.split(r"(^[ \t]*(?:```|~~~).*?^[ \t]*(?:```|~~~)[ \t]*$)",
                           text, flags=_re.MULTILINE | _re.DOTALL)
        for index in range(0, len(pieces), 2):
            pieces[index] = _re.sub(
                r"^(\s*)### (\d+\.\d+) (.+)$",
                lambda match: f'{match[1]}<h3 id="{_anchor(match[2])}">'
                              f'<span class="nb-number">{match[2]}</span> {match[3]}</h3>',
                pieces[index], flags=_re.MULTILINE)
        return mo.md("".join(pieces))

    def workflow_map():
        steps = "".join(
            f'<a class="nb-map-step" href="#{_anchor(number)}"><div class="nb-map-name">'
            f'<span>{number}</span>{name}</div><div class="nb-map-text">{detail}</div></a>'
            for number, (name, detail) in enumerate(_stages, start=1))
        return mo.Html(f'<div class="nb-map">{steps}</div>')

    def lesson_section(title, lead, stage=None, kicker=None):
        if stage is None:
            top = f'<div class="nb-eyebrow">{_markup.escape(kicker or "")}</div>'
            number, outline = "", ""
            anchor = _re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        else:
            rail = []
            for index, (name, _) in enumerate(_stages, start=1):
                state = "is-done" if index < stage else "is-current" if index == stage else ""
                if index > 1:
                    rail.append(f'<span class="nb-rail-arrow {"is-done" if index <= stage else ""}">&#8594;</span>')
                rail.append(f'<a class="nb-rail-step {state}" href="#{_anchor(index)}">'
                            f'<span class="nb-rail-number">{index}</span>{name}</a>')
            top = f'<div class="nb-rail">{"".join(rail)}</div>'
            number = f'<span class="nb-section-number">{stage}</span>'
            outline, anchor = f'<span class="nb-outline-only">{stage} </span>', _anchor(stage)
        return mo.Html(
            f'<div class="nb-section" id="{anchor}">{top}<div class="nb-section-head">{number}<div>'
            f'<h2 id="{anchor}-title">{outline}{_markup.escape(title)}</h2>'
            f'<div class="nb-standfirst">{lesson_md(lead).text}</div></div></div></div>')

    return lesson_md, lesson_section, workflow_map


@app.cell(hide_code=True)
def _(mo, workflow_map):
    mo.vstack([mo.md(r"""
    # Learn volatility dynamics from observed paths

    **The parameters stay fixed; the evidence grows.**

    The option lessons used prices across strikes and maturities. Here we observe
    **daily log returns**, in time order. Can one history tell us how volatility
    moves, how it relates to returns, and what drift is plausible?

    We start with rough Bergomi and a supplied variance curve:
    **model → simulate → train → infer → check → use**.
    The result is a joint parameter distribution, not a reconstructed volatility path.

    | Route | What you do | Model source |
    |:---|:---|:---|
    | **Train here** | Generate a small training set, fit the GRU and flow, then infer parameters | A new tutorial model, kept in this session |
    | **Optional pretrained model** | Load the supplied estimator and reveal more days | A larger, separately trained model; saved draws work without a download |

    First introduce the model and explore the forward simulator. Choose the
    observed history immediately before producing posteriors. Training and model loading are explicit actions;
    the small simulator runs locally.

    The supplied estimators learn useful information, but **all three failed
    their final marginal-calibration screen**. We will inspect that distinction.
    """), workflow_map(),
    mo.accordion({"Where BayesFlow is used": mo.md(r"""
    | Call | Role here |
    |:---|:---|
    | `simulation.generate` (NumPy) | Draw parameters and daily return histories before fitting |
    | `bf.Adapter` | Route transformed labels, return tokens, time context and mask |
    | `bf.networks.RecurrentNetwork` | Learn a causal GRU summary of the observed prefix |
    | `bf.networks.CouplingFlow` | Learn the conditional joint parameter density |
    | `bf.BasicWorkflow`, `fit_offline` | Fit the summary and density on simulated pairs |
    | `workflow.sample` through `inference.sample` | Condition on returns and restore physical parameter units |
    | `bf.diagnostics` | Inspect the tutorial fit's loss; saved cohort reports document calibration |
    """)})])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([mo.md(r"""
    ### The problem this notebook addresses

    An asset's daily returns are visible; its volatility process and generating
    parameters are hidden. This notebook addresses the inverse problem:
    **infer which fixed stochastic-volatility parameters can explain one ordered
    return history, while quantifying the uncertainty that remains.** Different
    parameter combinations can produce similar histories, making a single
    best-fitting vector an incomplete answer.

    We train BayesFlow on simulated parameter–history pairs to approximate the
    joint parameter posterior. Synthetic examples let us reveal more evidence,
    compare inference with known generating values, and check the learned
    uncertainty. We also clarify the additional information needed to turn
    parameter inference into a continuation forecast.
    """)])
    return


@app.cell(hide_code=True)
def _(mo, lesson_section):
    market_window = mo.ui.dropdown(options={
        "2020 to latest available · 13 November 2026": "voo-2020-20261113",
        "2020 crisis window · 256 trading days": "voo-covid-256"},
        value="2020 to latest available · 13 November 2026", label="VOO market window")
    mo.vstack([
        lesson_section("The data", "Compare market returns with a simulated history and a matched Gaussian.",
                       kicker="Before the workflow"),
        market_window,
        mo.md("This selects the **VOO observations in the next chart**. "
              "The illustrative Bergomi history stays fixed for comparison."),
    ])
    return (market_window,)


@app.cell(hide_code=True)
def _():
    from sbi4f_tutorial.trajectories import market_story
    # An illustrative forward simulation only: no fitting or market inference.
    illustrative_market_model = market_story.model_example()
    return illustrative_market_model, market_story


@app.cell(hide_code=True)
def _(illustrative_market_model, market_story, market_window, mo, trajectory):
    try:
        _market = trajectory.load_market_example(market_window.value)
        _metadata = _market["metadata"]
        _panel = mo.vstack([
            mo.md("### What behavior are we trying to explain?\n\n"
                "Compare VOO with an explicit rough-Bergomi simulation using the **same trace and "
                "tail-frequency views**. A Gaussian benchmark is matched to each displayed history's "
                "mean and standard deviation. Does that match its frequency of large moves?"),
            mo.md(f"**Selected VOO window: {_market['dates'][0]} to {_market['dates'][-1]} · "
                  f"{len(_market['dates']):,} daily returns.**"),
            market_story.comparison_figure(_market, illustrative_market_model),
            mo.md("**Bergomi can produce more large moves than a matched Gaussian.** "
                "The purple tail curve lies above its Gaussian benchmark for large daily returns, "
                "even though their mean and standard deviation match. Its instantaneous shocks are Gaussian, "
                "but their scale changes with hidden variance. Mixing returns from quiet and volatile "
                "periods changes the marginal return distribution. This illustrates non-Gaussian "
                "behavior; these parameters have not been fitted to VOO."),
            mo.md("**The synthetic lesson uses 1,260 trading days.** The longer VOO snapshot "
                "contains 1,726 returns and extends beyond the neural estimator's five-year "
                "training horizon. This descriptive market comparison does not extend that "
                "estimator's supported input length."),
            mo.accordion({"Data, model settings and interpretation": mo.md(
                "The dated VOO snapshot preserves adjusted-close returns, including corporate-action "
                "adjustments, and converts simple returns with `log1p`. No recentering, volatility "
                "normalization, filtering or market inference is performed. Each Gaussian uses its "
                "own history's mean and population standard deviation as a descriptive control.\n\n"
                "The Bergomi panel uses H = 0.12, eta = 1.2, rho = -0.7, drift = 0.07 and a flat "
                "mean-variance curve of 0.04. These settings and seed 20261114 were fixed for the "
                "illustration, not estimated from VOO. Time starts at the model's origin; its "
                "nonstationary long-history behavior is explained below. The same single simulated "
                "history supplies both the trace and its tail curve. Empirical probabilities are discrete; a finite "
                "tail plot is not a formal normality or model-adequacy test.\n\n"
                f"Source: {_metadata.get('source_url', _metadata['source'])}  \n"
                f"Last recorded trading date: **{_market['dates'][-1]}**. "
                "The notebook reads the accepted snapshot offline and does not download data on opening.")}),
            mo.md("**Next: a controlled synthetic history.** Its generating parameters and hidden "
                "variance are known for illustration. Only its returns enter the parameter estimator.")])
    except (OSError, ValueError, KeyError) as _error:
        _panel = mo.callout(mo.md(f"**Empirical motivation unavailable:** {_error}. "
                                "The synthetic lesson remains available below."), kind="warn")
    _panel
    return


@app.cell(hide_code=True)
def _(mo, trajectory):
    model_choice = mo.ui.dropdown(
        options={label: key for key, label in trajectory.MODEL_LABELS.items()},
        value="Rough Bergomi · known variance curve", label="Stochastic volatility model")
    mo.accordion({"Explore another model": mo.vstack([
        model_choice,
        mo.md("The other examples infer the rough variance curve or use Heston. "
              "Each model has its own synthetic histories and fitted estimator; "
              "switching models changes the data as well as the assumptions. "
              "This selects the supplied example; the small live-training exercise "
              "below remains rough Bergomi with the supplied curve."),
    ])})
    return (model_choice,)


@app.cell(hide_code=True)
def _(mo, model_choice, trajectory):
    mo.stop(not trajectory.CATALOGUE.is_file(), mo.callout(mo.md(
        "**Trajectory examples are not packaged in this copy.** See the tutorial "
        "README for the matching assets. No training or model download starts automatically."), kind="info"))
    cache_task, cache_report, cache_arrays = trajectory.load_example(model_choice.value)
    return cache_arrays, cache_report, cache_task


@app.cell(hide_code=True)
def _(cache_task, mo, lesson_md, lesson_section):
    parameter_labels = {
        "kappa": "Mean reversion κ", "long_variance": "Long variance L",
        "vol_of_vol": "Volatility of variance σ", "rho": "Leverage correlation ρ",
        "hurst": "Roughness H", "eta": "Log-variance dispersion η",
        "short_variance": "Short variance S", "decay": "Curve decay k", "drift": "Physical drift μ",
    }
    _meanings = {
        "kappa": "Speed of reversion toward the long variance level",
        "long_variance": "Long variance level",
        "vol_of_vol": "Strength of variance shocks", "rho": "Correlation of price and variance-driver shocks",
        "hurst": "Roughness of the Gaussian variance driver", "eta": "Dispersion of log variance at a fixed mean-variance curve",
        "short_variance": "Initial variance-curve level", "decay": "Speed of transition between curve levels",
        "drift": "Annual proportional price drift",
    }
    _priors = [{"parameter": parameter_labels[_name], "meaning": _meanings[_name],
                "prior": ("Log-uniform" if _kind == "log_uniform" else "Uniform")
                         + f" [{_bounds[0]:g}, {_bounds[1]:g}]"}
               for _name, _kind, _bounds in zip(cache_task["parameter_names"], cache_task["prior_types"],
                                               cache_task["prior_bounds"], strict=True)]
    _heston = cache_task["model"] == "heston"
    _curve = ("**Selected task: Heston.** Initial variance is supplied; the five "
              "parameters in the table are inferred. A supplied initial value "
              "does not fix the subsequent realized variance path."
              if _heston else
              r"**Selected task: rough Bergomi with a supplied curve.** We set $\xi(t)=0.04$, "
              "the deterministic ensemble mean of variance, and infer four parameters. "
              "Each realized variance path remains random. The joint-curve example relaxes this assumption."
              if cache_task["model"] == "rough-known" else
              r"**Selected task: rough Bergomi with an inferred curve.** We use $\xi(t)=L+(S-L)e^{-kt}$. "
              "The short level S, long level L and decay k join the other four parameters as joint targets. "
              "These coefficients are constant within each generated history; the realized variance is random. "
              "Learning a mean curve does not turn this estimator into a latent-state filter.")
    _steps = cache_task["numerics"].get("substeps", cache_task["numerics"].get("observation_substeps"))
    mo.vstack([
        lesson_section("Model", "Specify the return dynamics and parameter prior, then inspect one synthetic history.", stage=1), lesson_md(r"""
        ### 1.1 Dynamics and unknowns

        **The price equation is shared with Heston; the variance mechanism changes.**
        We model one ordered return history under the physical probability **P**.
        Here are the rough Bergomi components together, in continuous time:

        $$\begin{aligned}
        \frac{dS_t}{S_t}&=\mu\,dt+\sqrt{V_t}\,dW_t^S,
        &d\langle W^S,W^V\rangle_t&=\rho\,dt,\\
        Y_t&=\sqrt{2H}\int_0^t(t-s)^{H-1/2}\,dW_s^V,
        &q(t)&=\operatorname{Var}(Y_t)=t^{2H},\\
        V_t&=\xi(t)\exp\!\left(\eta Y_t-\tfrac12\eta^2q(t)\right),
        &Y_0&=0,\quad V_0=\xi(0).
        \end{aligned}$$

        Time is measured in years, with **252 observed trading days per year**.
        $W^S$ and $W^V$ are standard Brownian drivers under P.

        - **Price $S_t$:** μ is the annual proportional price drift. Variance $V_t$
          sets the noise scale; volatility is $\sqrt{V_t}$.
        - **Driver $Y_t$:** a weighted history of Gaussian innovations. The power-law
          kernel gives recent innovations more weight; H controls its roughness.
        - **Variance $V_t$:** exponentiating the driver makes it positive. η controls
          log-variance dispersion. The compensator $-\eta^2q(t)/2$ ensures
          $\mathbb E_P[V_t]=\xi(t)$; it does not pull each path toward that mean.
        - **Correlation ρ:** price and variance-driver innovations are correlated.
          Negative ρ creates leverage, linking negative price innovations with
          positive variance-driver innovations.

        We observe daily **log returns**, so Itô's formula supplies the correction

        $$d\log S_t=(\mu-V_t/2)\,dt+\sqrt{V_t}\,dW_t^S.$$

        **What differs from Heston?** Keep the same price equation and correlation,
        but replace the driver and exponential variance construction by

        $$dV_t=\kappa(L-V_t)\,dt+\sigma\sqrt{V_t}\,dW_t^V.$$

        | | Heston variance | Rough Bergomi variance above |
        |:---|:---|:---|
        | Random mechanism | Square-root diffusion | Exponential of a Gaussian Volterra driver |
        | Memory | Current variance is a Markov state | With $\eta>0$ and $H<1/2$, current variance alone does not summarize the past |
        | Mean reversion | $\kappa(L-V_t)$ pulls variance toward L | No corresponding restoring drift; $\xi(t)$ specifies the ensemble mean |

        These equations describe the mathematical models. **The estimator learns
        our declared finite-step simulator:** rough Bergomi uses the hybrid
        driver's actual variance $q_i=\operatorname{Var}_{\rm grid}(Y_i)$ in the
        compensator, rather than substituting $t_i^{2H}$. The numerical updates
        appear together in **The discrete model, step by step** below.

        The option lessons used a pricing probability **Q**, under which discounted
        prices are martingales. A parameter fitted under P need not have the same
        value under Q; this is a declared return-history law, not a fitted change
        of measure. Returns contain process randomness, with **no extra
        measurement-error or quote-noise term**. The variance constructions follow
        [Heston](https://doi.org/10.1093/rfs/6.2.327) and
        [Bayer, Friz and Gatheral](https://realvol.com/RVOL2Paper.pdf); our physical
        drift, initialization and training priors are explicit tutorial choices.
        """),
        mo.md(_curve),
        mo.ui.table(_priors, selection=None, pagination=False),
        mo.accordion({"Why this H prior, and what happens at H = 1/2?": mo.md(r"""
        **We assume the rough regime when training this estimator.** The uniform
        prior $H\in[0.03,0.4]$ covers strongly rough through less rough paths.
        High-frequency volatility studies motivate values often around 0.1
        ([Gatheral, Jaisson and Rosenbaum](https://arxiv.org/abs/1410.3394)),
        but do not prescribe our exact endpoints or a uniform density. Those
        are simulation/training design choices, kept away from the endpoints
        of the supported interval $0<H<1/2$.

        **Gaussianity and roughness are different properties.** The driver
        $Y_t$ above is Gaussian for every H: it is a linear integral of
        Brownian noise. Smaller H makes its paths
        rougher and changes dependence; it does not make the driver non-Gaussian.

        At the mathematical boundary $H=1/2$, the kernel becomes constant and
        $Y_t=W_t^V$, ordinary Brownian motion. For $\eta>0$, variance remains
        random and lognormal at a fixed time. This does **not** give a
        constant-volatility Gaussian-return model. That benchmark would use
        $\eta=0$ and a constant supplied variance curve.

        This estimator cannot test whether $H=1/2$ is adequate: its prior
        excludes that possibility. Our current rough-driver implementation
        also enforces $0<H<1/2$. Comparing the Brownian boundary needs a
        matching simulator and estimator; changing a label or slider cannot
        extend the existing checkpoint's training support.
        """)}) if not _heston else mo.md(""),
        mo.accordion({"Priors, supplied quantities and numerical details": mo.vstack([
            mo.md("The parameter priors are independent. Uniform priors assign equal probability "
                  "to equal-width intervals; log-uniform priors assign equal probability to equal "
                  "multiplicative ranges. A log-uniform density on [a,b] is 1/[x log(b/a)]. "
                  "Supplied quantities are modelling assumptions, not measurements of latent volatility."),
            mo.ui.table([{"supplied quantity": _name.replace("_", " "), "value": _value}
                         for _name, _value in cache_task["fixed"].items()], selection=None, pagination=False)
                if cache_task["fixed"] else mo.md("No fixed model parameters in this joint-curve task."),
            mo.md(f"The simulator uses **{_steps} internal steps per day**, then supplies one daily return. "
                  "Inference is trained for that finite-step law. Changing models does not change "
                  "the daily observation frequency."),
            mo.md("The Heston implementation uses a signed auxiliary variance state and its positive part. "
                  "The rough implementation uses a hybrid Gaussian driver and its matching variance compensator. "
                  "The step-by-step equations below describe the selected numerical law. "
                  "Each history starts at the declared origin; extending it reveals an older process, "
                  "rather than a stationary window shifted to a new starting date."),
            mo.ui.table([{"setting": _name.replace("_", " "), "value": _value}
                         for _name, _value in cache_task["numerics"].items()], selection=None, pagination=False),
        ])}),
    ])
    return (parameter_labels,)


@app.cell(hide_code=True)
def _(cache_task, mo, lesson_md):
    _memory = ("κ controls mean reversion; σ controls random variance fluctuations. "
               "Inspect persistence across several paths as you change them. The notebook fits the stated finite-step law."
               if cache_task["model"] == "heston" else
               "H controls the roughness of the latent Gaussian driver; η scales its contribution to log variance. "
               "A jagged price trace alone does not identify either. Inspect return-derived volatility across several paths.")
    _level = ("Initial variance is supplied; the long variance level is inferred. Variance still moves through time, "
              "while the generating parameters stay fixed within a history."
              if cache_task["model"] == "heston" else
              "The known-curve task fixes the mean variance scale at 0.04. Realized variance remains random."
              if cache_task["model"] == "rough-known" else
              "Short level S, long level L and decay k describe the deterministic mean-variance scale. "
              "They are inferred jointly; they are not observations of hidden variance.")
    _texts = [
        r"**1. Drift and log growth differ.** μ is proportional price drift. Daily log returns also contain "
        r"the variance correction $-V_t/2$. With variance and innovations fixed, raising μ by δ adds "
        r"$\delta/252$ to each daily log return.",
        "**2. Volatility dynamics leave patterns.** " + _memory,
        "**3. Leverage couples innovations.** A negative ρ links negative price innovations with positive "
        "variance-driver innovations. The co-movement is stochastic; the correlation between returns and a "
        "rolling volatility estimate is a different quantity.",
        "**4. Variance scale and evidence have different roles.** " + _level +
        " Revealing more days changes the evidence, rather than the generating parameter values.",
    ]
    _style = {"border": "1px solid #d1d5db", "border-radius": "8px", "padding": "10px 12px",
              "min-width": "240px", "min-height": "100%"}
    _cards = [mo.md(text).style(_style) for text in _texts]
    mo.vstack([lesson_md("### 1.2 Four features to carry into the simulator"),
               mo.hstack(_cards[:2], widths="equal", wrap=True, gap=1, align="stretch"),
               mo.hstack(_cards[2:], widths="equal", wrap=True, gap=1, align="stretch")], gap=1)
    return


@app.cell(hide_code=True)
def _(cache_task, mo):
    _warning = (mo.callout(mo.md(
        "**Heston numerical limitation.** The finite-step simulator can keep effective variance "
        "at zero for an entire day, making that return exactly drift × time. Its drift posterior "
        "can therefore contain a point mass that a continuous neural density cannot represent "
        "exactly. Resolving this needs a new numerical specification and fit."), kind="warn")
        if cache_task["model"] == "heston" else mo.md(""))
    _warning
    return


@app.cell(hide_code=True)
def _(cache_task, mo):
    _heston = cache_task["model"] == "heston"
    _innovations = (r"""
    The Heston simulator draws independent $Z_j^V,Z_j^\perp\sim N(0,1)$ at
    each internal step and for each path. A Brownian increment is $\sqrt h Z$.
    The variance shock uses $Z_j^V$; the price shock uses
    $\rho Z_j^V+\sqrt{1-\rho^2}Z_j^\perp$. Shared $Z_j^V$ gives the chosen
    price–variance correlation. These process shocks are not measurement noise.
    """ if _heston else r"""
    Rough Bergomi uses three independent standard-normal channels per internal
    step: $Z^W$, a Gaussian bridge residual $Z^B$, and $Z^\perp$.
    The hybrid convolution of past $Z^W$ values plus $Z^B$ builds the Gaussian
    driver $Y$. The bridge residual approximates the kernel's most recent,
    singular contribution; it is not a quote or measurement error.
    The price shock is $\rho Z^W+\sqrt{1-\rho^2}Z^\perp$.
    Paths draw independent channels. A fixed seed freezes these normal draws
    within the selected model; changing parameters still changes its variance
    and return paths.
    """)
    _calculation = (r"""
    **Eight internal steps make one observed day.** Write $h=1/(252\cdot8)$,
    let $\widetilde V_j$ be the signed numerical variance state, and set
    $v_j=\max(\widetilde V_j,0)$. Each step computes

    $$\Delta\ell_j=(\mu-v_j/2)h+\sqrt{v_jh}
       (\rho Z_j^V+\sqrt{1-\rho^2}Z_j^\perp),$$

    $$\widetilde V_{j+1}=\widetilde V_j+\kappa(L-v_j)h
       +\sigma\sqrt{v_jh}Z_j^V.$$

    Sum the eight $\Delta\ell_j$ values to obtain one daily log return.
    The signed state and its positive part are different objects; neither is
    an observed network input. The zero-variance-day limitation described above
    belongs to this actual finite-step law.
    """ if _heston else r"""
    **Four internal steps make one observed day.** With $h=1/(252\cdot4)$,
    the hybrid driver and return update are

    $$\begin{aligned}
    Y_i&=\sum_{j=0}^{i-1}w_{ij}(H,h)Z_j^W+b(H,h)Z_{i-1}^B,
       &q_i&=\operatorname{Var}_{\rm grid}(Y_i),\qquad i\ge1,\\
    V_i&=\xi(t_i)\exp(\eta Y_i-\tfrac12\eta^2q_i),\\
    \Delta\ell_i&=(\mu-V_i/2)h+\sqrt{V_i h}
       (\rho Z_i^W+\sqrt{1-\rho^2}Z_i^\perp).
    \end{aligned}$$

    Thus $Y_i$ uses earlier shocks, while $Z_i^W$ drives the next return step
    and contributes to $Y_{i+1}$. The return step uses left-endpoint variance. Four consecutive
    $\Delta\ell_i$ values are summed into one daily return. At the declared
    origin $Y_0=q_0=0$. The idealized continuous driver has variance $t^{2H}$;
    substituting it for $q_i$ would change this simulator's finite-grid law.
    """)
    _curve = (r"""
    Heston supplies the initial variance $V_0=0.04$, not a subsequent variance
    path. $L$ is the mean-reversion destination, $\kappa$ its speed, and $\sigma$
    the shock amplitude. Their P-law priors and the drift prior are shown above.
    This task does not supply a market-derived forward-variance curve.
    """ if _heston else r"""
    **Which curve is fixed?** The supplied example fixes only the deterministic
    mean-variance curve, not the latent path. Three distinct objects appear:

    | Object | Meaning | Our inference task |
    |:---|:---|:---|
    | $\xi(t)$ | Deterministic ensemble mean of variance | Supplied, or described by inferred curve coefficients |
    | $V_t$ | Random variance along one particular history | Hidden; integrated out of parameter inference |
    | $\Theta$ | Parameters shared across that history | Joint posterior target |

    In this declared synthetic
    P model, $\xi(t_i)=\mathbb E_P[V_i]$ at simulator grid nodes. The Gaussian
    compensator ensures this equality for the hybrid approximation. It is a
    physical-law variance-scale curve, not a Q curve estimated from option prices.

    **A constant mean does not make this process stationary.** At each grid
    time, $Y_i\sim N(0,q_i)$, so

    $$\mathbb E_P[V_i]=\xi(t_i),\qquad
      \operatorname{median}(V_i)=\xi(t_i)\exp(-\eta^2q_i/2).$$

    The contrast grows even while a typical path becomes quieter:

    $$\operatorname{Var}(\log V_i)=\eta^2q_i,\qquad
      \operatorname{Var}(V_i)=\xi(t_i)^2\bigl(e^{\eta^2q_i}-1\bigr).$$

    The driver variance $q_i$ grows from zero. Its distribution therefore
    widens, and its median falls relative to the mean. Rare high-variance
    realizations keep the ensemble mean at the curve level; there is no
    pathwise pull back to that level. This explains a tendency for typical
    histories to become quiet, especially with larger amplitude, but does
    not force an individual path to decline monotonically. The idealized
    continuous driver has $q(t)=t^{2H}$; the simulator uses its stated grid
    variance. Revealing more days changes both evidence and process age.

    **What is this model suitable for?** It gives a precisely specified synthetic
    task: learn which parameter combinations can generate an ordered history,
    and test whether inference recovers those combinations. Its finite-origin
    rough variance law is **not a stationary model of an arbitrary five-year
    market window**. Quiet later segments can be a model effect, rather than
    evidence that a market has permanently become calm. Before using this law
    for real-market continuation forecasts, we would need to review the physical
    variance dynamics and initialization/prehistory, specify the state-inference
    target, and validate a corresponding simulator and estimator. A different
    initialization or mean-reverting rough law would require a new training task;
    changing the plot or reusing the present checkpoint would not implement it.

    - **Supplied curve:** $\xi(t)=0.04$. Mean variance is constant at $0.2^2$;
      each realized $V_i$ still fluctuates. The constant does not say that the
      average realized volatility is exactly 20%.
    - **Inferred curve:** $\xi(t)=L+(S-L)e^{-kt}$. $S$ is the initial level,
      $L$ the long level, and $k$ the transition rate. These coefficients join
      $H,\eta,\rho,\mu$ as targets; they are not extra inputs supplied to the GRU.
      If $S$ and $L$ are similar, changing $k$ barely changes the curve, so
      decay can remain weakly informed.

    **To relax the supplied curve**, select the rough Bergomi joint-curve model
    in **Explore another model**. It has its own estimator: seven fixed
    parameters rather than four. Changing 0.04 under the supplied estimator
    would change its training task, so it cannot be treated as a free query input.

    This choice is separate from filtering. Even after learning $S,L,k$, we
    still have to infer a history's latent state and memory to continue it from
    its last observation. Curve uncertainty is uncertainty about the ensemble
    law; state uncertainty is uncertainty about this particular realization.

    At a fixed driver realization, $\partial_\eta\log V_i=Y_i-\eta q_i$ can
    be negative. Thus larger η means a wider log-variance distribution, not a
    guarantee that every displayed return or variance amplitude increases.
    """)
    mo.accordion({
        "P versus Q: what changes from the option lessons?": mo.md(r"""
        | | Return-history lesson | Option lessons |
        |:---|:---|:---|
        | Data | One ordered history of daily returns | Option prices across strikes and maturities |
        | Law | Physical probability P | Pricing probability Q |
        | Price drift | Inferred physical growth rate $\mu$ | Rate minus dividend yield; zero in the examples |
        | Randomness in an observation | Process shocks along that history | The declared option observation model |

        Fitting either model does not estimate the change of measure between them.
        Similar parameter names therefore do not make the fitted laws interchangeable.
        """),
        "What are the random innovations?": mo.md(_innovations),
        "The discrete model, step by step": mo.md(_calculation),
        "What does the variance curve mean?": mo.md(_curve),
    })
    return










@app.cell(hide_code=True)
def _(mo, lesson_md, lesson_section):
    mo.vstack([lesson_section("Simulate", "Choose parameters and generate ordered return histories before fitting a network.", stage=2), lesson_md(r"""
    ### 2.1 Try the forward model

    **Choose the parameters; see the histories they can generate.** The sliders
    use the model's declared parameter ranges. Eight independent five-year paths
    share those parameters. A fixed seed keeps the same innovations when a
    slider changes, so it is easier to see a parameter's effect.

    **One observed data set is one path's ordered return history.** The eight
    paths below are eight possible histories, not paths averaged into one price
    observation. Each training pair uses just one such history.

    The purple path is shown as daily returns and 21-day realized volatility
    on the right. Realized volatility is a summary of those returns, not the
    hidden instantaneous variance. This panel is a separate forward example;
    the posterior sections retain the selected supplied history.
    """)])
    return


@app.cell(hide_code=True)
def _(cache_task, mo, parameter_labels):
    _defaults = {"hurst": .10, "eta": 1.5, "rho": -.70, "drift": .05,
                 "kappa": 2., "long_variance": .04, "vol_of_vol": .30,
                 "short_variance": .04, "decay": 2.}
    _steps = {"hurst": .01, "eta": .05, "rho": .01, "drift": .01,
              "kappa": .05, "long_variance": .001, "vol_of_vol": .01,
              "short_variance": .001, "decay": .05}
    simulator_controls = mo.ui.dictionary({
        _name: mo.ui.slider(_bounds[0], _bounds[1], step=_steps[_name], value=_defaults[_name],
                           label=parameter_labels[_name], show_value=True, debounce=True)
        for _name, _bounds in zip(cache_task["parameter_names"], cache_task["prior_bounds"], strict=True)})
    _curve_names = {"short_variance", "long_variance", "decay"} if cache_task["model"] == "rough-joint" else set()
    _main = [_name for _name in cache_task["parameter_names"] if _name not in _curve_names]
    _panels = [mo.hstack([simulator_controls[_name] for _name in _main[_i:_i + 2]], justify="start")
               for _i in range(0, len(_main), 2)]
    if _curve_names:
        _panels.append(mo.accordion({"Set the variance-curve coefficients": mo.vstack([
            simulator_controls[_name] for _name in cache_task["parameter_names"] if _name in _curve_names])}))
    if cache_task["model"] != "heston":
        _panels.append(mo.md(r"**η controls log-variance dispersion at a fixed mean.** "
            r"Raising η spreads the distribution; it need not raise a particular variance path. "
            r"The compensator also changes: $\partial_\eta\log V=Y-\eta q$. "
            r"With fixed random inputs, many realized values can fall as η rises."))
    mo.vstack(_panels)
    return (simulator_controls,)


@app.cell(hide_code=True)
def _(mo, trajectory):
    simulator_scale = mo.ui.dropdown(options=list(trajectory.SIMULATOR_SCALES), value="Standard",
                                     label="Fixed simulator axes")
    simulator_view = mo.ui.radio(options=["Explore one setting", "Compare three settings"],
                                 value="Explore one setting", inline=True, label="Simulator view")
    simulator_refresh = mo.ui.button(value=0, on_click=lambda value: value + 1, label="New realizations")
    mo.vstack([simulator_view, mo.hstack([simulator_scale, simulator_refresh], justify="start"), mo.md(
        "The axis limits stay fixed when model parameters change. Choose a wider preset yourself. "
        "All simulated points remain in the data; counts on each panel report points outside the view. "
        "New realizations refreshes the random inputs for every setting together.")])
    return simulator_refresh, simulator_scale, simulator_view


@app.cell(hide_code=True)
def _(cache_task, mo, parameter_labels, simulator_view):
    _choices = ["vol_of_vol", "kappa"] if cache_task["model"] == "heston" else ["hurst", "eta"]
    simulator_compare = mo.ui.dropdown(options={parameter_labels[name]: name for name in _choices},
        value=parameter_labels[_choices[0]], label="Parameter to compare")
    simulator_compare if simulator_view.value == "Compare three settings" else mo.md("")
    return (simulator_compare,)


@app.cell(hide_code=True)
def _(
    cache_task,
    mo,
    simulator_compare,
    simulator_controls,
    simulator_refresh,
    simulator_scale,
    simulator_view,
    trajectory,
):
    try:
        _seed = 20261108 + simulator_refresh.value
        simulation_preview = None
        _limits = trajectory.SIMULATOR_SCALES[simulator_scale.value]
        if simulator_view.value == "Compare three settings":
            _comparison = trajectory.simulator_comparison(cache_task, simulator_controls.value,
                simulator_compare.value, paths=4, days=1260, seed=_seed)
            _display = trajectory.simulator_comparison_figure(_comparison, scale=simulator_scale.value)
            _detail = mo.md("**Held fixed:** " + ", ".join(
                f"{name} = {value:g}" for name, value in _comparison["fixed_parameters"].items())
                + f". Seed {_seed}; four matched realizations per setting. "
                "The selected parameter slider is replaced by the three displayed values. "
                "One realization need not change monotonically. Change the random inputs and inspect again.")
        else:
            simulation_preview = trajectory.simulator_preview(
                cache_task, simulator_controls.value, paths=8, days=1260, seed=_seed)
            _display = trajectory.forward_simulator_figure(simulation_preview, scale=simulator_scale.value)
            _detail = mo.accordion({"Exact axis limits, outside counts and data ranges": mo.ui.table(
                trajectory.simulator_axis_summary(simulation_preview, simulator_scale.value),
                selection=None, pagination=False)})
        _panel = mo.vstack([
            mo.md(f"**Fixed axes ({simulator_scale.value}):** asset price "
                  f"**{_limits['prices'][0]:g}–{_limits['prices'][1]:g}**; daily return "
                  f"**{_limits['returns'][0]:g}–{_limits['returns'][1]:g}%**; realized volatility "
                  f"**{_limits['realized'][0]:g}–{_limits['realized'][1]:g}%**."),
            _display, _detail,
        ])
    except (ValueError, FloatingPointError) as _error:
        simulation_preview = None
        _panel = mo.callout(mo.md("This forward example could not be simulated: " + str(_error)), kind="warn")
    _panel
    return


@app.cell(hide_code=True)
def _(cache_task, mo):
    _amplitude = "σ (volatility of variance)" if cache_task["model"] == "heston" else "η (log-variance dispersion)"
    _memory = "κ" if cache_task["model"] == "heston" else "H"
    mo.md(f"""
    **Predict → move one slider → inspect several paths.** Keep the innovations fixed.

    | Change | Predict and inspect |
    |:---|:---|
    | Raise μ only | The cumulative log-return paths shift upward; variance dynamics keep the same innovations. |
    | Change {_amplitude} | Which paths develop larger bursts? One fixed realization need not change monotonically. |
    | Move ρ toward zero | Compare price/volatility co-movement; rolling realized volatility is a return summary. |
    | Change {_memory} | Compare how volatility patterns persist or roughen, rather than reading the parameter from one price path. |

    Revealing more observed days later uses a longer prefix of one history. Rough
    histories retain their original starting point and memory; they are not restarted
    at the first day visible on a plot.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ```python
    # Forward simulation at chosen parameters; fixed seed keeps innovations fixed.
    parameters = simulator_controls.value
    preview = trajectory.simulator_preview(
        cache_task, parameters, paths=8, days=1260, seed=20261108)
    # Plot returns and return-based summaries; no posterior is fitted here.
    trajectory.forward_simulator_figure(preview, scale=simulator_scale.value)
    # Scale selection changes the view only; no path value is clipped or discarded.
    ```

    The helper calls `simulation.simulate_parameters`, which uses the **same
    numerical implementation** as training-pair generation. Only daily returns
    leave that simulator. No latent variance is supplied to inference.

    **Read the panels:** prices start at 100 and equal
    $100\exp(\sum_{j\le t}r_j)$. Daily-return bars multiply log returns by 100.
    Realized volatility is the highlighted path's rolling 21-day return standard
    deviation, multiplied by $\sqrt{252}$ and shown in percent. It is a noisy
    return summary, not the hidden $\sqrt{V_t}$.
    """)
    return


@app.cell(hide_code=True)
def _():
    # In-memory inputs only, kept separate for every complete task identity.
    diagnostic_panels = {}
    return (diagnostic_panels,)


@app.cell(hide_code=True)
def _(mo, lesson_md):
    lesson_md(r"""
    ### 2.2 Prior predictive: could this model generate a history like this?

    The sliders hold parameters fixed. A **prior-predictive check** instead draws
    parameters from their prior, then simulates one history for each draw. Compare
    200 histories at the same observed length before using the selected posterior.
    Each row is **one parameter vector and one process realization**, not an
    option-price average over many paths:

    $$p(r_{1:T})=\int p(r_{1:T}\mid\Theta)\,p(\Theta)\,d\Theta.$$

    By default compare with **VOO from 2020, using the same number of observed
    days**. You can instead select the supplied synthetic history. This is a
    prior-predictive adequacy check only: real returns are **not sent into the
    neural estimator**, and no real-data posterior is inferred here.

    **Does return order matter?** Six daily log returns, three +2% and three −3%,
    have cumulative log return −3%, annualized RMS 40.47%, and maximum daily size 3%
    in either order. Yet `+ + + − − −` has **8.61% maximum drawdown**, while
    `+ − + − + −` has **4.88%**. Magnitude summaries miss this difference.

    The default view connects return paths, cumulative log wealth and the return
    distribution. **Timing checks** compare signed and absolute-return dependence.
    A constant-variance **IID Gaussian fitted to this observation's mean and SD**
    provides a descriptive control. It is not another prior-predictive model.
    Finite Gaussian histories can also produce large moves and apparent clusters;
    compare their frequencies rather than declaring those observations impossible.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    prior_target = mo.ui.radio(options={"VOO from 2020, same number of days": "market",
        "Selected synthetic history": "synthetic"}, value="VOO from 2020, same number of days",
        label="Observation for the prior-predictive comparison")
    prior_target
    return (prior_target,)


@app.cell(hide_code=True)
def _(mo):
    ordering_shuffle = mo.ui.checkbox(value=False, label="Overlay a fixed shuffle of the observed prefix in ordering checks")
    mo.accordion({"Optional timing comparison": ordering_shuffle})
    return (ordering_shuffle,)


@app.cell(hide_code=True)
def _(mo, prior_target, selection):
    _selected_history = (selection,prior_target.value)
    prior_check_button = mo.ui.run_button(label="Simulate 200 histories from the prior")
    mo.accordion({"Optional: run the prior-predictive comparison": mo.vstack([
        prior_check_button,
        mo.md("Generate one history per parameter draw. The same complete histories are reused "
              "when changing the observed prefix; no inference or training runs here."),
    ])})
    return (prior_check_button,)


@app.cell(hide_code=True)
def _(active_history, cache_task, diagnostic_panels, mo, model_checks, path_checks,
      path_simulation, prior_check_button, prior_target, requested_days, trajectory):
    prior_predictive_result = None
    _status = None
    if prior_check_button.value:
        _key = path_simulation.task_fingerprint(cache_task)
        try:
            if _key not in diagnostic_panels:
                diagnostic_panels[_key] = path_checks.generate_panel(cache_task, count=200, seed=20261111)
            if prior_target.value == "market":
                _market = trajectory.load_market_example("voo-2020-20261113")
                _observation = _market["log_returns"][:requested_days, None]
                _label = f"VOO · {_market['dates'][0]} to {_market['dates'][requested_days-1]} · {requested_days} trading days"
            else:
                _observation = active_history["returns"]
                _label = f"Selected synthetic history · {requested_days} trading days"
            prior_predictive_result = model_checks.compare_prior(cache_task, diagnostic_panels[_key],
                        _observation, requested_days)
            prior_predictive_result["comparison_observation"] = _label
        except (OSError, KeyError, ValueError, RuntimeError, FloatingPointError) as _error:
            _status = mo.callout(mo.md(f"**Prior check stopped:** {_error}. No histories are replaced or filtered."), kind="warn")
    _status
    return (prior_predictive_result,)

@app.cell(hide_code=True)
def _(
    mo,
    model_checks,
    ordering_shuffle,
    predictive_visuals,
    prior_predictive_result,
):
    if prior_predictive_result is None:
        _display = mo.md("")
    else:
        _comparison = predictive_visuals.compare(prior_predictive_result)
        _control = ("Brown is a separate 200-history IID Gaussian control fitted to the observed prefix. "
                    "It matches its mean and variance by construction; tails and timing can still differ. "
                    if _comparison["metadata"]["gaussian_available"] else
                    "The fitted Gaussian control is unavailable for fewer than two varying observations; "
                    "model histories are still shown. ")
        _display = mo.vstack([
            mo.ui.tabs({"Paths and distributions": predictive_visuals.paths_and_distributions(_comparison),
                        "Summary checks": model_checks.plot_check(prior_predictive_result),
                        "Timing checks": predictive_visuals.timing(prior_predictive_result,_comparison,show_shuffle=ordering_shuffle.value)}),
            mo.ui.table(predictive_visuals.feature_rows(_comparison),selection=None,pagination=False),
            mo.md(f"**Observation: {prior_predictive_result['comparison_observation']}.** "
                  f"Prior simulations: {prior_predictive_result['length']} days each · 200 histories. "
                  "Blue bands are pointwise central 90% ranges, computed from all model histories. "
                  + _control +
                  "The return and cumulative-log-wealth views have fixed limits; crosses report values outside "
                  "the view without removing them. Distribution bins use all days in their denominators. "
                  "Zero tail frequency remains visible and means no exceedance in this finite history. "
                  "These are descriptive comparisons, not joint-support or calibrated rejection tests. "
                  "Undefined short/constant-series dependence remains explicit."),
            mo.accordion({"What this Gaussian comparison can and cannot establish":mo.md(
                "The IID Gaussian fixes its data-fitted mean and SD; blue simulations average over prior "
                "or posterior parameter uncertainty. Those are different uncertainty budgets, so this is "
                "a mechanism comparison, not a marginal-likelihood model-selection calculation. A Gaussian "
                "model with uncertainty about its variance can itself have heavy predictive tails. A "
                "stochastic-volatility model can also have Gaussian innovations and non-Gaussian marginal "
                "returns. Timing checks ask the additional question of whether large amplitudes cluster.")}),
        ])
    _display
    return


@app.cell(hide_code=True)
def _(mo, lesson_md):
    mo.vstack([lesson_md(r"""
    ### 2.3 Why not just fit this model in PyMC?

    **Rough Bergomi makes this a hidden-path problem.** Its Volterra kernel
    makes variance depend on earlier innovations, with weights that change
    with H. In the five-year conditional formulation below, roughly **10,000
    latent variance-driver and bridge coordinates** remain after integrating
    out the independent price shocks. We must account for those hidden
    histories alongside the four or seven fixed parameters.

    **PyMC can fit stochastic-volatility models.** The challenge here is
    constructing and sampling the joint parameter–path target, or integrating
    the hidden paths with a suitable specialized method.

    | Route | What has to be computed? | Where does the cost occur? |
    |:---|:---|:---|
    | Latent-path MCMC | Infer parameters together with hidden driving shocks, or integrate/filter those shocks using a specialized method | Repeated likelihood or latent-path work for each observed dataset |
    | Our BayesFlow estimator | Learn parameters from many simulated parameter–return pairs, with hidden paths averaged over by simulation | Upfront simulation and training; later conditional sampling is reusable |

    For one dataset a carefully built MCMC analysis may be the better choice.
    Amortization is attractive when we will analyze many histories or many
    prefixes under the **same simulation design and prior**. Its price is an
    approximate learned posterior that needs the checks below. We have not
    benchmarked PyMC against these networks, and training is not free.
    """), mo.accordion({"Where the likelihood and PyMC implementation become harder": mo.md(r"""
    For our discretized rough Bergomi task, a likelihood is available
    **conditional on the latent Gaussian innovations**.
    Let $U$ contain the variance-driving and hybrid-bridge shocks. For one
    observed day, summing its fine steps gives

    $$
    I_t=\sum_{j\in t} V_j h,\qquad
    J_t=\sum_{j\in t}\sqrt{V_jh}\,Z_{W,j}.
    $$

    $$
    r_t\mid\Theta,U\sim\mathcal{N}\!\left(
      \mu\Delta-\frac{1}{2}I_t+\rho J_t,\ (1-\rho^2)I_t\right).
    $$

    The Normal's second argument is its variance. The independent asset-shock
    channel has been integrated out. The parameter posterior still contains
    a latent-variable integral:

    $$p(\Theta\mid r_{1:T})\propto p(\Theta)
      \int \left[\prod_{t=1}^T p(r_t\mid\Theta,U)\right]\phi(U)\,dU.$$

    Here $\phi(U)$ is the standard Gaussian density of all the independent
    volatility-driving and bridge shocks. There are two conventional routes:

    - **Augment the target:** sample $\Theta,U$ together. Each log density is
      deterministic and computable; NUTS does not have to evaluate the integral
      inside every step. It must explore the much larger joint posterior and
      differentiate the variance construction as H and the shocks change.
    - **Marginalize the shocks:** evaluate or estimate the integral for each
      proposed $\Theta$, using a suitable integration or filtering method.
      Rough memory and long histories make that integration demanding too.

    Five years × 252 days × four fine steps × two Gaussian driving channels
    gives roughly **10,000 latent coordinates** in a direct augmented model.
    The H-dependent memory kernel couples these to the whole variance history.

    A PyMC/NUTS implementation would need a deterministic differentiable log
    density, validated gradients and a parameterization that samples well.
    Our NumPy/FFT forward simulator is not automatically that PyTensor model.
    Drawing fresh simulator randomness on every ordinary NUTS likelihood call
    does not create a valid deterministic target. Particle and pseudo-marginal
    approaches are alternatives with their own requirements.

    See PyMC's [stochastic-volatility example](https://www.pymc.io/projects/examples/en/latest/time_series/stochastic_volatility.html)
    and [external-likelihood guide](https://www.pymc.io/projects/examples/en/latest/howto/blackbox_external_likelihood_numpy.html).
    These explain feasible routes; they are not a matched performance comparison
    with the Bergomi task shown here.
    """)})])
    return


@app.cell(hide_code=True)
def _(mo, trajectory, lesson_md, lesson_section):
    _figure = trajectory.ASSETS / "illustrations" / "application-workflows-20261101" / "trajectory-prefix-workflow.png"
    mo.vstack([
        lesson_section("Train", "Train a compact posterior estimator on simulated parameter–history pairs.", stage=3), mo.md(""
              "**One training pair: parameters drawn from the prior and the daily returns they generate.** "
              "Only returns enter the estimator. Generating parameters are training labels; latent "
              "variance and shocks stay inside the simulator."),
        mo.image(_figure) if _figure.is_file() else mo.md(""),
        lesson_md(f"""
        ### 3.1 Generate parameter–history pairs

        ```python
        from sbi4f_tutorial.trajectories import simulation, inference

        # A physical-measure return model; process randomness needs no extra quote noise.
        task = simulation.teaching_task("rough-known", observations=1260)
        pairs = simulation.generate(
            task, "train", start=0, count=128, seed=20261107, backend="numpy")
        # A fresh, independent history is used only after training.
        example = simulation.generate(
            task, "validation", start=0, count=1, seed=20261108, backend="numpy")
        # 128 training histories, each with 1,260 daily returns (five years)
        # One independently generated history for the inference example
        ```

        The simulator samples both parameters and process innovations. It retains
        aligned parameter labels, daily returns and history IDs. This tutorial route
        uses **rough Bergomi with a supplied curve**, five-year histories and drift
        inference, regardless of the model selected for the supplied examples.
        The implementation lives in `src/sbi4f_tutorial/trajectories/`.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo, trajectory, lesson_md):
    from sbi4f_tutorial.trajectories import simulation as _input_simulation
    _task = _input_simulation.teaching_task("rough-known", observations=1260)
    mo.vstack([
        lesson_md("### 3.2 From a return sequence to network inputs\n\n"
              "The tutorial and supplied models both use a 1,260-day maximum (five trading years). Each observed day becomes a scaled return "
              "and a known time coordinate. Padding gets a false mask; a genuine zero return remains "
              "observed. Generating parameter labels follow a separate training route."),
        trajectory.input_concept_figure(_task),
        mo.md("The query supplies the upper route only. `training_data` also transforms the "
              "parameter labels for density scoring during training. Both models "
              "use the same input format and daily time step."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    from sbi4f_tutorial.trajectories import training_widget
    from sbi4f_tutorial.trajectories import simulation as _training_animation_simulation
    _animation_task = _training_animation_simulation.teaching_task("rough-known", observations=1260)
    # Exactly the first training pair used by the live tutorial route, generated once.
    _animation_pair = _training_animation_simulation.generate(
        _animation_task, "train", start=0, count=1, seed=20261107, backend="numpy")
    mo.vstack([mo.iframe(training_widget.render_html(
        _animation_task, _animation_pair["observations"][0], parameters=_animation_pair["parameters"][0]), height="950px"),
        mo.md("**What to learn:** changing the observed prefix changes tokens, padding and known time context; "
              "the generating parameter label stays fixed. Compare Training with Inference to see why that label "
              "scores a density during fitting but is absent from a posterior query. The original workflow "
              "graphic remains above. This guided illustration performs no fitting or sampling; the actual Train button below does.")])
    return (training_widget,)


@app.cell(hide_code=True)
def _(mo, training_widget):
    mo.Html(training_widget.encoder_comparison_html())
    return


@app.cell(hide_code=True)
def _(trajectory):
    from sbi4f_tutorial.trajectories import simulation as _architecture_simulation
    _task = _architecture_simulation.teaching_task("rough-known", observations=1260)
    trajectory.architecture_figure(_task, network=trajectory.TUTORIAL_NETWORK)
    return


@app.cell(hide_code=True)
def _(mo, lesson_md):
    lesson_md(r"""
    ### 3.3 Build the posterior estimator

    **Keep the order; ignore the padding.** A causal GRU reads returns in order
    and produces a learned summary. The conditional flow maps that summary to a
    joint parameter distribution. Both networks train together.

    The **encoder** extracts temporal features. The **density estimator** turns
    the summary and known time design into uncertainty about the parameters.
    The tutorial GRU emits **16 learned summary values**; the supplied models emit
    **32**. The flow also receives **two separate known time features**: observed
    length / maximum length and the observation step in reference daily units.
    Summary dimensions are compression choices, not parameter counts or guarantees
    of sufficient statistics. A feature discarded by the summary is unavailable
    to the flow; validation must check the resulting posteriors.
    A spline coupling block transforms part of a parameter vector conditional
    on the rest; several blocks can represent curved, dependent distributions.

    | Component | Tutorial model | Supplied model |
    |:---|:---|:---|
    | Causal GRU | 16 hidden units → 16 summary values | 64 hidden units → 32 summary values |
    | Spline coupling flow | 2 blocks, subnets [32, 32], 8 bins | 6 blocks, subnets [64, 64], 8 bins |
    | Training data | 128 five-year histories | One million five-year histories per model |

    **What changes from the option lessons?** The `BasicWorkflow` pattern stays:
    simulate pairs, adapt inputs, fit, then sample. The observation representation
    and the two networks change.

    | Piece | Heston / rough option examples | This return-history example |
    |:---|:---|:---|
    | Observations | Flat quote vector in `inference_conditions` | Ordered returns in `summary_variables`, with an observation mask |
    | Encoder | No sequence encoder in the displayed workflow | Causal GRU learns a summary of the observed prefix |
    | Density estimator | `FlowMatching` in the displayed small fit | Spline `CouplingFlow`, conditional on the summary and known time design |
    | Parameter bounds and units | Adapter constraints and fitted standardization | Fixed return units and prior-CDF logits in `training_data`; `sample` reverses the parameter transform |
    | Additional noise | Prescribed Gaussian quote errors | None; the return process is already stochastic |

    **Why these choices?**

    | Choice here | Reason | Other choices |
    |:---|:---|:---|
    | Causal GRU with a mask | Return order and observed length matter; padding is not data | An attention-based sequence encoder would require its own training; a permutation-invariant quote-set encoder discards order |
    | Spline `CouplingFlow` | Evaluates the learned density and samples through finite invertible blocks | `FlowMatching` learns a velocity field and samples by integration, as in the option fits |
    | Fixed return units and prior-CDF logits | Preserve amplitude information and the declared prior support | A learned preprocessing transform is a new training choice, not a change to apply to existing weights |
    | Random observed prefixes | Reuse one estimator across lengths within its exact model, prior and daily design | A different prior, observation law or initialization needs a new estimator |

    The adapter below routes arrays. It receives the already transformed targets
    and scaled return tokens; it does not fit another standardizer. This is the
    construction inside `inference.build_workflow(task, network=trajectory.TUTORIAL_NETWORK)`:

    ```python
    import bayesflow as bf

    # Preprocessing has already scaled returns and transformed bounded parameters.
    # Route the ordered history to a summary, and known times to the density.
    adapter = (bf.Adapter().convert_dtype("float64", "float32")
        .rename("parameters", "inference_variables")
        .rename("observations", "summary_variables")
        .rename("time_context", "inference_conditions"))
    # The encoder reads observed returns in order; the mask excludes padded days.
    summary = bf.networks.RecurrentNetwork(
        hidden_dim=16, summary_dim=16, recurrent_type="gru",
        bidirectional=False, time_axis=1, time_embed_dim=4,
        dropout=0.0, return_sequences=False)
    # This spline flow replaces FlowMatching in the displayed option fits.
    density = bf.networks.CouplingFlow(
        depth=2, transform="spline", subnet_kwargs={"widths": [32, 32]},
        transform_kwargs={"bins": 8})
    # BasicWorkflow is unchanged; now it trains the summary and density together.
    workflow = bf.BasicWorkflow(
        adapter=adapter, summary_network=summary,
        inference_network=density, standardize=None)
    ```

    `training_data` scales returns in fixed units, adds known time coordinates
    and a boolean mask, and transforms parameter labels to unbounded coordinates.
    The GRU state summarizes observations; it is not a fitted latent-volatility filter.
    """)
    return


@app.cell(hide_code=True)
def _(mo, lesson_md):
    mo.vstack([
        lesson_md(r"""
        ### 3.4 Train on different amounts of evidence

        **Reveal a random prefix, then mask its future.** This teaches one network
        to accept shorter or longer histories without treating padding as data.

        The executing function below prepares training and independent validation
        histories, constructs the GRU and flow, fits both together, then samples
        the independent example. Its loss curves use `bf.diagnostics.loss`.

        This continues the simulation example above. Two epochs on 128
        histories make **eight optimizer updates**. The last call infers parameters
        for the independent example using the newly trained model. It illustrates
        the workflow; no accuracy or calibration result is established.
        The 64 validation histories score the fit but do not update its weights. Two epochs cannot establish convergence or calibrated uncertainty. No training starts on opening the notebook.
        """),
        mo.accordion({"Training target and full-scale reproduction": mo.vstack([
            mo.md(r"The target integrates out hidden paths z: "
                  r"$p(\Theta\mid r_{1:T})\propto p(\Theta)\int p(r_{1:T},z\mid\Theta)dz$. "
                  "Training minimizes negative log density of transformed generating parameters, "
                  "conditional on the summary and time design. Labels never enter inference conditions."),
            mo.md("The supplied fits use one million independent training histories per model, "
                  "with independent validation/test histories, a 64-unit GRU, a 32-value summary "
                  "and six spline coupling layers. Generation precedes training. "
                  "A streamed runner reads bounded shards, with a small cache. "
                  "The fits completed five epochs; exact reproduction also requires their "
                  "recorded seeds, optimizer and checkpoint selection."),
            mo.md(r"""```bash
    # From lab/, with the matching native bank.
    python scripts/train_path_posterior.py \
        --bank /path/to/native-bank --out /path/to/new-run \
        --streaming --training-limit 1000000 --cache-shards 2 \
        --prefix-policy uniform_log_mixture --epochs 5 --batch-size 128
    ```"""),
            mo.md("The prefix policy mixes uniform integer lengths with more exposure to short histories. "
                  "Every integer length is possible; the prior and process law stay unchanged. "
                  "Supporting a length does not establish calibration at that length."),
        ])}),
    ])
    return


@app.cell(hide_code=True)
def _(mo, lesson_md):
    lesson_md(r"""
    ### 3.5 Run the tutorial training route

    The button executes the simulation, network, training and sampling steps
    above. It creates a **new five-year rough Bergomi model**. Its loss and
    posterior summary appear here, separately from the supplied model below.

    The button runs these steps on the CPU, keeps results only in this session,
    and reports generation, model-building/training and sampling time separately.
    First use includes backend imports and compilation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    train_demo_button = mo.ui.run_button(label="Train the tutorial model and infer its example")
    train_demo_button
    return (train_demo_button,)


@app.cell(hide_code=True)
def _(mo, path_simulation, trajectory):
    def train_tutorial_model():
        import time
        import bayesflow as bf
        import keras
        from sbi4f_tutorial.trajectories import inference

        started = time.perf_counter()
        task = path_simulation.teaching_task("rough-known", observations=1260)
        # Three independent random streams: fitting, validation loss, and the example.
        pairs = path_simulation.generate(task, "train", 0, 128, seed=20261107, backend="numpy")
        validation = path_simulation.generate(task, "validation", 0, 64, seed=20261112, backend="numpy")
        example = path_simulation.generate(task, "validation", 0, 1, seed=20261108, backend="numpy")
        generated = time.perf_counter()
        # Fixed units, bounded-label transforms, time tokens and masks are shared with inference.
        train = inference.training_data(pairs, task)
        validate = inference.training_data(validation, task)
        keras.utils.set_random_seed(20261107)
        workflow = inference.build_workflow(task, network=trajectory.TUTORIAL_NETWORK)
        prefixes = inference.random_prefixes(task, seed=20261109, policy="uniform_log_mixture")
        # The same offline API as SPX; jointly train summary and density for eight updates.
        history = workflow.fit_offline(data=train, validation_data=validate, epochs=2,
                                       batch_size=32, augmentations=prefixes, verbose=0)
        fitted = time.perf_counter()
        draws = inference.sample(workflow, task, example["observations"], draws=128, seed=20261110)[0]
        sampled = time.perf_counter()
        return task, workflow, draws, bf.diagnostics.loss(history, smoothing_factor=0, figsize=(8, 3)), {
            "generation": generated-started, "fit": fitted-generated, "sampling": sampled-fitted}

    mo.show_code()
    return (train_tutorial_model,)


@app.cell(hide_code=True)
def _(
    mo,
    path_simulation,
    train_demo_button,
    train_tutorial_model,
    trajectory,
):
    tutorial_task = path_simulation.teaching_task("rough-known", observations=1260)
    tutorial_workflow = None
    _output = mo.md("The training example is idle until requested.")
    if train_demo_button.value:
        try:
            tutorial_task, tutorial_workflow, _draws, _loss, _timing = train_tutorial_model()
            _output = mo.vstack([
                mo.md(f"**Source: newly trained tutorial model · eight optimizer updates.** "
                      f"Generation (128 train, 64 validation and one example): **{_timing['generation']:.2f} s**; "
                      f"model construction and fitting: **{_timing['fit']:.2f} s**; "
                      f"128 posterior draws: **{_timing['sampling']:.2f} s**. Imports precede these timers."),
                _loss,
                mo.md("Training loss scores the fitting histories; validation loss scores independent histories. "
                      "Validation uses their full five-year length while training sees random prefixes, "
                      "so a loss difference also changes the distribution of conditioning lengths. "
                      "Two loss values cannot diagnose convergence, overfitting, information preservation or calibrated posteriors. "
                      "Use the independent recovery and rank checks below."),
                mo.ui.table(trajectory.posterior_rows(tutorial_task, _draws), selection=None, pagination=False),
            ])
        except (ImportError, OSError, RuntimeError, ValueError) as _error:
            tutorial_workflow = None
            _output = mo.callout(mo.md(f"**Tutorial fit stopped:** {_error}."), kind="warn")
    _output
    return tutorial_task, tutorial_workflow


@app.cell(hide_code=True)
def _(mo, lesson_section):
    lesson_section("Infer", "Choose an observed history, condition the supplied estimator on its returns, and inspect parameter uncertainty.", stage=4)
    return

@app.cell(hide_code=True)
def _(cache_arrays, mo, lesson_md):
    history_source = mo.ui.radio(
        {"Supplied history": "saved", "Choose generating parameters": "custom"},
        value="Supplied history", inline=True, label="Observed-history source")
    case_choice = mo.ui.dropdown(
        options={f"History {index + 1}": index for index in range(len(cache_arrays["trajectory_id"]))},
        value="History 1", label="Supplied synthetic history")
    mo.vstack([lesson_md("### 4.1 The observed history"), history_source])
    return case_choice, history_source


@app.cell(hide_code=True)
def _(cache_arrays, cache_task, case_choice, history_source, mo, parameter_labels):
    # These controls use the selected trained task's exact physical parameter bounds.
    _controls = {}
    for _index, (_name, (_low, _high)) in enumerate(zip(cache_task["parameter_names"], cache_task["prior_bounds"])):
        _controls[_name] = mo.ui.slider(float(_low), float(_high), step=float((_high-_low)/200),
            value=float(cache_arrays["truth"][case_choice.value, _index]),
            label=parameter_labels[_name], show_value=True, include_input=True)
    observed_parameters = mo.ui.dictionary(_controls)
    history_seed = mo.ui.number(0, 2**31-1, value=20261114, step=1, label="Random realization seed")
    if history_source.value == "custom":
        _panel = mo.vstack([
            mo.md("Choose parameters within the trained bounds. Changes regenerate one complete 1,260-day history "
                  "with the same seed; change the seed for another realization. Posterior sampling is a separate action below."),
            observed_parameters, history_seed,
        ])
    else:
        _panel = case_choice
    _panel
    return history_seed, observed_parameters


@app.cell(hide_code=True)
def _(cache_arrays, cache_task, case_choice, history_seed, history_source, observed_parameters):
    from sbi4f_tutorial.trajectories import observations as history_data
    if history_source.value == "custom":
        active_history = history_data.custom_history(cache_task,
            [observed_parameters.value[name] for name in cache_task["parameter_names"]], seed=history_seed.value)
    else:
        active_history = history_data.saved_history(cache_task, cache_arrays, case_choice.value)
    return active_history, history_data


@app.cell(hide_code=True)
def _(active_history, mo):
    _options = ({"Use saved posterior samples": "saved", "Compute posterior for selected prefix": "live"}
                if active_history["source"] == "saved" else {"Compute posterior for selected prefix": "live"})
    inference_mode = mo.ui.radio(_options, value=next(iter(_options)), inline=True, label="Posterior calculation")
    inference_mode
    return (inference_mode,)


@app.cell(hide_code=True)
def _(active_history, cache_arrays, inference_mode, mo, trajectory):
    if inference_mode.value == "saved":
        observed_days = mo.ui.slider(steps=trajectory.saved_lengths(cache_arrays), value=1260,
                                    label="Observed trading days", show_value=True)
        _caption = "Saved samples belong to this supplied history at this exact prefix. No model is loaded."
    else:
        observed_days = mo.ui.slider(1, 1260, step=1, value=1260,
                                    label="Observed trading days", show_value=True, include_input=True)
        _caption = ("Choose a prefix, then press Compute posterior below. Only its returns enter the matching network. "
                    + ("Until then, the labelled saved snapshot is shown." if active_history["source"] == "saved"
                       else "A chosen history has no saved posterior; load the matching model to compute one."))
    mo.accordion({"Optional: change the prefix used for posterior calculations": mo.vstack([observed_days, mo.md(_caption)])})
    return (observed_days,)


@app.cell(hide_code=True)
def _(active_history, cache_task, history_data, mo, observed_days, trajectory):
    from sbi4f_tutorial.trajectories import history_widget
    requested_days = observed_days.value
    selection = history_data.selection_key(cache_task, active_history, requested_days)
    # Edit this title and these panel labels directly in marimo.
    _title = (f"{trajectory.MODEL_LABELS[cache_task['model']]} · "
              + (f"synthetic history {active_history['case']+1}" if active_history["source"] == "saved"
                 else "chosen synthetic history"))
    _labels = {
        "returns": "Observed daily log return (%)",
        "variance": "Actual hidden variance V (annualized variance)",
        "returnhist": "Revealed returns (%)",
        "variancehist": "Revealed boundary V",
    }
    _html = history_widget.render_html(active_history["paired_history"], end=requested_days,
        label=_title, labels=_labels, show_header=False, show_footer=False, show_references=False)
    mo.vstack([mo.md(f"**{_title}**"), mo.iframe(_html, height="650px")])
    return requested_days, selection


@app.cell(hide_code=True)
def _(mo, lesson_md):
    lesson_md(r"""
    ### 4.2 Produce and read the posterior

    **Condition the supplied estimator on this history.** Saved draws provide immediate
    results for the supplied examples. A chosen history needs a fresh query using the
    matching pretrained weights; no fitting is performed here.

    ```python
    task, workflow, record = trajectory.load_live(model_choice.value)
    # Only observed returns enter: no future returns, labels or hidden variance.
    revealed = active_history["returns"][None, :observed_days.value]
    draws = trajectory.sample(workflow, task, revealed, draws=512, seed=20261031)[0]
    ```

    Each query uses the original prior and the complete revealed prefix.
    """)
    return


@app.cell(hide_code=True)
def _(cache_task, mo, trajectory):
    mo.accordion({"The supplied model's architecture": mo.vstack([
        trajectory.architecture_figure(cache_task),
        mo.md("This uses the same GRU-plus-coupling-flow construction, with the supplied "
              "64-unit encoder, 32-value summary and six spline blocks. "
              "The loader verifies the task and checkpoint before sampling."),
    ])})
    return


@app.cell(hide_code=True)
def _(inference_mode, mo):
    checkpoint_path = mo.ui.text(value="", placeholder="local-checkpoints/path-rough-known (blank = selected model)",
                                 label="Model folder relative to tutorial/assets/ (optional)", full_width=True)
    if inference_mode.value == "live":
        _panel = mo.accordion({"Model download and folder": mo.vstack([
            mo.md("Download the matching native model as described in the tutorial README. "
                  "Place its unchanged `task.json`, `training.json` and `model.keras` under "
                  "`assets/local-checkpoints/path-rough-known/`, `path-rough-joint/` or `path-heston/`. "
                  "Leave this field blank for the selected model's default. No bank is needed."),
            checkpoint_path,
        ])})
    else:
        _panel = mo.md("")
    _panel
    return (checkpoint_path,)


@app.cell(hide_code=True)
def _(checkpoint_path, inference_mode, mo, requested_days, selection):
    _request = (selection, checkpoint_path.value, inference_mode.value)
    live_button = mo.ui.run_button(label=f"Compute posterior for {requested_days} observed days")
    if inference_mode.value == "live":
        _panel = mo.vstack([live_button, mo.md(
            "The computed samples update the posterior views below. Changing the model, history, "
            "day or model folder clears the previous result and requires a new click.")])
    else:
        _panel = mo.md("")
    _panel
    return (live_button,)


@app.cell(hide_code=True)
def _(active_history, cache_arrays, cache_report, cache_task, checkpoint_path, history_data,
      inference_mode, live_button, model_choice, requested_days, selection, trajectory):
    live_result, live_error, live_timing = None, None, None
    pretrained_workflow = None
    if inference_mode.value == "live" and live_button.value:
        from time import perf_counter as _live_clock
        try:
            _start = _live_clock()
            _task, _workflow, _record = trajectory.load_live(model_choice.value, checkpoint_path.value.strip() or None)
            if _task != cache_task or _record["model_sha256"] != cache_report["checkpoint_sha256"]:
                raise ValueError("checkpoint or task differs from this observed history")
            _load_seconds = _live_clock() - _start
            _revealed = active_history["returns"][None, :requested_days]
            _start = _live_clock()
            _draws = trajectory.sample(_workflow, _task, _revealed, draws=512, seed=20261031)[0]
            _sample_seconds = _live_clock() - _start
            _candidate = {"selection": selection, "draws": _draws}
            history_data.posterior_view(cache_task, cache_arrays, active_history, requested_days, live=_candidate)
            live_result, pretrained_workflow = _candidate, _workflow
            live_timing = (_load_seconds, _sample_seconds)
        except (FileNotFoundError, ValueError, RuntimeError, ImportError, OSError) as _error:
            live_error = str(_error)
    return live_error, live_result, live_timing, pretrained_workflow

@app.cell(hide_code=True)
def _(active_history, cache_arrays, cache_task, history_data, live_result, requested_days):
    posterior = history_data.posterior_view(cache_task, cache_arrays, active_history, requested_days, live=live_result)
    return (posterior,)

@app.cell(hide_code=True)
def _(active_history, inference_mode, live_error, live_timing, mo, posterior, requested_days):
    _contents = []
    if posterior is None:
        _contents.append(mo.md("**No posterior computed for this chosen history yet.** Use the matching pretrained model and press Compute posterior."))
    else:
        _contents.append(mo.callout(mo.md(f"**Source: supplied pretrained model. {posterior['label']}.**"), kind="neutral"))
        if posterior["source"] == "live":
            _contents.append(mo.md(f"Model verification/load: **{live_timing[0]:.3f} s**; "
                                  f"512 paired draws: **{live_timing[1]:.3f} s**. Fresh describes execution, not accuracy."))
        elif inference_mode.value == "live":
            _contents.append(mo.md(f"The display reveals {requested_days} days; the saved posterior uses "
                f"{posterior['length']} days. Press Compute posterior to query the selected prefix."))
    if live_error:
        _suffix = (" The labelled saved result remains visible." if active_history["source"] == "saved"
                   else " No other history's posterior is substituted.")
        _contents.append(mo.callout(mo.md("Posterior calculation is unavailable: " + live_error + _suffix), kind="warn"))
    mo.vstack(_contents)
    return

@app.cell(hide_code=True)
def _(
    active_history,
    cache_task,
    mo,
    np,
    parameter_labels,
    plt,
    posterior,
):
    if posterior is None:
        _display = mo.md("")
    else:
        _names = cache_task["parameter_names"]
        _figure, _axes = plt.subplots((len(_names) + 1) // 2, 2,
                                     figsize=(10, 2.45 * ((len(_names) + 1) // 2)), squeeze=False)
        _label = "fresh neural approximation" if posterior["source"] == "live" else "saved neural approximation"
        for _index, _axis in enumerate(_axes.flat):
            if _index >= len(_names):
                _axis.set_visible(False)
                continue
            _low, _high = cache_task["prior_bounds"][_index]
            _grid = np.linspace(_low, _high, 240)
            _density = (1 / (_grid * np.log(_high / _low))
                        if cache_task["prior_types"][_index] == "log_uniform" else np.full_like(_grid, 1 / (_high - _low)))
            _axis.hist(posterior["draws"][:, _index], bins=np.linspace(_low, _high, 29), density=True,
                       alpha=.42, color="#2563eb", label=_label)
            _axis.plot(_grid, _density, linestyle="--", color="#64748b", linewidth=1.5,
                       label="declared prior density")
            _axis.axvline(active_history["truth"][_index], linestyle=":", color="#111827",
                         linewidth=1, label="synthetic truth")
            _axis.set(xlim=(_low, _high), title=parameter_labels[_names[_index]], ylabel="Marginal density")
        _axes.flat[0].legend(fontsize=7)
        _figure.suptitle(f"Prior and approximate posterior · {posterior['length']} days", fontsize=12)
        _figure.tight_layout()
        _display = _figure
    _display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Read the comparison:** the dotted truth line is for evaluation only; it never
    enters the network. Dashed curves show the declared priors, including log-uniform
    densities. Contraction does not establish accuracy. There is no matched reference
    posterior for these histories and lengths.
    """)
    return


@app.cell(hide_code=True)
def _(active_history, cache_arrays, cache_task, case_choice, mo, parameter_labels, posterior, trajectory, lesson_md):
    if posterior is None:
        _display = mo.md("")
    else:
        _reading = trajectory.posterior_readout(cache_task, posterior["draws"],
                                              truth=active_history["truth"])
        _small = _reading["narrowest_relative_to_prior"]
        _large = _reading["widest_relative_to_prior"]
        _parameters = _reading["parameters"]
        _precision = (f"**Relative width.** {parameter_labels[_small]} has the narrowest central 90% interval "
                      f"relative to its own prior: **{100*_parameters[_small]['width_ratio90']:.1f}%** of the prior's "
                      f"central 90% width. {parameter_labels[_large]} retains **{100*_parameters[_large]['width_ratio90']:.1f}%**. "
                      "These compare equal-probability widths, including the log-uniform priors.")
        _pair = _reading["strongest_pair"]
        _dependence = (f"**Dependence.** The largest absolute Pearson correlation is between "
                       f"{parameter_labels[_pair['names'][0]]} and {parameter_labels[_pair['names'][1]]}: "
                       f"**{_pair['correlation']:+.2f}**. Keep their coordinates paired when using these draws."
                       if _pair else "**Dependence.** Fewer than two coordinates vary in these draws; a Pearson pair correlation is undefined.")
        _misses = _reading["truth_excluded_parameters"]
        _truth = ("**Synthetic truths.** Outside the central 90% marginal intervals: "
                  + ", ".join(parameter_labels[name] for name in _misses) + "."
                  if _misses else "**Synthetic truths.** All generating values lie within their central 90% marginal intervals for this history.")
        _display = mo.vstack([mo.md("**Read this posterior in three steps**\n\n" + _precision),
                   mo.md(_dependence), mo.md(_truth),
                   mo.md("These summaries describe the displayed **approximation** for this history and prefix. "
                         "Width reduction and truth inclusion in one case do not establish accuracy. "
                         "The final calibration failures and model-specific numerical limits still apply."),
                   mo.accordion({"Definitions behind this reading": mo.md(
                       "Relative width is (posterior q95 − q05) / (prior q95 − q05), in physical units. "
                       "It is not an information-distance or accuracy score. Pearson correlation measures linear dependence; "
                       "a small value does not rule out nonlinear dependence. Truth markers are evaluation labels and never network inputs.")})])
    _display
    return


@app.cell(hide_code=True)
def _():
    from sbi4f_tutorial.trajectories import evidence as evidence_frames, evidence_widget
    # Session-only frame/workflow contexts; no files are written by the player.
    evidence_contexts = {}
    return evidence_contexts, evidence_frames, evidence_widget


@app.cell(hide_code=True)
def _(mo, lesson_md):
    lesson_md(r"""
    ### 4.3 Watch the evidence grow

    **Reveal observations; watch the parameter distributions respond.** This local
    player follows the model and history selected above. The upper observed-day
    control sets its maximum; the player cursor does not change the other notebook
    views. Start with the prior, then play, pause, advance or scrub through actual
    posterior frames. Zoom into the first days and switch the time-axis units.

    Drift may remain uncertain after volatility parameters sharpen. Intervals can
    move or widen after surprising returns; the generating parameters stay fixed.
    The posteriors are actual saved or explicitly queried distributions, never
    interpolated. **This is evidence revelation, not hidden-state propagation or
    a particle filter.** No process is regenerated during playback.
    """)
    return


@app.cell(hide_code=True)
def _(mo, requested_days):
    _ends = {f"First {min(21, requested_days)} trading days": min(21, requested_days),
             f"First {min(63, requested_days)} trading days": min(63, requested_days),
             f"All {requested_days} selected trading days": requested_days}
    playback_horizon = mo.ui.dropdown(options=_ends, value=next(iter(_ends)), label="Exact-query playback horizon")
    playback_spacing = mo.ui.radio(options={"Daily": 1, "Weekly (5 trading days)": 5,
        "Monthly (21 trading days)": 21}, value="Daily", inline=True, label="Exact-query spacing")
    return playback_horizon, playback_spacing


@app.cell(hide_code=True)
def _(
    checkpoint_path,
    inference_mode,
    mo,
    playback_horizon,
    playback_spacing,
    selection,
):
    _identity = (selection, checkpoint_path.value, inference_mode.value, playback_horizon.value, playback_spacing.value)
    prepare_playback = mo.ui.run_button(label="Prepare exact-prefix playback")
    if inference_mode.value == "live":
        _panel = mo.accordion({"Optional: compute additional posterior playback frames": mo.vstack([
            playback_horizon, playback_spacing, prepare_playback,
            mo.md("This loads the matching supplied network if necessary and samples up to 61 exact "
                  "prefixes with 128 paired draws each; no training is performed. Daily spacing suits "
                  "the first 21 days. Use wider spacing for a longer horizon. Previously computed days "
                  "are reused within the same task, checkpoint, history and sampling design.")])})
    else:
        _panel = mo.accordion({"Optional: posterior playback source and extra query days": mo.md(
            "Saved playback uses actual cached days. Changing axis units creates no additional "
            "posteriors. To compute intermediate prefixes, choose **Compute posterior for selected prefix** "
            "in the Posterior calculation control above, then expand the additional-frame tools here.")})
    _panel
    return (prepare_playback,)


@app.cell(hide_code=True)
def _(
    active_history,
    cache_arrays,
    cache_report,
    cache_task,
    case_choice,
    checkpoint_path,
    evidence_contexts,
    evidence_frames,
    inference_mode,
    model_choice,
    playback_horizon,
    playback_spacing,
    prepare_playback,
    trajectory,
):
    from time import perf_counter as _playback_clock
    from hashlib import sha256 as _playback_hash
    _checkpoint = cache_report["checkpoint_sha256"]
    _history = active_history["id"]
    _path = checkpoint_path.value.strip() or None
    _history_hash = _playback_hash(active_history["returns"].tobytes()).hexdigest()
    _key = (cache_report["task_sha256"], _checkpoint, _history, _history_hash, inference_mode.value, _path)
    if _key not in evidence_contexts:
        _dense_error = None
        _manager = (evidence_frames.saved_manager(cache_task, cache_arrays, case_choice.value, _checkpoint)
                    if inference_mode.value == "saved" and active_history["source"] == "saved" else evidence_frames.native_manager(
                        cache_task, active_history["returns"], _checkpoint, _history,
                        draws=128, seed=20261113))
        if inference_mode.value == "saved" and active_history["source"] == "saved":
            try:
                from sbi4f_tutorial.trajectories import evidence_assets as _dense_evidence
            except ImportError:
                pass
            else:
                try:
                    _manager = _dense_evidence.load_dense(_manager)
                except (OSError, ValueError) as _error:
                    _dense_error = ("Optional denser replay unavailable: " + str(_error) +
                        ". The original verified snapshots remain available; no extra frames were accepted.")
        evidence_contexts[_key] = {"manager": _manager, "workflow": None, "dense_error": _dense_error}
    _context = evidence_contexts[_key]
    playback_manager = _context["manager"]
    playback_message, playback_error = None, _context.get("dense_error")
    if inference_mode.value == "live" and prepare_playback.value:
        try:
            _days = evidence_frames.exact_schedule(playback_horizon.value, step=playback_spacing.value)
            _started = _playback_clock()
            if _context["workflow"] is None:
                _loaded_task, _workflow, _record = trajectory.load_live(model_choice.value, _path)
                if _loaded_task != cache_task or _record["model_sha256"] != _checkpoint:
                    raise ValueError("playback checkpoint or task differs from this selected history")
                _context["workflow"] = _workflow
            _record = evidence_frames.prepare_frames(playback_manager, _context["workflow"], _days)
            playback_message = (f"Prepared {len(_record['new_lengths'])} new exact prefixes; "
                f"reused {len(_record['reused_lengths'])}. Total verification/query time: "
                f"{_playback_clock()-_started:.2f} s. Playing, scrubbing and changing axis units now reuse these draws.")
        except (ImportError, OSError, ValueError, RuntimeError, FloatingPointError) as _error:
            playback_error = ("Playback preparation stopped: " + str(_error) +
                ". No partial new frame batch is displayed. Any already prepared frames still belong to the labelled context.")
    return playback_error, playback_manager, playback_message


@app.cell(hide_code=True)
def _(
    active_history,
    cache_arrays,
    case_choice,
    evidence_frames,
    evidence_widget,
    inference_mode,
    mo,
    parameter_labels,
    playback_error,
    playback_manager,
    playback_message,
    requested_days,
):
    _frames = evidence_frames.frame_list(playback_manager, requested_days)
    _parts = []
    if playback_error:
        _parts.append(mo.callout(mo.md("**Evidence player notice:** " + playback_error), kind="warn"))
    if playback_message:
        _parts.append(mo.md(playback_message))
    if not _frames:
        _parts.append(mo.md("*The prior is ready. Prepare exact prefixes above before playing the native posterior sequence.*"))
    _html = evidence_widget.render_html(playback_manager["task"],
        active_history["returns"], _frames, end=requested_days,
        labels=parameter_labels, truth=active_history["truth"],
        history_label=f"{active_history['source'].title()} history · {'saved posterior samples' if inference_mode.value == 'saved' else 'computed posterior frames'}")
    _rows = (len(playback_manager["task"]["parameter_names"])+1)//2
    _parts.append(mo.iframe(_html, height=f"{380+210*_rows}px"))
    mo.vstack(_parts)
    return


@app.cell(hide_code=True)
def _(
    active_history,
    cache_arrays,
    cache_task,
    case_choice,
    mo,
    plt,
    requested_days,
    trajectory,
):
    _revealed = active_history["returns"][None, :requested_days]
    _inputs = trajectory.conditioning_data(_revealed, cache_task)
    _maximum = cache_task["n_observations"]
    _mask = _inputs["summary_mask"]
    _rows = sorted({0, requested_days - 1} | set(range(requested_days, min(requested_days + 2, _maximum))))
    _mask_figure, _mask_axis = plt.subplots(figsize=(9, 1.25), layout="constrained")
    _mask_axis.imshow(_mask, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                      extent=(.5, _maximum + .5, -.5, .5), interpolation="nearest")
    _mask_axis.axvline(requested_days + .5, color="#111827", linewidth=1)
    _mask_axis.set(yticks=[], xlabel="Day position in the padded input",
                   title=f"Actual GRU mask: {requested_days} observed days")
    mo.accordion({"Follow the actual network input and its mask": mo.vstack([
        mo.md(f"**{requested_days} observed days; {_maximum - requested_days} padded rows.** "
              "True masks mark observations, including genuine zero returns. False masks mark padding. "
              "Only the revealed prefix and known time design enter inference."),
        _mask_figure,
        mo.md(r"""```python
    revealed = active_history["returns"][None, :observed_days.value]
    inputs = trajectory.conditioning_data(revealed, cache_task)
    # observations, time_context, summary_mask
    ```"""),
        mo.ui.table([{"day": _i + 1, "role": "observed" if _mask[0, _i] else "padding",
                      "scaled return": float(_inputs["observations"][0, _i, 0]),
                      "normalized time": float(_inputs["observations"][0, _i, 1]), "mask": bool(_mask[0, _i])}
                     for _i in _rows], selection=None, pagination=False),
        mo.md(r"Returns use fixed units $r_t/\sqrt{0.04\Delta t}$; 0.04 is a reference variance, "
              "not an estimated latent variance. Normalizing each path by its own volatility would "
              "remove amplitude information. No future return, truth, hidden variance or shock enters these inputs."),
        mo.ui.table([{"input": _name, "shape": str(_value.shape), "dtype": str(_value.dtype)}
                     for _name, _value in _inputs.items()], selection=None, pagination=False),
        mo.md(f"Known time context: **{_inputs['time_context'][0].tolist()}** = "
              "[observed length / 1260, daily step / (1/252 year)]. Return tokens also carry normalized "
              "day position. Padded returns/times are zero with false masks. "
              "This input preview does not load a model or perform inference."),
    ])})
    return


@app.cell(hide_code=True)
def _(mo, lesson_md, lesson_section):
    mo.vstack([lesson_section("Check", "Check replicated histories, parameter recovery and uncertainty before interpreting the results.", stage=5), lesson_md(r"""
    | Check | Question | Evidence here |
    |:---|:---|:---|
    | Replicated histories | Can fitted parameters reproduce these return summaries? | Independent, zero-origin posterior-predictive histories |
    | Many independent histories | Does this estimator recover parameters and report uncertainty? | Saved final-cohort recovery and marginal coverage |
    | One conditional posterior | Does the approximation match an independent numerical reference? | No matched reference for these histories and lengths |

    The saved final-cohort results below answer a broader validation question.
    A successful small fit or a good-looking predictive plot does not replace them.

    ### 5.1 Posterior predictive: replicate, then compare

    Select complete paired parameter draws from the displayed posterior and
    generate a new history from the model's origin for each. Compare the **same
    length used by that posterior** with the observed prefix. These are independent
    replicas, not paths continuing from the end of the observed history.
    """)])
    return


@app.cell(hide_code=True)
def _(mo):
    predictive_count = mo.ui.radio(options={"32 histories: inspect individual curves": 32,
        "256 histories: optional finer simulation envelope": 256},
        value="32 histories: inspect individual curves", label="Posterior replication budget")
    return (predictive_count,)


@app.cell(hide_code=True)
def _(mo, posterior, predictive_count, selection):
    _source = (selection, None if posterior is None else (posterior["source"], posterior["length"]), predictive_count.value)
    replicate_button = mo.ui.run_button(label=f"Replicate {predictive_count.value} histories")
    if posterior is None:
        _panel = mo.md("Compute a posterior for the chosen history before generating its predictive replicas.")
    else:
        _panel = mo.accordion({"Optional: generate a posterior-predictive comparison": mo.vstack([
            predictive_count, replicate_button,
            mo.md(f"**Posterior source: {posterior['source']} · {posterior['length']} days.** "
                  "Independent replicas retain parameter uncertainty and fresh process randomness.")])})
    _panel
    return (replicate_button,)

@app.cell(hide_code=True)
def _(active_history, cache_task, mo, model_checks, posterior, predictive_count, replicate_button):
    posterior_predictive_result = None
    _status = None
    if posterior is not None and replicate_button.value:
        try:
            posterior_predictive_result = model_checks.replicate_check(cache_task,
                active_history["returns"], posterior["draws"], posterior["length"], count=predictive_count.value)
        except (ValueError, RuntimeError, FloatingPointError) as _error:
            _status = mo.callout(mo.md(f"**Replication stopped:** {_error}. No replicas are discarded or replaced."), kind="warn")
    _status
    return (posterior_predictive_result,)

@app.cell(hide_code=True)
def _(
    mo,
    model_checks,
    ordering_shuffle,
    posterior,
    posterior_predictive_result,
    predictive_visuals,
    requested_days,
):
    if posterior_predictive_result is None:
        _display = mo.md("")
    else:
        _result = posterior_predictive_result
        _comparison = predictive_visuals.compare(_result)
        _control = ("The brown IID Gaussian reference is fitted to this observed prefix's mean and SD. "
                    if _comparison["metadata"]["gaussian_available"] else
                    "The Gaussian reference is unavailable for a short or constant prefix. ")
        _display = mo.vstack([
            mo.ui.tabs({"Paths and distributions":predictive_visuals.paths_and_distributions(_comparison),
                        "Summary checks":model_checks.plot_check(_result),
                        "Timing checks":predictive_visuals.timing(_result,_comparison,show_shuffle=ordering_shuffle.value)}),
            mo.ui.table(predictive_visuals.feature_rows(_comparison),selection=None,pagination=False),
            mo.md(f"**{_result['metadata']['count']} independent replicas · {posterior['length']} days each.** "
                  f"The return plot reveals {requested_days} days; this check follows the actual "
                  "posterior conditioning length. The spread retains parameter uncertainty and process "
                  "randomness. The ordering view uses the same prefix, with at most 20 lags and at least "
                  "five observed days per lag; below 21 days it reports insufficient history. "
                  "A shuffled prefix is a teaching intervention, not another model simulation. "
                  "Agreement on these checks does not establish model adequacy or calibrated inference. "
                  "No latent state is filtered and no continuation forecast is generated."),
            mo.md(_control + "**Compare mechanisms:** which features differ from the data-fitted IID Gaussian control? "
                  "Which features does the volatility model still fail to reproduce? A Gaussian control is "
                  "fitted after seeing these returns, whereas blue replicas use the displayed approximate "
                  "parameter posterior. Neither shaded range is a simultaneous path or joint-feature test. "
                  "All model and Gaussian histories remain in the comparisons."),
        ])
    _display
    return


@app.cell(hide_code=True)
def _(json, trajectory):
    aggregate_report, aggregate_images, aggregate_problem = None, {}, None
    _folder = trajectory.ASSETS / "results" / "trajectories" / "closeout-20261031"
    if _folder.is_dir():
        try:
            if not _folder.resolve().is_relative_to(trajectory.ASSETS.resolve()):
                raise ValueError("aggregate evidence must remain within tutorial/assets")
            _accepted = json.loads((_folder / "ACCEPTED.json").read_text())
            for _name in ("closeout-summary.json", "figure-metadata.json", "drift-learning.png", "marginal-coverage.png"):
                _path = _folder / _name
                if (not _path.resolve().is_relative_to(trajectory.ASSETS.resolve())
                        or trajectory.asset_digest(_path) != _accepted["source_sha256"][_name]):
                    raise ValueError("packaged aggregate evidence hash differs: " + _name)
            _report = json.loads((_folder / "closeout-summary.json").read_text())
            if (_report.get("format") != "trajectory_workshop_closeout_summary_v1"
                    or _report.get("scientific_acceptance") is not False):
                raise ValueError("unexpected aggregate evidence scope")
            _catalogue = trajectory.catalogue()["artifacts"]
            if set(_report["models"]) != set(_catalogue):
                raise ValueError("aggregate report and trajectory catalogue model sets differ")
            for _model, _record in _report["models"].items():
                if (_record["task_sha256"] != _catalogue[_model]["task_sha256"]
                        or _record["model_sha256"] != _catalogue[_model]["checkpoint_sha256"]):
                    raise ValueError("aggregate evidence belongs to a different task/checkpoint")
            aggregate_report = _report
            aggregate_images = {"drift": _folder / "drift-learning.png", "coverage": _folder / "marginal-coverage.png"}
        except (OSError, ValueError, KeyError) as _error:
            aggregate_problem = str(_error)
    return aggregate_images, aggregate_problem, aggregate_report


@app.cell(hide_code=True)
def _(aggregate_images, aggregate_problem, aggregate_report, cache_task, mo, parameter_labels, lesson_md):
    _contents = [lesson_md("### 5.2 Saved final-cohort evidence\n\n"
                       "**Learning information and reporting reliable uncertainty are different achievements.** "
                       "Recovery asks whether estimates improve over a prior-only baseline. Calibration asks "
                       "whether nominal 90% intervals contain truth about 90% of the time across independent "
                       "histories drawn from the prior and simulator. One posterior plot cannot establish either.")]
    if aggregate_report is None:
        _contents.append(mo.callout(mo.md(
            "Final-cohort diagnostics are unavailable in this copy. "
            + (aggregate_problem or "The selected history remains an illustration.")), kind="warn"))
    else:
        _record = aggregate_report["models"][cache_task["model"]]
        _records = list(aggregate_report["models"].values())
        _failures = sum(not _entry["calibration_pass"] for _entry in _records)
        _contents.extend([
            mo.callout(mo.md(f"**{_failures} of {len(_records)} supplied estimators failed the final "
                             "marginal-calibration screen.** Lower estimation error and fast inference "
                             "do not establish accurate intervals. Heston also has the drift point-mass limitation."), kind="warn"),
            mo.image(aggregate_images["coverage"]),
            mo.md(f"Black outlines mark failed coverage checks. For the selected model, nominal 90% "
                  f"coverage ranges from **{100 * _record['coverage90_range'][0]:.2f}% to "
                  f"{100 * _record['coverage90_range'][1]:.2f}%** across parameters and histories. "
                  "A narrower interval can be informative and still too narrow."),
            mo.accordion({"Recovery, individual failures and evaluation design": mo.vstack([
                mo.md(f"The checks use **{_record['cases']:,} independent test histories per model**, "
                      f"**{_record['draws']} draws per history**, and twelve frozen history lengths. "
                      "These are separate from the four displayed validation histories. The independent unit "
                      "is a whole history; its nested prefixes are dependent. These reserved tests have "
                      "been examined, so new acceptance work needs fresh data."),
                mo.ui.table([{"observed days": _cell["days"], "parameter": parameter_labels.get(_cell["parameter"], _cell["parameter"]),
                              "coverage of nominal 90% interval": f"{100 * _cell['coverage90']:.2f}%",
                              "screen criterion": _cell["criterion"].replace("_", " ")}
                             for _cell in _record["failed_coverage_cells"]], selection=None, pagination=False),
                mo.md("These are the original joint-screen decisions, not tests selected after viewing "
                      "the figures. Heston drift uses an undercoverage screen only: central intervals "
                      "for posteriors with point masses can legitimately overcover. Even successful marginal "
                      "checks would not establish conditional accuracy or correct joint dependence."),
                mo.image(aggregate_images["drift"]),
                mo.md("**Recovery:** compare mean squared error with always returning the prior mean. "
                      "Below one means improvement; near one means little improvement. Each model has "
                      "its own prior and data. The table summarizes five-year histories."),
                mo.ui.table([{"parameter": parameter_labels.get(_name, _name),
                              "MSE / prior-mean baseline MSE": float(_ratio),
                              "estimation-error screen": _record["five_year_risk_labels"][_name].replace("_", " ")}
                             for _name, _ratio in _record["five_year_mse_to_prior_mean_ratio"].items()], selection=None, pagination=False),
                mo.md(f"{sum(_entry['execution_pass'] for _entry in _records)} estimators passed execution checks; "
                      f"{sum(_entry['risk_pass'] for _entry in _records)} passed their estimation-error screen. "
                      "Those results do not override the calibration failures."),
            ])}),
            mo.md("**Try it:** name a parameter whose estimation error improves, then find a failed "
                  "coverage check. Explain how both can be true."),
        ])
    mo.vstack(_contents)
    return


@app.cell(hide_code=True)
def _(mo, lesson_md, lesson_section):
    mo.vstack([
        lesson_section("Use", "Identify what must be added to turn parameter inference into a continuation forecast.", stage=6),
        lesson_md(r"""
        ### 6.1 What would a continuation forecast need?

        Independent replicated histories check the fitted model from its origin.
        Predicting the next part of the observed history additionally needs its
        current hidden state; rough Bergomi also needs past-shock memory. The GRU
        summary is not a joint state–parameter posterior, so these results are
        not state-conditioned forecasts of subsequent returns, VaR or expected
        shortfall.

        **Joint across parameters is different from joint across time.** We
        already learn one joint posterior over all four, five or seven fixed
        parameters. The separate density plots are marginals of those same
        paired draws; they are not independently trained parameter estimates.
        Keeping those rows together preserves parameter dependence when we
        simulate outcomes. It does not supply a posterior over the hidden
        variance trajectory.

        For example, one rough Bergomi posterior row might be
        $(H,\eta,\rho,\mu)=(0.1,1.2,-0.7,0.05)$. It is one plausible
        **combination**. Plotting H alone hides the other coordinates in the
        display; it does not train a separate H estimator. Taking H from one
        row and η from another would create combinations that the joint
        posterior did not necessarily support.

        **Which inference question did we train?** This distinction matters more
        than whether an encoder happens to be recurrent:

        | Question | Target distribution | Present estimator |
        |:---|:---|:---|
        | Static parameter inference | $p(\Theta\mid r_{1:t})$ | Yes: one joint parameter vector shared across the history |
        | Marginal state filtering | $p(V_t\mid r_{1:t})$ | No: the latent state is not a training target |
        | Joint trajectory smoothing | $p(V_{1:T}\mid r_{1:T})$ | No: past and future observations would inform an entire latent path |
        | Conditional continuation | $p(r_{T+1:T+h}\mid r_{1:T})$ | Needs a suitable joint state–parameter posterior or a directly trained predictive model |

        **Why didn't we train a state estimator?** We chose one teaching target:
        fixed parameters given an ordered return history. Keeping the parameter
        target from the option lessons lets us focus on what changes when the
        observations become a sequence. The training loss
        compares inferred parameters with their generating values; it never asks
        the network to recover the terminal variance or latent shocks. Those
        states exist in simulation, so adding them is possible. It is a different
        inference task that needs its own targets, training design and validation.
        BayesFlow itself does not restrict us to parameter inference. Marginalizing
        hidden paths while inferring static parameters is not the same task as
        marginal state filtering. Our flow retains dependence between the
        components of $\Theta$; it does not return independent parameter marginals.

        With continuous-time Heston, a joint posterior over parameters and today's variance
        would supply a Markov state from which to continue. For rough Bergomi,
        today's variance alone does not summarize how past shocks affect future
        variance. We would need a suitable memory representation, a validated
        finite-factor approximation, or a model trained directly for future
        returns conditional on the past. A GRU summary is not automatically a
        calibrated posterior over any of these states.
        State estimation need not itself be neural: another route is a
        conventional filter conditional on each parameter draw. This lesson
        currently implements neither that filter nor a learned state posterior.

        **Takeaways**

        1. Ordered returns are the data; latent variance and innovations are hidden.
        2. A recurrent summary and conditional flow learn a joint parameter distribution.
        3. Random prefixes and masks let one estimator use different amounts of history.
        4. Prior prediction, replicated-history fit, recovery and calibration ask different questions.
        5. A parameter point hides uncertainty; retain paired posterior draws when using the result.
        """),
        mo.accordion({"What the BayesFlow smoothing example actually changes": mo.md(r"""
        The official [bivariate smoothing tutorial](https://bayesflow.org/v2.0.14/_examples/Bivariate_Smoothing.html)
        ([pinned source notebook](https://github.com/bayesflow-org/bayesflow/blob/v2.0.14/examples/Bivariate_Smoothing.ipynb))
        infers a **time-varying latent trajectory**, not one fixed parameter vector.
        It uses all observations to learn a full joint smoother,

        $$q(z_{1:T}\mid r_{1:T})=
          \prod_{t=1}^T q(z_t\mid z_{1:t-1},r_{1:T}).$$

        Its encoder returns a sequence of representations. An **autoregressive
        decoder** then conditions each latent-state draw on the previously drawn
        states. That is a new target, decoder, training loss and validation task;
        it is not merely another name for our causal GRU summary.

        Here the GRU reads the observed prefix and emits one summary for a
        static-parameter density. Adding a sequence encoder is the central
        representation change from flat option quotes. The time design,
        parameter transforms and conditional-flow architecture also differ in
        the displayed workflows. A recurrent encoder alone does not make this
        an autoregressive posterior over latent states.

        Learning curve coefficients instead of supplying them is orthogonal
        to this choice. Either a fixed-curve or an inferred-curve model could
        support parameter inference, filtering or smoothing with suitable
        training targets and checks. Existing weights implement only the task
        for which they were trained.
        """), "Replicated histories, filtering and continuations": mo.md(r"""
        An independent replicate averages $p(r^{\rm rep}_{1:T}\mid\Theta)$ over
        $p(\Theta\mid r_{1:T})$. A conditional continuation instead needs

        $$p(r_{T+1:T+h}\mid r_{1:T})=
          \int p(r_{T+1:T+h}\mid z_T,\Theta)\,
          p(z_T,\Theta\mid r_{1:T})\,dz_T\,d\Theta.$$

        Here $z_T$ includes hidden state and, for rough Bergomi, memory. Filtering
        infers that state from past observations. This network infers fixed
        parameters after averaging over hidden paths.

        A parameter posterior and a conditional state filter could be combined
        through

        $$p(\Theta,z_T\mid r_{1:T})=
          p(z_T\mid\Theta,r_{1:T})p(\Theta\mid r_{1:T}).$$

        Sample a parameter vector, then a state conditional on that **same**
        vector and history. Independently sampling the two marginal posteriors
        would discard their dependence and can change the predictive result.
        The same principle explains why we keep parameter coordinates paired
        when generating replicated histories. Marginalizing hidden paths in parameter
        inference does not make the parameter posterior factorize.

        Heston's continuous-time state is its variance. Exact continuation of our
        full-truncation numerical scheme also retains its signed auxiliary
        variance; replacing it with its positive part loses numerical state.
        """), "How does amortized-fractals compare?": mo.md(r"""
        The [expanded MMAR tutorials](https://github.com/stefanradev93/amortized-fractals)
        also train **joint parameter-only** estimators: four coordinates in
        the univariate model, 13 in the multivariate model and 16 with its
        Hurst extension. Their one-dimensional posterior plots are views of
        joint draws, just as ours are. Their practical applications
        are more developed: rolling risk estimates, stock screening and
        allocation backtests. Their scenario code draws fresh cascade weights,
        a fresh cascade phase and new innovations after inferring parameters.
        It does not infer the observed window's terminal cascade state and
        continue that same latent realization.

        | | This trajectory lesson | amortized-fractals |
        |:---|:---|:---|
        | Inferred quantity | Parameters of a physical return model | MMAR parameters, including extensions for multiple assets and a fractional driver |
        | Time structure | Daily prefixes of one declared task | Fixed return windows; rolling-window applications |
        | Scenario calculation | Independent histories from the model's origin | New cascades/phases and return innovations |
        | Filtered state continuation | Not trained | Not implemented in the inspected scenario helpers |

        The stock screener applies separate joint-parameter fits to individual
        assets. Those independent asset fits do not specify cross-asset
        dependence. The multivariate MMAR instead trains a joint model with
        correlated innovations and shared cascade structure. Joint dependence
        across parameter coordinates, across assets and across latent times
        are three different modeling choices.

        The MMAR applications can still be evaluated prospectively: infer from
        past data, choose a risk estimate or allocation, then score it on later
        observations. This tests the usefulness of that restart-based rule; it
        does not turn it into a state-conditioned continuation. A random phase
        inside a finite cascade window is also not, by itself, a law for evolving
        that cascade beyond the window.

        Inspect the source
        [forward scenario helper](https://github.com/stefanradev93/amortized-fractals/blob/069d5dc5943dbe85505849f2fa8bfb563617ff97/mmar/univariate.py#L115),
        [rolling risk calculation](https://github.com/stefanradev93/amortized-fractals/blob/069d5dc5943dbe85505849f2fa8bfb563617ff97/mmar/risk.py#L55)
        and [allocation backtest](https://github.com/stefanradev93/amortized-fractals/blob/069d5dc5943dbe85505849f2fa8bfb563617ff97/mmar/allocation.py#L130).
        The Hurst extension likewise resimulates a fractional driver; it does
        not condition its future innovations on the observed past.
        """)}),
    ])
    return


@app.cell(hide_code=True)
def _(cache_report, mo):
    mo.accordion({"Source and reproducibility details": mo.md(
        "The supplied examples use a 64-unit causal GRU, a 32-value summary and six spline "
        "coupling layers. Saved example observations and paired draws are verified against "
        "their task and checkpoint identities; final-test diagnostics describe a separate cohort.  \n\n"
        f"Checkpoint SHA-256: `{cache_report['checkpoint_sha256']}`  \n"
        f"Task SHA-256: `{cache_report['task_sha256']}`  \n\n"
        "An exported HTML file is a static preview. Run the notebook in marimo for controls, "
        "optional training and new posterior calculations.")})
    return


if __name__ == "__main__":
    app.run()
