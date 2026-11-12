import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Learn volatility dynamics from observed paths")


@app.cell
def _():
    import json
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from sbi4f_tutorial.shared import plots
    from sbi4f_tutorial.trajectories import lesson as trajectory
    from sbi4f_tutorial.trajectories import diagnostics as path_checks, simulation as path_simulation, model_checks
    plots.style()
    return json, mo, model_checks, np, path_checks, path_simulation, plt, trajectory


@app.cell(hide_code=True)
def _(mo):
    def lesson_workflow(active=None):
        stages = [("Model", "Parameters and physical return law"),
                  ("Simulate", "Histories before seeing data"),
                  ("Train", "Learn a conditional density"),
                  ("Infer", "Condition on observed returns"),
                  ("Check", "Replicate, recover, calibrate"),
                  ("Use", "Interpret joint uncertainty")]
        boxes = []
        for name, detail in stages:
            selected = active is None or active == name
            border = "#2563eb" if selected else "#d1d5db"
            fill = "rgba(37,99,235,.07)" if selected else "transparent"
            boxes.append(f'<div style="border:1px solid {border};background:{fill};border-radius:8px;'
                         f'padding:9px 10px"><strong>{name}</strong>'
                         + (f'<div style="font-size:.8em;line-height:1.35;margin-top:4px">{detail}</div>'
                            if active is None else "") + '</div>')
        return mo.Html('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));'
                       'gap:6px;margin:8px 0">' + ''.join(boxes) + '</div>')
    return (lesson_workflow,)


@app.cell(hide_code=True)
def _(lesson_workflow, mo):
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

    First inspect an observed history, then explore the forward simulator. Both
    training and model loading are explicit actions; the small simulator runs locally.

    The supplied estimators learn useful information, but **all three failed
    their final marginal-calibration screen**. We will inspect that distinction.
    """), lesson_workflow(),
    mo.accordion({"Where BayesFlow is used": mo.md(r"""
    | Call | Role here |
    |:---|:---|
    | `simulation.generate` (NumPy) | Draw parameters and daily return histories before fitting |
    | `bf.Adapter` | Route transformed labels, return tokens, time context and mask |
    | `bf.networks.RecurrentNetwork` | Learn a causal GRU summary of the observed prefix |
    | `bf.networks.CouplingFlow` | Learn the conditional joint parameter density |
    | `bf.BasicWorkflow`, `fit_offline` | Fit the summary and density on simulated pairs |
    | `workflow.sample` through `inference.sample` | Condition on returns and restore physical parameter units |
    | `bf.diagnostics` | Plot loss, recovery, ranks and a selected posterior's parameter pairs |
    """)})])
    return


@app.cell
def _(mo, trajectory):
    model_choice = mo.ui.dropdown(
        options={label: key for key, label in trajectory.MODEL_LABELS.items()},
        value="Rough Bergomi · known variance curve", label="Stochastic volatility model")
    mo.accordion({"Explore another model": mo.vstack([
        model_choice,
        mo.md("The other examples infer the rough variance curve or use Heston. "
              "Each model has its own synthetic histories and fitted estimator; "
              "switching models changes the data as well as the assumptions."),
    ])})
    return (model_choice,)


@app.cell
def _(mo, model_choice, trajectory):
    mo.stop(not trajectory.CATALOGUE.is_file(), mo.callout(mo.md(
        "**Trajectory examples are not packaged in this copy.** See the tutorial "
        "README for the matching assets. No training or model download starts automatically."), kind="info"))
    cache_task, cache_report, cache_arrays = trajectory.load_example(model_choice.value)
    return cache_arrays, cache_report, cache_task


@app.cell
def _(cache_arrays, mo):
    case_choice = mo.ui.dropdown(
        options={f"History {index + 1}": index for index in range(len(cache_arrays["trajectory_id"]))},
        value="History 1", label="Supplied synthetic history")
    inference_mode = mo.ui.radio(
        options={"Saved results": "saved", "Fresh inference at any day": "live"},
        value="Saved results", inline=True, label="Supplied estimator: saved or loaded")
    mo.vstack([mo.md("## The observed history"), case_choice, inference_mode])
    return case_choice, inference_mode


@app.cell
def _(cache_arrays, inference_mode, mo, trajectory):
    _lengths = trajectory.saved_lengths(cache_arrays)
    if inference_mode.value == "saved":
        observed_days = mo.ui.slider(steps=_lengths, value=252,
                                    label="Observed trading days", show_value=True)
        _caption = "Saved mode selects an actual posterior snapshot. No model is loaded."
    else:
        observed_days = mo.ui.slider(1, 1260, step=1, value=252,
                                    label="Observed trading days", show_value=True, include_input=True)
        _caption = "Choose any day, then run fresh inference below. Until then, the labelled earlier snapshot is shown."
    mo.vstack([observed_days, mo.md(_caption)])
    return (observed_days,)


@app.cell
def _(cache_arrays, cache_task, case_choice, observed_days, trajectory):
    requested_days = observed_days.value
    selection = trajectory.selection_key(cache_task, cache_arrays, case_choice.value, requested_days)
    trajectory.plot_returns(cache_task, cache_arrays, case_choice.value, requested_days)
    return requested_days, selection


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A daily log return is $r_t=\log S_t-\log S_{t-1}$; 0.01 is about a 1% increase.
    The lower panel adds those log returns through time. **We observe returns,
    not the latent variance or shocks.** One observation is made per trading day,
    with $\Delta t=1/252$ year, for up to five years.

    **Try it:** keep the history fixed and reveal more days. Which parameters
    should be hardest to learn from a short history?
    """)
    return


@app.cell(hide_code=True)
def _(cache_task, lesson_workflow, mo):
    parameter_labels = {
        "kappa": "Mean reversion κ", "long_variance": "Long variance L",
        "vol_of_vol": "Volatility of variance σ", "rho": "Leverage correlation ρ",
        "hurst": "Roughness H", "eta": "Volatility amplitude η",
        "short_variance": "Short variance S", "decay": "Curve decay k", "drift": "Physical drift μ",
    }
    _meanings = {
        "kappa": "Speed of reversion toward the long variance level",
        "long_variance": "Long variance level",
        "vol_of_vol": "Strength of variance shocks", "rho": "Correlation of price and variance-driver shocks",
        "hurst": "Roughness of the Gaussian variance driver", "eta": "Amplitude of log-variance shocks",
        "short_variance": "Initial variance-curve level", "decay": "Speed of transition between curve levels",
        "drift": "Annual proportional price drift",
    }
    _priors = [{"parameter": parameter_labels[_name], "meaning": _meanings[_name],
                "prior": ("Log-uniform" if _kind == "log_uniform" else "Uniform")
                         + f" [{_bounds[0]:g}, {_bounds[1]:g}]"}
               for _name, _kind, _bounds in zip(cache_task["parameter_names"], cache_task["prior_types"],
                                               cache_task["prior_bounds"], strict=True)]
    _heston = cache_task["model"] == "heston"
    _story = (
        r"**Heston:** variance reverts toward a long-run level: "
        r"$dV_t=\kappa(L-V_t)dt+\sigma\sqrt{V_t}\,dW_t^V$. "
        "Initial variance is supplied; κ and physical drift are inferred here."
        if _heston else
        r"**Rough Bergomi:** variance depends on a weighted history of random shocks: "
        r"$V_t=\xi(t)\exp(\eta Y_t-\tfrac12\eta^2\operatorname{Var}_{\rm grid}(Y_t))$. "
        "H controls roughness, η the shock amplitude, and ξ the deterministic mean variance scale.")
    _curve = ("The supplied initial variance is not the realized subsequent variance."
              if _heston else
              "The variance-scale curve is supplied and constant at 0.04. The realized variance is still hidden."
              if cache_task["model"] == "rough-known" else
              r"Here the curve is inferred: $\xi(t)=L+(S-L)e^{-kt}$. Its coefficients are unknown parameters, not observed inputs.")
    _steps = cache_task["numerics"].get("substeps", cache_task["numerics"].get("observation_substeps"))
    mo.vstack([
        lesson_workflow("Model"), mo.md(r"""
        ## 1. Model: dynamics and unknowns

        This is a **physical-measure model** for returns, with an inferred drift μ:

        $$d\log S_t=(\mu-V_t/2)dt+\sqrt{V_t}\,dW_t^S,
        \qquad d\langle W^S,W^V\rangle_t=\rho\,dt.$$

        Volatility is $\sqrt{V_t}$. The physical measure **P** describes the
        statistical law of returns. The option lessons used a pricing measure
        **Q**, under which discounted prices are martingales. A drift or variance
        parameter fitted under one measure need not have the same value under
        the other. This lesson fits the declared P law, with **no extra
        measurement-error or quote-noise term**.
        """),
        mo.md(_story + "\n\n" + _curve),
        mo.ui.table(_priors, selection=None, pagination=False),
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
            mo.md(r"For rough Bergomi, the idealized Gaussian driver is "
                  r"$Y_t=\sqrt{2H}\int_0^t(t-s)^{H-1/2}\,dW_s^V$. The hybrid-grid "
                  r"implementation uses its own Gaussian variance in the exponential compensator, "
                  r"written $\operatorname{Var}_{\rm grid}(Y_t)$ above. Each history starts at the declared origin."),
            mo.ui.table([{"setting": _name.replace("_", " "), "value": _value}
                         for _name, _value in cache_task["numerics"].items()], selection=None, pagination=False),
        ])}),
    ])
    return (parameter_labels,)


@app.cell(hide_code=True)
def _(cache_task, mo):
    _memory = ("κ controls mean reversion; σ controls random variance fluctuations. "
               "Inspect persistence across several paths as you change them. The notebook fits the stated finite-step law."
               if cache_task["model"] == "heston" else
               "H controls the roughness of the latent Gaussian driver; η controls its amplitude. "
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
    mo.vstack([mo.md("### Four features to carry into the simulator"),
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
    the hybrid driver has the form
    $Y_i=\sum_{j=0}^{i-1}w_{ij}(H,h)Z_j^W+b(H,h)Z_{i-1}^B$ for $i\ge1$.
    Its known grid variance $q_i=\operatorname{Var}_{\rm grid}(Y_i)$ is used in

    $$V_i=\xi(t_i)\exp(\eta Y_i-\tfrac12\eta^2q_i),$$

    $$\Delta\ell_i=(\mu-V_i/2)h+\sqrt{V_i h}
       (\rho Z_i^W+\sqrt{1-\rho^2}Z_i^\perp).$$

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
    **A mean curve is not a realized variance path.** In this declared synthetic
    P model, $\xi(t_i)=\mathbb E_P[V_i]$ at simulator grid nodes. The Gaussian
    compensator ensures this equality for the hybrid approximation. It is a
    physical-law variance-scale curve, not a Q curve estimated from option prices.

    - **Supplied curve:** $\xi(t)=0.04$. Mean variance is constant at $0.2^2$;
      each realized $V_i$ still fluctuates. The constant does not say that the
      average realized volatility is exactly 20%.
    - **Inferred curve:** $\xi(t)=L+(S-L)e^{-kt}$. $S$ is the initial level,
      $L$ the long level, and $k$ the transition rate. These coefficients join
      $H,\eta,\rho,\mu$ as targets; they are not extra inputs supplied to the GRU.
      If $S$ and $L$ are similar, changing $k$ barely changes the curve, so
      decay can remain weakly informed.
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
def _(lesson_workflow, mo):
    mo.vstack([lesson_workflow("Simulate"), mo.md(r"""
    ## 2. Simulate: try the forward model

    **Choose the parameters; see the histories they can generate.** The sliders
    use the model's declared parameter ranges. Eight independent one-year paths
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


@app.cell
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
    mo.vstack(_panels)
    return (simulator_controls,)


@app.cell
def _(mo, trajectory):
    simulator_scale = mo.ui.dropdown(options=list(trajectory.SIMULATOR_SCALES), value="Standard",
                                     label="Fixed simulator axes")
    mo.vstack([simulator_scale, mo.md(
        "The axis limits stay fixed when model parameters change. Choose a wider preset yourself. "
        "All simulated points remain in the data; counts on each panel report points outside the view.")])
    return (simulator_scale,)


@app.cell
def _(cache_task, mo, simulator_controls, simulator_scale, trajectory):
    try:
        simulation_preview = trajectory.simulator_preview(
            cache_task, simulator_controls.value, paths=8, days=252, seed=20261108)
        _limits = trajectory.SIMULATOR_SCALES[simulator_scale.value]
        _panel = mo.vstack([
            mo.md(f"**Fixed axes ({simulator_scale.value}):** asset price "
                  f"**{_limits['prices'][0]:g}–{_limits['prices'][1]:g}**; daily return "
                  f"**{_limits['returns'][0]:g}–{_limits['returns'][1]:g}%**; realized volatility "
                  f"**{_limits['realized'][0]:g}–{_limits['realized'][1]:g}%**."),
            trajectory.forward_simulator_figure(simulation_preview, scale=simulator_scale.value),
            mo.accordion({"Exact axis limits, outside counts and data ranges": mo.ui.table(
                trajectory.simulator_axis_summary(simulation_preview, simulator_scale.value),
                selection=None, pagination=False)}),
        ])
    except (ValueError, FloatingPointError) as _error:
        simulation_preview = None
        _panel = mo.callout(mo.md("This forward example could not be simulated: " + str(_error)), kind="warn")
    _panel
    return (simulation_preview,)


@app.cell(hide_code=True)
def _(cache_task, mo):
    _amplitude = "σ (volatility of variance)" if cache_task["model"] == "heston" else "η (variance amplitude)"
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
        cache_task, parameters, paths=8, days=252, seed=20261108)
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
def _(mo):
    mo.md(r"""
    ### Prior predictive: could this model generate a history like this?

    The sliders above hold parameters fixed. A **prior-predictive check** instead
    draws parameters from their prior, then simulates one history for each draw.
    Compare the selected history with 200 such histories at the **same observed
    length**, before using its posterior. These supplied observations are synthetic;
    this is a workflow sanity check, not evidence of real-market fit.
    """)
    return


@app.cell(hide_code=True)
def _(mo, selection):
    _selected_history = selection
    prior_check_button = mo.ui.run_button(label="Compare this history with 200 prior simulations")
    prior_check_button
    return (prior_check_button,)


@app.cell(hide_code=True)
def _(cache_arrays, cache_task, case_choice, diagnostic_panels, mo, model_checks,
      path_checks, path_simulation, prior_check_button, requested_days):
    mo.stop(not prior_check_button.value, mo.md("*Optional: simulate from the prior and compare return summaries.*"))
    _key = path_simulation.task_fingerprint(cache_task)
    _reused = _key in diagnostic_panels
    try:
        if not _reused:
            diagnostic_panels[_key] = path_checks.generate_panel(cache_task, count=200, seed=20261111)
        _result = model_checks.compare_prior(cache_task, diagnostic_panels[_key],
                    cache_arrays["observations"][case_choice.value], requested_days)
        _output = mo.vstack([
            model_checks.plot_check(_result),
            mo.md(f"**Prior simulations · {requested_days} days each.** "
                  f"{'Reused the existing panel' if _reused else 'Generated 200 independent histories'}. "
                  "The black line is the selected observed prefix. Panels compare cumulative log return, "
                  "annualized realized RMS volatility and the largest absolute daily log return. "
                  "Realized volatility is a return statistic, not observed latent variance. "
                  "These summaries ignore much of the time ordering; a central position is neither a joint-support "
                  "certificate nor a posterior-calibration test."),
        ])
    except (ValueError, RuntimeError, FloatingPointError) as _error:
        _output = mo.callout(mo.md(f"**Prior check stopped:** {_error}. No histories are replaced or filtered."), kind="warn")
    _output
    return


@app.cell(hide_code=True)
def _(cache_task, lesson_workflow, mo, trajectory):
    _figure = trajectory.ASSETS / "illustrations" / "application-workflows-20261101" / "trajectory-prefix-workflow.png"
    mo.vstack([
        lesson_workflow("Train"), mo.md("## 3. Train a tutorial model here\n\n"
              "**One training pair: parameters drawn from the prior and the daily returns they generate.** "
              "Only returns enter the estimator. Generating parameters are training labels; latent "
              "variance and shocks stay inside the simulator."),
        mo.image(_figure) if _figure.is_file() else mo.md(""),
        mo.md(f"""
        ### Generate parameter–history pairs

        ```python
        from sbi4f_tutorial.trajectories import simulation, inference

        # A physical-measure return model; process randomness needs no extra quote noise.
        task = simulation.teaching_task("rough-known", observations=252)
        pairs = simulation.generate(
            task, "train", start=0, count=128, seed=20261107, backend="numpy")
        # A fresh, independent history is used only after training.
        example = simulation.generate(
            task, "validation", start=0, count=1, seed=20261108, backend="numpy")
        # 128 training histories, each with 252 daily returns
        # One independently generated history for the inference example
        ```

        The simulator samples both parameters and process innovations. It retains
        aligned parameter labels, daily returns and history IDs. This tutorial route
        uses **rough Bergomi with a supplied curve**, one-year histories and drift
        inference, regardless of the model selected for the supplied examples.
        The implementation lives in `src/sbi4f_tutorial/trajectories/`.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo, trajectory):
    from sbi4f_tutorial.trajectories import simulation as _input_simulation
    _task = _input_simulation.teaching_task("rough-known", observations=252)
    mo.vstack([
        mo.md("### From a return sequence to network inputs\n\n"
              "The tutorial model uses a 252-day maximum. Each observed day becomes a scaled return "
              "and a known time coordinate. Padding gets a false mask; a genuine zero return remains "
              "observed. Generating parameter labels follow a separate training route."),
        trajectory.input_concept_figure(_task),
        mo.md("The query supplies the upper route only. `training_data` also transforms the "
              "parameter labels for density scoring during training. The supplied five-year model "
              "uses the same input format with a 1,260-day maximum."),
    ])
    return


@app.cell(hide_code=True)
def _(trajectory):
    from sbi4f_tutorial.trajectories import simulation as _architecture_simulation
    _task = _architecture_simulation.teaching_task("rough-known", observations=252)
    trajectory.architecture_figure(_task, network=trajectory.TUTORIAL_NETWORK)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build the posterior estimator

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
    | Training data | 128 one-year histories | One million five-year histories per model |

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
def _(mo):
    mo.vstack([
        mo.md(r"""
        ### Train on different amounts of evidence

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
def _(mo):
    mo.md(r"""
    ### Run the tutorial training route

    The button executes the simulation, network, training and sampling steps
    above. It creates a **new one-year rough Bergomi model**. Its loss and
    posterior summary appear here, separately from the supplied model below.

    The button runs these steps on the CPU, keeps results only in this session,
    and reports generation, model-building/training and sampling time separately.
    First use includes backend imports and compilation.
    """)
    return


@app.cell
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
        task = path_simulation.teaching_task("rough-known", observations=252)
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


@app.cell
def _(mo, path_simulation, train_demo_button, train_tutorial_model, trajectory):
    tutorial_task = path_simulation.teaching_task("rough-known", observations=252)
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
                      "Validation uses their full one-year length while training sees random prefixes, "
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
def _(lesson_workflow, mo):
    mo.vstack([lesson_workflow("Infer"), mo.md(r"""
    ## 4. Infer: use the supplied model

    **Reuse a larger fitted estimator; change the evidence.** This route uses
    the supplied five-year model selected at the top, not the tutorial model
    trained above. Saved draws let you inspect it immediately. To draw a fresh
    posterior at any day, download the matching model and run:

    ```python
    # Restore the matching pretrained weights; no fitting is performed.
    task, workflow, record = trajectory.load_live(model_choice.value)
    # Only the observed prefix enters; future returns and generating labels stay out.
    revealed = cache_arrays["observations"][case_choice.value:case_choice.value + 1,
                                           :observed_days.value]
    # Same workflow.sample pattern, with preprocessing and inverse parameter transforms.
    draws = trajectory.sample(workflow, task, revealed, draws=512, seed=20261031)[0]
    ```

    Each query uses the **original prior and the entire revealed history**.
    We do not multiply posteriors from overlapping prefixes or feed the previous
    posterior into this unchanged network.
    """)])
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


@app.cell
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


@app.cell
def _(checkpoint_path, inference_mode, mo, requested_days, selection):
    _request = (selection, checkpoint_path.value, inference_mode.value)
    live_button = mo.ui.run_button(label=f"Run CPU inference for {requested_days} observed days")
    if inference_mode.value == "live":
        _panel = mo.vstack([live_button, mo.md(
            "Fresh draws will update every posterior view below. Changing the model, history, "
            "day or model folder clears the previous result and requires a new click.")])
    else:
        _panel = mo.md("")
    _panel
    return (live_button,)


@app.cell
def _(cache_arrays, cache_task, case_choice, checkpoint_path, inference_mode, live_button, model_choice, requested_days, selection, trajectory):
    live_result, live_error, live_timing = None, None, None
    pretrained_workflow = None
    if inference_mode.value == "live" and live_button.value:
        from time import perf_counter as _live_clock
        try:
            _start = _live_clock()
            _task, _workflow, _record = trajectory.load_live(
                model_choice.value, checkpoint_path.value.strip() or None)
            _load_seconds = _live_clock() - _start
            # This slice excludes future observations, parameter labels and hidden states.
            _revealed = cache_arrays["observations"][case_choice.value:case_choice.value + 1, :requested_days]
            _start = _live_clock()
            _draws = trajectory.sample(_workflow, _task, _revealed, draws=512, seed=20261031)[0]
            _sample_seconds = _live_clock() - _start
            _candidate = {"selection": selection, "draws": _draws}
            trajectory.posterior_view(cache_task, cache_arrays, case_choice.value, requested_days, live=_candidate)
            live_result = _candidate
            pretrained_workflow = _workflow
            live_timing = (_load_seconds, _sample_seconds)
        except (FileNotFoundError, ValueError, RuntimeError, ImportError, OSError) as _error:
            live_error = str(_error)
    return live_error, live_result, live_timing, pretrained_workflow


@app.cell
def _(cache_arrays, cache_task, case_choice, live_result, requested_days, trajectory):
    posterior = trajectory.posterior_view(
        cache_task, cache_arrays, case_choice.value, requested_days, live=live_result)
    return (posterior,)


@app.cell(hide_code=True)
def _(inference_mode, live_error, live_timing, mo, posterior, requested_days):
    _contents = [mo.callout(mo.md(f"**Source: supplied pretrained model. {posterior['label']}.**"), kind="neutral")]
    if live_error:
        _contents.append(mo.callout(mo.md(
            "Fresh inference is unavailable: " + live_error + " The labelled saved result remains visible."), kind="warn"))
    if posterior["source"] == "live":
        _contents.append(mo.md(f"Model verification/load: **{live_timing[0]:.3f} s**; "
                              f"512 draws: **{live_timing[1]:.3f} s**. First sampling may include compilation. "
                              "Fresh describes execution, not posterior accuracy."))
    elif inference_mode.value == "live":
        _contents.append(mo.md(f"The return plot reveals **{requested_days} days**. "
                              f"The posterior below uses the saved **{posterior['length']}-day** prefix. "
                              "Press the button to condition on every revealed return. Saved snapshots are never interpolated."))
    _contents.append(mo.md("Central 90% intervals and all density/dependence panels use the **same paired draw rows**. "
                          "The four example histories were selected before inspecting their posteriors."))
    mo.vstack(_contents)
    return


@app.cell
def _(cache_arrays, cache_task, case_choice, np, parameter_labels, plt, posterior):
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
        _axis.axvline(cache_arrays["truth"][case_choice.value, _index], linestyle=":", color="#111827",
                     linewidth=1, label="synthetic truth")
        _axis.set(xlim=(_low, _high), title=parameter_labels[_names[_index]], ylabel="Marginal density")
    _axes.flat[0].legend(fontsize=7)
    _figure.suptitle(f"Prior and approximate posterior · history {case_choice.value + 1} · {posterior['length']} days", fontsize=12)
    _figure.tight_layout()
    _figure
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
def _(cache_arrays, cache_task, case_choice, mo, parameter_labels, posterior, trajectory):
    _reading = trajectory.posterior_readout(cache_task, posterior["draws"],
                                          truth=cache_arrays["truth"][case_choice.value])
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
    mo.vstack([mo.md("### Read this posterior in three steps\n\n" + _precision),
               mo.md(_dependence), mo.md(_truth),
               mo.md("These summaries describe the displayed **approximation** for this history and prefix. "
                     "Width reduction and truth inclusion in one case do not establish accuracy. "
                     "The final calibration failures and model-specific numerical limits still apply."),
               mo.accordion({"Definitions behind this reading": mo.md(
                   "Relative width is (posterior q95 − q05) / (prior q95 − q05), in physical units. "
                   "It is not an information-distance or accuracy score. Pearson correlation measures linear dependence; "
                   "a small value does not rule out nonlinear dependence. Truth markers are evaluation labels and never network inputs.")})])
    return


@app.cell
def _(cache_task, mo, parameter_labels):
    _options = {parameter_labels[_name]: _name for _name in cache_task["parameter_names"]}
    _x = "vol_of_vol" if cache_task["model"] == "heston" else "eta"
    pair_x = mo.ui.dropdown(options=_options, value=parameter_labels[_x], label="Horizontal parameter")
    pair_y = mo.ui.dropdown(options=_options, value=parameter_labels["rho"], label="Vertical parameter")
    mo.vstack([mo.md("### Inspect parameter dependence"), mo.hstack([pair_x, pair_y], justify="start")])
    return pair_x, pair_y


@app.cell
def _(cache_arrays, cache_task, case_choice, mo, pair_x, pair_y, parameter_labels, plt, posterior):
    mo.stop(pair_x.value == pair_y.value, mo.md("Choose two different parameters for the joint view."))
    _x = cache_task["parameter_names"].index(pair_x.value)
    _y = cache_task["parameter_names"].index(pair_y.value)
    _figure, _axis = plt.subplots(figsize=(7, 4.2), layout="constrained")
    _axis.scatter(posterior["draws"][:, _x], posterior["draws"][:, _y], s=10, alpha=.28, color="#2563eb",
                  label="paired neural draws")
    _axis.scatter(cache_arrays["truth"][case_choice.value, _x], cache_arrays["truth"][case_choice.value, _y],
                  marker="x", color="#111827", s=60, label="synthetic truth")
    _axis.set(xlabel=parameter_labels[pair_x.value], ylabel=parameter_labels[pair_y.value],
              xlim=cache_task["prior_bounds"][_x], ylim=cache_task["prior_bounds"][_y],
              title=f"Joint parameter uncertainty · {posterior['length']} observed days")
    _axis.legend(fontsize=8)
    mo.vstack([_figure, mo.md("Each point preserves two coordinates from the **same posterior draw**. "
                              "Shuffling columns would destroy dependence. These trade-offs describe "
                              "the learned approximation, not an independently validated joint posterior."),
               mo.accordion({"Numerical posterior summary": mo.ui.table(
                   trajectory.posterior_rows(cache_task, posterior["draws"]), selection=None, pagination=False)})])
    return


@app.cell
def _(cache_arrays, cache_task, case_choice, mo, posterior, requested_days, trajectory):
    mo.vstack([mo.md("### Watch the evidence grow\n\n"
                     "Drift can remain uncertain after volatility parameters sharpen. Along one history, "
                     "intervals can move or widen after surprising returns. The parameters themselves stay fixed. "
                     "Grey shading marks prior support; blue steps and bars show the saved median and central "
                     "90% interval. A step is held between snapshots, not interpolated. A green diamond, "
                     "when present, marks fresh inference at the requested day."),
               trajectory.plot_evolution(cache_task, cache_arrays, case_choice.value, requested_days, posterior=posterior)])
    return


@app.cell(hide_code=True)
def _(cache_arrays, cache_task, case_choice, mo, plt, requested_days, trajectory):
    _revealed = cache_arrays["observations"][case_choice.value:case_choice.value + 1, :requested_days]
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
revealed = cache_arrays["observations"][case_choice.value:case_choice.value + 1,
                                       :observed_days.value]
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
def _(lesson_workflow, mo):
    mo.vstack([lesson_workflow("Check"), mo.md(r"""
    ## 5. Check: ask three different questions

    | Check | Question | Evidence here |
    |:---|:---|:---|
    | Replicated histories | Can fitted parameters reproduce these return summaries? | Independent, zero-origin posterior-predictive histories |
    | Many independent histories | Does this estimator recover parameters and report uncertainty? | The selected model's recovery, ranks and coverage |
    | One conditional posterior | Does the approximation match an independent numerical reference? | No matched reference for these histories and lengths |

    The saved final-cohort results below answer a broader validation question.
    A successful small fit or a good-looking predictive plot does not replace them.

    ### Posterior predictive: replicate, then compare

    Select complete paired parameter draws from the displayed posterior and
    generate a new history from the model's origin for each. Compare the **same
    length used by that posterior** with the observed prefix. These are independent
    replicas, not paths continuing from the end of the observed history.
    """)])
    return


@app.cell(hide_code=True)
def _(mo, posterior, selection):
    _source = (selection, posterior["source"], posterior["length"])
    replicate_button = mo.ui.run_button(label=f"Replicate histories from the {posterior['length']}-day posterior")
    mo.vstack([replicate_button, mo.md(
        f"**Posterior source: {posterior['source']} · {posterior['length']} days.** "
        "The button uses at most 32 complete paired draws and keeps all replicas." )])
    return (replicate_button,)


@app.cell(hide_code=True)
def _(cache_arrays, cache_task, case_choice, mo, model_checks, posterior, replicate_button, requested_days):
    mo.stop(not replicate_button.value, mo.md("*Optional: generate independent posterior-predictive histories.*"))
    try:
        _result = model_checks.replicate_check(cache_task,
            cache_arrays["observations"][case_choice.value], posterior["draws"], posterior["length"], count=32)
        _output = mo.vstack([
            model_checks.plot_check(_result),
            mo.md(f"**{_result['metadata']['count']} independent replicas · {posterior['length']} days each.** "
                  f"The return plot currently reveals {requested_days} days; this check follows the "
                  "posterior's stated conditioning length. Each blue value comes from a new path at one "
                  "paired parameter draw; the black line is the observed summary. This spread includes "
                  "parameter uncertainty and process randomness. A discrepancy motivates further model "
                  "or inference checks; agreement on three summaries does not establish full model adequacy. "
                  "No latent state is filtered and no continuation forecast is generated."),
        ])
    except (ValueError, RuntimeError, FloatingPointError) as _error:
        _output = mo.callout(mo.md(f"**Replication stopped:** {_error}. No replicas are discarded or replaced."), kind="warn")
    _output
    return


@app.cell(hide_code=True)
def _(mo, path_checks):
    _worked = mo.md(r"""
    **One history gives one rank.** Draw its parameters from the prior, simulate
    the observed prefix, then draw 200 parameter samples from its approximate
    posterior. If 140 draws are below the generating value, its rank is
    **140 / 200 = 0.70**. Generating truths remain evaluation labels.

    **Many histories give a curve.** Repeat over 200 independent histories. At
    a horizontal position z, count the share of ranks at or below z. This is
    the empirical CDF, $F_N(z)$. The plotted height is **$F_N(z)-z$**.
    Near zero means consistency with uniform ranks at this resolution.
    """).style({"min-width": "260px"})
    _shapes = mo.md(r"""
    **Read the direction, not just the grey band.**

    - **Above zero:** more ranks are below z than the uniform reference predicts.
    - **Below zero:** fewer ranks are below z.
    - **Above on the left, below on the right:** ranks tend toward the extremes;
      posterior intervals may be too narrow.
    - **The reverse:** ranks tend toward the centre; intervals may be too wide.
    - **An upper arch:** truths tend to fall below posterior draws; the fitted
      centres may be too high. A lower arch suggests the reverse.

    These are clues, rather than unique diagnoses.
    """).style({"min-width": "260px"})
    mo.accordion({"Read rank curves step by step": mo.vstack([
        mo.hstack([_worked, _shapes], widths="equal", wrap=True, align="start", gap=1),
        path_checks.rank_shape_guide(),
        mo.md("**Illustrations, not trajectory results.** These Normal examples share the same simulated "
              "data and random noise; only the approximate posterior's centre or spread changes. "
              "The correct posterior is known analytically."),
        mo.md(f"""
        **Three different intervals appear in this lesson:**

        | Display | What is uncertain? |
        |:---|:---|
        | Central 90% posterior interval | A parameter for one observed history, under this approximation |
        | Grey {100*path_checks.RANK_BAND_CONFIDENCE:.0f}% ECDF reference band | Sampling variation of one rank curve over independent histories |
        | 95% binomial interval in the coverage table | A coverage rate estimated using 200 histories |

        BayesFlow's grey band is simultaneous across its plotted rank grid **for
        one parameter curve**. It is not a 95% joint statement across all 4, 5
        or 7 parameter panels. Its continuous-uniform reference is approximate:
        200 draws give 201 possible rank positions, and ties need additional care,
        especially for the Heston drift atom. The band itself is simulation-based.

        Staying inside it is not a certificate of correct posteriors. A network
        that always returns the prior can have uniform ranks without learning.
        Read recovery, dependence, conditional checks and the saved final screens too.
        """),
    ])})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Optional: check the network on independent histories

    **A loss curve shows training progress. Does the resulting posterior recover
    generating parameters and express uncertainty?** Check the actual tutorial
    model you trained or the pretrained model you loaded above.

    This action generates **200 independent validation histories**, then draws
    **200 posterior samples for each**. Unlike an option surface, one history is
    one process realization: there is no Monte Carlo option-price average to
    compute. The full histories stay in session memory and are reused when you
    change the diagnostic prefix; generating labels never condition the network.
    Nothing here runs automatically or changes the supplied results.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    diagnostic_source = mo.ui.radio(
        options={"Tutorial network trained above": "tutorial", "Pretrained network loaded above": "pretrained"},
        value="Tutorial network trained above", label="Network to check", inline=True)
    diagnostic_source
    return (diagnostic_source,)


@app.cell(hide_code=True)
def _(cache_task, diagnostic_source, mo, trajectory, tutorial_task):
    diagnostic_task = tutorial_task if diagnostic_source.value == "tutorial" else cache_task
    diagnostic_days = mo.ui.slider(1, diagnostic_task["n_observations"], step=1, value=252,
                                   label="Days per diagnostic history", show_value=True, include_input=True)
    mo.vstack([
        mo.md(f"**Diagnostic task: {trajectory.MODEL_LABELS[diagnostic_task['model']]} · "
              f"maximum {diagnostic_task['n_observations']} days.** "
              "This separate control gives every diagnostic history the same observed length. "
              "It does not change the single example selected above. Later returns are masked."),
        diagnostic_days,
    ])
    return diagnostic_days, diagnostic_task


@app.cell(hide_code=True)
def _(diagnostic_days, diagnostic_source, diagnostic_task, mo, pretrained_workflow, tutorial_workflow):
    # Any changed source, prefix, task or model requires a fresh explicit click.
    _request = (diagnostic_source.value, diagnostic_days.value, diagnostic_task,
                id(tutorial_workflow), id(pretrained_workflow))
    run_diagnostics = mo.ui.run_button(label="Check this network on 200 independent histories")
    run_diagnostics
    return (run_diagnostics,)


@app.cell(hide_code=True)
def _(mo, path_checks):
    def check_independent_histories(workflow, task, panel, days):
        # The helper passes observed returns, times and masks to inference.sample.
        # Future returns and generating parameters never enter those conditions.
        draws = path_checks.evaluate(workflow, task, panel, days, draws=200, seed=20261111)
        # Native BayesFlow diagnostics compare physical-unit draws with separate truths.
        recovery, ranks = path_checks.plot_diagnostics(task, panel, draws)
        return draws, recovery, ranks

    mo.show_code()
    return (check_independent_histories,)


@app.cell(hide_code=True)
def _(check_independent_histories, diagnostic_days, diagnostic_panels, diagnostic_source,
      diagnostic_task, mo, path_checks, path_simulation, pretrained_workflow,
      run_diagnostics, tutorial_workflow):
    mo.stop(not run_diagnostics.value, mo.md("*Choose a network and run the check when ready. "
        "Changing the diagnostic prefix reuses the same independent histories.*"))
    _workflow = tutorial_workflow if diagnostic_source.value == "tutorial" else pretrained_workflow
    mo.stop(_workflow is None, mo.callout(mo.md("**No matching network is ready.** "
        "Train the tutorial network above, or select Fresh inference and load the matching "
        "pretrained model with the CPU-inference button above. Then run this check."), kind="info"))
    from time import perf_counter as _diagnostic_clock
    _key = path_simulation.task_fingerprint(diagnostic_task)
    _reused = _key in diagnostic_panels
    _started = _diagnostic_clock()
    try:
        if not _reused:
            # Fixed seed, independent validation IDs. Keep all rows; never redraw failures.
            diagnostic_panels[_key] = path_checks.generate_panel(diagnostic_task, count=200, seed=20261111)
        _panel = diagnostic_panels[_key]
        _generation_seconds = _diagnostic_clock() - _started
        _started = _diagnostic_clock()
        _diagnostic_draws, _recovery, _ranks = check_independent_histories(
            _workflow, diagnostic_task, _panel, diagnostic_days.value)
        _check_seconds = _diagnostic_clock() - _started
    except (ImportError, RuntimeError, ValueError, FloatingPointError) as _error:
        mo.stop(True, mo.callout(mo.md(f"**Diagnostic check stopped:** {_error}. "
            "No partial or selectively accepted cohort is displayed."), kind="warn"))
    diagnostic_result = {"task": diagnostic_task, "panel": _panel, "draws": _diagnostic_draws,
                         "length": diagnostic_days.value, "source": diagnostic_source.value}
    _coverage = path_checks.coverage_table(diagnostic_task, _panel, _diagnostic_draws)
    _rank_rows = path_checks.rank_readout(diagnostic_task, _panel, _diagnostic_draws, z=.5)
    mo.vstack([
        mo.md(f"**Source: {diagnostic_source.value} network · 200 independent histories · "
              f"{diagnostic_days.value} observed days each · 200 posterior draws per history.** "
              f"Input generation: **{'reused from this session' if _reused else f'{_generation_seconds:.3f} s'}**; "
              f"sampling and plotting: **{_check_seconds:.1f} s**, including compilation if needed."),
        _recovery, _ranks,
        mo.ui.table([{"parameter": row["label"], "inside central 90% interval": f'{row["hits"]}/{row["n"]}',
                      "empirical coverage": f'{100*row["coverage"]:.1f}%',
                      "95% binomial interval": f'{100*row["wilson95_low"]:.1f}–{100*row["wilson95_high"]:.1f}%'}
                     for row in _coverage], selection=None, pagination=False),
        mo.accordion({"Read one point from the actual rank curves": mo.vstack([
            mo.md("At **z = 0.5**, divide the number of ranks at or below 0.5 by the number of histories, "
                  "then subtract 0.5. This is the plotted ECDF difference at that position, using all retained cases."),
            mo.ui.table([{"parameter": row["label"], "ranks ≤ 0.5": f'{row["cases_below_z"]}/{row["n"]}',
                          "empirical CDF": f'{row["empirical_cdf"]:.3f}',
                          "CDF − 0.5": f'{row["departure"]:+.3f}'} for row in _rank_rows],
                        selection=None, pagination=False),
        ])}),
        mo.md("Recovery compares generating truths with posterior medians and central 90% intervals. "
              "The rank ECDF checks calibration averaged over the declared prior: near-zero differences "
              "mean roughly uniform ranks. Inspect both; returning the prior can pass a rank check without learning. "
              "These fixed-seed validation examples are reused across prefixes. Nested prefixes are not "
              "independent replicates, and these checks do not replace the final-test results below. "
              "A tiny fit may perform poorly; successful execution does not establish accuracy."),
    ])
    return (diagnostic_result,)


@app.cell(hide_code=True)
def _(diagnostic_result, mo, parameter_labels):
    _names = diagnostic_result["task"]["parameter_names"]
    recovery_parameter = mo.ui.dropdown(options={parameter_labels[name]: name for name in _names},
                                       value=parameter_labels[_names[0]], label="Parameter for recovery selection")
    mo.vstack([mo.md("### Inspect a history behind the recovery plot\n\n"
        "Drag a box around points, then choose a history from that selection. With no box, "
        "all 200 are available in generation order. The history and posterior below use the "
        "actual retained rows from this check; selecting a point runs no new inference."), recovery_parameter])
    return (recovery_parameter,)


@app.cell(hide_code=True)
def _(diagnostic_result, mo, parameter_labels, path_checks, plt, recovery_parameter):
    recovery_points = path_checks.recovery_case_data(diagnostic_result["task"], diagnostic_result["panel"],
                                                    diagnostic_result["draws"], recovery_parameter.value)
    _lo, _hi = diagnostic_result["task"]["prior_bounds"][diagnostic_result["task"]["parameter_names"].index(recovery_parameter.value)]
    _figure, _axis = plt.subplots(figsize=(6.5, 4), layout="constrained")
    _axis.plot([_lo, _hi], [_lo, _hi], "--", color="#64748b", linewidth=1)
    _axis.vlines(recovery_points["truth"], recovery_points["lower90"], recovery_points["upper90"], color="#2563eb", alpha=.22)
    _axis.scatter(recovery_points["truth"], recovery_points["median"], color="#2563eb", s=12, alpha=.7)
    _axis.set(xlabel="Generating value", ylabel="Posterior median", xlim=(_lo, _hi), ylim=(_lo, _hi),
              title=f"Select independent histories · {parameter_labels[recovery_parameter.value]}")
    recovery_picker = mo.ui.matplotlib(_axis, debounce=True)
    recovery_picker
    return recovery_picker, recovery_points


@app.cell(hide_code=True)
def _(mo, np, recovery_picker, recovery_points):
    _selection = (np.flatnonzero(recovery_picker.value.get_mask(recovery_points["truth"], recovery_points["median"]))
                  if recovery_picker.value else np.array([], dtype=int))
    _pool = _selection if len(_selection) else np.arange(len(recovery_points["truth"]))
    _options = {f"Diagnostic history {index+1}": int(index) for index in _pool}
    diagnostic_case = mo.ui.dropdown(options=_options, value=next(iter(_options)), label="Diagnostic history to inspect")
    show_diagnostic_pairs = mo.ui.checkbox(value=False, label="Show all parameter pairs for this diagnostic history")
    mo.vstack([mo.md(f"**{len(_pool)} histories available.** Default: the first in generation order, not a selected best or worst case."),
               diagnostic_case, show_diagnostic_pairs])
    return diagnostic_case, show_diagnostic_pairs


@app.cell(hide_code=True)
def _(diagnostic_case, diagnostic_result, mo, np, path_checks, plt, show_diagnostic_pairs, trajectory):
    _case = path_checks.case_view(diagnostic_result["task"], diagnostic_result["panel"],
        diagnostic_result["draws"], diagnostic_case.value, diagnostic_result["length"])
    _figure, _axis = plt.subplots(figsize=(8.5, 2.2), layout="constrained")
    _axis.plot(np.arange(1, _case["length"]+1), 100*_case["returns"], color="#2563eb", linewidth=.8)
    _axis.set(xlabel="Observed trading day", ylabel="Daily log return (%)",
              title=f"Diagnostic history {_case['index']+1} · {_case['length']} days")
    _rows = trajectory.posterior_rows(diagnostic_result["task"], _case["draws"])
    for _row, _truth in zip(_rows, _case["truth"], strict=True):
        _row["generating truth"] = float(_truth)
    _parts = [mo.md(f"**Source: {diagnostic_result['source']} network · diagnostic history {_case['index']+1}.** "
                   "These are separate from the four supplied example histories."), _figure,
              mo.ui.table(_rows, selection=None, pagination=False)]
    if show_diagnostic_pairs.value:
        _parts += [path_checks.native_pairs(diagnostic_result["task"], _case["draws"], _case["truth"]),
                   mo.md("Native `bf.diagnostics.pairs_posterior`: the same 200 paired draws, with generating truths marked. "
                         "Changing the selected history does not sample or fit another model.")]
    mo.vstack(_parts)
    return


@app.cell
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
                        or trajectory.sha256(_path) != _accepted["source_sha256"][_name]):
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
def _(aggregate_images, aggregate_problem, aggregate_report, cache_task, mo, parameter_labels, trajectory):
    _contents = [mo.md("### Saved final-cohort evidence\n\n"
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
def _(cache_task, lesson_workflow, mo, parameter_labels):
    _amplitude = "vol_of_vol" if cache_task["model"] == "heston" else "eta"
    _rho_bounds = cache_task["prior_bounds"][cache_task["parameter_names"].index("rho")]
    _amplitude_bounds = cache_task["prior_bounds"][cache_task["parameter_names"].index(_amplitude)]
    scenario_rho = mo.ui.slider(*_rho_bounds, step=.01, value=-.7,
                                 label="Correlation below", show_value=True)
    scenario_amplitude = mo.ui.slider(*_amplitude_bounds, step=.01,
        value=.3 if _amplitude == "vol_of_vol" else 1.5,
        label=f"{parameter_labels[_amplitude]} above", show_value=True)
    mo.vstack([lesson_workflow("Use"), mo.md(r"""
    ## 6. Use: a point or a probability?

    **Does one parameter vector express what we learned?** Ask whether correlation
    is below one threshold **and** variance amplitude is above another. Applying
    this rule to the posterior-mean parameters gives a yes/no answer. Applying it
    to complete paired posterior draws gives a posterior probability.
    """), scenario_rho, scenario_amplitude])
    return scenario_amplitude, scenario_rho


@app.cell(hide_code=True)
def _(cache_task, mo, np, parameter_labels, posterior, scenario_amplitude, scenario_rho):
    _names = cache_task["parameter_names"]
    _amplitude = "vol_of_vol" if cache_task["model"] == "heston" else "eta"
    _draws = posterior["draws"]
    _rho, _a = _names.index("rho"), _names.index(_amplitude)
    _event = (_draws[:, _rho] < scenario_rho.value) & (_draws[:, _a] > scenario_amplitude.value)
    _point = _draws.mean(axis=0)
    _plugin = _point[_rho] < scenario_rho.value and _point[_a] > scenario_amplitude.value
    mo.vstack([
        mo.md(f"**Source: {posterior['source']} posterior · {posterior['length']} days.** "
              f"Question: ρ < {scenario_rho.value:g} and {parameter_labels[_amplitude]} > {scenario_amplitude.value:g}."),
        mo.ui.table([{"answer from": "One posterior-mean parameter vector", "result": "yes" if _plugin else "no"},
                     {"answer from": "Complete paired posterior draws", "result": f'{100*np.mean(_event):.1f}% ({int(np.sum(_event))}/{len(_event)} draws)'}],
                    selection=None, pagination=False),
        mo.md("Both coordinates come from each same draw; multiplying marginal probabilities "
              "would impose independence. This is uncertainty about model parameters within the "
              "declared P task, not a probability of a future market event. Changing thresholds "
              "reuses the displayed posterior and runs no training or inference."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ### What would a continuation forecast need?

        Independent replicated histories check the fitted model from its origin.
        Predicting the next part of the observed history additionally needs its
        current hidden state; rough Bergomi also needs past-shock memory. The GRU
        summary is not a joint state–parameter posterior, so these results do not
        provide conditional return, VaR or expected-shortfall forecasts.

        **Takeaways**

        1. Ordered returns are the data; latent variance and innovations are hidden.
        2. A recurrent summary and conditional flow learn a joint parameter distribution.
        3. Random prefixes and masks let one estimator use different amounts of history.
        4. Prior prediction, replicated-history fit, recovery and calibration ask different questions.
        5. A parameter point hides uncertainty; retain paired posterior draws when using the result.
        """),
        mo.accordion({"Replicated histories, filtering and continuations": mo.md(r"""
        An independent replicate averages $p(r^{\rm rep}_{1:T}\mid\Theta)$ over
        $p(\Theta\mid r_{1:T})$. A conditional continuation instead needs

        $$p(r_{T+1:T+h}\mid r_{1:T})=
          \int p(r_{T+1:T+h}\mid z_T,\Theta)\,
          p(z_T,\Theta\mid r_{1:T})\,dz_T\,d\Theta.$$

        Here $z_T$ includes hidden state and, for rough Bergomi, memory. Filtering
        infers that state from past observations. This network infers fixed
        parameters after averaging over hidden paths.
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
        "optional training and fresh inference.")})
    return


if __name__ == "__main__":
    app.run()
