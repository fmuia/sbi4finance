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
    from sbi4f_tutorial.trajectories import diagnostics as path_checks, simulation as path_simulation
    plots.style()
    return json, mo, np, path_checks, path_simulation, plt, trajectory


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Learn volatility dynamics from observed paths

    **The parameters stay fixed; the evidence grows.**

    The option lessons used prices across strikes and maturities. Here we observe
    **daily log returns**, in time order. Can one history tell us how volatility
    moves, how it relates to returns, and what drift is plausible?

    We start with rough Bergomi and a supplied variance curve:
    **observe → simulate → train → infer → check**.

    | Route | What you do | Model source |
    |:---|:---|:---|
    | **Train here** | Generate a small training set, fit the GRU and flow, then infer parameters | A new tutorial model, kept in this session |
    | **Optional pretrained model** | Load the supplied estimator and reveal more days | A larger, separately trained model; saved draws work without a download |

    First inspect an observed history, then explore the forward simulator. Both
    training and model loading are explicit actions; the small simulator runs locally.

    The supplied estimators learn useful information, but **all three failed
    their final marginal-calibration screen**. We will inspect that distinction.
    """)
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
    mo.vstack([mo.md("## 1. Observe a history"), case_choice, inference_mode])
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
def _(cache_task, mo):
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
        mo.md(r"""
        ## 2. Define the model and its unknowns

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
def _(mo):
    mo.md(r"""
    ### Try the forward model

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
    """)
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
def _(cache_task, mo, trajectory):
    _figure = trajectory.ASSETS / "illustrations" / "application-workflows-20261101" / "trajectory-prefix-workflow.png"
    mo.vstack([
        mo.md("## 3. Train a tutorial model here\n\n"
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

        ```python
        # Scale returns in fixed units; map bounded labels to unbounded coordinates.
        train = inference.training_data(pairs, task)
        workflow = inference.build_workflow(task, network=trajectory.TUTORIAL_NETWORK)
        # Reveal random prefix lengths; mask every later day, including its time token.
        prefixes = inference.random_prefixes(
            task, seed=20261109, policy="uniform_log_mixture")
        # Same offline fit API; the GRU and flow receive eight optimizer updates.
        history = workflow.fit_offline(
            data=train, epochs=2, batch_size=32, augmentations=prefixes)
        # The helper conditions the workflow, samples, then restores original units.
        draws = inference.sample(workflow, task, example["observations"],
                                 draws=128, seed=20261110)[0]
        ```

        These calls continue the simulation example above. Two epochs on 128
        histories make **eight optimizer updates**. The last call infers parameters
        for the independent example using the newly trained model. It illustrates
        the workflow; no accuracy or calibration result is established.
        No training starts on opening the notebook.
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


@app.cell
def _(mo, path_simulation, plt, train_demo_button, trajectory):
    tutorial_task = path_simulation.teaching_task("rough-known", observations=252)
    tutorial_workflow = None
    _output = mo.md("The training example is idle until requested.")
    if train_demo_button.value:
        import os as _os
        from time import perf_counter as _clock
        _os.environ["KERAS_BACKEND"] = "jax"
        _os.environ["JAX_PLATFORMS"] = "cpu"
        from sbi4f_tutorial.trajectories import simulation as _simulation, inference as _inference
        try:
            _start = _clock()
            # Fresh prior/process draws supply training labels; no cached history is trained on.
            _task = tutorial_task
            _pairs = _simulation.generate(_task, "train", start=0, count=128,
                                          seed=20261107, backend="numpy")
            _example = _simulation.generate(_task, "validation", start=0, count=1,
                                            seed=20261108, backend="numpy")
            _generation_seconds = _clock() - _start
            _start = _clock()
            # Fixed scaling and parameter logits happen before the routing adapter.
            _train = _inference.training_data(_pairs, _task)
            _workflow = _inference.build_workflow(_task, network=trajectory.TUTORIAL_NETWORK)
            # Random-prefix augmentation masks all unrevealed returns and time tokens.
            _prefixes = _inference.random_prefixes(
                _task, seed=20261109, policy="uniform_log_mixture")
            _history = _workflow.fit_offline(
                data=_train, epochs=2, batch_size=32, augmentations=_prefixes, verbose=0)
            _training_seconds = _clock() - _start
            _start = _clock()
            # Infer only for the independently generated history; undo label transforms.
            _draws = _inference.sample(_workflow, _task, _example["observations"],
                                       draws=128, seed=20261110)[0]
            _sample_seconds = _clock() - _start
            _figure, _axis = plt.subplots(figsize=(6, 2.5), layout="constrained")
            _axis.plot([1, 2], _history.history["loss"], "o-", color="#7c3aed")
            _axis.set(xlabel="Epoch", ylabel="Training loss", xticks=[1, 2],
                      title="Separate one-year exercise: two training epochs")
            _output = mo.vstack([
                mo.md(f"**Source: newly trained tutorial model.** Generation: **{_generation_seconds:.2f} s**; "
                      f"model construction and training: **{_training_seconds:.2f} s**; "
                      f"128 draws: **{_sample_seconds:.2f} s**. Times describe this click only."),
                _figure,
                mo.ui.table(trajectory.posterior_rows(_task, _draws), selection=None, pagination=False),
                mo.md("These intervals come from the small exercise and its independently generated history. "
                      "Training loss and successful execution do not establish calibration. "
                      "The supplied five-year estimator and all plots below are unchanged."),
            ])
            tutorial_workflow = _workflow
        except (ImportError, OSError, RuntimeError, ValueError) as _error:
            _output = mo.callout(mo.md("The small training example could not complete: " + str(_error)), kind="warn")
    _output
    return tutorial_task, tutorial_workflow


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Optional: use the supplied pretrained model

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
def _():
    # In-memory inputs only. A complete task fingerprint keeps different process
    # laws, priors and maximum-history designs in separate panels.
    diagnostic_panels = {}
    return (diagnostic_panels,)


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
        return path_checks.plot_diagnostics(task, panel, draws)

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
        _recovery, _ranks = check_independent_histories(
            _workflow, diagnostic_task, _panel, diagnostic_days.value)
        _check_seconds = _diagnostic_clock() - _started
    except (ImportError, RuntimeError, ValueError, FloatingPointError) as _error:
        mo.stop(True, mo.callout(mo.md(f"**Diagnostic check stopped:** {_error}. "
            "No partial or selectively accepted cohort is displayed."), kind="warn"))
    mo.vstack([
        mo.md(f"**Source: {diagnostic_source.value} network · 200 independent histories · "
              f"{diagnostic_days.value} observed days each · 200 posterior draws per history.** "
              f"Input generation: **{'reused from this session' if _reused else f'{_generation_seconds:.3f} s'}**; "
              f"sampling and plotting: **{_check_seconds:.1f} s**, including compilation if needed."),
        _recovery, _ranks,
        mo.md("Recovery compares generating truths with posterior medians and central 90% intervals. "
              "The rank ECDF checks calibration averaged over the declared prior: near-zero differences "
              "mean roughly uniform ranks. Inspect both; returning the prior can pass a rank check without learning. "
              "These fixed-seed validation examples are reused across prefixes. Nested prefixes are not "
              "independent replicates, and these checks do not replace the final-test results below. "
              "A tiny fit may perform poorly; successful execution does not establish accuracy."),
    ])
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
    _contents = [mo.md("## 5. Check the uncertainty\n\n"
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
def _(mo):
    mo.vstack([
        mo.md(r"""
        ## 6. What comes next?

        **Parameter inference is not a continuation forecast.** Paired draws
        support parameter trade-offs and functions of the parameters. We could
        also generate an independent history from the model's origin as a
        posterior-predictive check. No replicated-history calculation is run here.

        Predicting the next part of the observed history additionally needs its
        current hidden state; rough Bergomi also needs the past-shock memory.
        This network does not infer that state/memory posterior, so it does not
        provide conditional return, VaR or expected-shortfall forecasts.

        **Takeaways**

        1. Ordered returns are the data; latent variance and shocks are hidden.
        2. A recurrent summary and conditional flow learn a joint parameter distribution.
        3. Random prefixes and masks let one estimator use different amounts of history.
        4. More data can improve estimates while intervals remain miscalibrated.
        """),
        mo.accordion({"Replicated histories, filtering and continuations": mo.md(r"""
        An independent replicate averages $p(r^{\rm rep}_{1:T}\mid\Theta)$ over
        $p(\Theta\mid r_{1:T})$. A conditional continuation instead needs

        $$p(r_{T+1:T+h}\mid r_{1:T})=
          \int p(r_{T+1:T+h}\mid z_T,\Theta)\,
          p(z_T,\Theta\mid r_{1:T})\,dz_T\,d\Theta.$$

        Here $z_T$ includes the relevant hidden state and, for rough Bergomi,
        memory. Filtering infers that state from past observations; this network
        infers fixed parameters after averaging over hidden paths. Its recurrent
        summary does not supply a joint state–parameter posterior.
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
