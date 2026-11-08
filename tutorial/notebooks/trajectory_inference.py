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
    plots.style()
    return json, mo, np, plt, trajectory


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Learn volatility dynamics from observed paths

    **The parameters stay fixed; the evidence grows.**

    The option lessons used prices across strikes and maturities. Here we observe
    **daily log returns**, in time order. Can one history tell us how volatility
    moves, how it relates to returns, and what drift is plausible?

    We start with rough Bergomi and a supplied variance curve:
    **observe → simulate → train → infer → check**. Saved results run without
    a model download; a small training exercise and fresh inference are optional.

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
        "README for the matching assets. No simulation or model download starts automatically."), kind="info"))
    cache_task, cache_report, cache_arrays = trajectory.load_example(model_choice.value)
    return cache_arrays, cache_report, cache_task


@app.cell
def _(cache_arrays, mo):
    case_choice = mo.ui.dropdown(
        options={f"History {index + 1}": index for index in range(len(cache_arrays["trajectory_id"]))},
        value="History 1", label="Synthetic history")
    inference_mode = mo.ui.radio(
        options={"Saved results": "saved", "Fresh inference at any day": "live"},
        value="Saved results", inline=True, label="Posterior source")
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

        Volatility is $\sqrt{V_t}$. Unlike option calibration, we do not infer
        a change of measure. The model's process randomness produces the returns;
        **there is no extra measurement-error or quote-noise term**.
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
def _(cache_task, mo, trajectory):
    _figure = trajectory.ASSETS / "illustrations" / "application-workflows-20261101" / "trajectory-prefix-workflow.png"
    mo.vstack([
        mo.md("## 3. Simulate and train\n\n"
              "**One training pair: parameters drawn from the prior and the daily returns they generate.** "
              "Only returns enter the estimator. Generating parameters are training labels; latent "
              "variance and shocks stay inside the simulator."),
        mo.image(_figure) if _figure.is_file() else mo.md(""),
        mo.md(f"""
        ### Generate parameter–history pairs

        ```python
        from sbi4f_tutorial.trajectories import simulation, inference

        task = simulation.teaching_task("{cache_task['model']}", observations=1260)
        pairs = simulation.generate(
            task, "train", start=0, count=128, seed=2026, backend="numpy")
        # pairs["observations"]: [128, 1260, 1]
        # pairs["parameters"]:   [128, number_of_unknowns]
        ```

        The simulator samples both parameters and process innovations. It retains
        aligned parameter labels, daily returns and history IDs. The implementation
        lives beside this lesson in `src/sbi4f_tutorial/trajectories/`.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build the posterior estimator

    **Keep the order; ignore the padding.** A causal GRU reads returns in order
    and produces a learned summary. The conditional flow maps that summary to a
    joint parameter distribution. Both networks train together.

    This is the construction inside `inference.build_workflow(task)`:

    ```python
    import bayesflow as bf

    adapter = (bf.Adapter().convert_dtype("float64", "float32")
        .rename("parameters", "inference_variables")
        .rename("observations", "summary_variables")
        .rename("time_context", "inference_conditions"))
    summary = bf.networks.RecurrentNetwork(
        hidden_dim=64, summary_dim=32, recurrent_type="gru",
        bidirectional=False, time_axis=1, time_embed_dim=4,
        dropout=0.0, return_sequences=False)
    density = bf.networks.CouplingFlow(
        depth=6, transform="spline", subnet_kwargs={"widths": [64, 64]},
        transform_kwargs={"bins": 8})
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
        train = inference.training_data(pairs, task)
        workflow = inference.build_workflow(task)
        prefixes = inference.random_prefixes(
            task, seed=2027, policy="uniform_log_mixture")
        history = workflow.fit_offline(
            data=train, epochs=5, batch_size=32, augmentations=prefixes)
        ```

        These calls continue the simulated-pairs example above. Five epochs on
        128 histories illustrate the API; they do not reproduce the supplied fit.
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
    ### Optional: run a small training example

    This separate **one-year rough Bergomi exercise** generates 128 new histories,
    trains a smaller network for two epochs (eight batches), then samples for one
    independent synthetic history. Its outputs do not replace the supplied results below.
    It illustrates the calls; it does not establish posterior accuracy.

    ```python
    from sbi4f_tutorial.trajectories import simulation, inference

    task = simulation.teaching_task("rough-known", observations=252)
    pairs = simulation.generate(task, "train", start=0, count=128,
                                seed=20261107, backend="numpy")
    example = simulation.generate(task, "validation", start=0, count=1,
                                  seed=20261108, backend="numpy")
    train = inference.training_data(pairs, task)
    workflow = inference.build_workflow(task, network={
        "hidden_dim": 16, "summary_dim": 16, "depth": 2,
        "bins": 8, "widths": [32, 32]})
    prefixes = inference.random_prefixes(
        task, seed=20261109, policy="uniform_log_mixture")
    history = workflow.fit_offline(
        data=train, epochs=2, batch_size=32, augmentations=prefixes)
    draws = inference.sample(workflow, task, example["observations"],
                             draws=128, seed=20261110)[0]
    ```

    The button runs these steps on the CPU, keeps results only in this session,
    and reports generation, model-building/training and sampling time separately.
    First use includes backend imports and compilation.
    """)
    return


@app.cell
def _(mo):
    train_demo_button = mo.ui.run_button(label="Generate 128 histories and train the small example")
    train_demo_button
    return (train_demo_button,)


@app.cell
def _(mo, plt, train_demo_button, trajectory):
    mo.stop(not train_demo_button.value, mo.md("The training example is idle until requested."))
    import os as _os
    from time import perf_counter as _clock
    _os.environ["KERAS_BACKEND"] = "jax"
    _os.environ["JAX_PLATFORMS"] = "cpu"
    from sbi4f_tutorial.trajectories import simulation as _simulation, inference as _inference
    try:
        _start = _clock()
        _task = _simulation.teaching_task("rough-known", observations=252)
        _pairs = _simulation.generate(_task, "train", start=0, count=128,
                                      seed=20261107, backend="numpy")
        _example = _simulation.generate(_task, "validation", start=0, count=1,
                                        seed=20261108, backend="numpy")
        _generation_seconds = _clock() - _start
        _start = _clock()
        _train = _inference.training_data(_pairs, _task)
        _workflow = _inference.build_workflow(_task, network={
            "hidden_dim": 16, "summary_dim": 16, "depth": 2,
            "bins": 8, "widths": [32, 32]})
        _prefixes = _inference.random_prefixes(
            _task, seed=20261109, policy="uniform_log_mixture")
        _history = _workflow.fit_offline(
            data=_train, epochs=2, batch_size=32, augmentations=_prefixes, verbose=0)
        _training_seconds = _clock() - _start
        _start = _clock()
        _draws = _inference.sample(_workflow, _task, _example["observations"],
                                   draws=128, seed=20261110)[0]
        _sample_seconds = _clock() - _start
        _figure, _axis = plt.subplots(figsize=(6, 2.5), layout="constrained")
        _axis.plot([1, 2], _history.history["loss"], "o-", color="#7c3aed")
        _axis.set(xlabel="Epoch", ylabel="Training loss", xticks=[1, 2],
                  title="Separate one-year exercise: two training epochs")
        _output = mo.vstack([
            mo.md(f"**Small training example completed.** Generation: **{_generation_seconds:.2f} s**; "
                  f"model construction and training: **{_training_seconds:.2f} s**; "
                  f"128 draws: **{_sample_seconds:.2f} s**. Times describe this click only."),
            _figure,
            mo.ui.table(trajectory.posterior_rows(_task, _draws), selection=None, pagination=False),
            mo.md("These intervals come from the small exercise and its independently generated history. "
                  "Training loss and successful execution do not establish calibration. "
                  "The supplied five-year estimator and all plots below are unchanged."),
        ])
    except (ImportError, OSError, RuntimeError, ValueError) as _error:
        _output = mo.callout(mo.md("The small training example could not complete: " + str(_error)), kind="warn")
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Infer parameters and reveal more days

    **Reuse the fitted estimator; change the evidence.** With the matching model
    downloaded, loading and querying the supplied estimator takes three calls:

    ```python
    task, workflow, record = trajectory.load_live(model_choice.value)
    revealed = cache_arrays["observations"][case_choice.value:case_choice.value + 1,
                                           :observed_days.value]
    draws = trajectory.sample(workflow, task, revealed, draws=512, seed=20261031)[0]
    ```

    Each query uses the **original prior and the entire revealed history**.
    We do not multiply posteriors from overlapping prefixes or feed the previous
    posterior into this unchanged network.
    """)
    return


@app.cell
def _(inference_mode, mo):
    checkpoint_path = mo.ui.text(value="", label="Model folder relative to tutorial/assets/ (optional)", full_width=True)
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
    if inference_mode.value == "live" and live_button.value:
        from time import perf_counter as _live_clock
        try:
            _start = _live_clock()
            _task, _workflow, _record = trajectory.load_live(
                model_choice.value, checkpoint_path.value.strip() or None)
            _load_seconds = _live_clock() - _start
            _revealed = cache_arrays["observations"][case_choice.value:case_choice.value + 1, :requested_days]
            _start = _live_clock()
            _draws = trajectory.sample(_workflow, _task, _revealed, draws=512, seed=20261031)[0]
            _sample_seconds = _live_clock() - _start
            _candidate = {"selection": selection, "draws": _draws}
            trajectory.posterior_view(cache_task, cache_arrays, case_choice.value, requested_days, live=_candidate)
            live_result = _candidate
            live_timing = (_load_seconds, _sample_seconds)
        except (FileNotFoundError, ValueError, RuntimeError, ImportError, OSError) as _error:
            live_error = str(_error)
    return live_error, live_result, live_timing


@app.cell
def _(cache_arrays, cache_task, case_choice, live_result, requested_days, trajectory):
    posterior = trajectory.posterior_view(
        cache_task, cache_arrays, case_choice.value, requested_days, live=live_result)
    return (posterior,)


@app.cell(hide_code=True)
def _(inference_mode, live_error, live_timing, mo, posterior, requested_days):
    _contents = [mo.callout(mo.md(f"**{posterior['label']}.**"), kind="neutral")]
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
                     "intervals can move or widen after surprising returns. The parameters themselves stay fixed."),
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
