import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Learn volatility dynamics from observed paths")


@app.cell
def _():
    import json
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from sbi4f_tutorial import plots, trajectory
    plots.style()
    return json, mo, np, plt, trajectory


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Learn volatility dynamics from observed paths

    **The parameters stay fixed; the evidence grows.**

    Here we observe the asset through time, rather than a set of option prices.
    Can its returns tell us how volatility moves, how it relates to price changes,
    and what drift is plausible? We learn a **joint approximate posterior** for
    those unknown parameters from a history of daily log returns.

    **The route:** stochastic model → simulated histories → recurrent summary
    and conditional flow → inference at a growing history → accuracy checks.

    Heston and both rough Bergomi examples share the same observation design:
    **one return per trading day, up to 1,260 days (five years)**. Each model has
    its own synthetic history. Switching models changes the observed data too;
    this is not a comparison of three models fitted to one shared path.

    We start with rough Bergomi and a supplied variance curve. Every integer
    history length is supported by the architecture. Saved posterior snapshots
    make the lesson usable without loading a model; exact-length CPU sampling
    is an optional action below.

    **Accuracy matters:** all three fitted estimators failed their final
    marginal-calibration screen. We will inspect useful learning alongside that
    failure, rather than treating a plausible-looking interval as a correct one.
    Heston also has a numerical limitation explained beside its model.
    """)
    return


@app.cell
def _(mo, trajectory):
    model_choice = mo.ui.dropdown(
        options={label: key for key, label in trajectory.MODEL_LABELS.items()},
        value="Rough Bergomi · known variance curve", label="Stochastic volatility model")
    observed_days = mo.ui.slider(1, 1260, step=1, value=252,
                                label="Observed trading days", show_value=True)
    mo.hstack([model_choice, observed_days], justify="start")
    return model_choice, observed_days


@app.cell
def _(mo, model_choice, trajectory):
    mo.stop(not trajectory.CATALOGUE.is_file(), mo.callout(mo.md(
        "**Trajectory examples are not packaged in this copy yet.** Obtain the "
        "matching tutorial trajectory assets described in the README. No model "
        "download, training, simulation or cluster connection starts automatically."), kind="info"))
    cache_task, cache_report, cache_arrays = trajectory.load_example(model_choice.value)
    return cache_arrays, cache_report, cache_task


@app.cell(hide_code=True)
def _(cache_task, mo):
    parameter_labels = {
        "kappa": "Mean reversion κ", "long_variance": "Long variance L",
        "vol_of_vol": "Volatility of variance σ", "rho": "Leverage correlation ρ",
        "hurst": "Roughness H", "eta": "Volatility amplitude η",
        "short_variance": "Short variance S", "decay": "Curve decay k", "drift": "Physical drift μ",
    }
    _meanings = {
        "kappa": "Speed at which Heston variance reverts toward its long level",
        "long_variance": "Long-run Heston variance level" if cache_task["model"] == "heston"
                         else "Long level of the deterministic variance-scale curve",
        "vol_of_vol": "Amplitude of Heston's variance shocks",
        "rho": "Correlation between price and variance-driver Brownian shocks",
        "hurst": "Memory-kernel roughness parameter of the Gaussian volatility driver",
        "eta": "Amplitude of the rough Gaussian driver in log variance",
        "short_variance": "Initial level of the deterministic variance-scale curve",
        "decay": "Rate joining the short and long curve levels",
        "drift": "Annual proportional price drift; log-return drift also contains −V/2",
    }
    _priors = [{"unknown parameter": parameter_labels[_name], "meaning": _meanings[_name],
                "prior family": "Log-uniform (density ∝ 1/x)" if _kind == "log_uniform" else "Uniform",
                "support": f"{_bounds[0]:g} to {_bounds[1]:g}"}
               for _name, _kind, _bounds in zip(cache_task["parameter_names"], cache_task["prior_types"],
                                               cache_task["prior_bounds"], strict=True)]
    _fixed = [{"supplied quantity": _name.replace("_", " "), "value": _value}
              for _name, _value in cache_task["fixed"].items()]
    _steps = cache_task["numerics"].get("substeps", cache_task["numerics"].get("observation_substeps"))
    _equation = (r"$$dV_t=\kappa(L-V_t)dt+\sigma\sqrt{V_t}\,dW_t^V.$$"
                 if cache_task["model"] == "heston" else
                 r"$$V_t=\xi(t)\exp\!\left(\eta Y_t-\tfrac12\eta^2\operatorname{Var}_{\rm grid}(Y_t)\right).$$")
    _model_story = (
        "**Heston: variance is pulled toward a long-run level, with random shocks around it.** "
        "Mean reversion κ sets the pull, L its destination, and σ the strength of the shocks. "
        "Correlation ρ connects price and variance shocks. Unlike the option example, κ is inferred here."
        if cache_task["model"] == "heston" else
        "**Rough Bergomi: today's variance depends on a weighted history of random shocks.** "
        "The Gaussian driver Y remembers past shocks through a kernel. H controls its roughness, "
        "η the size of its effects on log variance, and ρ the connection to price shocks. "
        "The deterministic curve ξ sets the mean variance scale; it is not the realized variance path.")
    _curve = ("The initial variance is supplied; the realized subsequent variance remains hidden."
              if cache_task["model"] == "heston" else
              "The deterministic variance-scale curve is the supplied constant level shown below; the realized variance remains hidden."
              if cache_task["model"] == "rough-known" else
              r"The curve is inferred: $\xi(t)=L+(S-L)e^{-kt}$. Its coefficients are targets, not observed inputs.")
    mo.vstack([
        mo.md(r"""
        ## 1 · The model and the observed data

        **Observed:** daily log returns $r_t=\log S_t-\log S_{t-1}$, in their time order.
        A return of 0.01 is about a 1% price increase. The time step is known:
        $\Delta t=1/252$ year. **Hidden:** the instantaneous variance $V_t$ and
        all random shocks. Volatility is $\sqrt{V_t}$, not $V_t$.

        This is a **physical-measure model**: an explicit statistical law for
        returns. It does not estimate a change of measure from option prices.
        Each history starts at the model's declared origin. In continuous-time
        notation, the price and variance drivers are related by

        $$
        d\log S_t=(\mu-V_t/2)dt+\sqrt{V_t}\,dW_t^S,
        \qquad d\langle W^S,W^V\rangle_t=\rho\,dt.
        $$
        """),
        mo.md(_model_story + "\n\n" + _equation + "\n\n" + _curve),
        mo.md("**Process randomness is already in the returns.** These tasks add no separate "
              "measurement-error or option-quote-noise term. The simulator integrates its internal "
              "price moves into each daily log return. Synthetic parameter values are known for "
              "training and evaluation; they are never part of the observed input."),
        mo.md("### What we infer, and what we assume\n\n"
              "The table defines the **prior**, our distribution over possible parameters before "
              "this history. A uniform prior gives equal probability to equal-width intervals. "
              "A log-uniform prior gives equal probability to equal multiplicative ranges, "
              "such as a doubling."),
        mo.ui.table(_priors, selection=None, pagination=False),
        mo.md("These declared parameter priors are independent. Supplied quantities are fixed modelling assumptions, not extra measurements of latent volatility."),
        mo.ui.table(_fixed, selection=None, pagination=False) if _fixed else mo.md("**No fixed model parameters** in this joint-curve task."),
        mo.md(f"**Same daily observations, internal numerical approximation.** This simulator uses "
              f"**{_steps} internal steps per day**, then supplies one daily return. "
              "Changing the model selector does not change the daily observation frequency. "
              "The equations explain the intended dynamics; inference is trained for the actual "
              "finite-step simulator, whose approximation matters."),
        mo.accordion({"Mathematical and numerical details": mo.vstack([
            mo.md(r"With $\Theta$ denoting the full parameter vector, these independent priors give "
                  r"$p(\Theta)=\prod_j p_j(\Theta_j)$. A log-uniform marginal on $[a,b]$ has density "
                  r"$1/[x\log(b/a)]$. For rough Bergomi, the idealized driver is "
                  r"$Y_t=\sqrt{2H}\int_0^t(t-s)^{H-1/2}\,dW_s^V$. The implemented hybrid grid "
                  r"uses its own Gaussian variance in the exponential compensator; "
                  r"$\operatorname{Var}_{\rm grid}(Y_t)$ in the displayed equation makes that numerical convention explicit."),
            mo.ui.table(
            [{"setting": _name.replace("_", " "), "declared value": _value}
             for _name, _value in cache_task["numerics"].items()], selection=None, pagination=False),
        ])}),
    ])
    return (parameter_labels,)


@app.cell(hide_code=True)
def _(cache_task, mo):
    _warning = (mo.callout(mo.md(
        "**A limitation in this Heston example.** Its finite-step simulator can keep effective "
        "variance at zero for an entire day. That day's return is then exactly drift × time. "
        "The parameter posterior can contain a point mass in drift: positive probability at "
        "one exact value. A continuous neural density cannot represent that mass exactly. "
        "These intervals therefore illustrate an approximation with a known representation "
        "problem. Fixing it requires a new numerical specification and a new fit."), kind="warn")
        if cache_task["model"] == "heston" else mo.md(""))
    _warning
    return



@app.cell(hide_code=True)
def _(mo, trajectory):
    _figure = trajectory.ASSETS / "illustrations" / "application-workflows-20261101" / "trajectory-prefix-workflow.png"
    mo.vstack([
        mo.md("## 2 · Simulate once, learn from many histories\n\n"
              "**A training pair is one set of parameters and the history it generated.** "
              "Draw the parameters from the declared prior, simulate the hidden process, "
              "and retain only daily returns as observations. The parameter label is the "
              "answer supplied during training. Hidden variances and innovations stay inside the simulator."),
        mo.image(_figure) if _figure.is_file() else mo.md(""),
        mo.md("**The supplied bank:** one million independent training histories per model, "
              "each containing 1,260 daily returns. Independent validation and test histories "
              "are stored separately. Generation happens before training; training batches "
              "read those saved histories rather than simulating new paths on every update."),
    ])
    return


@app.cell(hide_code=True)
def _(cache_task, mo):
    mo.md(f"""
    **The simulation API.** This author-side example runs from `lab/`; it is
    shown here rather than executed by the participant notebook.

    ```python
    from sbi4finance.path_inference import bank

    task = bank.teaching_task("{cache_task['model']}", observations=1260)
    pairs = bank.generate(task, "train", start=0, count=128, seed=2026)
    returns = pairs["observations"]   # [128, 1260, 1]
    parameters = pairs["parameters"] # [128, number_of_unknowns]
    ```

    `generate` samples both the prior and the process innovations. It returns
    aligned labels, daily returns and trajectory IDs; no latent variance channel
    is saved as an observation. The full-bank writer repeats this in bounded
    chunks with independent trajectory IDs. A new chunk seed or a changed task
    defines different data; this small example does not reproduce the published fit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · A summary for an ordered, variable-length history

    **Keep the order; ignore the padding.** A recurrent network reads returns in
    time order and compresses them into a learned summary. A conditional flow
    turns that summary into a distribution over parameters, including their
    trade-offs. The summary and density network are trained together.

    The published estimator uses a **64-unit causal GRU**, a **32-value summary**
    and **six spline coupling layers**. A GRU carries a learned hidden state
    from one observed day to the next. That neural state is a summary of the
    observations; it is not an observed volatility path or a fitted latent-state filter.

    This is the BayesFlow construction inside the author-side
    `inference.build_workflow(task)`:

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

    The boolean `summary_mask` accompanies `summary_variables`: true for an
    observed day, false for padding. A genuine zero return is still observed
    and has a true mask. Masking lets the **same architecture** accept every
    integer length from 1 to 1,260; it does not guarantee equal accuracy at every length.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        **Train on random prefixes of an offline bank.** On each training pass,
        reveal a randomly chosen prefix of each selected history and mask its
        future. The published training mixes uniform integer lengths with a
        distribution giving more exposure to short histories. Every integer
        length remains possible. This changes training exposure, not the prior
        or the return-process law.

        A small **author-side software illustration**, continuing the `pairs`
        example above, uses the public offline-training API:

        ```python
        from sbi4finance.path_inference import inference

        train = inference.training_data(pairs, task)
        workflow = inference.build_workflow(task)
        prefixes = inference.random_prefixes(
            task, seed=2027, policy="uniform_log_mixture")
        history = workflow.fit_offline(
            data=train, epochs=5, batch_size=32, augmentations=prefixes)
        ```

        `training_data` scales returns in fixed units, adds known times and a
        mask, and transforms parameter labels to unconstrained coordinates.
        The adapter routes labels as inference variables and returns as summary
        variables. **Labels are never inference conditions.** Five epochs on
        128 histories illustrate the calls; they do not recreate the million-history
        training or establish posterior accuracy.

        At full scale, the author runner reads bounded native shards, samples
        prefixes per batch, and keeps only a small shard cache in memory:

        ```bash
        # From lab/, after obtaining the matching native bank.
        python scripts/train_path_posterior.py \
            --bank /path/to/native-bank --out /path/to/new-run \
            --streaming --training-limit 1000000 --cache-shards 2 \
            --prefix-policy uniform_log_mixture --epochs 5 --batch-size 128
        ```

        The supplied million-history fits completed five epochs. Reproducing
        their weights additionally requires their recorded seeds, optimizer and
        checkpoint selection. This is an explicit author training recipe, not
        an action started by opening the notebook. The supplied pretrained
        examples remain the inference path for the workshop.
        """),
        mo.accordion({"What learning and updating mean mathematically": mo.md(r"""
        With $\Theta$ the fixed parameter vector and $z$ all hidden simulator
        variables, our target integrates out the hidden process:

        $$p(\Theta\mid r_{1:T})\propto p(\Theta)
          \int p(r_{1:T},z\mid\Theta)\,dz.$$

        We fit $q_\phi(\Theta\mid r_{1:T},T)$ using simulated parameter–history
        pairs. For this flow, training minimizes the negative log density of
        the transformed generating parameters, conditional on the recurrent summary.

        **Static-parameter inference and filtering differ.** Filtering estimates
        the current hidden process state from past observations; this network
        estimates fixed parameters after averaging over hidden paths. For rough
        Bergomi, the relevant hidden information includes memory of earlier shocks.

        At each new prefix we reuse the **original prior and the entire revealed
        history**. Multiplying approximate posteriors from overlapping prefixes,
        or feeding the previous posterior back into this unchanged estimator,
        would count observations more than once.
        """)}),
    ])
    return




@app.cell
def _(cache_arrays, mo):
    case_choice = mo.ui.dropdown(
        options={f"History {index}": index for index in range(len(cache_arrays["trajectory_id"]))},
        value="History 0", label="Synthetic history (each model has its own data)")
    mo.vstack([mo.md("## 4 · Reveal a history and infer its fixed parameters"), case_choice])
    return (case_choice,)


@app.cell
def _(cache_arrays, cache_task, case_choice, observed_days, trajectory):
    cached = trajectory.cached_posterior(cache_task, cache_arrays, case_choice.value, observed_days.value)
    return (cached,)


@app.cell(hide_code=True)
def _(cached, mo, observed_days):
    _detail = ("The cached posterior uses all currently revealed returns." if cached["exact"] else
               "The return plot reveals the requested history. Its cached posterior uses only the "
               "labelled earlier prefix; snapshots are never interpolated. Use live inference below "
               "to condition on every revealed return.")
    mo.vstack([
        mo.callout(mo.md(f"**{cached['label']}.** {_detail}"), kind="neutral"),
        mo.md(f"Elapsed observation time: **{observed_days.value / 252:.3g} trading years**. "
              "These four synthetic histories were selected before inspecting their posteriors. "
              "Their generating truth is known, but remains excluded from inference.\n\n"
              "**Try 137 days:** the return plot and actual network inputs use 137 returns, "
              "while the saved posterior uses the earlier **126-day** snapshot. The architecture "
              "supports 137 directly; use the optional CPU action for that exact conditioning length. "
              "Saved snapshots are never interpolated."),
    ])
    return


@app.cell(hide_code=True)
def _(cache_arrays, cache_task, cached, case_choice, mo, observed_days, plt, trajectory):
    _length = observed_days.value
    _revealed = cache_arrays["observations"][case_choice.value:case_choice.value + 1, :_length].copy()
    _inputs = trajectory.conditioning_data(_revealed, cache_task)
    _maximum = cache_task["n_observations"]
    _indices = sorted({0, max(0, _length - 2), _length - 1}
                      | set(range(_length, min(_length + 2, _maximum))))
    conditioning_preview = {
        "requested_length": _length,
        "shapes": [{"input tensor": _name, "shape": str(_value.shape), "dtype": str(_value.dtype)}
                   for _name, _value in _inputs.items()],
        "time_context": _inputs["time_context"][0].tolist(),
        "rows": [{"day index (1-based)": _index + 1,
                  "role": "observed" if _inputs["summary_mask"][0, _index] else "padding",
                  "scaled return": float(_inputs["observations"][0, _index, 0]),
                  "normalized time": float(_inputs["observations"][0, _index, 1]),
                  "mask": bool(_inputs["summary_mask"][0, _index])}
                 for _index in _indices],
    }
    _mask_figure, _mask_axis = plt.subplots(figsize=(9, 1.25), constrained_layout=True)
    _mask_axis.imshow(_inputs["summary_mask"], aspect="auto", cmap="Blues", vmin=0, vmax=1,
                      extent=(.5, _maximum + .5, -.5, .5), interpolation="nearest")
    _mask_axis.axvline(_length + .5, color="#111827", linewidth=1)
    _mask_axis.set(yticks=[], xlabel="day position in the padded input",
                   title=f"Actual GRU mask: {_length} observed days; {_maximum - _length} padded rows")
    mo.vstack([
        mo.md(f"### Follow the actual input into the network\n\n"
              f"**Selected history: {_length} revealed daily returns.** The dark mask is true for "
              "observed days; pale rows are padding. Padding is storage, not an observed zero return. "
              f"The separate cached posterior below still uses **{cached['length']} days**."),
        mo.md("""```python
revealed_returns = cache_arrays["observations"][case_choice.value:case_choice.value + 1,
                                                :observed_days.value]
inputs = trajectory.conditioning_data(revealed_returns, cache_task)
# inputs: observations, time_context, summary_mask
```"""),
        _mask_figure,
        mo.ui.table(conditioning_preview["rows"], selection=None, pagination=False),
        mo.md(r"Returns use the fixed scale $r_t/\sqrt{0.04\,\Delta t}$. "
              "The reference variance 0.04 is a choice of units, not an estimate of this path's latent variance. "
              "Dividing each history by its own observed volatility would remove variance-amplitude information. "
              "No future return, parameter truth, hidden variance or innovation enters these inputs."),
        mo.accordion({"Array shapes, time coordinates and padding details": mo.vstack([
            mo.ui.table(conditioning_preview["shapes"], selection=None, pagination=False),
            mo.md(f"Known `time_context`: **{conditioning_preview['time_context']}** = "
                  f"[observed length / {_maximum}, daily step / (1/252 year)]. "
                  "The observation tensor contains scaled return and normalized day position. "
                  "Padded returns and times are zero, with false masks. Constructing these inputs "
                  "does not load a model or run posterior inference."),
        ])}),
    ])
    return (conditioning_preview,)


@app.cell
def _(cache_arrays, cache_task, case_choice, observed_days, trajectory):
    trajectory.plot_returns(cache_task, cache_arrays, case_choice.value, observed_days.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Predict, then reveal.** Which parameter should be hardest to learn from a
    short history? Increase the observed days while keeping the model and
    history fixed. Does drift sharpen as quickly as volatility amplitude?
    The parameters are constant along the history: changing posterior intervals
    represent changing evidence, not parameters moving through time.

    Each query uses the original prior and the complete revealed history.
    We do not multiply posteriors from overlapping prefixes or feed the previous
    posterior back into this unchanged estimator.
    """)
    return


@app.cell
def _(cache_arrays, cache_task, case_choice, observed_days, trajectory):
    trajectory.plot_evolution(cache_task, cache_arrays, case_choice.value, observed_days.value)
    return


@app.cell(hide_code=True)
def _(cache_task, cached, mo, trajectory):
    mo.vstack([
        mo.md("**Approximate posterior at the labelled saved prefix.** Intervals are central 90% marginal intervals; "
              "the stored draw rows preserve the joint distribution."),
        mo.ui.table(trajectory.posterior_rows(cache_task, cached["draws"]), selection=None, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(cached, mo):
    mo.md(f"""
    ### Inspect uncertainty, one parameter and one pair at a time

    The following density and dependence views use **the same paired cached draw rows**
    at **{cached['length']} observed days**. They do not use a later prefix when
    the history control lies between cached snapshots.

    The dashed prior densities below are the declared marginal priors,
    not densities uniform over every parameter box. A histogram changing from its
    prior shows observation dependence; contraction alone does not establish accuracy.
    **There is no matched reference posterior for these displayed histories and lengths.**
    The plots compare the prior and neural approximation with known generating
    truth; truth is one fixed parameter value, not a reference distribution.
    """)
    return


@app.cell
def _(cache_arrays, cache_task, cached, case_choice, np, parameter_labels, plt):
    _names = cache_task["parameter_names"]
    _figure, _axes = plt.subplots((len(_names) + 1) // 2, 2,
                                 figsize=(10, 2.45 * ((len(_names) + 1) // 2)), squeeze=False)
    for _index, _axis in enumerate(_axes.flat):
        if _index >= len(_names):
            _axis.set_visible(False)
            continue
        _low, _high = cache_task["prior_bounds"][_index]
        _grid = np.linspace(_low, _high, 240)
        _density = (1 / (_grid * np.log(_high / _low))
                    if cache_task["prior_types"][_index] == "log_uniform" else np.full_like(_grid, 1 / (_high - _low)))
        _axis.hist(cached["draws"][:, _index], bins=np.linspace(_low, _high, 29), density=True,
                   alpha=.42, color="#2563eb", label="saved neural approximation")
        _axis.plot(_grid, _density, linestyle="--", color="#64748b", linewidth=1.5,
                   label="declared prior density")
        _axis.axvline(cache_arrays["truth"][case_choice.value, _index], linestyle=":", color="#111827",
                     linewidth=1, label="synthetic truth (evaluation only)")
        _axis.set(xlim=(_low, _high), title=parameter_labels[_names[_index]], ylabel="Marginal density")
    _axes.flat[0].legend(fontsize=7)
    _figure.suptitle(f"Prior and approximate posterior · history {case_choice.value} · {cached['length']} days", fontsize=12)
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(cache_task, mo, parameter_labels):
    _names = cache_task["parameter_names"]
    _options = {parameter_labels[_name]: _name for _name in _names}
    _x = "vol_of_vol" if cache_task["model"] == "heston" else "eta"
    pair_x = mo.ui.dropdown(options=_options, value=parameter_labels[_x], label="Joint view: horizontal parameter")
    pair_y = mo.ui.dropdown(options=_options, value=parameter_labels["rho"], label="Joint view: vertical parameter")
    mo.hstack([pair_x, pair_y], justify="start")
    return pair_x, pair_y


@app.cell
def _(cache_arrays, cache_task, cached, case_choice, mo, pair_x, pair_y, parameter_labels, plt):
    mo.stop(pair_x.value == pair_y.value,
            mo.md("Choose two different parameters to inspect their joint distribution."))
    _x = cache_task["parameter_names"].index(pair_x.value)
    _y = cache_task["parameter_names"].index(pair_y.value)
    _figure, _axis = plt.subplots(figsize=(7, 4.2), layout="constrained")
    _axis.scatter(cached["draws"][:, _x], cached["draws"][:, _y], s=10, alpha=.28, color="#2563eb",
                  label="paired neural draw rows")
    _axis.scatter(cache_arrays["truth"][case_choice.value, _x], cache_arrays["truth"][case_choice.value, _y],
                  marker="x", color="#111827", s=60, label="synthetic truth (evaluation only)")
    _axis.set(xlabel=parameter_labels[pair_x.value], ylabel=parameter_labels[pair_y.value],
              xlim=cache_task["prior_bounds"][_x], ylim=cache_task["prior_bounds"][_y],
              title=f"Joint parameter uncertainty · {cached['length']} observed days")
    _axis.legend(fontsize=8)
    mo.vstack([_figure, mo.md("Each point keeps two coordinates from the **same** posterior draw. "
                              "Independently shuffling columns would destroy dependence. "
                              "Visible trade-offs describe the learned approximation; checking their accuracy requires more than this picture.")])
    return


@app.cell(hide_code=True)
def _(cache_task, mo):
    _curve_lesson = (
        "**Curve uncertainty:** short variance, long variance and decay describe the deterministic "
        "variance-scale curve. Similar short and long levels make decay weakly informative; "
        "a broad interval can be appropriate even when other parameters sharpen."
        if cache_task["model"] == "rough-joint" else
        "**Known quantities:** Heston fixes the initial variance, while the known-curve rough task "
        "fixes its variance-scale curve. Neither task supplies the realized latent volatility path.")
    mo.md(rf"""
    **Drift is inferred in every model.** It is annual proportional price drift,
    $dS_t/S_t=\mu\,dt+\sqrt{{V_t}}\,dW_t$, so the log-return mean contains
    $(\mu-V_t/2)dt$. A single noisy path can leave substantial drift uncertainty.

    {_curve_lesson}

    More elapsed time and more frequent sampling provide different information.
    Moving this control extends the observed daily history. Posterior intervals
    along one realization can move or widen after surprising data. Their movement
    reflects learning about fixed parameters, not changing parameter values.

    The dotted lines show synthetic truth for interpretation only. Truth and
    latent volatility are excluded from neural inference conditions. These
    pictures alone do not establish accurate intervals for this history or
    accurate dependence between parameters.
    """)
    return


@app.cell(hide_code=True)
def _(cache_report, mo):
    mo.accordion({"Saved estimator identity and example scope": mo.md(
        "These fitted estimators use a **64-unit causal GRU**, a **32-value summary**, "
        "and **six spline coupling layers** for the joint parameter density. "
        "The GRU learns temporal features; the flow represents uncertainty and "
        "parameter trade-offs conditional on those features and the time design.  \n\n"
        f"Checkpoint SHA-256: `{cache_report['checkpoint_sha256']}`  \n"
        f"Task SHA-256: `{cache_report['task_sha256']}`  \n"
        "The cache contains validation observations and neural draws. Its file hashes, "
        "daily time design and checkpoint identities are verified on loading. "
        "Any aggregate final-test results describe a separate evaluation cohort.")})
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
    _contents = [mo.md("## 5 · Does learning useful information mean the intervals are right?\n\n"
                       "**Recovery and calibration ask different questions.** Recovery asks whether "
                       "parameter estimates move closer to the generating truth across many simulated "
                       "histories. Calibration asks whether uncertainty statements have their promised "
                       "frequency: for continuous posteriors, nominal 90% intervals should contain "
                       "truth about 90% of the time over independent histories drawn from this prior "
                       "and simulator. Point masses can make central intervals overcover; Heston's "
                       "drift check treats that exception explicitly.\n\n"
                       "One attractive posterior picture cannot answer either population question. "
                       "A narrower interval can be informative and still too narrow.")]
    if aggregate_report is None:
        _contents.append(mo.callout(mo.md(
            "**Packaged final-cohort diagnostics are unavailable in this copy.** "
            + (aggregate_problem or "The selected cached example remains an illustration, not an evaluation study.")), kind="warn"))
    else:
        _record = aggregate_report["models"][cache_task["model"]]
        _records = list(aggregate_report["models"].values())
        _execution_passes = sum(_entry["execution_pass"] for _entry in _records)
        _risk_passes = sum(_entry["risk_pass"] for _entry in _records)
        _calibration_failures = sum(not _entry["calibration_pass"] for _entry in _records)
        _contents.extend([
            mo.md(f"The checks use **{_record['cases']:,} independent test histories per model**, "
                  f"with **{_record['draws']} draws per history** at twelve frozen lengths. "
                  "They are separate from the four histories selected above. "
                  "A whole history is the independent unit; its nested prefixes are dependent. "
                  "These reserved tests have now been examined, so further acceptance work needs fresh data."),
            mo.callout(mo.md(f"**{_calibration_failures} of {len(_records)} estimators failed the final "
                             "marginal-calibration screen.** They can reduce estimation error while "
                             "giving intervals that miss their declared frequency. "
                             "Execution and lower average error do not establish accurate uncertainty. "
                             "Heston also has the point-mass limitation described above."), kind="warn"),
            mo.ui.table([{"selected model": trajectory.MODEL_LABELS[cache_task["model"]],
                          "final marginal-calibration screen": "passed" if _record["calibration_pass"] else "failed",
                          "90% coverage range across targets and lengths": f"{100 * _record['coverage90_range'][0]:.2f}% to {100 * _record['coverage90_range'][1]:.2f}%"}],
                        selection=None, pagination=False),
            mo.md("**Where the selected model failed.** These are the original joint-screen decisions "
                  "across parameters and lengths, not new tests chosen after looking at the plots."),
            mo.ui.table([{"observed days": _cell["days"], "parameter": parameter_labels.get(_cell["parameter"], _cell["parameter"]),
                          "empirical coverage of nominal 90% interval": f"{100 * _cell['coverage90']:.2f}%",
                          "screen criterion": _cell["criterion"].replace("_", " ")}
                         for _cell in _record["failed_coverage_cells"]], selection=None, pagination=False),
            mo.md("**Recovery: compare with always reporting the prior mean.** The drift figure follows "
                  "estimation error as more days become available. The table gives all parameters at "
                  "five years. Values below one mean lower mean squared error than that simple baseline; "
                  "a value near one means little improvement. Each model uses its own prior and data."),
            mo.image(aggregate_images["drift"]),
            mo.ui.table([{"parameter": parameter_labels.get(_name, _name), "MSE / prior-mean baseline MSE at 1,260 days": float(_ratio),
                          "estimation-error screen": _record["five_year_risk_labels"][_name].replace("_", " ")}
                         for _name, _ratio in _record["five_year_mse_to_prior_mean_ratio"].items()], selection=None, pagination=False),
            mo.accordion({"All interval-coverage cells and screening details": mo.vstack([
                mo.image(aggregate_images["coverage"]),
                mo.md(f"{_execution_passes} estimators passed execution checks and {_risk_passes} passed "
                      "their estimation-error screen. Black heatmap outlines mark failed final coverage checks. "
                      "Heston drift uses an undercoverage screen only: central intervals for a posterior with atoms can legitimately overcover. "
                      "Even a successful marginal check would not establish accuracy for every particular "
                      "history or correct joint parameter dependence."),
            ])}),
            mo.md("**Interpret the evidence:** name a parameter whose estimation error improves, "
                  "then find a failed coverage cell. Explain how both results can be true. "
                  "A fixed training bank and a fast network make these checks affordable; "
                  "they do not make the checks optional."),
        ])
    mo.vstack(_contents)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ## 6 · What can we do with a parameter posterior?

        **An uncertainty distribution over dynamics is useful; a conditional
        forecast needs the current hidden process too.** The paired draws let
        us inspect parameter trade-offs and propagate uncertainty through
        functions of the parameters. We can also draw parameters, start an
        independent process at the declared origin, and generate a replicated
        history for a posterior-predictive model check.

        That independent replicate is **not the next part of the displayed
        history**. Forecasting its continuation requires the hidden state
        conditional on what we observed; rough Bergomi additionally needs its
        past-shock memory. This notebook estimates fixed parameters and does
        not provide that state/memory posterior. It therefore shows no claimed
        conditional future-return, VaR or expected-shortfall forecast.

        **Before leaving:** identify the observations, one hidden quantity,
        one inferred parameter, how masking handles a shorter history, and a
        check the current estimator failed.
        """),
        mo.accordion({"Replicated histories and continuations are different integrals": mo.md(r"""
        An independent origin-starting replicate averages
        $p(r^{\rm rep}_{1:T}\mid\Theta)$ over $p(\Theta\mid r_{1:T})$.
        A history-conditional continuation instead needs

        $$p(r_{T+1:T+h}\mid r_{1:T})=
          \int p(r_{T+1:T+h}\mid z_T,\Theta)\,
          p(z_T,\Theta\mid r_{1:T})\,dz_T\,d\Theta,$$

        where $z_T$ must include whatever state or memory makes that conditional
        prediction valid. The parameter-only network does not supply this joint
        state–parameter distribution. The integrals describe possible checks
        and extensions; no replicated-history computation is claimed here.
        """)}),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Optional exact-length CPU inference

    Download the matching native model separately and place its unchanged
    `task.json`, `training.json` and `model.keras` under
    `tutorial/assets/local-checkpoints/path-heston/`, `path-rough-known/` or
    `path-rough-joint/`. An empty field selects the current model's default folder.
    The README describes the asset package. No bank is needed for inference.

    Press the button to load that model on the CPU and draw a fresh approximate posterior
    using **only the returns revealed by the history control**. Missing models
    leave the cached workflow available.

    **The inference call is short.** With the matching model downloaded, these
    are the same load/condition/sample steps used by the button. This displayed
    code does not execute until you explicitly request inference.

    ```python
    task, workflow, record = trajectory.load_live(model_choice.value)
    revealed_returns = cache_arrays["observations"][case_choice.value:case_choice.value + 1,
                                                    :observed_days.value]
    draws = trajectory.sample(workflow, task, revealed_returns, draws=512, seed=20261031)[0]
    ```

    The loader selects the matching model and checks its identity. Only the
    revealed prefix enters the estimator; generating labels and future returns
    stay excluded. “Exact length” describes the input, not exact posterior accuracy.

    Run this notebook in marimo for reactive controls. An exported HTML file is
    a static preview of its saved state; it does not run model switches or CPU inference.
    """)
    return


@app.cell
def _(mo):
    checkpoint_path = mo.ui.text(value="", label="Checkpoint folder relative to tutorial/assets/ (optional)", full_width=True)
    live_button = mo.ui.run_button(label="Run CPU inference at the exact observed history length")
    mo.vstack([checkpoint_path, live_button])
    return checkpoint_path, live_button


@app.cell
def _(
    cache_arrays,
    case_choice,
    checkpoint_path,
    live_button,
    mo,
    model_choice,
    observed_days,
    trajectory,
):
    mo.stop(not live_button.value, mo.md("CPU inference is idle until the button is pressed."))
    from time import perf_counter as _perf_counter
    _load_started = _perf_counter()
    try:
        _task, _workflow, _record = trajectory.load_live(model_choice.value, checkpoint_path.value.strip() or None)
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as _error:
        mo.stop(True, mo.callout(mo.md(str(_error)), kind="warn"))
    _load_seconds = _perf_counter() - _load_started
    _observed = cache_arrays["observations"][case_choice.value:case_choice.value + 1, :observed_days.value]
    _draw_started = _perf_counter()
    _fresh = trajectory.sample(_workflow, _task, _observed, 512, seed=20261031)[0]
    _draw_seconds = _perf_counter() - _draw_started
    mo.vstack([
        mo.callout(mo.md(f"**Fresh CPU approximate posterior at exactly {observed_days.value} observed days.** "
                         "These new draws can differ from the finite cached sample. "
                         "This completion message reports execution, not a calibration verdict."), kind="neutral"),
        mo.md(f"This click: model verification/load **{_load_seconds:.3f} s**; "
              f"512 posterior draws **{_draw_seconds:.3f} s**. The draw time can include "
              "first-call compilation. These measured elapsed times are not a performance guarantee."),
        mo.ui.table(trajectory.posterior_rows(_task, _fresh), selection=None, pagination=False),
    ])
    return


if __name__ == "__main__":
    app.run()
