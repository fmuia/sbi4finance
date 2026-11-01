import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Learn volatility dynamics from observed paths")


@app.cell
def _():
    import json
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from sbi4f_tutorial import trajectory
    return json, mo, np, plt, trajectory


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Learn volatility dynamics from observed paths

    Observe a return history, summarize its temporal structure, and obtain a
    **joint approximate posterior over fixed parameters**. Heston and both rough Bergomi
    tasks use the same daily observation grid: up to **1,260 trading days
    (five years)**. Volatility stays hidden from the estimator.

    A causal GRU summarizes the ordered returns; a conditional density network
    supplies parameter draws. The input interface accepts every integer history
    length; approximation quality can vary with the length and observed path.
    This notebook opens with cached examples. Moving the history control reveals
    observations and the available posterior snapshots; the optional CPU button
    requests fresh inference at the exact chosen length.

    Each model has its own synthetic validation histories. Switching the model
    changes the displayed history as well as the inference task. Observation
    frequency and maximum duration are held common across these examples.

    **These are candidate neural estimators, without a conditional-accuracy
    certificate.** The examples illustrate inference and uncertainty. Calibration
    and reference checks have their own recorded outcomes; fresh samples do not
    make an inaccurate approximation exact.

    The known-curve rough model is the opening illustration. Heston remains
    available as a numerical-development prototype with a model-specific warning.

    These are synthetic physical-measure experiments. Their parameters describe
    the specified return process; option-pricing parameter interpretations require
    a separately specified pricing measure.
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
    _curve = ("The initial variance is supplied; the realized subsequent variance remains hidden."
              if cache_task["model"] == "heston" else
              "The deterministic variance-scale curve is the supplied constant level shown below; the realized variance remains hidden."
              if cache_task["model"] == "rough-known" else
              r"The curve is inferred: $\xi(t)=L+(S-L)e^{-kt}$. Its coefficients are targets, not observed inputs.")
    mo.vstack([
        mo.md(r"""
        ## 1 · Specify the inference task

        **Observed:** ordered daily log returns, one channel per day, with the known time design.
        **Hidden:** the stochastic variance path and its driving shocks. The synthetic
        parameter labels shown later are evaluation aids, not inference conditions.

        Under the declared physical measure P,
        $$d\log S_t=(\mu-V_t/2)dt+\sqrt{V_t}\,dW_t^S,
        \qquad d\langle W^S,W^V\rangle_t=\rho\,dt.$$
        """),
        mo.md(_equation + "\n\n" + _curve),
        mo.ui.table(_priors, selection=None, pagination=False),
        mo.md("These declared parameter priors are independent. Supplied quantities are fixed modelling assumptions, not extra measurements of latent volatility."),
        mo.ui.table(_fixed, selection=None, pagination=False) if _fixed else mo.md("**No fixed model parameters** in this joint-curve task."),
        mo.md(f"**Numerical target:** `{cache_task['numerics']['scheme']}`, **{_steps} internal steps per observed day**. "
              f"The estimator accepts **{cache_task['n_observations']:,} daily returns at most**, "
              f"with dt = **{cache_task['dt_years']:.6g} years**. "
              "These equations explain the model; the task's finite-grid simulator defines the experiment."),
        mo.accordion({"Declared numerical-law settings": mo.ui.table(
            [{"setting": _name.replace("_", " "), "declared value": _value}
             for _name, _value in cache_task["numerics"].items()], selection=None, pagination=False)}),
    ])
    return (parameter_labels,)


@app.cell(hide_code=True)
def _(cache_task, mo):
    _targets = ", ".join(name.replace("_", " ") for name in cache_task["parameter_names"])
    mo.md(rf"""
    **How the amortization is learned**

    $\theta\sim p(\theta)$ → simulator → ordered daily returns → random prefix and mask
    → GRU summary → conditional flow $q_\phi(\theta\mid r_{{1:T}},T)$.

    Offline training uses **one million independent simulated histories per model**,
    with random prefix lengths from 1 to 1,260. The summary and flow learn jointly
    to assign high density to the generating parameters: **{_targets}**. These
    parameter labels, including drift, are training targets only. The estimator
    receives the revealed returns, their mask and known time design. Fixed return
    scaling preserves variance amplitude; each path is not divided by its own
    observed volatility.

    At each new prefix, the same trained estimator approximates the posterior
    under the **original prior and the full revealed history**. Do not multiply
    posteriors from overlapping prefixes or feed the previous posterior back into
    the unchanged estimator: that would reuse the same observations.
    """)
    return


@app.cell(hide_code=True)
def _(cache_task, mo):
    _warning = (mo.callout(mo.md(
        "**Heston: numerical-development prototype.** The current discretized simulator can "
        "produce a deterministic return when its effective variance stays at zero over a day. "
        "With drift inferred, the exact posterior can then contain a point mass in drift. "
        "A continuous neural density cannot represent that mass exactly. Treat the displayed "
        "Heston intervals as development illustrations, without an endorsed calibration claim. "
        "Resolving this requires a newly specified numerical target and refit; the frozen "
        "example and model are unchanged."), kind="warn")
        if cache_task["model"] == "heston" else mo.md(""))
    _warning
    return


@app.cell
def _(cache_arrays, mo):
    case_choice = mo.ui.dropdown(
        options={f"Validation case {index}": index for index in range(len(cache_arrays["trajectory_id"]))},
        value="Validation case 0", label="Synthetic case (case 0 is the primary example)")
    case_choice
    return (case_choice,)


@app.cell
def _(cache_arrays, cache_task, case_choice, observed_days, trajectory):
    cached = trajectory.cached_posterior(cache_task, cache_arrays, case_choice.value, observed_days.value)
    return (cached,)


@app.cell(hide_code=True)
def _(cache_report, cached, mo, observed_days):
    _detail = ("The cached posterior uses all currently revealed returns." if cached["exact"] else
               "The return plot reveals the requested history. Its cached posterior uses only the "
               "labelled earlier prefix; snapshots are never interpolated. Use live inference below "
               "to condition on every revealed return.")
    mo.vstack([
        mo.callout(mo.md(f"**{cached['label']}.** {_detail}"), kind="neutral"),
        mo.md(f"Elapsed observation time: **{observed_days.value / 252:.3g} trading years**. "
              "These are development examples from the first predeclared validation cases. "
              f"{cache_report['limitations']}"),
    ])
    return


@app.cell
def _(cache_arrays, cache_task, case_choice, observed_days, trajectory):
    trajectory.plot_returns(cache_task, cache_arrays, case_choice.value, observed_days.value)
    return


@app.cell
def _(cache_arrays, cache_task, case_choice, observed_days, trajectory):
    trajectory.plot_evolution(cache_task, cache_arrays, case_choice.value, observed_days.value)
    return


@app.cell(hide_code=True)
def _(cache_task, cached, mo, trajectory):
    mo.vstack([
        mo.md("**Current cached approximate posterior.** Intervals are central 90% marginal intervals; "
              "the stored draw rows preserve the joint distribution."),
        mo.ui.table(trajectory.posterior_rows(cache_task, cached["draws"]), selection=None, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(cached, mo):
    mo.md(f"""
    ## 2 · Inspect the selected approximate posterior

    The following density and dependence views use **the same paired cached draw rows**
    at **{cached['length']} observed days**. They do not use a later prefix when
    the history control lies between cached snapshots.

    The dashed prior densities below are the actual declared marginal priors,
    not densities uniform over every parameter box. A histogram changing from its
    prior shows observation dependence; contraction alone does not establish accuracy.
    **No matched conditional-reference draws are included in this selected replay.**
    These plots contain the prior and neural candidate, without a reference overlay.
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
                   alpha=.42, color="#2563eb", label="cached neural candidate")
        _axis.plot(_grid, _density, linestyle="--", color="#64748b", linewidth=1.5,
                   label="declared prior density")
        _axis.axvline(cache_arrays["truth"][case_choice.value, _index], linestyle=":", color="#111827",
                     linewidth=1, label="synthetic truth (evaluation only)")
        _axis.set(xlim=(_low, _high), title=parameter_labels[_names[_index]], ylabel="Marginal density")
    _axes.flat[0].legend(fontsize=7)
    _figure.suptitle(f"Prior and approximate posterior · validation case {case_choice.value} · {cached['length']} days", fontsize=12)
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
              title=f"Joint candidate geometry · {cached['length']} observed days")
    _axis.legend(fontsize=8)
    mo.vstack([_figure, mo.md("Each point keeps two coordinates from the **same** posterior draw. "
                              "Independently shuffling columns would destroy dependence. "
                              "Visible trade-offs describe the learned candidate; they do not certify its joint accuracy.")])
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
    latent volatility are excluded from neural inference conditions. Cached
    snapshots do not establish conditional calibration or a correct joint copula.
    """)
    return


@app.cell(hide_code=True)
def _(cache_report, mo):
    mo.accordion({"Network, provenance and scope": mo.md(
        "The published candidates use a **64-unit causal GRU**, a **32-value summary**, "
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
def _(aggregate_images, aggregate_problem, aggregate_report, cache_task, mo):
    _contents = [mo.md("## 3 · Check beyond the displayed case")]
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
            mo.md(f"These are **{_record['cases']:,} different, previously reserved histories per model**, "
                  f"with **{_record['draws']} draws per history** at twelve frozen lengths. "
                  "They are separate from the four development cases controlled above. "
                  "A whole simulated history is the independent evaluation unit; nested prefixes are dependent."),
            mo.callout(mo.md(f"**Final outcomes across {len(_records)} frozen candidates:** "
                             f"{_execution_passes} passed execution screens; {_risk_passes} passed risk screens; "
                             f"**{_calibration_failures} failed the declared final marginal-calibration screen.** "
                             "Lower mean estimation error and a working sampler do not imply calibrated posterior intervals. "
                             "Heston's atom-bearing numerical target has an additional representation limitation."), kind="warn"),
            mo.ui.table([{"selected model": cache_task["model"], "execution screen": "passed" if _record["execution_pass"] else "failed",
                          "risk screen": "passed" if _record["risk_pass"] else "failed",
                          "final marginal-calibration screen": "passed" if _record["calibration_pass"] else "failed",
                          "90% coverage range across targets and lengths": f"{100 * _record['coverage90_range'][0]:.2f}% to {100 * _record['coverage90_range'][1]:.2f}%"}],
                        selection=None, pagination=False),
            mo.md("**Selected model: failed final coverage cells.** These retain the sealed simultaneous screen's decisions; "
                  "they are not new pointwise tests."),
            mo.ui.table([{"observed days": _cell["days"], "parameter": _cell["parameter"],
                          "empirical coverage of nominal 90% interval": f"{100 * _cell['coverage90']:.2f}%",
                          "screen criterion": _cell["criterion"].replace("_", " ")}
                         for _cell in _record["failed_coverage_cells"]], selection=None, pagination=False),
            mo.md("**Five-year parameter estimation:** mean squared error divided by this model's own prior-mean baseline error. "
                  "A ratio near one can reflect little useful recovery; a lower ratio does not establish interval calibration."),
            mo.ui.table([{"parameter": _name, "MSE / prior-mean baseline MSE at 1,260 days": float(_ratio),
                          "frozen paired-risk screen": _record["five_year_risk_labels"][_name].replace("_", " ")}
                         for _name, _ratio in _record["five_year_mse_to_prior_mean_ratio"].items()], selection=None, pagination=False),
            mo.accordion({"Common-duration drift learning and all marginal coverage cells": mo.vstack([
                mo.image(aggregate_images["drift"]), mo.image(aggregate_images["coverage"]),
                mo.md("Each model uses its own prior-mean risk baseline. Black heatmap outlines mark failed sealed coverage screens. "
                      "Heston drift uses an undercoverage screen only: central intervals for a posterior with atoms can legitimately overcover. "
                      "No conditional or joint accuracy certificate follows from these marginal results."),
            ])}),
        ])
    mo.vstack(_contents)
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
