import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Learn volatility dynamics from observed paths")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from sbi4f_tutorial import trajectory
    return mo, np, trajectory


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
    mo.accordion({"Provenance and scope": mo.md(
        f"Checkpoint SHA-256: `{cache_report['checkpoint_sha256']}`  \n"
        f"Task SHA-256: `{cache_report['task_sha256']}`  \n"
        "The cache contains validation observations and neural draws. Its file hashes, "
        "daily time design and checkpoint identities are verified on loading. "
        "Any aggregate final-test results describe a separate evaluation cohort.")})
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
    try:
        _task, _workflow, _record = trajectory.load_live(model_choice.value, checkpoint_path.value.strip() or None)
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as _error:
        mo.stop(True, mo.callout(mo.md(str(_error)), kind="warn"))
    _observed = cache_arrays["observations"][case_choice.value:case_choice.value + 1, :observed_days.value]
    _fresh = trajectory.sample(_workflow, _task, _observed, 512, seed=20261031)[0]
    mo.vstack([
        mo.callout(mo.md(f"**Fresh CPU approximate posterior at exactly {observed_days.value} observed days.** "
                         "These new draws can differ from the finite cached sample. "
                         "This completion message reports execution, not a calibration verdict."), kind="neutral"),
        mo.ui.table(trajectory.posterior_rows(_task, _fresh), selection=None, pagination=False),
    ])
    return


if __name__ == "__main__":
    app.run()
