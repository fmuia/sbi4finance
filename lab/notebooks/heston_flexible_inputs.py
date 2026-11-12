import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Heston: learned summaries and flexible quotes")


@app.cell(hide_code=True)
def _():
    import hashlib
    import json
    import os
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from sbi4f_tutorial.heston_options.flexible_quotes import jitter_geometry, make_masks

    LAB = mo.notebook_dir().parent
    if str(LAB) not in sys.path:
        sys.path.insert(0, str(LAB))
    GRID = LAB / "evidence/pilot-results/2026-11-09/heston-spread-baseline/revised-grid/cv-preparation"
    DEFAULT_STUDY = LAB / "evidence/pilot-results/2026-11-12/heston-flexible-inputs"
    STUDY_ROOT = Path(os.environ.get("HESTON_EXPERIMENT_ROOT", str(DEFAULT_STUDY))).expanduser().resolve()
    BUNDLE = os.environ.get("HESTON_BUNDLE") or None
    PARAMETER_LABELS = ["initial variance v₀", "mean reversion κ", "long-run variance v∞",
                        "volatility of variance σᵥ", "correlation ρ", "quote-noise multiplier c"]

    def read_record(path):
        """Read compact evidence without importing or executing a model."""
        path = Path(path)
        if not path.is_file():
            return {}, f"{path.name} is not available"
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError("expected a JSON object")
            return value, None
        except (OSError, ValueError) as error:
            return {}, f"{path.name}: {error}"

    def run_directory(entry, parent):
        value = entry.get("directory", entry.get("run_dir", entry.get("path")))
        if value is None:
            return None
        path = Path(value).expanduser()
        return (path if path.is_absolute() else parent / path).resolve()

    def metric_rows(evaluation):
        """Accept explicit per-parameter results; do not manufacture absent diagnostics."""
        candidates = [evaluation.get("parameters"), evaluation.get("population", {}).get("parameters"),
                      evaluation.get("historical", {}).get("parameters")]
        metrics = next((item for item in candidates if isinstance(item, dict)), {})
        rows = []
        for name, values in metrics.items():
            if isinstance(values, dict):
                row = {"parameter": name}
                for key, value in values.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        row[key] = round(float(value), 5)
                if len(row) > 1:
                    rows.append(row)
        return rows

    def loss_series(history, key):
        source = history.get("history", history)
        if isinstance(source, dict):
            values = source.get(key, [])
        elif isinstance(source, list):
            values = [row.get(key) for row in source if isinstance(row, dict)]
        else:
            values = []
        if not isinstance(values, list):
            return np.asarray([])
        return np.asarray([float(value) if value is not None else np.nan for value in values], dtype=float)

    return (BUNDLE, GRID, LAB, PARAMETER_LABELS, STUDY_ROOT, Path, hashlib, json,
            jitter_geometry, loss_series, make_masks, metric_rows, mo, np, plt, read_record, run_directory)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Heston: learned summaries and flexible quotes

    Can a network compress an option surface without losing the posterior? What changes
    when some contracts disappear, or when the quoted strikes and expiries move?

    This experimental companion to **Heston SPX calibration** keeps the same joint target
    $(v_0,\kappa,v_\infty,\sigma_v,\rho,c)$ and observation law
    $Y_i=P_i(\boldsymbol\theta)+c\,h_i\,\epsilon_i$, with independent standard-normal errors.
    The spread $h_i$ is observed; $c$ is inferred.

    | Investigation | What changes | What we want to learn |
    |:---|:---|:---|
    | Learned summaries | Representation of the complete fixed grid | How small a learned summary can preserve useful posterior accuracy? |
    | Missing quotes | Which contracts are observed | Which parameter combinations lose information, and can a trained estimator track that loss? |
    | Moving grids | Observed strikes and maturities | Can one estimator condition on a supported range of contract locations? |

    **Development results:** the fixed-grid baseline already has conditional-reference
    errors. A summary matching that baseline is a useful comparison, not proof of an
    accurate posterior or a sufficient statistic. Six unknowns do not imply that six
    summary numbers suffice.
    """)
    return


@app.cell(hide_code=True)
def _(GRID, np, read_record):
    task, task_error = read_record(GRID / "task.json")
    conditional_panel, panel_error = read_record(GRID / "conditional-panel.json")
    geometry = task.get("records", {})
    parameter_names = task.get("target", {}).get("parameter_names", [])
    prior_bounds = np.asarray(task.get("target", {}).get("prior_bounds", []), dtype=float)
    full_cases = {case["id"]: np.asarray(case["observations"], dtype=float)
                  for case in conditional_panel.get("cases", [])}
    return (conditional_panel, full_cases, geometry, panel_error, parameter_names,
            prior_bounds, task, task_error)


@app.cell(hide_code=True)
def _(STUDY_ROOT, read_record, run_directory):
    study, study_error = read_record(STUDY_ROOT / "study.json")
    _entries = study.get("runs", [])
    if isinstance(_entries, dict):
        _entries = [dict(value, label=key) for key, value in _entries.items() if isinstance(value, dict)]
    run_records = []
    for _index, _entry in enumerate(_entries):
        if not isinstance(_entry, dict):
            continue
        _directory = run_directory(_entry, STUDY_ROOT)
        _metadata, _run_error = read_record(_directory / "run.json") if _directory else ({}, "run directory is not declared")
        _history, _history_error = read_record(_directory / "history.json") if _directory else ({}, "history unavailable")
        _evaluation, _evaluation_error = read_record(_directory / "evaluation.json") if _directory else ({}, "evaluation unavailable")
        run_records.append({"key": str(_index), "label": _entry.get("label", _entry.get("id", f"run {_index + 1}")),
                            "directory": _directory, "entry": _entry, "run": _metadata,
                            "history": _history, "evaluation": _evaluation,
                            "errors": [error for error in (_run_error, _history_error, _evaluation_error) if error]})
    return run_records, study, study_error


@app.cell(hide_code=True)
def _(BUNDLE, STUDY_ROOT, mo, run_records, study_error):
    _summary = [mo.md("## 1. Compare the learned summaries")]
    if study_error:
        _summary.append(mo.callout(mo.md(
            "**No experiment results are loaded.** The design exercises below still work. "
            "To compare trained estimators, set `HESTON_EXPERIMENT_ROOT` to the directory containing "
            "`study.json`, then restart this notebook. Missing results are left unmeasured."), kind="info"))
    else:
        _summary.append(mo.md(f"Loaded **{len(run_records)} declared runs** from `{STUDY_ROOT}`."))
    if not BUNDLE:
        _summary.append(mo.md(
            "`HESTON_BUNDLE` is not set. Saved experiment evidence can still be compared; "
            "the original training bank is needed when training new runs. See `lab/plans/training-data-guide.md`."))
    _summary.append(mo.md(r"""
    We keep the physical prior, quote-noise model, training rows and evaluation cases
    fixed while changing the representation. The inference network receives either
    the complete input, or a learned summary of dimension $d$.

    Compare calibration and reference agreement as well as loss and training cost.
    A prior-like posterior can pass population calibration while learning little about
    these particular quotes. The smallest **tested** useful dimension is an empirical
    result for this task and training recipe.
    """))
    mo.vstack(_summary)
    return


@app.cell(hide_code=True)
def _(mo, run_records):
    _options = {f"{record['label']} [{record['key']}]": record["key"] for record in run_records}
    selected_runs = mo.ui.multiselect(options=_options, value=list(_options)[:5], label="Compare saved runs")
    selected_runs
    return (selected_runs,)


@app.cell(hide_code=True)
def _(mo, run_records, selected_runs):
    compared_runs = [record for record in run_records if record["key"] in selected_runs.value]
    _rows = []
    _smoke_labels = []
    for _record in compared_runs:
        _run, _entry = _record["run"], _record["entry"]
        _config = _run.get("config", _run.get("configuration", {}))
        _smoke = _run.get("smoke_test_only", _entry.get("smoke_test_only"))
        if _smoke is True:
            _smoke_labels.append(_record["label"])
        _rows.append({"run": _record["label"], "status": _run.get("status", _entry.get("status", "unmeasured")),
                      "smoke_test_only": _smoke if _smoke is not None else "unrecorded",
                      "representation": _config.get("representation", _entry.get("representation", "unrecorded")),
                      "summary dimension": _config.get("summary_dim", _entry.get("summary_dim", "—")),
                      "training rows": _run.get("train_rows", _config.get("train_rows", "unrecorded")),
                      "completed epochs": _run.get("completed_epochs", _entry.get("completed_epochs", "unrecorded")),
                      "requested epochs": _run.get("epochs", _config.get("epochs", "unrecorded")),
                      "training seconds": _run.get("elapsed_seconds", _run.get("training_seconds", "unrecorded")),
                      "evidence": "; ".join(_record["errors"]) or "run, history and evaluation present"})
    _parts = [mo.ui.table(_rows, selection=None, pagination=False)] if _rows else [mo.md("*No runs selected.*")]
    if _smoke_labels:
        _parts.append(mo.callout(mo.md(
            "**Smoke tests are selected:** " + ", ".join(_smoke_labels) + ". "
            "These tiny runs check software execution, saving and evaluation. Their learning curves "
            "and posterior diagnostics do not establish scientific accuracy or a useful summary dimension."), kind="warn"))
    _view = mo.vstack(_parts)
    _view
    return (compared_runs,)


@app.cell(hide_code=True)
def _(compared_runs, loss_series, mo, np, plt):
    _fig, _ax = plt.subplots(figsize=(8.1, 3.6), constrained_layout=True)
    _shown = False
    for _record in compared_runs:
        for _key, _style in (("loss", "-"), ("val_loss", "--")):
            _values = loss_series(_record["history"], _key)
            if len(_values):
                _ax.plot(np.arange(1, len(_values) + 1), _values, _style,
                         label=f"{_record['label']}: {_key}", linewidth=1.2)
                _shown = True
    _ax.set(xlabel="epoch", ylabel="native training objective", title="Training and validation")
    if _shown:
        _ax.legend(fontsize=8)
        _view = _fig
    else:
        plt.close(_fig)
        _view = mo.md("*Learning curves will appear when a selected run has saved history.*")
    _view
    return


@app.cell(hide_code=True)
def _(compared_runs, metric_rows, mo):
    _rows = []
    _conditional_rows = []
    for _record in compared_runs:
        for _metric in metric_rows(_record["evaluation"]):
            _rows.append({"run": _record["label"], **_metric})
        for _case, _result in _record["evaluation"].get("cases", {}).items():
            if not isinstance(_result, dict):
                continue
            for _name, _values in _result.get("parameters", {}).items():
                if not isinstance(_values, dict):
                    continue
                _conditional_rows.append({
                    "run": _record["label"], "case": _case, "parameter": _name,
                    "mean error / reference SD": _values.get("mean_error_in_reference_sd", "unmeasured"),
                    "90% width / reference width": _values.get("width90_ratio", "unmeasured"),
                })
    mo.vstack([
        mo.md("### Recovery and calibration"),
        mo.ui.table(_rows, selection=None, pagination=True, page_size=12) if _rows else
        mo.md("*Per-parameter recovery and calibration are not yet available for the selected runs.*"),
        mo.md("Nominal 90% intervals should be checked across new simulations, with sampling uncertainty. "
              "Conditional MCMC comparisons ask the separate question of whether the learned posterior "
              "matches the posterior for these particular quotes."),
        mo.md("### Agreement with conditional references"),
        mo.ui.table(_conditional_rows, selection=None, pagination=True, page_size=12) if _conditional_rows else
        mo.md("*Conditional-reference comparisons have not yet been recorded for the selected runs.*"),
    ])
    return


@app.cell(hide_code=True)
def _(full_cases, mo, run_records):
    _options = {f"{record['label']} [{record['key']}]": record["key"] for record in run_records
                if record["directory"] is not None and (record["directory"] / "model.keras").is_file()}
    posterior_run = mo.ui.dropdown(options=_options or {"No saved model available": None},
                                   value=next(iter(_options)) if _options else "No saved model available",
                                   label="Pretrained experimental model")
    posterior_case = mo.ui.dropdown(options={key: key for key in full_cases} or {"No cases available": None},
                                    value=next(iter(full_cases)) if full_cases else "No cases available",
                                    label="Original complete-grid quotes")
    posterior_button = mo.ui.run_button(label="Load the selected model and draw its posterior")
    mo.vstack([mo.md("### Try a saved estimator on the original complete grid"),
               mo.hstack([posterior_run, posterior_case], justify="start"), posterior_button,
               mo.md("This button uses only the original complete 63-contract cases. "
                     "The missing-quote and moving-grid controls below are separate design previews.")])
    return posterior_button, posterior_case, posterior_run


@app.cell(hide_code=True)
def _(LAB, PARAMETER_LABELS, full_cases, hashlib, json, mo, np, parameter_names,
      posterior_button, posterior_case, posterior_run, prior_bounds, plt, run_records):
    mo.stop(not posterior_button.value, mo.md("*Select a pretrained run and press the button to sample it.*"))
    mo.stop(posterior_run.value is None or posterior_case.value is None,
            mo.md("*A saved model and a declared complete-grid case are needed.*"))
    _chosen = next(record for record in run_records if record["key"] == posterior_run.value)
    try:
        from scripts.heston_flexible_experiment import sample_run

        _samples = sample_run(_chosen["directory"], full_cases[posterior_case.value][None, :],
                              num_samples=1024, seed=261112)
        if isinstance(_samples, dict):
            if "draws" in _samples:
                _samples = _samples["draws"]
            else:
                _samples = np.stack([np.asarray(_samples[name]) for name in parameter_names], axis=-1)
        _samples = np.asarray(_samples, dtype=float).reshape(-1, len(parameter_names))
        if not np.isfinite(_samples).all():
            raise ValueError("the model returned nonfinite posterior draws")
        _reference = None
        _references = LAB / "evidence/pilot-results/2026-11-11/heston-actual63-notebook"
        _ref_info = _references / "reference-comparison.json"
        _ref_payload = _references / "reference-display-draws.npz"
        if _ref_info.is_file() and _ref_payload.is_file():
            _expected = json.loads(_ref_info.read_text())["display_cache"]["sha256"]
            if hashlib.sha256(_ref_payload.read_bytes()).hexdigest() != _expected:
                raise ValueError("reference display draws failed their recorded checksum")
            with np.load(_ref_payload, allow_pickle=False) as _archive:
                _reference = _archive[f"{posterior_case.value}__parameters"].reshape(-1, len(parameter_names))
        _fig, _axes = plt.subplots(2, 3, figsize=(9, 5.3), constrained_layout=True)
        for _index, _ax in enumerate(_axes.flat):
            _ax.hist(_samples[:, _index], bins=35, density=True, alpha=.6, label="experimental network")
            if _reference is not None:
                _ax.hist(_reference[:, _index], bins=35, density=True, histtype="step",
                         linewidth=1.5, label="PyMC reference")
            _ax.set(xlabel=PARAMETER_LABELS[_index], ylabel="density")
        _axes.flat[0].legend(fontsize=8)
        _outside = np.mean(np.any((_samples < prior_bounds[:, 0]) | (_samples > prior_bounds[:, 1]), axis=1))
        _view = mo.vstack([_fig, mo.md(
            f"**{_chosen['label']}**, {posterior_case.value}; 1,024 draws. "
            f"Outside the declared prior support: **{100 * _outside:.2f}%**. "
            "These are development diagnostics; no out-of-support draws have been hidden." +
            ("" if _reference is not None else " Reference display draws are unavailable."))])
    except (ImportError, OSError, ValueError, KeyError, RuntimeError) as _error:
        _view = mo.callout(mo.md(f"**The selected run could not be sampled:** {_error}. "
                                "Its saved results remain visible above."), kind="warn")
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Which quotes carry the information?

    Removing prices changes the information in the likelihood. Removing a distant
    expiry might matter more than removing the same number of nearby quotes.
    We want a trained estimator to reproduce that change, rather than remain
    artificially certain.

    The design controls here show **which contracts would remain**. They do not
    compute a sparse-data posterior. A missing-quote model must be trained for these
    patterns, with references using the same surviving quotes. The complete-grid
    whitening transform must not mix withheld prices into retained features.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mask_choice = mo.ui.dropdown(options={"Keep every quote": "full", "Remove one expiry": "expiry",
                                           "Remove the outer put wing": "put_wing", "Randomly retain 32": "random"},
                                value="Keep every quote", label="Illustrative missingness")
    preview_seed = mo.ui.number(start=0, stop=99999, step=1, value=42, label="Preview seed")
    mo.hstack([mask_choice, preview_seed], justify="start")
    return mask_choice, preview_seed


@app.cell(hide_code=True)
def _(geometry, make_masks, mask_choice, mo, np, preview_seed, plt, task_error):
    mo.stop(bool(task_error), mo.callout(mo.md(f"The committed grid metadata are unavailable: {task_error}"), kind="warn"))
    maturities = np.asarray(geometry["maturity"], dtype=float)
    log_moneyness = np.log(np.asarray(geometry["strike"]) / np.asarray(geometry["forward"]))
    observed = make_masks(geometry, 1, mode=mask_choice.value, seed=int(preview_seed.value), retained_counts=(32,))[0]
    _fig, _ax = plt.subplots(figsize=(8, 3.7), constrained_layout=True)
    _ax.scatter(365 * maturities[~observed], log_moneyness[~observed], marker="x", color="#a8a8a8", label="withheld")
    _ax.scatter(365 * maturities[observed], log_moneyness[observed], color="#296baf", label="observed")
    _ax.set(xlabel="time to expiry (days)", ylabel="log(K / F)", title="Availability on the original quote grid")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(f"**{observed.sum()} of {len(observed)} contracts remain.** "
                          "Compare different patterns at the same quote count before attributing a change to count alone."),
               mo.callout(mo.md("**Design preview only.** No masked posterior is computed by this exercise."), kind="info")])
    return log_moneyness, maturities, observed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. What if the market quotes a different grid?

    Here “fuzzy” means **varying, observed contract locations**. The network is told
    the true strike, maturity, spread and option type of each available quote; those
    coordinates are not latent measurement errors.

    A geometry-aware summary can aggregate quote records without tying their meaning
    to a particular position in a vector. That gives us a route to changing grids,
    but accuracy still has to be checked within its declared training range.

    Moving a contract requires a new model price. Relabeling the existing quote as
    a different strike or expiry would create an inconsistent training example.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    strike_jitter = mo.ui.slider(start=0.0, stop=0.04, step=0.002, value=0.01,
                                 label="Maximum change in log-moneyness", show_value=True)
    expiry_jitter = mo.ui.slider(start=0.0, stop=0.10, step=0.01, value=0.03,
                                 label="Maximum relative expiry change", show_value=True)
    mo.vstack([strike_jitter, expiry_jitter])
    return expiry_jitter, strike_jitter


@app.cell(hide_code=True)
def _(expiry_jitter, geometry, jitter_geometry, log_moneyness, maturities, mo, np,
      observed, preview_seed, plt, strike_jitter):
    _moved = jitter_geometry(geometry, 1, seed=int(preview_seed.value) + 1,
                             strike_log_width=float(strike_jitter.value),
                             maturity_relative_width=float(expiry_jitter.value))
    _moved_times = np.asarray(_moved["maturity"])[0]
    _moved_logk = np.log(np.asarray(_moved["strike"])[0] / np.asarray(_moved["forward"])[0])
    _fig, _ax = plt.subplots(figsize=(8, 3.7), constrained_layout=True)
    _ax.scatter(365 * maturities[observed], log_moneyness[observed], facecolors="none",
                edgecolors="#727272", s=40, label="original locations")
    _ax.scatter(365 * _moved_times[observed], _moved_logk[observed],
                color="#9467bd", s=20, label="illustrative moved locations")
    _ax.set(xlabel="time to expiry (days)", ylabel="log(K / F)", title="A synthetic grid proposal")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(
        "Each expiry moves as a block within the carry-curve support. This synthetic geometry proposal "
        "interpolates forward/discount curves and preserves each slot's normalized spread. "
        "The resulting spreads are engineering assumptions, not new market measurements. No quote price is transferred."),
        mo.callout(mo.md("**No moved-grid posterior is available here.** New contracts must be priced, "
                          "the carry/spread policy declared, and a matching geometry-conditioned model trained and checked."), kind="info")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Questions to investigate

    1. At what summary dimension do recovery and conditional-reference agreement begin
       to change? Is the result reproducible with another seed?
    2. At a fixed number of observations, do losing distant expiries and losing wing
       quotes affect the same parameter combinations? Inspect posterior correlations
       and forward-variance predictions as well as individual intervals.
    3. How far can contract locations move before a trained geometry-aware estimator
       loses calibration? Distinguish interpolation within training support from an
       extrapolation stress test.

    Wider intervals are not guaranteed for every realized subset. Random thinning
    also does not represent every market selection mechanism: liquidity and quote
    filters may depend on prices and need their own assumptions.

    The initial summary comparison can reuse the existing Heston bank. Missingness
    can be applied to those records before preprocessing. Moving contracts needs
    repricing, with fresh validation parameters and a checked numerical budget.
    """)
    return


if __name__ == "__main__":
    app.run()
