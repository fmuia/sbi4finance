import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Review the current inference backbones")


@app.cell
def _():
    import os
    from pathlib import Path
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from sbi4finance.calibration.full_surface_teaching import load_tutorial_artifact, load_tutorial_checkpoint

    MANIFEST = Path(os.environ.get("SBI4F_BACKBONE_MANIFEST", mo.notebook_dir().parents[1]
                                  / "data/synthetic/full-surface-backbones/artifacts.json"))
    return MANIFEST, load_tutorial_artifact, load_tutorial_checkpoint, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Review the current inference backbones

    Start with the **cached** role: the exact observation, joint parameter draws,
    all marginal distributions, and their provenance. Opening this notebook does
    not load a model, simulate prices, train a network, or run MCMC.

    Heston uses the checked interior development example and its PyMC reference.
    The two rough Bergomi examples are fixed **validation:0 software replays**,
    with no claim that they are interior or independently checked reference
    posteriors. A prior boundary in these cases remains visible.

    These synthetic observations are separate from the historical SPY examples
    in slots 4 and 5. Development results remain labelled as development.
    """)
    return


@app.cell
def _(mo):
    example = mo.ui.dropdown(options=["heston-final500", "rough-known50k", "rough-joint20k"],
                             value="heston-final500", label="Cached example")
    example
    return (example,)


@app.cell
def _(MANIFEST, example, load_tutorial_artifact):
    cache_task, cache_report, cache_arrays = load_tutorial_artifact(MANIFEST, example.value)
    return cache_arrays, cache_report, cache_task


@app.cell(hide_code=True)
def _(cache_arrays, cache_report, cache_task, mo, np):
    _draws = cache_arrays["neural_draws"]
    _bounds = np.asarray(cache_task["prior_bounds"])
    _q = np.quantile(_draws, [.005, .5, .995], axis=0)
    _rows = [{"parameter": name, "prior": f"{_bounds[i, 0]:.4g} to {_bounds[i, 1]:.4g}",
              "median": float(_q[1, i]), "central 99%": f"{_q[0, i]:.4g} to {_q[2, i]:.4g}",
              "near prior edge": bool(min(_q[0, i] - _bounds[i, 0], _bounds[i, 1] - _q[2, i])
                                       < .02 * (_bounds[i, 1] - _bounds[i, 0]))}
             for i, name in enumerate(cache_task["parameter_names"])]
    mo.vstack([
        mo.callout(mo.md(f"**Cached development data:** `{cache_report['case_id']}`. "
                         f"{len(_draws):,} joint neural draws; "
                         f"{'PyMC reference available' if 'reference_chains' in cache_arrays else 'no conditional reference yet'}. "
                         f"{cache_report['limitations']}"), kind="neutral"),
        mo.md(f"Model SHA-256: `{cache_report['checkpoint_sha256']}`  \n"
              f"Task SHA-256: `{cache_report['task_fingerprint']}`"),
        mo.ui.table(_rows, selection=None, pagination=False),
    ])
    return


@app.cell
def _(cache_arrays, cache_task, np, plt):
    _sets = {"neural": cache_arrays["neural_draws"]}
    if "reference_chains" in cache_arrays:
        _sets["PyMC reference"] = cache_arrays["reference_chains"].reshape(-1, len(cache_task["parameter_names"]))
    _names = cache_task["parameter_names"]; _n = len(_names)
    _fig, _axes = plt.subplots(_n, _n, figsize=(max(7, 1.9 * _n), max(7, 1.9 * _n)), squeeze=False)
    for _i in range(_n):
        for _j in range(_n):
            _ax = _axes[_i, _j]
            if _j > _i:
                _ax.set_visible(False)
                continue
            for _label, _draws in _sets.items():
                if _i == _j:
                    _ax.hist(_draws[:, _i], bins=35, density=True, histtype="step", label=_label)
                else:
                    _ids = np.linspace(0, len(_draws) - 1, min(1000, len(_draws)), dtype=int)
                    _ax.scatter(_draws[_ids, _j], _draws[_ids, _i], s=2, alpha=.18, rasterized=True)
            if _i == _j:
                _ax.axvline(cache_arrays["truth"][_i], color="black", linestyle=":", linewidth=1)
            else:
                _ax.plot(cache_arrays["truth"][_j], cache_arrays["truth"][_i], "k+", markersize=6)
            if _i == _n - 1: _ax.set_xlabel(_names[_j], fontsize=8)
            if _j == 0: _ax.set_ylabel(_names[_i], fontsize=8)
            _ax.tick_params(labelsize=7)
    _axes[0, 0].legend(fontsize=7)
    _fig.suptitle("All inferred parameters; dotted lines / crosses show synthetic truth", fontsize=11)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(cache_arrays, cache_task, mo):
    if "context_values" in cache_arrays:
        _knots = cache_task["model_context"]["forward_variance_knots"][1:]
        _context = mo.vstack([mo.md("**Known curve:** these seven observed levels accompany the 63 quotes. "
                                    "They are global inference conditions, not parameters being inferred."),
            mo.ui.table([{"maturity (years)": t, "forward variance": float(v)}
                         for t, v in zip(_knots, cache_arrays["context_values"])], selection=None, pagination=False)])
    elif cache_task["model"] == "rough":
        _context = mo.md("**Joint curve:** only the 63 quotes, design and known noise enter inference. "
                         "The generating curve is excluded; its three coefficients are among the six targets.")
    else:
        _context = mo.md("**Heston:** the variance curve follows from the inferred initial and long-run variances "
                         "and fixed mean-reversion rate; no external variance curve is supplied.")
    mo.vstack([_context, mo.md("The inference observation is the complete ordered 63-price surface. "
                              "Synthetic truth and reference draws are for checking the result only.")])
    return


@app.cell(hide_code=True)
def _(cache_arrays, mo, np):
    _rows = []
    for _name in ("variance_swap_1y", "forward_variance_0_5_to_1y"):
        for _prefix in ("neural_", "reference_"):
            if _prefix + _name in cache_arrays:
                _values = cache_arrays[_prefix + _name]
                _q = np.quantile(_values, [.1, .9])
                _rows.append({"annualized variance rate": _name, "posterior": _prefix.rstrip("_"),
                              "mean": float(_values.mean()), "q10": float(_q[0]), "q90": float(_q[1])})
    mo.vstack([mo.md("### Analytic variance products"), mo.ui.table(_rows, selection=None, pagination=False)]) if _rows else mo.md(
        "**Rough products pending:** this replay contains parameters, not validated product-price distributions. "
        "Independent pricing, reference and downstream-product checks remain separate work.")
    return


@app.cell(hide_code=True)
def _(cache_report, mo):
    mo.md(f"""
    ### Optional live role

    To obtain fresh neural draws for this same observation, first copy the
    matching checkpoint from `{cache_report['model_source']}` to your machine.
    Use the locked **core** environment (BayesFlow 2.1), enter its directory
    below, then press the button. The cached role needs no checkpoint or bank.

    Fresh draws can differ from the original batched cache. Editing an
    observation would require fresh inference; cached draws must stay paired
    with their original observation.
    """)
    return


@app.cell
def _(mo):
    checkpoint_path = mo.ui.text(value="", label="Local matching checkpoint directory", full_width=True)
    run_live = mo.ui.run_button(label="Load the matching model and draw 1,000 fresh samples")
    mo.vstack([checkpoint_path, run_live])
    return checkpoint_path, run_live


@app.cell
def _(MANIFEST, cache_arrays, checkpoint_path, example, load_tutorial_checkpoint, mo, np, run_live):
    mo.stop(not run_live.value, mo.md("Live model loading is off until explicitly requested."))
    mo.stop(not checkpoint_path.value.strip(), mo.md("Enter the copied checkpoint directory first."))
    _task, _workflow, _record = load_tutorial_checkpoint(MANIFEST, example.value, checkpoint_path.value.strip())
    from sbi4finance.calibration import full_surface
    _curves = cache_arrays.get("context_values")
    _fresh = full_surface.sample(_workflow, _task, cache_arrays["observations"][None], 1000,
        encoder=_record["network"]["encoder"], context_values=None if _curves is None else _curves[None], seed=0)[0]
    mo.ui.table([{"parameter": name, "fresh median": float(np.median(_fresh[:, i]))}
                 for i, name in enumerate(_task["parameter_names"])], selection=None, pagination=False)
    return


if __name__ == "__main__":
    app.run()
