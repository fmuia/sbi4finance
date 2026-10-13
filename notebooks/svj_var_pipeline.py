import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from sbi4finance.svj import model, plots, teaching

    ROOT = mo.notebook_dir().parent
    plots.style()
    return ROOT, json, mo, model, np, os, plots, plt, teaching


@app.cell
def _(mo):
    mo.md(r"""
    # Daily SVJ: the same inference workflow, a different financial question

    **Task → prior predictive checks → train and select → condition and diagnose →
    correct or fall back → propagate uncertainty → evaluate.** This is the same
    lifecycle as the HKDE option example. Here the question is physical-measure
    (**P**) return risk, with eight parameters and a latent terminal variance.
    The architecture, particle likelihood and downstream risk calculation serve
    this application.

    The default lesson reads a verified **synthetic** illustration. It needs no
    market file, network access, neural-backend initialization, training or live
    particle MCMC. Its small cached run is a workflow demonstration, **not a
    calibration or scientific acceptance study**.
    """)
    return


@app.cell
def _(ROOT, teaching):
    bundle, synthetic_window, cached, cached_report = teaching.load_bundle(
        ROOT / "data/synthetic/svj-workflow", checkpoint=ROOT / "checkpoints/svj"
    )
    return bundle, cached, cached_report, synthetic_window


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Specify observations, unknowns and the generative model

    The observation is one complete history of **1,000 daily log returns**.
    Saved data use decimal returns and an explicit `dt_years = 1/252` grid.
    The SVJ adapter converts to percent once, preserving the checkpoint's input
    channels `[y, log(y² + 0.01)]`. It never rescales each history to unit variance.

    In the contributed daily model, percent log returns and percent-squared
    daily variance obey

    $$y_t=\mu+\sqrt{v_{t-1}}\,\varepsilon_t^1+B_t\xi_t,
    \qquad B_t\sim\mathrm{Bernoulli}(\lambda),\quad
    \xi_t\sim\mathcal N(\mu_J,\sigma_J^2),$$

    $$v_t=\max\!\left\{v_{t-1}+\kappa(\theta-v_{t-1})+
    \sigma_v\sqrt{v_{t-1}}\left(\rho\varepsilon_t^1+
    \sqrt{1-\rho^2}\varepsilon_t^2\right),\;10^{-3}\right\}.$$

    The Gamma initial distribution, daily Bernoulli jump opportunity, variance
    floor and prior are part of this finite model. A continuous-time analogy
    does not make it interchangeable with the repository's substepped Heston
    kernel. The ninth inference target is
    $\log\sigma_T=\log(\sqrt{252v_T}/100)$, so forecasts can continue from the
    **matched parameter and terminal-state draw**.

    The [original design](https://github.com/fmuia/sbi4finance/blob/main/docs/design/svj-design-document.md) develops the
    model and adapted particle filter in more detail.
    """)
    return


@app.cell
def _(model):
    model.prior_table().round(2)
    return


@app.cell
def _(cached, model, np, plots, plt):
    _fig, (_top, _bottom) = plt.subplots(2, 1, figsize=(9.4, 4.8), sharex=True, constrained_layout=True)
    _days = np.arange(1, model.WINDOW + 1)
    _top.plot(_days, cached["returns"], color=plots.SERIES[0], linewidth=0.75)
    _top.scatter(_days[cached["jumps"]], cached["returns"][cached["jumps"]],
                 s=24, color=plots.SERIES[1], label="synthetic jump truth", zorder=3)
    _top.set_ylabel("Daily log return (%)")
    _top.legend(loc="lower left")
    _bottom.plot(_days, model.spot_vol_from_variance(cached["variance"]), color=plots.SERIES[0])
    _bottom.set_ylabel("Annualized spot volatility\n(synthetic truth)")
    _bottom.set_xlabel("Trading day")
    _fig.suptitle("Seed-3 illustration: observed returns and latent variance", x=0.01, ha="left", fontsize=11)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Check the prior predictive implications before fitting

    The path above has hand-chosen illustrative parameters. It is **not** a
    prior-predictive sample. The small independent panel below draws 24 complete
    histories from the declared prior. Inspect return scales, jump frequency
    and variance-floor contact before deciding whether the model/prior fits the
    intended application. These descriptive checks have no automatic pass rule.

    Synthetic variance, jump labels and true parameters are available for
    checking the simulator and evaluating recovery. They are **auxiliary truth**;
    only observed returns and known task context enter inference.
    """)
    return


@app.cell
def _(mo, model, np):
    _prior = model.simulate(24, seed=20261013)
    _jump_counts = _prior["jumps"].sum(axis=1)
    _daily_sd = _prior["returns"].std(axis=1)
    _floor_fraction = (_prior["variance"] == model.V_MIN).mean(axis=1)
    mo.ui.table([
        {"quantity across 24 prior histories": "Daily log-return standard deviation (%)",
         "10%": round(float(np.quantile(_daily_sd, .1)), 3), "median": round(float(np.median(_daily_sd)), 3),
         "90%": round(float(np.quantile(_daily_sd, .9)), 3)},
        {"quantity across 24 prior histories": "Jump days in 1,000 days",
         "10%": round(float(np.quantile(_jump_counts, .1)), 3), "median": round(float(np.median(_jump_counts)), 3),
         "90%": round(float(np.quantile(_jump_counts, .9)), 3)},
        {"quantity across 24 prior histories": "Fraction of days at variance floor",
         "10%": round(float(np.quantile(_floor_fraction, .1)), 4), "median": round(float(np.median(_floor_fraction)), 4),
         "90%": round(float(np.quantile(_floor_fraction, .9)), 4)},
    ], selection=None)
    return


@app.cell
def _(ROOT, bundle, json, mo):
    _training = json.loads((ROOT / "checkpoints/svj/training.json").read_text())
    mo.md(f"""
    ## 3. Train, select and bind an inference artifact

    New training uses the common lifecycle: independent training and validation
    examples, training-only normalization, validation-based checkpoint selection,
    then save/reload checks. Online simulation and saved banks feed the same SVJ
    network and target adapter. The test split stays reserved for evaluation.

    This illustration retains Francesco's **historical checkpoint**: its record
    reports **{_training['simulated_series']:,} simulated histories** over
    **{_training['epochs']} epochs**, with a 32-dimensional sequence summary and
    **50 Euler steps** in the flow sampler. Historical checkpoint-selection
    provenance is preserved as recorded; the integration does not retrospectively
    certify it under the new training lifecycle.

    The cache binds the checkpoint, observation, full task and inference sources.
    Task fingerprint: `{bundle['task_fingerprint'][:16]}…`.
    Architecture and summary-network choices belong to the application; validation,
    artifact compatibility and evidence reporting belong to the shared workflow.
    """)
    return


@app.cell
def _(mo, os):
    live_demo = mo.ui.checkbox(
        value=os.environ.get("SBI4FINANCE_SVJ_DEMO") == "1",
        label="Recompute this synthetic case with the pinned neural environment (optional)",
    )
    live_demo
    return (live_demo,)


@app.cell
def _(ROOT, cached, cached_report, live_demo, synthetic_window, teaching):
    active_report = cached_report
    active_draws = cached["posterior"]
    active_conditional_var = cached["conditional_var"]
    if live_demo.value:
        from sbi4finance.svj import amortized as _amortized, checks as _checks, pipeline as _pipeline

        _checkpoint = ROOT / "checkpoints/svj"
        _posterior = _amortized.AmortizedPosterior.load(_checkpoint)
        _reference = _checks.MisspecificationTest.load(_checkpoint / "misspecification.npz")
        active_report, active_draws = _pipeline.run_prepared_window(
            synthetic_window, _posterior, _reference, teaching.demo_config()
        )
        active_conditional_var = _pipeline.risk.conditional_var(active_draws)
    return active_conditional_var, active_draws, active_report


@app.cell
def _(active_report, live_demo, mo):
    _workflow = active_report["workflow"]
    _reasons = ", ".join(_workflow["diagnostic_reasons"]) or "No active diagnostic flags; this does not establish calibration."
    mo.md(f"""
    ## 4. Condition, diagnose, correct and choose a response

    **{'Live' if live_demo.value else 'Cached'} synthetic run · reliability:
    `{_workflow['reliability']}` · decision: `{_workflow['decision']}`**

    Selected estimator: `{_workflow['method']}`. Diagnostic reasons: **{_reasons}**

    The neural posterior proposes joint parameter/state draws. The particle
    importance stage fits a parameter proposal, evaluates a particle likelihood
    and supplies terminal states through that filter. It is not simply a
    reweighting of nine unchanged neural coordinates. Weight concentration and
    likelihood noise are checked against declared thresholds.

    The summary-space model check has a different role: an observation mismatch
    cannot be repaired just by changing the posterior estimator. When inference
    is flagged, the shared policy can report the flagged result, run a declared
    fallback, or decline downstream calculation. This bounded demo uses
    **report-only** and does not run PMMH. A PMMH run must pass its own mixing and
    noise diagnostics before its result is labelled locally checked.

    Budget: **128 neural draws, 256 proposals, 128 audit particles, 4 forward paths
    per joint draw**. Scientific acceptance: `{_workflow['scientific_acceptance']}`.
    """)
    return


@app.cell
def _(active_report, mo):
    _rows = []
    for _stage in active_report["workflow"]["stages"]:
        for _diagnostic in _stage["diagnostics"]:
            _metrics = ", ".join(
                f"{_name}={_value:.4g}" if isinstance(_value, (int, float)) else f"{_name}={_value}"
                for _name, _value in _diagnostic["metrics"].items()
            )
            _thresholds = ", ".join(
                f"{_name}={_value:.4g}" if isinstance(_value, (int, float)) else f"{_name}={_value}"
                for _name, _value in _diagnostic["thresholds"].items()
            )
            _rows.append({"stage": _stage["stage"], "method": _stage["method"],
                          "diagnostic": _diagnostic["name"], "status": _diagnostic["status"],
                          "scope": _diagnostic["scope"], "metrics": _metrics,
                          "declared thresholds": _thresholds})
    mo.ui.table(_rows, selection=None)
    return


@app.cell
def _(active_draws, cached, mo, model, np):
    _quantities = model.quantities(active_draws)
    _truth = model.quantities(cached["truth"])
    mo.ui.table([
        {"quantity": _name, "synthetic truth (comparison only)": round(float(_truth[_j]), 4),
         "posterior 5%": round(float(np.quantile(_quantities[:, _j], .05)), 4),
         "posterior median": round(float(np.median(_quantities[:, _j])), 4),
         "posterior 95%": round(float(np.quantile(_quantities[:, _j], .95)), 4)}
        for _j, _name in enumerate(model.QUANTITY_LABELS)
    ], selection=None)
    return


@app.cell
def _(active_report, mo):
    mo.md(f"""
    ## 5. Propagate joint uncertainty to the financial estimand

    Continue each **paired parameter and terminal-state row** into future returns.
    Weighted draws are jointly resampled before path simulation; weights and
    parameter/state dependence cannot be discarded. The plug-in comparison fixes
    parameters at their posterior median and re-filters terminal-state uncertainty.

    The retained risk definition is **percent log loss**: minus the cumulative
    percent log return. It preserves the meaning of the historical evidence.
    Percent-of-position losses would instead transform each future return as
    `100 * (1 - exp(R / 100))` before calculating quantiles and tail averages.
    The two ES definitions cannot be converted by transforming a final ES number.

    **Forecast reliability: `{active_report['risk']['reliability']}`.** The tiny
    forward sample illustrates uncertainty propagation; it is insufficient for
    precise tail-risk estimation. These synthetic outputs are not a risk limit
    or a claim of market performance.
    """)
    return


@app.cell
def _(active_report, mo):
    _rows = []
    for _method, _label in (("predictive", "joint predictive"), ("plugin_at_posterior_median", "plug-in parameters")):
        for _horizon, _numbers in active_report["risk"][_method].items():
            _rows.append({"horizon (trading days)": int(_horizon), "forecast": _label,
                          "99% VaR (percent log loss)": round(_numbers["var"], 3),
                          "97.5% ES (percent log loss)": round(_numbers["es"], 3),
                          "reliability": active_report["risk"]["reliability"]})
    mo.ui.table(_rows, selection=None)
    return


@app.cell
def _(active_conditional_var, active_report, plots, plt):
    _forecast = active_report["risk"]
    _predictive = _forecast["predictive"].get("1", _forecast["predictive"].get(1))
    _plugin = _forecast["plugin_at_posterior_median"].get("1", _forecast["plugin_at_posterior_median"].get(1))
    _fig, _ax = plt.subplots(figsize=(8.3, 3.5), constrained_layout=True)
    _ax.hist(active_conditional_var, bins=24, color=plots.SERIES[0], alpha=.85,
             edgecolor=plots.SURFACE, linewidth=.6)
    _ax.axvline(_predictive["var"], color=plots.INK, label="predictive VaR (small Monte Carlo sample)")
    _ax.axvline(_plugin["var"], color=plots.SERIES[1], label="plug-in VaR")
    _ax.set_xlabel("1-day 99% conditional VaR (percent log loss)")
    _ax.set_ylabel("Joint posterior draws")
    _ax.set_title(f"Uncertainty about the risk number · {_forecast['reliability']}", loc="left")
    _ax.grid(axis="y", visible=False)
    _ax.legend(fontsize=8)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Evaluate inference beyond a single illustration

    A truth comparison on one hand-chosen synthetic case does not establish
    population calibration or conditional accuracy. Keep the original evidence
    visible: **initial SBC: 8/9 targets inside the simultaneous bands; replicate:
    9/9**. The original conditional PMMH comparison was mixed, with flagged
    importance cases and timing limitations. A replicate pass does not erase
    the initial excursion, and a local diagnostic pass is not scientific acceptance.

    Read the [preserved pipeline study](https://github.com/fmuia/sbi4finance/blob/main/docs/svj-pipeline.md),
    [initial validation](https://github.com/fmuia/sbi4finance/blob/main/docs/results/svj-validation/validation.json),
    [SBC replicate](https://github.com/fmuia/sbi4finance/blob/main/docs/results/svj-validation/sbc_replicate.json), and
    [conditional reference comparison](https://github.com/fmuia/sbi4finance/blob/main/docs/results/svj-agreement/agreement.json).
    Those historical findings retain their original checkpoint and log-loss
    estimand; this lesson does not rerun or upgrade their scientific status.

    For the common workflow, ask separately: are inference ranks calibrated over
    independent prior examples, do individual cases agree with suitable reference
    methods, and are downstream predictions accurate on held-out futures? Real
    return windows are optional inputs through the same declared schedule and
    prepared-data adapter. They provide no true parameter/state labels, and one
    market window cannot establish repeated-outcome performance.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise: trace a saved bank through the same training lifecycle

    Run these commands manually in the pinned environment, choosing **new output
    directories**. The tiny run checks training/save/reload plumbing and does not
    replace the preserved checkpoint or establish posterior quality.

    ```sh
    pixi run python scripts/generate_svj_bank.py --config configs/tasks/svj_daily_smoke.json --out /tmp/my-svj-bank
    pixi run python scripts/check_svj_bank.py --bank /tmp/my-svj-bank
    pixi run svj-train --bank /tmp/my-svj-bank --epochs 1 --batch-size 8 --out /tmp/my-svj-training
    ```

    Inspect `manifest.json` for observation, target and identity roles, then
    `training.json` for validation-based selection. Confirm that test examples
    were reserved, and that changing synthetic truth or future outcomes cannot
    change the conditioning features. Compare the task fingerprint before
    attempting to load a checkpoint.

    To regenerate this lesson's bounded evidence explicitly, use:

    ```sh
    pixi run python scripts/prepare_svj_teaching.py --checkpoint checkpoints/svj --out /tmp/my-svj-lesson
    ```

    Extending the prior, window design, network or solver creates a new candidate
    needing its own validation. Increasing the particle or forecast budget is an
    explicit experiment; opening this notebook does not start one.
    """)
    return


if __name__ == "__main__":
    app.run()
