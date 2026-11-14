import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="MMAR: from turbulent returns to drawdown scenarios")


@app.cell
def _():
    import inspect
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from sbi4f_tutorial.shared import plots as workshop_plots
    from sbi4f_tutorial.shared.paths import ASSETS
    from sbi4f_tutorial.mmar_takehome import model, inference, diagnostics, lesson, plots
    workshop_plots.style()
    return ASSETS, diagnostics, inference, inspect, lesson, mo, model, np, plots, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # From turbulent returns to uncertain drawdown scenarios

    **A complete MMAR take-home · about one hour · saved results work offline**

    What produces clusters of quiet and turbulent days? How can we carry
    uncertainty about that mechanism into drawdown scenarios?

    We will observe one year of returns, build a small **multifractal model of asset
    returns (MMAR)**, follow its BayesFlow estimator, check it on known truths, then
    carry its posterior into drawdown scenarios.

    **Observe → Model → Simulate → Build → Check → Infer → Use**

    This lesson synthesizes the univariate and resolution experiments from
    [Stefan Radev's amortized-fractals](https://github.com/stefanradev93/amortized-fractals).
    Its market-wide and multivariate notebooks are the expanded story.
    """)
    return


@app.cell(hide_code=True)
def _(lesson):
    saved_report, saved = lesson.load_saved()
    market, market_dates, observed = lesson.load_market()
    return market, market_dates, observed, saved, saved_report


@app.cell(hide_code=True)
def _(mo):
    route = mo.ui.radio({"Explore saved results — complete offline route": "saved",
                         "Load the supplied estimator": "pretrained",
                         "Train a tutorial estimator here": "tutorial"}, value="Explore saved results — complete offline route",
                        label="Choose an execution route")
    mo.vstack([route, mo.md("Opening performs no training, downloads or neural sampling. "
        "All main-route questions can be answered with the accepted saved results. "
        "The component simulator below is a cheap local NumPy calculation.")])
    return (route,)


@app.cell(hide_code=True)
def _(market_dates, mo):
    mo.md(f"""
    ## 1 · Observe: what should a return model explain?

    **VOO · 256 daily simple returns · {market_dates[0]} to {market_dates[-1]}**

    Each observation is $r_t=P_t/P_{{t-1}}-1$, using the cached adjusted-price return
    window. The Gaussian reference uses the mean and standard deviation of these
    **same simple returns**. This selected crisis-era window motivates the problem;
    it is not an out-of-sample forecasting contest.

    **Predict first:** do large moves appear alone, or in clusters? Does a constant
    variance explain their magnitude and their placement in time equally well?
    """)
    return


@app.cell(hide_code=True)
def _(market_dates, observed, plots):
    plots.observed(observed, market_dates)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Model: separate location, scale, clustering and tails

    $$r_t=\mu+\bar\sigma\sqrt{\theta_t(q,d)}\,\varepsilon_t,
    \qquad \varepsilon_t=z_t\sqrt{\frac{\nu-2}{U_t}},$$
    $$z_t\sim N(0,1),\quad U_t\sim\chi^2_\nu,\quad z_t\perp U_t.$$

    | Quantity | What it does | What the estimator sees |
    | :-- | :-- | :-- |
    | μ | Daily return location | Unknown target |
    | σ̄ | Baseline scale | Unknown target |
    | q | Contrast between quiet and turbulent cascade blocks | Unknown target |
    | ν | Innovation-tail thickness; smaller means heavier tails | Unknown target |
    | d | Number of terminal volatility blocks | **Supplied condition** |
    | θ, z, U | Latent cascade and random innovations | Never observed inputs |

    At **d = 8**, the model has eight 32-day blocks. The cascade multiplies local
    intensity by **2q** or **2(1−q)** at every binary split. Its 256 intensities
    average one. A random circular phase avoids placing every block boundary at
    the same dates.

    q changes **clustering and variance mixing**; ν changes **innovation tails**.
    Both can change the marginal frequency of extreme returns.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([mo.md(r"""
    **The actual independent priors**

    | Target | Prior |
    | :-- | :-- |
    | μ | Normal(mean 0, daily standard deviation 0.0015) |
    | σ̄ | Uniform(0.005, 0.025) |
    | q | Uniform(0.52, 0.90) |
    | ν−2 | Log-uniform(0.05, 10), so ν lies in [2.05, 12] |

    d is supplied at inference. Training covers **4, 8, 16, 32, 64 and 128** blocks.
    A bound is not a complete prior: ν is **not uniformly distributed** between its bounds.
    """), mo.accordion({"The component math and the accepted-path law": mo.md(r"""
    A split divides a parent's mass between fractions q and 1−q. Dividing by
    the child's half-width gives relative intensities 2q and 2(1−q). Repeated
    splitting produces d terminal intensities, each repeated for 256/d days,
    normalized to average one and circularly shifted.

    The Student-t innovation has variance one **before the acceptance rule**:
    a tν variable has variance ν/(ν−2), so we multiply it by √((ν−2)/ν).
    Conditional on the random cascade, daily innovations are independent. After
    averaging over the shared cascade, returns are not IID.

    **Numerical model convention:** redraw the complete 256-day path if any
    simple return is at or below −100%, or nonfinite. Keep the parameter vector;
    do not drop difficult parameters. This acceptance condition belongs to the
    trained model. Shorter segments must come from an accepted full window.

    This is a finite-window block-cascade model. It is a different construction
    from Heston's variance diffusion or rough Bergomi's Volterra driver.
    """)})])
    return


@app.cell(hide_code=True)
def _(mo):
    contrast = mo.ui.slider(.52, .9, step=.01, value=.7, label="Cascade contrast q", show_value=True)
    tails = mo.ui.slider(2.05, 12., step=.05, value=4., label="Innovation-tail degrees ν", show_value=True)
    realization = mo.ui.number(0, 100, value=1, step=1, label="Simulated realization (seed)")
    mo.vstack([mo.md("### Try the components\n\nKeep σ̄ = 1.5% daily and μ = 0. "
        "Change q while holding ν and the realization fixed, then inspect the innovation "
        "density as you change ν. Axes stay fixed; crosses count values outside the view."),
        mo.hstack([contrast, tails, realization], wrap=True)])
    return contrast, realization, tails


@app.cell(hide_code=True)
def _(contrast, plots, realization, tails):
    plots.components(q=contrast.value, nu=tails.value, d=8, seed=realization.value)
    return


@app.cell(hide_code=True)
def _(mo, plots, tails):
    mo.vstack([plots.tail_mechanism(tails.value), mo.md("The density comparison holds variance one before path acceptance. "
        "Resetting a seed does not hold Student-t innovations fixed across different ν. "
        "At fixed ν and d, q comparisons reuse attempted tree orientations, phase and innovations; "
        "a whole-path rejection can change the accepted realization. No values are removed from the plot's data.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Check the prior's implications

    The sliders condition on one parameter vector. A **prior-predictive check**
    instead draws the parameters first, then one history per draw. Here 200
    independent simulations use d=8. The dark line marks the observed window.

    The absolute-return ACF checks temporal order. The magnitude summaries and
    drawdown ask different questions. A band is a pointwise simulation interval,
    not a simultaneous fit test or posterior-calibration certificate.
    """)
    return


@app.cell(hide_code=True)
def _(plots, saved):
    plots.check_comparison(saved["observed_summaries"], saved["prior_summaries"],
                           saved["observed_abs_acf"], saved["prior_abs_acf"], "Before inference · prior-predictive implications at d=8")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([mo.md(r"""
    ## 4 · Build: the sequence and the assumption have different roles

    **One training pair:** four parameters drawn from the prior, one known d,
    and the 256-day simple-return history they generate.
    """), mo.Html('''<div style="border:1px solid #b8a1d7;border-radius:12px;padding:18px;line-height:1.7">
    <div><b>256 × 1 observed returns</b> → full-window FusionTransformer → <b>16 summary values</b></div>
    <div style="margin:8px 0;color:#167e80"><b>Known log d</b> joins the summary: 16 + 1 = <b>17 conditions</b></div>
    <div>17 conditions → four-layer spline CouplingFlow → <b>4 posterior coordinates</b>: μ, σ̄, q, ν</div>
    </div>'''), mo.md(r"""
    Like the trajectory lesson, BayesFlow joins a learned sequence summary to a
    conditional flow. Here the **FusionTransformer uses the full fixed window**;
    it is not our masked causal GRU and is not trained for arbitrary prefixes.
    log d is a supplied design condition, not a fifth unknown parameter.

    The source encoder has two 64-dimensional embedding stages and a bidirectional
    LSTM template. Its serialized adapter and fitted standardization travel with
    the estimator. The returns, targets and log d are standardized; latent weights
    and innovations never enter the network.
    """)])
    return


@app.cell(hide_code=True)
def _(inference, inspect, mo):
    mo.vstack([mo.md("### The executing BayesFlow construction\n\nThis is the source of `build_workflow`, "
        "called by the optional training route. Bounds are reversed on sampling. "
        "The smaller tutorial architecture is a separate estimator."),
        mo.md("```python\n" + inspect.getsource(inference.build_workflow) + "\n```")])
    return


@app.cell(hide_code=True)
def _(diagnostics, inference, model):
    def train_tutorial_model(count=2048, epochs=5):
        import time
        # Independent histories: validation never updates network weights.
        training = model.simulate(count, seed=290061112)
        validation = model.simulate(256, seed=290061113)
        workflow = inference.build_workflow(small=True)
        started = time.perf_counter()
        history = workflow.fit_offline(data=training, validation_data=validation,
                                       epochs=epochs, batch_size=128)
        # A separate exploratory test, never the reserved supplied-model cohort.
        test = model.simulate(64, d=8, seed=290061114)
        draws = inference.sample(workflow, test["returns"], 8, count=128, seed=290061115)
        rule = inference.freeze_rule(n=64, draws=128)
        rule.update(null_replications=1000, null_seed=290061116)
        checks = diagnostics.evaluate(draws, model.parameter_array(test),
                                       null=diagnostics.null_statistics(rule))
        return {"workflow": workflow, "history": history, "checks": checks,
                "seconds": time.perf_counter()-started, "count": count, "epochs": epochs}
    return (train_tutorial_model,)


@app.cell(hide_code=True)
def _(inspect, mo, train_tutorial_model):
    mo.vstack([mo.md("### Fit offline; then check the estimator just trained\n\n"
        "This function is the actual optional action below. Training samples d from "
        "all six supported values; validation and exploratory test histories use "
        "independent seeds. Its checks describe this tiny fit only."),
        mo.md("```python\n" + inspect.getsource(train_tutorial_model) + "\n```")])
    return


@app.cell(hide_code=True)
def _(ASSETS, mo, route, saved_report):
    checkpoint = mo.ui.text(value=str(ASSETS / "local-checkpoints" / "mmar-sensitivity" / "model.keras"),
                            label="Matching checkpoint inside tutorial/assets/local-checkpoints", full_width=True)
    load_button = mo.ui.run_button(label="Verify and load the supplied estimator")
    _url = saved_report["checkpoint_url"]
    _hash = saved_report["checkpoint_sha256"]
    mo.vstack([mo.md("### Optional restoration\n\nThe complete saved route needs no checkpoint. "
        "To recompute, place the unchanged pinned source checkpoint in this cache. "
        "The loader verifies its SHA-256 and restores the whole estimator, including preprocessing."),
        mo.accordion({"Download recipe — run once from tutorial/": mo.md(f"""```bash
mkdir -p assets/local-checkpoints/mmar-sensitivity
curl -L '{_url}' -o assets/local-checkpoints/mmar-sensitivity/model.keras
```
Required SHA-256: `{_hash}`. Download is your explicit action; the notebook makes no network request.
""")}), checkpoint, load_button]) if route.value == "pretrained" else mo.md("")
    return checkpoint, load_button


@app.cell(hide_code=True)
def _(checkpoint, inference, load_button, mo, route, saved_report):
    supplied_model = None
    load_message = None
    if route.value == "pretrained" and load_button.value:
        try:
            supplied_model = inference.load_checkpoint(checkpoint.value, saved_report["checkpoint_sha256"])
            load_message = "Verified checkpoint restored. Fresh inference appears below."
        except Exception as _error:
            load_message = f"Restoration stopped: {_error}. Choose saved results to continue offline."
    mo.md(load_message or "")
    return (supplied_model,)


@app.cell(hide_code=True)
def _(mo, route):
    training_size = mo.ui.dropdown({"Short API exercise: 512 pairs, 2 epochs": (512, 2),
                                    "Longer exercise: 2,048 pairs, 5 epochs": (2048, 5)},
                                   value="Short API exercise: 512 pairs, 2 epochs", label="Training budget")
    train_button = mo.ui.run_button(label="Generate pairs, train and check a new estimator")
    mo.vstack([mo.md("### Optional live tutorial fit\n\nThe session-only model uses a smaller encoder and two spline layers. "
        "It illustrates the API; its accuracy comes from its own checks. "
        "Training includes backend compilation on first use. It never overwrites the supplied model or saved evidence."),
        training_size, train_button]) if route.value == "tutorial" else mo.md("")
    return train_button, training_size


@app.cell(hide_code=True)
def _(mo, route, train_button, train_tutorial_model, training_size):
    tutorial_result = None
    training_message = None
    if route.value == "tutorial" and train_button.value:
        try:
            tutorial_result = train_tutorial_model(*training_size.value)
            training_message = f"New session model: {tutorial_result['count']} pairs, {tutorial_result['epochs']} epochs, {tutorial_result['seconds']:.1f} s including fit and check."
        except Exception as _error:
            training_message = f"Tutorial fit stopped: {_error}. The saved route remains available."
    mo.md(training_message or "")
    return (tutorial_result,)


@app.cell(hide_code=True)
def _(mo, tutorial_result):
    mo.stop(tutorial_result is None)
    import bayesflow as bf
    mo.vstack([bf.diagnostics.loss(tutorial_result["history"]), mo.md(
        "**This new fit's exploratory check:** 64 independent histories × 128 draws at d=8. "
        "These are descriptive checks; the supplied model's reserved-cohort decisions do not apply."),
        mo.ui.table([{"Parameter":_r["parameter"],"MSE / prior baseline":round(_r["mse_to_prior_mean_ratio"],3),
                      "Central 90% coverage":f"{100*_r['coverage90']:.1f}%",
                      "Mean 90% width":f"{_r['mean_interval_width90']:.4g}"}
                     for _r in tutorial_result["checks"]["records"]], selection=None, pagination=False)])
    return (bf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Check: known truths before market interpretation

    Recovery asks whether posterior estimates improve on always returning the
    prior mean. Calibration asks whether posterior uncertainty behaves as claimed
    across independent simulated histories. Returning the prior can pass a rank
    check without learning anything; inspect both.

    The supplied model was evaluated on **200 independent histories × 512 draws**
    at each evaluated resolution (d=4, 8 and 128). The finite-rank null and decision rule were
    frozen before evaluation: 10,000 discrete-uniform null simulations, followed
    by Holm correction across **12 parameter/resolution tests** at familywise 5%.
    Central-90% coverage has a descriptive Wilson 95% interval; it is not a second
    pass/fail rule. These tests do not prove exact posterior accuracy.
    """)
    return


@app.cell(hide_code=True)
def _(mo, model):
    diagnostic_d = mo.ui.dropdown({"d=4 · 64-day blocks": 4, "d=8 · 32-day blocks": 8,
                                   "d=128 · 2-day blocks": 128}, value="d=8 · 32-day blocks", label="Independent diagnostic resolution")
    diagnostic_parameter = mo.ui.dropdown({name: j for j,name in enumerate(model.PARAMETERS)}, value="q", label="Recovery parameter")
    mo.hstack([diagnostic_d, diagnostic_parameter], wrap=True)
    return diagnostic_d, diagnostic_parameter


@app.cell(hide_code=True)
def _(diagnostic_d, diagnostic_parameter, mo, plots, saved, saved_report):
    _d = diagnostic_d.value
    _records = [row for row in saved_report["calibration"] if row["d"] == _d]
    _display = [{"Parameter":_r["parameter"],
                 "MSE / prior baseline":round(_r["mse_to_prior_mean_ratio"],3),
                 "90% coverage [Wilson 95%]":f"{100*_r['coverage90']:.1f}% [{100*_r['coverage90_wilson95'][0]:.1f}–{100*_r['coverage90_wilson95'][1]:.1f}]",
                 "Holm p":round(_r["rank_holm_p"],3),
                 "Rank check":"Reject" if _r["rank_holm_reject"] else "No rejection"}
                for _r in _records]
    mo.vstack([plots.diagnostics_figure(saved, _d, diagnostic_parameter.value),
        mo.md(f"**Reserved supplied-model check: {saved_report['calibration_rejections']}/12 Holm rejections.** "
              "No rejection is not proof of calibration. Recovery and interval widths show whether the estimator learns."),
        mo.ui.table(_display, selection=None, pagination=False),
        mo.md("An MSE ratio below 1 improves on the prior-mean estimate. Coverage need not equal 90% "
              "exactly in 200 histories; brackets describe its finite-cohort uncertainty."),
        mo.accordion({"Full numerical diagnostic record":mo.ui.table(_records,selection=None,pagination=False)})])
    return


@app.cell(hide_code=True)
def _(mo):
    diagnostic_example = mo.ui.slider(0,5,step=1,value=0,label="Inspect one retained test history",show_value=True)
    diagnostic_example
    return (diagnostic_example,)


@app.cell(hide_code=True)
def _(diagnostic_d, diagnostic_example, diagnostic_parameter, mo, model, plt, saved):
    _d, _case, _j = diagnostic_d.value, diagnostic_example.value, diagnostic_parameter.value
    _history = saved[f"diagnostic_{_d}_histories"][_case]
    _draws = saved[f"diagnostic_{_d}_examples"][_case]
    _fig, _axes = plt.subplots(1,2,figsize=(10,2.8))
    _axes[0].plot(100*_history, color="#283447", linewidth=.7)
    _axes[0].set(xlabel="Trading day",ylabel="Daily simple return (%)",title=f"Independent test history {_case}")
    _axes[1].hist(_draws[:,_j], bins=25,density=True,color="#7653ac",alpha=.7)
    _axes[1].axvline(saved[f"diagnostic_{_d}_truth"][_case,_j],color="#283447",label="Generating truth")
    _axes[1].set(xlabel=model.PARAMETERS[_j],ylabel="Posterior density")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig,mo.md("These are retained rows from the independent check. Selecting one runs no inference. "
                         "One successful history cannot establish cohort calibration.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Infer: use the same observations at d=8

    Condition on the **entire fixed 256-day VOO window**, initially assuming
    eight 32-day volatility blocks. The targets retain their original daily units.
    The dashed priors make contraction and boundary concentration visible.

    Compare q and ν: can clustered scale changes and heavy-tailed innovations
    partly explain the same large daily moves? Their paired posterior is more
    informative than four separate point estimates.
    """)
    return


@app.cell(hide_code=True)
def _(inference, lesson, observed, route, saved, supplied_model, tutorial_result):
    active_data = saved
    active_label = "Accepted saved draws from the supplied estimator"
    active_note = None
    _workflow = supplied_model if route.value == "pretrained" else tutorial_result["workflow"] if tutorial_result else None
    if route.value != "saved" and _workflow is not None:
        active_data = dict(saved)
        active_label = "Fresh supplied-estimator inference" if route.value == "pretrained" else "New tiny tutorial estimator — approximation quality is exploratory"
        _draws = inference.sample(_workflow, observed, 8, count=512, seed=310061120)[0]
        _scenarios = lesson.scenario_result(_draws, 8, seed=310071120)
        active_data["posterior_8"] = _draws
        active_data["mdd_8"] = _scenarios["mdd"]
        active_data["wealth_8"] = _scenarios["wealth"]
        active_data["predictive_8_summaries"] = _scenarios["summaries"]
        active_data["predictive_8_abs_acf"] = _scenarios["abs_acf"]
        active_data["point_mdd"] = _scenarios["point_mdd"]
    elif route.value != "saved":
        active_note = "Requested live model is not ready. The figures below explicitly retain the accepted saved results."
    return active_data, active_label, active_note


@app.cell(hide_code=True)
def _(active_data, active_label, active_note, mo, plots):
    mo.vstack([mo.md(f"**Source: {active_label}.** " + (active_note or "")),
               plots.posterior(active_data["posterior_8"])])
    return


@app.cell(hide_code=True)
def _(active_data, plt):
    _fig,_axis=plt.subplots(figsize=(6,3))
    _axis.scatter(active_data["posterior_8"][:,2],active_data["posterior_8"][:,3],s=6,alpha=.25,color="#7653ac")
    _axis.set(xlabel="Cascade contrast q",ylabel="Innovation-tail degrees ν",xlim=(.52,.90),ylim=(2.05,12),title="Complete paired posterior draws")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(active_data, mo, plots, saved):
    mo.vstack([plots.check_comparison(saved["observed_summaries"],active_data["predictive_8_summaries"],
                                      saved["observed_abs_acf"],active_data["predictive_8_abs_acf"],
                                      "After inference · independent replicated histories at d=8"),
        mo.md("**Interpret:** which observed summaries or lag patterns remain outside the simulated ranges? "
              "How can synthetic calibration look acceptable while the model's adequacy for real data remains limited? "
              "Calibration checks the estimator under its assumed simulator; predictive comparisons can reveal "
              "behavior that simulator does not reproduce.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Use: what drawdowns does the fitted model generate?

    A complete posterior row generates one new full-window path. Preserve its
    parameter dependence. **The one-vector comparison also simulates random paths**;
    it retains process uncertainty while ignoring parameter uncertainty.

    $$W_t=\prod_{s=1}^t(1+r_s),\quad W_0=1,\qquad
      \mathrm{MDD}=\max_t\left(1-\frac{W_t}{\max_{u\le t}W_u}\right).$$

    Initial wealth matters: a loss on day one is a drawdown. Calculations use
    cumulative log1p returns for numerical stability. Curves below show the
    distribution of one path statistic; a collection of visible paths is not a
    simultaneous prediction band.

    **Predict:** must integrating parameter uncertainty always raise a drawdown
    probability? Inspect rather than assume.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    drawdown_threshold = mo.ui.slider(.05,.8,step=.01,value=.2,label="Maximum drawdown threshold",show_value=True)
    drawdown_threshold
    return (drawdown_threshold,)


@app.cell(hide_code=True)
def _(active_data, active_label, drawdown_threshold, lesson, mo, plots):
    _threshold=drawdown_threshold.value
    _rows=[]
    for _label,_values in [("Paired posterior draws",active_data["mdd_8"]),("One posterior-mean vector",active_data["point_mdd"])]:
        _event=lesson.exceedance(_values,_threshold)
        _low,_high=_event["simulation_interval95_pct"]
        _rows.append({"parameter treatment":_label,"scenarios":_event["scenarios"],
                      "P(drawdown above threshold)":f"{_event['probability_pct']:.1f}%",
                      "95% simulation interval":f"{_low:.1f}–{_high:.1f}%"})
    mo.vstack([mo.md(f"**Source: {active_label} · threshold {100*_threshold:.0f}%.**"),
        plots.drawdown(active_data["wealth_8"],active_data["mdd_8"],active_data["point_mdd"],active_data["observed_wealth"],_threshold),
        mo.ui.table(_rows,selection=None,pagination=False),
        mo.md("The interval describes finite scenario-count precision, not posterior uncertainty about the probability. "
              "Changing the threshold reuses exactly these same scenarios.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Why this is a scenario experiment, not a continuation forecast": mo.md(r"""
    These histories begin from a new cascade with a new phase. They average over
    parameters learned from the observed window, but do not condition on the
    latent volatility phase at its last date. A continuation forecast would need
    the joint posterior of that state and the parameters. The sequence summary
    alone is not such a filter.

    Between-parameter and within-path randomness are both present in the mixture.
    To separate them, simulate many histories at each fixed posterior parameter
    row, estimate a conditional drawdown probability, then compare those
    probabilities across rows, recording finite Monte Carlo error. That optional
    nested experiment is different from treating pointwise wealth bands as
    uncertainty about a risk metric.
    """)})
    return


@app.cell(hide_code=True)
def _(mo, saved_report):
    mo.vstack([mo.md("### Continue with the expanded story\n\n"
        "[Univariate MMAR](https://github.com/stefanradev93/amortized-fractals/blob/069d5dc5943dbe85505849f2fa8bfb563617ff97/mmar_univariate.ipynb) · "
        "[Resolution sensitivity](https://github.com/stefanradev93/amortized-fractals/blob/069d5dc5943dbe85505849f2fa8bfb563617ff97/mmar_sensitivity.ipynb) · "
        "[Full amortized-fractals repository](https://github.com/stefanradev93/amortized-fractals)\n\n"
        "The full repository extends this story to rolling risk, reuse across many assets, "
        "joint markets and signed memory. Those additions require their own assumptions and checks."),
        mo.accordion({"Provenance and scientific scope":mo.md(
            "Model-design credit: Stefan Radev, amortized-fractals, pinned revision "
            + saved_report["model_design"]["upstream_revision"] + ". This workshop uses an independently written "
            "NumPy implementation with bit-identical parity at all six d values and a whole-path rejection stress case. "
            "The source checkpoint is preserved unchanged. Its historical training bank and seeds were not independently "
            "reconstructed; task metadata describe the verified design, not newly recovered historical provenance. "
            f"Checkpoint SHA-256: `{saved_report['checkpoint_sha256']}`. "
            "The reserved test panel and original full draws stay in the lab; compact results and bounded examples "
            "are accepted tutorial assets. No reference posterior or exact-posterior claim is supplied.")})])
    return


if __name__ == "__main__":
    app.run()
