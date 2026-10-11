import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import hashlib
    import json
    import os
    from time import perf_counter

    os.environ.setdefault("KERAS_BACKEND", "jax")

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from sbi4finance import heston_forecast, heston_inference, heston_reference
    from sbi4finance.heston_bank import generate_examples

    ROOT = mo.notebook_dir().parent
    return (
        ROOT,
        generate_examples,
        hashlib,
        heston_forecast,
        heston_inference,
        heston_reference,
        json,
        mo,
        np,
        pd,
        perf_counter,
        plt,
    )


@app.cell
def _(ROOT, json):
    evidence_dir = ROOT / "docs" / "results" / "heston-inference"
    bank_manifest = json.loads((evidence_dir / "bank-manifest.json").read_text())
    test_report = json.loads((evidence_dir / "neural-test.json").read_text())
    reference_report = json.loads((evidence_dir / "reference-test.json").read_text())
    development = json.loads((evidence_dir / "development-history.json").read_text())
    task = bank_manifest["configuration"]["task"]
    evaluation = test_report["evaluation"]
    return development, evaluation, bank_manifest, reference_report, task, test_report


@app.cell
def _(mo):
    mo.md(r"""
    # 5. Infer a persistent parameter and today's hidden state together

    **A working pipeline with a provisional posterior.** This milestone connects
    a Heston simulator, a saved training bank, Bayesian reference inference,
    a neural posterior and conditional forecasts. The shipped checkpoint failed
    its frozen conditional test checks. We inspect that failure rather than
    treating plausible plots or good average coverage as proof of accuracy.

    This notebook runs offline. It reads recorded results, regenerates four
    previously inspected synthetic histories, loads fixed weights and computes
    a small reference for the selected history. It does **not** retrain, submit
    cluster jobs or download market data. The interactive plots are illustrations
    on an existing regression set, not a new holdout evaluation.

    Follow the sequence: **define the observation → distinguish the targets →
    inspect recorded checks → compare a history's joint posterior → continue
    paired states into future losses.**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Key figures and the questions they answer

    These links show the **bounded-time candidate's development validation**;
    the interactive regression plots later in this notebook retain the original
    checkpoint. The reserved final test remains unevaluated.

    | Figure | What it checks |
    |---|---|
    | [Prior versus conditional posteriors](../docs/results/heston-diagnostic-review/prior-posterior-comparison.png) | Whether the observations change inference, and how neural/reference means and SDs compare |
    | [Marginal SBC ranks](../docs/results/heston-diagnostic-review/sbc-rank-ecdf.png) | Aggregate rank calibration for theta and the signed current state |
    | [Conditional reference errors](../docs/results/heston-diagnostic-review/conditional-reference-errors.png) | Means and predictive interval endpoints on 16 particular histories; **not SBC** |
    | [Refined calm-history comparison](../docs/results/heston-precision/case-0-comparison.png) | Remaining state and future-loss differences after increasing sampling/reference precision; **not SBC** |
    | [GRU gate saturation](../docs/results/heston-diagnostic-review/gru-gate-saturation.png) | A representation/scaling diagnosis; **not an inference acceptance gate** |

    **The candidate is learning from the histories, not simply copying the
    prior.** The prior/posterior figure shows different locations and widths
    across all 16 reference histories. The neural posterior generally follows
    those reference changes, especially for the current state. Some theta
    posteriors remain broad: weak identification can be a feature of the data,
    rather than a failure to learn. Bars are means ± one standard deviation,
    **not credible intervals** or markers of synthetic truth.

    **Yes, we perform marginal simulation-based calibration (SBC) diagnostics.**
    For each of 256 independently generated prior/simulator examples, theta is
    drawn from its prior and the simulator jointly produces the observed history
    and true terminal state. The state is not drawn independently of that history.
    We condition the neural posterior on the history and take 2,048 independent
    joint neural draws. For each target separately, its rank counts how many
    draws lie below the simulated truth. Under correct posterior sampling,
    these ranks are uniform across fresh joint generative examples.

    The figure plots the empirical rank CDF minus uniform, using the saved
    finite-draw ranks and exactly the same randomized normalization as the report.
    Its DKW band is a **development diagnostic**: these validation data were
    reused for model selection, so the band is not an independent calibration
    certificate. Marginal SBC cannot establish all joint dependence or accuracy
    conditional on every history. Coverage of a single 90% interval is only one
    aggregate summary; it is not the complete SBC check.
    A sampler that ignores observations and returns the marginal prior can also
    produce uniform marginal ranks: the truth and those draws share the same
    unconditional distribution. This is why we explicitly inspect the prior
    update and conditional reference agreement as well.

    The separate reference comparisons answer those conditional questions.
    Their 0.40 posterior-SD mean-error and 0.15 predictive-SD endpoint limits
    are **declared pilot engineering tolerances**, not universal statistical
    tests. Reference and sampling precision also matter near a limit. Gate
    saturation motivated one focused change; removing it did not eliminate
    the recorded posterior failures. All three checks inform the investigation
    without substituting for one another.

    [Figure source and artifact hashes](../docs/results/heston-diagnostic-review/figures-manifest.json)
    · [Offline reproduction script](../scripts/plot_heston_diagnostics.py)
    · [Diagnostic interpretation and precision review](../docs/heston-diagnostics.md)
    """)
    return


@app.cell
def _(ROOT, hashlib, json, mo):
    _prior = json.loads((ROOT / "docs/results/heston-diagnostic-review/prior-comparison.json").read_text())
    _report_path = ROOT / "docs/results/heston-time-context/neural-validation.json"
    _report = json.loads(_report_path.read_text())
    if (_prior["neural_report_sha256"] != hashlib.sha256(_report_path.read_bytes()).hexdigest()
            or _prior["bank_manifest_sha256"] != _report["bank_manifest_sha256"]):
        raise ValueError("Prior comparison must match the current recorded experiment.")
    mo.md(f"""
    Across these 16 histories and both targets, the reference-standardized mean
    RMSE is **{_prior['prior_copy_reference_mean_rmse_in_sd']:.3f} for an unchanged
    prior mean**, versus **{_prior['neural_reference_mean_rmse_in_sd']:.3f} for the
    neural posterior mean**. That is evidence of substantial data updating,
    not a certificate of full posterior accuracy. The conditional failures
    remain visible in the separate error plot.

    Theta's prior mean and SD are analytic log-uniform moments. The terminal
    state's prior is induced by the known initialization, theta prior and one
    year of dynamics; its mean and SD are estimated from all
    {_prior['prior_sample_count']:,} training simulations, without conditioning
    on their returns. See the [prior comparison record](../docs/results/heston-diagnostic-review/prior-comparison.json).
    """)
    return


@app.cell
def _(ROOT, mo):
    mo.image(ROOT / "docs/results/heston-diagnostic-review/prior-posterior-comparison.png")
    return


@app.cell
def _(mo, task):
    mo.md(rf"""
    ## What is one observation?

    One example is an **ordered history of {task['n_observations']} decimal log
    returns** $r_t=\log(S_t/S_{{t-1}})$, with array shape `[252, 1]`.
    A batch of $B$ examples has shape `[B, 252, 1]`. The entire history is the
    observation supplied to inference; individual days are not independent
    training examples.

    Here $\Delta=1/252$ is a year fraction: 252 modeled trading days span one
    year. A log return of 0.01 corresponds to a simple return
    $e^{{0.01}}-1\approx1.005\%$. No missing days, measurement errors or jumps
    are modeled in this restricted task.

    The **physical measure $P$** describes historical dynamics and future
    outcomes. These are not risk-neutral $Q$ parameters calibrated from option
    quotes. A $P$ posterior cannot be passed directly to an option pricer as a
    $Q$ calibration.

    | Quantity | Meaning | This task |
    |---|---|---|
    | $\theta$ | Persistent long-run variance | Unknown; log-uniform prior from 0.02 to 0.08 |
    | $a_T$ | Current signed numerical continuation state | Unknown; inferred jointly with $\theta$ |
    | $a_0$ | Known initial variance | 0.04 |
    | $\mu,\kappa,\xi,\rho$ | Growth, mean reversion, volatility of variance, shock correlation | Fixed at 0.05, 2, 0.2, −0.6 |

    Variance 0.04 corresponds to annualized volatility $\sqrt{{0.04}}=20\%$.
    The prior is uniform in $\log\theta$, not in $\theta$: its density is
    $p(\theta)=1/[\theta\log(0.08/0.02)]$ within the interval.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why distinguish the parameter from the state?

    The parameter $\theta$ controls where variance tends to revert over time.
    The state $a_T$ tells the numerical model where to start tomorrow. A calm
    recent week and a volatile preceding year can convey different information
    about these two quantities.

    This simulator uses **full-truncation Euler**, one update per observed day:

    $$v_t=\max(a_t,0),\qquad
    r_{t+1}=(\mu-v_t/2)\Delta+\sqrt{v_t\Delta}\,\varepsilon^S_t,$$
    $$a_{t+1}=a_t+\kappa(\theta-v_t)\Delta
                 +\xi\sqrt{v_t\Delta}\,\varepsilon^v_t,$$

    where the two standard-normal shocks have correlation $\rho$.
    The auxiliary state $a_t$ can be negative even though effective variance
    $v_t$ cannot. **Keep the signed state for continuation.** Clipping it first
    changes the next transition. This is a property of the declared numerical
    model, not a claim that economic variance is negative.

    The inference target is $p(\theta,a_T\mid r_{1:T})$ for this daily discrete
    model. Continuous-time approximation checks are a separate question.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## From the simulator to the posterior

    During bank generation, draw $\theta$ from its prior and simulate a whole
    history. Save its observed returns separately from the synthetic targets
    $(\theta,a_T)$. They are known during simulation, but unavailable when
    inferring from an ordinary history.

    ```python
    examples = generate_examples(config, "train", example_ids)
    labelled = heston_inference.training_data(examples, task)
    conditions = heston_inference.conditioning_data(
        examples["observations"], examples["dt_years"], task
    )
    ```

    Only `conditions` are passed when sampling a posterior. A GRU reads the
    ordered returns. Eight additional observed features summarize trailing
    8-, 32-, 128- and 252-day windows: mean log return and mean squared log
    return divided by $\Delta$. They are helpful features, **not claimed
    sufficient statistics**. The full history remains available.

    Scaling is learned globally from training data. Standardizing every history
    separately would erase much of the volatility evidence. Theta is mapped to
    the real line with a bounded logit; the signed state is not logged or clipped.
    A two-dimensional flow returns **matched** theta/state draws.

    Our reference instead combines quadrature over $\log\theta$ with a particle
    filter for the hidden state. It uses the same return/state shock correlation
    and daily transition. This is likelihood-based inference; SBI is useful for
    amortization, not mathematically required. The reference also has finite-grid
    and finite-particle error, checked by refinement and independent repeats.
    """)
    return


@app.cell
def _(evaluation, mo, pd):
    _names = {
        "mean_rmse_in_reference_sd": "Posterior mean RMSE / reference SD",
        "maximum_abs_mean_error_in_reference_sd": "Largest absolute mean error / reference SD",
        "maximum_relative_std_error": "Largest relative posterior SD error",
        "maximum_correlation_error": "Largest correlation error",
        "maximum_predictive_std_relative_error": "Largest relative future-loss SD error",
        "maximum_predictive_interval_error_in_sd": "Largest predictive interval endpoint error / SD",
    }
    gate_table = pd.DataFrame([
        {"Check": _names[key], "Frozen test": round(evaluation["metrics"][key], 4),
         "Maximum": limit,
         "Result": "PASS" if evaluation["metrics"][key] <= limit else "FAIL"}
        for key, limit in evaluation["thresholds"].items()
    ])
    mo.vstack([
        mo.md("""
        ## Recorded outcome: conditional checks failed

        These numbers come from the committed **frozen test report**, not the
        interactive samples below. The checkpoint was selected on validation
        data, frozen, then tested. The same predeclared gates were retained.
        Means and dependence concern four reference histories; the broader
        marginal diagnostics concern 256 histories with 2,048 draws each.
        """),
        mo.ui.table(gate_table, selection=None, show_column_summaries=False),
    ])
    return


@app.cell
def _(evaluation, np, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained")
    _coverage = [*evaluation["coverage_90"], evaluation["predictive_loss_interval_90_coverage"]]
    _axes[0].bar(["theta", "signed state", "future loss"], _coverage, color="#4d749b")
    _axes[0].axhline(0.90, color="black", linestyle="--", label="nominal 90%")
    _axes[0].set(ylim=(0, 1), ylabel="Coverage over 256 histories", title="Average coverage looks reasonable")
    _axes[0].legend(loc="lower right")
    _errors = np.array([r["mean_error_in_reference_sd"] for r in evaluation["reference_cases"]])
    _x = np.arange(len(_errors))
    _axes[1].bar(_x - 0.18, _errors[:, 0], width=0.36, label="theta", color="#4d749b")
    _axes[1].bar(_x + 0.18, _errors[:, 1], width=0.36, label="signed state", color="#c06b48")
    _limit = evaluation["thresholds"]["maximum_abs_mean_error_in_reference_sd"]
    _axes[1].axhspan(-_limit, _limit, color="#eeeeee", zorder=0)
    _axes[1].axhline(0, color="black", linewidth=0.8)
    _axes[1].set(xticks=_x, xticklabels=[f"test:{i}" for i in _x],
                 ylabel="Signed mean error / reference SD", title="Specific histories reveal errors")
    _axes[1].legend()
    _fig
    return


@app.cell
def _(evaluation, mo, reference_report):
    _ref_change = max(r["coarse_to_fine"]["maximum_mean_difference_in_sd"]
                      for r in reference_report["cases"])
    _repeat_change = max(r["independent_fine_repeat"]["maximum_mean_difference_in_sd"]
                         for r in reference_report["cases"])
    mo.md(f"""
    The marginal rank checks also passed, but neither rank checks nor average
    coverage guarantee accuracy for every observation or for joint dependence.
    The shaded band marks the ±0.40 SD individual-mean gate; the separate mean
    RMSE gate is 0.20 SD. A posterior can cover the truth often while having
    biased conditional means or unnecessarily wide intervals.

    The reference's largest coarse-to-fine mean change was **{_ref_change:.3f} SD**;
    its largest independent fine-repeat change was **{_repeat_change:.3f} SD**.
    Those checks are smaller than the largest neural error
    (**{evaluation['metrics']['maximum_abs_mean_error_in_reference_sd']:.3f} SD**),
    but do not make the reference exact. Coverage at this sample size has about
    **1.9 percentage points** nominal binomial standard error.

    **Acceptance remains open.** Software checks, numerical refinement and
    neural accuracy are distinct. Revisiting these test histories does not make
    them a fresh holdout; further model development needs a reserved independent
    evaluation set.
    """)
    return


@app.cell
def _(mo):
    _intro = mo.md("""
    ## Inspect one previously tested history

    Switch among all four reference cases. The saved error table remains the
    historical result. The following distributions use newly drawn illustrative
    samples from the **same provisional checkpoint** and a recomputed reference.
    They may differ slightly from the recorded aggregate because of Monte Carlo
    sampling and platform arithmetic; they do not replace its acceptance decision.
    """)
    case_selector = mo.ui.dropdown(
        options=["test:0", "test:1", "test:2", "test:3"],
        value="test:0", label="Previously inspected case",
    )
    mo.vstack([_intro, case_selector])
    return (case_selector,)


@app.cell
def _(bank_manifest, generate_examples):
    # Stable IDs regenerate only four histories, not the 73 MB training bank.
    illustration_examples = generate_examples(bank_manifest["configuration"], "test", range(4))
    return (illustration_examples,)


@app.cell
def _(ROOT, hashlib, heston_inference, json, task, test_report):
    checkpoint_dir = ROOT / "checkpoints" / "heston-theta-state-provisional"
    _design = json.loads((checkpoint_dir / "task.json").read_text())
    if _design != {"task": task, "adapter_id": heston_inference.ADAPTER_ID}:
        raise ValueError("The illustration requires the shipped task and adapter.")
    if hashlib.sha256((checkpoint_dir / "model.keras").read_bytes()).hexdigest() != test_report["checkpoint_sha256"]:
        raise ValueError("The checkpoint differs from the frozen report.")
    frozen_workflow = heston_inference.build_workflow(task)
    frozen_workflow.load_approximator(checkpoint_dir / "model.keras")
    return (frozen_workflow,)


@app.cell
def _(frozen_workflow, heston_inference, illustration_examples, np, perf_counter, task):
    _started = perf_counter()
    illustration_conditions = heston_inference.conditioning_data(
        illustration_examples["observations"], illustration_examples["dt_years"], task
    )
    _samples = frozen_workflow.sample(num_samples=2048, conditions=illustration_conditions, seed=12071)
    illustration_joint = np.concatenate([_samples["theta"], _samples["terminal_state"]], axis=-1)
    _low, _high = task["theta_bounds"]
    if (not np.isfinite(illustration_joint).all()
            or np.any(illustration_joint[..., 0] <= _low)
            or np.any(illustration_joint[..., 0] >= _high)):
        raise ValueError("Invalid joint draws: no clipping or selective redrawing is allowed.")
    sample_seconds = perf_counter() - _started
    return illustration_joint, sample_seconds


@app.cell
def _(case_selector, heston_reference, illustration_examples, np, perf_counter, reference_report, task):
    selected_case = int(case_selector.value.split(":")[1])
    _setting = reference_report["cases"][selected_case]["runs"][-1]
    _started = perf_counter()
    selected_reference = heston_reference.reference_posterior(
        illustration_examples["observations"][selected_case], illustration_examples["dt_years"], task,
        grid_size=_setting["grid_size"], particles=_setting["particles"], seed=_setting["seed"],
    )
    reference_seconds = perf_counter() - _started
    # Choose theta and terminal state from the SAME grid row.
    _rng = np.random.default_rng(12072 + selected_case)
    _rows = _rng.choice(len(selected_reference["theta_grid"]), size=2048,
                        p=selected_reference["grid_probabilities"])
    _columns = _rng.integers(selected_reference["terminal_particles"].shape[1], size=2048)
    reference_joint = np.column_stack([
        selected_reference["theta_grid"][_rows], selected_reference["terminal_particles"][_rows, _columns]
    ])
    return reference_joint, reference_seconds, selected_case, selected_reference


@app.cell
def _(illustration_examples, np, plt, selected_case, task):
    _fig, _ax = plt.subplots(figsize=(10, 2.8), layout="constrained")
    _ax.plot(np.arange(1, task["n_observations"] + 1) * task["dt_years"],
             100 * illustration_examples["observations"][selected_case, :, 0], linewidth=0.8)
    _ax.axhline(0, color="black", linewidth=0.6)
    _ax.set(xlabel="Time from start (trading-year fractions)", ylabel="Daily log return (%)",
            title=f"test:{selected_case}: all 252 ordered returns are observed")
    _fig
    return


@app.cell
def _(evaluation, mo, pd, selected_case):
    _record = evaluation["reference_cases"][selected_case]
    mo.vstack([
        mo.md("**Recorded frozen errors for this case** (separate from the illustrative samples):"),
        mo.ui.table(pd.DataFrame({
            "Target": ["theta", "signed state"],
            "Mean error / reference SD": _record["mean_error_in_reference_sd"],
            "Relative SD error": _record["relative_std_error"],
        }), selection=None, show_column_summaries=False),
    ])
    return


@app.cell
def _(illustration_examples, illustration_joint, np, plt, reference_joint, selected_case, task):
    _neural = illustration_joint[selected_case]
    _truth = np.array([illustration_examples["parameters"][selected_case, 0],
                       illustration_examples["terminal_state"][selected_case, 0]])
    _fig, _axes = plt.subplots(1, 3, figsize=(11, 3.6), layout="constrained")
    for _i, _name in enumerate(["Long-run variance theta", "Signed terminal state"]):
        _bins = np.linspace(min(_neural[:, _i].min(), reference_joint[:, _i].min()),
                            max(_neural[:, _i].max(), reference_joint[:, _i].max()), 28)
        _axes[_i].hist(reference_joint[:, _i], bins=_bins, density=True, histtype="step",
                       linewidth=1.7, color="#333333", label="particle/grid reference")
        _axes[_i].hist(_neural[:, _i], bins=_bins, density=True, alpha=0.45,
                       color="#4d749b", label="provisional neural")
        _axes[_i].axvline(_truth[_i], color="#c06b48", linestyle="--", label="synthetic truth")
        _axes[_i].set(xlabel=_name, ylabel="Density")
    _theta = np.linspace(*task["theta_bounds"], 200)
    _axes[0].plot(_theta, 1 / (_theta * np.log(task["theta_bounds"][1] / task["theta_bounds"][0])),
                  linestyle=":", color="#71894a", label="theta prior")
    _axes[0].legend(fontsize=7)
    _axes[2].scatter(reference_joint[:500, 0], reference_joint[:500, 1], s=7, alpha=0.25,
                     color="#333333", label="reference")
    _axes[2].scatter(_neural[:500, 0], _neural[:500, 1], s=7, alpha=0.25,
                     color="#4d749b", label="provisional neural")
    _axes[2].scatter(*_truth, marker="x", s=70, color="#c06b48")
    _axes[2].set(xlabel="theta", ylabel="Signed terminal state", title="Keep these draws paired")
    _axes[2].legend(fontsize=7)
    _fig
    return


@app.cell
def _(mo, reference_seconds, sample_seconds):
    mo.md(f"""
    Synthetic truth is shown **only for evaluation**; it was not an input to the
    neural or reference posterior. A credible interval need not center on the
    truth for a single random history. Reference agreement and coverage assess
    different properties.

    The theta reference is a finite quadrature distribution, so its histogram
    can look jagged. The reference is recomputed from observed returns using
    48 grid nodes and 4,096 particles per node. This selected run took
    **{reference_seconds:.2f} s**; neural sampling across all four histories took
    **{sample_seconds:.2f} s**, including any compilation. These unequal workloads
    are a notebook runtime check, not a speed comparison.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Continue the joint posterior into future losses

    For every matched pair $(\theta^{(s)},a_T^{(s)})$, draw independent future
    shocks and continue the same discrete law for ten trading days. Do not
    restart from $a_0$, replace the state by $\theta$, or independently shuffle
    the parameter and state samples.

    $$p(L\mid r_{1:T})=\int p(L\mid\theta,a_T)\,
       p(\theta,a_T\mid r_{1:T})\,d\theta\,da_T,$$
    $$L=1-\exp\!\left(\sum_{t=T+1}^{T+10}r_t\right).$$

    A positive $L$ means a loss on a long position; a negative $L$ means a gain.
    These are fractions of initial wealth, with no currency or position size.
    The central 90% interval illustrates the predictive mixture. Its quantiles
    are not averages of separate conditional quantiles. Two thousand draws are
    useful for this illustration, not a validated 99% tail-risk estimate.
    """)
    return


@app.cell
def _(heston_forecast, illustration_joint, np, plt, reference_joint, selected_case, task):
    # Shared fresh shocks reduce noise in comparing distributions. Both inputs
    # are posterior pairs; neither uses the bank's true parameter/state labels.
    _neural_losses = heston_forecast.simulate_futures(
        illustration_joint[selected_case], task, seed=12073 + selected_case
    )["relative_loss"]
    _reference_losses = heston_forecast.simulate_futures(
        reference_joint, task, seed=12073 + selected_case
    )["relative_loss"]
    _fig, _axes = plt.subplots(1, 2, figsize=(10.5, 3.3), layout="constrained")
    _bins = np.histogram_bin_edges(100 * np.concatenate([_neural_losses, _reference_losses]), bins=40)
    _axes[0].hist(100 * _reference_losses, bins=_bins, density=True, histtype="step",
                  linewidth=1.7, color="#333333", label="reference predictive")
    _axes[0].hist(100 * _neural_losses, bins=_bins, density=True, alpha=0.45,
                  color="#4d749b", label="provisional neural predictive")
    _axes[0].set(xlabel="Ten-day loss (% of initial wealth)", ylabel="Density")
    _axes[0].legend(fontsize=8)
    for _j, (_losses, _label, _color) in enumerate([
        (_reference_losses, "reference", "#333333"), (_neural_losses, "provisional neural", "#4d749b")
    ]):
        _q = 100 * np.quantile(_losses, [0.05, 0.50, 0.95])
        _axes[1].plot(_q[[0, 2]], [_j, _j], linewidth=4, color=_color)
        _axes[1].plot(_q[1], _j, "o", color=_color)
    _axes[1].set(yticks=[0, 1], yticklabels=["reference", "provisional neural"],
                 xlabel="Ten-day loss (%)", title="Illustrative central 90% interval", ylim=(-0.6, 1.6))
    _axes[1].axvline(0, color="black", linewidth=0.6)
    _fig
    return


@app.cell
def _(development, mo, pd):
    _trials = pd.DataFrame([
        {"Trial": r["trial"], "Train histories": r["train_histories"],
         "Validation mean RMSE / SD": round(r["validation_metrics"]["mean_rmse_in_reference_sd"], 3),
         "Validation passed": r["validation_passed"], "Change": r["development_decision"]}
        for r in development["trials"]
    ])
    mo.vstack([
        mo.md("""
        ## What this milestone established

        The bank-to-inference-to-forecast path is operational. Parameter/state
        pairing, target separation, global scaling and checkpoint reload have
        software tests. An independently computed reference enables meaningful
        conditional checks. The checkpoint itself remains provisional.

        The recorded development history explains the changes; all occurred on
        validation data before the frozen test. This notebook does not overwrite
        those records or select a better-looking subset of cases.
        """),
        mo.ui.table(_trials, selection=None, show_column_summaries=False),
        mo.md("""
        Moving from 16 to 64 flow integration steps changed conditional means
        by less than 3×10⁻⁷ reference SD in the recorded solver probe. That did
        not explain the learned-posterior error. A broader validation reference
        set and a new reserved holdout are appropriate for another inference
        iteration. No market-data acquisition is needed for this synthetic task.
        """),
    ])
    return


@app.cell
def _(ROOT, json, mo, pd):
    _followup = json.loads((ROOT / "docs/results/heston-inference-followup/neural-validation.json").read_text())
    _checks = _followup["evaluation"]
    _table = pd.DataFrame([
        {"Validation metric": _key, "Result": round(_checks["metrics"][_key], 4),
         "Maximum": _limit,
         "Gate": "PASS" if _checks["metrics"][_key] <= _limit else "FAIL"}
        for _key, _limit in _checks["thresholds"].items()
    ])
    mo.vstack([
        mo.md(r"""
        ## Follow-up: a different density estimator still needs better validation results

        After the failure above, we tested a spline coupling flow while keeping
        the same simulator, two targets, GRU and observed features. It learns by
        conditional density training; the original flow-matching objective
        samples auxiliary noise and times. Their loss values cannot be compared
        directly as measures of posterior quality.

        A new bank used root seed **43022**, and the validation reference set
        expanded from four to **16 preselected histories**. This was a development
        experiment, not a controlled claim about the effect of architecture alone.
        The same error limits were retained. The candidate **failed validation**:
        """),
        mo.ui.table(_table, selection=None, show_column_summaries=False),
        mo.md(r"""
        The current-state rank check also failed narrowly. **The new final test
        split was not evaluated.** Its files were integrity-checked, but its
        observations and truths did not enter loss, diagnostics or model selection.
        The wider validation set caught this candidate's failure before another
        final test was consumed.

        The interactive plots above deliberately retain the original checkpoint
        and original regression cases. They do not visualize this new candidate
        or replace either report. See the
        [follow-up design and results](../docs/heston-inference-followup.md).
        M3.3 inference acceptance remains pending; the next investigation can
        stay within the existing pipeline and the reserved holdout can remain
        untouched until a candidate passes validation.
        """),
    ])
    return


@app.cell
def _(ROOT, json):
    _directory = ROOT / "docs/results/heston-time-context"
    legacy_time_diagnosis = json.loads((_directory / "summary-legacy.json").read_text())
    explicit_time_diagnosis = json.loads((_directory / "summary-explicit-time.json").read_text())
    explicit_time_validation = json.loads((_directory / "neural-validation.json").read_text())
    spline_baseline_validation = json.loads(
        (ROOT / "docs/results/heston-inference-followup/neural-validation.json").read_text()
    )
    return (
        explicit_time_diagnosis,
        explicit_time_validation,
        legacy_time_diagnosis,
        spline_baseline_validation,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Focused experiment: put known time on a sensible scale

    The preceding spline failure motivated a smaller question: **how is time
    represented inside the recurrent summary?** A GRU reads the sequence one
    step at a time, using gates to control how previous information combines
    with the current input. BayesFlow also supplies a learned Time2Vec embedding.
    Its linear component has the form $w t+b$, alongside periodic components.
    Without an explicit time channel, it receives positions $0,1,\ldots,251$.
    A growing time component can dominate globally standardized return inputs.

    A sigmoid gate converts its input $z$ to

    $$\sigma(z)=\frac{1}{1+e^{-z}},\qquad
      \sigma'(z)=\sigma(z)[1-\sigma(z)].$$

    At $z=0$, the derivative is $1/4$. At $z=\pm5$, the output is already
    approximately 0.007 or 0.993 and the derivative is about 0.0066. Large
    absolute inputs therefore put the gate near an endpoint, where further
    changes have less effect. This motivates inspecting **gate preactivations**
    (the inputs before the sigmoid), not merely increasing training epochs.
    Saturation can be useful for memory; a high fraction alone neither proves
    a bug nor explains every posterior error.

    The bounded-time candidate instead supplies the known observation end times

    $$t_j=\sum_{i=1}^{j}\Delta_i,\qquad
      \widetilde t_j=\frac{t_j-\overline t_{\rm train}}{s_{t,\rm train}}.$$

    For this fixed 252-day grid, $t_j$ runs from $1/252$ to 1 year and global
    training scaling puts it approximately between −1.73 and 1.73. The
    recurrent network uses this channel via `time_axis=1`. Scaling statistics
    are fitted on training inputs and frozen for validation/application;
    returns are not normalized separately within each history.
    Centering and scaling integer positions would give the same values on this
    fixed grid; explicit end times also make the units and convention clear.

    **Nothing new has been observed.** The bank still stores `[B, T, 1]` decimal
    log returns. Adapter v2 constructs `[return, end_time]`, shape `[B, T, 2]`,
    for the network only. The known grid supplies time; neither latent variance,
    theta, nor future outcomes enter these conditions. Return amplitudes and
    the eight trailing moments remain the same. This task is still restricted
    to its saved fixed grid; explicit time does not establish support for
    arbitrary lengths or irregular observations.
    """)
    return


@app.cell
def _(explicit_time_diagnosis, legacy_time_diagnosis, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True, layout="constrained")
    for _axis, _gate in zip(_axes, ("update", "reset")):
        for _report, _label, _color in [
            (legacy_time_diagnosis, "Raw-index spline", "#4d749b"),
            (explicit_time_diagnosis, "Bounded-time spline", "#c06b48"),
        ]:
            _records = _report["gate_diagnostics"]
            _axis.plot([r["time_index"] + 1 for r in _records],
                       [100 * r[_gate]["fraction_abs_preactivation_above_5"] for r in _records],
                       marker="o", color=_color, label=_label)
        _axis.set(xlabel="Observed day", title=f"GRU {_gate} gate",
                  xticks=[1, 32, 128, 252], ylim=(-2, 102))
        _axis.legend(fontsize=8)
    _axes[0].set_ylabel("Preactivations with |z| > 5 (%)")
    _fig
    return


@app.cell
def _(explicit_time_diagnosis, legacy_time_diagnosis, mo):
    _old_error = legacy_time_diagnosis["manual_gru_hidden_state_max_abs_error"]
    _new_error = explicit_time_diagnosis["manual_gru_hidden_state_max_abs_error"]
    mo.md(f"""
    These are **saved-checkpoint diagnostics on the same 16 validation
    histories**, sampled at four sequence positions. The percentages pool
    histories and hidden units. Lines connect the recorded positions; the
    notebook does not rerun activation tracing.

    The final learned linear time component changes from
    **{legacy_time_diagnosis['final_embedded_time_trend']:.2f}** to
    **{explicit_time_diagnosis['final_embedded_time_trend']:.2f}**.
    The independently reconstructed GRU hidden states match their respective
    saved Keras layers to maximum absolute errors of {_old_error:.2g} and
    {_new_error:.2g}. This checks the diagnostic calculation. The two trained
    networks have different learned weights, so this is a before/after training
    comparison, not an isolated intervention on one gate. Posterior accuracy
    must still be evaluated separately.
    """)
    return


@app.cell
def _(explicit_time_validation, mo, pd, spline_baseline_validation):
    for _fingerprint in ("bank_manifest_sha256", "reference_report_sha256"):
        if explicit_time_validation[_fingerprint] != spline_baseline_validation[_fingerprint]:
            raise ValueError("The time-context comparison requires identical bank and reference reports.")
    _baseline = spline_baseline_validation["evaluation"]
    _bounded = explicit_time_validation["evaluation"]
    if _baseline["thresholds"] != _bounded["thresholds"]:
        raise ValueError("The time-context comparison requires unchanged validation thresholds.")
    _names = {
        "mean_rmse_in_reference_sd": "Posterior mean RMSE / reference SD",
        "maximum_abs_mean_error_in_reference_sd": "Largest absolute mean error / reference SD",
        "maximum_relative_std_error": "Largest relative posterior SD error",
        "maximum_correlation_error": "Largest correlation error",
        "maximum_predictive_std_relative_error": "Largest relative future-loss SD error",
        "maximum_predictive_interval_error_in_sd": "Largest predictive interval endpoint error / SD",
    }
    _rows = [
        {"Validation check": _names[k], "Raw-index spline": round(_baseline["metrics"][k], 4),
         "Bounded-time spline": round(_bounded["metrics"][k], 4), "Maximum": limit,
         "Bounded-time result": "PASS" if _bounded["metrics"][k] <= limit else "FAIL"}
        for k, limit in _bounded["thresholds"].items()
    ]
    for _i, _name in enumerate(("theta", "signed state")):
        _limit = _bounded["rank_dkw_bound_two_targets"]
        if _limit != _baseline["rank_dkw_bound_two_targets"]:
            raise ValueError("The time-context comparison requires the same rank-check bound.")
        _value = _bounded["rank_uniformity_max_ecdf_error"][_i]
        _rows.append({"Validation check": f"Marginal rank ECDF error: {_name}",
                      "Raw-index spline": round(_baseline["rank_uniformity_max_ecdf_error"][_i], 4),
                      "Bounded-time spline": round(_value, 4), "Maximum": round(_limit, 4),
                      "Bounded-time result": "PASS" if _value <= _limit else "FAIL"})
    _outcome = "PASSED" if _bounded["pilot_passed"] else "FAILED"
    mo.vstack([
        mo.md(f"""
        ### Did the conditional posterior improve?

        The bounded-time candidate **{_outcome} its recorded validation
        checks**. Both spline runs use the same root-43022 bank, 32,768 training
        histories, 2,048 validation histories, GRU32/summary16, eight trailing
        features, seed 871, 80 epochs and batch size 256. Each selects minimum
        validation loss. The intended change is the time-context adapter.

        Conditional comparisons use the same first 16 preselected reference
        histories and unchanged thresholds. Marginal rank checks compare the
        rank empirical CDF with uniformity over 256 histories. They use the same
        simultaneous DKW bound for the two targets. Errors below their individual
        limits are necessary checks, not guarantees for every possible history.
        """),
        mo.ui.table(pd.DataFrame(_rows), selection=None, show_column_summaries=False),
        mo.md("""
        **Validation failed, so the reserved final test remains unevaluated.**
        Both failed limits arise on `validation:0`, a low-current-variance
        history: the signed-state mean error is about 0.455 reference SD and
        the upper predictive-interval endpoint error is about 0.210 predictive
        SD. Their limits remain 0.40 and 0.15. Mean RMSE and both marginal rank
        checks pass, but cannot override these conditional failures.

        The recorded gate-saturation fractions are now zero at the four
        inspected positions. That confirms a change in representation behavior,
        while the remaining posterior errors demonstrate its limited evidential
        scope. No additional training or threshold changes follow this result;
        M3.3 inference acceptance remains pending.

        This is a **validation-based development comparison**. Improved gate
        behavior alone cannot certify a posterior; validation success alone
        cannot substitute for a frozen held-out test. The predeclared protocol
        allows the reserved root-43022 test only after validation passes, with
        separately checked references and no subsequent retuning on that test.
        The earlier interactive plots remain the original checkpoint's
        historical regression cases; they do not switch to this candidate.
        """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Exercise: scale, information and evidence.** Multiplying every time in
    this fixed grid by 252 expresses years as trading days. Show algebraically
    that global centering and scaling gives the same $\widetilde t$ in either
    unit. Why is this unlike normalizing every return history to unit variance?
    If a new network has less saturated gates but misses a posterior-mean gate,
    which result determines the inference acceptance decision?

    **Review hint:** multiplying time, its mean and its standard deviation by
    the same positive constant cancels in the ratio. Per-history return scaling
    can remove between-history volatility information. The numerical diagnostic
    supports a mechanism to investigate; it does not override failed posterior
    checks. No inference result here establishes empirical adequacy on real data.

    Inspect the [bounded-time design and outcome](../docs/heston-time-context.md),
    [v1/v2 adapter and recurrent network](../src/sbi4finance/heston_inference.py),
    [activation diagnostic script](../scripts/diagnose_heston_summary.py),
    [training/evaluation script](../scripts/train_heston.py),
    [raw-index diagnostic](../docs/results/heston-time-context/summary-legacy.json),
    [bounded-time diagnostic](../docs/results/heston-time-context/summary-explicit-time.json),
    and [recorded validation](../docs/results/heston-time-context/neural-validation.json).
    Those records identify the checkpoints and source hashes. The design lists
    the explicit reproduction command; opening this notebook only reads the
    saved comparison. Original checkpoints keep their v1 adapter and weights.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Does the remaining error survive a more precise calculation?

    We froze the latest bounded-time checkpoint and increased neural draws
    from 2,048 to 16,384 on all 16 validation histories, with three prespecified
    seeds. We also refined the calm-history reference twice from 48 grid nodes
    and 4,096 particles to 96 nodes and 16,384 particles. This is a **conditional
    precision diagnostic, not new SBC**. No weights, thresholds or reserved
    test observations changed.

    Across repeats, overall mean RMSE remains 0.171–0.177 reference SD. Against
    the second refined reference, the calm history's state mean is still high
    by 0.412–0.435 reference SD, with neural-mean sampling SE about 0.0096 SD.
    The two refined reference state means differ by 0.007 SD. Thus the nonzero
    discrepancy persists; these calculations do not establish statistically
    certain rejection at exactly the 0.40 engineering boundary. The original
    failed gate remains recorded without retrospective changes.

    Predictive endpoint errors are more sensitive to finite draws and straddle
    the 0.15 tolerance. For the first prespecified repeat shown below, the
    ten-step 95% loss quantile is **2.96% versus 2.74% of initial wealth**:
    a 0.22 percentage-point difference. Predictive SD is 8.4% larger. Those
    effects describe one synthetic history; they are not a market-risk claim.

    The checkpoint clearly learns from the observations, but its conditional
    accuracy still needs work. This is a restricted two-target Heston pilot,
    with other parameters fixed, not full-SVJ acceptance. Passing marginal SBC
    alone would not have answered the prior-copy or conditional-error questions.

    [Protocol, results and reproduction](../docs/heston-diagnostics.md)
    · [Complete precision report](../docs/results/heston-precision/report.json)
    · [Frozen diagnostic script](../scripts/check_heston_precision.py)
    """)
    return


@app.cell
def _(ROOT, hashlib, json, mo):
    _directory = ROOT / "docs/results/heston-precision"
    _precision = json.loads((_directory / "report.json").read_text())
    _arrays = _directory / _precision["plot_arrays"]["file"]
    if hashlib.sha256(_arrays.read_bytes()).hexdigest() != _precision["plot_arrays"]["sha256"]:
        raise ValueError("Precision plot arrays must match the recorded diagnostic.")
    mo.image(_directory / "case-0-comparison.png")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercises and further inspection

    1. Compare `test:0` and `test:3`. Which target has the largest recorded
       mean error, and does acceptable average coverage expose that failure?
    2. In the feature code, double every observed return. The mean features
       should double and squared-return features should quadruple. Why would
       standardizing each window to unit variance hide useful evidence?
    3. Explain why shuffling only the state column of the joint draws changes
       the forecasting calculation even when both marginal histograms stay
       unchanged. Use the scatter plot to identify the information being lost.
    4. Why is a repeat on these four histories a regression check rather than
       a new out-of-sample accuracy claim?
    5. Why can an algorithm that returns the prior pass marginal SBC? Which
       plot here rules out that behavior on the inspected histories, and why
       does it still not prove full posterior accuracy?

    **Review hints:** the state error on `test:0` is about 0.88 reference SD;
    marginal coverage averages across histories. Volatility scale informs both
    targets. Shuffling destroys their posterior dependence. Once inspected
    during development, these cases cannot provide untouched holdout evidence.
    Under the joint generative model, truth and independent prior draws have
    exchangeable marginal ranks. The prior/posterior plot instead checks that
    inference responds to each history; conditional reference checks still
    reveal approximation errors after that response is established.

    The expensive pipeline is explicit and separate from notebook execution:

    ```bash
    pixi run heston-bank --out /tmp/heston-bank-new
    pixi run heston-reference --bank /tmp/heston-bank-new \
      --out /tmp/heston-reference-validation-new
    pixi run heston-train --bank /tmp/heston-bank-new \
      --references /tmp/heston-reference-validation-new \
      --out /tmp/heston-trained-new --epochs 80 --batch-size 256
    ```

    Output directories must be new. These commands reproduce the task; they do
    not reserve a new holdout automatically. Use the guide's frozen-evaluation
    procedure and a deliberately independent bank identity before another
    scientific acceptance claim.

    Continue with the [joint-inference guide](../docs/heston-joint-inference.md),
    [simulator plan](../docs/simulator-plan/README.md),
    [inference and prediction notes](../docs/simulator-plan/notes/03-inference-and-pricing.md),
    [frozen test report](../docs/results/heston-inference/neural-test.json), or
    [checkpoint boundary](../checkpoints/heston-theta-state-provisional/README.md).
    Direct implementation artifacts: [task configuration](../configs/tasks/heston_inference.json),
    [bank manifest](../docs/results/heston-inference/bank-manifest.json),
    [reference source](../src/sbi4finance/heston_reference.py), and
    [network/adapter source](../src/sbi4finance/heston_inference.py).
    """)
    return


if __name__ == "__main__":
    app.run()
