import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import importlib.util
    import json

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm
    from sbi4finance import rough_reference

    ROOT = mo.notebook_dir().parent
    _spec = importlib.util.spec_from_file_location("rough_review_plots", ROOT / "scripts/plot_rough_inference.py")
    plot_helpers = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(plot_helpers)
    return ROOT, json, mo, norm, np, plot_helpers, plt, rough_reference


@app.cell
def _(mo):
    mo.md(r"""
    # 9. Rough Bergomi: from an option bank to a checked posterior

    **Review question:** can one small network recover the posterior for two
    rough-Bergomi parameters from nine noisy option prices, and preserve a
    downstream mean-price calculation?

    Notebook 8 built the forward simulator and checked its numerical behavior.
    This milestone adds a deliberately restricted inference experiment: an
    independent importance-sampling reference, a single joint-density candidate,
    conditional comparisons, marginal simulation-based calibration (**SBC**)
    and excluded-contract mean prices. Each result has its own acceptance gate.
    The original 4,096-row reference passed 19 of 20 cases and stopped before
    neural training. This review now follows one bounded extension to 8,192
    reference rows, preserving the original results below.

    This is the **S synthetic workflow under Q**, the pricing measure. It does
    not infer a physical-measure return model, today's hidden variance, or the
    probability of tomorrow's market losses. The forward-variance curve and
    all market context are known synthetic inputs. The eventual choice between
    a returns-based and an options-based live tutorial remains open.

    Default execution reads saved reports and runs one small scalar teaching
    example. It never generates a rough-Bergomi bank, trains a network, loads
    a checkpoint, reads market fixtures, or evaluates the reserved test set.
    Missing reports remain visibly missing rather than being replaced by
    invented results.
    """)
    return


@app.cell
def _(ROOT, json, plot_helpers):
    protocol = json.loads((ROOT / "configs/rough_reference8192_protocol.json").read_text())
    bank_config = json.loads((ROOT / protocol["bank_config"]).read_text())
    original_reports, _, original_report_paths = plot_helpers.load_reports(ROOT / "docs/results/rough-inference")
    reports, check_arrays, report_paths = plot_helpers.load_reports(ROOT / "docs/results/rough-reference8192")
    if any(_report["experiment_id"] != protocol["experiment_id"] for _report in reports.values()):
        raise ValueError("Follow-up evidence belongs to a different experiment")
    cached_figures = plot_helpers.figures(reports, check_arrays)
    return (
        bank_config,
        cached_figures,
        check_arrays,
        original_report_paths,
        original_reports,
        protocol,
        report_paths,
        reports,
    )


@app.cell
def _(mo, original_reports, reports):
    _rows = []
    _original = original_reports.get("reference")
    if _original:
        _rows.append({"Stage": "Original 4,096-row reference (historical)",
                      "Saved status": f"{sum(case['passed'] for case in _original['cases'])} / {len(_original['cases'])} cases passed; stopped before training"})
    _reference_stopped = reports.get("reference", {}).get("passed") is False
    for _name in ("reference", "training", "evaluation"):
        _report = reports.get(_name)
        _status = ("not run: reference stop rule (includes SBC)" if _report is None and _reference_stopped and _name == "evaluation" else
                   "not run: reference stop rule" if _report is None and _reference_stopped else
                   "not recorded" if _report is None else
                   "reload verified; inference acceptance separate" if _name == "training" and _report.get("same_seed_reload_exact") else
                   "PASS" if _report.get("passed") else "FAIL / unresolved")
        _rows.append({"Stage": "8,192-row follow-up: " + _name, "Saved status": _status})
    if _reference_stopped:
        _cases = reports["reference"]["cases"]
        _failed = [_case["case_id"] for _case in _cases if not _case["passed"]]
        _message = (f"The follow-up stopped at reference validation: {len(_failed)} of {len(_cases)} cases "
                    f"missed at least one gate ({', '.join(_failed)}). The stop rule permits no further automatic doubling. "
                    "Neural training and SBC require every reference gate to pass; software smoke runs do not satisfy these gates.")
    else:
        _message = ("The 8,192-row reference report is not yet present. The original failure remains recorded; no new pass, training, or SBC result is inferred."
                    if "reference" not in reports else
                    "This view displays the saved follow-up, including failed gates. It performs no tuning or final-test evaluation. Reference passage alone does not accept the neural posterior.")
    mo.vstack([
        mo.md("## Current evidence\n\nA delivered simulator, a sufficiently precise reference, and an accepted neural posterior are different achievements."),
        mo.ui.table(_rows, selection=None),
        mo.callout(_message, kind="warn" if _reference_stopped else "info"),
    ])
    return


@app.cell
def _(mo, np, original_reports, plt, protocol, reports):
    _items = [mo.md(r"""
        ## One reference-budget extension, on the same development cases

        Reuse the original 4,096 independently generated rows, then append
        4,096 new independent rows with the declared row IDs and random streams.
        The training bank, 20 conditional cases, likelihood and acceptance
        thresholds remain unchanged. This is **adaptive development after
        inspecting the original failure**, not a new independent validation
        cohort. It leaves the final test reserved.

        The comparison below shows every case. ESS should be at least 800 and the
        largest normalized weight should not exceed 0.005. These weight screens
        are only part of acceptance: parameter and target-price Monte Carlo
        precision and block agreement must also pass. With 8,192 rows, the two
        reference blocks contain 4,096 rows each: the original block and the
        new block. There is no automatic further doubling if a check fails.
        """)]
    _old = original_reports.get("reference")
    _new = reports.get("reference")
    if _old:
        _ids = [case["case_id"] for case in _old["cases"]]
        if _new and [case["case_id"] for case in _new["cases"]] != _ids:
            raise ValueError("Reference comparison requires the same ordered cases")
        _thresholds = protocol["reference"]["gates"]
        for _report in (_old, _new):
            if _report and _report["thresholds"] != _thresholds:
                raise ValueError("Reference comparison requires unchanged thresholds")
        _fig, _axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, layout="constrained")
        _x = np.arange(len(_ids))
        for _axis, _field, _gate, _ylabel in (
            (_axes[0], "ess", _thresholds["min_ess"], "Importance ESS (higher is better)"),
            (_axes[1], "max_weight", _thresholds["max_weight"], "Largest weight (lower is better)"),
        ):
            _axis.axhline(_gate, color="#B33E43", ls="--", lw=1, label="Unchanged gate")
            for _report, _label, _color, _offset in (
                (_old, "Original 4,096 rows", "#738095", -.1),
                (_new, "Follow-up 8,192 rows", "#087F8C", .1),
            ):
                if _report:
                    _axis.plot(_x + _offset, [case["reference"][_field] for case in _report["cases"]],
                               "o-", color=_color, ms=4, lw=.8, label=_label)
            _axis.set_ylabel(_ylabel)
            _axis.axvspan(15.5, 19.5, color="#EDF0F4", zorder=-5)
        _axes[0].legend(fontsize=8)
        _axes[1].set_xticks(_x, [case.replace("validation:", "").replace("stress:", "S") for case in _ids], fontsize=8)
        _axes[1].set_xlabel("Same development panel IDs; shaded S0–S3 are fixed stress panels")
        _fig.suptitle("Reference weight concentration before and after the bounded extension")
        _items.append(_fig)
        _items.append(mo.ui.table([
            {"Case": _case["case_id"], "Original ESS": _case["reference"]["ess"],
             "Follow-up ESS": _new["cases"][_i]["reference"]["ess"] if _new else "not recorded",
             "Original largest weight": _case["reference"]["max_weight"],
             "Follow-up largest weight": _new["cases"][_i]["reference"]["max_weight"] if _new else "not recorded",
             "Original all-gates pass": _case["passed"],
             "Follow-up all-gates pass": _new["cases"][_i]["passed"] if _new else "not recorded"}
            for _i, _case in enumerate(_old["cases"])
        ], selection=None))
        if not _new:
            _items.append(mo.callout("Only the original run is plotted. Follow-up values remain missing until reference.json is published.", kind="info"))
    else:
        _items.append(mo.callout("The original reference report is missing; no before/after comparison is available.", kind="warn"))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What do H and η mean?

    In this convention, a Gaussian Volterra driver is

    $$Z_t=\sqrt{2H}\int_0^t(t-s)^{H-1/2}dW_s^V,
    \qquad \operatorname{Var}(Z_t)=t^{2H}.$$

    Its kernel gives recent Brownian increments especially large influence.
    Smaller $H$ produces a rougher variance driver. The positive variance is

    $$V_t=\xi_0(t)\exp\left(\eta Z_t-\tfrac12\eta^2t^{2H}\right).$$

    The subtraction keeps $E[V_t]=\xi_0(t)$ for the continuum Gaussian driver.
    The hybrid numerical scheme uses its **own discrete variance** in that
    normalization. The known curve here is flat at $\xi_0(t)=0.04$.
    The amplitude $\eta$ controls how strongly the Gaussian driver moves log
    variance; its meaning depends on this driver normalization and year units.

    Under Q, the asset satisfies

    $$d\log S_t=(r-q-V_t/2)dt+\sqrt{V_t}\,dW_t^S,
    \qquad d\langle W^S,W^V\rangle_t=\rho\,dt.$$

    Here $\rho=-0.6$ is fixed. A European put price is an expectation,
    $P(K,T)=E_Q[e^{-rT}(K-S_T)^+]$, not a single simulated payoff. The
    numerical pricer estimates that expectation using many paths, with
    conditional Monte Carlo averaging to reduce noise.
    """)
    return


@app.cell
def _(bank_config, mo):
    _task = bank_config["task"]
    _context = _task["context"]
    mo.vstack([
        mo.md(r"""
        ## One observation is a panel, not a trajectory

        The conditioning input is a $[9,1]$ vector of noisy **currency** prices
        in the stored contract order. The training target is the paired vector
        $(H,\eta)$. Known spot, rates, curve, contract grid and quote-noise
        scale are fixed task context. Parameter truth, pricing standard errors
        and clean numerical prices are auxiliary training/audit information;
        none is silently appended to the observed panel.
        """),
        mo.ui.table([{"Parameter": _name, "Uniform lower": _bound[0], "Uniform upper": _bound[1]}
                     for _name, _bound in zip(_task["parameter_names"], _task["prior_bounds"])], selection=None),
        mo.ui.table([{"Contract index": _i, **_contract}
                     for _i, _contract in enumerate(_context["contracts"])], selection=None),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The numerical observation law is explicit

    Write $\theta=(H,\eta)$ and let $U$ contain the random numbers used in a
    262,144-path, 512-step hybrid pricing run. The simulated observation is

    $$\theta\sim\pi,\quad U\sim p_U,\qquad
    Y=\widehat C_{512,262144}(\theta,U)+0.25\epsilon,
    \quad \epsilon\sim N(0,I_9).$$

    There are **two noise sources**: finite-path price estimation and the
    deliberately added quote errors. The latter are independent across the
    nine contracts; shared pricing paths correlate the former. The bank saves
    the covariance of estimated price **means**, not of individual payoffs.

    This experiment targets the posterior of that finite numerical model.
    The preceding paired 512→1,024-step study supports using 512 steps for a
    bounded pilot; it does not prove that these are exact continuum prices.
    The per-panel MCSE screen is 0.025 currency, one tenth of the quote-noise
    standard deviation. Existing 128-step banks retain their separate identity.
    """)
    return


@app.cell
def _(bank_config, mo, protocol):
    _training = protocol["training"]
    _bases = bank_config["splits"]["train"]
    _augmentations = _training["quote_augmentations_per_base"]
    mo.vstack([
        mo.md(f"""
        ## Independence is counted at the base-panel level

        The frozen design contains **{_bases:,} independent training bases**.
        Each base has one parameter pair and one pricing-randomness draw. Adding
        {_augmentations} independent quote-error vectors produces **{_bases * _augmentations:,}
        training rows**, not that many independent base simulations. No
        augmentation of a training base may enter selection or audit data.

        The separate validation split contains 256 checkpoint-selection panels
        and 512 audit panels. Conditional checks use the first 16 audit panels
        plus four predeclared stress panels. The 256-case final test remains
        reserved unless all reference and neural gates pass; a final-test run
        would be a separate action, not a side effect of this notebook.

        Fixed-ID cluster chunks only change scheduling. The assembled arrays
        use the ordinary `option_bank_v1` schema, with exact coverage, source
        hashes and failed-ID preservation. A numerical failure is not redrawn.
        """),
        mo.ui.table([{"Role": "Training bases", "Count": _bases},
                     {"Role": "Training quote augmentations", "Count": _bases * _augmentations},
                     {"Role": "Independent reference rows", "Count": protocol["reference"]["rows"]},
                     {"Role": "Reserved final-test cases", "Count": bank_config["splits"]["test"]}], selection=None),
    ])
    return


@app.cell
def _(mo, original_reports, reports):
    _cluster = original_reports.get("cluster")
    _items = [mo.md(r"""
        ## Historical cluster generation cost

        Bank generation is a one-off cost shared by later training and audit
        calculations. The independent prior-reference bank is also reusable:
        a new compatible panel can reweight its existing rows without
        regenerating them, provided its importance diagnostics remain adequate.
        **Both routes therefore have amortization.** A neural speed advantage
        requires an actual comparison at comparable accuracy, with generation,
        training, posterior evaluation and downstream pricing counted separately.
        The table here is the **original pilot plus 4,096-row reference run**;
        it does not include the new 4,096-row extension or follow-up calculations.
        """)]
    if _cluster:
        _accounting = _cluster["accounting"]
        _allocation = _cluster["allocation"]
        _items.append(mo.ui.table([
            {"Quantity": "Generated base panels, total", "Recorded value": f"{_cluster['total_generated_panels']:,}"},
            {"Quantity": "Pilot / independent-reference panels", "Recorded value":
             f"{_cluster['banks']['pilot']['total_examples']:,} / {_cluster['banks']['reference']['total_examples']:,}"},
            {"Quantity": "Completed array tasks / submitted", "Recorded value":
             f"{_accounting['all_task_states'].get('COMPLETED', 0)} / {_allocation['array_tasks']}"},
            {"Quantity": "First task start to last task end (minutes)", "Recorded value":
             f"{_accounting['wall_span_seconds'] / 60:.2f}"},
            {"Quantity": "Allocated CPU hours (not utilized CPU time)", "Recorded value":
             f"{_accounting['allocated_cpu_hours']:.2f}"},
            {"Quantity": "Maximum simultaneous tasks, observed / requested", "Recorded value":
             f"{_allocation['observed_max_concurrent_tasks']} / {_allocation['max_concurrent_tasks_requested']}"},
            {"Quantity": "Allocated CPUs per task", "Recorded value": str(_allocation['cpus_per_task'])},
            {"Quantity": "Maximum single-task batch-step RSS (MiB)", "Recorded value":
             f"{_accounting['max_task_batch_rss_bytes'] / 2**20:.2f}"},
        ], selection=None))
        _items.append(mo.md(r"""
        The reference panel count includes two unused compatibility rows.
        Wall span excludes queueing before the first task starts. Allocated
        CPU hours sum each parent task's elapsed time times its allocated CPUs;
        they are **not measured utilized CPU time**, and job steps are not
        double-counted. RSS is the largest recorded batch-step resident-memory
        value, not the sum across simultaneous tasks.

        These measurements cover bank generation. Independent target-pricing
        reference calculations, neural training and downstream neural-draw
        pricing are separate costs. Successful cluster execution and portable
        bank delivery do not establish posterior or market-model acceptance.
        """))
    else:
        _items.append(mo.callout("No original cluster accounting report is present. Historical generation costs remain unreported.", kind="info"))
    _extension = reports.get("cluster")
    if _extension:
        _cost = _extension["accounting"]
        _items.append(mo.md("### Additional reference-bank generation\n\nThe following costs belong only to the new 4,096 rows, including unsuccessful infrastructure attempts."))
        _items.append(mo.ui.table([
            {"Quantity": "New / reused reference rows", "Recorded value": f"{_extension['new_generated_rows']:,} / {_extension['reused_reference_train_rows']:,}"},
            {"Quantity": "All task attempts / completed unique chunks", "Recorded value": f"{_cost['task_attempts']} / {_cost['unique_completed_task_ids']}"},
            {"Quantity": "Wall span, including retry gap (minutes)", "Recorded value": f"{_cost['wall_span_seconds'] / 60:.2f}"},
            {"Quantity": "Allocated CPU hours, all attempts", "Recorded value": f"{_cost['allocated_cpu_hours']:.2f}"},
            {"Quantity": "Of which timed-out attempts (CPU hours)", "Recorded value": f"{_cost['unsuccessful_allocated_cpu_hours']:.2f}"},
            {"Quantity": "Original prefix arrays exactly preserved", "Recorded value": str(_extension['prefix_all_arrays_bitwise_equal'])},
        ], selection=None))
        _items.append(mo.md(_cost["retry_cost_scope"]))
    else:
        _items.append(mo.callout("Extension cluster accounting is not yet recorded in the follow-up reports.", kind="info"))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Independent reference: weight the clean numerical prices

    The extended independently seeded bank contains 8,192 pairs $(\theta_i,U_i)$
    drawn from the prior and numerical simulator: 4,096 existing rows plus
    4,096 new rows. For an observed panel $y$, assign

    $$w_i(y)\propto\exp\left[-\frac{
        \|y-\widehat C(\theta_i,U_i)\|^2}{2(0.25)^2}\right].$$

    Normalize the weights to sum to one. Weighted parameter rows approximate
    $p(\theta\mid y)$ after integrating out $U$. This is **self-normalized
    importance sampling**, not an exact reference posterior. It is especially
    practical for this small prior box and deliberately restricted task.

    The likelihood is centered on `reference_prices`, never on an already
    quote-noised `observations` array. Do **not** add `pricing_covariance` to
    the quote likelihood: the independent $U_i$ rows already represent that
    numerical randomness, so doing so would count it twice.

    Effective sample size $1/\sum_iw_i^2$ describes weight concentration.
    Largest weight, Monte Carlo uncertainty and disagreement between two
    independent 4,096-row blocks provide additional checks. The original
    4,096-row run instead used two 2,048-row blocks. None can prove
    that a finite prior bank sampled every important region.

    This reference bank is itself reusable across compatible panels. In this
    two-parameter task, fast neural sampling alone therefore does not establish
    a speed advantage over the reference; their up-front and query costs must
    be compared separately.
    """)
    return


@app.cell
def _(mo):
    toy_observation = mo.ui.slider(-1.5, 1.5, step=.1, value=.7, show_value=True,
                                   label="Observed y in the scalar teaching model")
    mo.vstack([
        mo.md(r"""
        ### A small analogue you can manipulate

        This illustration is **not rough Bergomi and not pilot evidence**.
        Draw $\theta\sim U[-2,2]$, numerical noise $U\sim N(0,0.2^2)$,
        and observe $y=\theta+U+0.3\epsilon$. Weight independent $(\theta,U)$
        rows with $N(y;\theta+U,0.3^2)$. In this scalar example we can also
        integrate $U$ analytically: $y\mid\theta$ has variance $0.2^2+0.3^2$.
        The two calculations should agree up to finite-bank error.
        """),
        toy_observation,
    ])
    return (toy_observation,)


@app.cell
def _(mo, norm, np, plt, rough_reference, toy_observation):
    _rng = np.random.default_rng(91026)
    _theta = _rng.uniform(-2, 2, 8192)
    _numerical_noise = .2 * _rng.normal(size=len(_theta))
    _y = toy_observation.value
    _weights = rough_reference.importance_weights(
        (_theta + _numerical_noise)[:, None], np.array([_y]), .3)
    _grid = np.linspace(-2, 2, 600)
    _scale = np.sqrt(.2**2 + .3**2)
    _normalization = norm.cdf((2 - _y) / _scale) - norm.cdf((-2 - _y) / _scale)
    _density = norm.pdf((_grid - _y) / _scale) / (_scale * _normalization)
    _fig, _axis = plt.subplots(figsize=(9, 3.8), layout="constrained")
    _axis.hist(_theta, bins=np.linspace(-2, 2, 51), weights=_weights, density=True,
               color="#D97732", alpha=.4, label="Finite prior-importance reference")
    _axis.plot(_grid, _density, color="#087F8C", lw=2, label="Analytic marginal posterior")
    _axis.axhline(.25, color="#738095", ls="--", label="Prior density")
    _axis.set(xlabel="Scalar θ (toy units)", ylabel="Density", xlim=(-2, 2))
    _axis.legend(fontsize=8)
    mo.vstack([
        _fig,
        mo.md(f"Weighted mean: **{_weights @ _theta:.3f}**; importance ESS: **{1 / np.sum(_weights**2):.0f} / 8,192**. "
              "The analytic curve adds the two variances only after integrating out the numerical noise. "
              "The row-wise likelihood above uses the quote variance alone."),
    ])
    return


@app.cell
def _(cached_figures, mo, reports):
    _reference = reports.get("reference")
    _items = [mo.md("## Is the 8,192-row numerical reference precise enough?")]
    if _reference:
        _items.append(mo.ui.table([
            {"Gate": _name, "Recorded metric": _reference["metrics"].get(_name),
             "Threshold": _limit, "Passed": _reference["gates"].get(_name)}
            for _name, _limit in _reference["thresholds"].items()], selection=None))
        _items.append(mo.ui.table([
            {"Case": _case["case_id"], "Passed": _case["passed"],
             "Importance ESS": _case["reference"]["ess"],
             "Largest weight": _case["reference"]["max_weight"],
             "H mean MCSE / SD": _case["reference"]["mean_mcse_in_ref_sd"][0],
             "η mean MCSE / SD": _case["reference"]["mean_mcse_in_ref_sd"][1],
             "Failed gates": ", ".join(_name for _name, _passed in _case["gates"].items() if not _passed)}
            for _case in _reference["cases"]], selection=None))
        if "reference-precision" in cached_figures:
            _items.append(cached_figures["reference-precision"])
        _items.append(mo.ui.table([
            {"Reference-build stage": _stage, "Recorded seconds": _seconds}
            for _stage, _seconds in _reference.get("timing_seconds", {}).items()], selection=None))
    else:
        _items.append(mo.callout("No reference report is present. No precision or acceptance claim is available.", kind="info"))
    mo.vstack(_items)
    return


@app.cell
def _(mo, reports):
    _training_report = reports.get("training")
    _items = [mo.md(r"""
        ## One joint network, one bounded training run

        A four-layer spline coupling flow learns paired $(H,\eta)$ draws.
        Logit transforms handle the prior bounds; normalization is learned from
        training data, with no per-panel rescaling. The budget is 80 epochs,
        batch size 128, one seed and no architecture sweep. Selection uses the
        separate 256-panel selection set.

        For simulated training pairs $(\theta_i,Y_i)$, training minimizes
        $-\frac1N\sum_i\log q_\phi(\theta_i\mid Y_i)$. This is an inverse
        conditional-density objective: after training, a new compatible panel
        can be mapped to joint parameter draws cheaply. It does not train a
        network to regress option prices from parameters and contracts.

        A same-seed save/reload comparison checks checkpoint delivery. It says
        nothing by itself about posterior accuracy. Conditional, joint, SBC and
        price checks below decide whether that delivered candidate is accepted.
        """)]
    if _training_report:
        _items.append(mo.ui.table([
            {"Quantity": "Independent training bases", "Recorded value": str(_training_report["training_base_examples"])},
            {"Quantity": "Augmented training observations", "Recorded value": str(_training_report["training_examples"])},
            {"Quantity": "Exact same-seed reload", "Recorded value": str(_training_report["same_seed_reload_exact"])},
            *[{"Quantity": _name + " (seconds)", "Recorded value": f"{_value:.3f}"}
              for _name, _value in _training_report["timing_seconds"].items()],
        ], selection=None))
    else:
        _items.append(mo.callout(
            "Training was not run: the follow-up reference acceptance gate failed. A software smoke-test checkpoint is not this experiment's result."
            if reports.get("reference", {}).get("passed") is False else
            "No training report is present. The cached review does not load an optional checkpoint.", kind="info"))
    mo.vstack(_items)
    return


@app.cell
def _(cached_figures, mo, reports):
    _items = [mo.md(r"""
        ## Marginal SBC—and why SBC alone is insufficient

        The protocol calls for 512 independent audit panels with known
        synthetic truth and 512 posterior samples per panel, **after the
        reference gate passes**. Rank the truth among the samples,
        randomize ties and rank-bin positions, and compare the marginal rank
        distribution with uniform. When available, the diagnostic plots **empirical CDF
        minus the uniform CDF**; zero is ideal. Dashed DKW limits control the two marginal
        rank checks together at family $\alpha=0.05$.

        A method that ignores $y$ and always samples the prior can pass this
        check: under the prior-predictive experiment, truth and prior samples
        have the same marginal distribution. That is why a **prior control**
        and the conditional reference comparisons are integral to this pilot.
        SBC can detect some calibration errors, but cannot certify correct
        conditioning, dependence, every parameter regime, or market adequacy.
        """)]
    if "marginal-sbc" in cached_figures:
        _items.append(cached_figures["marginal-sbc"])
    else:
        _items.append(mo.callout(
            "SBC was not run for this follow-up because reference validation failed before training. No calibration claim follows from the software smoke runs."
            if reports.get("reference", {}).get("passed") is False else
            "The saved SBC report/ranks are absent; no ranks are synthesized here.", kind="info"))
    mo.vstack(_items)
    return


@app.cell
def _(mo, reports):
    _evaluation = reports.get("evaluation", {})
    available_case_ids = _evaluation.get("case_ids", [case["case_id"] for case in _evaluation.get("cases", [])])
    case_selector = mo.ui.dropdown(options=available_case_ids or ["No recorded cases"],
                                   value=(available_case_ids or ["No recorded cases"])[0],
                                   label="Inspect a fixed conditional case")
    mo.vstack([
        mo.md(r"""
        ## Is the network merely reproducing the prior?

        Compare **prior, reference and neural** means and 90% marginal intervals
        for the same observed panel. The reference says how much information
        that panel contains under the declared model. If its H marginal remains
        broad, a broad neural H marginal is not automatically a failure. Updating
        in η or in H–η dependence can coexist with weak marginal identification
        of H. Conversely, contraction alone is not proof of accurate learning.

        Select any predeclared case. These are equal-tail parameter intervals,
        not confidence intervals for the estimated means and not proof that
        the generating parameter must lie inside this particular interval.
        """),
        case_selector,
    ])
    return available_case_ids, case_selector


@app.cell
def _(available_case_ids, case_selector, mo, plot_helpers, reports):
    if available_case_ids:
        _index = available_case_ids.index(case_selector.value)
        _evaluation = reports["evaluation"]
        _case = _evaluation["cases"][_index]
        _rows = []
        for _dimension, _name in enumerate(("H", "η")):
            _rows.append({"Parameter": _name,
                          "Mean error / reference SD": _case["mean_error_in_ref_sd"][_dimension],
                          "Relative SD error": _case["relative_std_error"][_dimension],
                          "Reference mass of neural 90% interval": _case["reference_mass_of_90_interval"][_dimension]})
        _case_view = mo.vstack([
            plot_helpers.selected_case_figure(_evaluation, reports.get("reference", {}), _index),
            mo.ui.table(_rows, selection=None),
            mo.md(f"H–η correlation difference (neural minus reference): **{_case['correlation_error']:+.4f}**. "
                  "Inspecting dependence matters because marginal agreement can hide a different joint posterior."),
        ])
    else:
        _case_view = mo.callout("No conditional evaluation is recorded yet.", kind="info")
    _case_view
    return


@app.cell
def _(cached_figures, mo, reports):
    _evaluation = reports.get("evaluation")
    _items = [mo.md("## All fixed cases and frozen gates\n\nThe gray region identifies stress panels; every predeclared conditional case is shown. A failed gate remains part of the result.")]
    for _figure_name in ("posterior-intervals", "conditional-errors"):
        if _figure_name in cached_figures:
            _items.append(cached_figures[_figure_name])
    if _evaluation:
        _items.append(mo.ui.table([
            {"Gate": _name, "Recorded value": _evaluation["metrics"].get(_name),
             "Threshold": _limit, "Passed": _evaluation["gates"].get(_name)}
            for _name, _limit in _evaluation["thresholds"].items()], selection=None))
        _prior = _evaluation.get("prior_baseline", {}).get("metrics", {})
        if _prior:
            _items.append(mo.ui.table([
                {"Comparison": _name, "Neural": _evaluation["metrics"].get(_name), "Prior control": _prior.get(_name)}
                for _name in ("mean_rmse_in_ref_sd", "max_abs_mean_error_in_ref_sd", "max_abs_correlation_error")], selection=None))
        _items.append(mo.ui.table([
            {"Evaluation stage": _stage, "Recorded seconds": _seconds}
            for _stage, _seconds in _evaluation.get("timing_seconds", {}).items()], selection=None))
    else:
        _items.append(mo.callout("No neural gate outcomes are available.", kind="info"))
    mo.vstack(_items)
    return


@app.cell
def _(cached_figures, mo, protocol):
    _items = [mo.md(r"""
        ## What does posterior pricing mean here?

        SBI infers a distribution of parameters from the observed panel.
        Pricing then propagates those paired parameter draws through a target
        pricer. For two excluded puts, this pilot checks the **posterior mean**

        $$m(y)=E_{\theta\mid y}[P^{512}_{\rm target}(\theta)].$$

        The network here amortizes **calibration**. Each inferred parameter
        pair is still passed to the existing Monte Carlo pricer for the target
        contract. That downstream computation has its own cost, separate from
        neural posterior sampling. This rough experiment has no learned target
        price surrogate; that would require separate training and validation.
        The HKDE surrogate elsewhere in the project does not transfer to this
        model.

        Each target estimate uses fresh pricing randomness $V$, independent of
        the observation-pricing randomness $U$ given θ. Reusing U would select
        its lucky or unlucky price fluctuations through the conditioning weights
        and change the target. Reference rows use 4,096 fresh target paths;
        neural draws use 128. Pooling the estimates over posterior draws gives
        a mean-price estimate with outer and inner Monte Carlo uncertainty.

        A noisy target estimate is not an exact model-price draw. We therefore
        do **not** interpret quantiles of these estimates as posterior
        model-price intervals. That would require controlling the inner pricing
        error differently. These two target contracts also have no separate
        continuum-refinement certificate.

        The bars below are ±2 combined Monte Carlo standard errors. The frozen
        check is $|m_{\rm neural}-m_{\rm ref}|+2\,\mathrm{SE}\le0.06$ currency.
        It is a numerical screening criterion, not a simultaneous confidence
        statement or a bound on time-discretization bias.
        """),
        mo.ui.table(protocol["targets"], selection=None)]
    if "target-mean-errors" in cached_figures:
        _items.append(cached_figures["target-mean-errors"])
    else:
        _items.append(mo.callout("No downstream mean-price comparison is recorded yet.", kind="info"))
    mo.vstack(_items)
    return


@app.cell
def _(ROOT, mo, original_report_paths, report_paths, reports):
    _links = [f"- [{_name}]({_path})" for _name, _path in report_paths.items()]
    _links_text = "\n    ".join(_links)
    _original_links_text = "\n    ".join(f"- [Original {_name}]({_path})" for _name, _path in original_report_paths.items())
    _evaluation = reports.get("evaluation", reports.get("reference", {}))
    _test_state = str(_evaluation.get("reserved_test_evaluated", "not reported"))
    mo.md(f"""
    ## Where to inspect and reproduce the work

    - [Follow-up protocol and stop rule]({ROOT / 'configs/rough_reference8192_protocol.json'}) and [reference extension notes]({ROOT / 'docs/rough-reference-expansion.md'}).
    - [Original frozen protocol]({ROOT / 'docs/rough-inference-protocol.md'}) and [original machine-readable thresholds]({ROOT / 'configs/rough_inference_protocol.json'}).
    - [Run and milestone status]({ROOT / 'docs/rough-joint-inference.md'}) and [overall plan]({ROOT / 'docs/simulator-plan/README.md'}).
    - [Earlier numerical refinement]({ROOT / 'docs/bergomi-coupled-refinement.md'}).
    - [Plot renderer]({ROOT / 'scripts/plot_rough_inference.py'}); current reports use `docs/results/rough-reference8192/`.
    {_links_text}

    The historical artifacts remain in their original directory:
    {_original_links_text}

    Recorded final-test evaluation flag: **{_test_state}**. The default view
    reads only published audit artifacts. Simulator checks concern numerical
    implementation; conditional/SBC checks concern this synthetic inference
    experiment; empirical market assessment would require appropriate real
    observations and separately held-out outcomes. Those are distinct scopes.

    From the repository root, open this review with:

    ```bash
    PYTHONPATH=src pixi run marimo edit notebooks/09_rough_inference.py --no-token
    ```

    Regenerate standalone graphs from saved reports into a **new** directory:

    ```bash
    PYTHONPATH=src pixi run python scripts/plot_rough_inference.py \\
      --results docs/results/rough-reference8192 --out /tmp/rough-review-plots
    ```

    Bank generation, reference construction and training are explicit scripts
    (`assemble_option_bank.py`, `check_rough_reference.py`, `train_rough.py`,
    `evaluate_rough.py`) with frozen inputs. They are intentionally not live
    notebook cells. A delivered checkpoint is identified by its training
    report; cached review requires no ML
    framework import or checkpoint.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercises for review

    1. Move the scalar observation. Why does the importance-weighted parameter
       mean move while the prior mean stays fixed? Where does finite-bank noise
       appear relative to the analytic curve?
    2. Find a recorded case with a broad H interval. Compare the reference H
       interval, η interval and H–η correlation before deciding whether the
       network learned nothing.
    3. If the prior control passes marginal SBC but misses conditional means,
       what does that demonstrate about SBC's scope?
    4. Distinguish three quantities: posterior SD of H, Monte Carlo SE of its
       estimated posterior mean, and error in the neural posterior mean. Which
       graph or table addresses each?
    5. Explain why fresh target paths are required here, and why a distribution
       of noisy target estimates cannot directly supply price-uncertainty
       quantiles.
    6. Compare the original and extended reference diagnostics on all 20 cases.
       Why is reusing these development cases different from evaluating a new
       independent test cohort? Why do good ESS and maximum weight alone not
       establish reference precision?

    **Review decision:** judge the recorded gates and practical delivery
    separately. A failed bounded pilot can still deliver reusable simulators,
    banks and diagnostics. It should leave an explicit next-analysis note,
    preserve the frozen result, and keep the final test reserved rather than
    expanding the experiment until it passes.
    """)
    return


if __name__ == "__main__":
    app.run()
